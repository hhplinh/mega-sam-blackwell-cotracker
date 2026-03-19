import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

"""
Visualize motion from CoTracker and MegaSaM in a single reference frame.

CoTracker: arrows from tracked points (frame t to t+delta)
MegaSaM: arrows from flow/correspondence, or rigid flow from depth+K+pose, or heatmap from mask/probability

"""

def visualize_motion(
    image_t,
    cotracker_tracks,
    cotracker_visibility,
    t,
    delta,
    megasam_flow=None,
    megasam_lowres_flow=None,
    megasam_depth=None,
    megasam_K=None,
    megasam_pose_t=None,
    megasam_pose_t1=None,
    megasam_move_prob=None,
    megasam_conf=None,
    out_path="motion_overlay.png",
    flow_stride=12,
    point_stride=1,
    min_mag=1.0,
    alpha_mask=0.35,
    suppress_dynamic_arrows=False,
):
    """
    Overlay CoTracker and MegaSaM motion on a single frame.

    Args:
        image_t: np.ndarray [H, W, 3] RGB image at frame t
        cotracker_tracks: torch.Tensor [T, N, 2] or [B, T, N, 2] tracked points
        cotracker_visibility: torch.Tensor [T, N, 1] or [B, T, N, 1] visibility mask
        t: int, reference frame index
        delta: int, step to next frame
        megasam_flow: np.ndarray [H, W, 2] or torch.Tensor, dense flow field
        megasam_lowres_flow: np.ndarray [h, w, 2] or torch.Tensor, low-res flow
        megasam_depth: np.ndarray [H, W] or torch.Tensor, depth map
        megasam_K: np.ndarray [3, 3] or torch.Tensor, intrinsics
        megasam_pose_t: np.ndarray [4, 4] or torch.Tensor, pose at t
        megasam_pose_t1: np.ndarray [4, 4] or torch.Tensor, pose at t+delta
        megasam_move_prob: np.ndarray [H, W] or torch.Tensor, movement probability mask
        megasam_conf: np.ndarray [H, W] or torch.Tensor, confidence map
        out_path: str, output PNG path
        flow_stride: int, stride for MegaSaM arrows
        point_stride: int, stride for CoTracker points
        min_mag: float, minimum arrow magnitude
        alpha_mask: float, alpha for mask overlays
        suppress_dynamic_arrows: bool, suppress arrows in dynamic regions
    Returns:
        Saves overlay PNG to out_path.
    """
    if torch.is_tensor(image_t):
        image_t = image_t.cpu().numpy()
    if image_t.dtype != np.uint8:
        image_t = np.clip(image_t, 0, 255).astype(np.uint8)

    H, W = image_t.shape[:2]
    overlay = image_t.copy()

    # CoTracker: draw arrows from t to t+delta
    tracks = cotracker_tracks
    vis = cotracker_visibility
    # Squeeze batch dimension if present
    if tracks.ndim == 4:
        tracks = tracks[0]
    if vis.ndim == 4:
        vis = vis[0]
    print("DEBUG: tracks shape after squeeze:", tracks.shape)
    print("DEBUG: vis shape after squeeze:", vis.shape)
    print("DEBUG: t, delta:", t, delta)
    pts_t = tracks[t]
    pts_t1 = tracks[t + delta]
    print("DEBUG: pts_t shape:", pts_t.shape)
    print("DEBUG: pts_t1 shape:", pts_t1.shape)
    if vis.ndim == 3:
        # If shape is [batch, frame, points], use batch 0
        vis_t = vis[0, t, :] > 0.5
        vis_t1 = vis[0, t + delta, :] > 0.5
        print(f"DEBUG: vis[0, t, :].shape: {vis[0, t, :].shape}, vis[0, t+delta, :].shape: {vis[0, t+delta, :].shape}")
    else:
        vis_t = vis[t][0, :] > 0.5
        vis_t1 = vis[t + delta][0, :] > 0.5
        print(f"DEBUG: vis[t][0, :].shape: {vis[t][0, :].shape}, vis[t+delta][0, :].shape: {vis[t+delta][0, :].shape}")
    print("DEBUG: vis_t shape:", vis_t.shape, "vis_t1 shape:", vis_t1.shape, "vis_t type:", type(vis_t), "vis_t1 type:", type(vis_t1))
    visible = vis_t & vis_t1
    print("DEBUG: visible shape:", visible.shape, "visible type:", type(visible))
    if hasattr(visible, 'cpu'):
        visible_np = visible.cpu().numpy()
    else:
        visible_np = np.array(visible)
    print("DEBUG: sum visible:", np.sum(visible_np))
    pts_t = pts_t[visible][::point_stride]
    pts_t1 = pts_t1[visible][::point_stride]
    print("DEBUG: pts_t after visible/stride:", pts_t.shape)
    print("DEBUG: pts_t1 after visible/stride:", pts_t1.shape)
    if torch.is_tensor(pts_t):
        pts_t = pts_t.cpu().numpy()
    if torch.is_tensor(pts_t1):
        pts_t1 = pts_t1.cpu().numpy()

    for p0, p1 in zip(pts_t, pts_t1):
        disp = p1 - p0
        mag = np.linalg.norm(disp)
        if mag < min_mag:
            continue
        cv2.circle(overlay, tuple(np.round(p0).astype(int)), 3, (0, 255, 0), -1)
        cv2.arrowedLine(
            overlay,
            tuple(np.round(p0).astype(int)),
            tuple(np.round(p1).astype(int)),
            (0, 255, 0),
            2,
            tipLength=0.2,
        )

    # # MegaSaM: dense flow
    # if megasam_flow is not None:
    #     flow = megasam_flow
    #     if torch.is_tensor(flow):
    #         flow = flow.cpu().numpy()
    #     if flow.shape[0] == 2 and flow.shape[1] == H:
    #         flow = np.transpose(flow, (1, 2, 0))
    #     for y in range(0, H, flow_stride):
    #         for x in range(0, W, flow_stride):
    #             v = flow[y, x]
    #             mag = np.linalg.norm(v)
    #             if mag < min_mag:
    #                 continue
    #             if megasam_conf is not None:
    #                 conf = megasam_conf[y, x] if not torch.is_tensor(megasam_conf) else megasam_conf[y, x].item()
    #                 if conf < 0.3:
    #                     continue
    #             cv2.arrowedLine(
    #                 overlay,
    #                 (x, y),
    #                 (int(x + v[0]), int(y + v[1])),
    #                 (255, 0, 0),
    #                 2,
    #                 tipLength=0.2,
    #             )

    # # MegaSaM: low-res flow
    # elif megasam_lowres_flow is not None:
    #     flow = megasam_lowres_flow
    #     if torch.is_tensor(flow):
    #         flow = flow.cpu().numpy()
    #     h, w = flow.shape[:2]
    #     for y in range(0, h, 1):
    #         for x in range(0, w, 1):
    #             v = flow[y, x]
    #             mag = np.linalg.norm(v)
    #             if mag < min_mag:
    #                 continue
    #             X = int(x * W / w)
    #             Y = int(y * H / h)
    #             cv2.arrowedLine(
    #                 overlay,
    #                 (X, Y),
    #                 (int(X + v[0]), int(Y + v[1])),
    #                 (255, 0, 0),
    #                 2,
    #                 tipLength=0.2,
    #             )

    # # MegaSaM: rigid flow from depth + K + pose
    # elif (
    #     megasam_depth is not None
    #     and megasam_K is not None
    #     and megasam_pose_t is not None
    #     and megasam_pose_t1 is not None
    # ):
    #     depth = megasam_depth
    #     K = megasam_K
    #     pose0 = megasam_pose_t
    #     pose1 = megasam_pose_t1
    #     if torch.is_tensor(depth):
    #         depth = depth.cpu().numpy()
    #     if torch.is_tensor(K):
    #         K = K.cpu().numpy()
    #     if torch.is_tensor(pose0):
    #         pose0 = pose0.cpu().numpy()
    #     if torch.is_tensor(pose1):
    #         pose1 = pose1.cpu().numpy()
    #     for y in range(0, H, flow_stride):
    #         for x in range(0, W, flow_stride):
    #             z = depth[y, x]
    #             if z <= 0:
    #                 continue
    #             pt = np.array([x, y, 1.0])
    #             X = np.linalg.inv(K) @ pt * z
    #             X_h = np.concatenate([X, [1.0]])
    #             X0 = pose0 @ X_h
    #             X1 = pose1 @ X_h
    #             x1 = K @ X1[:3]
    #             x1 = x1[:2] / x1[2]
    #             v = x1 - pt[:2]
    #             mag = np.linalg.norm(v)
    #             if mag < min_mag:
    #                 continue
    #             cv2.arrowedLine(
    #                 overlay,
    #                 (x, y),
    #                 (int(x + v[0]), int(y + v[1])),
    #                 (255, 0, 0),
    #                 2,
    #                 tipLength=0.2,
    #             )

    # # MegaSaM: movement probability mask
    # elif megasam_move_prob is not None:
    #     mask = megasam_move_prob
    #     if torch.is_tensor(mask):
    #         mask = mask.cpu().numpy()
    #     mask = np.clip(mask, 0, 1)
    #     heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #     overlay = cv2.addWeighted(overlay, 1 - alpha_mask, heatmap, alpha_mask, 0)

    # Legend and scale
    legend = np.zeros((80, overlay.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "CoTracker: green arrows", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.putText(legend, "MegaSaM: blue arrows", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.arrowedLine(legend, (overlay.shape[1] - 120, 40), (overlay.shape[1] - 70, 40), (255, 255, 255), 2, tipLength=0.2)
    cv2.putText(legend, "50 px", (overlay.shape[1] - 65, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    combined = np.vstack([overlay, legend])

    plt.figure(figsize=(12, 8))
    plt.imshow(combined[..., ::-1])
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # For debug
    cv2.imwrite(out_path.replace(".png", "_frame.png"), image_t)
    cv2.imwrite(out_path.replace(".png", "_cotracker.png"), overlay)

    print(f"Saved combined overlay to {out_path}")
