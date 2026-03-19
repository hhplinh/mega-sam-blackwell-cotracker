"""
visualize_motion_cli.py

Command-line interface for visualize_motion utility.

Inputs:
- --tracks_path: path to CoTracker .pt file (dict with 'tracks' and 'visibility')
- --frame_path: path to RGB frame image
- --megasam_dir: path to MegaSaM output folder (per-frame .npz)

Usage example:
python visualize_motion_cli.py \
    --tracks_path output1s/split_vid_res/fish-1-of-9_tracks.pt \
    --frame_path output1s/split_vid_res/fish-1-of-9.png \
    --megasam_dir inference/output/fish/unidepth \
    --frame_idx 10 \
    --delta 1 \
    --output motion_overlay.png

The CoTracker .pt file must be a dict with keys 'tracks' ([T, N, 2]) and 'visibility' ([T, N, 1]).
"""

import argparse
import os
import numpy as np
import torch
import cv2
from visualize_motion import visualize_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_path", type=str, required=True, help="Path to CoTracker .pt file (dict with 'tracks' and 'visibility')")
    parser.add_argument("--frame_path", type=str, required=True, help="Path to RGB frame image")
    parser.add_argument("--megasam_dir", type=str, required=True, help="Path to MegaSaM output folder (per-frame .npz)")
    parser.add_argument("--frame_idx", type=int, required=True, help="Frame index t")
    parser.add_argument("--delta", type=int, default=1, help="Delta to next frame")
    parser.add_argument("--output", type=str, default="motion_overlay.png", help="Output PNG path")
    parser.add_argument("--flow_stride", type=int, default=12)
    parser.add_argument("--point_stride", type=int, default=1)
    parser.add_argument("--min_mag", type=float, default=1.0)
    parser.add_argument("--alpha_mask", type=float, default=0.35)
    parser.add_argument("--suppress_dynamic_arrows", action="store_true")
    args = parser.parse_args()

    # Load CoTracker tracks and visibility from dict .pt file
    cotracker_data = torch.load(args.tracks_path)
    cotracker_tracks = cotracker_data['tracks']
    cotracker_visibility = cotracker_data['visibility']

    # Load frame t
    image_t = cv2.cvtColor(cv2.imread(args.frame_path), cv2.COLOR_BGR2RGB)

    # Load MegaSaM per-frame .npz
    npz_path = os.path.join(args.megasam_dir, f"frame_{args.frame_idx:06d}.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"MegaSaM .npz not found: {npz_path}")
    data = np.load(npz_path)

    # Try to extract available fields
    megasam_flow = data['flow'] if 'flow' in data else None
    megasam_lowres_flow = data['lowres_flow'] if 'lowres_flow' in data else None
    megasam_depth = data['depth'] if 'depth' in data else None
    megasam_K = data['K'] if 'K' in data else None
    megasam_pose_t = data['pose'] if 'pose' in data else None
    megasam_move_prob = data['move_prob'] if 'move_prob' in data else None
    megasam_conf = data['conf'] if 'conf' in data else None

    # For pose_t1, load .npz for t+delta
    pose_t1 = None
    npz_path_t1 = os.path.join(args.megasam_dir, f"frame_{args.frame_idx + args.delta:06d}.npz")
    if os.path.isfile(npz_path_t1):
        data_t1 = np.load(npz_path_t1)
        pose_t1 = data_t1['pose'] if 'pose' in data_t1 else None

    # Decide visualization path
    if megasam_flow is not None:
        print("Using MegaSaM dense flow for visualization.")
    elif megasam_lowres_flow is not None:
        print("Using MegaSaM low-res flow for visualization.")
    elif (
        megasam_depth is not None
        and megasam_K is not None
        and megasam_pose_t is not None
        and pose_t1 is not None
    ):
        print("Using MegaSaM rigid flow from depth + K + pose.")
    elif megasam_move_prob is not None:
        print("Using MegaSaM movement probability mask for visualization.")
    else:
        print("No MegaSaM motion representation found. Only CoTracker will be shown.")

    visualize_motion(
        image_t,
        cotracker_tracks,
        cotracker_visibility,
        args.frame_idx,
        args.delta,
        megasam_flow=megasam_flow,
        megasam_lowres_flow=megasam_lowres_flow,
        megasam_depth=megasam_depth,
        megasam_K=megasam_K,
        megasam_pose_t=megasam_pose_t,
        megasam_pose_t1=pose_t1,
        megasam_move_prob=megasam_move_prob,
        megasam_conf=megasam_conf,
        out_path=args.output,
        flow_stride=args.flow_stride,
        point_stride=args.point_stride,
        min_mag=args.min_mag,
        alpha_mask=args.alpha_mask,
        suppress_dynamic_arrows=args.suppress_dynamic_arrows,
    )

if __name__ == "__main__":
    main()
