import argparse
import os
import numpy as np
import torch
import cv2
from visualize_motion import visualize_motion
import glob


"""
Command-line interface for visualize_motion utility.

Inputs:
- --tracks_path: path to CoTracker .pt file (dict with 'tracks' and 'visibility')
- --frame_dir: directory containing frame images (searches recursively for frame_XXXXXX.jpg)
- --megasam_dir: path to MegaSaM output folder (per-frame .npz)
- --start_frame: start frame index (inclusive)
- --end_frame: end frame index (inclusive)
- --delta: step to next frame (motion is visualized from t to t+delta)
- --output_dir: directory to save overlay PNGs for each frame
- --video_output: path to save the combined video (mp4)

Usage example:
python visualize_motion/visualize_motion_cli.py \
    --tracks_path output1s_combined/combined_tracks.pt \
    --frame_dir inference/data/test200 \
    --megasam_dir inference/output/fish/unidepth \
    --start_frame 1 \
    --end_frame 30 \
    --delta 1 \
    --output_dir output_motion_overlay \
    --video_output output_motion_overlay/motion_overlay.mp4

The CoTracker .pt file must be a dict with keys 'tracks' ([T, N, 2]) and 'visibility' ([T, N, 1]).
"""

def find_frame(frame_dir, frame_idx):
    pattern = f"frame_{frame_idx:06d}.jpg"
    matches = glob.glob(os.path.join(frame_dir, '**', pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"Frame image not found: {pattern} in {frame_dir}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_path", type=str, required=True, help="Path to CoTracker .pt file (dict with 'tracks' and 'visibility')")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory containing frame images (searches recursively for frame_XXXXXX.jpg)")
    parser.add_argument("--megasam_dir", type=str, required=True, help="Path to MegaSaM output folder (with *.npz)")
    parser.add_argument("--start_frame", type=int, required=True, help="Start frame index (inclusive)")
    parser.add_argument("--end_frame", type=int, required=True, help="End frame index (inclusive)")
    parser.add_argument("--delta", type=int, default=1, help="Step to next frame. Motion is visualized from frame t to frame t+delta. Default: 1 (next frame).")
    parser.add_argument("--output_dir", type=str, default="output_motion_overlay", help="Directory to save overlay PNGs for each frame.")
    parser.add_argument("--video_output", type=str, default="output_motion_overlay/motion_overlay.mp4", help="Path to save the combined video.")
    parser.add_argument("--flow_stride", type=int, default=12, help="Pixel stride for MegaSaM arrows.")
    parser.add_argument("--point_stride", type=int, default=1, help="Stride for CoTracker points.")
    parser.add_argument("--min_mag", type=float, default=1.0, help="Minimum arrow magnitude (in pixels) to draw.")
    parser.add_argument("--alpha_mask", type=float, default=0.35, help="Transparency for MegaSaM mask overlays.")
    parser.add_argument("--suppress_dynamic_arrows", action="store_true", help="Suppress MegaSaM arrows in highly dynamic regions.")
    args = parser.parse_args()

    # Load CoTracker tracks and visibility from dict .pt file
    cotracker_data = torch.load(args.tracks_path)
    cotracker_tracks = cotracker_data['tracks']
    cotracker_visibility = cotracker_data['visibility']

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    frame_indices = list(range(args.start_frame, args.end_frame + 1))
    image_paths = []
    # Debug visibility tensor shape
    print(f"DEBUG: cotracker_visibility shape: {cotracker_visibility.shape}, ndim: {cotracker_visibility.ndim}")
    num_vis_frames = cotracker_visibility.shape[1] if cotracker_visibility.ndim == 3 else cotracker_visibility.shape[0]
    num_vis_batches = cotracker_visibility.shape[0] if cotracker_visibility.ndim == 3 else None
    print(f"DEBUG: num_vis_frames: {num_vis_frames}, num_vis_batches: {num_vis_batches}")
    for t in frame_indices:
        # Add bounds check for visibility tensor
        if cotracker_visibility.ndim == 3:
            if t >= cotracker_visibility.shape[1]:
                print(f"Skipping frame {t}: t exceeds visibility tensor frame count ({cotracker_visibility.shape[1]})")
                continue
            if (t + args.delta) >= cotracker_visibility.shape[1]:
                print(f"Skipping frame {t}: t+delta ({t + args.delta}) exceeds visibility tensor frame count ({cotracker_visibility.shape[1]})")
                continue
        else:
            if t >= cotracker_visibility.shape[0]:
                print(f"Skipping frame {t}: t exceeds visibility tensor frame count ({cotracker_visibility.shape[0]})")
                continue
            if (t + args.delta) >= cotracker_visibility.shape[0]:
                print(f"Skipping frame {t}: t+delta ({t + args.delta}) exceeds visibility tensor frame count ({cotracker_visibility.shape[0]})")
                continue
        try:
            frame_path = find_frame(args.frame_dir, t)
            image_t = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Skipping frame {t}: {e}")
            continue

        npz_path = os.path.join(args.megasam_dir, f"frame_{t:06d}.npz")
        if not os.path.isfile(npz_path):
            print(f"Skipping frame {t}: MegaSaM .npz not found: {npz_path}")
            continue
        data = np.load(npz_path)

        megasam_flow = data['flow'] if 'flow' in data else None
        megasam_lowres_flow = data['lowres_flow'] if 'lowres_flow' in data else None
        megasam_depth = data['depth'] if 'depth' in data else None
        megasam_K = data['K'] if 'K' in data else None
        megasam_pose_t = data['pose'] if 'pose' in data else None
        megasam_move_prob = data['move_prob'] if 'move_prob' in data else None
        megasam_conf = data['conf'] if 'conf' in data else None

        # For pose_t1, load .npz for t+delta
        pose_t1 = None
        npz_path_t1 = os.path.join(args.megasam_dir, f"frame_{t + args.delta:06d}.npz")
        if os.path.isfile(npz_path_t1):
            data_t1 = np.load(npz_path_t1)
            pose_t1 = data_t1['pose'] if 'pose' in data_t1 else None

        overlay_path = os.path.join(args.output_dir, f"motion_overlay_{t:06d}.png")
        visualize_motion(
            image_t,
            cotracker_tracks,
            cotracker_visibility,
            t,
            args.delta,
            megasam_flow=megasam_flow,
            megasam_lowres_flow=megasam_lowres_flow,
            megasam_depth=megasam_depth,
            megasam_K=megasam_K,
            megasam_pose_t=megasam_pose_t,
            megasam_pose_t1=pose_t1,
            megasam_move_prob=megasam_move_prob,
            megasam_conf=megasam_conf,
            out_path=overlay_path,
            flow_stride=args.flow_stride,
            point_stride=args.point_stride,
            min_mag=args.min_mag,
            alpha_mask=args.alpha_mask,
            suppress_dynamic_arrows=args.suppress_dynamic_arrows,
        )
        image_paths.append(overlay_path)

    # Create video from overlays using ffmpeg
    if image_paths:
        print(f"Creating video {args.video_output} from {len(image_paths)} frames...")
        # ffmpeg expects input as motion_overlay_%06d.png
        ffmpeg_pattern = os.path.join(args.output_dir, "motion_overlay_%06d.png")
        cmd = f"ffmpeg -y -framerate 10 -start_number {args.start_frame} -i {ffmpeg_pattern} -c:v libx264 -pix_fmt yuv420p {args.video_output}"
        print(f"Running: {cmd}")
        os.system(cmd)
    else:
        print("No overlays generated, video not created.")

if __name__ == "__main__":
    main()
