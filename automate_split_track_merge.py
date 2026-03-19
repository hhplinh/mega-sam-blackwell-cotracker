import argparse
import os
import subprocess
import shlex

"""
Example usage:
python automate_split_track_merge.py \
  --inp_video_path fish.mp4 \
  --checkpoint checkpoints/scaled_online.pth \
  --split_size 1 \
  --grid_size 50 \
  --grid_query_frame 0 \
  --work_dir output1s \
  --output_type cotracker

This will:
1. Split input video into 1s(split_size) chunks in output/split_vid
2. Run cotracker3/online_demo.py on each chunk, saving results to output/split_vid_res
3. Merge the processed videos into output/merged
"""

def run_cmd(cmd):
    print("About to run:", shlex.join(cmd))
    subprocess.run(cmd, check=True)

def split_video(ffmpeg_split_path, video_path, split_size, split_dir, extra="-map 0"):
    os.makedirs(split_dir, exist_ok=True)
    # Pass --extra as a single argument: '-e=-map 0'
    cmd = [
        "python", ffmpeg_split_path,
        "-f", video_path,
        "-s", str(split_size),
        "-o", split_dir,
        f"-e={extra}"
    ]
    run_cmd(cmd)

def run_online_demo(online_demo_path, video_path, checkpoint, grid_size, grid_query_frame, output_video_path, output_track_path):
    output_video_dir = os.path.dirname(output_video_path)
    os.makedirs(output_video_dir, exist_ok=True)
    cmd = [
        "python", online_demo_path,
        "--video_path", video_path,
        "--checkpoint", checkpoint,
        "--grid_size", str(grid_size),
        "--grid_query_frame", str(grid_query_frame),
        "--output_video_path", output_video_path,
        "--output_track_path", output_track_path
    ]
    run_cmd(cmd)

def merge_videos(merge_videos_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python", merge_videos_path,
        "--merge_folder_mode",
        "--inp_folder_path", input_dir,
        "--output_path", output_dir
    ]
    run_cmd(cmd)

def pipeline(ffmpeg_split_path, online_demo_path, merge_videos_path, inp_video_path, checkpoint, split_size, grid_size, grid_query_frame, work_dir, output_type):
    split_dir = os.path.join(work_dir, "split_vid")
    split_res_dir = os.path.join(work_dir, "split_vid_res")
    merged_dir = os.path.join(work_dir, "merged")
    megasam_dir = os.path.join(work_dir, "megasam_output")
    vis_overlay_dir = os.path.join(work_dir, "motion_overlay")

    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(split_res_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(megasam_dir, exist_ok=True)
    os.makedirs(vis_overlay_dir, exist_ok=True)

    # Split video
    split_video(ffmpeg_split_path, inp_video_path, split_size, split_dir)

    # Processing
    for fname in sorted(os.listdir(split_dir)):
        if not fname.lower().endswith(".mp4"):
            continue
        split_path = os.path.join(split_dir, fname)
        out_video = os.path.join(split_res_dir, fname)
        out_track = os.path.join(split_res_dir, fname.replace(".mp4", "_tracks.pt"))
        split_name = fname.replace('.mp4', '')
        megasam_split_dir = os.path.join(megasam_dir, split_name)
        frame_dir = os.path.join(split_dir, split_name)
        tracks_path = out_track
        output_dir = os.path.join(vis_overlay_dir, split_name)
        video_output = os.path.join(vis_overlay_dir, f"{split_name}_motion_overlay.mp4")
        os.makedirs(output_dir, exist_ok=True)

        if output_type in ["cotracker", "both"]:
            run_online_demo(online_demo_path, split_path, checkpoint, grid_size, grid_query_frame, out_video, out_track)

        if output_type in ["megasam", "both"]:
            frame_files = [f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.jpg')]
            max_frame_idx = 0
            for f in frame_files:
                try:
                    idx = int(f.split('_')[1].split('.')[0])
                    if idx > max_frame_idx:
                        max_frame_idx = idx
                except Exception:
                    continue
            end_frame = str(max_frame_idx) if max_frame_idx > 0 else "1"
            cmd = [
                "python", "visualize_motion/visualize_motion_cli.py",
                "--tracks_path", tracks_path,
                "--frame_dir", frame_dir,
                "--megasam_dir", megasam_split_dir,
                "--start_frame", "1",
                "--end_frame", end_frame,
                "--delta", "1",
                "--output_dir", output_dir,
                "--video_output", video_output
            ]
            run_cmd(cmd)

    # Merge videos
    merged_megasam_dir = os.path.join(work_dir, "merged_megasam")
    os.makedirs(merged_megasam_dir, exist_ok=True)
    if output_type == "cotracker":
        merge_videos(merge_videos_path, split_res_dir, merged_dir)
    elif output_type == "megasam":
        merge_videos(merge_videos_path, vis_overlay_dir, merged_megasam_dir)
    elif output_type == "both":
        merge_videos(merge_videos_path, split_res_dir, merged_dir)
        merge_videos(merge_videos_path, vis_overlay_dir, merged_megasam_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated pipeline: split -> online_demo (+ MegaSaM + visualization) -> merge.")
    parser.add_argument("--ffmpeg_split_path", type=str, default="ffmpeg-split.py", help="Path to ffmpeg-split.py script")
    parser.add_argument("--online_demo_path", type=str, default="cotracker3/online_demo.py", help="Path to online_demo.py script")
    parser.add_argument("--merge_videos_path", type=str, default="merge_videos.py", help="Path to merge_videos.py script")
    parser.add_argument("--inp_video_path", type=str, required=True, help="Input video file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split_size", type=int, default=1, help="Split size in second(s)")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for online_demo")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="Grid query frame for online_demo")
    parser.add_argument("--work_dir", type=str, default="output", help="Working directory for intermediate and output files")
    parser.add_argument("--output_type", type=str, default="both", choices=["cotracker", "megasam", "both"], help="Type of merged video to produce: cotracker, megasam, or both")
    args = parser.parse_args()

    pipeline(
        args.ffmpeg_split_path,
        args.online_demo_path,
        args.merge_videos_path,
        args.inp_video_path,
        args.checkpoint,
        args.split_size,
        args.grid_size,
        args.grid_query_frame,
        args.work_dir,
        args.output_type
    )
