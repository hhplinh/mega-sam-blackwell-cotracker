import argparse
import os
import subprocess
import shlex

"""
Example usage:
python automate_split_track_merge.py \
  --video_path fish.mp4 \
  --checkpoint checkpoints/scaled_online.pth \
  --split_size 1 \
  --grid_size 50 \
  --grid_query_frame 0 \
  --work_dir output1s \
  --output_merge_video_path output1s/merged/merged1s.mp4

This will:
1. Split input video into 1s(split_size) chunks in output/split_vid
2. Run cotracker3/online_demo.py on each chunk, saving results to output/split_vid_res
3. Merge the processed videos into output/merged
"""

# Pipeline: split video -> run online_demo -> merge results

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

def pipeline(ffmpeg_split_path, online_demo_path, merge_videos_path, video_path, checkpoint, split_size, grid_size, grid_query_frame, work_dir):
    split_dir = os.path.join(work_dir, "split_vid")
    split_res_dir = os.path.join(work_dir, "split_vid_res")
    merged_dir = os.path.join(work_dir, "merged")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(split_res_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    # 1. Split video
    split_video(ffmpeg_split_path, video_path, split_size, split_dir)

    # 2. Run online_demo on each split
    for fname in sorted(os.listdir(split_dir)):
        if not fname.lower().endswith(".mp4"):
            continue
        split_path = os.path.join(split_dir, fname)
        out_video = os.path.join(split_res_dir, fname)
        out_track = os.path.join(split_res_dir, fname.replace(".mp4", "_tracks.pt"))
        run_online_demo(online_demo_path, split_path, checkpoint, grid_size, grid_query_frame, out_video, out_track)

    # 3. Merge processed videos
    merge_videos(merge_videos_path, split_res_dir, merged_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated pipeline: split -> online_demo -> merge.")
    parser.add_argument("--ffmpeg_split_path", type=str, default="ffmpeg-split.py", help="Path to ffmpeg-split.py script")
    parser.add_argument("--online_demo_path", type=str, default="cotracker3/online_demo.py", help="Path to online_demo.py script")
    parser.add_argument("--merge_videos_path", type=str, default="merge_videos.py", help="Path to merge_videos.py script")
    parser.add_argument("--video_path", type=str, required=True, help="Input video file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split_size", type=int, default=1, help="Split size in second(s)")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for online_demo")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="Grid query frame for online_demo")
    parser.add_argument("--work_dir", type=str, default="output", help="Working directory for intermediate and output files")
    parser.add_argument("--output_merge_video_path", type=str, default="output/merged/final_merged.mp4", help="Path to save final merged video")
    args = parser.parse_args()

    pipeline(
        args.ffmpeg_split_path,
        args.online_demo_path,
        args.merge_videos_path,
        args.video_path,
        args.checkpoint,
        args.split_size,
        args.grid_size,
        args.grid_query_frame,
        args.work_dir
    )
