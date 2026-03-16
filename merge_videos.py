import os
import argparse
import subprocess

def merge_videos(video1_path, video2_path, output_path):
    input1 = ffmpeg.input(video1_path)
    input2 = ffmpeg.input(video2_path)

    (
        ffmpeg
        .filter([input1, input2], 'hstack')
        .output(output_path)
        .overwrite_output()
        .run()
    )


def merge_folder(folder_path, output_path):
    videos = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
    ])

    if len(videos) % 2 != 0:
        raise ValueError("Number of videos in folder must be even.")

    os.makedirs(output_path, exist_ok=True)

    for i in range(0, len(videos), 2):
        video1 = os.path.join(folder_path, videos[i])
        video2 = os.path.join(folder_path, videos[i + 1])

        output_file = os.path.join(
            output_path,
            f"merged_{i//2:03d}.mp4"
        )

        print(f"Merging:\n  {video1}\n  {video2}\n -> {output_file}")

        merge_videos(video1, video2, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two videos side by side.")

    parser.add_argument("--merge_folder_mode", action="store_true",
                        help="Merge videos in a folder pairwise.")
    parser.add_argument("--inp_folder_path", type=str,
                        help="Path to the folder containing videos to merge.")
    parser.add_argument("--output_path", type=str,
                        help="Path to save merged videos.")

    parser.add_argument("--merge_pair_mode", action="store_true",
                        help="Merge two videos specified by paths.")
    parser.add_argument("--video1_path", type=str,
                        help="Path to the first video.")
    parser.add_argument("--video2_path", type=str,
                        help="Path to the second video.")

    args = parser.parse_args()

    if args.merge_pair_mode:
        if not args.video1_path or not args.video2_path or not args.output_path:
            raise ValueError("Pair mode requires --video1_path --video2_path and --output_path")

        merge_videos(args.video1_path, args.video2_path, args.output_path)
        print(f"Merged {args.video1_path} and {args.video2_path} into {args.output_path}")

    elif args.merge_folder_mode:
        if not args.inp_folder_path or not args.output_path:
            raise ValueError("Folder mode requires --inp_folder_path and --output_path")

        merge_folder(args.inp_folder_path, args.output_path)
        print(f"Merged videos in {args.inp_folder_path} and saved to {args.output_path}")

    else:
        parser.error("Please specify either --merge_pair_mode or --merge_folder_mode")

# Example usage:
# python merge_videos.py --merge_pair_mode --video1_path video1.mp4 --video2_path video2.mp4 --output_path merged_video.mp4
# python merge_videos.py --merge_folder_mode --inp_folder_path output/split_vid_no_ov_res --output_path output/split_vid_no_ov_res/merged/merged_video.mp4