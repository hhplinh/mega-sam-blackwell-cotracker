import argparse
import os
import torch

"""
combine_tracks.py

Combine multiple CoTracker track files (dicts with 'tracks' and 'visibility') from a folder into a single .pt file.

Usage:
python combine_tracks.py --input_dir output1s/split_vid_res --output_path output1s_combined/combined_tracks.pt

The output file will be a dict with keys:
- 'tracks': concatenated [T, N, 2] tensors
- 'visibility': concatenated [T, N, 1] tensors
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Folder containing *_tracks.pt files')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save combined .pt file')
    args = parser.parse_args()

    tracks_list = []
    visibility_list = []

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
    
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith('_tracks.pt'):
            continue
        fpath = os.path.join(args.input_dir, fname)
        data = torch.load(fpath)
        tracks = data['tracks']
        visibility = data['visibility']
        tracks_list.append(tracks)
        visibility_list.append(visibility)

    combined_tracks = torch.cat(tracks_list, dim=0)
    combined_visibility = torch.cat(visibility_list, dim=0)
    out = {'tracks': combined_tracks, 'visibility': combined_visibility}
    torch.save(out, args.output_path)
    print(f"Saved combined tracks to {args.output_path}")

if __name__ == '__main__':
    main()
