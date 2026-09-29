# Point-Tracking Consistency Evaluation Framework

A visual evaluation and diagnostic framework for analyzing **temporal track consistency** and **trajectory drift** in point-tracking models operating in feature-sparse, low-context scenes.

This repository combines a modified MegaSaM pipeline with CoTracker3 to make tracking failures easier to reproduce, inspect, and compare. The primary diagnostic mechanism periodically reinitializes CoTracker's tracking grid, then visualizes how the resulting trajectories behave across segment boundaries.

> **Status:** This project is experimental and under active development. Some comparison and overlay functionality is incomplete.

## Overview

Point trackers can appear stable in scenes with rich texture while drifting, losing identity, or accumulating error in scenes with few visual features. This project is designed to expose those failure modes through controlled visual experiments.

The evaluation workflow:

1. Splits a video into fixed-duration segments.
2. Reinitializes the CoTracker3 grid for each segment.
3. Tracks points independently within each segment.
4. Merges the processed segments into a single visualization.
5. Allows qualitative inspection of discontinuities, temporal inconsistency, and trajectory drift.

The main pipeline is implemented in [`automate_split_track_merge.py`](automate_split_track_merge.py).

## What This Framework Helps Diagnose

- **Temporal consistency:** Whether point trajectories remain smooth and stable over time.
- **Trajectory drift:** Whether tracks gradually diverge from the visual location of the point.
- **Reinitialization sensitivity:** How much tracking behavior changes when the point grid is restarted.
- **Feature-sparse scene behavior:** Failure modes in scenes with limited texture, repetitive appearance, motion blur, or weak visual context.
- **Segment-boundary artifacts:** Discontinuities introduced when tracking is restarted at regular intervals.
- **Model and pipeline comparisons:** A foundation for comparing CoTracker3, MegaSaM-derived motion estimates, and future point-tracking models.

## Quick Start

### Example: Reinitialize every second

```bash
python automate_split_track_merge.py \
  --inp_video_path fish.mp4 \
  --checkpoint checkpoints/scaled_online.pth \
  --split_size 1 \
  --grid_size 50 \
  --grid_query_frame 0 \
  --work_dir output1s \
  --output_type cotracker-point
```

This command:

1. Splits `fish.mp4` into one-second chunks under `output1s/split_vid`.
2. Runs CoTracker3 on each chunk and saves the results under `output1s/split_vid_res`.
3. Merges the processed chunks under `output1s/merged`.
4. Produces a visualization showing the effect of periodic tracking-grid reinitialization.

The split duration is the main experimental control. For example, use `--split_size 0.5`, `--split_size 1`, or `--split_size 5` to compare how frequently reinitialization affects tracking stability.

## Pipeline Components

| Stage | Script | Purpose |
| --- | --- | --- |
| Split | `ffmpeg-split.py` | Splits the input video into fixed-duration segments. FFmpeg re-encoding is used to improve timing accuracy at sub-second boundaries. |
| Track | `cotracker3/online_demo.py` | Runs CoTracker3 on each segment using a newly initialized tracking grid. |
| Merge | `merge_videos.py` | Combines the processed segment videos into a continuous output. |
| Track merge | `combine_tracks.py` | Combines track files from multiple processed segments into one `.pt` file. |

## Command-Line Arguments

### Required arguments

- `--inp_video_path`: Path to the input video.
- `--checkpoint`: Path to the CoTracker3 model checkpoint.

### Experiment and output settings

- `--work_dir`: Directory for intermediate files and final outputs. Default: `output`.
- `--output_type`: Visualization type:
  - `cotracker-point`: Displays CoTracker point tracks with periodic grid reinitialization.
  - `overlay`: Experimental visualization for comparing motion or flow estimates. This mode is not currently complete.
- `--split_size`: Duration of each video segment in seconds. Default: `1`.
- `--grid_size`: CoTracker grid size. Default: `10`.
- `--grid_query_frame`: Frame used to query the initial tracking grid. Default: `0`.

### Script path overrides

These options allow the pipeline to use scripts located outside their default paths:

- `--ffmpeg_split_path`: Path to `ffmpeg-split.py`. Default: `ffmpeg-split.py`.
- `--online_demo_path`: Path to `cotracker3/online_demo.py`. Default: `cotracker3/online_demo.py`.
- `--merge_videos_path`: Path to `merge_videos.py`. Default: `merge_videos.py`.

Run the following to inspect all available options:

```bash
python automate_split_track_merge.py --help
```

## Weights

Model weights are available here:

[Download checkpoints](https://drive.google.com/drive/folders/1V9Ela6ZxTYrXjnjmHKdBUZ3Aq0816CST?usp=sharing)

Place the required checkpoint at the path supplied with `--checkpoint`.

## Merging Track Files

To combine track outputs generated for multiple video segments:

```bash
python combine_tracks.py \
  --input_dir output1s/split_vid_res \
  --output_path output1s_combined/combined_tracks.pt
```

## Demo

The current visualization demonstrates periodic CoTracker grid reinitialization:

https://github.com/user-attachments/assets/c2cd21d7-c4e3-41bf-94e2-a05ec5a9d645

## Output Organization

By default, generated directories begin with `output` and are ignored by Git. A typical run may produce:

```text
output1s/
├── split_vid/       # Input video segments
├── split_vid_res/   # CoTracker outputs for each segment
└── merged/          # Merged visualization
```

If you change the work directory, update the commands above accordingly.

## Experimental Overlay Mode

The `overlay` output type is intended for side-by-side or overlaid comparisons of CoTracker motion and MegaSaM-derived motion. It is currently experimental for the following reasons:

- CoTracker may resize the video to account for points moving outside the image bounds.
- MegaSaM flow extraction is incomplete and currently commented out in `visualize_motion/visualize_motion.py`.
- RGB frames and CoTracker flow masks may require additional resizing or alignment before they can be compared reliably.
- Arrow-based flow visualizations can become noisy in low-context scenes.

The recommended output type for current experiments is `cotracker-point`.

## Installation

This repository is based on MegaSaM and includes code adapted to run on newer NVIDIA architectures, including Blackwell systems. The original MegaSaM pipeline remains available for depth, optical flow, camera motion, and 3D visualization experiments.

### Tested environment

The original MegaSaM code was tested with:

- Python 3.10
- CUDA 11.8
- PyTorch 2.0.1

A Conda environment is recommended.

### Create the environment

```bash
conda env create -f environment.yml
conda activate <environment-name>
```

### Install xFormers

Install xFormers using the instructions for your PyTorch and CUDA versions. For the original Python 3.10, CUDA 11.8, and PyTorch 2.0.1 environment, a prebuilt package can be installed with:

```bash
conda install https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
```

### Build the camera-tracking extension

Set `CUDA_HOME` to your CUDA installation and compile the extension:

```bash
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
cd base
python setup.py install
cd ..
```

On a cluster, CUDA may need to be loaded first, for example:

```bash
module load cuda/11.8
```

### Install visualization dependencies

```bash
pip install plotly
pip install -e viser
```

## MegaSaM Baseline Pipeline

The repository retains the original MegaSaM pipeline for processing videos represented as folders of frames.

Before running it, edit the paths in the first section of [`run_megasam.sh`](run_megasam.sh), then run:

```bash
bash run_megasam.sh
```

The pipeline writes intermediate results such as depth, optical flow, and reconstructions to the configured `OUT_DIR`, along with a final `sgd_cvd_hr.npz` result.

To visualize a result with Viser:

```bash
python viser/visualize_megasam.py \
  --data /path/to/sgd_cvd_hr.npz
```

When running remotely, forward the Viser port through SSH and open the local address printed by the visualizer, typically `http://localhost:8080`.

## Repository Background

This project is a modified version of the original [MegaSaM repository](https://github.com/mega-sam/mega-sam), with CoTracker3 added for 2D point correspondence and diagnostic experiments on Blackwell-compatible systems.

Original repository:

- [ludekcizinsky/mega-sam](https://github.com/ludekcizinsky/mega-sam)
- [MegaSaM project page](https://mega-sam.github.io/index.html)
- [MegaSaM paper](https://arxiv.org/abs/2412.04463)

## Citation

If you use the MegaSaM components, please cite:

```bibtex
@inproceedings{li2024_megasam,
  title     = {MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos},
  author    = {Li, Zhengqi and Tucker, Richard and Cole, Forrester and Wang, Qianqian and Jin, Linyi and Ye, Vickie and Kanazawa, Angjoo and Holynski, Aleksander and Snavely, Noah},
  booktitle = {arxiv},
  year      = {2024}
}
```

## License and Attribution

The original MegaSaM materials are distributed under the licenses described by the upstream project. Please review the upstream repository and included license files before redistributing modified code or model weights.

This is not an officially supported Google product.
