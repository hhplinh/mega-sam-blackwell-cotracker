#!/usr/bin/env python

from __future__ import print_function

import csv
import json
import math
import os
import shlex
import subprocess
import argparse


def split_by_manifest(filename, manifest, output_dir=None, vcodec="copy", acodec="copy",
                      extra="", **kwargs):
    """ Split video into segments based on the given manifest file.

    Arguments:
        filename (str)      - Location of the video.
        manifest (str)      - Location of the manifest file.
        output_dir (str)    - Directory to save the split files.
        vcodec (str)        - Controls the video codec for the ffmpeg video
                            output.
        acodec (str)        - Controls the audio codec for the ffmpeg video
                            output.
        extra (str)         - Extra options for ffmpeg.
    """
    if not os.path.exists(manifest):
        print("File does not exist: %s" % manifest)
        raise SystemExit

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(manifest) as manifest_file:
        manifest_type = manifest.split(".")[-1]
        if manifest_type == "json":
            config = json.load(manifest_file)
        elif manifest_type == "csv":
            config = csv.DictReader(manifest_file)
        else:
            print("Format not supported. File must be a csv or json file")
            raise SystemExit

        split_cmd = ["ffmpeg", "-i", filename, "-vcodec", vcodec,
                     "-acodec", acodec, "-y"] + shlex.split(extra)
        try:
            fileext = filename.split(".")[-1]
        except IndexError as e:
            raise IndexError("No . in filename. Error: " + str(e))

        for video_config in config:
            split_args = []
            try:
                split_start = video_config["start_time"]
                split_length = video_config.get("end_time", None)
                if not split_length:
                    split_length = video_config["length"]
                filebase = video_config["rename_to"]
                if fileext in filebase:
                    filebase = ".".join(filebase.split(".")[:-1])

                output_path = os.path.join(output_dir, filebase + "." + fileext) if output_dir else filebase + "." + fileext

                split_args += ["-ss", str(split_start), "-t",
                               str(split_length), output_path]
                print("########################################################")
                print("About to run: " + " ".join(split_cmd + split_args))
                print("########################################################")
                subprocess.check_output(split_cmd + split_args)
            except KeyError as e:
                print("############# Incorrect format ##############")
                if manifest_type == "json":
                    print("The format of each json array should be:")
                    print("{start_time: <int>, length: <int>, rename_to: <string>}")
                elif manifest_type == "csv":
                    print("start_time,length,rename_to should be the first line ")
                    print("in the csv file.")
                print("#############################################")
                print(e)
                raise SystemExit


def get_video_length(filename):
    output = subprocess.check_output(("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                      "default=noprint_wrappers=1:nokey=1", filename)).strip()
    video_length = int(float(output))
    print("Video length in seconds: " + str(video_length))

    return video_length


def ceildiv(a, b):
    return int(math.ceil(a / float(b)))


def split_by_seconds(filename, split_length, output_dir=None, vcodec="libx264", acodec="aac",
                     extra="-map 0", video_length=None,**kwargs):
    if split_length and split_length <= 0:
        print("Split length can't be 0")
        raise SystemExit
    
    if not video_length:
        video_length = get_video_length(filename)
    
    second_overlap = kwargs.get("second_overlap", 0)
    
    if second_overlap and second_overlap >= split_length:
        print("Second overlap can't be greater than or equal to split length.")
        raise SystemExit

    if video_length <= split_length:
        split_count = 1
    else:
        step_size = split_length - second_overlap
        split_count = 1 + math.ceil((video_length - split_length) / step_size)

    if split_count == 1:
        print("Video length is less than or equal to the target split length.")
        raise SystemExit

    # Ensure the output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filebase, fileext = os.path.splitext(os.path.basename(filename))
    fileext = fileext[1:] if fileext else "mp4"

    for n in range(0, split_count):
        split_start = n * (split_length - second_overlap)
        
        current_split_length = min(split_length, video_length - split_start)    

        output_path = os.path.join(output_dir, f"{filebase}-{n+1}-of-{split_count}.{fileext}") if output_dir else f"{filebase}-{n+1}-of-{split_count}.{fileext}"

        full_cmd = [
            "ffmpeg",
            "-ss", str(split_start), 
            "-i", filename, 
            "-t", str(current_split_length),
            "-vcodec", vcodec, 
            "-acodec", acodec
        ] + shlex.split(extra) + [output_path]
        
        print("About to run: " + " ".join(full_cmd))
        subprocess.check_output(full_cmd)
        
    print("Saved %d split files to %s" % (split_count, output_dir if output_dir else os.getcwd()))


def main():
    parser = argparse.ArgumentParser(description="Split video files using ffmpeg.")
    parser.add_argument("-f", "--file", dest="filename", help="File to split, for example sample.avi", type=str)
    parser.add_argument("-s", "--split-size", dest="split_length", help="Split or chunk size in seconds, for example 10", type=int)
    parser.add_argument("-c", "--split-chunks", dest="split_chunks", help="Number of chunks to split to", type=int)
    parser.add_argument("-S", "--split-filesize", dest="split_filesize", help="Split or chunk size in bytes (approximate)", type=int)
    parser.add_argument("--filesize-factor", dest="filesize_factor", help="with --split-filesize, use this factor in time to size heuristics", type=float, default=0.95)
    parser.add_argument("--chunk-strategy", dest="chunk_strategy", help="with --split-filesize, allocate chunks according to given strategy (eager or even)", choices=['eager', 'even'], default='eager')
    parser.add_argument("-m", "--manifest", dest="manifest", help="Split video based on a json manifest file.", type=str)
    parser.add_argument("-o", "--output-dir", dest="output_dir", help="Directory to save split files.", type=str)
    parser.add_argument("-O", "--second-overlap", dest="second_overlap", help="Number of seconds to overlap between chunks.", type=int, default=0) # Uppercase of o
    parser.add_argument("-v", "--vcodec", dest="vcodec", help="Video codec to use.", type=str, default="libx264")
    parser.add_argument("-a", "--acodec", dest="acodec", help="Audio codec to use.", type=str, default="aac")
    parser.add_argument("-e", "--extra", dest="extra", help="Extra options for ffmpeg, e.g. '-e -threads 8'.", type=str, default="-map 0")
    args = parser.parse_args()

    def bailout():
        parser.print_help()
        raise SystemExit

    if not args.filename:
        bailout()

    # Convert args to dict for compatibility
    options = vars(args)

    if options.get("manifest"):
        split_by_manifest(**options)
    else:
        video_length = None
        if not options.get("split_length"):
            video_length = get_video_length(options["filename"])
            file_size = os.stat(options["filename"]).st_size
            split_filesize = None
            if options.get("split_filesize"):
                split_filesize = int(options["split_filesize"] * options["filesize_factor"])
            if split_filesize and options["chunk_strategy"] == 'even':
                options["split_chunks"] = ceildiv(file_size, split_filesize)
            if options.get("split_chunks"):
                options["split_length"] = ceildiv(video_length, options["split_chunks"])
            if not options.get("split_length") and split_filesize:
                options["split_length"] = int(split_filesize / float(file_size) * video_length)
        if not options.get("split_length"):
            bailout()
        split_by_seconds(video_length=video_length, **options)


if __name__ == '__main__':
    main()


# Example usage:
# python ffmpeg-split.py -f fish.mp4 -s 5 -o split_vid_no_ov -e '-map 0'