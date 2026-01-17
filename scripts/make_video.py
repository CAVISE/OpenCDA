"""
Utility script for creating videos from image sequences using FFmpeg.

This module provides functionality to:
- Create videos from sequences of images
- Apply rotation to the output video
- Control frame rate and output format
- Handle different image file patterns
"""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Optional


def create_video(input_dir: str, output_path: str, framerate: int = 20, rotate: int = 0, pattern: str = "*.png"):
    """
    Create a video from a sequence of images using FFmpeg.

    This function creates a video file from a sequence of images in a directory.
    It supports optional rotation of the output video and various image formats.

    Parameters
    ----------
    input_dir : str
        Path to directory containing source images.
    output_path : str
        Path where the output video will be saved.
    framerate : int, optional
        Frame rate in frames per second. Default is 20.
    rotate : int, optional
        Rotation angle in degrees counter-clockwise.
        Valid values are 90, 180, or 270. Default is 0 (no rotation).
    pattern : str, optional
        File pattern for image selection using glob format. Default is "*.png".

    Notes
    -----
    Requires FFmpeg to be installed and available in the system PATH.
    The output video will be in MP4 format with H.264 codec.
    """
    try:
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Directory {input_dir} not found")

        # Creates tmp file if rotation needed
        temp_file: Optional[Path] = None
        if rotate:
            temp_file = Path(tempfile.mktemp(suffix=".mp4"))

        intermediate_path = temp_file if rotate else output_path

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(framerate),
            "-pattern_type",
            "glob",
            "-i",
            str(Path(input_dir) / pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(intermediate_path),
        ]

        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

        if rotate:
            match rotate:
                case 90:
                    rotate_str = "transpose=2"
                case 180:
                    rotate_str = "transpose=1,transpose=1"
                case 270:
                    rotate_str = "transpose=1"
                case _:
                    print("Invalid rotate angle")
                    return -1

            rotate_cmd = ["ffmpeg", "-y", "-i", str(intermediate_path), "-vf", rotate_str, "-c:a", "copy", str(output_path)]
            subprocess.run(rotate_cmd, check=True, stderr=subprocess.PIPE)
            os.remove(intermediate_path)

        print(f"Video successfully created: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.output}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from frames using ffmpeg-python")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_path", default="video.mp4", help="Output video file path (e.g., video.mp4)")
    parser.add_argument("--framerate", type=int, default=20, help="Frame rate in frames per second (FPS)")
    parser.add_argument("--rotate", type=int, default=0, help="Rotation angle in degrees counter-clockwise (valid values: 90, 180, 270)")
    parser.add_argument("--pattern", default="*.png", help="Filename pattern for image sequence (glob format)")

    args = parser.parse_args()
    create_video(**vars(args))
