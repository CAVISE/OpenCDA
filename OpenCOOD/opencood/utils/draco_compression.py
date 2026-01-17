"""
Point cloud compression using Google's Draco library.

This module provides functionality to compress and analyze point cloud data
using Google's Draco compression library. It supports saving point clouds in PLY
format and compressing them using the Draco encoder.

Notes
-----
To use this module, Draco must be installed from:
https://github.com/google/draco.git
"""

import random
import os
import re
import numpy as np
import torch
from glob import glob
import subprocess
from typing import List

draco = "/media/hdd/yuan/draco/build_dir/draco_encoder"


def save_ply(path: str, batch_coords: List[torch.Tensor], batch_features: List[torch.Tensor]) -> None:
    """
    Save point cloud data to PLY files.

    Parameters
    ----------
    path : str
        Directory path where to save the PLY files.
    batch_coords : list of torch.Tensor
        List of coordinate tensors, each with shape (N, 3).
    batch_features : list of torch.Tensor
        List of feature tensors, each with shape (N, C).

    Notes
    -----
    - Creates a new directory with a random 6-digit name under the given path.
    - Saves each point cloud in the batch as a separate PLY file.
    - Only processes elements after the first one in the batch (batch_coords[1:]).
    """
    # path = "/media/hdd/yuan/OpenCOOD/opencood/logs/fpvrcnn_intermediate_fusion/cpms/"
    dirname = "{:06d}".format(random.randint(0, 999999))
    os.mkdir(path + dirname)
    for bi, (coords, features) in enumerate(zip(batch_coords[1:], batch_features[1:])):
        header = f"ply\nformat ascii 1.0\nelement vertex {len(coords)}\nproperty float x\nproperty float y\nproperty float z\n"
        header = header + "".join([f"property float feat{i}\n" for i in range(32)]) + "end_header"
        data = torch.cat([coords, features], dim=1).detach().cpu().numpy()
        np.savetxt(path + dirname + f"/{bi + 1}.ply", data, delimiter=" ", header=header, comments="")


def draco_compression(ply_path: str) -> List[int]:
    """
    Compress all PLY files in a directory using Draco encoder.

    Parameters
    ----------
    ply_path : str
        Path to the directory containing PLY files.

    Returns
    -------
    list of int
        List of compressed file sizes in bytes for each PLY file.
    """
    files = glob(os.path.join(ply_path, "*/*.ply"))
    cpm_sizes = list(map(draco_compression_one, files))
    return cpm_sizes


def draco_compression_one(file: str) -> int:
    """
    Compress a single PLY file using Draco encoder.

    Parameters
    ----------
    file : str
        Path to the input PLY file.

    Returns
    -------
    int
        Size of the compressed file in bytes, or 0 if compression failed.
    """
    out_file = file.replace("ply", "drc")
    std_out = subprocess.getoutput(f"{draco} -point_cloud -i {file} -o {out_file}")
    size_str = re.findall("[0-9]+ bytes", std_out)
    if len(size_str) < 1:
        print("Compression failed:", file)
        cpm_size = 0
    else:
        cpm_size = int(size_str[0].split(" ")[0])

    return cpm_size


def cal_avg_num_kpts(ply_path: str) -> List[float]:
    """
    Calculate the average number of keypoints in PLY files.

    Parameters
    ----------
    ply_path : str
        Path to the directory containing PLY files.

    Returns
    -------
    list of float
        List of keypoint sizes in KB for each PLY file (vertices * 4 * 32 / 1024).
    """
    files = glob(os.path.join(ply_path, "*/*.ply"))

    def read_vertex_num(file: str) -> float:
        """
        Extract vertex count from PLY file header and calculate size.

        Parameters
        ----------
        file : str
            Path to PLY file.

        Returns
        -------
        float
            Size in KB (number of vertices * 4 * 32 / 1024).
        """
        with open(file, "r") as f:
            size_str = re.findall("element vertex [0-9]+", f.read())[0]
        return float(size_str.split(" ")[-1]) * 4 * 32 / 1024

    sizes = list(map(read_vertex_num, files))

    return sizes


if __name__ == "__main__":
    cpm_sizes = cal_avg_num_kpts("/media/hdd/yuan/OpenCOOD/opencood/logs/fpvrcnn_intermediate_fusion/cpms")
    # cpm_sizes = draco_compression("/media/hdd/yuan/OpenCOOD/opencood/logs/fpvrcnn_intermediate_fusion/cpms")
    print(np.array(cpm_sizes).mean())
