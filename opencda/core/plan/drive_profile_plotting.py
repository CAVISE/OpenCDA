# -*- coding: utf-8 -*-

"""
Visualization tools for vehicle dynamics profiles.

This module provides plotting functions for velocity, acceleration, time-to-collision,
time gap, and distance gap profiles for vehicle trajectory analysis.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License:  TDG-Attribution-NonCommercial-NoDistrib

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def draw_velocity_profile_single_plot(velocity_list: List[npt.NDArray[np.float64]]) -> None:
    """
    Draw velocity profiles in a single plot.

    Parameters
    ----------
    velocity_list : List[npt.NDArray[np.float64]]
        List of velocity profile arrays for each vehicle. Each array contains
        velocity values sampled at 0.05s intervals.
    """

    for i, v in enumerate(velocity_list):
        x_s = np.arange(len(v)) * 0.05
        plt.plot(x_s, v)

    plt.ylim([0, 34])

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    fig = plt.gcf()
    fig.set_size_inches(11, 5)


def draw_acceleration_profile_single_plot(acceleration: List[npt.NDArray[np.float64]]) -> None:
    """
    Draw acceleration profiles in a single plot.

    Parameters
    ----------
    acceleration : List[npt.NDArray[np.float64]]
        List of acceleration profile arrays for each vehicle. Each array contains
        acceleration values sampled at 0.05s intervals.
    """

    for i, v in enumerate(acceleration):
        x_s = np.arange(len(v)) * 0.05
        plt.plot(x_s, v)

    plt.ylim([-8, 8])

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m^2/s)")
    fig = plt.gcf()
    fig.set_size_inches(11, 5)


def draw_ttc_profile_single_plot(ttc_list: List[npt.NDArray[np.float64]]) -> None:
    """
    Draw time-to-collision (TTC) profiles in a single plot.

    Parameters
    ----------
    ttc_list : List[npt.NDArray[np.float64]]
        List of TTC profile arrays for each vehicle. Each array contains
        TTC values sampled at 0.05s intervals.
    """
    # this is used to find the merging vehicle position since its inter gap
    # length is always smaller

    for i, v in enumerate(ttc_list):
        x_s = np.arange(len(v)) * 0.05
        plt.plot(x_s, v)

    plt.xlabel("Time (s)")
    plt.ylabel("TTC (s)")
    plt.ylim([0, 30])
    fig = plt.gcf()
    fig.set_size_inches(11, 5)


def draw_time_gap_profile_singel_plot(gap_list: List[npt.NDArray[np.float64]]) -> None:
    """
    Draw time gap profiles in a single plot.

    Parameters
    ----------
    gap_list : List[npt.NDArray[np.float64]]
        List of time gap profile arrays for each vehicle. Each array contains
        front time gap values sampled at 0.05s intervals.
    """

    for i, v in enumerate(gap_list):
        x_s = np.arange(len(v)) * 0.05
        plt.plot(x_s, v)

    plt.xlabel("Time (s)")
    plt.ylabel("Time Gap (s)")
    plt.ylim([0.0, 1.8])
    fig = plt.gcf()
    fig.set_size_inches(11, 5)


def draw_dist_gap_profile_singel_plot(gap_list: List[npt.NDArray[np.float64]]) -> None:
    """
     Draw distance gap profiles in a single plot.

    Parameters
    ----------
     gap_list : list
         The vehicle front distance gap profile saved in a list.
    """
    for i, v in enumerate(gap_list):
        x_s = np.arange(len(v)) * 0.05
        plt.plot(x_s, v)

    plt.xlabel("Time (s)")
    plt.ylabel("Distance Gap (m)")
    plt.ylim([5, 35])
    fig = plt.gcf()
    fig.set_size_inches(11, 5)


def draw_sub_plot(
    velocity_list: List[npt.NDArray[np.float64]],
    acceleration_list: List[npt.NDArray[np.float64]],
    time_gap_list: List[npt.NDArray[np.float64]],
    distance_gap_list: List[npt.NDArray[np.float64]],
    ttc_list: List[npt.NDArray[np.float64]],
) -> plt.Figure:
    """
    This is a specific function that draws 4 in 1 images
    for trajectory following task.

    Parameters
    ----------
    velocity_list : list
        The vehicle velocity profile saved in a list.
    distance_gap_list : list
        The vehicle distance gap profile saved in a list.
    time_gap_list : list
        The vehicle time gap profile saved in a list.
    acceleration_list : list
        The vehicle acceleration profile saved in a list.
    ttc_list : list
        The ttc list.
    """
    fig = plt.figure()
    plt.subplot(511)
    draw_velocity_profile_single_plot(velocity_list)

    plt.subplot(512)
    draw_acceleration_profile_single_plot(acceleration_list)

    plt.subplot(513)
    draw_time_gap_profile_singel_plot(time_gap_list)

    plt.subplot(514)
    draw_dist_gap_profile_singel_plot(distance_gap_list)

    plt.subplot(515)
    draw_dist_gap_profile_singel_plot(distance_gap_list)

    label = []
    for i in range(1, len(velocity_list) + 1):
        label.append("Leading Vehicle, id: %d" % int(i - 1) if i == 1 else "Following Vehicle, id: %d" % int(i - 1))

    fig.legend(label, loc="upper right")

    return fig
