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
