"""Typed metadata exchanged through communication adapters."""

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class PoseFrameMetadata:
    """Pose and capture frame used to derive pairwise transformations.

    Attributes
    ----------
    lidar_pose : list[float] | tuple[float, ...]
        Agent LiDAR pose at feature capture time.
    capture_frame : int
        Dataset frame in which the model input was captured.
    """

    lidar_pose: list[float] | tuple[float, ...]
    capture_frame: int


@dataclass(frozen=True, slots=True, kw_only=True)
class V2XViTMetadata:
    """Metadata consumed by the V2X-ViT fusion model.

    Attributes
    ----------
    velocity : float
        Sender velocity normalized by 30 km/h.
    agent_type : int
        ``1`` for an infrastructure agent and ``0`` for a vehicle.
    capture_frame : int
        Dataset frame in which the model input was captured.
    spatial_correction_matrix : numpy.typing.NDArray[Any]
        Transformation correcting the captured feature to the receive frame.
    """

    velocity: float
    agent_type: int
    capture_frame: int
    spatial_correction_matrix: npt.NDArray[Any]


IntermediateMetadata: TypeAlias = PoseFrameMetadata | V2XViTMetadata
