"""Early-fusion V2X communication adapter."""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from opencood.models.communication_adapters.base import ModelCommunicationAdapter
from opencood.models.communication_adapters.inference_metadata import PoseFrameMetadata
from opencood.models.communication_adapters.wire_payload import EarlyFusionWirePayload
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2


@runtime_checkable
class EarlyFusionCommunicationDataset(Protocol):
    """Dataset capabilities required by the early-fusion adapter."""

    def extract_data(
        self,
        idx: int,
        agent_payload_builder: Callable[[npt.NDArray[Any], PoseFrameMetadata], object],
    ) -> None: ...


class EarlyFusionCommunicationAdapter(ModelCommunicationAdapter):
    """Exchange local LiDAR points and project them only at the receiver."""

    def prepare_transmission_payloads(self, dataset: Any, idx: int) -> None:
        """
        Build and publish an early-fusion payload for every local agent.

        Parameters
        ----------
        dataset : Any
            Dataset exposing local early-fusion agent observations.
        idx : int
            Dataset frame to publish.

        Raises
        ------
        TypeError
            If the selected dataset cannot build early-fusion observations.
        """
        if not isinstance(dataset, EarlyFusionCommunicationDataset):
            raise TypeError("Early fusion requires a dataset capable of building local LiDAR observations")
        dataset.extract_data(idx, agent_payload_builder=self._build_wire_payload)

    @staticmethod
    def _build_wire_payload(
        lidar_points: npt.NDArray[Any],
        metadata: PoseFrameMetadata,
    ) -> EarlyFusionWirePayload:
        """
        Build a payload without applying a receiver-specific transformation.

        Parameters
        ----------
        lidar_points : numpy.typing.NDArray[Any]
            LiDAR points expressed in the sender coordinate frame.
        metadata : PoseFrameMetadata
            Sender pose and capture frame.

        Returns
        -------
        EarlyFusionWirePayload
            Local LiDAR points and the metadata needed by a receiver.
        """
        return EarlyFusionWirePayload(
            lidar_points=lidar_points,
            metadata=metadata,
        )

    def decode_received_payload(
        self,
        payload: object,
        receiver_pose: list[float],
    ) -> dict[str, Any]:
        """
        Project one sender-local point cloud into the receiver frame.

        Parameters
        ----------
        payload : object
            Deserialized early-fusion payload delivered over V2X.
        receiver_pose : list[float]
            Receiver LiDAR pose for the current inference frame.

        Returns
        -------
        dict[str, Any]
            Point cloud expressed in the receiver coordinate frame.

        Raises
        ------
        TypeError
            If a payload belonging to another contract is received.
        """
        if not isinstance(payload, EarlyFusionWirePayload):
            raise TypeError(f"Expected EarlyFusionWirePayload, got {type(payload).__name__}")

        transformation_matrix = x1_to_x2(payload.metadata.lidar_pose, receiver_pose)
        projected_lidar = np.array(payload.lidar_points, copy=True)
        if len(projected_lidar) > 0:
            projected_lidar[:, :3] = box_utils.project_points_by_matrix_torch(
                projected_lidar[:, :3],
                transformation_matrix,
            )
        return {"projected_lidar": projected_lidar}
