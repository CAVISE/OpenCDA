"""Communication adapter for distributed FPV-RCNN inference."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch

from opencood.models.communication_adapters.base import ModelCommunicationAdapter
from opencood.models.communication_adapters.inference_input import (
    FpvrcnnAgentInferenceInput,
)
from opencood.models.communication_adapters.inference_metadata import PoseFrameMetadata
from opencood.models.communication_adapters.wire_payload import FpvrcnnWirePayload
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2


@runtime_checkable
class FpvrcnnCommunicationDataset(Protocol):
    """Dataset capabilities required by the FPV-RCNN adapter."""

    pre_processor: Any
    post_processor: Any

    def extract_data(
        self,
        idx: int,
        agent_payload_builder: Callable[[dict[str, Any], PoseFrameMetadata], object] | None = None,
    ) -> None: ...


class FpvrcnnCommunicationAdapter(ModelCommunicationAdapter):
    """Run FPV-RCNN stage one and VSA before serializing an agent payload."""

    def prepare_transmission_payloads(self, dataset: Any, idx: int) -> None:
        """
        Encode and publish FPV-RCNN outputs for every local agent.

        Parameters
        ----------
        dataset : Any
            Dataset exposing the FPV-RCNN agent-sample capabilities.
        idx : int
            Dataset frame to encode and publish.

        Raises
        ------
        TypeError
            If the selected dataset cannot prepare FPV-RCNN agent samples.
        """
        if not isinstance(dataset, FpvrcnnCommunicationDataset):
            raise TypeError("FPV-RCNN communication requires a dataset capable of building local agent observations")
        dataset.extract_data(
            idx,
            agent_payload_builder=lambda sample, metadata: self._build_wire_payload(
                dataset,
                sample,
                metadata,
            ),
        )

    def _build_wire_payload(
        self,
        dataset: FpvrcnnCommunicationDataset,
        sample: dict[str, Any],
        metadata: PoseFrameMetadata,
    ) -> FpvrcnnWirePayload:
        """
        Run sender-side FPV-RCNN and build its typed wire payload.

        Parameters
        ----------
        dataset : FpvrcnnCommunicationDataset
            Dataset providing the FPV-RCNN preprocessor and anchors.
        sample : dict[str, Any]
            Local sender model input without supervision fields.
        metadata : PoseFrameMetadata
            Sender pose and feature capture frame.

        Returns
        -------
        FpvrcnnWirePayload
            Typed payload containing proposals and VSA keypoint features.

        Raises
        ------
        TypeError
            If the model output is not the expected tensor mapping.
        """
        encode_agent = getattr(self.model, "encode_agent", None)
        if encode_agent is None:
            raise TypeError("FpvrcnnCommunicationAdapter requires a model with encode_agent()")

        processed_lidar = dataset.pre_processor.collate_batch([sample["processed_features"]])
        origin_lidar = np.pad(sample["projected_lidar"], ((0, 0), (1, 0)), mode="constant")
        batch = self.move_to_device(
            {
                "processed_lidar": processed_lidar,
                "origin_lidar": torch.from_numpy(origin_lidar),
                "anchor_box": torch.from_numpy(np.asarray(dataset.post_processor.generate_anchor_box())),
                "record_len": torch.ones(1, dtype=torch.long),
            }
        )
        self.model.eval()
        with torch.no_grad():
            output = encode_agent(batch)
        if not isinstance(output, Mapping):
            raise TypeError("FPV-RCNN encode_agent() must return a tensor mapping")

        return FpvrcnnWirePayload(
            inference_input=FpvrcnnAgentInferenceInput(
                proposals=self._tensor_to_numpy(output, "proposals"),
                proposal_scores=self._tensor_to_numpy(output, "proposal_scores"),
                point_coords=self._tensor_to_numpy(output, "point_coords"),
                point_features=self._tensor_to_numpy(output, "point_features"),
            ),
            metadata=metadata,
        )

    def decode_received_payload(
        self,
        payload: object,
        receiver_pose: list[float],
    ) -> dict[str, Any]:
        """
        Decode and project one FPV-RCNN payload into the receiver frame.

        Parameters
        ----------
        payload : object
            Deserialized FPV-RCNN payload delivered over V2X.
        receiver_pose : list[float]
            Receiver LiDAR pose for the current inference frame.

        Returns
        -------
        dict[str, Any]
            Remote proposals and VSA keypoint data in the receiver frame.

        Raises
        ------
        TypeError
            If a payload belonging to another model contract is received.
        """
        if not isinstance(payload, FpvrcnnWirePayload):
            raise TypeError(f"Expected FpvrcnnWirePayload, got {type(payload).__name__}")

        transformation_matrix = x1_to_x2(payload.metadata.lidar_pose, receiver_pose)
        point_coords = np.array(payload.inference_input.point_coords, copy=True)
        if len(point_coords) > 0:
            point_coords[:, :3] = box_utils.project_points_by_matrix_torch(
                point_coords[:, :3],
                transformation_matrix,
            )

        proposals = np.array(payload.inference_input.proposals, copy=True)
        if len(proposals) > 0:
            proposals[:, :3] = box_utils.project_points_by_matrix_torch(
                proposals[:, :3],
                transformation_matrix,
            )
            proposals[:, 6] += np.arctan2(
                transformation_matrix[1, 0],
                transformation_matrix[0, 0],
            )

        return {
            "proposals": proposals,
            "proposal_scores": np.array(payload.inference_input.proposal_scores, copy=True),
            "point_coords": point_coords,
            "point_features": np.array(payload.inference_input.point_features, copy=True),
        }

    @staticmethod
    def _tensor_to_numpy(output: Mapping[str, Any], field: str) -> npt.NDArray[Any]:
        """
        Detach one required model output as a NumPy array.

        Parameters
        ----------
        output : collections.abc.Mapping[str, Any]
            Sender-side model output.
        field : str
            Required tensor field to convert.

        Returns
        -------
        numpy.typing.NDArray[Any]
            Detached CPU representation of the tensor.

        Raises
        ------
        TypeError
            If the required field is not a tensor.
        """
        value = output.get(field)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'FPV-RCNN output field "{field}" must be a tensor')
        return value.detach().cpu().numpy()
