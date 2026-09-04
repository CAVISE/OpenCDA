"""Late-fusion V2X communication adapter."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from opencood.models.communication_adapters.base import ModelCommunicationAdapter
from opencood.models.communication_adapters.inference_input import (
    DetectionInferenceInput,
    InferenceInput,
    inference_input_to_dict,
)
from opencood.models.communication_adapters.inference_metadata import PoseFrameMetadata
from opencood.models.communication_adapters.wire_payload import LateFusionWirePayload
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2


@runtime_checkable
class LateFusionCommunicationDataset(Protocol):
    """Dataset capabilities required by the late-fusion adapter."""

    pre_processor: Any
    post_processor: Any

    def extract_data(
        self,
        idx: int,
        agent_payload_builder: Callable[[InferenceInput, PoseFrameMetadata], object],
    ) -> None:
        raise NotImplementedError


class LateFusionCommunicationAdapter(ModelCommunicationAdapter):
    """Run a complete detector locally and exchange detection candidates."""

    def prepare_transmission_payloads(self, dataset: Any, idx: int) -> None:
        """
        Run sender-side late-fusion inference for every local agent.

        Parameters
        ----------
        dataset : Any
            Dataset exposing local detector inputs and post-processing.
        idx : int
            Dataset frame to publish.

        Raises
        ------
        TypeError
            If the selected dataset lacks late-fusion capabilities.
        """
        if not isinstance(dataset, LateFusionCommunicationDataset):
            raise TypeError("Late fusion requires a dataset capable of building local detector inputs")
        dataset.extract_data(
            idx,
            agent_payload_builder=lambda inference_input, metadata: self._build_wire_payload(
                dataset,
                inference_input,
                metadata,
            ),
        )

    def _build_wire_payload(
        self,
        dataset: LateFusionCommunicationDataset,
        inference_input: InferenceInput,
        metadata: PoseFrameMetadata,
    ) -> LateFusionWirePayload:
        """
        Run the local detector and serialize its decoded candidates.

        Parameters
        ----------
        dataset : LateFusionCommunicationDataset
            Dataset providing pre- and post-processing implementations.
        inference_input : InferenceInput
            Preprocessed input belonging only to the sender.
        metadata : PoseFrameMetadata
            Sender pose and capture frame.

        Returns
        -------
        LateFusionWirePayload
            Decoded sender-frame boxes and their confidence scores.

        Raises
        ------
        TypeError
            If the model output or postprocessor result is invalid.
        """
        processed_lidar = dataset.pre_processor.collate_batch([inference_input_to_dict(inference_input)])
        anchor_box = dataset.post_processor.generate_anchor_box()
        model_input = self.move_to_device(
            {
                "processed_lidar": processed_lidar,
                "batch_size": 1,
                "anchor_box": (None if anchor_box is None else torch.from_numpy(np.asarray(anchor_box))),
            }
        )

        self.model.eval()
        with torch.no_grad():
            output = self.model(model_input)
        if not isinstance(output, Mapping):
            raise TypeError("Late-fusion model must return a tensor mapping")

        decode_predictions = getattr(dataset.post_processor, "decode_agent_predictions", None)
        if decode_predictions is None:
            raise TypeError("Late-fusion postprocessor must implement decode_agent_predictions()")
        boxes, scores = decode_predictions(model_input, output)
        if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("decode_agent_predictions() must return box and score tensors")

        return LateFusionWirePayload(
            detections=DetectionInferenceInput(
                boxes=boxes.detach().cpu().numpy(),
                scores=scores.detach().cpu().numpy(),
            ),
            metadata=metadata,
        )

    def decode_received_payload(
        self,
        payload: object,
        receiver_pose: list[float],
    ) -> dict[str, Any]:
        """
        Project one sender's detections into the receiver frame.

        Parameters
        ----------
        payload : object
            Deserialized late-fusion payload delivered over V2X.
        receiver_pose : list[float]
            Receiver LiDAR pose for the current inference frame.

        Returns
        -------
        dict[str, Any]
            Detection candidates expressed in the receiver frame.

        Raises
        ------
        TypeError
            If the payload or box representation is unsupported.
        """
        if not isinstance(payload, LateFusionWirePayload):
            raise TypeError(f"Expected LateFusionWirePayload, got {type(payload).__name__}")

        boxes = np.array(payload.detections.boxes, copy=True)
        transformation_matrix = x1_to_x2(payload.metadata.lidar_pose, receiver_pose)
        if boxes.ndim != 3 or boxes.shape[-2:] not in {(4, 2), (8, 3)}:
            raise TypeError(f"Unsupported late-fusion box shape: {boxes.shape}")
        if boxes.shape[0] > 0:
            if boxes.shape[-1] == 3:
                boxes = box_utils.project_box3d(boxes, transformation_matrix)
            else:
                box3d = np.pad(boxes, ((0, 0), (0, 0), (0, 1)))
                boxes = box_utils.project_points_by_matrix_torch(
                    box3d.reshape(-1, 3),
                    transformation_matrix,
                )[:, :2].reshape(-1, 4, 2)

        return {
            "pred_box_tensor": boxes,
            "pred_score": np.array(payload.detections.scores, copy=True),
        }
