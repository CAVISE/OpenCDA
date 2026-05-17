"""
AdvCP intermediate-fusion attack.

In an intermediate-fusion pipeline each CAV produces its own spatial
feature map (after the per-CAV backbone but before the fusion head).
The cooperative model concatenates / fuses these per-CAV feature maps
and produces a single detection output.

This module attacks at exactly that layer: it perturbs the attacker's
spatial feature map before the fusion head sees it, in the immediate
neighbourhood of the target box. The perturbation is the solution of
an Adam optimization that:

- Increases the model's confidence at the target location for spoofing.
- Decreases the model's confidence at the target location for removal.

Loss formulation
----------------
Let ``s_i`` be the sigmoided proposal score at anchor ``i`` and
``w_i`` the IoU between the proposal box and the target box. Writing
``log_p(x) = log(clamp(x, 1e-6))``:

- Spoofing loss = sum over all targets t of:
    ``sum_{i : w_i^t >= 0.01} w_i^t * log_p(1 - s_i)``
  Minimising this pushes scores up for proposals overlapping the
  target.
- Removal loss = sum over all targets t of:
    ``sum_{i : w_i^t >= 0.01} w_i^t * log_p(s_i)``
  Minimising this pushes scores down for proposals overlapping the
  target.

The perturbation tensor itself is constrained to an L-infinity ball
of radius ``max_perturb`` (clamped before each forward pass), and
spatially confined to a square patch of side ``2 * feature_size``
centred on the target's pixel coordinates in the feature map.

Single attacker vs multiple attackers
-------------------------------------
- Single attacker: :meth:`_optimize_spoofing` runs one Adam optimizer
  on a single perturbation tensor.
- Multiple attackers: :meth:`_optimize_joint` runs one shared Adam
  optimizer over all attackers' perturbation tensors and a single
  combined forward pass per step. This is necessary because the
  cooperative backbone fuses the attackers' features together;
  optimizing them independently would ignore the interaction and
  produce weaker attacks.

Sync, init, online
------------------
- ``sync``: when both the previous and the current tick contain all
  configured attackers, optimize the perturbation against the
  previous-tick batch instead of the current-tick batch. The previous
  tick's spatial features are usually closer to a stable optimum.
- ``init``: warm-start the perturbation extraction by ray-tracing a
  synthetic mesh into the attacker's lidar cloud (reusing the
  early-fusion ``_apply_init_*_to_memory`` paths) so the base
  perturbation already encodes a synthetic-object signal before Adam
  begins.
- ``online``: persist the converged perturbation and reuse it as the
  initial value for the next tick (significantly speeds up
  convergence when the scene changes slowly).
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from functools import wraps
import logging
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from opencood.tools import inference_utils
from opencood.utils import box_utils

from opencda.core.attack.advcp.attack_helper import AdvCPAttackHelper
from opencda.core.attack.advcp.early_fusion_attack import AdvCoperceptionEarlyFusionAttack
from opencda.core.attack.advcp.types import (
    AdvCPAttackResult,
    AdvCPConfig,
    AdvCPIntermediateAttackState,
    AdvCPMemoryData,
    AdvCPMemoryRecord,
    AdvCPScenarioData,
    AdvCPVisualizationContext,
    AttackerId,
    BoxLwhBottomCenter,
)
from opencda.core.common.coperception_data_processor import LiveMemorySnapshot

logger = logging.getLogger("cavise.opencda.opencda.core.attack.advcp.intermediate_fusion_attack")


class AdvCoperceptionIntermediateFusionAttack:
    """
    Intermediate-fusion AdvCP attack runner.

    All entry points are class / static methods; instantiation is
    unnecessary. The only persistent state lives in
    :class:`AdvCPIntermediateAttackState`, which the manager passes in
    through ``run(..., attack_state=...)``.
    """

    @staticmethod
    def run(
        batch_data: Any,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData | None = None,
        attack_state: AdvCPIntermediateAttackState | None = None,
    ) -> AdvCPAttackResult:
        """
        Execute the intermediate-fusion attack for a single tick.

        Resolves the present attackers, dispatches to either the
        single-attacker (:meth:`_optimize_spoofing`) or joint
        (:meth:`_optimize_joint`) optimization path, updates the
        intermediate-state buffers, and returns the cooperative
        prediction.

        Parameters
        ----------
        batch_data : Any
            Collated batch.
        model : Any
            Cooperative perception model with the standard
            intermediate-fusion structure (PointPillars or VoxelNet).
        dataset : Any
            OpenCOOD dataset.
        device : torch.device
            Device the model lives on.
        advcp_config : AdvCPConfig
            Resolved AdvCP config.
        memory_data : Optional[AdvCPMemoryData]
            Per-tick memory data; required for any non-fallback path.
        attack_state : Optional[AdvCPIntermediateAttackState]
            Persistent attack state. Created on demand if missing.

        Returns
        -------
        AdvCPAttackResult

        Raises
        ------
        NotImplementedError
            If the AdvCP mode is not one of ``"spoofing"`` or
            ``"removal"``.
        ValueError
            If ``memory_data`` is missing or no attackers are
            configured.
        """
        mode = AdvCPAttackHelper.require_config_value(advcp_config, "mode")
        advcp_context = AdvCPVisualizationContext(mode=mode)

        match mode:
            case "removal" | "spoofing":
                pass
            case _:
                raise NotImplementedError(f"AdvCP mode '{mode}' is not available for intermediate fusion.")
        if memory_data is None:
            raise ValueError(f"AdvCP intermediate {mode} requires current memory data.")

        intermediate_state: AdvCPIntermediateAttackState = attack_state if attack_state is not None else {}
        online_enabled = bool(AdvCPAttackHelper.require_config_value(advcp_config, "online"))
        configured_attacker_ids = AdvCPAttackHelper.resolve_configured_attacker_ids(advcp_config)
        if len(configured_attacker_ids) == 0:
            AdvCPAttackHelper.raise_no_configured_attackers("intermediate")

        AttackResultTuple = tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]

        def stateful(executor: Callable[[], tuple[AttackResultTuple | None, dict[str, list[npt.NDArray]] | None]]) -> Callable[[], AdvCPAttackResult]:
            @wraps(executor)
            def wrapped() -> AdvCPAttackResult:
                attack_result, init_perturbation = executor()
                AdvCoperceptionIntermediateFusionAttack._update_attack_state(
                    intermediate_state,
                    memory_data,
                    init_perturbation=init_perturbation,
                    online=online_enabled,
                )
                if attack_result is None:
                    return (*inference_utils.inference_intermediate_fusion(batch_data, model, dataset), advcp_context)

                pred_box_tensor, pred_score, gt_box_tensor = attack_result
                return pred_box_tensor, pred_score, gt_box_tensor, advcp_context

            return wrapped

        @stateful
        def execute_attack() -> tuple[AttackResultTuple | None, dict[str, list[npt.NDArray]] | None]:
            current_scenario_data = next(iter(memory_data.values()))
            present_attacker_ids, _ = AdvCPAttackHelper.resolve_present_and_missing_attackers(
                configured_attacker_ids,
                current_scenario_data.keys(),
            )
            if not present_attacker_ids:
                AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                    configured_attacker_ids,
                    current_scenario_data.keys(),
                    fusion_name="intermediate",
                )
                return None, None

            sync_enabled = bool(AdvCPAttackHelper.require_config_value(advcp_config, "sync"))
            stored_init_map: dict[str, list[npt.NDArray]] = intermediate_state.get("init_perturbation") or {}

            valid_attacker_infos: list[tuple[str, int, list[npt.NDArray]]] = []
            for attacker_id in present_attacker_ids:
                attacker_index = AdvCoperceptionIntermediateFusionAttack._resolve_attacker_index(batch_data, attacker_id)
                if attacker_index is None:
                    AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                        [attacker_id],
                        AdvCPAttackHelper.resolve_batch_agent_ids(batch_data),
                        fusion_name="intermediate",
                    )
                    continue
                ego_boxes = AdvCoperceptionIntermediateFusionAttack._resolve_ego_attack_boxes(
                    current_scenario_data,
                    advcp_config,
                    attacker_id,
                )
                valid_attacker_infos.append((attacker_id, attacker_index, ego_boxes))

            if not valid_attacker_infos:
                return None, None

            first_attacker_boxes = valid_attacker_infos[0][2]
            current_target_boxes = torch.stack(
                [AdvCPAttackHelper.convert_box_for_model(box, dataset).to(device) for box in first_attacker_boxes],
                dim=0,
            )
            target_corners = box_utils.boxes_to_corners_3d(
                current_target_boxes,
                order=dataset.post_processor.params.get("order", "hwl"),
            )
            match mode:
                case "removal":
                    advcp_context.removed_box_tensor = target_corners
                case "spoofing":
                    advcp_context.fake_box_tensor = target_corners
                case _:
                    raise NotImplementedError(f"AdvCP mode '{mode}' is not available for intermediate fusion.")
            advcp_context.attacker_ids = [aid for aid, _, _ in valid_attacker_infos]

            try:
                if len(valid_attacker_infos) == 1:
                    attacker_id, _, ego_boxes = valid_attacker_infos[0]
                    pred_box, pred_score, gt_box, init_pert = AdvCoperceptionIntermediateFusionAttack._optimize_spoofing(
                        model,
                        dataset,
                        device,
                        advcp_config,
                        attacker_id,
                        memory_data,
                        ego_boxes,
                        intermediate_state.get("previous_memory_data"),
                        sync_enabled,
                        stored_init_map.get(attacker_id),
                        mode=mode,
                    )
                    new_init = {attacker_id: init_pert} if init_pert is not None else {}
                else:
                    pred_box, pred_score, gt_box, new_init = AdvCoperceptionIntermediateFusionAttack._optimize_joint(
                        model,
                        dataset,
                        device,
                        advcp_config,
                        memory_data,
                        valid_attacker_infos,
                        intermediate_state.get("previous_memory_data"),
                        sync_enabled,
                        stored_init_map,
                        mode=mode,
                    )
            finally:
                dataset.update_database(memory_data=memory_data)

            return (pred_box, pred_score, gt_box), new_init or None

        return execute_attack()

    @staticmethod
    def _update_attack_state(
        attack_state: AdvCPIntermediateAttackState,
        memory_data: AdvCPMemoryData,
        init_perturbation: dict[AttackerId, list[npt.NDArray]] | None,
        online: bool,
    ) -> None:
        """
        Advance the persistent attack state by one tick.

        The two reusable buffers are swapped: yesterday's "current"
        becomes today's "previous"; the empty buffer is then
        repopulated from the freshly received ``memory_data``. The
        last best perturbation is stored when ``online`` is enabled,
        otherwise discarded.
        """
        previous_buffer = attack_state.get("previous_memory_data")
        current_buffer = attack_state.get("current_memory_data")
        if previous_buffer is None:
            previous_buffer = OrderedDict()
        if current_buffer is None:
            current_buffer = OrderedDict()

        # Swap the two reusable buffers and overwrite only the current one
        # with a structural copy of fresh tick data.
        previous_buffer, current_buffer = current_buffer, previous_buffer
        AdvCoperceptionIntermediateFusionAttack._refresh_memory_buffer(current_buffer, memory_data)

        attack_state["previous_memory_data"] = previous_buffer if len(previous_buffer) > 0 else None
        attack_state["current_memory_data"] = current_buffer
        attack_state["init_perturbation"] = init_perturbation if online else None

    @staticmethod
    def _refresh_memory_buffer(
        target_buffer: AdvCPMemoryData,
        source_memory_data: AdvCPMemoryData,
    ) -> None:
        """
        Re-populate ``target_buffer`` with the structure of
        ``source_memory_data``, keeping numpy payloads by reference.

        Only the dict / OrderedDict scaffold is recreated; the bulky
        ``lidar_np`` and ``params`` payloads are aliased rather than
        copied, so the buffer is cheap to maintain across ticks.
        """
        target_buffer.clear()
        for batch_idx, source_batch in source_memory_data.items():
            batch_copy: AdvCPScenarioData = OrderedDict()
            target_buffer[batch_idx] = batch_copy
            for agent_id, source_agent_record in source_batch.items():
                agent_record_copy: AdvCPMemoryRecord = OrderedDict()
                batch_copy[agent_id] = agent_record_copy
                for record_key, record_value in source_agent_record.items():
                    if isinstance(record_value, Mapping):
                        agent_record_copy[record_key] = cast(LiveMemorySnapshot, dict(record_value))
                    else:
                        agent_record_copy[record_key] = record_value

    @staticmethod
    def _resolve_ego_attack_boxes(
        scenario_data: AdvCPScenarioData,
        advcp_config: AdvCPConfig,
        attacker_id: AttackerId,
    ) -> list[BoxLwhBottomCenter]:
        """
        Convenience wrapper returning only the ego-frame target boxes.

        Equivalent to discarding the first three elements of
        :meth:`AdvCPAttackHelper.resolve_spoof_boxes_for_ego`.
        """
        _, _, _, attack_boxes = AdvCPAttackHelper.resolve_spoof_boxes_for_ego(
            scenario_data,
            advcp_config,
            attacker_id,
        )
        return attack_boxes

    @staticmethod
    def _resolve_attacker_index(batch_data: Mapping[str, Any], attacker_id: AttackerId) -> int | None:
        """
        Find the attacker's slot in the cooperatively-stacked feature
        tensor.

        OpenCOOD stacks per-CAV spatial features in the order given by
        ``ego.origin_lidar_agent_ids``. To inject a perturbation into
        the right slot we must know that ordering. Returns ``None``
        when the attacker is absent from the batch.
        """
        ego_entry = batch_data.get("ego")
        if not isinstance(ego_entry, Mapping):
            return None

        agent_ids = ego_entry.get("origin_lidar_agent_ids")
        if not isinstance(agent_ids, Sequence):
            return None

        agent_ids_list = [str(agent_id) for agent_id in agent_ids]
        if attacker_id not in agent_ids_list:
            return None
        return agent_ids_list.index(attacker_id)

    @classmethod
    def _apply_init_spoof_to_memory(
        cls,
        memory_data: AdvCPMemoryData,
        advcp_config: AdvCPConfig,
        attacker_id: AttackerId,
    ) -> AdvCPMemoryData:
        """
        Build memory data with the early-fusion spoofing pipeline
        applied to a single attacker.

        Used as an initialization helper by intermediate fusion: when
        ``init: true`` is configured, the attacker's lidar is first
        rewritten by the early-fusion ray-tracing pipeline so the
        backbone produces a "spoofed-looking" feature map; the base
        perturbation is then the difference between that map and the
        unmodified one. Reuses
        :meth:`AdvCoperceptionEarlyFusionAttack._apply_sampled_ray_traced_spoof`
        on a deep copy of the memory.
        """
        attacked_memory = copy.deepcopy(memory_data)
        attacked_scenario_data = next(iter(attacked_memory.values()))
        original_scenario_data = next(iter(memory_data.values()))

        _, _, _, attack_boxes = AdvCPAttackHelper.resolve_spoof_boxes_for_agent(original_scenario_data, advcp_config, attacker_id)
        attacked_snapshot = AdvCPAttackHelper.resolve_agent_snapshot(attacked_scenario_data, attacker_id)
        attacker_lidar = AdvCPAttackHelper.require_agent_lidar(attacked_snapshot, attacker_id, "AdvCP intermediate init")
        lidar_poses = AdvCPAttackHelper.build_lidar_pose_map(original_scenario_data)
        density = AdvCPAttackHelper.resolve_density(AdvCPAttackHelper.require_config_value(advcp_config, "density"))

        spoofed_lidar = np.asarray(attacker_lidar, dtype=np.float32)
        spoofing_mask = np.zeros((spoofed_lidar.shape[0],), dtype=np.bool_)
        for attack_box in attack_boxes:
            spoofed_lidar, spoofing_mask = AdvCoperceptionEarlyFusionAttack._apply_sampled_ray_traced_spoof(
                spoofed_lidar,
                spoofing_mask,
                attack_box,
                lidar_poses,
                attacker_id,
                advcp_config,
                density,
            )

        attacked_snapshot["lidar_np"] = spoofed_lidar
        attacked_snapshot["spoofing_mask"] = spoofing_mask
        return attacked_memory

    @classmethod
    def _apply_init_removal_to_memory(
        cls,
        memory_data: AdvCPMemoryData,
        advcp_config: AdvCPConfig,
        attacker_id: AttackerId,
    ) -> AdvCPMemoryData:
        """
        Build memory data with the early-fusion removal pipeline
        applied to a single attacker.

        Mirror of :meth:`_apply_init_spoof_to_memory` for removal mode.
        Used as warm-start initialization for the intermediate-fusion
        optimizer when ``init: true``.
        """
        attacked_memory = copy.deepcopy(memory_data)
        attacked_scenario_data = next(iter(attacked_memory.values()))
        original_scenario_data = next(iter(memory_data.values()))

        _, _, _, removal_boxes = AdvCPAttackHelper.resolve_spoof_boxes_for_agent(original_scenario_data, advcp_config, attacker_id)
        attacked_snapshot = AdvCPAttackHelper.resolve_agent_snapshot(attacked_scenario_data, attacker_id)
        attacker_lidar = AdvCPAttackHelper.require_agent_lidar(attacked_snapshot, attacker_id, "AdvCP intermediate removal init")
        lidar_poses = AdvCPAttackHelper.build_lidar_pose_map(original_scenario_data)
        density = AdvCPAttackHelper.resolve_density(AdvCPAttackHelper.require_config_value(advcp_config, "density"))
        advshape_enabled = AdvCoperceptionEarlyFusionAttack._resolve_advshape_enabled(advcp_config)

        removed_lidar = np.asarray(attacker_lidar, dtype=np.float32)
        for removal_box in removal_boxes:
            removed_lidar = AdvCoperceptionEarlyFusionAttack._apply_sampled_ray_traced_remove(
                removed_lidar,
                removal_box,
                lidar_poses,
                attacker_id,
                advcp_config,
                density,
                advshape_enabled,
            )

        attacked_snapshot["lidar_np"] = removed_lidar
        attacked_snapshot.pop("spoofing_mask", None)
        return attacked_memory

    @classmethod
    def _optimize_spoofing(
        cls,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        attacker_id: AttackerId,
        memory_data: AdvCPMemoryData,
        current_attack_boxes: Sequence[BoxLwhBottomCenter],
        previous_memory_data: AdvCPMemoryData | None,
        sync_enabled: bool,
        stored_init_perturbation: list[npt.NDArray] | None,
        mode: str = "spoofing",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[npt.NDArray] | None]:
        """
        Optimize a single-attacker perturbation against the cooperative
        model.

        Adam loop over ``optimization_steps`` iterations. Each step:

        1. Forward-pass the model with the current ``base_perturbation
           + learnable_perturbation`` applied to the attacker's slot
           in the spatial feature tensor.
        2. Compute the spoofing or removal loss (see module docstring).
        3. Track the best (lowest loss) prediction so far, since Adam
           updates after each step are not monotone.
        4. Backpropagate and update the perturbation tensors.

        When ``sync_enabled`` and the previous tick contained the
        attacker, the optimization runs on the **previous** tick and
        the resulting perturbation is evaluated on the **current**
        tick. This decouples the optimization from the current tick's
        possibly noisy spatial features.

        Parameters
        ----------
        model, dataset, device : Any
            Cooperative perception machinery.
        advcp_config : AdvCPConfig
            Resolved AdvCP config (provides ``step``, ``lr``,
            ``max_perturb``, ``feature_size``, ``init``).
        attacker_id : AttackerId
        memory_data : AdvCPMemoryData
            Current-tick memory.
        current_attack_boxes : Sequence of BoxLwhBottomCenter
            Target boxes in the ego lidar frame.
        previous_memory_data : Optional[AdvCPMemoryData]
            Previous-tick memory (for sync mode).
        sync_enabled : bool
        stored_init_perturbation : Optional[list of npt.NDArray]
            Warm-start perturbation from the previous tick (online
            mode).
        mode : {"spoofing", "removal"}

        Returns
        -------
        tuple
            ``(pred_box_tensor, pred_score, gt_box_tensor, init_perturbation_for_next_tick)``.
            The fourth element is ``None`` when no improvement was
            found.
        """
        max_perturb = float(AdvCPAttackHelper.require_config_value(advcp_config, "max_perturb"))
        learning_rate = float(AdvCPAttackHelper.require_config_value(advcp_config, "lr"))
        optimization_steps = int(AdvCPAttackHelper.require_config_value(advcp_config, "step"))
        feature_size = int(AdvCPAttackHelper.require_config_value(advcp_config, "feature_size"))
        use_init = bool(AdvCPAttackHelper.require_config_value(advcp_config, "init"))
        init_memory_fn = cls._apply_init_spoof_to_memory if mode == "spoofing" else cls._apply_init_removal_to_memory
        compute_loss_fn = cls._compute_spoof_loss if mode == "spoofing" else cls._compute_removal_loss

        optimize_memory_data = memory_data
        optimize_attack_boxes = current_attack_boxes
        real_memory_data: AdvCPMemoryData | None = None
        real_attack_boxes: Sequence[npt.NDArray] | None = None

        if sync_enabled and previous_memory_data is not None:
            previous_scenario_data = next(iter(previous_memory_data.values()))
            if attacker_id in previous_scenario_data:
                optimize_memory_data = previous_memory_data
                optimize_attack_boxes = cls._resolve_ego_attack_boxes(
                    previous_scenario_data,
                    advcp_config,
                    attacker_id,
                )
                real_memory_data = memory_data
                real_attack_boxes = current_attack_boxes
            else:
                logger.warning(
                    "AdvCP intermediate previous-tick optimization skipped previous tick because attacker '%s' was not present. "
                    "Falling back to current-tick optimization.",
                    attacker_id,
                )

        original_optimize_batch = AdvCPAttackHelper.build_batch_from_memory(dataset, device, optimize_memory_data)
        optimize_batch = (
            AdvCPAttackHelper.build_batch_from_memory(dataset, device, init_memory_fn(optimize_memory_data, advcp_config, attacker_id))
            if use_init
            else original_optimize_batch
        )

        optimize_attacker_index = cls._resolve_attacker_index(optimize_batch, attacker_id)
        if optimize_attacker_index is None:
            AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                [attacker_id],
                AdvCPAttackHelper.resolve_batch_agent_ids(optimize_batch),
                fusion_name="intermediate",
            )
            return (*inference_utils.inference_intermediate_fusion(original_optimize_batch, model, dataset), None)

        real_original_batch: Mapping[str, Any] | None = None
        real_batch: Mapping[str, Any] | None = None
        real_attacker_index: int | None = None
        if real_memory_data is not None and real_attack_boxes is not None:
            real_original_batch = AdvCPAttackHelper.build_batch_from_memory(dataset, device, real_memory_data)
            real_batch = (
                AdvCPAttackHelper.build_batch_from_memory(dataset, device, init_memory_fn(real_memory_data, advcp_config, attacker_id))
                if use_init
                else real_original_batch
            )
            real_attacker_index = cls._resolve_attacker_index(real_batch, attacker_id)
            if real_attacker_index is None:
                AdvCPAttackHelper.report_missing_attackers_from_current_batch(
                    [attacker_id],
                    AdvCPAttackHelper.resolve_batch_agent_ids(real_batch),
                    fusion_name="intermediate",
                )
                return (*inference_utils.inference_intermediate_fusion(real_original_batch, model, dataset), None)

        with torch.no_grad():
            _, optimize_spatial_features = cls._attack_forward(
                optimize_batch,
                model,
                optimize_attacker_index,
                perturbations=None,
                centers=None,
                max_perturb=max_perturb,
            )
            _, original_spatial_features = cls._attack_forward(
                original_optimize_batch,
                model,
                optimize_attacker_index,
                perturbations=None,
                centers=None,
                max_perturb=max_perturb,
            )

            real_spatial_features: torch.Tensor | None = None
            real_original_spatial_features: torch.Tensor | None = None
            if real_batch is not None and real_original_batch is not None and real_attacker_index is not None:
                _, real_spatial_features = cls._attack_forward(
                    real_batch,
                    model,
                    real_attacker_index,
                    perturbations=None,
                    centers=None,
                    max_perturb=max_perturb,
                )
                _, real_original_spatial_features = cls._attack_forward(
                    real_original_batch,
                    model,
                    real_attacker_index,
                    perturbations=None,
                    centers=None,
                    max_perturb=max_perturb,
                )

        feature_dim = int(optimize_spatial_features[optimize_attacker_index].shape[0])
        optimize_centers = [cls._point_to_feature_index(attack_box, dataset) for attack_box in optimize_attack_boxes]
        real_centers = [cls._point_to_feature_index(attack_box, dataset) for attack_box in (real_attack_boxes or optimize_attack_boxes)]
        base_perturbations = cls._extract_base_perturbations(
            optimize_spatial_features[optimize_attacker_index],
            original_spatial_features[optimize_attacker_index],
            optimize_centers,
            feature_size,
        )
        real_base_perturbations = (
            cls._extract_base_perturbations(
                real_spatial_features[real_attacker_index],
                real_original_spatial_features[real_attacker_index],
                real_centers,
                feature_size,
            )
            if real_spatial_features is not None and real_original_spatial_features is not None and real_attacker_index is not None
            else [torch.zeros((feature_dim, 2 * feature_size, 2 * feature_size), device=optimize_spatial_features.device) for _ in optimize_centers]
        )

        perturbations = cls._initialize_perturbations(
            feature_dim,
            feature_size,
            optimize_spatial_features.device,
            stored_init_perturbation,
            len(optimize_centers),
        )
        optimizer = torch.optim.Adam(perturbations, lr=learning_rate)

        optimize_target_boxes = torch.stack(
            [
                AdvCPAttackHelper.convert_box_for_model(attack_box, dataset).to(optimize_spatial_features.device)
                for attack_box in optimize_attack_boxes
            ],
            dim=0,
        )
        best_loss = float("inf")
        best_pred_box_tensor: torch.Tensor | None = None
        best_pred_score: torch.Tensor | None = None
        best_gt_box_tensor: torch.Tensor | None = None
        best_init_perturbation: list[npt.NDArray] | None = None

        for _ in range(optimization_steps):
            # Forward pass on the optimization batch with the current
            # perturbations applied. The base perturbation captures the
            # init-mode warm start; the learnable tensor is the actual
            # gradient target.
            output_dict, _ = cls._attack_forward(
                original_optimize_batch,
                model,
                optimize_attacker_index,
                perturbations=[base_perturbation + perturbation for base_perturbation, perturbation in zip(base_perturbations, perturbations)],
                centers=optimize_centers,
                max_perturb=max_perturb,
            )
            loss = compute_loss_fn(output_dict, original_optimize_batch, dataset, optimize_target_boxes)

            # Evaluate the current perturbation on the **real** (current
            # tick) batch when sync mode optimized against the previous
            # tick. The post-processed prediction is the artifact we
            # actually return.
            with torch.no_grad():
                if real_original_batch is not None and real_attacker_index is not None:
                    eval_output_dict, _ = cls._attack_forward(
                        real_original_batch,
                        model,
                        real_attacker_index,
                        perturbations=[
                            real_base_perturbation + perturbation
                            for real_base_perturbation, perturbation in zip(real_base_perturbations, perturbations)
                        ],
                        centers=real_centers,
                        max_perturb=max_perturb,
                    )
                    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(real_original_batch, eval_output_dict)
                else:
                    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(original_optimize_batch, output_dict)

            # Adam updates are not monotone in the loss, so we keep the
            # snapshot of the best step so far. Halving on save lets the
            # next tick's warm start sit closer to the unperturbed
            # tensor (helps convergence when scenes change slowly).
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_pred_box_tensor = pred_box_tensor
                best_pred_score = pred_score
                best_gt_box_tensor = gt_box_tensor
                best_init_perturbation = [
                    torch.clamp(perturbation.detach(), min=-max_perturb, max=max_perturb).cpu().numpy() / 2.0 for perturbation in perturbations
                ]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if best_gt_box_tensor is None:
            inference_batch = real_original_batch if real_original_batch is not None else original_optimize_batch
            return (*inference_utils.inference_intermediate_fusion(inference_batch, model, dataset), None)

        return best_pred_box_tensor, best_pred_score, best_gt_box_tensor, best_init_perturbation

    @classmethod
    def _optimize_joint(
        cls,
        model: Any,
        dataset: Any,
        device: torch.device,
        advcp_config: AdvCPConfig,
        memory_data: AdvCPMemoryData,
        attacker_infos: list[tuple[AttackerId, int, list[BoxLwhBottomCenter]]],
        previous_memory_data: AdvCPMemoryData | None,
        sync_enabled: bool,
        stored_init_map: dict[AttackerId, list[npt.NDArray]],
        mode: str = "spoofing",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[AttackerId, list[npt.NDArray]]]:
        """
        Optimize multiple attackers' perturbations jointly.

        Identical structure to :meth:`_optimize_spoofing` but with one
        Adam optimizer holding all attackers' learnable tensors and a
        single combined forward pass per step. This is required because
        the cooperative backbone fuses the attackers' features
        together: independently optimized perturbations would each
        assume the other attackers' features are unchanged, which is
        not what happens at evaluation time.

        Parameters
        ----------
        attacker_infos : list of tuple
            ``(attacker_id, attacker_index, ego_attack_boxes)`` for
            every present attacker. The ``attacker_index`` is the
            slot in the cooperative spatial-feature tensor.
        stored_init_map : dict
            Per-attacker warm starts from the previous tick.

        Returns
        -------
        tuple
            ``(pred_box_tensor, pred_score, gt_box_tensor, init_perturbation_map)``
            where the last element maps attacker id to its current
            best perturbation (for next-tick reuse).
        """
        max_perturb = float(AdvCPAttackHelper.require_config_value(advcp_config, "max_perturb"))
        learning_rate = float(AdvCPAttackHelper.require_config_value(advcp_config, "lr"))
        optimization_steps = int(AdvCPAttackHelper.require_config_value(advcp_config, "step"))
        feature_size = int(AdvCPAttackHelper.require_config_value(advcp_config, "feature_size"))
        use_init = bool(AdvCPAttackHelper.require_config_value(advcp_config, "init"))
        init_memory_fn = cls._apply_init_spoof_to_memory if mode == "spoofing" else cls._apply_init_removal_to_memory
        compute_loss_fn = cls._compute_spoof_loss if mode == "spoofing" else cls._compute_removal_loss

        use_prev_for_opt = False
        optimize_memory = memory_data
        if sync_enabled and previous_memory_data is not None:
            prev_scenario = next(iter(previous_memory_data.values()))
            if all(aid in prev_scenario for aid, _, _ in attacker_infos):
                optimize_memory = previous_memory_data
                use_prev_for_opt = True
            else:
                logger.warning(
                    "AdvCP joint: not all attackers present in previous tick; falling back to current-tick optimization.",
                )

        optimize_scenario = next(iter(optimize_memory.values()))
        original_optimize_batch = AdvCPAttackHelper.build_batch_from_memory(dataset, device, optimize_memory)
        real_original_batch = AdvCPAttackHelper.build_batch_from_memory(dataset, device, memory_data) if use_prev_for_opt else None

        per_attacker: list[dict] = []
        all_learnable_perturbations: list[torch.Tensor] = []

        for attacker_id, _, current_boxes in attacker_infos:
            optimize_idx = cls._resolve_attacker_index(original_optimize_batch, attacker_id)
            if optimize_idx is None:
                logger.warning("AdvCP joint: attacker '%s' not found in optimize batch, skipping.", attacker_id)
                continue

            optimize_boxes = cls._resolve_ego_attack_boxes(optimize_scenario, advcp_config, attacker_id) if use_prev_for_opt else current_boxes
            optimize_centers = [cls._point_to_feature_index(b, dataset) for b in optimize_boxes]

            with torch.no_grad():
                if use_init:
                    init_opt_batch = AdvCPAttackHelper.build_batch_from_memory(
                        dataset, device, init_memory_fn(optimize_memory, advcp_config, attacker_id)
                    )
                    _, init_feats = cls._attack_forward(init_opt_batch, model, optimize_idx, None, None, max_perturb)
                else:
                    _, init_feats = cls._attack_forward(original_optimize_batch, model, optimize_idx, None, None, max_perturb)
                _, orig_feats = cls._attack_forward(original_optimize_batch, model, optimize_idx, None, None, max_perturb)

            base_perts = cls._extract_base_perturbations(
                init_feats[optimize_idx],
                orig_feats[optimize_idx],
                optimize_centers,
                feature_size,
            )
            feature_dim = int(orig_feats[optimize_idx].shape[0])
            learnable_perts = cls._initialize_perturbations(
                feature_dim,
                feature_size,
                device,
                stored_init_map.get(attacker_id),
                len(optimize_centers),
            )
            all_learnable_perturbations.extend(learnable_perts)

            if use_prev_for_opt and real_original_batch is not None:
                real_idx = cls._resolve_attacker_index(real_original_batch, attacker_id)
                if real_idx is not None:
                    real_centers = [cls._point_to_feature_index(b, dataset) for b in current_boxes]
                    with torch.no_grad():
                        if use_init:
                            real_init_batch = AdvCPAttackHelper.build_batch_from_memory(
                                dataset, device, init_memory_fn(memory_data, advcp_config, attacker_id)
                            )
                            _, real_init_feats = cls._attack_forward(real_init_batch, model, real_idx, None, None, max_perturb)
                        else:
                            _, real_init_feats = cls._attack_forward(real_original_batch, model, real_idx, None, None, max_perturb)
                        _, real_orig_feats = cls._attack_forward(real_original_batch, model, real_idx, None, None, max_perturb)
                    real_base_perts = cls._extract_base_perturbations(
                        real_init_feats[real_idx],
                        real_orig_feats[real_idx],
                        real_centers,
                        feature_size,
                    )
                else:
                    real_idx = optimize_idx
                    real_centers = optimize_centers
                    real_base_perts = base_perts
            else:
                real_idx = optimize_idx
                real_centers = optimize_centers
                real_base_perts = base_perts

            per_attacker.append(
                {
                    "attacker_id": attacker_id,
                    "optimize_idx": optimize_idx,
                    "optimize_centers": optimize_centers,
                    "base_perts": base_perts,
                    "learnable_perts": learnable_perts,
                    "real_idx": real_idx,
                    "real_centers": real_centers,
                    "real_base_perts": real_base_perts,
                    "optimize_boxes": optimize_boxes,
                }
            )

        if not per_attacker:
            return (*inference_utils.inference_intermediate_fusion(original_optimize_batch, model, dataset), {})

        optimize_target_boxes = torch.stack(
            [AdvCPAttackHelper.convert_box_for_model(b, dataset).to(device) for b in per_attacker[0]["optimize_boxes"]],
            dim=0,
        )

        optimizer = torch.optim.Adam(all_learnable_perturbations, lr=learning_rate)
        best_loss = float("inf")
        best_pred_box: torch.Tensor | None = None
        best_pred_score: torch.Tensor | None = None
        best_gt_box: torch.Tensor | None = None
        best_init_map: dict[str, list[npt.NDArray]] = {}

        for _ in range(optimization_steps):
            # Build the combined per-attacker spec list. The forward
            # helper applies each attacker's perturbation to its slot
            # in the shared spatial feature tensor, so the loss sees
            # the joint effect of all attackers and gradients flow into
            # all of their learnable tensors at once.
            optimize_specs: list[tuple[int, list[torch.Tensor], list[npt.NDArray]]] = [
                (d["optimize_idx"], [b + p for b, p in zip(d["base_perts"], d["learnable_perts"])], d["optimize_centers"]) for d in per_attacker
            ]
            output_dict, _ = cls._attack_forward(
                original_optimize_batch,
                model,
                attacker_index=per_attacker[0]["optimize_idx"],
                perturbations=None,
                centers=None,
                max_perturb=max_perturb,
                all_attacker_specs=optimize_specs,
            )
            loss = compute_loss_fn(output_dict, original_optimize_batch, dataset, optimize_target_boxes)

            # Sync-mode evaluation: same trick as _optimize_spoofing,
            # but run as a combined forward pass on the current tick
            # with the same set of perturbations.
            with torch.no_grad():
                if real_original_batch is not None:
                    real_specs: list[tuple[int, list[torch.Tensor], list[npt.NDArray]]] = [
                        (d["real_idx"], [b + p for b, p in zip(d["real_base_perts"], d["learnable_perts"])], d["real_centers"]) for d in per_attacker
                    ]
                    eval_output, _ = cls._attack_forward(
                        real_original_batch,
                        model,
                        attacker_index=per_attacker[0]["real_idx"],
                        perturbations=None,
                        centers=None,
                        max_perturb=max_perturb,
                        all_attacker_specs=real_specs,
                    )
                    pred_box, pred_score, gt_box = dataset.post_process(real_original_batch, eval_output)
                else:
                    pred_box, pred_score, gt_box = dataset.post_process(original_optimize_batch, output_dict)

            # Track best snapshot across all attackers simultaneously,
            # so the warm-start map for the next tick is internally
            # consistent.
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_pred_box, best_pred_score, best_gt_box = pred_box, pred_score, gt_box
                for d in per_attacker:
                    best_init_map[d["attacker_id"]] = [
                        torch.clamp(p.detach(), min=-max_perturb, max=max_perturb).cpu().numpy() / 2.0 for p in d["learnable_perts"]
                    ]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if best_gt_box is None:
            fallback = real_original_batch if real_original_batch is not None else original_optimize_batch
            return (*inference_utils.inference_intermediate_fusion(fallback, model, dataset), {})

        return best_pred_box, best_pred_score, best_gt_box, best_init_map

    @staticmethod
    def _initialize_perturbations(
        feature_dim: int,
        feature_size: int,
        device: torch.device | str,
        stored_init_perturbation: list[npt.NDArray] | None,
        expected_count: int,
    ) -> list[torch.Tensor]:
        """
        Allocate one learnable perturbation tensor per target box.

        Each tensor has shape ``(feature_dim, 2 * feature_size, 2 * feature_size)``
        and is initialised either from the stored numpy warm start
        (when available and within range) or to zeros. The returned
        tensors have ``requires_grad = True`` and are ready to be
        passed to ``torch.optim.Adam``.
        """
        perturbations: list[torch.Tensor] = []
        for index in range(expected_count):
            if stored_init_perturbation is not None and index < len(stored_init_perturbation):
                perturbation = torch.from_numpy(stored_init_perturbation[index]).to(device=device, dtype=torch.float32)
            else:
                perturbation = torch.zeros((feature_dim, 2 * feature_size, 2 * feature_size), device=device, dtype=torch.float32)
            perturbation.requires_grad = True
            perturbations.append(perturbation)
        return perturbations

    @staticmethod
    def _extract_base_perturbations(
        spoofed_features: torch.Tensor,
        original_features: torch.Tensor,
        centers: Sequence[npt.NDArray],
        feature_size: int,
    ) -> list[torch.Tensor]:
        """
        Compute the per-target base perturbation patches.

        The base perturbation is the spatial difference between the
        feature map produced from the warm-started lidar (post
        early-fusion ray tracing) and the original lidar, restricted
        to a square patch around each target's pixel coordinates. It
        encodes "what the spatial features would look like if the
        spoof object were really there", which is then the starting
        signal that Adam refines.

        Returns
        -------
        list of torch.Tensor
            One detached patch per target.
        """
        base_perturbations = []
        for center in centers:
            patch = AdvCoperceptionIntermediateFusionAttack._extract_feature_patch(
                spoofed_features - original_features,
                center,
                feature_size,
            )
            base_perturbations.append(patch.detach())
        return base_perturbations

    @staticmethod
    def _extract_feature_patch(features: torch.Tensor, center: npt.NDArray, feature_size: int) -> torch.Tensor:
        """
        Crop a square patch from a feature tensor with zero padding.

        The patch is ``2 * feature_size`` pixels on each side and
        centred on ``center`` (rounded to integers). When the patch
        spills past the feature-tensor borders, the out-of-bounds
        portion is zero-filled.

        Parameters
        ----------
        features : torch.Tensor
            ``(C, H, W)`` feature tensor.
        center : npt.NDArray
            Pixel-coordinate centre ``(x, y)``.
        feature_size : int
            Half side length in pixels.

        Returns
        -------
        torch.Tensor
            ``(C, 2 * feature_size, 2 * feature_size)`` patch.
        """
        _, height, width = features.shape
        center_x = int(center[0])
        center_y = int(center[1])
        patch_height = 2 * feature_size
        patch_width = 2 * feature_size
        patch = torch.zeros((features.shape[0], patch_height, patch_width), device=features.device, dtype=features.dtype)

        # Compute the intersection of [center - half, center + half]
        # with [0, dim) for both axes; the corresponding source slice
        # is copied into the same position inside the zero-filled
        # patch.
        x_start = max(center_x - feature_size, 0)
        x_end = min(center_x + feature_size, width)
        y_start = max(center_y - feature_size, 0)
        y_end = min(center_y + feature_size, height)

        patch_x_start = x_start - (center_x - feature_size)
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_start = y_start - (center_y - feature_size)
        patch_y_end = patch_y_start + (y_end - y_start)

        patch[:, patch_y_start:patch_y_end, patch_x_start:patch_x_end] = features[:, y_start:y_end, x_start:x_end]
        return patch

    @classmethod
    def _attack_forward(
        cls,
        batch_data: Mapping[str, Any],
        model: Any,
        attacker_index: int,
        perturbations: Sequence[torch.Tensor] | None,
        centers: Sequence[npt.NDArray] | None,
        max_perturb: float,
        all_attacker_specs: list[tuple[int, list[torch.Tensor], list[npt.NDArray]]] | None = None,
    ) -> tuple[OrderedDict[str, dict[str, torch.Tensor]], torch.Tensor]:
        """
        Run a forward pass with optional perturbations applied to the
        spatial feature tensor.

        Dispatches by model class to the PointPillars or VoxelNet
        variant. Both variants reproduce the model's standard
        backbone pipeline up to (and including) the spatial feature
        tensor, then either:

        - apply a single attacker's perturbation set (single-attacker
          path: ``perturbations`` and ``centers`` are non-None), or
        - apply all attackers' perturbations in one go (joint path:
          ``all_attacker_specs`` is non-None).

        After the perturbation is applied, the rest of the head is run
        normally to produce the cooperative output.

        Returns
        -------
        tuple
            ``(output_dict, spatial_features)``. The spatial features
            are returned (post-perturbation) so callers can use them
            as the warm-start signal for the next tick.
        """
        model_name = type(model).__name__
        if model_name == "VoxelNetIntermediate":
            return cls._attack_forward_voxelnet(batch_data, model, attacker_index, perturbations, centers, max_perturb, all_attacker_specs)
        return cls._attack_forward_point_pillar(batch_data, model, attacker_index, perturbations, centers, max_perturb, all_attacker_specs)

    @classmethod
    def _attack_forward_point_pillar(
        cls,
        batch_data: Mapping[str, Any],
        model: Any,
        attacker_index: int,
        perturbations: Sequence[torch.Tensor] | None,
        centers: Sequence[npt.NDArray] | None,
        max_perturb: float,
        all_attacker_specs: list[tuple[int, list[torch.Tensor], list[npt.NDArray]]] | None = None,
    ) -> tuple[OrderedDict[str, dict[str, torch.Tensor]], torch.Tensor]:
        """
        PointPillars-family forward pass with perturbation injection.

        Reproduces the standard PointPillars backbone (PFE +
        scatter), inserts the perturbation into the resulting spatial
        feature tensor, and then dispatches to the appropriate fusion
        head variant via :meth:`_run_point_pillar_head`.
        """
        ego_entry = batch_data["ego"]
        processed_lidar = ego_entry["processed_lidar"]
        record_len = ego_entry["record_len"]
        pairwise_t_matrix = ego_entry.get("pairwise_t_matrix")

        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "record_len": record_len,
        }
        if "PointPillarintermediateV2VAM" in type(model).__name__:
            batch_dict["voxel_features"] = batch_dict["voxel_features"].float()

        batch_dict = model.pillar_vfe(batch_dict)
        batch_dict = model.scatter(batch_dict)
        if all_attacker_specs is not None:
            spatial_features = batch_dict["spatial_features"]
            for idx, perts, cents in all_attacker_specs:
                spatial_features = cls._apply_perturbations_to_attacker_features(spatial_features, idx, perts, cents, max_perturb)
        else:
            spatial_features = cls._apply_perturbations_to_attacker_features(
                batch_dict["spatial_features"],
                attacker_index,
                perturbations,
                centers,
                max_perturb,
            )
        batch_dict["spatial_features"] = spatial_features
        output = cls._run_point_pillar_head(model, batch_dict, ego_entry, record_len, pairwise_t_matrix)
        return OrderedDict(ego=output), spatial_features

    @classmethod
    def _attack_forward_voxelnet(
        cls,
        batch_data: Mapping[str, Any],
        model: Any,
        attacker_index: int,
        perturbations: Sequence[torch.Tensor] | None,
        centers: Sequence[npt.NDArray] | None,
        max_perturb: float,
        all_attacker_specs: list[tuple[int, list[torch.Tensor], list[npt.NDArray]]] | None = None,
    ) -> tuple[OrderedDict[str, dict[str, torch.Tensor]], torch.Tensor]:
        """
        VoxelNet forward pass with perturbation injection.

        Reproduces the SVFE + voxel-indexing + CML backbone of
        ``VoxelNetIntermediate`` up to the 4D-to-2D reshape, applies
        the perturbations there, then runs fusion + RPN heads
        normally.
        """
        ego_entry = batch_data["ego"]
        processed_lidar = ego_entry["processed_lidar"]
        record_len = ego_entry["record_len"]
        voxel_coords = processed_lidar["voxel_coords"]

        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": voxel_coords,
            "voxel_num_points": processed_lidar["voxel_num_points"],
        }
        record_len_tmp = record_len.cpu() if getattr(voxel_coords, "is_cuda", False) else record_len
        model.N = int(sum(record_len_tmp.detach().cpu().numpy().tolist()))

        voxelwise_features = model.svfe(batch_dict)["pillar_features"]
        voxel_coords_np = voxel_coords.detach().cpu().numpy()
        voxelwise_features = model.voxel_indexing(voxelwise_features, voxel_coords_np)
        voxelwise_features = model.cml(voxelwise_features)
        spatial_features = voxelwise_features.view(model.N, -1, model.H, model.W)
        if getattr(model, "compression", False):
            spatial_features = model.compression_layer(spatial_features)

        if all_attacker_specs is not None:
            for idx, perts, cents in all_attacker_specs:
                spatial_features = cls._apply_perturbations_to_attacker_features(spatial_features, idx, perts, cents, max_perturb)
        else:
            spatial_features = cls._apply_perturbations_to_attacker_features(
                spatial_features,
                attacker_index,
                perturbations,
                centers,
                max_perturb,
            )
        fused_features = model.fusion_net(spatial_features, record_len)
        psm, rm = model.rpn(fused_features)
        return OrderedDict(ego={"psm": psm, "rm": rm}), spatial_features

    @staticmethod
    def _run_point_pillar_head(
        model: Any,
        batch_dict: dict[str, Any],
        ego_entry: Mapping[str, Any],
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """
        Dispatch the post-spatial-features stages of PointPillars
        cooperative variants.

        Each supported model class wires its fusion / compression /
        head stages slightly differently. This helper applies the
        right sequence based on ``type(model).__name__`` and returns
        the ``{"psm", "rm"}`` head outputs.

        Raises
        ------
        NotImplementedError
            If the model class is not one of the supported families.
        """
        model_name = type(model).__name__
        batch_dict = model.backbone(batch_dict)
        spatial_features_2d = batch_dict["spatial_features_2d"]

        if getattr(model, "shrink_flag", False):
            spatial_features_2d = model.shrink_conv(spatial_features_2d)

        if model_name == "PointPillarWhere2comm":
            psm_single = model.cls_head(spatial_features_2d)
            if getattr(model, "compression", False):
                spatial_features_2d = model.naive_compressor(spatial_features_2d)
            if model.multi_scale:
                fused_feature, _ = model.fusion_net(batch_dict["spatial_features"], psm_single, record_len, pairwise_t_matrix, model.backbone)
                if getattr(model, "shrink_flag", False):
                    fused_feature = model.shrink_conv(fused_feature)
            else:
                fused_feature, _ = model.fusion_net(spatial_features_2d, psm_single, record_len, pairwise_t_matrix)
            return {"psm": model.cls_head(fused_feature), "rm": model.reg_head(fused_feature)}

        if getattr(model, "compression", False):
            spatial_features_2d = model.naive_compressor(spatial_features_2d)

        if model_name == "PointPillarV2VNet":
            fused_feature = model.fusion_net(spatial_features_2d, record_len, pairwise_t_matrix)
            return {"psm": model.cls_head(fused_feature), "rm": model.reg_head(fused_feature)}

        if model_name in {"PointPillarFCooper", "PointPillarintermediateV2VAM"}:
            fused_feature = model.fusion_net(spatial_features_2d, record_len)
            return {"psm": model.cls_head(fused_feature), "rm": model.reg_head(fused_feature)}

        if model_name in {"PointPillarIntermediate", "PointPillarIntermediateV2", "PointPillarCoAlign"}:
            return {"psm": model.cls_head(spatial_features_2d), "rm": model.reg_head(spatial_features_2d)}

        raise NotImplementedError(
            f"AdvCP intermediate spoofing is not implemented for model '{model_name}'. "
            "Supported models are point-pillar intermediate, V2VNet, F-Cooper/V2VAM, Where2comm, and VoxelNet."
        )

    @classmethod
    def _apply_perturbations_to_attacker_features(
        cls,
        spatial_features: torch.Tensor,
        attacker_index: int,
        perturbations: Sequence[torch.Tensor] | None,
        centers: Sequence[npt.NDArray] | None,
        max_perturb: float,
    ) -> torch.Tensor:
        """
        Add a list of perturbation patches to one CAV's slot in the
        spatial features tensor.

        Each perturbation is clamped to the L-infinity ball of radius
        ``max_perturb`` before being summed in via
        :meth:`_build_perturbation_feature_map`. Falls through and
        returns the input tensor unchanged when no perturbations are
        provided (used when the same forward helper handles both
        clean and attacked passes).

        Returns
        -------
        torch.Tensor
            Same shape as ``spatial_features``.
        """
        if not perturbations or not centers:
            return spatial_features

        attacked_features = spatial_features.clone()
        attacker_features = attacked_features[attacker_index]
        for perturbation, center in zip(perturbations, centers):
            clipped_perturbation = torch.clamp(perturbation, min=-max_perturb, max=max_perturb)
            attacker_features = attacker_features + cls._build_perturbation_feature_map(attacker_features, clipped_perturbation, center)
        attacked_features[attacker_index] = attacker_features
        return attacked_features

    @staticmethod
    def _build_perturbation_feature_map(
        attacker_features: torch.Tensor,
        perturbation: torch.Tensor,
        center: npt.NDArray,
    ) -> torch.Tensor:
        """
        Place a perturbation patch into a full-size feature map with
        sub-pixel registration.

        The perturbation tensor is anchored at integer pixel
        coordinates (the floor of ``center``), then resampled with a
        ``grid_sample`` affine grid to apply the sub-pixel offset
        ``center - floor(center)``. This avoids quantising the target
        location to a coarse 2D grid.

        Returns
        -------
        torch.Tensor
            Same shape as ``attacker_features``; zero everywhere except
            in the patch region.
        """
        channels, height, width = attacker_features.shape
        center_x = float(center[0])
        center_y = float(center[1])
        aligned_center_x = int(np.floor(center_x))
        aligned_center_y = int(np.floor(center_y))

        perturbation_map = torch.zeros_like(attacker_features)
        half_patch_height = perturbation.shape[1] // 2
        half_patch_width = perturbation.shape[2] // 2
        x_start = max(aligned_center_x - half_patch_width, 0)
        x_end = min(aligned_center_x + half_patch_width, width)
        y_start = max(aligned_center_y - half_patch_height, 0)
        y_end = min(aligned_center_y + half_patch_height, height)

        patch_x_start = x_start - (aligned_center_x - half_patch_width)
        patch_x_end = patch_x_start + (x_end - x_start)
        patch_y_start = y_start - (aligned_center_y - half_patch_height)
        patch_y_end = patch_y_start + (y_end - y_start)
        perturbation_map[:, y_start:y_end, x_start:x_end] = perturbation[:, patch_y_start:patch_y_end, patch_x_start:patch_x_end]

        theta = torch.tensor(
            [
                [
                    [1.0, 0.0, (center_y - aligned_center_y) * 2.0 / max(width, 1)],
                    [0.0, 1.0, (center_x - aligned_center_x) * 2.0 / max(height, 1)],
                ]
            ],
            dtype=attacker_features.dtype,
            device=attacker_features.device,
        )
        grid = F.affine_grid(theta, [1, channels, height, width], align_corners=False)
        return F.grid_sample(perturbation_map.unsqueeze(0), grid, align_corners=False)[0]

    @staticmethod
    def _point_to_feature_index(attack_box: BoxLwhBottomCenter, dataset: Any) -> npt.NDArray:
        """
        Convert a target box's world centre into pixel coordinates in
        the spatial feature map.

        The OpenCOOD voxel grid spans ``cav_lidar_range`` and uses the
        configured ``voxel_size``. The pixel index of the box centre
        is therefore ``floor((centre - range_origin) / voxel_size)``.

        Returns
        -------
        npt.NDArray
            Length-3 ``int32`` array (typically only the first two
            components are used as ``(x, y)`` pixel coordinates).
        """
        lidar_range = np.asarray(dataset.pre_processor.params["cav_lidar_range"][:3], dtype=np.float32)
        voxel_size = np.asarray(dataset.pre_processor.params["args"]["voxel_size"], dtype=np.float32)
        return np.floor((attack_box[:3] - lidar_range) / voxel_size).astype(np.int32)

    @classmethod
    def _compute_spoof_loss(
        cls,
        output_dict: Mapping[str, Any],
        batch_data: Mapping[str, Any],
        dataset: Any,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Spoofing loss: encourage high confidence at every target box.

        Computes the model's sigmoided proposal scores and the
        per-proposal IoU with each target box. Proposals with
        IoU >= 0.01 contribute a term ``iou * log(1 - score)``; the
        sum is the loss to **minimise**, which pushes scores toward 1
        for proposals overlapping the targets.

        Returns
        -------
        torch.Tensor
            Zero-dim loss tensor. Returns a zero tensor that still
            tracks gradients when no proposal overlaps any target
            (so the optimizer step is a no-op).
        """
        # psm has shape (batch, num_anchors_per_cell, H, W); permute
        # to put anchors last and flatten so each proposal becomes a
        # single row.
        probabilities = torch.sigmoid(output_dict["ego"]["psm"].permute(0, 2, 3, 1)).reshape(-1)
        proposals = dataset.post_processor.delta_to_boxes3d(output_dict["ego"]["rm"], batch_data["ego"]["anchor_box"])[0]
        proposals_lwh = AdvCPAttackHelper.model_boxes_to_lwh(proposals, dataset)
        target_boxes_lwh = AdvCPAttackHelper.model_boxes_to_lwh(target_boxes, dataset)

        loss_terms: list[torch.Tensor] = []
        for target_box in target_boxes_lwh:
            iou_weights = AdvCPAttackHelper.compute_iou_weights(proposals_lwh, target_box)
            # 0.01 threshold filters out proposals so distant from the
            # target that they would contribute negligible gradient.
            box_mask = iou_weights >= 0.01
            if torch.any(box_mask):
                # log(1 - p) is unbounded as p -> 1, so clamp the
                # complement away from zero before logging.
                log_prob = torch.log(torch.clamp(1.0 - probabilities[box_mask], min=1e-6))
                loss_terms.append((iou_weights[box_mask] * log_prob).sum())

        if not loss_terms:
            # Return a zero tensor still tied to the graph so .backward()
            # on it remains a valid no-op.
            return probabilities.sum() * 0.0
        return torch.stack(loss_terms).sum()

    @classmethod
    def _compute_removal_loss(
        cls,
        output_dict: Mapping[str, Any],
        batch_data: Mapping[str, Any],
        dataset: Any,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Removal loss: encourage low confidence at every target box.

        Symmetric counterpart of :meth:`_compute_spoof_loss`: the
        per-proposal contribution is ``iou * log(score)`` instead of
        ``iou * log(1 - score)``. Minimising it drives scores toward
        zero for proposals overlapping the targets.

        Returns
        -------
        torch.Tensor
            Zero-dim loss tensor.
        """
        probabilities = torch.sigmoid(output_dict["ego"]["psm"].permute(0, 2, 3, 1)).reshape(-1)
        proposals = dataset.post_processor.delta_to_boxes3d(output_dict["ego"]["rm"], batch_data["ego"]["anchor_box"])[0]
        proposals_lwh = AdvCPAttackHelper.model_boxes_to_lwh(proposals, dataset)
        target_boxes_lwh = AdvCPAttackHelper.model_boxes_to_lwh(target_boxes, dataset)

        loss_terms: list[torch.Tensor] = []
        for target_box in target_boxes_lwh:
            iou_weights = AdvCPAttackHelper.compute_iou_weights(proposals_lwh, target_box)
            box_mask = iou_weights >= 0.01
            if torch.any(box_mask):
                # Clamp away from zero to keep log() finite.
                log_prob = torch.log(torch.clamp(probabilities[box_mask], min=1e-6))
                loss_terms.append((iou_weights[box_mask] * log_prob).sum())

        if not loss_terms:
            return probabilities.sum() * 0.0
        return torch.stack(loss_terms).sum()
