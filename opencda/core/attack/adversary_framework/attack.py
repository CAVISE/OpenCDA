"""Generic runtime attack object backed by a structured attack spec."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.scenario_testing.types import SimulationSnapshot

from .attack_stage_protocol import AttackStage
from .condition_evaluator import evaluate_condition
from .models import AttackSpec, AttackStageResult, RuntimeStatus, StageRuntime, Status
from .stage_registry import AttackStageRegistry
from .utils import ServiceResolver, match_services, resolve_targets as resolve_configured_targets

if TYPE_CHECKING:
    pass


class Attack:
    """Generic attack runtime built entirely from an `AttackSpec`."""

    def __init__(self, spec: AttackSpec, stages: tuple[AttackStage, ...]) -> None:
        if len(spec.stages) != len(stages):
            raise ValueError(f"Attack '{spec.name}' defines {len(spec.stages)} stage specs, but received {len(stages)} runtime stage instances.")

        stage_ids = [stage_spec.id for stage_spec in spec.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(f"Attack '{spec.name}' defines duplicate stage ids.")

        self.attack_name = spec.name
        self.spec = spec
        self.stages = stages
        self.is_active = False
        self.status = RuntimeStatus.INACTIVE
        self.previous_status = RuntimeStatus.INACTIVE
        self.stage_runtimes = tuple(StageRuntime(spec=stage_spec, stage=stage) for stage_spec, stage in zip(spec.stages, stages, strict=True))

    @classmethod
    def from_spec(cls, spec: AttackSpec) -> Attack:
        """Build a generic runtime attack from config and stage registry."""
        stages = tuple(AttackStageRegistry.create_stage(stage_spec.type) for stage_spec in spec.stages)
        return cls(spec=spec, stages=stages)

    def mark_started(self) -> None:
        """Mark attack runtime as started."""
        self.is_active = True
        self._set_attack_status(RuntimeStatus.STARTED)

    def mark_active(self) -> None:
        """Mark attack runtime as active."""
        self.is_active = True
        self._set_attack_status(RuntimeStatus.ACTIVE)

    def mark_succeeded(self) -> None:
        """Mark attack runtime as completed successfully."""
        self.is_active = False
        self._set_attack_status(RuntimeStatus.SUCCESS)

    def mark_failed(self) -> None:
        """Mark attack runtime as failed."""
        self.is_active = False
        self._set_attack_status(RuntimeStatus.FAIL)

    def mark_stopped(self) -> None:
        """Mark attack runtime as stopped."""
        self.is_active = False
        self._set_attack_status(RuntimeStatus.STOPPED)

    def reset_runtime(self) -> None:
        """Reset attack and stage runtime state."""
        self.is_active = False
        self._set_attack_status(RuntimeStatus.INACTIVE)
        for stage_runtime in self.stage_runtimes:
            self._set_stage_status(stage_runtime, RuntimeStatus.INACTIVE)
            stage_runtime.last_result = None

    def get_stage_runtime(self, stage_id: str) -> StageRuntime:  # noqa: DC04
        """Return runtime state for a configured stage by id."""
        for stage_runtime in self.stage_runtimes:
            if stage_runtime.spec.id == stage_id:
                return stage_runtime

        raise KeyError(f"Attack '{self.attack_name}' does not define stage id '{stage_id}'.")

    def get_stage_history(self) -> tuple[AttackStageResult, ...]:
        """Return accumulated stage results."""
        return tuple(stage_runtime.last_result for stage_runtime in self.stage_runtimes if stage_runtime.last_result is not None)

    def should_start(
        self,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        """Return whether the attack should start on this tick."""
        if self.is_active:
            return False

        return self.check_requirements(previous_snapshot, current_snapshot) and self.check_start_trigger(
            previous_snapshot,
            current_snapshot,
        )

    def should_stop(
        self,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        """Return whether the attack should stop on this tick."""
        if not self.is_active:
            return False

        return self.check_stop_trigger(previous_snapshot, current_snapshot)

    def deactivate(self) -> None:
        """Deactivate all stages and reset runtime state."""
        for stage_runtime in self.stage_runtimes:
            self._deactivate_stage(stage_runtime)

        self.reset_runtime()

    def resolve_targets(
        self,
        current_snapshot: SimulationSnapshot,
        service_resolver: ServiceResolver,
    ) -> tuple[BehaviorService[Any, Any], ...]:
        """Resolve live target services according to `spec.targets`."""
        return resolve_configured_targets(self.spec.targets, current_snapshot, service_resolver)

    def run_stage_lifecycle(
        self,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
        available_services: tuple[BehaviorService[Any, Any], ...],
    ) -> tuple[AttackStageResult, ...]:
        """Advance stage runtime according to configured/default lifecycle rules."""
        emitted_results: list[AttackStageResult] = []

        for stage_runtime in self.stage_runtimes:
            if stage_runtime.status != RuntimeStatus.ACTIVE:
                continue

            stop_trigger = stage_runtime.spec.stage_stop_trigger
            if stop_trigger is None:
                continue

            if self._evaluate_condition(stop_trigger, previous_snapshot, current_snapshot):
                self._deactivate_stage(stage_runtime)
                stop_result = AttackStageResult(
                    stage_name=stage_runtime.stage.stage_name,
                    status=Status.SUCCESS,
                    reason="Stage stop trigger satisfied.",
                )
                stage_runtime.last_result = stop_result
                self._set_stage_status(stage_runtime, RuntimeStatus.SUCCESS)
                emitted_results.append(stop_result)

        progress = True
        while progress:
            progress = False
            for index, stage_runtime in enumerate(self.stage_runtimes):
                if stage_runtime.status != RuntimeStatus.INACTIVE:
                    continue

                if stage_runtime.spec.requirements is not None and not self._evaluate_condition(
                    stage_runtime.spec.requirements,
                    previous_snapshot,
                    current_snapshot,
                ):
                    continue

                if not self._should_start_stage(index, stage_runtime, previous_snapshot, current_snapshot):
                    continue

                matched_services = tuple(match_services(available_services, stage_runtime.stage.required_capabilities))
                if not matched_services:
                    fail_result = AttackStageResult(
                        stage_name=stage_runtime.stage.stage_name,
                        status=Status.FAIL,
                        reason=f"No services matched required capabilities for stage '{stage_runtime.spec.id}'.",
                    )
                    stage_runtime.last_result = fail_result
                    self._set_stage_status(stage_runtime, RuntimeStatus.FAIL)
                    self.mark_failed()
                    emitted_results.append(fail_result)
                    return tuple(emitted_results)

                stage_result = stage_runtime.stage.execute(matched_services)
                stage_runtime.last_result = stage_result
                emitted_results.append(stage_result)

                if stage_result.status == Status.FAIL:
                    self._set_stage_status(stage_runtime, RuntimeStatus.FAIL)
                    self.mark_failed()
                    return tuple(emitted_results)

                if stage_result.status == Status.STOP:
                    self._set_stage_status(stage_runtime, RuntimeStatus.STOPPED)
                    self.mark_stopped()
                    return tuple(emitted_results)

                if stage_runtime.spec.stage_stop_trigger is None:
                    self._set_stage_status(stage_runtime, RuntimeStatus.SUCCESS)
                else:
                    self._set_stage_status(stage_runtime, RuntimeStatus.ACTIVE)

                progress = True

        if self.stage_runtimes and all(stage_runtime.status == RuntimeStatus.SUCCESS for stage_runtime in self.stage_runtimes):
            self.mark_succeeded()

        return tuple(emitted_results)

    def check_requirements(
        self,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        """Evaluate attack-level requirements against the environment."""
        requirements = self.spec.requirements
        if requirements is None:
            return True
        return evaluate_condition(self, requirements, previous_snapshot, current_snapshot)

    def check_start_trigger(
        self,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        """Evaluate attack start trigger."""
        start_trigger = self.spec.start_trigger
        if start_trigger is None:
            return False
        return evaluate_condition(self, start_trigger, previous_snapshot, current_snapshot)

    def check_stop_trigger(
        self,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        """Evaluate attack stop trigger."""
        stop_trigger = self.spec.stop_trigger
        if stop_trigger is None:
            return False
        return evaluate_condition(self, stop_trigger, previous_snapshot, current_snapshot)

    def _should_start_stage(
        self,
        index: int,
        stage_runtime: StageRuntime,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        start_trigger = stage_runtime.spec.stage_start_trigger
        if start_trigger is not None:
            return evaluate_condition(self, start_trigger, previous_snapshot, current_snapshot)

        if index == 0:
            return self.is_active

        return self.stage_runtimes[index - 1].status == RuntimeStatus.SUCCESS

    def _evaluate_condition(
        self,
        condition: Any,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        return evaluate_condition(self, condition, previous_snapshot, current_snapshot)

    @staticmethod
    def _deactivate_stage(stage_runtime: StageRuntime) -> None:
        deactivate = getattr(stage_runtime.stage, "deactivate", None)
        if callable(deactivate):
            deactivate()

    def _set_attack_status(self, status: RuntimeStatus) -> None:
        self.previous_status = self.status
        self.status = status

    @staticmethod
    def _set_stage_status(stage_runtime: StageRuntime, status: RuntimeStatus) -> None:
        stage_runtime.previous_status = stage_runtime.status
        stage_runtime.status = status
