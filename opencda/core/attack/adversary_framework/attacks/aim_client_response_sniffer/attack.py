from __future__ import annotations

from opencda.core.application.behavior.behavior_service_protocol import BehaviorService
from opencda.core.application.behavior.services.aim_server.types import AIMServerState
from opencda.core.attack.adversary_framework.attack_protocol import ServiceResolver
from opencda.core.attack.adversary_framework.registry import AttackRegistry
from opencda.core.attack.adversary_framework.stages.sniffer import SnifferStage
from opencda.scenario_testing.types import SimulationSnapshot


@AttackRegistry.register
class AimClientResponseSnifferAttack:
    """Start sniffing aim-client responses once AIM begins tracking vehicles."""

    attack_name = "aim_client_response_sniffer"

    def __init__(self) -> None:
        self._sniffer_stage = SnifferStage()
        self.stages = (self._sniffer_stage,)
        self.is_active = False

    def should_start(
        self,
        previous_snapshot: SimulationSnapshot | None,
        current_snapshot: SimulationSnapshot,
    ) -> bool:
        if self.is_active:
            return False

        previous_counts = self._collect_tracked_vehicle_counts(previous_snapshot)
        current_counts = self._collect_tracked_vehicle_counts(current_snapshot)

        for node_id, current_count in current_counts.items():
            if current_count >= 1 and current_count > previous_counts.get(node_id, 0):
                return True

        return False

    def resolve_targets(
        self,
        current_snapshot: SimulationSnapshot,
        service_resolver: ServiceResolver,
    ) -> tuple[BehaviorService[object, object], ...]:
        target_services: list[BehaviorService[object, object]] = []
        tracked_vehicle_ids = self._collect_tracked_vehicle_ids(current_snapshot)

        for vehicle_id in sorted(tracked_vehicle_ids):
            service = service_resolver(vehicle_id, "aim_client")
            if service is not None:
                target_services.append(service)

        return tuple(target_services)

    def mark_active(self) -> None:
        self.is_active = True

    def deactivate(self) -> None:
        self._sniffer_stage.deactivate()
        self.is_active = False

    @staticmethod
    def _collect_tracked_vehicle_counts(snapshot: SimulationSnapshot | None) -> dict[str, int]:
        if snapshot is None:
            return {}

        counts: dict[str, int] = {}
        for rsu_node in snapshot.rsu_nodes:
            aim_server_state = rsu_node.service_states.get("aim_server")
            if isinstance(aim_server_state, AIMServerState):
                counts[rsu_node.node_id] = aim_server_state.tracked_vehicle_count

        return counts

    @staticmethod
    def _collect_tracked_vehicle_ids(snapshot: SimulationSnapshot) -> set[str]:
        tracked_vehicle_ids: set[str] = set()

        for rsu_node in snapshot.rsu_nodes:
            aim_server_state = rsu_node.service_states.get("aim_server")
            if isinstance(aim_server_state, AIMServerState):
                tracked_vehicle_ids.update(aim_server_state.tracked_vehicle_ids)

        return tracked_vehicle_ids
