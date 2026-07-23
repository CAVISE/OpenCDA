"""Tests for localization delivery through SelfInformer."""

from types import SimpleNamespace
from unittest.mock import Mock

from opencda.core.application.behavior import BROADCAST_SERVICE_TYPE
from opencda.core.application.behavior.services.self_informer import SelfInformer
from opencda.core.application.behavior.types import Transform
from opencda.core.sensing.localization import LocalizationSource, LocalizationState


class _Owner:
    def __init__(self, localization: LocalizationState) -> None:
        self.id = "node-1"
        self.agent = SimpleNamespace(localizer=Mock())
        self.agent.localizer.get_state.return_value = localization


def test_process_publishes_localizer_state() -> None:
    localization = LocalizationState(
        transform=Transform(),
        speed_kmh=12.5,
        source=LocalizationSource.GT,
    )
    owner = _Owner(localization)
    service = SelfInformer()
    service.on_attach(owner)

    (message,) = service.process(())

    assert message.src_owner_id == owner.id
    assert message.dst_owner_id == owner.id
    assert message.dst_service_type == BROADCAST_SERVICE_TYPE
    assert message.payload.localization is localization
    assert service.get_state().localization is localization
    owner.agent.localizer.get_state.assert_called_once_with()
