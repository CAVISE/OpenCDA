"""Tests for retained, unavailable platooning APIs."""

import pytest

from opencda.core.application.platooning.fsm import FSM
from opencda.core.application.platooning.platoon_behavior_agent import PlatooningBehaviorAgent
from opencda.core.application.platooning.platooning_manager import PlatooningManager
from opencda.core.application.platooning.platooning_plugin import PlatooningPlugin


@pytest.mark.parametrize(
    "platooning_type",
    [PlatooningBehaviorAgent, PlatooningManager, PlatooningPlugin],
)
def test_platooning_components_are_explicitly_unavailable(platooning_type) -> None:
    with pytest.raises(NotImplementedError, match="not implemented in the unified agent runtime"):
        platooning_type()


def test_platooning_fsm_remains_importable() -> None:
    assert FSM.SEARCHING.value == 0
