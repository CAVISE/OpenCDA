"""AIM server behavior service implementation."""

from __future__ import annotations
import weakref

import logging
from typing import Sequence, cast, TYPE_CHECKING

from opencda.core.application.behavior.registry import BehaviorServiceRegistry

from AIM import get_model

if TYPE_CHECKING:
    from opencda.core.common.rsu_manager import RSUManager

    from .messages import AIMServerRequestMessage
    from .results import AIMServerResult
from .aim_model_manager import AIMModelManager

logger = logging.getLogger("cavise.opencda.opencda.core.application.behavior.services.aim_server")


@BehaviorServiceRegistry.register
class AIMServer:
    """Behavior service that runs AIM predictions for a batch of CAV requests."""

    service_name = "aim_server"

    def __init__(
        self,
        **aim_config,
    ):
        """
        Initialize the AIM-backed behavior service.

        Parameters
        ----------
        model : AIMModel
            Loaded AIM model used for trajectory prediction.
        """
        self._owner_ref = None

        aim_model_name = cast(str, aim_config.pop("model", "MTP"))
        self.model = get_model(aim_model_name, **aim_config)

    def on_attach(self, owner: RSUManager) -> None:
        """Initialize the service for a particular participant instance."""
        self._owner_ref = weakref.ref(owner)

        self._owner_ref().localizer.localize()
        control_center = self._owner_ref().localizer.get_ego_pos()
        self.aim_model_manager = AIMModelManager(self.model, control_center, self.service_name, self._owner_ref().id)

    def on_detach(self) -> None:
        """Release service resources before the participant is destroyed."""
        self._owner_ref = None

    def process(self, messages: Sequence[AIMServerRequestMessage]) -> AIMServerResult:
        return self.aim_model_manager.process(messages)
