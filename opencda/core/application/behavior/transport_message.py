"""Typed transport envelope used by behavior services."""

from typing import Generic, TypeVar

from dataclasses import dataclass

BROADCAST_OWNER_ID = "broadcast"
BROADCAST_SERVICE_TYPE = "broadcast"

payloadT = TypeVar("payloadT", covariant=True)


@dataclass(frozen=True)
class TransportMessage(Generic[payloadT]):
    """Message wrapper carrying typed payloads between behavior services."""

    src_owner_id: str
    src_service_type: str
    dst_owner_id: str
    dst_service_type: str
    payload: payloadT  # Service message
