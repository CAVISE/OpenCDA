"""Wire models shared by runtime responses and generated native OpenAPI."""

from typing import Literal
from pydantic import BaseModel, Field


class Task(BaseModel):
    """Observed task identity and phase; missing results are never synthesized."""

    id: str = Field(min_length=1)
    state: Literal["PREPARING", "RUNNING", "UNKNOWN", "SUCCEEDED", "FAILED", "CANCELLED"]


class Snapshot(BaseModel):
    """Complete atomic Runner observation; every field is required."""

    registration_id: str
    runner_id: str
    instance_id: str
    api_version: Literal["v1"]  # noqa: DC01 - Pydantic reads this field for validation and serialization.
    version: str  # noqa: DC01 - Pydantic reads this field for validation and serialization.
    capabilities: list[str]
    revision: int = Field(ge=0)
    ready: bool  # noqa: DC01 - Pydantic reads this field for validation and serialization.
    occupancy: Literal["idle", "busy", "unknown"]  # noqa: DC01 - Pydantic reads this field for validation and serialization.
    current_task: Task | None  # noqa: DC01 - Pydantic reads this field for validation and serialization.
