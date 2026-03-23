"""Scalar sample model used by raw metric series."""

from dataclasses import asdict, dataclass
from typing import TypeVar 


@dataclass(frozen=True, slots=True)
class MetricSample:
    """
    A single scalar metric observation collected during simulation.

    Parameters
    ----------
    tick : int
        Simulation tick when the sample was collected.
    value : float
        Scalar metric value.
    """

    tick: int
    value: TypeVar

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation of the sample."""
        return asdict(self)
