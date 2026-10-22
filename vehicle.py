"""Small data model for one straight-through intersection vehicle."""
from dataclasses import dataclass


@dataclass
class Vehicle:
    """A car's progress along one approach, plus its accumulated stopped time."""

    vehicle_id: int
    direction: str
    progress: float = 0.0
    wait_seconds: float = 0.0
    color_index: int = 0
