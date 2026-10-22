"""Safety-enforced two-axis traffic signal phases."""
from dataclasses import dataclass

ACTIONS = ("North", "South", "East", "West")
MIN_GREEN_SECONDS = 5.0
MAX_GREEN_SECONDS = 17.0
ALL_RED_SECONDS = 1.0


def axis_for_action(action: int) -> str:
    """Map N/S actions to one legal axis and E/W actions to the other."""
    if action not in range(4):
        raise ValueError("Action must be 0 (North), 1 (South), 2 (East), or 3 (West).")
    return "NS" if action < 2 else "EW"


@dataclass
class TrafficSignal:
    """Only NS or EW can be green; axis switches include an all-red interval."""

    phase: str = "NS"
    mode: str = "green"
    green_elapsed: float = 0.0
    all_red_remaining: float = 0.0
    pending_phase: str = "NS"
    requested_action: int = 0
    switches: int = 0

    def __post_init__(self) -> None:
        if self.phase not in ("NS", "EW"):
            raise ValueError("Signal phase must be NS or EW.")
        self.pending_phase = self.phase
        self.requested_action = 0 if self.phase == "NS" else 2

    def request_action(self, action: int) -> None:
        target = axis_for_action(action)
        self.requested_action = action
        if self.mode == "all-red":
            self.pending_phase = target

    def is_green(self, direction: str) -> bool:
        if self.mode != "green":
            return False
        return (self.phase == "NS" and direction in ("north", "south")) or (
            self.phase == "EW" and direction in ("east", "west")
        )

    @property
    def label(self) -> str:
        if self.mode == "all-red":
            return "ALL RED"
        return "N–S GREEN" if self.phase == "NS" else "E–W GREEN"

    def advance(self, dt: float, intersection_clear: bool) -> None:
        """Start/advance clearance; only the environment may grant green."""
        if self.mode == "green":
            self.green_elapsed += dt
            target = axis_for_action(self.requested_action)
            if self.green_elapsed >= MAX_GREEN_SECONDS and target == self.phase:
                target = "EW" if self.phase == "NS" else "NS"
            if target != self.phase and self.green_elapsed >= MIN_GREEN_SECONDS and intersection_clear:
                self.mode = "all-red"
                self.pending_phase = target
                self.all_red_remaining = ALL_RED_SECONDS
        else:
            self.all_red_remaining = max(0.0, self.all_red_remaining - dt)

    def finish_clearance(self, intersection_clear: bool) -> None:
        """Commit an axis switch only after the full red interval and a clear box."""
        if self.mode == "all-red" and self.all_red_remaining <= 0.0 and intersection_clear:
            self.mode = "green"
            if self.pending_phase != self.phase:
                self.phase = self.pending_phase
                self.green_elapsed = 0.0
                self.switches += 1
