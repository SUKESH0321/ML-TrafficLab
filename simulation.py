"""Stochastic straight-through intersection simulation shared by training and evaluation."""
import random
from typing import Any

from traffic_signal import TrafficSignal
from vehicle import Vehicle

DT = 1 / 30
APPROACHES = ("north", "south", "east", "west")
QUEUE_STOP = 0.38
QUEUE_EXIT = 0.62
VEHICLE_GAP = 0.066
VEHICLE_SPEED = 0.16
RATES_PER_MINUTE = {
    "low": {"north": 4, "south": 3, "east": 3, "west": 2},
    "medium": {"north": 8, "south": 5, "east": 7, "west": 4},
    "high": {"north": 13, "south": 9, "east": 11, "west": 7},
}


def queue_bucket(length: int) -> int:
    """Return 0=Low (0–2), 1=Medium (3–5), or 2=High (6+)."""
    if length <= 2:
        return 0
    if length <= 5:
        return 1
    return 2


def encode_state(queue_lengths: list[int] | tuple[int, ...], phase: str) -> int:
    """Encode four ternary queue categories and the current axis phase (162 states)."""
    if len(queue_lengths) != 4:
        raise ValueError("The state requires North, South, East, and West queue lengths.")
    queue_code = 0
    for length in queue_lengths:
        queue_code = queue_code * 3 + queue_bucket(length)
    return (81 if phase == "EW" else 0) + queue_code


class IntersectionSimulator:
    """Four independent approach queues, two legal green phases, no turning movements."""

    def __init__(self, density: str = "medium", seed: int = 1, start_phase: str = "NS") -> None:
        if density not in RATES_PER_MINUTE:
            raise ValueError(f"Unknown density: {density}")
        self.density = density
        self.random = random.Random(seed)
        self.signal = TrafficSignal(phase=start_phase)
        self.lanes: dict[str, list[Vehicle]] = {name: [] for name in APPROACHES}
        self.pending_arrivals = {name: 0 for name in APPROACHES}
        self.arrival_counts = {name: 0 for name in APPROACHES}
        self.next_vehicle_id = 1
        self.time = 0.0
        self.served = 0
        self.waiting_time_of_served = 0.0
        self.total_waiting_seconds = 0.0
        self.queue_sum = 0
        self.queue_samples = 0
        self.cumulative_reward = 0.0

    def set_density(self, density: str) -> None:
        if density not in RATES_PER_MINUTE:
            raise ValueError(f"Unknown density: {density}")
        self.density = density

    def request_action(self, action: int) -> None:
        self.signal.request_action(action)

    def queue_lengths(self) -> list[int]:
        return [sum(car.progress < QUEUE_STOP for car in self.lanes[name]) for name in APPROACHES]

    def queue_by_direction(self) -> dict[str, int]:
        return dict(zip(APPROACHES, self.queue_lengths()))

    @property
    def state(self) -> int:
        return encode_state(self.queue_lengths(), self.signal.phase)

    @property
    def total_vehicles(self) -> int:
        return sum(len(self.lanes[name]) for name in APPROACHES)

    @property
    def average_wait(self) -> float:
        return self.waiting_time_of_served / self.served if self.served else 0.0

    @property
    def average_queue(self) -> float:
        if not self.queue_samples:
            return sum(self.queue_lengths()) / 4.0
        return self.queue_sum / (self.queue_samples * 4.0)

    def max_wait_by_approach(self) -> dict[str, float]:
        return {
            name: max((car.wait_seconds for car in self.lanes[name]), default=0.0)
            for name in APPROACHES
        }

    def _intersection_clear(self) -> bool:
        return all(
            car.progress < QUEUE_STOP or car.progress >= QUEUE_EXIT
            for name in APPROACHES for car in self.lanes[name]
        )

    def _spawn(self, dt: float) -> None:
        rates = RATES_PER_MINUTE[self.density]
        for direction in APPROACHES:
            lane = self.lanes[direction]
            tail = lane[-1] if lane else None
            available = tail is None or tail.progress >= VEHICLE_GAP
            # Draw once per approach on every step. This keeps paired-seed arrival events
            # identical even when different signal policies change lane occupancy.
            arrival_event = self.random.random() < rates[direction] * dt / 60.0
            if arrival_event:
                self.pending_arrivals[direction] += 1
                self.arrival_counts[direction] += 1
            if available and self.pending_arrivals[direction] > 0:
                lane.append(Vehicle(
                    vehicle_id=self.next_vehicle_id,
                    direction=direction,
                    color_index=self.next_vehicle_id % 5,
                ))
                self.next_vehicle_id += 1
                self.pending_arrivals[direction] -= 1

    def step(self, dt: float = DT, spawn: bool = True) -> dict[str, Any]:
        if dt <= 0:
            raise ValueError("Simulation step must be a positive duration.")
        served_before = self.served
        switches_before = self.signal.switches
        waiting_seconds = 0.0
        self.time += dt
        if spawn:
            self._spawn(dt)
        self.signal.advance(dt, self._intersection_clear())

        for direction in APPROACHES:
            lane = self.lanes[direction]
            for index, car in enumerate(lane):
                front_progress = 1.1 if index == 0 else lane[index - 1].progress
                gap_limit = front_progress - VEHICLE_GAP
                desired = car.progress + VEHICLE_SPEED * dt
                next_progress = min(desired, gap_limit)
                entering = car.progress < QUEUE_STOP and desired >= QUEUE_STOP
                if entering and not self.signal.is_green(direction):
                    next_progress = min(next_progress, QUEUE_STOP - 0.0001)
                if QUEUE_STOP <= car.progress < QUEUE_EXIT:
                    # A car already in the conflict box clears during all-red.
                    next_progress = min(desired, gap_limit)
                if next_progress < desired - 0.00001:
                    car.wait_seconds += dt
                    waiting_seconds += dt
                car.progress = max(0.0, next_progress)

            completed = [car for car in lane if car.progress > 1.0]
            for car in completed:
                self.served += 1
                self.waiting_time_of_served += car.wait_seconds
            self.lanes[direction] = [car for car in lane if car.progress <= 1.0]

        self.signal.finish_clearance(self._intersection_clear())
        queue_total = sum(self.queue_lengths())
        self.queue_sum += queue_total
        self.queue_samples += 1
        self.total_waiting_seconds += waiting_seconds
        return {
            "served": self.served - served_before,
            "waiting_seconds": waiting_seconds,
            "switches": self.signal.switches - switches_before,
            "queue_total": queue_total,
            "max_wait_by_approach": self.max_wait_by_approach(),
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "simulation_seconds": self.time,
            "vehicles_in_system": self.total_vehicles,
            "vehicles_served": self.served,
            "average_wait": self.average_wait,
            "average_queue": self.average_queue,
            "switches": self.signal.switches,
            "waiting_seconds": self.total_waiting_seconds,
            "queues": self.queue_by_direction(),
            "arrivals": dict(self.arrival_counts),
        }
