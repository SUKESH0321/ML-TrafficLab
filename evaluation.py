"""Matched-seed evaluation against a non-learning fixed-time reference."""
from typing import Any

from q_learning_agent import QLearningAgent
from simulation import DT, IntersectionSimulator
from training import reward_for_transition

TICKS_PER_DECISION = round(1.0 / DT)
DECISIONS_PER_EPISODE = 120
SEED_BASE = 73191


class MatchedComparison:
    """Evaluate greedy RL, then fixed timing on the exact same seed and arrival process."""

    def __init__(
        self,
        agent: QLearningAgent,
        density: str = "medium",
        seeds: int = 5,
        decisions_per_episode: int = DECISIONS_PER_EPISODE,
    ) -> None:
        self.agent = agent
        self.density = density
        self.seed_count = seeds
        self.decisions_per_episode = decisions_per_episode
        self.seed_index = 0
        self.controller_index = 0  # greedy RL first, fixed-time second for each seed
        self.decision = 0
        self.reward = 0.0
        self.complete = False
        self.records: dict[str, list[dict[str, Any]]] = {"rl": [], "fixed": []}
        self.sim = self._new_simulator()

    @property
    def controller_name(self) -> str:
        return "rl" if self.controller_index == 0 else "fixed"

    @property
    def completed_runs(self) -> int:
        return len(self.records["rl"]) + len(self.records["fixed"])

    @property
    def progress(self) -> float:
        return (self.seed_index * 2 + self.controller_index) / (self.seed_count * 2)

    def _seed(self) -> int:
        return SEED_BASE + self.seed_index * 101

    def _new_simulator(self) -> IntersectionSimulator:
        return IntersectionSimulator(density=self.density, seed=self._seed(), start_phase="NS")

    def advance_one(self) -> dict[str, Any] | None:
        if self.complete:
            return None
        if self.controller_index == 0:
            action = self.agent.select_action(self.sim.state, explore=False)
        else:
            phase = "NS" if int(self.sim.time // 12) % 2 == 0 else "EW"
            action = 0 if phase == "NS" else 2
        self.sim.request_action(action)
        for _ in range(TICKS_PER_DECISION):
            self.reward += reward_for_transition(self.sim.step(DT))
        self.decision += 1
        if self.decision < self.decisions_per_episode:
            return None

        record = {**self.sim.metrics(), "reward": self.reward, "seed": self._seed()}
        self.records[self.controller_name].append(record)
        self.decision = 0
        self.reward = 0.0
        if self.controller_index == 0:
            self.controller_index = 1
            self.sim = self._new_simulator()
        else:
            self.controller_index = 0
            self.seed_index += 1
            if self.seed_index >= self.seed_count:
                self.complete = True
            else:
                self.sim = self._new_simulator()
        return record

    def advance_batch(self, decisions: int = 1) -> bool:
        for _ in range(max(0, decisions)):
            if self.complete:
                break
            self.advance_one()
        return self.complete

    def summary(self) -> dict[str, Any]:
        def mean(records: list[dict[str, Any]], key: str) -> float:
            return sum(record[key] for record in records) / len(records) if records else 0.0

        return {
            "rl": {
                "average_wait": mean(self.records["rl"], "average_wait"),
                "average_queue": mean(self.records["rl"], "average_queue"),
                "vehicles_served": mean(self.records["rl"], "vehicles_served"),
                "switches": mean(self.records["rl"], "switches"),
                "reward": mean(self.records["rl"], "reward"),
            },
            "fixed": {
                "average_wait": mean(self.records["fixed"], "average_wait"),
                "average_queue": mean(self.records["fixed"], "average_queue"),
                "vehicles_served": mean(self.records["fixed"], "vehicles_served"),
                "switches": mean(self.records["fixed"], "switches"),
                "reward": mean(self.records["fixed"], "reward"),
            },
            "seeds": self.seed_count,
            "density": self.density,
        }


def compare_controllers(
    agent: QLearningAgent,
    density: str = "medium",
    seeds: int = 5,
    decisions_per_episode: int = DECISIONS_PER_EPISODE,
) -> dict[str, Any]:
    comparison = MatchedComparison(agent, density, seeds, decisions_per_episode)
    while not comparison.complete:
        comparison.advance_batch(128)
    return comparison.summary()
