"""Episode-based Q-learning with incremental steps for a responsive Pygame UI."""
import random
from typing import Any

from q_learning_agent import QLearningAgent
from simulation import IntersectionSimulator

DT = 0.2


def reward_for_transition(metrics: dict[str, Any]) -> float:
    """Measured reward: throughput − waiting − queues − switches − starvation."""
    starvation_excess = sum(max(0.0, seconds - 15.0) for seconds in metrics["max_wait_by_approach"].values())
    return (
        2.5 * metrics["served"]
        - 0.03 * metrics["waiting_seconds"]
        - 0.025 * metrics["queue_total"]
        - 0.8 * metrics["switches"]
        - 0.005 * starvation_excess
    )


class TrainingSession:
    """One Q-learning interaction per decision; call advance_batch from the UI loop."""

    def __init__(
        self,
        agent: QLearningAgent,
        episodes: int = 80,
        density: str = "medium",
        decisions_per_episode: int = 90,
        seconds_per_decision: float = 1.0,
        seed: int = 60213,
    ) -> None:
        self.agent = agent
        self.episode_target = max(1, int(episodes))
        self.density = density
        self.decisions_per_episode = decisions_per_episode
        self.ticks_per_decision = round(seconds_per_decision / DT)
        self.seed = seed
        self.random = random.Random(seed + 17)
        self.completed_episodes = 0
        self.decision = 0
        self.episode_reward = 0.0
        self.history: list[dict[str, float | int]] = []
        self.complete = False
        self.sim = self._new_simulator(0)

    def _new_simulator(self, episode: int) -> IntersectionSimulator:
        return IntersectionSimulator(
            density=self.density,
            seed=(self.seed + episode * 7919) & 0xFFFFFFFF,
            start_phase="EW" if episode % 2 else "NS",
        )

    def set_density(self, density: str) -> None:
        self.density = density
        self.sim.set_density(density)

    def advance_one(self) -> dict[str, float | int] | None:
        if self.complete:
            return None
        state = self.sim.state
        action = self.agent.select_action(state, explore=True, rng=self.random)
        self.sim.request_action(action)
        reward = 0.0
        for _ in range(self.ticks_per_decision):
            reward += reward_for_transition(self.sim.step(DT))
        next_state = self.sim.state
        self.decision += 1
        terminal = self.decision >= self.decisions_per_episode
        self.agent.update(state, action, reward, next_state, terminal)
        self.episode_reward += reward
        self.sim.cumulative_reward = self.episode_reward
        if not terminal:
            return None

        metrics = self.sim.metrics()
        entry: dict[str, float | int] = {
            "episode": self.completed_episodes + 1,
            "reward": self.episode_reward,
            "average_wait": metrics["average_wait"],
            "average_queue": metrics["average_queue"],
            "vehicles_served": metrics["vehicles_served"],
            "switches": metrics["switches"],
            "epsilon": self.agent.epsilon,
        }
        self.history.append(entry)
        self.completed_episodes += 1
        self.agent.decay_exploration()
        self.decision = 0
        self.episode_reward = 0.0
        if self.completed_episodes >= self.episode_target:
            self.complete = True
        else:
            self.sim = self._new_simulator(self.completed_episodes)
        return entry

    def advance_batch(self, decisions: int = 1) -> list[dict[str, float | int]]:
        completed = []
        for _ in range(max(0, decisions)):
            if self.complete:
                break
            result = self.advance_one()
            if result is not None:
                completed.append(result)
        return completed


def train_agent(
    agent: QLearningAgent,
    episodes: int = 80,
    density: str = "medium",
    seed: int = 60213,
) -> list[dict[str, float | int]]:
    """Convenience function for scripts; the Pygame interface uses TrainingSession."""
    session = TrainingSession(agent, episodes=episodes, density=density, seed=seed)
    while not session.complete:
        session.advance_batch(128)
    return session.history
