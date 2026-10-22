"""Small, inspectable tabular Q-Learning agent (162 states × 4 actions)."""
from pathlib import Path
import random

import numpy as np

STATE_COUNT = 162
ACTION_COUNT = 4


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.18,
        gamma: float = 0.94,
        epsilon: float = 0.92,
        epsilon_decay: float = 0.985,
        min_epsilon: float = 0.06,
        q_table: np.ndarray | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if q_table is None:
            self.q_table = np.zeros((STATE_COUNT, ACTION_COUNT), dtype=np.float64)
        else:
            table = np.asarray(q_table, dtype=np.float64)
            if table.shape != (STATE_COUNT, ACTION_COUNT):
                raise ValueError(f"Expected a {(STATE_COUNT, ACTION_COUNT)} Q-table, got {table.shape}.")
            self.q_table = table.copy()
        self.updates = 0

    def select_action(self, state: int, explore: bool = True, rng: random.Random | None = None) -> int:
        if state < 0 or state >= STATE_COUNT:
            raise ValueError(f"State index must be in [0, {STATE_COUNT - 1}].")
        chooser = rng or random
        if explore and chooser.random() < self.epsilon:
            return chooser.randrange(ACTION_COUNT)
        # np.argmax returns the first maximum, making greedy ties deterministic.
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, terminal: bool = False) -> float:
        """Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]."""
        current = self.q_table[state, action]
        future_value = 0.0 if terminal else float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * future_value
        self.q_table[state, action] = current + self.alpha * (target - current)
        self.updates += 1
        return float(self.q_table[state, action])

    def decay_exploration(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, self.q_table)
        return destination

    @classmethod
    def load(cls, path: str | Path, **parameters: float) -> "QLearningAgent":
        table = np.load(Path(path), allow_pickle=False)
        return cls(q_table=table, **parameters)
