"""Headless core tests: no display is required; run from python-traffic-rl/ with unittest."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from q_learning_agent import QLearningAgent
from simulation import APPROACHES, IntersectionSimulator, QUEUE_STOP, encode_state, queue_bucket
from training import TrainingSession
from vehicle import Vehicle


class CoreTests(unittest.TestCase):
    def test_state_encoding_covers_162_states(self):
        self.assertEqual(queue_bucket(0), 0)
        self.assertEqual(queue_bucket(2), 0)
        self.assertEqual(queue_bucket(3), 1)
        self.assertEqual(queue_bucket(5), 1)
        self.assertEqual(queue_bucket(6), 2)
        states = {
            encode_state([n, s, e, w], phase)
            for phase in ("NS", "EW")
            for n in (0, 3, 6)
            for s in (0, 3, 6)
            for e in (0, 3, 6)
            for w in (0, 3, 6)
        }
        self.assertEqual(len(states), 162)
        self.assertEqual(max(states), 161)

    def test_q_learning_update_including_terminal_target(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0, q_table=np.zeros((162, 4)))
        agent.q_table[1] = [0, 0, 4, 2]
        self.assertAlmostEqual(agent.update(0, 1, 2, 1), 2.8)
        self.assertAlmostEqual(agent.update(0, 0, 2, 1, terminal=True), 1.0)
        self.assertEqual(agent.updates, 2)

    def test_conflicting_axes_are_never_green_simultaneously(self):
        sim = IntersectionSimulator(density="high", seed=21)
        for tick in range(350):
            sim.request_action((tick // 31) % 4)
            sim.step(0.2)
            self.assertEqual(sim.signal.is_green("north"), sim.signal.is_green("south"))
            self.assertEqual(sim.signal.is_green("east"), sim.signal.is_green("west"))
            self.assertFalse(sim.signal.is_green("north") and sim.signal.is_green("east"))
            if sim.signal.mode == "all-red":
                self.assertFalse(any(sim.signal.is_green(name) for name in APPROACHES))

    def test_cancelled_clearance_returns_to_same_phase_without_a_false_switch(self):
        sim = IntersectionSimulator(seed=7)
        sim.step(5, spawn=False)
        sim.request_action(2)
        sim.step(0.2, spawn=False)
        self.assertEqual(sim.signal.mode, "all-red")
        sim.request_action(1)  # North and South both map to the existing NS phase.
        for _ in range(8):
            sim.step(0.2, spawn=False)
        self.assertEqual(sim.signal.phase, "NS")
        self.assertEqual(sim.signal.mode, "green")
        self.assertEqual(sim.signal.switches, 0)

    def test_minimum_green_and_full_clearance_before_axis_switch(self):
        sim = IntersectionSimulator(seed=9)
        sim.request_action(2)
        sim.step(4.8, spawn=False)
        self.assertEqual(sim.signal.mode, "green")
        self.assertEqual(sim.signal.phase, "NS")
        sim.step(0.2, spawn=False)
        self.assertEqual(sim.signal.mode, "all-red")
        self.assertEqual(sim.signal.phase, "NS")
        self.assertEqual(sim.signal.switches, 0)
        sim.step(0.8, spawn=False)
        self.assertEqual(sim.signal.mode, "all-red")
        sim.step(0.2, spawn=False)
        self.assertEqual(sim.signal.phase, "EW")
        self.assertEqual(sim.signal.mode, "green")
        self.assertEqual(sim.signal.switches, 1)

    def test_red_signal_stops_a_vehicle_before_its_stop_line(self):
        sim = IntersectionSimulator(seed=2)
        sim.lanes["west"].append(Vehicle(1, "west", progress=QUEUE_STOP - 0.01))
        sim.step(0.2, spawn=False)
        self.assertLess(sim.lanes["west"][0].progress, QUEUE_STOP)

    def test_arrival_draws_are_matched_even_under_different_actions(self):
        first = IntersectionSimulator(density="high", seed=81)
        second = IntersectionSimulator(density="high", seed=81)
        for tick in range(600):
            first.request_action(0 if tick % 50 < 25 else 2)
            second.request_action((tick // 17) % 4)
            first.step(0.2)
            second.step(0.2)
        self.assertEqual(first.arrival_counts, second.arrival_counts)

    def test_q_table_save_and_load(self):
        agent = QLearningAgent()
        agent.q_table[12, 3] = 1.25
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "q_table.npy"
            agent.save(path)
            loaded = QLearningAgent.load(path)
        self.assertEqual(loaded.q_table.shape, (162, 4))
        self.assertAlmostEqual(loaded.q_table[12, 3], 1.25)

    def test_incremental_training_updates_q_values_and_completes(self):
        agent = QLearningAgent()
        session = TrainingSession(agent, episodes=1, decisions_per_episode=8, seed=14)
        while not session.complete:
            session.advance_batch(2)
        self.assertEqual(session.completed_episodes, 1)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(agent.updates, 8)
        self.assertGreater(float(np.abs(agent.q_table).sum()), 0)


if __name__ == "__main__":
    unittest.main()
