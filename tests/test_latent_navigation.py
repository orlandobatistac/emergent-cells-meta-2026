"""Sensor-only navigation state and causal ablation tests."""
import unittest
import numpy as np

from emergent_cells.experiments.latent_navigation import (
    OnlineAgent, fit_prototypes, run_episode, benchmark)
from emergent_cells.experiments.recurrent_state import RecurrentPredictor
from emergent_cells.experiments.predictive_context import traces
from emergent_cells.experiments.sensor_world import SensorEpisode


class LatentNavigationTests(unittest.TestCase):
    def test_learned_embedding_has_no_position_labels(self):
        model = RecurrentPredictor(seed=2, hidden=8)
        source = traces(44, steps=400, episode_length=30)
        proto = fit_prototypes(model, source, centers=4, max_points=100, seed=7)
        self.assertEqual(proto.shape, (4, 8))
        self.assertTrue(np.isfinite(proto).all())

    def test_memory_updates_after_actions(self):
        model = RecurrentPredictor(seed=2, hidden=8)
        proto = np.zeros((3, 8), dtype=np.float32)
        agent = OnlineAgent("recurrent_novelty", 7, model, proto)
        reading = SensorEpisode().reset(12)
        before = agent.h.copy()
        a = agent.act(reading)
        world = SensorEpisode()
        world.reset(12)
        new_reading, _, _ = world.step(a)
        agent.observe(new_reading)
        self.assertGreater(float(np.max(np.abs(agent.h - before))), .001)
        self.assertEqual(agent.step_index, 1)
        self.assertTrue(all(isinstance(i, int) for i in agent.last_state))

    def test_reactive_counts_are_persistent(self):
        agent = OnlineAgent("reactive", 7)
        reading = SensorEpisode().reset(2)
        for _ in range(10):
            agent.act(reading)
        self.assertTrue(agent.reactive_counts)

    def test_deterministic_small_benchmark(self):
        params = {"train_seed": 11, "train_steps": 240,
                  "epochs": 1, "hidden": 8, "centers": 3,
                  "limit": 30, "goals": 1}
        a = benchmark([31], **params)
        b = benchmark([31], **params)
        self.assertEqual(a["summary"], b["summary"])
        self.assertFalse(a["protocol"]["oracle_position_or_map_inputs"])


if __name__ == "__main__":
    unittest.main()
