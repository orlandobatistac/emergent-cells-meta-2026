"""Safety boundaries and controls for sensor-driven goal-seeking policies."""
import unittest
import numpy as np
from emergent_cells.experiments.beacon_navigation import (
    BeaconEpisode, BeaconReading, SignalAgent, episode, benchmark)
from emergent_cells.experiments.recurrent_state import RecurrentPredictor


class GoalSeekingTests(unittest.TestCase):
    def test_observation_no_hidden_coordinates(self):
        scene = BeaconEpisode(seed=24)
        obs = scene.reset(13)
        self.assertEqual(set(obs.__dataclass_fields__), {
            "walls", "nearby_target", "floor_signal",
            "previous_reward", "proximity"})
        self.assertFalse(hasattr(obs, "position"))
        self.assertFalse(hasattr(obs, "goal"))
        self.assertTrue(0 <= obs.proximity <= 1)

    def test_masked_signal_is_zero_and_action_interface_stays_local(self):
        scene = BeaconEpisode(seed=11, beacon_enabled=False)
        obs = scene.reset(18)
        self.assertEqual(obs.proximity, 0)
        for action in (0, 1, 2, 3):
            obs, reward, done = scene.step(action)
            self.assertEqual(obs.proximity, 0)
            self.assertIsInstance(done, bool)

    def test_signal_is_deterministic_under_seeded_actions(self):
        a = BeaconEpisode(seed=42)
        b = BeaconEpisode(seed=42)
        self.assertEqual(a.reset(8), b.reset(8))
        for act in (2, 0, 1, 3, 2, 2, 0):
            self.assertEqual(a.step(act), b.step(act))

    def test_reward_learning_and_eval_freeze(self):
        model = RecurrentPredictor(seed=11, hidden=8)
        probes = np.zeros((2, 8), dtype=np.float32)
        agent = SignalAgent("signal_latent", 11, model, probes, training=True)
        scene = BeaconEpisode(seed=11)
        obs = scene.reset(17)
        agent.reset_episode(17)
        for _ in range(15):
            action = agent.act(obs)
            obs, _, done = scene.step(action)
            agent.observe(obs)
            if done:
                break
        training_count = sum(agent.learned_counts.values())
        self.assertGreater(training_count, 0)
        counts_before = dict(agent.learned_counts)
        episode("signal_latent", 88, model, probes, warm=agent,
                limit=25, training=False)
        self.assertEqual(dict(agent.learned_counts), counts_before)

    def test_benchmark_reproducible_with_tiny_budget(self):
        opts = dict(train_seed=19, train_steps=250, epochs=1,
                    hidden=8, centers=3, warm_episodes=2,
                    limit=25, goals=1)
        a = benchmark([9], **opts)
        b = benchmark([9], **opts)
        self.assertEqual(a["summary"], b["summary"])
        self.assertFalse(a["protocol"]["global_state_inputs"])
        self.assertFalse(a["protocol"]["central_planner"])
        for condition in a["summary"].values():
            for score in condition.values():
                self.assertTrue(0 <= score["finish_rate"] <= 1)


if __name__ == "__main__":
    unittest.main()
