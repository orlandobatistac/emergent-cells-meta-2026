"""Contract tests for the agent-facing sensor-only interface."""
import unittest
from dataclasses import fields

from emergent_cells.experiments.sensor_world import (
    Reading, SensorEpisode, benchmark, choose)


class SensorWorldTests(unittest.TestCase):
    def test_reading_has_no_absolute_position_or_destination(self):
        self.assertEqual({f.name for f in fields(Reading)},
                         {"walls", "nearby_target", "floor_signal", "previous_reward"})
        self.assertFalse(hasattr(SensorEpisode().reset(13), "position"))
        self.assertFalse(hasattr(SensorEpisode().reset(13), "destination"))

    def test_deterministic_same_seed_and_action_trace(self):
        a, b = SensorEpisode(), SensorEpisode()
        self.assertEqual(a.reset(42), b.reset(42))
        for action in (1, 3, 1, 3, 2, 0, 0, 1):
            self.assertEqual(a.step(action), b.step(action))

    def test_ambiguous_sensor_signature(self):
        episode = SensorEpisode()
        episode.reset(12)
        readings = {}
        # Evaluation may inspect hidden coordinates, but only readings enter a policy.
        for position in episode._maze.states():
            episode._position = position
            obs = episode._read()
            signature = (obs.walls, obs.nearby_target, obs.floor_signal)
            if signature in readings and readings[signature] != position:
                return
            readings[signature] = position
        self.fail("World should contain aliased sensory states")

    def test_sparse_target_reward_and_completion(self):
        episode = SensorEpisode(limit=50)
        episode.reset(13, goals=1)
        destination = episode._targets[0]
        adjacent = [(x, a) for x in episode._maze.states()
                    for a in range(4)
                    if episode._maze.step(x, a) == destination and x != destination]
        self.assertTrue(adjacent)
        s, action = adjacent[0]
        episode._position = s
        reading = episode._read()
        self.assertTrue(reading.nearby_target[action])
        new_reading, reward, done = episode.step(action)
        self.assertTrue(done)
        self.assertGreater(reward, 0)
        self.assertTrue(episode.metrics()["completed"])

    def test_action_validation(self):
        episode = SensorEpisode()
        episode.reset(7)
        for invalid in (-1, 4, True, "1"):
            with self.assertRaises(ValueError):
                episode.step(invalid)

    def test_baselines_deterministic_and_bounded(self):
        a = benchmark([11, 12], goals=2, limit=30)
        b = benchmark([11, 12], goals=2, limit=30)
        self.assertEqual(a, b)
        self.assertEqual(len(a["records"]), 8)
        for row in a["summary"].values():
            self.assertGreaterEqual(row["complete_rate"], 0)
            self.assertLessEqual(row["complete_rate"], 1)


if __name__ == "__main__":
    unittest.main()
