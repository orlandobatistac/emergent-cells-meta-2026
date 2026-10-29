"""Contract, ablation and deterministic tests for local-value cells."""
import unittest

from emergent_cells.experiments.local_value_cells import (
    LocalValueCells, run_benchmark)
from emergent_cells.experiments.sensor_world import SensorEpisode


class LocalValueCellTests(unittest.TestCase):
    def test_policy_receives_no_coordinates(self):
        observation = SensorEpisode().reset(13)
        self.assertEqual(set(observation.__dataclass_fields__),
                         {"walls", "nearby_target", "floor_signal",
                          "previous_reward"})
        self.assertFalse(hasattr(observation, "position"))
        self.assertFalse(hasattr(observation, "target_coordinates"))

    def test_relative_position_reconstructed_only_from_actions(self):
        for changed in (False, True):
            for seed in (3, 11, 24, 71):
                world = SensorEpisode(changed=changed, limit=160)
                reading = world.reset(seed)
                start = world._position  # Privileged truth in test harness only.
                agent = LocalValueCells(seed)
                for _ in range(70):
                    action = agent.act(reading)
                    reading, _, done = world.step(action)
                    agent.observe(reading)
                    absolute = world._position  # Never passed to the agent.
                    self.assertEqual(
                        agent.position, (absolute[0] - start[0],
                                         absolute[1] - start[1]))
                    if done:
                        break

    def test_reward_resets_phase_counts_without_goal_identity(self):
        world = SensorEpisode()
        reading = world.reset(18)
        agent = LocalValueCells(18)
        action = agent.act(reading)
        # Positive reward is a normal public sensor event.
        from dataclasses import replace
        reward_reading = replace(reading, previous_reward=.99)
        agent.observe(reward_reading)
        self.assertEqual(agent.completed_objectives, 1)
        self.assertEqual(sum(agent.goal_phase_visits.values()), 0)
        self.assertIn(action, range(4))

    def test_freezing_internal_odometry_is_a_causal_ablation(self):
        world = SensorEpisode()
        reading = world.reset(18)
        agent = LocalValueCells(18, freeze_memory=True)
        for _ in range(8):
            action = agent.act(reading)
            reading, _, done = world.step(action)
            agent.observe(reading)
            if done:
                break
        self.assertEqual(agent.position, (0, 0))
        self.assertEqual(len(agent.open_actions), 1)

    def test_determinism_with_seeded_experiment(self):
        a = run_benchmark([101, 102], rounds=3)
        b = run_benchmark([101, 102], rounds=3)
        self.assertEqual(a["summary"], b["summary"])
        self.assertEqual(
            a["protocol"]["relative_odometry"], "programmed motor update")
        self.assertTrue(a["protocol"]["centralized_python_scheduler"])
        for scene in a["summary"].values():
            for scores in scene.values():
                self.assertGreaterEqual(scores["success_rate"], 0)
                self.assertLessEqual(scores["success_rate"], 1)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            LocalValueCells(1, rounds=-1)
        with self.assertRaises(ValueError):
            LocalValueCells(1, decay=1.5)
        with self.assertRaises(ValueError):
            LocalValueCells(1, frontier_reward=0)


if __name__ == "__main__":
    unittest.main()
