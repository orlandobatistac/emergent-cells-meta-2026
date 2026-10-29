"""Experimental controls for local motor-command calibration."""
import unittest

from emergent_cells.experiments.learned_motor_code import (
    MotorCodeWorld, MotorCodeLearner, PERMUTATIONS, learn_mapping,
    play, benchmark)
from emergent_cells.experiments.sensor_world import SensorEpisode


class LearnedMotorCodeTests(unittest.TestCase):
    def test_motor_permutation_hypothesis_space(self):
        self.assertEqual(len(PERMUTATIONS), 24)
        self.assertEqual(len(set(PERMUTATIONS)), 24)
        for candidate in PERMUTATIONS:
            self.assertEqual(set(candidate), set(range(4)))

    def test_only_local_wall_reading_and_contact_train_calibrator(self):
        mapping = (2, 3, 0, 1)
        learner = MotorCodeLearner()
        world = MotorCodeWorld(mapping, seed=13)
        observation = world.reset(9)
        self.assertEqual(set(observation.__dataclass_fields__),
                         {"walls", "nearby_target", "floor_signal",
                          "previous_reward"})
        self.assertFalse(hasattr(observation, "position"))
        for action in range(4):
            wall_before = observation.walls
            observation, contact, _, _ = world.step(action)
            learner.observe(wall_before, action, contact)
        self.assertEqual(learner.samples, 4)

    def test_known_motor_mapping_recovered_without_position_supervision(self):
        for i, mapping in enumerate(PERMUTATIONS):
            learner = learn_mapping(4100+i, mapping,
                                    calibration_steps=100, noise=.05)
            self.assertEqual(learner.estimate(), mapping,
                             msg=f"failed mapping {mapping}")
            self.assertGreater(learner.posterior_confidence(), .9)

    def test_command_mapping_affects_real_action(self):
        mapping = (2, 3, 0, 1)
        world = MotorCodeWorld(mapping, seed=17)
        reading = world.reset(11)
        start = world._env._position  # Test harness privilege only.
        for command, direction in enumerate(mapping):
            world = MotorCodeWorld(mapping, seed=17)
            reading = world.reset(11)
            start = world._env._position
            next_reading, bump, _, _ = world.step(command)
            expected = world._env._maze.step(start, direction)
            self.assertEqual(world._env._position, expected)
            self.assertEqual(bump, reading.walls[direction])

    def test_equal_motor_mapping_yields_identical_navigation(self):
        mapping = (3, 2, 0, 1)
        a = play(61, mapping, mapping)
        b = play(61, mapping, mapping)
        self.assertEqual(a, b)
        self.assertLessEqual(a["steps"], 160)

    def test_deterministic_paired_benchmark_and_noise(self):
        opts = dict(calibration_steps=55, noise=.1, rounds=3)
        a = benchmark([101, 102], **opts)
        b = benchmark([101, 102], **opts)
        self.assertEqual(a["summary"], b["summary"])
        self.assertEqual(a["calibration"], b["calibration"])
        self.assertTrue(a["protocol"]["separate_calibration_budget"])
        self.assertTrue(a["protocol"]["compass_aligned_wall_sensor"])

    def test_bad_hypothesis_rejected(self):
        with self.assertRaises(ValueError):
            MotorCodeWorld((0, 0, 1, 2))
        with self.assertRaises(ValueError):
            MotorCodeLearner(error_rate=.5)


if __name__ == "__main__":
    unittest.main()
