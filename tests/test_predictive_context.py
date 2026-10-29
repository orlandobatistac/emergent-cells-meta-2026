"""Integrity and predictive-memory tests for R1."""
import math
import unittest
from collections import deque

from emergent_cells.experiments.predictive_context import (
    ALPHABET, MODES, PredictiveContext, benchmark, key, signature, traces)
from emergent_cells.experiments.sensor_world import Reading


class PredictiveContextTests(unittest.TestCase):
    def test_sensor_only_no_coordinates(self):
        a = Reading((True, False, True, False),
                    (False, False, False, False), 1, -0.01)
        self.assertEqual(signature(a), 21)
        self.assertEqual(set(a.__dataclass_fields__),
                         {"walls", "nearby_target", "floor_signal", "previous_reward"})

    def test_context_uses_observation_action_history_only(self):
        history = deque([(4, 1), (9, 3)], maxlen=3)
        self.assertEqual(key(history, 6, 0, 0), ((), 6, 0))
        self.assertEqual(key(history, 6, 0, 1), (((9, 3),), 6, 0))
        self.assertEqual(key(history, 6, 0, 2), (((4, 1), (9, 3)), 6, 0))

    def test_prediction_probabilities_valid_and_pruned(self):
        data = [[(0, 1, 3), (3, 2, 4), (4, 3, 3)]] * 7
        m = PredictiveContext(3, min_visits=5).fit(data)
        self.assertGreater(m.size()[1], 0)
        self.assertEqual(m.size()[3], 0)
        for mode in MODES:
            probs = m.probs([(0, 1)], 3, 2, mode)
            self.assertEqual(len(probs), ALPHABET)
            self.assertAlmostEqual(sum(probs), 1, places=12)
            self.assertTrue(all(p > 0 for p in probs))

    def test_history_can_learn_aliased_future(self):
        # Same present local sensor/action, different next sensor signatures.
        training = ([[(7, 1, 2), (2, 3, 11)],
                     [(8, 1, 2), (2, 3, 13)]] * 25)
        m = PredictiveContext(2, min_visits=2).fit(training)
        from_7 = m.probs([(7, 1)], 2, 3, "adaptive_predictive")
        from_8 = m.probs([(8, 1)], 2, 3, "adaptive_predictive")
        self.assertGreater(from_7[11], from_7[13])
        self.assertGreater(from_8[13], from_8[11])
        coarse = m.probs([], 2, 3, "observation_only")
        self.assertAlmostEqual(coarse[11], coarse[13], places=12)

    def test_deterministic_sensor_rollouts_and_benchmark(self):
        self.assertEqual(traces(23, steps=35, episode_length=12),
                         traces(23, steps=35, episode_length=12))
        a = benchmark([12], train_steps=220, test_steps=55,
                      episode_length=20)
        b = benchmark([12], train_steps=220, test_steps=55,
                      episode_length=20)
        self.assertEqual(a["summary"], b["summary"])
        self.assertEqual(a["protocol"]["training_goal_labels"], False)
        for scenario in a["summary"].values():
            for mode in scenario.values():
                self.assertTrue(0 <= mode["accuracy"] <= 1)
                self.assertTrue(math.isfinite(mode["nll_nats"]))


if __name__ == "__main__":
    unittest.main()
