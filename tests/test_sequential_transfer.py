"""Regression tests for the sequential transfer research benchmark."""
import unittest
import numpy as np
from emergent_cells.experiments.sequential_transfer import (
    STRATEGIES, RULES, target, initialize, stream, trial)
from emergent_cells.experiments.dynamic_memory import GrowthSpec


class SequentialTransferTests(unittest.TestCase):
    def test_boolean_truth_tables(self):
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        self.assertEqual(target(bits, "or").tolist(), [0, 1, 1, 1])
        self.assertEqual(target(bits, "nand").tolist(), [1, 1, 1, 0])

    def test_deterministic_stream(self):
        a = list(stream(7, "probe", 3, 8))
        b = list(stream(7, "probe", 3, 8))
        for x, y in zip(a, b):
            for xi, yi in zip(x, y):
                np.testing.assert_array_equal(xi, yi)

    def test_init_strategies_do_not_alias_source(self):
        banks = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        original = banks.copy()
        for strategy in STRATEGIES:
            w = initialize(banks, np.array([0, 1]), np.array([2, 2]),
                           np.ones((2, 4)), strategy)
            w[0, 0] = -100
            np.testing.assert_array_equal(banks, original)

    def test_trial_has_independent_transfer_and_frozen_prior(self):
        spec = GrowthSpec(pairs=4, phase_steps=240)
        a = trial(19, RULES[0], spec, 240, 25, 210)
        b = trial(19, RULES[0], spec, 240, 25, 210)
        self.assertEqual(a, b)
        self.assertTrue(a["prior_banks_unchanged"])
        self.assertEqual(set(a["conditions"]), set(STRATEGIES))
        for v in a["conditions"].values():
            for key in ("first200", "last200", "post_adapt_test_accuracy"):
                self.assertGreaterEqual(v[key], 0)
                self.assertLessEqual(v[key], 1)


if __name__ == "__main__":
    unittest.main()
