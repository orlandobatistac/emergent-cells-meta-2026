"""Held-out scenario separation, local-reward transfer and replay controls."""
import copy
import unittest
import numpy as np

from emergent_cells.experiments.learned_curiosity import (
    ALL_SCENARIOS, HOLDOUT_SCENARIOS, SCENARIOS, CuriositySpec,
    LocalCuriosity, train_policy,
)
from emergent_cells.experiments.transfer_stress import (
    METHODS, benchmark, evaluate,
)


class TransferStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = CuriositySpec(
            pairs=8, pretrain=220, meta_episodes=25,
            meta_horizon=22, eval_horizon=100
        )
        cls.intrinsic = train_policy(5, cls.spec)
        plain_spec = CuriositySpec(
            **{**vars(cls.spec), "intrinsic_weight": 0.}
        )
        cls.plain = train_policy(5, plain_spec)

    def test_holdouts_not_in_training(self):
        self.assertFalse(set(HOLDOUT_SCENARIOS) & set(SCENARIOS))
        self.assertEqual(len(ALL_SCENARIOS), len(SCENARIOS)+len(HOLDOUT_SCENARIOS))
        self.assertEqual(len(HOLDOUT_SCENARIOS), 6)

    def test_communication_same_across_training_methods(self):
        np.testing.assert_array_equal(self.intrinsic[1], self.plain[1])
        np.testing.assert_array_equal(self.intrinsic[2], self.plain[2])

    def test_frozen_train_q(self):
        before = self.intrinsic[0].q.copy()
        for scenario in HOLDOUT_SCENARIOS:
            row = evaluate(5, scenario, self.spec, self.intrinsic,
                           self.plain, "learned_intrinsic")
            self.assertIn("by_case_correctness", row)
        np.testing.assert_array_equal(before, self.intrinsic[0].q)

    def test_case_frequency_and_budget(self):
        row = evaluate(5, "mixed_change_10_scarcity_01", self.spec,
                       self.intrinsic, self.plain, "manual_uncertain")
        self.assertEqual(sum(row["by_case_counts"].values()),
                         self.spec.pairs * self.spec.eval_horizon)
        for value in row["by_case_correctness"].values():
            self.assertTrue(0 <= value <= 1)
        self.assertGreaterEqual(row["requests"], 0)
        self.assertLessEqual(row["requests"], self.spec.pairs*2*self.spec.eval_horizon)

    def test_guard_reduces_repeat_probes_without_q_training(self):
        q = self.intrinsic[0].q.copy()
        before = evaluate(5, 'scarcity_01', self.spec, self.intrinsic, self.plain, 'learned_intrinsic')
        after = evaluate(5, 'scarcity_01', self.spec, self.intrinsic, self.plain, 'learned_intrinsic_guard2')
        self.assertLessEqual(after['requests'], before['requests'])
        np.testing.assert_array_equal(self.intrinsic[0].q, q)

    def test_explicit_no_probe_baseline(self):
        row = evaluate(5, "changed_10", self.spec,
                       self.intrinsic, self.plain, "passive")
        self.assertEqual(row["requests"], 0)
        self.assertEqual(row["probe_cost"], 0.)

    def test_replay_exactly_and_all_modes(self):
        a = benchmark([2], self.spec)
        self.assertEqual(a, benchmark([2], self.spec))
        self.assertEqual(len(a["rows"]), len(HOLDOUT_SCENARIOS) * len(METHODS))
        for scenario in HOLDOUT_SCENARIOS:
            self.assertEqual(set(a["summary"][scenario]), set(METHODS))

    def test_scenario_unknown_rejected(self):
        learner = LocalCuriosity(self.spec, 3)
        with self.assertRaises(ValueError):
            learner.run_episode(self.intrinsic[1], self.intrinsic[2],
                                ["secret_answer"]*self.spec.pairs,
                                np.random.default_rng(1),
                                steps=5, train=False, epsilon=0)
        with self.assertRaises(ValueError):
            evaluate(5, "stable", self.spec, self.intrinsic,
                     self.plain, "learned_intrinsic")


if __name__ == "__main__":
    unittest.main()
