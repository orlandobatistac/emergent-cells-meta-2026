"""Causal and reproducibility tests for an engineered two-cue cooperation task."""
import unittest
import numpy as np

from emergent_cells.experiments.dual_cue import (
    GateCells, GateSpec, benchmark, trial, CONDITIONS
)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.spec = GateSpec(pairs=4, train_episodes=20, test_episodes=12)

    def test_validations(self):
        for spec in (GateSpec(pairs=1), GateSpec(message_cost=-1),
                     GateSpec(radius=0), GateSpec(train_episodes=0)):
            with self.assertRaises(ValueError):
                spec.validate()

    def test_local_cues_exclude_other_pairs(self):
        p = GateCells(self.spec, 2)
        bits = np.array([[1, -1], [-1, 1], [1, 1], [-1, -1]])
        before, messages = p.features(bits, "local")
        self.assertEqual(messages, 8)
        self.assertEqual(before[0, 1], 1)
        self.assertEqual(before[0, 2], -1)
        self.assertEqual(before[0, 3], -1)
        bits[3] = [1, -1]
        after, _ = p.features(bits, "local")
        np.testing.assert_array_equal(before[:2], after[:2])

    def test_breaking_contact_blocks_signal(self):
        p = GateCells(self.spec, 1)
        bits = np.ones((4, 2), dtype=int)
        p.y[1] += 2
        feat, messages = p.features(bits, "local")
        self.assertEqual(messages, 6)
        self.assertEqual(feat[0, 2], 0)
        self.assertEqual(feat[1, 3], 0)

    def test_no_message_masks_interaction_feature(self):
        p = GateCells(self.spec, 2)
        bits = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
        feat, n = p.features(bits, "none")
        self.assertEqual(n, 0)
        np.testing.assert_allclose(feat[:, 2:], 0)
        p.y[1::2] += 2  # Move every partner beyond physical message radius.
        feat, n = p.features(bits, "far")
        self.assertEqual(n, 0)
        np.testing.assert_allclose(feat[:, 2:], 0)

    def test_gate_requires_both_correct_without_leaking_label(self):
        p = GateCells(self.spec, 2)
        bits = np.ones((4, 2), dtype=int)
        # For identical cues, hidden parity target is action 0 at both sites.
        p.weights[:] = 0
        p.weights[:, 0] = -9
        before = p.energy.copy()
        ok = p.episode(bits, np.full(8, 0.5), "local", train=False)
        self.assertEqual(ok["opened"], 4)
        self.assertTrue((p.energy > before).all())
        self.assertEqual(ok["messages"], 8)
        # A single wrong site closes exactly its own gate.
        p.weights[1, 0] = 9
        bad = p.episode(bits, np.full(8, 0.5), "local", train=False)
        self.assertEqual(bad["opened"], 3)

    def test_local_learning_and_frozen_control(self):
        bits = np.array([[1, -1], [-1, 1], [1, 1], [-1, -1]])
        p = GateCells(self.spec, 8)
        old = p.weights.copy()
        for i in range(10):
            p.episode(bits, np.random.default_rng(i).random(8),
                      "local", train=True)
        self.assertFalse(np.array_equal(old, p.weights))
        self.assertTrue(np.isfinite(p.weights).all())
        self.assertTrue((p.energy >= 0).all())
        self.assertTrue((p.energy <= self.spec.maximum_energy).all())
        fixed = GateCells(self.spec, 8)
        old_fixed = fixed.weights.copy()
        for i in range(10):
            fixed.episode(bits, np.random.default_rng(i).random(8),
                          "frozen", train=True)
        np.testing.assert_array_equal(old_fixed, fixed.weights)

    def test_corruption_is_independent_of_local_pair(self):
        p = GateCells(self.spec, 3)
        bits = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
        a, _ = p.features(bits, "scrambled", np.random.default_rng(8))
        b, _ = p.features(bits, "scrambled", np.random.default_rng(8))
        np.testing.assert_array_equal(a, b)

    def test_benchmark_all_conditions_reproducible(self):
        a = benchmark([1, 2], self.spec)
        b = benchmark([1, 2], self.spec)
        self.assertEqual(a, b)
        self.assertEqual(set(a["summary"]), set(CONDITIONS))
        self.assertEqual(len(a["rows"]), 10)
        self.assertIsNotNone(next(x for x in a["rows"] if x["condition"] == "local")
                             ["trained_local_without_messages"])

    def test_engineered_task_reports_causal_gain(self):
        # Threshold intentionally coarse: protection against wrong wiring;
        # NOT a significance test and NOT evidence of emergent cognition.
        spec = GateSpec(pairs=32, train_episodes=800, test_episodes=120)
        real = trial(1, "local", spec)
        absent = trial(1, "none", spec)
        self.assertGreater(real["test_fraction"], absent["test_fraction"] + 0.18)
        self.assertGreater(real["test_fraction"],
                           real["trained_local_without_messages"] + 0.18)


if __name__ == "__main__":
    unittest.main()
