"""Falsifiable tests for local third-rule memory creation and baselines."""
import unittest
import numpy as np
from emergent_cells.experiments.dynamic_memory import (
    GrowthSpec, GrowingCells, rule_targets, rng_stream, run_trial, benchmark
)


class DynamicMemoryTests(unittest.TestCase):
    def setUp(self):
        self.spec = GrowthSpec(pairs=8, phase_steps=400, dropout=.08)

    def test_experimental_bounds(self):
        for spec in (GrowthSpec(pairs=0), GrowthSpec(dropout=-1),
                     GrowthSpec(relative_trigger=0), GrowthSpec(max_banks=8)):
            with self.assertRaises(ValueError):
                spec.validate()

    def test_world_rule_is_distinct_and_external(self):
        cues = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        self.assertEqual(rule_targets(cues, "xor").tolist(), [0, 1, 1, 0])
        self.assertEqual(rule_targets(cues, "xnor").tolist(), [1, 0, 0, 1])
        self.assertEqual(rule_targets(cues, "and").tolist(), [0, 0, 0, 1])
        with self.assertRaises(ValueError):
            rule_targets(cues, "unsupported")

    def test_no_growth_at_initialization(self):
        cells = GrowingCells(self.spec, 3, "grow")
        self.assertEqual(cells.allocated.tolist(), [1] * cells.n)
        self.assertEqual(cells.growth_events, 0)
        self.assertTrue((cells.active == 0).all())
        self.assertEqual(cells.weights.shape, (cells.n, 1, 4))

    def test_lazy_storage_expands_only_on_demand(self):
        p = GrowingCells(self.spec, 31, "grow")
        self.assertEqual(p.weights.shape, (p.n, 1, 4))
        self.assertEqual(p.baseline.shape, (p.n, 1))
        initial = p.weights[:, 0].copy()
        # One cell asks for another logical bank. The vectorized array
        # grows for the WHOLE population; other cells remain logically 1-bank.
        select = np.zeros(p.n, dtype=bool)
        select[0] = True
        p._allocate(select)
        self.assertEqual(p.weights.shape, (p.n, 2, 4))
        self.assertEqual(p.baseline.shape, (p.n, 2))
        self.assertEqual(p.allocated[0], 2)
        self.assertTrue((p.allocated[1:] == 1).all())
        np.testing.assert_array_equal(p.weights[:, 0], initial)
        select[:] = False
        select[1] = True
        p._allocate(select)
        self.assertEqual(p.weights.shape[1], 2)
        select[:] = False
        select[0] = True
        p._allocate(select)
        self.assertEqual(p.weights.shape[1], 3)
        self.assertEqual(p.allocated[0], 3)
        np.testing.assert_array_equal(p.weights[:, 0], initial)

    def test_private_policy_and_task_rule_no_direct_target_features(self):
        cells = GrowingCells(self.spec, 5, "grow")
        bits, send, action, noise = next(rng_stream(7, 1, cells.n, "A"))
        initial_weights = cells.weights.copy()
        # With identical randomness, the gate rule changes reward only.
        a = GrowingCells(self.spec, 5, "grow")
        b = GrowingCells(self.spec, 5, "grow")
        a.episode((bits, send, action, noise), "xor",
                  train=False, detect=False)
        b.episode((bits, send, action, noise), "and",
                  train=False, detect=False)
        np.testing.assert_array_equal(a.weights, b.weights)
        np.testing.assert_array_equal(initial_weights, a.weights)
        np.testing.assert_array_equal(a.active, b.active)

    def test_locality_in_private_weight_updates(self):
        p, q = GrowingCells(self.spec, 9, "grow"), GrowingCells(self.spec, 9, "grow")
        input1 = next(rng_stream(9, 1, p.n, "A"))
        input2 = tuple(a.copy() for a in input1)
        input2[0][2:] = 1 - input2[0][2:]
        p.episode(input1, "xor", train=True, detect=True, train_sender=True)
        q.episode(input2, "xor", train=True, detect=True, train_sender=True)
        np.testing.assert_allclose(p.sender[:2], q.sender[:2])
        np.testing.assert_allclose(p.weights[:2], q.weights[:2])
        np.testing.assert_allclose(p.score[:2], q.score[:2])

    def test_bound_and_probe_existing_bank(self):
        p = GrowingCells(self.spec, 3, "grow")
        all_cells = np.ones(p.n, dtype=bool)
        p._allocate(all_cells)
        self.assertTrue((p.allocated == 2).all())
        self.assertEqual(p.weights.shape[1], 2)
        p._allocate(all_cells)
        self.assertTrue((p.allocated == 3).all())
        self.assertEqual(p.weights.shape[1], 3)
        p._allocate(all_cells)
        self.assertEqual(p.allocated.max(), 3)

    def test_seed_replay_and_controls(self):
        spec = GrowthSpec(pairs=8, phase_steps=350, test_steps=80, revisit_steps=250)
        a = benchmark([1, 2], spec)
        self.assertEqual(a, benchmark([1, 2], spec))
        self.assertEqual(len(a["rows"]), 10)
        self.assertEqual(set(a["summary"]),
                         {"grow", "cap2", "single", "oracle", "fast_detector"})
        for item in a["rows"]:
            self.assertEqual(len(item["stages"]), 5)
            for stage in item["stages"]:
                self.assertTrue(0 <= stage["tail_success"] <= 1)
                self.assertLessEqual(stage["max_allocated_banks"], 3)
                self.assertEqual(set(stage["case_accuracy_last200"]),
                                 {"00", "01", "10", "11"})

    def test_growth_on_known_pilot(self):
        # Pilot regression, NOT independently held-out confirmation.
        spec = GrowthSpec(pairs=32, phase_steps=4500, dropout=.08)
        growing = run_trial(11, "grow", spec)
        capped = run_trial(11, "cap2", spec)
        oracle = run_trial(11, "oracle", spec)
        self.assertGreater(growing["stages"][2]["tail_success"], .82)
        self.assertGreater(growing["stages"][2]["case_accuracy_last200"]["11"], .75)
        self.assertGreater(growing["stages"][2]["mean_allocated_banks"], 2.8)
        self.assertLess(capped["stages"][2]["tail_success"], .6)
        self.assertEqual(oracle["stages"][2]["mean_allocated_banks"], 3)


if __name__ == "__main__":
    unittest.main()
