"""Unannounced context changes detected by each cell's OWN reward history."""
import unittest
import numpy as np
from emergent_cells.experiments.context_switch import SwitchSpec
from emergent_cells.experiments.reward_surprise import (
    SurpriseCells, trial_hidden, benchmark
)


class RewardSurpriseTests(unittest.TestCase):
    def setUp(self):
        self.spec = SwitchSpec(pairs=8, train_a=25, train_b=30, test_steps=10)

    def test_detector_reads_no_context_label(self):
        cells = SurpriseCells(self.spec, 7)
        self.assertEqual(cells.active_bank.shape, (16,))
        self.assertTrue((cells.active_bank == 0).all())
        cues = np.zeros(cells.n, dtype=int)
        draws = np.full(cells.n, .5)
        with self.assertRaises(TypeError):
            cells.episode_hidden(cues, draws, draws, context=1, world_rule=1,
                                  train_sender=False, train_receiver=False, detect=True)

    def test_reward_failure_locally_triggers_change(self):
        p = SurpriseCells(self.spec, 5)
        # Old bank confidently produces action 0 for all inputs.
        p.receiver[:, 0, :] = -12.0
        p.peak[:, 0] = .95
        p.score[:, 0] = .95
        p.visits[:, 0] = 400
        cues = np.zeros(p.n, dtype=int)
        draws = np.full(p.n, .5)
        for _ in range(36):
            p.episode_hidden(cues, draws, draws, world_rule=1, detect=True,
                             train_sender=False, train_receiver=False)
        self.assertTrue((p.active_bank == 1).all())
        self.assertTrue((p.switches == 1).all())
        self.assertTrue((p.receiver[:, 0, :] == -12).all())

    def test_no_switch_if_detector_is_paused(self):
        p = SurpriseCells(self.spec, 5)
        p.receiver[:, 0, :] = -12
        p.peak[:, 0] = .95
        p.score[:, 0] = .95
        p.visits[:, 0] = 400
        cues = np.zeros(p.n, dtype=int)
        draws = np.full(p.n, .5)
        for _ in range(36):
            p.episode_hidden(cues, draws, draws, world_rule=1, detect=False,
                             train_sender=False, train_receiver=False)
        self.assertFalse(p.switches.any())

    def test_distant_pair_does_not_update_focal_local_detector(self):
        a, b = SurpriseCells(self.spec, 9), SurpriseCells(self.spec, 9)
        cues_a = np.zeros(a.n, dtype=int)
        cues_b = cues_a.copy()
        cues_b[2:] = 1
        draws = np.full(a.n, .5)
        a.episode_hidden(cues_a, draws, draws, world_rule=0,
                         detect=True, train_sender=True, train_receiver=True)
        b.episode_hidden(cues_b, draws, draws, world_rule=0,
                         detect=True, train_sender=True, train_receiver=True)
        np.testing.assert_array_equal(a.sender[:2], b.sender[:2])
        np.testing.assert_array_equal(a.receiver[:2], b.receiver[:2])
        np.testing.assert_array_equal(a.score[:2], b.score[:2])
        np.testing.assert_array_equal(a.active_bank[:2], b.active_bank[:2])

    def test_deterministic_replay_and_control_structure(self):
        one = trial_hidden(5, self.spec)
        self.assertEqual(one, trial_hidden(5, self.spec))
        report = benchmark([3, 4], self.spec)
        self.assertEqual(len(report["rows"]["reward_surprise"]), 2)
        self.assertEqual(report, benchmark([3, 4], self.spec))
        self.assertIn("provided_context", report["aggregate"])
        self.assertIn("blind_memory", report["aggregate"])

    def test_frozen_policy_retention_and_adaptive_recovery(self):
        p = trial_hidden(21, SwitchSpec(pairs=40, train_a=4200,
                                        train_b=3200, test_steps=100))
        self.assertGreater(p["a_before"], .85)
        self.assertGreater(p["b_after"], .85)
        self.assertGreater(p["a_after_detection_no_weight_training"], .85)
        self.assertGreater(p["a_return_pair_detection_fraction"], .90)
        self.assertLess(p["a_return_detection_latency_mean"], 80)
        self.assertEqual(p["sender_max_delta_after_a"], 0)
        # During the detection delay, B failures may transiently update bank A.
        # Retention is PERFORMANCE, not bit-identical retention of weights.

    def test_detection_only_does_not_retrain_cell_weights(self):
        cells = SurpriseCells(self.spec, 9)
        cells.peak[:, 0] = .98
        cells.score[:, 0] = .98
        cells.visits[:, 0] = 350
        cells.receiver[:, 0, :] = -12
        sender_before = cells.sender.copy()
        receiver_before = cells.receiver.copy()
        bits = np.zeros(cells.n, dtype=int)
        samples = np.full(cells.n, .5)
        for _ in range(30):
            cells.episode_hidden(bits, samples, samples, world_rule=1,
                                 train_sender=False, train_receiver=False,
                                 detect=True)
        self.assertGreater(cells.switches.sum(), 0)
        np.testing.assert_array_equal(cells.sender, sender_before)
        np.testing.assert_array_equal(cells.receiver, receiver_before)


if __name__ == "__main__":
    unittest.main()
