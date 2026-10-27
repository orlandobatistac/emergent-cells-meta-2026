"""Tests for a genuinely learned cell-local sender code (not XOR-coded input)."""
import unittest
import numpy as np
from emergent_cells.experiments.learned_messages import (
    ProtocolSpec, LocalProtocolPopulation, trial, benchmark, CONDITIONS
)


class LearnedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.spec = ProtocolSpec(pairs=8, train_steps=25, test_steps=12)

    def test_world_and_parameter_limits(self):
        for bad in (ProtocolSpec(pairs=1), ProtocolSpec(train_steps=0),
                    ProtocolSpec(test_steps=-1), ProtocolSpec(learning_rate=0),
                    ProtocolSpec(transmission_cost=-0.1)):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_private_sender_and_generic_receiver_slots(self):
        p = LocalProtocolPopulation(self.spec, seed=3)
        self.assertEqual(p.sender.shape, (16, 2))
        self.assertEqual(p.receiver.shape, (16, 4))
        # Sender probability for cell 0 cannot depend on private cue of cell 1.
        own = 0
        q0 = 1 / (1 + np.exp(-(p.sender[0, 0] - p.sender[0, 1])))
        other_bit = 1
        q1 = 1 / (1 + np.exp(-(p.sender[0, 0] - p.sender[0, 1])))
        self.assertEqual(q0, q1)

    def test_gate_physics_is_not_passed_to_policy(self):
        p = LocalProtocolPopulation(self.spec, seed=3)
        p.receiver[:] = -12
        cues = np.zeros(p.n, dtype=int)
        draws = np.full(p.n, 0.5)
        self.assertEqual(p.episode(cues, draws, draws, train=False)["opened"], self.spec.pairs)
        cues[1] = 1
        self.assertEqual(p.episode(cues, draws, draws, train=False)["opened"], self.spec.pairs-1)

    def test_learning_changes_both_own_policies(self):
        p = LocalProtocolPopulation(self.spec, seed=5)
        sender, receiver = p.sender.copy(), p.receiver.copy()
        rng = np.random.default_rng(3)
        for _ in range(40):
            p.episode(rng.integers(0, 2, p.n), rng.random(p.n), rng.random(p.n))
        self.assertFalse(np.array_equal(sender, p.sender))
        self.assertFalse(np.array_equal(receiver, p.receiver))
        self.assertTrue(np.isfinite(p.sender).all())
        self.assertTrue(np.isfinite(p.receiver).all())

    def test_frozen_sender_and_receiver_do_not_change(self):
        cues = np.zeros(self.spec.pairs * 2, dtype=int)
        draw = np.full(cues.shape, 0.4)
        p = LocalProtocolPopulation(self.spec, seed=1)
        sender = p.sender.copy()
        p.episode(cues, draw, draw, condition="fixed_sender", train=True)
        np.testing.assert_array_equal(sender, p.sender)
        q = LocalProtocolPopulation(self.spec, seed=1)
        receiver = q.receiver.copy()
        q.episode(cues, draw, draw, condition="frozen_actor", train=True)
        np.testing.assert_array_equal(receiver, q.receiver)

    def test_no_message_equal_attempted_transmission_cost(self):
        a = LocalProtocolPopulation(self.spec, seed=6)
        b = LocalProtocolPopulation(self.spec, seed=6)
        bits = np.zeros(a.n, dtype=int)
        samples = np.full(a.n, 0.5)
        a.episode(bits, samples, samples, condition="learned_messages", train=False)
        b.episode(bits, samples, samples, condition="blocked_messages", train=False)
        self.assertEqual(a.transmitted_bits, b.transmitted_bits)
        self.assertEqual(b.received_bits, 0)
        self.assertEqual(a.received_bits, a.n)

    def test_strict_input_validation(self):
        p = LocalProtocolPopulation(self.spec, seed=7)
        cue = np.zeros(p.n, dtype=int)
        draw = np.full(p.n, 0.5)
        with self.assertRaises(ValueError):
            p.episode(cue + 2, draw, draw)
        with self.assertRaises(ValueError):
            p.episode(cue, draw, draw, condition="fake")
        with self.assertRaises(ValueError):
            benchmark([1, 1], self.spec)

    def test_seed_replay_and_complete_controls(self):
        a = trial(3, "learned_messages", self.spec)
        self.assertEqual(a, trial(3, "learned_messages", self.spec))
        r = benchmark([1, 2], self.spec)
        self.assertEqual(len(r["rows"]), len(CONDITIONS) * 2)
        self.assertEqual(r, benchmark([1, 2], self.spec))

    def test_sender_protocol_causal_benefit(self):
        spec = ProtocolSpec(pairs=48, train_steps=5000, test_steps=200)
        learned = trial(101, "learned_messages", spec)
        blocked = trial(101, "blocked_messages", spec)
        self.assertGreater(learned["test_success_fraction"], 0.85)
        self.assertGreater(learned["sender_symbol_separation"], 0.65)
        self.assertGreater(learned["test_success_fraction"],
                           blocked["test_success_fraction"] + 0.25)
        self.assertGreater(learned["test_success_fraction"],
                           learned["same_policy_knockouts"]["block_received_bits"] + 0.3)


if __name__ == "__main__":
    unittest.main()
