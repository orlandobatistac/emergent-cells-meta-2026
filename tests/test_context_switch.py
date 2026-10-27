"""Scientific controls for A(XOR)->B(XNOR)->A task switching."""
import unittest
import numpy as np
from emergent_cells.experiments.context_switch import (
    SwitchCells, SwitchSpec, trial, benchmark, CONDITIONS
)


class TestSwitch(unittest.TestCase):
    def setUp(self):
        self.spec = SwitchSpec(pairs=8, train_a=25, train_b=35, test_steps=12)

    def test_strict_parameter_validation(self):
        for bad in (SwitchSpec(pairs=2), SwitchSpec(train_a=0),
                    SwitchSpec(train_b=0), SwitchSpec(test_steps=0),
                    SwitchSpec(message_cost=-1)):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_world_rule_reverses_without_leaking_answer_to_cells(self):
        pop = SwitchCells(self.spec, 4)
        pop.receiver[:] = -12
        cue = np.zeros(pop.n, dtype=int)
        sample = np.full(pop.n, .5)
        self.assertEqual(pop.episode(cue, sample, sample, rule=0,
                                     context=0, condition="single_policy",
                                     train_sender=False, train_receiver=False), 1)
        self.assertEqual(pop.episode(cue, sample, sample, rule=1,
                                     context=1, condition="single_policy",
                                     train_sender=False, train_receiver=False), 0)

    def test_bank_separation_with_same_local_inputs(self):
        pop = SwitchCells(self.spec, 1)
        pop.sender[:, 0] = 12
        pop.sender[:, 1] = 0
        pop.receiver[:, 0, :] = -12
        pop.receiver[:, 1, :] = 12
        cue = np.zeros(pop.n, dtype=int)
        draws = np.full(pop.n, .5)
        # The policy sees context ONLY by selecting its private receiver bank.
        success_a = pop.episode(cue, draws, draws, rule=0, context=0,
                                condition="context_memory",
                                train_sender=False, train_receiver=False)
        success_b = pop.episode(cue, draws, draws, rule=1, context=1,
                                condition="context_memory",
                                train_sender=False, train_receiver=False)
        success_blind = pop.episode(cue, draws, draws, rule=1, context=1,
                                    condition="blind_memory",
                                    train_sender=False, train_receiver=False)
        self.assertEqual((success_a, success_b, success_blind), (1, 1, 0))

    def test_blocking_messages_really_blocks_reception(self):
        pop = SwitchCells(self.spec, 5)
        cue = np.ones(pop.n, dtype=int)
        draws = np.full(pop.n, .5)
        pop.episode(cue, draws, draws, rule=0, context=0,
                    condition="no_messages", train_sender=False,
                    train_receiver=False)
        self.assertEqual(pop.messages_sent, pop.n)
        self.assertEqual(pop.messages_received, 0)

    def test_b_phase_freezes_sender_for_every_condition(self):
        for condition in CONDITIONS:
            r = trial(2, condition, self.spec)
            self.assertEqual(r["sender_weight_change_after_A"], 0.0)

    def test_memory_preserves_a_weights_exactly(self):
        result = trial(3, "context_memory", self.spec)
        self.assertEqual(result["a_decoder_weight_change_after_A"], 0)

    def test_reset_and_single_are_not_just_a_stored_memory(self):
        reset = trial(3, "reset_on_switch", self.spec)
        self.assertGreater(reset["a_decoder_weight_change_after_A"], 0)
        single = trial(3, "single_policy", self.spec)
        self.assertGreater(single["a_decoder_weight_change_after_A"], 0)

    def test_deterministic_and_full_trial_grid(self):
        r1 = benchmark([1, 2], self.spec)
        r2 = benchmark([1, 2], self.spec)
        self.assertEqual(r1, r2)
        self.assertEqual(len(r1["results"]), 2 * len(CONDITIONS))
        self.assertEqual(set(r1["aggregate"]), set(CONDITIONS))
        for row in r1["results"]:
            for field in ("a_before", "b_before", "b_after",
                          "a_return_no_training"):
                self.assertTrue(0 <= row[field] <= 1)

    def test_learning_b_and_remembering_a_in_pilot(self):
        spec = SwitchSpec(pairs=40, train_a=4200, train_b=3500, test_steps=120)
        memory = trial(11, "context_memory", spec)
        reset = trial(11, "reset_on_switch", spec)
        self.assertGreater(memory["a_before"], 0.85)
        self.assertGreater(memory["b_after"], 0.85)
        self.assertGreater(memory["a_return_no_training"], 0.85)
        self.assertGreater(reset["b_after"], 0.80)
        self.assertLess(reset["a_return_no_training"], 0.35)


if __name__ == "__main__":
    unittest.main()
