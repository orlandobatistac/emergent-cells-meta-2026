"""Causal tests for local coordinated exploration and matched controls."""
import unittest
import numpy as np
from emergent_cells.experiments.active_probes import (
    ProbeSpec, MODES, SCENARIOS, pretrain, evaluate, benchmark,
    binary_entropy, expected_info,
)

class CounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=ProbeSpec(pairs=20, pretrain=3200, horizon=420)
        cls.model=pretrain(11,cls.spec)

    def test_invalid_parameters(self):
        for s in (ProbeSpec(pairs=2),ProbeSpec(pretrain=3),
                  ProbeSpec(horizon=2),ProbeSpec(request_cost=-.1),
                  ProbeSpec(entropy_threshold=5)):
            with self.assertRaises(ValueError):
                s.validate()

    def test_entropy_and_information_gain(self):
        self.assertAlmostEqual(float(binary_entropy(np.array([[.5,.5,0]]))[0]),1.)
        p=np.array([[.02,.49,.49],[1.,0.,0.]])
        diff=expected_info(p,np.array([.012,.012,.92])) - (
            expected_info(p,np.array([.92,.012,.012])))
        self.assertGreater(diff[0],.1)
        self.assertAlmostEqual(float(diff[1]),0.,places=7)

    def test_locally_learned_one_bit_code_is_used(self):
        send,recv=self.model
        self.assertEqual(send.shape,(40,2))
        self.assertEqual(recv.shape,(40,4))
        self.assertTrue(np.isfinite(send).all())
        self.assertTrue(np.isfinite(recv).all())

    def test_active_intervention_against_passive(self):
        passive=evaluate(11,"passive","changed_11",self.spec,self.model)
        active=evaluate(11,"information_gain","changed_11",self.spec,self.model)
        self.assertLess(passive["critical_11_success"],.10)
        self.assertGreater(active["critical_11_success"],.80)
        self.assertGreater(active["pairs_both_discovered_fraction"],.75)
        self.assertGreater(active["information_request_bits"],0)

    def test_scarcity_is_not_automatically_new_memory(self):
        result=evaluate(11,"information_gain","scarcity_11",self.spec,self.model)
        self.assertLessEqual(result["false_new_memory"],2)
        self.assertGreater(result["critical_11_success"],.85)

    def test_passive_control_spends_no_probe_cost(self):
        result=evaluate(11,"passive","stable",self.spec,self.model)
        self.assertEqual(result["coordinated_probe_count"],0)
        self.assertEqual(result["synthetic_probe_cost"],0)
        self.assertEqual(result["information_request_bits"],0)

    def test_global_rule_change_is_learnable(self):
        result=evaluate(11,"information_gain","global_change",self.spec,self.model)
        self.assertGreater(result["pair_success_rate"],.75)
        self.assertGreater(result["inverted_slot_count"],20)

    def test_replay_and_full_factorial_controls(self):
        s=ProbeSpec(pairs=8,pretrain=250,horizon=130)
        one=benchmark([7,8],s)
        self.assertEqual(one,benchmark([7,8],s))
        self.assertEqual(len(one["rows"]),2*len(MODES)*len(SCENARIOS))
        for x in one["rows"]:
            self.assertTrue(0<=x["pair_success_rate"]<=1)
            self.assertTrue(0<=x["critical_11_success"]<=1)

    def test_unsupported_conditions(self):
        with self.assertRaises(ValueError):
            evaluate(11,"made_up","stable",self.spec,self.model)
        with self.assertRaises(ValueError):
            evaluate(11,"passive","unknown",self.spec,self.model)
        with self.assertRaises(ValueError):
            benchmark([1,1],self.spec)

if __name__=="__main__":
    unittest.main()
