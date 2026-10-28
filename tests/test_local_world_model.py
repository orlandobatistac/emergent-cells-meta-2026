"""First falsifiable tests for a local predictive reward model."""
import unittest
import numpy as np
from emergent_cells.experiments.local_world_model import (
    ModelSpec, PrivatePredictors, MODES, CASES, run, benchmark,
)

class LocalWorldModelTests(unittest.TestCase):
    def test_invalid_ranges(self):
        for x in (ModelSpec(pairs=3),ModelSpec(steps=5),
                  ModelSpec(pulse_probability=0),
                  ModelSpec(probe_cost=-.1)):
            with self.assertRaises(ValueError):x.validate()

    def test_private_predictor_learns_conditional_action_reward(self):
        p=PrivatePredictors(4)
        ids=np.array([0,1])
        slots=np.array([3,1])
        alternate=np.ones(2,dtype=int)
        p.update(ids,slots,alternate,np.array([1,0]))
        posterior=p.prediction()
        self.assertGreater(posterior[0,3,1],.25)
        self.assertLess(posterior[1,1,1],.25)
        self.assertEqual(p.visits[2].sum(),0)
        self.assertEqual(p.visits[3].sum(),0)

    def test_independent_cells_do_not_share_reward_tables(self):
        p=PrivatePredictors(12)
        p.update(np.array([0]),np.array([2]),np.array([0]),np.array([0]))
        np.testing.assert_array_equal(p.visits[1:],np.zeros_like(p.visits[1:]))
        self.assertEqual(p.visits[0,2,0],1)
        self.assertEqual(p.hits[0,2,0],0)

    def test_local_conditioned_prediction_bounded(self):
        p=PrivatePredictors(8)
        for _ in range(10):
            p.update(np.array([0,3]),np.array([3,2]),
                     np.array([1,0]),np.array([0,1]))
        a=p.prediction()
        self.assertTrue(np.all((a>0)&(a<1)))
        self.assertEqual(p.visits[0,3,1],10)
        self.assertEqual(p.visits[3,2,0],10)

    def test_pilot_changed_rule_and_scarcity_control(self):
        spec=ModelSpec(pairs=16,sender_training=1900,steps=250)
        active=run(11,"changed_11","predictive_progress",spec)
        passive=run(11,"changed_11","passive",spec)
        scarcity=run(11,"scarcity_11","predictive_progress",spec)
        self.assertGreater(active["case11_correct"],.75)
        self.assertLess(passive["case11_correct"],.15)
        self.assertGreater(scarcity["case11_correct"],.80)
        self.assertGreater(active["requests"],0)
        self.assertEqual(passive["requests"],0)

    def test_seed_reproducible(self):
        spec=ModelSpec(pairs=8,sender_training=500,steps=140)
        a=benchmark([3,4],spec)
        self.assertEqual(a,benchmark([3,4],spec))
        self.assertEqual(len(a["rows"]),2*len(MODES)*len(CASES))
        self.assertIn("reward_prediction_brier",a["rows"][0])
        for row in a["rows"]:
            self.assertTrue(0<=row["correct"]<=1)
            self.assertTrue(0<=row["case11_correct"]<=1)
            self.assertTrue(0<=row["reward_prediction_brier"]<=1)

if __name__=="__main__":
    unittest.main()
