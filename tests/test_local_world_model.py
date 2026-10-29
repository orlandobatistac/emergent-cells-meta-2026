"""Fast reproducibility and locality checks for local world models."""
import unittest
import numpy as np
from emergent_cells.experiments.local_world_model import (
    Spec, LocalPredictor, benchmark, target, stage,
)

class WorldModelTests(unittest.TestCase):
    def test_boolean_world_rules_are_distinct(self):
        x=np.array([0,0,0,1,1,0,1,1])
        self.assertEqual(target(x,"xor").tolist(),[0,1,1,0])
        self.assertEqual(target(x,"xnor").tolist(),[1,0,0,1])
        self.assertEqual(target(x,"and").tolist(),[0,0,0,1])
        with self.assertRaises(ValueError):target(x,"missing")

    def test_local_weights_and_no_rule_labels(self):
        spec=Spec(pairs=8)
        learner=LocalPredictor(spec,7)
        self.assertEqual(learner.p.shape,(16,8))
        self.assertFalse(hasattr(learner,"rule"))
        self.assertFalse(hasattr(learner,"target"))
        bits=np.zeros(16,dtype=int);actions=np.zeros(16,dtype=int)
        before=learner.p.copy()
        learner.step(bits,actions,np.ones(8,dtype=bool),"xor",True)
        self.assertTrue((learner.p[:2,0]!=before[:2,0]).all())
        np.testing.assert_array_equal(learner.p[:,1:],before[:,1:])

    def test_frozen_predictor_never_trains(self):
        s=Spec(pairs=8)
        x=LocalPredictor(s,3,False)
        stage(3,s,"xor",x,120,123,True)
        self.assertTrue((x.p==.5).all())

    def test_replay_and_independent_seed_scoring(self):
        spec=Spec(pairs=8,phase_steps=250,test_steps=50)
        a=benchmark([2,3],spec)
        self.assertEqual(a,benchmark([2,3],spec))
        self.assertEqual(len(a["rows"]),2)
        self.assertEqual([x["rule"] for x in a["aggregate"]],
                         ["xor","xnor","and","xor"])
        for part in a["aggregate"]:
            self.assertLess(part["test_mse"],.21)
            self.assertGreater(part["relative_gain_vs_static"],.15)

    def test_invalid_parameters(self):
        for s in [Spec(pairs=0),Spec(lr=-1),Spec(test_steps=2),
                  Spec(reward_dropout=.8)]:
            with self.assertRaises(ValueError):s.validate()

if __name__=="__main__":
    unittest.main()
