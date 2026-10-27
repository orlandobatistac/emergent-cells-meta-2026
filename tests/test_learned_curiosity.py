"""Scientific controls for reward-learned local probe-vs-exploit policy."""
import unittest
import numpy as np
from emergent_cells.experiments.learned_curiosity import (
    CuriositySpec, LocalCuriosity, local_state, train_policy,
    evaluate_policy, benchmark, SCENARIOS,
)


class LearnedCuriosityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=CuriositySpec(pairs=12,pretrain=2100,meta_episodes=120,
                              meta_horizon=60,eval_horizon=200)
        cls.trained=train_policy(11,cls.spec)

    def test_reject_invalid_parameters(self):
        for bad in (CuriositySpec(pairs=2),CuriositySpec(pretrain=0),
                    CuriositySpec(meta_episodes=0),CuriositySpec(alpha=0),
                    CuriositySpec(initial_epsilon=-.1)):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_state_uses_only_private_beliefs(self):
        x=np.array([[.90,.05,.05],[.50,.25,.25],
                    [.20,.40,.40],[.05,.90,.05],[.05,.05,.90]])
        self.assertEqual(local_state(x).tolist(),[0,1,2,3,4])

    def test_private_tables_train_without_truth_labels(self):
        model=self.trained[0]
        self.assertEqual(model.q.shape,(24,6,2))
        self.assertTrue(np.isfinite(model.q).all())
        self.assertGreater(np.max(np.abs(model.q-.25)), .05)

    def test_repeated_failed_probe_has_distinct_local_state(self):
        agent=LocalCuriosity(self.spec,9)
        self.assertEqual(agent.failed_probes.shape,(24,4))
        self.assertTrue((agent.failed_probes==0).all())
        # The context is private observed failures, not world identity.
        agent.failed_probes[0,0]=3
        self.assertNotEqual(agent.failed_probes[0,0],agent.failed_probes[1,0])

    def test_frozen_untrained_control(self):
        x=evaluate_policy(11,"changed_11",self.spec,self.trained)
        cold=evaluate_policy(11,"changed_11",self.spec,self.trained,untrained=True)
        self.assertEqual(x["method"],"locally_learned_probe")
        self.assertEqual(cold["method"],"untrained_probe_policy")
        # A 50% probe-everywhere untrained policy can accidentally solve
        # a flipped rule, so changed-rule correctness alone is misleading.
        # The learned decision should save resources when probes are futile.
        rich=evaluate_policy(11,"scarcity_11",self.spec,self.trained)
        cold_scarce=evaluate_policy(11,"scarcity_11",self.spec,self.trained,untrained=True)
        self.assertGreater(rich["critical"],cold_scarce["critical"]+.30)
        self.assertLess(rich["requests"],cold_scarce["requests"]*.10)

    def test_scarcity_no_mass_false_memories(self):
        x=evaluate_policy(11,"scarcity_11",self.spec,self.trained)
        self.assertGreater(x["critical"],.70)
        self.assertLess(x["false_corrections"],8)

    def test_training_never_receives_correct_answer(self):
        # No action-ground-truth labels appear in the learner state. The world
        # simulator receives scenario IDs; the cell Q learner only net reward.
        x=LocalCuriosity(self.spec,9)
        self.assertFalse(hasattr(x,"world_rule"))
        self.assertFalse(hasattr(x,"correct_answer"))
        self.assertEqual(x.q.shape[-1],2)

    def test_replay_across_same_seed(self):
        m2=train_policy(11,self.spec)
        np.testing.assert_allclose(self.trained[0].q,m2[0].q)
        a=evaluate_policy(11,"changed_01",self.spec,self.trained)
        b=evaluate_policy(11,"changed_01",self.spec,self.trained)
        self.assertEqual(a,b)

    def test_private_q_updates_are_local_to_pair(self):
        a=LocalCuriosity(self.spec,9)
        b=LocalCuriosity(self.spec,9)
        send,recv=self.trained[1],self.trained[2]
        scenes_a=["stable"]*self.spec.pairs
        scenes_b=["stable"]+["changed_11"]*(self.spec.pairs-1)
        a.run_episode(send,recv,scenes_a,np.random.default_rng(99),
                      steps=25,train=True,epsilon=.2)
        b.run_episode(send,recv,scenes_b,np.random.default_rng(99),
                      steps=25,train=True,epsilon=.2)
        np.testing.assert_array_equal(a.q[:2],b.q[:2])
        np.testing.assert_array_equal(a.belief[:2],b.belief[:2])

    def test_benchmark_controls_and_new_input_cases(self):
        sp=CuriositySpec(pairs=8,pretrain=200,meta_episodes=24,
                         meta_horizon=25,eval_horizon=110)
        x=benchmark([2,3],sp)
        self.assertEqual(set(x["aggregate"]),set(SCENARIOS))
        self.assertEqual(len(x["rows"]),2*(5*6+2*2))
        self.assertIn("information_gain",x["aggregate"]["changed_11"])
        self.assertIn("untrained_probe_policy",x["aggregate"]["changed_01"])
        self.assertEqual(x,benchmark([2,3],sp))


if __name__=="__main__":
    unittest.main()
