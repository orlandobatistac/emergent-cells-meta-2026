"""Gradient and information-boundary checks for memory-dependent policy."""
import unittest
import numpy as np

from emergent_cells.experiments.delayed_goal import DelayedCuePolicy, benchmark
from emergent_cells.experiments.policy_gradient import (
    ActorMemory, LinearActor, episode as grid_episode)
from emergent_cells.experiments.recurrent_state import RecurrentPredictor
from emergent_cells.experiments.sensor_world import SensorEpisode


class DelayedGoalTests(unittest.TestCase):
    def test_cue_is_visible_only_initially(self):
        policy=DelayedCuePolicy(seed=1)
        cues=np.array([0,1])
        obs=policy.inputs(cues,12)
        self.assertNotEqual(obs[0,0].tolist(),obs[1,0].tolist())
        np.testing.assert_array_equal(obs[0,1:],obs[1,1:])
        self.assertEqual(obs.shape,(2,12,3))

    def test_training_learns_reward_only_memory_and_erasure(self):
        policy=DelayedCuePolicy(seed=7)
        train=policy.fit(seed=29,batches=140,batch=64,length=8)
        self.assertGreater(train["final_25_batches"],.88)
        full=policy.evaluate(seed=111,length=8,cases=500)
        erased=policy.evaluate(seed=111,length=8,cases=500,erased=True)
        inverted=policy.evaluate(seed=111,length=8,cases=500,inverted_cue=True)
        self.assertGreater(full["accuracy"],.95)
        self.assertLess(erased["accuracy"],.6)
        self.assertLess(inverted["accuracy"],.06)
        self.assertLess(policy.parameter_bytes(),2000)

    def test_protocol_reproducible_and_new_length(self):
        a=benchmark([88,89],batches=80,batch=64,eval_cases=50)
        b=benchmark([88,89],batches=80,batch=64,eval_cases=50)
        self.assertEqual(a["summary"],b["summary"])
        self.assertEqual(a["protocol"]["training_signal"],
                         "terminal +1 correct or -1 incorrect, sampled action only")


class RecurrentGoalRLTests(unittest.TestCase):
    def test_actor_masks_illegal_actions(self):
        a=LinearActor(seed=11)
        feat=np.zeros(79,dtype=np.float32)
        p=a.probabilities(feat,[1,3],np.zeros(4,dtype=np.float32))
        self.assertAlmostEqual(p.sum(),1.,places=9)
        self.assertEqual(p[0],0.)
        self.assertEqual(p[2],0.)

    def test_actor_updates_weights_from_reward(self):
        actor=LinearActor(seed=4)
        feat=np.zeros(79,dtype=np.float32)
        feat[2]=1.
        records=[]
        for target in (0,1,0,1):
            probs=actor.probabilities(feat,[0,1],np.zeros(4))
            records.append([(feat,(probs.astype(np.float32),target),
                             1. if target==0 else -1.)])
        old=actor.w.copy()
        actor.update(records)
        self.assertGreater(np.max(np.abs(actor.w-old)),1e-8)

    def test_policy_observation_boundary(self):
        w=SensorEpisode(limit=20)
        reading=w.reset(18,goals=1)
        model=RecurrentPredictor(seed=3,hidden=32)
        centers=np.zeros((4,32),np.float32)
        agent=ActorMemory("recurrent_actor",model,centers,seed=5)
        action,info,prev,state=agent.choose(reading,LinearActor(seed=3))
        self.assertIn(action,(0,1,2,3))
        self.assertEqual(info[0].shape,(79,))
        self.assertFalse(hasattr(reading,"position"))
        nxt,_,_=w.step(action)
        agent.update_state(prev,action)
        self.assertGreater(float(np.max(np.abs(agent.h))),0.)


if __name__=="__main__":
    unittest.main()
