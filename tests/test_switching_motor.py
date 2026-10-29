"""Hidden motor-remapping adaptation and detection tests."""
import unittest

from emergent_cells.experiments.switching_motor import (
    SwitchedMotorWorld, RecentMotorEvidence, episode, benchmark)
from emergent_cells.experiments.learned_motor_code import PERMUTATIONS


class SwitchingMotorTests(unittest.TestCase):
    def test_mapping_changes_after_private_action_index(self):
        first=(0,1,2,3)
        second=(2,3,0,1)
        world=SwitchedMotorWorld(first,second,seed=7,switch_at=2)
        observation=world.reset(13)
        for _ in range(2):
            observation,contact,_,done=world.step(0)
            self.assertEqual(world._mapping,first)
            if done:break
        observation,contact,_,done=world.step(0)
        self.assertEqual(world._mapping,second)
        self.assertFalse(hasattr(observation,"mapping"))

    def test_windowed_detector_recovers_new_code(self):
        first=(0,1,2,3)
        second=(2,3,0,1)
        detector=RecentMotorEvidence(first,contact_noise=.01,
                                      window=48,margin=3,min_samples=8)
        # Independent randomized local wall patterns identify motor codes;
        # no position or correct action labels are supplied to detector.
        import random
        rng=random.Random(42)
        flipped=False
        for step in range(90):
            command=rng.randrange(4)
            wall=tuple(bool(rng.randrange(2)) for _ in range(4))
            contact=wall[second[command]]
            flipped |= detector.observe(wall,command,contact,step+1)
        self.assertTrue(flipped)
        self.assertEqual(detector.estimate,second)

    def test_oracle_can_use_new_mapping_without_global_position(self):
        first=(0,1,2,3)
        second=(2,3,0,1)
        a=episode(33,first,second,method="oracle_switch",
                  switch_at=12,rounds=2)
        self.assertLessEqual(a["steps"],160)
        self.assertTrue(a["switch_occurred"])

    def test_online_detector_and_frozen_controls_deterministic(self):
        first=(0,1,2,3)
        second=(2,3,0,1)
        kwargs=dict(switch_at=12,window=15,margin=3,rounds=2)
        for method in ("online_reset","online_keep_map","frozen_mapping"):
            self.assertEqual(
                episode(44,first,second,method=method,**kwargs),
                episode(44,first,second,method=method,**kwargs))

    def test_small_benchmark_and_information_contract(self):
        a=benchmark([61,62],calibration_steps=40,switch_at=16,
                    window=15,margin=3,rounds=2)
        self.assertTrue(a["protocol"]["motor_change_not_announced_to_policy"])
        self.assertTrue(a["protocol"]["known_action_permutation_hypothesis_class"])
        self.assertIn("stable",a["summary"])
        self.assertEqual(len(a["records"]),2*2*4)


if __name__=="__main__":
    unittest.main()
