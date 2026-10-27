"""Validation: no privileged rule labels, local evidence, deterministic replay."""
import unittest
import numpy as np
from emergent_cells.experiments.dynamic_memory import GrowthSpec, rng_stream
from emergent_cells.experiments.robust_drift import (
    GuardedCells, METHODS, run_sequence, run_stationary
)


class RobustDriftTests(unittest.TestCase):
    def setUp(self):
        self.spec = GrowthSpec(pairs=12, phase_steps=500,
                               revisit_steps=200, dropout=.08)

    def test_method_validation_and_lazy_memory(self):
        with self.assertRaises(ValueError):
            GuardedCells(self.spec, 3, "unknown")
        for method in METHODS:
            p=GuardedCells(self.spec,3,method)
            self.assertEqual(p.weights.shape,(24,1,4))
            self.assertTrue((p.allocated==1).all())

    def test_recent_sample_is_per_cell_not_global(self):
        p=GuardedCells(self.spec,3,"credible_guard")
        allzero=np.zeros(p.n)
        allzero[2]=1
        p.last_slot=np.zeros(p.n,dtype=int)
        p._detect(allzero)
        self.assertEqual(int(p.hit_count[2,0]),1)
        self.assertEqual(int(p.hit_count[0,0]),0)
        self.assertEqual(int(p.samples[0,0]),1)
        self.assertEqual(int(p.samples[2,0]),1)

    def test_absolute_guard_does_not_alarm_on_moderate_dropout(self):
        p=GuardedCells(self.spec,3,"absolute_floor")
        p.age[:]=1000
        p.peak[:]=.92
        p.score[:]=.73
        self.assertFalse(p._novelty_alarm(np.ones(p.n,dtype=bool),p.spec).any())
        p.score[:]=.22
        self.assertTrue(p._novelty_alarm(np.ones(p.n,dtype=bool),p.spec).all())

    def test_credible_guard_requires_statistical_evidence(self):
        p=GuardedCells(self.spec,3,"credible_guard")
        p.age[:]=1000
        p.peak[:]=.95
        p.score[:]=.23
        p.samples[:,0]=100
        p.hit_count[:,0]=20
        self.assertTrue(p._novelty_alarm(np.ones(p.n,dtype=bool),p.spec).all())
        p.hit_count[:,0]=73
        self.assertFalse(p._novelty_alarm(np.ones(p.n,dtype=bool),p.spec).any())

    def test_actions_cannot_access_change_labels(self):
        p=GuardedCells(self.spec,5,"credible_guard")
        inputs=next(rng_stream(9,1,p.n,"A"))
        with self.assertRaises(TypeError):
            p.episode(inputs,"xor",train=False,detect=True,world_label="A")

    def test_conditional_surprise_finds_selective_failure_not_uniform_noise(self):
        p=GuardedCells(self.spec,3,"conditional_signature")
        p.age[:]=1000
        p.peak[:]=.92
        p.score[:]=.70
        p.samples[:,0]=100
        p.hit_count[:,0]=70
        p.cond_count[:,0,:]=40
        p.cond_hits[:,0,:]=32
        # One observation context became nearly impossible, others remain good.
        p.cond_hits[:,0,0]=0
        flagged=p._novelty_alarm(np.ones(p.n,dtype=bool),p.spec)
        self.assertTrue(flagged.all())
        # Uniform reward erosion should not create task-specific novelty.
        p.cond_hits[:,0,:]=26
        self.assertFalse(p._novelty_alarm(np.ones(p.n,dtype=bool),p.spec).any())

    def test_explicit_local_availability_sensor_is_extra_information(self):
        sensed=GuardedCells(self.spec,8,"availability_sensor")
        reward_only=GuardedCells(self.spec,8,"conditional_signature")
        for population in (sensed,reward_only):
            population.last_slot=np.zeros(population.n,dtype=int)
            population.last_available=np.zeros(population.n,dtype=bool)
        empty=np.zeros(sensed.n)
        sensed._detect(empty)
        reward_only._detect(empty)
        # Only the additional availability sensor excludes absent resources
        # from being interpreted as mistakes; policies still see no task label.
        self.assertTrue((sensed.score > reward_only.score).all())

    def test_old_baseline_and_controls_replay(self):
        s=GrowthSpec(pairs=8,phase_steps=400,revisit_steps=200)
        a=run_sequence(11,"credible_guard",s)
        self.assertEqual(a,run_sequence(11,"credible_guard",s))
        self.assertEqual([x["stage"] for x in a["stages"]],
                         ["A","B","C","returnA","returnB"])
        x=run_stationary(11,"credible_guard",.20,s,stress_steps=300)
        self.assertEqual(x,run_stationary(11,"credible_guard",.20,s,stress_steps=300))

    def test_false_alert_is_explicitly_measured(self):
        s=GrowthSpec(pairs=16,phase_steps=3400,revisit_steps=350)
        base=run_stationary(11,"relative_ema",.20,s,stress_steps=1000)
        robust=run_stationary(11,"credible_guard",.20,s,stress_steps=1000)
        self.assertGreater(base["false_memory_cells"],0)
        self.assertEqual(robust["false_memory_cells"],0)


if __name__=="__main__":
    unittest.main()
