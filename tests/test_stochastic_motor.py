"""Tests of stochastic motor dynamics and no-oracle sensor fusion."""
import unittest

from emergent_cells.experiments.stochastic_motor import (
    MotionReading, SlipWorld, LearnedMotorReliability, infer_move,
    calibrate, rollout, benchmark, METHODS)


class StochasticMotorTests(unittest.TestCase):
    def test_public_observations_exclude_pose_and_hidden_dynamics(self):
        world = SlipWorld(12, rates=(.45,.6,.75,.9))
        reading = world.reset(22)
        self.assertEqual(set(reading.__dataclass_fields__), {
            "walls", "nearby_target", "floor_signal",
            "previous_reward", "motion_tick"})
        self.assertFalse(hasattr(reading,"position"))
        self.assertFalse(hasattr(reading,"motor_success_rates"))
        self.assertEqual(len(reading.walls),4)

    def test_zero_slip_has_exact_motion_inference_with_perfect_sensors(self):
        world = SlipWorld(32, rates=(1.,1.,1.,1.),
                          motion_noise=0.,floor_noise=0.)
        reading=world.reset(44)
        for t in range(30):
            action=t % 4
            prior=reading
            reading,done=world.step(action)
            expected=not prior.walls[action]
            self.assertEqual(reading.motion_tick,expected)
            self.assertEqual(prior.floor_signal!=reading.floor_signal,expected)
            if done:break

    def test_no_move_when_all_commands_fail(self):
        world = SlipWorld(12,rates=(0.,0.,0.,0.),
                          motion_noise=0.,floor_noise=0.)
        reading = world.reset(11)
        first=world._env._position  # Test-harness-only hidden truth.
        for _ in range(20):
            previous=reading
            reading,done=world.step(0)
            self.assertFalse(reading.motion_tick)
            self.assertEqual(reading.floor_signal,previous.floor_signal)
            self.assertEqual(world._env._position,first)
        self.assertFalse(done)

    def test_noise_corrected_probability_estimator(self):
        model=LearnedMotorReliability(tick_error=.2,smoothing=2.)
        # Synthetic local movement sensor outcome statistics; not true labels.
        for i in range(1000):
            model.observe(0,True,(i%10)<8)
            model.observe(1,True,(i%10)<3)
            model.observe(2,False,True)
        result=model.estimate()
        self.assertAlmostEqual(result[0],.995,places=2)
        self.assertAlmostEqual(result[1],(0.3-.2)/.6,delta=.02)
        self.assertEqual(model.samples[2],0)

    def test_sensor_evidence_changes_posterior(self):
        a=MotionReading((False,True,False,True),(False,)*4,0,0.,False)
        b=MotionReading((False,True,False,True),(False,)*4,1,0.,True)
        c=MotionReading((False,True,False,True),(False,)*4,0,0.,False)
        self.assertTrue(infer_move(a,b,prior=.5,tick_error=.2,
                                   floor_error=.1))
        self.assertFalse(infer_move(a,c,prior=.5,tick_error=.2,
                                    floor_error=.1))

    def test_calibration_uses_no_coordinate_supervision(self):
        rates=(.45,.6,.75,.9)
        learned=calibrate(17,rates,actions=400,tick_error=.08,
                          floor_error=.15)
        estimated=learned.estimate()
        self.assertTrue(all(0<v<1 for v in estimated))
        self.assertGreater(sum(learned.samples),100)
        self.assertLess(sum(abs(a-b) for a,b in zip(rates,estimated))/4,.15)

    def test_rollout_and_map_vs_single_sensor_controls(self):
        rates=(.5,.65,.75,.9)
        for method in METHODS:
            result=rollout(41,rates,method=method,learned_rates=rates,
                           tick_error=.15,floor_error=.1,rounds=2)
            self.assertLessEqual(result["steps"],160)
            self.assertTrue(0<=result["movement_prediction_accuracy"]<=1)
            self.assertGreaterEqual(result["relative_pose_error"],0)

    def test_reproducible_paired_procedural_results(self):
        kwargs=dict(calibration_actions=45,tick_error=.1,
                    floor_error=.1,rounds=2,procedural=True)
        a=benchmark([53,54],**kwargs)
        b=benchmark([53,54],**kwargs)
        self.assertEqual(a["summary"],b["summary"])
        self.assertTrue(a["protocol"]["no_global_pose_or_map_input"])
        self.assertEqual(len(a["records"]),2*len(METHODS))
        self.assertIn("procedural",a["summary"])


if __name__=="__main__":
    unittest.main()
