"""R1 learned recurrent state: protocol, gradient, and regression tests."""
import unittest
import numpy as np

from emergent_cells.experiments.recurrent_state import (
    GeneratedMaze, RecurrentPredictor, generated_traces, packed, trial)
from emergent_cells.experiments.predictive_context import traces


class RecurrentStateTests(unittest.TestCase):
    def test_packed_from_sensor_traces_not_coordinates(self):
        episodes = [[(1, 3, 7), (7, 1, 6)], [(9, 0, 4)]]
        x, target, mask = packed(episodes)
        self.assertEqual(x.shape, (2, 2, 36))
        self.assertEqual(target.tolist(), [[7, 6], [4, 0]])
        self.assertEqual(mask.tolist(), [[1, 1], [1, 0]])
        self.assertEqual(int(x[0, 0].sum()), 2)
        self.assertEqual(int(x[1, 1].sum()), 0)

    def test_recurrent_gradient_matches_finite_difference(self):
        model = RecurrentPredictor(seed=7, hidden=5)
        data = [[(1, 2, 5), (5, 0, 8), (8, 3, 1)]]
        x, target, mask = packed(data)
        _, analytic = model._gradients(x, target, mask)
        eps = 1e-3
        for name, index in (("wx", (1, 2)), ("wh", (1, 1)),
                            ("wy", (2, 5)), ("bh", (1,))):
            original = float(model.p[name][index])
            model.p[name][index] = original + eps
            lp, _ = model._gradients(x, target, mask)
            model.p[name][index] = original - eps
            lm, _ = model._gradients(x, target, mask)
            model.p[name][index] = original
            self.assertAlmostEqual(
                (lp-lm) / (2*eps), float(analytic[name][index]), delta=.004,
                msg=name)

    def test_training_reduces_loss_on_familiar_pattern(self):
        training = [[(0, 1, 3), (3, 2, 4), (4, 3, 2),
                     (2, 0, 0)]] * 32
        model = RecurrentPredictor(seed=3, hidden=8)
        model.fit(training, epochs=12, batch_size=8, lr=.015, seed=9)
        self.assertLess(model.training_history[-1],
                        model.training_history[0] * .5)
        self.assertEqual(len(model.probabilities(training[0])), 4)
        reset = model.probabilities(training[0], clear_history=True)
        full = model.probabilities(training[0])
        np.testing.assert_allclose(reset[0], full[0], rtol=1e-5, atol=1e-6)
        self.assertGreater(float(np.abs(reset[1:] - full[1:]).max()), 1e-6)
        self.assertLess(model.parameter_bytes(), 5000)

    def test_procedural_world_variants_without_agent_labels(self):
        a = GeneratedMaze(42)
        b = GeneratedMaze(43)
        self.assertNotEqual(a.walls, b.walls)
        self.assertEqual(a.walls, GeneratedMaze(42).walls)
        trace = generated_traces(48, steps=35, episode_length=20)
        self.assertEqual(trace, generated_traces(48, steps=35, episode_length=20))
        for episode in trace:
            for record in episode:
                self.assertEqual(len(record), 3)
                self.assertTrue(all(isinstance(v, int) for v in record))

    def test_short_same_data_benchmark_and_no_oracle(self):
        a = trial(19, train_steps=320, test_steps=80, epochs=1, hidden=8,
                  episode_length=20)
        b = trial(19, train_steps=320, test_steps=80, epochs=1, hidden=8,
                  episode_length=20)
        for scenario in ("stable", "changed", "novel"):
            self.assertEqual(a["results"][scenario], b["results"][scenario])
            for method in ("observation_only", "history_table", "trainable_recurrent"):
                self.assertGreaterEqual(a["results"][scenario][method]["accuracy"], 0)
                self.assertLessEqual(a["results"][scenario][method]["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
