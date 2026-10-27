"""Falsifiable tests for local contact slots and their ablations."""
import unittest
import numpy as np

from emergent_cells.cells import CellSpec
from emergent_cells.experiments.connectivity import (
    LinkedPopulation, LinkSpec, benchmark, trial
)
from emergent_cells.world import World, WorldSpec


class TestLinks(unittest.TestCase):
    def setUp(self):
        self.world = World(WorldSpec(width=24, height=24), seed=7)
        self.spec = CellSpec(count=4, communication_radius=5)
        self.link_spec = LinkSpec(max_links=2, radius=5)

    def test_local_only_and_bounded_degree(self):
        pop = LinkedPopulation(self.world.spec, self.spec, seed=1,
                               link_spec=self.link_spec, plastic_links=True)
        pop.x[:] = [1, 2, 3, 15]
        pop.y[:] = [1, 1, 1, 15]
        pop.last_harvest[:] = [0.12] * 4
        signal, edges = pop.perceive(self.world)
        self.assertEqual(edges, 6)
        self.assertTrue((pop.links[:3] < 3).all())
        self.assertTrue((pop.links[3] == -1).all())
        np.testing.assert_allclose(signal[3, :, 2], 0)
        self.assertLessEqual((pop.links >= 0).sum(axis=1).max(), 2)

    def test_link_expires_when_neighbor_leaves_range(self):
        pop = LinkedPopulation(self.world.spec, self.spec, seed=1,
                               link_spec=self.link_spec, plastic_links=True)
        pop.x[:] = [1, 2, 15, 18]
        pop.y[:] = [1, 1, 15, 18]
        pop.perceive(self.world)
        self.assertIn(1, pop.links[0])
        pop.x[1] = 12
        pop.perceive(self.world)
        self.assertNotIn(1, pop.links[0])
        self.assertGreaterEqual(pop.contacts_expired, 1)

    def test_equal_slots_fixed_and_adaptive(self):
        fixed = LinkedPopulation(self.world.spec, self.spec, seed=9,
                                 link_spec=self.link_spec, plastic_links=False)
        adapt = LinkedPopulation(self.world.spec, self.spec, seed=9,
                                 link_spec=self.link_spec, plastic_links=True)
        fixed.x[:] = adapt.x[:] = [1, 2, 3, 4]
        fixed.y[:] = adapt.y[:] = [1, 1, 1, 1]
        fixed.last_harvest[:] = adapt.last_harvest[:] = [0.12] * 4
        fixed.perceive(self.world)
        adapt.perceive(self.world)
        np.testing.assert_array_equal(fixed.links, adapt.links)
        np.testing.assert_array_equal(fixed.trust, adapt.trust)

    def test_fixed_trust_does_not_update(self):
        pop = LinkedPopulation(self.world.spec, self.spec, seed=3,
                               link_spec=self.link_spec, plastic_links=False)
        pop.x[:] = [1, 2, 3, 4]
        pop.y[:] = [1, 1, 1, 1]
        pop.last_harvest[:] = 0.12
        pop.step(self.world, np.random.default_rng(4))
        self.assertEqual(pop.trust_update_count, 0)
        np.testing.assert_allclose(pop.trust, 1.0)

    def test_plastic_link_receives_only_local_credit(self):
        # Place three cells in range; the fourth stays isolated.
        pop = LinkedPopulation(self.world.spec, self.spec, seed=2,
                               link_spec=self.link_spec, plastic_links=True)
        pop.x[:] = [1, 2, 3, 15]
        pop.y[:] = [1, 1, 1, 15]
        pop.last_harvest[:] = [0.12] * 4
        pop.step(self.world, np.random.default_rng(4))
        self.assertGreater(pop.trust_update_count, 0)
        np.testing.assert_allclose(pop.trust[3], 1.0)
        self.assertTrue(np.isfinite(pop.trust).all())
        self.assertGreaterEqual(pop.trust.min(), 0)
        self.assertLessEqual(pop.trust.max(), 2.5)

    def test_communication_ablations_deterministic(self):
        for condition in ("none", "broadcast", "fixed", "adaptive"):
            first = trial(2, steps=12, condition=condition, shock_step=6, width=24)
            second = trial(2, steps=12, condition=condition, shock_step=6, width=24)
            self.assertEqual(first, second)
            self.assertTrue(0 <= first["survival_fraction"] <= 1)
        self.assertEqual(trial(2, 12, "none", width=24)["neighbor_messages"], 0)

    def test_benchmark_full_factorial(self):
        result = benchmark([1, 3], steps=8, width=24)
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(set(result["summary"]),
                         {"none", "broadcast", "fixed", "adaptive"})

    def test_invalid_link_parameters(self):
        for cfg in (LinkSpec(max_links=0), LinkSpec(radius=0),
                    LinkSpec(minimum_trust=1.0)):
            with self.assertRaises(ValueError):
                cfg.validate()


if __name__ == "__main__":
    unittest.main()
