"""Behavioral tests of local boundaries and deterministic world mechanics."""
import unittest
import numpy as np

from emergent_cells.cells import CellPopulation, CellSpec
from emergent_cells.simulation import RunSpec, compare, run
from emergent_cells.world import World, WorldSpec


class TestWorld(unittest.TestCase):
    def test_invalid_spec_rejected(self):
        with self.assertRaises(ValueError):
            World(WorldSpec(width=0), seed=1)
        with self.assertRaises(ValueError):
            World(WorldSpec(regeneration=-0.2), seed=1)

    def test_finite_shared_harvest(self):
        world = World(WorldSpec(width=12, height=12, resource_patches=0,
                                hazard_patches=0, regeneration=0), seed=1)
        world.resource.fill(0)
        world.resource[4, 5] = 0.1
        xs = np.array([5, 5, 5, 0])
        ys = np.array([4, 4, 4, 0])
        got = world.harvest(xs, ys, np.array([True, True, True, False]))
        np.testing.assert_allclose(got[:3], [0.1 / 3] * 3)
        self.assertEqual(got[3], 0)
        self.assertAlmostEqual(float(world.resource[4, 5]), 0.0, places=12)
        self.assertGreaterEqual(world.resource.min(), 0.0)

    def test_probe_is_local(self):
        world = World(WorldSpec(width=24, height=24), seed=2)
        pop = CellPopulation(world.spec, CellSpec(count=1), seed=1)
        pop.x[:] = 1
        pop.y[:] = 1
        before, _ = pop.perceive(world)
        world.resource[15, 15] += 0.5
        after, _ = pop.perceive(world)
        np.testing.assert_array_equal(before, after)

    def test_shock_applies_once(self):
        world = World(WorldSpec(shock_step=3), seed=5)
        before = world.capacity.copy()
        world.step(1)
        self.assertFalse(world.shock_applied)
        world.step(3)
        self.assertTrue(world.shock_applied)
        np.testing.assert_array_equal(world.capacity,
                                      np.roll(before, world.spec.width // 2, axis=1))
        world.step(3)
        np.testing.assert_array_equal(world.capacity,
                                      np.roll(before, world.spec.width // 2, axis=1))


class TestCells(unittest.TestCase):
    def test_no_messages_outside_radius(self):
        world = World(WorldSpec(width=24, height=24), seed=3)
        pop = CellPopulation(world.spec, CellSpec(count=2, communication_radius=2), seed=1)
        pop.x[:] = [1, 15]
        pop.y[:] = [1, 15]
        pop.last_harvest[:] = [0.1, 0.1]
        sensed, edges = pop.perceive(world)
        self.assertEqual(edges, 0)
        np.testing.assert_allclose(sensed[:, :, 2], 0.0)

    def test_neighbor_messages_only_locally(self):
        world = World(WorldSpec(width=24, height=24), seed=3)
        pop = CellPopulation(world.spec, CellSpec(count=2, communication_radius=2), seed=1)
        pop.x[:] = [1, 2]
        pop.y[:] = [1, 1]
        pop.last_harvest[:] = [0.0, 0.1]
        sensed, edges = pop.perceive(world)
        self.assertEqual(edges, 2)
        self.assertGreater(sensed[0, 1, 2], 0)

    def test_frozen_weights_unchanged(self):
        world = World(WorldSpec(width=24, height=24), seed=5)
        pop = CellPopulation(world.spec, CellSpec(count=12), seed=6)
        before = pop.weights.copy()
        pop.step(world, np.random.default_rng(9), mode="frozen")
        np.testing.assert_array_equal(pop.weights, before)

    def test_learning_local_weights_and_energy_bounds(self):
        world = World(WorldSpec(width=24, height=24, resource_patches=12),
                      seed=5)
        pop = CellPopulation(world.spec, CellSpec(count=18), seed=4)
        before = pop.weights.copy()
        rng = np.random.default_rng(2)
        for i in range(14):
            pop.step(world, rng, mode="learned")
            world.step(i + 1)
        self.assertFalse(np.array_equal(before, pop.weights))
        self.assertTrue(np.isfinite(pop.weights).all())
        self.assertTrue((pop.energy >= 0).all())
        self.assertTrue((pop.energy <= world.spec.maximum_energy).all())


class TestRuns(unittest.TestCase):
    def test_same_seed_reproduces_bit_for_bit(self):
        spec = RunSpec(seed=22, steps=30, cells=CellSpec(count=16),
                       world=WorldSpec(width=24, height=24))
        self.assertEqual(run(spec), run(spec))

    def test_mode_controls_and_histories(self):
        results = [run(RunSpec(seed=2, steps=20, mode=mode,
                               cells=CellSpec(count=16),
                               world=WorldSpec(width=24, height=24)))
                   for mode in ("random", "frozen", "learned")]
        self.assertEqual(len(results), 3)
        for item in results:
            self.assertEqual(item["initial_cells"], 16)
            self.assertTrue(0 <= item["survival_fraction"] <= 1)
            self.assertTrue(item["history"])
            self.assertGreaterEqual(item["harvested_total"], 0)

    def test_comparison_runs_each_seed_for_every_mode(self):
        r = compare(seeds=[1, 3], steps=10, cells=8)
        self.assertEqual(len(r["results"]), 6)
        self.assertEqual(set(r["aggregate"]), {"learned", "frozen", "random"})


if __name__ == "__main__":
    unittest.main()
