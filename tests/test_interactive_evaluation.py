import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from emergent_cells.cells import CellPopulation, CellSpec
from emergent_cells.interactive import Session
from emergent_cells.simulation import RunSpec, run
from emergent_cells.world import World, WorldSpec


class TestInteractive(unittest.TestCase):
    def spec(self):
        return RunSpec(seed=7, cells=CellSpec(count=16), world=WorldSpec(width=24, height=24))

    def test_same_seed_interactive_matches_batch(self):
        session = Session(self.spec())
        session.advance(30)
        result = run(RunSpec(seed=7, steps=30, cells=self.spec().cells, world=self.spec().world))
        self.assertEqual(session.status()['survivors'], result['survivors'])
        self.assertEqual(session.status()['harvested'], result['harvested_total'])
        self.assertEqual(session.status()['simulated_energy'], result['remaining_energy'])

    def test_damage_no_resurrection_and_logged(self):
        session = Session(self.spec())
        self.assertEqual(session.damage(0.5), 8)
        dead = ~session.population.alive.copy()
        weights = session.population.weights[dead].copy()
        session.advance(10)
        self.assertFalse(session.population.alive[dead].any())
        np.testing.assert_array_equal(session.population.weights[dead], weights)
        self.assertEqual(session.report()['events'][0]['action'], 'damage')
        session.damage(1)
        self.assertEqual(session.advance(10)['survivors'], 0)

    def test_frozen_preserves_weights_after_learning(self):
        session = Session(self.spec())
        session.advance(10)
        before = session.population.weights.copy()
        session.set_mode('frozen')
        session.advance(10)
        np.testing.assert_array_equal(before, session.population.weights)

    def test_shock_preserves_resource_quantity(self):
        session = Session(self.spec())
        before = session.world.resource_total()
        session.shock()
        self.assertAlmostEqual(before, session.world.resource_total())

    def test_cli_interaction_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'session.json'
            proc = subprocess.run([sys.executable, '-m', 'emergent_cells', '--interactive', '--cells', '8'],
                input=f'avanzar 5\ndano 0.5\nmodo frozen\nchoque\nexportar {target}\nsalir\n',
                text=True, capture_output=True, timeout=20, check=True)
            self.assertIn('Informe guardado', proc.stdout)
            report = json.loads(target.read_text())
            self.assertEqual(report['status']['step'], 5)
            self.assertEqual(report['status']['mode'], 'frozen')
            self.assertEqual(report['status']['survivors'], 4)
            self.assertEqual(len(report['events']), 4)


class TestControls(unittest.TestCase):
    def test_greedy_uses_local_food_and_does_not_learn(self):
        world = World(WorldSpec(width=24, height=24, hazard_patches=0), 1)
        world.resource[:] = 0
        world.resource[4, 5] = 1
        pop = CellPopulation(world.spec, CellSpec(count=1), 3)
        pop.x[:] = 4
        pop.y[:] = 4
        before = pop.weights.copy()
        pop.step(world, np.random.default_rng(2), mode='greedy')
        self.assertEqual((int(pop.x[0]), int(pop.y[0])), (5, 4))
        np.testing.assert_array_equal(before, pop.weights)

    def test_remote_cell_state_does_not_change_focal_update(self):
        worlds = [World(WorldSpec(width=24, height=24), 3) for _ in range(2)]
        pops = [CellPopulation(worlds[0].spec, CellSpec(count=2, communication_radius=2), 4) for _ in range(2)]
        for pop in pops:
            pop.x[:] = [1, 15]
            pop.y[:] = [1, 15]
        pops[1].energy[1] = 0.1
        pops[1].weights[1] = [5, -5, 5]
        pops[1].last_harvest[1] = 1
        for pop, world in zip(pops, worlds):
            pop.step(world, np.random.default_rng(8))
        np.testing.assert_array_equal(pops[0].weights[0], pops[1].weights[0])
        self.assertEqual(pops[0].energy[0], pops[1].energy[0])

    def test_benchmark_subprocess_report(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'benchmark.json'
            subprocess.run([sys.executable, '-m', 'emergent_cells.evaluation.benchmark',
                            '--seeds', '1', '--steps', '3', '--cells', '4', '--shock-step', '2',
                            '--output', str(target)], check=True, capture_output=True, timeout=30)
            report = json.loads(target.read_text())
            self.assertEqual(len(report['results']), 5)
            self.assertEqual(report['learned_minus_control']['greedy']['pairs'], 1)
            for row in report['results']:
                self.assertGreater(row['wall_seconds'], 0)
                self.assertTrue(row['shock_applied'])
                self.assertEqual(row['spec']['seed'], 1)


if __name__ == '__main__':
    unittest.main()
