"""Scientific controls for privately learned exploit/probe/rest actions."""
import unittest
import numpy as np

from emergent_cells.experiments.three_action_energy import (
    ThreeActionCells, ThreeActionSpec, TEST_SCENARIOS, SCENARIO_NAMES,
    EXPLOIT, PROBE, REST, benchmark,
)
from emergent_cells.experiments.learned_curiosity import SCENARIOS


class ThreeActionTests(unittest.TestCase):
    def setUp(self):
        self.spec = ThreeActionSpec(
            pairs=8, pretrain=150,
            train_episodes=22, train_steps=18, eval_steps=105
        )

    def test_options_and_input_ranges(self):
        self.assertEqual((EXPLOIT, PROBE, REST), (0,1,2))
        for spec in (ThreeActionSpec(pairs=2),
                     ThreeActionSpec(train_episodes=1),
                     ThreeActionSpec(active_cost=-1),
                     ThreeActionSpec(epsilon_end=2)):
            with self.assertRaises(ValueError):
                spec.validate()
        with self.assertRaises(ValueError):
            ThreeActionCells(self.spec, 3, mode="fictional")

    def test_holdouts_excluded_from_training(self):
        self.assertEqual(set(SCENARIOS)&set(TEST_SCENARIOS),set())
        self.assertEqual(len(TEST_SCENARIOS), 8)
        self.assertEqual(len(SCENARIO_NAMES),len(SCENARIOS)+len(TEST_SCENARIOS)+1)

    def test_world_goal_and_resource_rates(self):
        bits = np.array([0,0,0,1,1,0,1,1],dtype=int)
        scenarios = np.array(["stable"]*4)
        goal,rate,pair=ThreeActionCells.world(bits,scenarios,.92)
        np.testing.assert_array_equal(goal,[0,1,1,0])
        np.testing.assert_allclose(rate,[.92]*4)
        scenarios = np.array(["changed_10","scarcity_01",
                              "severe_drought","global_change_blackout"])
        goal,rate,pair=ThreeActionCells.world(bits,scenarios,.92)
        np.testing.assert_array_equal(goal,[0,1,1,1])
        np.testing.assert_allclose(rate,[.92,0.,.05,.05])

    def test_actual_rest_blocks_resource_and_cost(self):
        sim = ThreeActionCells(self.spec, 1,mode="always_rest")
        sender = np.zeros((sim.n,2))
        recv = np.zeros((sim.n,4))
        result = sim.run(sender, recv, ["stable"]*self.spec.pairs,
                         np.random.default_rng(12),
                         steps=110,train=False,epsilon=0)
        self.assertEqual(result["correct"],0)
        self.assertEqual(result["pulses"],0)
        self.assertEqual(result["net_energy"],0)
        self.assertEqual(result["rest_fraction"],1)
        self.assertEqual(result["requests"],0)
        self.assertEqual(result["synthetic_cost"],0)

    def test_learned_two_has_no_rest_actions(self):
        cell = ThreeActionCells(self.spec,4,mode="learned_two")
        state=np.zeros(cell.n,dtype=int)
        for seed in (1,2,3):
            intent=cell.select(state,np.random.default_rng(seed),
                               train=True,epsilon=1.)
            self.assertFalse(np.any(intent == REST))

    def test_ema_modes_have_identical_capacity(self):
        three = ThreeActionCells(self.spec,4,mode="learned_three_ema")
        two = ThreeActionCells(self.spec,4,mode="learned_two_ema")
        self.assertEqual(three.q.shape,two.q.shape)
        self.assertEqual(three.q.shape,(16,288,3))
        selected = two.select(np.zeros(16,dtype=int),np.random.default_rng(1),train=True,epsilon=1)
        self.assertFalse((selected == REST).any())

    def test_tabular_state_is_private_per_cell(self):
        sim=ThreeActionCells(self.spec,7,mode="learned_three")
        self.assertEqual(sim.q.shape,(self.spec.pairs*2,96,3))
        sim.q[0,0,REST]=100
        state=np.zeros(sim.n,dtype=int)
        intent=sim.select(state,np.random.default_rng(8),
                          train=False,epsilon=0)
        self.assertEqual(intent[0],REST)
        self.assertTrue(np.all(intent[1:]==EXPLOIT))

    def test_small_full_benchmark_repeats(self):
        a=benchmark([2],self.spec)
        b=benchmark([2],self.spec)
        self.assertEqual(a,b)
        self.assertEqual(len(a["rows"]),8*10)
        self.assertEqual(set(a["summary"]),set(TEST_SCENARIOS))
        for r in a["rows"]:
            self.assertLessEqual(r["rest_fraction"],1)
            self.assertGreaterEqual(r["rest_fraction"],0)
            if r["method"] not in ("learned_three", "learned_three_no_history", "learned_three_low_supply", "learned_three_ema", "always_rest"):
                self.assertEqual(r["rest_fraction"],0)

    def test_world_labels_not_in_cell_action_update(self):
        sim=ThreeActionCells(self.spec,1)
        self.assertFalse(hasattr(sim,"scenario"))
        self.assertFalse(hasattr(sim,"correct_target"))


if __name__ == "__main__":
    unittest.main()
