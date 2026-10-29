"""Reproducibility and negative controls for multi-step local world modeling."""
import unittest

from emergent_cells.experiments.multistep_world import (
    HELDOUT_GOALS, SOURCE_GOALS, LocalTransitionCells, Maze, benchmark,
    missions, play, train_source,
)


class MultiStepWorldTests(unittest.TestCase):
    def test_geometry_and_unannounced_closure(self):
        before, after = Maze(), Maze(changed=True)
        self.assertEqual(before.step((3, 2), 3), (4, 2))
        self.assertEqual(after.step((3, 2), 3), (3, 2))
        self.assertEqual(after.step((3, 6), 3), (4, 6))
        self.assertIsNotNone(after.shortest((1, 2), (7, 2)))

    def test_cell_updates_transition_on_surprise(self):
        cells = LocalTransitionCells()
        cells.observe((3, 2), 3, (4, 2))
        self.assertEqual(cells.decide((3, 2), (4, 2)), 3)
        cells.observe((3, 2), 3, (3, 2))
        self.assertEqual(cells.revisions, 1)
        self.assertEqual(cells.edges[(3, 2)][3], (3, 2))
        # Local memory update must not modify an existing clone.
        clone = cells.clone()
        clone.observe((3, 2), 3, (4, 2))
        self.assertEqual(cells.revisions, 1)
        self.assertEqual(clone.revisions, 2)

    def test_goal_holdout_and_three_step_missions(self):
        self.assertFalse(set(SOURCE_GOALS) & set(HELDOUT_GOALS))
        m = missions(13, 12, Maze())
        self.assertEqual(m, missions(13, 12, Maze()))
        self.assertEqual(len(m), 12)
        for _, goals in m:
            self.assertEqual(len(goals), 3)
            self.assertEqual(len(set(goals)), 3)
            self.assertTrue(set(goals).issubset(HELDOUT_GOALS))

    def test_source_training_reproducible(self):
        a, b = train_source(9, 550), train_source(9, 550)
        self.assertEqual(a.edges, b.edges)
        self.assertEqual(a.visits, b.visits)
        self.assertGreater(len(a.visits), 20)

    def test_replanning_but_not_frozen_memories(self):
        normal = Maze()
        cells = LocalTransitionCells()
        for state in normal.states():
            for action in range(4):
                cells.observe(state, action, normal.step(state, action))
        altered = Maze(changed=True)
        online = play(altered, cells.clone(), (1, 2),
                      ((7, 2), (1, 6)), "reusable_model", 7, limit=160)
        frozen = play(altered, cells.clone(), (1, 2),
                      ((7, 2), (1, 6)), "frozen_model", 7, limit=160)
        self.assertTrue(online["complete"])
        self.assertGreaterEqual(online["revisions"], 1)
        self.assertFalse(frozen["complete"])
        self.assertEqual(frozen["revisions"], 0)

    def test_short_benchmark_metrics(self):
        a = benchmark([41], train_steps=400, eval_missions=2, step_limit=90)
        self.assertEqual(len(a["trials"]), 1)
        self.assertEqual(set(a["summary"]), {
            "reusable_model", "frozen_model", "cold_model", "greedy", "random"})
        for row in a["summary"].values():
            self.assertGreaterEqual(row["heldout_complete_rate"], 0.0)
            self.assertLessEqual(row["heldout_complete_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
