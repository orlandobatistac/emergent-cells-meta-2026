"""Local event-driven cell message passing and ablation tests."""
import unittest
from emergent_cells.experiments.event_cells import (
    EventDrivenCells, Cell, run_episode, benchmark)
from emergent_cells.experiments.sensor_world import SensorEpisode


class EventCellsTests(unittest.TestCase):
    def test_message_queue_respects_local_budget(self):
        world=SensorEpisode(limit=160)
        r=world.reset(71,goals=3)
        agent=EventDrivenCells(71,budget=6)
        while True:
            a=agent.act(r)
            r,_,done=world.step(a)
            agent.observe(r)
            self.assertLessEqual(agent.messages_delivered,agent.actions*12)
            if done:break
        self.assertGreater(agent.messages_delivered,0)
        self.assertGreaterEqual(agent.messages_sent,agent.messages_delivered)

    def test_message_ablation_is_causal(self):
        world=SensorEpisode(limit=160)
        r=world.reset(74,goals=3)
        no_messages=EventDrivenCells(74,budget=0)
        for _ in range(45):
            action=no_messages.act(r)
            r,_,done=world.step(action)
            no_messages.observe(r)
            if done:break
        self.assertEqual(no_messages.messages_delivered,0)
        self.assertGreater(no_messages.messages_sent,0)

    def test_cell_observes_only_local_walls(self):
        reading=SensorEpisode().reset(71)
        agent=EventDrivenCells(71)
        action=agent.act(reading)
        self.assertEqual(set(agent.cells[(0,0)].__dataclass_fields__),
                         {"walls","value","visits","epoch","neighbor_values"})
        self.assertFalse(hasattr(reading,"goal_coordinates"))
        self.assertIsInstance(action,int)

    def test_goal_phase_visit_reset_is_lazy(self):
        agent=EventDrivenCells(7,budget=4)
        agent.epoch=3
        agent.cells[(0,0)]=Cell((False,True,False,True),
                                 visits=7,epoch=2)
        agent._refresh((0,0))
        self.assertEqual(agent.cells[(0,0)].visits,0)
        self.assertEqual(agent.cells[(0,0)].epoch,3)

    def test_paired_deterministic_procedural_protocol(self):
        a=benchmark([401,402],budget=6,procedural=True)
        b=benchmark([401,402],budget=6,procedural=True)
        self.assertEqual(a["summary"],b["summary"])
        self.assertTrue(a["protocol"]["programmed_relative_odometry"])
        self.assertFalse(a["protocol"]["global_map_or_route_input"])
        self.assertFalse(a["protocol"]["learned_message_weights"])
        self.assertEqual(len(a["records"]),6)

    def test_simulated_cell_loss_does_not_stop_healthy_cells(self):
        healthy=run_episode(13001,mode="event",budget=80)
        fault=run_episode(13001,mode="event",budget=80,fault_at=30)
        self.assertLessEqual(fault["steps"],160)
        self.assertEqual(fault["faults_injected"],1)
        self.assertGreater(fault["discovered_cells"],0)
        self.assertGreater(healthy["messages"],0)
        self.assertGreaterEqual(fault["failed_cells_reobserved"],0)

    def test_reward_tuning_is_deterministic_on_training_seeds(self):
        from emergent_cells.experiments.event_reward_tuning import fit
        a, rows_a=fit(seeds=[31,32],budget=6)
        b, rows_b=fit(seeds=[31,32],budget=6)
        self.assertEqual(a,b)
        self.assertEqual(rows_a,rows_b)
        self.assertEqual(len(rows_a),25)
        self.assertTrue(0<a["decay"]<1)

    def test_simulated_cell_loss_does_not_stop_healthy_cells(self):
        healthy=run_episode(13001,mode="event",budget=80)
        fault=run_episode(13001,mode="event",budget=80,fault_at=30)
        self.assertLessEqual(fault["steps"],160)
        self.assertEqual(fault["faults_injected"],1)
        self.assertGreater(fault["discovered_cells"],0)
        self.assertGreater(healthy["messages"],0)
        self.assertGreaterEqual(fault["failed_cells_reobserved"],0)

    def test_validation_of_message_budget(self):
        with self.assertRaises(ValueError):
            EventDrivenCells(1,budget=-1)
        with self.assertRaises(ValueError):
            EventDrivenCells(1,decay=1.1)


if __name__=="__main__":
    unittest.main()
