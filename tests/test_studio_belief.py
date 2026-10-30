"""Consumer-facing studio and belief-state scientific integrity tests."""
import importlib.util
import unittest

from emergent_cells.experiments.studio import Laboratory
from emergent_cells.experiments.belief_navigation import (
    BeliefExplorer, benchmark)
from emergent_cells.experiments.stochastic_motor import SlipWorld

HAS_WEB = importlib.util.find_spec("fastapi") is not None and (
    importlib.util.find_spec("httpx") is not None)


class LocalProductTests(unittest.TestCase):
    def test_local_offline_mission_is_interactive_and_reproducible(self):
        a = Laboratory()
        b = Laboratory()
        x = a.reset(seed=13001, changed=False, stochastic=False)
        y = b.reset(seed=13001, changed=False, stochastic=False)
        self.assertEqual(x,y)
        for n in (1,8,20,50):
            self.assertEqual(a.step(n),b.step(n))
        self.assertLessEqual(a.status()["steps"],160)
        self.assertNotIn("true_position",a.status())
        self.assertNotIn("target_coordinates",a.status())
        self.assertGreater(a.status()["discovered_cells"],0)

    def test_chat_commands_do_not_use_arbitrary_execution(self):
        lab=Laboratory()
        response,state=lab.local_command("run 20")
        self.assertIn("Executed",response)
        self.assertEqual(state["steps"],20)
        response,state=lab.local_command("status")
        self.assertIn("Mission status",response)
        self.assertIsNone(lab.local_command("write arbitrary code to disk"))
        self.assertNotIn("global_map",state)
        self.assertTrue(state["foundation_model_distinct_from_cells"])

    def test_invalid_inputs_and_terminal_behavior(self):
        lab=Laboratory()
        for seed in (-1,10**9,True,1.2):
            with self.assertRaises(ValueError):
                lab.reset(seed=seed,changed=False,stochastic=False)
        for count in (-2,0,161,True,1.5):
            with self.assertRaises(ValueError):
                lab.step(count)
        lab.reset(seed=20,changed=True,stochastic=False)
        end=lab.step(160)
        self.assertTrue(end["done"])
        self.assertEqual(lab.step(10)["steps"],end["steps"])

    def test_stochastic_mode_with_bounded_belief_memory(self):
        lab=Laboratory()
        s=lab.reset(seed=18,changed=True,stochastic=True)
        self.assertEqual(s["policy"],"multi-hypothesis belief explorer")
        self.assertLessEqual(s["hypotheses"],24)
        self.assertIn("motion_tick",s["current_reading"])
        out=lab.step(10)
        self.assertEqual(out["steps"],10)
        self.assertLessEqual(out["hypotheses"],24)
        self.assertNotIn("true_motor_success",out)

    def test_live_cell_memory_loss_and_local_recovery(self):
        lab=Laboratory()
        lab.reset(seed=13001,changed=False,stochastic=False,
                  controller="event")
        lab.step(20)
        before=lab.status()["faults_injected"]
        after=lab.inject_cell_failure()
        self.assertEqual(after["faults_injected"],before+1)
        later=lab.step(20)
        self.assertGreaterEqual(later["steps"],20)
        self.assertLessEqual(later["steps"],160)
        self.assertEqual(later["policy"],"event-driven local cells")

    @unittest.skipUnless(HAS_WEB,"optional Studio dependencies unavailable")
    def test_http_app_session_and_bounded_actions(self):
        from fastapi.testclient import TestClient
        from emergent_cells.experiments.studio import create_app
        c=TestClient(create_app(Laboratory()))
        self.assertTrue(c.get("/api/health").json()["local_only"])
        self.assertIn("Learned spatial memory",c.get("/").text)
        self.assertEqual(c.post("/api/reset",json={
            "seed":44,"changed":True,"stochastic":False}).status_code,200)
        self.assertEqual(c.post("/api/step",json={"count":8}).json()["steps"],8)
        self.assertEqual(c.post("/api/fault",json={}).status_code,200)
        self.assertEqual(c.post("/api/step",json={"count":1000}).status_code,422)
        self.assertEqual(c.post("/api/chat",json={"message":"status"}).status_code,200)
        self.assertEqual(c.post("/api/chat",json={"message":"run 4"}).json()["status"]["steps"],12)


class BeliefFilterTests(unittest.TestCase):
    def test_no_absolute_state_observation_and_bounded_beam(self):
        w=SlipWorld(9,rates=(.6,.7,.8,.9))
        old=w.reset(9)
        b=BeliefExplorer(9,width=8,rounds=2)
        b.reset(old)
        self.assertEqual(b.beam[0].position,(0,0))
        self.assertFalse(hasattr(old,"position"))
        for _ in range(14):
            action=b.act(old)
            nxt,done=w.step(action)
            b.observe(old,nxt,action)
            self.assertTrue(1<=len(b.beam)<=8)
            self.assertAlmostEqual(sum(
                __import__("math").exp(p.log_weight) for p in b.beam),1.,
                delta=1e-5)
            old=nxt
            if done:break

    def test_reproducible_belief_protocol_and_no_oracle(self):
        opts=dict(calibration_actions=25,width=4,rounds=1)
        a=benchmark([35],**opts)
        b=benchmark([35],**opts)
        self.assertEqual(a["summary"],b["summary"])
        self.assertTrue(a["protocol"]["no_agent_pose_or_oracle_map"])
        self.assertTrue(a["protocol"]["authored_bayesian_filter_and_local_value_diffusion"])


if __name__=="__main__":
    unittest.main()
