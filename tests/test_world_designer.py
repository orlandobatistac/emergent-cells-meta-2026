"""Testing world-designer provenance, locality boundary and server integration.

No real network Meta calls, and no API spend, happen in these tests.
"""
import http.client
import json
from types import SimpleNamespace
import os
import threading
import unittest
from unittest.mock import patch

from emergent_cells.meta.world_designer import (
    DesignError, META_MODEL, design_world, evaluate_world, validate_proposal
)
from emergent_cells.demo.server import LabServer
from emergent_cells.cells import CellSpec
from emergent_cells.simulation import RunSpec


class FakeMetaClient:
    def __init__(self, proposal):
        self.proposal = proposal
        self.requests = []
        msg = SimpleNamespace(content=json.dumps(proposal))
        self.response = SimpleNamespace(id="test-meta-response-not-live",
                                        choices=[SimpleNamespace(message=msg)])
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


class TestDesign(unittest.TestCase):
    def setUp(self):
        self.description = "Make a resource-scarce world with an environmental shock."
        self.template = design_world(self.description, source="template")

    def test_template_explicit_provenance_and_valid_world(self):
        self.assertEqual(self.template["source"], "template")
        self.assertIsNone(self.template["model"])
        self.assertIsNone(self.template["meta_response_id"])
        self.assertEqual(self.template["world"]["resource_patches"], 5)
        self.assertEqual(self.template, design_world(self.description, "template"))

    def test_fake_client_meta_path_and_json_schema(self):
        fake = FakeMetaClient(dict(self.template["proposal"]))
        result = design_world(self.description, source="meta", client=fake)
        self.assertEqual(result["source"], "meta")
        self.assertEqual(result["model"], META_MODEL)
        self.assertEqual(result["meta_response_id"], "test-meta-response-not-live")
        self.assertEqual(len(fake.requests), 1)
        call = fake.requests[0]
        self.assertEqual(call["model"], META_MODEL)
        self.assertTrue(call["response_format"]["json_schema"]["strict"])
        self.assertEqual(call["response_format"]["json_schema"]["schema"]["additionalProperties"], False)

    def test_meta_no_key_fails_closed_without_template_switch(self):
        with patch.dict(os.environ, {"MODEL_API_KEY": ""}):
            with self.assertRaisesRegex(DesignError, "MODEL_API_KEY"):
                design_world(self.description, "meta")

    def test_world_rejects_invalid_model_output(self):
        good = dict(self.template["proposal"])
        for field, value in [("resource_patches", -5),
                             ("regeneration", float("nan")),
                             ("shock_step", 0), ("hazard_damage", True),
                             ("name", ""), ("patch_radius", 100)]:
            bad = dict(good, **{field: value})
            with self.subTest(field=field), self.assertRaises(DesignError):
                validate_proposal(bad)
        with self.assertRaises(DesignError):
            validate_proposal(dict(good, commands="delete local files"))
        with self.assertRaises(DesignError):
            design_world("bad", "template")
        with self.assertRaises(DesignError):
            design_world("hello world", "surprise")

    def test_compare_replay_and_strong_greedy_baseline(self):
        a = evaluate_world(self.template, seeds=[5, 6], steps=16, count=8)
        b = evaluate_world(self.template, seeds=[5, 6], steps=16, count=8)
        self.assertEqual(a, b)
        self.assertEqual(len(a["rows"]), 10)
        self.assertIn("greedy", a["aggregate"])
        off = [r for r in a["rows"] if r["variant"] == "learned_no_messages"]
        self.assertTrue(all(x["messages"] == 0 for x in off))
        with self.assertRaises(DesignError):
            evaluate_world(self.template, seeds=[1, 1])


class TestWebDesigner(unittest.TestCase):
    def setUp(self):
        self.server = LabServer(("127.0.0.1", 0), RunSpec(cells=CellSpec(count=8)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def call(self, method, path, body=None, origin=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        headers = {"Content-Type": "application/json"}
        if origin is not None:
            headers["Origin"] = origin
        conn.request(method, path, json.dumps(body).encode() if body is not None else None, headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, json.loads(raw) if raw.startswith(b"{") else raw

    def test_full_workflow_dryrun_and_replay(self):
        desc = {"description": "Mundo con recursos escasos y peligros",
                "source": "template"}
        code, data = self.call("POST", "/api/challenge/design", desc)
        self.assertEqual(code, 200)
        self.assertEqual(data["challenge"]["source"], "template")
        self.assertIsNone(data["challenge"]["model"])
        proposed = data["challenge"]["world"]
        self.assertEqual(proposed["resource_patches"], 5)
        code, data = self.call("POST", "/api/challenge/evaluate", {})
        self.assertEqual(code, 200)
        self.assertEqual(len(data["evaluation"]["rows"]), 15)
        self.assertEqual(data["evaluation"]["designer_source"], "template")
        code, state = self.call("POST", "/api/challenge/apply", {})
        self.assertEqual(code, 200)
        self.assertEqual(state["status"]["step"], 0)
        self.assertEqual(self.server.session.world.spec.resource_patches, 5)
        code, state = self.call("POST", "/api/action",
                                {"action": "advance", "steps": 5})
        self.assertEqual(code, 200)
        self.assertEqual(state["status"]["step"], 5)
        code, report = self.call("GET", "/api/report")
        self.assertEqual(code, 200)
        self.assertEqual(report["world_design"]["source"], "template")
        self.assertEqual(report["world_evaluation"]["schema"],
                         "emergent-world-evaluation-v1")

    def test_invalid_and_unauthorized_requests_do_not_advance(self):
        self.assertEqual(self.call("POST", "/api/challenge/evaluate", {})[0], 400)
        self.assertEqual(self.call("POST", "/api/challenge/apply", {})[0], 400)
        self.assertEqual(self.call("POST", "/api/challenge/design",
                                   {"description": "bad", "source": "template"})[0], 400)
        with patch.dict(os.environ, {"MODEL_API_KEY": ""}):
            self.assertEqual(self.call("POST", "/api/challenge/design",
                                       {"description": "A challenge", "source": "meta"})[0], 400)
        payload = {"description": "A challenging world", "source": "template"}
        self.assertEqual(self.call("POST", "/api/challenge/design", payload,
                                   origin="https://attacker.example")[0], 403)
        self.assertEqual(self.server.session.step_index, 0)

    def test_html_contains_integrated_world_designer(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        conn.request("GET", "/")
        r = conn.getresponse()
        html = r.read().decode()
        conn.close()
        self.assertEqual(r.status, 200)
        self.assertIn("challenge-source", html)
        self.assertIn("/api/challenge/design", html)
        self.assertIn("Plantilla local", html)


if __name__ == "__main__":
    unittest.main()
