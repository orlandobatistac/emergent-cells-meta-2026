"""Live local HTTP integration of A-B-A contextual memory test."""
import http.client
import json
import threading
import unittest
from emergent_cells.demo.server import LabServer
from emergent_cells.cells import CellSpec
from emergent_cells.simulation import RunSpec


class ContextSwitchHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = LabServer(("127.0.0.1", 0), RunSpec(cells=CellSpec(count=8)))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, value=None, extra=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=90)
        headers = {"Content-Type": "application/json", **(extra or {})}
        body = json.dumps(value).encode() if value is not None else None
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        code = response.status
        conn.close()
        return code, json.loads(raw) if raw.startswith(b"{") else raw

    def test_01_initial_and_invalid_calls(self):
        status, data = self.request("GET", "/api/context-switch")
        self.assertEqual(status, 200)
        self.assertEqual(data["source"], "independent-local-context-memory-no-meta")
        self.assertIsNone(data["report"])
        self.assertEqual(self.request("POST", "/api/context-switch/run", {"pairs": 8})[0], 400)
        self.assertEqual(self.request("POST", "/api/context-switch/run", {},
                         {"Origin": "https://attacker.test"})[0], 403)

    def test_02_real_a_b_a_run_and_report(self):
        before = self.server.session.status()
        status, result = self.request("POST", "/api/context-switch/run", {})
        self.assertEqual(status, 200)
        self.assertEqual(result["source"], "local-contextual-memory-not-meta")
        self.assertIn("Season indicator", result["caveat"])
        data = result["context_switch"]
        self.assertEqual(data["seeds"], [11, 12, 13])
        self.assertEqual(len(data["results"]), 18)
        memory = data["aggregate"]["context_memory"]
        reset = data["aggregate"]["reset_on_switch"]
        self.assertGreater(memory["b_after"]["mean"], .85)
        self.assertGreater(memory["a_return_no_training"]["mean"], .85)
        self.assertGreater(reset["b_after"]["mean"], .85)
        self.assertLess(reset["a_return_no_training"]["mean"], .30)
        self.assertEqual(self.server.session.status(), before)
        _, report = self.request("GET", "/api/report")
        self.assertEqual(report["context_switch_evaluation"], data)


if __name__ == "__main__":
    unittest.main()
