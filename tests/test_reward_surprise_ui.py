"""Real local HTTP route test for reward-only context inference."""
import http.client
import json
import threading
import unittest
from emergent_cells.demo.server import LabServer
from emergent_cells.cells import CellSpec
from emergent_cells.simulation import RunSpec


class SurpriseHTTPTests(unittest.TestCase):
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
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=120)
        headers = {"Content-Type": "application/json", **(extra or {})}
        body = json.dumps(value).encode() if value is not None else None
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        code = response.status
        conn.close()
        return code, json.loads(raw) if raw.startswith(b"{") else raw

    def test_01_request_validation(self):
        code, data = self.request("GET", "/api/reward-surprise")
        self.assertEqual(code, 200)
        self.assertIsNone(data["report"])
        self.assertEqual(self.request("POST", "/api/reward-surprise/run",
                         {"task_label": 1})[0], 400)
        self.assertEqual(self.request("POST", "/api/reward-surprise/run", {},
                         {"Origin": "https://attacker.test"})[0], 403)

    def test_02_real_unannounced_switch_and_audit_report(self):
        before = self.server.session.status()
        code, response = self.request("POST", "/api/reward-surprise/run", {})
        self.assertEqual(code, 200)
        self.assertEqual(response["source"], "local-reward-based-context-inference-not-meta")
        report = response["reward_surprise"]
        self.assertEqual(report["seeds"], [21, 22])
        self.assertEqual(len(report["rows"]["reward_surprise"]), 2)
        x = report["aggregate"]["reward_surprise"]
        self.assertGreater(x["b_after"], .85)
        self.assertGreater(x["a_after_detection_no_weight_training"], .85)
        self.assertGreater(x["a_return_pair_detection_fraction"], .9)
        self.assertEqual(before, self.server.session.status())
        code, session_report = self.request("GET", "/api/report")
        self.assertEqual(code, 200)
        self.assertEqual(session_report["reward_surprise_evaluation"], report)


if __name__ == "__main__":
    unittest.main()
