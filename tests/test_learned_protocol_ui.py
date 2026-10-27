"""October 27: learned discrete messages are exposed through the REAL server."""
import http.client
import json
import threading
import unittest
from emergent_cells.demo.server import LabServer
from emergent_cells.cells import CellSpec
from emergent_cells.simulation import RunSpec


class LearnedMessageHTTPTests(unittest.TestCase):
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
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=45)
        headers = {"Content-Type": "application/json", **(extra or {})}
        body = json.dumps(value).encode() if value is not None else None
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        code = response.status
        conn.close()
        return code, json.loads(raw) if raw.startswith(b"{") else raw

    def test_01_empty_report_then_rejects_unexpected_arguments(self):
        code, value = self.request("GET", "/api/protocol")
        self.assertEqual(code, 200)
        self.assertEqual(value["source"], "independent-local-symbol-learning-no-meta")
        self.assertIsNone(value["report"])
        self.assertEqual(self.request("POST", "/api/protocol/run", {"seed": 1})[0], 400)
        self.assertEqual(self.request("POST", "/api/protocol/run", {},
                         {"Origin": "https://attacker.test"})[0], 403)
        self.assertIsNone(self.server.learned_protocol_report)

    def test_02_true_protocol_benchmark_without_modifying_foraging_world(self):
        before = self.server.session.status()
        code, result = self.request("POST", "/api/protocol/run", {})
        self.assertEqual(code, 200)
        self.assertEqual(result["source"], "local-independent-learning-not-meta")
        data = result["protocol"]
        self.assertEqual(data["seeds"], [11, 12, 13])
        self.assertEqual(len(data["rows"]), 15)
        self.assertGreater(data["aggregate"]["learned_messages"]["test_success_fraction"]["mean"],
                           data["aggregate"]["blocked_messages"]["test_success_fraction"]["mean"])
        self.assertEqual(self.server.session.status(), before)
        code, full = self.request("GET", "/api/report")
        self.assertEqual(code, 200)
        self.assertEqual(full["learned_protocol_evaluation"], data)
        self.assertEqual(full["status"]["step"], 0)


if __name__ == "__main__":
    unittest.main()
