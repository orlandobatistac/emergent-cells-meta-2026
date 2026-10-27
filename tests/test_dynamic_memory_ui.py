"""Real localhost evidence for the third-rule growth UI and strict API."""
import http.client
import json
import threading
import unittest
from emergent_cells.cells import CellSpec
from emergent_cells.demo.server import LabServer
from emergent_cells.simulation import RunSpec


class DynamicMemoryWebTests(unittest.TestCase):
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

    def query(self, method, path, data=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port,
                                          timeout=90)
        h = {"Content-Type":"application/json", **(headers or {})}
        conn.request(method, path, json.dumps(data).encode()
                     if data is not None else None, headers=h)
        res = conn.getresponse()
        body = res.read()
        code = res.status
        conn.close()
        return code, json.loads(body) if body.startswith(b"{") else body

    def test_01_invalid_requests_are_blocked(self):
        status, result = self.query("GET", "/api/dynamic-memory")
        self.assertEqual(status, 200)
        self.assertIsNone(result["report"])
        self.assertEqual(self.query("POST", "/api/dynamic-memory/run",
                                    {"world_rule": "and"})[0], 400)
        self.assertEqual(self.query("POST", "/api/dynamic-memory/run", {},
                        {"Origin": "https://example.net"})[0], 403)

    def test_02_local_run_and_persistent_report(self):
        initial = self.server.session.status()
        status, data = self.query("POST", "/api/dynamic-memory/run", {})
        self.assertEqual(status, 200)
        self.assertEqual(data["source"], "local-heuristic-bank-growth-no-meta")
        report = data["dynamic_memory"]
        self.assertEqual(report["seeds"], [11,12])
        self.assertEqual(len(report["rows"]), 10)
        grow = report["summary"]["grow"]
        self.assertGreater(grow["C"]["tail_success_mean"], .80)
        self.assertGreater(grow["C"]["avg_banks"], 2.5)
        self.assertGreater(grow["returnA"]["tail_success_mean"], .80)
        self.assertGreater(grow["returnB"]["tail_success_mean"], .80)
        self.assertLess(report["summary"]["cap2"]["C"]["tail_success_mean"], .6)
        self.assertEqual(self.server.session.status(), initial)
        _, record = self.query("GET", "/api/report")
        self.assertEqual(record["dynamic_memory_evaluation"], report)


if __name__ == "__main__":
    unittest.main()
