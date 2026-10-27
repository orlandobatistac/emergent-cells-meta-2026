"""Integration proof for the October 27 cooperative-demo API."""
import http.client
import json
import threading
import unittest

from emergent_cells.cells import CellSpec
from emergent_cells.demo.server import LabServer
from emergent_cells.simulation import RunSpec


class CooperationUITests(unittest.TestCase):
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
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=30)
        headers = {"Content-Type": "application/json"}
        headers.update(extra or {})
        body = json.dumps(value).encode() if value is not None else None
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        code = response.status
        conn.close()
        return code, json.loads(raw) if raw.startswith(b"{") else raw

    def test_01_only_local_data_before_explicit_run(self):
        code, result = self.request("GET", "/api/cooperation")
        self.assertEqual(code, 200)
        self.assertFalse(result["meta_used"])
        self.assertIsNone(result["report"])

    def test_02_rejects_payload_and_cross_origin_without_execution(self):
        response = self.request("POST", "/api/cooperation/run", {"seeds": [1]})
        self.assertEqual(response[0], 400)
        response = self.request("POST", "/api/cooperation/run", {},
                                {"Origin": "https://attacker.example"})
        self.assertEqual(response[0], 403)
        self.assertIsNone(self.server.cooperation_report)

    def test_03_actual_cooperation_run_replayed_in_report(self):
        before = self.server.session.status()
        code, result = self.request("POST", "/api/cooperation/run", {})
        self.assertEqual(code, 200)
        self.assertEqual(result["source"], "local-engineered-gate-not-meta")
        data = result["cooperation"]
        self.assertEqual(data["seeds"], [201, 202, 203])
        self.assertEqual(len(data["rows"]), 15)
        self.assertEqual(set(data["summary"]),
                         {"local", "none", "far", "scrambled", "frozen"})
        self.assertGreater(data["summary"]["local"]["test_fraction"]["mean"],
                           data["summary"]["none"]["test_fraction"]["mean"])
        self.assertEqual(before, self.server.session.status())
        code, report = self.request("GET", "/api/report")
        self.assertEqual(code, 200)
        self.assertEqual(report["cooperation_evaluation"], data)
        self.assertEqual(report["status"]["step"], 0)


if __name__ == "__main__":
    unittest.main()
