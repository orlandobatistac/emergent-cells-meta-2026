"""Local HTTP acceptance test of robust drift experiment UI backend."""
import http.client
import json
import threading
import unittest
from emergent_cells.cells import CellSpec
from emergent_cells.demo.server import LabServer
from emergent_cells.simulation import RunSpec


class RobustDriftWebTests(unittest.TestCase):
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

    def request(self, method, path, data=None, extra=None):
        con=http.client.HTTPConnection("127.0.0.1", self.server.server_port,timeout=120)
        headers={"Content-Type":"application/json",**(extra or {})}
        con.request(method,path,json.dumps(data).encode() if data is not None else None,headers)
        response=con.getresponse()
        raw=response.read()
        code=response.status
        con.close()
        return code,json.loads(raw) if raw.startswith(b"{") else raw

    def test_01_safe_params_and_provenance(self):
        code,x=self.request("GET","/api/robust-drift")
        self.assertEqual(code,200)
        self.assertIsNone(x["report"])
        self.assertEqual(x["source"],"local-conditional-reward-surprise-no-meta")
        self.assertEqual(self.request("POST","/api/robust-drift/run",{"key":"ignored"})[0],400)
        self.assertEqual(self.request("POST","/api/robust-drift/run",{},
                         {"Origin":"https://attacker.example"})[0],403)

    def test_02_real_run_and_report(self):
        original=self.server.session.status()
        code,result=self.request("POST","/api/robust-drift/run",{})
        self.assertEqual(code,200)
        self.assertEqual(result["source"],"local-conditional-surprise-not-meta")
        data=result["robust_drift"]
        self.assertEqual(data["seeds"],[11])
        self.assertEqual(len(data["task_rows"]),2)
        self.assertEqual(len(data["stationary_rows"]),6)
        self.assertIn("conditional_signature",data["summary"])
        self.assertIn("relative_ema",data["summary"])
        self.assertEqual(self.server.session.status(),original)
        code,report=self.request("GET","/api/report")
        self.assertEqual(code,200)
        self.assertEqual(report["robust_drift_evaluation"],data)


if __name__=="__main__":
    unittest.main()
