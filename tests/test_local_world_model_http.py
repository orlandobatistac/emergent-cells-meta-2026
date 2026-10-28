"""Real local HTTP integration for the Phase 8 predictor prototype."""
import http.client
import json
import threading
import unittest
from emergent_cells.cells import CellSpec
from emergent_cells.demo.server import LabServer
from emergent_cells.simulation import RunSpec

class LocalPredictorHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=LabServer(("127.0.0.1",0),RunSpec(cells=CellSpec(count=8)))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
    def call(self,method,path,data=None,extra=None):
        c=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=75)
        payload=json.dumps(data).encode() if data is not None else None
        c.request(method,path,payload,headers={"Content-Type":"application/json",**(extra or {})})
        r=c.getresponse()
        raw=r.read()
        code=r.status
        c.close()
        return code,json.loads(raw) if raw.startswith(b"{") else raw
    def test_invalid_inputs(self):
        code,d=self.call("GET","/api/local-world-model")
        self.assertEqual(code,200)
        self.assertIsNone(d["report"])
        self.assertEqual(self.call("POST","/api/local-world-model/run",{"seed":9})[0],400)
        self.assertEqual(self.call("POST","/api/local-world-model/run",{},{"Origin":"https://attacker.example"})[0],403)
    def test_real_run_and_report(self):
        original=self.server.session.status()
        code,result=self.call("POST","/api/local-world-model/run",{})
        self.assertEqual(code,200)
        raw=result["local_world_model"]
        self.assertEqual(raw["seeds"],[11,12])
        self.assertEqual(len(raw["rows"]),70)
        self.assertGreater(raw["summary"]["changed_11"]["predictive_progress"]["case11_correct"],.70)
        self.assertEqual(self.server.session.status(),original)
        _,report=self.call("GET","/api/report")
        self.assertEqual(report["local_world_model_evaluation"],raw)
