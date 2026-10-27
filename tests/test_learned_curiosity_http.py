"""HTTP integration for local independently trained curiosity demo."""
import http.client
import json
import threading
import unittest
from emergent_cells.cells import CellSpec
from emergent_cells.demo.server import LabServer
from emergent_cells.simulation import RunSpec

class LearnedCuriosityHTTPTests(unittest.TestCase):
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
    def request(self,method,path,payload=None,extra=None):
        connection=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=100)
        body=json.dumps(payload).encode() if payload is not None else None
        connection.request(method,path,body,headers={"Content-Type":"application/json",**(extra or {})})
        response=connection.getresponse()
        data=response.read()
        code=response.status
        connection.close()
        return code,json.loads(data) if data.startswith(b"{") else data
    def test_01_strict_request_validation(self):
        code,x=self.request("GET","/api/learned-curiosity")
        self.assertEqual(code,200)
        self.assertIsNone(x["report"])
        self.assertEqual(self.request("POST","/api/learned-curiosity/run",{"task":"secret"})[0],400)
        self.assertEqual(self.request("POST","/api/learned-curiosity/run",{},
                         {"Origin":"https://other.example"})[0],403)
    def test_02_real_demo_and_same_origin_report(self):
        initial=self.server.session.status()
        code,result=self.request("POST","/api/learned-curiosity/run",{})
        self.assertEqual(code,200)
        report=result["learned_curiosity"]
        self.assertEqual(report["seeds"],[11,12])
        self.assertIn("locally_learned_probe",report["aggregate"]["changed_11"])
        self.assertIn("untrained_probe_policy",report["aggregate"]["changed_11"])
        self.assertGreater(len(report["rows"]),50)
        self.assertEqual(self.server.session.status(),initial)
        _,full=self.request("GET","/api/report")
        self.assertEqual(full["learned_curiosity_evaluation"],report)
