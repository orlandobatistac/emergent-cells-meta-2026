"""HTTP contract for the active exploration study."""
import http.client
import json
import threading
import unittest
from emergent_cells.cells import CellSpec
from emergent_cells.demo.server import LabServer
from emergent_cells.simulation import RunSpec

class InquiryHTTPTests(unittest.TestCase):
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
    def request(self,method,path,payload=None):
        c=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=90)
        body=json.dumps(payload).encode() if payload is not None else None
        c.request(method,path,body,{"Content-Type":"application/json"})
        result=c.getresponse()
        data=result.read()
        code=result.status
        c.close()
        return code,json.loads(data) if data.startswith(b"{") else data
    def test_api(self):
        status,x=self.request("GET","/api/active-probes")
        self.assertEqual(status,200)
        self.assertIsNone(x["report"])
        self.assertEqual(self.request("POST","/api/active-probes/run",{"extra":1})[0],400)
        initial=self.server.session.status()
        status,x=self.request("POST","/api/active-probes/run",{})
        self.assertEqual(status,200)
        report=x["active_probes"]
        self.assertEqual(len(report["rows"]),40)
        self.assertGreater(report["aggregate"]["changed_11"]["information_gain"]["critical_11_success"],.75)
        self.assertEqual(self.server.session.status(),initial)
        _,full=self.request("GET","/api/report")
        self.assertEqual(full["active_probes_evaluation"],report)
