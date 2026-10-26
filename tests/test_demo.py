import http.client
import json
import threading
import unittest
from emergent_cells.demo.server import LabServer
from emergent_cells.simulation import RunSpec
from emergent_cells.cells import CellSpec


class TestDemo(unittest.TestCase):
    def setUp(self):
        self.server = LabServer(('127.0.0.1', 0), RunSpec(cells=CellSpec(count=8)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        conn.request(method, path, body, headers or {})
        response = conn.getresponse()
        code, content = response.status, response.read()
        conn.close()
        return code, content

    def post(self, data, headers=None):
        return self.request('POST', '/api/action', json.dumps(data),
                            {'Content-Type': 'application/json', **(headers or {})})

    def test_real_engine_api_and_reset(self):
        code, html = self.request('GET', '/')
        self.assertEqual(code, 200)
        self.assertIn(b'<canvas', html)
        code, result = self.post({'action': 'advance', 'steps': 7})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(result)['status']['step'], 7)
        self.post({'action': 'damage', 'fraction': .5})
        code, result = self.request('GET', '/api/report')
        self.assertEqual(json.loads(result)['status']['survivors'], 4)
        code, result = self.post({'action': 'reset'})
        self.assertEqual(json.loads(result)['status']['step'], 0)
        self.assertEqual(json.loads(result)['status']['survivors'], 8)

    def test_bad_actions_do_not_advance(self):
        for data in ({'action': 'advance', 'steps': True}, {'action': 'advance', 'steps': 999999},
                     {'action': 'damage', 'fraction': -1}, {'action': 'damage', 'fraction': float('nan')},
                     {'action': 'mode', 'mode': 'invalid'}, {'action': 'invalid'}, []):
            self.assertEqual(self.post(data)[0], 400)
        self.assertEqual(self.server.session.step_index, 0)

    def test_cross_origin_and_rebinding_rejected(self):
        self.assertEqual(self.post({'action': 'advance'}, {'Origin': 'https://example.com'})[0], 403)
        self.assertEqual(self.request('GET', '/api/state', headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.request('GET', '/missing')[0], 404)


if __name__ == '__main__':
    unittest.main()
