"""Serve the real Python engine on loopback; no API key or cloud service."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from emergent_cells.interactive import Session
from emergent_cells.simulation import RunSpec


class LabServer(HTTPServer):
    def __init__(self, address, spec):
        self.session = Session(spec)
        super().__init__(address, Handler)


def snapshot(session):
    p, w = session.population, session.world
    return {'status': session.status(), 'width': w.spec.width, 'height': w.spec.height,
            'resource': w.resource.round(4).tolist(), 'hazard': w.hazard.round(4).tolist(),
            'cells': [{'x': int(x), 'y': int(y), 'energy': float(e), 'alive': bool(a)}
                      for x, y, e, a in zip(p.x, p.y, p.energy, p.alive)],
            'maximum_energy': w.spec.maximum_energy}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, status, data, mime='application/json; charset=utf-8'):
        body = data.encode() if isinstance(data, str) else json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def allowed_host(self):
        port = self.server.server_port
        return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

    def do_GET(self):
        if not self.allowed_host():
            self.respond(403, {'error': 'Local host required'})
        elif self.path == '/':
            self.respond(200, Path(__file__).with_name('index.html').read_text(encoding='utf-8'), 'text/html; charset=utf-8')
        elif self.path == '/api/state':
            self.respond(200, snapshot(self.server.session))
        elif self.path == '/api/report':
            self.respond(200, self.server.session.report())
        else:
            self.respond(404, {'error': 'Not found'})

    def do_POST(self):
        origin = self.headers.get('Origin')
        if not self.allowed_host() or (origin and origin != f'http://{self.headers.get("Host")}'):
            return self.respond(403, {'error': 'Local same-origin request required'})
        if self.path != '/api/action':
            return self.respond(404, {'error': 'Not found'})
        if self.headers.get_content_type() != 'application/json':
            return self.respond(415, {'error': 'JSON required'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Invalid payload size')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected object')
            command = data.get('action')
            s = self.server.session
            if command == 'advance':
                steps = data.get('steps', 10)
                if type(steps) is not int or not 1 <= steps <= 200:
                    raise ValueError('steps: integer 1..200')
                s.advance(steps)
            elif command == 'shock':
                s.shock()
            elif command == 'damage':
                value = data.get('fraction', .2)
                if type(value) not in (int, float):
                    raise ValueError('fraction: number 0..1')
                s.damage(value)
            elif command == 'mode':
                s.set_mode(data.get('mode'))
            elif command == 'reset':
                self.server.session = Session(s.spec)
            else:
                raise ValueError('Unknown action')
            self.respond(200, snapshot(self.server.session))
        except (ValueError, TypeError) as error:
            self.respond(400, {'error': str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with LabServer(('127.0.0.1', args.port), RunSpec(seed=args.seed)) as server:
        print(f'Emergent Cells: http://127.0.0.1:{server.server_port}', flush=True)
        print('Ctrl+C para detener. Procesamiento local; energía simulada ≠ joules.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
