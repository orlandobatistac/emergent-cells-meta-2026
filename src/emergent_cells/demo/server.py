"""Serve the real Python engine on loopback; no API key or cloud service."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from emergent_cells.interactive import Session
from emergent_cells.simulation import RunSpec
from emergent_cells.experiments.dual_cue import GateSpec, benchmark as benchmark_cooperation
from emergent_cells.experiments.learned_messages import (
    ProtocolSpec, benchmark as benchmark_learned_messages,
)
from emergent_cells.experiments.context_switch import (
    SwitchSpec, benchmark as benchmark_context_switch,
)
from emergent_cells.experiments.reward_surprise import (
    benchmark as benchmark_reward_surprise,
)
from emergent_cells.meta.world_designer import (
    DesignError, design_world, evaluate_world, validate_proposal,
)


class LabServer(HTTPServer):
    def __init__(self, address, spec):
        self.session = Session(spec)
        self.challenge = None
        self.challenge_report = None
        self.cooperation_report = None
        self.learned_protocol_report = None
        self.context_switch_report = None
        self.reward_surprise_report = None
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
            report = self.server.session.report()
            report["world_design"] = self.server.challenge
            report["world_evaluation"] = self.server.challenge_report
            report["cooperation_evaluation"] = self.server.cooperation_report
            report["learned_protocol_evaluation"] = self.server.learned_protocol_report
            report["context_switch_evaluation"] = self.server.context_switch_report
            report["reward_surprise_evaluation"] = self.server.reward_surprise_report
            self.respond(200, report)
        elif self.path == '/api/reward-surprise':
            self.respond(200, {
                'report': self.server.reward_surprise_report,
                'source': 'local-reward-based-context-inference-no-meta',
            })
        elif self.path == '/api/context-switch':
            self.respond(200, {
                'report': self.server.context_switch_report,
                'source': 'independent-local-context-memory-no-meta',
            })
        elif self.path == '/api/protocol':
            self.respond(200, {
                'report': self.server.learned_protocol_report,
                'source': 'independent-local-symbol-learning-no-meta',
            })
        elif self.path == '/api/cooperation':
            self.respond(200, {
                'report': self.server.cooperation_report,
                'task': 'engineered-dual-cue-gate',
                'meta_used': False,
            })
        elif self.path == '/api/challenge':
            self.respond(200, {
                "challenge": self.server.challenge,
                "evaluation": self.server.challenge_report,
            })
        else:
            self.respond(404, {'error': 'Not found'})

    def do_POST(self):
        origin = self.headers.get('Origin')
        if not self.allowed_host() or (origin and origin != f'http://{self.headers.get("Host")}'):
            return self.respond(403, {'error': 'Local same-origin request required'})
        if self.path not in ('/api/action', '/api/challenge/design',
                             '/api/challenge/evaluate', '/api/challenge/apply',
                             '/api/cooperation/run', '/api/protocol/run',
                             '/api/context-switch/run', '/api/reward-surprise/run'):
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
            if self.path == '/api/reward-surprise/run':
                if data:
                    raise ValueError('Reward-surprise demo takes empty JSON only')
                outcome = benchmark_reward_surprise(
                    [21, 22], SwitchSpec(pairs=48, train_a=4200,
                                        train_b=3200, test_steps=150)
                )
                self.server.reward_surprise_report = outcome
                return self.respond(200, {
                    'reward_surprise': outcome,
                    'source': 'local-reward-based-context-inference-not-meta',
                    'caveat': 'Detector thresholds and two memory banks are engineered',
                })
            if self.path == '/api/context-switch/run':
                if data:
                    raise ValueError('Context-switch demo takes empty JSON only')
                outcome = benchmark_context_switch(
                    [11, 12, 13],
                    SwitchSpec(pairs=64, train_a=4500, train_b=3500,
                               test_steps=200)
                )
                self.server.context_switch_report = outcome
                return self.respond(200, {
                    'context_switch': outcome,
                    'source': 'local-contextual-memory-not-meta',
                    'caveat': 'Season indicator is GIVEN; no hidden-change detection',
                })
            if self.path == '/api/protocol/run':
                if data:
                    raise ValueError('Learned-message demo takes empty JSON only')
                # Deliberately bounded live demo; raw 10-seed study is separate.
                report = benchmark_learned_messages(
                    [11, 12, 13],
                    ProtocolSpec(pairs=64, train_steps=3500, test_steps=150)
                )
                self.server.learned_protocol_report = report
                return self.respond(200, {
                    'protocol': report,
                    'source': 'local-independent-learning-not-meta',
                })
            if self.path == '/api/cooperation/run':
                if data:
                    raise ValueError('Cooperation demo takes an empty JSON object')
                # Bounded, deterministic engineering demo; NO Meta request.
                # The underlying XOR-capable simulator is Oct 26 prior work.
                report = benchmark_cooperation(
                    [201, 202, 203],
                    GateSpec(pairs=64, train_episodes=600, test_episodes=120)
                )
                self.server.cooperation_report = report
                return self.respond(200, {
                    'cooperation': report,
                    'source': 'local-engineered-gate-not-meta',
                    'new_integration_date': '2026-10-27',
                    'prior_experiment_date': '2026-10-26',
                })
            if self.path == '/api/challenge/design':
                if set(data) != {'description', 'source'}:
                    raise ValueError('Only description and source are allowed')
                proposed = design_world(data['description'], data['source'])
                self.server.challenge = proposed
                self.server.challenge_report = None
                return self.respond(200, {'challenge': proposed})
            if self.path == '/api/challenge/evaluate':
                if data:
                    raise ValueError('Evaluation uses fixed development controls')
                if self.server.challenge is None:
                    raise ValueError('Design a world before evaluating')
                outcome = evaluate_world(self.server.challenge)
                self.server.challenge_report = outcome
                return self.respond(200, {'evaluation': outcome})
            if self.path == '/api/challenge/apply':
                if data:
                    raise ValueError('Apply takes an empty JSON object')
                if self.server.challenge is None:
                    raise ValueError('Design a world before applying')
                _, world_spec = validate_proposal(self.server.challenge['proposal'])
                old = self.server.session
                self.server.session = Session(
                    RunSpec(seed=old.spec.seed, mode=old.mode,
                            world=world_spec, cells=old.spec.cells,
                            steps=old.spec.steps)
                )
                return self.respond(200, snapshot(self.server.session))
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
        except (ValueError, TypeError, DesignError) as error:
            self.respond(400, {'error': str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with LabServer(('127.0.0.1', args.port), RunSpec(seed=args.seed)) as server:
        print(f'Emergent Cells: http://127.0.0.1:{server.server_port}', flush=True)
        print('Ctrl+C para detener. Procesamiento local; la energia simulada NO son joules.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
