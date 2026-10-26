"""Interactive access to the real engine; commands are explicit, not an LLM."""
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import time
import numpy as np
from .cells import CellPopulation
from .simulation import RunSpec
from .world import World


class Session:
    def __init__(self, spec: RunSpec):
        spec.validate()
        self.spec = spec
        self.world = World(spec.world, spec.seed)
        self.population = CellPopulation(spec.world, spec.cells, spec.seed + 1000003)
        self.rng = np.random.default_rng(spec.seed + 2000003)
        self.step_index = 0
        self.mode = spec.mode
        self.events = []
        self.elapsed = 0.0

    def advance(self, steps=20):
        if not 1 <= steps <= 2000:
            raise ValueError('Pasos permitidos: 1..2000')
        start = time.perf_counter()
        completed = 0
        for _ in range(steps):
            if not self.population.alive.any():
                break
            self.population.step(self.world, self.rng, self.mode)
            self.step_index += 1
            completed += 1
            self.world.step(self.step_index)
        self.elapsed += time.perf_counter() - start
        self.events.append({'action': 'advance', 'requested': steps, 'completed': completed,
                            'at_step': self.step_index, 'mode': self.mode})
        return self.status()

    def shock(self):
        # User intervention displaces resources exactly once per command.
        shift = self.world.spec.width // 2
        self.world.capacity = np.roll(self.world.capacity, shift, axis=1)
        self.world.resource = np.roll(self.world.resource, shift, axis=1)
        self.events.append({'action': 'resource_shift', 'at_step': self.step_index})

    def damage(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError('Fracción permitida: 0..1')
        live = np.flatnonzero(self.population.alive)
        killed = self.rng.choice(live, size=int(len(live) * fraction), replace=False)
        self.population.alive[killed] = False
        self.population.energy[killed] = 0
        self.population.last_harvest[killed] = 0
        self.events.append({'action': 'damage', 'fraction': fraction,
                            'killed': killed.tolist(), 'at_step': self.step_index})
        return len(killed)

    def set_mode(self, mode):
        if mode not in ('learned', 'frozen', 'random', 'greedy'):
            raise ValueError('Modos: learned, frozen, random, greedy')
        self.mode = mode
        self.events.append({'action': 'mode', 'mode': mode, 'at_step': self.step_index})

    def status(self):
        p = self.population
        return {'step': self.step_index, 'mode': self.mode,
                'survivors': int(p.alive.sum()), 'initial_cells': self.spec.cells.count,
                'simulated_energy': float(p.energy.sum()),
                'harvested': p.total_harvested, 'neighbor_messages': p.total_messages,
                'mean_abs_weights': float(np.abs(p.weights).mean()),
                'engine_wall_seconds': self.elapsed}

    def report(self):
        return {'schema': 'emergent-session-v1', 'spec': asdict(self.spec),
                'status': self.status(), 'events': self.events,
                'limitations': ['Command interface, not language understanding.',
                                'No cell growth or resurrection implemented.',
                                'Simulation energy is not joules; wall time is hardware-dependent.',
                                'Manual interventions are not controlled evidence of recovery.']}


def shell(spec):
    session = Session(spec)
    print('Emergent Cells | Laboratorio interactivo v0.1')
    print('Motor real de forrajeo. Aún sin conversación, crecimiento ni agentes.')
    help_text = ('Comandos: estado | avanzar [20] | choque | dano 0.2 | '
                 'modo learned/frozen/random/greedy | reiniciar | exportar ruta.json | salir')
    print(help_text)
    while True:
        try:
            parts = shlex.split(input('cells> '))
            if not parts:
                continue
            command, *args = parts
            if command == 'salir':
                return
            if command == 'ayuda':
                print(help_text)
            elif command == 'estado':
                print(json.dumps(session.status(), ensure_ascii=False, indent=2))
            elif command == 'avanzar':
                print(json.dumps(session.advance(int(args[0]) if args else 20), indent=2))
            elif command == 'choque':
                session.shock()
                print('Recursos desplazados. Avanza para observar la respuesta.')
            elif command == 'dano':
                print(f'Células eliminadas: {session.damage(float(args[0]))}')
            elif command == 'modo':
                session.set_mode(args[0])
                print(f'Modo: {session.mode}')
            elif command == 'reiniciar':
                session = Session(spec)
                print('Misma semilla, estado inicial restaurado.')
            elif command == 'exportar':
                target = Path(args[0])
                if target.exists():
                    raise ValueError('El archivo existe; elige otra ruta.')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(session.report(), indent=2, allow_nan=False) + '\n', encoding='utf-8')
                print(f'Informe guardado: {target}')
            else:
                print(help_text)
        except (EOFError, KeyboardInterrupt):
            print('\nSesión terminada.')
            return
        except (ValueError, IndexError, OSError) as error:
            print(f'Entrada inválida: {error}')
