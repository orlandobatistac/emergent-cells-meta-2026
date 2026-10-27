"""Run matched controls in fresh child processes; retain every outcome."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from emergent_cells.cells import CellSpec
from emergent_cells.simulation import RunSpec, run
from emergent_cells.world import WorldSpec

VARIANTS = ('random', 'frozen', 'greedy', 'learned', 'learned_no_communication')


def trial(seed, variant, steps, cells, shock_step):
    mode = 'learned' if variant == 'learned_no_communication' else variant
    spec = RunSpec(seed=seed, mode=mode, steps=steps,
                   world=WorldSpec(shock_step=shock_step),
                   cells=CellSpec(count=cells, communication_radius=0 if variant == 'learned_no_communication' else 3))
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    result = run(spec)
    result.update(wall_seconds=time.perf_counter() - start_wall,
                  cpu_seconds=time.process_time() - start_cpu, variant=variant)
    # Peak RSS includes interpreter/imports, NumPy native arrays and the whole
    # worker. No subtraction of a dubious "idle baseline". Not an energy meter.
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
    except ImportError:
        result['peak_rss_bytes'] = None
    result['spec'] = asdict(spec)
    return result


def summarize(rows):
    aggregates = {}
    for variant in VARIANTS:
        group = [r for r in rows if r['variant'] == variant]
        aggregates[variant] = {}
        for key in ('survival_fraction', 'alive_cell_steps', 'harvested_total',
                    'wall_seconds', 'cpu_seconds', 'peak_rss_bytes'):
            values = [r[key] for r in group if r[key] is not None]
            aggregates[variant][key] = None if not values else {
                'mean': float(np.mean(values)), 'std_population': float(np.std(values))}
    paired = {}
    learned = {r['seed']: r for r in rows if r['variant'] == 'learned'}
    for variant in VARIANTS:
        if variant == 'learned':
            continue
        diffs = [learned[r['seed']]['survival_fraction'] - r['survival_fraction']
                 for r in rows if r['variant'] == variant]
        paired[variant] = {'mean_survival_difference': float(np.mean(diffs)),
                           'wins': sum(d > 0 for d in diffs),
                           'ties': sum(d == 0 for d in diffs), 'pairs': len(diffs)}
    return {'aggregate': aggregates, 'learned_minus_control': paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='101,102,103,104,105,106,107,108,109,110')
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--cells', type=int, default=128)
    parser.add_argument('--shock-step', type=int, default=80)
    parser.add_argument('--output', type=Path, default=Path('results/runs/benchmark.json'))
    parser.add_argument('--worker', choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        parser.error('Provide distinct nonnegative seeds')
    if args.worker:
        print(json.dumps(trial(seeds[0], args.worker, args.steps, args.cells, args.shock_step), allow_nan=False))
        return
    RunSpec(steps=args.steps, cells=CellSpec(count=args.cells),
            world=WorldSpec(shock_step=args.shock_step)).validate()
    rows = []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for i, seed in enumerate(seeds):
        # Rotate order deterministically to reduce systematic warm-up/order bias.
        order = VARIANTS[i % len(VARIANTS):] + VARIANTS[:i % len(VARIANTS)]
        for variant in order:
            proc = subprocess.run([sys.executable, '-m', __name__ if __name__ != '__main__' else
                                   'emergent_cells.evaluation.benchmark', '--worker', variant,
                                   '--seeds', str(seed), '--steps', str(args.steps), '--cells', str(args.cells),
                                   '--shock-step', str(args.shock_step)],
                                  capture_output=True, text=True, check=True, env=env, timeout=300)
            rows.append(json.loads(proc.stdout))
        print(f'Seed {seed}: 5 controls completed', file=sys.stderr)
    report = {'schema': 'emergent-benchmark-v1', 'status': 'exploratory, not confirmatory',
              'seeds': seeds, 'steps': args.steps, 'cells': args.cells, 'shock_step': args.shock_step,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'platform': platform.platform(), 'machine': platform.machine(),
                              'thread_limit': 1},
              'results': rows, **summarize(rows),
              'limitations': ['No language, agent capability, growth or retention tested.',
                              'Peak RSS includes whole fresh worker; unavailable on Windows.',
                              'Wall and CPU time cover simulation, exclude interpreter startup.',
                              'Different survival lengths change the actual amount of work.',
                              'The no-communication condition changes cost as well as information.',
                              'No physical joules measured. Pairwise neighbor discovery remains O(N^2).',
                              'These seeds are now development data; freeze fresh seeds for confirmation.',
                              'Scheduled resource displacement is not cell damage or memory retention.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
