"""One Python entry point for the scientific cellular experiments."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

from .cells import CellSpec
from .simulation import RunSpec, compare, run
from .world import WorldSpec

EXPERIMENTS = {
    "messages": "learned_messages",
    "connectivity": "connectivity",
    "dual-cue": "dual_cue",
    "context-switch": "context_switch",
    "reward-surprise": "reward_surprise",
    "dynamic-memory": "dynamic_memory",
    "stationary-noise": "stationary_noise",
    "conditional-scarcity": "conditional_scarcity",
    "robust-drift": "robust_drift",
    "active-probes": "active_probes",
    "learned-curiosity": "learned_curiosity",
    "curiosity-ablation": "curiosity_ablation",
    "sequential-transfer": "sequential_transfer",
}


def simulate(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python run.py simulate",
        description="Baseline local cells in finite-resource environments",
    )
    parser.add_argument("--mode", choices=("learned", "frozen", "random", "greedy"),
                        default="learned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--cells", type=int, default=256)
    parser.add_argument("--shock-step", type=int, default=-1)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.compare:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        data = compare(seeds, args.steps, args.cells, args.shock_step)
    else:
        data = run(RunSpec(
            seed=args.seed, mode=args.mode, steps=args.steps,
            world=WorldSpec(shock_step=args.shock_step),
            cells=CellSpec(count=args.cells),
        ))
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(payload, end="")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "list", "experiments"):
        print("Emergent Cells — Python-only scientific lab")
        print("Usage: python run.py <experiment> [experiment-specific args]")
        print("       python run.py simulate [--compare --seeds ...]")
        print("Available experiments:")
        for command, module in EXPERIMENTS.items():
            print(f"  {command:21s} {module}")
        return
    command, *params = argv
    if command == "simulate":
        simulate(params)
    elif command in EXPERIMENTS:
        old_args = sys.argv
        try:
            sys.argv = [command, *params]
            import_module(
                f"emergent_cells.experiments.{EXPERIMENTS[command]}"
            ).main()
        finally:
            sys.argv = old_args
    else:
        raise SystemExit(
            f"Unknown experiment: {command!r}. Run 'python run.py list'."
        )


if __name__ == "__main__":
    main()
