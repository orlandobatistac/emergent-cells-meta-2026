"""Command-line runs and honest baseline comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cells import CellSpec
from .simulation import RunSpec, compare, run
from .world import WorldSpec


def main() -> None:
    p = argparse.ArgumentParser(
        description="Local cellular learner prototype; no Meta API dependency"
    )
    p.add_argument("--mode", choices=("learned", "frozen", "random", "greedy"), default="learned")
    p.add_argument("--interactive", action="store_true", help="Consola interactiva en español")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--cells", type=int, default=256)
    p.add_argument("--shock-step", type=int, default=-1)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--seeds", default="1,2,3",
                   help="Comma-separated seeds for --compare")
    p.add_argument("--output", type=Path,
                   help="Save JSON; omit to print JSON to stdout")
    args = p.parse_args()

    if args.interactive:
        from .interactive import shell
        shell(RunSpec(seed=args.seed, mode=args.mode, steps=args.steps,
                      world=WorldSpec(shock_step=args.shock_step),
                      cells=CellSpec(count=args.cells)))
        return

    if args.compare:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        data = compare(seeds, args.steps, args.cells, args.shock_step)
    else:
        data = run(RunSpec(
            seed=args.seed, mode=args.mode, steps=args.steps,
            world=WorldSpec(shock_step=args.shock_step),
            cells=CellSpec(count=args.cells)
        ))
    body = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body + "\n", encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(body)


if __name__ == "__main__":
    main()

