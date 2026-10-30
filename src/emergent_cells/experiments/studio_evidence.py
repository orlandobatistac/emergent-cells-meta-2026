"""Paired, reproducible messaging ablations for the interactive laboratory.

These comparisons rerun identical environment seeds using authored controllers.
They are exploratory demonstrations, not preregistered research results.
"""
from __future__ import annotations

from statistics import mean
from time import process_time

from .event_cells import run_episode

ARMS = ("event", "no_messages", "synchronous")


def compare_controllers(*, start_seed: int, trials: int = 8, changed: bool = False) -> dict:
    """Test three matched controls without touching the active Studio mission."""
    if type(start_seed) is not int or not 0 <= start_seed <= 10_000_000:
        raise ValueError("start_seed must be an integer between 0 and 10000000")
    if type(trials) is not int or not 1 <= trials <= 12:
        raise ValueError("trials must be an integer between 1 and 12")
    if start_seed + trials - 1 > 10_000_000:
        raise ValueError("seed range exceeds 10000000")
    if type(changed) is not bool:
        raise ValueError("changed must be a boolean")
    started = process_time()
    records = []
    for seed in range(start_seed, start_seed + trials):
        outcomes = {}
        for mode in ARMS:
            result = run_episode(seed, mode=mode, budget=80, changed=changed)
            outcomes[mode] = {
                "completed": bool(result["completed"]),
                "waypoints": int(result["waypoints"]),
                "steps": int(result["steps"]),
                "messages": int(result["messages"]),
                "discovered_cells": int(result["discovered_cells"]),
            }
        records.append({"seed": seed, "outcomes": outcomes})
    summary = {}
    for mode in ARMS:
        runs = [r["outcomes"][mode] for r in records]
        summary[mode] = {
            "completed": sum(r["completed"] for r in runs),
            "completion_rate": mean(int(r["completed"]) for r in runs),
            "mean_steps": mean(r["steps"] for r in runs),
            "mean_waypoints": mean(r["waypoints"] for r in runs),
            "mean_messages": mean(r["messages"] for r in runs),
        }
    event_wins = sum(r["outcomes"]["event"]["completed"]
                     and not r["outcomes"]["no_messages"]["completed"]
                     for r in records)
    no_message_wins = sum(r["outcomes"]["no_messages"]["completed"]
                          and not r["outcomes"]["event"]["completed"]
                          for r in records)
    return {
        "protocol": {
            "seed_start": start_seed,
            "trials": trials,
            "changed": changed,
            "goals": 3,
            "max_actions": 160,
            "event_message_budget": 80,
            "paired_seeds": True,
            "exploratory_not_preregistered": True,
            "authored_policy_and_messages": True,
            "zero_budget_still_has_direct_neighbor_handshake": True,
            "synchronous_message_count_represents_local_update_operations": True,
            "no_physical_node_failures": True,
            "process_cpu_seconds": round(process_time() - started, 5),
        },
        "summary": summary,
        "paired_event_only_successes": event_wins,
        "paired_no_message_only_successes": no_message_wins,
        "records": records,
    }


def main() -> None:
    """Run and save a bounded exploratory experiment from the CLI."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=56001)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--changed", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_controllers(start_seed=args.seed,
                                 trials=args.trials, changed=args.changed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output),
                      "summary": result["summary"],
                      "paired_event_only_successes":
                          result["paired_event_only_successes"],
                      "paired_no_message_only_successes":
                          result["paired_no_message_only_successes"]}, indent=2))


if __name__ == "__main__":
    main()
