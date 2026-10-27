"""False memory growth during stationary XOR reward streams and reward dropout.

This does NOT simulate changing task labels, only missing resource pulses.
Used to quantify a known confound of reward-only change detectors.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .dynamic_memory import GrowthSpec, GrowingCells, rng_stream


def stationary_case(seed: int, base_dropout: float, stress_dropout: float,
                    train_steps=4500, stress_steps=1400, pairs=32):
    spec = GrowthSpec(pairs=pairs, phase_steps=train_steps,
                      dropout=base_dropout)
    population = GrowingCells(spec, seed, "grow")
    for inputs in rng_stream(seed, train_steps, population.n, "A"):
        population.episode(inputs, "xor", train=True, detect=True, train_sender=True)
    initial_allocated = int((population.allocated > 1).sum())
    initial_events = population.change_events

    population.spec = GrowthSpec(pairs=pairs, phase_steps=train_steps,
                                 dropout=stress_dropout)
    rates = []
    for inputs in rng_stream(seed, stress_steps, population.n, "stationary"):
        correct, _ = population.episode(inputs, "xor",
                                        train=False, detect=True,
                                        train_sender=False)
        rates.append(correct)
    return {
        "seed": seed, "training_dropout": base_dropout,
        "stationary_dropout": stress_dropout,
        "new_memory_before_stress": initial_allocated,
        "new_memory_during_stress": int(
            (population.allocated > 1).sum()) - initial_allocated,
        "change_alarms_during_stress": population.change_events-initial_events,
        "success_after_stress_last200": float(
            sum(rates[-200:]) / len(rates[-200:])),
        "steps": stress_steps,
    }


def benchmark(seeds, train_steps=4500, stress_steps=1400, pairs=32):
    conditions = [(0.08, 0.08), (0.08, 0.20), (0.0, 0.0)]
    rows = [stationary_case(seed, start, challenge,
                            train_steps, stress_steps, pairs)
            for seed in seeds for start, challenge in conditions]
    return {
        "schema": "stationary-noise-false-memory-v1",
        "date": "2026-10-27",
        "rows": rows,
        "conditions": conditions,
        "scientific_limitations": [
            "Noise stress is reward dropout, not an actual task change.",
            "A local reward-surprise detector cannot fully distinguish two causes of reward loss.",
            "No independent biological substrate, no measured joules, and no dynamic cells.",
        ]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds", default="11,12,13")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=benchmark([int(s) for s in args.seeds.split(",")])
    text=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf8")
        print("Saved",args.output)
    else:
        print(text)

if __name__=="__main__":
    main()
