"""Reproducible trial runner. Metrics are measured by observer code only."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import numpy as np

from .cells import CellPopulation, CellSpec
from .world import World, WorldSpec


@dataclass(frozen=True)
class RunSpec:
    seed: int = 42
    steps: int = 240
    mode: str = "learned"
    world: WorldSpec = field(default_factory=WorldSpec)
    cells: CellSpec = field(default_factory=CellSpec)

    def validate(self) -> None:
        self.world.validate()
        self.cells.validate()
        if not 1 <= self.steps <= 20000:
            raise ValueError("steps must be 1..20000")
        if self.mode not in ("learned", "frozen", "random"):
            raise ValueError("mode must be learned, frozen or random")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def run(spec: RunSpec) -> dict:
    spec.validate()
    world = World(spec.world, spec.seed)
    population = CellPopulation(spec.world, spec.cells, seed=spec.seed + 1000003)
    rng = np.random.default_rng(spec.seed + 2000003)
    history = []
    alive_cell_steps = 0
    for t in range(spec.steps):
        state = population.step(world, rng, mode=spec.mode)
        alive_cell_steps += state["alive"]
        world.step(t + 1)
        if (t + 1) == 1 or (t + 1) % max(1, spec.steps // 12) == 0 or (t + 1) == spec.steps:
            history.append({"step": t + 1, **state,
                            "world_resource": world.resource_total()})
        if not state["alive"]:
            break
    return {
        "engine": "locally-learning-cells-prototype-v0",
        "run_hash": spec.digest(),
        "seed": spec.seed,
        "mode": spec.mode,
        "steps_requested": spec.steps,
        "steps_completed": t + 1,
        "initial_cells": spec.cells.count,
        "survivors": int(population.alive.sum()),
        "survival_fraction": float(population.alive.mean()),
        "alive_cell_steps": alive_cell_steps,
        "harvested_total": population.total_harvested,
        "hazard_cost_total": population.total_hazard_cost,
        "remaining_energy": float(population.energy.sum()),
        "neighbor_messages": population.total_messages,
        "mean_abs_weights": float(np.abs(population.weights).mean()),
        "shock_applied": world.shock_applied,
        "world_spec": asdict(spec.world),
        "cell_spec": asdict(spec.cells),
        "history": history,
        "limitations": [
            "Simulation units are not measured physical joules.",
            "Vectorized execution is not evidence of a physical substrate-independent device.",
            "No autonomous cell reproduction; no semantic general intelligence demonstrated.",
            "No Meta Model API integration in prototype v0.",
        ],
    }


def compare(seeds: list[int], steps: int, cells: int = 256,
            shock_step: int = -1) -> dict:
    if not seeds:
        raise ValueError("At least one seed required")
    outcomes = []
    for seed in seeds:
        for mode in ("random", "frozen", "learned"):
            result = run(RunSpec(
                seed=seed, mode=mode, steps=steps,
                world=WorldSpec(shock_step=shock_step),
                cells=CellSpec(count=cells)
            ))
            outcomes.append({
                "seed": seed, "mode": mode,
                "survival_fraction": result["survival_fraction"],
                "alive_cell_steps": result["alive_cell_steps"],
                "remaining_energy": result["remaining_energy"],
                "harvested_total": result["harvested_total"],
                "neighbor_messages": result["neighbor_messages"],
                "run_hash": result["run_hash"],
            })
    aggregates = {}
    for mode in ("random", "frozen", "learned"):
        group = [r for r in outcomes if r["mode"] == mode]
        aggregates[mode] = {
            key: {
                "mean": float(np.mean([r[key] for r in group])),
                "std_population": float(np.std([r[key] for r in group])),
            }
            for key in ("survival_fraction", "alive_cell_steps",
                        "remaining_energy", "harvested_total")
        }
    return {
        "status": "exploratory - not a pre-registered or statistically conclusive study",
        "seeds": seeds, "steps": steps, "cells": cells,
        "shock_step": shock_step,
        "results": outcomes, "aggregate": aggregates
    }
