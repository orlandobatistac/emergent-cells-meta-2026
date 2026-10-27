"""Reward decline vs task-rule change: guarded local change detectors.

Based on on-demand dynamic_memory engine. NO model gets world rule labels.
Reward-only ablations: legacy relative EMA, hard evidence guard, and
beta-binomial upper confidence guard for one cell's recent success pulses.

The confidence guard is heuristic, not full Bayesian change-point filtering.
A collapse in resource availability can be observationally indistinguishable
from an action-to-reward rule change, so no universal detector is claimed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np

from .dynamic_memory import GrowthSpec, GrowingCells, rng_stream

METHODS = ("relative_ema", "absolute_floor", "credible_guard", "conditional_signature", "availability_sensor")
STAGES = (("A", "xor"), ("B", "xnor"), ("C", "and"),
          ("returnA", "xor"), ("returnB", "xnor"))


class GuardedCells(GrowingCells):
    def __init__(self, spec: GrowthSpec, seed: int, method: str):
        if method not in METHODS:
            raise ValueError("unknown detector")
        super().__init__(spec, seed, "grow")
        self.method = method
        self.window_size = 100
        # Each cell has a private observation history for each memory bank;
        # never includes rule labels or a population aggregate.
        self.recent = np.zeros((self.n, spec.max_banks, self.window_size),
                               dtype=np.uint8)
        self.write_index = np.zeros((self.n, spec.max_banks), dtype=np.int32)
        self.samples = np.zeros((self.n, spec.max_banks), dtype=np.int32)
        self.hit_count = np.zeros((self.n, spec.max_banks), dtype=np.int32)
        # Tiny local, input-conditional histories. No gate rule/target labels.
        self.conditional_size = 48
        shape = (self.n, spec.max_banks, 4)
        self.cond_recent = np.zeros(shape+(self.conditional_size,), dtype=np.uint8)
        self.cond_write = np.zeros(shape, dtype=np.int16)
        self.cond_count = np.zeros(shape, dtype=np.int16)
        self.cond_hits = np.zeros(shape, dtype=np.int16)

    def _detect(self, pulse):
        # This optional arm has an EXPLICIT additional physical-style sensor:
        # Was a resource pulse available at this local gate this episode?
        # A missing opportunity supplies no evidence about the gate rule.
        # Imputing "no failure" for unavailable opportunities is deliberately
        # optimistic; compare against reward-only detectors fairly.
        if self.method == "availability_sensor":
            pulse = np.where(self.last_available, pulse, 1.0)
        # Record own pulse in currently active private memory (including probes).
        i = self.ids
        bank = self.active
        pos = self.write_index[i, bank]
        prior = self.recent[i, bank, pos].astype(int)
        success = pulse.astype(np.uint8)
        self.recent[i, bank, pos] = success
        self.hit_count[i, bank] += success.astype(int) - prior
        self.write_index[i, bank] = (pos + 1) % self.window_size
        self.samples[i, bank] = np.minimum(
            self.samples[i, bank] + 1, self.window_size)
        # Separately track reward success by local observation slot.
        slot = self.last_slot
        j = self.cond_write[i, bank, slot]
        old = self.cond_recent[i, bank, slot, j].astype(int)
        self.cond_recent[i, bank, slot, j] = success
        self.cond_hits[i, bank, slot] += success.astype(int) - old
        self.cond_write[i, bank, slot] = (j + 1) % self.conditional_size
        self.cond_count[i, bank, slot] = np.minimum(
            self.cond_count[i, bank, slot] + 1, self.conditional_size)
        super()._detect(pulse)

    def _novelty_alarm(self, eligible, spec):
        change = super()._novelty_alarm(eligible, spec)
        if self.method == "relative_ema":
            return change
        if self.method == "absolute_floor":
            return change & (self.score < .55)
        # Beta(1,1) posterior and normal upper 95% approximation.
        # A cautious rejection test against a .55 success threshold.
        i = self.ids
        n = self.samples[i, self.active].astype(float)
        good = self.hit_count[i, self.active].astype(float)
        alpha = good + 1.0
        beta = n - good + 1.0
        mu = alpha / (alpha + beta)
        sigma = np.sqrt((alpha*beta) /
                        ((alpha+beta)**2 * (alpha+beta+1)))
        upper = mu + 1.96*sigma
        if self.method == "credible_guard":
            return change & (n >= 50) & (upper < .55)

        # New hypothesis: uniform reward dropout versus selective failures.
        # Under a new Boolean gate, some own-cue/peer-message cases may
        # become wrong even if aggregate success remains ~75%.
        # Under uniform reward dropout, success falls across ALL slots.
        cs = self.cond_count[i,self.active].astype(float)
        ch = self.cond_hits[i,self.active].astype(float)
        ca, cb = ch+1., cs-ch+1.
        cm = ca/(ca+cb)
        cstd = np.sqrt(ca*cb/((ca+cb)**2*(ca+cb+1)))
        case_upper = cm + 1.96*cstd
        case_lower = cm - 1.96*cstd
        well_sampled = cs >= 24
        severe_case = np.any(well_sampled & (case_upper < .43), axis=1)
        healthy_case = np.any(well_sampled & (case_lower > .52), axis=1)
        all_cases_collapse = (n >= 60) & (upper < .43)
        return change & (all_cases_collapse | (severe_case & healthy_case))


def run_sequence(seed: int, method: str, spec: GrowthSpec):
    p = GuardedCells(spec, seed, method)
    stages = []
    for stage, rule in STAGES:
        duration = spec.phase_steps if stage in ("A", "B", "C") else spec.revisit_steps
        initial = p.allocated.copy()
        sample = []
        by_case_hits = np.zeros(4)
        by_case_total = np.zeros(4)
        for step, inputs in enumerate(rng_stream(seed, duration, p.n, stage)):
            score, _ = p.episode(inputs, rule,
                                train=(stage in ("A", "B", "C")),
                                train_sender=(stage == "A"), detect=True)
            sample.append(score)
            if step >= duration - 200:
                category = inputs[0][::2]*2+inputs[0][1::2]
                by_case_total += np.bincount(category, minlength=4)
                by_case_hits += np.bincount(
                    category, weights=p.last_good.astype(float), minlength=4)
        stages.append({
            "stage": stage,
            "late_success": float(np.mean(sample[-200:])),
            "early_success": float(np.mean(sample[:50])),
            "allocated_average": float(p.allocated.mean()),
            "new_memory_cells": int(np.count_nonzero(p.allocated > initial)),
            "case_11_success": float(by_case_hits[3]/by_case_total[3]),
        })
    return {"seed": seed, "method": method, "stages": stages,
            "allocations_total": p.growth_events,
            "reused_old_banks": p.probe_reuses}


def run_stationary(seed: int, method: str, stress_dropout: float,
                   spec: GrowthSpec, stress_steps: int=1000):
    if not 0 <= stress_dropout <= .6:
        raise ValueError("invalid reward dropout")
    p = GuardedCells(spec, seed, method)
    for inputs in rng_stream(seed, spec.phase_steps, p.n, "A"):
        p.episode(inputs, "xor", train=True,
                  train_sender=True, detect=True)
    initial = p.allocated.copy()
    first_events = p.change_events
    p.spec = replace(spec, dropout=stress_dropout)
    scores = []
    for inputs in rng_stream(seed, stress_steps, p.n, "stationary"):
        score, _ = p.episode(inputs, "xor", train=False,
                             train_sender=False, detect=True)
        scores.append(score)
    return {"seed": seed, "method": method,
            "train_dropout": spec.dropout,
            "stress_dropout": stress_dropout,
            "false_memory_cells": int(np.count_nonzero(p.allocated > initial)),
            "false_alarms": int(p.change_events-first_events),
            "late_success": float(np.mean(scores[-200:]))}


def benchmark(seeds, spec=GrowthSpec(pairs=32, phase_steps=4000),
              stress=(.08, .20, .35, .45), stress_steps=1000,
              methods=METHODS):
    spec.validate()
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("select unique known methods")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique nonempty seeds required")
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("invalid seed")
    detail = [run_sequence(seed, method, spec)
              for seed in seeds for method in methods]
    stationary = [run_stationary(seed, method, noise, spec, stress_steps)
                  for seed in seeds for method in methods for noise in stress]
    summary = {}
    for method in methods:
        own = [x for x in detail if x["method"] == method]
        summary[method] = {
            "stages": {stage: {
                "late_success": float(np.mean([
                    next(y["late_success"] for y in x["stages"]
                         if y["stage"] == stage)
                    for x in own])),
                "banks": float(np.mean([
                    next(y["allocated_average"] for y in x["stages"]
                         if y["stage"] == stage)
                    for x in own])),
            } for stage, _ in STAGES},
            "noise": {str(n): {
                "false_memory_cells": float(np.mean([
                    x["false_memory_cells"] for x in stationary
                    if x["method"] == method and x["stress_dropout"] == n])),
                "false_alarms": float(np.mean([
                    x["false_alarms"] for x in stationary
                    if x["method"] == method and x["stress_dropout"] == n])),
            } for n in stress}
        }
    return {"schema": "reward-noise-versus-task-drift-v1",
            "date": "2026-10-27",
            "seeds": seeds, "spec": asdict(spec),
            "noise_levels": list(stress),
            "methods": list(methods),
            "summary": summary, "task_rows": detail,
            "stationary_rows": stationary,
            "limitations": [
                "Thresholds selected on development seeds. Evaluate independent seeds.",
                "Beta normal-interval approximation is NOT BOCPD.",
                "World task IDs are inaccessible to adaptive policy and detector.",
                "The synthetic reward dropout value is NOT observed by the cells.",
                "A pure reward stream cannot identify the cause of all changes.",
                "Memory ceiling 3, fixed partners, binary actions and authored task rules.",
                "Synthetic rewards are not actual measured energy joules.",
            ]}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--seeds", default="11,12,13")
    a.add_argument("--pairs", type=int, default=32)
    a.add_argument("--phase-steps", type=int, default=4000)
    a.add_argument("--stress-steps", type=int, default=1000)
    a.add_argument("--output", type=Path)
    a.add_argument("--methods", default=",".join(METHODS))
    args=a.parse_args()
    seeds=[int(s) for s in args.seeds.split(",") if s.strip()]
    result=benchmark(seeds,GrowthSpec(pairs=args.pairs,
                                      phase_steps=args.phase_steps),
                     stress_steps=args.stress_steps,
                     methods=[m.strip() for m in args.methods.split(",")])
    body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(body,encoding="utf8")
        print("SAVED",args.output)
    else:
        print(body)


if __name__ == "__main__":
    main()
