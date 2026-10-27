"""Local reward-surprise-driven *on-demand* memory growth, three Boolean worlds.

Experiments, not an implementation of a biological or general intelligence.
Each cell emits independently learned 1-bit cues, and has privately learned
per-bank conditional response tables. A change is unannounced; a simple
authored reward-history detector probes old banks and, if none is useful,
creates one NEW memory bank. Physical rule labels NEVER reach adaptive cells.

Simulated pair-level reward dropout can create false surprise. Agents do not
receive the dropout bit. All reward/energy counts are synthetic, not joules.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from .learned_messages import sigmoid

CONDITIONS = ("grow", "cap2", "single", "oracle", "fast_detector")


@dataclass(frozen=True)
class GrowthSpec:
    pairs: int = 48
    phase_steps: int = 5000
    test_steps: int = 180
    revisit_steps: int = 900
    max_banks: int = 3
    learning_rate: float = 0.10
    dropout: float = 0.08
    smoothing: float = 0.005
    relative_trigger: float = 0.10
    maturity: float = 0.78
    maturity_steps: int = 550
    probe_steps: int = 160
    probe_accept: float = 0.78
    cooldown: int = 140
    exploration: float = 0.03

    def validate(self):
        if not 4 <= self.pairs <= 2048 or not 200 <= self.phase_steps <= 30000:
            raise ValueError("bad pairs or phase length")
        if not 50 <= self.test_steps <= 4000 or not 100 <= self.revisit_steps <= 10000:
            raise ValueError("bad test or revisit length")
        if self.max_banks != 3:
            raise ValueError("benchmark requires 3 bank capacity")
        if not 0 < self.learning_rate <= 1 or not 0 <= self.dropout <= .5:
            raise ValueError("bad learning rate or reward noise")
        if not .001 <= self.smoothing <= .3 or not .04 <= self.relative_trigger <= .5:
            raise ValueError("bad change-point settings")
        if not 0 < self.maturity < 1 or not 30 <= self.maturity_steps <= 10000:
            raise ValueError("bad maturity")
        if not 20 <= self.probe_steps <= 1000 or not .4 <= self.probe_accept <= 1:
            raise ValueError("bad probe settings")
        if not 10 <= self.cooldown <= 3000 or not 0 <= self.exploration < .4:
            raise ValueError("bad cooldown or exploration")


def rule_targets(bits: np.ndarray, rule: str):
    a, b = bits[0::2], bits[1::2]
    if rule == "xor":
        return a ^ b
    if rule == "xnor":
        return 1 - (a ^ b)
    if rule == "and":
        return a & b
    raise ValueError("world rule must be xor, xnor or and")


def rng_stream(seed, steps, n, stage):
    # Independent reproducible streams; physical rule absent from policy input.
    salts = {"A": 100, "B": 200, "C": 300, "returnA": 400,
             "returnB": 500, "stationary": 600, "eval": 700}
    rng = np.random.default_rng(seed + 3000001 + salts[stage])
    action = np.random.default_rng(seed + 4000001 + salts[stage])
    sender = np.random.default_rng(seed + 5000001 + salts[stage])
    noise = np.random.default_rng(seed + 6000001 + salts[stage])
    for _ in range(steps):
        yield (rng.integers(0, 2, n),
               sender.random(n), action.random(n),
               noise.random(n // 2))


class GrowingCells:
    def __init__(self, spec: GrowthSpec, seed: int, condition: str):
        spec.validate()
        if condition not in CONDITIONS:
            raise ValueError("unknown condition")
        self.spec = spec
        self.condition = condition
        n = 2 * spec.pairs
        self.n = n
        self.ids = np.arange(n)
        self.peer = self.ids ^ 1
        init = np.random.default_rng(seed + 1000003)
        self.sender = init.normal(0, .12, (n, 2))
        # Physically store ONE bank initially; append when needed. Oracle
        # deliberately starts with three banks because it knows task IDs.
        initial_banks = spec.max_banks if condition == "oracle" else 1
        self.weights = np.zeros((n, initial_banks, 4), dtype=float)
        self.weights[:, 0] = init.normal(0, .12, (n, 4))
        self.baseline = np.zeros((n, initial_banks))
        self.allocated = np.full(n, initial_banks, dtype=int)
        self.active = np.zeros(n, dtype=int)
        self.score = np.full(n, .25)
        self.peak = np.full(n, .25)
        self.age = np.zeros(n, dtype=int)
        self.cooldown_left = np.zeros(n, dtype=int)
        # probing -1: ordinary training, >=0: evaluate old policy bank
        self.probing = np.full(n, -1, dtype=int)
        self.probe_sum = np.zeros(n)
        self.probe_count = np.zeros(n, dtype=int)
        self.probe_origin = np.zeros(n, dtype=int)
        self.change_events = 0
        self.growth_events = 0
        self.probe_reuses = 0
        self.failed_probes = 0
        self.energy_balance = 0.

    def _switch_to(self, choose: np.ndarray, target: np.ndarray, probe=False):
        # Sets private state only. No global reward averaging.
        self.active[choose] = target[choose]
        self.score[choose] = .25
        self.peak[choose] = .25
        self.age[choose] = 0
        self.cooldown_left[choose] = self.spec.cooldown
        if probe:
            self.probing[choose] = target[choose]
            self.probe_sum[choose] = 0
            self.probe_count[choose] = 0
        else:
            self.probing[choose] = -1

    def _allocate(self, choose: np.ndarray):
        if not choose.any():
            return
        nidx = self.ids[choose]
        cap = self.spec.max_banks if self.condition not in ("cap2",) else 2
        can_add = self.allocated[choose] < cap
        recipients = nidx[can_add]
        if recipients.size:
            slots = self.allocated[recipients]
            # Lazy physical NumPy storage growth. The population scheduler
            # extends its packed tensor, while logical bank counts remain
            # independently owned by cells. No physical substrate claim.
            required_columns = int(slots.max()) + 1
            while self.weights.shape[1] < required_columns:
                self.weights = np.concatenate(
                    (self.weights, np.zeros((self.n, 1, 4))), axis=1)
                self.baseline = np.concatenate(
                    (self.baseline, np.zeros((self.n, 1))), axis=1)
            self.weights[recipients, slots] = 0.0  # generic new policy, no XOR rule
            self.baseline[recipients, slots] = 0.0
            self.allocated[recipients] += 1
            move = np.zeros(self.n, dtype=bool)
            move[recipients] = True
            next_slot = self.active.copy()
            next_slot[recipients] = slots
            self._switch_to(move, next_slot)
            self.growth_events += len(recipients)
        stuck = nidx[~can_add]
        if stuck.size:
            # Capacity-limited control reuses an old bank and can overwrite it.
            select = np.zeros(self.n, dtype=bool)
            select[stuck] = True
            next_slot = self.active.copy()
            next_slot[stuck] = 0
            self._switch_to(select, next_slot)
            self.failed_probes += len(stuck)

    def _detect(self, pulse):
        if self.condition in ("single", "oracle"):
            return
        s = self.spec
        self.cooldown_left[self.cooldown_left > 0] -= 1
        # During a bounded *probe* of an old memory, freeze local weights.
        probing = self.probing >= 0
        if probing.any():
            self.probe_sum[probing] += pulse[probing]
            self.probe_count[probing] += 1
            done = probing & (self.probe_count >= s.probe_steps)
            if done.any():
                success = (self.probe_sum / np.maximum(self.probe_count, 1)) >= s.probe_accept
                reuse = done & success
                if reuse.any():
                    self.probe_reuses += int(reuse.sum())
                    # Empirical local probe success IS the evidence that this
                    # reused memory is mature; do not reset its maturity clock.
                    recovered = self.probe_sum[reuse] / self.probe_count[reuse]
                    self.score[reuse] = recovered
                    self.peak[reuse] = recovered
                    self.age[reuse] = s.maturity_steps
                    self.probing[reuse] = -1
                    self.cooldown_left[reuse] = s.cooldown
                failed = done & ~success
                if failed.any():
                    # If there are more older banks, try them; otherwise grow.
                    still = failed & (self.probe_origin > 0)
                    if still.any():
                        next_bank = self.active.copy()
                        next_bank[still] = self.probe_origin[still] - 1
                        self.probe_origin[still] -= 1
                        self._switch_to(still, next_bank, probe=True)
                    exhausted = failed & ~still
                    if exhausted.any():
                        self.failed_probes += int(exhausted.sum())
                        self._allocate(exhausted)
        # Work from current (non-probing) observations only.
        eligible = (self.probing < 0) & (self.cooldown_left == 0)
        self.score[eligible] = (
            (1-s.smoothing)*self.score[eligible] + s.smoothing*pulse[eligible])
        self.peak[eligible] = np.maximum(self.peak[eligible], self.score[eligible])
        self.age[eligible] += 1
        threshold = self.peak - (
            .115 if self.condition == "fast_detector" else s.relative_trigger)
        alarm = (eligible & (self.age >= s.maturity_steps) &
                 (self.peak >= s.maturity) & (self.score < threshold))
        if not alarm.any():
            return
        self.change_events += int(alarm.sum())
        # Try all old memories, most recently created -> oldest, excluding
        # current active slot. When none exists, grow. No world rule label.
        existing = alarm & (self.allocated > 1)
        if existing.any():
            self.probe_origin[existing] = self.allocated[existing] - 1
            target = self.active.copy()
            target[existing] = self.allocated[existing] - 1
            same = existing & (target == self.active)
            target[same] = np.maximum(0, target[same] - 1)
            self.probe_origin[existing] = target[existing]
            self._switch_to(existing, target, probe=True)
        only = alarm & (self.allocated == 1)
        self._allocate(only)

    def episode(self, inputs, rule, *, train, detect, train_sender=False):
        if rule not in ("xor", "xnor", "and"):
            raise ValueError("invalid environmental rule")
        bits, um, ua, ur = inputs
        if bits.shape != (self.n,) or not np.isin(bits, [0, 1]).all():
            raise ValueError("invalid cell cues")
        own = bits * 2 - 1
        pmsg = sigmoid(self.sender[:, 0] + self.sender[:, 1] * own)
        symbols = (um < pmsg).astype(int)
        received = symbols[self.peer]
        slot = bits * 2 + received
        if self.condition == "oracle":
            # Research-only upper bound receives environment's rule ID.
            index = {"xor": 0, "xnor": 1, "and": 2}[rule]
            self.active[:] = index
            self.allocated[:] = 3
        if self.condition == "single":
            self.active[:] = 0
            self.allocated[:] = 1
        p = sigmoid(self.weights[self.ids, self.active, slot])
        p = (1 - self.spec.exploration) * p + self.spec.exploration * .5
        actions = (ua < p).astype(int)
        goal = rule_targets(bits, rule)
        good = (actions[::2] == goal) & (actions[1::2] == goal)
        self.last_good = good.copy()
        earned = good & (ur >= self.spec.dropout)
        own_reward = np.repeat(earned.astype(float), 2)
        net = own_reward - .04 - .02
        self.energy_balance += float(net.sum())
        if train:
            valid = self.probing == -1
            ownadv = net - self.baseline[self.ids, self.active]
            if train_sender:
                # Encoder-learning phase is controlled by experimental schedule,
                # NOT by giving the hidden world rule to the cell.
                update = self.spec.learning_rate * ownadv * (symbols - pmsg)
                self.sender[:, 0] += update
                self.sender[:, 1] += update * own
                np.clip(self.sender, -12, 12, out=self.sender)
            self.weights[self.ids[valid], self.active[valid], slot[valid]] += (
                self.spec.learning_rate * ownadv[valid] * (actions[valid]-p[valid]))
            np.clip(self.weights, -12, 12, out=self.weights)
            self.baseline[self.ids[valid], self.active[valid]] = (
                .98*self.baseline[self.ids[valid], self.active[valid]]
                + .02*net[valid])
        if detect:
            self._detect(own_reward)
        return float(good.mean()), float(earned.mean())


def run_trial(seed: int, condition="grow", spec: GrowthSpec = GrowthSpec()):
    pop = GrowingCells(spec, seed, condition)
    schedule = [
        ("A", "xor", spec.phase_steps),
        ("B", "xnor", spec.phase_steps),
        ("C", "and", spec.phase_steps),
        ("returnA", "xor", spec.revisit_steps),
        ("returnB", "xnor", spec.revisit_steps),
    ]
    metrics = []
    false_after_A = None
    for stage, rule, steps in schedule:
        history = []
        old_banks = pop.allocated.copy()
        start_events = pop.change_events
        start_allocated = pop.allocated.copy()
        per_case_hits = np.zeros(4)
        per_case_total = np.zeros(4)
        for idx, inp in enumerate(rng_stream(seed, steps, pop.n, stage)):
            correct, harvested = pop.episode(
                inp, rule, train=(stage not in ("returnA", "returnB")),
                train_sender=(stage == "A"),
                detect=(condition not in ("single", "oracle")))
            if idx >= max(0, steps-200):
                case_index = inp[0][::2]*2 + inp[0][1::2]
                per_case_total += np.bincount(case_index, minlength=4)
                per_case_hits += np.bincount(
                    case_index, weights=pop.last_good.astype(float), minlength=4)
            history.append(correct)
        tail = float(np.mean(history[-min(200, len(history)):]))
        metrics.append({
            "stage": stage, "rule_hidden_from_cells": rule,
            "tail_success": tail,
            "first50_success": float(np.mean(history[:50])),
            "total_success": float(np.mean(history)),
            "mean_allocated_banks": float(pop.allocated.mean()),
            "allocated_tensor_columns": int(pop.weights.shape[1]),
            "packed_policy_storage_bytes": int(pop.weights.nbytes + pop.baseline.nbytes),
            "max_allocated_banks": int(pop.allocated.max()),
            "case_accuracy_last200": {
                format(k, "02b"): float(per_case_hits[k]/per_case_total[k])
                if per_case_total[k] else None for k in range(4)
            },
            "newly_grown_cells": int(np.count_nonzero(pop.allocated > start_allocated)),
            "local_detection_events": pop.change_events - start_events,
        })
    return {"seed": seed, "condition": condition, "stages": metrics,
            "growth_events": pop.growth_events, "reuse_events": pop.probe_reuses,
            "bank_count_mean": float(pop.allocated.mean()),
            "energy_proxy": pop.energy_balance}


def benchmark(seeds, spec=GrowthSpec()):
    spec.validate()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique seeds required")
    rows = [run_trial(seed, c, spec) for seed in seeds for c in CONDITIONS]
    summary = {}
    for mode in CONDITIONS:
        subset = [r for r in rows if r["condition"] == mode]
        summary[mode] = {stage: {
            "tail_success_mean": float(np.mean([
                next(s for s in r["stages"] if s["stage"] == stage)["tail_success"]
                for r in subset])),
            "avg_banks": float(np.mean([
                next(s for s in r["stages"] if s["stage"] == stage)["mean_allocated_banks"]
                for r in subset])),
        } for stage in ("A", "B", "C", "returnA", "returnB")}
    return {"schema": "dynamic-memory-third-rule-v1", "seeds": seeds,
            "spec": asdict(spec), "summary": summary, "rows": rows,
            "limitations": [
                "Fixed pairs, binary noiseless messages, task functions and detector are authored.",
                "Rule is unavailable to growing cells; oracle is explicitly privileged.",
                "Memory growth is heuristic state allocation, not biological cell growth.",
                "Synthetic costs are not measured joules; third-rule constant-zero baseline is 75%.",
                "Probe success uses local reward and can be fooled by reward noise.",
            ]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="11,12,13")
    p.add_argument("--pairs", type=int, default=48)
    p.add_argument("--phase-steps", type=int, default=5000)
    p.add_argument("--dropout", type=float, default=.08)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    data = benchmark(seeds, GrowthSpec(pairs=args.pairs,
                        phase_steps=args.phase_steps, dropout=args.dropout))
    serialized = json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf8")
        print("Saved", args.output)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
