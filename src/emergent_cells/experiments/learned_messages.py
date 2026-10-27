"""Learn a 1-bit local message protocol with private reward only.

Date: 2026-10-27. New independent experiment on a fixed pair-gate task.
No XOR term, parity label, global critic or other-cell weight enters the
private sender or receiver update. A four-slot observation/action table
supports arbitrary binary rules; its structure and pairing are designed.
This does not demonstrate natural intelligence or physical joule efficiency.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

CONDITIONS = ("learned_messages", "blocked_messages", "random_messages",
              "fixed_sender", "frozen_actor")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


@dataclass(frozen=True)
class ProtocolSpec:
    pairs: int = 128
    train_steps: int = 8000
    test_steps: int = 500
    learning_rate: float = 0.1
    baseline_decay: float = 0.98
    sender_initial_std: float = 0.12
    receiver_initial_std: float = 0.12
    max_abs_weight: float = 12.0
    resource_pulse: float = 1.0
    metabolism: float = 0.04
    transmission_cost: float = 0.02

    def validate(self):
        if not 2 <= self.pairs <= 4096:
            raise ValueError("pairs out of range")
        if not 1 <= self.train_steps <= 100000 or not 1 <= self.test_steps <= 10000:
            raise ValueError("steps out of range")
        if not 0 < self.learning_rate <= 2 or not 0 < self.baseline_decay < 1:
            raise ValueError("invalid training hyperparameters")
        if not 0 <= self.sender_initial_std <= 5 or not 0 <= self.receiver_initial_std <= 5:
            raise ValueError("invalid initial weights")
        if not 1 <= self.max_abs_weight <= 30:
            raise ValueError("invalid weight bounds")
        if not 0 < self.resource_pulse <= 10:
            raise ValueError("invalid resource pulse")
        if not 0 <= self.metabolism <= 1 or not 0 <= self.transmission_cost <= 1:
            raise ValueError("invalid synthetic energy costs")


class LocalProtocolPopulation:
    """Each cell owns independent encoder, decoder and homeostatic baseline."""

    def __init__(self, spec: ProtocolSpec, seed: int):
        spec.validate()
        self.spec = spec
        self.n = spec.pairs * 2
        init = np.random.default_rng(seed + 1000003)
        self.sender = init.normal(0, spec.sender_initial_std, (self.n, 2))
        self.receiver = init.normal(0, spec.receiver_initial_std, (self.n, 4))
        self.baseline = np.zeros(self.n)
        self.cumulative_energy_delta = np.zeros(self.n)
        self.transmitted_bits = 0
        self.received_bits = 0
        self.opened_gates = 0
        self.episode_count = 0
        self.ids = np.arange(self.n)
        self.neighbor = self.ids ^ 1

    def episode(self, cues, send_uniform, act_uniform, condition="learned_messages", train=True):
        if condition not in CONDITIONS:
            raise ValueError("unknown condition")
        if cues.shape != (self.n,) or not np.isin(cues, [0, 1]).all():
            raise ValueError("private cues must be binary")
        if send_uniform.shape != (self.n,) or act_uniform.shape != (self.n,):
            raise ValueError("random draw shape mismatch")
        own = cues.astype(float) * 2 - 1
        send_p = sigmoid(self.sender[:, 0] + self.sender[:, 1] * own)
        symbol = (send_uniform < send_p).astype(int)
        if condition == "random_messages":
            symbol = (send_uniform < 0.5).astype(int)
        received = (np.zeros(self.n, dtype=int) if condition == "blocked_messages"
                    else symbol[self.neighbor])
        index = cues * 2 + received  # generic observation slot; NOT XOR
        action_p = sigmoid(self.receiver[self.ids, index])
        action = (act_uniform < action_p).astype(int)

        # External WORLD rule. The cell learner only sees its own reward.
        target = (cues[0::2] != cues[1::2]).astype(int)
        opened = (action[0::2] == target) & (action[1::2] == target)
        local_delta = (np.repeat(opened, 2).astype(float) * self.spec.resource_pulse
                       - self.spec.metabolism - self.spec.transmission_cost)
        self.cumulative_energy_delta += local_delta
        self.transmitted_bits += self.n  # same attempted bandwidth and cost
        self.received_bits += 0 if condition == "blocked_messages" else self.n
        self.opened_gates += int(opened.sum())
        self.episode_count += 1
        if train:
            advantage = local_delta - self.baseline
            if condition in ("learned_messages", "frozen_actor"):
                grad = advantage * (symbol - send_p)
                self.sender[:, 0] += self.spec.learning_rate * grad
                self.sender[:, 1] += self.spec.learning_rate * grad * own
            if condition != "frozen_actor":
                self.receiver[self.ids, index] += (
                    self.spec.learning_rate * advantage * (action - action_p))
            np.clip(self.sender, -self.spec.max_abs_weight,
                    self.spec.max_abs_weight, out=self.sender)
            np.clip(self.receiver, -self.spec.max_abs_weight,
                    self.spec.max_abs_weight, out=self.receiver)
            self.baseline = (self.spec.baseline_decay * self.baseline
                             + (1 - self.spec.baseline_decay) * local_delta)
        return {"opened": int(opened.sum()), "opened_fraction": float(opened.mean())}

    def symbol_separation(self):
        p0 = sigmoid(self.sender[:, 0] - self.sender[:, 1])
        p1 = sigmoid(self.sender[:, 0] + self.sender[:, 1])
        return float(np.mean(np.abs(p0 - p1)))


def streams(seed, steps, n, phase, bias=.5):
    salt = {"train": 3000001, "test": 6000001, "biased_test": 9000001}[phase]
    a = np.random.default_rng(seed + salt)
    b = np.random.default_rng(seed + salt + 1000000)
    c = np.random.default_rng(seed + salt + 2000000)
    for _ in range(steps):
        yield (a.random(n) < bias).astype(int), b.random(n), c.random(n)


def evaluate(pop, seed, episodes, mode, phase="test", bias=.5):
    opened = 0
    for cue, send, action in streams(seed, episodes, pop.n, phase, bias):
        opened += pop.episode(cue, send, action, mode, train=False)["opened"]
    return float(opened / (episodes * pop.spec.pairs))


def trial(seed, condition, spec=ProtocolSpec()):
    spec.validate()
    if condition not in CONDITIONS:
        raise ValueError("unsupported condition")
    p = LocalProtocolPopulation(spec, seed)
    for cue, send, action in streams(seed, spec.train_steps, p.n, "train"):
        p.episode(cue, send, action, condition, train=True)
    separation = p.symbol_separation()
    test = evaluate(copy.deepcopy(p), seed, spec.test_steps, condition)
    knockouts = None
    shifted = None
    if condition == "learned_messages":
        blocked = evaluate(copy.deepcopy(p), seed, spec.test_steps, "blocked_messages")
        random = evaluate(copy.deepcopy(p), seed, spec.test_steps, "random_messages")
        shifted = evaluate(copy.deepcopy(p), seed, spec.test_steps,
                           "learned_messages", phase="biased_test", bias=.8)
        knockouts = {"block_received_bits": blocked, "replace_with_random_bits": random}
    return {
        "seed": int(seed), "condition": condition,
        "test_success_fraction": test, "test_gate_count": spec.pairs * spec.test_steps,
        "sender_symbol_separation": separation,
        "biased_cue_success_fraction": shifted, "same_policy_knockouts": knockouts,
        "training_messages_attempted": p.transmitted_bits,
        "training_messages_received": p.received_bits,
        "training_gates_opened": p.opened_gates,
        "cumulative_training_energy_proxy": float(p.cumulative_energy_delta.sum()),
    }


def benchmark(seeds, spec=ProtocolSpec()):
    spec.validate()
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty distinct seed list required")
    if any(type(seed) is not int or seed < 0 or seed > 2**32 - 10000000 for seed in seeds):
        raise ValueError("invalid seed range")
    rows = [trial(seed, condition, spec) for seed in seeds for condition in CONDITIONS]
    agg = {}
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        agg[condition] = {
            field: {"mean": float(np.mean([r[field] for r in group])),
                    "std_population": float(np.std([r[field] for r in group]))}
            for field in ("test_success_fraction", "sender_symbol_separation",
                          "cumulative_training_energy_proxy")
        }
    learned = [r for r in rows if r["condition"] == "learned_messages"]
    agg["same_policy_knockouts"] = {
        "mean_received_blocked": float(np.mean(
            [r["same_policy_knockouts"]["block_received_bits"] for r in learned])),
        "mean_randomized_messages": float(np.mean(
            [r["same_policy_knockouts"]["replace_with_random_bits"] for r in learned])),
        "mean_biased_cues": float(np.mean(
            [r["biased_cue_success_fraction"] for r in learned]))
    }
    lookup = {(r["seed"], r["condition"]): r for r in rows}
    diffs = {
        control: {
            "mean_paired_difference": float(np.mean([
                lookup[seed, "learned_messages"]["test_success_fraction"]
                - lookup[seed, control]["test_success_fraction"] for seed in seeds])),
            "wins": sum(lookup[seed, "learned_messages"]["test_success_fraction"]
                        > lookup[seed, control]["test_success_fraction"] for seed in seeds),
            "pairs": len(seeds),
        }
        for control in CONDITIONS if control != "learned_messages"
    }
    return {
        "schema": "learned-local-symbol-protocol-v1",
        "created": "2026-10-27",
        "status": "exploratory, task rules and channel fixed; not AGI",
        "seeds": seeds, "spec": asdict(spec), "conditions": CONDITIONS,
        "aggregate": agg, "paired_contrasts": diffs, "rows": rows,
        "limitations": [
            "Every receiver has an engineered four-slot table; no XOR term is supplied.",
            "Task is the external XOR gate, same at training and testing.",
            "Fixed partner, noiseless one-bit channel, immediate local reward.",
            "Cell learning uses own energy delta only; Python still batches actions.",
            "No autonomous topology, growth, substrate independence, or AGI.",
            "Synthetic energy proxy is NOT physical joules.",
        ],
    }


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--seeds", default="101,102,103,104,105")
    a.add_argument("--pairs", type=int, default=128)
    a.add_argument("--train", type=int, default=8000)
    a.add_argument("--test", type=int, default=500)
    a.add_argument("--output", type=Path)
    args = a.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    data = benchmark(seeds, ProtocolSpec(pairs=args.pairs,
                     train_steps=args.train, test_steps=args.test))
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print("Saved", args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
