"""Cooperation-required dual-cue local-gate experiment (Oct 26, exploratory).

Two adjacent cells receive independent private bits. A shared environmental
gate yields finite resources only if BOTH cells select the XOR parity lever.
The gate computes whether the environment yields food, but neither cell is
given that answer or any global reward. Each cell learns from its energy delta.

CRITICAL LIMITATION: the product feature private*peer is engineered, not
self-discovered. This tests causal information-sharing, NOT emergent cognition.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

CONDITIONS = ("none", "local", "scrambled", "frozen", "far")


@dataclass(frozen=True)
class GateSpec:
    pairs: int = 128
    train_episodes: int = 1200
    test_episodes: int = 300
    radius: float = 1.5
    learning_rate: float = 4.0
    temperature: float = 1.0
    baseline_decay: float = 0.975
    resource_gain: float = 0.11
    metabolism: float = 0.018
    message_cost: float = 0.002
    initial_energy: float = 2.0
    maximum_energy: float = 4.0
    weight_limit: float = 9.0

    def validate(self):
        if not 2 <= self.pairs <= 4096:
            raise ValueError("pairs must be 2..4096")
        if not 1 <= self.train_episodes <= 100000 or not 1 <= self.test_episodes <= 100000:
            raise ValueError("episodes must be 1..100000")
        if not 0 < self.radius <= 8:
            raise ValueError("radius must be (0, 8]")
        if not 0 < self.learning_rate <= 20:
            raise ValueError("learning rate must be (0,20]")
        if not 0.1 <= self.temperature <= 10:
            raise ValueError("temperature must be 0.1..10")
        if not 0 < self.baseline_decay < 1:
            raise ValueError("baseline decay must be (0,1)")
        if not 0 <= self.metabolism <= 1 or not 0 <= self.message_cost <= 1:
            raise ValueError("invalid energy costs")
        if not 0 < self.resource_gain <= 1:
            raise ValueError("resource_gain must be (0,1]")
        if not 0 < self.initial_energy <= self.maximum_energy <= 100:
            raise ValueError("invalid energy bounds")
        if not 1 <= self.weight_limit <= 100:
            raise ValueError("invalid weight limit")


class GateCells:
    """Each fixed local pair is two cells with separate private learned weights.

    Pair identities and positions are predefined (not self-organized topology).
    Policies have no global information; the experiment evaluator does.
    """

    def __init__(self, spec: GateSpec, seed: int):
        spec.validate()
        self.spec = spec
        self.count = 2 * spec.pairs
        self.weights = np.random.default_rng(seed + 1000003).normal(
            0, 0.05, size=(self.count, 4))
        self.baseline = np.zeros(self.count, dtype=float)
        self.energy = np.full(self.count, spec.initial_energy, dtype=float)
        self.alive = np.ones(self.count, dtype=bool)
        self.messages = 0
        self.resource_received = 0.0
        columns = int(np.ceil(np.sqrt(spec.pairs)))
        base_x = (np.arange(spec.pairs) % columns) * 4
        base_y = (np.arange(spec.pairs) // columns) * 4
        self.x = np.repeat(base_x, 2) + np.tile([0, 1], spec.pairs)
        self.y = np.repeat(base_y, 2)

    def features(self, bits: np.ndarray, condition: str,
                 corrupt_rng: np.random.Generator | None = None):
        if condition not in CONDITIONS:
            raise ValueError("Unknown condition")
        if bits.shape != (self.spec.pairs, 2):
            raise ValueError("Exactly two private cues per pair required")
        own = bits.reshape(self.count).astype(float)
        received = np.zeros(self.count)
        if condition in ("local", "scrambled", "frozen", "far"):
            # Only an in-range, alive physical partner may supply this cue.
            delta_x = self.x[0::2] - self.x[1::2]
            delta_y = self.y[0::2] - self.y[1::2]
            within_range = (delta_x**2 + delta_y**2 <= self.spec.radius**2)
            within_range &= self.alive.reshape(-1, 2).all(axis=1)
            if np.any(within_range):
                neighbor_bits = bits[:, ::-1].reshape(self.count)
                if condition == "scrambled":
                    if corrupt_rng is None:
                        raise ValueError("Corrupted-message control needs RNG")
                    shift = int(corrupt_rng.integers(1, self.spec.pairs))
                    neighbor_bits = np.roll(bits, shift, axis=0)[:, ::-1].reshape(self.count)
                received[np.repeat(within_range, 2)] = neighbor_bits[np.repeat(within_range, 2)]
        # On absence of neighbor, the product term is identically zero.
        features = np.column_stack((np.ones(self.count), own,
                                    received, own * received))
        return features, int(np.count_nonzero(received[self.alive]))

    def episode(self, bits: np.ndarray, action_uniform: np.ndarray,
                condition: str, train: bool,
                corrupt_rng: np.random.Generator | None = None):
        if action_uniform.shape != (self.count,):
            raise ValueError("One action random number per cell required")
        features, messages = self.features(bits, condition, corrupt_rng)
        self.messages += messages
        logit = np.clip(np.sum(features * self.weights, axis=1)
                        / self.spec.temperature, -25, 25)
        probability = 1.0 / (1.0 + np.exp(-logit))
        action = action_uniform < probability

        # Shared resource rule belongs to world physics, not to local policy.
        # Both agents must operate the gate according to cue parity.
        hidden_target = bits[:, 0] != bits[:, 1]
        pair_alive = self.alive.reshape(-1, 2).all(axis=1)
        joint_correct = (action.reshape(-1, 2) == hidden_target[:, None]).all(axis=1)
        opened = pair_alive & joint_correct

        before = self.energy.copy()
        reward = np.repeat(opened, 2) * self.spec.resource_gain
        message_paid = (features[:, 2] != 0).astype(float)
        cost = self.spec.metabolism + self.spec.message_cost * message_paid
        self.energy[self.alive] = np.clip(
            self.energy[self.alive] + reward[self.alive] - cost[self.alive],
            0.0, self.spec.maximum_energy)
        own_gain = self.energy - before
        self.alive &= self.energy > 0
        self.resource_received += float(reward.sum())

        if train and condition != "frozen":
            # Purely local homeostatic policy update; no parity label here.
            advantage = own_gain - self.baseline
            adjustment = (self.spec.learning_rate * advantage[:, None]
                          * (action.astype(float) - probability)[:, None]
                          * features)
            self.weights[self.alive] += adjustment[self.alive]
            np.clip(self.weights, -self.spec.weight_limit,
                    self.spec.weight_limit, out=self.weights)
            self.baseline[self.alive] = (
                self.spec.baseline_decay * self.baseline[self.alive]
                + (1.0 - self.spec.baseline_decay) * own_gain[self.alive])
        return {
            "opened": int(opened.sum()),
            "active_pairs": int(pair_alive.sum()),
            "messages": messages,
            "living_cells": int(self.alive.sum())
        }


def trial(seed: int, condition: str, spec: GateSpec = GateSpec()) -> dict:
    spec.validate()
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    pop = GateCells(spec, seed)
    if condition == "far":
        # Move each second partner two additional lattice units away.
        # No rule or hidden cue is altered; the communication range is exceeded.
        pop.y[1::2] += 2
    cues_rng = np.random.default_rng(seed + 3000001)
    choice_rng = np.random.default_rng(seed + 4000001)
    bad_rng = np.random.default_rng(seed + 5000001)
    total_train_opened = 0
    train_snapshots = []
    for t in range(spec.train_episodes):
        bits = cues_rng.integers(0, 2, size=(spec.pairs, 2)) * 2 - 1
        state = pop.episode(bits, choice_rng.random(pop.count), condition,
                            train=True, corrupt_rng=bad_rng)
        total_train_opened += state["opened"]
        if t in (0, spec.train_episodes // 2, spec.train_episodes - 1):
            train_snapshots.append({"episode": t + 1,
                                    "fraction": state["opened"] / spec.pairs})

    # Frozen clone tests whether the SAME trained policies rely on messages.
    # This is a direct message knock-out, not an independently retrained control.
    blocked_pop = deepcopy(pop) if condition == "local" else None

    # Held-out stimulus sequences and stochastic choices, not novel task rules.
    test_cues = np.random.default_rng(seed + 6000001)
    test_actions = np.random.default_rng(seed + 7000001)
    test_corrupt = np.random.default_rng(seed + 8000001)
    total_opened = 0
    total_active_pairs = 0
    for _ in range(spec.test_episodes):
        bits = test_cues.integers(0, 2, size=(spec.pairs, 2)) * 2 - 1
        state = pop.episode(bits, test_actions.random(pop.count), condition,
                            train=False, corrupt_rng=test_corrupt)
        total_opened += state["opened"]
        total_active_pairs += state["active_pairs"]

    # Same held-out cues/actions, same trained local weights, but no messages.
    blocked_fraction = None
    if blocked_pop is not None:
        cue_rng = np.random.default_rng(seed + 6000001)
        action_rng = np.random.default_rng(seed + 7000001)
        blocked_opened = 0
        for _ in range(spec.test_episodes):
            test_bits = cue_rng.integers(0, 2, size=(spec.pairs, 2)) * 2 - 1
            blocked = blocked_pop.episode(test_bits, action_rng.random(blocked_pop.count),
                                           "none", train=False)
            blocked_opened += blocked["opened"]
        blocked_fraction = blocked_opened / (spec.test_episodes * spec.pairs)

    return {
        "seed": seed, "condition": condition,
        "training_fraction": total_train_opened / (spec.train_episodes * spec.pairs),
        "test_fraction": total_opened / (spec.test_episodes * spec.pairs),
        "test_given_active": total_opened / total_active_pairs if total_active_pairs else 0.0,
        "trained_local_without_messages": blocked_fraction,
        "living_cells": int(pop.alive.sum()),
        "remaining_energy": float(pop.energy.sum()),
        "resource_received": float(pop.resource_received),
        "message_count": int(pop.messages),
        "mean_abs_weights": float(np.abs(pop.weights).mean()),
        "train_snapshots": train_snapshots,
    }


def benchmark(seeds: list[int], spec: GateSpec = GateSpec()) -> dict:
    spec.validate()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Distinct non-empty seeds are required")
    rows = [trial(seed, condition, spec) for seed in seeds
            for condition in CONDITIONS]
    summaries = {}
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        summaries[condition] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in group])),
                "std_population": float(np.std([r[metric] for r in group]))
            }
            for metric in ("training_fraction", "test_fraction", "test_given_active",
                           "living_cells", "remaining_energy", "message_count")
        }
    return {
        "schema": "dual-cue-local-cooperation-v1",
        "status": "Exploratory Oct 26 experiment; not qualifying hackathon evidence",
        "question": "Can cell-local messages enable a gate requiring two complementary private cues?",
        "parameters": asdict(spec),
        "seeds": seeds,
        "conditions": {
            "none": "Private cue only; learning active.",
            "local": "Receives true adjacent peer cue; learning active.",
            "scrambled": "Negative control: received cue replaced from different pair.",
            "frozen": "True local cue but cell policy never updates.",
            "far": "Same learning, peer out of communication range."
        },
        "limitations": [
            "XOR-capable private*peer feature is engineered, not learned spontaneously.",
            "Predefined fixed partners, not emergent topology or growth.",
            "Gate's parity rule is external simulated physics; only the resulting own energy delta reaches learner.",
            "Test uses independent cue sequences under the same rule, not held-out tasks.",
            "Direct message knock-out holds trained weights fixed and replays the same held-out stimuli/actions.",
            "Message channel is noiseless and predefined; scrambled condition intentionally violates locality for a negative control.",
            "Finite synthetic metabolic energy is not measured physical joules.",
            "Oct 26 work predates guide's Oct 27 stated competition build window."
        ],
        "summary": summaries, "rows": rows
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--pairs", type=int, default=128)
    p.add_argument("--train", type=int, default=1200)
    p.add_argument("--test", type=int, default=300)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    result = benchmark(seeds, GateSpec(pairs=args.pairs,
                        train_episodes=args.train, test_episodes=args.test))
    body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
        print(f"Saved {args.output}")
    else:
        print(body)


if __name__ == "__main__":
    main()
