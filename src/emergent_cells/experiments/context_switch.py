"""A->B->A continual learning of *locally rewarded* learned cell communication.

World: each fixed pair receives two private bits, transmits learned discrete
symbols, and chooses its own lever. World A opens at XOR; world B at XNOR.
No cell policy observes parity labels or other cells' parameters. A visible
local season/context cue (0/1) is supplied ONLY to the contextual-memory arm.
This is deliberately easier than undetected nonstationarity; it tests
retention with per-cell memory slots, not autonomous task discovery.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from .learned_messages import ProtocolSpec, LocalProtocolPopulation, sigmoid

CONDITIONS = ("context_memory", "single_policy", "reset_on_switch",
              "blind_memory", "frozen_on_switch", "no_messages")


@dataclass(frozen=True)
class SwitchSpec:
    pairs: int = 128
    train_a: int = 8000
    train_b: int = 4000
    test_steps: int = 400
    learning_rate: float = 0.1
    baseline_decay: float = 0.98
    message_cost: float = 0.02
    metabolism: float = 0.04
    resource_pulse: float = 1.0

    def validate(self):
        if not 4 <= self.pairs <= 4096:
            raise ValueError("pairs must be 4..4096")
        if not 1 <= self.train_a <= 100000 or not 1 <= self.train_b <= 100000:
            raise ValueError("training steps must be 1..100000")
        if not 1 <= self.test_steps <= 10000:
            raise ValueError("test_steps must be 1..10000")
        if not 0 < self.learning_rate <= 2 or not 0 < self.baseline_decay < 1:
            raise ValueError("invalid learning settings")
        if not 0 <= self.message_cost <= 1 or not 0 <= self.metabolism <= 1:
            raise ValueError("energy cost out of range")
        if not 0 < self.resource_pulse <= 10:
            raise ValueError("resource_pulse out of range")


class SwitchCells:
    """Independent private policies, retained sender code, optional local memory."""

    def __init__(self, spec: SwitchSpec, seed: int):
        spec.validate()
        self.spec = spec
        self.n = 2 * spec.pairs
        # Match A-stage initial weights with learned_messages v1 exactly.
        init = np.random.default_rng(seed + 1000003)
        self.sender = init.normal(0, 0.12, (self.n, 2))
        initial = init.normal(0, 0.12, (self.n, 4))
        self.receiver = np.stack((initial.copy(), initial.copy()), axis=1)
        self.baseline = np.zeros((self.n, 2), dtype=float)
        self.ids = np.arange(self.n)
        self.peer = self.ids ^ 1
        self.messages_sent = 0
        self.messages_received = 0
        self.energy_balance = np.zeros(self.n)

    def episode(self, cues, draws_msg, draws_action, *,
                rule, condition, context, train_sender, train_receiver):
        if rule not in (0, 1) or context not in (0, 1):
            raise ValueError("rule/context are binary environmental configuration")
        if condition not in CONDITIONS:
            raise ValueError("unknown control")
        if cues.shape != (self.n,) or not np.isin(cues, [0, 1]).all():
            raise ValueError("invalid private cue")
        if draws_msg.shape != (self.n,) or draws_action.shape != (self.n,):
            raise ValueError("invalid action draws")
        # Each sender encodes its own cue; no parity or global gradient.
        own = cues * 2 - 1
        msg_p = sigmoid(self.sender[:, 0] + self.sender[:, 1] * own)
        sent = (draws_msg < msg_p).astype(int)
        received = (np.zeros(self.n, dtype=int) if condition == "no_messages"
                    else sent[self.peer])
        self.messages_sent += self.n
        self.messages_received += self.n if condition != "no_messages" else 0
        # The *memory* arm gets a local environmental season bit, not parity.
        bank = context if condition == "context_memory" else 0
        index = cues * 2 + received  # generic 4-state lookup, NOT XOR-coded
        action_p = sigmoid(self.receiver[self.ids, bank, index])
        actions = (draws_action < action_p).astype(int)

        # Reward is world physics. Cells only receive OWN energy change.
        goal = (cues[0::2] != cues[1::2]).astype(int)
        if rule == 1:
            goal = 1 - goal  # XNOR world B
        opens = ((actions[0::2] == goal) & (actions[1::2] == goal))
        advantage_raw = (np.repeat(opens.astype(float), 2) * self.spec.resource_pulse
                         - self.spec.metabolism - self.spec.message_cost)
        self.energy_balance += advantage_raw
        if train_sender or train_receiver:
            baseline = self.baseline[:, bank]
            advantage = advantage_raw - baseline
            if train_sender and condition != "no_messages":
                pg = advantage * (sent - msg_p)
                self.sender[:, 0] += self.spec.learning_rate * pg
                self.sender[:, 1] += self.spec.learning_rate * pg * own
                np.clip(self.sender, -12.0, 12.0, out=self.sender)
            if train_receiver:
                self.receiver[self.ids, bank, index] += (
                    self.spec.learning_rate * advantage * (actions - action_p))
                np.clip(self.receiver, -12.0, 12.0, out=self.receiver)
            self.baseline[:, bank] = (
                self.spec.baseline_decay * baseline +
                (1 - self.spec.baseline_decay) * advantage_raw
            )
        return float(opens.mean())

    def sender_separation(self):
        p0 = sigmoid(self.sender[:, 0] - self.sender[:, 1])
        p1 = sigmoid(self.sender[:, 0] + self.sender[:, 1])
        return float(np.mean(np.abs(p0 - p1)))


def draws(seed, steps, count, stream):
    # Separate, reproducible world states and action RNG, no hidden labels.
    salt = {"train_A": 101, "train_B": 202,
            "eval_A": 303, "eval_B": 404}[stream]
    a = np.random.default_rng(seed + 3000001 + salt)
    b = np.random.default_rng(seed + 4000001 + salt)
    c = np.random.default_rng(seed + 5000001 + salt)
    for _ in range(steps):
        yield a.integers(0, 2, count), b.random(count), c.random(count)


def evaluate(cells, spec, seed, condition, rule, stream):
    total = 0.0
    for cues, send, action in draws(seed, spec.test_steps, cells.n, stream):
        total += cells.episode(cues, send, action, rule=rule,
                               condition=condition, context=rule,
                               train_sender=False, train_receiver=False)
    return total / spec.test_steps


def trial(seed: int, condition: str, spec: SwitchSpec = SwitchSpec()) -> dict:
    spec.validate()
    if condition not in CONDITIONS:
        raise ValueError("invalid condition")
    cells = SwitchCells(spec, seed)
    # Phase A, all arms choose bank 0. Both decoder and sender learn.
    for cues, send, action in draws(seed, spec.train_a, cells.n, "train_A"):
        cells.episode(cues, send, action, rule=0, context=0,
                      condition=condition, train_sender=True, train_receiver=True)
    sender_A = cells.sender.copy()
    decoder_A = cells.receiver[:, 0, :].copy()
    accuracy_A_before = evaluate(cells, spec, seed, condition, 0, "eval_A")
    accuracy_B_before = evaluate(cells, spec, seed, condition, 1, "eval_B")

    # A deliberately memory-free control clears its learned A decoder to
    # escape policy saturation at task reversal; this can learn B but
    # overwrites its only decoder. The reset is externally triggered.
    if condition == "reset_on_switch":
        cells.receiver[:, 0, :] = cells.receiver[:, 1, :].copy()
        cells.baseline[:, 0] = 0.0

    # Phase B: freezes ALL senders, so changing the message code cannot break A.
    # This isolates whether a local decoder can learn B without overwriting A.
    history = []
    total = 0.0
    for step, (cues, send, action) in enumerate(
            draws(seed, spec.train_b, cells.n, "train_B"), 1):
        outcome = cells.episode(cues, send, action, rule=1, context=1,
                                condition=condition,
                                train_sender=False,
                                train_receiver=condition != "frozen_on_switch")
        total += outcome
        if step % max(1, spec.train_b // 8) == 0:
            history.append({"training_step": step,
                            "window_success": total / max(1, spec.train_b // 8)})
            total = 0.0

    accuracy_B_after = evaluate(cells, spec, seed, condition, 1, "eval_B")
    accuracy_A_return = evaluate(cells, spec, seed, condition, 0, "eval_A")
    return {
        "seed": seed, "condition": condition,
        "a_before": accuracy_A_before,
        "b_before": accuracy_B_before,
        "b_after": accuracy_B_after,
        "a_return_no_training": accuracy_A_return,
        "a_retention_drop": accuracy_A_before - accuracy_A_return,
        "sender_weight_change_after_A": float(np.max(np.abs(sender_A-cells.sender))),
        "a_decoder_weight_change_after_A": float(
            np.max(np.abs(decoder_A-cells.receiver[:, 0, :]))),
        "private_sender_symbol_separation": cells.sender_separation(),
        "parameter_count": int(cells.sender.size + cells.receiver.size),
        "b_training_history": history,
        "messages_sent": cells.messages_sent,
        "messages_received": cells.messages_received,
        "simulated_energy_balance": float(cells.energy_balance.sum()),
    }


def benchmark(seeds, spec=SwitchSpec()):
    spec.validate()
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty unique seeds")
    if any(type(seed) is not int or not 0 <= seed <= 2**32-10000000 for seed in seeds):
        raise ValueError("invalid seed")
    rows = [trial(seed, condition, spec)
            for seed in seeds for condition in CONDITIONS]
    grouped = {}
    for mode in CONDITIONS:
        rr = [r for r in rows if r["condition"] == mode]
        grouped[mode] = {
            field: {
                "mean": float(np.mean([r[field] for r in rr])),
                "std_population": float(np.std([r[field] for r in rr]))
            }
            for field in ("a_before", "b_before", "b_after",
                          "a_return_no_training", "a_retention_drop")
        }
    return {
        "schema": "contextual-A-B-A-memory-v1",
        "date": "2026-10-27",
        "status": "exploratory task-switching and memory benchmark",
        "spec": asdict(spec), "seeds": seeds, "results": rows,
        "aggregate": grouped,
        "task_design": {
            "A": "XOR shared gate", "B": "XNOR shared gate",
            "context_memory": "season indicator 0/1 externally provided to a cell",
            "single_policy": "one shared decoder trained sequentially",
            "reset_on_switch": "one decoder externally reset at change, then retrained B",
            "blind_memory": "two banks available but environment context ignored",
            "frozen_on_switch": "decoder frozen after A; no adaptation",
            "no_messages": "bit reception masked in both phases",
        },
        "limitations": [
            "Environment rule and visible season context are engineered by authors.",
            "Only XOR->XNOR->XOR on fixed pair gates, not arbitrary task transfer.",
            "The context memory arm has 8 used decoder slots versus 4 in single-policy.",
            "Reset-on-switch is a deliberately external action, not autonomous change detection.",
            "Senders freeze during phase B for all variants; protocol is retained by design.",
            "No adaptive topology, growth, discovered task changes or measured joules.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="11,12,13")
    p.add_argument("--pairs", type=int, default=128)
    p.add_argument("--train-a", type=int, default=8000)
    p.add_argument("--train-b", type=int, default=4000)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    report = benchmark(seeds, SwitchSpec(pairs=a.pairs,
                       train_a=a.train_a, train_b=a.train_b,
                       test_steps=a.test))
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if a.output:
        a.output.parent.mkdir(exist_ok=True, parents=True)
        a.output.write_text(text+"\n", encoding="utf8")
        print("Saved", a.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
