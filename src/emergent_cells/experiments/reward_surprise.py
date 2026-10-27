"""Unannounced XOR->XNOR->XOR switching via each cell's own reward surprise.

Unlike context_switch.context_memory, agents do NOT receive a season label.
They use a hand-designed local reward moving-average detector to switch between
two private learned action-policy banks. Sender weights freeze after phase A.

This proves neither spontaneous invention of change detection nor open-ended
intelligence: the two-bank capacity, threshold and reward detector are authored.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from .context_switch import SwitchCells, SwitchSpec, draws, evaluate, trial
from .learned_messages import sigmoid


class SurpriseCells(SwitchCells):
    def __init__(self, spec: SwitchSpec, seed: int,
                 decay: float = .95, trigger_below: float = .38,
                 mature_above: float = .85, min_visits: int = 300,
                 cooldown_steps: int = 100):
        super().__init__(spec, seed)
        self.active_bank = np.zeros(self.n, dtype=np.int64)
        self.score = np.full((self.n, 2), .25)
        self.peak = np.full((self.n, 2), .25)
        self.visits = np.zeros((self.n, 2), dtype=np.int64)
        self.cooldown = np.zeros(self.n, dtype=np.int64)
        self.switches = np.zeros(self.n, dtype=np.int64)
        self.decay = decay
        self.trigger_below = trigger_below
        self.mature_above = mature_above
        self.min_visits = min_visits
        self.cooldown_steps = cooldown_steps

    def episode_hidden(self, cues, draws_message, draws_action, *,
                       world_rule: int, train_sender: bool,
                       train_receiver: bool, detect: bool):
        if world_rule not in (0, 1):
            raise ValueError("world rule is binary")
        if cues.shape != (self.n,) or not np.isin(cues, [0, 1]).all():
            raise ValueError("invalid private cues")
        if draws_message.shape != (self.n,) or draws_action.shape != (self.n,):
            raise ValueError("invalid sampling input")
        ids = self.ids
        own = cues * 2 - 1
        p_message = sigmoid(self.sender[:, 0] + self.sender[:, 1] * own)
        send_symbol = (draws_message < p_message).astype(int)
        received_symbol = send_symbol[self.peer]
        feature_slot = cues * 2 + received_symbol
        active = self.active_bank.copy()
        p_action = sigmoid(self.receiver[ids, active, feature_slot])
        action = (draws_action < p_action).astype(int)

        # WORLD alone knows rule. No target nor context enters the cell policy.
        target = (cues[0::2] != cues[1::2]).astype(int)
        if world_rule:
            target = 1 - target
        gate_open = ((action[::2] == target) & (action[1::2] == target))
        pulse = np.repeat(gate_open.astype(float), 2)
        net = (pulse * self.spec.resource_pulse
               - self.spec.metabolism - self.spec.message_cost)
        self.energy_balance += net
        self.messages_sent += self.n
        self.messages_received += self.n

        if train_sender or train_receiver:
            baseline = self.baseline[ids, active]
            advantage = net - baseline
            if train_sender:
                grad = self.spec.learning_rate * advantage * (send_symbol - p_message)
                self.sender[:, 0] += grad
                self.sender[:, 1] += grad * own
                np.clip(self.sender, -12, 12, out=self.sender)
            if train_receiver:
                self.receiver[ids, active, feature_slot] += (
                    self.spec.learning_rate * advantage * (action - p_action))
                np.clip(self.receiver, -12, 12, out=self.receiver)
            self.baseline[ids, active] = (
                self.spec.baseline_decay * baseline
                + (1 - self.spec.baseline_decay) * net
            )

        if detect:
            # Detection sees ONLY each cell's own positive energy pulse.
            # No global schedule, world rule or externally supplied label.
            success = pulse
            old = self.score[ids, active]
            score = self.decay * old + (1 - self.decay) * success
            self.score[ids, active] = score
            # Advanced NumPy indexing returns a COPY; assign back explicitly.
            self.peak[ids, active] = np.maximum(self.peak[ids, active], score)
            self.visits[ids, active] += 1
            self.cooldown[self.cooldown > 0] -= 1
            change = ((self.peak[ids, active] > self.mature_above)
                      & (score < self.trigger_below)
                      & (self.visits[ids, active] >= self.min_visits)
                      & (self.cooldown == 0))
            self.active_bank[change] = 1 - active[change]
            self.switches[change] += 1
            self.cooldown[change] = self.cooldown_steps
        return float(gate_open.mean())


def eval_hidden(cells, seed, spec, world_rule):
    total = 0.
    stream = "eval_A" if world_rule == 0 else "eval_B"
    for cues, msg, action in draws(seed, spec.test_steps, cells.n, stream):
        total += cells.episode_hidden(cues, msg, action, world_rule=world_rule,
                                      train_sender=False, train_receiver=False,
                                      detect=False)
    return total / spec.test_steps


def return_stream(seed, steps, n):
    a = np.random.default_rng(seed + 3100909)
    b = np.random.default_rng(seed + 4100909)
    c = np.random.default_rng(seed + 5100909)
    for _ in range(steps):
        yield a.integers(0, 2, n), b.random(n), c.random(n)


def trial_hidden(seed: int, spec: SwitchSpec = SwitchSpec(),
                 return_detection_episodes: int = 120):
    spec.validate()
    if not 1 <= return_detection_episodes <= 1000:
        raise ValueError("invalid return detection length")
    cells = SurpriseCells(spec, seed)
    for cues, msg, action in draws(seed, spec.train_a, cells.n, "train_A"):
        cells.episode_hidden(cues, msg, action, world_rule=0,
                             train_sender=True, train_receiver=True, detect=True)

    sender_after_a = cells.sender.copy()
    decoder_a_after_a = cells.receiver[:, 0].copy()
    pre_a = eval_hidden(cells, seed, spec, 0)
    pre_b = eval_hidden(cells, seed, spec, 1)
    stage_a_bank1 = float(np.mean(cells.active_bank))
    switches_before_b = cells.switches.copy()

    for cues, msg, action in draws(seed, spec.train_b, cells.n, "train_B"):
        cells.episode_hidden(cues, msg, action, world_rule=1,
                             train_sender=False, train_receiver=True, detect=True)
    post_b = eval_hidden(cells, seed, spec, 1)
    before_return_bank1 = float(np.mean(cells.active_bank))
    switches_before_return = cells.switches.copy()

    # RULE flips without any season signal. Permit ONLY local
    # reward-surprise detection during this stage, not weight training.
    # Evaluate the first 20 episodes and later recovery separately.
    first20 = []
    following = []
    first_flip = np.full(cells.n, -1, dtype=int)
    for step, (cues, msg, action) in enumerate(
        return_stream(seed, return_detection_episodes, cells.n), 1):
        score = cells.episode_hidden(cues, msg, action, world_rule=0,
                                     train_sender=False, train_receiver=False,
                                     detect=True)
        if step <= 20:
            first20.append(score)
        else:
            following.append(score)
        just_flipped = (cells.switches > switches_before_return) & (first_flip < 0)
        first_flip[just_flipped] = step

    recovered_a = eval_hidden(cells, seed, spec, 0)
    paired = first_flip.reshape(-1, 2)
    return {
        "seed": seed,
        "a_before": pre_a, "b_before": pre_b,
        "b_after": post_b,
        "a_return_first20": float(np.mean(first20)),
        "a_return_detection_period": float(np.mean(following)) if following else 0.,
        "a_after_detection_no_weight_training": recovered_a,
        "a_return_detected_fraction": float(np.mean(first_flip > 0)),
        "a_return_detection_latency_mean": float(np.mean(first_flip[first_flip > 0])) if (first_flip > 0).any() else None,
        "a_return_pair_detection_fraction": float(np.mean((paired > 0).all(axis=1))),
        "a_receiver_max_delta_after_a": float(np.max(np.abs(
            decoder_a_after_a - cells.receiver[:, 0]))),
        "sender_max_delta_after_a": float(np.max(np.abs(
            sender_after_a - cells.sender))),
        "b_phase_switch_fraction": float(np.mean(cells.switches > switches_before_b)),
        "bank1_fraction_after_a": stage_a_bank1,
        "bank1_fraction_after_b": before_return_bank1,
        "bank1_fraction_after_return": float(np.mean(cells.active_bank)),
        "total_local_memory_switches": int(cells.switches.sum()),
        "simulated_energy_balance": float(cells.energy_balance.sum()),
    }


def benchmark(seeds: list[int], spec: SwitchSpec = SwitchSpec()):
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("distinct non-empty seeds required")
    observed = [trial_hidden(seed, spec) for seed in seeds]
    oracle = [trial(seed, "context_memory", spec) for seed in seeds]
    blind = [trial(seed, "blind_memory", spec) for seed in seeds]
    def ag(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    return {
        "schema": "local-reward-surprise-context-inference-v1",
        "created": "2026-10-27",
        "status": "exploratory, engineered local detector and two-bank prior",
        "spec": asdict(spec), "seeds": seeds,
        "aggregate": {
            "reward_surprise": {k: ag(observed, k) for k in
                               ("a_before", "b_before", "b_after",
                                "a_return_first20",
                                "a_after_detection_no_weight_training",
                                "a_return_detected_fraction",
                                "a_return_detection_latency_mean",
                                "a_return_pair_detection_fraction",
                                "b_phase_switch_fraction")},
            "provided_context": {
                "a_before": ag(oracle, "a_before"),
                "b_after": ag(oracle, "b_after"),
                "a_after_return": ag(oracle, "a_return_no_training")},
            "blind_memory": {
                "a_before": ag(blind, "a_before"),
                "b_after": ag(blind, "b_after"),
                "a_after_return": ag(blind, "a_return_no_training")}
        },
        "rows": {"reward_surprise": observed, "provided_context": oracle,
                 "blind_memory": blind},
        "limitations": [
            "Reward detector thresholds, two banks, fixed pairs and XOR/XNOR rule are engineered.",
            "Cells receive own energy pulse only; no season bit reaches the reward-surprise arm.",
            "A return uses reward feedback for context detection but NO weight updates.",
            "Protocol encoder is frozen after A, and all costs are simulated units not joules.",
            "This is not spontaneous change-detection algorithm discovery or general intelligence."
        ]
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="21,22,23")
    p.add_argument("--pairs", type=int, default=64)
    p.add_argument("--train-a", type=int, default=4500)
    p.add_argument("--train-b", type=int, default=3500)
    p.add_argument("--test", type=int, default=200)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data = benchmark(seeds, SwitchSpec(pairs=args.pairs, train_a=args.train_a,
                                      train_b=args.train_b, test_steps=args.test))
    text = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf8")
        print("Saved", args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
