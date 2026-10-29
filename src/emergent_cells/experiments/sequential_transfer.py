"""Controlled sequential transfer study: frozen prior banks and new unseen Boolean rules.

Research simulation ONLY. Pretraining bank IDs and probe scheduling are experimenter-
controlled; novel-task policies receive only private cues, learned peer messages and
local pair reward. Transfer mechanisms are heuristics, not state-of-the-art replicas.
No physical energy measurement or task-general intelligence is demonstrated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .dynamic_memory import GrowthSpec, GrowingCells, rng_stream
from .learned_messages import sigmoid

STRATEGIES = ("scratch", "scratch_equal_budget", "last_clone", "selected_clone",
              "selected_tempered", "slot_mosaic", "worst_clone")
RULES = ("or", "nand")


def target(bits, rule):
    a, b = bits[::2], bits[1::2]
    if rule == "or":
        return a | b
    if rule == "nand":
        return 1 - (a & b)
    raise ValueError("unknown novel rule")


def stream(seed, stage, steps, n):
    # Every strategy sees the same stochastic inputs, messages, actions and dropout.
    salt = {"probe": 71, "adapt": 81, "pretest": 91, "posttest": 101}[stage]
    r = np.random.default_rng(seed * 997 + salt + 1)
    for _ in range(steps):
        yield (r.integers(0, 2, n), r.random(n), r.random(n), r.random(n // 2))


def evaluate(sender, weights, seed, rule, steps=300, dropout=.08):
    n = len(sender)
    peer = np.arange(n) ^ 1
    good = []
    per_case_good = np.zeros(4)
    per_case_n = np.zeros(4)
    for bits, um, ua, ur in stream(seed, "pretest", steps, n):
        msg = (um < sigmoid(sender[:, 0] + sender[:, 1] * (2 * bits - 1))).astype(int)
        slot = 2 * bits + msg[peer]
        p = .97 * sigmoid(weights[np.arange(n), slot]) + .015
        actions = (ua < p).astype(int)
        success = (actions[::2] == target(bits, rule)) & (actions[1::2] == target(bits, rule))
        good.append(success.mean())
        case = 2 * bits[::2] + bits[1::2]
        per_case_n += np.bincount(case, minlength=4)
        per_case_good += np.bincount(case, weights=success.astype(float), minlength=4)
    return {"accuracy": float(np.mean(good)),
            "case_accuracy": (per_case_good / np.maximum(per_case_n, 1)).tolist()}


def probe_sources(sender, banks, seed, rule, probe_per_bank, dropout):
    n = len(sender)
    ids = np.arange(n)
    peer = ids ^ 1
    pair_n = n // 2
    success = np.zeros((3, pair_n))
    slot_hits = np.zeros((3, n, 4))
    slot_counts = np.zeros((3, n, 4))
    trials = list(stream(seed, "probe", probe_per_bank, n))
    for bank in range(3):
        for bits, um, ua, ur in trials:
            msg = (um < sigmoid(sender[:, 0] + sender[:, 1] * (2 * bits - 1))).astype(int)
            slot = 2 * bits + msg[peer]
            p = .97 * sigmoid(banks[ids, bank, slot]) + .015
            acts = (ua < p).astype(int)
            good = (acts[::2] == target(bits, rule)) & (acts[1::2] == target(bits, rule))
            # Availability is not exposed to the learner: it only sees actual reward.
            pulse = good & (ur >= dropout)
            success[bank] += pulse
            for cell in range(n):
                slot_counts[bank, cell, slot[cell]] += 1
                slot_hits[bank, cell, slot[cell]] += float(pulse[cell // 2])
    local_pair_best = np.argmax(success, axis=0)
    local_pair_worst = np.argmin(success, axis=0)
    best = np.repeat(local_pair_best, 2)
    worst = np.repeat(local_pair_worst, 2)
    mosaic = np.zeros((n, 4))
    # Local, input-conditional reward score, shrinkage helps scarce observations.
    smoothed = (slot_hits + 1) / (slot_counts + 2)
    for cell in range(n):
        for slot in range(4):
            k = int(np.argmax(smoothed[:, cell, slot]))
            mosaic[cell, slot] = banks[cell, k, slot]
    return best, worst, mosaic, (success / probe_per_bank).mean(axis=1).tolist()


def initialize(banks, best, worst, mosaic, strategy):
    n = len(best)
    idx = np.arange(n)
    if strategy in ("scratch", "scratch_equal_budget"):
        return np.zeros((n, 4))
    if strategy == "last_clone":
        return banks[:, 2, :].copy()
    if strategy == "selected_clone":
        return banks[idx, best, :].copy()
    if strategy == "selected_tempered":
        return .35 * banks[idx, best, :].copy()
    if strategy == "slot_mosaic":
        return .55 * mosaic.copy()
    if strategy == "worst_clone":
        return banks[idx, worst, :].copy()
    raise ValueError(strategy)


def learn_new(sender, weights, seed, rule, spec, adapt_steps, stage="adapt"):
    n = len(sender)
    ids = np.arange(n)
    peer = ids ^ 1
    policy = weights.copy()
    baseline = np.zeros(n)
    history = []
    for bits, um, ua, ur in stream(seed, stage, adapt_steps, n):
        own = 2 * bits - 1
        msg = (um < sigmoid(sender[:, 0] + sender[:, 1] * own)).astype(int)
        slot = 2 * bits + msg[peer]
        p = .97 * sigmoid(policy[ids, slot]) + .015
        actions = (ua < p).astype(int)
        goal = target(bits, rule)
        good = (actions[::2] == goal) & (actions[1::2] == goal)
        pulse = np.repeat((good & (ur >= spec.dropout)).astype(float), 2)
        net = pulse - .06
        advantage = net - baseline
        policy[ids, slot] += spec.learning_rate * advantage * (actions - p)
        np.clip(policy, -12, 12, out=policy)
        baseline = .98 * baseline + .02 * net
        history.append(float(good.mean()))
    x = np.asarray(history)
    roll = np.convolve(x, np.ones(150) / 150, mode="valid")
    cross = np.flatnonzero(roll >= .9)
    return policy, {"first100": float(x[:100].mean()),
                    "first200": float(x[:200].mean()),
                    "last200": float(x[-200:].mean()),
                    "all": float(x.mean()),
                    "steps_to_90pct_150window": int(cross[0]+150) if len(cross) else None}


def trial(seed, novel_rule, spec, pre_steps, probe_steps, adapt_steps):
    pop = GrowingCells(spec, seed, "oracle")
    for stage, source, count in (("A", "xor", pre_steps),
                                 ("B", "xnor", pre_steps),
                                 ("C", "and", pre_steps)):
        for inp in rng_stream(seed, count, pop.n, stage):
            pop.episode(inp, source, train=True, detect=False, train_sender=(stage == "A"))
    sender = pop.sender.copy()
    banks = pop.weights.copy()
    prior_hash = hashlib.sha256(banks.tobytes()).hexdigest()
    best, worst, mosaic, probe_means = probe_sources(
        sender, banks, seed, novel_rule, probe_steps, spec.dropout)
    results = {}
    for strategy in STRATEGIES:
        initial = initialize(banks, best, worst, mosaic, strategy)
        if strategy == "scratch_equal_budget":
            # Stronger control: learn from an equal 3 * probe_steps local
            # interaction budget rather than discarding the probe experience.
            initial, _ = learn_new(sender, initial, seed, novel_rule, spec,
                                   3 * probe_steps, stage="probe")
        init_eval = evaluate(sender, initial, seed + 31, novel_rule, dropout=spec.dropout)
        final, metrics = learn_new(sender, initial, seed, novel_rule, spec, adapt_steps)
        final_eval = evaluate(sender, final, seed + 131, novel_rule, dropout=spec.dropout)
        results[strategy] = {**metrics, "pre_adapt_test_accuracy": init_eval["accuracy"],
                             "post_adapt_test_accuracy": final_eval["accuracy"]}
    # Frozen prior banks must be byte-identical, but that alone does not
    # establish autonomous context inference or general lifelong learning.
    assert prior_hash == hashlib.sha256(banks.tobytes()).hexdigest()
    return {"seed": seed, "novel_rule": novel_rule,
            "source_probe_reward_means": probe_means,
            "selected_bank_distribution": np.bincount(best[::2], minlength=3).tolist(),
            "prior_banks_unchanged": True, "conditions": results}


def benchmark(seeds, *, pairs=24, pre_steps=3800, probe_steps=120,
              adapt_steps=1100, dropout=.08):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("nonempty unique seeds required")
    if pairs < 4 or pre_steps < 200 or probe_steps < 20 or adapt_steps < 200:
        raise ValueError("invalid experiment size")
    spec = GrowthSpec(pairs=pairs, phase_steps=pre_steps, dropout=dropout)
    results = [trial(seed, rule, spec, pre_steps, probe_steps, adapt_steps)
               for seed in seeds for rule in RULES]
    summary = {}
    for rule in RULES:
        matching = [row for row in results if row["novel_rule"] == rule]
        summary[rule] = {}
        for strategy in STRATEGIES:
            summary[rule][strategy] = {
                key: {"mean": float(np.mean([r["conditions"][strategy][key] for r in matching])),
                      "sd": float(np.std([r["conditions"][strategy][key] for r in matching], ddof=1))
                            if len(matching) > 1 else 0.0}
                for key in ("pre_adapt_test_accuracy", "first100", "first200",
                            "last200", "post_adapt_test_accuracy")}
    return {"protocol": {"seeds": seeds, "pairs": pairs,
                         "pre_steps_each_source": pre_steps,
                         "probe_steps_per_prior_bank": probe_steps,
                         "novel_adapt_steps": adapt_steps,
                         "dropout": dropout,
                         "strategy_names": STRATEGIES,
                         "source_task_bank_ids": "Privileged only during pretraining",
                         "novel_task_ids_visible_to_learners": False,
                         "prior_memory": "Frozen by design; no autonomous task routing"},
            "summary": summary, "raw": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--pre-steps", type=int, default=3800)
    parser.add_argument("--probe-steps", type=int, default=120)
    parser.add_argument("--adapt-steps", type=int, default=1100)
    parser.add_argument("--dropout", type=float, default=.08)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    start = time.perf_counter()
    results = benchmark([int(s) for s in args.seeds.split(",")],
                        pairs=args.pairs, pre_steps=args.pre_steps,
                        probe_steps=args.probe_steps, adapt_steps=args.adapt_steps,
                        dropout=args.dropout)
    results["elapsed_seconds"] = time.perf_counter() - start
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"summary": results["summary"],
                      "elapsed_seconds": results["elapsed_seconds"],
                      "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
