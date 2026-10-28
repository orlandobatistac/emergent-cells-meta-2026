"""Zero-update transfer tests on previously untrained cellular task combinations.

Holdout task *families* are excluded from local curiosity meta-training.
The Bayesian representation, four-cell input slots, pairwise code and
counterfactual probe primitive remain ENGINEERED; this is not AGI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from .learned_curiosity import (
    ALL_SCENARIOS, HOLDOUT_SCENARIOS, SCENARIOS, CuriositySpec,
    LocalCuriosity, train_policy,
)


METHODS = (
    "learned_intrinsic", "learned_extrinsic",
    "learned_intrinsic_guard2", "learned_intrinsic_guard4",
    "manual_uncertain", "passive", "always_probe", "untrained_random_ties",
)


def evaluate(seed, scenario, spec, trained_intrinsic, trained_extrinsic, method):
    if scenario not in HOLDOUT_SCENARIOS or method not in METHODS:
        raise ValueError("test scenario and method must be preregistered")
    fresh = LocalCuriosity(spec, seed + 17)
    if method in ("learned_intrinsic", "learned_intrinsic_guard2",
                  "learned_intrinsic_guard4"):
        fresh.q = trained_intrinsic[0].q.copy()
    elif method == "learned_extrinsic":
        fresh.q = trained_extrinsic[0].q.copy()
    elif method == "manual_uncertain":
        # Authored qualitative uncertainty rule, NOT Bayesian optimal.
        fresh.q[:, :, 0] = .75
        fresh.q[:, :, 1] = .20
        fresh.q[:, [1, 2], 1] = .95
    elif method == "passive":
        fresh.q[:, :, 0] = .85
        fresh.q[:, :, 1] = .15
    elif method == "always_probe":
        fresh.q[:, :, 0] = .15
        fresh.q[:, :, 1] = .85
    # Random-tie condition keeps both zero-knowledge action values equal.
    sender, receiver = trained_intrinsic[1:3]
    rng = np.random.default_rng(
        seed + 79813 + ALL_SCENARIOS.index(scenario)*100003
    )
    result = fresh.run_episode(
        sender, receiver, [scenario] * spec.pairs, rng,
        steps=spec.eval_horizon, train=False, epsilon=0,
        diagnostic=True,
        stop_after_failed_probes=(
            2 if method == "learned_intrinsic_guard2"
            else 4 if method == "learned_intrinsic_guard4"
            else None
        )
    )
    return {
        **result,
        "seed": seed,
        "scenario": scenario,
        "method": method,
        "net_reward_per_pair_episode": float(
            result["earned"] - result["probe_cost"]/(spec.pairs*spec.eval_horizon)
        ),
        "request_bits_per_pair_episode": float(
            result["requests"]/(spec.pairs*spec.eval_horizon)
        ),
    }


def benchmark(seeds, spec=CuriositySpec()):
    spec.validate()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty distinct seeds")
    if any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("nonnegative int seeds required")
    assert not set(HOLDOUT_SCENARIOS).intersection(SCENARIOS)
    rows = []
    training = []
    for seed in seeds:
        train_intr = train_policy(seed, spec)
        no_bonus_spec = CuriositySpec(**{
            **asdict(spec), "intrinsic_weight": 0.
        })
        train_plain = train_policy(seed, no_bonus_spec)
        # Validate that the sender/receiver data were identically trained.
        # Only the curiosity policy's reward bonus differs across arms.
        if not (np.array_equal(train_intr[1], train_plain[1]) and
                np.array_equal(train_intr[2], train_plain[2])):
            raise AssertionError("unmatched communication pretraining")
        training.append({
            "seed": seed,
            "trained_curiosity_q_mean": train_intr[0].q.mean(axis=0).tolist(),
            "no_intrinsic_q_mean": train_plain[0].q.mean(axis=0).tolist()
        })
        for scenario in HOLDOUT_SCENARIOS:
            for method in METHODS:
                rows.append(
                    evaluate(seed, scenario, spec, train_intr, train_plain, method)
                )
    summary = {}
    fields = ("correct", "earned", "critical", "requests",
              "probe_cost", "false_corrections",
              "net_reward_per_pair_episode",
              "request_bits_per_pair_episode")
    for scenario in HOLDOUT_SCENARIOS:
        summary[scenario] = {}
        for method in METHODS:
            group = [r for r in rows if r["scenario"] == scenario and
                     r["method"] == method]
            summary[scenario][method] = {
                field: float(np.mean([r[field] for r in group]))
                for field in fields
            }
            summary[scenario][method]["case_correctness"] = {
                k: float(np.mean([r["by_case_correctness"][k] for r in group]))
                for k in ("00", "01", "10", "11")
            }
    return {
        "schema": "unseen-task-curiosity-transfer-v1",
        "date": "2026-10-27",
        "status": "exploratory holdout-task families; no on-test adaptation of Q values",
        "seeds": list(seeds),
        "spec": asdict(spec),
        "train_scenarios": list(SCENARIOS),
        "never_trained_scenarios": list(HOLDOUT_SCENARIOS),
        "methods": list(METHODS),
        "summary": summary,
        "rows": rows,
        "training_statistics": training,
        "limits": [
            "Test scenario IDs drive the WORLD only, not the cells.",
            "The local Bayesian belief states were engineered for the task family.",
            "Both trained controllers use the SAME trained sender/receiver policy.",
            "No evaluation-time update to the learned probe Q policy is permitted.",
            "A fixed four-slot parity decoder and XOR-like task family remain.",
            "The synthetic cost does not price memory, CPU or physical power.",
            "Critical score is NOT meaningful in globally reversed/drought worlds.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="11,12")
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--pretrain", type=int, default=2400)
    p.add_argument("--meta-episodes", type=int, default=140)
    p.add_argument("--meta-horizon", type=int, default=55)
    p.add_argument("--eval-horizon", type=int, default=300)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    spec = CuriositySpec(
        pairs=a.pairs, pretrain=a.pretrain,
        meta_episodes=a.meta_episodes,
        meta_horizon=a.meta_horizon, eval_horizon=a.eval_horizon
    )
    data = benchmark([int(z) for z in a.seeds.split(",") if z.strip()], spec)
    body = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if a.output:
        a.output.parent.mkdir(exist_ok=True, parents=True)
        a.output.write_text(body, encoding="utf-8")
        print("SAVED", a.output)
    else:
        print(body)


if __name__ == "__main__":
    main()
