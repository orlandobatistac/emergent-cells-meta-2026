"""R1 exploratory predictive state from action-observation history.

CPU-only self-supervised conditional prediction. Sensor state and actions are
the ONLY learner inputs; targets are observed next sensory signatures.
Training never uses coordinates, maze labels, shortest paths, or oracle states.

This is a small variable-order categorical context model, not a neural latent
world model or learned autonomous navigation. History keys are engineered;
their predictive probabilities are learned from observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
import random
import time
import tracemalloc

from .sensor_world import SensorEpisode

ALPHABET = 32
HISTORY = 3
MODES = ("observation_only", "one_step_history", "adaptive_predictive")


def signature(reading):
    value = sum((int(blocked) << a) for a, blocked in enumerate(reading.walls))
    return value + (16 if reading.floor_signal else 0)


def traces(seed, *, steps, episode_length, changed=False):
    """A uniform local-action exploration policy, shared by all models.

    Only the experiment harness instantiates a world. A generated example
    contains observed symbols and actions, not world or goal coordinates.
    """
    rng = random.Random(seed * 10547 + 103)
    samples = []
    episode_count = (steps + episode_length - 1) // episode_length
    for episode_id in range(episode_count):
        world = SensorEpisode(changed=changed, limit=episode_length + 1)
        reading = world.reset(seed * 100003 + episode_id, goals=3)
        sequence = []
        count = min(episode_length, steps - episode_id * episode_length)
        for _ in range(count):
            s = signature(reading)
            action = rng.randrange(4)
            reading, reward, done = world.step(action)
            ns = signature(reading)
            sequence.append((s, action, ns))
            if done:
                # Episode boundary resets history in the next generated run.
                break
        samples.append(sequence)
    return samples


def key(history, observed, action, depth):
    # Record past input/action pairs, with the current observed input.
    return (tuple(history)[-depth:] if depth else (), observed, action)


class PredictiveContext:
    def __init__(self, depth, alpha=5.0, min_visits=5):
        self.depth = depth
        self.alpha = alpha
        self.min_visits = min_visits
        self.counts = [defaultdict(Counter) for _ in range(depth + 1)]
        self.total_updates = 0

    def fit(self, episodes):
        for episode in episodes:
            hist = deque(maxlen=self.depth)
            for s, action, outcome in episode:
                for depth in range(self.depth + 1):
                    if depth <= len(hist):
                        self.counts[depth][key(hist, s, action, depth)][outcome] += 1
                hist.append((s, action))
                self.total_updates += 1
        # Only supported longer contexts participate in inference. Physically
        # discard sparse keys to avoid accumulating unused memory.
        for depth in range(1, self.depth + 1):
            self.counts[depth] = {
                context: counts for context, counts in self.counts[depth].items()
                if sum(counts.values()) >= self.min_visits
            }
        return self

    def probs(self, history, observed, action, mode):
        max_depth = {
            "observation_only": 0,
            "one_step_history": 1,
            "adaptive_predictive": self.depth,
        }[mode]
        prior = [1.0 / ALPHABET] * ALPHABET
        for depth in range(max_depth + 1):
            if len(history) < depth:
                continue
            counts = self.counts[depth].get(key(history, observed, action, depth))
            if not counts:
                continue
            total = sum(counts.values())
            if depth and total < self.min_visits:
                continue
            # Shrink sparse history-specific estimates toward a lower-order
            # learner, avoiding brittle exact-context lookup.
            denominator = total + self.alpha
            prior = [(counts.get(symbol, 0) + self.alpha * prior[symbol]) /
                     denominator for symbol in range(ALPHABET)]
        return prior

    def size(self):
        return [len(d) for d in self.counts]


def assess(model, episodes, mode):
    logloss = 0.0
    correct = 0
    n = 0
    ambiguity = 0
    ambiguous_correct = 0
    # Ambiguous means a current sensory class/action had at least two known
    # possible outcomes in the TRAINING set; this label uses no true positions.
    for episode in episodes:
        hist = deque(maxlen=HISTORY)
        for s, action, nxt in episode:
            dist = model.probs(hist, s, action, mode)
            answer = max(range(ALPHABET), key=lambda c: dist[c])
            correct += int(answer == nxt)
            logloss -= math.log(max(1e-12, dist[nxt]))
            coarse = model.counts[0].get(((), s, action))
            if coarse and len(coarse) > 1:
                ambiguity += 1
                ambiguous_correct += int(answer == nxt)
            hist.append((s, action))
            n += 1
    return {
        "samples": n,
        "accuracy": correct / n if n else 0,
        "nll_nats": logloss / n if n else 0,
        "perplexity": math.exp(logloss / n) if n else None,
        "ambiguous_samples": ambiguity,
        "ambiguous_accuracy": ambiguous_correct / ambiguity if ambiguity else None,
    }


def run_seed(seed, *, train_steps=32000, test_steps=8000, episode_length=100,
             alpha=5.0, min_visits=5):
    train = traces(seed, steps=train_steps, episode_length=episode_length)
    test = traces(seed + 600001, steps=test_steps,
                  episode_length=episode_length)
    changed = traces(seed + 700001, steps=test_steps,
                     episode_length=episode_length, changed=True)
    start = time.process_time()
    tracemalloc.start()
    model = PredictiveContext(HISTORY, alpha=alpha,
                              min_visits=min_visits).fit(train)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    train_cpu = time.process_time() - start
    stable = {mode: assess(model, test, mode) for mode in MODES}
    shifted = {mode: assess(model, changed, mode) for mode in MODES}
    return {
        "seed": seed, "train_steps": model.total_updates,
        "state_context_counts": model.size(),
        "train_cpu_seconds": train_cpu,
        "peak_traced_allocation_bytes": peak,
        "stable_test": stable, "changed_test": shifted
    }


def benchmark(seeds, *, train_steps=32000, test_steps=8000,
              episode_length=100, alpha=5.0, min_visits=5):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("nonempty unique seeds required")
    if train_steps < 200 or test_steps < 50 or episode_length < 10:
        raise ValueError("invalid benchmark settings")
    runs = [run_seed(seed, train_steps=train_steps, test_steps=test_steps,
                     episode_length=episode_length, alpha=alpha,
                     min_visits=min_visits) for seed in seeds]
    summary = {}
    for scenario in ("stable_test", "changed_test"):
        summary[scenario] = {}
        for mode in MODES:
            rows = [r[scenario][mode] for r in runs]
            summary[scenario][mode] = {
                field: sum(r[field] for r in rows) / len(rows)
                for field in ("accuracy", "nll_nats", "ambiguous_accuracy")
            }
    return {
        "protocol": {
            "inputs": "local walls, floor texture, past actions; no coordinates",
            "target": "next local walls and floor texture (self-supervised)",
            "history_pairs": HISTORY,
            "model_type": "engineered context keys; learned categorical predictive distributions",
            "training_goal_labels": False,
            "train_steps": train_steps,
            "test_steps_per_scenario": test_steps,
            "episode_length": episode_length,
            "smoothing_alpha": alpha,
            "min_history_visits": min_visits,
            "seeds": seeds
        }, "summary": summary, "runs": runs
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--train-steps", type=int, default=32000)
    parser.add_argument("--test-steps", type=int, default=8000)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--min-visits", type=int, default=5)
    parser.add_argument("--output", default="")
    a = parser.parse_args()
    if not 0 < a.alpha < 1000 or not 1 <= a.min_visits <= 1000:
        parser.error("invalid alpha or min-visits")
    start = time.perf_counter()
    data = benchmark([int(seed) for seed in a.seeds.split(",")],
                     train_steps=a.train_steps, test_steps=a.test_steps,
                     episode_length=a.episode_length,
                     alpha=a.alpha, min_visits=a.min_visits)
    data["elapsed_wall_seconds"] = time.perf_counter() - start
    if a.output:
        target = Path(a.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": data["summary"],
        "average_contexts": [sum(r["state_context_counts"][i]
                              for r in data["runs"]) / len(data["runs"])
                             for i in range(HISTORY + 1)],
        "elapsed_wall_seconds": data["elapsed_wall_seconds"],
        "output": a.output
    }, indent=2))


if __name__ == "__main__":
    main()
