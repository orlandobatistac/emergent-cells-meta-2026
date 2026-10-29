"""R1 trainable recurrent latent state on self-supervised sensory trajectories.

A small NumPy RNN with full-episode backpropagation through time and Adam.
No coordinates, map IDs, goal coordinates, or correct route labels enter
the learner. The simulator retains privileged state for episode generation.
"""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import random
import time
import tracemalloc
import numpy as np

from .multistep_world import Maze, MOVES, HELDOUT_GOALS
from .predictive_context import (
    ALPHABET, HISTORY, PredictiveContext, assess, signature, traces)
from .sensor_world import SensorEpisode

FEATURES = ALPHABET + 4


class GeneratedMaze:
    """Evaluation-only modified maze; never passed to a learning model."""

    def __init__(self, seed):
        rng = random.Random(seed)
        original = Maze()
        blocked = set(original.blocked())
        protected = set(HELDOUT_GOALS) | {(1, 1), (7, 7), (4, 2), (4, 6),
                                          (3, 2), (5, 2), (3, 6), (5, 6)}
        removable = sorted(blocked - protected)
        if removable:
            blocked.remove(rng.choice(removable))
        candidates = sorted(set(original.states()) - protected)
        rng.shuffle(candidates)
        for p in candidates[:3]:
            blocked.add(p)
        self.walls = blocked

    def free(self, point):
        x, y = point
        return 0 <= x < 9 and 0 <= y < 9 and point not in self.walls

    def states(self):
        return [(x, y) for y in range(9) for x in range(9)
                if self.free((x, y))]

    def step(self, state, action):
        dx, dy = MOVES[action]
        nxt = (state[0] + dx, state[1] + dy)
        return nxt if self.free(nxt) else state


def generated_traces(seed, *, steps, episode_length=100):
    """Held-out procedural layouts, generated only within the harness."""
    rng = random.Random(seed * 10547 + 103)
    episodes = []
    for index in range((steps + episode_length - 1) // episode_length):
        world = SensorEpisode(limit=episode_length + 1)
        world._maze = GeneratedMaze(seed * 8191 + index)
        current = world.reset(seed * 100003 + index, goals=3)
        episode = []
        for _ in range(min(episode_length, steps - index * episode_length)):
            state = signature(current)
            action = rng.randrange(4)
            current, _reward, done = world.step(action)
            episode.append((state, action, signature(current)))
            if done:
                break
        episodes.append(episode)
    return episodes


def packed(episodes, length=None):
    if not episodes:
        raise ValueError("at least one episode needed")
    n = len(episodes)
    t = length or max(len(e) for e in episodes)
    x = np.zeros((n, t, FEATURES), dtype=np.float32)
    target = np.zeros((n, t), dtype=np.int64)
    mask = np.zeros((n, t), dtype=np.float32)
    for i, e in enumerate(episodes):
        for j, (obs, action, next_obs) in enumerate(e[:t]):
            x[i, j, obs] = 1
            x[i, j, ALPHABET + action] = 1
            target[i, j] = next_obs
            mask[i, j] = 1
    return x, target, mask


class RecurrentPredictor:
    """Trainable 12-32-dimensional hidden state, learned through BPTT."""

    def __init__(self, *, seed=1, hidden=24):
        if not 4 <= hidden <= 128:
            raise ValueError("hidden units must be 4..128")
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        self.p = {
            "wx": (rng.normal(size=(FEATURES, hidden)) / np.sqrt(FEATURES)).astype(np.float32),
            "wh": (rng.normal(size=(hidden, hidden)) * .35 / np.sqrt(hidden)).astype(np.float32),
            "bh": np.zeros(hidden, dtype=np.float32),
            "wy": (rng.normal(size=(hidden, ALPHABET)) / np.sqrt(hidden)).astype(np.float32),
            "by": np.zeros(ALPHABET, dtype=np.float32),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._step = 0
        self.training_history = []

    def _forward(self, x, mask):
        wx, wh, bh = (self.p[k] for k in ("wx", "wh", "bh"))
        n, steps, _ = x.shape
        h = np.zeros((n, self.hidden), dtype=np.float32)
        states = [h]
        proposed = []
        logits = []
        for t in range(steps):
            candidate = np.tanh(x[:, t] @ wx + h @ wh + bh)
            visible = mask[:, t:t+1]
            h = visible * candidate + (1-visible) * h
            states.append(h)
            proposed.append(candidate)
            logits.append(h @ self.p["wy"] + self.p["by"])
        return states, proposed, np.stack(logits, axis=1)

    def _gradients(self, x, target, mask):
        states, proposed, logits = self._forward(x, mask)
        prob = np.exp(logits - logits.max(axis=2, keepdims=True))
        prob /= prob.sum(axis=2, keepdims=True)
        n, steps = mask.shape
        ri, ti = np.indices((n, steps))
        valid = max(float(mask.sum()), 1.)
        loss = -np.log(np.maximum(prob[ri, ti, target], 1e-8))
        loss = float((loss * mask).sum() / valid)
        dout = prob
        dout[ri, ti, target] -= 1
        dout *= (mask / valid)[:, :, None]
        grad = {k: np.zeros_like(v) for k, v in self.p.items()}
        upstream = np.zeros((n, self.hidden), dtype=np.float32)
        for t in range(steps - 1, -1, -1):
            grad["wy"] += states[t+1].T @ dout[:, t]
            grad["by"] += dout[:, t].sum(axis=0)
            dh = upstream + dout[:, t] @ self.p["wy"].T
            active = mask[:, t:t+1]
            delta = (dh * active) * (1. - proposed[t] * proposed[t])
            grad["wx"] += x[:, t].T @ delta
            grad["wh"] += states[t].T @ delta
            grad["bh"] += delta.sum(axis=0)
            upstream = (delta @ self.p["wh"].T) + dh * (1-active)
        total_norm = np.sqrt(sum(float(np.sum(g * g)) for g in grad.values()))
        if total_norm > 2:
            for k in grad:
                grad[k] *= 2.0 / total_norm
        return loss, grad

    def _adam(self, gradient, lr):
        self._step += 1
        for k in self.p:
            self._m[k] = .9*self._m[k] + .1*gradient[k]
            self._v[k] = .999*self._v[k] + .001*(gradient[k]*gradient[k])
            m = self._m[k] / (1 - .9**self._step)
            v = self._v[k] / (1 - .999**self._step)
            self.p[k] -= lr * m / (np.sqrt(v) + 1.e-7)

    def fit(self, episodes, *, epochs=5, batch_size=16, lr=.007, seed=1):
        if epochs < 1 or batch_size < 1 or lr <= 0:
            raise ValueError("invalid optimization schedule")
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            ordering = rng.permutation(len(episodes))
            total_loss, total_steps = 0., 0
            for start in range(0, len(episodes), batch_size):
                subset = [episodes[int(i)] for i in ordering[start:start+batch_size]]
                x, target, mask = packed(subset)
                loss, grad = self._gradients(x, target, mask)
                self._adam(grad, lr)
                count = int(mask.sum())
                total_loss += loss * count
                total_steps += count
            self.training_history.append(total_loss / total_steps)
        return self

    def probabilities(self, episode, *, clear_history=False):
        x, _, mask = packed([episode])
        if clear_history:
            # Causal ablation: suppress all remembered hidden state while
            # keeping the identical trained weights and test observations.
            h = np.tanh(x[0] @ self.p["wx"] + self.p["bh"])
            logits = h @ self.p["wy"] + self.p["by"]
        else:
            _states, _candidates, outputs = self._forward(x, mask)
            logits = outputs[0]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return p

    def parameter_bytes(self):
        return sum(p.nbytes for p in self.p.values())


def evaluate(model, episodes, coarse, *, clear_history=False):
    correct, n, nll = 0, 0, 0.
    ambiguous, ambiguous_correct = 0, 0
    for ep in episodes:
        if not ep:
            continue
        probs = model.probabilities(ep, clear_history=clear_history)
        for t, (obs, action, next_obs) in enumerate(ep):
            pred = int(np.argmax(probs[t]))
            correct += int(pred == next_obs)
            nll -= float(np.log(max(float(probs[t, next_obs]), 1e-10)))
            known = coarse.counts[0].get(((), obs, action))
            if known and len(known) > 1:
                ambiguous += 1
                ambiguous_correct += int(pred == next_obs)
            n += 1
    return {"samples": n,
            "accuracy": correct / n,
            "nll_nats": nll / n,
            "ambiguous_accuracy": ambiguous_correct / ambiguous if ambiguous else None,
            "ambiguous_samples": ambiguous}


def trial(seed, *, train_steps=16000, test_steps=4000, epochs=5, hidden=24,
          episode_length=100):
    train = traces(seed, steps=train_steps, episode_length=episode_length)
    stable = traces(seed+600001, steps=test_steps, episode_length=episode_length)
    changed = traces(seed+700001, steps=test_steps, episode_length=episode_length, changed=True)
    novel = generated_traces(seed+800001, steps=test_steps, episode_length=episode_length)
    tabular = PredictiveContext(HISTORY).fit(train)
    start = time.process_time()
    tracemalloc.start()
    model = RecurrentPredictor(seed=seed, hidden=hidden).fit(
        train, epochs=epochs, seed=seed+17)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    cpu_time = time.process_time() - start
    scenarios = {"stable": stable, "changed": changed, "novel": novel}
    metrics = {}
    for name, episodes in scenarios.items():
        metrics[name] = {
            "observation_only": assess(tabular, episodes, "observation_only"),
            "history_table": assess(tabular, episodes, "adaptive_predictive"),
            "trainable_recurrent": evaluate(model, episodes, tabular),
            "recurrent_memory_ablation": evaluate(
                model, episodes, tabular, clear_history=True),
        }
    return {
        "seed": seed, "training_loss_per_epoch": model.training_history,
        "train_cpu_seconds": cpu_time,
        "peak_traced_allocation_bytes": peak,
        "recurrent_parameter_bytes": model.parameter_bytes(),
        "tabular_context_count": sum(tabular.size()),
        "results": metrics
    }


def benchmark(seeds, *, train_steps=16000, test_steps=4000, epochs=5,
              hidden=24, episode_length=100):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("unique seeds required")
    rows = [trial(seed, train_steps=train_steps, test_steps=test_steps,
                  epochs=epochs, hidden=hidden, episode_length=episode_length)
            for seed in seeds]
    summary = {}
    for scenario in ("stable", "changed", "novel"):
        summary[scenario] = {}
        for method in ("observation_only", "history_table",
                       "trainable_recurrent", "recurrent_memory_ablation"):
            fields = [r["results"][scenario][method] for r in rows]
            summary[scenario][method] = {
                key: float(np.mean([a[key] for a in fields]))
                for key in ("accuracy", "nll_nats", "ambiguous_accuracy")}
    return {"protocol": {"seeds": seeds, "train_steps": train_steps,
                         "test_steps": test_steps,
                         "epochs": epochs, "hidden_units": hidden,
                         "episode_length": episode_length,
                         "novel_layouts": "per-episode procedural, held-out",
                         "world_state_labels_in_training": False,
                         "sensor_channels": "current wall bits, texture, action",
                         "comparison": "identical trajectories, outcome and inputs"},
            "summary": summary, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="11,12")
    p.add_argument("--train-steps", type=int, default=16000)
    p.add_argument("--test-steps", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--output", default="")
    a = p.parse_args()
    start = time.perf_counter()
    result = benchmark([int(s) for s in a.seeds.split(",")],
                       train_steps=a.train_steps, test_steps=a.test_steps,
                       epochs=a.epochs, hidden=a.hidden,
                       episode_length=a.episode_length)
    result["elapsed_wall_seconds"] = time.perf_counter() - start
    if a.output:
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"],
                      "cpu_seconds_mean": sum(x["train_cpu_seconds"]
                          for x in result["rows"]) / len(result["rows"]),
                      "elapsed_wall_seconds": result["elapsed_wall_seconds"],
                      "output": a.output}, indent=2))


if __name__ == "__main__":
    main()
