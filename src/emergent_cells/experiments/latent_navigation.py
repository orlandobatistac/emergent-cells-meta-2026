"""R1 closed-loop exploration from sensor-only recurrent states.

The agent never receives world position, goal coordinates, a wall map or BFS.
Goal detection is possible only from an adjacent local sensor. Its policy is
an engineered curiosity/exploration heuristic over self-supervised learned
state embeddings, NOT a trained goal-conditioned RL policy.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import random
import time
import numpy as np

from .predictive_context import signature, traces, ALPHABET
from .recurrent_state import RecurrentPredictor
from .sensor_world import SensorEpisode, choose

MODES = ("random", "reactive", "obs_novelty", "history_novelty",
         "recurrent_novelty", "recurrent_no_memory")


def fit_prototypes(model, episodes, centers=24, max_points=8000, seed=0):
    """Learn unsupervised latent centroids on TRAINING sensor histories only."""
    vectors = []
    for episode in episodes:
        if not episode:
            continue
        x = np.zeros((1, len(episode), ALPHABET + 4), dtype=np.float32)
        for t, (obs, action, _next) in enumerate(episode):
            x[0, t, obs] = 1.
            x[0, t, ALPHABET + action] = 1.
        masks = np.ones((1, len(episode)), dtype=np.float32)
        states, _, _ = model._forward(x, masks)
        vectors.extend(s[0].copy() for s in states[1:])
    data = np.asarray(vectors, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(data) > max_points:
        data = data[rng.choice(len(data), max_points, replace=False)]
    prototypes = data[rng.choice(len(data), centers, replace=False)].copy()
    for _ in range(12):
        assignment = ((data[:, None] - prototypes[None, :])**2).sum(axis=2).argmin(axis=1)
        for k in range(centers):
            chosen = data[assignment == k]
            if len(chosen):
                prototypes[k] = chosen.mean(axis=0)
    return prototypes


class OnlineAgent:
    def __init__(self, mode, seed, model=None, prototypes=None):
        self.mode = mode
        self.rng = random.Random(seed)
        self.model = model
        self.prototypes = prototypes
        self.h = np.zeros(model.hidden, dtype=np.float32) if model is not None else None
        self.history = deque(maxlen=3)
        self.action_visits = Counter()
        self.state_visits = Counter()
        self.transitions = {}
        self.reactive_counts = {}
        self.last_state = None
        self.last_action = None
        self.last_signature = None
        self.last_move = None
        self.step_index = 0

    def encode(self, reading):
        sig = signature(reading)
        if self.mode == "obs_novelty":
            return (sig,)
        if self.mode == "history_novelty":
            return (sig, tuple(self.history))
        if self.mode in ("recurrent_novelty", "recurrent_no_memory"):
            differences = ((self.prototypes - self.h[None, :])**2).sum(axis=1)
            return (sig, int(np.argmin(differences)))
        return (sig,)

    def act(self, reading):
        if self.mode in ("random", "reactive"):
            a = choose(reading, self.rng, self.reactive_counts, self.mode)
            return a
        key = self.encode(reading)
        self.state_visits[key] += 1
        possible = [a for a in range(4) if not reading.walls[a]]
        if not possible:
            possible = list(range(4))
        adjacent = [a for a in possible if reading.nearby_target[a]]
        if adjacent:
            action = adjacent[0]
        else:
            scored = []
            for action in possible:
                visits = self.action_visits[(key, action)]
                known_next = self.transitions.get((key, action))
                future_visits = (self.state_visits[known_next]
                                 if known_next is not None else 0)
                reversal = int(self.last_move is not None and action == (self.last_move ^ 1))
                # Uncertainty reduction is an engineered intrinsic-motivation
                # heuristic, identical for all state encodings.
                utility = -2.5 * visits - 0.4 * future_visits - 0.45 * reversal
                utility += self.rng.random() * .24
                scored.append((utility, action))
            action = max(scored)[1]
        self.action_visits[(key, action)] += 1
        self.last_state, self.last_action = key, action
        self.last_signature = signature(reading)
        self.last_move = action
        self.step_index += 1
        return action

    def observe(self, reading):
        if self.last_action is None or self.mode in ("random", "reactive"):
            return
        if self.model is not None and self.mode in ("recurrent_novelty", "recurrent_no_memory"):
            x = np.zeros(ALPHABET + 4, dtype=np.float32)
            x[self.last_signature] = 1.
            x[ALPHABET + self.last_action] = 1.
            base = (np.zeros_like(self.h) if self.mode == "recurrent_no_memory"
                    else self.h)
            self.h = np.tanh(x @ self.model.p["wx"] + base @ self.model.p["wh"] +
                             self.model.p["bh"])
        self.history.append((self.last_signature, self.last_action))
        new_key = self.encode(reading)
        self.transitions[(self.last_state, self.last_action)] = new_key


def run_episode(mode, seed, model, prototypes, changed=False, limit=160, goals=3):
    world = SensorEpisode(changed=changed, limit=limit)
    reading = world.reset(seed, goals=goals)
    agent = OnlineAgent(mode, seed + 19, model=model, prototypes=prototypes)
    while True:
        action = agent.act(reading)
        reading, reward, done = world.step(action)
        agent.observe(reading)
        if done:
            break
    return {**world.metrics(),
            "unique_internal_states": len(agent.state_visits),
            "internal_action_entries": len(agent.action_visits)}


def benchmark(seeds, *, train_seed=19, train_steps=16000, epochs=20,
              hidden=32, centers=24, limit=160, goals=3):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty unique evaluation seeds required")
    train = traces(train_seed, steps=train_steps, episode_length=100)
    start = time.process_time()
    model = RecurrentPredictor(seed=train_seed, hidden=hidden).fit(
        train, epochs=epochs, seed=train_seed+17)
    centers_array = fit_prototypes(model, train, centers=centers, seed=train_seed)
    training_cpu = time.process_time() - start
    rows = []
    for seed in seeds:
        for changed in (False, True):
            for mode in MODES:
                result = run_episode(mode, seed, model, centers_array,
                                     changed=changed, limit=limit, goals=goals)
                rows.append({"seed": seed, "changed": changed,
                             "mode": mode, **result})
    summary = {}
    for changed in (False, True):
        label = "changed" if changed else "stable"
        summary[label] = {}
        for mode in MODES:
            subset = [r for r in rows if r["changed"] == changed and r["mode"] == mode]
            summary[label][mode] = {
                "complete_rate": sum(r["completed"] for r in subset)/len(subset),
                "mean_waypoints": sum(r["waypoints"] for r in subset)/len(subset),
                "mean_steps": sum(r["steps"] for r in subset)/len(subset),
                "mean_internal_states": sum(r["unique_internal_states"] for r in subset)/len(subset)}
    return {"protocol": {
        "policy": "Engineered intrinsic novelty over learned recurrent or history states",
        "oracle_position_or_map_inputs": False,
        "goal_signal": "Only adjacent goal detector; no distant bearing",
        "pretraining_task": "Self-supervised next sensory signature",
        "train_seed": train_seed, "train_steps": train_steps,
        "epochs": epochs, "hidden": hidden, "centers": centers,
        "limit": limit, "goals": goals, "eval_seeds": seeds,
        "train_cpu_seconds": training_cpu,
        "parameter_bytes": model.parameter_bytes(),
        "centroid_bytes": centers_array.nbytes},
        "summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="11,12,13,14,15")
    ap.add_argument("--train-seed", type=int, default=19)
    ap.add_argument("--train-steps", type=int, default=16000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--centers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--goals", type=int, default=3)
    ap.add_argument("--output", default="")
    a = ap.parse_args()
    start = time.perf_counter()
    report = benchmark([int(s) for s in a.seeds.split(",")],
                       train_seed=a.train_seed, train_steps=a.train_steps,
                       epochs=a.epochs, hidden=a.hidden, centers=a.centers,
                       limit=a.limit, goals=a.goals)
    report["elapsed_wall_seconds"] = time.perf_counter() - start
    if a.output:
        dest = Path(a.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "protocol": report["protocol"],
                      "elapsed_wall_seconds": report["elapsed_wall_seconds"],
                      "output": a.output}, indent=2))


if __name__ == "__main__":
    main()
