"""R1 closed-loop learning with a local noisy goal signal.

Unlike R0's adjacent-only goal detection, this *separate* benchmark adds a
coarse noisy scalar proximity measurement. This is an informative sensor,
not a route, global map, target coordinate, or shortest-path oracle.
Compare beacon and beacon-masked controls to avoid hidden-task leakage.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
import random
import time
import numpy as np

from .sensor_world import SensorEpisode, Reading
from .latent_navigation import OnlineAgent, fit_prototypes
from .predictive_context import signature, traces
from .recurrent_state import RecurrentPredictor


@dataclass(frozen=True, slots=True)
class BeaconReading:
    walls: tuple
    nearby_target: tuple
    floor_signal: int
    previous_reward: float
    proximity: float


class BeaconEpisode(SensorEpisode):
    """Measurement derived by harness; only scalar noisy signal is exposed."""

    def __init__(self, *, changed=False, limit=160, beacon_enabled=True, seed=0):
        super().__init__(changed=changed, limit=limit)
        self.beacon_enabled = beacon_enabled
        self.noise = random.Random(seed * 1777 + 593)

    def _read(self):
        base = super()._read()
        if self.beacon_enabled:
            target = self._targets[min(self._index, len(self._targets)-1)]
            distance = (abs(target[0]-self._position[0]) +
                        abs(target[1]-self._position[1]))
            # Scalar only: truncated and noisy, not a bearing or position.
            strength = max(0., 1.0 - distance / 12.0)
            strength = max(0., min(1., strength + self.noise.gauss(0., .025)))
        else:
            strength = 0.
        return BeaconReading(base.walls, base.nearby_target,
                             base.floor_signal, base.previous_reward,
                             round(strength, 3))


class SignalAgent(OnlineAgent):
    """Local reward-progress bandit; learned action values, no global BFS."""

    def __init__(self, mode, seed, model, prototypes, *, training=False):
        base_mode = {
            "signal_latent": "recurrent_novelty",
            "signal_history": "history_novelty",
            "signal_memory_ablation": "recurrent_no_memory",
            "signal_observation": "obs_novelty",
            "no_signal_latent": "recurrent_novelty",
        }[mode]
        super().__init__(base_mode, seed, model=model, prototypes=prototypes)
        self.signal_mode = mode
        self.learned_reward = Counter()
        self.learned_counts = Counter()
        self.current_proximity = None
        self.last_proximity = None
        self.last_bandit_key = None
        self.learning = training
        self.episode_actions = 0

    def reset_episode(self, seed):
        self.rng = random.Random(seed)
        self.h[:] = 0
        self.history.clear()
        self.action_visits.clear()
        self.state_visits.clear()
        self.transitions.clear()
        self.last_state = self.last_action = self.last_signature = self.last_move = None
        self.last_proximity = None
        self.current_proximity = None
        self.last_bandit_key = None
        self.episode_actions = 0

    def act(self, reading):
        # Active goal is only directly visible when adjacent, as in R0.
        vicinity = [a for a in range(4) if reading.nearby_target[a]]
        if vicinity:
            action = vicinity[0]
        else:
            signal = reading.proximity if self.signal_mode != "no_signal_latent" else 0
            key = self.encode(reading)
            trend = 0 if self.last_proximity is None else (
                1 if signal > self.last_proximity + .03 else
                -1 if signal < self.last_proximity - .03 else 0)
            # A local learning context; no absolute position or goal coordinate.
            context = (key, trend, self.last_move)
            open_actions = [a for a in range(4) if not reading.walls[a]]
            candidates = open_actions if open_actions else list(range(4))
            values = []
            for action in candidates:
                visits = self.action_visits[(key, action)]
                attempts = self.learned_counts[(context, action)]
                progress = (self.learned_reward[(context, action)] /
                            max(1, attempts))
                known = self.transitions.get((key, action))
                recurring = self.state_visits[known] if known is not None else 0
                reverse = int(self.last_move is not None and action == (self.last_move ^ 1))
                # Learned progress estimate + local visit novelty.
                value = 12 * progress + 1.15 / np.sqrt(1 + attempts)
                value += -1.1 * visits - .15 * recurring - .35 * reverse
                value += self.rng.random() * .16
                values.append((value, action))
            action = max(values)[1]
            self.last_bandit_key = (context, action)
        self.current_proximity = reading.proximity if self.signal_mode != "no_signal_latent" else 0.
        self.last_proximity = self.current_proximity
        self.action_visits[(self.encode(reading), action)] += 1
        self.state_visits[self.encode(reading)] += 1
        self.last_state, self.last_action = self.encode(reading), action
        self.last_signature = signature(reading)
        self.last_move = action
        self.episode_actions += 1
        return action

    def observe(self, reading):
        if self.learning and self.last_bandit_key is not None:
            after = reading.proximity if self.signal_mode != "no_signal_latent" else 0.
            prior = self.current_proximity
            progress = max(-.2, min(.2, after-prior))
            if reading.previous_reward > .5:
                # The objective found signal is a sparse task reward.
                progress += .5
            context, action = self.last_bandit_key
            self.learned_reward[(context, action)] += progress
            self.learned_counts[(context, action)] += 1
        super().observe(reading)
        # A new target becomes active after reaching the previous one.
        if reading.previous_reward > .5:
            self.last_proximity = None
            self.action_visits.clear()
            self.state_visits.clear()
        self.last_bandit_key = None


def episode(mode, seed, model, prototypes, *, warm=None, changed=False,
            limit=160, goals=3, training=False):
    agent = warm if warm is not None else SignalAgent(mode, seed, model, prototypes,
                                                     training=training)
    agent.learning = training
    agent.reset_episode(seed * 127 + 11)
    world = BeaconEpisode(changed=changed, limit=limit,
                          beacon_enabled=mode != "no_signal_latent", seed=seed)
    reading = world.reset(seed, goals=goals)
    while True:
        action = agent.act(reading)
        reading, reward, done = world.step(action)
        agent.observe(reading)
        if done:
            break
    return world.metrics()


def benchmark(seeds, *, train_seed=71, train_steps=16000, epochs=20,
              hidden=32, centers=24, warm_episodes=250, limit=160,
              goals=3, train_modes=("signal_latent", "signal_history",
                                    "signal_observation", "no_signal_latent",
                                    "signal_memory_ablation")):
    train = traces(train_seed, steps=train_steps, episode_length=100)
    start = time.process_time()
    model = RecurrentPredictor(seed=train_seed, hidden=hidden).fit(
        train, epochs=epochs, seed=train_seed+17)
    prototypes = fit_prototypes(model, train, centers=centers, seed=train_seed)
    training_cpu = time.process_time() - start
    modes = tuple(train_modes) + ("signal_latent_cold",)
    learned = {}
    warm_cpu = {}
    for mode in train_modes:
        agent = SignalAgent(mode, train_seed, model, prototypes, training=True)
        warm_start = time.process_time()
        for index in range(warm_episodes):
            episode(mode, train_seed*100009 + index, model, prototypes,
                    warm=agent, changed=(index%4 == 0),
                    limit=limit, goals=goals, training=True)
        learned[mode] = agent
        warm_cpu[mode] = time.process_time() - warm_start
    rows = []
    for seed in seeds:
        for changed in (False, True):
            for mode in modes:
                if mode == "signal_latent_cold":
                    result = episode("signal_latent", seed, model, prototypes,
                                     changed=changed, limit=limit, goals=goals)
                else:
                    result = episode(mode, seed, model, prototypes,
                                     warm=learned[mode], changed=changed,
                                     limit=limit, goals=goals)
                rows.append({"seed": seed, "changed": changed, "mode": mode,
                             **result})
    summary = {}
    for changed in (False, True):
        label = "changed" if changed else "stable"
        summary[label] = {}
        for mode in modes:
            records = [r for r in rows if r["changed"] == changed and r["mode"] == mode]
            summary[label][mode] = {
                "finish_rate": sum(r["completed"] for r in records)/len(records),
                "mean_waypoints": sum(r["waypoints"] for r in records)/len(records),
                "mean_steps": sum(r["steps"] for r in records)/len(records)}
    return {"protocol": {
        "pretrained_state": "Self-supervised RNN from local observation/action history",
        "target_information": "Noisy scalar proximity sensor, or masked no-signal control",
        "global_state_inputs": False, "central_planner": False,
        "learning_algorithm": "Local per-context empirical reward-progress bandit",
        "train_seed": train_seed, "source_steps": train_steps,
        "recurrent_epochs": epochs, "hidden": hidden, "centers": centers,
        "warm_episodes": warm_episodes, "limit": limit, "goals": goals,
        "eval_seeds": seeds, "pretraining_cpu_seconds": training_cpu,
        "goal_policy_training_cpu_seconds": warm_cpu,
        "total_goal_training_episodes_per_mode": warm_episodes},
        "summary": summary, "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="11,12,13,14,15")
    p.add_argument("--train-seed", type=int, default=71)
    p.add_argument("--train-steps", type=int, default=16000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--centers", type=int, default=24)
    p.add_argument("--warm-episodes", type=int, default=250)
    p.add_argument("--limit", type=int, default=160)
    p.add_argument("--goals", type=int, default=3)
    p.add_argument("--output", default="")
    a = p.parse_args()
    now = time.perf_counter()
    data = benchmark([int(s) for s in a.seeds.split(",")],
                     train_seed=a.train_seed, train_steps=a.train_steps,
                     epochs=a.epochs, hidden=a.hidden, centers=a.centers,
                     warm_episodes=a.warm_episodes, limit=a.limit, goals=a.goals)
    data["elapsed_wall_seconds"] = time.perf_counter() - now
    if a.output:
        dest = Path(a.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({"summary": data["summary"],
                      "elapsed_wall_seconds": data["elapsed_wall_seconds"],
                      "output": a.output}, indent=2))


if __name__ == "__main__":
    main()
