"""A partially observable navigation benchmark.

The simulation uses coordinates internally; the policy is called only with
local sensor readings. This is infrastructure for research milestone R0.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

from .multistep_world import HELDOUT_GOALS, MOVES, Maze


@dataclass(frozen=True, slots=True)
class Reading:
    walls: tuple[bool, bool, bool, bool]
    nearby_target: tuple[bool, bool, bool, bool]
    floor_signal: int
    previous_reward: float


class SensorEpisode:
    def __init__(self, *, changed=False, limit=160):
        self._maze = Maze(changed=changed)
        self._limit = limit
        self._position = None
        self._targets = ()
        self._index = 0
        self._steps = 0
        self._reward = 0.0

    def _read(self):
        s = self._position
        walls = tuple(self._maze.step(s, a) == s for a in range(4))
        target = self._targets[min(self._index, len(self._targets)-1)]
        nearby = tuple(
            not walls[a] and
            (s[0] + delta[0], s[1] + delta[1]) == target
            for a, delta in enumerate(MOVES))
        return Reading(walls, nearby, (s[0] + s[1]) % 2, self._reward)

    def reset(self, seed, *, goals=3):
        if not 1 <= goals <= 5:
            raise ValueError("goals must be between 1 and 5")
        rng = random.Random(seed * 711 + 53)
        self._position = rng.choice(self._maze.states())
        choices = [g for g in HELDOUT_GOALS if g != self._position]
        self._targets = tuple(rng.sample(choices, goals))
        self._index = 0
        self._steps = 0
        self._reward = 0.0
        return self._read()

    def step(self, action):
        if not isinstance(action, int) or isinstance(action, bool) or action not in range(4):
            raise ValueError("action must be an integer in [0,3]")
        if self._position is None or self._index == len(self._targets) or self._steps >= self._limit:
            raise RuntimeError("start or restart an episode")
        self._position = self._maze.step(self._position, action)
        self._steps += 1
        self._reward = -0.01
        if self._position == self._targets[self._index]:
            self._index += 1
            self._reward += 1
        done = self._index == len(self._targets) or self._steps >= self._limit
        return self._read(), self._reward, done

    def metrics(self):
        """Only the external evaluation harness should call this method."""
        return {"completed": self._index == len(self._targets),
                "waypoints": self._index, "steps": self._steps}


def choose(reading, rng, counts, mode):
    open_actions = [a for a, wall in enumerate(reading.walls) if not wall]
    if mode == "reactive":
        for a, target in enumerate(reading.nearby_target):
            if target:
                return a
    if not open_actions:
        return rng.randrange(4)
    if mode == "random":
        return rng.choice(open_actions)
    key = (reading.walls, reading.floor_signal)
    local = counts.setdefault(key, [0, 0, 0, 0])
    low = min(local[a] for a in open_actions)
    action = rng.choice([a for a in open_actions if local[a] == low])
    local[action] += 1
    return action


def benchmark(seeds, goals=3, limit=160):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide distinct seeds")
    records = []
    for seed in seeds:
        for changed in (False, True):
            for mode in ("random", "reactive"):
                episode = SensorEpisode(changed=changed, limit=limit)
                reading = episode.reset(seed, goals=goals)
                rng = random.Random(seed + 101)
                counts = {}
                while True:
                    reading, reward, done = episode.step(
                        choose(reading, rng, counts, mode))
                    if done:
                        break
                records.append({"seed": seed, "changed": changed,
                                "mode": mode, **episode.metrics()})
    summary = {}
    for changed in (False, True):
        for mode in ("random", "reactive"):
            rows = [r for r in records
                    if r["changed"] == changed and r["mode"] == mode]
            summary[f"{'changed' if changed else 'stable'}_{mode}"] = {
                "complete_rate": sum(r["completed"] for r in rows) / len(rows),
                "average_goals_reached": sum(r["waypoints"] for r in rows) / len(rows)}
    return {"sensor_fields": list(Reading.__dataclass_fields__),
            "summary": summary, "records": records}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="11,12,13,14,15")
    p.add_argument("--goals", type=int, default=3)
    p.add_argument("--limit", type=int, default=160)
    p.add_argument("--output", default="")
    a = p.parse_args()
    result = benchmark([int(v) for v in a.seeds.split(",")], a.goals, a.limit)
    if a.output:
        dest = Path(a.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "output": a.output}, indent=2))


if __name__ == "__main__":
    main()
