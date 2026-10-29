"""Observed local graph with programmed relative odometry and value diffusion.

The agent gets only local walls, adjacent-goal cues and its own reward.
It builds cell-local transition memory from its actions, but its motion model,
value-propagation rule and central simulation scheduler are PROGRAMMED.
This is a strong non-oracle engineering baseline, NOT an emergent neural planner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

from .sensor_world import SensorEpisode
from .latent_navigation import run_episode, OnlineAgent
from .recurrent_state import GeneratedMaze

MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))
REVERSE = (1, 0, 3, 2)


class LocalValueCells:
    """Local visit memories exchange neighbor values over discovered edges."""

    def __init__(self, seed, *, mode="diffusion", rounds=10, decay=.92,
                 frontier_reward=2.5, freeze_memory=False):
        if mode not in ("diffusion", "greedy"):
            raise ValueError("unknown mode")
        if rounds < 0 or not 0 < decay < 1 or frontier_reward <= 0:
            raise ValueError("invalid propagation parameters")
        self.random = random.Random(seed + 19)
        self.mode = mode
        self.rounds = rounds
        self.decay = decay
        self.frontier_reward = frontier_reward
        self.freeze_memory = freeze_memory
        self.position = (0, 0)  # Relative origin; never simulator coordinates.
        self.open_actions = {}
        self.total_visits = Counter()
        self.goal_phase_visits = Counter()
        self.local_edge_visits = Counter()
        self.last_action = None
        self.last_position = None
        self.local_message_updates = 0
        self.completed_objectives = 0
        self.action_count = 0

    @staticmethod
    def successor(point, action):
        dx, dy = MOVES[action]
        return point[0] + dx, point[1] + dy

    def _diffuse(self):
        """Programmed synchronous local value updates, no BFS or oracle map.

        A Python loop schedules updates across all discovered cells; execution
        is centralized even though each value accesses only neighboring cells.
        """
        value = {p: 0. for p in self.open_actions}
        for _ in range(self.rounds):
            updated = {}
            for point, actions in self.open_actions.items():
                visits = self.goal_phase_visits[point]
                reward = 0.9 / math.sqrt(1 + visits)
                if any(self.successor(point, a) not in self.open_actions
                       for a in actions):
                    reward += self.frontier_reward
                neighbors = [value[self.successor(point, a)] for a in actions
                             if self.successor(point, a) in value]
                updated[point] = max(
                    reward, self.decay * max(neighbors, default=0.))
                self.local_message_updates += len(neighbors)
            value = updated
        return value

    def act(self, reading):
        """Only the public reading fields are used for choosing an action."""
        legal = tuple(a for a, wall in enumerate(reading.walls) if not wall)
        self.open_actions[self.position] = legal
        self.total_visits[self.position] += 1
        self.goal_phase_visits[self.position] += 1
        adjacent = [a for a in legal if reading.nearby_target[a]]
        if adjacent:
            action = adjacent[0]
        else:
            potential = self._diffuse() if self.mode == "diffusion" else {}
            scores = []
            for candidate in legal:
                nxt = self.successor(self.position, candidate)
                backward = int(
                    self.last_action is not None and
                    candidate == REVERSE[self.last_action])
                if self.mode == "greedy":
                    utility = 6. if nxt not in self.open_actions else 0.
                    utility -= self.goal_phase_visits[nxt]
                    utility -= 0.35 * self.local_edge_visits[self.position, candidate]
                    utility -= 0.10 * backward
                    utility += self.random.random() * .1
                elif nxt not in self.open_actions:
                    utility = self.frontier_reward + 2.
                    utility += self.random.random() * .02
                else:
                    utility = potential[nxt]
                    utility -= 0.04 * self.goal_phase_visits[nxt]
                    utility -= 0.07 * self.local_edge_visits[self.position, candidate]
                    utility -= 0.12 * backward
                    utility += self.random.random() * .02
                scores.append((utility, candidate))
            action = max(scores)[1] if scores else self.random.randrange(4)
        self.last_position = self.position
        self.last_action = action
        self.local_edge_visits[self.position, action] += 1
        self.action_count += 1
        return action

    def observe(self, reading):
        """Only self-action history and observed reward update the state.

        Deterministic action displacement is an ENGINEERED prior, not a learned
        motor model. The previous wall reading prevented choosing blocked edges.
        """
        if self.last_position is None:
            raise RuntimeError("act before observe")
        if not self.freeze_memory:
            if self.last_action in self.open_actions[self.last_position]:
                self.position = self.successor(
                    self.last_position, self.last_action)
        if reading.previous_reward > .5:
            self.completed_objectives += 1
            self.goal_phase_visits.clear()
            self.local_edge_visits.clear()


def run_benchmark(seeds, *, rounds=10, decay=.92, frontier_reward=2.5,
                  procedural=False):
    """Paired comparison on identical simulator seeds and goal sequences."""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide nonempty distinct seeds")
    records = []
    start = time.process_time()
    for changed in ((False,) if procedural else (False, True)):
        for seed in seeds:
            if procedural:
                baseline_world = SensorEpisode(changed=False, limit=160)
                baseline_world._maze = GeneratedMaze(seed)  # Harness only.
                baseline_reading = baseline_world.reset(seed, goals=3)
                baseline_agent = OnlineAgent(
                    "history_novelty", seed + 19, model=None, prototypes=None)
                while True:
                    action = baseline_agent.act(baseline_reading)
                    baseline_reading, _, finished = baseline_world.step(action)
                    baseline_agent.observe(baseline_reading)
                    if finished:
                        break
                baseline = baseline_world.metrics()
            else:
                baseline = run_episode(
                    "history_novelty", seed, None, None, changed=changed)
            records.append({
                "seed": seed, "changed": changed, "mode": "history_novelty",
                "completed": baseline["completed"],
                "waypoints": baseline["waypoints"], "steps": baseline["steps"],
                "cells": None, "messages": None,
            })
            for mode, freeze in (("greedy", False), ("diffusion", False),
                                 ("diffusion", True)):
                env = SensorEpisode(changed=changed, limit=160)
                if procedural:
                    env._maze = GeneratedMaze(seed)  # Harness only.
                reading = env.reset(seed, goals=3)
                agent = LocalValueCells(
                    seed, mode=mode, rounds=rounds, decay=decay,
                    frontier_reward=frontier_reward, freeze_memory=freeze)
                while True:
                    action = agent.act(reading)
                    reading, _, done = env.step(action)
                    agent.observe(reading)
                    if done:
                        break
                records.append({
                    "seed": seed, "changed": changed,
                    "mode": "frozen_odometry" if freeze else mode,
                    **env.metrics(),
                    "cells": len(agent.open_actions),
                    "messages": agent.local_message_updates,
                })
    summary = {}
    for changed in ((False,) if procedural else (False, True)):
        label = "procedural" if procedural else "changed" if changed else "stable"
        summary[label] = {}
        for mode in ("history_novelty", "greedy", "diffusion",
                     "frozen_odometry"):
            rows = [r for r in records
                    if r["changed"] == changed and r["mode"] == mode]
            summary[label][mode] = {
                "success_rate": sum(r["completed"] for r in rows) / len(rows),
                "mean_waypoints": sum(r["waypoints"] for r in rows) / len(rows),
                "mean_steps": sum(r["steps"] for r in rows) / len(rows),
                "mean_message_updates":
                    sum(r["messages"] or 0 for r in rows) / len(rows),
            }
    return {
        "protocol": {
            "seeds": list(seeds), "goals": 3, "action_limit": 160,
            "procedural_test_only_layouts": procedural,
            "rounds": rounds, "decay": decay,
            "frontier_reward": frontier_reward,
            "relative_odometry": "programmed motor update",
            "spatial_memory": "observed online, no global state supplied",
            "value_diffusion": "engineered neighbor-only updates",
            "centralized_python_scheduler": True,
            "oracle_coordinates_or_bfs_in_agent": False,
            "process_cpu_seconds": time.process_time() - start,
        },
        "summary": summary, "episodes": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="13001,13002,13003,13004,13005")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--decay", type=float, default=.92)
    parser.add_argument("--frontier-reward", type=float, default=2.5)
    parser.add_argument("--procedural", action="store_true",
                        help="Use held-out procedurally modified maze layouts")
    parser.add_argument("--output", default="")
    options = parser.parse_args()
    result = run_benchmark(
        [int(s) for s in options.seeds.split(",")],
        rounds=options.rounds, decay=options.decay,
        frontier_reward=options.frontier_reward,
        procedural=options.procedural)
    if options.output:
        output = Path(options.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "protocol": result["protocol"], "summary": result["summary"]},
        indent=2))


if __name__ == "__main__":
    main()
