"""Multi-step goal transfer using locally acquired transition memories.

A small CPU-only research benchmark, NOT autonomous general intelligence.
Position and active target coordinates are privileged structured observations;
route search is a programmed central planner, not emergent cell reasoning.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time
import tracemalloc

MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0))
SOURCE_GOALS = ((1, 1), (7, 7))
HELDOUT_GOALS = ((1, 2), (1, 6), (2, 1), (2, 7), (6, 1), (6, 7), (7, 2), (7, 6))
MODES = ("reusable_model", "frozen_model", "cold_model", "greedy", "random")


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class Maze:
    changed: bool = False
    size: int = 9

    def blocked(self):
        wall = {(4, y) for y in range(self.size) if y not in (2, 6)}
        wall.update(((2, 4), (2, 5), (6, 3), (6, 4), (7, 4)))
        if self.changed:
            wall.add((4, 2))  # Unannounced closure; lower passage stays open.
        return wall

    def free(self, point):
        x, y = point
        return 0 <= x < self.size and 0 <= y < self.size and point not in self.blocked()

    def states(self):
        return [(x, y) for y in range(self.size) for x in range(self.size)
                if self.free((x, y))]

    def step(self, state, action):
        dx, dy = MOVES[action]
        nxt = (state[0] + dx, state[1] + dy)
        return nxt if self.free(nxt) else state

    def shortest(self, source, target):
        if source == target:
            return 0
        q = deque([(source, 0)])
        seen = {source}
        while q:
            state, distance = q.popleft()
            for action in range(4):
                nxt = self.step(state, action)
                if nxt == target:
                    return distance + 1
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, distance + 1))
        return None


class LocalTransitionCells:
    """Each visited position owns four independent, locally updated edges."""

    def __init__(self):
        self.edges = {}     # (x, y) -> {local_action: locally observed successor}
        self.visits = {}    # (x, y, action) -> observations
        self.revisions = 0

    def clone(self):
        other = LocalTransitionCells()
        other.edges = {s: choices.copy() for s, choices in self.edges.items()}
        other.visits = self.visits.copy()
        other.revisions = self.revisions
        return other

    def observe(self, state, action, next_state):
        choices = self.edges.setdefault(state, {})
        if action in choices and choices[action] != next_state:
            self.revisions += 1
        choices[action] = next_state
        key = (state, action)
        self.visits[key] = self.visits.get(key, 0) + 1

    def coverage(self, world):
        possible = len(world.states()) * 4
        observed = sum(int(a in self.edges.get(s, {}))
                       for s in world.states() for a in range(4))
        return observed / possible

    def stored_bytes_estimate(self):
        # Approximate shallow Python payload only, not RSS or physical memory.
        return (sys.getsizeof(self.edges) + sys.getsizeof(self.visits) +
                sum(sys.getsizeof(s) + sys.getsizeof(a) +
                    sys.getsizeof(c) for s, c in self.edges.items()
                    for a in c))

    def decide(self, state, target):
        """Search learned edges; otherwise explore a reachable local frontier.

        The planner is an explicit, central BFS (not learned/distributed).
        Neither the world map nor world.step is used in action selection.
        """
        queue = deque([state])
        first_action = {state: None}
        depth = {state: 0}
        while queue:
            position = queue.popleft()
            if position == target and position != state:
                return first_action[position]
            for action, nxt in sorted(self.edges.get(position, {}).items()):
                if nxt == position or nxt in first_action:
                    continue
                first_action[nxt] = action if position == state else first_action[position]
                depth[nxt] = depth[position] + 1
                queue.append(nxt)
        frontiers = [position for position in first_action
                     if len(self.edges.get(position, {})) < 4]
        if frontiers:
            # First seek an unexplored action that is cheaply reachable.
            frontier = min(frontiers, key=lambda p:
                           (depth[p] + manhattan(p, target), depth[p], p))
            if frontier != state:
                return first_action[frontier]
            unknown = [a for a in range(4) if a not in self.edges.get(state, {})]
            return min(unknown, key=lambda a: (
                manhattan((state[0] + MOVES[a][0],
                           state[1] + MOVES[a][1]), target), a))
        return min(range(4), key=lambda a: (self.visits.get((state, a), 0), a))


def train_source(seed, steps):
    """Alternate between two source destinations using a local learned map.

    The target cue changes after successful arrival, without a task ID.
    Random resets periodically diversify exploration and are privileged.
    """
    world = Maze()
    rng = random.Random(seed * 917 + 171)
    states = world.states()
    model = LocalTransitionCells()
    state = rng.choice(states)
    goal_index = 0
    for t in range(steps):
        if t % 83 == 0:
            state = rng.choice(states)
        goal = SOURCE_GOALS[goal_index]
        action = (rng.randrange(4) if rng.random() < .35
                  else model.decide(state, goal))
        nxt = world.step(state, action)
        model.observe(state, action, nxt)
        if nxt == goal:
            goal_index = 1 - goal_index
        state = nxt
    return model


def missions(seed, n, world):
    rng = random.Random(seed * 1307 + 41)
    starts = world.states()
    cases = []
    for _ in range(n):
        start = rng.choice(starts)
        goals = []
        current = start
        for _ in range(3):  # Multi-step, three consecutive unseen targets.
            eligible = [g for g in HELDOUT_GOALS
                        if g != current and manhattan(g, current) >= 5
                        and g not in goals and world.free(g)]
            if not eligible:
                eligible = [g for g in HELDOUT_GOALS if g != current and g not in goals]
            goal = rng.choice(eligible)
            goals.append(goal)
            current = goal
        cases.append((start, tuple(goals)))
    return cases


def play(world, model, start, goals, mode, seed, limit=160):
    rng = random.Random(seed)
    state = start
    target_i = 0
    steps = 0
    original_revisions = model.revisions if model else 0
    path_optimum = 0
    current = start
    for g in goals:
        distance = world.shortest(current, g)
        if distance is None:
            raise ValueError("unreachable goal")
        path_optimum += distance
        current = g
    while steps < limit and target_i < len(goals):
        target = goals[target_i]  # Structured goal cue, not a task ID.
        if mode == "random":
            action = rng.randrange(4)
        elif mode == "greedy":
            action = min(range(4), key=lambda a: (
                manhattan((state[0] + MOVES[a][0],
                           state[1] + MOVES[a][1]), target), a))
        else:
            action = model.decide(state, target)
        next_state = world.step(state, action)
        if model is not None and mode != "frozen_model":
            model.observe(state, action, next_state)
        state = next_state
        steps += 1
        if state == target:
            target_i += 1
    return {"complete": target_i == len(goals),
            "waypoints": target_i,
            "steps": steps,
            "oracle_steps": path_optimum,
            "path_excess": steps - path_optimum if target_i == len(goals) else None,
            "revisions": model.revisions - original_revisions if model else 0}


def summarize(rows):
    completed = [r for r in rows if r["complete"]]
    count = len(rows)
    return {"missions": count,
            "complete_rate": sum(r["complete"] for r in rows) / count,
            "mean_waypoints": sum(r["waypoints"] for r in rows) / count,
            "mean_steps": sum(r["steps"] for r in rows) / count,
            "mean_extra_steps_if_complete": (
                sum(r["path_excess"] for r in completed) / len(completed)
                if completed else None),
            "total_revisions": sum(r["revisions"] for r in rows)}


def trial(seed, train_steps=7000, eval_missions=24, step_limit=160):
    world = Maze()
    t0 = time.process_time()
    tracemalloc.start()
    trained = train_source(seed, train_steps)
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    train_cpu = time.process_time() - t0
    cases = missions(seed, eval_missions, world)
    out = {}
    for mode in MODES:
        t = time.process_time()
        rows = []
        for i, (start, goals) in enumerate(cases):
            model = (trained.clone() if mode in ("reusable_model", "frozen_model")
                     else LocalTransitionCells() if mode == "cold_model" else None)
            rows.append(play(world, model, start, goals, mode,
                             seed=seed * 333 + i, limit=step_limit))
        out[mode] = {**summarize(rows),
                     "eval_cpu_seconds": round(time.process_time() - t, 6),
                     "episodes": rows}
    # Chosen hard negative: a formerly open passage closes unexpectedly.
    shifted = Maze(changed=True)
    dynamic = {}
    for mode in MODES:
        model = (trained.clone() if mode in ("reusable_model", "frozen_model")
                 else LocalTransitionCells() if mode == "cold_model" else None)
        dynamic[mode] = play(shifted, model, (1, 2), ((7, 2), (1, 6)), mode,
                             seed=seed * 811 + 7, limit=step_limit)
    return {"seed": seed, "training_actions": train_steps,
            "source_observed_edge_fraction": trained.coverage(world),
            "source_edge_count": len(trained.visits),
            "source_training_cpu_seconds": round(train_cpu, 6),
            "source_peak_tracemalloc_bytes": peak_memory,
            "shallow_source_memory_bytes": trained.stored_bytes_estimate(),
            "heldout": out, "door_closure": dynamic,
            "training_world_source_goal_cues": SOURCE_GOALS,
            "heldout_goal_pool": HELDOUT_GOALS}


def benchmark(seeds, train_steps=7000, eval_missions=24, step_limit=160):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("unique nonempty seeds required")
    if train_steps < 100 or eval_missions < 1 or step_limit < 20:
        raise ValueError("invalid benchmark size")
    trials = [trial(s, train_steps, eval_missions, step_limit) for s in seeds]
    summary = {}
    for mode in MODES:
        field = [t["heldout"][mode] for t in trials]
        changed = [t["door_closure"][mode] for t in trials]
        summary[mode] = {
            "heldout_complete_rate": sum(x["complete_rate"] for x in field) / len(field),
            "heldout_mean_waypoints": sum(x["mean_waypoints"] for x in field) / len(field),
            "heldout_mean_extra_steps_if_complete": (
                sum(x["mean_extra_steps_if_complete"] for x in field
                    if x["mean_extra_steps_if_complete"] is not None) /
                max(1, sum(x["mean_extra_steps_if_complete"] is not None for x in field))),
            "door_closure_success_rate": sum(x["complete"] for x in changed) / len(changed),
            "door_closure_mean_revisions": sum(x["revisions"] for x in changed) / len(changed),
            "evaluation_cpu_seconds_per_seed": sum(x["eval_cpu_seconds"] for x in field) / len(field)}
    return {"protocol": {
        "description": "Tabular local transition cells plus authored central BFS search",
        "goal_observation": "Exact (x,y) target coordinates, supplied by environment",
        "position_observation": "Exact (x,y), supplied by environment",
        "wall_map_visible_to_policy": False,
        "source_exploration": "Two alternating visible goal cues, epsilon=0.35, periodic privileged random state resets",
        "source_goal_cues": SOURCE_GOALS,
        "train_steps": train_steps, "eval_missions": eval_missions,
        "mission_waypoints": 3, "step_limit": step_limit, "seeds": seeds,
        "heldout_goals": HELDOUT_GOALS},
        "summary": summary, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--train-steps", type=int, default=7000)
    parser.add_argument("--missions", type=int, default=24)
    parser.add_argument("--step-limit", type=int, default=160)
    parser.add_argument("--output", default="")
    a = parser.parse_args()
    start = time.perf_counter()
    data = benchmark([int(s.strip()) for s in a.seeds.split(",")],
                     train_steps=a.train_steps,
                     eval_missions=a.missions, step_limit=a.step_limit)
    data["elapsed_wall_seconds"] = time.perf_counter() - start
    if a.output:
        p = Path(a.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": data["protocol"],
                      "summary": data["summary"],
                      "mean_source_coverage": sum(t["source_observed_edge_fraction"]
                          for t in data["trials"]) / len(data["trials"]),
                      "mean_source_training_cpu_seconds": sum(
                          t["source_training_cpu_seconds"] for t in data["trials"]
                          ) / len(data["trials"]),
                      "mean_source_tracemalloc_peak_bytes": sum(
                          t["source_peak_tracemalloc_bytes"] for t in data["trials"]
                          ) / len(data["trials"]),
                      "elapsed_wall_seconds": data["elapsed_wall_seconds"],
                      "output": a.output}, indent=2))


if __name__ == "__main__":
    main()
