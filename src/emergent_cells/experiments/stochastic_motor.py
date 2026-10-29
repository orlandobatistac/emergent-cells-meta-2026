"""Stochastic motor dynamics from local sensors; CPU-only R1 experiment.

Unknown command-specific slip/no-op rates are learned from noisy local
movement ticks. The independent floor parity sensor gives an additional,
possibly corrupted movement cue. No global pose/goal/map labels reach learner.

Limits: motor directions and sensor-noise rates are authored; the tracker
collapses uncertain motion to a hard relative displacement. It is not SLAM.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

from .local_value_cells import LocalValueCells
from .sensor_world import SensorEpisode
from .recurrent_state import GeneratedMaze


@dataclass(frozen=True, slots=True)
class MotionReading:
    walls: tuple[bool, bool, bool, bool]
    nearby_target: tuple[bool, bool, bool, bool]
    floor_signal: int
    previous_reward: float
    motion_tick: bool


class SlipWorld:
    """Harness-owned stochastic no-op motor failures, not agent-accessible."""

    def __init__(self, seed, *, rates, motion_noise=.2, floor_noise=.15,
                 changed=False, procedural=False, limit=160):
        if len(rates) != 4 or any(not 0 <= p <= 1 for p in rates):
            raise ValueError("rates must contain four probabilities")
        self._rng = random.Random(seed * 53003 + 17)
        self._rates = tuple(rates)
        self._motion_noise = motion_noise
        self._floor_noise = floor_noise
        self._env = SensorEpisode(changed=changed, limit=limit)
        if procedural:
            self._env._maze = GeneratedMaze(seed)
        self._obs = None
        self._previous_true_motion = None

    def _sense(self, reading, motion):
        parity = reading.floor_signal
        if self._rng.random() < self._floor_noise:
            parity ^= 1
        tick = bool(motion)
        if self._rng.random() < self._motion_noise:
            tick = not tick
        return MotionReading(reading.walls, reading.nearby_target, parity,
                             reading.previous_reward, tick)

    def reset(self, seed):
        base = self._env.reset(seed, goals=3)
        self._obs = self._sense(base, False)
        return self._obs

    def step(self, direction):
        if direction not in range(4):
            raise ValueError("direction must be in 0..3")
        if self._obs is None:
            raise RuntimeError("reset before step")
        before = self._env._read()
        can_move = not before.walls[direction]
        succeeded = can_move and self._rng.random() < self._rates[direction]
        if succeeded:
            base, _, done = self._env.step(direction)
        else:
            # Simulation-only state update: no movement, one action consumed.
            if self._env._steps >= self._env._limit:
                raise RuntimeError("episode finished")
            self._env._steps += 1
            self._env._reward = -.01
            done = self._env._steps >= self._env._limit
            base = self._env._read()
        self._previous_true_motion = bool(succeeded)
        self._obs = self._sense(base, succeeded)
        return self._obs, done


class LearnedMotorReliability:
    """Empirical Bernoulli dynamics model from noisy *local* encoder ticks."""

    def __init__(self, *, tick_error=.2, smoothing=2.):
        if not 0 <= tick_error < .5 or smoothing <= 0:
            raise ValueError("invalid noise or smoothing")
        self.tick_error = tick_error
        self.smoothing = smoothing
        self.samples = [0] * 4
        self.reported_moves = [0] * 4

    def observe(self, command, available, motion_tick):
        if command not in range(4):
            raise ValueError("invalid direction")
        # Blocked commands are not informative about slip probability.
        if available:
            self.samples[command] += 1
            self.reported_moves[command] += int(motion_tick)

    def estimate(self):
        result = []
        error = self.tick_error
        for index in range(4):
            n = self.samples[index]
            observed_rate = ((self.reported_moves[index] + self.smoothing * .5) /
                             (n + self.smoothing))
            true_rate = (observed_rate - error) / (1. - 2. * error)
            result.append(min(.995, max(.005, true_rate)))
        return tuple(result)


def infer_move(previous, current, *, prior, tick_error, floor_error,
               use_tick=True, use_floor=True):
    """Bayesian fusion with explicitly authored binary sensor noise models."""
    if not 0 < prior < 1:
        raise ValueError("prior must be between 0 and 1")
    yes = prior
    no = 1. - prior
    if use_tick:
        actual_tick = bool(current.motion_tick)
        yes *= (1. - tick_error) if actual_tick else tick_error
        no *= tick_error if actual_tick else (1. - tick_error)
    if use_floor:
        flipped = previous.floor_signal != current.floor_signal
        same_noise = (1. - floor_error)**2 + floor_error**2
        different_noise = 1. - same_noise
        yes *= same_noise if flipped else different_noise
        no *= different_noise if flipped else same_noise
    return yes / (yes + no) >= .5


def calibrate(seed, rates, *, actions=120, tick_error=.2, floor_error=.15):
    model = LearnedMotorReliability(tick_error=tick_error)
    probe = SlipWorld(seed * 5 + 81, rates=rates,
                      motion_noise=tick_error, floor_noise=floor_error,
                      limit=160)
    rng = random.Random(seed * 719 + 93)
    count = 0
    while count < actions:
        observation = probe.reset(seed * 913 + count)
        for _ in range(min(25, actions-count)):
            cmd = rng.randrange(4)
            available = not observation.walls[cmd]
            observation, done = probe.step(cmd)
            model.observe(cmd, available, observation.motion_tick)
            count += 1
            if done:
                break
    return model


def rollout(seed, rates, *, method, learned_rates,
            tick_error=.2, floor_error=.15, changed=False,
            procedural=False, rounds=10):
    env = SlipWorld(seed, rates=rates, motion_noise=tick_error,
                    floor_noise=floor_error, changed=changed,
                    procedural=procedural)
    reading = env.reset(seed)
    absolute_origin = env._env._position  # Harness-only validation; not passed to agent.
    agent = LocalValueCells(seed, rounds=rounds)
    prediction_correct = 0
    attempts = 0
    while True:
        action = agent.act(reading)
        old = reading
        reading, done = env.step(action)
        if old.walls[action]:
            predicted_move = False
        elif method == "always_move":
            predicted_move = True
        elif method == "encoder_only":
            predicted_move = bool(reading.motion_tick)
        elif method == "floor_only":
            predicted_move = reading.floor_signal != old.floor_signal
        else:
            p = (rates[action] if method in ("oracle_probabilities", "oracle_map_fusion")
                 else learned_rates[action])
            predicted_move = infer_move(
                old, reading, prior=p, tick_error=tick_error,
                floor_error=floor_error)
            if method in ("learned_map_fusion", "oracle_map_fusion"):
                # In a static topology, staying in place never changes its
                # wall signature. This is an agent-accessible LOCAL sensor.
                if old.walls != reading.walls:
                    predicted_move = True
                else:
                    # The neighbor's wall signature is remembered from a
                    # previous visit, not obtained from simulator coordinates.
                    destination = agent.successor(agent.position, action)
                    known_neighbor = agent.open_actions.get(destination)
                    if known_neighbor is not None:
                        known_walls = tuple(a not in known_neighbor for a in range(4))
                        if known_walls != reading.walls:
                            predicted_move = False
        # Explicit non-learned state integration, conditioned on inferred
        # local movement outcome. No global position is observed by policy.
        if predicted_move:
            agent.observe(reading)
        else:
            agent.last_action = None
            agent.last_position = None
            if reading.previous_reward > .5:
                agent.completed_objectives += 1
                agent.goal_phase_visits.clear()
                agent.local_edge_visits.clear()
        # In the evaluation harness only, score true motion prediction.
        prediction_correct += int(predicted_move == env._previous_true_motion)
        attempts += 1
        if done:
            break
    return {
        **env._env.metrics(),
        "movement_prediction_accuracy": prediction_correct / attempts,
        "relative_pose_error": (abs(agent.position[0] - (env._env._position[0] - absolute_origin[0]))
                                + abs(agent.position[1] - (env._env._position[1] - absolute_origin[1]))),
        "internal_cells": len(agent.open_actions),
        "message_updates": agent.local_message_updates,
    }


METHODS = ("always_move", "encoder_only", "floor_only",
           "learned_fusion", "oracle_probabilities",
           "learned_map_fusion", "oracle_map_fusion")


def benchmark(seeds, *, calibration_actions=120, tick_error=.2,
              floor_error=.15, rounds=10, procedural=False):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("unique nonempty seeds required")
    start = time.process_time()
    records = []
    calibrations = []
    for seed in seeds:
        rng = random.Random(seed * 41 + 919)
        rates = tuple(rng.choice((.45,.60,.75,.90)) for _ in range(4))
        model = calibrate(seed, rates, actions=calibration_actions,
                          tick_error=tick_error, floor_error=floor_error)
        estimated = model.estimate()
        calibrations.append({
            "seed": seed, "rates": rates, "learned_rates": estimated,
            "mean_absolute_rate_error": sum(abs(a-b) for a,b
                                             in zip(rates,estimated))/4.,
            "training_samples": model.samples,
        })
        for changed in ((False,) if procedural else (False, True)):
            for method in METHODS:
                outcome = rollout(seed, rates, method=method,
                                  learned_rates=estimated,
                                  tick_error=tick_error,
                                  floor_error=floor_error,
                                  changed=changed, procedural=procedural,
                                  rounds=rounds)
                records.append({
                    "seed":seed, "changed":changed, "method":method,
                    **outcome})
    summary = {}
    for changed in ((False,) if procedural else (False, True)):
        scene = "procedural" if procedural else "changed" if changed else "stable"
        summary[scene] = {}
        for method in METHODS:
            rows = [x for x in records if x["changed"] == changed
                    and x["method"] == method]
            summary[scene][method] = {
                "success_rate": sum(x["completed"] for x in rows)/len(rows),
                "mean_waypoints": sum(x["waypoints"] for x in rows)/len(rows),
                "mean_steps": sum(x["steps"] for x in rows)/len(rows),
                "mean_movement_prediction_accuracy": sum(
                    x["movement_prediction_accuracy"] for x in rows)/len(rows)}
    return {"protocol":{
        "seeds":seeds,"max_navigation_actions":160,"goals":3,
        "calibration_actions_extra":calibration_actions,
        "motor_rates_hidden_per_seed":True,
        "motor_rates_range":[.45,.90],
        "motion_tick_flip_rate":tick_error,
        "parity_bit_flip_rate":floor_error,
        "motor_action_directions_known":True,
        "relative_odometry_engineered":True,
        "probability_fusion_model_engineered":True,
        "learned_stochastic_dynamics":"per-command Bernoulli success rate",
        "no_global_pose_or_map_input":True,
        "procedural":procedural,"cpu_seconds":time.process_time()-start},
        "mean_rate_mae":sum(c["mean_absolute_rate_error"] for c in
                            calibrations)/len(calibrations),
        "summary":summary,"calibrations":calibrations,"records":records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",default="101,102,103,104,105")
    p.add_argument("--calibration-actions",type=int,default=120)
    p.add_argument("--tick-error",type=float,default=.20)
    p.add_argument("--floor-error",type=float,default=.15)
    p.add_argument("--rounds",type=int,default=10)
    p.add_argument("--procedural",action="store_true")
    p.add_argument("--output",default="")
    a=p.parse_args()
    if not 0 <= a.tick_error < .5 or not 0 <= a.floor_error < .5:
        p.error("sensor error probabilities must be in [0,.5)")
    report=benchmark([int(x) for x in a.seeds.split(",")],
                     calibration_actions=a.calibration_actions,
                     tick_error=a.tick_error,
                     floor_error=a.floor_error,
                     rounds=a.rounds,procedural=a.procedural)
    if a.output:
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"summary":report["summary"],
                      "mean_rate_mae":report["mean_rate_mae"],
                      "cpu_seconds":report["protocol"]["cpu_seconds"],
                      "output":a.output},indent=2))


if __name__=="__main__":
    main()
