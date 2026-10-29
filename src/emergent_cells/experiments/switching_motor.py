"""On-policy recovery after an unannounced discrete motor-code change.

This is a sensorimotor calibration experiment, NOT general learned geometry.
Local wall indicators and a contact sensor suffice for online permutation
change detection. Values and relative-coordinate integration remain authored.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import random
import time

from .learned_motor_code import MotorCodeWorld, MotorCodeLearner, PERMUTATIONS, learn_mapping
from .local_value_cells import LocalValueCells


class SwitchedMotorWorld(MotorCodeWorld):
    """Unknown secret command remapping after a fixed number of interactions."""

    def __init__(self, first, second, *, switch_at=35, changed=False, seed=0,
                 contact_noise=.04, procedural=False):
        super().__init__(first, changed=changed,
                         procedural_seed=seed if procedural else None,
                         contact_flip_probability=contact_noise, seed=seed)
        if first == second:
            raise ValueError("old and new maps must differ")
        self._first = tuple(first)
        self._second = tuple(second)
        self._switch_at = switch_at
        self._actions = 0

    def step(self, motor_command):
        if self._actions >= self._switch_at:
            self._mapping = self._second
        output = super().step(motor_command)
        self._actions += 1
        return output


class RecentMotorEvidence:
    """Windowed Bayesian log likelihood over last local contact observations."""

    def __init__(self, first_mapping, *, contact_noise=.04, window=30,
                 margin=5., min_samples=12):
        if window < min_samples or margin <= 0:
            raise ValueError("invalid evidence window")
        self.first_mapping = first_mapping
        self.estimate = first_mapping
        self.history = deque(maxlen=window)
        self.contact_noise = contact_noise
        self.margin = margin
        self.min_samples = min_samples
        self.transitions = 0
        self.change_steps = []

    def observe(self, walls, command, contact, step):
        self.history.append((tuple(walls), command, bool(contact)))
        if len(self.history) < self.min_samples:
            return False
        yes = math.log(1-self.contact_noise)
        no = math.log(self.contact_noise)
        scores = [0.] * len(PERMUTATIONS)
        for walls, command, contact in self.history:
            for i, candidate in enumerate(PERMUTATIONS):
                scores[i] += yes if bool(walls[candidate[command]]) == contact else no
        winner = max(range(len(scores)), key=lambda i: scores[i])
        index = PERMUTATIONS.index(self.estimate)
        best = PERMUTATIONS[winner]
        if best != self.estimate and scores[winner] - scores[index] > self.margin:
            self.estimate = best
            self.transitions += 1
            self.change_steps.append(step)
            self.history.clear()
            return True
        return False


MODES = ("online_reset", "online_keep_map", "frozen_mapping", "oracle_switch")


def episode(seed, first, second, *, method, switch_at=35,
            changed=False, procedural=False,
            contact_noise=.04, window=30, margin=5., rounds=10,
            first_estimate=None):
    env = SwitchedMotorWorld(first, second, switch_at=switch_at,
                             changed=changed, seed=seed,
                             procedural=procedural, contact_noise=contact_noise)
    reading = env.reset(seed, goals=3)
    controller = LocalValueCells(seed, rounds=rounds)
    guessed = first if first_estimate is None else first_estimate
    detector = RecentMotorEvidence(guessed, contact_noise=contact_noise,
                                    window=window, margin=margin)
    errors = 0
    count = 0
    false_mapping_changes = 0
    earliest_correct_recovery = None
    while True:
        if method == "oracle_switch" and count == switch_at:
            guessed = second
            controller = LocalValueCells(seed + 17, rounds=rounds)
        desired = controller.act(reading)
        inverse = {direction: cmd for cmd, direction in enumerate(guessed)}
        actual_command = inverse[desired]
        previous = reading
        reading, contact, _, done = env.step(actual_command)
        # Harness-only truth for error metric. Never used for policy decisions.
        actual_dir = first[actual_command] if count < switch_at else second[actual_command]
        errors += int(actual_dir != desired)
        if method.startswith("online"):
            changed_map = detector.observe(
                previous.walls, actual_command, contact, count+1)
            if changed_map:
                guessed = detector.estimate
                if count < switch_at or guessed != second:
                    false_mapping_changes += 1
                if count >= switch_at and guessed == second and earliest_correct_recovery is None:
                    earliest_correct_recovery = count + 1 - switch_at
                if method == "online_reset":
                    # Discard potentially corrupted relative map, not a
                    # privileged pose reset. New origin is locally defined.
                    controller = LocalValueCells(seed + 17 + count, rounds=rounds)
                else:
                    controller.observe(reading)
            else:
                controller.observe(reading)
        else:
            controller.observe(reading)
        count += 1
        if done:
            break
    return {
        **env._env.metrics(),
        "motor_error_actions": errors,
        "last_mapping_correct": guessed == second,
        "changes_detected": detector.transitions,
        "false_or_incorrect_mapping_changes": false_mapping_changes,
        "change_detected_after": earliest_correct_recovery,
        "switch_occurred": count > switch_at,
    }


def benchmark(seeds, *, calibration_steps=100, contact_noise=.04,
              switch_at=35, window=30, margin=5.,
              rounds=10, procedural=False):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("distinct seeds required")
    started = time.process_time()
    rows = []
    for seed in seeds:
        rng = random.Random(seed * 157 + 227)
        first = PERMUTATIONS[rng.randrange(24)]
        candidates = [p for p in PERMUTATIONS if p != first]
        second = rng.choice(candidates)
        trained = learn_mapping(seed, first, calibration_steps=calibration_steps,
                                noise=contact_noise)
        first_estimate = trained.estimate()
        for changed in ((False,) if procedural else (False,True)):
            for mode in MODES:
                record = episode(seed, first, second, method=mode,
                                 switch_at=switch_at, changed=changed,
                                 procedural=procedural,
                                 contact_noise=contact_noise,
                                 window=window, margin=margin,
                                 rounds=rounds,
                                 first_estimate=first_estimate if mode != "oracle_switch" else first)
                rows.append({
                    "seed":seed,"changed":changed,"method":mode,
                    "initial_calibration_correct":first_estimate==first,
                    **record})
    summary={}
    for changed in ((False,) if procedural else (False, True)):
        name = "procedural" if procedural else ("changed" if changed else "stable")
        summary[name] = {}
        for mode in MODES:
            selected = [r for r in rows if r["changed"]==changed
                        and r["method"]==mode]
            delays = [r["change_detected_after"] for r in selected
                      if r["change_detected_after"] is not None]
            summary[name][mode] = {
                "success_rate":sum(r["completed"] for r in selected)/len(selected),
                "mean_waypoints":sum(r["waypoints"] for r in selected)/len(selected),
                "mean_steps":sum(r["steps"] for r in selected)/len(selected),
                "mean_motor_errors":sum(r["motor_error_actions"] for r in selected)/len(selected),
                "final_map_correct_rate":sum(r["last_mapping_correct"] for r in selected)/len(selected),
                "detection_rate":len(delays)/len(selected),
                "mean_false_or_incorrect_changes":sum(
                    r["false_or_incorrect_mapping_changes"] for r in selected)/len(selected),
                "mean_detection_delay_if_detected":
                    sum(delays)/len(delays) if delays else None}
    return {"protocol":{
        "seeds":seeds,"switch_at_action_index":switch_at,
        "known_compass_sensor_axes":True,"contact_sensor_added":True,
        "initial_calibration_extra_source_actions":calibration_steps,
        "motor_change_not_announced_to_policy":True,
        "known_action_permutation_hypothesis_class":True,
        "value_diffusion_and_relative_odometry_programmed":True,
        "window":window,"change_margin":margin,"contact_noise":contact_noise,
        "procedural":procedural,"cpu_seconds":time.process_time()-started
        }, "summary":summary,"records":rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",default="101,102,103,104,105")
    p.add_argument("--calibration-steps",type=int,default=100)
    p.add_argument("--switch-at",type=int,default=35)
    p.add_argument("--contact-noise",type=float,default=.04)
    p.add_argument("--window",type=int,default=30)
    p.add_argument("--margin",type=float,default=5.)
    p.add_argument("--rounds",type=int,default=10)
    p.add_argument("--procedural",action="store_true")
    p.add_argument("--output",default="")
    a=p.parse_args()
    report=benchmark([int(x) for x in a.seeds.split(",")],
                     calibration_steps=a.calibration_steps,
                     contact_noise=a.contact_noise,
                     switch_at=a.switch_at,window=a.window,margin=a.margin,
                     rounds=a.rounds,procedural=a.procedural)
    if a.output:
        dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"summary":report["summary"],
                      "cpu_seconds":report["protocol"]["cpu_seconds"],
                      "output":a.output},indent=2))


if __name__ == "__main__":
    main()
