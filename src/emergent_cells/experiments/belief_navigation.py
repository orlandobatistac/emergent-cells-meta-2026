"""Locally-observed multi-hypothesis motion belief for stochastic environments.

CPU-only beam search over *self-integrated* relative positions and the walls
observed at each visited relative location. No simulator pose/map enters policy.

This is an engineered Bayesian/beam-filter controller, not a learned neural
state model or true SLAM. It is an experiment in maintaining uncertainty
rather than committing every ambiguous motor action to a single movement.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

from .local_value_cells import MOVES, REVERSE
from .stochastic_motor import SlipWorld, calibrate


@dataclass
class Hypothesis:
    position: tuple[int,int]
    log_weight: float
    walls: dict
    visits: Counter
    edge_visits: Counter

    def fork(self, position, log_weight, seen):
        wall_map = self.walls.copy()
        wall_map[position] = seen
        visits = self.visits.copy()
        visits[position] += 1
        return Hypothesis(position, log_weight, wall_map, visits,
                          self.edge_visits.copy())


class BeliefExplorer:
    def __init__(self, seed, *, rates=(.65,)*4, tick_error=.2,
                 floor_error=.15, width=24, rounds=6,
                 aggregation="consensus"):
        if width < 1 or rounds < 0:
            raise ValueError("belief width must be positive")
        self.random=random.Random(seed+19)
        self.rates=rates
        self.tick_error=tick_error
        self.floor_error=floor_error
        self.width=width
        self.rounds=rounds
        if aggregation not in ("consensus","best"):
            raise ValueError("aggregation must be consensus or best")
        self.aggregation=aggregation
        self.beam=[]
        self.last_action=None
        self.message_ops=0
        self.belief_entropy=[]
        self.weight_of_best=[]
        self.phase=0

    def reset(self, reading):
        seen={ (0,0): tuple(reading.walls)}
        self.beam=[Hypothesis((0,0), 0., seen, Counter({(0,0):1}), Counter())]
        self.last_action=None
        self.belief_entropy=[]
        self.weight_of_best=[]

    @staticmethod
    def _next(pos, action):
        dx,dy=MOVES[action]
        return (pos[0]+dx,pos[1]+dy)

    def _scores(self, particle, reading):
        available=[a for a,w in enumerate(reading.walls) if not w]
        if not available:return {a:-1e3 for a in range(4)}
        map_walls=particle.walls
        pos=particle.position
        potentials={p:0. for p in map_walls}
        for _ in range(self.rounds):
            next_values={}
            for p,walls in map_walls.items():
                open_actions=[a for a,blocked in enumerate(walls) if not blocked]
                frontier=any(self._next(p,a) not in map_walls for a in open_actions)
                reward=.9/math.sqrt(1+particle.visits[p])+(2.5 if frontier else 0.)
                neighbors=[potentials[self._next(p,a)] for a in open_actions
                           if self._next(p,a) in map_walls]
                next_values[p]=max(reward,.92*max(neighbors,default=0.))
                self.message_ops+=len(neighbors)
            potentials=next_values
        scores={}
        for a in available:
            p=self._next(pos,a)
            backwards=int(self.last_action is not None and a==REVERSE[self.last_action])
            if p not in map_walls:
                val=4.5
            else:
                val=potentials[p]-.04*particle.visits[p]
            val-=.09*particle.edge_visits[pos,a]
            val-=.13*backwards
            scores[a]=val
        return scores

    def act(self, reading):
        legal=[a for a,w in enumerate(reading.walls) if not w]
        if not legal:return 0
        goals=[a for a in legal if reading.nearby_target[a]]
        if goals:
            a=goals[0]
        else:
            # Use a weighted consensus of local plans over uncertain states;
            # no global pose is supplied to any hypothesis.
            candidates=(self.beam[:1] if self.aggregation=="best"
                        else self.beam)
            best=max(p.log_weight for p in candidates)
            probs=[math.exp(p.log_weight-best) for p in candidates]
            denom=sum(probs)
            scores={a:0. for a in legal}
            for particle,weight in zip(candidates,probs):
                local=self._scores(particle,reading)
                for action in legal:
                    scores[action]+=(weight/denom)*local[action]
            a=max(legal,key=lambda action:(scores[action]+self.random.random()*.018,-action))
        for particle in self.beam:
            particle.edge_visits[particle.position,a]+=1
        self.last_action=a
        return a

    def observe(self, before, after, action):
        if self.last_action is None:
            raise RuntimeError("call act before observe")
        if before.walls[action]:
            # Never choose wall directions; no hidden movement truth.
            pass
        floor_same=(1-self.floor_error)**2+self.floor_error**2
        observed_flip=before.floor_signal!=after.floor_signal
        expanded=[]
        for p in self.beam:
            prior=self.rates[action] if not before.walls[action] else .0001
            for moved in (False,True):
                if moved and before.walls[action]:continue
                pos=self._next(p.position,action) if moved else p.position
                known=p.walls.get(pos)
                # In the static world the wall signature is a deterministic
                # local observation. Inconsistent loops are eliminated.
                if known is not None and known!=tuple(after.walls):
                    continue
                if moved:
                    reverse=REVERSE[action]
                    if after.walls[reverse]:
                        continue
                ll=math.log(prior if moved else (1.-prior))
                observed_tick=bool(after.motion_tick)
                ll+=math.log((1-self.tick_error) if observed_tick==moved
                             else self.tick_error)
                likelihood_floor=(floor_same if observed_flip==moved
                                  else 1.-floor_same)
                ll+=math.log(likelihood_floor)
                child=p.fork(pos,p.log_weight+ll,tuple(after.walls))
                if after.previous_reward>.5:
                    child.visits.clear()
                    child.visits[pos]=1
                    child.edge_visits.clear()
                expanded.append(child)
        if not expanded:
            # Do not invent a privileged fix. Keep prior particle and record
            # a failure to reconcile locally observed wall evidence.
            parent=self.beam[0]
            expanded=[parent.fork(parent.position,parent.log_weight,
                                  tuple(after.walls))]
        expanded.sort(key=lambda p:p.log_weight,reverse=True)
        # Bound state by retaining competing locations, with max two distinct
        # histories per candidate location and fixed overall beam capacity.
        counts=Counter()
        chosen=[]
        for p in expanded:
            if counts[p.position]>=2:continue
            counts[p.position]+=1
            chosen.append(p)
            if len(chosen)>=self.width:break
        best=chosen[0].log_weight
        denominator=sum(math.exp(p.log_weight-best) for p in chosen)
        entropy=0.
        for p in chosen:
            weight=math.exp(p.log_weight-best)/denominator
            entropy-=weight*math.log(max(weight,1e-12))
            p.log_weight=math.log(max(weight,1e-12))
        self.belief_entropy.append(entropy)
        self.weight_of_best.append(math.exp(chosen[0].log_weight))
        self.beam=chosen


def rollout(seed, rates, *, mode, calibrated, tick_error=.20,
            floor_error=.15, changed=False, procedural=False,
            width=24, rounds=6):
    world=SlipWorld(seed,rates=rates,motion_noise=tick_error,
                    floor_noise=floor_error,changed=changed,
                    procedural=procedural,limit=160)
    obs=world.reset(seed)
    origin=world._env._position # Validation harness only.
    tracker=BeliefExplorer(seed,rates=calibrated,tick_error=tick_error,
                           floor_error=floor_error,
                           width=1 if mode=="single" else width,rounds=rounds,
                           aggregation="best" if mode=="beam_best" else "consensus")
    tracker.reset(obs)
    while True:
        chosen=tracker.act(obs)
        prior=obs
        obs,done=world.step(chosen)
        tracker.observe(prior,obs,chosen)
        if done:break
    truth=(world._env._position[0]-origin[0],
           world._env._position[1]-origin[1])
    error=min(abs(p.position[0]-truth[0])+abs(p.position[1]-truth[1])
              for p in tracker.beam)
    best=tracker.beam[0]
    best_error=(abs(best.position[0]-truth[0])+
                abs(best.position[1]-truth[1]))
    return {**world._env.metrics(),
            "best_pose_error":best_error,
            "min_hypothesis_pose_error":error,
            "mean_entropy":sum(tracker.belief_entropy)/max(1,len(tracker.belief_entropy)),
            "mean_best_probability":sum(tracker.weight_of_best)/max(1,len(tracker.weight_of_best)),
            "unique_hypotheses":len(tracker.beam),
            "message_ops":tracker.message_ops}


def benchmark(seeds, *, calibration_actions=120,
              tick_error=.20,floor_error=.15,width=24,rounds=6,
              procedural=False):
    if not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError("unique seeds required")
    start=time.process_time()
    records=[]
    for seed in seeds:
        rng=random.Random(seed*41+919)
        rates=tuple(rng.choice((.45,.60,.75,.90)) for _ in range(4))
        model=calibrate(seed,rates,actions=calibration_actions,
                        tick_error=tick_error,floor_error=floor_error)
        calibrated=model.estimate()
        for changed in ((False,) if procedural else (False,True)):
            for mode in ("single","beam","beam_best","beam_true_rates"):
                result=rollout(seed,rates,mode=mode,
                    calibrated=rates if mode=="beam_true_rates" else calibrated,
                    tick_error=tick_error,floor_error=floor_error,
                    changed=changed,procedural=procedural,
                    width=width,rounds=rounds)
                records.append({"seed":seed,"changed":changed,
                                "mode":mode,**result})
    summary={}
    for changed in ((False,) if procedural else (False,True)):
        label="procedural" if procedural else "changed" if changed else "stable"
        summary[label]={}
        for mode in ("single","beam","beam_best","beam_true_rates"):
            rows=[r for r in records if r["changed"]==changed and r["mode"]==mode]
            summary[label][mode]={
                "success_rate":sum(r["completed"] for r in rows)/len(rows),
                "mean_waypoints":sum(r["waypoints"] for r in rows)/len(rows),
                "mean_steps":sum(r["steps"] for r in rows)/len(rows),
                "mean_best_pose_error":sum(r["best_pose_error"] for r in rows)/len(rows),
                "mean_min_hypothesis_error":sum(r["min_hypothesis_pose_error"] for r in rows)/len(rows),
                "mean_cell_messages":sum(r["message_ops"] for r in rows)/len(rows)}
    return {"protocol":{
        "seeds":seeds,"max_actions":160,"goal_count":3,"beam_width":width,
        "value_rounds":rounds,"proprioceptive_noise":tick_error,
        "floor_parity_noise":floor_error,"separate_calibration_actions":calibration_actions,
        "no_agent_pose_or_oracle_map":True,
        "authored_bayesian_filter_and_local_value_diffusion":True,
        "prior_rates_learned_from_sensor_ticks":True,
        "cpu_seconds":time.process_time()-start,
        "procedural":procedural,
    }, "summary":summary,"records":records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",default="101,102,103")
    p.add_argument("--calibration-actions",type=int,default=120)
    p.add_argument("--tick-error",type=float,default=.2)
    p.add_argument("--floor-error",type=float,default=.15)
    p.add_argument("--width",type=int,default=24)
    p.add_argument("--rounds",type=int,default=6)
    p.add_argument("--procedural",action="store_true")
    p.add_argument("--output",default="")
    a=p.parse_args()
    result=benchmark([int(s) for s in a.seeds.split(",")],
                     calibration_actions=a.calibration_actions,
                     tick_error=a.tick_error,floor_error=a.floor_error,
                     width=a.width,rounds=a.rounds,
                     procedural=a.procedural)
    if a.output:
        dest=Path(a.output)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"summary":result["summary"],
                      "cpu_seconds":result["protocol"]["cpu_seconds"],
                      "output":a.output},indent=2))


if __name__=="__main__":
    main()
