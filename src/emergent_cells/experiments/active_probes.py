"""Causal information-seeking by locally rewarded communicating cell pairs.

New October 27 2026 exploratory experiment. Cells privately learn a 1-bit
message before deployment. After deployment, a cell sees its private cue,
the received partner symbol, its own reward, and optional one-bit PROBE request
from its partner. It neither sees world rule, target, nor resource availability.

Two observationally indistinguishable worlds for baseline actions:
(1) true new action mapping on cue 11;
(2) resource pulse permanently absent on cue 11.
A coordinated counterfactual action can discriminate these worlds.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import copy
import numpy as np

from emergent_cells.experiments.learned_messages import (
    LocalProtocolPopulation, ProtocolSpec, sigmoid, streams,
)

MODES = ("passive", "random", "periodic", "information_gain")
SCENARIOS = ("stable", "scarcity_11", "changed_11", "uniform_drought", "global_change")


@dataclass(frozen=True)
class ProbeSpec:
    pairs: int = 48
    pretrain: int = 4500
    horizon: int = 450
    resource_probability: float = 0.92
    drought_probability: float = 0.55
    request_cost: float = 0.004
    intervention_cost: float = 0.025
    entropy_threshold: float = 0.06
    learned_threshold: float = 0.96

    def validate(self):
        if not 4 <= self.pairs <= 1000 or not 100 <= self.pretrain <= 100000:
            raise ValueError("invalid pretraining")
        if not 100 <= self.horizon <= 4000:
            raise ValueError("invalid test horizon")
        if not .3 < self.resource_probability < 1:
            raise ValueError("invalid resource pulse probability")
        if not .1 <= self.drought_probability < 1:
            raise ValueError("invalid drought pulse probability")
        if not 0 <= self.request_cost <= .1 or not 0 <= self.intervention_cost <= .5:
            raise ValueError("invalid simulated costs")
        if not 0 <= self.entropy_threshold < 2 or not .7 <= self.learned_threshold < 1:
            raise ValueError("invalid exploration thresholds")


def pretrain(seed: int, spec: ProbeSpec):
    """Use previously validated INDEPENDENT, local reward sender/receiver learner."""
    learner = LocalProtocolPopulation(
        ProtocolSpec(pairs=spec.pairs, train_steps=spec.pretrain,
                     test_steps=200), seed)
    for cues, u_send, u_act in streams(seed, spec.pretrain, learner.n, "train"):
        learner.episode(cues, u_send, u_act, "learned_messages", train=True)
    return learner.sender.copy(), learner.receiver.copy()


def binary_entropy(p):
    x=np.clip(p,1e-12,1.)
    return -np.sum(x*np.log2(x), axis=-1)


def expected_info(posterior, likelihood):
    # Matrix n,3 (normal/scarcity/mapping flip), binary observation reward.
    # Number of bits expected to learn from a *single intervention*.
    q=np.einsum("ij,j->i",posterior,likelihood)
    posterior_yes=posterior*likelihood[None,:]/np.maximum(q[:,None],1e-12)
    posterior_no=(posterior*(1-likelihood)[None,:]) / np.maximum((1-q)[:,None],1e-12)
    return (binary_entropy(posterior) -
            q*binary_entropy(posterior_yes)-
            (1-q)*binary_entropy(posterior_no))


def sample_world(seed, spec, scenario):
    if scenario not in SCENARIOS:
        raise ValueError("invalid scenario")
    rng=np.random.default_rng(seed + 5000019)
    for t in range(spec.horizon):
        bits=rng.integers(0,2,size=spec.pairs*2)
        # Each pair has its own next emission/draw, common to all methods.
        u_symbol=rng.random(spec.pairs*2)
        u_resource=rng.random(spec.pairs)
        u_probe=rng.random(spec.pairs*2)
        yield t,bits,u_symbol,u_resource,u_probe


def evaluate(seed: int, mode: str, scenario: str, spec: ProbeSpec,
             pretrained=None):
    spec.validate()
    if mode not in MODES or scenario not in SCENARIOS:
        raise ValueError("invalid condition")
    if pretrained is None:
        pretrained=pretrain(seed,spec)
    sender, receiver=pretrained
    n=2*spec.pairs
    ix=np.arange(n)
    peer=ix^1
    # Per cell, 4 PRIVATE observation slots, 3 authored competing hypotheses.
    # H0 unchanged rule, H1 scarce resource for this input, H2 altered rule.
    prior=np.array([.90,.05,.05])
    p=np.broadcast_to(prior,(n,4,3)).copy()
    counts=np.zeros((n,4),dtype=int)
    misses=np.zeros((n,4),dtype=int)
    grown=np.zeros((n,4),dtype=bool)
    transmitted_requests=0
    probing_pair_steps=0
    success=0
    earned=0
    challenge_success=0
    challenge_count=0
    decisions=[]
    correct_by_case=np.zeros(4,dtype=int)
    seen_by_case=np.zeros(4,dtype=int)
    cost=0.
    first_discovery=np.full(spec.pairs, -1, dtype=int)
    q_base=np.array([spec.resource_probability,.012,.012])
    q_invert=np.array([.012,.012,spec.resource_probability])
    for t,bits,u_symbol,u_resource,u_probe in sample_world(seed,spec,scenario):
        own=bits*2-1
        sym_p=sigmoid(sender[:,0]+sender[:,1]*own)
        symbols=(sym_p>=.5).astype(int) # learned, deterministic communication at test
        neighbor=symbols[peer]
        slot=bits*2+neighbor
        basis=(sigmoid(receiver[ix,slot])>=.5).astype(int)
        joint_pattern=bits[::2]*2+bits[1::2]
        target=bits[::2]^bits[1::2]
        changed=(joint_pattern==3) if scenario=="changed_11" else (
            np.ones(spec.pairs,dtype=bool) if scenario=="global_change"
            else np.zeros(spec.pairs,dtype=bool))
        target=np.where(changed,1-target,target)
        if scenario=="scarcity_11":
            resource_limit=np.where(joint_pattern==3,0.,spec.resource_probability)
        elif scenario=="uniform_drought":
            resource_limit=np.full(spec.pairs,spec.drought_probability)
        else:
            resource_limit=np.full(spec.pairs,spec.resource_probability)

        own_p=p[ix,slot,:]
        flip_confident=own_p[:,2]>spec.learned_threshold
        # The *only* memory growth is a private slot-specific correction
        # created by the cell's own observed evidence.
        grown[ix,slot]|=flip_confident
        action_base=(basis^grown[ix,slot].astype(int))
        request=np.zeros(n,dtype=bool)
        uncertain=(own_p[:,0]<.50) & (own_p[:,2]>.025) & (~grown[ix,slot])
        if mode=="random":
            request=uncertain&(u_probe<.10)
        elif mode=="periodic":
            request=uncertain&(counts[ix,slot]%7==0)
        elif mode=="information_gain":
            # Explicit local Bayesian intervention choice. Compare expected
            # reduction of hypothesis entropy vs repeating usual action.
            gain=expected_info(own_p,q_invert)-expected_info(own_p,q_base)
            request=uncertain&(gain>spec.entropy_threshold)
        # Each requesting cell sends an independent 1-bit request, its
        # neighbor interprets request to INVERT baseline policy in unison.
        transmitted_requests+=int(request.sum())
        coordinated=(request[::2]|request[1::2])
        probing_pair_steps+=int(coordinated.sum())
        action=(action_base^np.repeat(coordinated,2).astype(int))
        opened=(action[::2]==target)&(action[1::2]==target)
        pulses=opened&(u_resource<resource_limit)
        rew=np.repeat(pulses.astype(int),2)
        success+=int(opened.sum())
        earned+=int(pulses.sum())
        challenge_count+=int((joint_pattern==3).sum())
        challenge_success+=int(np.sum(opened[joint_pattern==3]))
        seen_by_case+=np.bincount(joint_pattern,minlength=4)
        correct_by_case+=np.bincount(
            joint_pattern,weights=opened.astype(int),minlength=4).astype(int)
        # Normal action samples inferred likelihood "old mapping". If we
        # have already learned to invert, that IS an intervention too.
        action_inverted=(grown[ix,slot]^np.repeat(coordinated,2))
        model=np.where(action_inverted[:,None],q_invert,q_base)
        current=p[ix,slot]
        likelihood=np.where(rew[:,None]==1,model,1-model)
        updated=current*likelihood
        updated/=np.maximum(updated.sum(axis=1,keepdims=True),1e-12)
        p[ix,slot]=updated
        counts[ix,slot]+=1
        misses[ix,slot]+=(1-rew)
        newly=grown[ix,slot]
        newly_pairs=newly[::2]&newly[1::2]
        mark=(first_discovery<0)&newly_pairs
        first_discovery[mark]=t
        cost+=spec.request_cost*request.sum()+spec.intervention_cost*coordinated.sum()
    allocated=int(grown.sum())
    # On changed_11 only, count allocated memory on each cell's challenge slot
    # vs unrelated slots; true new memory means at least one bit.
    return {
        "seed":seed,"mode":mode,"scenario":scenario,
        "pair_success_rate":success/(spec.pairs*spec.horizon),
        "resource_pulse_rate":earned/(spec.pairs*spec.horizon),
        "critical_11_success":challenge_success/max(challenge_count,1),
        "inverted_slot_count":allocated,
        "information_request_bits":transmitted_requests,
        "coordinated_probe_count":probing_pair_steps,
        "synthetic_probe_cost":float(cost),
        "first_pair_discovery_mean":(
            float(np.mean(first_discovery[first_discovery>=0]))
            if np.any(first_discovery>=0) else None),
        "pairs_both_discovered_fraction":float(np.mean(first_discovery>=0)),
        "by_case_correctness": {
            format(i,"02b"):float(correct_by_case[i]/max(seen_by_case[i],1))
            for i in range(4)
        },
        "false_new_memory": allocated if scenario in ("stable","scarcity_11","uniform_drought") else None,
    }


def benchmark(seeds, spec=ProbeSpec(), methods=MODES, cases=SCENARIOS):
    spec.validate()
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s)!=int or s<0 for s in seeds):
        raise ValueError("seeds must be nonempty distinct nonnegative integers")
    if any(m not in MODES for m in methods) or any(c not in SCENARIOS for c in cases):
        raise ValueError("invalid benchmark selection")
    rows=[]
    for seed in seeds:
        model=pretrain(seed,spec)
        for case in cases:
            for mode in methods:
                rows.append(evaluate(seed,mode,case,spec,model))
    aggregate={}
    for case in cases:
        aggregate[case]={}
        for mode in methods:
            group=[r for r in rows if r["scenario"]==case and r["mode"]==mode]
            aggregate[case][mode]={
                field:float(np.mean([r[field] for r in group]))
                for field in ("pair_success_rate","critical_11_success",
                              "inverted_slot_count","information_request_bits",
                              "coordinated_probe_count","synthetic_probe_cost",
                              "pairs_both_discovered_fraction")
            }
    return {
        "schema":"paired-active-causal-probing-v1",
        "date":"2026-10-27",
        "spec":asdict(spec),"seeds":list(seeds),
        "modes":list(methods),"scenarios":list(cases),
        "aggregate":aggregate,"rows":rows,
        "boundaries":[
            "Sender/decoder protocol learned beforehand from local energy; no global critic.",
            "All probes jointly invert both partners' old actions using one-bit requests.",
            "Three hypothesis likelihoods and intervention value are hand-designed.",
            "Challenge rule/resource availability and pair matching are engineered.",
            "No privileged correct answer or availability sensor is delivered to cells.",
            "The test is synthetic and does not measure actual joules or general intelligence.",
            "One bit per cell per signature allocated as learned action correction, not physical growth.",
        ],
    }


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--seeds",default="11,12,13")
    a.add_argument("--pairs",type=int,default=48)
    a.add_argument("--pretrain",type=int,default=4500)
    a.add_argument("--horizon",type=int,default=450)
    a.add_argument("--output",type=Path)
    args=a.parse_args()
    spec=ProbeSpec(pairs=args.pairs,pretrain=args.pretrain,horizon=args.horizon)
    data=benchmark([int(s) for s in args.seeds.split(",") if s.strip()],spec)
    text=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(exist_ok=True,parents=True)
        args.output.write_text(text,encoding="utf8")
        print("SAVED",args.output)
    else:
        print(text)


if __name__=="__main__":
    main()
