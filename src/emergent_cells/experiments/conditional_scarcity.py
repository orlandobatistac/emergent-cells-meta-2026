"""Adversarial stationary-resource scarcity conditioned on private gate inputs.

Task NEVER changes: XOR throughout. The hidden availability of a resource
changes only for specific world input pairs, producing a pattern of reward
failures identical to some genuinely altered gate mappings.

This exposes an *identifiability* limit of local reward-only novelty detectors.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .dynamic_memory import GrowthSpec, rng_stream
from .robust_drift import GuardedCells

def trial(seed, method, spec=GrowthSpec(pairs=24, phase_steps=3500),
          stress_steps=1000):
    pop=GuardedCells(spec,seed,method)
    for inputs in rng_stream(seed,spec.phase_steps,pop.n,"A"):
        pop.episode(inputs,"xor",train=True,detect=True,train_sender=True)
    before=pop.allocated.copy()
    events=pop.change_events
    outcomes=[]
    for bits, sender_u, action_u, reward_u in rng_stream(
            seed,stress_steps,pop.n,"stationary"):
        # World availability hazard: 11 pairs lose ALL resource pulses,
        # while other pairs retain baseline 8% random pulse loss.
        withheld=(bits[::2] == 1) & (bits[1::2] == 1)
        available=reward_u.copy()
        available[withheld]=0.
        accuracy,_=pop.episode((bits,sender_u,action_u,available),"xor",
                               train=False,detect=True,train_sender=False)
        outcomes.append(accuracy)
    return {"seed":seed,"method":method,
            "novelty_without_task_change":int(pop.change_events-events),
            "extra_memory_cells":int((pop.allocated > before).sum()),
            "late_gate_correctness":float(sum(outcomes[-200:])/200),
            "experiment":"same XOR rule, 11-pattern-specific resource dropout"}

def benchmark(seeds):
    rows=[trial(seed,method) for seed in seeds
          for method in ("relative_ema","conditional_signature","availability_sensor")]
    return {"schema":"conditional-dropout-identifiability-v1",
            "seeds":seeds,"rows":rows,
            "limitations":[
                "The world resource process explicitly depends on the private cue combination.",
                "The task rule remains XOR throughout.",
                "Passive reward histories alone need not identify the cause of the same observed failures; independent sensors or active interventions may resolve some cases.",
            ]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",default="11,12,13")
    p.add_argument("--output",type=Path)
    x=p.parse_args()
    result=benchmark([int(s) for s in x.seeds.split(",") if s.strip()])
    body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if x.output:
        x.output.parent.mkdir(parents=True,exist_ok=True)
        x.output.write_text(body,encoding="utf8")
        print("SAVED",x.output)
    else:
        print(body)

if __name__=="__main__":
    main()
