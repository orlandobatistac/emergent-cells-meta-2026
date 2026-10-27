"""Paired same-seed ablation of intrinsic learning-progress reward.

No training or evaluation rule is changed between the two conditions except
the intrinsic weight. Private cell learning and world seeds remain identical.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
from emergent_cells.experiments.learned_curiosity import (
    CuriositySpec, train_policy, evaluate_policy,
)
CASES=("changed_11","scarcity_11","stable","uniform_drought","global_change")
def benchmark(seeds, pairs=48, pretrain=3600, meta_episodes=320,
              meta_horizon=75, eval_horizon=500):
    rows=[]
    for seed in seeds:
        for weight in (0.,.25):
            s=CuriositySpec(pairs=pairs, pretrain=pretrain,
                            meta_episodes=meta_episodes,
                            meta_horizon=meta_horizon,
                            eval_horizon=eval_horizon,intrinsic_weight=weight)
            trained=train_policy(seed,s)
            for case in CASES:
                row=evaluate_policy(seed,case,s,trained)
                row["intrinsic_weight"]=weight
                rows.append(row)
    summary={}
    for c in CASES:
        summary[c]={}
        for w in (0.,.25):
            a=[r for r in rows if r["scenario"]==c and r["intrinsic_weight"]==w]
            summary[c][str(w)]={field:float(np.mean([r[field] for r in a]))
                for field in ("correct","critical","requests","probe_cost",
                              "false_corrections")}
    return {"schema":"local-curiosity-intrinsic-vs-plain-paired-v1",
            "seeds":seeds,"cases":list(CASES),"rows":rows,"summary":summary,
            "limitations":[
                "Both controllers have engineered Bayesian belief representations.",
                "Only same-seed training and scenario distributions are matched.",
                "No physically measured energy; costs are synthetic.",
                "The intrinsic formula was chosen after development pilots."
            ]}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",default="2001,2002,2003,2004,2005")
    p.add_argument("--pairs",type=int,default=48)
    p.add_argument("--pretrain",type=int,default=3600)
    p.add_argument("--meta-episodes",type=int,default=320)
    p.add_argument("--meta-horizon",type=int,default=75)
    p.add_argument("--eval-horizon",type=int,default=500)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    data=benchmark([int(s) for s in a.seeds.split(",") if s.strip()],
                   a.pairs,a.pretrain,a.meta_episodes,a.meta_horizon,a.eval_horizon)
    out=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(out,encoding="utf8")
        print("SAVED",a.output)
    else:
        print(out)
if __name__=="__main__":
    main()
