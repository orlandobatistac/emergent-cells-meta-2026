"""Fast falsifiable local predictive-model experiment; no rule labels in cells.

Each pair exchanges a fixed binary cue (a noiseless engineered communication
channel, NOT the learned sender from previous experiments). Independent
cells predict their own reward conditional on own cue, received cue, and their
own selected action. World rules are XOR/XNOR/AND; only world uses rule ID.
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Spec:
    pairs:int=32
    phase_steps:int=1200
    test_steps:int=240
    lr:float=.09
    reward_dropout:float=.08
    def validate(self):
        if not 4<=self.pairs<=1024 or not 100<=self.phase_steps<=20000 or not 20<=self.test_steps<=5000:
            raise ValueError("invalid scale")
        if not 0<self.lr<=1 or not 0<=self.reward_dropout<.5:
            raise ValueError("invalid training settings")

RULES=("xor","xnor","and")
def target(bits,rule):
    a,b=bits[::2],bits[1::2]
    if rule=="xor": return a^b
    if rule=="xnor": return 1-(a^b)
    if rule=="and": return a&b
    raise ValueError("unknown world")

class LocalPredictor:
    def __init__(self,spec,seed,learn=True):
        spec.validate()
        self.spec=spec; self.learn=learn; self.n=2*spec.pairs
        self.ids=np.arange(self.n)
        self.peer=self.ids^1
        self.p=np.full((self.n,8),.5)
        self.visits=np.zeros((self.n,8),dtype=int)
        self.global_mean=np.full(self.n,.5)
        self.global_count=np.zeros(self.n,dtype=int)
        self.rng=np.random.default_rng(seed)
    def step(self,bits,action,resource,rule,train):
        # 3 local features: own cue, peer's one-bit message and OWN action.
        peer_message=bits[self.peer]
        slot=bits*4+peer_message*2+action
        predicted=self.p[self.ids,slot].copy()
        good=(action[::2]==target(bits,rule))&(action[1::2]==target(bits,rule))
        pulse=np.repeat((good & resource).astype(float),2)
        # A stronger comparison: each cell learns its overall reward rate
        # WITHOUT features. It is updated on the same observed samples.
        baseline_prediction=self.global_mean.copy()
        if train and self.learn:
            self.p[self.ids,slot]+=self.spec.lr*(pulse-predicted)
            self.visits[self.ids,slot]+=1
            counts=self.global_count+1
            self.global_mean+=(pulse-self.global_mean)/counts
            self.global_count=counts
        return np.abs(predicted-pulse), (predicted-pulse)**2, (baseline_prediction-pulse)**2

def stage(seed,spec,rule,learner,steps,salt,train):
    rng=np.random.default_rng(seed*100003+salt)
    error=[]
    for _ in range(steps):
        bits=rng.integers(0,2,learner.n)
        # Independent actions allow observations of counterfactual branches.
        act=rng.integers(0,2,learner.n)
        available=rng.random(spec.pairs)>=spec.reward_dropout
        _,square,global_square=learner.step(bits,act,available,rule,train)
        error.append((float(square.mean()),float(global_square.mean())))
    return np.asarray(error)

def trial(seed,spec):
    spec.validate()
    learned=LocalPredictor(spec,seed,True)
    static=LocalPredictor(spec,seed,False)
    rows=[]
    for index,rule in enumerate(("xor","xnor","and","xor")):
        training=stage(seed,spec,rule,learned,spec.phase_steps,100+index,True)
        # Frozen control always predicts .5 (baseline MSE .25).
        reference=stage(seed,spec,rule,static,spec.phase_steps,100+index,True)
        # Evaluate future samples without learning, before next shift.
        test=stage(seed,spec,rule,learned,spec.test_steps,900+index,False)
        control=stage(seed,spec,rule,static,spec.test_steps,900+index,False)
        rows.append({
            "rule_world_only":rule,"phase":index,
            "train_first100_mse":float(training[:100,0].mean()),
            "train_last100_mse":float(training[-100:,0].mean()),
            "test_mse":float(test[:,0].mean()),
            "control_test_mse":float(control[:,0].mean()),
            "global_mean_baseline_mse":float(test[:,1].mean()),
            "relative_gain_vs_static":float(1-test[:,0].mean()/control[:,0].mean()),
            "relative_gain_vs_global_mean":float(1-test[:,0].mean()/test[:,1].mean()),
            "visited_cell_slots_fraction":float(np.mean(learned.visits>0)),
        })
    return {"seed":seed,"stages":rows}

def benchmark(seeds,spec=Spec()):
    spec.validate()
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s)!=int or s<0 for s in seeds):
        raise ValueError("invalid seeds")
    rows=[trial(s,spec) for s in seeds]
    aggregate=[]
    for phase in range(4):
        stage_rows=[r["stages"][phase] for r in rows]
        aggregate.append({
            "phase":phase,"rule":stage_rows[0]["rule_world_only"],
            **{key:float(np.mean([r[key] for r in stage_rows])) for key in
               ("train_first100_mse","train_last100_mse","test_mse",
                "control_test_mse","global_mean_baseline_mse","relative_gain_vs_static","relative_gain_vs_global_mean")}
        })
    return {"schema":"local-conditional-reward-predictor-v1",
            "seeds":list(seeds),"spec":asdict(spec),
            "aggregate":aggregate,"rows":rows,
            "limitations":[
                "WORLD ONLY knows XOR/XNOR/AND; cell model sees private cue, noiseless peer bit, own action, own reward.",
                "Peer bit is a perfect preprogrammed cue channel: learned communication is NOT evaluated here.",
                "Memory is overwritten on task changes; no automatic retention or context inference proven.",
                "Fixed 8-slot tabular predictor per cell; not a learned general representation.",
                "This measures forecast MSE vs frozen p=.5 baseline, not AGI or physical joules.",
            ]}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",default="11,12,13")
    p.add_argument("--pairs",type=int,default=32)
    p.add_argument("--phase-steps",type=int,default=1200)
    p.add_argument("--test-steps",type=int,default=240)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    result=benchmark([int(s) for s in a.seeds.split(",") if s.strip()],
                     Spec(pairs=a.pairs,phase_steps=a.phase_steps,test_steps=a.test_steps))
    data=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(data,encoding="utf8")
        print("Saved",a.output)
    else: print(data)
if __name__=="__main__":main()
