"""World-model v0: each cell learns reward predictions for own local cue,
received learned symbol, and two coordinated action choices.

This is an *empirical model* (beta-Bernoulli conditional reward predictions),
not a general-purpose neural world model. It contains NO hand-coded
"mapping changed" vs "resource scarcity" hypotheses. Probe selection
heuristics are still authored; model predictions are learned per cell.
"""
from __future__ import annotations
import argparse, json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from .active_probes import ProbeSpec, pretrain, sigmoid

MODES=("passive","random","predictive_greedy","predictive_ucb","predictive_progress")
CASES=("stable","changed_11","scarcity_11","changed_01","scarcity_10",
       "global_change","uniform_drought")


@dataclass(frozen=True)
class ModelSpec:
    pairs: int=48
    sender_training: int=3600
    steps: int=550
    pulse_probability: float=.92
    drought_probability: float=.55
    probe_cost: float=.025
    message_cost: float=.004
    random_probe: float=.08
    uncertainty_scale: float=.5

    def validate(self):
        if not 4<=self.pairs<=2048 or not 100<=self.sender_training<=100000 or not 100<=self.steps<=10000:
            raise ValueError("invalid study size")
        if not .2<self.pulse_probability<1 or not .05<self.drought_probability<1:
            raise ValueError("invalid probability")
        if not 0<=self.random_probe<=.5 or not 0<=self.uncertainty_scale<=4:
            raise ValueError("invalid exploration")
        if self.probe_cost<0 or self.message_cost<0:
            raise ValueError("invalid cost")


class PrivatePredictors:
    def __init__(self,n):
        # Weak structural prior from previously learned gate baseline;
        # NOT a world-rule label. Each cell owns 4Ã—2 independent models.
        self.hits=np.zeros((n,4,2),dtype=float)
        self.visits=np.zeros((n,4,2),dtype=float)
        self.alpha=np.ones((n,4,2))
        self.beta=np.ones((n,4,2))
        self.alpha[:,:,0]=5.0
        self.beta[:,:,1]=3.0
        self.errors=[]
    def prediction(self):
        return self.alpha/(self.alpha+self.beta)
    def policy(self,idx,slot,mode,rng,random_probability=.08,uncertainty_scale=.5):
        normal=self.prediction()[idx,slot,0]
        alternate=self.prediction()[idx,slot,1]
        vn=self.visits[idx,slot,0]
        va=self.visits[idx,slot,1]
        if mode=="passive":
            return np.zeros(idx.size,dtype=bool)
        if mode=="random":
            return rng.random(idx.size)<random_probability
        if mode=="predictive_greedy":
            # Simple empirical model decision with an initial 1/12
            # randomized experiment opportunity, not a rule-specific cue.
            return (alternate-normal>.12) | ((va==0)&(rng.random(idx.size)<.12))
        # Both active learned-prediction options require evidence and
        # charge for probe actions. Stop trying unrewarded alternatives
        # once enough empirical observations show low predicted return.
        initial=(va<3)
        if mode=="predictive_ucb":
            ucb=alternate+2.3*uncertainty_scale/np.sqrt(va+1)
            return (ucb-normal>.10) & (
                (vn>=2)|(rng.random(idx.size)<.2))
        if mode=="predictive_progress":
            # Error reduction matters: avoid noisy situations whose
            # observed reward is low for BOTH actions.
            uncertainty=np.sqrt(alternate*(1-alternate)/(va+2))
            score=alternate+1.4*uncertainty_scale*uncertainty-normal
            no_future_gain=(va>=5)&(alternate<.15)&(normal<.15)
            return ((score>.05)|(initial&(rng.random(idx.size)<.16))) & ~no_future_gain
        raise ValueError("unknown mode")
    def update(self,ids,slots,performed_action,reward):
        p=self.prediction()[ids,slots,performed_action]
        self.errors.append(float(np.mean((reward-p)**2)))
        np.add.at(self.visits,(ids,slots,performed_action),1)
        np.add.at(self.hits,(ids,slots,performed_action),reward)
        np.add.at(self.alpha,(ids,slots,performed_action),reward)
        np.add.at(self.beta,(ids,slots,performed_action),1-reward)


def run(seed,scenario,mode,spec=ModelSpec(),encoder=None):
    spec.validate()
    if scenario not in CASES or mode not in MODES:
        raise ValueError("unsupported condition")
    if encoder is None:
        encoder=pretrain(seed,ProbeSpec(pairs=spec.pairs,pretrain=spec.sender_training))
    sender,decoder=encoder
    n=spec.pairs*2
    idx=np.arange(n)
    peer=idx^1
    model=PrivatePredictors(n)
    rng=np.random.default_rng(seed*1003+CASES.index(scenario)*173+81001)
    correct=0
    pulses=0
    requests=0
    probes=0
    experimental_probes=0
    committed_alternate_steps=0
    challenge_correct=0
    challenge_total=0
    active_memories=np.zeros((n,4),dtype=bool)
    first100=[]
    last100=[]
    regret_cost=0.
    for t in range(spec.steps):
        bits=rng.integers(0,2,n)
        joint_slot=2*bits[::2]+bits[1::2]
        symbol=(sigmoid(sender[:,0]+sender[:,1]*(2*bits-1))>=.5).astype(int)
        local_slot=2*bits+symbol[peer]
        action0=(sigmoid(decoder[idx,local_slot])>=.5).astype(int)
        request=model.policy(
            idx,local_slot,mode,rng,spec.random_probe,spec.uncertainty_scale)
        estimates=model.prediction()[idx,local_slot,:]
        confident_alternate=(model.visits[idx,local_slot,1]>=4)&(
            estimates[:,1]>estimates[:,0]+.18)
        # Once the model has LEARNED the alternative has higher expected
        # return, it is a normal exploitation action, not an experiment.
        if mode.startswith("predictive_"):
            request|=confident_alternate
        exploratory=request & ~confident_alternate
        pair_probe=request[::2]|request[1::2]
        pair_experiment=exploratory[::2]|exploratory[1::2]
        performed=np.repeat(pair_probe,2).astype(int)
        choice=action0^performed
        target=bits[::2]^bits[1::2]
        changed=np.zeros(spec.pairs,dtype=bool)
        if scenario=="changed_11":changed=joint_slot==3
        elif scenario=="changed_01":changed=joint_slot==1
        elif scenario=="global_change":changed=np.ones(spec.pairs,dtype=bool)
        target=target^changed.astype(int)
        available=np.full(spec.pairs,spec.pulse_probability,dtype=float)
        if scenario=="scarcity_11":available[joint_slot==3]=0
        elif scenario=="scarcity_10":available[joint_slot==2]=0
        elif scenario=="uniform_drought":available[:]=spec.drought_probability
        opened=(choice[::2]==target)&(choice[1::2]==target)
        obtained=opened&(rng.random(spec.pairs)<available)
        own_pulse=np.repeat(obtained.astype(int),2)
        model.update(idx,local_slot,performed,own_pulse)
        # learned prediction for alt exceeding normal by margin -> local
        # correction suggestion; counts only after a few successful
        # alternative observations. No correct label or task id.
        prediction=model.prediction()[idx,local_slot,:]
        seen_alt=model.visits[idx,local_slot,1]
        active_memories[idx,local_slot] |= (seen_alt>=3)&(
            prediction[:,1] > prediction[:,0]+.25)
        requests+=int(request.sum())
        probes+=int(pair_probe.sum())
        experimental_probes+=int(pair_experiment.sum())
        committed_alternate_steps+=int((pair_probe & ~pair_experiment).sum())
        correct+=int(opened.sum())
        pulses+=int(obtained.sum())
        relevant=(joint_slot==3)
        challenge_total+=int(relevant.sum())
        challenge_correct+=int((opened&relevant).sum())
        regret_cost+=spec.probe_cost*int(pair_experiment.sum())+spec.message_cost*int(request.sum())
        if t<100:first100.append(float(opened.mean()))
        if t>=spec.steps-100:last100.append(float(opened.mean()))
    return {
        "seed":seed,"scenario":scenario,"mode":mode,
        "correct":correct/(spec.steps*spec.pairs),
        "last100_correct":float(np.mean(last100)),
        "first100_correct":float(np.mean(first100)),
        "case11_correct":challenge_correct/max(challenge_total,1),
        "received_resource_rate":pulses/(spec.steps*spec.pairs),
        "requests":requests,"pair_probes":probes,
        "experimental_probe_count":experimental_probes,
        "committed_alternate_pair_steps":committed_alternate_steps,
        "synthetic_probe_cost":regret_cost,
        "candidate_correction_bits":int(active_memories.sum()),
        "reward_prediction_brier":float(np.mean(model.errors)),
        "local_model_weights_per_cell":8*2,
    }


def benchmark(seeds,spec=ModelSpec(),modes=MODES,cases=CASES):
    spec.validate()
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError("distinct seeds required")
    rows=[]
    for seed in seeds:
        # Learned sender / decoder shared *without retraining* per condition.
        sender,decoder=pretrain(seed,ProbeSpec(pairs=spec.pairs,pretrain=spec.sender_training))
        for scenario in cases:
            for mode in modes:
                rows.append(run(seed,scenario,mode,spec,(sender,decoder)))
    summary={}
    for case in cases:
        summary[case]={}
        for mode in modes:
            grouped=[r for r in rows if r["scenario"]==case and r["mode"]==mode]
            summary[case][mode]={
                key:float(np.mean([r[key] for r in grouped]))
                for key in ("correct","last100_correct","case11_correct",
                            "requests","pair_probes","experimental_probe_count",
                            "committed_alternate_pair_steps","synthetic_probe_cost",
                            "candidate_correction_bits","reward_prediction_brier")
            }
    return {
        "schema":"local-empirical-world-model-v0","date":"2026-10-28",
        "seeds":list(seeds),"spec":asdict(spec),"summary":summary,"rows":rows,
        "limitations":[
            "Predictor learns Bernoulli reward rates for local cue/message and joint probe action.",
            "No three pre-written task-cause hypotheses in the predictor.",
            "The model class, priors and probe policies are authored, not discovered by cells.",
            "A pair's shared gate reward can misattribute outcomes to either cell.",
            "Corrected-action memory flags are analysis only; this run does not apply them to future actions.",
            "Fixed pair topology and single learned binary communication code.",
            "Cost proxy excludes CPU and memory energy; not measured joules.",
            "No claim of general intelligence or unseen task-family generalization."
        ]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",default="11,12")
    p.add_argument("--pairs",type=int,default=32)
    p.add_argument("--steps",type=int,default=450)
    p.add_argument("--sender-training",type=int,default=3000)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    s=ModelSpec(pairs=a.pairs,steps=a.steps,sender_training=a.sender_training)
    result=benchmark([int(t) for t in a.seeds.split(",") if t.strip()],s)
    txt=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if a.output:
        a.output.parent.mkdir(exist_ok=True,parents=True)
        a.output.write_text(txt,encoding="utf8")
        print("SAVED",a.output)
    else:print(txt)
if __name__=="__main__":main()
