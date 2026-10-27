"""Cell-private tabular probe policies learned from real local net reward.

A fixed Bayesian *belief representation* and four-bit corrected-action slots
are retained from the previous counterfactual-probing study. The ACTION POLICY
probe / exploit is newly learned, individually per cell, from experience;
there is no privileged world label, target or resource sensor in agent updates.

Crucially, this is not an autonomous invention of curiosity, hypotheses,
signal representations or topology. Exploration choices are learned within
a manually structured POMDP and a designed counterfactual action primitive.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
from .active_probes import ProbeSpec, pretrain, sigmoid, sample_world, evaluate as fixed_evaluate

STATES = ("normal", "weak_uncertain", "strong_uncertain", "likely_scarce", "likely_flip", "recent_failed_probes")
SCENARIOS = ("stable", "changed_11", "scarcity_11", "uniform_drought", "global_change",
             "changed_01", "scarcity_10")


@dataclass(frozen=True)
class CuriositySpec:
    pairs: int=48
    pretrain: int=3600
    meta_episodes: int=320
    meta_horizon: int=75
    eval_horizon: int=500
    alpha: float=.06
    initial_epsilon: float=.18
    final_epsilon: float=.04
    probe_cost: float=.025
    message_cost: float=.004
    pulse_rate: float=.92
    intrinsic_weight: float=.25

    def validate(self):
        if not 4<=self.pairs<=2000 or not 100<=self.pretrain<=100000:
            raise ValueError("invalid population/pretraining")
        if not 20<=self.meta_episodes<=5000 or not 20<=self.meta_horizon<=500:
            raise ValueError("invalid learning duration")
        if not 100<=self.eval_horizon<=10000:
            raise ValueError("invalid evaluation length")
        if not 0<self.alpha<=.5 or not 0<=self.final_epsilon<=self.initial_epsilon<=.9:
            raise ValueError("invalid learning settings")
        if not 0<=self.intrinsic_weight<=2:
            raise ValueError('invalid curiosity weight')
        if not 0<=self.probe_cost<=.2 or not 0<=self.message_cost<=.1 or not .3<=self.pulse_rate<1:
            raise ValueError("invalid rewards")


def local_state(belief):
    """Hand-designed coarse state from local evidence; NOT task label."""
    normal=belief[:,0]>=.72
    scarcity=belief[:,1]>=.80
    flipped=belief[:,2]>=.80
    weak=belief[:,0]>=.30
    return np.where(normal,0,np.where(scarcity,3,np.where(flipped,4,
                    np.where(weak,1,2)))).astype(np.int64)


def update_belief(p, observed, inverted, pulse_rate=.92):
    """Likelihood update using private received reward, no hidden world label."""
    base=np.array([pulse_rate,.012,.012])
    flip=np.array([.012,.012,pulse_rate])
    likelihood=np.where(inverted[:,None],flip[None,:],base[None,:])
    likelihood=np.where(observed[:,None],likelihood,1-likelihood)
    result=p*likelihood
    return result/np.maximum(result.sum(axis=1,keepdims=True),1e-12)


class LocalCuriosity:
    def __init__(self, spec: CuriositySpec, seed: int):
        spec.validate()
        self.spec=spec
        self.n=2*spec.pairs
        self.ids=np.arange(self.n)
        self.peer=self.ids^1
        # Every cell has its own private Q value per observed state/action.
        self.q=np.zeros((self.n,len(STATES),2),dtype=float)
        self.q[:,:,0]=.25
        self.q[:,:,1]=.25
        self.seed=seed
        self.reset_environment()

    def reset_environment(self):
        self.belief=np.broadcast_to(np.array([.90,.05,.05]),
                                    (self.n,4,3)).copy()
        self.corrected=np.zeros((self.n,4),dtype=bool)
        self.failed_probes=np.zeros((self.n,4),dtype=np.int16)

    def get_requests(self, local_state_ids, rng, train=False, epsilon=0):
        # No hand-coded exploration rule. Q is a private learned value table.
        greedy=self.q[self.ids,local_state_ids,1]>self.q[self.ids,local_state_ids,0]
        ties=np.abs(self.q[self.ids,local_state_ids,1]-
                    self.q[self.ids,local_state_ids,0])<1e-9
        # A neutral coin toss on untrained ties, explicitly accounted for.
        greedy=np.where(ties,rng.random(self.n)<.5,greedy)
        if train:
            greedy=np.where(rng.random(self.n)<epsilon,
                            rng.random(self.n)<.5,greedy)
        # A known-corrected slot no longer needs experimental probes. This
        # fixed suppression is part of prior experiment's memory controller.
        return greedy

    def run_episode(self, sender, decoder, scenarios, rng, *,
                    steps, train, epsilon, diagnostic=False):
        self.reset_environment()
        if len(scenarios)!=self.spec.pairs or any(s not in SCENARIOS for s in scenarios):
            raise ValueError("invalid private environment assignment")
        scenario=np.asarray(scenarios)
        critical=np.array([3 if s in ("changed_11","scarcity_11") else
                           1 if s=="changed_01" else
                           2 if s=="scarcity_10" else -1 for s in scenario])
        n=self.n
        earned=0.
        correct=0.
        cost=0.
        messages=0
        probes=0
        crit_hit=0
        crit_total=0
        for _ in range(steps):
            bits=rng.integers(0,2,n)
            pair_cues=bits[::2]*2+bits[1::2]
            s_p=sigmoid(sender[:,0]+sender[:,1]*(bits*2-1))
            msg=(s_p>=.5).astype(int)
            slot=bits*2+msg[self.peer]
            learned_action=(sigmoid(decoder[self.ids,slot])>=.5).astype(int)
            own_belief=self.belief[self.ids,slot]
            state=local_state(own_belief)
            # Private, locally observed history: two probe failures in the
            # same cue/message slot may justify declining further tests.
            failed=self.failed_probes[self.ids,slot]
            state=np.where((failed>=2) & np.isin(state,[1,2]),5,state)
            request=self.get_requests(state,rng,train,epsilon)
            # No reason to explore once existing cell-local memory has fully
            # learned the flipped action, same rule in all methods.
            request&=~self.corrected[self.ids,slot]
            actual_probe=request[::2]|request[1::2]
            active=request|request[self.peer]
            inverted=self.corrected[self.ids,slot]^active
            act=learned_action ^ inverted.astype(int)
            target=bits[::2]^bits[1::2]
            flipped=((scenario=="global_change") |
                     ((scenario=="changed_11") & (pair_cues==3)) |
                     ((scenario=="changed_01") & (pair_cues==1)))
            target ^= flipped.astype(int)
            available=np.where(scenario=="uniform_drought",.55,self.spec.pulse_rate)
            available=np.where(
                ((scenario=="scarcity_11")&(pair_cues==3)) |
                ((scenario=="scarcity_10")&(pair_cues==2)),0.,available)
            opened=(act[::2]==target)&(act[1::2]==target)
            reward=opened & (rng.random(self.spec.pairs)<available)
            own_reward=np.repeat(reward.astype(float),2)
            previously_failed=self.failed_probes[self.ids,slot]
            previously_failed=np.where(active & (own_reward==0),
                np.minimum(10,previously_failed+1),previously_failed)
            previously_failed=np.where(active & (own_reward>0),0,previously_failed)
            self.failed_probes[self.ids,slot]=previously_failed
            net=own_reward - self.spec.probe_cost*active - self.spec.message_cost*request
            likelihood=update_belief(own_belief,own_reward.astype(int),inverted,self.spec.pulse_rate)
            # Learn from private request/reward trials whose physical outcome
            # was not caused by an independent request from the peer.
            if train:
                independent=~request[self.peer]
                chosen=request.astype(int)
                old=self.q[self.ids,state,chosen]
                h_before=-np.sum(own_belief*np.log2(np.maximum(own_belief,1e-12)),axis=1)
                h_after=-np.sum(likelihood*np.log2(np.maximum(likelihood,1e-12)),axis=1)
                progress=np.maximum(0.,h_before-h_after)
                intrinsic=self.spec.intrinsic_weight*progress*active
                delta=self.spec.alpha*(net+intrinsic-old)
                self.q[self.ids[independent],state[independent],
                       chosen[independent]] += delta[independent]
                np.clip(self.q,-1.0,1.0,out=self.q)
            self.belief[self.ids,slot]=likelihood
            self.corrected[self.ids,slot]|=(likelihood[:,2]>.96)
            messages+=int(request.sum())
            probes+=int(actual_probe.sum())
            earned+=float(reward.sum())
            correct+=float(opened.sum())
            cost+=float((self.spec.probe_cost*actual_probe.sum()+
                         self.spec.message_cost*request.sum()))
            if diagnostic:
                c=(pair_cues==critical)
                crit_hit+=int((opened&c).sum())
                crit_total+=int(c.sum())
        return {
            "correct":correct/(steps*self.spec.pairs),
            "earned":earned/(steps*self.spec.pairs),
            "requests":messages,
            "probes":probes,
            "probe_cost":cost,
            "critical":crit_hit/max(1,crit_total),
            "false_corrections":int(self.corrected.sum()),
        }


def train_policy(seed,spec):
    spec.validate()
    base=ProbeSpec(pairs=spec.pairs,pretrain=spec.pretrain,
                   horizon=spec.eval_horizon,
                   intervention_cost=spec.probe_cost,
                   request_cost=spec.message_cost,
                   resource_probability=spec.pulse_rate)
    sender,decoder=pretrain(seed,base)
    learner=LocalCuriosity(spec,seed)
    rng=np.random.default_rng(seed+3141590)
    categories=np.array(["stable","changed_11","scarcity_11",
                         "changed_01","scarcity_10","global_change","uniform_drought"])
    freq=np.array([.16,.18,.18,.14,.14,.10,.10])
    stats=[]
    for ep in range(spec.meta_episodes):
        scenarios=rng.choice(categories,size=spec.pairs,p=freq)
        eps=spec.initial_epsilon+(spec.final_epsilon-spec.initial_epsilon)*ep/max(1,spec.meta_episodes-1)
        r=learner.run_episode(sender,decoder,scenarios,rng,
                              steps=spec.meta_horizon,train=True,
                              epsilon=eps)
        if ep%max(1,spec.meta_episodes//8)==0:
            stats.append({"episode":ep,"reward_pulses":r["earned"],
                          "probe_cost":r["probe_cost"]})
    return learner,sender,decoder,stats


def evaluate_policy(seed, scenario, spec, trained=None, untrained=False):
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    if trained is None:
        trained=train_policy(seed,spec)
    agent,send,recv,_=trained
    eval_agent=LocalCuriosity(spec,seed+17)
    if not untrained:
        eval_agent.q=agent.q.copy()
    rng=np.random.default_rng(seed+79813+SCENARIOS.index(scenario)*100003)
    result=eval_agent.run_episode(
        send,recv,[scenario]*spec.pairs,rng,steps=spec.eval_horizon,
        train=False,epsilon=0,diagnostic=True)
    return {**result,"seed":seed,"scenario":scenario,
            "method":("untrained_probe_policy" if untrained
                      else "locally_learned_probe")}


def benchmark(seeds,spec=CuriositySpec()):
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("seeds must be unique")
    specs=ProbeSpec(pairs=spec.pairs,pretrain=spec.pretrain,
                    horizon=spec.eval_horizon,
                    resource_probability=spec.pulse_rate,
                    intervention_cost=spec.probe_cost,
                    request_cost=spec.message_cost)
    rows=[]
    training=[]
    for seed in seeds:
        trained=train_policy(seed,spec)
        training.append({"seed":seed,"q_per_cell_mean":trained[0].q.mean(axis=0).tolist(),
                         "training_log":trained[3]})
        for case in SCENARIOS[:5]:
            rows.append(evaluate_policy(seed,case,spec,trained))
            rows.append(evaluate_policy(seed,case,spec,trained,untrained=True))
            for old_mode in ("passive","random","periodic","information_gain"):
                o=fixed_evaluate(seed,old_mode,case,specs,
                                 (trained[1],trained[2]))
                rows.append({"seed":seed,"scenario":case,"method":old_mode,
                             "correct":o["pair_success_rate"],
                             "critical":o["critical_11_success"],
                             "requests":o["information_request_bits"],
                             "probe_cost":o["synthetic_probe_cost"],
                             "false_corrections":o["inverted_slot_count"]})
        for case in SCENARIOS[5:]:
            rows.append(evaluate_policy(seed,case,spec,trained))
            rows.append(evaluate_policy(seed,case,spec,trained,untrained=True))
    aggregates={}
    for case in SCENARIOS:
        aggregates[case]={}
        for method in sorted({r["method"] for r in rows if r["scenario"]==case}):
            rr=[r for r in rows if r["scenario"]==case and r["method"]==method]
            aggregates[case][method]={
                k:float(np.mean([r[k] for r in rr]))
                for k in ("correct","critical","requests",
                          "probe_cost","false_corrections")}
    return {
        "schema":"independently-learned-local-curiosity-v1",
        "date":"2026-10-27",
        "seeds":seeds,
        "spec":asdict(spec),
        "aggregate":aggregates,
        "rows":rows,
        "learned_local_values":training,
        "limitations":[
            "Probe-vs-act policy learned from each cell's own realized net reward.",
            "Four-slot Bayesian hypotheses and local correction flags remain engineered.",
            "Partner coordination, two actions, zero-noise message channel fixed.",
            "Meta training uses hand-designed mixture of XOR-like synthetic worlds.",
            "No learned intrinsic predictive model, AGI, energy joules or arbitrary agents.",
            "Score includes simulated probe/message costs but not compute/memory cost."
        ]}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--seeds",default="11,12,13")
    a.add_argument("--pairs",type=int,default=48)
    a.add_argument("--meta-episodes",type=int,default=320)
    a.add_argument("--meta-horizon",type=int,default=75)
    a.add_argument("--pretrain",type=int,default=3600)
    a.add_argument("--eval-horizon",type=int,default=500)
    a.add_argument("--output",type=Path)
    z=a.parse_args()
    spec=CuriositySpec(pairs=z.pairs,meta_episodes=z.meta_episodes,
                       meta_horizon=z.meta_horizon,pretrain=z.pretrain,
                       eval_horizon=z.eval_horizon)
    data=benchmark([int(x) for x in z.seeds.split(",") if x.strip()],spec)
    out=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if z.output:
        z.output.parent.mkdir(parents=True,exist_ok=True)
        z.output.write_text(out,encoding="utf8")
        print("SAVED",z.output)
    else:
        print(out)

if __name__=="__main__":
    main()
