"""A diagnostic delayed-cue policy trained end-to-end from sparse reward.

Agent receives an initial one-bit local cue, then visually identical corridor
steps, then selects one of two branches. No position, map, hidden goal or route
enters the policy. This is a tiny memory benchmark, NOT maze navigation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np


class DelayedCuePolicy:
    """Trainable leaky recurrent memory and policy using only terminal reward."""

    def __init__(self, seed=1, units=12, retention=.88, learning_rate=.025):
        rng=np.random.default_rng(seed)
        self.units=units
        self.retention=retention
        self.lr=learning_rate
        self.p={
            "wx":rng.normal(0,.3,(3,units)).astype(np.float32),
            "bias":np.zeros(units,np.float32),
            "actor":rng.normal(0,.2,(units,2)).astype(np.float32),
            "action_bias":np.zeros(2,np.float32)}
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}
        self.n=0

    def inputs(self, cues, length, *, shuffle=False, noise=0., rng=None):
        if length<2:
            raise ValueError("length must be >=2")
        b=len(cues)
        x=np.zeros((b,length,3),np.float32)
        observed=1-cues if shuffle else cues
        x[np.arange(b),0,observed]=1
        x[:,1:,2]=1
        if noise:
            if rng is None:
                raise ValueError("rng required when adding noise")
            x+=rng.normal(0,noise,x.shape).astype(np.float32)
        return x

    def forward(self,x):
        h=np.zeros((len(x),self.units),np.float32)
        states=[h]
        candidate=[]
        for t in range(x.shape[1]):
            z=np.tanh(x[:,t]@self.p["wx"]+self.p["bias"])
            h=self.retention*h+(1-self.retention)*z
            candidate.append(z)
            states.append(h)
        logits=h@self.p["actor"]+self.p["action_bias"]
        probability=np.exp(logits-logits.max(axis=1,keepdims=True))
        probability/=probability.sum(axis=1,keepdims=True)
        return probability,states,candidate

    def step(self,cues,length,rng):
        x=self.inputs(cues,length)
        probs,states,derived=self.forward(x)
        # Policy samples an action and receives ONLY terminal scalar reward.
        sampled=(rng.random(len(cues))>probs[:,0]).astype(np.int64)
        reward=np.where(sampled==cues,1.,-1.).astype(np.float32)
        advantage=reward-reward.mean()
        logits_delta=-probs
        logits_delta[np.arange(len(cues)),sampled]+=1
        logits_delta*=advantage[:,None]
        grad={k:np.zeros_like(v) for k,v in self.p.items()}
        n=max(len(cues),1)
        grad["actor"]=states[-1].T@logits_delta/n
        grad["action_bias"]=logits_delta.mean(axis=0)
        dh=logits_delta@self.p["actor"].T
        for t in range(x.shape[1]-1,-1,-1):
            dz=dh*(1-self.retention)*(1-derived[t]**2)
            grad["wx"]+=x[:,t].T@dz/n
            grad["bias"]+=dz.mean(axis=0)
            dh=dh*self.retention
        norm=np.sqrt(sum((g*g).sum() for g in grad.values()))
        if norm>2:
            for k in grad:grad[k]*=2/norm
        self.n+=1
        for k in self.p:
            self.m[k]=.9*self.m[k]+.1*grad[k]
            self.v[k]=.999*self.v[k]+.001*grad[k]*grad[k]
            self.p[k]+=self.lr*(self.m[k]/(1-.9**self.n))/(
                np.sqrt(self.v[k]/(1-.999**self.n))+1.e-8)
        return float(np.mean(reward>0))

    def fit(self, *, seed, batches=280, batch=64, length=8):
        rng=np.random.default_rng(seed)
        scores=[]
        for i in range(batches):
            cues=rng.integers(0,2,size=batch)
            score=self.step(cues,length,rng)
            scores.append(score)
        return {"initial_25_batches":float(np.mean(scores[:25])),
                "final_25_batches":float(np.mean(scores[-25:])),
                "training_reward_updates":batches,
                "train_episodes":batches*batch}

    def evaluate(self, *, seed, length, cases=500, erased=False,
                 inverted_cue=False, noise=0.):
        rng=np.random.default_rng(seed)
        # Balanced targets and a fresh independent test seed.
        cues=rng.integers(0,2,size=cases)
        x=self.inputs(cues,length,shuffle=inverted_cue,noise=noise,rng=rng)
        probs,states,_=self.forward(x)
        if erased:
            # Same learned weights and final current observation: no cue state.
            blank=np.zeros((cases,self.units),np.float32)
            last=np.tanh(x[:,-1]@self.p["wx"]+self.p["bias"])
            blank=self.retention*blank+(1-self.retention)*last
            logits=blank@self.p["actor"]+self.p["action_bias"]
            probs=np.exp(logits-logits.max(axis=1,keepdims=True))
            probs/=probs.sum(axis=1,keepdims=True)
        chosen=np.argmax(probs,axis=1)
        return {"accuracy":float(np.mean(chosen==cues)),
                "n_cases":cases,"length":length}

    def parameter_bytes(self):
        return int(sum(p.nbytes for p in self.p.values()))


def benchmark(seeds, *, batches=280, batch=64, units=12,
              training_length=8, eval_cases=500):
    if not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError("nonempty unique seeds required")
    rows=[]
    start=time.process_time()
    for seed in seeds:
        model=DelayedCuePolicy(seed=seed,units=units)
        training=model.fit(seed=seed+50009,batches=batches,batch=batch,
                           length=training_length)
        results={}
        for length in (training_length,training_length+4,training_length+8):
            for condition in ("memory","erased","inverted_cue","noisy"):
                results[f"{length}_{condition}"]=model.evaluate(
                    seed=seed*2003+length,cases=eval_cases,length=length,
                    erased=(condition=="erased"),
                    inverted_cue=(condition=="inverted_cue"),
                    noise=.05 if condition=="noisy" else 0.)
        rows.append({"seed":seed,"train":training,"results":results})
    keys=rows[0]["results"]
    summary={key:sum(row["results"][key]["accuracy"] for row in rows)/len(rows)
             for key in keys}
    return {"protocol":{
        "task":"short cue then blank corridor then final binary branch",
        "oracle_observation":False,
        "agent_input":"initial binary local cue followed by blank/noisy readings",
        "training_signal":"terminal +1 correct or -1 incorrect, sampled action only",
        "policy_train":"REINFORCE through trainable recurrent encoder",
        "fixed_retention":.88,
        "units":units,
        "training_length":training_length,
        "batches":batches,"batch_size":batch,
        "training_episodes_per_seed":batches*batch,
        "eval_cases_per_condition":eval_cases,
        "seeds":seeds,
        "parameter_bytes":DelayedCuePolicy(units=units).parameter_bytes(),
        "total_process_cpu_seconds":time.process_time()-start},
        "summary":summary,"rows":rows}


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--seeds",default="11,12,13")
    a.add_argument("--batches",type=int,default=280)
    a.add_argument("--batch",type=int,default=64)
    a.add_argument("--units",type=int,default=12)
    a.add_argument("--length",type=int,default=8)
    a.add_argument("--cases",type=int,default=500)
    a.add_argument("--output",default="")
    p=a.parse_args()
    if p.batches<1 or p.batch<2 or p.cases<1: a.error("positive counts required")
    t=time.perf_counter()
    r=benchmark([int(x) for x in p.seeds.split(",")],
                batches=p.batches,batch=p.batch,units=p.units,
                training_length=p.length,eval_cases=p.cases)
    r["wall_seconds"]=time.perf_counter()-t
    if p.output:
        dst=Path(p.output);dst.parent.mkdir(exist_ok=True,parents=True)
        dst.write_text(json.dumps(r,indent=2),encoding="utf-8")
    print(json.dumps({"summary":r["summary"],
                      "initial_training":sum(x["train"]["initial_25_batches"] for x in r["rows"])/len(r["rows"]),
                      "late_training":sum(x["train"]["final_25_batches"] for x in r["rows"])/len(r["rows"]),
                      "wall_seconds":r["wall_seconds"],
                      "output":p.output},indent=2))


if __name__=="__main__":
    main()
