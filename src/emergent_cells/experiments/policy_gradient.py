"""R1 direct policy-gradient decisions from local sensory recurrent state.

No position, map, target coordinates or BFS enters the actor. The actor learns
action logits from goal rewards and intrinsic revisitation penalties. An
engineered local novelty prior is disclosed and shared across policy controls.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import numpy as np
from .predictive_context import traces, signature, ALPHABET
from .recurrent_state import RecurrentPredictor
from .latent_navigation import fit_prototypes, run_episode
from .sensor_world import SensorEpisode

FEATURES = 32 + 4 + 4 + 4 + 32 + 3
MODES = ("recurrent_actor", "observation_actor")
REVERSE = (1,0,3,2)


class LinearActor:
    def __init__(self, seed=1, lr=.02):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0,.004,(FEATURES,4)).astype(np.float32)
        self.b = np.zeros(4,np.float32)
        self.critic = np.zeros(FEATURES,np.float32)
        self.critic_b=0.
        self.lr=lr
        self.updates=0

    def probabilities(self, features, legal, prior):
        logits = (features@self.w + self.b + prior).astype(np.float64)
        for a in range(4):
            if a not in legal:
                logits[a]=-1e6
        probabilities=np.exp(logits-logits.max())
        return probabilities/probabilities.sum()

    def update(self, trajectories, gamma=.99):
        batch=[]
        for episode in trajectories:
            expected=0.
            for transition in reversed(episode):
                expected=transition[2]+gamma*expected
                batch.append((transition[0],transition[1],expected))
        feats=np.stack([item[0] for item in batch])
        probs=np.stack([item[1][0] for item in batch])
        chosen=np.array([item[1][1] for item in batch],np.int64)
        targets=np.array([item[2] for item in batch],np.float32)
        values=feats@self.critic+self.critic_b
        advantages=targets-values
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-6)
        probs[np.arange(len(batch)),chosen]-=1
        derivative=-(probs*advantages[:,None])
        derivative-=.01*(probs+1/4)
        grad=feats.T@derivative/len(batch)
        gradb=derivative.mean(axis=0)
        norm=np.sqrt((grad**2).sum()+(gradb**2).sum())
        if norm>1.5:
            grad*=1.5/norm
            gradb*=1.5/norm
        self.w+=self.lr*grad
        self.b+=self.lr*gradb
        delta=np.clip(targets-values,-3,3)
        self.critic+=.015*(feats.T@delta)/len(batch)
        self.critic_b+=.015*float(delta.mean())
        self.updates+=1

    def parameter_bytes(self):
        return self.w.nbytes+self.b.nbytes+self.critic.nbytes+4


class ActorMemory:
    def __init__(self, mode, model, centers, seed):
        self.mode=mode
        self.model=model
        self.centers=centers
        self.h=np.zeros(model.hidden,np.float32)
        self.visits=Counter()
        self.actions=Counter()
        self.last=None
        self.rng=random.Random(seed)
        self.t=0

    def key(self, obs):
        s=signature(obs)
        if self.mode=="observation_actor":
            return (s,)
        embedding=self.h if self.mode=="recurrent_actor" else np.zeros_like(self.h)
        cluster=int(np.argmin(((self.centers-embedding)**2).sum(axis=1)))
        return (s,cluster)

    def features(self, obs, state):
        x=np.zeros(FEATURES,np.float32)
        x[signature(obs)]=1.
        x[32:36]=obs.nearby_target
        x[36:40]=obs.walls
        if self.last is not None:
            x[40+self.last]=1
        if self.mode=="recurrent_actor":
            n=min(32,len(self.h))
            x[44:44+n]=self.h[:n]
        x[76]=min(1.,self.visits[state]/8)
        x[77]=min(1.,self.t/160)
        x[78]=min(1.,self.t/40)
        return x

    def choose(self, obs, actor):
        state=self.key(obs)
        self.visits[state]+=1
        legal=[a for a,blocked in enumerate(obs.walls) if not blocked]
        if not legal: legal=list(range(4))
        prior=np.zeros(4,np.float32)
        for a in legal:
            prior[a]=-1.3*np.log1p(self.actions[(state,a)])
            if self.last is not None and REVERSE[self.last]==a:
                prior[a]-=.25
            if obs.nearby_target[a]:
                prior[a]+=12.
        feature=self.features(obs,state)
        p=actor.probabilities(feature,legal,prior)
        action=self.rng.choices(range(4),weights=p,k=1)[0]
        self.actions[(state,action)]+=1
        old=signature(obs)
        self.last=action
        self.t+=1
        return action,(feature,(p.astype(np.float32),action),0.),old,state

    def update_state(self, old, action):
        if self.mode=="observation_actor": return
        x=np.zeros(ALPHABET+4,np.float32)
        x[old]=1
        x[ALPHABET+action]=1
        prev=self.h if self.mode=="recurrent_actor" else np.zeros_like(self.h)
        self.h=np.tanh(x@self.model.p["wx"]+prev@self.model.p["wh"]+
                        self.model.p["bh"])


def episode(mode, seed, actor, model, centers, *, changed=False,
            train=False, intrinsic=.06, limit=160, goals=3):
    world=SensorEpisode(changed=changed,limit=limit)
    observation=world.reset(seed,goals=goals)
    memory=ActorMemory(mode,model,centers,seed*17+7)
    samples=[]
    while True:
        action,item,old,key=memory.choose(observation,actor)
        observation,reward,done=world.step(action)
        memory.update_state(old,action)
        if train:
            bonus=intrinsic/np.sqrt(memory.visits[key])
            samples.append((item[0],item[1],reward+bonus))
        if done:
            break
    return world.metrics(),samples


def train_actor(mode, model, centers, *, seed, episodes=600, batch=32,
                intrinsic=.06, lr=.02):
    actor=LinearActor(seed,lr=lr)
    success=[]
    t=time.process_time()
    for start in range(0,episodes,batch):
        trajectories=[]
        for n in range(start,min(episodes,start+batch)):
            result,exp=episode(mode,seed*10001+31*n+1,actor,model,centers,
                               changed=(n%3==0),train=True,intrinsic=intrinsic)
            success.append(result["completed"])
            trajectories.append(exp)
        actor.update(trajectories)
    return actor,{"cpu_seconds":time.process_time()-t,
                  "early_100_success":sum(success[:100])/len(success[:100]),
                  "last_100_success":sum(success[-100:])/len(success[-100:]),
                  "updates":actor.updates}


def benchmark(seeds, *, source_seed=73, source_steps=12000,
              source_epochs=16, episodes=600, hidden=32,
              centers=24, intrinsic=.06):
    if not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError("nonempty unique evaluation seeds required")
    data=traces(source_seed,steps=source_steps,episode_length=100)
    start=time.process_time()
    model=RecurrentPredictor(seed=source_seed,hidden=hidden).fit(
        data,epochs=source_epochs,seed=source_seed+17)
    centroids=fit_prototypes(model,data,centers=centers,seed=source_seed)
    model_cpu=time.process_time()-start
    trained={}
    stats={}
    for mode in MODES:
        trained[mode],stats[mode]=train_actor(
            mode,model,centroids,seed=source_seed+101,episodes=episodes,
            intrinsic=intrinsic)
    untrained=LinearActor(source_seed+101)
    rows=[]
    for seed in seeds:
        for changed in (False,True):
            for mode in (*MODES,"recurrent_erased_eval","recurrent_untrained","history_novelty"):
                if mode=="history_novelty":
                    result=run_episode("history_novelty",seed,model,centroids,
                                       changed=changed)
                else:
                    m=("memory_erased_actor" if mode=="recurrent_erased_eval"
                       else "recurrent_actor" if mode=="recurrent_untrained" else mode)
                    actor=(untrained if mode=="recurrent_untrained"
                           else trained["recurrent_actor"] if mode=="recurrent_erased_eval"
                           else trained[m])
                    result,_=episode(m,seed,actor,model,centroids,changed=changed)
                rows.append({"seed":seed,"changed":changed,"mode":mode,**result})
    summary={}
    for changed in (False,True):
        label="changed" if changed else "stable"
        summary[label]={}
        for mode in (*MODES,"recurrent_erased_eval","recurrent_untrained","history_novelty"):
            r=[x for x in rows if x["changed"]==changed and x["mode"]==mode]
            summary[label][mode]={"completed":sum(x["completed"] for x in r)/len(r),
                                  "mean_waypoints":sum(x["waypoints"] for x in r)/len(r),
                                  "mean_steps":sum(x["steps"] for x in r)/len(r)}
    return {"protocol":{
        "agent_oracle_access":False,"task":"three sequential hidden goals",
        "sensor":"four local walls, ambiguous floor, adjacent target, reward",
        "policy":"linear actor trained by REINFORCE with learned linear critic",
        "shared_action_prior":"engineered per-state/action visit novelty",
        "source_seed":source_seed,"source_steps":source_steps,
        "source_epochs":source_epochs,"train_episodes_per_mode":episodes,
        "hidden":hidden,"centers":centers,"intrinsic_bonus":intrinsic,
        "rnn_pretraining_cpu_seconds":model_cpu,
        "rnn_weight_bytes":model.parameter_bytes(),
        "actor_weight_bytes":trained[MODES[0]].parameter_bytes(),
        "training":stats,"test_seeds":seeds},
        "summary":summary,"rows":rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds",default="11,12,13,14,15")
    ap.add_argument("--source-seed",type=int,default=73)
    ap.add_argument("--source-steps",type=int,default=12000)
    ap.add_argument("--source-epochs",type=int,default=16)
    ap.add_argument("--episodes",type=int,default=600)
    ap.add_argument("--hidden",type=int,default=32)
    ap.add_argument("--centers",type=int,default=24)
    ap.add_argument("--intrinsic",type=float,default=.06)
    ap.add_argument("--output",default="")
    a=ap.parse_args()
    t=time.perf_counter()
    output=benchmark([int(s) for s in a.seeds.split(",")],
        source_seed=a.source_seed,source_steps=a.source_steps,
        source_epochs=a.source_epochs,episodes=a.episodes,
        hidden=a.hidden,centers=a.centers,intrinsic=a.intrinsic)
    output["wall_seconds"]=time.perf_counter()-t
    if a.output:
        p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps({"summary":output["summary"],
                      "training":output["protocol"]["training"],
                      "wall_seconds":output["wall_seconds"],"output":a.output},
                     indent=2))


if __name__=="__main__":
    main()
