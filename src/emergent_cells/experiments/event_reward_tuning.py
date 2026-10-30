"""Fit two shared event-message coefficients from episodic task reward.

This is **black-box reward-based hyperparameter learning** of two shared
authored weights, not neural message learning or per-cell local plasticity.
Source training and independent evaluation seeds never overlap.
"""
import json
from pathlib import Path
import statistics
import time

from .event_cells import run_episode


def fit(*, seeds, budget=80):
    choices=[]
    for decay in (.65,.75,.85,.92,.97):
        for frontier in (1.,1.75,2.5,3.5,5.):
            records=[]
            for seed in seeds:
                for changed in (False,True):
                    r=run_episode(seed,mode="event",budget=budget,
                                  changed=changed,decay=decay,
                                  frontier_reward=frontier)
                    records.append(r)
            rate=statistics.mean(r["completed"] for r in records)
            goals=statistics.mean(r["waypoints"] for r in records)/3.
            actions=statistics.mean(r["steps"] for r in records)/160.
            messages=statistics.mean(r["messages"] for r in records)
            # Reward uses only online interactions, no hidden state/route.
            score=rate+.10*goals-.015*actions-.015*(messages/5000.)
            choices.append({"decay":decay,"frontier":frontier,
                            "score":score,"success":rate,
                            "mean_messages":messages,"mean_goals":goals})
    choices.sort(key=lambda x:x["score"],reverse=True)
    return choices[0],choices


def main():
    train=list(range(51001,51026))
    heldout=list(range(52001,52101))
    t=time.process_time()
    best,candidates=fit(seeds=train)
    records=[]
    for scenario in ("stable","changed","procedural"):
        for seed in heldout:
            for variant in ("learned_shared","fixed_default","no_messages"):
                kwargs=dict(mode="no_messages" if variant=="no_messages" else "event",
                            budget=80,changed=scenario=="changed",
                            procedural=scenario=="procedural")
                if variant=="learned_shared":
                    kwargs.update(decay=best["decay"],
                                  frontier_reward=best["frontier"])
                r=run_episode(seed,**kwargs)
                records.append({"seed":seed,"scenario":scenario,
                                "variant":variant,**r})
    summary={}
    for scenario in ("stable","changed","procedural"):
        summary[scenario]={}
        for variant in ("learned_shared","fixed_default","no_messages"):
            x=[r for r in records if r["scenario"]==scenario and
               r["variant"]==variant]
            summary[scenario][variant]={
                "success":statistics.mean(r["completed"] for r in x),
                "mean_goals":statistics.mean(r["waypoints"] for r in x),
                "mean_messages":statistics.mean(r["messages"] for r in x)}
    output={"protocol":{
        "train_seeds":train,"test_seeds":heldout,
        "train_missions_per_candidate":len(train)*2,
        "candidate_count":len(candidates),
        "optimizer":"enumerated black-box scalar message hyperparameters",
        "full_message_rule_engineered":True,
        "training_goal":"episode completion, partial goal reward, action and messaging penalty",
        "no_hidden_state_labels_or_oracle_routes_to_optimizer":True,
        "cpu_process_seconds":time.process_time()-t},
        "chosen_parameters":best,"candidates":candidates,
        "summary":summary,"records":records}
    target=Path("results/audited/2026-10-29-event-reward-weights.json")
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps({"chosen":best,"summary":summary,
                      "cpu_seconds":output["protocol"]["cpu_process_seconds"]},indent=2))


if __name__=="__main__":
    main()
