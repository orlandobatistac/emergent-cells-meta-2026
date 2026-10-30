"""Paired CPU process-time profile, not an electrical energy measurement."""
import json
from pathlib import Path
import statistics
import time

from .event_cells import run_episode


def main():
    seeds=list(range(44001,44031))
    rows=[]
    for mode in ("event","no_messages","synchronous"):
        for seed in seeds:
            t=time.process_time()
            outcome=run_episode(seed,mode=mode,budget=80)
            elapsed=time.process_time()-t
            rows.append({"seed":seed,"mode":mode,"process_cpu_seconds":elapsed,
                         **outcome})
    summary={}
    for mode in ("event","no_messages","synchronous"):
        x=[r for r in rows if r["mode"]==mode]
        summary[mode]={
            "mean_cpu_seconds_per_mission":statistics.mean(r["process_cpu_seconds"] for r in x),
            "median_cpu_seconds_per_mission":statistics.median(r["process_cpu_seconds"] for r in x),
            "mean_messages_per_mission":statistics.mean(r["messages"] for r in x),
            "completion":statistics.mean(r["completed"] for r in x)}
    r={"protocol":{
        "machine":"LANDITO-LAPTOP",
        "seed_range":"44001..44030",
        "measurement":"Python time.process_time(), not joules",
        "randomized_run_order":False,
        "run_order":"event first, then no_messages, then synchronous",
        "limitations":"Single machine and execution order; thermal, CPU contention and cache effects may bias times.",
        "budget":80},
       "summary":summary,"rows":rows}
    p=Path("results/audited/2026-10-29-event-cpu-profile.json")
    p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(json.dumps(r,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
