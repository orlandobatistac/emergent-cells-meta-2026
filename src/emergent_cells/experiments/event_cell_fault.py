"""Simulated single-cell memory loss with local rediscovery, not a PC shutdown."""
import json
from pathlib import Path
import time
from .event_cells import run_episode


def main():
    seeds=list(range(45001,45101))
    start=time.process_time()
    rows=[]
    for changed in (False,True):
        for seed in seeds:
            for mode,fault in (("event_healthy",False),
                               ("event_cell_loss",True)):
                result=run_episode(seed,mode="event",budget=80,
                                   changed=changed,fault_at=30 if fault else None)
                rows.append({"seed":seed,"changed":changed,"mode":mode,**result})
    summary={}
    for changed in (False,True):
        label="changed" if changed else "stable"
        summary[label]={}
        for mode in ("event_healthy","event_cell_loss"):
            data=[r for r in rows if r["changed"]==changed and r["mode"]==mode]
            summary[label][mode]={
                "complete_rate":sum(x["completed"] for x in data)/len(data),
                "mean_waypoints":sum(x["waypoints"] for x in data)/len(data),
                "mean_steps":sum(x["steps"] for x in data)/len(data),
                "failure_triggered_rate":sum(x["faults_injected"] for x in data)/len(data),
                "reobserved_lost_cell_rate":sum(x["failed_cells_reobserved"] for x in data)/len(data)}
    output={"protocol":{
        "seeds":seeds,"max_actions":160,"goals":3,"fault_action":30,
        "fault_type":"erase one visited cell's local observations and pending neighbor messages",
        "real_OS_process_or_network_node_failure":False,
        "no_map_or_pose_oracle_input":True,
        "process_cpu_seconds":time.process_time()-start},
        "summary":summary,"records":rows}
    p=Path("results/audited/2026-10-29-event-cell-fault-holdout.json")
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
