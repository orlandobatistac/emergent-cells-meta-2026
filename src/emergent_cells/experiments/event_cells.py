"""Event-driven local cell-to-cell value messages on an observed graph.

Independent cells own their observations, neighbor messages, visit counts, and
local value. A bounded scheduler merely delivers messages between directly
discovered neighbors; it never reads the entire map to compute global values,
does not call BFS or receive true maze coordinates/goal positions.

Movement geometry, exploration reward, message update, and the local scheduler
are *authored*, not trained. This establishes a lower-compute engineering
comparator for fully learned decentralized policies.
"""
from __future__ import annotations
import argparse
from collections import deque
from dataclasses import dataclass,field
import json
import math
from pathlib import Path
import random
import time

from .local_value_cells import LocalValueCells,MOVES,REVERSE
from .sensor_world import SensorEpisode
from .recurrent_state import GeneratedMaze


@dataclass
class Cell:
    walls: tuple
    value: float=0.
    visits: int=0
    epoch: int=0
    neighbor_values: dict=field(default_factory=dict)


class EventDrivenCells:
    def __init__(self,seed,*,budget=80,decay=.92,frontier_reward=2.5,
                 tolerance=.015):
        if budget<0 or not 0<decay<1 or frontier_reward<=0 or tolerance<=0:
            raise ValueError("invalid local event parameters")
        self.rng=random.Random(seed+19)
        self.position=(0,0)
        self.cells={}
        self.queue=deque()
        self.budget=budget
        self.decay=decay
        self.frontier_reward=frontier_reward
        self.tolerance=tolerance
        self.epoch=0
        self.last_action=None
        self.action_counts={}
        self.messages_sent=0
        self.messages_delivered=0
        self.actions=0
        self.peak_queue=0
        self.local_updates=0
        self.failures_injected=0
        self.reobserved_failed_cells=0
        self._failed_positions=set()

    def drop_cell(self,position):
        """Fault injection: erase one known cell and its queued messages.

        This models simulated cell-memory loss; it does NOT terminate a
        separate operating-system process or physical network node.
        """
        if position not in self.cells:
            return False
        lost=self.cells.pop(position)
        self.failures_injected+=1
        self._failed_positions.add(position)
        self.queue=deque((src,dst,value) for src,dst,value in self.queue
                         if src!=position and dst!=position)
        for a,wall in enumerate(lost.walls):
            if wall:continue
            neighbor=self._successor(position,a)
            if neighbor in self.cells:
                self.cells[neighbor].neighbor_values.pop(position,None)
                self._refresh(neighbor)
        return True

    def _successor(self,pos,action):
        dx,dy=MOVES[action]
        return (pos[0]+dx,pos[1]+dy)

    def _available_neighbors(self,pos):
        cell=self.cells[pos]
        return [self._successor(pos,a) for a,w in enumerate(cell.walls)
                if not w and self._successor(pos,a) in self.cells]

    def _emit(self,src):
        for dst in self._available_neighbors(src):
            self.queue.append((src,dst,self.cells[src].value))
            self.messages_sent+=1
        self.peak_queue=max(self.peak_queue,len(self.queue))

    def _refresh(self,pos):
        c=self.cells[pos]
        if c.epoch!=self.epoch:
            c.epoch=self.epoch
            c.visits=0
        frontier=any(self._successor(pos,a) not in self.cells
                     for a,w in enumerate(c.walls) if not w)
        reward=.9/math.sqrt(1+c.visits)+(self.frontier_reward if frontier else 0.)
        value=max(reward,self.decay*max(c.neighbor_values.values(),default=0.))
        self.local_updates+=1
        if abs(value-c.value)>self.tolerance:
            c.value=value
            self._emit(pos)

    def _drain(self):
        remaining=self.budget
        while remaining and self.queue:
            src,dst,value=self.queue.popleft()
            remaining-=1
            if src not in self.cells or dst not in self.cells:
                continue
            self.messages_delivered+=1
            cell=self.cells[dst]
            if abs(cell.neighbor_values.get(src,0.)-value)>self.tolerance:
                cell.neighbor_values[src]=value
                self._refresh(dst)

    def _sense(self,reading):
        current=self.cells.get(self.position)
        if current is None:
            self.cells[self.position]=Cell(tuple(reading.walls),epoch=self.epoch)
            if self.position in self._failed_positions:
                self._failed_positions.remove(self.position)
                self.reobserved_failed_cells+=1
        else:
            current.walls=tuple(reading.walls)
        # Local undirected edge handshake when both neighbors are observed.
        for neighbor in self._available_neighbors(self.position):
            # No distant cell data read, just an existing direct neighbor.
            if self.position not in self.cells[neighbor].neighbor_values:
                self.cells[neighbor].neighbor_values[self.position]=self.cells[self.position].value
                self._refresh(neighbor)
            self.cells[self.position].neighbor_values[neighbor]=self.cells[neighbor].value
        self._refresh(self.position)

    def act(self,reading):
        self._sense(reading)
        c=self.cells[self.position]
        c.visits+=1
        self._refresh(self.position)
        self._drain()
        legal=[a for a,w in enumerate(reading.walls) if not w]
        targets=[a for a in legal if reading.nearby_target[a]]
        if targets:
            action=targets[0]
        elif legal:
            def score(a):
                p=self._successor(self.position,a)
                visits=self.action_counts.get((self.position,a),0)
                rev=int(self.last_action is not None and a==REVERSE[self.last_action])
                potential=(self.cells[p].value if p in self.cells else
                           self.frontier_reward+2.)
                return potential-.06*visits-.1*rev+self.rng.random()*.025
            action=max(legal,key=score)
        else:
            action=self.rng.randrange(4)
        self.action_counts[self.position,action]=self.action_counts.get(
            (self.position,action),0)+1
        self.last_action=action
        self.actions+=1
        return action

    def observe(self,reading):
        if self.last_action is None:
            raise RuntimeError("act before observe")
        self.position=self._successor(self.position,self.last_action)
        if reading.previous_reward>.5:
            self.epoch+=1
            self.action_counts.clear()
            # Epoch means the cell visit state will lazily reset on encounter.
        self._sense(reading)


def run_episode(seed,*,mode,budget=80,changed=False,procedural=False,
                fault_at=None,decay=.92,frontier_reward=2.5):
    world=SensorEpisode(changed=changed,limit=160)
    if procedural:
        world._maze=GeneratedMaze(seed)
    reading=world.reset(seed,goals=3)
    agent=(EventDrivenCells(seed,budget=budget,decay=decay,
                            frontier_reward=frontier_reward) if mode=="event"
           else EventDrivenCells(seed,budget=0,decay=decay,
                                  frontier_reward=frontier_reward) if mode=="no_messages"
           else LocalValueCells(seed,rounds=10))
    injected=False
    while True:
        if (fault_at is not None and not injected and
                isinstance(agent,EventDrivenCells) and agent.actions>=fault_at):
            known=[p for p in agent.cells if p!=agent.position]
            if known:
                selected=max(sorted(known),key=lambda p:agent.cells[p].visits)
                injected=agent.drop_cell(selected)
        action=agent.act(reading)
        reading,_,done=world.step(action)
        agent.observe(reading)
        if done:break
    if isinstance(agent,EventDrivenCells):
        measured={"messages":agent.messages_delivered,
                  "messages_sent":agent.messages_sent,
                  "local_updates":agent.local_updates,
                  "peak_queue":agent.peak_queue,
                  "discovered_cells":len(agent.cells),
                  "faults_injected":agent.failures_injected,
                  "failed_cells_reobserved":agent.reobserved_failed_cells}
    else:
        measured={"messages":agent.local_message_updates,
                  "messages_sent":agent.local_message_updates,
                  "local_updates":None,
                  "peak_queue":None,
                  "discovered_cells":len(agent.open_actions),
                  "faults_injected":0,
                  "failed_cells_reobserved":0}
    return {**world.metrics(),**measured}


def benchmark(seeds,*,budget=80,procedural=False):
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError("unique seeds required")
    t=time.process_time()
    records=[]
    for changed in ((False,) if procedural else (False,True)):
        for seed in seeds:
            for mode in ("event","no_messages","synchronous"):
                outcome=run_episode(seed,mode=mode,budget=budget,
                                    changed=changed,procedural=procedural)
                records.append({"seed":seed,"changed":changed,"mode":mode,
                                **outcome})
    summary={}
    for changed in ((False,) if procedural else (False,True)):
        label="procedural" if procedural else "changed" if changed else "stable"
        summary[label]={}
        for mode in ("event","no_messages","synchronous"):
            rows=[r for r in records if r["changed"]==changed and r["mode"]==mode]
            summary[label][mode]={
                "completion_rate":sum(r["completed"] for r in rows)/len(rows),
                "mean_goals":sum(r["waypoints"] for r in rows)/len(rows),
                "mean_actions":sum(r["steps"] for r in rows)/len(rows),
                "mean_message_deliveries":sum(r["messages"] for r in rows)/len(rows),
                "mean_cells_discovered":sum(r["discovered_cells"] for r in rows)/len(rows)}
    return {"protocol":{
        "seeds":seeds,"goals":3,"navigation_budget":160,
        "event_message_budget_per_action":budget,
        "programmed_relative_odometry":True,
        "programmed_local_value_update":True,
        "learned_message_weights":False,
        "central_event_delivery_not_cpu_distributed":True,
        "global_map_or_route_input":False,
        "procedural":procedural,
        "process_cpu_seconds":time.process_time()-t},
        "summary":summary,"records":records}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds",default="51,52,53")
    ap.add_argument("--budget",type=int,default=80)
    ap.add_argument("--procedural",action="store_true")
    ap.add_argument("--output",default="")
    a=ap.parse_args()
    data=benchmark([int(s) for s in a.seeds.split(",")],
                   budget=a.budget,procedural=a.procedural)
    if a.output:
        p=Path(a.output);p.parent.mkdir(exist_ok=True,parents=True)
        p.write_text(json.dumps(data,indent=2),encoding="utf-8")
    print(json.dumps({"summary":data["summary"],
                      "cpu_seconds":data["protocol"]["process_cpu_seconds"],
                      "output":a.output},indent=2))


if __name__=="__main__":
    main()
