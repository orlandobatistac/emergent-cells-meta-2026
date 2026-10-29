# R0 Prototype — Agent-Only Local Sensory Interface
**Date:** 2026-10-28 | **Status:** initial benchmark and baselines, not learned navigation.

## Research question

What remains of multi-goal competence when we stop supplying an agent with exact spatial coordinates, goal coordinates and a central graph planner?

We intentionally keep the old privileged navigation experiment intact as a comparison. This new environment offers a minimal, strictly defined policy input.

## Agent-facing contract

`Reading` contains only:
- `walls`: four Boolean egocentric-adjacent blockage indicators, in a fixed action orientation;
- `nearby_target`: whether the current target is exactly one reachable action away in each direction;
- `floor_signal`: deliberately ambiguous two-state texture; several distinct positions share the same observation;
- `previous_reward`: latest earned local scalar reward.

`SensorEpisode.reset(seed, goals)` returns a reading. `SensorEpisode.step(action)` returns a new reading, reward and terminal flag. Position, target sequence, wall layout and scoring are held by the **simulation harness**. The agent policy is called with **only a reading and its own saved local state**. The harness's `metrics()` method exists exclusively for external evaluation.

**Important boundary:** this is a source-code interface convention and audited test contract, not operating-system isolation or a Python security sandbox. For a stronger anti-leakage guarantee, eventually place agents in separate processes exposing only serialized observations.

## Baselines

Two intentionally weak, strictly observation-based controls:
- `random`: chooses randomly from locally unblocked actions, without history.
- `reactive`: follows the target if immediately adjacent; otherwise performs visit-count-balanced exploration keyed to the **ambiguous sensory signature**, not coordinates.

The 9×9 world has three sequential unknown targets, each episode limited to 160 primitive actions. The maze remains fixed within a run. The changed variant closes one upper crossing before episode start; this is a topology generalization control, **not** an unannounced mid-episode closure detection test.

Ten independent exploratory seeds (101–110), one mission per world condition and baseline:

| World | Control | Full 3-goal missions completed | Average goals reached |
|---|---|---:|---:|
| Stable | Random | 0/10 | 0.4 |
| Stable | Reactive | 2/10 | 1.4 |
| Changed | Random | 0/10 | 0.6 |
| Changed | Reactive | 0/10 | 0.9 |

The result is **negative** for simple reactive behavior and demonstrates a real competency gap; it does not establish learned state or decentralized planning. These runs do not compare equal training budgets because neither control receives source training. They are baseline calibration for R1.

## Tests and scope

Six new tests check: schema contains no absolute positions or targets, deterministic action traces, sensory aliasing, positive local target reward and completion, action validity, and deterministic bounded baseline output.

The internal simulator still uses `Maze` and absolute coordinates **inside the harness**. The agent-facing action selection does not use `Maze.shortest`, coordinates or BFS. Older benchmarks still use privileged coordinates and BFS: their existence in the codebase does not imply this milestone has solved R3.

## Next experiment: R1

Replace the reactive visit count with a compact per-agent recurrent predictive state trained only from observation/action/reward sequences. Compare recurrent table, contrastive transition embedding and local-prediction updates. Demand improvement on unseen sequential goals, measure prediction accuracy on held-out action traces and preserve sensor-aliasing controls. Do not train with ground-truth coordinates.

The next stage must eventually link to the original population's locally learned messages. Do not present this standalone sensor shell as an emergent cellular reasoning agent.

## Reproduce

```powershell
python -m pip install -e .
python -m unittest discover -s tests -p test_sensor_world.py -v
python run.py sensor-world --seeds 101,102,103,104,105,106,107,108,109,110 --goals 3 --limit 160 --output results/runs/sensor-baselines.json
```

Raw exploratory results: `results/exploratory/sensor-world-r0-baselines.json`.
