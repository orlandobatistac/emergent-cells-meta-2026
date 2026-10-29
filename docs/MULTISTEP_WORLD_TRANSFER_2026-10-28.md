# Multi-Step World Transfer — Local Transition Cells and Online Replanning
**Experimental date:** 2026-10-28. **Status:** completed controlled synthetic prototype.

## Research question

Can a resource-light modular system learn how an environment changes under local actions, reuse that knowledge for new **sequences of goals**, and correct its own predictions after a previously passable route is blocked?

This benchmark intentionally moves beyond Boolean truth tables. Its goal is **not** to claim a discovered generalized reasoning algorithm. The learner is a modular tabular dynamics memory, paired with a **hand-written centralized breadth-first graph planner**. Exact position and target coordinates are supplied by the simulator; these remain strong engineered representations.

## Relevant work and design rationale

- Liu et al., **Continual Reinforcement Learning by Planning with Online World Models** (ICML 2025). Reusable online dynamics estimation with planning for changing tasks: https://proceedings.mlr.press/v267/liu25p.html
- GX-Chen, Marino, and Fergus, **Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction** (ICLR 2025). Abstract state/action representations and longer-horizon planning: https://proceedings.iclr.cc/paper_files/paper/2025/hash/1f8b45077e53a39a8b657101bb1a09b4-Abstract-Conference.html
- Barreto et al., **Successor Features for Transfer in Reinforcement Learning** (NeurIPS 2017). Separating reusable environment structure from reward-specific policies: https://papers.nips.cc/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html
- Fu et al., **Knowledge Retention in Continual Model-Based Reinforcement Learning** (ICML 2025). Model retention across changing reward tasks: https://proceedings.mlr.press/v267/fu25f.html
- Zhang et al., **Mixture-of-World Models: Scaling Multi-Task Reinforcement Learning with Modular Latent Dynamics** (ICLR 2026). Modular dynamics, but at a vastly larger learning scale than our CPU pilot: https://proceedings.iclr.cc/paper_files/paper/2026/hash/33b47b3d2441a17b95344cd635f3dd01-Abstract-Conference.html
- Farebrother et al., **Compositional Planning with Jumpy World Models** (ICML 2026). Longer-horizon planning through composable temporally extended skills: https://proceedings.mlr.press/v306/farebrother26a.html

These papers motivate the **questions**, not implementations replicated here. Our prototype uses a simpler explicit transition table and BFS.

## Architecture and information boundary

A 9×9 maze has an internal wall separating its two halves, two passages, and several stationary obstacles. At each step the simulator exposes exactly the current **(x, y) position**, current active **(x, y) goal** and four available action codes. It does **not** reveal the maze's obstacle set or the shortest route. A local transition entry records the observed outcome of each action at a position: `position → action → successor`. Every such entry acts as an independently updateable local memory.

A generic **central** graph search reads these learned entries to plan routes and explore unknown edges. After taking an action, only that action's local memory is updated. A contradiction (predicted successor differs from observed successor) records a revision. This is **not** decentralized multi-cell planning; its central BFS remains a hand-authored engineering scaffold.

Source training alternates between **(1, 1)** and **(7, 7)** destination cues. It begins with no map and uses goal-seeking plus 35% random actions. Every 83 source interactions, the simulator can reset the position to a random accessible tile. This privileged reset improves coverage and cannot be assumed in a deployed robot. Training uses 7,000 action observations; goal coordinates and precise location are engineered sensory channels. World transitions, rather than a learned natural-language task representation, drive transfer.

## Frozen test protocol and controls

Two external source goals are never included among the eight candidate held-out goals. Each seed generates 30 separate missions, each containing three **consecutive held-out targets**. Each mission restarts from the identical seeded position for every policy; a new instance of the corresponding model is used to prevent evaluation leakage between missions. The maximum is 160 primitive actions per mission.

Five conditions:
1. **Reusable model:** source-trained transition memory, updated online on contradictions.
2. **Frozen model:** same pretrained memory, but no updates after training.
3. **Cold model:** same BFS and learning rules, with no pretrained environment memory.
4. **Greedy:** memoryless Manhattan-distance actions, an intentionally weak geometric heuristic.
5. **Random:** uniformly random actions.

We deliberately include the cold model as the more informative planning baseline; it uses **less total training experience** than warm models. Therefore this is an evaluation of transfer after pretraining, **not** a claim that the warm method uses fewer lifetime interactions or computational resources.

Dynamic negative control: in a separate post-training task, the original upper crossing at **(4, 2)** is closed without announcement; the bottom passage remains open. The agent must travel from (1, 2) to (7, 2), then (1, 6). Frozen memory cannot correct the closed crossing. This is **one engineered change**, not comprehensive robustness under topology or observation drift.

Development exploration: an initial uniform-random-source pilot (seeds 11–12) showed that complete transition coverage made the task nearly deterministic but did not actually train toward source destinations. It is retained as `results/exploratory/multistep-world-pilot-11-12.json` for transparency and is **not** the final training protocol. A revised source-goal-driven pilot used seeds 11–13 with 4,500 training actions and 16 test missions. The final fixed protocol uses **fresh held-out seeds 2001–2010** with 7,000 training steps, 30 missions per seed, three waypoints, and a 160-step horizon.

## Held-out results (10 seeds × 30 missions = 300 missions per policy)

| Policy | Entire 3-goal mission solved | Mean excess steps on successful missions | Unannounced door closure solved (10 seeds) |
|---|---:|---:|---:|
| Reusable pretrained local model | **100.0%** | **1.04** | **10/10** |
| Frozen pretrained local model | **100.0%** | **1.04** | **0/10** |
| Cold local model + identical BFS | **100.0%** | **20.63** | **10/10** |
| Greedy Manhattan | 6.0% | 0.0 (successful cases only) | 0/10 |
| Random policy | 0.0% | not applicable | 0/10 |

The warm model reached **90.62% local transition edge coverage** on average after source training. Its trained graph was incomplete, so successful navigation still sometimes required additional exploration. Against the cold model, it saved an average **19.59 primitive actions per mission**. Bootstrap resampling the ten **independent seed-level differences** (20,000 draws, fixed seed 9024) yielded a descriptive 95% interval of **[17.68, 21.40]** extra steps saved per mission. This interval is limited to this synthetic setup and does not establish generalization to other worlds.

The dynamic change produced an average **1.10 locally recorded transition revisions** per evaluated seed for the online model, which succeeded in all ten runs. Frozen reuse failed all ten. This identifies the need for online state-change correction; it is not evidence of a learned change-detection policy.

### Compute and memory

Measured using Python 3.12 on the connected Windows laptop in ten held-out seeds:
- Mean source-training **process CPU time:** approximately **1.22 seconds per seed** for 7,000 world action updates, excluding evaluation.
- Mean reused-model evaluation **process CPU time:** approximately **0.075 seconds per seed** for 30 missions (wall time and energy use may differ; Windows timer resolution is limited).
- Approximate **shallow Python object accounting** for the learned transition entries: **88,599 bytes** averaged across seeds. This is *not* real RAM, process RSS, a packed lower bound, or a hardware-independent memory measurement.
- The study also stores `tracemalloc` peak allocated bytes while tracing source training; this measure excludes some existing allocations and must **not** be directly compared against the shallow total or interpreted as device RAM.
- **No electrical energy measurements were made.** No Meta API calls, large models, or GPU are required.

## Scientific interpretation and limitations

**Supported:** an ordinary laptop can run a small modular transition learner that uses prior action observations to execute new multi-step goal sequences with less exploration than an otherwise identical model initialized without memory. It can locally overwrite contradicted transitions and replan after one externally introduced obstacle change.

**Not supported:** autonomous hierarchical task discovery, deep reasoning, generalized language grounding, decentralized route computation, transfer to new sensor formats or maze families, true task-free partial-observation learning, generic robust lifelong learning, or orders-of-magnitude resource advantages versus strong neural RL/world-model baselines. The exact (x,y) position/goal cues and the BFS algorithm supply much of the skill. Source-goal policy and learned transition memory are not direct extensions of the earlier one-bit message network; this is a **separate modular research branch**, not proof the original cellular architecture has acquired those capabilities.

## Next falsifiable test

Replace privileged coordinates with partial local observations and learned place signatures, vary maze layouts/door schedules across seeds, and measure whether a compact distributed latent transition representation still supports multi-step transfer. Compare source-trained vs cold agents with matched total lifetime action budgets and include a strong goal-conditioned RL baseline. Separately measure true process RSS, CPU utilization and externally sampled watts or joules, not just synthetic reward.

## Reproduction

```powershell
python -m pip install -e .
python -m unittest discover -s tests -q
python run.py multistep-world --seeds 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010 --train-steps 7000 --missions 30 --step-limit 160 --output results/runs/multistep-reproduction.json
```

Raw data: `results/audited/2026-10-28-multistep-world-holdout.json`. Development results: `results/exploratory/multistep-world-goal-driven-pilot.json`. This document is in English under the repository contribution policy; historical data are not rewritten.
