# Emergent Cells Lab

**CPU-first research into local learning, adaptive memory, communication, and autonomous intelligence.**

Emergent Cells Lab investigates whether modular learners can build useful world representations, share knowledge, and act toward multi-step goals without an externally supplied global map or a hand-coded central planner. Our target is an experimentally verifiable architecture that can train and run on ordinary computers, not a claim of artificial general intelligence.

## Active research objective

**Eliminate two artificial dependencies:**

1. **Privileged representation:** the agent should learn internal state from observations, actions, and local rewards rather than receive exact world/goal coordinates or hidden task IDs.
2. **Centralized planning:** the agent should learn or coordinate actions using its own local state and bounded inter-cell messages, rather than call global BFS or an oracle route.

The environment may hold hidden truth **only for evaluation**. An agent must not receive that truth through its input interface.

**[Full research roadmap (R0-R5)](docs/RESEARCH_ROADMAP_NO_ORACLES_2026-10-28.md)** | **[R1: predictive latent state](https://github.com/orlandobatistac/emergent-cells-meta-2026/issues/21)**

## Current status (October 28, 2026)

| Research line | Evidence | Status |
|---|---|---|
| Learned local communication | Around 98% accuracy on a fixed, synthetic XOR task | Verified in bounded tests |
| Reward-triggered memory expansion | Approximately 93-95% final-stage accuracy on scheduled Boolean tasks, but false allocation under increased reward dropout | Promising, with known failures |
| Sequential policy transfer | OR 92.04% and NAND 91.73% held-out accuracy using locally selected memory fragments, five seeds | Verified for related Boolean tasks |
| Spatial world-model transfer | Reused transition tables complete 300/300 three-waypoint missions, reducing average excess actions from 20.63 to 1.04 | **Uses privileged coordinates and coded BFS; comparator only** |
| **R0: sensor-only input** | Policy receives nearby obstacles, adjacent target signal, ambiguous floor cue, and past reward. Reactive baseline solves 2/10 stable missions and 0/10 changed missions | **Initial interface and negative baseline implemented; security isolation incomplete** |
| **R1: predictive state** | A trained 48-unit recurrent state reached **93.56%** held-out next-sensor accuracy (vs **86.96%** history table); on unseen maze variants, **76.16%** (vs **72.85%** table). Erasing recurrent memory reduced stable accuracy to **58.33%** | **Trainable predictive state experimentally validated; goal-conditioned learning and cell integration still pending** |
| **R1: closed-loop action experiments** | With **only sparse adjacent-goal sensing**, explicit history exploration completed 52/100 stable and 36/100 changed three-goal missions; recurrent-state novelty completed 37/100 and 33/100. In a **separate task with an engineered noisy goal-distance signal**, reward-progress training completed 100/100 and 50/100; erasing recurrent memory did not reduce performance in that task | **Learned action selection demonstrated in an informative-sensor setting; recurrent-memory causal advantage for navigation remains unproven** |
| **R1: reward-trained recurrent policy** | A 12-unit recurrent action policy learned to choose correctly after an initial cue: **100% at delays 8 and 12**, **85.19% at 16**, versus ~50% with memory erased (20 held-out seeds). However, on the original sparse-goal maze, a separate recurrent REINFORCE policy solved **18% stable / 11% changed**, worse than explicit-history exploration (**50% / 39%**) | **Causal reward-trained memory proven only in a delayed-cue diagnostic; general three-goal navigation remains unsolved** |
| **R1: local-value spatial-memory baseline (Oct 29)** | Programmed relative odometry plus locally observed cell-memory/value diffusion completed **97/100** stable and **88/100** closed-passage three-goal missions in an isolated sandbox, versus matched history **50/100** and **31/100**. The relative motor model and value-update policy are authored; these results were **not run on the laptop** | **Strong engineered non-oracle baseline; learned recurrent motor-state/planning still unsolved** |
| R2-R5 | Communication, decentralized planning, generalization, and measured efficiency | Planned; no success claims |

Tests verify a defined source-code observation contract, **not** a sandboxed security barrier against a malicious agent. Python world attributes remain available to the evaluation harness. We have not demonstrated a decentralized reasoning agent.

## Run locally

Python 3.10+; NumPy. No GPU, web interface, API key, or Meta API is needed for the numerical experiments.

```powershell
python -m pip install -e .
python run.py list
python -m unittest discover -s tests -q

# First no-coordinate agent-facing benchmark (negative controls):
python run.py sensor-world --seeds 101,102,103,104,105,106,107,108,109,110 --goals 3 --limit 160 --output results/runs/sensor-baselines.json

# First R1 sensor-only predictive memory benchmark:
python run.py predictive-context --seeds 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010 --train-steps 24000 --test-steps 6000 --output results/runs/r1-context.json

# R1 learned 48-dimensional recurrent state (no coordinate labels):
python run.py recurrent-state --seeds 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010 --train-steps 24000 --test-steps 4000 --epochs 30 --hidden 48 --output results/runs/r1-recurrent.json

# R1 action selection with only sparse adjacent-goal sensing:
python run.py latent-navigation --seeds 4001,4002,4003,4004,4005 --train-steps 16000 --epochs 20 --hidden 32 --centers 24 --output results/runs/strict-closed-loop.json

# Separate R1 goal task with a noisy local target-proximity sensor:
python run.py beacon-navigation --seeds 3001,3002,3003,3004,3005 --warm-episodes 300 --output results/runs/goal-informed-policy.json

# R1 reward-trained policy on original sparse-goal maze (negative outcome):
python run.py policy-gradient --seeds 7101,7102,7103,7104,7105 --episodes 900 --source-steps 16000 --source-epochs 20 --intrinsic .08 --output results/runs/reward-policy.json

# R1 delayed-cue memory-dependent policy (positive diagnostic):
python run.py delayed-goal --seeds 6101,6102,6103,6104,6105 --batches 280 --batch 64 --length 8 --cases 1000 --output results/runs/delayed-cue.json

# Oct 29: observed local memory and programmed neighbor-value propagation:
python run.py local-value-cells --seeds 13001,13002,13003,13004,13005 --rounds 10 --decay 0.92 --frontier-reward 2.5 --output results/runs/r1-local-value.json

# Earlier privileged coordinate+BFS comparison (not the target architecture):
python run.py multistep-world --seeds 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010 --train-steps 7000 --missions 30 --output results/runs/multistep.json

# Earlier local-reward memory transfer:
python run.py sequential-transfer --seeds 2001,2002,2003,2004,2005 --pairs 24 --pre-steps 3800 --probe-steps 120 --adapt-steps 1100 --output results/runs/transfer.json
```

For specific runner options, use `python run.py <experiment> --help`. Results written to `results/runs/` are generated locally and should not be mistaken for preregistered held-out evidence.

## Reproducible research

- **[No-oracle roadmap](docs/RESEARCH_ROADMAP_NO_ORACLES_2026-10-28.md)**: six falsifiable stages and completion gates.
- **[R0 sensor-only protocol](docs/SENSOR_ONLY_WORLD_R0_2026-10-28.md)**: information boundary, baseline failures, and tests.
- **[R1 predictive memory experiment](docs/R1_PREDICTIVE_CONTEXT_2026-10-28.md)**: next-observation prediction from observation/action history, matched baselines and unseen-seed results.
- **[R1 trainable recurrent state](docs/R1_TRAINABLE_RECURRENT_STATE_2026-10-28.md)**: learned 48-dimensional state, hidden-memory causal ablation, held-out maze variants and limitations.
- **[R1 closed-loop goal learning](docs/R1_CLOSED_LOOP_GOAL_LEARNING_2026-10-28.md)**: strict sensor-only navigation, distinct noisy-proximity experiment, trained action values, causal controls and negative results.
- **[R1 locally observed value cells](docs/R1_LOCAL_VALUE_CELLS_2026-10-29.md)**: 100-seed alternative-sandbox validation of internal odometry and neighbor-only value diffusion, including ablations and explicit engineering limitations.
- **[R1 reward-trained recurrent policies](docs/R1_REWARD_TRAINED_POLICY_2026-10-28.md)**: delayed-cue memory-dependent success, failed sparse-maze REINFORCE, higher-resolution state controls and full raw evidence.
- **[Three-waypoint model study](docs/MULTISTEP_WORLD_TRANSFER_2026-10-28.md)**: positive transfer and explicit coordinate/BFS limitations.
- **[Sequential transfer study](docs/SEQUENTIAL_TRANSFER_2026-10-28.md)**: local memory recombination and negative transfer controls.
- **[Dynamic memory study](docs/DYNAMIC_MEMORY_THIRD_RULE_2026-10-27.md)** and **[learned curiosity](docs/LEARNED_CURIOSITY_2026-10-27.md)**: prior results and failures.
- `src/emergent_cells/`: Python world, cell and experiment modules.
- `tests/`: deterministic unit and research integrity checks.
- `results/audited/`: held-out numerical evidence; `results/exploratory/`: development trials and negative results.
- `docs/EXPERIMENT_PROTOCOL.md`: research procedure.

Old results remain intact. Report controls, separate development and holdout seeds, and measure model quality, training interactions, CPU time, memory, and actual energy **only when measured**. Simulation reward is not electrical energy.

## Contributor and AI-agent policy

**All code, comments, docstrings, documentation, issues, pull requests, and research artifacts must use clear, modern professional English, regardless of the language spoken by contributors or used by their AI driver.** Discussions can happen in any language. Keep older material for provenance, including the archived Spanish research synthesis.

Do not silently edit historical data or claim new capabilities from a prototype that uses hidden labels, central planners, privileged sensors, or unmeasured hardware efficiency. Document negative results and do not substitute stronger scripted logic for learning.
