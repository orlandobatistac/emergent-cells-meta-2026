# R1 Experiment — Locally Observed Spatial Memory and Neighbor Value Diffusion

**Date:** 2026-10-29. **Status:** strong *engineered spatial-memory baseline*, not learned recurrent navigation.

## Research question

Can the existing sensor-only three-target world be navigated without *supplied* coordinates, target locations, a global wall map, or BFS? What is the contribution of (a) relative self-motion memory and (b) local value propagation over discovered cell connections?

The previous frozen recurrent world model obtained 18% stable / 11% shifted mission completion under reward-trained policy gradients, while the explicit three-action history explorer attained 50% / 39% in that study. These are separate prior seeds; the new study re-runs the history control on matched seeds.

Related work and motivation:

- Jeen et al., *Zero-Shot Reinforcement Learning Under Partial Observability* (RLC 2025), https://rlj.org/papers/jeen2025zero — memory and task transfer under partial observability.
- Machado et al., *Temporal Abstraction in Reinforcement Learning with the Successor Representation* (JMLR 2023), https://www.jmlr.org/papers/v24/21-1213.html — predicting state occupancy and composing temporally extended skills.
- Malegaonkar et al., *Reward Structure Shapes the Interaction Between Episodic Exploration and Neural Memory in Reinforcement Learning* (August 2026 preprint), https://arxiv.org/abs/2608.05111 — why exploration and what memory retains must be tested jointly.
- Meng et al., *Multi-step first: A lightweight deep reinforcement learning strategy for robust continuous control with partial observability* (Neural Networks 2026), https://doi.org/10.1016/j.neunet.2025.108521 — multi-step learning considerations.

Our work is an original lightweight prototype, **not** an implementation or replication of those papers.

## Exact information boundary

The simulator supplies to the agent only:
1. Four local wall-block indicators,
2. Four indicators for a goal immediately adjacent to the agent,
3. A two-state ambiguous floor reading,
4. The latest reward from its own action.

The agent **does not** receive the simulator's global coordinates, distant goal coordinates, wall layout, shortest-path distance, global route, or goal ID.

**But the agent is given additional *programmed motion semantics*.** It knows how each of four action labels changes a location in a self-initialized `(0,0)` relative coordinate frame and trusts the wall detector to mark collision-free moves. It locally integrates its chosen actions into a relative position. This explicit 2D motor/odometry prior is **not learned**, and would need replacement/calibration for noisy or rotated real-world sensors. The relative cell memory is constructed online, not imported from the simulator.

The value field is also **programmed**: over every discovered cell, ten synchronous rounds compute local desirability from inverse goal-phase visit count and nearby unknown passage frontiers, then pass it to directly observed neighboring cells using decay 0.92. The Python loop is a **central scheduling mechanism**, although each value update reads only discovered neighbor values. It is not fully asynchronous distributed computation or neural/plastic message-passing.

The controller does not perform BFS, does not receive a global map, and requires no prior maze-specific obstacle labels.

## Reproducible study

Development: 30 simulator seeds 101–130, two world conditions. These seeds guided selection of 10 diffusion rounds, decay 0.92 and frontier reward 2.5. No test seed overlap.

Independent study: 100 untouched seeds **13001–13100**, original maze and the alternative maze with one passage blocked **before each episode**. Every condition uses the exact same start and target sequence per seed, three sequential targets, 160 primitive actions maximum.

Comparators:
- Existing explicit-history exploration policy, identical input restrictions.
- **Greedy relative memory**: same action integration and local visitation counts, no value messages.
- **Local value diffusion**: same relative memory plus ten authored neighbor-only value propagation rounds.
- **Frozen-odometry ablation**: identical diffusion code, but never updates the internal relative location after an action. This tests reliance on position integration, not just hyperparameter count.

### Results — held-out 100 missions per condition

| Agent | Stable all three goals | Closed-passage all three goals | Mean action count (stable / changed) |
|---|---:|---:|---:|
| Explicit-history novelty | 50/100 | 31/100 | 127.28 / 140.26 |
| Relative greedy exploration | 82/100 | 77/100 | 102.34 / 112.12 |
| **Relative memory + neighbor diffusion** | **97/100** | **88/100** | **90.39 / 101.12** |
| Same diffusion with internal odometry frozen | 18/100 | 12/100 | 146.37 / 152.81 |

The diffusion controller improves over the paired history baseline by **+47 percentage points** stable and **+57 points** with passage closed. Descriptive paired bootstrap confidence intervals (20,000 draws resampling 100 episode seeds, fixed RNG seed 20261029) are **[36, 58]** and **[47, 67]** percentage points, respectively. They quantify only synthetic seed variation, not cross-domain generalization.

Each diffusion episode produces about **101,313 neighbor-value updates stable**, **120,372 changed**, averaged over seeds. These are *simulator message-update operations*, not actual network packets or measured joules. The learned cell memory covers about 60 discovered local locations per episode. This cost may be excessive if run continuously with hundreds of cells or realistic sensors; true process RAM, laptop energy and hardware efficiency remain unverified.

## Data provenance, integrity and limitations

The connected laptop was **offline** during this study. The experiment was run in an alternative local Python sandbox using a reference simulator reconstructed to match the versioned `SensorEpisode` and `Maze` from the GitHub repository. Five focused unit tests passed in that environment (sensor API, action-based relative integration against harness truth, reward-phase transition, memory ablation, deterministic seeded reproduction). Therefore **the new code has not yet passed the full repository suite or been executed on LANDITO-LAPTOP**. The GitHub branch carries six additional tests for execution against the actual repository modules.

Raw 800 paired per-seed episode outcomes, step counts and cell/message counts were generated in the sandbox; a compact seed-indexed evidence summary can be retained alongside the full standalone JSON in the experiment conversation. Reproduce exact repository modules using `python run.py local-value-cells --seeds 13001,...,13100 --rounds 10 --decay 0.92 --frontier-reward 2.5 --output results/runs/r1-local-value.json`.

**Supported:** A powerful non-privileged-observation controller can navigate synthetic multi-goal environments effectively using internally maintained relative motion memory and hand-authored local value diffusion. This reveals an information/algorithm gap in the earlier recurrent policy design: the previous network was not retaining the action history needed for robust state localization/exploration.

**NOT supported:** the RNN has learned odometry, the value messages are learned or physically decentralized, a system can navigate with unknown/noisy action directions, or the original R1 goal of learned autonomous recurrent navigation is solved. Relative coordinates and programmed value diffusion are still architectural scaffolding. This positive result is an **engineering ceiling / candidate teacher and ablation tool**, not proof of novel artificial intelligence.

## Next scientific step

1. **Learn rather than assume relative motion.** Train a tiny recurrent motor-state update from action/observation consistency and loop closure. Test rotated, noisy and drifting motor codes. Compare with this privileged *internal motion-model* baseline and frozen memory.
2. **Train local message weights** using goal and exploration outcomes, then ablate learned vs fixed diffusion and shuffled/zero messages. Compare at matched interaction and compute budget.
3. **Replace the central synchronous scheduler** with asynchronous cell-local neighborhood updates and measure communication stability/cost.
4. **Test generalization** to layout families with different wall topologies, sensory aliases, resets and dynamic mid-episode closures (these were not present here).
5. **Validate on laptop** once Desktop Commander reconnects, run the full regression suite, and measure CPU/RSS independently before reporting resource advantages.

All project text, comments and tests are written in professional contemporary English.
