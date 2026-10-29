# Research Roadmap — From Privileged Simulation to Locally Learned Cellular Intelligence
**Project:** Emergent Cells Lab | **Opened:** 2026-10-28 | **Status:** active research plan

## North star

Build a modular architecture whose cells collectively learn useful predictive representations, coordinate and pursue multi-step goals on ordinary laptop hardware **without** experimenter-provided world coordinates or a centralized scripted planner.

The target is falsifiable generalization and resource efficiency, **not** a premature claim of AGI. This document supersedes the assumption that more Boolean-world accuracy alone constitutes progress.

### What we already have

- Local reward-modulated learning and learned binary messages, with causal ablations.
- Dynamic context-specific memory (including documented noise-triggered failure cases).
- Reward-conditioned recombination of response policies across related Boolean functions.
- A goal-sequenced navigation prototype with locally stored transition entries, but exact coordinates and programmer-supplied BFS. The prototype is a valuable **upper-bound engineering comparator**, not proof of cellular autonomous navigation.

### Two dependencies to remove

1. **Representation oracle:** the agent receives exact position, target coordinates, orientation/world structure, or a task ID that should instead be inferred from sensory experience. The simulator may retain hidden ground truth **only for reproducibility and scoring**.
2. **Planning oracle:** the agent calls global BFS, shortest-path computation, an omniscient controller, or a hand-authored transition graph that solves the mission. A central scheduler may run independent cells but may not pick their actions or route messages based on the hidden world.

Do not confuse locally implemented rules (necessary for learning algorithms) with task-specific solutions hardcoded for a benchmark.

## Scientific foundation

Three complementary, non-exclusive experimental hypotheses:

- **H1 — Predictive representation:** local observation/action history can form a reusable compact state that disambiguates repeated-looking places; test recurrence, contrastive predictive learning and local predictive-map updates.
- **H2 — Communication as memory:** sparse signals between independent learners convey history-dependent information beyond the immediate sensory input; test causal channel interventions, message dropout, bit counts and mixed observational access.
- **H3 — Local planning:** learned transition/goal value can propagate through bounded neighborhood interactions without global shortest-path access; test asynchronous value propagation, learned neighbor credit and distributed goal-conditioned representations.

Sources to investigate—not to claim as reproduced:
- Cognitive map learning under partial observability with local prediction (2025): https://www.sciencedirect.com/science/article/pii/S2667305325000778
- Causal-JEPA and structured partial observation (ICML 2026): https://proceedings.mlr.press/v306/nam26c.html
- Learning Disentangled Multi-Agent World Model for Decentralized Control (ICML 2026): https://proceedings.mlr.press/v306/xue26e.html
- Dreamer-CPC history-aware latent messages (2026): https://arxiv.org/abs/2607.19809
- Efficient Communication with Skill Neurons (TPAMI 2026): https://pubmed.ncbi.nlm.nih.gov/42678856/
- Decentralized Collective World Model for Emergent Communication and Coordination (ICDL 2025): https://ieeexplore.ieee.org/abstract/document/11204457/

## Execution status

- **R0: in progress.** The initial sensor-only environment, simple reactive/random controls and six contract tests are implemented. Its **negative baseline** (2/10 stable three-goal missions for reactive, 0/10 for random; zero complete missions after an alternate wall closure) is documented in [the R0 report](SENSOR_ONLY_WORLD_R0_2026-10-28.md). This is not evidence of state learning or distributed planning; more anti-leakage tests and agent isolation are pending.
- **R1: recurrent prediction and first closed-loop policies evaluated; full milestone open.** A 48-dimensional recurrent state improved sensory predictions in ten-seed tests, but task-directed behavior was harder: with sparse local goal detection, explicit-history exploration completed 52/100 original and 36/100 changed missions, while recurrent-state novelty completed 37/100 and 33/100. A **separate informative-sensor benchmark** using an engineered noisy target-distance signal and reward-trained local action preferences completed 100/100 and 50/100, but recurrent-memory erasure was not detrimental. See [R1 recurrent state report](R1_TRAINABLE_RECURRENT_STATE_2026-10-28.md), [predictive-context study](R1_PREDICTIVE_CONTEXT_2026-10-28.md) and [closed-loop goal learning](R1_CLOSED_LOOP_GOAL_LEARNING_2026-10-28.md). Still needed: a learned goal policy that benefits causally from recurrent state *without* a distance-like target oracle, tougher unseen worlds, matched-budget baselines and cell-to-cell messaging.
- **R1: reward-trained recurrent decision memory verified in a separate diagnostic; original maze task remains open.** A 12-unit recurrent actor learned decisions from terminal rewards using an initial cue, achieving 100% at eight/twelve-step delays and 85.19% at sixteen steps, versus approximately 50% after memory erasure (20 validation seeds). However, a separate actor trained in the original sparse-target 9x9 maze achieved 18% original / 11% changed completion, below history novelty (50% / 39%). Increasing recurrent centroid resolution did not provide a consistent benefit. This **does not establish recurrent goal navigation**; see [R1 reward-trained policy report](R1_REWARD_TRAINED_POLICY_2026-10-28.md). Next: action-relevant state objectives, memory rewriting, and non-oracle multi-step control.
- **R1: local-value memory baseline established (Oct 29), but learned motor-state and policy remain open.** An isolated sandbox experiment using programmed action-relative odometry plus neighbor-only value diffusion solved 97/100 stable and 88/100 closed-passage missions, plus 95/100 on a separate set of 100 procedural layouts (paired history 50/100 and 31/100). This used no externally supplied global pose/map/goal route, but it **did use engineered relative motion and a centralized Python update scheduler**; not a demonstration of learned recurrent navigation or physical decentralized cells. It has not yet run on the connected laptop. See [R1 local-value cells report](R1_LOCAL_VALUE_CELLS_2026-10-29.md). Next: learned action-state integration and reward-trained message propagation, with odometry/message ablations and new layout families.
- **R2–R5: open research tasks.** Their completion is not asserted.

## Execution order and GitHub tickets

| Stage | Remove or learn | Executable deliverable | Acceptance evidence | Ticket |
|---|---|---|---|---|
| **R0** Sensor firewall | Hide global coordinates, world map and task IDs from agents | Partially observable world and non-oracle controls; strict API tests | Deterministic sensory episodes, aliased states, no oracle data in agent inputs | [#20](https://github.com/orlandobatistac/emergent-cells-meta-2026/issues/20) |
| **R1** Learn latent state | Replace coordinate lookup with compact memory of observation/action history | Two or more predictive state learners vs recurrent/tabular baselines | Better unseen transition prediction under ambiguous observations | [#21](https://github.com/orlandobatistac/emergent-cells-meta-2026/issues/21) |
| **R2** Learn messaging | Replace shared global information with locally learned messages | 1–8-bit message passing and causal channel ablations | Message benefit under matched compute; costs/robustness reported | [#22](https://github.com/orlandobatistac/emergent-cells-meta-2026/issues/22) |
| **R3** Decentralize planning | Replace global BFS with local goal-reward/value propagation | No-BFS action policy operating over learned local representations | Held-out 3-waypoint completion above controls; oracle ratio measured | [#23](https://github.com/orlandobatistac/emergent-cells-meta-2026/issues/23) |
| **R4** Transfer without task IDs | Unannounced topology changes and new worlds | Multi-goal, dynamic maze and distribution-shift suite | Forgetting, adaptation latency, cross-seed confidence intervals | [#24](https://github.com/orlandobatistac/emergent-cells-meta-2026/issues/24) |
| **R5** Efficiency | Measure real laptop performance and scaling | 64/256/1024-cell profiler and strong matched-budget baselines | CPU, RSS, wall time, bandwidth and quality trade-offs | [#25](https://github.com/orlandobatistac/emergent-cells-meta-2026/issues/25) |

### First executable task: R0

Build a **new** agent-facing world interface:
`reset(seed) -> local_observation`; `step(local_action) -> (next_observation, private_reward, done)`.
The agent must not receive the environment object, internal coordinates, the goal coordinate, global map or any oracle route. Instead it receives egocentric wall/texture/short-range goal/previous-reward observations. Multiple positions must deliberately share the same observation to block trivial identity lookup.

The harness alone may inspect hidden truth for metrics and compare against the existing privileged planner as an **oracle upper bound**. Begin with random/reactive baselines, record incomplete missions and failure cases as results. Do not add a new centralized planner to make R0 performance look positive.

Next, R1 explicitly attempts to learn enough state to beat those baselines. R2 connects that latent state with the earlier learned-message code, then R3 removes the remaining planning oracle. Failures become focused follow-up experiments, not pretexts to reinsert privileged data.

## Measurement protocol and acceptance gates

- **Scientific isolation:** disjoint development and holdout seeds/worlds; save protocol, code SHA, machine specs, raw JSON, negative outcomes and comparisons.
- **Primary quality:** full three-target mission success rate, action count versus privileged optimum, recovery after a hidden obstacle shift, and per-seed variability.
- **Representational learning:** held-out one- and multi-step prediction accuracy/calibration; explicit sensor-aliasing comparisons; no supervised ground-truth coordinates.
- **Communication causal tests:** fixed policy with normal vs shuffled/disabled messages and matched randomness; bits per episode and packet drops.
- **Memory/efficiency:** memory-state bytes, process peak RSS, CPU seconds, wall time, number of samples and messages; report electrical watts/joules only if actually measured.
- **Stronger baselines:** memoryless reactive/random, the existing coordinate+BFS privileged oracle (evaluation only), coordinate-free tabular recurrent method, and compact model-based or RL baselines with matched **total** experiences.
- **Go/no-go:** do not claim a dependency eliminated unless the corresponding API/behavioral ablation confirms its removal. Investigate failures before increasing model complexity, but do not reintroduce the oracle to manufacture successes.

### Resource-aware implementation constraints

- Python-first, CPU-first, modular and testable on LANDITO-LAPTOP; GPU optional.
- Retain independent cell-local state and bounded neighborhood messaging.
- Keep initial sensor/latent dimensions and message width compact; measure rather than assume lower energy use.
- If a decentralized algorithm underperforms, compare against a legitimate centralized baseline **without hiding the difference**.

### Hackathon scope vs research horizon

For the current Meta Hackathon submission, an honest demo can showcase a reproducible R0/R1 boundary (and R2 only if supported by actual observations). **R3–R5 remain explicit research milestones**, not invented completed features. Do not trade away scientific integrity for an interface or demo deadline.

## Repository engineering policy

All project code, comments, tests, documentation, issues and AI-contributor artifacts are in professional contemporary English. The human conversation can remain in Spanish.

Roadmap owners: research, experimentation and code agents. Each ticket requires an isolated implementation, tests and saved results. The previous experiments and raw datasets remain immutable. Work in small reviewable increments; promote only reproducible conclusions to `main`.
