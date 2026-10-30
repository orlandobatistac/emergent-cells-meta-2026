# Emergent Cells Studio: Event-Driven Local Intelligence, Belief-State Investigation, and Usable Laptop Product

**Research date:** October 29, 2026

**Execution:** LANDITO-LAPTOP, Windows 11, 16 GB physical RAM, 16 logical CPUs, NVIDIA RTX 3050 Ti (4 GB VRAM)

**Scope:** CPU-first synthetic navigation; an optional local foundation-language-model interface; no claim of AGI.

## Product delivered

The standalone local web application **Emergent Cells Studio** is now part of the repository:

- Run `python -m pip install -e ".[studio]"`, then `python run.py studio --host 127.0.0.1 --port 8765`.
- Open `http://127.0.0.1:8765/` from the laptop.
- Select a reproducible seed, an original/changed world, and either an **event-driven cell network** or the older **synchronous diffusion** comparator. A third choice runs the experimental **stochastic motor belief memory**.
- Start/reset the three-goal mission; advance one step, twenty steps, or finish up to 160 allowed steps.
- Inspect a live, **self-constructed relative-cell map**, predicted agent location, discovered cells, sequential targets reached, recent motor actions, belief-state count, and local message counts. The visualization does not expose hidden world coordinates or hidden target positions to the agent.
- Use the integrated chat to issue **bounded deterministic offline agent commands** (`status`, `run 20`, `reset`). For unconstrained research questions, a separate preinstalled **Ollama Qwen 2.5 Coder 7B** model can optionally generate explanations locally; this is a conventional existing pretrained foundation model and must not be attributed to cells. This experimental Ollama response path was not included in the deterministic scientific model checks.

The application binds only to `127.0.0.1`; neither chat messages nor model output execute arbitrary code, shell commands or file edits. There are no cloud API credentials and no real-world robot/machine control. Laboratory sessions are single-user, local-only, and not authenticated for multi-user exposure.

## Key scientific advance: bounded messages instead of global sweeps

The earlier baseline `LocalValueCells` used ten synchronous value-propagation rounds over **every discovered cell after each action**, with a Python loop scheduling all cells. That achieved high navigation success but many repeated value computations.

New `EventDrivenCells` gives each observed cell its own wall observations, local visit counts, local value, and neighboring-message cache. It updates when its own local state changes or when a directly observed neighbor sends a value message. Only a **bounded number of queued direct-neighbor messages** are delivered per primitive action (80 in the final study). A simple deterministic Python scheduler carries messages; it does not compute a global path or read distant cells to update a cell's value. Local value and frontier-reward formulas are still **engineered and not learned**.

### Independent paired validation: 100 untouched seeds per scenario

Original/closed-passage seeds 41001–41100. Procedurally modified maze seeds 42001–42100. All three controls receive the same seeded simulator conditions, three hidden goals and a maximum 160 primitive actions.

| Scenario | No message passing | Bounded event-driven messages | Global synchronous sweeps |
|---|---:|---:|---:|
| Original 9x9 maze | 77/100 | **92/100** | **96/100** |
| Upper passage closed | 63/100 | **91/100** | **91/100** |
| 100 new same-family maze layouts | 78/100 | **94/100** | **96/100** |

Mean neighbor-message deliveries versus global value-update operations:

| Scenario | Event-driven delivered messages | Synchronous neighbor value-update operations | Approximate ratio |
|---|---:|---:|---:|
| Original | 2,368 | 95,699 | **40.4x fewer** |
| Closed passage | 2,657 | 105,690 | **39.8x fewer** |
| Procedural | 2,392 | 98,465 | **41.2x fewer** |

The two quantities measure *related but not identical work*: delivered scalar neighbor messages versus neighbor value inspections in a synchronous Python scheduler. They are **not actual network packets**, hardware energy, or fully distributed multi-process execution. The event-driven system sacrifices only four percentage points of mission completion on the original map and two on procedural maps while maintaining parity on the passage-changed condition. Crucially, **zero-message ablation loses 15, 28 and 16 percentage points**. Descriptive paired seed-bootstrap 95% confidence intervals are **[+6,+24] points** in the original maze, **[+17,+38]** with a closed passage and **[+8,+25]** on procedural worlds (10,000 seed-level resamples, RNG seed 20261029). This supports a causal role for local value messages under the fixed engineered exploration rule, not general learned intelligence.

Raw final protocols:
- `results/audited/2026-10-29-event-cells-bounded-holdout.json`
- `results/audited/2026-10-29-event-cells-bounded-procedural.json`

### Measured laptop CPU cost (distinct 30-seed sample)

An additional independent, single-machine profile used 30 sequential missions per controller (seeds 44001–44030, original maze), measured by Python's `time.process_time()`:

| Controller | Mean process CPU seconds per mission | Full mission success |
|---|---:|---:|
| Event-driven local messages | **0.00990 s** | 27/30 |
| No-message ablation | 0.00365 s | 28/30 |
| Global synchronous sweeps | **0.10677 s** | 29/30 |

The event controller consumed **about 10.8x less measured process CPU time** per episode than the synchronous comparator in this microbenchmark. The time is below 0.1 seconds per mission and Windows CPU timing resolution can quantize individual runs; the experiment was executed in a fixed order, not counterbalanced, so its ratio is indicative rather than hardware-invariant. No physical electrical power, GPU energy or actual network-packet cost was measured. The 30-seed success ordering differs from the stronger paired 100-seed experiments and should not be used to assert a no-message advantage.

Raw: `results/audited/2026-10-29-event-cpu-profile.json`.

Earlier results generated before correcting the per-action message-budget enforcement are kept in `results/exploratory/` for provenance; they are **not** the canonical final event-driven benchmark.

## Reward-driven tuning of shared message coefficients: negative efficiency result

To test whether task rewards can improve the authored event-message controller, an additional black-box search evaluated **25 candidate pairs** of value-decay and frontier-reward coefficients on **25 separate training seeds** (51001–51025), each under the original and closed-passage world. The choice maximizing completed missions, intermediate goal reward, action economy and message budget was **decay 0.92 / frontier 1.75** (default decay 0.92 / frontier 2.5).

Independent evaluation on **100 unseen seeds 52001–52100** in each of three scenarios yielded:

| Scenario | Learned shared coefficients | Original fixed coefficients | Zero-message control |
|---|---:|---:|---:|
| Original maze | 92/100 | **93/100** | 79/100 |
| Closed passage | 87/100 | 87/100 | 54/100 |
| Procedural layouts | **94/100** | 92/100 | 78/100 |

The learned coefficients also caused **more** local message deliveries: approximately 2,807 vs 2,384 original; 3,277 vs 2,818 changed; 2,723 vs 2,402 procedural. There is **no consistent learning benefit** in completion or resource efficiency. Therefore **the product retains the authored default coefficients**. This is a crucial negative control: tuning two scalar weights with mission reward does not constitute discovering local communication rules or decentralized learning.

Raw 25-candidate training comparison and 900 independent evaluation episodes: `results/audited/2026-10-29-event-reward-weights.json`.

## Simulated loss of one cell's local memory

A second holdout study used **100 novel seeds 45001–45100** on each original/passage-closed world. At the 30th primitive action (if the mission was still active and a noncurrent cell existed), the test harness deleted one previously visited cell's remembered local wall pattern, value state and queued messages. Direct neighbors treat the missing cell as unexplored and can relearn it upon revisiting.

| Scenario | Event-driven, no failure | Event-driven, one simulated cell-memory loss |
|---|---:|---:|
| Original maze | 89/100 completed | 94/100 completed |
| Closed passage | 83/100 completed | 88/100 completed |

The fault was actually triggered in **93/100 original** and **98/100 changed** runs; some missions finished before the action-30 injection point. Lost-cell observations were subsequently reconstructed in **69/100** and **74/100** runs, respectively. The experimental controller continued operating when a cell record disappeared.

**This is not the shutdown of another laptop, process, network server or real distributed computation node.** It models loss of *one local memory record* within the same Python process. The numerical increase in mission completion under failure is not evidence failures improve intelligence; perturbing stale local value/visit memory can alter exploration and this experiment did not test statistical significance of that increase.

The live Studio includes a **Simulate loss of one cell** control which performs the same limited fault injection and displays resulting cell/message state.

Source: `src/emergent_cells/experiments/event_cell_fault.py`; 400 per-episode records: `results/audited/2026-10-29-event-cell-fault-holdout.json`.

## Secondary investigation: multi-hypothesis navigation under motor slip

A new `BeliefExplorer` maintains a bounded set of candidate relative coordinates and candidate **locally observed wall maps**. Each issued action forks a belief into movement/no-op outcomes according to per-command slip rates learned on separate sensor-only source trials. Noisy movement ticks, noisy floor-parity changes, wall-map loop consistency and an authored likelihood model reweight candidate histories. The beam uses a maximum width and does not receive simulator absolute pose or goal coordinates.

An early exploratory 24-hypothesis consensus policy suffered **more mission failures and >10x more local value operations** than its one-hypothesis baseline. This is a meaningful negative result: multiplying uncertain positions and aggregating a separate exploration plan for each is not automatically useful, and may harm both accuracy and compute efficiency.

A second 8-hypothesis test used two value-message rounds and separately evaluated (a) **weighted consensus** across candidate local plans and (b) using only the maximum-posterior candidate's plan. This reduced cost compared with 24 hypotheses, but the development pilot remained mixed.

### Independent belief filter validation: 40 unseen seeds, 43001–43040

All methods share the same slip generation, local contact/parity noise, source calibration and target-sequence generation per seed.

| Belief policy | Stable three-goal success | Passage-closed success | Mean local value-message work (stable) |
|---|---:|---:|---:|
| Single-hypothesis commitment | 6/40 (15.0%) | 9/40 (22.5%) | 30,697 |
| **Eight-hypothesis weighted consensus** | **17/40 (42.5%)** | **13/40 (32.5%)** | 151,288 |
| Eight hypotheses, best-only action plan | 10/40 (25.0%) | 12/40 (30.0%) | 31,434 |
| Eight-hypothesis consensus with true rate parameters | 9/40 (22.5%) | 11/40 (27.5%) | 174,013 |

The weighted-consensus beam improves stable success over single-hypothesis by **27.5 percentage points**, with a descriptive 95% seed-resampling interval **[+10, +45] points** (10,000 draws, RNG seed 20261029). The changed-world advantage is **+10 points**, interval **[−10, +30] points**, not conclusive. Computing a plan per candidate increases local value-update work approximately **4.9x** compared with hard commitment.

The true-rate control receives exact *slip probabilities*, not realized motion labels; it can underperform the estimated-rate method in a small finite sample after different action trajectories. The Studio uses **best-only** belief action selection by default to control CPU latency, rather than the more expensive consensus. Even the best held-out 42.5% completion is substantially below deterministic-grid event-cell performance. The belief state remains an experimental feature.

Raw: `results/audited/2026-10-29-belief-holdout.json`, with 40 seed-level outcomes in each original and passage-closed condition for all four controllers.

### Scientific distinction

- The **belief** is a bounded engineered Bayes/beam filter, not an end-to-end learned recurrent representation.
- Movement direction and wall-sensor orientation are authored and consistent; the filter can still lose the true path under aliased observations.
- Cell-local messages remain programmed, not tuned from reward; Python still centrally schedules message delivery.
- A useful empirical controller is not evidence of a new language foundation model. The optional Ollama model is an independent existing model with a different architecture.

## Independent research context

- Kirchner, Purschke & Knoll, *A survey of world models for physical AI with uncertainty representation and control* (September 2026), https://link.springer.com/article/10.1007/s44163-026-02122-1 — emphasizes uncertainty-aware closed-loop world models.
- Chen et al., *PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making* (ICML 2026), https://proceedings.mlr.press/v306/chen26i.html — supports testing multiple candidate world hypotheses against calibrated observations.
- Omotuyi & Kumar, *Scalable decentralized controllers for cohesive navigation using graph reinforcement learning* (2026), https://www.sciencedirect.com/science/article/pii/S0921889026003398 — motivates locality, communication boundedness and transfer comparisons.

These works motivate testing but the reported code **does not reproduce their models**.

## Acceptance criteria and honest limits

The delivered application is usable today for locally running, observing and talking to an agent-controlled synthetic exploration world. Its action policy runs on an ordinary CPU, and the scientific messages are explicit. It does **not** implement a ChatGPT-class emergent foundation model, learn general language from scratch, discover an arbitrary physical substrate, independently schedule across PCs, or demonstrate true hardware-independent intelligence. Those original long-term aims remain **unproven**.

New falsifiable next gates: (1) learned cell-specific message weights and independent process/node failure tests; (2) probabilistic localization that beats hard commitment on unseen worlds at matched interaction and CPU budgets; (3) learned recurrent goal policy exceeding authored local explorers; (4) a sustained user-facing agent with robust constrained tools and continual memory; (5) actual electricity and process-RSS benchmarks before low-energy claims.

## Verification

A full regression test suite and GitHub Actions scientific tests are part of the merge gate. New tests exercise live API semantics, boundary validation, deterministic local simulation, limited command interpretation, stochastic belief constraints and locally bounded direct-neighbor message updates. Optional FastAPI TestClient tests run only when studio dependencies exist, avoiding a hard dependency for numerical-only environments.

User-visible model interactions are bounded to the simulator and a separate local Qwen explanation model; no code-generation response is treated as proof of cellular language intelligence.
