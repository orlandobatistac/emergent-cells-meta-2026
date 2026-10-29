# R1 Laboratory: Stochastic Motor Dynamics and Unexpected Motor Remapping

**Experiment date:** October 29, 2026

**Execution:** LANDITO-LAPTOP, Python 3.12, CPU-only

**Scientific status:** online motor-model adaptation studied; unstructured learned odometry and fully decentralized planning remain open.

## Question and prior work

Does a small learned local dynamics model enable useful control when actions sometimes fail, sensory movement cues are noisy, and a motor-command-to-direction correspondence can change without warning?

Related contemporary research:
- Altawaitan, *Online Model Learning and Control for Autonomous Robot Navigation* (2026), https://escholarship.org/uc/item/6c03534v — motor models must adapt when actual platform behavior deviates from authored physics.
- Paudel, Xiao & Stein, *Multi-Strategy Deployment-Time Learning and Adaptation for Navigation under Uncertainty* (CoRL 2024 proceedings, published 2025), https://proceedings.mlr.press/v270/paudel25a.html — adaptation during deployment and the need for matched performance controls.
- *Bayesian autoregressive online change-point detection with time-varying parameters* (2025), https://doi.org/10.1016/j.cnsns.2024.108500 — probabilistic detection of changes in dynamics.

This experiment is not an implementation or reproduction of these works.

## Pre-existing control and experimental integrity

The earlier R1 baseline, `local_value_cells.py`, builds its own relative coordinates by integrating *known* four-direction motor commands and propagates authored local cell utility over previously visited neighbors. It was reproduced on the laptop with **97/100 stable**, **88/100 closed passage**, **95/100 procedural** three-goal completion. It does not use simulator pose, goal coordinates, a global map or BFS, but its motor-vector basis, compass-aligned four-wall sensor and synchronous value propagation are **programmed**.

This new study keeps the same 9x9 task, three hidden sequential goals and **160 navigation actions** per mission. For each matched episode seed, every method receives the same initial world/goal generation. A simulator-only reference retains private pose and true action results for scoring; agent code never accesses them to choose actions. Separate calibration interactions are counted **in addition** to navigation actions. No energy claim is made without physical power data.

## A. Slippage/no-op dynamics with partially observed movement

Source file: `src/emergent_cells/experiments/stochastic_motor.py`

For each seed, four hidden command-specific movement probabilities are independently drawn from **0.45, 0.60, 0.75 or 0.90**. An attempted movement may result in a stationary no-op. The observation interface includes the four local wall bits, adjacent target cue, checkerboard floor cue, reward and a **new noisy binary proprioceptive movement tick**.

To make inference nontrivial, the tick flips independently with **20%** probability and the floor cue flips independently with **15%** probability. The floor cue would otherwise reveal the movement/no-op bit perfectly in a checkerboard world; the added noise is a critical information-boundary control. The agent cannot see actual position, achieved displacement, true motor success rates or goal coordinates.

### Model

For each command, collect noisy ticks on wall-free attempts and estimate its Bernoulli true movement rate with a corrected likelihood and smoothing (a parameter learned from observed local sensor outcomes). At test time, a predefined Bayesian fusion rule combines that learned prior with noisy tick and floor-parity changes.

Separate spatial-memory consistency adds a comparison of **remembered** wall signatures at previously explored neighbor cells. A changed current wall signature implies movement in the static world; a remembered destination mismatch can suggest a no-op. That check never queries the simulator's hidden map, but its local consistency logic is authored.

The controller still uses a **hard position commitment** at every step. This causes compounding location errors when both movement cues fail. It is **not** a probabilistic SLAM/belief-state filter. The source calibration world may reset to random hidden simulator locations (privileged coverage in the harness), but the learner receives only local wall/tick observations.

### Controls

- **Always move:** assume a movement succeeds whenever the local wall sensor says the direction is open.
- **Tick only:** infer movement exclusively from the noisy proprioceptive bit.
- **Floor only:** infer movement exclusively from the noisy floor-parity difference.
- **Learned Bayesian fusion:** combine both sensors using rates learned during separate local calibration.
- **Known rates:** same Bayesian rule with true per-command probabilities, an evaluation-only parameter oracle.
- **Learned local-map fusion:** learned Bayesian rule plus walls observed on previous visits.
- **Known rates + local-map fusion:** oracle-probability counterpart.

**Pilot:** 10 seeds 101–110 first showed movement prediction improved from approximately 69% (always move) to 84% (learned sensor fusion). Nevertheless, mission completion did not consistently improve. A second 15-seed development pilot (111–125) indicated that adding previously observed wall consistency improved motion classification to roughly 86%, while multi-goal navigation remained difficult. Both raw pilots are saved under `results/exploratory/`.

Independent study: **35 seeds 32001–32035** with no prior use for model tuning. All seven methods are evaluated on stable and passage-changed worlds with the same seeds and action cap.

### Independent slippage results

**Completed:** 35 seeds, 32001–32035, each evaluated in the stable and corridor-closed maze using 160 navigation actions maximum, plus 120 separate source-calibration actions.

| Method | Stable three-goal success | Passage-closed success | Mean motor-outcome accuracy, stable / changed |
|---|---:|---:|---:|
| Always assume motion | 15/35 (42.9%) | 8/35 (22.9%) | 66.3% / 65.6% |
| Noisy movement tick only | 16/35 (45.7%) | 13/35 (37.1%) | 80.3% / 80.2% |
| Learned-rate tick+floor fusion | 18/35 (51.4%) | **19/35 (54.3%)** | 82.7% / 82.6% |
| Learned-rate fusion + observed wall-map consistency | **22/35 (62.9%)** | 17/35 (48.6%) | **87.4% / 86.2%** |
| Known-rate oracle + wall-map consistency | 20/35 (57.1%) | 16/35 (45.7%) | 88.3% / 86.6% |

The average absolute error of four learned command success probabilities was **0.1249** on the 0–1 scale. Importantly, the known-rate comparator is an oracle **only for motor success probabilities**, not for actual random motor outcomes or maps. Its mission completion can be below the learned model for a small paired test sample because different policy predictions change its subsequent visited-state trajectories; this is not proof learned rates are superior to true rates.

Paired seed-level bootstrap (10,000 draws, RNG seed 102926):
- Stable learned-map fusion vs always-move: **+20.0 points**, descriptive 95% interval **[−2.9, +42.9] points**; interval includes zero.
- Passage-closed learned sensor fusion vs always-move: **+31.4 points**, interval **[+11.4, +51.4] points**.

Thus the independent data support a reliable **movement-prediction advantage**, a promising but uncertain stable-maze completion gain, and a stronger changed-maze navigation gain. In a harness-only diagnostic, the mean final relative-position error was **19.11 / 21.89 grid units** for always-move in stable/changed scenes, **10.29 / 11.66** for learned sensor fusion, and **7.00 / 6.91** for learned local-map fusion. These relative-position errors are **evaluation metrics, never supplied as training labels**. Remaining errors accumulate because the controller maintains just one estimated position, rather than a posterior over plausible locations.

Full source: `results/audited/2026-10-29-stochastic-motor-laptop-holdout.json`, including all 490 matched episodes, calibrated parameters, actions and harness-only terminal pose-error scores. The batch used approximately **134.42 process CPU seconds** across all methods and seeds. Not wall-power.

## B. An unannounced discrete motor-code remapping

Source file: `src/emergent_cells/experiments/switching_motor.py`.

Initial motor mapping is learned from **100 separate source actions**, as in R1's previous discrete calibration. The simulator secretly changes the command permutation **after 35 navigation actions**. It does not signal the change. The agent sees only compass-aligned wall measurements, local target cue, the issued command, and a binary wall-contact signal with **4% noise**. It retains the same 24-permutation hypothesis class used in the earlier study.

A **windowed Bayesian sensorimotor detector** scores all 24 mappings using only the last 28 action/wall/contact observations and triggers a remapping when the alternative has a log-likelihood margin above 4. The learned controller can then discard its possibly corrupted relative spatial memory and restart its local coordinate origin. This reset never uses simulator position.

### Controls

- **Online recalibration + local map reset:** learn changed command mapping and restart self-constructed relative memory.
- **Online recalibration without map reset:** same evidence rule, but retain the potentially corrupted local spatial map.
- **Frozen original mapping:** fail to adapt after an unannounced command change.
- **Privileged known switch:** simulator truth determines exact switch time/new code; it is an **upper bound only**, not autonomous behavior.

Development pilot: 15 seeds 101–115. Online + reset completed **9/15** missions stable and **9/15** changed; frozen mapping **0/15** for both; known-switch control **12/15** stable and **13/15** changed. Mean first detection delay was approximately 11–17 navigation actions when recovery was detected. A later code change distinguished genuinely correct recovery from merely switching to some other mapping and counts incorrect remaps.

Independent validation: 50 unseen seeds **31001–31050** in both stable and passage-closed worlds. See `results/audited/2026-10-29-switching-motor-laptop-holdout.json` for per-seed completion, adaptation, delay, wrong-command count and false-recovery metrics.

### Independent remapping results

**Completed:** 50 seeds, 31001–31050, in original and upper-passage-closed mazes. Every evaluation episode has a hidden mapping switch after the 35th attempted action; the motor mapping was initially calibrated using 100 additional source interactions.

| Method | Stable three-goal success | Passage-closed success |
|---|---:|---:|
| Frozen initial motor code | **2/50 (4%)** | **2/50 (4%)** |
| Online recalibration + reset local memory | **33/50 (66%)** | **33/50 (66%)** |
| Online recalibration, retain old relative map | **36/50 (72%)** | **31/50 (62%)** |
| Known exact switch/mapping (oracle) | **45/50 (90%)** | **44/50 (88%)** |

For online calibration with map reset, the recovery detector eventually reached the true new permutation in **88% stable** and **96% passage-closed** of runs. Among successful detections, the mean lag from the hidden switch was roughly **36.8** and **35.8 actions**, respectively. The detector also made on average **1.86 / 1.36 incorrect or premature remapping transitions** in these two conditions. This explains the remaining gap to the privileged upper-bound controller. Interestingly, retaining the old map sometimes improved success because resetting discards useful explored-space coverage; a reset is not unconditionally beneficial.

Against the frozen controller, the paired improvement from online-reset is **+62 points** in both scenarios; descriptive seed-resampling 95% intervals are **[+48, +76] points** each (10,000 draws, RNG seed 102926). This is a large and directly validated online-adaptation effect, not evidence of spontaneous motor geometry discovery.

Full source: `results/audited/2026-10-29-switching-motor-laptop-holdout.json` with 400 matched episodes, true simulator mappings retained only for scoring, detector-event counts and delay metrics. The study consumed roughly **113.22 process CPU seconds**, not measured energy.

## Engineering validation

The full local regression suite passed **162/162 tests** on LANDITO-LAPTOP, including 8 new stochastic-motor tests and 5 new remapping tests. These exercise sensor schema isolation, simulated slip extremes, noisy-probability calibration, posterior fusion, deterministic paired runs, unannounced switching and local change-point detection. The research branches are CPU-only and require no PyTorch, GPU or external API.

## Conclusions and remaining scientific gap

**Learned from experience:** command-conditioned slip/no-op probabilities from noisy local movement ticks and a finite-window change in sensorimotor command mapping after an unexpected switch. Both use local observations, not simulator coordinates or hidden goal locations.

**Authored priors remain:** compass-aligned four-wall sensors; four-command directions and known 24-permutation class for remapping; Bernoulli no-op slip class; known sensor-noise probabilities; Bayesian fusion and change detection; relative two-dimensional position integration; frontier incentives and local-value propagation under a centralized Python scheduler.

Navigation under repeated ambiguous slip still needs a **multi-hypothesis belief state** (particle filter or equivalent local predictive memory), online loop closure, and learned message weighting. Even correct command-probability estimates cannot prevent global relative pose drift if each uncertain action is collapsed prematurely to one guessed outcome.

Future research must compare navigation, learned transition accuracy and compute at equal lifetime sample budgets. Source calibration probes are not free. Energy use in joules and actual physical power are **not measured**.

## Reproducibility

```powershell
python -m unittest discover -s tests -q

python run.py stochastic-motor --seeds 32001,32002,32003,32004,32005 --calibration-actions 120 --tick-error .20 --floor-error .15 --rounds 10 --output results/runs/stochastic-reproduce.json

python run.py switching-motor --seeds 31001,31002,31003,31004,31005 --switch-at 35 --window 28 --margin 4 --rounds 10 --output results/runs/switching-reproduce.json
```

All work is written in modern professional English; Spanish is used only in investigator discussion.
