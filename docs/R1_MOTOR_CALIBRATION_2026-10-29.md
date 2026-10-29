# R1 — Learning a Motor Command Code from Local Contact Feedback

**Date:** October 29, 2026 | **Hardware:** LANDITO-LAPTOP, Windows 11, Python 3.12, NumPy CPU-only

## Research goal

The previous R1 controller achieved 97/100 three-goal completions in a known maze, 88/100 after an upper-passage closure, and 95/100 in held-out same-family procedural layouts. But its relative odometry assumed the four motor commands were already aligned with the sensor's four directional wall channels.

This experiment removes **that command-to-direction assumption** while retaining the explicitly programmed relative coordinate basis and spatial value-diffusion rules. It is a *learned sensorimotor calibration* result, not an emergent coordinate-free spatial representation.

### Relevant research

- Tanaka & Imamizu (2025), *Sense of agency for a new motor skill emerges via the formation of a structural internal model*, Communications Psychology, https://doi.org/10.1038/s44271-025-00240-7 — an action-outcome internal model learned through exploration.
- Kamireddypalli et al. (2026), *BayesContact: Uncertain Pose Estimation via Visuo-Tactile Proposals and Simulation-Based Inference*, https://www.max-robotics.com/en/research/publications/cmrt2m85n001rqr1mghx9d2js — combining contact evidence with probabilistic hypotheses.
- Hu et al. (NeurIPS 2025), *Real-World Reinforcement Learning of Active Perception Behaviors*, https://papers.nips.cc/paper_files/paper/2025/hash/aa1b1a959c80086cba61d0fd66de412f-Abstract-Conference.html — active perception under hidden task-relevant state.

The experiment below does not replicate those methods.

## Problem specification and information access

A simulator secretly assigns each of four motor command IDs to one of four compass directions. It selects a random permutation of these commands per seed. The agent sees only:

- The existing locally measured four-wall pattern (compass-aligned) and neighboring-goal cue,
- The motor command it issued,
- A new **binary contact/bumper observation** after that motor command,
- Its own reward feedback and memories.

No global pose, destination coordinate, wall map, inverse motor mapping or correct-action label is supplied to the learner.

**Remaining engineered knowledge:** The learner knows that there are precisely four possible directions and that every motor code maps to exactly one direction. It enumerates all 24 permutations. The local wall channels are already compass-aligned; this is a powerful *sensor-alignment prior*. The Bayesian likelihood uses a predefined independent contact-noise rate. The original value-diffusion spatial controller is retained; odometry integrates a now-learned motor mapping but still uses a hard-coded two-dimensional displacement basis.

The training harness uses privileged random source-world resets to provide diverse wall observations, **but none of those reset coordinates are passed to the learner**. Each calibration source action is counted **in addition to** the 160 actions allowed per target navigation mission. No evaluation outcome is used to tune a seed's mapping.

## Method

For each observed tuple (wall pattern, motor command, contact/no-contact), accumulate a likelihood for each candidate motor permutation. The predicted collision for a candidate is the observed wall value at that candidate's selected direction. Multiply likelihoods (or equivalently, add log-likelihoods), given the hypothesized bumper noise. Select the maximum-posterior candidate after the training actions.

The trained mapping converts the spatial policy's chosen direction into a motor command. The policy continues to maintain its *self-constructed relative location* and to diffuse local utility among visited cells via a centralized Python scheduler and programmed neighbor update rule.

A calibrated motor code is not a learned dynamics model in continuous unstructured space. An unexplored motor code cannot be resolved without additional interactions in this setting.

### Controls

- **Learned command mapping:** 24-candidate Bayesian sensorimotor calibration from wall/contact pairs, then frozen evaluation.
- **Known mapping:** evaluation harness provides the true permutation *only* to the comparator; this is the oracle upper bound.
- **No calibration:** blindly assumes the fixed original command-to-direction mapping, which can corrupt internal odometry when motor codes are permuted.

All three controls face the same evaluation seeds, goals, and maze changes. The known mapping control excludes the calibration interactions from its own learning, so comparison of **total lifetime sample cost** must explicitly charge the learned method for its source probes.

## Development sample-efficiency curve (50 seeds 601–650)

Exact recovery of all four hidden motor-command correspondences:

| Contact sensor flip probability | 24 probes | 48 probes | 72 probes | 100 probes | 140 probes |
|---|---:|---:|---:|---:|---:|
| 0% | 66% | 96% | 100% | 100% | 100% |
| 8% | 56% | 88% | 98% | 100% | 100% |
| 20% | 44% | 80% | 92% | 94% | 98% |

The 12-probe and confidence results are available in `results/exploratory/2026-10-29-motor-calibration-sample-efficiency.json`. These curves are exploratory and do not constitute cross-domain generalization.

## Independent evaluation

Calibration source seeds are disjoint from the 50-seed development sweep. The normal stable/changed holdout uses seeds **9101–9200**, 100 separate motor permutations and calibration runs, with 100 independent source action probes per seed, 8% simulated contact-label flips. Each policy faces three sequential hidden goals in a maximum of 160 navigation actions.

The second holdout uses **50 procedural world seeds 9301–9350**, 140 calibration probes and 20% contact noise, with one obstacle removed and three added in each new map. The calibration uses the same source-world family; only the navigation test layout is changed.

### Procedural layout / 20% contact-noise result

| Method | Three-goal completion |
|---|---:|
| Learned motor mapping | **47/50 (94%)** |
| Known true mapping | **47/50 (94%)** |
| No calibration (identity guess) | **0/50 (0%)** |

The calibrator recovered the exact motor permutation for **50/50 seeds**, averaging posterior confidence 0.9953. These results were run on LANDITO-LAPTOP with the original 160-action mission limit. The uncalibrated policy's relative location becomes inaccurate because it assumes a wrong movement code.

Raw episodes: `results/audited/2026-10-29-learned-motor-laptop-procedural-noisy.json` (and local run in `results/runs/2026-10-29-motor-calibration-procedural-noisy50.json`).

### Standard-world / 8% contact-noise result

The independent 100-seed stable/changed evaluation **completed on LANDITO-LAPTOP**, including all 100 seeded permutations. Calibration recovered **100/100 correct motor mappings** after 100 contact observations per seed and 8% training contact flips (average posterior confidence 0.99909).

| Control | Stable 3-goal missions | Closed-passage 3-goal missions |
|---|---:|---:|
| Learned motor-code mapping | **93/100** | **93/100** |
| Known mapping oracle | **93/100** | **93/100** |
| Uncalibrated identity | **5/100** | **4/100** |

The learned mapping matched the known-mapping comparator exactly in both settings, under the fixed programmed cell-based navigation policy. The calibrated controller used a mean **96.03** stable and **96.33** changed navigation actions, *not counting the 100 separate source calibration interactions*. Total measured CPU process time for this 100-seed paired study was about **173.20 seconds**, which includes calibration and all three evaluation conditions per seed, not actual energy.

Raw per-seed records: `results/audited/2026-10-29-learned-motor-laptop-holdout.json`; original local run: `results/runs/2026-10-29-motor-calibration-holdout-100.json`.

## Engineering checks and limitations

A first exploratory 10-seed run incorrectly allowed 200 actions, and is intentionally marked **noncomparable** (`results/exploratory/r1-motor-code-pilot.json`). The code was corrected to a 160-action limit; the corrected development pilot (`results/exploratory/r1-motor-code-pilot-160.json`) recovered ten mappings and completed all ten missions per layout. Seven focused tests independently check sensor schema, exact motor mapping recovery across 24 permutations, contact effects, deterministic benchmarks, parameter rejection and action-count bounds.

No measured joules or physical electrical power. The control still has a centralized simulation update scheduler, discrete four-direction map, compass-aligned walls and a newly introduced bumper sensor. Therefore **R1's full goal of learned unstructured motor dynamics and autonomous distributed planning remains open**.

## Next falsifiable stage

1. Test **nonpermutation motor dynamics** (slippage, stochastic actuation, unknown movement lengths, orientation rotations), where the current 24-hypothesis calibration is insufficient.
2. Learn per-action local transition probabilities using a compact recurrent predictive model from proprioceptive/bump and sensory history, without direct coordinate supervision.
3. Introduce explicit hypothesis uncertainty and active probe selection under finite battery/action budgets; include a no-contact/no-compass ablation to expose observability boundaries.
4. Replace authored value propagation with trained cell-local message updates and test message shuffle/zeroing against matched compute/sample controls.
5. Collect real process CPU/RSS/power measurements before making claims of resource superiority.

All project code, documentation, tests, comments, CLI help and contributor artifacts use modern professional English.
