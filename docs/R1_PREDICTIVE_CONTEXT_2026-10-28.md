# R1 Pilot — Predictive State from Local Observation-Action History
**Date:** October 28, 2026 | **Status:** partial R1 milestone; independent synthetic validation.

## Question

Does remembering several previous local sensor readings and actions improve predictions when two hidden maze locations look identical to an agent?

The immediate task is **self-supervised prediction**, not goal navigation. A model observes only four egocentric wall indicators, a two-valued floor signal and a chosen action, then predicts the next wall/floor signature. World position, destination coordinates, wall map, shortest path, and task ID are **not used as learning inputs or labels**. The benchmark harness alone owns the hidden maze and reset logic.

## Research rationale

- Wang et al., *Learning Belief Representations for Partially Observable Deep RL* (ICML 2023), motivates history-conditioned belief representations: https://proceedings.mlr.press/v202/wang23p.html
- *Predictive Representations: Building Blocks of Intelligence* (Neural Computation, 2024) surveys predictive representations and generalization: https://doi.org/10.1162/neco_a_01705
- *Observable quotient world models for knowledge-state abstraction under partial observability* (2026) discusses preservation of observable predictive distinctions rather than unnecessary latent detail: https://www.sciencedirect.com/science/article/pii/S0950705126014942
- Kirchner, Purschke and Knoll, *A survey of world models for physical AI with uncertainty representation and control* (2026), frames world models as action/observation-history predictors: https://link.springer.com/article/10.1007/s44163-026-02122-1

Our model is **not** a reproduction of these methods. It is a compact discrete variable-order predictor for a tiny symbolic world.

## Implementation and limits

`src/emergent_cells/experiments/predictive_context.py` trains three comparators using **exactly the same recorded observations, selected actions and self-supervised next-observation labels**:

1. **Observation-only:** predicts the next 5-bit sensor pattern from current pattern and action (no memory).
2. **One-step history:** additionally conditions on one preceding observation-action pair.
3. **Adaptive predictive context:** uses as many as three preceding observation-action pairs, with additive-smoothed conditional distributions and backoff for rare patterns.

The learner updates counts from its own local experience, with Laplace-style smoothing toward lower-order probabilities. History contexts with fewer than five observations are physically removed after training; predictions use only evidence-supported history. This implements learned predictive probabilities over **engineered, discrete history features**, not a neural latent state, fully autonomous state discovery, task-free navigation, or self-organized cell communication.

Training and evaluation are CPU-only, independent of GPU or Meta APIs. A uniform random action generator supplies trajectories and the simulator may reset position between episodes. This is a **privileged training-coverage mechanism**, but reset state coordinates and any goal positions are not passed to the learner.

## Exploratory development

Two development seeds 11-12 with 12,000 observations produced 83.03% adaptive accuracy against 65.32% observation-only, on unseen random trajectories in the stationary maze. After low-count context pruning, a second two-seed development pilot (21-22) measured 83.42% vs 64.54% accuracy. These pilot seeds were used only for development, not final confirmation.

Raw files:
- `results/exploratory/r1-predictive-context-pilot.json`
- `results/exploratory/r1-predictive-pruned-pilot.json`

## Held-out validation: ten previously unused seeds

Seeds 2001–2010, each trained with 24,000 random-action interactions. Two separate held-out trajectory sets requesting up to 6,000 action observations per seed (some episodes terminate early after completing goals): (A) same maze; (B) modified maze, with one passage blocked before the trajectory. Models do not receive that change as a label. This is **zero-shot test prediction under a changed transition graph**, not online retraining on the new graph.

| Test condition and metric | Observation-only | One-step history | Adaptive history |
|---|---:|---:|---:|
| Stable world: next-observation accuracy | 64.85% | 82.96% | **86.89%** |
| Changed passage: next-observation accuracy | 66.62% | 81.41% | **84.55%** |
| Stable world: ambiguous-condition accuracy | 46.14% | 73.89% | **79.91%** |
| Changed passage: ambiguous-condition accuracy | 48.74% | 71.46% | **76.28%** |
| Stable world: negative log likelihood, nats | 0.7419 | 0.3462 | **0.2719** |
| Changed passage: negative log likelihood, nats | 0.8090 | 0.5020 | **0.4535** |

The adaptive-history advantage over the memoryless comparator was **+22.04 percentage points** in stable environments and **+17.93 points** for changed topology. Bootstrap resampling ten seed-level differences (10,000 draws; random seed 28026) gives descriptive 95% intervals of **[21.23, 22.89]** and **[17.33, 18.58]** percentage points. These intervals quantify variability across seeded synthetic runs, not external-world uncertainty.

Mean retained dictionary keys by history depth: **84**, **624**, **2,234**, **622**, or approximately **3,564 context keys total**. Average CPU process time for fitting a seed was approximately **0.44 seconds** on the user's Windows laptop; Python `tracemalloc` training peak averaged **7.35 MB**, which is **not** process RSS or energy consumption. The training/test world and memory budget are tiny, so these figures are not general resource-efficiency claims.

Raw per-seed results: `results/audited/2026-10-28-r1-predictive-history-holdout.json`.

## What this establishes

Supported: compact history-conditioned predictive probabilities improve sensor-transition forecasts over a memoryless comparator and a one-step-history comparator in a finite, aliased synthetic maze. The model requires no coordinate labels, map, or privileged goal features for training.

Not supported: an emergently discovered neural belief state; learned sensor-to-goal policy; multi-step planning without BFS; decentralized reasoning or learned inter-cell messaging. Three previous observation/action pairs are **fixed, authored representation capacity**; the posterior probabilities within them are learned. Test-time obstruction is preconfigured before each new test episode, not dynamically switched in the middle. Random source resets are favorable simulator privileges.

## Next experiments required to close R1

- Train a small recurrent predictive representation and a compact predictive-state clustering model from the same trajectories; compare both against this history-count baseline, equalizing sample count and memory where possible.
- Test difficult unseen layouts and randomized textures to separate memorization of this maze from general predictive abstractions.
- Predict multi-step future observations and rewards, not only one-step wall/texture.
- Pass the best **learned state** into R2's locally learned cell-to-cell messages; retain a no-history ablation and explicit hardware accounting.

## Reproduce

```powershell
python -m pip install -e .
python -m unittest discover -s tests -q
python run.py predictive-context --seeds 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010 --train-steps 24000 --test-steps 6000 --episode-length 100 --alpha 5 --min-visits 5 --output results/runs/r1-predictive-history.json
```

All new code and documentation use modern professional English. Do not alter historical raw results.
