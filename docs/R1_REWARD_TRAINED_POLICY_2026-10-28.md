# R1 — Reward-Trained Recurrent Policy: Memory Test and Maze Failure

**Date:** October 28, 2026. **Research status:** verified diagnostic success, strict maze objective unsolved.

## Goal and scientific references

Can we train the *policy itself* from reward so that prior sensor information influences decisions, without providing world coordinates, a wall map or a centralized BFS planner?

The design is informed by:
- Pedamonti et al., *Hippocampus supports multi-task reinforcement learning under partial observability*, Nature Communications (2025): https://www.nature.com/articles/s41467-025-64591-9 — recurrent policies tested in cue-dependent T-mazes.
- Anjarlekar et al., *Scalable Policy-Based RL Algorithms for POMDPs*, NeurIPS (2025): https://proceedings.neurips.cc/paper_files/paper/2025/hash/8b078da44614b6e249dc9e9474dc8d11-Abstract-Conference.html — finite-history control under partial observability.
- Eberhard et al., *Partially Observable Reinforcement Learning with Memory Traces*, ICML (2025): https://proceedings.mlr.press/v267/eberhard25a.html — compressed memory and the cost of long histories.
- Shchendrigin et al., *Memory Retention Is Not Enough to Master Memory Tasks in Reinforcement Learning* (2026 preprint): https://arxiv.org/abs/2601.15086 — motivates tests of later memory updates, not just retention.

This study **does not reproduce** their architectures or results. All prototypes run with Python/NumPy on the user's conventional laptop.

## Experiment A — Direct policy optimization in the original sparse-sensor maze

We reuse the original SensorEpisode. Policy inputs: nearby wall bits, ambiguous checkerboard texture, immediately adjacent target flags, and own reward/action history. The model never sees its true (x,y), goal coordinates, a global map, or BFS. Three targets must be discovered and reached sequentially within 160 actions.

The new policy_gradient.py implements a **linear stochastic actor and learned linear value baseline**, trained using discounted REINFORCE, a locally computed visit-novelty bonus and 900 whole goal-seeking episodes per model condition. The state encoder is a 32-dimensional recurrent world model pretrained self-supervised on 16,000 sensor transitions, with 20 epochs. The recurrent weights are *frozen during actor training*. This distinction matters: we are not claiming end-to-end training of the recurrent maze representation.

The actor still receives a **shared, hand-designed local visit-count exploration prior** to make sparse-reward learning less brittle; this is not a learned planner. Each actor receives the same total episode budget. The original history_novelty exploration policy is retained unchanged as a stronger reference, not a comparable trained actor.

Development experiments: 150 training episodes (results/exploratory/r1-policy-gradient-pilot-150.json), then 900 (results/exploratory/r1-policy-gradient-900.json). Both failed to establish an advantage. A later control fix added **recurrent_erased_eval** which uses the exact recurrent actor's learned policy weights but zeroes its recurrent hidden memory, unlike a separate trained feedforward control.

### Independent test: 100 seeds 7101–7200; 900 training episodes per actor

| Control | Stable mission completion | Passage-changed completion |
|---|---:|---:|
| **Trained recurrent actor** | **18/100** | **11/100** |
| Trained observation-only actor | 25/100 | 15/100 |
| Same recurrent actor weights, **memory erased** | 26/100 | 15/100 |
| Untrained recurrent actor | 18/100 | 8/100 |
| **Explicit-history novelty control, not reward-trained** | **50/100** | **39/100** |

The actor learns parameters from reward, but **not a successful navigational advantage**. The genuine memory-erasure intervention does not decrease its score. Strong baseline history remains superior. It would be incorrect to describe the grid navigation objective as solved.

Evidence: results/audited/r1-sparse-goal-policy-gradient-holdout.json, including individual episodes and training statistics. Evaluation topology changes are made before episodes, not during episodes.

## Experiment B — A memory-isolating reward-trained decision benchmark

To test whether *reward-driven recurrent decision making itself* works, rather than measuring the difficult global exploration problem, we separately implement delayed_goal.py. Each episode delivers a brief **binary local cue at the start of a corridor**, followed by visually identical readings without the cue. At the end, an agent chooses one of **two possible branches**. A terminal scalar reward is +1 for correct or -1 for incorrect. **The correct action is not provided as a training label**. The simulator knows only the hidden cue-reward mapping and uses it to calculate rewards.

We train a **12-unit leaky recurrent state and policy head end-to-end** with REINFORCE and analytic backpropagation through eight time steps. Memory retention factor 0.88 is *hand-selected and fixed*; sensory-to-memory and memory-to-action weights are trained. This is a controlled proof of memory-dependent choice, not unconstrained architecture discovery and **not the original three-goal maze**.

Comparisons: same trained model with recurrent state zeroed immediately before action; same trained model with the cue flipped at input (causal intervention); unseen corridor length 12/16 without retraining; independent additive Gaussian observation noise at standard deviation 0.05. A memoryless agent has no cue information at the decision point, so the maximum expected correctness on balanced trials is 50%.

Development: seeds 11–13, 200 minibatches and 64 episodes/minibatch, with high performance by the end. Development raw: results/exploratory/r1-delayed-goal-pilot.json.

### Independent validation: 20 seeds 6101–6120, 280 minibatches of 64 episodes per seed

Total = 17,920 training decisions per independently trained seed, 1,000 held-out choices per seed/condition, no target labels given to the policy gradient.

| Test | Learned memory | Erased memory | Inverted initial cue | Added input noise |
|---|---:|---:|---:|---:|
| Trained 8-step delay | **100.00%** | 49.84% | 0.00% | 87.13% |
| Unseen 12-step delay | **100.00%** | 49.98% | 0.00% | 73.70% |
| Unseen 16-step delay | **85.19%** | 50.32% | 15.19% | 65.18% |

Training accuracy averaged 52.12% in the first 25 training batches and 96.78% in the final 25 batches. The consistent memory-vs-erasure gap across 20 seeds and 20,000 held-out choices per condition supports **causal memory dependence in this synthetic task**. For the 16-step delay the seed-level mean accuracy advantage is **34.88 percentage points**; a descriptive 95% bootstrap interval resampling 20 independent training seeds (20,000 draws; RNG seed 20261028) is **[24.71, 44.65] points**. This is uncertainty only over the synthetic seed distribution. Learned sensor-to-state and state-to-action weights operate without a global model, coordinates, routes or supervised correct-action labels.

The fixed retention factor, binary delayed cue, single final branch and stable reward mapping are deliberate *experimental conveniences*. This does **not** demonstrate navigation under arbitrary hidden goals or energy independence. At longer delays and with noise, performance degrades. Memory rewriting under changing cues has **not yet been tested**.

Evidence: results/audited/r1-delayed-cue-reward-holdout.json.

## Experiment C — Higher-resolution recurrent state still does not beat explicit history

We tested an exploratory 48-unit pretrained recurrent encoder and 96 unsupervised state centroids (previously 32 and 24). Development seeds 31–45 improved apparent state discrimination, but 100 new evaluation seeds 8101–8200 showed no consistent advantage:

| Control | Stable completion | Passage-changed completion |
|---|---:|---:|
| Explicit history novelty | **43/100** | 33/100 |
| Higher-resolution recurrent novelty | 38/100 | **34/100** |
| Same recurrent weights, memory erased | 41/100 | 29/100 |

Even with more model capacity, the local one-step sensory prediction objective does not guarantee useful state distinctions for multi-goal planning. The original challenge requires either a policy-specific memory representation, better credit assignment, informative but **legitimate** goal cues, or explicit composition of learned skills—without smuggling in coordinates or a BFS route.

Evidence: results/audited/r1-high-resolution-latent-holdout.json. Development: results/exploratory/r1-high-resolution-latent-states.json.

## Resource accounting

The delayed-cue model has 74 float32 scalars, **296 bytes of weight parameters** (excluding optimizer state, arrays, NumPy and process RSS). The earlier recurrent maze feature extractor is much larger, and maze policy learning took many full exploratory episodes. These figures are **not measures of real electrical energy** or generalized efficiency. The run JSONs record CPU/wall time where instrumented. No GPU required; all learning was performed on the laptop.

## Interpretation and next experiment

**Established:** (1) reward-trained recurrent action selection with causal memory dependence in a controlled delayed-cue task; (2) learned maze actor prototypes and clear reproducible failure against a strong memory-based explorer; (3) no coordinates or route oracle were delivered to either agent.

**Not established:** reliable multi-step navigation from a sparse local target signal; a causal recurrent memory advantage in the real 9x9 maze; learned decentralized planning; strong generalization to unrelated world structures; long-term memory updating; or device-power savings.

### Next action

Add **task-relevant self-supervised auxiliary objectives** or learned predictive successor representations that preserve action-history information useful for future rewards. Pair that with goal-specific cues that are *locally observable* and a reward-trained **memory-updating controller**, not merely a one-step predictor plus externally authored exploration bias. First prove cue updates and multiple sequential choices in partially observed corridors, then transfer the causal memory mechanism back to the unchanged sensor-only 9x9 three-goal benchmark. The acceptance gate remains beating both the strong history policy and same-weights memory erasure on independent maze seeds.

## Reproduce

    python -m unittest discover -s tests -q
    python run.py delayed-goal --seeds 6101,6102,6103,6104,6105 --batches 280 --batch 64 --length 8 --cases 1000 --output results/runs/delayed-goal.json
    python run.py policy-gradient --seeds 7101,7102,7103,7104,7105 --episodes 900 --source-steps 16000 --source-epochs 20 --intrinsic .08 --output results/runs/maze-policy.json

The code, tests, data and README use professional contemporary English, regardless of the language of research discussions.
