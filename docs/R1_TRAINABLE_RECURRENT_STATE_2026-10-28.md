# R1 — Trainable Recurrent Predictive State from Local Sensors

**Date:** October 28, 2026. **Research status:** recurrent predictive subexperiment validated; full R1 not complete.

## Motivation and references

R1 tests whether a compact, learned hidden state can retain useful information that a single partial observation does not expose. Relevant research: Eberhard et al., *Partially Observable Reinforcement Learning with Memory Traces* (ICML 2025), https://proceedings.mlr.press/v267/eberhard25a.html; Zhao et al., *Why Linear Recurrent Memory Works in Partially Observable Reinforcement Learning* (ICML 2026), https://proceedings.mlr.press/v306/zhao26j.html; Lillemark et al., *Flow Equivariant World Models* (ICML 2026), https://proceedings.mlr.press/v306/lillemark26a.html. The study is independent experimental work, not a reproduction of those algorithms.

## Design

- The learner sees only four local adjacent-wall bits, one binary ambiguous floor texture, and its selected action. It predicts the next 5-bit sensory signature. It never receives map coordinates, goal coordinates, hidden state labels, world layout ID, or shortest-path labels.
- An action-conditioned NumPy tanh RNN uses 36 one-hot input features, a 48-dimensional hidden vector, and a 32-class categorical output. Recurrent weights, input weights, and output weights are all learned by full-episode backpropagation through time, Adam, gradient clipping, and self-supervised cross entropy.
- Hidden state resets at each independent episode and persists between actions within an episode. This is a *learned parametric recurrent state*, not an explicit hand-coded window.
- The fixed-history comparison uses the earlier, independently validated variable-order memory table with up to three engineered observation/action history pairs.
- The memoryless comparison learns conditional next-sensor distributions using only current observation/action. The recurrent-memory ablation evaluates the *trained* RNN after erasing its prior hidden state at every step; this is an intervention, not a separately trained matched-capacity feedforward neural control.
- All three trained approaches see identical local training trajectories. Training-world resets can use hidden simulator coordinates, but that information never reaches the learner. The external simulator maintains privileged truth only for generating outcomes.

## Development attempts and failures

- **V1, negative:** seed 11; 4,000 training observations, 3 epochs, 16 hidden units: RNN stable-world accuracy 19.00% and novel-world accuracy 16.25%, far below historical controls. Saved in results/exploratory/r1-recurrent-dev-11.json.
- **V2, underperforming:** seed 12; 24,000 training observations, 12 epochs, 24 hidden units: RNN stable accuracy 71.17% versus 85.97% for the history table; novel-world 62.23% versus 73.09%. Saved in results/exploratory/r1-recurrent-dev-12.json.
- **V3, promising pilot:** seed 13; 24,000 observations, 30 epochs, 48 hidden units: RNN stable 94.47% versus table 87.55%; novel 78.63% versus table 74.47%. Saved in results/exploratory/r1-recurrent-dev-13-48x30.json.

All variants are maintained as exploratory evidence. The final schedule was selected before running the RNN on final study seeds.

## Held-out evaluation

**Seeds:** 2001–2010; 24,000 training observations and 30 training epochs per seed; 48 hidden units; up to 4,000 test actions in each separate scenario. These seeds were used in an earlier history-table study but were not used for V1–V3 recurrent hyperparameter selection. They are independent of *this* tuning stage, not globally untouched project seed IDs.

Evaluation scenarios:
- **Stable:** new action trajectories in the original world.
- **Changed:** one upper corridor is closed before new trajectories. No online model updates are allowed; this is not a mid-episode change recovery demonstration.
- **Novel:** test-only procedurally modified 9x9 layouts, each adding three obstacle cells and removing one existing obstacle. These share the original action vocabulary and local sensor encoding.

| Test case | Observation only | History table | Trainable RNN | RNN with memory erased |
|---|---:|---:|---:|---:|
| Stable sensor accuracy | 64.78% | 86.96% | **93.56%** | 58.33% |
| Changed sensor accuracy | 66.64% | 84.59% | **89.42%** | 58.84% |
| Novel layout accuracy | 57.96% | 72.85% | **76.16%** | 52.14% |
| Stable negative log likelihood (nats) | 0.7422 | 0.2715 | **0.2131** | 0.9851 |
| Changed negative log likelihood (nats) | 0.8022 | 0.4449 | **0.4041** | 1.0476 |
| Novel negative log likelihood (nats) | 1.4711 | 1.3470 | **1.3007** | 1.7370 |
| Stable ambiguous sensor accuracy | 46.22% | 80.08% | **90.26%** | 37.37% |
| Novel ambiguous sensor accuracy | 40.71% | 64.94% | **71.62%** | 32.88% |

The RNN exceeds the strong history-table comparison by **6.60 percentage points** in the stable maze, **4.83** under a shifted passage, and **3.31** on new layout variants. The memory-erasure ablation substantially reduces accuracy, consistent with a causal role for the learned recurrent state. The drop also changes the distribution of hidden inputs seen during training, so it is not a fair separately trained feedforward comparison.

Raw per-seed metrics: results/audited/2026-10-28-r1-trainable-recurrent-holdout.json.

## Compute profile and scientific boundaries

- **22,592 bytes** of recurrent model parameter arrays at float32 precision, excluding optimizer states, Python objects, training sequences, and runtime buffers.
- Mean RNN training **process CPU time** about **13.84 seconds per seed** in the ten-seed study. A seeded full run took about **156.7 seconds wall clock** on LANDITO-LAPTOP; workloads can overlap with other processes.
- Mean peak Python allocation measured by tracemalloc approximately **1.61 MB** during recurrent training. This is **not process RSS or physical energy**.
- Memory table may be faster and/or lower total computational cost; comparison of fitted model parameter bytes alone is insufficient for an efficiency claim.
- The training data are stationary within episodes and have privileged random resets. Novel maps share the same two-dimensional coordinate topology, wall sensor and movement structure.
- The RNN predicts one-step sensory outcomes; it does **not** yet discover a goal-conditioned action policy or coordinate with multiple cellular agents. True decentralized planning is still R3.

## Remaining R1 work

1. Study true long-horizon partial observability and test the hidden-state capacity needed for reliable multi-step predictions and rewards.
2. Compare against an independently trained, matched-budget feedforward model and a compact recurrent baseline with stronger holdout layouts and textures.
3. Learn an actionable goal-conditioned state abstraction and test it without global position/state access.
4. Wire recurrent state updates into cell-local agents and communication (R2); eliminate any hidden world-state dependency from the policy interface.
5. Profile actual process RSS and hardware power only when measurement tools are available.

## Reproduction

Install the project with Python and NumPy; execute its Python test suite, then:

    python run.py recurrent-state --seeds 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010 --train-steps 24000 --test-steps 4000 --epochs 30 --hidden 48 --output results/runs/r1-recurrent-reproduction.json

All new work is documented in professional English; historical results are preserved.
