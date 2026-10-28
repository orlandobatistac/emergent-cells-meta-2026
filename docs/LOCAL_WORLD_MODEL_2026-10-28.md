# Phase 8: Local Empirical Outcome Models — Fast Scientific Prototype

**October 28, 2026.** This is a CPU-friendly proof of concept for independently learned outcome predictions, **not** a general generative world model, new physical cells, AGI, biological intelligence, or measured physical energy efficiency.

## Research question

Earlier agents selected between authored hypotheses: original rule, changed rule or scarce resources. Can a cell instead learn a much simpler **empirical predictor from its own cue/message/action/reward history**, use that to choose a counterfactual action and handle both a changed rule and selective resource scarcity?

The motivation is drawn from stochastic world-model exploration and the noisy-TV failure mode:

- Hou, An & Du, *Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring*, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/803b9c4a8e4784072fdd791c54d614e2-Abstract-Conference.html
- Jarrett et al., *Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments*, ICML 2023: https://proceedings.mlr.press/v202/jarrett23a.html
- A formal model-based or deep representation-learning implementation of either cited work is **not** attempted here.

## Implementation

- Fixed pairs of individually parameterized cells, each with its own learned **one-bit sender/decoder protocol** from earlier experiments; the sender/decoder is pretrained and frozen during this phase.
- Each cell sees only its own binary cue, a partner's learned bit, whether the pair tried the familiar or alternate coordinated action, and its own obtained reward (same gate pulse shared by both pair members).
- Each cell stores a **4 local cue/message contexts × 2 possible actions** empirical Bernoulli reward model. Beta pseudo-counts start with small, explicitly authored prior preferences for the familiar action. Success/failure observations independently update the posterior for *the action actually executed*. No three authored **cause hypotheses** are present in these models.
- A separate authored exploration heuristic chooses actions from these *learned probabilities*. We test: passive no probe, uniformly random 8%-probability probes, greedy empirical predictions, optimistic/UCB predictions, and a "predictive progress" policy with exploration suppressed when repeated failures indicate neither action is rewarding.
- If the alternate action becomes empirically better than the familiar action with at least four observed opportunities, the policy can select it as **ordinary learned exploitation**. These are not billed as novel *experimental probes*. Switching still requires a neighbor message and its simulated signaling cost.
- The environment still supplies the **fixed XOR-like gate task**, predetermined pair topology, binary channel, resource pulse chance and hand-designed probe mechanism. It computes targets, but no correct-action label or world-rule ID is supplied to the cell learner.

## Development and fresh-seed confirmation

Development seeds 11 and 12 (24 pairs, 3000 sender training steps, 450 test episodes) were used to inspect methods and correct a cost-accounting error. The original v1 incorrectly counted *every execution of a learned alternate action* as another costly experiment. That v1 trial is preserved as an exploratory artifact; the final model separates exploratory probing from committed exploitation. The change does not alter joint action correctness, only the meaning of the modeled experimental cost.

**Independent confirmatory seeds 7001–7005:** 48 fixed pairs = 96 cells, 3600 sender training steps, 550 test interactions, seven synthetic scenarios and five methods; **175 per-seed/scenario/method records**.

| Method | Changed mapping for cue 11: correct joint action | Selective scarcity for cue 11: correct joint action | Exploratory probe count, changed case | Synthetic request/probe cost, changed case |
|---|---:|---:|---:|---:|
| Passive | 0.4% | 99.6% | 0 | 0 |
| Random 8% | 15.9% | 84.6% | 3,225 | 97.52 |
| Empirical greedy | 87.5% | 97.4% | 337 | 54.77 |
| Optimistic UCB | **98.5%** | **55.3%** | 1,378 | 88.05 |
| **Predictive progress** | **97.0%** | **96.0%** | **951** | **76.29** |

The success metric is **correct joint gate action when cues=11**, irrespective of the later stochastic availability of a resource pulse. Under unchanged stable XOR, the predictive-progress method also achieved 97.8% cue-11 correctness; passive achieved 99.6%.

Thus optimistic uncertainty **over-explores resource scarcity** while the progress-limited predictor provides a better empirical tradeoff in these tested conditions. Random baseline correctness on changed action remains low because it does not retain a persistent empirical preference for alternatives.

The predictive-progress strategy still sends many **action-coordination bits** under the changed mapping. Its **experimental probe count** is separate from committed alternate actions. Simulated cost is not electrical joules and does not account for posterior computations, memory allocation, CPU, battery use or system overhead.

Raw results and all per-seed rows: [confirmatory JSON](../results/audited/2026-10-28-local-world-model-confirmatory.json).

SHA-256: `29e13f1aeb7993ba4b4a6dfab16f97103311d30353b5c1d7a8d2b05bd6e4b113`.

## Interpretation and what we cannot claim

**Supported:** every cell can learn a tiny *conditional action-outcome predictor* from its own rewards; these predictions can support adaptation when an expected action mapping changes while avoiding much of the reward-noise over-exploration of a purely optimistic uncertainty policy. The comparison shows a real dependence on empirical model predictions, not fixed labels for three environment causes.

**NOT supported:** representation learning, a learned latent dynamics model, long-horizon planning, autonomous hypothesis creation, stochastic task transfer beyond this toy family, decentralized implementation on physical devices, energy savings in joules, or general intelligence.

Caveats:
- The predictor is a **Beta–Bernoulli conditional table with engineered priors and four known observation slots**. We have not learned an arbitrary state representation.
- Both peers observe the same pair-level reward and share fixed topology; if one peer changes its action unexpectedly, each local model may misattribute the outcome. No arbitrary network learning.
- The five compared modes use the same pretraining seed and deterministic pseudorandom streams, but consume random draws differently when choosing actions, so this is not a strict identical-trajectory paired causal intervention comparison.
- Reward-prediction Brier scores are measured **on each method's own selected action distribution**, not on a common held-out labeled transition set. These scores therefore cannot establish unbiased predictive superiority versus passive behavior.
- The stored candidate correction-bit flags are diagnostic: the applied learned alternate action is determined by the current empirical prediction; the flags are NOT a new separate applied memory component.
- The method is *not* proven better than the authored Bayesian information-gain controller from the prior research phase. Comparing them under equal samples/communication/compute budgets is a next milestone.

## Reproduce rapidly on the owner's laptop

```powershell
python -m pip install -e .
python -m unittest discover -s tests -q
python -m emergent_cells.experiments.local_world_model --seeds 7001,7002,7003,7004,7005 --pairs 48 --steps 550 --sender-training 3600 --output results/runs/replicate-local-world-model.json
python -m emergent_cells.demo.server --port 18773
```

The web UI adds a separate **Phase 8 · Learning Local Outcome Predictions** panel. It uses a smaller two-seed development demo and strict localhost endpoints; it is not part of the moving foraging-cells canvas. No Meta Model API call or charge occurs.

## Next falsifiable experiment

Learn a **compact latent state representation** from local recurrent inputs, actions and rewards rather than prescribing four cue/message slots. Compare to this empirical tabular model and the authored Bayesian exploration controller with **equal memory and message budgets**. Then introduce unpredictable partner changes and noisy messages without task-label leakage. Only claim physical resource efficiency after measurement on hardware.
