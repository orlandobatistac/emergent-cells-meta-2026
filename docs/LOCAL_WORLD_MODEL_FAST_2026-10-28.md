# Fast Local World-Model Experiment — October 28, 2026

**Scope:** Falsifiable first step toward locally learned predictive world models. This narrow experiment tests **predicting local reward from observations and chosen actions**, not full physical dynamics, autonomous causal discovery or general intelligence.

## Setup

We started from the `experiment/learned-local-curiosity` branch, leaving prior experiment history unchanged.

- Each of 64 artificial cells (32 fixed pairs) has a distinct eight-parameter tabular conditional reward predictor: 2 own-cue values × 2 peer-message values × 2 own-action values. It updates with its **own observed reward** and local incremental prediction error (learning rate 0.09).
- The neighbor's message is **a perfect direct copy of its private cue**, and action selection is randomly sampled by the external experiment runner for coverage. These are engineered scaffolds. **This run does not validate the previously learned message protocol or learned exploration policy.**
- The world alone implements fixed XOR, XNOR, AND, then XOR again. Rule IDs and correct target actions are NOT passed to the cell predictors. Gate reward is shared by each pair and randomly omitted 8% of the time.
- The predictor estimates reward probability given only its observable features. Prediction quality uses held-out **mean squared error (MSE)**. Comparator is a frozen 0.5-probability predictor with MSE 0.25 for binary rewards.
- Each phase trains 1200 episodes and tests on 240 independent new samples with updates frozen. There is no manual context indication or reset between phases. The one learned table is simply overwritten/adapted when conditions change.

## Pilot (development seeds 11–13)

| World phase | Held-out MSE | Relative improvement over fixed predictor |
|---|---:|---:|
| XOR | 0.1295 | 48.2% |
| XNOR | 0.1287 | 48.5% |
| AND | 0.1282 | 48.7% |
| Return XOR | 0.1295 | 48.2% |

## New-seed test (no additional tuning, seeds 5001–5005)

| World phase | Held-out MSE | Fixed comparator MSE | Relative improvement | First 100 training MSE → Last 100 |
|---|---:|---:|---:|---|
| XOR | **0.1303** | 0.2500 | 47.9% | 0.180 → 0.130 |
| XNOR | **0.1301** | 0.2500 | 47.9% | 0.218 → 0.131 |
| AND | **0.1287** | 0.2500 | 48.5% | 0.152 → 0.129 |
| Return XOR | **0.1292** | 0.2500 | 48.3% | 0.196 → 0.130 |

All five seeds and all four phases are included; no failed seeds removed. See [machine-readable observations](../results/audited/2026-10-28-local-world-model-fresh-seeds.json).

## Interpretation and limitations

This is a **positive local conditional-reward prediction result** across three authored Boolean world rules. It supports the existence of a small learnable predictive model in each cell, not emergence of a general world representation.

**Major controls missing:** compare against an empirically estimated constant reward baseline (rather than only 0.5), a *no-peer-message* control, learned vs perfect messaging, learned vs random action selection, and no-adaptation evaluation immediately after a switch. The fixed 0.5 comparator is intentionally weak because reward frequencies are not balanced; future tests must include stronger baselines before claiming superiority in learned causal world modeling.

**No retention demonstrated:** returning to XOR causes a fresh adaptation from a table modified by other tasks. In this run there is no learned context bank, continual memory protection or proven task-free transfer.

**No physical resource measurement:** synthetic reward is not real electrical energy; table size, process time, and byte allocations are not evaluated.

**No Meta Model API calls** or real-world general-agent control. The code and comments are in professional English.

## Reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
python -m emergent_cells.experiments.local_world_model --seeds 5001,5002,5003,5004,5005 --pairs 32 --phase-steps 1200 --test-steps 240 --output results/runs/reproduce-world-model.json
```

## Immediate next gate

Replace the perfect peer cue with the **independently learned message code from PR #13–#14**, add a strong empirical constant-predictor and message-knockout controls, and test whether learned local predictions improve *action selection* under missing resources, not merely MSE. Use new independent seeds and retain negative findings.

## Stronger context-free statistical baseline (added before closeout)

We also compared each private conditional predictor against an **individually learned global mean reward predictor**. That reference gets the same per-cell training reward stream but no private cue, partner message or chosen-action features. It is **substantially stronger** than a frozen probability 0.5 comparator.

| World phase | Contextual predictor MSE | Learned global reward-rate MSE | Relative error reduction |
|---|---:|---:|---:|
| XOR | **0.1303** | 0.1769 | **26.4%** |
| XNOR | **0.1301** | 0.1768 | **26.4%** |
| AND | **0.1287** | 0.1758 | **26.8%** |
| Return XOR | **0.1292** | 0.1767 | **26.9%** |

Same five held-out seeds 5001–5005, same phase durations and update schedule, no hyperparameter changes. The saved raw audit was regenerated to contain BOTH baselines plus all per-seed stage results. Even this improved comparison does not replace a model with only some features ablated, e.g. without the peer bit, or the actually learned message interface. Those are the next required tests.
