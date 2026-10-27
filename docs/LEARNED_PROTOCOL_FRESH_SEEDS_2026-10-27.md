# Independent-seed replication of the learned-symbol experiment

**Date:** October 27, 2026. Executed AFTER inspecting development cohorts; algorithm, hyperparameters, task-rule and test procedure were **not changed**.

These 5 seeds (1001, 1002, 1003, 1004, 1005) were not used for tuning the algorithm or for the prior 10-seed report. Same fixed XOR gate, fixed partner identity, one-bit channel, 128 pairs, 8000 training and 500 test episodes per run. This only tests reproduction across independent random initializations/cue sequences; it **does not** demonstrate transfer to a new task rule or unknown environments.

## Repeated result

| Condition | Average held-out-seed joint success |
|---|---:|
| Learned private sender + private receiver | **97.93%** |
| Blocked reception | 49.27% |
| Cue-independent random messages | 48.35% |
| Fixed initially random sender | 48.30% |
| Frozen receiver | 25.60% |
| Same learned weights, received symbols blocked | 25.17% |
| Same learned weights, received symbols randomized | 25.09% |
| Same learned weights, input cue frequency changed to 80% ones | 97.91% |

Learned-vs-blocked mean matched difference: **48.66 percentage points**; learned condition won **5 of 5** fresh-seed pairs. Individual learned-condition success fractions were **0.9811, 0.9775, 0.9845, 0.9774, 0.9759**.

Raw data: [machine-readable held-out-seed results](../results/audited/2026-10-27-learned-messages-unseen-seeds.json).

SHA-256: `85c9e3fc7668c508cd8313fb702a62c698107209876d7940375f2c966742aa49`

## Reproduction

    python -m emergent_cells.experiments.learned_messages --seeds 1001,1002,1003,1004,1005 --pairs 128 --train 8000 --test 500 --output results/runs/fresh-seed-reproduction.json

## What remains unproven

- Our cells were given the representation as a four-slot conditional receiver table, a fixed pairing, and a noiseless one-bit channel.
- The world rule itself remains XOR, entirely supplied by the researcher. Training/test rules are identical.
- All modeled energy is symbolic; we have measured no electricity or physical device efficiency.
- Good held-out-initialization generalization is NOT the same as general intelligence, language creation, dynamic link formation or cross-task learning.
- A stronger next falsifiable gate is task-rule reversal or new task families, measured before and after retraining, using only local energy rewards and explicit memory/communication budgets.

This is an independent-seed replication of a narrow laboratory claim, not confirmation of the full emergent-intelligence hypothesis.
