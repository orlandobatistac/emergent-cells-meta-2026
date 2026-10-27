# Dual-cue cooperation: exploratory result — Oct 26, 2026

**Preparatory research outside the participant guide's stated Oct 27–Nov 4 building window. Not claimed as eligible competition contribution.**

## Setup

- 256 artificial cells, forming 128 fixed local pairs.
- 1,200 training episodes, then 300 fresh-cue evaluation episodes (learning frozen).
- Ten independent seeds: 1–10; identical seeded trial interfaces for all five conditions.
- Each resource gate opens only when **both** private-cue holders make the correct parity action.
- Simulated metabolism: 0.018 units/episode; message 0.002 units; successful gate reward 0.11 units per cell; finite energy bound 4.0.

## Measured outcomes

| Condition | Mean joint gate success, held-out cues | Population SD across 10 seeds | Interpretation |
|---|---:|---:|---|
| Local true partner messages | **98.04%** | 0.07 percentage points | Learners can solve this specific engineered task |
| No messages | 48.69% | 0.27 percentage points | Private information insufficient for consistent correct choice |
| Physically separated partners | 48.69% | 0.27 percentage points | Communication unavailable beyond radius |
| Corrupted messages from other pairs | 48.14% | 0.41 percentage points | Receiving irrelevant information does not solve gate |
| True message, frozen random policy | 24.97% | 0.23 percentage points | Information alone without learning is insufficient |

**Direct causal knockout**: keeping *exactly the trained local cell weights* and replaying the same held-out cues/action RNG while blocking messages reduces mean success from 98.04% to **24.98%**.

The paired difference between independently trained local and no-message treatments averages **49.35 percentage points** across these 10 seeds. These were *development seeds*, not a preregistered confirmatory set.

Each run contains 128 pairs × 300 evaluation episodes; all 10 local runs had successful gate rates between 97.94% and 98.18%. This is a task intentionally constructed so local information transfer is advantageous.

Full raw per-seed outcomes and every reported metric are in [the associated JSON](../results/exploratory/dual-cue-2026-10-26.json).

## Interpretation

This supports a **narrow causal result**: communicating the missing bit lets separately learning cells open a parity-based shared resource gate much more often than when their cues cannot be shared. Their learning updates consume only local energy-change feedback, not explicit correct-action labels.

It does **not** demonstrate spontaneous creation of an XOR representation, autonomous discovery of communication protocols, emergence of new biology, task-general cognition, or physical energy efficiency. We provided an XOR-capable product feature, fixed partners, a perfect bit channel and a world with a known parity gate rule. A conventional hard-coded logic gate would solve this toy benchmark perfectly.

No evidence of Meta Model API use, structural growth, substrate independence, hardware joules or learning under previously unseen task *rules* is presented.

## Local verification

- Python 3.12.10, NumPy 1.26.4; Windows laptop.
- All 28 unit/integration tests passed locally after correcting the distance-ablation test.
- The ten-seed benchmark ran successfully with no crash and identical run settings.
- Raw JSON SHA-256: 82868d012cc29c49c6637725d4b38e4226e77cab66697d9f1b8f543fbba1c816.

## Next gate

A stronger scientific milestone is **learned message encoding** with local physical costs and dynamic connectivity, with an explicit random/frozen/hand-coded-policy comparison on a task the agent has not received as an engineered interaction feature. A positive result there would be more relevant to Emergent Cells Lab than this necessary-information sanity test.
