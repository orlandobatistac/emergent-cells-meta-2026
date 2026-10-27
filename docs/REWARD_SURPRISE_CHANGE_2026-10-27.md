# Unannounced task reversals detected via local reward history

**Research date:** October 27, 2026. This is an exploratory **engineered detector** experiment, not proof of general intelligence or spontaneous invention of memory.

## Scientific motivation

Our prior phase-A XOR / phase-B XNOR / phase-A contextual-memory experiment achieved strong learning and retention only because an externally provided season/context bit selected the correct memory bank. The stronger question here is whether cells can select a previously learned bank **without being told when the environment reverses its rule**.

Related literature:
- Foerster et al. (NeurIPS 2016), *Learning to Communicate with Deep Multi-Agent Reinforcement Learning*: https://arxiv.org/abs/1605.06676
- Hamadanian et al. (ICLR 2025), *Online Reinforcement Learning in Non-Stationary Context-Driven Environments*: https://proceedings.iclr.cc/paper_files/paper/2025/hash/fb21dae9e8710a272c0a848f71553-Abstract-Conference.html

These works motivate the problem; we do not claim to replicate their algorithms.

## Experiment design

- 64 fixed pairs of cells (128 individually trained cells). World provides two independent private bits and shared gate reward. A opens with XOR; B with XNOR. **Neither cell sees the XOR/XNOR target**.
- Each cell emits its own learned, 1-bit cue-dependent symbol, selects an action using its own cue plus peer's symbol, and updates independent local weights from its own synthetic energy pulse.
- **No season/task label is supplied to the cells.** World rules are set by experiment runner; cells only learn from energy delta.
- Two private receiver-memory banks exist from the start; the engine does **not** discover how many banks to allocate.
- A designer-authored *surprise detector* monitors the cell's own exponentially weighted recent resource pulse (decay .95), remembers that bank's historical peak, and switches to the other bank if a previously mature bank (peak above .85; minimum 300 visits) loses performance below .38. Switching has a 100-episode cooldown. All thresholds were selected during exploratory development, not discovered by RL.
- A training: 4500 steps; B training: 3500 steps; test: 200 new-input episodes. The sender encoder is frozen after A, preserving the communication code. Only private receiver weights update in B.
- On unannounced return to A, agents are given 120 episodes of **local reward observation and possible memory selection, without ANY weight training**. The first 20 and final recovery are measured separately. This is NOT zero-shot recovery.
- Matched comparison: **provided_context** (receives explicit season bit, upper-bound policy memory), and **blind_memory** (never receives the marker and never activates a change detector). Same fixed pair gates, same world inputs, seeded initial cell weights, common energy-reward rules and evaluation sequence.

## Positive exploratory pilot and fresh-seed replication

Development pilot seeds 21,22: approximately 96.48% success on B, 96.42% recovered success on A, 100% of pairs detecting return, mean detection delay 18.52 episodes.

Without algorithm/threshold changes after that successful pilot, previously unused seeds **2001,2002,2003,2004,2005** were evaluated under the same conditions:

| Condition / metric | Mean |
|---|---:|
| Reward-surprise, A before switch | 96.51% |
| Reward-surprise, B before adaptation | 0.13% |
| Reward-surprise, B after adaptation | **96.29%** |
| Reward-surprise, A success in first 20 return steps | 6.75% |
| Reward-surprise, A success after 120 detection-only steps | **96.51%** |
| Pair members detecting return | **99.69%** |
| Mean reward-only switch latency | **18.61 episodes** |
| Oracle with explicitly supplied context, B | 96.32% |
| No context or detector, B | 0.18% |

| Fresh seed | B learned | A recovered | Both pair members detected return |
|---|---:|---:|---:|
| 2001 | 96.23% | 96.46% | 100.00% |
| 2002 | 96.52% | 96.83% | 100.00% |
| 2003 | 96.08% | 95.98% | 98.44% |
| 2004 | 96.34% | 96.52% | 100.00% |
| 2005 | 96.27% | 96.74% | 100.00% |

**Important interference discovered by a failing test:** during the detection delay at the start of B, some cells still update their old A bank. Consequently A weights are **not always bit-identical** to their state just after A. The experiment shows recovery in **performance**, not perfect unmodified memory storage. During the 120 return-detection episodes, weights are frozen and only the detector state changes; an independent behavioral test checks this. The first test implementation had a NumPy advanced-indexing-copy mistake in the historical-peak update; it was fixed BEFORE the successful pilot. This was a coding bug, not negative scientific evidence.

Full per-seed raw observations: [reward-surprise data](../results/audited/2026-10-27-unannounced-switch-fresh-seeds.json).

Raw JSON SHA-256: `a1482a6b8e04bd1ed090bb55ed3d73b0568af94837ed0d845d81b8c5c572e592`.

## Limits and unanswered questions

- **The local detector and number of memory banks were manually engineered**. The system did not invent change detection, memory bank selection or communication channel structure.
- World task family consists only of invertible XOR and XNOR with fixed paired collaborators and deterministic shared gate reward. This result cannot be generalized to unknown tasks or natural language.
- Detection depends on visible energy reward decrease, not an external season indicator. It has nonzero latency and can in principle false alarm under resource drought or reward noise. Long-term false-alarm rates, repeated alternating switches, task-switch scaling and stochastic non-stationarity are NOT tested.
- Pair detectors are independent but receive an identical shared gate reward in a pair; synchronous switching follows from the engineered shared environment and aligned detector history, not negotiated consensus.
- **Memory capacity, switch computations, communication and simulated energy are not costed in measured joules**. Hardware independence is not demonstrated.
- Neither Meta AI nor a global model controls cellular decisions in this experiment. Meta Model API has not been invoked by this route.

## Reproduce

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python -m emergent_cells.experiments.reward_surprise --seeds 2001,2002,2003,2004,2005 --pairs 64 --train-a 4500 --train-b 3500 --test 200 --output results/runs/reproduce-unannounced.json

## Scientific follow-up

1. Test noisy/disappearing rewards and repeated A/B/A/B/A reversals; log false positives per 1000 stable episodes and recovery times.
2. Test a third previously unseen rule (AND/NAND) requiring additional learnable memory capacity; measure whether the detector can create new state rather than toggling one of two fixed slots.
3. Add an explicit finite cost for each memory bank and message and compare performance/resource cost against fixed-gradient and random controllers.
4. Measure where communication fails under channel noise, changing partner identities and asymmetric local rewards.
5. Design pre-registered evaluation seeds and rule families before tuning to avoid circular success claims.
