# On-demand third memory from local reward surprise — 2026-10-27

**Scope:** original, exploratory digital simulation. A trained one-bit pairwise message protocol, a hand-authored change detector, preconfigured maximum capacity of 3 memory banks, fixed pair identities and externally programmed Boolean gate rules. This is **not** AGI, self-organizing biology, a self-invented detector, or energy-efficient hardware.

## Scientific references and design alternatives

1. [Statistical Context Detection for Deep Lifelong RL (CoLLAs/PMLR 2025)](https://proceedings.mlr.press/v274/dick25a.html) discusses statistical task inference from experience and the challenge of reward-function changes.
2. [Dynamics-Adaptive Continual Reinforcement Learning via Progressive Contextualization (DaCoRL)](https://pubmed.ncbi.nlm.nih.gov/37285252/) investigates detecting contexts and *expanding* capacity for newly encountered contexts without given task IDs.
3. [Progress & Compress (ICML 2018)](https://proceedings.mlr.press/v80/schwarz18a.html) protects existing knowledge with a different, constant-parameter memory architecture.
4. [On the Pitfalls of Measuring Emergent Communication (Meta AI Research, 2019)](https://ai.meta.com/research/publications/on-the-pitfalls-of-measuring-emergent-communication/) shows why communication needs causal ablations; reward advantage alone can mislead.

Our implementation is a much smaller **heuristic experiment**, not a faithful reproduction of these papers or a novel state-of-the-art RL result.

## Research question

Can individually learned, local binary-message cells (a) retain existing behavior on XOR and XNOR gates, (b) infer from their *own local shared-gate reward* that **neither known memory bank works** under a new AND gate, (c) allocate a third receiver-memory bank as needed, (d) learn AND and (e) recover XOR/XNOR later **without an externally supplied rule identifier**?

## Methods tested, including failure

**V1: fast exponential-average detector.** Reward EMA factor 0.025 and trigger relative to best reward 0.14; **FAILED**: created extra memory in a stationary A world and caused poor A/B performance. Raw initial-development output is in [the failed pilot](../results/exploratory/dynamic-pilot-v1-failed.json). DO NOT treat it as proof of success.

**V2: slower reward-change detector.** EMA 0.005, relative trigger 0.10, maturity criterion 0.78 and 550 local visits; stage A/B/C were learned, but on returning to B the agent delayed recognizing changes because an already-probed old bank was reset to immature status. This failed to fully recover B in the original 900-step return window.

**V3: slow detector + mature reinstatement after old-bank probing.** On choosing an existing memory, its **measured local reward over the probe window** becomes the new baseline/peak, and age is marked mature. This is not using a correct-action label. The decision to trigger, probe, allocate, overwrite or reuse remains a prescribed heuristic; two-bit inputs, fixed one-bit channel and a 4-entry receiver table remain engineered.

We kept seeds 1001–1005 untouched during V1–V3 tuning. The preregistered **evaluation parameters for this development holdout** were 32 independent pairs, phases A/B/C=4500 episodes each, returns A/B=900 episodes each, reward dropout=8%, five modes and identical input/action RNG streams; last 200 episodes per stage are reported. Each cell's encoder is trained during the first phase and frozen thereafter by the experiment schedule, not autonomously. The world physics holds the Boolean rule; learners receive individual energy change only, without rule ID or correct target.

**Controls**: `grow` dynamically makes 1→2→3 private banks after failing known-policy probes, `cap2` cannot exceed two banks, `single` always uses one bank, `fast_detector` is a more reactive relative-threshold variant, and `oracle` is explicitly privileged with the rule ID for an upper-bound comparison. Capacities and oracle access are **NOT equal**; report that mismatch rather than claim equal-footing supremacy.

## Untouched-seed validation: 1001–1005

| Variant | A XOR | B XNOR | C AND | Return A | Return B |
|---|---:|---:|---:|---:|---:|
| Growth with surprise | **93.7%** | **94.3%** | **95.2%** | **94.1%** | **94.5%** |
| Capacity 2 | 93.7% | 93.6% | **24.1%** | 94.3% | 94.5% |
| Single policy | 93.9% | 0.5% | 24.6% | 93.7% | 0.9% |
| Fast detector variant | 93.9% | 94.8% | 95.2% | 94.1% | 94.5% |
| Privileged rule-aware oracle | 93.9% | 94.9% | 95.4% | 94.2% | 95.2% |

Values are end-of-phase last-200 success rates averaged over 5 seeds, NOT physical energy efficiencies or statistical significance.

The **AND cue pair '11'**, which requires the rarer action, reached **92.16–95.41%** success in the five growth runs. A trivial always-zero policy solves only 75% of the balanced AND cases and **fails on '11'**. This additional per-input check rejects that trivial shortcut.

Growth model mean allocated memory banks: A **1.006**, B **2.006**, C **2.987**. Thus allocation is locally triggered for most, but *not absolutely all*, cells. This is growing *stored decoder slots*, not creating new biological or computational cells. Receiver storage initially contains only one private policy bank per cell; when any cell needs another slot, the NumPy simulator **physically extends the packed storage tensor**. The per-cell allocated count remains independent, but the new tensor column is provisioned centrally for batched Python execution. This is actual software memory allocation on demand, NOT growth of biological cells, physical circuitry or an unbounded topology. Old trained banks can still be modified during detection latency; perfect weight preservation has NOT been demonstrated.

The final on-demand NumPy allocation reproduced all stage-level success rates of the fixed-storage prototype on the same five seeds.

Raw, every-condition/per-seed/per-phase results: [JSON](../results/audited/2026-10-27-dynamic-third-rule-holdout.json), SHA-256 `aeebf5f7fd854b3d45f951ca4f20b134c06ab56f6f04402c7e119553a9f526c2`.

## Critical noise confound and negative result

Separate stationary-rule stress (XOR only, 3 development seeds, 4500 training + 1400 further episodes without changing task):

| Training dropout → stable stress dropout | Unnecessary new memory allocations |
|---|---|
| 8% → 8% | 0 in all 3 seeds |
| 0% → 0% | 0 in all 3 seeds |
| **8% → 20%** | **62–64 false allocations per 64 cells**, in all 3 seeds |

**The detector confuses a sudden reward availability loss with a task-rule change.** This is a real failure. The C-rule result is valid only under the tested reward dropout and task schedule; it does NOT solve robust non-stationary environment identification. This problem must be solved before a trustworthy autonomous lifelong-learning claim.

Raw stress data: [JSON](../results/audited/2026-10-27-noise-stress.json); SHA-256 `ca0edf1edd448c0fa980061c039e64584141cfd7fb28001f33f7097034d4ef95`.

## Interpretation, caveats, next falsifiable milestone

Supported: a **programmed local algorithm** uses performance loss to probe existing learned memories and allocate a previously unused receiver bank when none of the old policies yields useful reward; reward-modulated local policy updates can then train a genuinely different, asymmetric AND rule and later reuse prior policies.

Not supported: spontaneous invention of a brain-like dynamic growth algorithm, task-general cognition, learned arbitrary-length protocols, independent hardware/joule claims, task-free detector robustness, persistent long-horizon memory stability, new physical cells, or comparisons fairly accounting for larger memory footprint.

Next study must distinguish noise/dropout from a different reward mapping using **per-input surprise signatures or local posterior likelihoods**, test common vs asymmetric pair reward noise, preregister disjoint task families, and price memory bits and control computations. Distinguish **inference of environment conditions** from performance-optimized task rules.

## Reproduce

From repository root (Python with NumPy):

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m emergent_cells.experiments.dynamic_memory --seeds 1001,1002,1003,1004,1005 --pairs 32 --phase-steps 4500 --dropout .08 --output results/runs/third-rule-reproduction.json
python -m emergent_cells.experiments.stationary_noise --seeds 11,12,13 --output results/runs/stationary-noise-reproduction.json
```

**Meta API was not called or charged.** This task runs locally and is separate from the Meta-based bounded world designer; do not present it as using the model.

## Engineering audit: physical storage grows lazily

The implementation now begins a non-oracle cell population with **one physical NumPy receiver policy bank**, `weights.shape == (2 * pairs, 1, 4)` and a matching per-bank baseline column. When the FIRST cell requests another bank, the scheduler appends a *new full-population NumPy column*; a third column is appended when some cell requires its third bank. Previously, all three columns were allocated up front.

This is **actual lazy allocation of packed storage at population level**, not true per-cell ragged physical allocation. Every appended column reserves space for all cells even if one cell requested it, while `allocated[cell]` enforces independently permitted/used memory counts. The experiment does NOT grow the number of cells or discover an unbounded number of slots. The max capacity remains three by research design, and the task-aware oracle starts with all three.

For 64 cells (32 pairs), policy logits plus baselines occupy **2,560 bytes per physical bank** (float64, excluding NumPy object/container overhead): initial one bank 2,560 B, then 5,120 B, then 7,680 B. The executable report now includes `allocated_tensor_columns` and `packed_policy_storage_bytes` for each stage. These are **resident array payload sizes, not measured RAM footprint, energy or resource efficiency across implementations**.

A regression test checks that no second/third physical column exists initially, one cell's request causes the global column to be appended, other cells retain just one *logical* bank, and old logits are bitwise preserved through the extension. A fresh-seed run checks numerical outcome equivalence to the original five-seed data; see the separate regeneration file rather than retrospectively changing historic raw results. Physical RAM use remains a vectorized implementation choice, not per-cell adaptive matter growth.
