# Distinguishing task-rule drift from reward scarcity using cell-local conditional surprise

**Experiment date:** October 27, 2026. Research prototype, not proof of general intelligence, substrate independence, biological cell growth, autonomous invention of a detector, or measured physical energy efficiency.

## Scientific question and references

The previous reward-only detector could learn a new Boolean gate and grow memory, but a stationary XOR world suffering an abrupt rise of resource-pulse dropout from 8% to 20% caused false bank allocations in almost every cell.

Can locally learned, communicating cells tell whether they are failing at **particular observable situations** (suggesting a changed task rule) rather than suffering **roughly uniform reward loss** (suggesting less available resources)?

Relevant background:

- [Statistical Context Detection for Deep Lifelong Reinforcement Learning, Dick et al., PMLR/CoLLAs 2025](https://proceedings.mlr.press/v274/dick25a.html): context inference from experience and difficulty of detecting changing reward functions.
- [Bayesian Online Changepoint Detection, Adams & MacKay 2007](https://arxiv.org/abs/0710.3742): sequential posterior inference over change points. The program below is **not** an implementation of this full method.
- [Concept drift survey, Lukats et al., 2024/2025](https://link.springer.com/article/10.1007/s41060-024-00620-y): warns about distribution drift in streaming signals.

## Mechanism tested

We build on the existing *engineered*, fixed-pair, one-bit-message experiment with task progression A=XOR -> B=XNOR -> C=AND -> return A -> return B. Cells never receive the task identifier or correct gate action; the simulated world computes reward. A sender encoder learns in A, then remains frozen by the experiment's prescribed schedule. Each cell's receiver has a four-slot action lookup indexed by **its own input bit and the neighbor's learned one-bit symbol**, with up to three private memory banks. Old memories are probed using local received reward, and a new NumPy bank is allocated on demand when the detector declares existing banks inadequate. The *structure and maximum capacity* are not discovered by learning.

Compared four hand-authored detector methods on development seeds 11–12:

1. `relative_ema` (pre-existing control): compare short and historical reward moving averages. Fast adaptation but false positives under reward scarcity.
2. `absolute_floor`: require a severe drop below 55% recent success. Reduces some false alarms, but misses changes in a *specific* input case when average reward stays high.
3. `credible_guard`: Beta(1,1) success posterior with a normal-approximate 95% upper bound below 55%. This is an approximate binomial confidence heuristic, **not full Bayesian change-point detection**. Also misses high-average yet selectively wrong policies.
4. `conditional_signature` (new experimental mechanism): maintain **separate local rolling reward histories for four input/message slots** in each private memory bank. Raise a novelty alarm if the preexisting relative-average detector detects a decline AND either (i) overall success posterior's upper bound is under 43%, or (ii) an adequately sampled input slot's upper bound is under 43% **while** another slot's lower bound is above 52%. Thus uniformly reduced resources are treated differently from a pattern of selective failures. All thresholds, slot count and memory size were selected by us during development.

Each cell sees only its own reward pulses, private cue and received peer symbol. No task-rule labels, hidden targets, true dropout probabilities or global metrics are passed to the policy or detector. NumPy executes all cells centrally for efficiency, but updates are independently computed.

## Why the first improvements failed

With trained XNOR (B) policy in a new AND world (C), the two rules agree on **three of four input combinations**, so a good old policy may still obtain roughly 75% success on C. A detector using only mean success sees a modest drop, mistakes the old bank as still good and fails to create the third memory. This is why both `absolute_floor` and `credible_guard` achieved only ~66–72% C in pilot conditions even though they suppressed false memory growth under a 20% resource pulse loss.

The conditional signature can identify the *specific* incorrectly handled input combination even when total reward is still relatively high.

## Development pilot

Two seeds (11,12), 24 cell-pairs, 3500 episodes for each A/B/C training stage, 900 per return; environmental resource pulse dropout 8% during learning. Four methods compared against unchanged A->B->C->A->B schedule plus unchanged stationary-rule reward dropout 8%,20%,35%,45% stress probes of 1000 episodes per seed.

| Method | Final C gate correctness | False memory allocations per 48 cells at 20%, 35%, 45% dropout (average) |
|---|---:|---|
| Relative EMA (control) | 92.7% | 48 / 48 / 48 |
| Absolute floor | 71.7% | 0 / 12 / 48 |
| Credible guard | 71.7% | 0 / 9 / 47 |
| Conditional signature | **94.1%** | **0 / 0 / 6** |

This pilot was used to select the conditional method, **not** as unseen confirmation.

## Independent seed validation

We fixed the code/thresholds and used never-tuned seeds **3001,3002,3003,3004,3005**. Matching settings: 24 pairs (48 cells), training A/B/C 3500 episodes each, returns A/B 900 episodes each, dropout 8% during learning; independent stationary XOR stress 1000 episodes at 8%,20%,35%,45% resource-pulse dropout.

| Method | A | B | C | Return A | Return B |
|---|---:|---:|---:|---:|---:|
| Prior relative EMA | 92.1% | 92.5% | 93.2% | 92.7% | 92.9% |
| Conditional signature | 92.1% | 92.5% | **93.1%** | 92.7% | 91.4% |

| Stable XOR resource dropout | Legacy false new memory cells /48 | Conditional false new memory cells /48 |
|---|---:|---:|
| 8% | 0.4 | **0** |
| 20% | 47.2 | **0** |
| 35% | 47.2 | **0** |
| 45% | 47.2 | **0.4** |

These are means across the five new seeds. Under 45% scarcity, exactly **two** needless extra bank allocations occurred in a single seed; the other four seeds had zero. AND C test per-seed late gate success (conditional): **91.69%, 94.21%, 90.92%, 94.17%, 94.69%**. No significance hypothesis test is asserted from only five seeds.

Full raw results (10 task-sequence evaluations + 40 stationary reward-stress evaluations): [JSON](../results/audited/2026-10-27-conditional-drift-holdout.json).

SHA-256: `232d6c35021169b13f46cd5ec98f0564f51fab4352780f10ded7a7e864aba403`.

The task-success percentages measure **correct gate actions** even when its resource pulse is randomly withheld; this is intentional to separate policy competence from available resources. Memory creation decisions use *received reward pulses*, including random withholding.

## Cost and methodological limitations

- The new detector uses extra per-cell storage: a 100-step rolling reward history per bank plus four 48-entry rolling histories per bank, **292 bytes per possible bank per cell** for byte-valued buffers alone, plus counters and model weights. A three-bank cell would need 876 such bytes, versus only 96 float64 bytes for its 12 action logits. This experiment has **not** optimized memory-energy tradeoffs; a more expressive detector spends more resources.
- At 45% reward withholding, two false allocations still occurred across the five validation runs. More severe loss or selective loss correlated with input cues can mimic a changed rule. Without informative interactions, the cause of a reward change can be **non-identifiable** from local observations alone.
- The comparison is intentionally not equal in memory cost: the new detector has extra input-conditional history, while prior EMA does not.
- Fixed pairs, one-bit messages, three authored Boolean task rules, engineered detector thresholds, sender frozen after A, central execution and a three-bank cap remain.
- Only five independent seeds and these synthetic conditions were tested; no guarantee for real-world tasks, human language, agentic capabilities or autonomous abstract reasoning.
- Local callback code and its result semantics are separate from the Meta World Designer; **no Meta Model API call occurred**, and no claim is made about credit balance or charges.

## Reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m emergent_cells.experiments.robust_drift --seeds 3001,3002,3003,3004,3005 --pairs 24 --phase-steps 3500 --stress-steps 1000 --methods relative_ema,conditional_signature --output results/runs/conditional-reproduction.json
```

## Next genuinely stronger falsifiable milestone

1. Same task under **cue-dependent dropout** (rewards disappear preferentially for one input combination): the conditional detector may then falsely infer a new task. Run this adversarial stress, log false alarms and measure per-context calibration.
2. Test **stochastic task changes** (partially probabilistic XOR/XNOR/AND rather than abrupt deterministic switches) and asynchronous independent rewards.
3. Replace manually fixed thresholds with local posterior predictive hypotheses for both *resource availability* and *task mapping*, with a bounded memory budget and pre-registered held-out tasks/seeds.
4. Measure implementation time, byte writes and actual electrical energy on the laptop before any energy-efficiency claim.

## Follow-up: input-correlated scarcity and an additional physical-style sensor

A stronger negative control withheld resources specifically when both hidden private cues were `11` while **keeping the XOR rule unchanged**. On new seeds 4001–4003, reward-only conditional detection and the old EMA each created **44–46 unnecessary memories among 48 cells**. This illustrates a causal ambiguity: input-correlated resource scarcity resembles a task-rule change in passive reward observations.

In a **separate additional-information condition**, each cell received an idealized binary *local resource available / unavailable* observation independent of whether it operated the gate correctly. Missed opportunities were excluded (optimistically imputed as non-failures) from its change detector; its learning reward still reflected real obtained simulated energy. This additional sensor prevented false bank allocations on all three fresh adversarial seeds (0/48), while retaining **93.0% C-rule performance** and **91.3% / 92.1%** upon return to A/B. The sensor itself is programmed and not available from physical hardware; it makes the comparison **unequal-information**, not an improvement in reward-only inference.

See [adversarial experiment, caveats and raw data](ADVERSARIAL_SCARCITY_LOCAL_SENSOR_2026-10-27.md). No universal robustness, real joule efficiency, unsupervised sensor discovery or new-task-general cognition is established.
