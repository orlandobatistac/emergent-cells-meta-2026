# Emergent Curiosity: learning when to explore using private reward and local learning progress

**Research milestone:** October 27, 2026. This experiment is a small, digitally simulated multi-agent local learning system. It does **not** prove biological intelligence, general artificial intelligence, physical energy efficiency, autonomous discovery of a curiosity formula, or a universal agent framework.

## Why this is the next falsifiable test

Earlier counterfactual probing experiments demonstrated that an **authored** Bayesian information-gain rule could coordinate useful inverted-action tests of a changed XOR gate. However, the artificial cells did not learn **when to issue that test**. Our new research question is whether each cell can learn a tiny probe/exploit decision policy from **its own actual rewards**, and whether local intrinsic learning-progress rewards help it stop wasting probes during resource scarcity.

Relevant peer-reviewed work:
- Pathak et al., *Curiosity-driven Exploration by Self-supervised Prediction*, ICML 2017: https://proceedings.mlr.press/v70/pathak17a.html — motivating intrinsic prediction-based exploration. Our implementation uses discrete belief-entropy reduction, **not** their ICM model.
- Xie et al., *Learning to Explore in POMDPs with Informational Rewards*, ICML 2024: https://proceedings.mlr.press/v235/xie24a.html — learning information-gathering policies in partially observed environments. Our solver is a much smaller reward-driven tabular method.
- Parisi et al., *Interesting Object, Curious Agent: Learning Task-Agnostic Exploration*, NeurIPS 2021: https://ai.meta.com/research/publications/interesting-object-curious-agent-learning-task-agnostic-exploration/ — separate train/test exploration and transferring exploration skill.
- Burda et al., *Large-Scale Study of Curiosity-Driven Learning*, ICLR 2019: https://openai.com/index/large-scale-study-of-curiosity-driven-learning/ — intrinsic reward can be confused by stochastic environmental factors.

We borrow a research question and experimental patterns, **not** results or an existing model from these papers.

## Exactly what learns, and what remains engineered

- A fixed local binary sender/receiver protocol is learned first by individual cells using private, reward-modulated updates. Its sender and receiver are then frozen during the probe-decision experiment.
- Each cell has a private table with **six coarse states × two choices**: *continue exploiting* or *send a 1-bit request to probe*. Its state is derived from the prior experiment's hand-designed Bayesian hypotheses (old rule, resource scarcity, flipped rule), **plus whether its own last probes repeatedly failed**. These six observation features and all Bayesian likelihoods are hard-coded. The **probe/exploit choice**, in contrast, comes from a **locally trained action-value table**, not a fixed information-gain formula.
- The world supplies a private cue bit and a partner's learned 1-bit symbol. It does **not** send the current rule, the correct action, resource-availability flags or the reason a reward disappeared.
- A probe request from either cell causes the fixed pair to invert its normal gate actions **together** on that episode; this coordination primitive is engineered. The cell then updates its private belief from the received reward and stores a binary action correction for confident mapping flips.
- Each cell trains its own action values with a reward-modulated local incremental update. Net reward equals locally obtained resource pulse minus synthetic intervention/request costs. A positive **intrinsic reward** is additionally available to probe actions when that action *actually reduces its local Bayesian uncertainty*. This is **realized posterior learning progress**, not a privileged world label and not a hard-coded **decision** to probe. The intrinsic reward function is manually selected. Cells do not share Q weights or a global critic.
- A known credit-assignment problem was discovered during development: when the partner initiates a probe, our own no-probe choice did not cause the action taken. The final training update only credits private requests where the partner did not independently request one.

### Training and controls

For each seed, 48 pairs (=96 locally learning cells) first learn the binary communication protocol for 3600 steps. The curiosity policy then trains for **320 episodes of 75 interactions**, with a seeded curriculum mixture of *unchanged XOR*, *changed XOR for cue 11*, *scarcity for cue 11*, *changed XOR for cue 01*, *scarcity for cue 10*, *global action flip*, and *uniform reward drought*. All of those scenario families are **known to the training distribution**, though world identities are private from the cell learners. **Do not claim that changed_01 or scarcity_10 represent novel task-family transfer**.

Evaluation freezes each local Q table and runs 500 fresh-step interactions per scenario. We compare against:
- An **untrained version** of the exact same local probe policy (no Q updates).
- The old passive policy, random coordinated probing, periodic probing, and authored Bayesian information-gain decision (with the same sender/decoder weights for each seed in the common five scenarios).

The old controls and new learned controller use the same distribution and saved sender weights for each seed, but not bit-identical environmental RNG samples: individual trial draws differ. Comparisons are per-seed and distribution-matched, **not exact matched random-trajectory causal replay**.

We report correctness (even if a resource pulse is withheld), critical-11 correctness where defined, request bits, correction flags, and synthetic request/intervention costs. No CPU time, RAM usage or real joules are measured.

## Development attempts (all retained)

1. **V1** locally trained two-choice action table without explicit previous-probe-failure observation: **negative**. It explored far too often under scarcity (e.g., roughly 1000 requests on a 24-pair, 280-step scarcity pilot), performing worse than authored Bayesian decision.
2. **V2** added a locally observed history of repeated failed probes to its state: **partially positive** in a 2-seed pilot, but a later 5-seed evaluation (1001–1005) showed only **92.1% changed-11 correctness** compared with **98.0%** for authored Bayes, with higher scarcity costs. The first attempt at heldout generalization was therefore **not** a confirmation and is now considered DEVELOPMENT evidence.
3. **V3** credited joint interventions as if each cell requested them: **negative**, worsened performance (about 89% changed-11 in pilot).
4. **V4** distinguished previously untested uncertainty from previously probed uncertainty in a larger state table: **negative**, over-suppressed useful probes (~76% changed-11 in pilot).
5. **V5** retained private credit assignment and failed-probe history while adding an **intrinsic learning-progress reward** from local posterior entropy reduction. This was selected only after development seeds 11–12. Initial pilot: **96.1% changed-11 accuracy, 97.3% scarcity-11 accuracy**, compared with **94.9%/97.2%** for authored Bayes in that pilot, with more requests than random probing. This is the candidate for FRESH seed verification.

Saved negative, intermediate and selected-pilot JSON files are retained in `results/exploratory/`.

## Scientific uncertainty and remaining gaps

- The control state, belief hypotheses, likelihoods and probe coordination primitive are authored. Only the *decision to request* and its private action values are learned. A positive result would demonstrate **bounded local policy learning of information-gathering behavior**, **not** a discovered general-purpose curiosity algorithm.
- Posterior-entropy improvement can suffer the same noisy-TV pathology as prediction-error curiosity when randomness cannot be controlled. This is why noise and scarcity controls are essential.
- An untrained actor probes randomly on exact-value ties and therefore wastes huge numbers of requests. A successful trained-vs-untrained comparison is necessary but **not sufficient**; random and authored Bayesian policies remain serious baselines.
- The simulated binary world uses two-cell fixed pairings, fixed rule families, a noiseless learned message bit, an extra probe bit and hand-coded 4-slot correction memory. The higher information/memory cost of Bayesian belief tracking and the intrinsic reward computation are NOT fully priced.
- Results for uniform drought and a globally flipped rule should be reported separately; the **critical 11** metric is only defined for the scenarios where a particular input combination is designated. For other scenarios use overall correctness.
- No autonomous physical cell growth, independent hardware substrate, real electricity use, natural language reasoning, tool-using general agents, or AGI has been demonstrated.

## Reproduction

From the repo root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m emergent_cells.experiments.learned_curiosity --seeds 2001,2002,2003,2004,2005 --pairs 48 --pretrain 3600 --meta-episodes 320 --meta-horizon 75 --eval-horizon 500 --output results/runs/reproduce-curiosity.json
```


## Next stricter test

1. Learn the state representation itself rather than supplying a hand-coded three-hypothesis Bayesian table; validate posteriors are calibrated under independent reward noise.
2. Compare learned curiosity, Bayesian and random baselines with a **pre-registered equal intervention/message/compute budget**; use matched identical environment samples and confidence intervals with more seeds.
3. Test unseen task *families* (not merely new seeds or cue combinations previously present in training) and new partner relationships, noisy message channels and asymmetric pair rewards.
4. Learn multi-step experiments (more than just XOR action inversion) and test when the cell should not probe because additional information is fundamentally unavailable.

## Final independent-seed results (2001–2005, unchanged selected algorithm)

No tuning was performed using these five seeds. Fixed testing configuration: 48 pairs/96 individual learners, 3600 sender training steps, 320 private curiosity-training episodes × 75 encounters, then 500 evaluation encounters per scenario.

**Critical target metric:** correctly operating the gate when joint private cues are `11`. This is meaningful for selectively changed/scarce `11` worlds. In globally flipped and ordinary worlds there is **no designated critical cue**, so use overall correctness rather than the stored zero-valued `critical` placeholder.

| Method | Changed 11 critical accuracy | Scarcity 11 critical accuracy | Changed 11 request bits | Scarcity 11 request bits |
|---|---:|---:|---:|---:|
| Authored Bayesian expected information gain | **98.0%** | **97.1%** | 113.4 | 193.2 |
| **Trained private Q + intrinsic progress** | **96.6%** | **95.7%** | 138.6 | 272.6 |
| Random coordinated testing | 94.4% | 97.5% | 61.6 | 103.2 |
| Trained private Q without intrinsic progress (ablation) | 92.4% | 95.4% | 109.2 | 296.8 |
| Untrained random-tie Q table | 98.9% | 25.0% | ~17,959 | ~23,827 |

The **untrained** agent can score highly on a flipped mapping simply by probing almost every time; in the unchanged mapping with scarcity, it fails and spends enormous amounts of resources. Therefore cherry-picking changed-rule correctness would be misleading.

**Resource/cost comparison:** for a changed `11` world, trained curiosity costs ~**2.55** symbolic units in requests and interventions per seed/test vs **1.89** for authored Bayes and **1.71** for random. Under scarcity for `11`, trained curiosity costs ~**5.78** units vs **3.22** authored Bayes and **2.87** random. These do not include Bayesian calculations, state-table memory, CPU use or real electrical joules.

**Other scenario results**, learned intrinsic policy (overall correctly selected gate action, not resource payout): stable XOR **98.8%**, uniform drought **97.7%**, global action flip **95.1%**, switched rule for cue `01` **93.6% critical-cue correctness**, resource scarcity only for cue `10` **96.0% critical-cue correctness**. Both changed-01 and scarce-10 types were represented in the meta-training distribution; this is NOT transfer to an unseen task family.

In the five changed-11 seed comparisons, learned curiosity beat authored Bayes in **2/5** seeds, and its **mean was lower** (96.6% vs 98.0%). The stronger claim supported is that intrinsic reward makes the *learned local policy more effective than the same local learner without intrinsic reward*, not that it is globally optimal or better than an authored exploration rule.

Raw 170 evaluations (5 seeds × [5 familiar scenarios × 6 controls + 2 additional scenarios × 2 local controllers]), Q table means and training logs: [five-seed study](../results/audited/2026-10-27-intrinsic-curiosity-fresh-seeds.json). SHA-256 `2ca13884ee1d2cc835b171898bf839b0196b8ff5d5a2a4de674dd8a8c0f582a0`.

## Causal ablation of the added intrinsic reward

To avoid comparing different seeds, we repeated training with `intrinsic_weight=0` vs `0.25`, **same five seeds 2001–2005**, same local population, training schedule, private observations, Bayesian representations and world/task scenario mixture. The intrinsic reward is the only selected training setting changed; during evaluation each local Q table is frozen.

| Outcome | No intrinsic bonus | 0.25 × realized local posterior learning-progress bonus |
|---|---:|---:|
| Changed-11 target correctness | 92.39% | **96.65%** |
| Global-flip overall correctness | 92.41% | **95.14%** |
| Scarcity-11 target correctness | 95.42% | **95.65%** |
| Scarcity-11 request bits | 296.8 | **272.6** |
| Stable XOR overall correctness | **98.96%** | 98.84% |
| Uniform-drought request bits | **462.6** | 530.2 |
| Stable false correction flags per 96 cells | 1.4 | 1.4 |
| Scarcity-11 false correction flags per 96 cells | **0.6** | 0.8 |

**Interpretation:** realized information progress improves selected rule-change learning and reduces requests under cue-specific scarcity in this setting, at the cost of more probes in stable/uniform-noise settings and a slightly worse false-correction count under selective scarcity. This is a *tradeoff*, not dominance.

Raw 50 paired same-seed intrinsic ablation rows: [ablation data](../results/audited/2026-10-27-curiosity-intrinsic-ablation.json). SHA-256 `95aaaa118304a233a0abb07e7aa1168754966fbc1756b1fa575e86eb09346a3d`.

**Limit:** all tested seed values are new to the final algorithm selection but the task families are synthetic and already known to the curriculum. Five seeds do not establish statistical significance or real-world robustness. This work is a bounded proof of *trained* information-seeking policy, not a general curiosity engine.
