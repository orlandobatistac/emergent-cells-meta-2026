# Locally coordinated causal probes: distinguish a changed action rule from missing resources

**Date:** October 27, 2026. New experimental work for Emergent Cells Lab. This is a small digital simulation of paired-cell communication and intervention, **not** general intelligence, true substrate-independent computation or physical joule-efficient hardware.

## Scientific motivation and relevant literature

- Xie et al., *Learning to Explore in POMDPs with Informational Rewards* (ICML 2024): https://proceedings.mlr.press/v235/xie24a.html — information-gathering actions in partially observed tasks.
- Doshi-Velez, Pineau and Roy, *Reinforcement learning with limited reinforcement: Using Bayes risk for active learning in POMDPs* (Artificial Intelligence, 2012): https://doi.org/10.1016/j.artint.2012.04.006 — belief-informed choice of informative actions.
- Liu et al., *Cooperative Exploration for Multi-Agent Deep Reinforcement Learning* (ICML 2021): https://proceedings.mlr.press/v139/liu21j.html — coordination can improve which information agents collect.
- Eccles et al., *Biases for Emergent Communication in Multi-agent Reinforcement Learning* (NeurIPS 2019): https://papers.neurips.cc/paper_files/paper/2019/hash/fe5e7cb609bdbe6d62449d61849c38b0-Abstract.html — independent communicating learners struggle with joint exploration.

These sources inform our problem formulation. This study does not claim to reproduce their algorithms.

## Why the previous result was limited

Our prior reward-signature detector was fooled by a stationary XOR world in which resources disappeared *only for private input combination 11*. If the paired cells always performed their original successful action and received no reward, they could not tell whether (a) their action was now wrong or (b) they were correct but no resource was available.

A separate idealized resource-availability sensor solved that particular test, but it provides **extra information** that may be hard or expensive to realize on hardware.

## New question and experimental method

Can cells **actively test an alternative action**, without any ideal resource sensor or correct-answer label, and use the result to determine whether a new memory is needed?

Each of 48 pairs (96 cells) first independently learns an encoder and decoder for a single-bit message using local reward (reuses `LocalProtocolPopulation` from earlier research). Learning lasts 4500 episodes per seed. The local sender/receiver weights are **frozen** during the test, so newly obtained improvements are due to information gathering, not silently re-training the prior protocol.

A cell thereafter has only: (1) its own private binary cue, (2) its neighbor's independently learned one-bit symbol, (3) its own gate reward (shared within the pair), and (4) an optional **one-bit probe request** from its neighbor. There is **no task label, correct target or resource availability sensor**.

Three competing hypotheses are encoded in every cell, for every one of its four observed (cue, symbol) slots:

- Old rule still applies and resources are normal.
- Old rule still applies but resources for this observation are scarce.
- The expected action for this observation is reversed.

The prior probabilities, hypothetical reward likelihoods and entropy thresholds are **hand-coded** as part of the experimental controller. In information-gain mode, each cell asks whether trying the **inverse of its existing learned action** would reduce uncertainty more than repeating the old action. It can send a one-bit probe request. If **either** member asks to probe, both invert their respective actions on that episode. This hand-designed negotiation ensures the joint gate test is informative when both participants would otherwise act correctly. Their observed rewards independently update their private Bayesian-like hypothesis vectors.

When a cell accumulates strong evidence of a flipped mapping in one input slot, it stores one learned **binary correction bit** for that slot and applies it on later encounters. That is *new small software memory*, **not** adding new cells, general-purpose learned procedures, autonomous discovery of Bayesian rules or physical memory growth. A classical NumPy scheduler still executes the simulation centrally.

### Controls

- `passive`: no counterfactual probes; cells repeat their pretrained actions.
- `random`: uncertain cells request probes stochastically.
- `periodic`: uncertain cells request a probe at a fixed visit interval.
- `information_gain`: requests selected using expected reduction in hypothesis entropy.

All modes share **identical** pretrained weights, world cue streams, resource-dropout draws and pair membership for each seed. No mode gets world-rule labels or availability bits.

### World scenarios

- `stable`: original XOR gate, resource probability .92.
- `scarcity_11`: still XOR, but **no resources at all** for joint private cues 11; other cases retain normal availability.
- `changed_11`: the required action flips **only** when joint cues are 11; resources remain available.
- `uniform_drought`: still XOR, but resource availability falls uniformly to .55.
- `global_change`: action mapping flips for all cue combinations.

Critically, for a cell that does not intervene, `scarcity_11` and `changed_11` both produce nearly identical failure patterns on input 11. A successful alternate action distinguishes the changed mapping from absent resources. Pretraining errors explain the small nonzero passive success in the changed-rule case.

## Exploratory pilot and threshold development

The pilot (seeds 11,12; 24 pairs, 3500 pretraining, 450 evaluation episodes) demonstrated that all three active approaches can learn a changed action rule. The original `information_gain` threshold .06 was evaluated against .12, .22, .35 and .5. Thresholds up to .22 were similarly effective; .35 and .5 slowed adaptation. The **original .06** value was retained for independent validation, not selected to maximize held-out performance. A higher threshold may reduce messages but also lower success.

## Independent validation on new seeds

After the above development, seeds **1001–1005** were run with a fixed model: 48 pairs, 4500 local-message training episodes, 550 evaluation episodes, five scenarios × four policies × five seeds = **100 paired comparisons**.

| Method | Critical 11 accuracy under actual changed mapping | Overall gate accuracy under changed mapping |
|---|---:|---:|
| Passive | **0.4%** | 74.8% |
| Periodic coordinated probes | 93.6% | 98.1% |
| Random coordinated probes | 95.7% | 98.6% |
| **Local Bayesian information gain** | **98.1%** | **99.2%** |

Under **scarcity without a rule change**, information-gain cells retained **97.7%** correctness on cue 11, and allocated only **0.4 incorrect correction bits on average per 96 cells**, versus zero for the passive no-probe baseline. Under stable conditions, the information-gain mode also averaged about **0.6 incorrect correction bits per 96 cells**; low is not zero.

| Scenario | Bayesian conditional-11 correct actions | Avg learned correction bits, all 96 cells | Avg probe request bits (per 48 pairs × 550 episodes) |
|---|---:|---:|---:|
| Stable XOR | 99.2% | 0.6 | 6.4 |
| Scarcity only for 11 | 97.7% | 0.4 | 195.4 |
| Changed mapping only for 11 | **98.1%** | 95.6 | 110.4 |
| Uniform drought | 99.3% | 0.8 | 199.2 |
| Global mapping change | 98.0% | 380.8 | 444.0 |

The correction-bit counts for `global_change` span all four private signature slots for both members of every pair; they are **not** numbers of new biological cells or entire decoder banks. The experiment starts with a fixed capacity of four correction flags per cell.

**Costs matter:** in the changed-11 world, the Bayesian approach averaged 110.4 extra request bits and **1.85 simulated cost units** for coordinated probes/request signaling; random exploration averaged 58.2 request bits and 1.62 simulated cost units, while achieving 95.7% cue-11 correctness. Bayesian exploration yields better task success in this synthetic benchmark **at additional probe/message cost**. It is NOT automatically best on an objective that heavily prices information or energy.

Per-seed critical-11 information-gain success: **98.55%, 98.51%, 98.48%, 98.51%, 96.57%**. Pair-matched passive: **0%, 0%, 0%, 2.13%, 0%**. All five seeded comparisons favor information gathering. Five seeds are still too few for robust uncertainty estimates over arbitrary settings.

Full unedited records, including 100 per-method/scenario seed-level rows, costs and per-cue correctness:

[Machine-readable results](../results/audited/2026-10-27-active-causal-probing-fresh-seeds.json)

SHA-256: `ca5e120b768eedcf0448841c4196bed6918c1660098f211f280f597669a03351`

## Claims supported and limits

**Supported in this engineered experiment:** after learning a local binary message, independent artificial cells can coordinate inexpensive counterfactual actions that reveal a changed rule, avoid most false memory growth under resource scarcity, and preserve original actions when the rule still holds. Comparing several active controllers against passive controls makes the *causal contribution of intervening* explicit.

**Not demonstrated:**
- An autonomously learned curiosity objective. The Bayesian likelihoods, three hypotheses, information-gain formula and intervention triggers are authored, not discovered by cell learning.
- Arbitrary task mapping changes or long-term control with more than two hypotheses, unknown real-world signal structures, real physical reward interpretation or self-invented sensors.
- Truly substrate-agnostic computation, physical power savings, neural AGI, arbitrary natural language, general tool-use agents, or automatically expanding topology.
- Provably optimal diagnosis when resources are permanently absent under *both* possible actions. Some worlds remain causally unidentifiable even with probing.
- Equal-resource comparison: the information-gain agent uses probability vectors and compute beyond passive/random exploration, and only approximate symbolic costs are accounted for.

## Reproduction

```bash
python -m pip install -e .
python -m unittest discover -s tests -q
python -m emergent_cells.experiments.active_probes --seeds 1001,1002,1003,1004,1005 --pairs 48 --pretrain 4500 --horizon 550 --output results/runs/replicate-active-probes.json
python -m emergent_cells.demo.server
```

Visit localhost in the laptop browser for a **smaller live demo** with development seeds 11 and 12; do not confuse it with the held-out analysis.

## Next experiments that would be genuinely stronger

1. Learn the **intervention decision itself** using individual reward and an explicit information budget, compare against hand-coded information gain, random and periodic exploration on untouched task families.
2. Remove perfect shared-gate reward, use asymmetric rewards and imperfect communication; quantify coordination failure and extra messaging cost.
3. Generate stochastic nonstationary task rules and scarcity processes with hidden states; measure calibration of agent posteriors, not just success.
4. Include active exploration alternatives that are **not just action inversion**, and ask cells to choose which experiment to perform.
5. Measure real CPU time, resident memory bytes, and energy through calibrated hardware meters before claims of low consumption.
