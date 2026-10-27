# October 27 — Learning a local one-bit communication protocol

**Research milestone, not a claim of general intelligence.** This implementation and its results were produced October 27, 2026. The existing engineered paired-gate task from October 26 remains separate historically. No proprietary research source was copied.

## Question and relevant literature

Can individually learning cellular units discover a *useful discrete symbol mapping* using only their own reward change, without an XOR basis term in their policy?

This follows the scientific question addressed by Foerster et al., *Learning to Communicate with Deep Multi-Agent Reinforcement Learning* (NeurIPS 2016, https://arxiv.org/abs/1605.06676 ), which developed RIAL/DIAL for emergent communication. Our demonstration is much smaller and specifically tests **independent private reward updates**. Unlike DIAL, there is no gradient across the message channel; unlike earlier Oct 26 work, we do not supply a private-cue × neighbor-cue interaction feature.

## Task and protocol

- 128 independent fixed two-cell resource gates, 256 cells total (10-seed study).
- At each episode, the environment gives each cell one private random binary cue.
- Each cell **emits one stochastic binary symbol**, sampled using only its own cue and two private trainable encoder weights.
- It receives the partner's emitted symbol over a fixed one-bit local link.
- It selects its binary lever action using its own cue and the received symbol. Each cell's receiver is a **generic four-entry 2×2 conditional action table**, with four private trainable values; it is **not a hard-coded parity function**.
- Only the world knows that both cell lever actions must equal the XOR/parity of the two private cues to open the finite-resource gate.
- The world returns a local resource pulse to each participant if the gate opens; each cell adjusts its **own** encoder and decoder weights with a local reward-modulated stochastic policy update. Neither cell observes the gate target, the other's weights or a global population score.
- Explicit simulated metabolism and bit-transmission costs are paid in every condition, including when delivery is blocked, to keep attempted bandwidth/cost comparable. These energy values are **bookkeeping proxies, not joules**.

## Reproducible, matched controls

All seeds use the same starting local weights, train/test cue streams and stochastic samples for each condition. Training is 8,000 episodes, testing is 500 *new cue samples under the SAME gate rule*, per seed. The learned weights are frozen during evaluation.

| Condition | Meaning |
|---|---|
| learned_messages | Each cell learns its own sender and receiver; real 1-bit neighbor channel |
| blocked_messages | Same sender attempts and costs; receiver gets constant zero, no sender learning |
| random_messages | Sender bit independent of private cue, receiver trains |
| fixed_sender | Weak random-initial sender remains fixed, receiver trains |
| frozen_actor | Sender may train but receiver never changes from random initialization |

Additional same-trained-policy tests replay identical fresh cues and random draws after **blocking reception** or replacing received bits with random ones. There is also an 80%-ones cue-distribution test; **this is input-frequency shift only**, not a new task.

## Results — 10 seeds, 128 pairs

| Condition | Mean jointly opened resource gates |
|---|---:|
| learned_messages | **98.23%** |
| blocked_messages | 49.35% |
| random_messages | 48.36% |
| fixed_sender | 48.31% |
| frozen_actor | 25.63% |
| already-trained model, reception blocked | 25.03% |
| already-trained model, message randomized | 25.05% |
| already-trained model, 80%-ones inputs | 98.16% |

Learned-vs-blocked paired mean difference: **48.89 percentage points**; learned condition wins on each of the ten development seeds. Average private sender separation between its message probabilities for the two cue values is approximately **0.9953** (absolute difference), indicating useful symbols whose polarity may differ by individual cell.

These are *exploratory development outcomes*, not a preregistered confirmatory trial or inference about unseen tasks. See [raw 50 condition-by-seed outcomes](../results/audited/2026-10-27-learned-messages-10seeds.json).

## Reproduce

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python -m emergent_cells.experiments.learned_messages --seeds 101,102,103,104,105,106,107,108,109,110 --pairs 128 --train 8000 --test 500 --output results/runs/learned-10seeds.json


## What this supports — and what it DOES NOT

**Supported, narrowly:** learned local, *arbitrarily polarised* binary messages carry a necessary missing cue; agents learn their own send/receive mapping using local reward, and the learned policy fails on a direct message knockout. No explicit XOR calculation was programmed inside an agent.

**Not demonstrated:** spontaneously chosen partner assignment, self-organised network topology, learned size/bandwidth of communication channels, compositional language, new task-rule transfer, persistent memories, material/energy-constrained growth, substrate independence, low electrical consumption or general intelligence. The external gate XOR rule, fixed pairing, generic four-entry receiver table, direct payoff, one-bit noiseless channel and simulator remain designed by us.

**Next falsifiable gate:** randomise and change gate functions (not just cue samples), require cells to infer task context through sparse local interactions, measure training-to-transfer and no-message controls, and reduce hand-designed lookup capacity while preserving locally learned policies.
