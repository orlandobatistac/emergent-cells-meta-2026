# Dual-cue cooperative gate: experiment design

**Research status:** exploratory engineering experiment created October 26, 2026. The participant guide gives October 27–November 4 as the hackathon building period. No part of this October 26 experiment is claimed as eligible new hackathon work without explicit organizer clarification.

## Research question

Can a population of independently learning cells demonstrate a causal, measurable benefit from local information sharing when neither cell can infer the correct group action from its own observations?

The prior foraging experiment showed no survival benefit from local messages. This new task deliberately removes complete local knowledge. It tests a *necessary-information* condition rather than the broader existence of emergent cognition.

## Experimental world and rules

- 128 predefined local pairs (256 cells); two nearby, fixed-position cells operate one resource gate.
- Each cell observes only its own independent binary cue (+1 / -1).
- The gate provides a shared finite resource pulse **only when both cells select the action equal to the XOR parity of both hidden cues**.
- In the local condition, each cell transmits its own cue to its adjacent partner over a one-hop channel, paying a simulated message cost.
- Cell action is sampled from a logistic policy with four local features: constant, private cue, received cue, and **private cue multiplied by received cue**.
- Each cell independently changes its own four weights using reward-modulated policy gradient, where the reward is solely its own before/after energy change.
- The gate's success criterion is computed in the simulated environment. The target/parity bit is *never passed to cell observations or the local weight update*.
- Training and testing use independent cue/action random streams. During evaluation, learning is frozen.

## Conditions

| Name | Cell perception | Updates |
|---|---|---|
| none | private cue only | independent local weight learning |
| local | private cue + true nearby partner cue | independent local weight learning |
| scrambled | private cue + foreign-pair cue | same learning; deliberately invalid negative-control channel |
| frozen | private cue + nearby partner cue | fixed random initialized policy |
| far | partner physically displaced beyond radius | local weight learning without messages |

An extra **message knockout** replays held-out cues and action randomness against the *same trained local weights* with the channel disabled. It isolates dependence on the message after training, avoiding the confound of comparing different trained parameter sets.

## Fairness and confounders

- All conditions use the same per-seed initial policy RNG, cue streams, action sampling streams, finite resource budget, metabolic cost, and training/test episode count.
- The local, frozen and scrambled conditions incur a message energy cost; the no-message and far conditions do not. These are simulated units only.
- Policies start from the same weights at each seed. Their learned trajectories may diverge and are not forced to match.
- No generalization across *different gate rules* is measured. The test only changes private cue sequences and stochastic action draws.
- The local channel is fixed and lossless; **message format and pair partner assignments are supplied**, not learned.
- The product feature enabling an XOR solution is explicitly **engineered by us**. This experiment does **not** show discovery of parity representation or emergent intelligence.
- The task construction creates a hard information deficit; strong local-vs-none differences are expected under a successful learning rule. Classical engineered solutions could solve this directly.
- Two cells have a common resource consequence but no separately trained global critic or centralized neural network. The simulator still batches all updates using Python/NumPy, and the environment's gate physics uses a global implementation.

## How to reproduce

From a Python 3.10+ repo environment with NumPy installed:

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python -m emergent_cells.experiments.dual_cue --seeds 1,2,3,4,5,6,7,8,9,10 --pairs 128 --train 1200 --test 300 --output results/runs/dual-cue-reproduced.json

Output includes all 50 seed×condition rows and separate local-knockout values.

## Falsifiable next directions

1. Remove the engineered interaction/product feature; allow cells to **learn their own message encodings** instead of forwarding an input bit.
2. Replace predetermined pairs with dynamically available contacts, finite message bandwidth and imperfect/attenuated signals.
3. Introduce novel environmental rules and delayed rewards to test adaptation beyond a memorized four-case task.
4. Explicitly account for connection maintenance, memory, metabolic/message costs and constrained material growth.
5. Connect a Meta model as an external *experiment designer*, never as a cell policy.
