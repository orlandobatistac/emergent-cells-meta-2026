# Contextual cellular memory: A -> B -> A — October 27, 2026

**Scope:** exploratory repeated-task learning with privately trained sender and receiver policies. This is NOT demonstration of autonomous change detection, general task transfer, general intelligence, or hardware efficiency.

## Research basis

- Foerster et al., *Learning to Communicate with Deep Multi-Agent Reinforcement Learning* (NeurIPS 2016), https://arxiv.org/abs/1605.06676 : multi-agent communication and discrete messages.
- Hamadanian et al., *Online Reinforcement Learning in Non-Stationary Context-Driven Environments* (ICLR 2025), https://proceedings.iclr.cc/paper_files/paper/2025/hash/fb21dae9e8710a272c0a848f71553-Abstract-Conference.html : policy forgetting under non-stationary context.
- These papers informed the *question and controls*; this project does **not** reproduce or claim their methods.

## Exact experimental task

128 cells arranged in 64 **predefined fixed neighbor pairs**. Each receives one binary private cue. In world A, both must select the parity/XOR action of their two private cues to open their gate. World B reverses the environmental rule to XNOR. **No agent observes the required output/action label.** Each agent has a private stochastic 1-bit cue encoder (two parameters) and four generic cue+message action logits. It updates via its **own** energy reward, no global critic or inter-agent backpropagation.

Phase A: 4,500 training episodes per pair; test A on new inputs and test the unseen reversed B rule, with no updates during testing.

Phase B: 3,500 training episodes on the reversed world; **the trained sender is frozen in every variant**, isolating recipient-policy adaptation and preserving the original message vocabulary.

Return A: immediately test the original world **without training**, also test the learned B rule again.

The **context-memory** variant receives a one-bit **externally provided local season marker**, 0 in A and 1 in B, used only to select one of two 4-entry receiver memory banks. The reward still trains the chosen local bank. The inactive bank's weights are simply not modified. The marker is **not the correct XOR/XNOR answer**, but it IS a supervised change/context indicator and is important to the result.

## Controls

| Variant | Explanation |
|---|---|
| context_memory | Eight receiver values (two banks × four slots), season-selective training |
| single_policy | Four used receiver values (bank 0) learned sequentially with no context |
| reset_on_switch | Resets the ONLY active bank at phase B onset, then re-learns B; memory of A is overwritten |
| blind_memory | Two banks allocated, but no season signal; always uses bank 0 |
| frozen_on_switch | Policy trained A, then decoder learning disabled for B |
| no_messages | Neighbor symbols never reach the receiver; all other world rules remain |

Every variant uses equal individual energy reward and message cost, same initial weight RNG, task cues, action stochastic streams, evaluation seeds and training horizon. The contextual variant gets **extra memory capacity and a context bit**; these costs are not physically priced in the simulation. There is no claim of equal parameter-count/memory efficiency. The reset control uses a deliberately externally triggered reset, not a naturally learned change detector.

## Exploratory pilot and untouched-seed replication

The pilot on development seeds 11,12,13 used 64 pairs, train-A=4500, train-B=3500, test=200:
- Context memory ~96.3% on A before, 96.4% on B after, 96.3% on A immediately upon return.
- Single policy ~96.3% A, ~0.3% B, ~96.1% A return. It *failed to adapt*, so it did NOT exhibit catastrophic forgetting.
- Reset-on-switch ~96.3% A, ~96.4% B, ~0% A return. Reset restores plasticity but destroys its only learned decoder.

With algorithm and hyperparameters fixed, **five independent seeds 1001–1005** were tested identically. No parameter modifications were made after the pilot.

| Variant | A before | B before training | B after | A at return without training |
|---|---:|---:|---:|---:|
| context_memory | 96.51% | 24.62% | **96.40%** | **96.51%** |
| single_policy | 96.51% | 0.18% | 0.19% | 96.43% |
| reset_on_switch | 96.51% | 0.18% | **96.40%** | 0.19% |
| blind_memory | 96.51% | 0.18% | 0.19% | 96.43% |
| frozen_on_switch | 96.51% | 0.18% | 0.18% | 96.51% |
| no_messages | 48.79% | 48.45% | 49.13% | 49.45% |

Individual contextual-memory results:

| Seed | A before | B learned | A returned |
|---|---:|---:|---:|
| 1001 | 96.58% | 96.52% | 96.58% |
| 1002 | 96.84% | 96.45% | 96.84% |
| 1003 | 96.02% | 95.92% | 96.02% |
| 1004 | 96.63% | 96.45% | 96.63% |
| 1005 | 96.51% | 96.70% | 96.51% |

Full machine-readable raw output: [five-seed record](../results/audited/2026-10-27-context-switch-heldout.json)

SHA-256: `a7b8b35733e689dc59e69d701530a74c72c947acecab7d5739dac194b0998541`

## Scientific interpretation

**Positive but narrow:** separately stored context-conditioned decoder policies can learn two conflicting fixed-pair coordination rules while preserving one another. Return-to-A performance is preserved *by design* because the old bank is not updated and senders stay frozen. This is a demonstrated function of modular state selection, **not spontaneous discovery of modular memory**.

**Negative and informative:** a saturated single-bank policy cannot relearn the inverted rule with the current small local reward gradients; it did not forget because it never adapted. An explicit reset learns B, but then loses A, exposing the tradeoff that modular state sidesteps.

**Unproven:** adaptation to *unannounced* changes, inference of a context signal from local reward, novel non-parity task rules, autonomous memory creation, change-detection latency, message meaning surviving a changing encoder, material/energy costs of memory and links, growth, hardware joules or general intelligence.

## Reproduction

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python -m emergent_cells.experiments.context_switch --seeds 1001,1002,1003,1004,1005 --pairs 64 --train-a 4500 --train-b 3500 --test 200 --output results/runs/fresh-ABA.json

## Next genuinely stronger falsifiable test

1. Remove the externally supplied season bit; cells must **infer changes from local reward surprise**, negotiate a shared active memory, and recover without a global task label.
2. Run periodic A/B/A/B switches, track how quickly both cells agree, and compare against fresh retraining and fixed-context upper bound.
3. Assign an explicit cost to each private memory slot and transmitted bit, measure success per simulated resource unit, and report the extra capacity required.
4. Test at least three environmental boolean rules (e.g. XOR, AND, NAND) and unseen transitions, not merely symmetric XOR/XNOR inversion. An agent cannot adapt to arbitrary rules without informative rewards and sufficient expressivity.
5. Predefine holdout seeds and measures before tuning. Preserve failures and do not infer natural cognition from an engineered test.
