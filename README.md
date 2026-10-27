# Emergent Cells — Adaptive Worlds

**Meta Global AI Developer Hackathon 2026 · Indie Developer · Push the Frontier**

A public, **new-work** experiment exploring whether locally learning, energy-constrained cells can adapt collectively to environments proposed by Meta AI models **without an AI model directing individual cell decisions**.

## Status

- **Oct 26, 2026:** Public repository initialized for hackathon coordination and planning only.
- **Oct 27–Nov 4, 2026:** Event's stated *building period*. Competition implementation will be committed separately during this window.
- **Submission deadline:** Nov 4, 2026, 9:00 PM Eastern Time.

**No functioning cellular simulator, Meta integration, measurements or demo are claimed in this initial commit.**

## Architecture in brief

1. **Meta world designer:** model proposes typed environment specifications, not cell actions.
2. **World validator/compiler:** turns a validated scenario into local physical stimuli and resource fields.
3. **Cellular system:** each cell has local signals, memory/plasticity and a bounded energy/resource budget; collective behavior emerges from interactions.
4. **Independent evaluator:** reproducible trials, fixed/random/Meta-generated scenario controls, multiple seeds, reported failures and computational-cost proxies.
5. **Demo:** visualization of the world, cells, changed conditions and transparent metrics.

## Scientific and provenance boundary

The long-term **Emergent Cells Lab** research predates this hackathon and is **not** being claimed as new. This project will independently implement a scoped competition prototype; any historical inspiration will be disclosed, with no copied private source code or historical results presented as hackathon output.

Physical substrate independence is a **design hypothesis**, not evidence of an implementation beyond this digital simulator. Simulated resource/energy units and activity counts are **not measured joules**.

## Participation and links

- [Hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)
- [Submission page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit)
- [Meta AI developer documentation](https://ai.meta.com/resources/models-and-libraries/)

## Contributions and rights

Code will be published with explicit provenance and applicable licenses when selected; until a LICENSE is committed, **no open-source license is granted by this repository**. Do not submit secrets, personally identifying data, or rights-restricted assets.

*This public repository is separate from the existing private research repository.*


## Prototype cellular engine v0 (Oct 26 — exploratory, not a competition claim)

A working *headless* simulator has been added on branch `feat/local-cellular-engine`, with independent cell-level energy and local weight updates, finite resource consumption, local neighbor messages, repeatable world perturbations and three benchmark modes. There is **no live Meta Model API integration or visual demo yet**.

The Participant Guide lists Oct 27–Nov 4 as the building period. This Oct 26 engine is explicitly recorded as *pre-period exploratory work*. Do not label it a new work-period contribution without written rules clarification or a distinct eligible implementation.

Run it with Python 3.10+:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m emergent_cells --mode learned --seed 42 --steps 240 --cells 256 --output results/runs/single.json
python -m emergent_cells --compare --seeds 1,2,3 --steps 240 --cells 256 --shock-step 120 --output results/runs/compare.json
```

See [engineering limitations](docs/ENGINE_V0.md), [validation log](docs/ENGINE_TEST_LOG.md), and the [proposed evaluation protocol](docs/EXPERIMENT_PROTOCOL.md). Do not confuse resource/energy in the simulator with measured joules or demonstrated general intelligence.


## Interactive validation milestone (Oct 26)

The real engine now has a local browser laboratory, explicit-command console,
and fresh-process five-control benchmark. This remains exploratory research;
no language, agents, growth or physical energy advantage is demonstrated.

```sh
python -m pip install -e .
python -m emergent_cells.demo.server
```

Open http://127.0.0.1:8765 on the same computer.
See [usage, evaluation and evidence gates](docs/VALIDATION_AND_INTERACTION.md)
and [all-control results](results/audited/2026-10-26-SUMMARY.md).
The simple fixed local rule outperformed the learner in this first evaluation.
This Oct 26 milestone retains the existing event-eligibility caveat.

## Adaptive Worlds designer (Oct 26 prep prototype)

This branch extends the existing interactive lab with a **human challenge -> validated WorldSpec -> reproducible multi-policy evaluation -> live visual simulation** flow.

- Pick `Plantilla local` for a clearly labeled, deterministic local template. It makes **zero** Meta API requests and lets you test the entire workflow offline.
- Pick `Meta Muse Spark` for a real Meta Model API structured-JSON design; this mode **requires** private `MODEL_API_KEY` and the optional Python OpenAI SDK.
- The design is restricted to resource/hazard fields, replenishment and shock timing; it can **not** instruct individual cell actions or invent cooperative physics.
- Controls are random, frozen, a hand-coded greedy local baseline, learned, and learned with messages blocked (development seeds only).
- Once a world is validated, use `Aplicar al mapa` to play it in the real engine.

Detailed [implementation and limits](docs/META_WORLD_DESIGNER.md) and [activation instructions](docs/META_WORLD_SETUP.md). Real Meta requests have **not** been executed in this prototype; no credits or billing availability is assumed. This code was written **Oct 26** and remains pre-period research unless organizers confirm eligibility for an Oct 27 start.

## Unified research lab: cooperative comparisons (October 27 integration)

The new **cooperative gate panel** brings the pre-existing October 26 dual-cue experiments into the existing local browser lab. It runs fixed bounded development seeds, displays all five local-message ablations and the knockout of already-trained policies, and attaches results to the session report. The task is an engineered XOR gate with fixed pair assignments and does **not** run Meta, create a natural-language agent, or demonstrate general intelligence. The Oct 27 integration code is distinct from the Oct 26 scientific prototype; neither is silently represented as original contest-period research. See [integration notes](docs/UNIFIED_LAB_2026-10-27.md).

## Learned one-bit messages — October 27 milestone

A NEW local learning experiment removes the hard-coded XOR *policy feature* from the older dual-cue gate. Each cell independently trains a private stochastic sender encoder (one-bit symbol) and a generic four-slot local receiver table, using its own simulated energy delta. **The world still provides XOR gate physics, a predefined pair, a one-bit channel, and the four-slot receiver representation.**

The 10-seed exploratory run (128 fixed pairs, 8000 training and 500 new-input testing episodes per seed) obtained **98.23%** group-gate success for learned local symbols, versus **49.35%** with reception blocked and **48.36%** with random symbols. Blocking messages in the SAME trained policies gave **25.03%**. No unseen task-function transfer or general intelligence has been demonstrated.

See [methodology, citations and limitations](docs/LEARNED_PROTOCOL_2026-10-27.md), [raw outcomes](results/audited/2026-10-27-learned-messages-10seeds.json), and [new browser workflow](docs/LEARNED_PROTOCOL_UI_2026-10-27.md). This is new October 27 work; Oct 26 background modules retain their original provenance.

## Contextual memory under A -> B -> A reversal (October 27)

A new local reward experiment learns XOR world A, adapts to inverted XNOR world B, and retests A **without retraining**. An externally supplied local season marker selects one of two private decoder banks; the cell never receives its correct gate action. Senders are **frozen after A** to preserve the already learned one-bit language.

Untouched after-pilot seeds 1001..1005: context-memory cells reached **96.40% B** and immediately recovered **96.51% A**; explicit one-bank reset learned **96.40% B** but returned to only **0.19% A**; a saturated single-bank learner failed to learn B. These narrow results are expected under the supplied season marker and deliberately frozen protocol. They **do not** demonstrate autonomous change detection, general intelligence or a capacity/energy-matched improvement.

See [experimental report](docs/CONTEXT_SWITCH_2026-10-27.md), [raw fresh-seed data](results/audited/2026-10-27-context-switch-heldout.json), and [live UI instructions](docs/CONTEXT_SWITCH_UI_2026-10-27.md).

## Unannounced environmental reversal detection (October 27)

A new **local energy-reward surprise detector** can switch among two private learned receiver-memory banks WITHOUT any externally supplied season identifier. Two-cell gates use an independently learned one-bit message; the external environment changes XOR to XNOR and back. The detector's exponential moving average, maturity threshold, two slots and switch cooldown are **explicitly designed** and are not emergent themselves.

On five fresh seeds (2001..2005, 64 paired gates), the cells learned B with **96.29%** success; after a reward-only detector delay averaging **18.61 episodes**, they recovered A to **96.51%**, with **99.69%** of pairs detecting the return. This is recovery in task PERFORMANCE: some weights of old A change during the initial detection delay on B. An explicit seasonal-context oracle got 96.32% B; the model with no detector and no season cue got only 0.18% B.

Reproduce the exact experiment and examine caveats in [reward-only detector study](docs/REWARD_SURPRISE_CHANGE_2026-10-27.md), [raw fresh-seed observations](results/audited/2026-10-27-unannounced-switch-fresh-seeds.json) and [browser panel guide](docs/REWARD_SURPRISE_UI_2026-10-27.md). No Meta Model API calls or measured physical joules are involved.

## Dynamic, on-demand third memory (October 27)

In a new **A XOR → B XNOR → C AND → A → B** experiment, individually rewarded fixed-pair cells learn one-bit messages and use an engineered local reward-surprise mechanism to **probe existing memory banks and allocate a new conditional-response bank only when old policies fail**. The world does not tell the adaptive learner the Boolean rule. The memory-allocation algorithm, noise sensitivity threshold, maximum three slots, fixed pairs and Boolean tasks are author designed.

With **five previously unused seeds**, 32 pairs and 8% missing resource pulses, growth reached **93.7 / 94.3 / 95.2 / 94.1 / 94.5%** success across those stages. The two-bank control reached **24.1%** on C. The rare AND input `11` was handled at **92.2–95.4%**, excluding an always-zero shortcut. A first, faster detector **failed** due to false memory allocation and was preserved in the results.

**Key negative result:** the final detector produces no false new memories in stationary 8% dropout, but mistakes a sudden resource-dropout increase to 20% for a task change, allocating 62–64 needless new memories in each stress seed. Thus robust context inference is **not** solved.

See [the scientific report](docs/DYNAMIC_MEMORY_THIRD_RULE_2026-10-27.md), [complete holdout data](results/audited/2026-10-27-dynamic-third-rule-holdout.json), [noise stress](results/audited/2026-10-27-noise-stress.json), and [interactive demo](docs/DYNAMIC_MEMORY_UI_2026-10-27.md). **No Meta API call, measured joules, agentic AI or general intelligence is implied.**

## Robust local task-change detection, October 27

We tested **four** authored detectors for distinguishing changing Boolean gate rules from simply lower resource-pulse availability: prior relative reward EMA, absolute reward floor, aggregate binomial-confidence guard and a **new private conditional reward-signature detector**. The key discovery: XNOR and AND have 3/4 matching cue cases, so global reward may remain high despite a genuinely changed rule. Tracking each cell's own success in its four cue/message categories fixes that particular blind spot.

Five unused seeds showed **93.1% final-third-rule gate correctness** with conditional detection, 92.7% recovered A and 91.4% recovered B; while stable-task resource withholding at 20% and 35% created **zero false memories** in the new detector versus ~47 of 48 cells on the old detector. **At 45% scarcity, two false allocations still occurred** across five tests. The new method also consumes additional byte-valued history and is NOT a universally reliable, physically energy-efficient context detector.

Read [methods and all results](docs/ROBUST_DRIFT_2026-10-27.md), [raw independent-seed data](results/audited/2026-10-27-conditional-drift-holdout.json) and [local demo instructions](docs/ROBUST_DRIFT_UI_2026-10-27.md). No Meta API requests were made.

### A deliberate adversarial scarcity control (separate study)

If the XOR task is stationary but resource availability disappears selectively for input `11`, **reward-only** detectors still allocate needless banks (44–46 of 48 cells on independent seeds). An **explicit extra local resource-presence sensor** prevented these false allocations (0 of 48) and still learned the third rule (93.0% success, return A 91.3%, return B 92.1%). This is **unequal-information testing**, not a reward-only breakthrough. Physical availability sensing, extra compute/memory cost and transfer to other environmental rules remain unknown. [Full report and raw data](docs/ADVERSARIAL_SCARCITY_LOCAL_SENSOR_2026-10-27.md).

## Locally coordinated counterfactual exploration (October 27)

New experimental milestone: after learning one-bit communication privately, paired artificial cells can **coordinate their own alternate-action probes** when a previously successful world stops rewarding them. The benchmark distinguishes a selectively **flipped XOR gate mapping** from selectively **absent resources** *without an ideal resource-availability sensor*. Each cell updates hand-designed Bayesian-style beliefs from its own gate reward; sender/receiver communication weights are frozen during diagnosis. This is engineered active inference, **not learned curiosity** or general intelligence.

On five independent seeds (48 pairs), information-guided probes reached **98.1%** critical-case success when a mapping changed, versus **0.4%** with no probes. Under unchanged mapping but missing resources for that case, they retained **97.7%** correct actions with only **0.4 false correction bits per 96 cells** on average. Random exploration reached 95.7% changed-case success with *fewer messages and lower simulated cost* than information-gain exploration. Both performance and costs are reported.

Read [scientific protocol and results](docs/ACTIVE_CAUSAL_EXPLORATION_2026-10-27.md), [full raw results](results/audited/2026-10-27-active-causal-probing-fresh-seeds.json), and [local browser instructions](docs/ACTIVE_CAUSAL_EXPLORATION_UI_2026-10-27.md). No Meta API call, measured physical joules or general task transfer is claimed.
