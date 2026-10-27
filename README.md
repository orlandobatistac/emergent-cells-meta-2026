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
