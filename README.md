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
