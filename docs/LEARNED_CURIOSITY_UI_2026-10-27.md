# Learned local curiosity — interactive demo

The interactive HTML laboratory has a new panel **Curiosidad aprendida — ¿cuándo investigar?**. It is **a separate gate-world experiment**, not the mobile foraging-cell canvas.

The demo:
- Trains private, 1-bit cue-communication policies.
- Teaches each cell a private 6-state × 2-choice local Q-table via actual reward plus an authored intrinsic posterior-learning-progress reward. A decision to request a coordinated probe is read from **learned Q**, not a fixed expected-information-gain trigger.
- Compares learned Q, untrained Q, random coordinated exploration and authored Bayesian information gain. Reports changed-rule success, scarcity correctness, request bits and per-seed raw details.
- Uses **two DEVELOPMENT seeds**, 16 pairs, 2000 message-policy training episodes, 130 meta episodes of 55 interactions, and 260 test steps. This is deliberately smaller than the separately audited new-seed experiment.
- Stores report in GET `/api/learned-curiosity` and `/api/report`. POST `/api/learned-curiosity/run` accepts ONLY an empty JSON object. Cross-origin requests are rejected.

From the local repository:

```powershell
python -m pip install -e .
python -m emergent_cells.demo.server --port 18772
```

On **that same laptop**, navigate to `http://127.0.0.1:18772`. The unauthenticated server must remain bound to localhost, not be exposed on the public Internet.

**Do not claim any use of Meta Model API, general-purpose tool agents, substrate-agnostic physical computing or real joule measurement.** For the full scientific comparison, see [the study report](LEARNED_CURIOSITY_2026-10-27.md).
