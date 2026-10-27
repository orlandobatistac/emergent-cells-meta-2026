# Unified Interactive Research Lab — October 27, 2026

This milestone integrates the **existing** October 26 experimental branches without changing their original provenance. The user-facing cooperation panel and HTTP controls documented below were implemented October 27.

## Features

- Browser-based 64x64 resource world with existing interactive cells and controls.
- Validated structured world proposals, choosing explicitly between a deterministic **local template** and Meta Model API. Meta is not called when local is selected.
- Seeded independent evaluation against random, frozen, greedy, learned and learned-without-messages policies for the resource world.
- New **cooperative two-cue gate** panel (separate experimental world) using the imported October 26 research code: 64 pairs (128 cells), three development seeds 201–203, 600 train episodes, 120 evaluation episodes; all five conditions and the same-trained-policy message-knockout replay. It does **not** change the moving cells in the foraging canvas, because the gate is a different task with fixed positions and dynamics.
- The UI labels separate world types and their limitations. No automatic conversion of a natural-language Meta world into an XOR cooperation task is claimed.
- Entire cooperation report appears in GET /api/report and GET /api/cooperation, allowing independent audit.

## Provenance

**Before October 27:** local foraging cells, research connectivity experiments, the engineered dual-cue/XOR simulator, dashboard v0, structured-world Meta adapter. All retained as prior exploratory work.
**October 27 integration:** assembled historical research branches with their original Git history; added cooperation HTTP endpoint, UI panel, browser-control coverage and integrated report. Do not claim the prior code was written October 27 merely because a merge occurred.

## Reproduce

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python -m emergent_cells.demo.server

Open http://127.0.0.1:8765 from the host machine and use "Cooperación celular".
To verify browser event handlers over real localhost HTTP, run `node tests/browser_controls.cjs` while the server is running.

## Authentication and reality checks

- The Meta adapter uses the officially documented OpenAI-compatible endpoint, model muse-spark-1.3, strict JSON-schema output and MODEL_API_KEY.
- **No real Meta API request or billing/credit status has been verified.** The default local-template and cooperative tasks do not invoke any model or incur model-API expense.
- Both services are local research prototypes. Never expose the development server directly on a public network.
- The task's XOR-capable input feature is engineered; cells learn only their own parameters and do not invent language, connections, abstract symbols or software-agent tools.
- Resource/energy units are simulator proxies, not physical joules.
- Confirm official event eligibility guidance before treating pre-period work as a contest submission.
