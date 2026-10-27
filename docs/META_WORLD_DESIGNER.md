# Adaptive Worlds — Meta integration prototype, October 26, 2026

**Provenance and eligibility:** this prototype was written October 26 (kickoff), while the participant guide states the building period is October 27–November 4. This code is not represented as eligible hackathon new work without organizer clarification. The private research and its older results are independent.

## Functional pipeline

A person enters a natural-language challenge description in the existing local visual lab.

1. Explicitly choose the origin: deterministic offline template or Meta Model API.
2. The designer returns a strictly bounded world proposal and source identity. Offline templates are NOT Meta requests.
3. A local type/range validator rejects extra keys, invalid numbers, NaN/Inf, booleans passed as integers, unsupported fields or unsafe world parameters.
4. The world is compiled to the existing WorldSpec type, with default 64×64 geometry.
5. Five independent policies are evaluated against *the same* seeded worlds: random, frozen, greedy chemotaxis, local learning, and local learning without neighbor messages. Three development seeds 101–103; 90 steps, 64 cells, by default.
6. The approved world can be applied to the **real live visualization**, resetting the session with the same configured seed and cell count. The user can then advance and intervene manually.
7. The report keeps the design source, model identifier when used, individual outcomes, limitations and per-seed results. These exploratory data are not scientific confirmatory evidence.

## Meta Model API (real integration, not yet live-tested)

Meta's official developer API supports an OpenAI-compatible client and chat-completions JSON schema output:
- Base URL: https://api.meta.ai/v1
- Model: muse-spark-1.3
- Authentication environment variable: MODEL_API_KEY
- Request: chat.completions.create with response_format type json_schema and strict true
- Schema: name, rationale and exactly six numeric parameters; no generated executable code

References:
- https://dev.meta.ai/docs/sdks
- https://dev.meta.ai/docs/structured-output

To install the optional client:
    python -m pip install -e ".[meta]"

Configure the API key privately in the OS environment; **never paste it into the browser, issue, chat or repository**, and never save it in a .env tracked by Git. The server listens only on 127.0.0.1. When the user selects Meta explicitly, a single request is sent; no silent fallback occurs on error.

**Current local verification:** the OpenAI SDK is installed, but MODEL_API_KEY was not configured in the process inspected. No real paid request was attempted. Integration with a fake structured-response client and the full offline path passed tests. Full live Meta authorization, credits, model response quality and billing remain unverified.

## How to run

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python -m emergent_cells.demo.server

Open http://127.0.0.1:8765 on the same laptop. Use the "Diseñar desafíos" panel. Select "Plantilla local (sin Meta API)" until an API key and any credit/billing limits have been confirmed.

To verify UI event handlers through real localhost HTTP, with server running:
    node tests/browser_controls.cjs

This is a synthetic DOM/canvas integration check, **not browser pixel/visual QA**.

## Boundaries / next experiment

- The Meta model only chooses *environment parameters*. It never chooses cell actions or modifies cell policy weights. The validation protocol is identical for model and local-template sources.
- Resource scarcity and hazards do not automatically create a collective-information task. A request "require cooperation" is disclosed as not enforceable by the current foraging world.
- The separate exploratory dual-cue research experiment has true missing-information pairs but engineered XOR-capable features. It has NOT been merged into the interactive product.
- A Meta model could design task families, intervention timing and counterfactuals only after those mechanisms are implemented and validated; model language does not itself expand the physics engine.
- Measures are simulated energy and software steps, **not physical joules**. Greedy may outperform learned policy. This prototype does not demonstrate scalable, general or emergent intelligence.
- Fixed development seeds do not establish out-of-sample generalization. Pre-register evaluation worlds, compare with procedural random-world and human-authored curricula, and keep holdout outcomes out of model prompts.
- This server is a *single-user local research demo* and not safe to expose to the Internet without authentication, rate limiting, isolation and multitenancy work.

## Acceptance evidence for Oct 26 research-only milestone

- 30 Python unit/integration tests passed on Windows Python 3.12.10.
- Local HTML/JS actions tested through actual localhost HTTP: previous controls plus design, evaluate and apply actions.
- No API key logged or committed and no live Meta credits spent.
- Source branch remains separate from the private research, dual-cue experiment branch and default main.
