# Interactive research milestone — October 26, 2026

This work extends `8568ad0aa4ad52ef06e86b8346a39eae3f87e505` from
`feat/local-cellular-engine`. It was developed on October 26. It is an
exploratory research milestone, not a claim of eligibility for the event's
October 27–November 4 building window. Preserve the existing provenance notes.
No historical private research implementation or results were imported.

## Use the actual engine

From the repository root on Python 3.10+:

```sh
python -m pip install -e .
python -m emergent_cells.demo.server
```

Open http://127.0.0.1:8765 in the browser **on the same computer**.
The server runs locally, not at a public URL. No Meta API key is required.
Stop with Ctrl+C. One process serves one shared session; this is a research
prototype, not a multi-user application.

The browser displays the real resource/hazard fields and population. Start or
pause, advance twenty steps, displace resources, remove 20% of surviving cells,
freeze learning, switch to a simple local rule, reset to the original seed,
and download a metrics/event report. Mode switches preserve the current weights;
reset returns to initialization. Removed cells do not regrow. Downloaded reports
are records, not restorable checkpoints. Closing the server loses the session.

A text interface is also available:

```sh
python -m emergent_cells --interactive
```

Commands: `estado`, `avanzar 20`, `choque`, `dano 0.2`, `modo frozen`,
`reiniciar`, `exportar results/runs/session.json`, `salir`.
This interface implements explicit commands, not natural-language reasoning.

## First matched evaluation

```sh
python -m unittest discover -s tests -v
python -m emergent_cells.evaluation.benchmark --output results/runs/benchmark.json
```

The default experiment uses ten development seeds (101–110), 128 cells, 160
steps, and a resource-field displacement at step 80. Each seed runs all five
variants in fresh child processes, with deterministic rotated order and one
BLAS thread. All outcomes are retained, including the stronger control's wins.

- Random and frozen controls preserve the prior comparison.
- `greedy` uses a fixed local food-plus-danger gradient, with random tie breaking;
  it does not learn or use social steering. This is a reasonable simple control,
  not an exhaustive search for the best classical policy.
- `learned_no_communication` ablates communication at the same initial positions,
  policy initialization, environment and RNG seed. Trajectories may diverge.
- Primary outcome: survival fraction at the end of the requested horizon.
- Also report alive-cell-steps, harvest, timing and whole-worker peak RSS.

Wall/CPU measurements cover simulation execution, not interpreter startup.
Peak RSS includes Python, NumPy imports and native allocations, in a fresh
worker (unavailable on Windows). Timing is hardware/load dependent. Early death
reduces actual work, so equal requested steps are not equal computational work.
These values are not measured electricity consumption or evidence of efficiency
at matched task quality. The old neighbor search is still quadratic in cell count.

See `results/audited/2026-10-26-SUMMARY.md` and the associated raw JSON.
These exploratory outcomes are now development data; use new, frozen seeds and
world families for a confirmatory evaluation. A resource shift is not a retention
or cell-damage recovery benchmark.

## Evidence gates toward a product

| Claim | Current evidence | Gate still needed |
|---|---|---|
| Local learning | Own-cell update, bounded sensors, remote-cell perturbation test | Broader audit and locality tests under edge cases |
| Collective benefit | Communication implemented; first ablation has no clear gain | Cooperative task outperforming no-communication control |
| Better adaptation | Beats random/frozen here, loses to greedy | Held-out tasks where learning beats tuned simple policies |
| Growth adds capacity | Not implemented in this engine | Constrained growth and capacity/retention controls |
| Retention | Not measured in this engine | Sequential A→B learning, test A and B, matched replay budgets |
| Damage recovery | Manual removal is available | Automated matched lesions, recovery metric and controls |
| Low resource use | CPU, wall-time and RSS instrumentation | Quality-matched alternatives, scaling curves, physical energy measurements |
| General AI with agents | No language model, planner or agent runtime | Real tool-use tasks with verifiable outcomes and budgets |

**Next scientific priority:** identify a task where temporal information and
learning are necessary; current immediately visible resource gradients favor a
fixed rule. Establish that advantage before attributing general intelligence
or product benefits to the cellular mechanism. The original ambition remains;
this milestone exposes the engine and its limitations for direct experimentation.

## Verification of this milestone

- 22 Python behavioral/integration tests passed, including actual HTTP actions,
  invalid inputs, local-only request boundaries, deterministic session/batch
  agreement, frozen weights and no resurrection after removal.
- `node tests/browser_controls.cjs` passed against a real running Python server:
  advance, damage, displacement, mode, play/pause, report download and reset.
  It uses DOM/canvas stubs and is **not** a browser-rendering test.
- A wheel was successfully built without dependency downloads and checked for
  inclusion of the HTML interface and server module.
- Full browser/visual QA remains pending: Chromium was unavailable and its
  attempted download failed. No desktop/mobile screenshot is claimed.
