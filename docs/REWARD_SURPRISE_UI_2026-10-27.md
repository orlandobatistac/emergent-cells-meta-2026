# Interactive reward-only change-detector panel

The browser lab has a distinct panel **Detectar un cambio sin aviso A -> B -> A**. It invokes the separate resource-gate experiment through a strict server endpoint. These are not the roaming foraging cells in the canvas.

- POST /api/reward-surprise/run accepts ONLY `{}`; fixed two pilot seeds, 48 pairs, A 4200 train steps, B 3200, 150 test steps.
- GET /api/reward-surprise gives most recent stored experiment.
- GET /api/report includes `reward_surprise_evaluation` to reproduce and audit the comparative results.
- Comparison shows independently inferred context from local energy reward, provided season cue oracle, and blind baseline without detector. It also shows switch latency, detection fraction and the early loss during change.
- The gate XOR/XNOR rule, two banks and threshold are designed; the system does not independently invent them.
- Offline only: no Meta Model API request or credits used. The localhost server should not be exposed publicly.

Run `python -m emergent_cells.demo.server` after `python -m pip install -e .`, then visit `http://127.0.0.1:8765` on the laptop.

This was developed October 27, 2026, separately from preparatory October 26 research.
