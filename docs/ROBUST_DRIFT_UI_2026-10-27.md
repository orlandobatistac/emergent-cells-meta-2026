# Interactive test for reward scarcity versus a changed task

Open `http://127.0.0.1:18770` from the connected Windows laptop while the corresponding local development server is running. The default server can also be started with:

```bash
python -m pip install -e .
python -m emergent_cells.demo.server
```

Open its terminal-specified `http://127.0.0.1:8765` URL on the laptop.

The **¿Nueva regla o simplemente menos recursos?** panel runs a bounded **pilot**, not the full independent-seed audit: seed 11; 16 pairs; 3000 episodes A/B/C; 900 episodes per revisit; and stationary XOR resource-dropout cases 8%, 20%, 35%, 850 stress steps each. The panel compares the original `relative_ema` with `conditional_signature`. It displays learning on A/B/C, return to A/B, and the average number of unnecessary bank creations under stationary noise stress.

Endpoints: `GET /api/robust-drift` (last report), `POST /api/robust-drift/run` (strict empty JSON), `GET /api/report` (provenance and full audit output). Browser buttons verified via real localhost HTTP in CI.

**The two experimental tasks remain separate**: the interactive foraging-cells canvas is not the same as the pair-gate experiment. The panel does not invoke Meta or consume Model API credits; its thresholds, gate rules and 3-bank memory cap are engineered.

Scientific method, holdout measurements, all four candidate strategies and unsuccessful results: [full report](ROBUST_DRIFT_2026-10-27.md).
