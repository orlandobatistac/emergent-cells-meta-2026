# Emergent Cells Lab — Python Scientific Core

**An experimental research laboratory for decentralized learning with local rewards, learned one-bit messages, context-dependent memory, and simulated resource constraints.**

This repository contains reproducible Python/NumPy experiments, tests, numerical results, and research notes. It does **not** contain a browser UI or an API server. Most historical cell experiments use fixed pairs, digital Boolean environments, authored schedules, and synthetic reward/energy accounting. The newer multi-step world prototype adds tabular local transition memories and a programmed central planner.

> **Scientific boundary:** High accuracy on these deliberately small simulations does not establish AGI, physical energy efficiency, autonomous growth of biological cells, hardware independence, or substrate-agnostic intelligence. The simulated "energy" variable is not measured in joules.

## Run locally

```powershell
python -m pip install -e .
python run.py list
python run.py simulate --compare --seeds 1,2,3 --steps 240 --cells 256 --output results/runs/simulation.json
python run.py dynamic-memory --seeds 1001,1002,1003,1004,1005 --pairs 32 --phase-steps 4500 --dropout 0.08 --output results/runs/third-rule.json
python run.py sequential-transfer --seeds 2001,2002,2003,2004,2005 --pairs 24 --pre-steps 3800 --probe-steps 120 --adapt-steps 1100 --output results/runs/transfer.json
python run.py multistep-world --seeds 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010 --train-steps 7000 --missions 30 --output results/runs/multistep.json
python -m unittest discover -s tests -q
```

Individual experiment runners expose `--help` and optional JSON `--output` flags.

## Current research outcomes

- **Learned pairwise communication:** approximately 98% task accuracy on controlled XOR environments; no informative-message baselines are near chance.
- **Locally triggered decoder memory expansion:** under the original five-seed protocol, approximately 93–95% late-phase accuracy across XOR, XNOR, AND and subsequent re-exposure. The same detector can falsely allocate memory after a large reward-availability drop.
- **Locally trained exploratory actions:** realized information-progress reward improves several change-detection outcomes but increases intervention costs in some stable/noisy settings.
- **Sequential transfer (October 28, 2026):** locally reward-selected pieces of old response tables improve first-200-step adaptation to unseen OR/NAND Boolean rules compared with cold-start and equal-interaction-budget controls. Mean held-out post-training accuracy was **92.04% for OR** and **91.73% for NAND** across five seeds. Source policies were frozen by experimental design; this is same-family transfer, not autonomous general learning.
- **Three-waypoint world-model transfer (October 28, 2026):** a separate prototype of locally stored spatial transitions with an explicit BFS planner completed **300/300 unseen three-target missions** across ten holdout seeds. A cold version with the same planner also completed all 300, but required **19.59 additional primitive steps per mission** on average. Online correction handled a specific unannounced door closure in **10/10** tests; frozen memory failed **10/10**. Structured coordinates and central planning are supplied by the experimenter, not learned or decentralized.

Read [the multi-step world-model report](docs/MULTISTEP_WORLD_TRANSFER_2026-10-28.md), [the sequential transfer report](docs/SEQUENTIAL_TRANSFER_2026-10-28.md), [the original dynamic-memory report](docs/DYNAMIC_MEMORY_THIRD_RULE_2026-10-27.md), [learned curiosity notes](docs/LEARNED_CURIOSITY_2026-10-27.md), and [the experimental protocol](docs/EXPERIMENT_PROTOCOL.md). Raw machine-readable study data are in `results/audited/`; developmental and negative results are retained in `results/exploratory/`.

## Repository structure

- `run.py` and `src/emergent_cells/cli.py`: Python experiment dispatcher.
- `src/emergent_cells/cells/`: private local learners.
- `src/emergent_cells/experiments/`: scientific benchmarks and ablations.
- `src/emergent_cells/evaluation/`: metrics.
- `src/emergent_cells/world/`: simulated resource worlds.
- `tests/`: deterministic regression and experimental integrity tests.
- `docs/`: hypotheses, protocols, limits and study reports.
- `results/`: immutable historical observations and new recorded runs.

## Research and contribution policy

**All new code, comments, docstrings, README updates, research reports, issues, pull requests and developer-facing documentation must be written in clear, modern professional English, regardless of the human discussion language or AI agent/driver language.** Human conversations may occur in Spanish or any other language. Agents should translate their work artifacts into English before submission.

Preserve experiment provenance: preregister test conditions and holdout seeds when practicable, report negative outcomes and limitations, compare against meaningful controls with matched observation budgets, and keep numerical datasets reproducible. Do not silently rewrite historical results. One legacy Spanish-language research synthesis remains in `docs/REPORTE_CIENTIFICO_2026-10-27.md` solely for historical provenance; new material must be in English.

The historical hackathon UI is recoverable from previous Git history, but is not part of the current Python-only scientific core. No Meta Model API calls were required for these numerical experiments.
