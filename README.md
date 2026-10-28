# Emergent Cells Lab — Python Research Core

An experimental, reproducible research lab for decentralized cellular learning, local communication, contextual memory, and **simulated** energy economics. The repository contains Python algorithms, essential tests, numerical outputs, and scientific reports. No frontend, HTTP, JavaScript, or HTML.

**Scientific boundaries:** results demonstrate narrow behaviors in explicitly designed mathematical environments. They are **not** evidence of AGI, real-world electrical efficiency, substrate independence, or physically scalable cellular networks. Simulated energy is a reward proxy, not joules.

## Current Research Priority

**Close the functional intelligence loop before optimizing compute or energy.** Build and evaluate an integrated process of observation, local communication, learning, distributed memory, action, and environmental feedback. The immediate next experiment is **Closed-Loop Cellular Learning v1**, evaluated against isolated cells and a compact centralized baseline. Agent orchestration and real resource savings remain future research questions.

See [Research Goals and Workflow](docs/RESEARCH_GOAL_AND_WORKFLOW_2026-10-28.md) and [Scientific Architecture](docs/ARCHITECTURE.md).

## Language Standard — Mandatory for All Contributors and AI Agents

**All content committed to this repository MUST be in modern, precise, professional US English, regardless of the language used by the human operator, AI driver, assistant, coding agent, or research conversation.** This includes code comments, docstrings, identifiers, tests, CLI messages, documentation, experimental reports, commit messages, pull requests, and Issues.

Discussion with the project owner may happen in Spanish; **repository artifacts must always be English**. Keep a clear AI research/startup engineering voice, without inflated claims. Preserve historical evidence and compatibility when translating. Every contributing agent must follow [AGENTS.md](AGENTS.md), which is the authoritative policy.

## Run Locally (Windows)

Install the package: `python -m pip install -e .`

- List experiments: `python run.py list`
- Base simulation: `python run.py simulate --compare --seeds 1,2,3 --steps 240 --cells 256 --output results/runs/simulation.json`
- Dynamic memory: `python run.py dynamic-memory --seeds 11,12 --pairs 32 --phase-steps 4500 --dropout 0.08 --output results/runs/third-rule.json`
- Learned curiosity: `python run.py learned-curiosity --help`
- Three-action policies: `python run.py three-action --help`
- Python tests: `python -m unittest discover -s tests -q`

Where supported, `--output` writes machine-readable JSON for analysis without a browser or visualization layer.

## Repository Layout

- `run.py`: single Python entrypoint.
- `src/emergent_cells/cells/`: cell-local state and learning mechanisms.
- `src/emergent_cells/world/`, `simulation.py`: environment mechanics and reference policies.
- `src/emergent_cells/experiments/`: controlled hypotheses, algorithms, and evaluations.
- `src/emergent_cells/evaluation/`: metrics and benchmarks.
- `tests/`: essential scientific and reproducibility checks.
- `results/audited/`: independent-seed evaluation records.
- `results/exploratory/`: pilots and negative findings.
- `docs/`: methods, architecture, limitations, and historical reports.

## Research History and Evidence

1. **Local connectivity:** ablations, including negative findings.
2. **Learned binary communication:** approximately 98% task success in a designed XOR problem with fixed pairs; controls without informative messages around 50%.
3. **Context switching:** A/B memory and local reward-surprise detection; recovery depends partly on engineered detection.
4. **Dynamic memory:** five-seed sequence XOR (93.7%) → XNOR (94.3%) → AND (95.2%) → XOR recovery (94.1%) → XNOR recovery (94.5%) under 8% random resource loss. Two-memory control achieved 24.1% on AND.
5. **Known failure:** raising resource loss from 8% to 20% without a rule change triggered false memory creation.
6. **Curiosity and task transfer:** held-out combinations within the same XOR/flip/scarcity family, not genuinely new mathematical tasks. On seeds 3001–3005, an authored two-failed-probe stopping policy improved scarcity-01 accuracy from 86.3% to 96.0% and reduced message requests from 405.4 to 98.2 per seed; global-change-with-drought accuracy declined from 65.4% to 64.1%. Manual controls remained stronger in multiple settings.
7. **Learned EXPLOIT / PROBE / REST:** previously audited five-seed experiments showed improved symbolic net outcomes under extreme scarcity, but reduced gate accuracy and reward in usable resource conditions. Rest is a genuine no-activity action; the decision space and costs are engineered.
8. **EMA development and independent-seed check (October 28, 2026):** held-out seeds 41–43 supported the same tradeoff. Three-action EMA vs two-action EMA net symbolic reward: severe drought -0.061 vs -0.088; global-change blackout -0.072 vs -0.114; changed task 0.476 vs 0.523. EMA did not consistently outperform the version without EMA.

Evidence and reproduction details: `results/three_action_ema_report_2026-10-28.md`, `docs/TRANSFERENCIA_CURIOSIDAD_2026-10-27.md`, `docs/TRES_ACCIONES_ENERGIA_2026-10-27.md`, and experiment JSON files. Some legacy research reports are still in Spanish and are queued for carefully audited English migration; **the language policy applies immediately to all new changes**.

## Repository Policy

Prioritize Python/NumPy algorithms and measured results over UI development. Preserve negative findings and immutable experimental outputs. Keep controls, seeds, and measurement definitions comparable. Use feature branches and reviewed PRs; do not merge into `main` automatically. Legacy hackathon UI artifacts remain recoverable from Git history but are not maintained in the science core.
