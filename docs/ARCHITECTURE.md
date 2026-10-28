# Scientific Architecture — Python Core

**Purpose:** test falsifiable hypotheses about decentralized learning, not claim that general intelligence has been demonstrated.

- `cells/population.py`: local cell states and update rules; no learned global policy.
- `world/environment.py` and `simulation.py`: simulated resources, environmental dynamics, and baselines. Target rules belong to the environment, never the learner.
- `experiments/`: private signal encoding, learned binary messages, contextual memory, local change detection, conditional rewards, causal probing, and learned curiosity. Experiments own their configurations and assumptions.
- `evaluation/benchmark.py`: paired controls and evaluation metrics.
- `run.py` and `cli.py`: Python-only experiment entrypoints.
- `tests/`: reproducibility, local independence, absence of privileged target labels, bounded behavior, and regression checks.
- `results/`: negative exploratory findings and audited runs. Never rewrite original experimental records.

NumPy vectorization and simulation orchestration are features of the **environment implementation**; they do not establish physically decentralized computing. Pairwise connectivity, one-bit messages, detection rules, and memory capacity are partly engineered assumptions. Simulated energy is not measured in joules.

No visualization frontend or HTTP endpoints are required: they are not part of the scientific evidence.

For the current roadmap, see [Research Goals and Workflow](RESEARCH_GOAL_AND_WORKFLOW_2026-10-28.md). Follow the mandatory language policy in [AGENTS.md](../AGENTS.md).
