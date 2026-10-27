# Prototype validation log (non-submission evidence)

**Date:** 2026-10-26, before the explicit Oct 27–Nov 4 building-period window in the participant guide.

This file records engineering validation only, **not official qualifying hackathon experiment results**.

- Platform: Windows + Python 3.12.10 + NumPy 1.26.4, local connected machine.
- Basic run: 11 Python unittest cases passed (determinism, bounds, conservation, locality, learning update, frozen control, comparison shape).
- Exploratory sanity check: 256 cells, 240 steps, seeds 1–3, intervention at step 120, 3 modes (random, frozen, learned).
- Preliminary survival means: learned 0.951; random 0.784; frozen 0.779.
- These unregistered three-seed numbers are *development diagnostics*, not statistically rigorous evidence, not proof of collective intelligence, not measured energy in joules, not Meta-enabled results and **not a hackathon claim**.
- Follow-up: run more seeds, ensure identical environments and report full per-seed outcomes, test a stronger classical gradient-following baseline, communication ablation and out-of-distribution shocks. Check for algorithmic mistakes and data leakage.

Raw runs are excluded from the repository by default. Public reproducible results, if eligible, must come from explicitly dated event-period source/experiment commands.


## Broader exploratory run (10 seeds)

Re-ran with 10 seeds (1–10), 256 cells, 240 steps, one spatial resource shock at step 120 in each mode, on the same Windows/Python environment.

| Mode | Mean survival fraction | Population standard deviation | Mean total remaining cell energy |
|---|---:|---:|---:|
| random | 0.7512 | 0.0860 | 638.1 |
| frozen | 0.7535 | 0.0795 | 644.3 |
| learned | 0.9230 | 0.0563 | 926.2 |

Command: `python -m emergent_cells --compare --seeds 1,2,3,4,5,6,7,8,9,10 --steps 240 --cells 256 --shock-step 120 --output results/runs/compare-10seeds-provisional.json`

These are exploratory *in-sample* sanity checks with an intentionally weak random/frozen comparator, no held-out curriculum or adjusted classical chemotaxis baseline. They are **not eligibility-confirmed hackathon results**. Do not interpret these means as evidence of an emergent cognitive system.
