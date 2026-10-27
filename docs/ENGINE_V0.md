# Cellular engine v0 — exploratory prototype

**Date of code creation: Oct 26, 2026 (planning/kickoff day).**

> **Important eligibility caveat:** the Cerebral Valley participant guide identifies Oct 27–Nov 4 as the *building period*, while event kickoff was Oct 26. This prototype is explicitly dated Oct 26 and must **not** be presented as qualifying event-period work unless organizer rules confirm eligibility. A subsequent event-eligible implementation must be clearly delineated; a new commit date alone would not make old code newly developed.

This initial implementation is intended to make the physics/local learning contract testable before proceeding to Meta integration. It does **not** use any private research source code or historical results.

## What actually exists

- **256 independently located cells** by default, each with private position, metabolic energy, learned policy weights, homeostatic baseline and last harvested signal.
- A finite periodic 2D world with Gaussian resource fields, shared tile depletion, local replenishment, hazards, and an optional mid-run spatial resource shock.
- Four local directional probes of resource and hazard at current/adjacent tiles.
- Communication only with currently nearby surviving cells within a radius, using lagged local harvested-resource signals.
- Independent locally reward-modulated stochastic weight updates using *each cell's* own energy change, previous local baseline and chosen observation/action. There is no global gradient or optimizer adjusting cell policies.
- Modes: `random`, `frozen`, `learned`, matched environments, seeds and steps.
- Deterministic run metadata, JSON output, per-step diagnostic snapshots and simple multi-seed control comparisons.
- Tests for resource conservation, sensory locality, communication radius, frozen policy, bounded state and deterministic replay.

## Important boundaries

- This is a **population of simple foraging units**, not yet evidence of emergent general intelligence.
- Local message passing is implemented, but positive contribution of communication to a collective task is **not yet demonstrated**. A communication-on/off ablation is needed.
- Cell-specific energy and reward are homeostatic simulation feedback, not an independent measured physical property.
- World updates, collision/resource allocation and final metrics are performed by a conventional CPU for simulation bookkeeping; local policies never read aggregate performance.
- No new cells are produced in v0; autonomous growth and lineage integration remain preserved research goals rather than finished features.
- Gaussian resource patches and toroidal wrapping are modeling conveniences, not arbitrary-substrate physical proof.
- **No Meta Model API call** has been implemented in v0; future work will use an explicit validated WorldSpec boundary.
- Do not claim that a favorable result proves the system "thinks." Report all control comparisons including failures.

## Installation / execution

From repo root:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
python -m emergent_cells --mode learned --seed 42 --steps 240 --cells 256 --output results/runs/seed42.json
python -m emergent_cells --compare --seeds 1,2,3 --steps 240 --cells 256 --shock-step 120 --output results/runs/compare.json
```

`results/runs/` is excluded by .gitignore. Only audited, deliberately selected, dated reports should enter the public repository.

## Next scientific checks

1. Ablate neighbor communication at fixed seed/difficulty.
2. Evaluate out-of-distribution world radii/patch densities.
3. Distinguish learning gains from local gradients already visible to the sensor.
4. Add truly local structural plasticity and material-constrained budding with explicit accounting.
5. Formalize and test a WorldSpec validator for scientifically controlled environments.
6. Compare against a reasonable classical chemotaxis baseline (not solely random motion).
