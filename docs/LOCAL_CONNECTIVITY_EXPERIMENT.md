# Exploratory local connectivity / structural-plasticity experiment

**Created: 2026-10-26 (pre-building-period research).**
This branch is NOT claimed as qualifying Meta Hackathon development or final scientific evidence. The participant guide gives Oct 27–Nov 4 as its building period; get organizer clarification before reusing any pre-period implementation.

## Why run this?

The previous engine demonstrated individual local feedback. It did not establish whether communication or structural changes among cells contribute to population behavior.

The experiment tests a narrower, falsifiable question: under identical worlds and initial local policies, does limited local communication improve survival compared with no intercellular messages, and does locally adjusted link trust improve results compared with fixed link trust?

## Four conditions

1. **none**: learning cells receive no neighbor messages (radius 0).
2. **broadcast**: existing local-radius communication, all peers within radius (uncapped, exploratory reference).
3. **fixed**: each cell maintains up to 4 directed local neighbor contacts, with unchanged trust strength (1.0). Contacts naturally expire as cells move out of range and new neighbors enter. *This is fixed trust, not fixed physical topology*.
4. **adaptive**: identical 4-slot bounded contacts, but each cell locally adjusts link trust based on its **own** energy-change advantage, direction of movement relative to the neighbor and that neighbor's previous food message. Very low-trust directed contacts are pruned, with a local cooldown before reconnecting.

Each individual cell stores its link IDs and strengths in its own row of a NumPy array. This is a **digital simulation of local rules**; the CPU still computes a distance matrix (quadratic global bookkeeping). This is *not* evidence of energy-efficient hardware or unbounded modular growth.

## Matched controls

All conditions use the same seed, world parameters, starting coordinates, energy, per-cell initial weights, action-selection random generator, baseline and time horizon. The capped **fixed vs adaptive** pairing is the primary plasticity comparison because it has identical slot capacity. The broadcast condition is an unbounded-contact reference, not a matched-density control.

## Tests and caveats

- No model/API involvement: Meta integration remains a separate stage.
- Locality: messages only from live cells within configured radius.
- No global rewards or optimizer: local per-cell energy change modulates only the cell's own link weights.
- Link strength != physical synapse or material-constrained growth.
- The old engine has no real topological rewiring or cell birth; dynamic contact slots are provisional structural plasticity.
- Absence of better survival for the adaptive condition must be reported, not hidden.
- Simulated metabolic units and message counts **are not measured joules**.
- The perturbation at step 120 shifts the resource field, but comparing final survival alone does not establish *recovery* from the shock. Separate before/after time series and further controls are necessary.

## Run

Install once at repo root:

\`\`\`sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -m emergent_cells.experiments.connectivity --seeds 1,2,3 --steps 240 --shock-step 120 --output results/runs/connectivity.json
\`\`\`

Result rows contain per-seed condition outcomes. Git ignores generated output; intentionally publish only audited reports with clear date and interpretation.
