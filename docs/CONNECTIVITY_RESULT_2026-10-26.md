# Connectivity ablation — exploratory report, October 26, 2026

**Preparatory research outside the explicitly stated Oct 27–Nov 4 hackathon building period. Not an eligible result unless organizers clarify.**

## Hypothesis and method

Local neighbor communication and per-cell changes to directed contacts might improve survival over independent learning. This falsifiable test was allowed to return a negative result.

- Original synthetic world: 64 × 64 grid, 256 cells with per-cell action policy learning.
- Seeds: 1, 2, 3, 4, 5; 180 steps in each condition; resource map shifts after step 90.
- Matched world and initial cell locations, energy, weights and random streams for each seed.
- No communication; local broadcast within radius 5; at most four contacts with fixed trust; same four slots with locally adaptive trust, pruning and cooldown.
- All conditions update each cell's own action weights. Only the neighbor channel changes.

## Results (exploratory)

| Condition | Mean survival | Mean remaining simulated energy | Mean neighbor messages | Mean local trust prunes |
|---|---:|---:|---:|---:|
| none | 94.61% | 947.57 | 0 | 0 |
| broadcast | 93.75% | 936.66 | 322,575 | 0 |
| fixed | 94.53% | 947.01 | 160,777 | 0 |
| adaptive | 94.45% | 940.52 | 160,977 | 189.4 |

| Seed | none | broadcast | fixed | adaptive |
|---|---:|---:|---:|---:|
| 1 | 0.9570 | 0.9336 | 0.9531 | 0.9375 |
| 2 | 0.9805 | 0.9688 | 0.9844 | 0.9844 |
| 3 | 0.9219 | 0.9062 | 0.9180 | 0.9180 |
| 4 | 0.9844 | 0.9844 | 0.9922 | 0.9922 |
| 5 | 0.8867 | 0.8945 | 0.8789 | 0.8906 |

Exact values and all raw per-seed metrics are in [the machine-readable output](../results/exploratory/connectivity-2026-10-26.json).

**Outcome:** no survival advantage for messages or locally adaptive trust was observed here. The adaptive rule did produce contact creation, trust adjustments and pruning. Mechanism functioning does not mean it improves performance.

## Limitations

1. Local sensors already expose nearby food and hazards; social cues could be redundant in this task. It does not require cooperation.
2. Fixed means *fixed trust*, not permanent edges; contacts expire with movement in both limited-link conditions.
3. There is no metabolic cost for messages, trust updates or links. Counts of messages are not physical joules, and no architecture efficiency claim is justified.
4. Five seeds, no holdout or classical chemotaxis baseline, and no inference about generalization. No claim of scientific significance.
5. Simulated local policy updates are not physical synapses, emergent general intelligence, independent hardware or autonomous cell growth.
6. Research dates precede the guide's stated building period.

## Scientific decision

Preserve the negative result rather than tuning parameters to manufacture a win. The next experiment should be a cooperation-required world with complementary partial observations, so that information exchange can be tested against a genuine no-message control under matching resource budgets.

## Reproduce

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python -m emergent_cells.experiments.connectivity --seeds 1,2,3,4,5 --steps 180 --shock-step 90 --output results/runs/connectivity-reproduction.json

Previously run under Windows / Python 3.12.10 / NumPy 1.26.4. This historical developmental run must not be presented as qualifying hackathon evidence without clarification.
