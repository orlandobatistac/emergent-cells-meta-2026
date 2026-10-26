# MVP architecture — new competition implementation

**Planning dated 2026-10-26.** This is a design proposal only; no functionality is implemented yet.

## Objective

Implement a simulator in which approximately 256 simple cells adapt collectively to spatial energy/resource perturbations. Constraints: local perception, local memory, local interaction and local plasticity; no centralized policy, reward function that secretly encodes correct actions, or LLM-produced cell action vectors.

## Modules (to implement during event building period)

- `cells/`: cell states, neighbor coupling, local plasticity, local energy sharing and optional local growth. Distinguish internal metabolic pool from simulated environment resources.
- `world/`: deterministic 2D environment, spatial resource fields, hazards, sensory channels, occupancy/transport and seeded world dynamics.
- `meta/`: Meta model to structured, auditable world proposal, schema validator, bounded retry and deterministic replay fixtures.
- `evaluation/`: saved runs, seeds, baseline agents, matching budgets, ablations, confidence intervals, held-out experiments.
- `demo/`: visualization of local cell activations, environment, interventions and observed performance.

## Physical/simulation boundary

This is a *digital simulator of physical local-interaction hypotheses*, not a substrate-independent physical implementation. Hypothetical cellular behaviors should be expressible as local transfers and gradients. Any global simulator operations are bookkeeping (e.g. synchronized stepping, aggregate metrics), not a control signal delivered to the network.

## Critical data paths

```text
Human task -> Meta proposed WorldSpec JSON -> Schema + feasibility validation
                                                |
                                                v
                  Deterministic local environment (seeded)
                                                |
            local energy/sensor interactions + neighbor links
                                                |
                 autonomous local cell dynamics
                                                |
                motors/forces -> collective motion
                                                |
                 independent metrics collector
                                                |
             development summary -> next Meta proposal
```

Meta must never provide a per-cell action, direct synaptic update, target label for an episode or a handcrafted winning path.

## Locality acceptance tests

1. Destroying a direct neighbor edge prevents direct information flow over that edge.
2. Changing a distant world tile does not instantly change an unconnected cell's observation.
3. No global leaderboard/fitness selects individual weight updates during an episode.
4. Cell updates remain bounded in state, weight and simulated energy.
5. Baseline random/untrained/learned share comparable interfaces and energy/step budgets.

The schematic is not evidence that the properties already hold in code.
