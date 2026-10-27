# Adversarial reward availability and an extra local resource sensor — October 27, 2026

This is a **separate evidence stream** from reward-only contextual drift detection. It was designed and tried after seeing that the reward-only conditional detector handled uniform reward shortages. The five-seed reward-only held-out study was **not re-tuned** or overwritten after this additional investigation.

## Scientific premise

When resource scarcity selectively affects particular input states, observed reward failures can mimic a changed mapping from actions to reward. A passive learner that observes only its existing actions and rewards need not be able to distinguish both causes. Carefully chosen *exploratory interventions* may resolve some cases; alternatively, one can supply an independent, **physically local measurement of resource availability**.

The latter changes the information available to the agent. It must **not** be presented as a breakthrough in task identification from reward alone.

## Adversarial negative control

The shared-gate task remains XOR throughout 3,500 training episodes and 1,000 stationary test episodes. In the test interval, the world still withholds resources randomly 8% of the time for most cue combinations but withholds 100% when the pair's private cue combination is `11`. The cells do **not** receive a task label or a label for the chosen scarce cue case.

Three development seeds 11–13: all **48/48 cells** in both the relative-average and reward-only conditional-signature arms falsely allocated new memories despite an unchanged XOR rule. This was an intentionally hostile, input-correlated resource process, **not** a changing task.

## Alternative with a local resource-availability sensor

We added an **explicit idealized binary local sensor** indicating whether the resource pulse was physically available to a given pair in a given episode, independent of gate success. This signal is not an answer about which action is correct and it does not identify the current task rule. However, it **is** extra information about the world that reward-only agents do not have.

As a simple control implementation, reward evidence from an unavailable opportunity is *optimistically imputed as non-failure* for novelty detection; ordinary cell policy training still uses only actual obtained energy and sender messages are trained as before. This is an engineered heuristic, not a learned estimate of resource availability, and it could suppress real task-change detection when resource availability is itself very low.

### Fresh independent seeds 4001,4002,4003

| Seed | False extra memory cells, old EMA | False extra cells, reward-only conditional | False extra cells, extra local sensor |
|---|---:|---:|---:|
| 4001 | 44 | 44 | **0** |
| 4002 | 46 | 46 | **0** |
| 4003 | 46 | 46 | **0** |

Same seeds, new-rule A→B→C→A→B sequence with 24 pairs, 3,500 training episodes for each rule, 900 return episodes per revisit and 8% baseline reward dropout:

| A | B | C | Recovered A | Recovered B |
|---:|---:|---:|---:|---:|
| **91.9%** | **93.2%** | **93.0%** | **91.3%** | **92.1%** |

A separate stationary XOR resource-withholding test at 8%, 20%, 35% and 45% produced **zero false new memories** in each of those three sensor-enabled trials.

The 4001–4003 seeds were not used during development of the resource-availability sensor method. However, the broad task rules and algorithm family had been explored previously, so the outcomes do not demonstrate transfer to an unknown natural task.

[Fresh sensor task-and-noise raw data](../results/audited/2026-10-27-local-sensor-fresh-seeds.json) (SHA-256 `9f3b845682e16151f970becb9512ab47188e9fab339f235c1d4dddc4c8a59ce5`)

[Fresh adversarial comparison raw data](../results/audited/2026-10-27-adversarial-scarcity-fresh-seeds.json) (SHA-256 `714912c9d635b75e675ea297720eddea70d366f3cb7beff00c79c340f523f181`)

[Development adversarial raw data](../results/audited/2026-10-27-adversarial-scarcity-local-sensor.json) (SHA-256 `fd47ad3987efb42534fea373606e5fb025d9d2d6e96af83ab544d3cbfa53b2e2`)

## What this experiment does **not** establish

- There is no actual physical resource sensor on the laptop. The simulator directly supplies an idealized world availability bit that future hardware would have to measure independently.
- Perfectly distinguishing scarcity and new task rules under every possible noise process is not proven. Absent sensor information, **active intervention** and more sophisticated local inference could provide information that a passive reward history lacks.
- The sensor has no accounted material, power or time cost. We have not measured real electrical consumption.
- No new partners, learned new sensor modalities, discovered topology, language, open-ended skills, or general intelligence.
- This research did **not** call Meta Model API or confirm billable credits.

## Reproduce

```bash
python -m emergent_cells.experiments.conditional_scarcity --seeds 4001,4002,4003 --output results/runs/reproduce-adversarial.json
python -m emergent_cells.experiments.robust_drift --seeds 4001,4002,4003 --pairs 24 --phase-steps 3500 --stress-steps 1000 --methods availability_sensor --output results/runs/reproduce-sensor.json
```

## Next research question

Can an agent **learn** to distinguish two causal hypotheses — mapping change versus resource unavailability — using actions, sparse experimental probes and local rewards, *without* assuming a perfect sensor? This would test active inference, causal interventions and explicit energy budgets. It would be scientifically stronger than simply increasing the model's private memory.
