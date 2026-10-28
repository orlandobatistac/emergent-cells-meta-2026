# Three-Action EMA Study — October 28, 2026

## Hypothesis
A private exponential moving average (EMA) of recently observed reward could help cells distinguish resource drought from actual rule changes, improving EXPLOIT / PROBE / REST decisions.

## Experimental setup
All 12 methods share the same evaluation setup and paired random streams: 12 cell pairs, 600 pretraining steps, 45 training episodes of 30 steps, and 130 evaluation steps per scenario. Development seeds: 11, 12, 13. Untouched validation seeds: 41, 42, 43. The evaluated scenarios are combinations within the same designed task family, not unfamiliar task families.

## Independent-seed validation results
Mean **simulated net reward per opportunity** (higher is better):

| Scenario | Three-action EMA | Two-action EMA | Three-action no EMA | Engineered guard, two actions |
|---|---:|---:|---:|---:|
| changed_10 | 0.476 | 0.523 | 0.552 | 0.566 |
| scarcity_01 | 0.359 | 0.379 | 0.398 | 0.381 |
| changed_10_drought | 0.148 | 0.195 | 0.191 | 0.221 |
| scarcity_01_partial | 0.395 | 0.405 | 0.412 | 0.411 |
| mixed_change_10_scarcity_01 | 0.344 | 0.348 | 0.355 | 0.396 |
| global_change_drought | 0.090 | 0.091 | 0.121 | 0.146 |
| severe_drought | -0.061 | -0.088 | -0.059 | -0.088 |
| global_change_blackout | -0.072 | -0.114 | -0.069 | -0.114 |

Under severe drought, three-action EMA rested 37.5% of the time and achieved 29.6% gate correctness. Under global-change blackout, it rested 36.3% of the time and achieved only 7.6% correctness. In changed_10, it rested 18.6% of the time, reaching 62.2% correctness versus 69.7% for two-action EMA.

## Mechanism audit
REST truly avoids active-resource attempts and its own activity/message costs. A partner may still incur costs by attempting action. EMA changes only from private received rewards on active pair interactions. Its initial rate of 0.75, state thresholds of 0.25/0.65, and decay rates of 0.985/0.92 were designed by researchers rather than discovered by cells.

EMA agents use 288 state rows and three Q columns per cell. Two-action variants mask REST. Activity, probe, and communication costs are separately reported; JSON summaries include inter-seed standard deviations. Matched evaluation random seeds are used, but policies can consume different random samples through different actions.

## Conclusion
**The hypothesis of broadly improved task performance through EMA was not supported.** REST can reduce simulated losses in extreme scarcity, but the additional reward-history feature does not consistently outperform the non-EMA three-action policy. Resting also suppresses reward observations and may amplify passivity. Bayesian beliefs, action inversion, information bonuses, and key detectors remain engineered mechanisms.

No claim is made of general intelligence, substrate independence, or physical energy savings. Three validation seeds limit precision.

## Next experiment
Move to **Closed-Loop Cellular Learning v1**, integrating input, communication, memory, action, and feedback. Revisit sleep/wake policies only after this loop is demonstrated. Do not tune against seeds 41–43.

## Reproduction
```powershell
.\.venv\Scripts\python.exe run.py three-action --seeds 11,12,13 --pairs 12 --pretrain 600 --train-episodes 45 --train-steps 30 --eval-steps 130 --output results/three_action_ema_dev_11_12_13.json
.\.venv\Scripts\python.exe run.py three-action --seeds 41,42,43 --pairs 12 --pretrain 600 --train-episodes 45 --train-steps 30 --eval-steps 130 --output results/three_action_ema_validation_41_42_43.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_three_action_energy.py -q
```
