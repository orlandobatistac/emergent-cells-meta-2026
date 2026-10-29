# R1 — Closed-Loop Action Selection from Sensor History and Learned Recurrent State

**Research date:** October 28, 2026. **Status:** two new agent policies evaluated; R1 not finished.

## Question

Can a CPU-only agent translate self-supervised sensory predictions into *actual actions* on a partially observed multi-goal task, without an externally supplied map, coordinates or BFS? When does predictive memory actually improve goal completion?

We explicitly investigate **two observation regimes** so that successful performance is not misattributed to either extra information or learned state.

## Research context

- Jeen, Bewley & Cullen, *Zero-Shot Reinforcement Learning Under Partial Observability* (Reinforcement Learning Journal, 2025): memory is useful when objectives and state are partially observed. https://rlj.org/papers/jeen2025zero
- Eberhard, Muehlebach & Vernade, *Partially Observable Reinforcement Learning with Memory Traces* (ICML 2025): history compression for partial observability. https://proceedings.mlr.press/v267/eberhard25a.html
- Fu et al., *Knowledge Retention in Continual Model-Based Reinforcement Learning* (ICML 2025): revisiting useful states and retaining world-model knowledge. https://proceedings.mlr.press/v267/fu25f.html
- Ma et al., *Efficient Exploration via Observation-Action Constraint on Trajectory-Based Intrinsic Reward* (2026): novelty from trajectories rather than single frames. https://doi.org/10.1049/cit2.70100

These works motivate the experiments, but the algorithms below are our own simplified CPU prototypes, not reproductions.

## A. Original sensor-only task — no distant-goal signal

The agent receives four adjacent-wall bits, ambiguous floor texture, an adjacent-target detection vector and private reward. The world chooses three sequential goals and never transmits global coordinates or a shortest route.

The agent's **engineered** exploration heuristic chooses actions with fewer visits to its current internal state, discourages repeated known transitions, and takes the goal action if a goal is detected adjacent to the current room. Five controls use exactly the same action-utility formula with different internal state definitions:
- Observation-only current reading.
- Explicit three-pair observation/action history (state memorization but not trained latent dynamics).
- Self-supervised pretrained recurrent hidden state projected onto 24 unsupervised centroids (learned from training trajectories; no position labels).
- The same recurrent weights with recurrent history erased at each action.
- Memoryless reactive and random references.

The pretrained RNN uses **16,000 source sensor observations, 20 epochs, 32 hidden units**, one fixed training seed 19. Prototypes are trained on sensor/action sequences with unsupervised cluster averaging. The recurrent model does **not** learn its action policy from extrinsic task rewards here: its action choice is a designed novelty strategy over learned states.

### Independent evaluation: 100 episode seeds 4001–4100 per world type, 160 primitive actions, three goals

| Agent | Original world complete | Modified passage complete |
|---|---:|---:|
| Random | 3/100 | 5/100 |
| Reactive observation | 9/100 | 9/100 |
| Novelty using observation only | 25/100 | 20/100 |
| Novelty using explicit history | **52/100** | **36/100** |
| Novelty using recurrent state | 37/100 | 33/100 |
| Recurrent memory erased | 33/100 | 23/100 |

**Interpretation:** learning an internal predictive state is not sufficient for strong goal-seeking. It helps relative to a memoryless sensor-only agent, but the engineered history-key memory still provides stronger completion on this task. The modified passage reduces both methods' success. These are completion rates on finite, closely related synthetic worlds, not autonomous general navigation.

Raw study data: `results/audited/r1-strict-sensor-navigation.json`. The small-sample earlier prototype `results/exploratory/r1-latent-nav-pilot.json` is retained.

## B. Separate task with a local target-proximity measurement

The sparse task above has *no information about a goal until the agent is next to it*. To test explicitly goal-informed control, we add a **new, informative local sensor**: a scalar proximity measurement representing a noisy, clipped signal strength from the current target. Inside the evaluation harness the sensor is generated from hidden Manhattan target distance, rescaled to [0,1] and perturbed with zero-mean Gaussian noise (standard deviation 0.025). The agent receives only the scalar, not a direction, route, target ID or coordinates. This is an **engineered distance proxy**, not a measurement of a physical radio beacon; such a sensor makes the task substantially easier and may not be present in a real-world application. The earlier no-signal environment is preserved unchanged.

The agent combines its local state key with previous-action and signal-trend context. It **learns** an empirical local action preference from observed positive/negative signal change and sparse goal reward over **300 three-goal training episodes per condition**. An engineered novelty/recency component encourages exploration. There is no centralized shortest-path search and the policy does not receive a global map.

Conditions:
- Learned recurrent latent state + noisy signal + reward-progress training.
- Same with explicit three-step history instead of recurrent embedding.
- Same signal and training but only the current observation as the state key.
- Recurrent agent with all signal readings masked to zero, retaining sparse adjacent-target rewards.
- Same trained recurrent weights, but recurrent state erased on every update.
- Cold recurrent signal agent with no previous action-value training.

The recurrent feature extractor is pretrained for 16,000 sensory observations, 20 epochs, 32 hidden units, using exactly the same sensor-only data sources as A. The goal policy's reward-progress training is separate.

### Independent evaluation: 100 episode seeds 3001–3100 per world type, 160 primitive actions, three goals

| Policy | Original world complete | Modified passage complete |
|---|---:|---:|
| Trained recurrent + noisy signal | **100/100** | 50/100 |
| Trained explicit-history + signal | **100/100** | 53/100 |
| Trained current-observation-only + signal | 83/100 | 35/100 |
| Trained recurrent, **signal masked** | 46/100 | 29/100 |
| Trained recurrent, hidden state erased | **100/100** | **56/100** |
| Recurrent with signal, **no action-value training** | 46/100 | 35/100 |

**Interpretation:** direct action-value learning with an informative goal signal closes the loop from observation to action and substantially improves task completion in the original maze. However, the recurrent-state ablation is **not negative**: when given the strong proximity cue, erasing recurrent memory does not hurt and slightly helps in the modified maze. Therefore we **cannot claim** that recurrent memory is responsible for signal-driven navigation success. The explicit-history alternative is competitive.

The proximity experiment is intentionally separate from the sparse sensor experiment. **Do not merge their rates or use proximity results to claim that coordinate-free observation alone achieves 100% completion.**

Raw: `results/audited/r1-beacon-policy-100-seeds.json` (with per-seed records); earlier fixed-protocol run: `results/audited/r1-beacon-holdout.json`; development runs: `results/exploratory/r1-beacon-pilot.json`, `results/exploratory/r1-beacon-300-dev.json`.

## CPU resource and reproducibility

The sensor-only recurrent predictor has 32 float32 hidden units. The source-model weight arrays occupy 13,056 bytes and the 24 latent centroids occupy 3,072 bytes, excluding Python objects, reward tables, trajectory buffers and runtime RSS. The source-model pretraining CPU process time in the strict sensor evaluation was approximately 2.95 seconds; this **excludes** goal-policy episode training. See machine-readable protocol timings for later per-policy process CPU costs. No electrical energy was measured.

The comparator is not a general neural RL baseline; local action preferences use a custom engineered progress bandit. The layout family and action vocabulary are fixed. Changes are applied before evaluation episodes, not injected midway. Source pretraining uses privileged randomized simulator resets but no labels are sent to learning models.

### Reproduce

    python -m pip install -e .
    python -m unittest discover -s tests -q
    python run.py latent-navigation --seeds 4001,4002,4003,4004,4005 --train-steps 16000 --epochs 20 --hidden 32 --centers 24 --output results/runs/strict-nav.json
    python run.py beacon-navigation --seeds 3001,3002,3003,3004,3005 --warm-episodes 300 --output results/runs/beacon-nav.json

Full 100-seed seed lists and results are stored in the JSON artifacts and supplied in the benchmark command-line options.

## Next falsifiable experiment and decision gate

The priority is **not** to add more perfect target signals. It is to train an action selection policy using only partial local observations and *learned recurrent state*, then show:
1. A meaningful absolute improvement in completed multi-goal missions over both the explicit-history novelty control and the same policy with recurrent memory erased.
2. Improvement under held-out maze layouts, shifted sensors, blocked corridors and strict total-action-budget accounting.
3. No action-selection use of hidden environment coordinates, global graph-search routines or experimenter-provided task IDs.
4. CPU and actual RSS profiling, plus message-passing between independent cells in later R2/R3.

If an engineered history table remains stronger, preserve it as the leading non-oracle controller until a neural recurrent policy genuinely wins. This is an honest research milestone, not a declaration of general autonomous intelligence.
