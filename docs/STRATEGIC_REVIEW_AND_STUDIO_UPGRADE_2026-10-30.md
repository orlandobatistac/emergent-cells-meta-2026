# Strategic Review and Studio Upgrade — October 30, 2026

## Executive assessment

Emergent Cells Lab now has a reproducible CPU-first experimental platform and a working localhost Studio. It does **not** yet have a general-purpose artificial intelligence, a trained decentralized planner, an autonomous network of physical nodes, or an electrical-efficiency advantage measured on hardware. The core research opportunity is to learn representations **and local messages** jointly, then establish a causal improvement on unseen worlds at matched compute budgets.

### Verified baseline entering this session

- The October 29 research reported 92/100 stable, 91/100 changed and 94/100 procedural missions for bounded event-driven local updates. The corresponding zero-*queued-message* controls reported 77/100, 63/100 and 78/100, respectively.
- The existing controller has authored relative odometry, frontier reward/value propagation, explicit direction semantics and a central delivery scheduler; only some separate diagnostics have reward-trained memories/messages.
- The existing optional offline language assistant is a separate Qwen model via Ollama.
- The benchmark includes a direct observed-neighbor handshake even when the queued-message budget is zero. Therefore **zero queued messages is not zero inter-cell information**. Compare claims must retain that qualification.

These numbers are prior evidence in the October 29 document, **not new independent confirmation** in this upgrade.

## Highest-value research gaps

| Priority | Gap | Falsifiable acceptance criterion |
| --- | --- | --- |
| P0 | Authored scalar values/messages instead of learned communication | Train sparse messages from local data/reward; evaluate unseen maps, shuffled messages, fully isolated neighbors and equal interaction budgets |
| P0 | Uncertain localization and dynamics drift | Adapt to remapped/slipping motors across unseen layouts with held-out success, calibration, CPU and memory all reported |
| P1 | Central Python message scheduler and direct object access | Run process-isolated cell actors over a bounded transport; inject process loss, delay, reordering and packet drops; compare outcomes |
| P1 | Limited transfer and small synthetic task family | Blind procedural holdouts, obstacle shifts and new goal rules with no environment identifiers reaching the policy |
| P1 | No real-world efficiency measurement | Compare matched success and model quality with process CPU, RSS, latency, message bytes and measured watts/joules when instrumentation is available |
| P2 | A domain-constrained research demonstration rather than general agent usefulness | Introduce carefully bounded observer/tool actions driven by a foundation-model copilot, clearly distinct from cell cognition |

Independent references: IEEE *Decentralized Collective World Model for Emergent Communication and Coordination* (2025), DOI 10.1109/ICDL63968.2025.11204457; IEEE 2026 review *A Review of Emergent Communication in Multi-Agent Reinforcement Learning*, DOI 10.1109/ICMLT69916.2026.11688719; Meta Model API docs https://dev.meta.ai/docs/overview.

## Implemented in this session

1. **Counterfactual evidence panel:** same deterministic seed interval for three controls: event-driven queued messages, queued messages disabled (immediate neighbor handshake remains), synchronous diffusion. Results include paired completion, average actions, mean message deliveries and per-seed exportable JSON. Bounded to 1–12 seeds per request.
2. **Independent experiment state:** `POST /api/evidence` does not reset, step or mutate the current Studio mission. It does not train or alter scientific policies.
3. **Live zero-queued-message mode:** users can explicitly start a local event-network mission with queue delivery budget zero.
4. **Optional Meta Model API research copilot:** `provider=meta` uses Muse Spark 1.3 over Meta Model API only if the **server environment** contains `MODEL_API_KEY`. It receives limited observer summary, not the hidden simulator world. It is an existing hosted language model and does not direct the cells or execute tools.
5. **Offline-first continuity:** the default provider remains preinstalled local Ollama Qwen 2.5 Coder 7B; simulator commands never need the Meta key.
6. **Tests:** paired reproducibility, control isolation, bounded inputs, optional-provider mocking and HTTP semantics. A network test with actual Meta credentials is **not** claimed when a key is absent.

## October 30 exploratory comparison (not an independent held-out claim)

Two deterministic 12-seed trials using fresh exploratory intervals were saved under `results/exploratory/2026-10-30-studio-paired-{stable,changed}.json`. On stable missions, queued-message event cells solved **12/12**, queued-message-disabled cells **9/12**, and synchronous diffusion **12/12**. On closed-passage missions, the corresponding outcomes were **10/12**, **4/12**, and **11/12**. These are small same-family test samples and do not establish broad statistical generalization. The synchronous counter counts *local value-update operations*, whereas the event counter counts *queued deliveries*: they are not identical bandwidth metrics. No electrical watts/joules were measured.

Reproduce with `python -m emergent_cells.experiments.studio_evidence --seed 56001 --trials 12 --output results/exploratory/2026-10-30-studio-paired-stable.json` (use `--seed 57001 --changed` for the second scenario).

## Run and reproduce

```powershell
python -m pip install -e ".[studio]"
python -m unittest discover -s tests -q
python run.py studio --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765/ on LANDITO-LAPTOP. Start a seeded mission, advance 20 actions, inject one **simulated cell-memory loss**, run the side-by-side paired comparison, and export the record. To use Meta instead of local Ollama, set `MODEL_API_KEY` in the server's environment **before starting** the Studio, then select Meta in the research assistant.

## Scientific boundary and next experiment

The next genuinely original advance should **not** be another aesthetic UI change. Train a minimal sender/receiver parameterization with a 1–4 bit local message under sparse rewards and compare against authored values, shuffled packets, zero queued delivery **and full neighbor isolation**. Use fixed development seeds and a fresh preregistered holdout; measure per-task success plus CPU time, memory and bytes transferred. If it fails, retain the negative result and revise the model rather than reintroducing a hidden map or global BFS.

The present Studio is a runnable experimental product with explanatory AI assistance. Model API access is optional and has not been verified live without a configured key.
