# Research Goals and Scientific Workflow — October 28, 2026

## Long-term vision
Investigate an interactive, modular AI architecture built from locally learning cells that exchange information, develop and retrieve memories, select actions, and eventually orchestrate agents. Lower computational and physical energy costs and substrate-independent implementations are **long-term hypotheses, not demonstrated outcomes**.

## Current priority: close the functional intelligence loop
Observation → local processing and communication → learning and distributed memory → action selection → environment response → feedback → updated cell state and subsequent interaction.

Never expose privileged task labels or correct target actions to the cells. The simulation may coordinate environmental mechanics but must not act as a centralized learned policy controller.

REST and energy optimization are deferred until the integrated learning loop works. Preserve all prior results and code; do not erase negative findings.

## Roadmap: testable milestones
1. **Closed-Loop Cellular Learning v1.** Build a Python/NumPy environment with partial observations, sequential tasks, and consequential actions. Measure task accuracy, reward, adaptation, and learning without task labels.
2. **Communication and participation.** Compare isolated cells, communicating cells, and a compact centralized baseline under matched budgets. Perform causal message ablations.
3. **Persistent memory and recall.** Evaluate A → B → C → A/B sequences, retention, interference, and recovery. Separate engineered memory indexing from actually learned representations.
4. **Generalization.** Distinguish novel combinations from truly novel task families. Freeze hyperparameters before untouched validation seeds; report failures.
5. **Practical interaction.** Provide a Python CLI for entering test observations or instructions, receiving predictions/actions, and supplying feedback. No web UI or HTTP service. Tool use and agent orchestration are future milestones, not current capabilities.
6. **Scalability.** Vary cell count, connectivity, and workload. Determine whether capability scales without centralized policy control. Distinguish simulation scale from physical distribution.
7. **Efficiency optimization.** Only after functional success, test REST, selective activation, sparse messaging, memory footprint, runtime, RAM, and measured physical power. Compare equivalent task quality against relevant baselines. Substrate independence remains exploratory.

## Experiment workflow
- Define a falsifiable hypothesis, evaluation criteria, and comparable baselines before implementation.
- Use separate development seeds and previously untouched validation seeds.
- Track task accuracy, reward, resource capture, transfer, memory retention, communication, computation, per-seed variability, and negative results.
- Run unit tests and development pilots first; freeze the design before final evaluation.
- Save configuration, reproducible command, version/commit, machine-readable JSON, and an **English** scientific report.
- Clearly label engineered mechanisms vs. learned behaviors; symbolic energy is not joules.
- Develop and publish through feature branches and pull requests. Never change `main` automatically.

## Immediate experiment
**Closed-Loop Cellular Learning v1:** compare isolated cells, communicating cells, and a compact centralized agent on partial-observation decisions and contextual shifts. Then test persistent memory and task transfer. Do not introduce REST optimization or frontend development in this phase.

## Advancement criteria
Advance only when the target behavior reproduces under controlled comparisons with explicit limitations. If performance does not beat relevant baselines, record the failure and investigate before adding complexity.
