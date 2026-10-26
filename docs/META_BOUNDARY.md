# Meaningful Meta AI integration without centralizing learning

**Design specification only — Oct 26, 2026.**

The Meta Model API is the **experimental scientist**, not the policy controller.

## Permitted

- Translate human descriptions into structured world proposals and test hypotheses.
- Condition new *development* worlds on aggregated trial metrics and failure descriptions.
- Suggest interventions to discriminate competing hypotheses about adaptation.
- Produce explanations that clearly distinguish data from interpretation.

## Prohibited interface

- Directing cell positions, movements, motor outputs, synapse weights or rewards at each step.
- Seeing private held-out seeds/outcomes when proposing a final test.
- Generating runnable Python to bypass sandbox or mutate scientific results.
- Selecting only favorable runs for presentation.

## Contract (to implement)

An audited `WorldSpec` object will include: version; seeded map dimensions; food/resource fields; hazards; transport/attenuation; sensor encoding; time horizon; resource replenishment; intervention timestamp and parameter overrides. Fields must have explicit bounds and default values. Deterministic world IDs hash canonical normalized specs.

Input to Meta: high-level goal, allowed spec schema, scientific limitations, aggregated development metrics.

Output from Meta: **only** JSON representing a proposed world and a brief reason. A validator clamps or rejects unsafe/impossible configurations; do not silently rewrite results.

## Evidence that Meta is substantial

Compare at fixed budgets and comparable difficulties:
- Human-designed curriculum.
- Uniform random procedural curriculum.
- Meta-designed *non-adaptive* curriculum.
- Meta-designed adaptive curriculum.

Freeze held-out worlds before development selection. Attribute benefit to Meta only if the evaluation demonstrates it; otherwise state observed limitations. Cache Meta proposals with timestamp, model ID and request settings, but never include API keys or user personal data.
