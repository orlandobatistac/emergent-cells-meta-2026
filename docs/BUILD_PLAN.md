# Day-by-day execution — Meta 2026

**Official event guide lists building period Oct 27–Nov 4; kickoff Oct 26.** This planning-only setup is dated Oct 26; confirm with organizers whether any feature implemented on Oct 26 is eligible before claiming it as competition work.

| Date (ET) | Work | Acceptance gate |
|---|---|---|
| Oct 26 | Public repository, scope, boundaries, scientific protocol, team/credit/Discord setup | No pre-event private code imported; public repo accessible |
| Oct 27 | New deterministic world + 256-cell local interaction model | Headless run, fixed seed, unit tests for locality/resource bounds |
| Oct 28 | Independent untrained/random/learned baselines | Same worlds, reproducible CSV and honest early comparison |
| Oct 29 | Meta Model API -> structured world specification + validator/cache | Live API call or explicit offline fixture, no code execution from model |
| Oct 30 | Adaptive challenge loop with holdout split | Proven input/output audit, Meta does not steer cells |
| Oct 31 | Visual demo and perturbation/sensor swap | Stable recording; show failures and recovery metrics |
| Nov 1 | Controlled local plasticity and energy ablations | Guardrails; skip growth if baseline breaks |
| Nov 2 | Held-out seeded comparisons, error bars, negative controls | Frozen runs and reproducible report |
| Nov 3 | Clean-room install, video script + 3-minute video, README, license/IP review | External test of repo and video |
| Nov 4 | Full submission rehearsal and delivery before 9 PM ET | Submission acknowledgment, anonymous URLs work |

## Kill criteria

- Prioritize stable working local learner over flashy demo.
- If growth destabilizes core by Oct 31, mark it as future research, not a hackathon claim.
- If Meta credit is unavailable, use *previously recorded legitimate API-generated fixtures* where permitted, noting that live calls are disabled; don't fabricate Meta usage.
- If repeated trials show no adaptation, report that accurately and emphasize experimental infrastructure rather than inventing success.

## First code milestone

Implement **only** a deterministic headless world, independently updated cells, energy constraints, reproducible seed and a small test suite. Do not begin with a chatbot or dashboard.
