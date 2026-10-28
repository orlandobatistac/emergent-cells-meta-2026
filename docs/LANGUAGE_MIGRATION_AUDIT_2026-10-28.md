# Language Migration Audit — October 28, 2026

## Policy
All new or edited repository artifacts must be professional US English. See `AGENTS.md` and `README.md`. The language used to communicate with the project owner or operate an AI coding agent does not change this requirement.

## Completed in this change
- `AGENTS.md`: machine-readable contributor language policy and scientific safeguards.
- `README.md`: English project overview, findings, execution instructions, and contribution standards.
- `docs/ARCHITECTURE.md`: English architecture summary.
- `docs/RESEARCH_GOAL_AND_WORKFLOW_2026-10-28.md`: English research roadmap and acceptance criteria.
- `results/three_action_ema_report_2026-10-28.md`: English report with preserved data values and methods.

## Remaining legacy documentation
A repository scan detected historical Spanish-language scientific documents, including:
- `docs/ACTIVE_CAUSAL_EXPLORATION_2026-10-27.md`
- `docs/CONTEXT_SWITCH_2026-10-27.md`
- `docs/REPORTE_CIENTIFICO_2026-10-27.md`
- `docs/TRANSFERENCIA_CURIOSIDAD_2026-10-27.md`
- `docs/TRES_ACCIONES_CELULARES_2026-10-27.md`
- `docs/TRES_ACCIONES_ENERGIA_2026-10-27.md`

These files are **legacy research evidence, not compliant final English documentation**. Translate each with a line-by-line scientific check against the original, keeping source traceability, reported statistics, definitions, and experiment identifiers intact. Rename only when cross-references and links can also be updated safely. Preserve originals in Git history. Other markdown files should receive a complete editorial audit; the keyword scan is not a certification.

## Source code
An initial keyword scan found no obvious Spanish prose in tracked Python source or tests. This is **not a complete linguistic or naming audit**. Do not rename Python symbols, JSON keys, CLI commands, or files solely for style if doing so breaks reproducibility or compatibility. Future code and comments must be English.

## Completion criteria
- All active documentation, code comments/docstrings, CLI strings, test descriptions, and contributor-facing text reviewed for English.
- Legacy reports translated accurately and their numeric claims cross-checked.
- Tests rerun after any code changes.
- Results, scientific conclusions, and reproducible experiment IDs preserved.
