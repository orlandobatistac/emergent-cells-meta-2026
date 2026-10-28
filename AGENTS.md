# Contributor and AI Agent Instructions

## Language policy — mandatory

**All repository content MUST be written in clear, modern, professional US English, in the style of a high-quality AI research startup.** This applies regardless of the human conversation language, agent model, IDE, operator, prompt, or AI driver.

Scope: README, docs, research reports, experiment protocols, PR titles and descriptions, commit messages, GitHub Issues, test names, code comments and docstrings, CLI help text, logs and user-facing messages, configuration labels, variable/function/class names, and JSON keys where practical.

Use concise, technically precise language. Write for engineers, researchers, and early-stage startup contributors. Avoid marketing hype, exaggerated claims, unnecessary jargon, and claims of demonstrated AGI or physical energy efficiency.

Conversations and research reviews with the project owner may be conducted in Spanish. That does not authorize Spanish in repository content. Translate any Spanish draft before saving or committing it.

Preserve numeric evidence, seed IDs, experimental conditions, benchmark outputs, public APIs, and reproducibility. Do NOT translate identifiers or filenames when doing so would break imports, commands, published links, or data formats. Historical experimental records must remain immutable; create an English companion or editorial translation with traceability instead of changing original results.

## Scientific development

- Python and NumPy only for experimental code. No frontend, JavaScript, or HTTP service.
- State hypotheses, controls, benchmarks, limitations, and failure cases explicitly.
- Separate development seeds from untouched evaluation seeds.
- Keep measurements of simulated energy separate from real energy in joules.
- Work in feature branches / PRs; never merge into main automatically.
- Before submitting, run relevant Python tests and inspect diffs for language consistency.
