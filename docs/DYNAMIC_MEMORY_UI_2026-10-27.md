# Local browser demonstration: third-rule memory allocation

Available in the **Crear una tercera memoria** panel after starting:
```bash
python -m pip install -e .
python -m emergent_cells.demo.server
```
Open `http://127.0.0.1:8765` on the same laptop, not an external device. The demo uses 2 development seeds, 24 pairs, 3500 episodes per A/B/C stage, 900 per return and 8% random resource-pulse loss. It evaluates five algorithms and displays last-200 success for all five phases and average allocated banks.

Server routes: `GET /api/dynamic-memory`, `POST /api/dynamic-memory/run` (strictly empty JSON object), and session report `GET /api/report` includes `dynamic_memory_evaluation`.

This user-facing development demo does not use exact untouched-seed holdout settings; see [scientific results](DYNAMIC_MEMORY_THIRD_RULE_2026-10-27.md) instead. It requires no Model API key and sends no network calls to Meta. The server is local single-user research software and must not be exposed publicly.
