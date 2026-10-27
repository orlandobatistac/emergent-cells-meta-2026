# A -> B -> A browser integration

The existing single-user local browser lab now contains a separate panel **Aprendizaje continuo — A → B → A**. This task uses fixed paired-cell gate physics, not the moving-world canvas.

The server exposes:
- GET /api/context-switch — saved report, labeled local/no Meta.
- POST /api/context-switch/run — accepts **only** empty JSON; bounded to three development seeds 11,12,13, 64 pairs, A 4500 episodes, B 3500 episodes, 200 evaluation steps.
- GET /api/report — full session now includes the contextual-learning report.

UI displays **A before**, **B after training**, and **A immediately on return** for each of six variants. It states that external season cue is supplied, that extra memory banks are engineered and that Meta API is **not** called.

Run local demo:
    python -m pip install -e .
    python -m emergent_cells.demo.server

Open http://127.0.0.1:8765 **on the same host**. Never bind unauthenticated development server to public interfaces.

This is October 27 development; old independent research modules keep their original dates.
