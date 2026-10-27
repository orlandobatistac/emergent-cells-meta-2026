# Use the local coordinated-probe experiment in the interactive lab

This branch adds a separate panel, **Exploracion cooperativa**, to the existing local browser laboratory. The resource-grid canvas is a different artificial environment from the paired-gate experiment; do not treat its moving cells as the gate-learning cells.

Start the lab on the machine carrying the repository:

```powershell
python -m pip install -e .
python -m emergent_cells.demo.server --port 18771
```

Open `http://127.0.0.1:18771` **on that same laptop**. The server binds to loopback, has no user login, and must never be exposed to the public Internet.

Click **Comparar pruebas** under *Exploracion cooperativa*. This triggers a bounded 2-seed development pilot: 24 pairs, 3000 local-message training episodes, 350 diagnostic episodes in each of five experimental worlds. The server compares passive, random, periodic and Bayesian information-gain intervention. Use the browser Download report action to obtain complete metrics including request bits, new memory flags, symbolic costs and per-input correctness.

The program offers GET `/api/active-probes` (current report) and POST `/api/active-probes/run` (exactly an empty JSON object), and saves a copy in GET `/api/report`.

**This demo is smaller than the independent five-seed audit**. No Meta API calls or credentials are required. Its agents do not control tools, invent natural language or demonstrate physical low-power computing.

Full independent-seed research report: [ACTIVE_CAUSAL_EXPLORATION_2026-10-27.md](ACTIVE_CAUSAL_EXPLORATION_2026-10-27.md).
