# Learned-message UI integration (October 27)

This panel is a separate executable scientific task, not a new behaviour of the roaming cells shown on the resource-grid canvas. The server offers POST /api/protocol/run (empty JSON only) and GET /api/protocol; the session report attaches the complete benchmark under learned_protocol_evaluation.

The bounded browser demo trains 128 separately learning cells in 64 fixed pairs for 3,500 episodes per seed (seeds 11,12,13), then tests for 150 new cue episodes per seed. It runs five ablations and two direct same-trained-weights message knockouts.

The implementation is purely local and never calls Meta or uses API credentials. Meta Model API remains a separately implemented world-designer integration whose real credentialed call has not yet been confirmed.

The October 26 pre-period experimental gate is retained for audit; this October 27 learned encoder + receiver algorithm and its integration are new. Existing simulation, world generator, and private scientific project remain separate. This is one narrow building-period milestone, not a claim about the eligibility of all earlier code.

From repo root:

    python -m pip install -e .
    python -m unittest discover -s tests -v
    python -m emergent_cells.demo.server

Open http://127.0.0.1:8765 on the same machine. Choose **Mensajes aprendidos** -> **Entrenar y comparar**. The server binds to loopback and has no authentication; do not expose it on public interfaces.
