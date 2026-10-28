# Emergent Cells Lab — Núcleo científico en Python

Laboratorio experimental reproducible para aprendizaje celular descentralizado, comunicación local, memoria y energía **simulada**. Contiene algoritmos Python, tests científicos, resultados numéricos y reportes, sin frontend, HTTP, JavaScript, HTML ni Meta API.

**Alcance:** las simulaciones demuestran capacidades limitadas en ambientes matemáticos diseñados; NO son evidencia de AGI, eficiencia energética física, independencia del sustrato ni crecimiento de células físicas. La energía del simulador es una recompensa, no joules.

## Ejecutar en la laptop

Instalar dependencias: python -m pip install -e .

- Ver experimentos: python run.py list
- Simulación base: python run.py simulate --compare --seeds 1,2,3 --steps 240 --cells 256 --output results/runs/simulation.json
- Memoria dinámica: python run.py dynamic-memory --seeds 11,12 --pairs 32 --phase-steps 4500 --dropout 0.08 --output results/runs/third-rule.json
- Curiosidad aprendida: python run.py learned-curiosity --help
- Todos los tests: python -m unittest discover -s tests -q

Cada experimento incluye sus opciones --help. El argumento --output, cuando existe, guarda JSON para que ChatGPT analice directamente los resultados sin interfaz.

## Estructura

- run.py: punto único de entrada Python.
- src/emergent_cells/cells/: unidades de aprendizaje local.
- src/emergent_cells/world/, simulation.py: mundos con recursos y controles.
- src/emergent_cells/experiments/: hipótesis, algoritmos, variantes y evaluaciones reproducibles.
- src/emergent_cells/evaluation/: mediciones y análisis.
- tests/: tests Python de reproducibilidad y comportamiento.
- results/audited/: datos de replicación y semillas independientes.
- results/exploratory/: pilotos y resultados negativos conservados.
- docs/: informes científicos, métodos y limitaciones.

## Línea científica hasta el 27 de octubre de 2026

1. Ablación de conectividad local, incluyendo hallazgos negativos.
2. Protocolo binario aprendido mediante recompensas privadas: alrededor del 98% de éxito en una tarea XOR diseñada con parejas fijas, frente a controles de ~50% sin mensajes informativos.
3. Memoria de contextos A/B y detector de sorpresa por recompensa local: buena recuperación bajo un detector programado.
4. Memoria dinámica: cinco semillas, éxito 93.7% XOR → 94.3% XNOR → 95.2% AND → 94.1% recuperación XOR → 94.5% recuperación XNOR, con pérdida aleatoria estable del 8% de recursos. El control de dos memorias logró 24.1% en AND.
5. Límite negativo: aumentar la pérdida de recompensa del 8% al 20% sin cambiar la regla causó creación falsa de memorias.
6. Estudios exploratorios posteriores sobre deriva contextual, exploración causal y curiosidad aprendida. Consultar sus informes originales y JSON antes de generalizar resultados.

Síntesis científica en español: docs/REPORTE_CIENTIFICO_2026-10-27.md.

## Política del repositorio

La antigua interfaz del hackathon permanece recuperable en el historial Git y las ramas anteriores, pero no forma parte del código actual. No se reescriben resultados históricos. Los nuevos estudios deben conservar controles equivalentes, semillas nuevas, resultados negativos, documentación de límites y verificaciones Python. Después de revisión se integran en main. No requiere mantenimiento de interfaz visual.

## Latest scientific investigation — curiosity transfer to unseen configurations

A Python-only held-out task study tested previously **untrained combinations** of XOR rule flips and local reward scarcity with frozen cell-local curiosity Q policies. In five fresh seeds (3001–3005), adding an authored two-failed-probe stopping rule raised scarce-01 target accuracy from **86.3% to 96.0%** while reducing request bits from **405.4 to 98.2** per seed. It increased overall mixed-rule-plus-scarcity accuracy from **93.8% to 96.3%**, but **reduced global-change-with-drought accuracy from 65.4% to 64.1%**. A manually specified uncertainty controller remained more effective on several comparisons; no universal advantage was shown.

This tests transfer within the same designed XOR/flip/scarcity family, NOT unfamiliar mathematics or spontaneously discovered curiosity. Read the full Spanish methods/results report at docs/TRANSFERENCIA_CURIOSIDAD_2026-10-27.md. Raw per-seed dataset is results/audited/2026-10-27-curiosity-ood-scenarios-fresh-seeds.json. Execute with: python run.py task-transfer --help.
