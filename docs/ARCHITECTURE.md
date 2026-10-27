# Arquitectura científica — Python solamente (27-oct-2026)

Objetivo: probar mecanismos falsables de aprendizaje descentralizado, no presentar inteligencia general como demostrada.

- cells/population.py: estados y reglas locales; no hay política global aprendida.
- world/environment.py y simulation.py: recursos físicos simulados, dinámica y baselines. Las reglas objetivo pertenecen al MUNDO, nunca al agente.
- experiments/: codificación de señales privadas, mensajes binarios aprendidos, memorias contextuales, detectores locales, recompensa condicional, sondas causales y curiosidad aprendida. Cada experimento puede poseer su CLI, datos y supuestos.
- evaluation/benchmark.py: métricas y controles emparejados.
- run.py y cli.py: despacho a experimentos Python existentes.
- tests/: reproducibilidad, independencia local, ausencia de filtración de etiquetas objetivo, límites y no regresión.
- results/: experimentos exploratorios negativos y estudios auditados, sin modificar sus registros originales.

Los arreglos NumPy vectorizados y la ejecución de reglas en el simulador constituyen coordinación del MUNDO, no prueban hardware descentralizado. El detector, la conexión fija por pares, el protocolo de un bit y la capacidad de memoria son hipótesis diseñadas. La energía no es medida en joules.

Se eliminan los artefactos de visualización y endpoints HTTP: una interfaz no aporta evidencia adicional al núcleo científico.
