# Objetivo y workflow científico — Emergent Cells Lab (28-oct-2026)

## Visión de largo plazo
Investigar una arquitectura de IA útil, interactiva y modular formada por células con estados y actualizaciones locales, capaz de aprender mediante experiencia, intercambiar información, conservar y recuperar memoria, elegir acciones y gestionar agentes. Aspiramos a reducir coste computacional y energético y explorar posibles implementaciones agnósticas al sustrato; **nada de esto se considera demostrado**.

## Prioridad actual: cerrar el ciclo funcional
Entrada observable → procesamiento y comunicación entre células → aprendizaje y memoria local/distribuida → elección de acción → respuesta al entorno → recompensa/retroalimentación → actualización y siguiente interacción. No imponer etiquetas de tarea ni objetivos correctos dentro de las células. La infraestructura del simulador puede coordinar el mundo, pero no actuar como controlador de políticas celulares.

REST y la optimización energética pasan a la **fase posterior**: preservar resultados y variantes existentes, sin hacer del descanso un requisito del siguiente prototipo. No eliminar ni reescribir los experimentos previos.

## Workflow por hitos (cada hito es falsable)
1. **Closed-Loop Cellular Learning v1.** Entorno Python/NumPy con observaciones parciales, tareas secuenciales y acciones con consecuencias verificables. Ejecutar episodios continuos sin revelar tipo de tarea; medir acierto, recompensa y adaptación. Entregar baseline antes de escalar.
2. **Comunicación y selección de participación.** Comparar células independientes, comunicantes y control centralizado pequeño. Aislar el aporte causal de los mensajes con ablaciones y costes de comunicación.
3. **Memoria persistente y recuperación.** Tareas A→B→C→A/B con retención y pruebas de olvido; diferenciar memoria aprendida de índices, detectores y reglas codificados.
4. **Transferencia real.** Separar combinaciones de la misma familia de tareas de familias y reglas genuinamente nuevas. Congelar parámetros antes de semillas retenidas; reportar fallos.
5. **Interacción usable.** Ofrecer interfaz **de Python (CLI)** para introducir observaciones/instrucciones de prueba, recibir respuestas/acciones y realimentar el sistema, sin frontend web ni HTTP. La capacidad de usar herramientas o manejar agentes es un hito futuro, no una habilidad implícita.
6. **Escalabilidad.** Variar población, conectividad y carga; medir si aumenta capacidad sin controlador central. Separar escala simulada de distribución física.
7. **Optimización posterior.** Estudiar REST, cómputo selectivo, mensajes escasos, memoria, tiempo CPU, RAM y consumo eléctrico *medido*; comparar calidad igualada contra arquitecturas de referencia. Explorar independencia de sustrato solo como investigación posterior.

## Protocolo de cada experimento
- Hipótesis y criterio de falsación predefinidos; métodos y controles con presupuestos comparables.
- Semillas de desarrollo separadas de semillas de validación nunca usadas para ajustar.
- Métricas: exactitud, recompensa, recursos obtenidos, transferencia, retención, comunicación, cómputo, variación entre semillas y resultados negativos. Energía simulada se etiqueta como simbólica.
- Ejecutar unit tests y pilotos de desarrollo primero. Congelar diseño; luego validación retenida.
- Guardar JSON, comando, configuración, versión Git y reporte científico en español.
- Documentar qué es mecanismo diseñado y qué comportamiento es aprendido.
- Trabajar en rama/PR; **no modificar main automáticamente**.

## Próximo experimento concreto
Implementar Closed-Loop Cellular Learning v1 con 3 baselines: células sin comunicación, células comunicantes y agente centralizado pequeño. Primero tarea de decisión parcial y cambio de contexto, después memoria y transferencia. No añadir optimizaciones de descanso ni interfaces web durante esta fase.

## Criterio de avance
Pasar de fase solo al demostrar funcionamiento reproducible con comparación causal y límites explícitos. Si el sistema no supera los controles, registrar fracaso e investigar su causa antes de acumular nuevas funciones.
