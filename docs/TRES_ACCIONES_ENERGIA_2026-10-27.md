# Emergent Cells Lab — Aprendizaje local de tres acciones
Fecha: 27 de octubre de 2026. Python y NumPy. Estudio exploratorio reproducido en semillas nuevas.

## Pregunta e hipótesis
¿Puede una célula aprender, usando exclusivamente su propia recompensa y las señales privadas, cuándo elegir EXPLOTAR, EXPLORAR o DESCANSAR sin un umbral manual de parada?

Antecedentes conceptuales: Krale et al., Act-Then-Measure (ICAPS 2023), https://doi.org/10.1609/icaps.v33i1.27197; Nam et al., Reinforcement Learning with State Observation Costs (NeurIPS 2021), https://proceedings.neurips.cc/paper/2021/hash/83e8fe6279ad25f15b23c6298c6a3584-Abstract.html. Nuestros algoritmos NO reproducen estos trabajos.

## Mundo y agentes
Parejas fijas de células con bits privados, un símbolo binario aprendido por emisor y receptor y recompensa propia procedente de abrir una compuerta. Cada célula aprende una tabla Q individual. El mundo determina la regla XOR, sus inversiones y la tasa de recursos; ninguna etiqueta de escenario, solución correcta ni tasa de recursos llega a la célula.

EXPLOTAR usa la acción aprendida, con coste sintético de actividad 0.12. EXPLORAR ensaya la acción contraria, con coste adicional 0.025 por intervención y 0.004 por solicitud. DESCANSAR significa inactividad real de esa célula: ahorra coste, pero no obtiene muestras nuevas ni abre compuertas si la otra célula no participa. Las dos células pueden tomar decisiones incompatibles; esa discordancia no está instrumentada todavía como métrica separada. La recompensa neta por pareja/episodio = pulsos de recurso obtenidos menos costes sintéticos de actividad, intervenciones y mensajes. NO son joules ni mediciones eléctricas.

Las creencias bayesianas, cuatro slots por célula, protocolo binario, tarea, acción REST y costes fueron DISEÑADOS. Se aprenden preferencias de acción dentro del espacio, no se inventa espontáneamente una nueva arquitectura.

## Iteraciones del piloto (antes de fijar semillas nuevas)
V1: tres acciones con creencias y fallos de sonda. Descansa durante sequía pero pierde compuertas rentables.
V2: memoria de fallos de acción normal en cada slot privado. Mejoras limitadas en precisión y conservación de recursos.
V3: escenario de ENTRENAMIENTO con disponibilidad del 8%; favorece descanso extremo y sobretransfiere aversión a actuar.
V4: tasa reciente de recompensa local codificada en 3 niveles, EMA lenta con tasa 0.015. Evidencia local, NO sensor privilegiado de recursos.
V5: EMA rápida con tasa 0.08, para reaccionar antes y comparar estabilidad.
Los umbrales y tipos de estado se eligieron antes de analizar las nuevas semillas.

## Controles
Los métodos learned_three_ema y learned_two_ema tienen la misma tabla de 288 estados por 3 valores privados y el MISMO currículo de entrenamiento, excepto que REST está enmascarada en learned_two_ema. También se comparan EMA rápida (misma capacidad y currículo), versiones sin memoria temporal, con currículo normal, guard_two manual, manual_two y always_rest. El control always_rest obtiene exactamente 0 recompensa neta y 0 compuertas abiertas; es mejor que una pérdida neta en mundos inviables, pero no aprende ni gana recursos.

El currículo de entrenamiento contiene siete escenarios XOR/flip/escasez, más un escenario separado con 8% de recursos para las variantes EMA/low_supply. Los ocho mundos de evaluación no estaban etiquetados exactamente en el entrenamiento: changed_10, scarcity_01, changed_10_drought, scarcity_01_partial, mixed_change_10_scarcity_01, global_change_drought, severe_drought y global_change_blackout. Son combinaciones sintéticas dentro de la MISMA familia XOR/flip/escasez; no hay generalización demostrada a lógica o agentes arbitrarios.

## Protocolo de replicación
Piloto con semillas 11,12,13: 16 parejas, 1600 pasos de preentrenamiento del protocolo, 90 episodios de aprendizaje de 40 pasos y 180 episodios de prueba por escenario. Tras comparar V1–V5 se fijaron hiperparámetros.
Evaluación posterior con semillas NUEVAS 4001–4005, tamaños idénticos, 12 métodos, 8 escenarios: 480 ejecuciones. Tabla Q, codificadores y decodificadores congelados durante la evaluación. Datos crudos:
results/audited/2026-10-27-three-action-energy-fresh-seeds.json
SHA256: aae4103eae4b04454884c453808a7d3bbcfd4bf62c7603b613b80576c3467319.

## Recompensa neta simbólica por pareja/episodio, medias de cinco semillas

| Mundo de prueba | Tres acciones + EMA lenta | Dos acciones + EMA lenta | Tres + EMA rápida | Dos + EMA rápida | Manual guard_two |
|---|---:|---:|---:|---:|---:|
| changed_10 | 0.7002 | 0.7531 | 0.7521 | 0.7622 | 0.7601 |
| changed_10_drought | 0.2680 | 0.3481 | 0.2576 | 0.3528 | 0.3621 |
| global_change_blackout | -0.0229 | -0.1175 | -0.0334 | -0.1169 | -0.1161 |
| global_change_drought | 0.2010 | 0.1867 | 0.1745 | 0.1910 | 0.2758 |
| mixed_change_10_scarcity_01 | 0.5100 | 0.5386 | 0.5350 | 0.5369 | 0.5462 |
| scarcity_01 | 0.5179 | 0.5464 | 0.5460 | 0.5438 | 0.5475 |
| scarcity_01_partial | 0.5425 | 0.5854 | 0.5793 | 0.5822 | 0.5883 |
| severe_drought | -0.0194 | -0.0769 | -0.0306 | -0.0785 | -0.0763 |

## Descanso aprendido, resultados negativos y emparejamiento
En severe_drought, EMA lenta descansó el 81.1% del tiempo, obtuvo -0.0194, mientras dos acciones obtuvo -0.0769. Diferencia promedio emparejada +0.0575, favorable en las cinco semillas.
En global_change_blackout descansó 79.6%, obtuvo -0.0229 frente a -0.1175. Mejora emparejada +0.0946, cinco de cinco.
En changed_10_drought, abrió sólo 64.9% de las compuertas frente al 85.1% con dos acciones, y la energía neta cayó 0.2680 frente a 0.3481; ninguna de las cinco semillas mejoró.
En scarcity_01, acierto 78.5% frente a 94.2%; energía neta 0.5179 frente a 0.5464; ninguna semilla mejoró.
El control siempre descansar gana en conservación puramente inmediata durante las sequías inviables, pues tiene energía neta 0; la política aprendida conserva aún una pérdida pequeña.
El currículo de escasez severa hizo descansar demasiado a algunas políticas en entornos rentables. EMA rápida mitiga parcialmente, pero no resuelve la tensión.

## Conclusión científica
La acción REST es técnicamente aprendible mediante recompensas celulares locales y reduce las pérdidas durante sequías extremas. Sin embargo, el objetivo de obtener una política general que sepa cuándo aprovechar y cuándo descansar sin perjudicar la adaptación FALLÓ. No supera consistentemente al control de dos acciones ni al guard_two diseñado. La capacidad de ahorro y precisión son objetivos opuestos en esta arquitectura.

El protocolo de pareja requiere participación de ambas células, pero ninguna garantiza que la otra también actuará; la ausencia de una señal local de coordinación de descanso probablemente contribuye a pérdidas evitables. Además, REST no genera información: una célula inactiva puede tardar demasiado en detectar que el mundo vuelve a disponer de recursos.

No se midieron joules, electricidad, escalamiento físico, negociación espontánea de reposo, nuevos operadores lógicos ni inteligencia general.

## Próximo estudio
Añadir una señal local de consentimiento para descanso y un mecanismo de comprobación periódica de bajo costo; diseñar controles para separar negociación aprendida de reglas programadas. Simular temporadas alternas de disponibilidad 5% → 92% → 5%, medir parejas con decisiones discordantes, tiempo de reactivación, recompensas y coste de mensajes. Reportar cualquier señal proporcionada por el mundo que haga más fácil la tarea.

## Reproducción
python run.py three-action --seeds 4001,4002,4003,4004,4005 --pairs 16 --pretrain 1600 --train-episodes 90 --train-steps 40 --eval-steps 180 --output results/runs/three-action-replica.json
python -m unittest discover -s tests -q

Sólo Python/NumPy. Sin interfaz web, Meta API ni servicio remoto.
