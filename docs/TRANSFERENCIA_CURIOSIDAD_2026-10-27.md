# Emergent Cells Lab — Transferencia de curiosidad a escenarios no entrenados
Fecha: 27 de octubre de 2026. Estudio Python exploratorio con control negativo.

## Pregunta
¿La política de exploración aprendida por células con recompensas locales se transfiere a nuevas combinaciones de reglas y escasez sin reentrenarse?

Motivación académica: IJCAI 2024, Causal Knowledge in Task-Agnostic Exploration (https://ijcai.org/proceedings/2024/591), e IJCAI 2025, Continual Multi-Agent Coordination (https://ijcai.org/proceedings/2025/759). Este trabajo NO reproduce sus algoritmos.

## Entrenamiento y pruebas
El currículo ORIGINAL de entrenamiento incluye stable, changed_11, scarcity_11, changed_01, scarcity_10, global_change y uniform_drought. Se añadió solamente evaluación para seis combinaciones nunca usadas en entrenamiento: changed_10, scarcity_01, changed_10_drought, scarcity_01_partial, mixed_change_10_scarcity_01 y global_change_drought.

Estos escenarios son NOVEDOSOS dentro de la familia XOR/flip/escasez, no nuevas familias matemáticas arbitrarias. La representación bayesiana por cuatro slots, canal de comunicación y mundo de parejas fueron diseñados por nosotros. El agente recibe su información privada, mensajes y recompensas, no etiquetas de tarea ni respuestas correctas. La Q de exploración permanece congelada en evaluación.

Condiciones comparadas: Q con recompensa intrínseca; Q sin bonificación intrínseca; Q más parada tras 2 o 4 exploraciones fallidas por slot; política manual de explorar en incertidumbre; pasivo; explorar siempre; y Q no entrenada con desempate aleatorio. La parada después de k fallos es una HEURÍSTICA local diseñada, no un comportamiento aprendido.

## Desarrollo y validación independiente
Piloto: semillas 11,12,13 con 16 parejas, preentrenamiento de 1600 pasos, 90 episodios de metaprendizaje de 40 pasos y 180 evaluaciones. Se fijaron las dos guardias k=2 y k=4.

Validación posterior: semillas 3001–3005, 24 parejas, preentrenamiento de 2400 pasos, 140 episodios de 55 pasos, 300 encuentros de evaluación por mundo. Se evaluaron 240 combinaciones semilla × mundo × método. No se modificaron algoritmos, estados ni umbrales tras examinar estas semillas.

## Resultados principales — promedios de 5 semillas

| Mundo no entrenado | Indicador | Curiosidad Q | Guardia 2 | Guardia 4 | Manual incertidumbre | Pasivo |
|---|---|---:|---:|---:|---:|---:|
| changed_10 | Acierto en 10 | 92.5% | 92.5% | 92.5% | 95.5% | 0.9% |
| scarcity_01 | Acierto en 01 | 86.3% | 96.0% | 95.7% | 94.3% | 98.5% |
| scarcity_01_partial | Acierto en 01 | 85.3% | 94.2% | 93.9% | 93.9% | 98.2% |
| changed_10_drought | Acierto en 10 | 63.6% | 63.6% | 63.6% | 70.4% | 0.8% |
| mixed_change_10_scarcity_01 | Acierto general | 93.8% | 96.3% | 96.2% | 96.5% | 73.2% |
| global_change_drought | Acierto general | 65.4% | 64.1% | 65.0% | 71.7% | 1.5% |

El indicador interno de «critical» no tiene significado en global_change_drought: en esa fila solo se muestra el acierto general, nunca el cero convencional de critical.

## Costes, ventajas y fallos
- En scarcity_01, la política Q original solicitó 405.4 exploraciones por semilla, frente a 98.2 de guardia 2: reducción del 75.8%. Acierto general: 94.4% frente a 97.1%; rendimiento neto simbólico de 0.667 frente a 0.670 por pareja/episodio. Las cinco semillas mejoraron en ambos indicadores.
- En mixed_change_10_scarcity_01, solicitudes 423.2 frente a 136.4: reducción del 67.8%. Acierto general 93.8% frente a 96.3%; las cinco semillas mejoraron. La ventaja de rendimiento neto fue pequeña: +0.0008 por pareja/episodio.
- En global_change_drought, la guardia 2 empeoró el acierto general de 65.4% a 64.1% en las cinco semillas y perdió 0.0074 de recompensa neta simbólica por pareja/episodio.
- El método manual de explorar solo con incertidumbre fue superior en algunos mundos. La política aprendida NO domina a las reglas manuales.
- En scarcity_01 el método pasivo obtuvo 98.5% de acierto específico, pero falla cuando la regla cambia. No hay un método dominante universal.

El acierto de la compuerta y la recogida de recursos son indicadores distintos. Las unidades de exploración y mensaje NO son joules ni coste real de hardware. Cinco semillas permiten replicación exploratoria, no intervalos de confianza universalmente válidos.

## Conclusión científica
La curiosidad aprendida se transfiere parcialmente a posiciones de cambio no practicadas y las políticas de parada local pueden hacerla más eficiente cuando la escasez vuelve infructuosas las exploraciones. No hemos demostrado aprender de forma autónoma CUÁNDO detenerse: esa regla sigue siendo manual, y puede impedir exploración útil durante cambios verdaderos de reglas con recursos escasos.

Próximo experimento falsable: entrenar directamente una política local de tres decisiones — explorar, explotar o descansar — y comparar con los controles manuales k=2/k=4 en mundos reservados, con costes de tiempo, RAM y mensajes medidos por separado. La política debe resolver el compromiso entre cambio real y escasez sin modificar reglas a partir de los resultados de evaluación.

## Reproducir
Ejecutar desde la carpeta local:

    .\.venv\Scripts\python.exe run.py task-transfer --seeds 3001,3002,3003,3004,3005 --pairs 24 --pretrain 2400 --meta-episodes 140 --meta-horizon 55 --eval-horizon 300 --output results/runs/transfer-replica.json
    .\.venv\Scripts\python.exe -m unittest discover -s tests -q

Datos: results/audited/2026-10-27-curiosity-ood-scenarios-fresh-seeds.json
SHA-256: 718c8258d02d10e70c976c0736e4f255cc24908943046a63e0c46698df2ed066

Ninguna interfaz web ni Meta Model API participó en estas pruebas.
