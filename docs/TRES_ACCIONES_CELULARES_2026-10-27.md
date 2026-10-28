# Emergent Cells Lab — Aprender EXPLOTAR / EXPLORAR / DESCANSAR

**27 de octubre de 2026 — evaluación exploratoria en Python.** Ninguna interfaz web ni Meta API. La energía aquí son **unidades simbólicas**, no joules.

## Pregunta y criterio de éxito

¿Pueden las células aprender una tercera decisión **DESCANSAR** que ahorre recursos, y usarla *solo cuando compense*, sin deteriorar de forma importante el aprendizaje de reglas cambiantes? La mera aparición de acciones REST no es evidencia suficiente. Se comparan **acierto en la compuerta, recursos realmente cosechados, energía simbólica neta, frecuencia de descanso, bits solicitados y exploraciones**.

Investigación relacionada:
- [Asking for Information by Evaluating the Communication and Coordination Trade-Off in Multi-Agent POMDPs, IEEE RA-L 2026](https://doi.org/10.1109/LRA.2026.3703246) — información valiosa solo si el coste de comunicación merece la pena.
- [Interpretable abstractions of neural networks predict behavior during human information gathering, Nature Neuroscience 2026](https://www.nature.com/articles/s41593-026-02342-9) — decisión de cuándo dejar de recopilar información.
- [First return, then explore, Nature 2021](https://www.nature.com/articles/s41586-020-03157-9) — dificultad de RL ante recompensas escasas y engañosas.

Estas referencias motivan la pregunta; NO hemos replicado esos métodos.

## Mundo e información accesible a las células

Dos células fijas por compuerta. Cada una recibe su bit privado, símbolo de 1 bit de su compañera y recompensa local. El código binario de comunicación ya se aprendió en entrenamiento separado. El mundo determina XOR, cambios condicionales de regla y disponibilidad del recurso; **las células no reciben etiquetas de escenarios, respuestas correctas ni disponibilidad del recurso**.

Acciones por célula:
1. EXPLOTAR: accionar palanca con política aprendida y correcciones privadas.
2. EXPLORAR: solicitar prueba contrafactual conjunta; cuando ambas participan, el par invierte su acción habitual.
3. DESCANSAR: no accionar palanca, no obtener recursos, no consumir coste de actividad. Si alguna descansa, la compuerta no abre. El hecho de que la pareja participara es observable; esto es una condición diseñada del entorno, no un sensor de recursos.

Presupuesto sintético: cada célula activa gasta 0.12 por episodio, una prueba coordinada gasta 0.025 y cada solicitud binaria cuesta 0.004. El retorno neto por pareja descuenta el coste activo medio de ambas, la prueba y las solicitudes. Una pareja que siempre descansa recibe **0.0 recompensa neta**. El éxito de la compuerta exige dos acciones correctas **y participación**; un descanso se cuenta como ausencia de éxito incluso cuando es óptimo en energía. Todas las variantes se evalúan con esta misma física.

La política nueva es una Q-tabla **privada de cada célula**, 3 acciones, recompensa local neta y bonificación local de progreso en creencias bayesianas; no hay crítico global. La representación de creencias, la primitiva de prueba, el coste de actividad, la forma de la Q-tabla y sus señales de entrada están diseñados.

## Iteraciones probadas antes de mirar semillas nuevas

- **V1:** 3 acciones con solo creencias y fallos de prueba; aprendió descanso moderado bajo sequía extrema, pero siguió gastando recursos.
- **V2:** añadió cuenta por slot de intentos ACTIVOS que no recibieron pulso (estado local) y mantuvo una Q de 96 estados, 3 acciones. Mejoró algunos aciertos pero no resolvió el problema de adaptar descanso al contexto global.
- **V3:** añadió una categoría de ENTRENAMIENTO con pulsos disponibles solo 8% de las veces. Con ello aprendió a descansar casi siempre en el mundo de prueba con 5% de pulsos (NUNCA usado en entrenamiento), pero también descansó demasiado en mundos donde sí convenía actuar. El control de dos acciones recibió exactamente el mismo curriculum de escasez.
- **V4:** añadió una media exponencial de recompensas ACTIVAS recibidas por la propia célula (no sensor perfecto) y su discretización en 3 niveles. La Q de estas dos variantes contiene 288 estados por 3 acciones; el control de dos acciones tiene **la misma cantidad de parámetros**, con REST inhabilitado. Ambos se entrenaron en las mismas condiciones, incluidas las de 8% de pulso. En el piloto la observación global mejoró la lectura de sequía frente a V2, pero la política siguió descansando en exceso en múltiples mundos con oportunidad real.

Controles:
- learnt_three / learned_two: Q local entrenada, con o sin tercera acción, misma capacidad de 96×3 valores por célula.
- learned_three_no_history: Q local 3 acciones, sin observación del historial de fallos de acciones normales.
- learned_three_low_supply / learned_two_low_supply: Q de 96×3, mismo curriculum que añade sequía de entrenamiento al 8%.
- learned_three_ema / learned_two_ema: Q de 288×3, mismo curriculum, datos locales de recompensa reciente, con o sin descanso.
- manual_two: probar ante incertidumbre, regla prescrita.
- guard_two: probar ante incertidumbre con límite de 2 pruebas fallidas por slot, regla prescrita.
- always_rest: abstenerse siempre, control superior de energía en apagones prolongados pero inútil para abrir compuertas.

**Importante:** estas parejas controladas tienen igualdad en número de parámetros, pero distintas representaciones y habilidades. La capacidad mayor de las EMA no es igual a la de 96 estados. Los valores Q no se actualizan durante el periodo de evaluación. El estado de las creencias, el detector y la intervención siguen siendo diseñados.

## Curriculum y mundos no entrenados

Entrenamiento base: las 7 tareas históricas (stable, changed_11, scarcity_11, changed_01, scarcity_10, uniform_drought y global_change). Curriculum ampliado: agrega el mundo de 8% disponibilidad, 25% de probabilidad en entrenamiento. Ninguno de los mundos específicos de prueba se usa para entrenar Q.

Evaluación: changed_10, scarcity_01, scarcity_01_partial, changed_10_drought, mixed_change_10_scarcity_01, global_change_drought, severe_drought (5% de disponibilidad) y global_change_blackout (5% con regla global invertida).

Estos son escenarios nuevos dentro de una **familia sintética común**, no funciones matemáticas arbitrarias. Los controles fueron comparados con flujos estocásticos reproducibles y semillas emparejadas.

## Hallazgos del desarrollo

En los pilotos, la política con descanso aprendió acciones REST a partir de recompensas. Cuando el curriculum incluyó escasez severa, descansó alrededor de **90%** en severe_drought y elevó la recompensa neta aproximada de **−0.08** (2 acciones activas) a **−0.01** por pareja/episodio. Pero en mundos donde todavía había recursos también descansó demasiado y redujo el acierto. Ninguna de las cuatro iteraciones dominó de forma uniforme a los controles de 2 acciones. El control REST permanente obtiene neto 0 durante severe_drought, por lo que **la mejor política aprendida todavía no alcanza ese límite energético**.

Pilotos completos, con fallos conservados en:
- results/exploratory/2026-10-27-three-action-v1-short-pilot.json
- results/exploratory/2026-10-27-three-action-v2-pilot.json
- results/exploratory/2026-10-27-three-action-v3-low-supply-pilot.json
- results/exploratory/2026-10-27-three-action-v4-ema-pilot.json

## Interpretación y líneas futuras

Este mecanismo comprueba una **capacidad de abstención aprendida en un laboratorio numérico restringido**, no una política óptima general. La Q de un paso y observaciones agregadas no bastan para decidir cuándo descartar información que podría ser útil más tarde. Además, REST reduce el flujo de recompensa informativa y puede crear un bucle de falta de evidencia: precisamente la tensión entre exploración y explotación que queríamos estudiar.

Siguiente hipótesis: sustituir Q de recompensa inmediata por una **política temporal/recurrente con opción REST de duración acotada**, y medir rendimiento acumulado por episodio, latencia de retorno a la exploración, tasa de falsas abstenciones, incertidumbre causal y presupuesto real de memoria y CPU. Mantener un conjunto independiente de familias de evaluación, sin retocar umbrales después de verlo.

No existe evidencia de autonomía general, agentes LLM, hardware descentralizado, sustrato independiente ni eficiencia real en joules.


## Validación en cinco semillas nuevas, sin retocar el algoritmo

Se ejecutaron semillas **7001–7005**, con 16 parejas (32 células), 1600 pasos de preentrenamiento de la comunicación, 90 episodios de Q local de 40 pasos, y 180 evaluaciones por escenario. Total: **400 registros completos** (5 semillas × 8 escenarios × 10 métodos). La Q se mantuvo congelada en evaluación. Los números son medias exploratorias, no intervalos de confianza universales.

| Escenario | Método | Acierto | Descanso | Recompensa neta simbólica |
|---|---|---:|---:|---:|
| severe_drought (pulso 5%) | 2 acciones, curriculum escaso | 87.62% | 0% | **−0.0765** |
| severe_drought | 3 acciones, mismo curriculum/capacidad | 5.81% | **89.67%** | **−0.0101** |
| severe_drought | siempre descansar | 0% | 100% | **0.0000** |
| global_change_blackout (5%, regla invertida) | 2 acciones, curriculum escaso | 8.78% | 0% | **−0.1169** |
| global_change_blackout | 3 acciones, mismo curriculum/capacidad | 4.59% | **88.95%** | **−0.0116** |
| changed_10 (recursos normales) | 2 acciones, curriculum escaso | **95.01%** | 0% | **0.7561** |
| changed_10 | 3 acciones, mismo curriculum/capacidad | 82.68% | 15.51% | 0.6616 |
| mixed_change_10_scarcity_01 | 2 acciones, curriculum escaso | **92.81%** | 0% | **0.5211** |
| mixed_change_10_scarcity_01 | 3 acciones, mismo curriculum/capacidad | 61.89% | 35.39% | 0.4804 |

En los DOS apagones extremos, el modelo de tres acciones superó en retorno energético a su modelo pareado de dos acciones **en 5/5 semillas**, pero aprendió a descansar demasiado en changed_10 y en el mundo mixto. La reducción relativa de pérdida en severe_drought fue 86.8% aproximadamente, sin llegar al límite neto cero del controlador que siempre descansa. Es ahorro selectivo en ambiente extremo, NO superioridad universal.

### Ablaciones negativas y positivas

| Mundo y métrica neta | Q 2 acciones original | Q 3 acciones original | Q 3 acciones con EMA y curriculum de escasez |
|---|---:|---:|---:|
| severe_drought | −0.0743 | **−0.0462** | **−0.0201** |
| scarcity_01 | 0.5439 | **0.5564** | 0.5273 |
| mixed_change_10_scarcity_01 | 0.5225 | **0.5338** | 0.4887 |
| global_change_drought | 0.2392 | 0.2386 | 0.2028 |

Con el curriculum ORIGINAL, la tercera acción mejoró retorno neto en scarcity_01 **5/5 semillas**, y en el mundo mixto **4/5**, aunque disminuyó la tasa de apertura. Añadir la media reciente de recompensas al estado no resolvió el problema de generalización: con frecuencia produjo demasiadas abstenciones y redujo el retorno en mundos con recursos aprovechables.

El experimento demuestra que una célula puede APRENDER cuándo usar un tercer tipo de acción ya proporcionado, pero no que invente REST ni la representación para detectar sequía. Nuestro algoritmo aprende recompensa inmediata como un bandit contextual, no es un agente con planificación temporal ni RL recurrente; al descansar elimina temporalmente la información necesaria para identificar si el entorno se recuperó.

### Datos, hash y reproducción

Datos: results/audited/2026-10-27-three-action-unseen-seeds.json
SHA-256: e3126978658bbebff93ea5fa2fc572f5a12160953dcd870ef898764d1845e6a5
Pilotos V1–V4: results/exploratory/2026-10-27-three-action-v*.json

Desde la laptop:
    .\.venv\Scripts\python.exe run.py three-action --seeds 7001,7002,7003,7004,7005 --pairs 16 --pretrain 1600 --train-episodes 90 --train-steps 40 --eval-steps 180 --output results/runs/reproduce-three-action.json

No se midió consumo eléctrico real, período de recuperación tras apagones, independencia del sustrato ni escalamiento masivo. **No se alcanzó el requisito de superar o igualar todos los controles de dos acciones**: resultado parcialmente positivo para ahorro, negativo para generalización robusta.
