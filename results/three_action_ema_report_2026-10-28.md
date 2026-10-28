# Informe científico: tres decisiones y EMA (2026-10-28)

## Hipótesis
Una media móvil de recompensa privada debería ayudar a distinguir sequías de cambios reales y mejorar la política EXPLOIT/PROBE/REST.

## Método
Mismo preentrenamiento, condiciones y generadores de evaluación emparejados para 12 métodos. 12 parejas, 600 pasos de preentrenamiento, 45 episodios de 30 pasos, evaluación 130 pasos por escenario. Desarrollo: semillas 11,12,13. Validación sin ajustar parámetros: 41,42,43. Escenarios de evaluación pertenecen a la familia conocida de tareas; las semillas de validación son nuevas, no constituyen nuevas familias de tareas.

## Resultados (media validación; recompensa neta simbólica, mayor mejor)
| Escenario | Tres EMA | Dos EMA | Tres sin EMA | Guard dos |
|---|---:|---:|---:|---:|
| changed_10 | 0.476 | 0.523 | 0.552 | 0.566 |
| scarcity_01 | 0.359 | 0.379 | 0.398 | 0.381 |
| changed_10_drought | 0.148 | 0.195 | 0.191 | 0.221 |
| scarcity_01_partial | 0.395 | 0.405 | 0.412 | 0.411 |
| mixed_change_10_scarcity_01 | 0.344 | 0.348 | 0.355 | 0.396 |
| global_change_drought | 0.090 | 0.091 | 0.121 | 0.146 |
| severe_drought | -0.061 | -0.088 | -0.059 | -0.088 |
| global_change_blackout | -0.072 | -0.114 | -0.069 | -0.114 |

En validación, tres EMA descansó 37.5% en sequía extrema y 36.3% en apagón; precisión de compuerta 29.6% y 7.6%, respectivamente. En changed_10 descansó 18.6% con precisión 62.2%, frente a 69.7% en dos EMA. Resultados negativos: EMA no supera de forma consistente a tres acciones sin EMA ni a dos acciones fuera de sequías extremas.

## Auditoría del mecanismo
REST es acción real sin coste propio de actividad, mensajes ni recurso. Las acciones de pares pueden dejar costos activos en su compañero, conforme al diseño. EMA se alimenta de recompensas locales observadas solo cuando ambos participan; el valor inicial 0.75, umbrales 0.25/0.65, y las escalas 0.985/0.92 son parámetros diseñados. Q tiene 288 estados por célula con EMA, tres columnas; dos acciones enmascara REST. Los costos están separados en actividad, prueba y comunicación. Se informa desviación estándar entre semillas en JSON. Se preserva la comparación aleatoria emparejada, aunque distinta política consume distinto número de muestreos debido a acciones.

## Interpretación
Hipótesis de mejora global por EMA NO respaldada. El descanso evita pérdidas bajo escasez severa, pero inhibe oportunidades de explotación y reconocimiento de tareas. La EMA congelada durante descanso podría reforzar políticas de pasividad. El aprendizaje depende de creencias bayesianas, inversión de acciones, señal intrínseca y recompensas simbólicas diseñadas. No se demuestra inteligencia general, independencia física del sustrato, ni ahorro medido en joules. Tres semillas de validación implican incertidumbre considerable.

## Próximo experimento
Hipótesis nueva: introducir una prueba de vigilia con intervalo controlado y coste energético explícito, y comparar con dos acciones con igual presupuesto, ablando la EMA. Especificar antes de evaluar un escenario realmente novedoso, más semillas y criterio Pareto de recursos/precisión/coste. Reservar nuevas semillas sin optimizarlas. Nunca ajustar sobre 41,42,43.

## Reproducción
.\.venv\Scripts\python.exe run.py three-action --seeds 11,12,13 --pairs 12 --pretrain 600 --train-episodes 45 --train-steps 30 --eval-steps 130 --output results/three_action_ema_dev_11_12_13.json

.\.venv\Scripts\python.exe run.py three-action --seeds 41,42,43 --pairs 12 --pretrain 600 --train-episodes 45 --train-steps 30 --eval-steps 130 --output results/three_action_ema_validation_41_42_43.json

.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_three_action_energy.py -q
