# Emergent Cells Lab — Reporte científico del crecimiento de memoria
Fecha: 27 de octubre de 2026. Estado: experimento exploratorio reproducible, NO inteligencia general.

## Pregunta de investigación
¿Pueden las células identificar que una tarea previamente desconocida requiere una memoria adicional, crearla bajo demanda y recuperar las reglas anteriores? Probamos XOR (A), XNOR (B), AND (C), retorno a A y retorno a B. Los agentes no reciben la regla ni la acción correcta. Cada célula procesa un bit privado, el mensaje binario aprendido de su pareja y una recompensa energética local compartida en el par. El simulador tiene parejas y canal fijos.

## Métodos investigados y ejecutados
La versión V1 del detector EMA muy sensible FALLÓ: creó memorias nuevas sin cambio de tarea. V2 redujo falsas activaciones, pero recuperaba B lentamente porque trataba memorias reutilizadas como inmaduras. V3 mejoró el reingreso: una memoria antigua se marca madura según la recompensa obtenida al probarla, sin acceder al identificador de tarea. Todas estas decisiones son heurísticas diseñadas por nosotros.

La comparación usa cinco semillas nuevas 1001–1005, 32 parejas, 4,500 episodios por regla, 900 por retorno, 8% de recompensas aleatoriamente ausentes y éxito en los últimos 200 episodios de cada fase.

| Método | XOR A | XNOR B | AND C | Retorno A | Retorno B |
|---|---:|---:|---:|---:|---:|
| Crear memoria nueva según recompensa local | 93.7% | 94.3% | **95.2%** | 94.1% | 94.5% |
| Capacidad máxima de dos memorias | 93.7% | 93.6% | **24.1%** | 94.3% | 94.5% |
| Política única | 93.9% | 0.5% | 24.6% | 93.7% | 0.9% |
| Detector más rápido | 93.9% | 94.8% | 95.2% | 94.1% | 94.5% |
| Control que conoce la regla | 93.9% | 94.9% | 95.4% | 94.2% | 95.2% |

El caso privado '11' de AND alcanzó entre 92.2% y 95.4%. Por eso el resultado no se explica por responder siempre cero, que solo resolvería alrededor del 75% de los casos balanceados de AND. Los promedios de memorias lógicas asignadas por célula fueron aproximadamente 1.006 (A), 2.006 (B) y 2.987 (C).

## Prueba negativa: confusión con ruido
Manteniendo XOR sin cambiar la regla, tres semillas mostraron cero asignaciones falsas al mantener 0% o 8% de recompensas ausentes. Al pasar de 8% a 20% de recompensas ausentes, el sistema produjo **62–64 asignaciones innecesarias por población de 64 células**. El detector todavía no distingue fiablemente entre una caída de recursos y una regla distinta.

## Auditoría de asignación física de memoria
La versión original preasignaba tres columnas NumPy de política. La revisión utiliza inicialización de una sola columna y amplía el arreglo cuando aparece una demanda. Para 64 células, cada columna de cuatro pesos y un baseline float64 representa **2,560 bytes de payload**; una, dos y tres columnas representan 2,560, 5,120 y 7,680 bytes respectivamente, sin contar objetos Python ni otros estados.

La ampliación física se efectúa para TODA la población en el instante en que alguna célula requiere una columna nueva. Cada célula conserva un contador lógico de sus memorias autorizadas. Por tanto, esto es **asignación física perezosa a nivel de población**, no asignación física exacta e independiente de cada célula. Tampoco crece el número de células ni se miden joules. Las pruebas comprueban esa distinción y la preservación de bancos previos al ampliar almacenamiento.

## Conclusión científica
Se ha logrado una forma funcional de aprendizaje continuo limitado: los agentes reutilizan memorias conocidas o habilitan una nueva cuando un detector local de recompensa considera inadecuadas las anteriores. El tercer mundo AND se aprende y las dos tareas previas conservan un alto rendimiento. Este mecanismo está diseñado y se comprueba en un simulador con tareas simples; NO equivale a inteligencia emergente general, detección robusta de cambios ambientales, crecimiento celular físico ni una máquina agnóstica al sustrato.

## Próxima hipótesis falsable
Diseñar un detector que distinga cambios de regla de cambios en tasa de recompensa estudiando la evidencia local por combinación de señal privada, mensaje recibido, acción y recompensa. Comparar contrastes estadísticos por ventanas y una versión bayesiana con penalización de crecimiento. Evaluar ruido estable y cambiante, recompensas asimétricas, nuevas reglas y falsas alarmas con semillas separadas. En paralelo, medir costes reales de RAM/CPU y presupuesto de mensajes para evitar declaraciones de eficiencia energética no comprobadas.

## Referencias académicas verificadas
- Dick et al. (2025), Statistical Context Detection for Deep Lifelong Reinforcement Learning: https://proceedings.mlr.press/v274/dick25a.html
- Zhang et al. (2024), Dynamics-Adaptive Continual Reinforcement Learning via Progressive Contextualization: https://pubmed.ncbi.nlm.nih.gov/37285252/
- Schwarz et al. (2018), Progress & Compress: https://proceedings.mlr.press/v80/schwarz18a.html

## Evidencia adjunta
- results/audited/2026-10-27-dynamic-third-rule-holdout.json
- results/audited/2026-10-27-noise-stress.json
- results/exploratory/dynamic-pilot-v1-failed.json
- docs/DYNAMIC_MEMORY_THIRD_RULE_2026-10-27.md
- src/emergent_cells/experiments/dynamic_memory.py
- src/emergent_cells/experiments/stationary_noise.py

No se invocó Meta Model API ni se generaron cargos por este experimento.
