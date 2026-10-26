# Emergent Cells: plan maestro de investigación y producto

Versión 1.0 · 26 de octubre de 2026 · Responsable de decisiones: Orlando Batista.
Estado: plan de ejecución; sus metas y estimaciones no son resultados demostrados.
Base verificada: rama `feat/interactive-validation-lab`, commit `6f318f80478e1ee9b397df6ab1bf7648fb536aef`, PR #3.

## 1. Visión y resultados que buscamos

Construir una IA personal de propósito general que pueda conversar, aprender de la experiencia, utilizar herramientas y coordinar agentes en equipos convencionales, con un presupuesto reducido de memoria, cómputo y energía. Investigar si una arquitectura de células con aprendizaje local, comunicación limitada, plasticidad y crecimiento puede sostener una parte cada vez mayor de estas capacidades.

Hay dos entregables relacionados, con evidencia separada:

1. **Motor celular validado:** mecanismos locales, capacidad, retención, adaptación y costo medidos frente a controles fuertes.
2. **Producto usable:** aplicación local con lenguaje, memoria persistente, herramientas, supervisión y tareas verificables. Puede incorporar temporalmente un modelo existente; debe identificar claramente esa dependencia.

El primer producto puede ser útil antes de demostrar una arquitectura general nueva. Si necesita un modelo externo, su desempeño no se atribuirá a las células. La sustitución progresiva de ese modelo permanece como línea de investigación, sin una fecha garantizada. Tampoco supondremos que recordar hechos equivale a razonar, ni que coordinar agentes demuestra inteligencia emergente.

**Primera experiencia objetivo:** abrir la aplicación, dar una tarea de software o documentos, revisar el plan y los resultados, corregir una decisión, cerrar y reabrir conservando el estado, y comprobar qué aprendió y cuánto recurso utilizó.

## 2. Punto de partida real

| Elemento | Estado al inicio | Evidencia / límite |
|---|---|---|
| Motor de forrajeo local | Implementado | Células con sensores, energía simulada y actualización propia |
| Laboratorio visual local y consola | Implementados | Controles conectados al motor Python; revisión visual real pendiente |
| Pruebas automatizadas | 22 aprobadas en la ejecución anterior | Comportamiento, localidad parcial, sesiones y HTTP; no prueba científica general |
| Evaluación comparativa | 50 ejecuciones exploratorias | 10 semillas × 5 variantes; 128 células; 160 pasos; desplazamiento de recursos en el paso 80 |
| Supervivencia media | Azar 85.70%; congelado 85.39%; aprendizaje 92.73%; sin comunicación 92.81%; regla sencilla 100% | El control sencillo gana; no hay ventaja colectiva clara en este escenario |
| Búsqueda de vecinos | Cuadrática actualmente | Obstáculo de escalabilidad del simulador |
| Crecimiento, retención secuencial y recuperación automatizada | Pendientes | No importar como evidencia actual resultados de experimentos históricos distintos |
| Lenguaje, herramientas y agentes | Pendientes | Ninguna capacidad conversacional actual |
| Energía eléctrica y sustratos alternativos | No demostrados | Unidades simuladas y tiempo de CPU no son joules físicos |

Documentos base: `VALIDATION_AND_INTERACTION.md`, `ENGINE_V0.md`, `EXPERIMENT_PROTOCOL.md` y `results/audited/2026-10-26-SUMMARY.md`.

## 3. Reglas para avanzar

- Cada etapa entrega código, protocolo, resultados completos y una decisión escrita: avanzar, corregir o conservar como experimento negativo.
- Una característica no obtiene estatus de ventaja por estar implementada. Debe superar su ablación o una alternativa pertinente.
- Fijar métricas, presupuestos, configuración y umbrales antes de abrir el conjunto final de prueba.
- Las semillas 101–110 y los mundos ya inspeccionados son datos de desarrollo. El conjunto confirmatorio será nuevo, separado y registrado por hash.
- Igualar oportunidades de ajuste: mismo número de configuraciones o mismo presupuesto de cómputo por alternativa. Incluir entrenamiento, replay, indexación, comunicación y memoria auxiliar en el costo.
- Limitar cambios simultáneos: introducir primero memoria, después plasticidad y después crecimiento; comparar cada incorporación con la versión anterior.
- Separar mediciones del motor, del modelo de lenguaje y del sistema completo.
- No comprar hardware, contratar GPU ni abrir nuevos servicios pagados por defecto. Primero medir en el equipo disponible. Documentar cualquier necesidad posterior antes de gastar.
- No copiar investigación privada al repositorio público. Mantener procedencia y distinguir trabajo histórico, trabajo del 26 de octubre y cualquier implementación elegible para el evento. Este plan no confirma las reglas del hackathon.

## 4. Arquitectura del producto y de la investigación

| Capa | Responsabilidad | Cómo se comprueba |
|---|---|---|
| Interfaz local | Conversación, archivos, plan, progreso, correcciones y cancelar | Pruebas de usuario y pruebas de navegador reales |
| Runtime de tareas | Estado, límites, herramientas, permisos, historial y reanudación | Resultados de tareas y fallos inyectados |
| Adaptador de lenguaje | Interpretar y producir lenguaje; modelo local reemplazable | Exactitud, seguimiento de instrucciones, latencia, memoria |
| Motor celular | Aprender una función delimitada y ampliar su alcance según evidencia | Ablaciones con y sin células y comparación con reglas/modelos sencillos |
| Persistencia | Guardar memoria, versiones, checkpoints y procedencia | Reinicio, restauración, migración y borrado |
| Evaluador independiente | Calcular métricas a partir de resultados observables | No delegar la decisión de éxito únicamente a un LLM |

La selección inicial de herramientas es una hipótesis candidata para integrar el motor, no una decisión irrevocable. Antes de elegirla debe probarse que el motor aprende secuencias o asociaciones relevantes. No convertir la arquitectura celular en un adorno detrás de una aplicación convencional.

Para pocos recursos, varios agentes pueden compartir una única instancia del modelo y ejecutarse por turnos. Agregar concurrencia solo cuando una medición demuestre beneficio bajo el límite de RAM y latencia.

## 5. Etapas con entregables y puertas de salida

### E0 — Congelar la línea base y el contrato del producto

**Construir:** inventario del repositorio, etiqueta de referencia, archivo de dependencias reproducible, registro de hardware/OS, lista de afirmaciones y ficha del producto. Elegir una PC de referencia y registrar CPU, RAM disponible, almacenamiento y SO; no asumir sus especificaciones.

**Comprobar:** instalación desde cero y reproducción de las cinco variantes. Deben repetirse los resultados deterministas en un entorno fijado; en otras plataformas se declarará la tolerancia numérica. El rendimiento temporal no debe exigirse idéntico.

**Entregar:** `research/claims.yaml`, `research/hardware.json`, instrucciones de reproducción y veinte tareas iniciales de uso; esta etapa crea esos archivos, el plan no afirma que existan hoy.

**Salida:** otra instalación ejecuta el laboratorio y genera un informe válido. Especificación del producto y presupuesto aprobados como decisiones de trabajo.

**Esfuerzo estimado:** 2–4 horas.

### E1 — Auditar localidad, conservación y reproducibilidad

**Construir:** contrato explícito de información accesible por célula, flujo de mensajes y contabilidad de recursos. Separar los observadores del estado que consume la política.

**Comprobar:** perturbar células y campos fuera del radio y comprobar que no alteren el siguiente paso focal sin una vía causal local. Verificar límites periódicos, células coincidentes, radio cero, muerte, ausencia de recursos, pesos finitos, repetición y reinicio. Revisar que la creación o eliminación de recursos responda a reglas documentadas.

Una recompensa propia puede usar el cambio de energía de la célula; no debe ocultar etiquetas globales, un gradiente de toda la red ni decisiones de un LLM. La vectorización central del simulador debe distinguirse de centralización de la política.

**Salida:** pruebas de invariantes aprobadas, trazas auditables y explicación completa de cualquier señal global necesaria. Si existe coordinación global de aprendizaje, corregirla o cambiar la descripción del mecanismo.

**Esfuerzo:** 6–10 horas.

### E2 — Medir y reducir el costo del simulador

**Construir:** perfiles por componente y búsqueda espacial de vecinos mediante celdas/buckets. Conservar una versión de referencia para verificar equivalencia; no optimizar antes de tener mediciones.

**Comprobar:** tamaños 64, 128, 256, 512 y 1,024 células; ampliar solo cuando sea viable. Medir tiempo por paso, CPU, pico de RAM, mensajes y trabajo útil. Probar poblaciones dispersas y concentradas: un índice espacial puede mantener un peor caso cuadrático cuando se concentran demasiados vecinos.

**Salida propuesta:** equivalencia de resultados dentro de tolerancia registrada, sin romper localidad, y mejora medible en los tamaños objetivo. Si no mejora, conservar la versión más simple y documentar el cuello de botella.

**Esfuerzo:** 8–16 horas.

### E3 — Demostrar que hace falta aprender

**Construir:** una batería que no entregue toda la respuesta mediante un gradiente visible:

- Señal breve seguida de una decisión posterior: evaluar memoria temporal.
- Asociación entre señal y recompensa que cambia por bloques: evaluar adaptación.
- Información distribuida entre vecinos con comunicación limitada: evaluar beneficio colectivo.
- Mantener el forrajeo original como control de regresión, sin exigirle demostrar lo que no puede distinguir.

**Controles:** azar, pesos congelados, regla local ajustada, pequeño método tabular cuando proceda y modelo sencillo supervisado/recurrente si recibe la misma información y presupuesto. Incluir versiones con memoria propia sin comunicación y con comunicación sin aprendizaje.

**Diseño:** desarrollo y validación separados de prueba. Piloto para estimar variabilidad; como punto de partida, 30 semillas nuevas por comparación principal. Ajustar tamaño por variabilidad y efecto mínimo detectable antes de la evaluación final, no detener al aparecer significación favorable.

**Salida propuesta:** al menos una tarea temporal donde el tratamiento supere al control sencillo más fuerte por ≥10 puntos porcentuales de éxito y el intervalo de confianza pareado del 95% excluya cero. Para afirmar beneficio colectivo, repetir el criterio frente a la ablación sin comunicación en una tarea cooperativa. Estos umbrales son decisiones de ingeniería por congelar, no estándares científicos universales.

**Si falla:** revisar información disponible, representación temporal y regla local; dedicar como máximo dos ciclos acotados de rediseño antes de revisar la hipótesis del mecanismo. No cambiar de tarea para ocultar resultados.

**Esfuerzo:** 12–24 horas; los rediseños pueden ampliar el plazo.

### E4 — Añadir memoria y medir aprendizaje continuo

**Construir:** estados persistentes locales, escritura con presupuesto y política de consolidación. Definir qué se almacena en pesos, estados y memoria externa. Distinguir recordar un registro de aprender una habilidad.

**Comprobar:** aprender A, medir A; aprender B, medir A y B; continuar con C. Comparar sin replay, replay uniforme, replay propuesto y controles con la misma cantidad de memoria y ejemplos. Registrar el costo de conservar y reproducir experiencias. Incluir ejemplos nuevos y cambios de reglas, no solo claves ya vistas.

**Métricas:** desempeño por tarea antes/después, olvido en puntos porcentuales, transferencia, bytes de memoria, actualizaciones, tiempo y procedencia de ejemplos de replay. Un replay generado externamente requiere control del costo y riesgo de errores.

**Salida propuesta:** pérdida en tareas anteriores ≤5 puntos, aprendizaje nuevo dentro de 5 puntos del entrenamiento aislado y mejora frente a un control con el mismo presupuesto. No aceptar congelar todo como solución de retención si impide aprender B.

**Esfuerzo:** 16–32 horas.

### E5 — Plasticidad estructural, crecimiento y daño

**Construir:** enlaces explícitos si aún no existen, reglas locales de poda y creación; después, incorporación de células bajo límites de espacio, material y energía simulada. El motor actual no posee todas estas estructuras. La contabilidad debe incluir creación, mantenimiento y mensajes.

**Experimentos separados:** conectividad fija vs. plástica; población fija vs. crecimiento; crecimiento aleatorio vs. crecimiento propuesto. Igualar presupuestos totales y comparar también con una red estática que disponga de la capacidad final desde el inicio, contabilizando su costo.

**Daño:** lesiones aleatorias y por región al 10%, 30% y 50%; control sin mecanismo de recuperación; misma intensidad y distribución por semilla. Evaluar habilidad y retención, no solamente número de sobrevivientes. Distinguir restauración desde respaldo de recuperación autónoma.

**Salida propuesta:** mejorar la curva de capacidad/costo respecto a controles, conservar el criterio de retención y recuperar ≥90% del desempeño anterior a una lesión del 30% dentro de un número de pasos fijado en el piloto. Reportar si la recuperación exige ejemplos, replay o recursos adicionales. No extrapolar a daño ilimitado.

**Esfuerzo:** 20–40 horas; puede requerir varios experimentos.

### E6 — Validar bajo consumo y escalabilidad en CPU

**Construir:** mediciones repetibles del motor y, más adelante, de toda la aplicación. Fijar modo de energía, hilos, calentamiento, alimentación de la laptop y carga de fondo.

**Comprobar:** comparar a calidad equivalente o presentar la frontera calidad/costo. Reportar instalación, carga inicial, aprendizaje, inferencia, almacenamiento, comunicación y mantenimiento. Medir latencias p50/p95, CPU, pico de RAM y rendimiento sostenido.

**Energía:** usar medición física si está disponible; indicar si corresponde al equipo completo o a componentes. Medir consumo en reposo y consumo bruto por separado; si se resta reposo, publicar también el bruto y el método. Una estimación debe llamarse estimación. Si no hay instrumento adecuado, dejar la afirmación energética pendiente.

**Salida propuesta:** ≥25% de reducción en una métrica de costo preseleccionada manteniendo calidad dentro de 2 puntos del mejor control. Solo afirmar ahorro energético si se mide energía comparable. Esta puerta no equivale a superar un LLM general.

**Esfuerzo:** 12–24 horas; una confirmación física depende del equipo disponible.

### E7 — Persistencia y SDK estable

**Construir:** API versionada para observar, actualizar, inferir, guardar y cargar; configuración declarativa; eventos; checkpoints completos; límites duros de recursos y formato seguro sin deserialización ejecutable.

**Comprobar:** cerrar/reabrir, interrupción durante escritura, archivo corrupto, versión anterior, falta de espacio y memoria al límite. Reanudar debe conservar el estado y el progreso; el informe JSON actual no sustituye un checkpoint.

**Salida:** instalación de un comando, sesión restorable, migración o rechazo claro de formatos y contratos documentados. Un proceso externo debe poder usar el SDK sin conocer detalles de la simulación.

**Esfuerzo:** 12–24 horas.

### E8 — Primera conversación usable e integración del motor

**Construir:** chat local, sesiones, entrada de archivos de texto, correcciones, registro de decisiones y adaptador de modelo reemplazable. Elegir un modelo local mediante una prueba de candidatos en la PC real; no fijar uno por popularidad. `llama.cpp` es un candidato de runtime de inferencia, no una implementación del aprendizaje celular.

**Prueba causal del motor:** mismo modelo, prompts, contexto, herramientas y presupuesto con tres variantes: sin motor celular, con una regla/memoria convencional y con motor celular. Si la función elegida es selección de herramientas, medir tanto selección como éxito final. No atribuirle al motor respuestas producidas por el LLM sin esa comparación.

**Salida funcional:** usuario puede conversar, enseñar una preferencia, verificar una acción y reabrir su sesión. **Salida de integración científica:** mejora ≥10 puntos en una tarea o ≥25% de reducción de costo con calidad equivalente, definida antes de probar. Si no la hay, el producto híbrido puede continuar como referencia funcional, pero la integración celular seguirá experimental.

**Esfuerzo:** 16–32 horas, una vez disponible el contrato del SDK.

### E9 — Un agente que ejecuta tareas verificables

**Construir:** ciclo planear→actuar→observar→verificar; herramientas locales para leer, buscar, transformar archivos de una carpeta de trabajo y ejecutar código en un entorno acotado. Añadir cancelar, límites de pasos, tiempo y memoria; mostrar errores y resultados parciales.

**Tareas de primera beta:** resumir con referencias a archivos, extraer datos a JSON/CSV, limpiar un dataset pequeño, modificar una función con pruebas, y ejecutar un procedimiento enseñado. Las transformaciones deben preservar archivos originales hasta confirmar el resultado.

**Comprobar:** tareas no usadas al ajustar prompts; éxito mediante validadores y revisión humana cuando sea necesario. Probar archivos con instrucciones engañosas, bucles, herramientas fallidas, cancelación y agotamiento de presupuesto. Confirmación explícita antes de envíos, compras, publicación o cambios externos irreversibles.

**Salida propuesta:** ≥80% de éxito en 30 tareas independientes congeladas, sin acciones externas no autorizadas en la batería, con límites y cancelación funcionando. Ese resultado describe esas tareas y no certifica seguridad universal.

**Esfuerzo:** 20–40 horas.

### E10 — Coordinación de agentes con recursos limitados

**Construir:** roles de planificador, ejecutor y verificador; cola con estado y presupuestos por tarea. Comenzar secuencialmente compartiendo el modelo; permitir dos trabajadores concurrentes solo tras comprobar la memoria real. Registrar qué contexto ve cada rol.

**Comprobar:** comparar agente único y varios agentes con igual límite total de tokens, tiempo y herramientas. Medir éxito, costo por éxito, latencia, ciclos redundantes, errores de coordinación, recuperación al fallar un rol y tareas duplicadas. Añadir una regla de decisión convencional como control del coordinador celular.

**Salida propuesta:** ≥10 puntos de mejora de éxito, o ≥25% de reducción de costo por tarea exitosa sin caída material de calidad, en una batería donde exista una razón para delegar. Si no mejora, conservar un agente como configuración por defecto y la coordinación como opción experimental.

**Esfuerzo:** 16–32 horas.

### E11 — Producto local beta y uso diario

**Construir:** instalador o paquete sencillo, onboarding, gestión de sesiones y permisos, registro de recursos, exportar/borrar datos, recuperación tras reinicio y mensajes comprensibles. Cerrar la revisión visual en navegadores reales; incluir accesibilidad básica, teclado y diseño adaptable.

**Piloto:** Orlando primero; después 3–5 personas con invitación explícita. Seleccionar tareas de software, documentos y automatización que ocurran realmente. Durante dos semanas registrar éxito validado, correcciones, tiempo total frente a hacerlo manualmente, fallos y uso repetido.

**Salida propuesta de beta:** ≥80% de éxito en tareas soportadas, instalación ≤15 minutos siguiendo instrucciones en dos equipos, reanudación fiable, ninguna pérdida de datos en pruebas de fallos y al menos tres usuarios que vuelvan a usarlo durante el piloto. No inferir demanda comercial de una demo; obtener señales de uso y disposición a pagar antes de ampliar gastos.

**Salida v1:** requisitos de beta sostenidos, guía de tareas soportadas, versión fijada, incidencias conocidas, ruta de actualización y decisión explícita sobre distribución/licencia. No activar cobros como parte automática del plan.

**Esfuerzo:** 16–32 horas, más dos semanas calendario de observación.

### E12 — Red compartida opcional

**Dependencias:** producto local estable y evidencia de que una carga concreta se beneficia de dividirse. No es requisito para tener la primera versión usable.

**Construir:** primero dos procesos locales; luego dos PCs en LAN. Protocolo de mensajes limitado, identificación/autenticación, asignación de tareas, idempotencia, checkpoints y recuperación ante desconexión. Definir qué datos, pesos o resúmenes salen de cada PC. Aprendizaje federado no se implementa por defecto: distribuir ejecución y distribuir entrenamiento son problemas distintos.

**Comprobar:** desconectar una PC, retrasar/repetir mensajes, capacidades diferentes, pérdida de red y presupuesto de comunicación. Medir energía y RAM sumadas de todos los equipos, no solo del coordinador.

**Salida propuesta:** tarea verificable termina sin pérdida ni duplicación de efectos después de una desconexión; la red aporta capacidad o rendimiento medible respecto a una PC. No declarar ahorro si únicamente traslada el costo a otra máquina.

**Esfuerzo:** 24–60 horas; ejecutar después de v1 salvo evidencia que justifique adelantarlo.

### E13 — Ampliación hacia un núcleo celular general

**Objetivo:** comprobar hasta dónde puede extenderse el mecanismo original, conservando la visión de una IA general de bajo consumo.

Secuencia experimental: asociaciones y memoria temporal → secuencias simbólicas → composición de reglas y planificación pequeña → generación de texto acotada → herramientas → tareas abiertas. Cada salto necesita datos nuevos, controles y comparación de calidad/costo. Incluir un modelo diminuto entrenado con métodos convencionales al mismo presupuesto de datos y cómputo.

Distinguir entrenamiento desde cero, distilación, embeddings de otro modelo y conocimiento introducido por un profesor externo. Contabilizar esas dependencias. Sustituir una función del LLM solo si el núcleo celular iguala su desempeño en el alcance definido y mejora un criterio relevante.

**Salida:** una capacidad más amplia demostrada reproduciblemente. Repetir por capacidad. La afirmación de igualar o superar una IA de propósito general requiere un programa de evaluación mucho más amplio y no tiene plazo ni viabilidad confirmados.

**Sustratos alternativos:** formalizar operaciones locales, precisión, sincronización, ruido y flujos físicos antes de seleccionar hardware. Un diseño matemático compatible no prueba una implementación química, óptica o neuromórfica. Esa línea requiere prototipo físico y medición propia, después de justificarla con resultados digitales.

**Esfuerzo:** investigación abierta; sin fecha de entrega responsable.

## 6. Definición provisional de producto usable

Un usuario puede instalarlo, conversar, asignar una tarea soportada, ver y detener el trabajo, revisar un resultado correcto, corregirlo y continuar después de reiniciar. El sistema declara si una capacidad depende del modelo existente o del motor celular.

Objetivos iniciales de ingeniería para negociar con el hardware real en E0:

| Dimensión | Objetivo inicial | Cómo se valida |
|---|---|---|
| Equipo de referencia | PC convencional, CPU, 16 GB de RAM; sin GPU obligatoria | Inventario real; ajustar antes de congelar pruebas |
| Memoria de la aplicación completa | Pico ≤8 GB en tareas soportadas | Incluir modelo, contexto, agentes y procesos auxiliares |
| Reposo | CPU media <5% tras estabilizar, sin trabajos programados | Ventana de 5 minutos en el equipo registrado |
| Respuesta | Primera señal visible <2 s; primer contenido útil p95 <15 s en prompts cortos fijados | Carga inicial reportada aparte; no confundir spinner con respuesta útil |
| Tarea breve | p95 <120 s bajo presupuesto acordado | Batería congelada y resultado correcto |
| Aprendizaje personal incremental | Sesión ≤15 min y dentro del límite de RAM | No equivale a entrenar desde cero un modelo general |
| Uso sin conexión | Tareas locales soportadas funcionan offline | Bloqueo de red en prueba; distinguir herramientas web opcionales |
| Éxito | ≥80% en batería soportada | Validadores y revisión de casos ambiguos |
| Persistencia | Reanudar sin perder cambios confirmados | Reinicio e interrupción de escritura |
| Control | Cancelación, límites y registro de acciones | Fallos inyectados y tareas largas |

Estos son objetivos de diseño, no promesas de rendimiento ya verificadas. Si un modelo no cabe o no alcanza latencia, reducir contexto/modelo/capacidades y publicar la restricción. La frase «entrenable en cualquier PC» se reemplazará por una especificación de hardware y de qué componente se entrena.

## 7. Protocolo común y fórmulas

Cada experimento registra: identificador, pregunta, hipótesis nula, tratamiento/control, información permitida, datos de entrenamiento/validación/prueba, semillas, presupuesto, hiperparámetros, commit, dependencias, hardware, métrica primaria, efecto mínimo, análisis, errores y decisión final.

- **Éxito:** tareas correctas / tareas intentadas; los fallos, timeouts y agotamientos de presupuesto cuentan en el denominador.
- **Olvido de A:** puntuación de A después de aprender A menos puntuación de A después de aprender B. Publicar también ambas puntuaciones.
- **Recuperación:** desempeño posterior / desempeño inmediatamente anterior al daño, junto con pasos, recursos y ejemplos usados. Tratar por separado casos con referencia cero.
- **Costo por éxito:** costo total de todos los intentos / número de tareas exitosas. Si no hay éxitos, declarar que no puede calcularse una ventaja; no excluir los fallos.
- **Ahorro relativo:** 1 − costo del tratamiento / costo del control, únicamente para calidad comparable y costo de referencia positivo.
- **Capacidad:** número/dificultad de tareas o asociaciones que mantiene por encima de un umbral fijado; no contar células como capacidad por definición.

La unidad estadística será la semilla/mundo o tarea independiente, no cada paso de una misma trayectoria. Publicar diferencias pareadas e intervalos del 95%, además de medias y dispersión. No mezclar resultados de distintas tareas en una media sin explicar ponderaciones. Si se prueban varias hipótesis principales, fijar corrección o jerarquía de pruebas; el resto será exploratorio.

Reservar un conjunto final no visto y una reproducción independiente. Cuando una prueba se use para tomar decisiones, deja de ser prueba final y pasa a desarrollo. Registrar todos los ensayos, no solo los mejores. Separar prueba de corrección de software de validación de una hipótesis científica.

## 8. Secuencia, calendario y carga sostenible

Estimar por horas de trabajo efectivo, no por velocidad de generación de código. Los rangos anteriores son presupuestos iniciales y pueden cambiar cuando se conozcan resultados; no garantizan resolver mecanismos abiertos.

| Horizonte | Trabajo prioritario | Entregable |
|---|---|---|
| Próximas 1–2 sesiones | E0 y comienzo de E1 | Línea base reproducible y contrato verificable |
| Siguiente bloque | E2 y E3 | Simulador perfilado y primera tarea que exige aprender |
| Investigación intermedia | E4, E5 y E6 | Evidencia de memoria, crecimiento, recuperación y costo, o resultados negativos documentados |
| Ingeniería del producto | E7, E8 y E9 | Asistente local persistente con una tarea útil y un agente |
| Ampliación y piloto | E10 y E11 | Coordinación justificada y producto beta/v1 |
| Después de v1 | E12 y E13 | Red compartida y ampliación del núcleo según evidencia |

E0–E11 suman aproximadamente **156–310 horas** según los rangos de esta versión; verificar y revisar la suma al modificar etapas. A 10–12 horas semanales representa aproximadamente **13–31 semanas**, más esperas, rediseños y observación del piloto. No reservar ese tiempo como compromiso rígido: E3–E6 contienen incertidumbre científica y pueden requerir volver atrás.

Para obtener interacción útil antes, E7–E9 pueden avanzar en una vía de ingeniería con un modelo existente mientras E4–E6 siguen en investigación, una vez definidos sus contratos. Esa vía entrega una referencia usable; no permite declarar completadas las puertas científicas pendientes. Mantener como máximo una tarea de investigación y una de producto activas; no implica agentes autónomos trabajando en paralelo.

El hackathon es una entrega acotada dentro de este programa. Seleccionar una demostración honesta, verificar reglas y preservar procedencia; no prometer completar todas las etapas para el 4 de noviembre. Toda integración Meta debe tener una función evaluable y no dirigir ocultamente decisiones locales si se afirma aprendizaje descentralizado.

## 9. Riesgos y decisiones predefinidas

| Hallazgo | Respuesta |
|---|---|
| La regla sencilla sigue ganando | Revisar si la tarea necesita memoria y aprendizaje; comparar de nuevo sin ocultar el control |
| Comunicación no aporta | Mantenerla como mecanismo experimental y no pagar su costo en el producto por defecto |
| Memoria externa explica toda la mejora | Informar una mejora de recuperación/memoria; no atribuir aprendizaje interno no medido |
| El crecimiento solo agrega parámetros | Comparar con capacidad estática y costos acumulados antes de afirmar ventaja |
| Replay evita olvido pero cuesta demasiado | Reducir/reasignar presupuesto y reportar la frontera, sin excluir su costo |
| Varios agentes empeoran costo o calidad | Un agente como configuración estable; delegación solo en tareas donde ayude |
| El producto funciona gracias al LLM | Describirlo como híbrido y mantener el motor en investigación hasta demostrar aporte |
| No hay instrumento energético | Publicar RAM/CPU/tiempo, dejar joules pendientes |
| Se incumple el presupuesto de PC | Reducir alcance o elegir otra configuración con requisitos explícitos |
| No se demuestra ventaja tras dos ciclos de E3 | Revisión escrita del mecanismo y del experimento; conservar resultados y visión, decidir el siguiente experimento antes de seguir ampliando |

## 10. Organización del trabajo y definición de terminado

Ruta sugerida, por crear a medida que se necesite: `research/` para protocolos e hipótesis; `experiments/` para ejecuciones; `results/audited/` para informes; `src/emergent_cells/` para el motor y SDK; `src/product/` para runtime/interfaz; `tests/` para invariantes e integración; `docs/` para decisiones y operación. No reestructurar el repositorio completo antes de necesitarlo.

Cada etapa se convierte en una tarea de GitHub con objetivo, archivos afectados, criterios de aceptación, dependencias, presupuesto y comando de validación. Este plan no crea todavía esas tareas ni programa ejecuciones futuras.

Una etapa está terminada cuando: su código está versionado; las pruebas pertinentes pasan; el experimento se reproduce; los resultados negativos están incluidos; los costos y dependencias se declaran; las instrucciones permiten repetirlo; y hay una decisión de siguiente etapa. Un dashboard bonito no sustituye estos requisitos.

## 11. Backlog inmediato, en orden

1. Reproducir el commit base en la PC real y completar inventario de hardware.
2. Cerrar revisión visual real del laboratorio y comprobar instalación/reinicio.
3. Fijar el protocolo de localidad y agregar las invariantes que falten.
4. Perfilar búsqueda de vecinos y comprobar una alternativa espacial contra referencia.
5. Especificar una tarea de señal retardada y sus controles antes de implementarla.
6. Ejecutar piloto, congelar umbrales y reservar nuevas semillas de prueba.
7. Tomar la decisión de E3 antes de atribuir al motor memoria útil, lenguaje o capacidad de agentes.

## 12. Referencias técnicas y cómo se utilizarán

Estas referencias orientan controles e integración; no prueban la viabilidad del proyecto ni sustituyen nuestros experimentos. Consultadas el 26 de octubre de 2026.

- [Rigging the Lottery / RigL](https://proceedings.mlr.press/v119/evci20a.html): antecedente de entrenamiento disperso con cambios de topología; ayuda a delimitar qué sería nuevo.
- [Titans: Learning to Memorize at Test Time](https://research.google/pubs/titans-learning-to-memorize-at-test-time/): antecedente de memoria que aprende durante el uso; no asumir que nuestro mecanismo es equivalente.
- [Building effective agents](https://www.anthropic.com/research/building-effective-agents): referencia conceptual para distinguir flujos y agentes y evaluar su costo. El artículo advierte que sus herramientas han evolucionado; no fija nuestra implementación.
- [llama.cpp](https://github.com/ggml-org/llama.cpp): candidato de inferencia local para el adaptador; verificar soporte, licencia del modelo y desempeño concreto al seleccionarlo.

**Decisión central:** producir evidencia y una aplicación usable sin dar por resuelta la investigación. La visión completa se conserva; las afirmaciones se amplían únicamente cuando las mediciones las sostengan.
