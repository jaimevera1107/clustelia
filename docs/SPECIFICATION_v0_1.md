# Especificación metodológica y de API — librería de clustering automático

Versión del documento: 0.1. Fecha: 30 de octubre de 2026.

Estado: diseño inicial para implementación y validación. No constituye software implementado ni evidencia de rendimiento. El nombre del paquete queda pendiente; `package` en los ejemplos es un marcador. El notebook `clustering_models.ipynb` es el antecedente funcional, no la arquitectura que se conservará.

## 1. Propósito y alcance

Construir una librería Python que explore soluciones de clustering bajo objetivos explícitos, seleccione candidatos compatibles y entregue evidencia de calidad, cobertura, estabilidad, sensibilidad y limitaciones. El modo automático debe usar el mismo motor que el modo experto.

El alcance acordado cubre núcleo conceptual y diferenciación: datos tabulares numéricos, categóricos y mixtos; representaciones y geometrías explícitas; familias de modelos complementarias; restricciones entre observaciones; multivista; consenso; soluciones alternativas e interpretación. La ejecución inicial será local y en CPU, sin servicios externos obligatorios.

Quedan para extensiones posteriores: series temporales como objetos, seguimiento de grupos entre períodos, restricciones espaciales, comunidades en grafos como dominio propio, FLASC, deep clustering, transporte óptimo y asistencia mediante LLM. Los embeddings precalculados sí pueden recibirse como matrices numéricas si el usuario declara su geometría.

La librería no promete recuperar una partición verdadera universal, demostrar ausencia de clusters ni producir probabilidades universales de corrección. Una recomendación expresa preferencia bajo un contrato de análisis, un presupuesto y una política versionada.

## 2. Vocabulario y unidades de análisis

| Concepto | Definición contractual |
|---|---|
| Observación | Entidad con identificador estable; la fila no sustituye su identidad en multivista o restricciones. |
| Vista | Bloque de variables que describe las mismas entidades; puede tener un esquema propio. |
| Representación | Transformación ajustada de la entrada, con parámetros y semántica identificables. |
| Geometría de referencia | Disimilitud y ponderación fijadas para evaluar un objetivo. |
| Candidato | Especificación completa de representación, algoritmo, hiperparámetros y semillas. |
| Ajuste | Ejecución de un candidato; réplicas distintas no son candidatos científicos distintos. |
| Objetivo | Contrato que fija significado de similitud, restricciones y criterios de admisibilidad. |
| Partición | Etiquetas excluyentes, con ruido posible. Los enteros de las etiquetas no tienen orden. |
| Pertenencias | Matriz con semántica declarada: posterior de mezcla, pertenencia difusa o fuerza de asociación. |
| Alternativa | Solución defendible que no es redundante respecto a las ya presentadas. |
| Consenso | Nueva solución derivada del acuerdo entre particiones, evaluada como candidato propio. |

Una partición plana puede representar un corte de una jerarquía; la jerarquía completa se conserva cuando el algoritmo la proporciona. Grupos pequeños y grupos desbalanceados no son inválidos por definición.

## 3. Contrato de entrada

### 3.1 Formatos

- `DataFrame` con nombres de columnas únicos, o matriz NumPy bidimensional con esquema explícito cuando sea necesario.
- Matrices dispersas exclusivamente para adaptadores y transformaciones que declaren soporte; nunca densificar silenciosamente.
- Multivista: mapa nombre → tabla/matriz, alineado por identificador. Un índice ausente exige declarar explícitamente que el orden ya está alineado.
- Disimilitud precomputada mediante objeto específico, separada de una matriz de características. Se validan dimensiones, finitud, no negatividad, simetría y diagonal con tolerancias declaradas.
- Afinidades y distancias son tipos diferentes. Una conversión requiere política explícita.

### 3.2 Esquema y preprocesamiento

`Schema` declara columnas numéricas, nominales, ordinales y binarias; las ordinales requieren orden explícito. Los enteros no se interpretan automáticamente como continuos si el usuario ha declarado otro tipo. Se pueden sugerir tipos, pero el informe debe identificar las inferencias.

Identificadores, categorías de alta cardinalidad, columnas constantes y posibles fugas de información generan diagnósticos. Los identificadores declarados se excluyen del ajuste; una sospecha no autoriza exclusión silenciosa. Las columnas constantes se excluyen con registro y se preservan en el esquema original.

Faltantes: política por defecto `error`. La imputación debe activarse y configurarse, ajustarse dentro de cada réplica y conservar indicadores si se solicitan. Imputación, normalización y reducción nunca se ajustan usando etiquetas externas de validación.

La entrada no se modifica. El orden de filas, nombres, tipos, transformaciones y exclusiones se registran. En predicción se rechazan columnas faltantes o adicionales por defecto; una reordenación por nombres solo se permite con comprobación completa y registro. Categorías nuevas requieren una política declarada del adaptador.

No habrá soporte implícito de pesos muestrales. Si un objetivo solicita pesos, todos los componentes activos deben soportar la semántica declarada; en caso contrario se rechaza o se excluye el candidato con motivo. Pesos de filas y pesos de variables son conceptos distintos.

### 3.3 Multivista incompleta

El primer adaptador multivista exige entidades completas en todas las vistas. La interfaz representa máscaras de disponibilidad para evolucionar sin cambiar identidades. El soporte de vistas incompletas forma parte del alcance de diferenciación, pero se habilita solo en adaptadores con denominadores, normalización y validación propios; el resto produce un error de capacidad, sin rellenar vistas enteras de forma automática.

## 4. Objetivos y geometrías

Se distinguirán `compact`, `density`, `probabilistic` y `consensus`; una jerarquía puede ofrecer candidatos a varias resoluciones dentro del objetivo compatible. Un objetivo personalizado puede declarar sus métricas y una política de selección propia, conservando los controles de validez.

| Datos | Geometrías contempladas | Condiciones |
|---|---|---|
| Numéricos | Euclídea; Manhattan donde el algoritmo lo admita | Escalado declarado; no todas las combinaciones son admisibles. |
| Embeddings | Coseno; euclídea tras normalización cuando proceda | No equiparar objetivos sin demostrar la relación aplicable. |
| Categóricos | Disimilitud por coincidencia, con pesos explícitos | No convertir categorías en enteros ordinales arbitrarios. |
| Mixtos | Gower y geometría de KPrototypes | Tienen supuestos y ponderaciones diferentes; objetivos separados por defecto. |
| Multivista | Distancia combinada o afinidad combinada | Normalización por vista y pesos fijados o explorados de forma declarada. |

Para comparación directa, todos los candidatos de un objetivo se evalúan sobre la misma geometría de referencia y las mismas observaciones de evaluación, salvo restricciones documentadas de ruido. También se guardan métricas en el espacio de ajuste como diagnósticos secundarios.

Buscar escaladores o ponderaciones puede cambiar la pregunta. El modo automático fija por defecto un contrato de referencia y explora representaciones de ajuste dentro de él. `geometry_search=True` genera objetivos alternativos, sin fabricar una puntuación universal entre ellos.

La ruta predeterminada numérica propone escalado estándar y debe reportar su efecto; el usuario puede elegir robusto, sin escalado o pesos expertos. Esta es una decisión de ingeniería transparente, no una elección neutral. Para datos mixtos, el modo automático propone Gower con pesos iguales por variable original; el rango, tratamiento ordinal, constantes y faltantes se documentan en `GeometrySpec`. KPrototypes se presenta como alternativa bajo su propia geometría cuando corresponda.

Se incluye siempre representación sin reducción. PCA puede conservar una fracción de varianza declarada; no se fuerza a dos dimensiones. UMAP, si se habilita, es opcional, con parámetros y semillas explícitos. Las proyecciones gráficas nunca sustituyen automáticamente la geometría de evaluación.

## 5. Matriz de modelos y capacidades

Esta matriz describe el soporte previsto por la librería, no todas las capacidades posibles de cada implementación externa. Cada adaptador publicará una ficha verificable con backend, versiones y capacidades realmente implementadas.

| Adaptador | Entrada prevista | Resultado principal | Nuevas observaciones | Restricciones iniciales |
|---|---|---|---|---|
| KMeans / MiniBatchKMeans | Numérica euclídea | Partición, centroides | Nativa | Sin restricciones; adaptador específico separado |
| KMedoids | Características o disimilitud compatible | Partición, medoids | Por distancias a medoids si están disponibles | Sin restricciones inicialmente |
| GaussianMixture | Numérica euclídea | Componentes y posteriores | Nativa | Sin restricciones |
| BayesianGaussianMixture | Numérica euclídea | Componentes y posteriores | Nativa | Sin restricciones |
| DBSCAN | Características o disimilitud compatible | Partición y ruido | No nativa en el contrato inicial | Sin restricciones |
| HDBSCAN | Entrada admitida por backend | Partición, ruido y jerarquía de densidad | Solo extensión explícita de backend | Sin restricciones |
| OPTICS | Entrada admitida por backend | Orden de alcanzabilidad y cortes | No nativa | Sin restricciones |
| Aglomerativo | Características o disimilitud según enlace | Jerarquía y cortes | No nativa | Conectividad solo cuando se implemente y declare |
| BisectingKMeans | Numérica euclídea | Partición y estructura de bisección | Nativa si se conserva en backend | Sin restricciones |
| Spectral | Características o afinidades admitidas | Partición | No nativa inicialmente | Sin restricciones |
| Fuzzy C-Means | Numérica euclídea | Pertenencias difusas y prototipos | Con prototipos ajustados | Sin restricciones |
| KModes | Categórica | Partición y modos | Nativa según adaptador | Sin restricciones |
| KPrototypes | Mixta | Partición y prototipos | Nativa según adaptador | Sin restricciones |
| BIRCH | Numérica euclídea | Resumen y partición | Según adaptador; actualización separada | Sin restricciones |
| ConstrainedMedoids | Disimilitud declarada | Partición con diagnóstico de restricciones | Asignación separada sin prometer restricciones sobre casos nuevos | Duras y suaves según modo |
| Consensus | Conjunto de particiones compatibles | Partición de consenso | No nativa | No hereda restricciones sin verificarlas |

Ward se limita a condiciones euclídeas compatibles. No aplicar centros o CH/DB a una disimilitud arbitraria. Especificar parámetros de backend explícitamente, evitando depender de valores por defecto que cambien entre versiones. La implementación de HDBSCAN queda fijada en cada adaptador: no tratar backends como intercambiables, especialmente por convenciones de `min_samples`.

Las probabilidades de mezclas, la pertenencia difusa y las fuerzas de asociación se almacenan como tipos semánticos diferentes. No habrá `predict_proba` genérico en el selector si su significado depende del ganador.

## 6. Restricciones

`ConstraintSet` contiene pares de identificadores, relación `must_link` o `cannot_link`, modo `hard` o `soft` y peso no negativo para preferencias. Los pares se canonicalizan y deduplican. Se rechazan identificadores desconocidos, auto-relaciones incompatibles y pesos inválidos.

Para restricciones duras se calcula el cierre de componentes must-link y se detectan cannot-link dentro de un componente. Esta prueba encuentra contradicciones evidentes, pero no certifica factibilidad para cualquier número de grupos. El adaptador comprueba además la factibilidad relevante a su algoritmo; un fallo heurístico no se etiqueta como imposibilidad matemática demostrada.

En modo duro, un candidato con alguna violación no es admisible. Se publica el número de pares verificables, violaciones y cobertura. En modo suave se publican coste ponderado y restricciones satisfechas; no se suman automáticamente a silhouette con unidades incompatibles. La política fija un límite aceptable o una prioridad lexicográfica.

El ruido no puede usarse para satisfacer restricciones triviales: por defecto, los extremos de pares duros deben quedar asignados a grupos válidos. Una política distinta debe activarse expresamente.

No se promete envolver cualquier algoritmo para que satisfaga restricciones. El adaptador inicial previsto es ConstrainedMedoids: grupos must-link contraídos, asignación con cannot-link, verificación final y registro de intentos. Su diseño algorítmico tendrá una revisión específica antes de implementarse. La factibilidad y el valor del objetivo se prueban exhaustivamente en casos pequeños. El clustering activo se reserva como evolución de esta interfaz, sin ser condición de la primera entrega.

## 7. Multivista y consenso

### 7.1 Multivista

Dos rutas iniciales:

1. Fusión temprana de bloques numéricos transformados: solo bajo normalización por bloque que haga visible su contribución; se publica el número de variables y el peso efectivo.
2. Fusión de disimilitudes o afinidades: cada vista especifica escala y normalización; las vistas de distancia y de afinidad no se suman entre sí. Para distancias, una regla inicial es suma ponderada de matrices normalizadas por una estadística positiva de referencia; se registra la muestra usada para estimarla. Una vista degenerada genera diagnóstico y requiere exclusión explícita o política registrada.

Los pesos por defecto son iguales por vista después de normalización, no iguales por columna concatenada. La búsqueda de pesos produce alternativas de objetivos. Se reporta sensibilidad leave-one-view-out y acuerdo de la solución con cada vista; una vista conflictiva no se elimina silenciosamente.

Vistas incompletas requieren algoritmos específicos: combinar solo vistas observadas por par puede generar disimilitudes no métricas. Esa salida se marca como disimilitud general y se restringen adaptadores; no se ofrece como distancia métrica garantizada.

### 7.2 Consenso

El consenso se calcula entre particiones sobre las mismas entidades y dentro de un objetivo o conjunto de objetivos autorizado. La entrada se deduplica para evitar que una familia con muchos hiperparámetros domine. Por defecto se conserva un representante por grupo de particiones casi equivalentes y se ponderan familias explícitamente.

Para cada par, el acuerdo considera únicamente ejecuciones donde ambos casos estén asignados; se conserva el denominador y no se interpreta la coincidencia de ruido como evidencia de cluster. La afinidad por pares con denominadores distintos no se declara kernel positivo semidefinido. Un par sin soporte se marca desconocido: el adaptador debe declarar cómo lo trata o rechazar la ejecución.

La matriz completa se usa solo si cabe en presupuesto. Para conjuntos grandes se requieren consensos aproximados por anclas o vecinos con diagnóstico de cobertura. El consenso es un candidato adicional: debe reevaluarse y no hereda ni la calidad ni las restricciones de sus miembros.

## 8. Protocolo de evaluación

### 8.1 Validez y ruido

- Contar clusters excluyendo etiquetas especiales de ruido o inválidos.
- Particiones con menos de dos grupos o tantos grupos como observaciones evaluables no reciben métricas internas que requieran otro dominio.
- Un resultado de un solo grupo puede conservarse como diagnóstico, sin inventar una silhouette.
- Métricas no aplicables se registran como ausentes con motivo, no como cero.
- Reportar cobertura, ruido, tamaños, mínimo y máximo; no penalizar desbalance sin un requisito explícito.
- Calcular métricas de grupos sobre casos asignados y publicar el conjunto utilizado. Al variar el ruido, la métrica condicionada usa poblaciones distintas: no presentarla como comparación perfectamente controlada. El umbral de cobertura y el diagnóstico en soporte común deben acompañar el ranking.

### 8.2 Métricas por objetivo

| Objetivo | Evidencia principal | Diagnósticos adicionales |
|---|---|---|
| Compacto | Silhouette en referencia, estabilidad y cobertura | CH y DB solo en espacios compatibles; tamaños y representantes |
| Densidad | Validez de densidad compatible, cobertura y estabilidad | Silhouette secundaria; persistencia del backend con su significado propio |
| Probabilístico | Criterio de información o log-verosimilitud de validación comparable | Convergencia, degeneración, estabilidad y separación |
| Difuso | Objetivo y diagnósticos de partición difusa compatibles | Entropía, sensibilidad a fuzzifier y evaluación de partición inducida |
| Consenso | Soporte de coasignación, estabilidad y calidad en referencia | Diversidad y contribución de miembros |

Para densidad se prevé DBCV cuando su definición, distancia y backend sean compatibles. Si falta una métrica primaria, no se sustituye silenciosamente: se emite una alternativa parcial o se solicita una política explícita. Para mezclas, BIC se compara sobre la misma entrada y número de observaciones; cambiar representación invalida comparaciones ingenuas. Una mezcla puede usar varios componentes para representar un grupo: `n_components` no se presenta como número demostrado de subpoblaciones.

La estabilidad de persistencia de HDBSCAN y la estabilidad por remuestreo son evidencias diferentes. La estabilidad por sí sola no valida utilidad ni estructura real.

### 8.3 Muestreo de métricas

Las métricas costosas usan una muestra reproducible común dentro del objetivo, seleccionada antes de mirar etiquetas candidatas. Si quedan pocos grupos o casos asignados, la métrica es no evaluable y se registra. Se puede añadir un diagnóstico estratificado por cluster, claramente separado del valor usado para selección. Se conservan tamaños, índices o identificadores de muestra y semillas.

### 8.4 Estabilidad

Dos ejes obligatorios para candidatos finalistas: aleatoriedad y composición muestral. Las perturbaciones de variables y de hiperparámetros cercanos son diagnósticos adicionales de diferenciación.

Cada réplica reajusta imputación, escalado, reducción y modelo. Los ajustes se comparan sobre intersecciones de identificadores. Se reportan ARI sobre asignados comunes, tamaño de la intersección y acuerdo de asignación/ruido por separado. ARI no exige alinear los números de etiquetas; los perfiles por grupo usan correspondencias por solapamiento y dejan visibles divisiones/fusiones.

Las comparaciones con menos de dos grupos comunes, soporte insuficiente o varianza degenerada no aportan evidencia positiva. No se convierte una coincidencia trivial en máxima estabilidad. El umbral mínimo de soporte se configura y se registra.

Reportar mediana, cuantiles descriptivos y fracción de réplicas evaluables; no llamar a estos cuantiles intervalos de confianza sin un procedimiento inferencial específico. En datos dependientes se exige una estrategia de remuestreo por bloques o grupos proporcionada; el remuestreo independiente solo se asume para datos adecuados a esa hipótesis.

## 9. Búsqueda y selección: política inicial P0

La política P0 es una regla de ingeniería explícita y provisional, pendiente de benchmark; sus umbrales no son significancias ni niveles de confianza. No se publica una versión estable mientras estos parámetros no estén evaluados y documentados.

### 9.1 Planificación

1. Resolver esquema, objetivo, geometría, restricciones y capacidades.
2. Materializar un plan determinista de candidatos y semillas; un cambio de presupuesto puede cambiar el plan y debe registrarse.
3. Explorar con presupuesto y métricas baratas. `max_fits` cuenta todo ajuste, incluidas réplicas y confirmación; `max_candidates` cuenta especificaciones únicas.
4. Reservar de antemano presupuesto de confirmación. Si no alcanza, devolver `incomplete`, sin presentar exploración como confirmada.
5. Seleccionar una lista corta diversa de hasta cinco candidatos por objetivo, evitando variantes casi idénticas cuando sea posible.
6. Confirmar la lista fija con semillas y submuestras nuevas; no generar hiperparámetros tras observar esa confirmación en la misma ejecución.

Propuesta inicial de confirmación para calibrar: diez submuestras sin reemplazo del 80% y tres semillas del ajuste completo cuando el algoritmo sea estocástico. Se conservan el coste y el número realmente completado. Esto no elimina el sesgo de selección ni garantiza detección de grupos reales.

### 9.2 Admisibilidad

Se comprueban éxito y convergencia relevante, finitud, dominio de métricas, restricciones duras, límites de grupos definidos por el usuario, cobertura y soporte evaluable. Los umbrales no universales se publican en `SelectionPolicy`.

Valores iniciales propuestos para P0: cobertura ≥ 0.80 cuando se admite ruido; al menos 80% de réplicas evaluables; mediana ARI ≥ 0.70 como filtro operativo de estabilidad. No se impone igualdad de tamaños ni silhouette positiva universal. El usuario puede cambiar o desactivar estos filtros; el informe registra la política efectiva y advierte su carácter no calibrado en la etapa experimental.

El soporte mínimo de una comparación de estabilidad es `max(20, 2*k)` casos comunes asignados, donde `k` es el mayor número de grupos representados en esa comparación. Para muestras menores se requiere una política de pequeño tamaño específica; no reducir el umbral silenciosamente.

### 9.3 Preferencia y alternativas

Se forma una frontera no dominada dentro de cada objetivo con calidad primaria orientada correctamente, estabilidad y cobertura. La pertenencia a esta frontera no es prueba estadística de superioridad.

En el objetivo compacto, P0 prioriza calidad primaria entre candidatos admisibles; diferencias de silhouette ≤ 0.02 se tratan como equivalencia práctica provisional y se desempatan por estabilidad, cobertura, coste y `candidate_id` en ese orden. En densidad se usa una tolerancia propia de su métrica, nunca se hereda 0.02 automáticamente. En probabilístico se aplica el criterio comparable declarado. Para objetivos sin una tolerancia validada, P0 conserva alternativas y no fuerza un ganador único.

La deduplicación de alternativas usa acuerdo de particiones y similitud de conjuntos de ruido; sus tolerancias pertenecen a la política y se reportan. Los tiempos no se usan para declarar superioridad científica.

Se puede devolver un candidato preferido por la política junto con alternativas. Entre geometrías distintas se publican recomendaciones por objetivo; sin preferencia explícita no existe ganador global. Cambiar la política, la geometría o el espacio de búsqueda puede cambiar la selección.

### 9.4 Estados

| Estado | Significado |
|---|---|
| `recommended` | Hay una preferencia defendible bajo P0 dentro del objetivo y confirmación completada. |
| `ambiguous` | Varias alternativas permanecen sin preferencia suficiente o hay objetivos no ordenables. |
| `no_admissible_solution` | Ninguna solución evaluada cumple la política; no demuestra ausencia de estructura. |
| `incomplete` | El presupuesto o la cancelación impidieron completar el protocolo. |

Entrada inválida o restricciones contradictorias detectadas producen excepciones antes de la búsqueda. Si todos los ajustes fallan por errores de ejecución, se eleva `SearchFailedError` con diagnóstico; no se etiqueta como ausencia de clusters. Un resultado parcial puede contener candidatos exploratorios, pero no convertirse en recomendado sin completar su protocolo.

## 10. API pública propuesta

Los siguientes nombres son diseño propuesto, sin prometer importación o ejecución actuales.

```python
from package import AutoCluster, Schema, GeometrySpec, SelectionPolicy, SearchBudget

model = AutoCluster(
    schema=Schema(numeric=["edad", "ingreso"], categorical=["actividad"]),
    geometry=GeometrySpec(kind="gower"),
    objective="compact",
    selection=SelectionPolicy.p0(),
    budget=SearchBudget(max_fits=160, max_candidates=40, timeout_s=600),
    random_state=42,
    n_jobs=4,
)
model.fit(df)
result = model.result_
print(result.selection_status)
print(result.candidates_table())
chosen = result.select(result.recommendation.candidate_id)  # solo si existe
labels = chosen.labels_
```

El ejemplo exige comprobar `recommendation is not None` en código de producción. `select` crea una vista/objeto de solución sobre el pipeline ya ajustado, sin reentrenar. No modifica la recomendación original ni la tabla histórica.

```python
from package import ConstraintSet, MultiViewCluster

constraints = ConstraintSet(
    must_link=[("empresa_001", "empresa_009")],
    cannot_link=[("empresa_001", "empresa_020")],
    mode="hard",
)
multi = MultiViewCluster(
    view_specs=view_specs,  # esquema y geometría por vista
    fusion="distance",
    view_weights={"economica": 0.5, "laboral": 0.5},
    constraints=constraints,
    random_state=42,
)
multi.fit({"economica": df_economia, "laboral": df_laboral})
```

`AutoCluster.fit(X, y=None)` ignora `y` para compatibilidad y nunca usa etiquetas externas para seleccionar. Se informará si se proporciona `y`; la evaluación externa tiene método separado. Los objetos de configuración son clonables o compatibles con la clonación del estimador y no almacenan estado ajustado mutable compartido.

| Método/atributo | Contrato |
|---|---|
| `plan(X)` | Valida y devuelve candidatos previstos, costes orientativos y exclusiones; no hace la búsqueda. Si ajusta estadísticas de planificación, lo declara y no reutiliza estados ocultos. |
| `fit(X)` | Devuelve `self`; conserva resultado y pipelines. Un segundo fit sustituye el estado ajustado de forma explícita. |
| `fit_predict(X)` | Devuelve etiquetas solo con recomendación única; si hay ambigüedad, devuelve excepción específica con resultado accesible. |
| `result_` | Experimento con política, candidatos, evidencia y estados. |
| `best_estimator_`, `labels_` | Disponibles solo si hay recomendación única válida; si no, se consulta `result_`. |
| `results_` | Tabla de candidatos con esquema estable y métricas sin redondeo para decisiones. |
| `result_.select(id)` | Obtiene una solución ya ajustada; seleccionar manualmente no la convierte en validada. |
| `solution.assign(X_new)` | Resultado con capacidad y método explícitos; error si no soportado. |
| `solution.membership_` | Objeto que identifica semántica, etiquetas de columnas y normalización. |
| `solution.describe()` | Perfiles, representantes y diagnósticos en variables originales. |
| `result_.compare(ids)` | Compara soporte, partición, cobertura y supuestos; no inventa orden entre objetivos. |
| `result_.external_evaluate(y)` | Evaluación externa posterior; no altera selección. |
| `save` / `load` | Persistencia con entorno, esquema y política de compatibilidad. |

No habrá `score` universal destinado a optimizar ciegamente el selector con validación cruzada supervisada. La compatibilidad de estimador no implica que cualquier rutina de tuning tenga sentido estadístico.

## 11. Interpretación

Los perfiles se calculan en variables originales: tamaños, medianas y cuantiles, proporciones categóricas, diferencias respecto al conjunto y representatividad de medoids o casos cercanos. Valores ausentes y grupos pequeños deben ser visibles.

Las reglas descriptivas de un árbol sustituto se evalúan en observaciones separadas para medir fidelidad al modelo cuando la capacidad de asignación lo permite. En algoritmos transductivos, el reporte debe distinguir reproducción de etiquetas observadas de generalización a casos nuevos. Una explicación no se presenta como efecto causal ni como prueba de validez del clustering.

La ambigüedad individual puede derivarse de desacuerdo entre réplicas, frontera o pertenencias nativas; se conserva su fuente. No se promedian estas evidencias para inventar una probabilidad genérica.

## 12. Ejecución, trazabilidad y persistencia

- `timeout_s` es límite suave entre tareas por defecto; un límite duro requiere backend con procesos aislados y terminación comprobada.
- Presupuestos incluyen ajustes de búsqueda, réplicas, consenso y confirmación. El plan reserva trabajo final antes de gastar todo en exploración.
- Memoria: estimar matrices por pares, rechazar densificaciones no autorizadas y declarar límite como preventivo salvo aislamiento con control efectivo.
- Semillas derivadas de identificadores estables del candidato y fase; su asignación no depende del orden de finalización paralelo.
- Registrar advertencias y fallos por candidato; errores de programación no se silencian como incompatibilidad estadística. `error_policy="raise"` facilita depuración.
- Paralelismo externo e interno controlados para evitar sobresuscripción. Tiempo máximo y número de hilos efectivos figuran en el registro.
- Caché por representación y datos identificados, con separación de réplicas. Nunca reutilizar un transformador ajustado en una muestra distinta como si hubiera sido reajustado.
- Guardar versión de paquete, metodología, backends, dependencias, esquema y semillas. Las huellas de datos son opcionales y no contienen datos brutos por defecto; índices e identificadores persistidos son configurables.
- La persistencia no promete compatibilidad universal entre versiones. No cargar objetos serializados de origen no confiable; el formato documenta su modelo de confianza.
- El reporte de metadatos no incluye datos completos salvo solicitud explícita. Guardar un pipeline puede conservar ejemplos o prototipos: documentarlo.

## 13. Arquitectura interna

| Módulo | Responsabilidad |
|---|---|
| `schema` | Tipos, entidades y validación |
| `geometry` | Distancias, afinidades, pesos y reglas de compatibilidad |
| `representations` | Transformadores y pipelines clonables |
| `adapters` | Backends y registro de capacidades |
| `constraints` | Validación, cierre y verificación |
| `search` | Plan, presupuesto, ejecución y caché |
| `evaluation` | Métricas, estabilidad y muestreo |
| `selection` | Políticas versionadas y alternativas |
| `multiview` | Alineación y fusión |
| `consensus` | Diversidad, soporte y agregación |
| `results` | Contratos de resultados y selección manual |
| `inspection` | Perfiles, representantes y reglas |
| `persistence` | Metadatos y compatibilidad |

El registro de adaptadores debe validar capacidades, no solo mapear nombres a clases. Un candidato incompatible se excluye antes de ejecutar con motivo legible. No se necesita un sistema público de plugins hasta estabilizar estos contratos internos.

## 14. Criterios de aceptación

| Área | Evidencia exigida |
|---|---|
| Pipeline | Las etiquetas conservadas coinciden con el candidato evaluado; ninguna reducción se encadena accidentalmente. |
| Independencia | Un fallo o ajuste no modifica otro candidato ni la entrada. |
| Esquema | Errores claros en columnas, tipos, identidades y categorías incompatibles. |
| Métricas | Casos degenerados y ruido verificados; invariancia a renombrar etiquetas. |
| Restricciones | Contradicciones detectables y verificación exhaustiva de soluciones pequeñas; nunca reportar factibilidad no comprobada. |
| Multivista | Alineación por ID; pesos efectivos comprobados; errores en vistas ausentes según capacidad. |
| Consenso | Duplicar una partición redundante no aumenta arbitrariamente su influencia; soporte desconocido visible. |
| Estabilidad | Se reajusta todo el pipeline; ruido y soporte se reportan; degeneraciones no producen estabilidad ficticia. |
| Selección | Casos con candidato único, empate, ausencia de admisibles, búsqueda incompleta y fallo total. |
| Presupuesto | Contadores reales incluyen confirmación; prueba de cancelación y liberación de recursos. |
| Persistencia | Round trip bajo entorno soportado y diagnóstico de incompatibilidad. |
| Ecosistema | Comprobaciones aplicables de estimadores, instalación limpia y dependencias opcionales realmente opcionales. |

No exigir invariancia exacta al orden de filas para algoritmos que no la garantizan; sí documentar y medir sensibilidad. Una semilla no promete identidad binaria entre hardware o versiones.

Benchmarks: datos con grupos esféricos, alargados, no convexos, densidades distintas, desbalance, contaminación, columnas irrelevantes, correlación y datos sin separación clara. Incluir categóricos, mixtos, multivista concordante y conflictiva, restricciones correctas y erróneas. Separar conjuntos de desarrollo y evaluación. Etiquetas externas solo en evaluación. Comparar baselines con presupuesto equivalente, medir calidad externa cuando proceda, cobertura, abstenciones, tiempo y memoria. Hacer ablaciones de estabilidad, representación, restricciones y consenso. No prometer un porcentaje mínimo de victorias antes de ejecutar el benchmark.

## 15. Implementación por hitos dentro del alcance

1. **M1 — Contratos y núcleo numérico:** esquema, registro, pipeline conservado, KMeans, KMedoids, mezcla gaussiana, aglomerativo y densidad; evaluación y políticas; pruebas de regresión del notebook.
2. **M2 — Cobertura de familias:** MiniBatch, BIRCH, OPTICS, Bisecting, espectral, difuso y mezcla bayesiana; jerarquías y alternativas.
3. **M3 — Categóricos y mixtos:** distancias y tipos, KModes, KPrototypes, Gower/medoids y evaluación coherente.
4. **M4 — Diferenciación:** restricciones, multivista, consenso y sensibilidad; adaptadores especializados verificados.
5. **M5 — Interpretación y validación integral:** perfiles, reglas, benchmark reservado, rendimiento, documentación y distribución.

Cada hito debe funcionar por sí mismo, pero no se declara completado el alcance acordado antes de M4–M5. La primera implementación puede ser experimental; la condición estable exige calibrar P0 y demostrar los contratos.

## 16. Decisiones fijadas y cuestiones que requieren experimentos

Fijadas: selección dentro de objetivos; geometría visible; pipelines conservados; ruido separado; ausencia de ganador universal entre geometrías; alternativas; restricciones por capacidad; extensiones opcionales; ninguna promesa inferencial sin calibración.

Pendientes de evidencia, sin bloquear la arquitectura: umbrales definitivos P0, tolerancias por objetivo, número óptimo de réplicas, backend de medoids y densidad, rendimiento de ConstrainedMedoids, estrategia escalable de consenso, mecanismo especializado de vistas incompletas y compatibilidad de versiones a publicar. Estas decisiones se cierran mediante pruebas y registros de decisión, no por preferencias ocultas.

## 17. Referencias técnicas de apoyo

Las referencias apoyan familias y convenciones; no validan la política P0 propuesta en este documento.

- Scikit-learn, desarrollo de estimadores: https://scikit-learn.org/stable/developers/develop.html
- Scikit-learn, clustering y evaluación: https://scikit-learn.org/stable/modules/clustering.html
- Scikit-learn, paralelismo: https://scikit-learn.org/stable/computing/parallelism.html
- Scikit-learn, persistencia: https://scikit-learn.org/stable/model_persistence.html
- HDBSCAN, referencia de API y validez: https://hdbscan.readthedocs.io/en/latest/api.html
- UMAP, uso para clustering y advertencias: https://umap-learn.readthedocs.io/en/latest/clustering.html

## 18. Migración del notebook

El notebook queda como ejemplo histórico. Se reemplaza la clase monolítica por componentes independientes. Las primeras regresiones cubren reducción accidentalmente encadenada, discrepancia entre selección y estimación final, ruido contado como grupo, divisiones por cero en el índice compuesto, errores silenciados, tratamiento mixto incorrecto y variables no definidas de visualización.

No se conserva el índice compuesto como criterio predeterminado. Si se ofrece por compatibilidad, será diagnóstico heredado con advertencias y sin pretensión de validez universal. Los gráficos son consumidores de soluciones ya ajustadas y no modifican los datos de entrenamiento ni la selección.
