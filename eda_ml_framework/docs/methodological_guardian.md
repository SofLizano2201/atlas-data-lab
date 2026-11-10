# Guardia metodológica

## Auditoría del framework

| Riesgo | Implementación inspeccionada y reutilizada |
|---|---|
| Datos vacíos, esquema y límites | `ExploratoryDataAnalysis._validate`, al importar |
| Ausentes, duplicados, constantes | `profile`, `missing_values`, `constant_columns` |
| Tipos, categorías, extremos, redundancia | EDA: `feature_roles`, `outliers`, correlaciones y VIF; disponibles, no convertidos automáticamente en errores |
| Infinitos | `validate_features`; el servicio también los identifica como hallazgo estructurado |
| Fuga por ajuste | `SupervisedLearning.train`: división antes de `Pipeline`, ajuste del preprocesador dentro de cada fold |
| Clases, estratificación, folds | `SupervisedLearning.train`, `_cv`; el diagnóstico de configuración reutiliza `_cv` |
| Métricas indefinidas | `regression_metrics` conserva NaN cuando corresponde; `clustering_metrics` omite silueta fuera de su dominio |
| K, silueta | `Clustering.train`, `silhouette_plot`; servicio añade límite por observaciones únicas |
| Comparabilidad | `experiment.compare`, cohortes y procedencia; servicio añade dataset, objetivo, folds y scoring |
| Procedencia | `Experiment.record_provenance`, índices de particiones, fingerprint y configuración |
| Eventos y agentes | `WorkflowController`, `EventType`, `AgentEventMixin`; se conservan los cinco agentes y su razonamiento existente |
| Interfaz | Panel persistente `intelligence_panel`; tarjetas y registro acotado añadidos dentro del mismo panel |
| Confirmación | `ExploratoryDataAnalysis.apply(confirm=False)` y propuestas existentes; Guardian nunca transforma datos |

## Contrato y ejecución

`MethodologicalFinding` separa evidencia, explicación, severidad y componente. El servicio del Model no necesita credenciales y no entrena algoritmos. `enforce` rechaza solo hallazgos bloqueantes. Entrenamiento supervisado y clustering ejecutan comprobaciones inmediatamente antes del ajuste y guardan hallazgos en `Experiment.details`.

El controlador mantiene hasta 30 resultados en caché por identidad, versión, contenido, configuración y experimentos; el historial conserva 100 revisiones. Los eventos existentes y `VALIDATION_CHANGED` actualizan el contexto sin entrenar. Los cinco agentes interpretan el mismo contrato a través del comportamiento compartido.

## Cobertura y límites explícitos

Se detectan ausentes, constantes, duplicados, objetivo incluido o inexistente, objetivo constante, tipos incompatibles, desbalance (clase menor al 10%, heurística), dominio de MAPE cerca de cero, infinitos, K inválido, distancia incompatible con K-Means y silueta degenerada. Las validaciones previas de clases, particiones y CV siguen activas.

La API acepta `train`, `test`, `metadata` y `labels`. Metadatos explícitos `preprocessing_fit_scope`, `feature_selection_fit_scope` y `resampling_fit_scope` con valor `full_dataset` bloquean evaluación contaminada. `group_column` permite advertir entidades compartidas; `time_column` junto con `future_prediction=True` bloquea solapamiento temporal. `target_derived_features` y `post_outcome_features` solo se interpretan si fueron declarados. Estos metadatos no se infieren por nombres ni se inventan. El flujo actual de entrenamiento no recoge automáticamente esos metadatos de dominio.

La igualdad de filas entre particiones genera advertencia, no bloqueo: puede ser legítima. No se garantiza detección semántica de fugas. No se diagnostican automáticamente unidades, supuestos de residuos, estabilidad entre semillas ni disponibilidad real de una variable en producción. La corrección disponible excluye el objetivo de los predictores mediante el controlador de configuración existente y requiere confirmación explícita. El botón Revisar muestra orientación; todavía no enfoca un control específico. El historial es de sesión. AppTest verifica el renderizado y la continuación de avisos; la inspección visual en navegador sigue pendiente.

Pruebas: `python -m pytest tests/test_methodological_validation.py tests/test_models.py tests/test_agent_events.py tests/test_agent_integration.py`.
