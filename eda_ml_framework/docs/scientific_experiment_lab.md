# Laboratorio científico de ATLAS

## Auditoría y reutilización

| Capacidad inspeccionada | Reutilización |
|---|---|
| `Experiment`, `record_provenance`, `runtime_versions` | Registro individual, configuración, índices, métricas, artefactos y librerías |
| `SupervisedLearning.train`, `_cv` | Hold-out coordinado por semilla y CV dentro del entrenamiento; estratificación en clasificación |
| `make_preprocessor`, `Pipeline` | Ajuste de imputación, codificación y escala dentro de cada partición; no ajuste global nuevo |
| `Classification`, `Regression` | Estimadores y métricas existentes; el laboratorio no implementa otro pipeline |
| `Clustering.train`, `clustering_metrics` | Agrupamientos, métricas internas, ruido y etiquetas sobre las mismas filas |
| `experiment.compare` | Su cohorte incluye predictores y preprocesamiento: no sirve directamente para experimentos donde precisamente cambia ese factor. El laboratorio valida el factor y las filas de prueba antes de resumir diferencias pareadas |
| `MethodologicalValidation` | Diagnóstico de objetivos, leakage, CV, métricas, presupuesto y compatibilidad |
| `DatasetVersion`, `fingerprint`, `ExploratoryDataAnalysis.df` | Referencias verificadas, copia profunda del dataset, reproducción de versión original o diseño nuevo con versión actual |
| `ExperimentService.remember` | Un único registro compartido con las pantallas existentes; investigaciones en el mismo servicio |
| `ReportService`, `html_report`, `_pdf` | Informes HTML/PDF con la infraestructura actual |
| Eventos y cinco agentes | Nuevos hechos LAB; recomendaciones de especialistas y lectura de resultados por el contrato compartido; ningún agente adicional |
| `figure`, `chart_theme` | Gráficos Plotly con el diseño actual del frontend |

Las capacidades nuevas son definiciones tipadas de investigación, control de factores, coordinación de repeticiones, diferencias pareadas, estabilidad ARI y bootstrap condicional. Los cálculos viven en `models/scientific_experiment.py`; el controlador administra confirmación y ejecución. No se ejecuta Python proporcionado por usuarios.

## Uso

1. Confirma un dataset y abre **Laboratorio científico** en el menú.
2. Elige una plantilla: algoritmos, transformación, variables, estabilidad del modelo, estabilidad de clusters, incertidumbre, sensibilidad o comparación personalizada.
3. Escribe pregunta e hipótesis; revisa alternativas, condiciones, métrica, repeticiones y presupuesto. Las hipótesis nula/alternativa son editables; no se ejecuta un test formal por el mero hecho de declararlas.
4. **Validar y guardar diseño** no entrena. Los errores bloqueantes impiden preparar la ejecución.
5. En **Ejecución y resultados**, confirma explícitamente el presupuesto. Cada paso ejecuta una evaluación por los métodos existentes. El progreso se actualiza también en Inteligencia ATLAS.
6. Revisa variabilidad, diferencias por semilla y estabilidad disponible. Las ejecuciones fallidas conservan el resto de los resultados y pueden reanudarse con una confirmación nueva.
7. Exporta el paquete ZIP. Contiene manifiesto, snapshot Parquet con tipos e índice, CSV, informe HTML y gráficos interactivos con una copia compartida de Plotly para uso sin conexión. HTML/PDF/CSV/JSON también se descargan por separado.
8. Reabre una investigación o prepara una reproducción con ID nuevo. La versión original guardada y el dataset activo son opciones explícitas. Importar el ZIP solo valida y prepara; nunca entrena ni carga código serializado.

La persistencia durante el uso es de sesión. Para conservar una investigación fuera de esa sesión hay que descargar el ZIP; no se presenta la memoria de Streamlit como almacenamiento permanente. Los modelos entrenados se registran también en el historial de experimentos existente, desde el que pueden descargarse con su mecanismo actual.

## Diseño y límites

- Una tarea, dataset, objetivo y protocolo por investigación; métricas y direcciones registradas por tarea.
- Dos a cuatro alternativas, o una alternativa con al menos dos semillas para estabilidad. Hasta cinco semillas, doce evaluaciones, 72 ajustes contando folds, 5000 filas, 30 predictores y diez investigaciones por sesión. Se respeta además el límite de 30 modelos del registro compartido.
- Máximo de 50 inicializaciones y 2000 iteraciones declaradas por ajuste; se reutilizan los límites existentes para otros parámetros. Los estimadores con búsqueda CV interna se bloquean para no subestimar el presupuesto: usa el estimador base y sensibilidad.
- Clustering con coste potencialmente cuadrático tiene un límite adicional de 2000 filas en los algoritmos correspondientes.
- Las configuraciones de algoritmo incluyen sus hiperparámetros. Para variar simultáneamente variables o preprocesamiento se requiere un diseño personalizado con justificación. Sensibilidad cambia un hiperparámetro y conserva el resto.
- Se reutilizan evaluaciones completadas al reanudar; preparar dos veces el mismo diseño pendiente no crea duplicados. Una reproducción explícita usa un ID nuevo y vuelve a ejecutar para poder verificarla.
- La cancelación se procesa entre evaluaciones; no interrumpe un ajuste sklearn ya iniciado. Los fallos recuperables se registran con sus intentos anteriores.
- Los protocolos disponibles son hold-out aleatorio coordinado con CV interna (estratificada en clasificación) e inicializaciones de clustering sobre filas fijas. Protocolos agrupados, temporales y CV anidada se bloquean: nunca se sustituyen silenciosamente por particiones aleatorias.
- Comparar investigaciones anteriores permite una tabla descriptiva. Ordenar requiere diseños, datos y protocolos compatibles y ejecuciones completas. Cambiar de versión no crea una comparación válida automáticamente.

## Incertidumbre y límites de interpretación

La media, desviación estándar muestral, mediana y percentiles 25/75 describen las repeticiones. Con una repetición la desviación estándar no se define. Los folds y las repeticiones que comparten filas no se tratan como muestras independientes; no se construyen intervalos normales alrededor de sus medias ni p-valores automáticos. Cambiar la semilla supervisada cambia también la partición: esa variabilidad combina cambios del entrenamiento y de los datos evaluados.

Para dos alternativas se muestran diferencias pareadas por semilla; positivo significa mejora de B respecto a A según la dirección de la métrica. La interpretación de apoyo/contradicción es descriptiva y relativa al umbral práctico declarado, no una declaración de significación ni una interpretación automática de cualquier hipótesis escrita.

Bootstrap está limitado a dos alternativas, una partición fija, Accuracy o MAE, al menos diez etiquetas alineadas y confirmación explícita de observaciones independientes. Se remuestrean diferencias por observación con `scipy.stats.bootstrap`, método percentil, 95%, semilla registrada y lotes limitados. El intervalo es condicional a esos modelos y esa prueba: no incluye variación de reentrenamiento ni corrige selección repetida de modelos. No se admite en datos declarados temporales o agrupados. Con más de dos alternativas no se realizan múltiples tests.

La estabilidad usa `sklearn.metrics.adjusted_rand_score` entre soluciones alineadas de la misma alternativa. Se excluye el ruido de ambas soluciones y se registra cobertura. Soluciones degeneradas no generan ARI ficticio. Los pares de semillas comparten soluciones y no se tratan como observaciones independientes para inferencia. Repetir un algoritmo determinista puede producir siempre lo mismo; eso no verifica robustez a perturbaciones. No se implementa todavía estabilidad por resampling de filas.

Escalar cambia la geometría utilizada por índices internos como silueta. Sus valores son resultados en cada representación: no justifican por sí solos una escala universalmente superior. Ningún resultado del módulo demuestra causalidad, ausencia de fuga semántica ni utilidad de grupos naturales. Las curvas de aprendizaje siguen disponibles en los módulos científicos actuales; el laboratorio no dispara ajustes adicionales sin presupuesto explícito.

Fuentes metodológicas primarias: [bootstrap en SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [ARI en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html).

## Verificación y demostración reproducible

Desde `eda_ml_framework`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_scientific_experiment.py tests/test_experiment_lab_ui.py
.\.venv\Scripts\python.exe tools/demo_scientific_lab.py --confirm
```

La demostración ejecuta seis ajustes K-Means sobre las 178 filas y 13 variables de Wine, y otros seis para reproducirlos. Mantiene K=3, n_init=10 y semillas 42/43/44. Verifica bloqueo sin consentimiento, dataset intacto e igualdad de resultados dentro del entorno. Exporta resultados reales en `artifacts/scientific-lab-demo/`.

En la ejecución inicial observada, la silueta media fue 0,571138 sin escalar y 0,284859 estandarizando; ARI entre semillas fue 1 en ambas alternativas. La hipótesis de mejora de la silueta por estandarización no recibió apoyo en ese diseño. Esto no demuestra superioridad sustantiva de la representación sin escalar. El archivo `demonstration.json` y ambos manifiestos contienen la evidencia completa; no se utilizan estos números como resultados predeterminados de la interfaz.

Verificación del 10 de noviembre de 2026: **341 pruebas aprobadas** en la suite completa, incluyendo el flujo Streamlit de confirmación, ejecución, exportación y preparación de una reproducción. Se registraron siete advertencias en pruebas existentes: una media de datos vacíos y seis avisos de convergencia en pruebas que limitan MLP a veinte iteraciones. El laboratorio registra las advertencias numéricas y de convergencia reales de sus ajustes. Ruff pasó en los archivos nuevos y los servicios modificados. El resultado XML de la suite se guarda en `artifacts/scientific-lab-demo/test-results.xml`.
