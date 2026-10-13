# Catálogo de métodos

Los docstrings y anotaciones del código especifican parámetros y tipos de retorno.

## EDA

Carga: `cargar`, `cargar_csv`, `cargar_excel`, `cargar_json`, `cargar_parquet`.
Consulta: `vista_previa`, `dimensiones`, `columnas`, `tipos_datos`, `informacion`,
`memoria`, `valores_unicos`, `constantes`, `duplicados`, `resumen`.
Estadística: `estadisticas`, `estadistica`, `media`, `mediana`, `moda`, `minimo`,
`maximo`, `rango`, `varianza`, `desviacion_estandar`, `cuartiles`, `percentiles`,
`rango_intercuartilico`, `coeficiente_variacion`, `asimetria`, `curtosis`,
`frecuencias`, `contingencia`, `estadisticas_agrupadas`.
Calidad: `nulos`, `outliers`, `inconsistencias`, `categorias_raras`.
Estado: `copia_seguridad`, `aplicar`, `restaurar`, propiedades `historial`, `versiones`, `df`.
Salidas: `exportar`, `reporte`, `grafico`.
Inferencia exploratoria: `correlacion` (Pearson/Spearman/Kendall), `vif`, `prueba`
(shapiro, normaltest, levene, ttest, anova, chi2, mannwhitney).

`transformar(operacion, columnas, **parametros)` acepta:

| Operación | Parámetros adicionales |
|---|---|
| mean, median, mode | Imputar; rechaza columna sin valores de apoyo |
| constant | valor |
| knn | n_neighbors |
| drop_rows, duplicates | columnas seleccionadas |
| drop_null_columns | umbral entre 0 y 1 |
| standard, minmax, robust, normalizer | escalamiento exploratorio |
| onehot, ordinal | codificación exploratoria |
| label | una variable objetivo sin faltantes |
| log, boxcox, yeojohnson | restricciones de dominio de cada transformación |
| binning | n_bins |
| select, drop_columns | columnas seleccionadas |
| convert | tipo: float64, int64, string, category o datetime64[ns] |
| normalize_names | quitar acentos y espacios; rechaza colisiones |
| drop_outliers | metodo: iqr/zscore/isolation; umbral |
| clip | límites IQR |
| winsor | limite: proporción por cola, < 0.5 |

`transformar_personalizado(funcion)` solo para funciones locales de confianza.

Figuras: histogram, box, violin, scatter, bar, line, pie (≤8 categorías), heatmap,
correlation, pairplot, kde, qq, count, missing, outliers. `before_after` compara
distribuciones y puede usarse desde Python. Pairplot usa ≤8 variables y ≤500 filas.

## No supervisado y clustering

`preparar(columnas, scaler=..., encoding=...)`, `reducir(columnas, metodo='pca',
dimensiones=2, ...)`, `visualizar_representacion`, `distancias` (euclidean,
cityblock/Manhattan, cosine), `comparar_representaciones`.
PCA devuelve varianza explicada y modelo con cargas; las cargas evalúan contribuciones.
`Clustering.entrenar` admite kmeans, minibatch, kmedoids opcional, dbscan, optics,
agglomerative, hierarchical, gaussian, spectral, birch.
`codo`, `silhouette_plot`, `dendrograma`, `caracterizar`, `representantes`, `comparar`.
Etiquetas exportables en `Experiment.predictions`; -1 indica ruido. Representantes
son medias de cada grupo en espacio transformado, no necesariamente medoids del algoritmo.

## Supervisado

`identificar_problema`, `entrenar`, `predecir`, `importancia`, `coeficientes`,
`curva_aprendizaje`, `comparar`. División y validación cruzada se configuran mediante
test_size, random_state del constructor y cv. Predictores explícitos excluyen objetivo.
`busqueda` acepta listas de parámetros de estimador; también claves de pipeline con `__`.
`tipo_busqueda='grid'|'random'`, `n_iter`. Se incluyen modelo y preprocesamiento en búsqueda.
Experiment: `model_bytes()` exporta joblib, `config` es exportable como JSON;
`predictions` y tabla `compare` mediante `export_frame`. `html_report` exporta reportes.

Clasificación: logistic, knn, tree, forest, extra_trees, naive_bayes, svm_linear,
svm_rbf, svm_poly, adaboost, gradient_boosting, hist_gradient_boosting, mlp, xgboost opcional.
Accuracy, balanced accuracy, precision/recall/F1 macro, F1 weighted, métricas binarias,
confusión, classification report, ROC/AP OvR y log loss solo con probabilidades.
`grafico_resultado`: confusion, confusion_normalized, roc, pr, tree, correctness.

Regresión: linear_simple, linear_multiple, ridge/ridge_cv, lasso/lasso_cv,
elasticnet/elasticnet_cv, svr_linear/rbf/poly, tree, forest, extra_trees,
gradient_boosting, hist_gradient_boosting, knn, mlp, xgboost opcional.
MAE, MSE, RMSE, R2, R2 ajustado para regresión lineal, MAPE, sMAPE, ER.
MAPE/sMAPE/ER son proporciones, no porcentajes. MAPE se omite con reales cero.
R2 ajustado es diagnóstico con p igual al número de características transformadas;
no se interpreta como prueba inferencial sobre el holdout.
`grafico_resultado`: actual_predicted, residuals, histogram, qq.
`comparar_lasso`, `comparar_ridge`, `comparar_svm` filtran experimentos sin reentrenar.

## Supuestos y límites

Pruebas t/ANOVA se rechazan si diagnósticos de normalidad y homogeneidad resultan
incompatibles. No rechazar un diagnóstico no demuestra un supuesto. Independencia,
muestreo y diseño deben justificarse externamente. Chi2 rechaza esperadas <5.
Shapiro usa muestra reproducible de 5000 si es necesario; D'Agostino exige ≥8 filas.
VIF requiere más observaciones completas que predictores. Coeficiente de variación
es difícil de interpretar con escalas sin cero significativo o medias casi cero.

La aplicación limita 25 MB subidos, 100 MB descomprimidos, 100000 filas, 500 columnas,
20 versiones. ML supervisado limita matrices densas a 10 millones de celdas;
CV 2–10 folds. Clustering/reducción: 5000 filas; algoritmos cuadráticos 2000;
distancias 1000; silhouette individual 2000; búsqueda 100 combinaciones.
Las operaciones largas son síncronas, con indicadores de progreso; no hay cancelación
de trabajos ni almacenamiento persistente de sesiones.
