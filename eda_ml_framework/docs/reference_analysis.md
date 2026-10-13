# Fase 1: análisis de las referencias

Se leyeron `GuiaRegresiones.py`, `GuiaRegresiones (1).py` y
`GuiaRegresiones (2).py` de Descargas. Los tres tienen el mismo SHA-256:
`376D28499B485CE9EF09E6E74C4FB9652EBA02C1CC936A22F2115F31EE7DEA03`.
Son tres copias del mismo contenido, no tres implementaciones distintas.
No se ejecutaron como scripts. La fotografía requiere EDA → NoSupervisado →
Clustering y EDA → Supervisado → Clasificacion/Regresion.

## Equivalencias conservadas

| Guía | API nueva / algoritmo |
|---|---|
| RegLinSimp | entrenar('linear_simple', objetivo, [predictor]) |
| RegLinMult | entrenar('linear_multiple', objetivo, predictores) |
| RegLasso | entrenar('lasso', ...); alpha predeterminado 0.1 |
| RegLassoCV | entrenar('lasso_cv', ...); búsqueda de alpha dentro del pipeline |
| RegRidge | entrenar('ridge', ...); alpha predeterminado 1 |
| RegRidgeCV | entrenar('ridge_cv', ...); búsqueda dentro del pipeline |
| ComparacionLasso | comparar_lasso() sobre resultados almacenados |
| ComparacionRidge | comparar_ridge() |
| SVM / ComparacionSVM | svr_rbf, svr_linear, svr_poly / comparar_svm() |
| DecisionTreeReg | entrenar('tree', ..., parametros={'max_depth': 3}) |
| RandomForestReg | entrenar('forest', ..., parametros={'max_depth': 2}) |
| XGBoostingReg | gradient_boosting; NO era XGBoost |
| ComparacionALL | comparar(), sin reentrenamiento |
| Max / ResVarPred | maximo(), minimo(), cuartiles() |

El error relativo académico ER se conserva exactamente como
`sum(abs(y - pred)) / sum(abs(y))`. Si el denominador es cero se devuelve NaN.
SVR conserva C=100 y epsilon=0.1 por defecto, con kernels explícitos.

## Problemas corregidos

- La clase madre de la referencia no estaba definida; se implementa la herencia completa.
- Su constructor no llama a super() y duplica el DataFrame: ahora el estado vive en EDA.
- Limpieza atada a columnas de vehículos/inmuebles y mutación silenciosa: ahora columnas
  seleccionables, propuestas sin mutación y aplicación confirmada.
- Códigos categóricos arbitrarios crean orden artificial: one-hot predeterminado,
  ordinal solo por elección explícita.
- Inconsistencia de reshape en regresión simple: siempre DataFrames 2D.
- CV=10 fijo falla con datasets pequeños: CV configurable y validado.
- Reentrenamientos al comparar y al aplicar alpha seleccionado: resultados almacenados
  y best_estimator_ reutilizado.
- store_cv_values es una API sensible a versión: no se usa esa opción antigua.
- Imports obligatorios de umap, prince y sklearn_extra para operaciones no relacionadas:
  imports opcionales solo cuando se elige la integración; prince no es necesario.
- warnings.filterwarnings('ignore') eliminada; advertencias no se ocultan.
- print() como interfaz de resultados sustituido por tablas, métricas y figuras.
- GradientBoostingRegressor correctamente separado de la integración xgboost.

RidgeCV/LassoCV/ElasticNetCV se implementan como equivalentes funcionales mediante
GridSearchCV del pipeline completo. Usar el estimador CV al final de un pipeline
ajustaría el preprocesamiento antes de sus folds internos; esta alternativa mantiene
aislamiento de cada fold. La cuadrícula es configurable usando el algoritmo base.
