# Verificación ejecutada

La versión actual supera **155 pruebas**, Ruff y la comprobación de dependencias.
La comprobación final del paquete y la aplicación consta en
[verificación de publicación](release_verification_1_1.md).
El [estándar de visualización](visualization_standard.md) y la
[adaptación de los mockups](mockups_2026_10_27.md) documentan las últimas mejoras.
La ampliación inicial con agentes y PDF del 27 de octubre se documenta en
`verification_2026_10_27.md` (135 pruebas en esa etapa). Este archivo conserva la
evidencia histórica de la primera entrega y del rediseño inicial del sidebar.

Fecha local: 12 de octubre de 2026 (America/Costa_Rica).
Entorno: Windows, Python 3.11, entorno virtual aislado en la carpeta superior.
Dependencias exactas: `requirements-lock.txt` (sin rutas locales ni instalación editable).
Entre las versiones ejecutadas: pandas 2.3.3, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.7.2, Streamlit 1.65.0 y Plotly 6.9.0.

## Resultados

- `python -m pytest -q`: **98 passed, 6 warnings in 17.13s** en la ejecución final.
- `python -m ruff check .`: **All checks passed!**.
- `python -m ruff format .`: archivos formateados; sin cambios funcionales posteriores.
- Servidor `streamlit run app.py --server.address 127.0.0.1 --server.port 8501`:
  inició correctamente. Endpoint `/_stcore/health` respondió **ok**.
- AppTest comprobó todas las páginas con dataset activo, navegación desde Inicio,
  estado entre reruns, confirmación de limpieza y entrenamiento desde las pantallas
  de clasificación, regresión y clustering, además de PCA de los clusters.

## Cobertura funcional comprobada

Carga/exportación CSV, XLSX, JSON, Parquet; configuración CSV; rechazo de formatos
no admitidos, archivos vacíos y exceso de tamaño. Herencia real y métodos de EDA
accesibles desde todas las clases. Copias defensivas, transformaciones propuestas,
aplicación confirmada y restauración.

Estadísticas, faltantes, outliers, imputación numérica/categórica, escaladores,
codificadores, potencias, binning, selección/conversión de columnas, gráficos,
correlaciones y pruebas estadísticas con aceptación/rechazo de supuestos.
Seguridad de exportaciones y HTML escapado.

Algoritmos principales de clasificación y regresión probados por parametrización;
todos los algoritmos principales de clustering. PCA y t-SNE, distancias,
codo, dendrograma y silhouette. Ruido, un solo cluster y un cluster por fila.
Objetivo inválido, fuga del objetivo, objetivo con faltantes, clases insuficientes,
CV insuficiente, categorías nuevas en predicción y datos mixtos.

Métricas académicas ER, errores con denominadores cero, curvas con y sin
probabilidades, importancia, coeficientes, aprendizaje, búsquedas grid/random,
comparación compatible/incompatible y modelo serializado.
La media aprendida por StandardScaler se contrastó con la media de **x_train**.
Las búsquedas de regularización CV incluyen el pipeline completo en cada fold.

## Advertencias y límites de lo verificado

Las seis ConvergenceWarning provienen de MLP con max_iter=20 usado deliberadamente
para una prueba rápida de integración; no indican convergencia satisfactoria del modelo.
La aplicación permite más iteraciones y conserva las advertencias.

UMAP, K-Medoids y XGBoost no están instalados en este entorno. Sus motores **no fueron
ejecutados**. Se verificaron errores explícitos de ausencia de dependencia y el adaptador
de codificación de etiquetas con un clasificador sklearn. Instalar las integraciones y
validarlas sobre los datos deseados antes de utilizarlas en una entrega académica que
requiera esos motores. K-Medoids puede requerir NumPy 1.26 por compatibilidad binaria.

La inspección visual manual/automatizada del navegador **no pudo ejecutarse**: el runtime
de browser automation falló al iniciar el sandbox de Windows. Esto no impidió la prueba
HTTP ni AppTest, pero no se afirma QA visual a nivel de píxeles o responsive en dispositivos.
La pestaña local fue solicitada mediante el panel de Codex; puede abrirse por su URL.

Se corrigieron fallos encontrados durante ejecuciones previas: validación del None de
Ridge.max_iter y backend Tk no disponible. El backend estadístico definitivo es Agg.
Los resultados anteriores fallidos no se presentan como verificación final.

## Verificación posterior: rediseño del sidebar

- Suite completa después del cambio: **99 passed, 6 warnings in 20.56s**.
- Revisión estática completa: **All checks passed!**.
- Revisado con Edge headless instalado: navegación real, activación de dataset y
  ancho de 260 px / 72 px. En viewport móvil de 390 px el documento mide 390 px,
  sin desbordamiento horizontal; nueve botones visibles de 44 px de altura.
- Capturas inspeccionadas de escritorio, modo compacto y móvil. Esta revisión posterior
  pudo hacerse mediante Edge local, a diferencia del runtime de browser automation que
  falló durante la primera entrega.
- El dataset, navegación y resultados persisten al seleccionar páginas y al contraer/
  expandir. Se verificó escape HTML y nombre de dataset dinámico.
- Detalles, archivos y límites responsive en `docs/sidebar.md`.
