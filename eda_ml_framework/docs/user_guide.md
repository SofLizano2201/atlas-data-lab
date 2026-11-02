# Guía del usuario

1. **Inicio** muestra estado, filas, variables, faltantes y experimentos. Los botones
   abren Datos, Exploración y Clasificación.
2. **Datos**: selecciona un CSV/XLSX/JSON/Parquet (25 MB máximo), configura separador,
   codificación o hoja; revisa la vista previa y confirma. Los ejemplos Iris, Diabetes,
   Cáncer de mama y clusters sintéticos funcionan sin internet.
3. **Exploración**: consulta resumen, estadísticas, frecuencias, agrupación, calidad,
   correlaciones, distribuciones, VIF y pruebas. Cada gráfico calculado puede exportarse.
4. **Preparación**: elige columnas y operación; previsualiza antes/después. Solo
   **Confirmar estos cambios** modifica el estado. Restaura una versión cuando sea necesario.
5. **Clustering**: elige variables, escalamiento y codificación; calcula representación
   PCA/t-SNE/UMAP. Agrupa, revisa ruido, métricas, tamaños y estadísticas de grupos.
   Compara experimentos en Experimentos. Descarga etiquetas.
6. **Clasificación / Regresión**: utiliza preferentemente dataset original. Selecciona
   objetivo, predictores, partición, folds, preprocesamiento y algoritmos. Excluye IDs.
   Configura controles guiados o parámetros JSON avanzados; éstos son datos, no código.
   Entrena, consulta métricas, gráficos y predicciones. Las SVM sin probabilidades usan
   puntuaciones para curvas; log loss no estará disponible.
7. **Experimentos**: selecciona resultados compatibles, métrica y sentido de optimización.
   El ganador depende del criterio. Si cambias dataset, variables o partición, compara
   cada cohorte por separado. Tiempo incluye entrenamiento y validación cruzada.
8. **Reportes**: descarga datos y reportes EDA/limpieza. Las pantallas de modelos ofrecen
   reporte HTML, modelo joblib y configuración JSON. Exporta antes de cerrar la sesión.

## Interpretación responsable

Las métricas provienen del holdout indicado y pueden variar con semilla, tamaño y
composición de la muestra. No uses el holdout reiteradamente para escoger variables,
algoritmos o configuraciones. Selecciona usando CV y confirma con una prueba independiente.
Para series temporales o personas con observaciones repetidas necesitas divisiones por
tiempo/grupo: la partición aleatoria de esta versión no es apropiada.
Normaliza/imputa dentro del pipeline al modelar; hacerlo en Preparación sobre todos los
datos puede dar evaluaciones optimistas. Escoge ponderación balanceada en clasificación
cuando corresponda y considera recall/F1, además de accuracy.

## Resolución de errores

- Objetivo con nulos: decide si eliminar filas o corregir datos, no se elimina en silencio.
- CV insuficiente: reduce folds o añade muestras de cada clase.
- Columna completamente vacía para imputación EDA: usa constante o elimina tras confirmar.
- Log/Box-Cox: requieren datos positivos; Yeo-Johnson admite valores no positivos.
- Categorías excesivas: excluye identificadores; ordinal introduce orden y debe justificarse.
- UMAP/K-Medoids/XGBoost: instalación opcional indicada en README.
- Archivo ilegible: revisa formato, tamaño, encoding, separador y hoja.
- Comparación incompatible: selecciona la misma cohorte de datos y partición.

No subas archivos joblib/pickle: la aplicación solo permite exportar modelos. Para cargar
tu propio modelo exportado en Python, hazlo exclusivamente si confías en su origen.

## ATLAS 1.2: asistencia opcional

Abre el botón del agente en la pantalla que estás usando. Las acciones rápidas
ejecutan herramientas locales; la conversación requiere proveedor configurado y
consentimiento. Las propuestas de configuración y los entrenamientos se muestran
antes de ejecutarse y requieren confirmación. Aplicar una configuración actualiza
los controles reales del formulario; luego puedes entrenar manualmente. Cerrar el
panel cancela sus confirmaciones pendientes. Al cambiar de dataset se reinician los
resultados dependientes. Los datos permanecen en memoria durante esta sesión.

La distribución técnica actual está en `src/atlas/`; consulta [MVC](mvc.md) y
[agentes](agents.md). Las importaciones antiguas siguen disponibles.
