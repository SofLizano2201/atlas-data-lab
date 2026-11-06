# Rediseño de visualizaciones ATLAS

## Actualización de tablas y tema interactivo

Las sesiones locales que continúan abiertas entre cambios de código reciben
una actualización visual versionada mediante `services/presentation_service.py`.
Se actualizan explícitamente los módulos de presentación una vez, se regeneran
los artefactos EDA y el informe automático, y se conservan los objetos entrenados,
el original y el dataset de trabajo. Los histogramas guardados también reciben
el morado actual. Los gráficos EDA automáticos excluyen identificadores,
constantes y fechas sin preparar; los diagnósticos de calidad los conservan.

La migración se verificó en la sesión activa `8513` con Housing.csv: 21.613 filas,
siete experimentos conservados, distribución de `price` en morado `#694D75` y
fondo lavanda `#F5F2F8`. Evidencia local:
`artifacts/atlas-2.0/housing-chart-fixed-8513.png`.

Las vistas usan `views/components/tables.py` para conservar los controles
nativos de Streamlit y presentar encabezados lavanda, cuerpo blanco y filas
alternas gris azulado. Los decimales se muestran sin ceros de relleno (hasta
ocho cifras significativas); los valores originales y las exportaciones no se
modifican. Tablas con más de 20.000 celdas o índices duplicados usan el renderizado
nativo sin Styler, manteniendo el tema claro y evitando costes excesivos.

La configuración de Streamlit define superficies de tablas claras; el CSS
mantiene el fondo general crema. El estilo también se aplica a tablas de
informes HTML mediante `TABLE_CSS`, compartido con el informe autónomo.

Todos los gráficos de la aplicación se renderizan con Plotly y `theme=None`.
La vista reaplica el tema ATLAS central sin alterar valores. Se retiró la ruta
`st.pyplot`; una figura externa que no sea Plotly recibe un error explícito,
en lugar de mostrarse silenciosamente como imagen estática. El área de datos
es lavanda suave, los histogramas morados y las categorías mantienen su mapa
de colores y símbolos. No se asignan colores arbitrarios a intervalos numéricos.

Verificación de esta actualización: 279 pruebas aprobadas, siete advertencias
conocidas; Ruff aprobado. Tras ajustar el formato numérico se repitieron las
pruebas específicas de tablas e integración. Navegador: selección, zoom, pan y
descarga PNG comprobados; diez pantallas sin errores. Captura de tablas:
`artifacts/atlas-2.0/table-clean.png`; gráfico:
`artifacts/atlas-2.0/chart-exploration.png`.

## Arquitectura y archivos

Los modelos conservan los cálculos científicos. `src/atlas/models/visualization.py`
centraliza la presentación Plotly, paleta, tipografía, márgenes, tooltips,
muestreo visual, exportación de observaciones e interpretación de coeficientes.
Los modelos `classification.py`, `regression.py`, `clustering.py`,
`unsupervised.py` y `comparison.py` construyen figuras a partir de sus resultados.

`src/atlas/controllers/charts.py`, `clustering_controller.py` y
`capabilities.py` conectan esas capacidades con las páginas y los agentes.
Las vistas `ui.py`, `workspace.py` y `modeling.py` ofrecen los controles;
`workspace.css` define sus tarjetas. `dataset_service.py` limpia preferencias
de gráficos al activar otro dataset. No se añadieron dependencias.

## Inventario revisado y mejoras

- EDA: histogramas, barras, líneas, dispersión, boxplots, violines, correlaciones,
  faltantes, KDE, Q-Q y matriz de relaciones. KDE, Q-Q y pairplots ahora son Plotly.
- Clustering: proyecciones 2D/3D, tamaños, perfiles, silueta, codo y dendrograma.
  Los puntos conservan su índice y el ruido tiene una identidad visual propia.
- Clasificación: confusión absoluta/normalizada, ROC, Precision-Recall,
  importancia y árbol interactivo construido desde el árbol ya entrenado.
- Regresión: valores reales frente a predichos, residuos, distribución, Q-Q,
  importancia y coeficientes. Los tooltips muestran valores y errores reales.
- Comparación: ranking según criterio, tiempo, rendimiento frente a tiempo,
  curvas, validación cruzada y facetas con escalas independientes.

Se corrigió el selector de correlación: la vista enviaba `method`, pero la
función leía únicamente `metodo`. Se preservaron codo y dendrograma entre
interacciones, evitando repetir el cálculo. Se retiró un gráfico comparativo
duplicado. Las correlaciones indefinidas se muestran como N/D y su máscara
visual no se exporta como si fuera un resultado numérico.

## Interacciones

El componente común permite filtrar series, ampliar el gráfico, activar zoom
con rueda y seleccionar puntos compatibles. La selección muestra las
observaciones y conserva estado de sesión. La barra Plotly permite zoom,
desplazamiento, visibilidad de series y descarga PNG. Se ofrecen CSV de datos
representados y HTML interactivo autónomo, generado a petición.

Las recomendaciones de agentes siguen utilizando las capacidades autorizadas
existentes; sus artefactos de gráficos utilizan el mismo componente. El codo
del agente devuelve la tabla calculada y su figura sin entrenar dos veces.

## Validación realizada

- Suite completa: **267 pruebas aprobadas**, 7 advertencias, 65,41 segundos.
  Las advertencias corresponden a medias de columnas vacías y convergencia
  de MLP en pruebas breves.
- Tras el último ajuste de etiquetas: **10 pruebas de gráficos aprobadas**.
- Ruff: comprobación aprobada.
- Navegación y resultados entrenados revisados con navegador automatizado.
- Interacciones reales verificadas: selección de puntos, zoom, desplazamiento
  y creación del archivo `atlas_grafico.png`.
- Galería de 23 figuras reales y capturas de escritorio y móvil inspeccionadas.
  Se corrigió el recorte de títulos en pantallas estrechas.

Reproducción desde la raíz del proyecto:

```powershell
..\.venv\Scripts\python.exe -m pytest
..\.venv\Scripts\python.exe tools/verify_charts.py
..\.venv\Scripts\python.exe tools/sidebar_browser_check.py --port 8511 --all-pages --trained-results --chart-interactions
```

El último comando requiere una instancia de ATLAS en ese puerto y Edge con
depuración local disponible, según las opciones del script.

Capturas generadas: `artifacts/charts/desktop.png`, `matrices.png`,
`modelos.png`, `mobile.png` y `artifacts/atlas-2.0/chart-selection.png`.
Los artefactos locales no se incluyen en Git.

## Límites y significado de las descargas

- El muestreo se limita a la representación y se informa en la descripción;
  no cambia métricas ni entrenamiento. Los límites se configuran donde procede.
- El CSV de histogramas, boxplots y violines contiene observaciones de entrada,
  no bins o cuartiles calculados por el navegador.
- El PNG nativo exporta la figura; el HTML también incluye su explicación.
- Las figuras 3D permiten rotar y ampliar; la selección de observaciones del
  componente se ofrece para dispersión 2D compatible.
- Las curvas requieren puntuaciones/probabilidades y clases compatibles;
  no se inventan curvas o métricas para resultados sin esa información.
- Las verificaciones cubren datasets representativos, no todas las posibles
  combinaciones de archivos, dependencias opcionales y parámetros.

## Extensión

Una nueva figura debe recibir resultados del modelo, reutilizar `style` y
`note`, proporcionar etiquetas españolas y tooltips de valores reales, y
devolverse al componente `figure` de la vista. Añadir pruebas de correspondencia
entre datos y trazas en `tests/test_modern_charts.py`; evitar recalcular métricas
en vistas, controladores o agentes.
