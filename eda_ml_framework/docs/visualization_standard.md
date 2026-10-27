# Estándar de visualización de ATLAS

Los gráficos representan cálculos del framework. El estilo se centraliza en
`framework/visualization.py`; Streamlit solo presenta y exporta las figuras.
No se aplican colores automáticos de Streamlit que sustituyan la identidad Atlas.

## Reglas de lectura

- Título descriptivo, ejes rotulados, leyendas y texto con tamaños consistentes.
  Se mantienen los nombres originales de variables y sus unidades cuando forman
  parte del nombre; no se inventan unidades ni interpretaciones de columnas.
- Colores azules, morados y variantes de estos tonos, sobre una superficie cálida.
  El crema funciona como fondo o centro de escala, no como categoría casi invisible.
- Categorías: asignación determinista para la misma colección. Dispersión añade
  símbolos; curvas añaden trazos. La interpretación no depende únicamente del color.
- Variables numéricas continuas en dispersión: escala continua con barra. Valores
  numéricos de hasta 12 niveles se tratan como categorías al elegir color.
- Líneas ordenadas por X. No se agregan filas ni se suavizan curvas sin indicarlo.
  El usuario debe convertir fechas de texto si necesita orden cronológico real.
- Histogramas por categoría: frecuencias, intervalos compartidos y barras agrupadas.
  La selección Y no se usa como suma o ponderación oculta del histograma.
- Conteos: barras horizontales ordenadas; incluyen valores faltantes. Los gráficos
  circulares se reservan para hasta ocho categorías y muestran proporciones.
- Correlación: escala divergente fija de −1 a +1, centrada en cero. `N/D` tiene
  una superficie diferenciada para evitar confundir un coeficiente indefinido con cero.
  Correlación no implica causalidad ni independencia cuando el coeficiente es cero.
- Faltantes: dos estados fijos, Presente/Faltante, incluso sin valores faltantes.
- Confusión: intensidad secuencial creciente. La versión normalizada usa porcentajes
  por clase real y una escala de 0 a 100 %; no se cambia su denominador.
- ROC/Precision–Recall: ejes de proporción con porcentajes, identificación de clases
  y trazos diferenciados. ROC incluye una referencia aleatoria. Las curvas no se
  generan para estimadores sin las puntuaciones necesarias.
- Regresión: iguales rangos numéricos para valores reales/predichos, con línea
  `y = x`. Residuos: real menos predicho, con referencia horizontal en cero.
- Clusters: color y símbolo por grupo, ruido explícito con una × gris lavanda.
  PCA incorpora varianza explicada en los ejes; no se atribuye esa varianza a t-SNE/UMAP.
  La silueta muestra distribuciones individuales por grupo, excluye ruido e incluye
  la media. El dendrograma indica método y truncamiento.
- Antes/después: histogramas porcentuales con intervalos compartidos, leyenda
  explícita y tamaños de ambas versiones. No se confunden cambios de cantidad de
  filas con cambios de distribución.
- Importancia por permutación: barras horizontales con desviación de cinco
  repeticiones; esa desviación no es un intervalo de confianza.

## Límites de legibilidad

Máximo 12 categorías de color y ocho de circular. Conteos muestran las 30 categorías
más frecuentes si hay más; se indica cuántas observaciones cubren. Correlación muestra
hasta 30 variables y omite cifras dentro de celdas de matrices grandes. Dispersión y
gráficos EDA muestran las primeras 10.000 filas cuando se excede ese límite, con aviso.
Faltantes usa las primeras 1.000 filas. Pairplot usa hasta seis variables y 500 filas
muestreadas con semilla 42, informadas en el título.

KDE exige variación y al menos tres valores finitos. Al agrupar, las curvas tienen
área 1 individualmente y se avisa de la omisión de grupos insuficientes. La identidad
de una categoría puede cambiar al cambiar el universo de categorías; no se promete
una identidad global entre datasets diferentes. Muchas categorías, nombres muy largos
o matrices densas siguen requiriendo seleccionar variables o ampliar el gráfico.

## Verificación y uso

`tests/test_visualization.py` comprueba rangos, denominadores, datos representados,
colores/símbolos, ruido, referencias y conservación de los datos originales.
La suite completa terminó con **155 pruebas aprobadas y 7 advertencias** (las mismas
advertencias de MLP limitado a 20 iteraciones y estadística sobre una columna vacía).

```powershell
& '..\.venv\Scripts\python.exe' -m pytest -q
& '..\.venv\Scripts\python.exe' tools/verify_charts.py
```

La herramienta genera una galería HTML autónoma y capturas en `artifacts/charts/`:
12 figuras Plotly y tres figuras estadísticas, con Iris, Diabetes y agrupamiento
sintético calculados. Se revisaron escritorio, móvil, matrices y diagnósticos de
modelos en Edge. No utiliza datos privados de sesiones ni llamadas externas.
La revisión visual es manual, no una certificación formal de accesibilidad.

Los gráficos ya guardados en memoria conservan su instantánea. Para aplicar el
estilo a ellos, vuelve a generar la figura; no hace falta volver a cargar el dataset.
La exportación HTML conserva título, colores y advertencias; el botón de cámara de
Plotly exporta PNG al doble de resolución. Matplotlib se exporta como PNG.
