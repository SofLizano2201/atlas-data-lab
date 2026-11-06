# ATLAS · Diseño de producto

## Resumen automático comprensible

El panel automático distingue una recomendación principal para predecir
(acento azul) de una alternativa para explorar grupos (acento morado). Cada
tarjeta separa propósito, modelo recomendado, dato relevante e interpretación.
El error absoluto medio se presenta como diferencia media, con formato español
y referencia a las unidades del objetivo, sin inferir una moneda ni garantizar
un límite para cada error. El objetivo provisional queda visible en un bloque
propio. El tiempo de ejecución pasa a la sección secundaria de decisiones.

Los botones de cada tarjeta abren gráficos ya calculados, sin nuevos
entrenamientos. La comparación técnica queda dentro de un desplegable y la
descarga se agrupa bajo «Conserva tus resultados». Se verificó la acción de
abrir/ocultar predicciones tanto con AppTest como en la sesión 8513. Veintiuna
pruebas de integración, recorrido autónomo y tema visual aprobadas; Ruff
aprobado. Captura: `artifacts/atlas-2.0/recommendations-clear-8513.png`.

## Espacio de trabajo compacto e interactivo

La barra «Vista» permite mostrar la guía, desplegar el contexto completo y
reducir el movimiento. Estas preferencias permanecen en la sesión y no alteran
los datos ni vuelven a entrenar modelos. Un resumen compacto conserva siempre
dataset, versión, dimensiones, experimentos y avisos de nulos/duplicados.

«Estado del dataset y próximos pasos» reúne los detalles y seis botones reales
para navegar entre etapas. Sus etiquetas de estado proceden del controlador de
flujo; no se presentan avances ficticios. En modelado, el acceso conserva la
especialización de clasificación o regresión cuando corresponde.

Inicio separa el dashboard y el análisis automático. EDA separa exploración
manual y resultados automáticos, evitando tablas duplicadas antes de los
controles principales. Los botones de ampliación y exploración de gráficos
comparten una barra de acciones.

Se añadieron entradas suaves de 140–200 ms, respuesta al pulsar y despliegue de
paneles. No hay animación continua ni progreso simulado. Tanto la opción
«Reducir movimiento» como la preferencia del sistema desactivan efectos.

Validación de esta reorganización: batería completa de 283 pruebas aprobadas
con siete advertencias conocidas. Se añadieron pruebas de navegación entre
etapas y conservación de datos/preferencias. Ruff aprobado. La sesión activa
8513 se actualizó conservando Housing.csv y siete experimentos.

## Intención

Las diez pantallas comparten una jerarquía: nombre de la sección, propósito y
orientación breve, contexto real del dataset, configuración y resultados.
Las instrucciones numeradas explican el uso; no son botones ni indican que
una operación se haya completado. El progreso real conserva su estado existente.

## Color y composición

- Ciruela `#331832`: identidad, navegación lateral y texto de alta importancia.
- Morado `#694D75`: orientación, selección y bloques secundarios.
- Azul `#1B5299`: acción principal, cifras y énfasis interactivo.
- Azul claro `#9FC2CC`: superficies informativas, bordes y foco acompañado de
  un contorno azul oscuro; no se usa como texto pequeño sobre crema.
- Crema `#F1ECCE`: fondo cálido; tarjetas derivadas claras separan controles.

Las tarjetas alternan acentos discretos, con bordes de 1 px, radio de 12–20 px,
sombras suaves y espaciado consistente. La tipografía mantiene Manrope/Inter
con alternativas del sistema, sin descargas obligatorias de fuentes.

La capa compartida `src/atlas/views/styles/product.css` se carga al final de
los estilos existentes. Evita modificar cálculos o introducir dependencias.
Utiliza clases propias, roles y selectores semánticos; no clases CSS generadas.

## Cambios de interacción

- Inicio, Datos, EDA, Preparación, Clustering, Clasificación, Regresión,
  Experimentos, Comparador y Reportes tienen instrucciones contextualizadas.
- EDA agrupa el contenido de cada pestaña dentro de una superficie consistente.
- Preparación comienza por una necesidad: información incompleta, repetidos y
  extremos, escalas/distribuciones o categorías/columnas. Se conservan todas las
  operaciones existentes y se muestran nombres comprensibles cuando procede.
- La revisión antes/después y el historial de restauración tienen espacios
  distintos. Confirmar sigue siendo una acción explícita y destacada.
- Reportes separa configuración, historial/descargas y exportación. Después de
  generar un informe indica dónde encontrarlo. La programación no configurada
  deja de ocupar una sección principal.
- Acciones principales azules; secundarias de fondo claro y borde morado.
  Estados hover, foco de teclado, selección y deshabilitado diferenciados.
- Errores recuperables en una tarjeta con título, mensaje del requisito y ayuda
  para corregirlo. No se borran resultados al fallar una operación.
- Estados vacíos sin dataset y comparaciones sin resultados ofrecen navegación
  real hacia un primer paso.
- Cargas conservan spinners/status reales y reciben una superficie informativa.
- Hover y transiciones de 180 ms aportan respuesta, sin animación constante.
  `prefers-reduced-motion` desactiva movimientos y transiciones.
- Pestañas desplazables, tarjetas adaptables y sidebar expandida/compacta siguen
  disponibles en pantallas pequeñas.

## Archivos principales

`app.py`, `views/components/layout.py`, `views/styles/product.css`,
`views/pages/workspace.py`, `views/pages/reports.py`, `views/pages/modeling.py`,
`tests/test_product_ux.py` y `tools/sidebar_browser_check.py`.

## Verificación

- Batería existente: 273 pruebas aprobadas, 7 advertencias NumPy/MLP conocidas.
- Dos nuevas pruebas UX aprobadas: recuperación de un logaritmo incompatible sin
  modificar el dataset y generación de reportes con la nueva navegación.
- Revisión final específica: 18 pruebas de UX, integración y navegación aprobadas.
- Ruff aprobado.
- Navegador Edge: diez pantallas capturadas, modos de sidebar de 260/72 px y
  viewport móvil de 390 px comprobados. No hubo errores de ejecución en el recorrido.
- Capturas inspeccionadas de Inicio, Preparación, Reportes y móvil.

Las capturas locales se encuentran en `artifacts/atlas-2.0/`; no se versionan.
La revisión no sustituye una auditoría completa con lectores de pantalla ni
comprueba cada combinación de widgets, contenidos y tamaños de dispositivo.
Los gráficos y tablas mantienen sus propios controles de zoom/desplazamiento.

```powershell
..\.venv\Scripts\python.exe -m pytest tests/test_product_ux.py tests/test_integration.py tests/test_mockup_navigation.py -q
..\.venv\Scripts\python.exe tools/sidebar_browser_check.py --port 8514 --all-pages --trained-results --autonomous
```

## Panel de asistencia: siguiente paso y actividad

Se separaron la navegación a resultados, las ideas de predicción y la actividad
completada en tarjetas con jerarquía visual y colores ATLAS. Las sugerencias de
objetivo distinguen una propuesta automática de una elección del usuario.
La actividad agrupa repeticiones consecutivas y omite reutilizaciones de caché;
el registro original se conserva. El botón de exploración navega sin entrenar.

Validación de este cambio: 21 pruebas de panel, integración y flujo autónomo
aprobadas; Ruff aprobado. Revisión visual con Iris en una sesión de prueba.

## Comparador guiado

El comparador organiza la selección, el criterio, la interpretación y las descargas
en cuatro pasos. Los conjuntos conservan las claves originales de compatibilidad,
pero se presentan con objetivo, tipo de tarea y número de pruebas. La selección
usa IDs estables internamente y nombres legibles en pantalla. Se necesitan dos
pruebas; cambiar filtros o indicadores no vuelve a entrenar.

Se muestran solo vistas respaldadas por resultados guardados; los indicadores
sin valores válidos en todas las pruebas no se ofrecen para ordenar. Los empates
se explican y los parámetros JSON quedan en detalles técnicos. La corrección de
procedencia permite interpretar resultados de análisis conservados tras recargar
sus clases durante desarrollo. Se verificó visualmente con Housing, conservando
sus siete experimentos y sin entrenamientos adicionales.
