# Reportes

La pantalla Reportes permite elegir información, estadística, calidad, limpieza,
modelos/métricas y decisiones de agentes. Genera HTML autónomo o PDF real mediante
ReportLab y conserva hasta diez documentos en memoria de la sesión. Cada documento
registra huella de datos, secciones, título, formato y fecha UTC de generación.
Cambiar el dataset conserva los documentos ya generados e indica cuando pertenecen
a otra versión; su contenido no se recalcula silenciosamente.

Los modelos se listan con sus configuraciones, versiones y dataset. Listar modelos
de cohortes diferentes no implica compararlos; la comparación rigurosa sigue en
el módulo Experimentos y exige cohorte compatible.

PDF usa A4 horizontal, cabeceras repetidas y división de tablas anchas en grupos de
columnas. Limita tablas a 200 filas y celdas a 1500 caracteres; se indica en el
documento. Exporta CSV/Excel cuando necesites los datos completos.
Los gráficos interactivos se descargan aparte como HTML o PNG en sus pantallas y
en el selector de gráficos de Reportes. Esta versión no los incrusta en el PDF.

No hay tareas programadas ni servicio de correo configurados. No se ofrecen
controles de programación/envío como si estuvieran disponibles.

Verificación opcional del PDF:

```powershell
python -m pip install -e '.[test,pdf_qa]'
python tools/verify_pdf.py
```

El verificador genera un documento con datos públicos, reabre su contenido,
comprueba límites de texto y renderiza páginas de muestra en `artifacts/reports/`.
No incluye los datos privados de la sesión y sus resultados no se suben a Git.
