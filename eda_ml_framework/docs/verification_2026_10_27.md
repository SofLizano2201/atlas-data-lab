# Verificación de la integración Atlas 1.1

Fecha: 27 de octubre de 2026, America/Costa_Rica.

## Resultados ejecutados

| Comprobación | Resultado |
|---|---|
| Suite antes de modificar | 99 passed, 6 warnings in 19.83s |
| Suite final | **135 passed, 7 warnings in 25.76s** |
| Ruff sobre todo el proyecto | **All checks passed!** |
| Servidor local de prueba | Streamlit iniciado en 127.0.0.1:8502 |
| Endpoint de salud | Respuesta `ok` |
| Navegador Edge con datos públicos | Nueve páginas y cinco especialidades capturadas sin errores de ejecución |
| Sidebar | Anchos de 260/72 px y nueve filas de 44 px |
| Móvil | Viewport y documento de 390 px: sin desbordamiento horizontal |
| PDF | Tres páginas; reabierto, texto comprobado, límites revisados y páginas renderizadas |

Seis advertencias corresponden a MLP limitado deliberadamente a 20 iteraciones en
pruebas; no indican convergencia satisfactoria. La séptima procede de una muestra
numérica completamente vacía: el contrato devuelve `null`, no una media inventada.
No se ocultan advertencias globalmente.

## Qué se probó

- Jerarquía, cálculos, carga/exportación, transformaciones, clustering, clasificación,
  regresión, pipelines, búsquedas, métricas y comparaciones existentes.
- Nueve contratos compartidos y serialización JSON estricta.
- Cinco especialidades con herramientas reales; fallos, permisos, sesiones cruzadas,
  versiones obsoletas, caducidad, cancelación y repetición de confirmaciones.
- Consentimiento antes del LLM; ausencia de filas/tablas en el contexto enviado.
- Planes con herramientas no permitidas y datos con instrucciones maliciosas.
- Proveedores Responses y Chat Completions contra un servidor HTTP **local de prueba**,
  configuración ausente, URLs inválidas y errores de red sin revelar claves.
- Botones de los cinco agentes, conversación deshabilitada sin proveedor, acción EDA
  real, entrenamiento solo tras confirmar y acceso compartido a experimentos.
- Versiones del original/trabajo, hashes de esquema y contenido, original defensivo,
  procedencia de experimentos y selección contextual del comparador.
- Recursos acotados: modelos/búsquedas, MLP, propuestas/artefactos, KNN, one-hot,
  contingencias y versiones. Preservación de máscaras de datos faltantes y tipos nullable.
- Reportes HTML/PDF, contenido reabierto con pypdf, historial al navegar y al activar
  otro dataset. Los documentos previos conservan su hash y se identifican como instantáneas.
- Correcciones de scores SVM OvO/OvR, gráficos con roles de columna repetidos y
  codificación ordinal de alta cardinalidad.

## Revisión visual

Capturas en `artifacts/atlas-1.1/`: Inicio, Datos, Exploración, Preparación, Clustering,
Clasificación, Regresión, Experimentos y Reportes; además de paneles de agentes,
sidebar compacto y móvil. Se inspeccionaron imágenes para comprobar jerarquía,
legibilidad, paleta, controles y ausencia de errores visibles.

El primer intento se interrumpió por un timeout de Edge al repetir una captura
idéntica del panel EDA. La ejecución posterior esperó a que Streamlit terminara
de renderizar y reutilizó la captura cuando no había cambiado el estado visual;
completó las nueve pantallas y cinco especialidades sin errores de ejecución.

Posteriormente se recibieron ocho mockups y se adaptó la composición de las nueve
páginas. La implementación, las nuevas capturas y sus diferencias respecto a las
referencias se documentan en [adaptación visual](mockups_2026_10_27.md). La prueba
del proveedor LLM externo sigue pendiente de configuración.

El PDF de QA se conserva en `artifacts/reports/atlas-validation.pdf`, con PNG de la
primera y última página. Se verificó texto, márgenes y ausencia de contenido fuera
del área de página. Usa fuente estándar para impresión; las figuras interactivas
se descargan separadamente y no están incrustadas en este PDF.

## Límites restantes y no verificado

1. No hubo una inferencia externa real: no se configuró/probó una clave y modelo de
   un proveedor con consentimiento del usuario. Las pruebas HTTP locales **no** son
   resultados de un LLM real. Sin proveedor, el chat se identifica como no disponible.
2. UMAP, K-Medoids y XGBoost siguen siendo integraciones opcionales; sus motores no
   fueron ejecutados en esta entrega. La detección de ausencia sí se prueba.
3. No hay programación ni envío de reportes, base de datos persistente ni autenticación
   multiusuario. No se ofrecen acciones de envío ficticias.
4. El historial reside en memoria de la sesión. Los informes sobreviven a cambios
   de dataset durante esa sesión, pero deben descargarse antes de cerrar/reiniciar.
5. La selección de prueba sigue siendo aleatoria, no por tiempo/grupos. Para inferencia
   estadística el diseño y la independencia no pueden demostrarse automáticamente.
6. La importancia en holdout es interpretativa y no debe usarse para seleccionar
   predictores. Entrenar con datos exploratorios previamente transformados puede
   introducir sesgo; el flujo supervisado recomienda el original y pipeline.

No se declara Atlas finalizado respecto a fidelidad a mockups ni validación con
un proveedor LLM externo. La implementación local comprobada y los puntos pendientes
se distinguen explícitamente.
