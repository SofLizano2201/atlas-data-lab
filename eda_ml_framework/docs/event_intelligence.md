# Inteligencia por eventos y exploración automática

Implementación sobre el proyecto existente, 3 de noviembre de 2026. No se creó una
aplicación paralela ni se trasladaron algoritmos a los agentes.

## Diagnóstico inicial

La base tenía seis clases científicas con herencia real, cinco agentes polimórficos,
21 herramientas registradas, pipelines de entrenamiento y pruebas de integración.
La ayuda requería abrir un diálogo; confirmar datos no generaba un EDA persistente.
No había un contrato común de eventos ni un panel visible durante la navegación.
La batería inicial ejecutada pasó 238 pruebas con siete advertencias conocidas.

## Archivos de esta integración

| Capa | Archivos | Responsabilidad |
|---|---|---|
| Core | `src/atlas/core/events.py` | Enum de eventos, contexto tipado, IDs estables y deduplicación por sesión y versión |
| Core | `schemas.py`, `session.py`, `dependencies.py` | Feedback tipado, evidencia cacheada y construcción local sin inicializar LLM |
| Models | `src/atlas/models/eda.py` | Reutilización de análisis existentes; muestra reproducible solo para visualización y conteo temporal |
| Controllers | `automatic_eda_controller.py` | Dos pasadas controladas, catálogo, presupuestos, caché por versión e informe |
| Controllers | `workflow_controller.py` | Detectar transiciones, emitir hechos y despachar al contrato común de los agentes |
| Controllers | `agent_controller.py`, `comparison_controller.py` | Construcción offline y registro de comparación realmente ejecutada |
| Agents | `src/atlas/agents/agent.py` | Recepción de eventos, prioridades, interpretación, objetivos candidatos y herramientas de lectura autorizadas |
| Services | `dataset_service.py`, `experiment_service.py`, `report_service.py` | Eventos ligados al ciclo real de datos, experimentos y reportes |
| Views | `components/intelligence_panel.py`, `components/agent_panel.py` | Panel reutilizable, detalles EDA y diálogo con confirmaciones existente |
| Views | `pages/workspace.py`, `pages/modeling.py`, `utils/state.py`, `styles/workspace.css`, `app.py` | Integración en las páginas, progreso y composición responsive |
| Tests | `test_automatic_eda.py`, `test_agent_events.py`, `test_intelligence_panel.py` | Resultados reales, caché, permisos, límites y continuidad |
| Herramientas | `tools/demo_event_workflow.py`, `tools/sidebar_browser_check.py` | Demostración reproducible y capturas Edge/CDP |

Otros archivos del árbol ya contenían cambios locales de las mejoras anteriores;
no se descartaron ni se publicaron automáticamente en GitHub.

## Arquitectura y contratos

Models conserva los cálculos. Los controladores coordinan métodos existentes y
eventos; las vistas presentan resultados. Todas las políticas agénticas siguen en
`agent.py`, con BaseAgent, EDAAgent, ClusteringAgent, ClassificationAgent,
RegressionAgent y ModelComparisonAgent.

`on_event()` es la interfaz común. `read_event_evidence()` admite únicamente
perfil, distribución de clases, métricas guardadas y comparación. Rechaza cualquier
entrenamiento o mutación. Se usa NoProvider para el circuito automático: no se
consulta una cuenta ni se inicializa un transporte de lenguaje. La conversación
opcional existente sigue necesitando una acción y consentimiento del usuario.

Los contratos científicos y académicos previos se conservan. Los identificadores y
docstrings nuevos son ingleses; se mantienen etiquetas históricas españolas de
tablas/métricas y mensajes de validación anteriores por compatibilidad. Esta entrega
no realiza una migración global de todos esos mensajes heredados.

## EDA automático real

Confirmar un archivo o ejemplo activa el EDA sin pulsar un botón adicional.

1. Pasada inicial: perfil, esquema, nulos, duplicados, tipos, memoria y estadísticas
   numéricas calculadas sobre todos los registros.
2. Pasada complementaria: percentiles, outliers IQR/Z-score, Isolation Forest cuando
   cabe en los límites, Pearson/Spearman/Kendall, VIF compatible, frecuencias,
   categorías escasas, contingencias, estadísticas agrupadas y un catálogo de
   gráficos compatibles. Se priorizan representaciones distintas antes de repetir
   gráficos para todas las variables.
3. Cada operación guarda `completed`, `skipped` o `failed`, con motivo y alcance.
   Un fallo local no convierte el resto del análisis en falso éxito.
4. Se genera un HTML con tablas, cobertura y gráficos Plotly exportados con runtime
   incluido, sin depender de una CDN. La descarga incorpora la interpretación y
   objetivos candidatos del agente cuando están disponibles.

En Exploración se pueden consultar los resultados calculados, gráficos y registro
de ejecución, descargar el informe y actualizar el análisis explícitamente.
El EDA manual y las pruebas avanzadas anteriores siguen disponibles.

Las pruebas de hipótesis no se ejecutan sin una pregunta y supuestos definidos.
No se inventan reglas de negocio para declarar inválidos precios, edades u otros
valores. Las contingencias describen asociaciones; no afirman causalidad.

## Rendimiento y persistencia

`EDALimits` permite configurar tiempo, columnas, categorías, pares y gráficos.
Valores predeterminados: 20 segundos para iniciar operaciones complementarias,
20 columnas numéricas en diagnósticos acotados, 12 categóricas/discretas, 6 pares,
6 gráficos y hasta 1000 puntos por gráfico. Kendall se limita a 3000 filas;
Isolation Forest y VIF tienen límites adicionales registrados en la cobertura.

El presupuesto temporal es cooperativo: se comprueba **entre operaciones**, no
interrumpe una llamada científica que ya está ejecutándose. La pasada inicial puede
superar ese tiempo. No hay un worker ni una tarea en segundo plano ficticia.
Existe cancelación programática antes de operaciones mediante
`eda_cancel_requested`; no se ofrece un botón que prometa detener inmediatamente
un cálculo en curso.

Las estadísticas usan todo el dataset. Los gráficos muestreados se etiquetan;
el subconjunto no sustituye las métricas ni modifica los datos. Se retienen como
máximo dos ejecuciones por versión en la sesión y 30 lecturas de evidencia. La
identidad incluye huella, ID y versión. Un rerun normal no repite el EDA.

Los resultados permanecen en memoria de la sesión. No se garantiza persistencia
tras reiniciar el servidor o perder la sesión. La muestra manual de Clustering
sigue separada del agente y se informa de esa limitación.

## Eventos y feedback

El enum incluye todos los eventos solicitados: carga/confirmación/cambio de datos,
inicio/fin/fallo de EDA, limpieza, selección de variables/objetivo/modelo,
inicio/fin de entrenamiento, evaluación, clustering, experimento guardado,
comparación, reporte y error de análisis. Los eventos de carga corresponden al
dataset validado que se entrega al servicio de activación; no a cada byte subido.

Se generan desde servicios/controladores. El coordinador observa transiciones de
selección y versión; no utiliza polling. Los hechos idénticos se deduplican; volver
a una selección anterior es una transición nueva, no un entrenamiento nuevo.
Los eventos de otra versión se marcan desactualizados.

El EDAAgent prioriza calidad y sugiere hasta tres objetivos por nombre/tipo/
cardinalidad, excluyendo identificadores, fechas y constantes reconocidos. No
selecciona el objetivo definitivamente. La detección es heurística y requiere
revisión del dominio, especialmente por fuga de información.

Los otros especialistas usan contexto y resultados existentes. Clasificación
comprueba la distribución real del objetivo y advierte sobre categorías únicas o
dominantes. Tras entrenar se consultan métricas guardadas. Comparación aplica el
comparador existente a cohortes compatibles y expone el criterio; no mezcla tareas
ni promete un ganador universal.

## Panel y autorización

En escritorio amplio aparecen navegación izquierda, espacio de trabajo y panel
derecho. Por debajo de 1450 px el panel se coloca debajo para no estrechar los
gráficos. Puede ocultarse y reabrirse conservando estado. La ayuda detallada se abre
en el diálogo existente y no modifica el contenido de la página por sí sola.

El panel muestra contexto, hallazgo, hasta tres ideas, evidencia y actividad real.
Los avisos pueden descartarse sin borrar resultados. Al cambiar de versión se
invalidan. No se muestran JSON ni nombres de herramientas en el panel normal.

Entrenar, optimizar, buscar grupos de forma costosa o aplicar limpieza sigue
requiriendo una acción explícita del usuario. El botón manual de entrenamiento
constituye esa autorización; las propuestas de los agentes utilizan su confirmación
separada. Las descargas y transmisiones tampoco se ejecutan automáticamente.

## Verificación realizada

- Base antes de cambios: **238 passed**, siete advertencias.
- Última batería completa integrada: **257 passed**, siete advertencias, **72.99 s**.
- Tras el ajuste final de exportación y actualización EDA: **14 passed** específicos.
- Ajuste final del control visual y evidencia visible: **9 passed** específicos.
- Ruff: sin errores en los archivos revisados.
- Edge/CDP: diez pantallas y cinco agentes, entrenamientos reales de ejemplo,
  escritorio de 1672 px y documento móvil de 390 px sin desbordamiento horizontal.
- Demostración ejecutable de cinco escenarios con CSV real, clasificación,
  comparación, clustering y regresión. Resultados en `event_demo_results.json`.

Las siete advertencias conocidas son una media sobre columna vacía y seis avisos de
convergencia de MLP con entrenamientos cortos de prueba. No se presentan como fallos
ocultos ni se suprimen para aparentar una ejecución limpia.

Las capturas están en `artifacts/atlas-2.0/` (ignoradas por Git), incluyendo
`intelligence-desktop.png`, páginas de EDA, resultados y asistentes móviles.
La revisión de controles no certifica cada combinación posible de archivo,
dependencia opcional, algoritmo e hiperparámetros. Las integraciones opcionales de
LLM/UMAP/K-Medoids/XGBoost dependen del entorno y no se certifican por esta entrega.

## Instalación, uso y extensión

Desde la raíz del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/demo_event_workflow.py
```

En el entorno de desarrollo actual el venv está en la carpeta superior:

```powershell
& '..\.venv\Scripts\python.exe' -m streamlit run app.py --server.port 8510
```

Para ampliar análisis, primero implemente o identifique el método en Models. Después
regístrelo en el controlador/gateway y añada compatibilidad y límites. Una política
nueva se añade únicamente a `agent.py`. Un evento nuevo pertenece a `core/events.py`
y se emite al completar una operación real, con pruebas de deduplicación y permisos.
