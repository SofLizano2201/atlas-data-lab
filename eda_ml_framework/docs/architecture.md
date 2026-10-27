# Arquitectura y contratos

## Ampliación 1.1

`contracts/` aporta modelos Pydantic compartidos. `agents/` añade contexto privado,
proveedores configurables, registro cerrado de herramientas, orquestación y cinco
especialidades. Ninguna clase académica importa agentes ni depende de una API LLM.
EDA utiliza DatasetVersion para exponer procedencia; Experiment guarda versión del
modelo, huella de datos, índices de partición y versiones de bibliotecas.

La interfaz de agentes se concentra en `frontend/components/agent_panel.py` y utiliza
los mismos objetos/historiales que el flujo manual. `frontend/pages/reports.py`
utiliza `framework/reporting.py` para generar HTML/PDF y conservar documentos en sesión.
Los tokens de `frontend/styles/tokens.css` sustituyen colores de presentación sin
alterar el router ni los controles de modelado existentes.

Flujo del agente:

```text
Interfaz → AgentRequest → especialidad / BaseAgent
                        ├── LLMProvider (solo con configuración y consentimiento)
                        └── registro de herramientas y validación
                                  ↓
                        confirmación si hay efectos/coste
                                  ↓
                        métodos del framework existente
                                  ↓
                        ToolResult → AgentResponse → interfaz
```

Los contratos no contienen modelos ejecutables ni DataFrames: tablas JSON acotadas
y referencias a artefactos locales. El executor no evalúa código del LLM. Los tickets
son de un uso, con caducidad y ligados a la versión activa. Contextos se reinician
al activar un dataset diferente para impedir cruces accidentales de datos.

```text
AnalisisDatosExploratorio
├── NoSupervisado
│   └── Clustering
└── Supervisado
    ├── Clasificacion
    └── Regresion
```

EDA posee datos, validación, versiones, transformaciones y diagnósticos compartidos.
NoSupervisado usa esas validaciones y datos para construir matrices y representaciones.
Clustering reutiliza preparación, reducción y visualización. Supervisado hereda
exploración y añade un único ciclo de partición, pipeline, búsqueda y evaluación.
Sus hijas proporcionan catálogo de estimadores y métricas especializadas.
Cada constructor llama a `super().__init__`. Las hijas pueden usar `nulos()`,
`estadisticas()`, `transformar()` y todos los demás métodos EDA directamente.

## Contratos

- Constructor: DataFrame no vacío, nombres de columnas de texto únicos.
- `df`: copia defensiva. No existe setter que modifique silenciosamente el estado.
- `transformar`: nuevo DataFrame, registra propuesta; no modifica el actual.
- `aplicar(..., confirmar=True)`: valida y registra versión; restaurar también requiere confirmación.
- `cargar`: no acepta código, pickle ni joblib; lectura tabular limitada.
- `grafico`: devuelve Plotly, Figure de Matplotlib o PairGrid; nunca Streamlit.
- `preparar`: ajusta datos exploratorios no supervisados, nunca usado para partición supervisada.
- `entrenar`: devuelve Experiment con modelo, métricas, predicciones, configuración,
  tiempo total y hash de cohorte.
- `comparar`: solo datos, predictores, objetivo y partición compatibles; no vuelve a entrenar.
- `predecir`: valida columnas y utiliza transformadores ya ajustados.

Los auxiliares `preprocessing`, `metrics`, `visualization`, `results` evitan colocar
serialización, presentación o fórmulas ajenas a la jerarquía en las clases académicas.
Frontend llama al framework y solo organiza controles, estado y descargas.

Las figuras estadísticas usan el backend Agg de Matplotlib para evitar dependencias
de Tk/escritorio en servidores. La interacción visual proviene de Plotly. La exportación
de clustering incluye estimador, preprocesador y configuración en un diccionario joblib;
la jerarquía scipy exporta su matriz de enlaces, no un predictor de nuevos puntos.

## Aislamiento y seguridad

Datos privados y modelos viven en `st.session_state`, nunca en cachés globales.
Solo ejemplos públicos usan `st.cache_data`. No hay un recurso global compartido que
justifique `st.cache_resource`. Reiniciar el proceso/cerrar sesión puede perder datos.
No se ejecuta código subido ni se evalúan expresiones. La API de transformaciones
personalizadas acepta funciones Python del desarrollador de confianza, nunca texto del usuario.
No se cargan modelos externos. Fórmulas de Excel/CSV en textos se neutralizan al exportar.
HTML de reportes escapa contenido. El CSS estático usa testids semánticos y HTML propio.

Pipeline: split → imputación/escalamiento/codificación dentro de folds → selección
en entrenamiento → ajuste final en entrenamiento → evaluación sobre prueba.
La importancia en prueba es descriptiva: usarla para seleccionar variables vuelve a
utilizar el holdout y sesga la evaluación. CV de búsqueda es una medida de selección;
para estimar su generalización usa evaluación independiente o CV anidada externa.

## Extensión

Agregar un estimador en CLASSIFIERS/REGRESSORS/ALGORITHMS y su nombre aparece en la UI.
Para requerir parámetros guiados nuevos, extender `parameters()`. Para una métrica nueva,
añadirla a `evaluar()` o `metrics.py` con prueba de su definición y casos inválidos.
Para una transformación, añadir rama validada en `transformar()` y control de UI;
la confirmación/historial se conserva automáticamente. Para una nueva página, mantener
algoritmos en framework y agregar el enrutamiento en `app.py`.
