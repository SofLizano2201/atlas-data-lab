# Agentes y orquestación de Atlas

## Separación de responsabilidades

La jerarquía académica permanece en `framework`. Los agentes no calculan métricas
ni implementan algoritmos: el registro `agents/tools.py` delega en los métodos de
EDA, Clustering, Clasificacion y Regresion. `contracts/models.py` contiene los nueve
contratos tipados compartidos y rechaza campos adicionales/coerciones ambiguas.

`AgentContext` pertenece a una sesión. Conserva referencias locales a los mismos
experimentos de Streamlit, artefactos, propuestas, planes y conversaciones.
`BaseAgent` valida especialidad, sesión, versión de dataset y argumentos antes de
ejecutar. Las cinco clases de `specialists.py` definen identidad, instrucciones y
herramientas permitidas. `frontend/components/agent_panel.py` renderiza los contratos.

| Agente | Herramientas principales |
|---|---|
| EDAAgent | profile, statistics, correlation, plot, test, preview, apply |
| ClusteringAgent | profile, reduce, distances, train, metrics |
| ClassificationAgent | profile, train, metrics, plot, importance |
| RegressionAgent | profile, train, metrics, plot, importance |
| ModelComparisonAgent | compare, metrics |

## Configurar un LLM

Sin configuración, aparece **Asistente conversacional no disponible** y se
deshabilita la conversación. Las acciones rápidas se identifican como deterministas
y ejecutan herramientas reales. No hay una respuesta simulada ni un modelo oculto.

Ejemplo en PowerShell, antes de iniciar Streamlit:

```powershell
$env:ATLAS_LLM_PROVIDER = 'responses'
$env:ATLAS_LLM_MODEL = '<modelo disponible en tu cuenta con JSON Schema>'
$env:ATLAS_LLM_API_KEY = '<tu clave privada>'
$env:ATLAS_LLM_BASE_URL = 'https://api.openai.com/v1'
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Introduce la clave en tu propia terminal; no la compartas en el chat ni la guardes
en archivos versionados. `.env.example` es una plantilla, no un cargador automático.
La configuración no utiliza la suscripción de ChatGPT como credencial API.

Para proveedores compatibles con Chat Completions y JSON Schema estricto:
`ATLAS_LLM_PROVIDER=chat_completions`, modelo y base URL del servidor correspondiente.
Para un servidor local se permite HTTP exclusivamente en localhost/127.0.0.1/::1.
Para un protocolo diferente, implementa `LLMProvider.plan()` y `.explain()` y pásalo
al constructor del agente. No necesitas modificar las clases académicas.

La documentación de [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
fundamenta el contrato JSON del adaptador Responses. El soporte del protocolo no
garantiza que todos los modelos/proveedores acepten ese esquema: los rechazos se
muestran como errores y no se sustituyen por respuestas ficticias.

## Flujo y permisos

1. Selecciona el botón del agente en la pantalla del módulo.
2. Usa una acción rápida con argumentos editables o, si existe proveedor, formula
   una pregunta y consiente el envío de contexto.
3. El LLM devuelve como máximo tres nombres de herramientas del registro y argumentos JSON.
4. El executor valida toda la propuesta antes de ejecutar cualquier herramienta.
5. Entrenar, reducir, calcular distancias/importancia y aplicar cambios requieren
   **Confirmar y ejecutar plan**. No basta con que el modelo escriba «confirmado».
6. El ticket se conserva solo en el servidor, pertenece a sesión/agente/versión,
   caduca a los cinco minutos y se consume una vez. Cambiar/restaurar datos invalida planes.
7. Las transformaciones primero generan una propuesta. «Preparar aplicación» rellena
   `eda.apply`; aún es necesario ejecutar esa acción y confirmar el cambio.
8. Los resultados numéricos y gráficos son objetos calculados. La prosa de un LLM
   se etiqueta como interpretación y no se utiliza para llenar tablas de métricas.

Nunca se ejecuta Python generado, `eval`, shell o importaciones suministradas por
el LLM. Las instrucciones incluidas en celdas no forman parte del prompt. Los
nombres de columnas son metadatos no confiables y no pueden ampliar permisos.

Con consentimiento se envían la pregunta, esquema (máximo 80 nombres/tipos), conteos,
identificadores de experimentos pertinentes y hasta seis interacciones previas ya
asistidas por LLM. Para explicar resultados se envían métricas y advertencias; no
tablas, filas, categorías, índices, modelos ni archivos. La API del proveedor puede
tener sus propias condiciones de retención; Responses utiliza `store=false`.
Se rechazan redirecciones HTTP para evitar reenviar Authorization a otro host.

## Uso desde Python

```python
from framework import AnalisisDatosExploratorio
from agents import AgentContext, EDAAgent
from contracts import ToolCall

context = AgentContext(AnalisisDatosExploratorio(df))
agent = EDAAgent(context)
request = agent.request("Diagnóstico", [ToolCall(tool="eda.profile")])
response = agent.run(request)
print(response.model_dump_json(indent=2))
```

El modo Python permite ejecutar acciones explícitas sin LLM. Si conectas un
proveedor, las llamadas conversacionales exigen `consent=True`. Para ejecutar un
plan sensible después de la aprobación humana: `agent.confirm(response.confirmation_id)`.
Ese método es una API local de confianza, no un endpoint público para el LLM.

## Trazabilidad y límites

Dataset original encapsulado y copias defensivas; hash de nombres/tipos/índice/valores,
versiones de trabajo y registro de operaciones aplicadas. Cada experimento incluye
dataset, variables, preprocesamiento, semilla, partición/índices, configuración,
versión del modelo y versiones de las bibliotecas. `Experiment.model_bytes()`
exporta el pipeline; la configuración puede exportarse separadamente.

Máximo 30 experimentos, 10 planes pendientes, 30 respuestas por especialidad y
500 decisiones de auditoría por contexto. Tablas de respuestas limitadas a 200 filas.
Se conservan hasta 100 artefactos y cinco propuestas, estas últimas con un total de
128 MB. El historial de datasets se limita a 256 MB y 20 versiones; KNN a 5000 filas.
El permiso de confirmación también considera el volumen de datos y las transformaciones
costosas, además de los efectos explícitos de cada herramienta.
No existe persistencia de conversaciones entre reinicios. Al activar otro dataset,
se invalida el contexto previo para evitar cruces de datos. Las operaciones son síncronas.

Los tests de proveedores usan un servidor HTTP local de prueba. No equivalen a una
llamada real a OpenAI u otro servicio. Sin una clave/modelo y consentimiento del
usuario no puede verificarse una inferencia real; debe declararse esa limitación.
