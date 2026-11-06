# Atlas 2.0 · Data Lab

Framework académico modular de EDA y Machine Learning, con aplicación Streamlit en
español. La jerarquía del profesor tiene herencia real y responsabilidades funcionales:
EDA → NoSupervisado → Clustering; EDA → Supervisado → Clasificacion / Regresion.
Las clases funcionan desde Python sin importar Streamlit.

## Espacio de trabajo y agentes offline

Al confirmar datos, ATLAS ejecuta una exploración inicial con las operaciones
compatibles, límites explícitos y resultados guardados por versión. El panel
**Inteligencia ATLAS** permanece disponible al navegar y actualiza sus consejos
con eventos del flujo. No inicia entrenamientos automáticamente.
Consulta [la implementación, verificación y limitaciones](docs/event_intelligence.md).

Los cinco asistentes se abren en un diálogo contextual con dos o tres ideas basadas
en el dataset y los resultados actuales. Cada idea explica su utilidad y permite
continuar con una acción clara. Las herramientas técnicas quedan ocultas hasta
solicitar detalles. Consulta
[el flujo simplificado de asistencia](docs/assistant_ux.md).

La asistencia predeterminada funciona sin claves API ni servicios de pago: selecciona
una idea y el agente evalúa requisitos, llama una herramienta real y presenta
evidencia. Los entrenamientos y cambios sensibles requieren confirmación. Este motor
local usa políticas explícitas; la conversación generativa es opcional mediante un
modelo compatible, incluido Ollama instalado localmente.

La navegación agrupa Espacio de trabajo, Machine Learning y Resultados. Experimentos
y Comparador son pantallas diferentes. La barra de contexto y el progreso utilizan
datos reales; la preparación incluye impacto previo y deshacer confirmado. Los gráficos
ofrecen ajustes interactivos, selección, ampliación y exportación. No se reentrenan
modelos al cambiar una preferencia de visualización.

- [Guía 2.0, instalación, ejecución y limitaciones](docs/atlas_2_0.md).
- [Auditoría funcional y reparaciones](docs/audit_2_0.md).
- [Inventario de controles](docs/ui_controls_2_0.json).
- [Resultados de validación 2.0](docs/verification_2_0.md).
- `python tools/demo_offline.py`: demostración de los cinco agentes con datos públicos.

## Arquitectura MVC y agentes polimórficos

La implementación canónica está en `src/atlas/`: modelos científicos en `models/`,
coordinación en `controllers/`, interfaz española en `views/`, contratos y contexto
en `core/`, y ciclo de datasets, experimentos y reportes en `services/`.
**Cada agente tiene su propio módulo en `src/atlas/agents/`; `agent.py` conserva las importaciones antiguas.**
`framework/`, `frontend/`, `agents/` y `contracts/` en la raíz conservan importaciones
antiguas mediante adaptadores; no son implementaciones científicas paralelas.

Consulta [MVC y extensión](docs/mvc.md), [arquitectura](docs/architecture.md),
[configuración de agentes](docs/agents.md) e [inventario verificable](docs/capabilities_mvc.json).
El backend nuevo usa nombres ingleses; la API académica española sigue disponible.
Se conservan etiquetas históricas de métricas/tablas para mantener compatibilidad.

```python
from sklearn.datasets import load_iris
from atlas.models import ExploratoryDataAnalysis, Classification

data = load_iris(as_frame=True).frame
analysis = ExploratoryDataAnalysis(data)
print(analysis.profile())
assert issubclass(Classification, ExploratoryDataAnalysis)
```

Demostración reproducible de los cinco agentes, con cálculos reales sobre datasets
públicos y sin simular inferencia de un LLM:

```powershell
python tools/demo_agents.py --confirm-training
python tools/demo_event_workflow.py
python tools/audit_capabilities.py
python -m pytest -q
```

## Agentes, contratos y reportes

Se conservan las seis clases académicas. `agents/` incorpora EDAAgent,
ClusteringAgent, ClassificationAgent, RegressionAgent y ModelComparisonAgent.
`contracts/` define solicitudes, respuestas, llamadas/resultados de herramientas,
análisis, experimentos, modelos, decisiones y versiones de dataset con Pydantic.

Cada pantalla especializada incluye el botón de su agente. Sin proveedor/modelo
configurados, la conversación se muestra no disponible y las acciones deterministas
siguen funcionando. Un plan sensible necesita confirmación local; el LLM no puede
autorizarse a sí mismo ni ejecutar Python. La pregunta y el contexto reducido solo
se envían con consentimiento. No se envían filas completas ni tablas al proveedor.

Reportes ofrece HTML y PDF con selección de secciones e historial por sesión.
Los experimentos registran dataset, versiones, bibliotecas, variables y partición.
Consulta [agentes y configuración LLM](docs/agents.md), [reportes](docs/reports.md)
y [diagnóstico inicial](docs/diagnosis_2026_10_27.md).

La paleta se centraliza en `src/atlas/views/styles/tokens.css`, sobre la composición
de los ocho mockups proporcionados. Las adaptaciones se documentan en
`docs/mockups_2026_10_27.md`. Las fuentes Manrope/Inter usan alternativas del sistema
si no están instaladas; no se descargan automáticamente.

## Requisitos e instalación

Python 3.11–3.13. Recomendado 3.11; conexión para instalar paquetes. Ejemplos y ejecución
normal no requieren servicios externos. Desde la carpeta `eda_ml_framework`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

En macOS/Linux: `python3 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`,
`.venv/bin/python -m streamlit run app.py`.

En la entrega actual el entorno está en la carpeta superior. Desde esa carpeta:

```powershell
cd eda_ml_framework
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Para reproducir las versiones verificadas en un entorno nuevo, instala primero
`python -m pip install -r requirements-lock.txt` y después `python -m pip install -e .`.
El ZIP entregado contiene fuentes y documentación; no incluye el entorno virtual.

Abre la dirección local que imprime Streamlit (habitualmente http://localhost:8501).
En Windows también puedes ejecutar `powershell -ExecutionPolicy Bypass -File .\run.ps1`;
el lanzador busca el entorno virtual en el proyecto o su carpeta superior y no modifica
la política de ejecución del sistema.
La aplicación incluye navegación, estado por sesión, controles guiados, descargas,
gráficos interactivos y cambios confirmados con versiones.

## Estructura

```text
app.py                    Entrada Streamlit
src/atlas/models/         EDA, herencia académica, métricas y auxiliares
src/atlas/controllers/    Coordinación, validación y gateway de capacidades
src/atlas/views/          Páginas, componentes, estado visual y recursos
src/atlas/core/           Contratos tipados, contexto y dependencias
src/atlas/services/       Ciclo de datasets, experimentos y reportes
src/atlas/agents/         BaseAgent, cinco especialistas y módulos compartidos; ver docs/agents_structure.md
framework/, frontend/    Adaptadores de compatibilidad (también agents/, contracts/)
.streamlit/config.toml    Tema y límites del servidor
datasets/examples/        Generador reproducible de ejemplos
tests/                    Unitarias e integración Streamlit AppTest
docs/                     Arquitectura, catálogo, guía, referencias y verificación
pyproject.toml            Dependencias, empaquetado y configuración de pruebas
requirements.txt          Instalación editable con herramientas de prueba
```

## Uso desde Python

```python
from sklearn.datasets import load_iris, load_diabetes, make_blobs
import pandas as pd
from framework import (
    AnalisisDatosExploratorio,
    NoSupervisado,
    Supervisado,
    Clustering,
    Clasificacion,
    Regresion,
)

iris = load_iris(as_frame=True).frame
eda = AnalisisDatosExploratorio(iris)
print(eda.estadisticas())
propuesta = eda.transformar("standard", ["sepal length (cm)"])
eda.aplicar(propuesta, "Escalamiento exploratorio", confirmar=True)
eda.restaurar(0, confirmar=True)

no_sup = NoSupervisado(iris.drop(columns="target"))
pca = no_sup.reducir(no_sup.columnas(), "pca", dimensiones=2)
figura = no_sup.visualizar_representacion(pca["datos"])

sup = Supervisado(iris)
print(sup.identificar_problema("target"))
print(sup.nulos())  # Método heredado de EDA

x, _ = make_blobs(n_samples=100, centers=3, random_state=42)
clustering = Clustering(pd.DataFrame(x, columns=["x", "y"]))
grupos = clustering.entrenar("kmeans", ["x", "y"], n_clusters=3)
print(grupos.metrics)

clasificacion = Clasificacion(iris)
clas = clasificacion.entrenar("logistic", "target", list(iris.columns[:-1]), cv=3)
print(clas.metrics)
print(clasificacion.nulos())  # Herencia funcional, también en nietas

datos = load_diabetes(as_frame=True).frame
regresion = Regresion(datos)
predictores = [c for c in datos if c != "target"]
lineal = regresion.entrenar("linear_multiple", "target", predictores, cv=3)
ridge = regresion.entrenar("ridge_cv", "target", predictores, cv=3)
print(regresion.comparar())  # No reentrena

from pathlib import Path

Path("modelo.joblib").write_bytes(ridge.model_bytes())
Path("eda.html").write_text(eda.reporte(), encoding="utf-8")
```

`AnalisisDatosExploratorio.cargar_csv('datos.csv', separador=';', codificacion='latin-1')`
también acepta archivos/bytes para Excel, JSON y Parquet. Cada operación destructiva
requiere `confirmar=True`; las propuestas conservan el estado actual.

## Funcionalidades

- Estadística, faltantes, duplicados, constantes, outliers, tipos y versiones.
- Imputaciones, escaladores, codificaciones, potencias, discretización y exportación.
- Figuras Plotly y Matplotlib/Seaborn; correlación, VIF y pruebas con validaciones.
- PCA con varianza explicada, t-SNE, UMAP opcional y distancias.
- Diez familias de clustering con métricas válidas y manejo de ruido.
- Clasificación binaria/multiclase y todos los regresores solicitados.
- Pipeline/ColumnTransformer, CV por fold, búsqueda grid/random y reproducibilidad.
- Predicciones, residuos, curvas, coeficientes, importancia, aprendizaje y comparación.
- HTML autónomo, CSV/XLSX/JSON/Parquet, gráficos HTML/PNG, joblib y configuración JSON.

Consulta [catálogo de métodos](docs/methods.md), [guía del usuario](docs/user_guide.md),
[arquitectura y extensión](docs/architecture.md) y [análisis de las guías](docs/reference_analysis.md).

## Dependencias opcionales

```powershell
python -m pip install '.[umap]'
python -m pip install '.[xgboost]'
python -m pip install '.[medoids]'
```

No son necesarias para iniciar. Las selecciones muestran errores de instalación claros.
K-Medoids depende de extensiones binarias de scikit-learn-extra, que pueden ser
incompatibles con NumPy 2; usar entorno separado con NumPy 1.26 cuando corresponda.
XGBoost es distinto de Gradient Boosting. Las versiones CV de regularización usan
búsqueda del pipeline completo para evitar fuga dentro de folds internos.

## Pruebas

```powershell
python -m pytest -q
python -m ruff check .
```

En el entorno ya creado: `..\.venv\Scripts\python.exe -m pytest -q`.
Casos sintéticos y sklearn; carga de formatos, herencia, transformaciones, métricas,
algoritmos principales, ruido, errores, comparación, aislamiento train/test y acciones
reales de UI mediante AppTest. Los resultados ejecutados constan en
[verificación](docs/verification.md); las integraciones opcionales se distinguen.

## Limitaciones

- Aplicación de análisis local con sesiones en memoria; no incluye autenticación,
  despliegue multiusuario ni persistencia de experimentos entre reinicios.
- División aleatoria; no es adecuada para series temporales ni grupos repetidos.
- Los supuestos estadísticos dependen del diseño y no pueden garantizarse automáticamente.
- Se aplican límites de recursos documentados; datos de alta cardinalidad pueden requerir
  selección de variables o muestreo externo.
- No hay ejecución de código arbitrario, carga de modelos externos ni pickle subido.
- Los reportes HTML son de tablas; los gráficos se exportan separadamente.
- Coeficientes son del espacio transformado; representantes de clusters son medias.
- No se proporciona un modelo universalmente mejor; se compara según métrica elegida.

## Cómo extender

Agrega estimadores al registro de la clase especializada; conserva el pipeline común.
Agrega transformaciones validadas en EDA o funciones locales confiables mediante composición.
No introduzcas Streamlit en `framework`. Añade tests de casos normales, límites y
aislamiento antes de ampliar la UI. Los contratos completos están en arquitectura.md.

## Referencias visuales de ATLAS

La adaptación de los ocho mockups, los componentes modificados y las diferencias
restantes se describen en [la guía visual](docs/mockups_2026_10_27.md).


El [estándar de visualización](docs/visualization_standard.md) explica colores,
escalas, referencias estadísticas, límites y cómo verificar las figuras.

El [informe del rediseño de gráficos](docs/visualization_redesign.md) documenta
componentes, interacciones, correcciones, pruebas y límites de exportación.

El [recorrido autónomo](docs/autonomous_workflow.md) explica cómo ATLAS genera
gráficos, entrena alternativas, compara resultados y prepara un informe local
al confirmar un archivo, manteniendo el modo manual disponible.

La [guía de diseño de producto](docs/product_redesign.md) describe la composición
de las pantallas, los roles de color, estados interactivos y validación UX.
