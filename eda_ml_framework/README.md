# Atlas · Data Lab

Framework académico modular de EDA y Machine Learning, con aplicación Streamlit en
español. La jerarquía del profesor tiene herencia real y responsabilidades funcionales:
EDA → NoSupervisado → Clustering; EDA → Supervisado → Clasificacion / Regresion.
Las clases funcionan desde Python sin importar Streamlit.

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
framework/                EDA, clases académicas, métricas y auxiliares
frontend/pages/           Carga, EDA, preparación, modelos y comparación
frontend/components/      Figuras y descargas compartidas
frontend/utils/           Estado y ejemplos públicos cacheados
frontend/styles/          Tema CSS de la paleta solicitada
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
