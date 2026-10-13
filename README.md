# Atlas · Data Lab / EDA + ML

Framework académico de análisis exploratorio de datos y Machine Learning con una
aplicación web Streamlit en español. Incluye limpieza con versiones, visualizaciones,
clustering, clasificación, regresión, comparación de experimentos y exportaciones.

El código completo está en [`eda_ml_framework/`](eda_ml_framework/).

## Instalación y ejecución

Python 3.11 recomendado. Desde la carpeta del repositorio:

```powershell
cd eda_ml_framework
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

En macOS/Linux, usa `python3 -m venv .venv` y `.venv/bin/python` en lugar del
ejecutable de Windows. Streamlit muestra la dirección local al iniciar.

## Documentación

- [Guía completa y uso desde Python](eda_ml_framework/README.md).
- [Arquitectura y extensión](eda_ml_framework/docs/architecture.md).
- [Catálogo de métodos](eda_ml_framework/docs/methods.md).
- [Guía del usuario](eda_ml_framework/docs/user_guide.md).
- [Verificación ejecutada](eda_ml_framework/docs/verification.md).
- [Sidebar y navegación](eda_ml_framework/docs/sidebar.md).

## Pruebas

Desde `eda_ml_framework`, con el entorno instalado:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

La entrega verificada registra 99 pruebas aprobadas. UMAP, K-Medoids y XGBoost son
integraciones opcionales; consulta las instrucciones y limitaciones en la guía completa.

Los datos de ejemplo son públicos de scikit-learn. El entorno virtual, credenciales,
modelos exportados, capturas de sesiones y archivos ZIP no forman parte del repositorio.
