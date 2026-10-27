"""Flujos visuales de carga, exploración, limpieza y exportación."""

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from framework import AnalisisDatosExploratorio
from framework.results import html_report
from framework.visualization import before_after
from frontend.components.agent_panel import render_agent_panel
from frontend.components.layout import metric_cards, page_header
from frontend.components.sidebar import NAV_ITEMS, icon_svg, navigate
from frontend.components.ui import figure
from frontend.utils.state import activate, example


def overview() -> None:
    """Dashboard con navegación real y resumen del dataset."""
    page_header("home")
    if "eda" not in st.session_state:
        st.info("Tu espacio está listo. Carga un archivo o comienza con un dataset de ejemplo.")
        st.caption(
            "Cinco agentes especializados disponibles tras activar un dataset. "
            "Conversación requiere un proveedor LLM; cálculos manuales siguen disponibles."
        )
        return
    eda = st.session_state.eda
    df = eda.df
    numeric = list(df.select_dtypes(include=np.number).columns)
    metric_cards(
        [
            ("Dataset activo", st.session_state.dataset_name, "Datos privados de esta sesión"),
            (
                "Variables",
                len(df.columns),
                f"{len(numeric)} numéricas · {len(df.columns) - len(numeric)} restantes",
            ),
            ("Filas", f"{len(df):,}", "Registros totales"),
            (
                "Estado del proyecto",
                "En análisis",
                f"{len(st.session_state.experiments)} experimentos registrados",
            ),
        ]
    )
    st.subheader("Accesos rápidos")
    st.caption("Herramientas principales para tu flujo de análisis de datos.")
    with st.container(key="atlas_quicklinks"):
        for col, index, description in zip(
            st.columns(5),
            [2, 4, 5, 6, 8],
            [
                "Estructura, distribuciones y relaciones.",
                "Descubre grupos y patrones.",
                "Entrena y evalúa categorías.",
                "Realiza predicciones numéricas.",
                "Exporta informes y resultados.",
            ],
        ):
            item = NAV_ITEMS[index]
            with col.container(border=True, key=f"atlas_card_shortcut_{item.slug}"):
                st.markdown(
                    f'<div class="atlas-shortcut-icon">{icon_svg(item.icon)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{item.label}**")
                st.caption(description)
                st.button(
                    "Abrir",
                    key=f"shortcut_{item.slug}",
                    on_click=navigate,
                    args=(item.route,),
                    type="primary",
                )
    summary, assistants = st.columns([1.3, 1])
    with summary.container(border=True, key="atlas_card_summary"):
        st.subheader("Resumen del dataset")
        preview, stats, types, correlations = st.tabs(
            ["Vista previa", "Estadísticas", "Tipos de variables", "Correlaciones"]
        )
        preview.dataframe(eda.vista_previa(), use_container_width=True, hide_index=True)
        stats.dataframe(eda.resumen(), use_container_width=True)
        types.dataframe(eda.informacion(), use_container_width=True)
        if len(numeric) > 1:
            correlations.dataframe(df[numeric].corr(), use_container_width=True)
        else:
            correlations.caption("Se requieren al menos dos variables numéricas.")
    with assistants.container(border=True, key="atlas_card_assistants"):
        st.subheader("Asistentes inteligentes")
        st.caption("Herramientas reales; conversación disponible con un proveedor LLM configurado.")
        render_agent_panel("EDAAgent", show_button=False)
        for index, name in [
            (4, "ClusteringAgent"),
            (5, "ClassificationAgent"),
            (6, "RegressionAgent"),
            (7, "ModelComparisonAgent"),
        ]:

            def open_assistant(route: str, agent: str) -> None:
                st.session_state[f"agent_open_{agent}"] = True
                navigate(route)

            st.button(
                f"Agente · {NAV_ITEMS[index].label}",
                key=f"home_{name}",
                on_click=open_assistant,
                args=(NAV_ITEMS[index].route, name),
                use_container_width=True,
            )


def data_page() -> None:
    """Importación en dos paneles con validación y activación explícitas."""
    page_header("data")
    source, configuration = st.columns([1.65, 1])
    candidate = None
    with source.container(border=True, key="atlas_card_source"):
        st.subheader("Cargar datos")
        st.caption("Selecciona el origen de tus datos. Formatos: CSV, Excel, JSON y Parquet.")
        upload, samples = st.tabs(["Archivo local", "Dataset de ejemplo"])
        with upload:
            file = st.file_uploader(
                "Archivo tabular · máximo 25 MB", type=["csv", "xlsx", "json", "parquet"]
            )
            st.caption(
                "Los archivos se procesan en esta sesión y no se envían a un proveedor de IA."
            )
        with samples:
            name = st.selectbox(
                "Ejemplo", ["Iris", "Diabetes", "Cáncer de mama", "Clusters sintéticos"]
            )
            df = example(name)
            st.dataframe(df.head(5), use_container_width=True, hide_index=True)
            if st.button("Activar ejemplo", type="primary"):
                activate(df, name)
                st.success("Dataset activado.")
    with configuration.container(border=True, key="atlas_card_import_config"):
        st.subheader("Configuración de importación")
        a, b = st.columns(2)
        separator = a.selectbox("Separador CSV", [",", ";", "\t", "|"])
        encoding = b.selectbox("Codificación CSV", ["utf-8", "utf-8-sig", "latin-1", "cp1252"])
        sheet = 0
        if file and file.name.lower().endswith(".xlsx"):
            if file.size > 25 * 1024**2:
                st.error("El archivo supera 25 MB.")
                return
            from zipfile import BadZipFile, ZipFile

            try:
                with ZipFile(BytesIO(file.getvalue())) as archive:
                    if sum(i.file_size for i in archive.infolist()) > 100 * 1024**2:
                        st.error("Excel supera 100 MB descomprimido.")
                        return
            except BadZipFile as exc:
                raise ValueError(
                    "Archivo XLSX inválido; confirma que es un libro Excel válido."
                ) from exc
            sheet = st.selectbox(
                "Hoja de Excel", pd.ExcelFile(BytesIO(file.getvalue())).sheet_names
            )
        if file:
            candidate = AnalisisDatosExploratorio.cargar(
                file, separador=separator, codificacion=encoding, hoja=sheet
            )
            st.caption("Vista previa con la configuración actual")
            st.dataframe(
                candidate.vista_previa(), use_container_width=True, hide_index=True, height=190
            )
            if len(candidate.columnas()) == 1 and file.name.lower().endswith(".csv"):
                st.warning("Se detectó una columna. Confirma que el separador es correcto.")
            if st.button("Confirmar dataset activo", type="primary"):
                activate(candidate.df, file.name)
                st.success("Dataset activado.")
        else:
            st.info("Carga un archivo para revisar su lectura antes de confirmarlo.")
            st.caption(
                "Conexiones externas no configuradas. Utiliza un archivo local o un ejemplo."
            )
    visible = candidate or st.session_state.get("eda")
    if visible is not None:
        active = st.session_state.get("eda")
        confirmed = active is not None and (
            candidate is None
            or (
                st.session_state.get("dataset_name") == file.name
                and candidate.dataset_version.content_hash == active.dataset_version.content_hash
            )
        )
        frame = visible.df
        metric_cards(
            [
                (
                    "Estado del dataset",
                    "Activo" if confirmed else "Vista previa",
                    f"{len(frame):,} registros · {len(frame.columns)} variables",
                ),
                ("Valores faltantes", int(frame.isna().sum().sum()), "Celdas sin información"),
                (
                    "Duplicados encontrados",
                    int(frame.duplicated().sum()),
                    "Filas idénticas; aún sin eliminar",
                ),
            ]
        )
        preview, schema = st.columns([1.1, 1])
        with preview.container(border=True, key="atlas_card_import_preview"):
            st.subheader("Vista previa del dataset")
            st.dataframe(visible.vista_previa(), use_container_width=True, hide_index=True)
        with schema.container(border=True, key="atlas_card_schema"):
            st.subheader("Esquema y tipos de variables")
            st.dataframe(visible.informacion(), use_container_width=True)


def eda_page() -> None:
    """EDA, pruebas y figuras devueltas por el framework."""
    page_header("exploration")
    eda = st.session_state.eda
    tabs = st.tabs(["Panorama", "Estadísticas", "Calidad", "Visualizaciones", "Pruebas"])
    numeric = list(eda.df.select_dtypes(include=np.number).columns)
    with tabs[0]:
        st.dataframe(eda.informacion(), use_container_width=True)
        st.dataframe(eda.resumen(), use_container_width=True)
        st.write("Columnas constantes:", eda.constantes())
    with tabs[1]:
        if numeric:
            st.dataframe(eda.estadisticas(), use_container_width=True)
        column = st.selectbox("Analizar variable", eda.columnas())
        st.dataframe(eda.frecuencias(column), use_container_width=True)
        with st.expander("Categorías poco frecuentes y contingencia"):
            rare = st.slider("Frecuencia relativa mínima", 0.0, 1.0, 0.01)
            st.dataframe(eda.categorias_raras(column, rare), use_container_width=True)
            choices = [c for c in eda.columnas() if c != column]
            if choices:
                other = st.selectbox("Cruzar con", choices)
                if st.button("Crear tabla de contingencia"):
                    st.dataframe(eda.contingencia(column, other))
        if column in numeric:
            st.write("Percentiles", eda.percentiles(column, [5, 25, 50, 75, 95]))
            group = st.selectbox(
                "Agrupar por", ["Sin agrupar"] + [c for c in eda.columnas() if c != column]
            )
            if group != "Sin agrupar":
                st.dataframe(eda.estadisticas_agrupadas(group, column))
    with tabs[2]:
        st.dataframe(eda.nulos(), use_container_width=True)
        st.write("Filas duplicadas", len(eda.duplicados()))
        st.dataframe(eda.duplicados().head(100))
        st.dataframe(eda.inconsistencias())
        figure(eda.grafico("missing"), "missing")
        if numeric:
            outcols = st.multiselect("Variables para outliers", numeric, default=numeric[:1])
            method = st.selectbox("Detección", ["iqr", "zscore", "isolation"])
            threshold = st.number_input(
                "Umbral IQR/Z-score", 0.1, 10.0, 3.0 if method == "zscore" else 1.5
            )
            if st.button("Detectar outliers"):
                mask = eda.outliers(outcols, method, threshold)
                st.write("Filas detectadas:", int(mask.any(axis=1).sum()))
                st.dataframe(eda.df[mask.any(axis=1)].head(100))
    with tabs[3]:
        kind = st.selectbox(
            "Gráfico",
            [
                "histogram",
                "box",
                "violin",
                "scatter",
                "bar",
                "line",
                "pie",
                "correlation",
                "pairplot",
                "kde",
                "qq",
                "count",
                "outliers",
            ],
        )
        x = st.selectbox("Variable X", eda.columnas())
        y = st.selectbox(
            "Variable Y",
            ["Ninguna"] + eda.columnas(),
            disabled=kind not in {"scatter", "line"},
            help="Y se utiliza en dispersión y líneas. Los histogramas muestran frecuencias de X.",
        )
        color = st.selectbox(
            "Color por variable",
            ["Ninguna"] + eda.columnas(),
            help="Categorías: colores y símbolos. En dispersión, una variable numérica continua usa una escala con barra.",
        )
        corr = st.selectbox("Correlación", ["pearson", "spearman", "kendall"])
        if st.button("Generar gráfico", type="primary"):
            st.session_state.graphs["eda"] = eda.grafico(
                kind,
                x,
                None if y == "Ninguna" else y,
                None if color == "Ninguna" else color,
                metodo=corr,
            )
        if "eda" in st.session_state.graphs:
            figure(st.session_state.graphs["eda"], "eda")
        if len(numeric) >= 2 and st.button("Calcular VIF"):
            st.dataframe(eda.vif(numeric).to_frame())
    with tabs[4]:
        st.info(
            "Para pruebas de grupos, cada columna debe contener una muestra independiente. "
            "En datos pareados, este diseño no corresponde. Las pruebas son orientativas."
        )
        test = st.selectbox(
            "Prueba", ["shapiro", "normaltest", "levene", "ttest", "anova", "chi2", "mannwhitney"]
        )
        selected = st.multiselect("Muestras / variables", eda.columnas(), key="test_cols")
        alpha = st.number_input("Nivel alpha", 0.001, 0.2, 0.05)
        if st.button("Ejecutar prueba"):
            st.json(eda.prueba(test, selected, alpha))


def cleaning_page() -> None:
    """Propuesta, comparación, confirmación y restauración explícitas."""
    page_header("preparation")
    eda = st.session_state.eda
    st.info(
        "Estas transformaciones sirven para explorar. Para evaluar modelos, usa los datos "
        "originales y configura el pipeline en la pantalla de entrenamiento."
    )
    operations = {
        "Imputar media": "mean",
        "Imputar mediana": "median",
        "Imputar moda": "mode",
        "Imputar constante": "constant",
        "Imputación KNN": "knn",
        "Eliminar filas nulas": "drop_rows",
        "Eliminar columnas por nulos": "drop_null_columns",
        "Eliminar duplicados": "duplicates",
        "StandardScaler": "standard",
        "MinMaxScaler": "minmax",
        "RobustScaler": "robust",
        "Normalizer": "normalizer",
        "OneHotEncoder": "onehot",
        "OrdinalEncoder": "ordinal",
        "LabelEncoder (objetivo)": "label",
        "Logaritmo": "log",
        "Box-Cox": "boxcox",
        "Yeo-Johnson": "yeojohnson",
        "Discretización": "binning",
        "Seleccionar columnas": "select",
        "Eliminar columnas": "drop_columns",
        "Convertir tipo": "convert",
        "Normalizar nombres": "normalize_names",
        "Clipping IQR": "clip",
        "Winsorización": "winsor",
        "Eliminar outliers": "drop_outliers",
    }
    metric_cards(
        [
            ("Filas de trabajo", len(eda.df), "Registros de la versión activa"),
            ("Variables", len(eda.columnas()), "Columnas disponibles"),
            ("Nulos restantes", int(eda.df.isna().sum().sum()), "Celdas sin información"),
            ("Versión actual", f"v{eda.dataset_version.version}", "Historial restaurable"),
        ]
    )
    configuration, preview, history = st.columns([1, 1.8, 1])
    with configuration.container(border=True, key="atlas_card_clean_config"):
        st.subheader("Configuración de limpieza")
        name = st.selectbox("Operación", list(operations))
        selected = st.multiselect("Columnas", eda.columnas(), default=eda.columnas()[:1])
        op = operations[name]
        params = {}
        if op == "constant":
            mode = st.radio("Tipo del valor", ["Número", "Texto"], horizontal=True)
            params["valor"] = (
                st.number_input("Valor", value=0.0) if mode == "Número" else st.text_input("Valor")
            )
        if op in {"drop_null_columns", "drop_outliers"}:
            params["umbral"] = st.number_input(
                "Umbral", 0.01, 10.0, 0.5 if op == "drop_null_columns" else 1.5
            )
        if op == "drop_outliers":
            params["metodo"] = st.selectbox("Método", ["iqr", "zscore", "isolation"])
        if op == "winsor":
            params["limite"] = st.slider("Proporción por cola", 0.0, 0.49, 0.05)
        if op in {"knn", "binning"}:
            params["n_neighbors" if op == "knn" else "n_bins"] = st.number_input("Número", 2, 30, 5)
        if op == "convert":
            params["tipo"] = st.selectbox(
                "Tipo", ["float64", "int64", "string", "category", "datetime64[ns]"]
            )
        if st.button("Previsualizar transformación", type="primary"):
            st.session_state.pending = {
                "data": eda.transformar(op, selected, **params),
                "name": name,
                "columns": selected,
            }
    with preview.container(border=True, key="atlas_card_clean_preview"):
        st.subheader("Vista previa de los cambios")
        pending = st.session_state.pending
        if pending:
            a, b = st.columns(2)
            a.subheader("Antes")
            a.dataframe(eda.vista_previa())
            b.subheader("Después · " + pending["name"])
            b.dataframe(pending["data"].head(10))
            common = [
                c
                for c in pending["columns"]
                if c in pending["data"]
                and pd.api.types.is_numeric_dtype(pending["data"][c])
                and pd.api.types.is_numeric_dtype(eda.df[c])
            ]
            if common:
                figure(before_after(eda.df, pending["data"], common[0]), "before_after")
            if st.button("Confirmar estos cambios"):
                eda.aplicar(pending["data"], pending["name"], confirmar=True)
                st.session_state.work = eda.df
                st.session_state.pending = None
                st.session_state.graphs = {}
                st.rerun()
        if not pending:
            st.caption("Previsualiza una operación para comparar antes y después.")
            st.dataframe(eda.vista_previa(), use_container_width=True)
    with history.container(border=True, key="atlas_card_clean_history"):
        st.subheader("Historial y versiones")
        st.dataframe(eda.historial, use_container_width=True)
        version = st.selectbox("Versión a restaurar (0 = original)", range(eda.versiones))
        if st.button("Confirmar restauración"):
            eda.restaurar(version, confirmar=True)
            st.session_state.work = eda.df
            st.session_state.pending = None
            st.session_state.graphs = {}
            st.rerun()
    downloads(eda)


def downloads(eda: AnalisisDatosExploratorio) -> None:
    """Descargas del dataset y reportes de exploración/limpieza."""
    st.subheader("Exportar")
    fmt = st.selectbox("Formato de datos", ["csv", "xlsx", "json", "parquet"])
    st.download_button("Descargar dataset", eda.exportar(fmt), f"dataset.{fmt}")
    st.download_button("Reporte EDA HTML", eda.reporte(), "eda.html", "text/html")
    st.download_button(
        "Reporte de limpieza",
        html_report(
            "Atlas · Limpieza", {"Operaciones": eda.historial, "Estado actual": eda.informacion()}
        ),
        "limpieza.html",
        "text/html",
    )
