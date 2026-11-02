"""Spanish Streamlit views: rendering and input collection only."""

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from atlas.controllers.charts import before_after
from atlas.controllers.dataset_controller import DatasetController
from atlas.controllers.eda_controller import EDAController
from atlas.controllers.report_controller import html_report
from atlas.views.components.agent_panel import render_agent_panel
from atlas.views.components.layout import metric_cards, page_header
from atlas.views.components.sidebar import NAV_ITEMS, icon_svg, navigate
from atlas.views.components.ui import figure
from atlas.views.utils.state import activate, example


def overview() -> None:
    """Provide overview using validated inputs."""
    page_header("home")
    if "eda" not in st.session_state:
        st.info("Tu espacio está listo. Carga un archivo o comienza con un dataset de ejemplo.")
        st.caption(
            "Cinco agentes con decisiones locales disponibles tras activar un dataset. No necesitas API ni un servicio de pago."
        )
        return
    eda = EDAController(st.session_state.eda)
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
        preview.dataframe(eda.preview(), use_container_width=True, hide_index=True)
        stats.dataframe(eda.summary(), use_container_width=True)
        types.dataframe(eda.information(), use_container_width=True)
        if len(numeric) > 1:
            correlations.dataframe(eda.correlation(), use_container_width=True)
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
    """Provide data page using validated inputs."""
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
            with st.expander("Vista previa del ejemplo"):
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)
            if st.button("Activar ejemplo", type="primary"):
                activate(df, name)
                st.success("Dataset activado.")
    with configuration.container(border=True, key="atlas_card_import_config"):
        st.subheader("Configuración de importación")
        a, b = st.columns(2)
        separator, encoding = ",", "utf-8"
        if file and file.name.lower().endswith(".csv"):
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
            candidate = DatasetController.load(
                file, separator=separator, text_encoding=encoding, sheet=sheet
            )
            st.caption(
                f"Lectura validada: {file.name}. Revisa la tabla y el esquema debajo antes de confirmar."
            )
            if len(candidate.columns()) == 1 and file.name.lower().endswith(".csv"):
                st.warning("Se detectó una columna. Confirma que el separador es correcto.")
            if st.button("Confirmar dataset activo", type="primary"):
                activate(candidate.df, file.name)
                st.success("Dataset activado.")
        else:
            st.info("Carga un archivo para revisar su lectura antes de confirmarlo.")
            st.caption(
                "Conexiones externas no configuradas. Utiliza un archivo local o un ejemplo."
            )
    visible = candidate or (
        EDAController(st.session_state.eda) if "eda" in st.session_state else None
    )
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
                ("Valores faltantes", visible.profile()["missing_cells"], "Celdas sin información"),
                (
                    "Duplicados encontrados",
                    visible.profile()["additional_duplicates"],
                    "Filas idénticas; aún sin eliminar",
                ),
            ]
        )
        preview, schema = st.columns([1.1, 1])
        with preview.container(border=True, key="atlas_card_import_preview"):
            st.subheader("Vista previa del dataset")
            st.dataframe(visible.preview(), use_container_width=True, hide_index=True)
        with schema.container(border=True, key="atlas_card_schema"):
            st.subheader("Esquema y tipos de variables")
            st.dataframe(visible.information(), use_container_width=True)


def eda_page() -> None:
    """Provide eda page using validated inputs."""
    page_header("exploration")
    eda = EDAController(st.session_state.eda)
    profile = eda.profile()
    metric_cards(
        [
            ("Dimensiones", f"{profile['rows']:,} × {profile['columns']}", "Filas y columnas"),
            ("Faltantes", f"{profile['missing_percentage']:.2f}%", "Porcentaje de celdas"),
            ("Duplicados", profile["additional_duplicates"], "Filas adicionales idénticas"),
            (
                "Tipos",
                f"{profile['numeric_columns']} numéricas",
                f"{profile['categorical_columns']} no numéricas",
            ),
        ]
    )
    tabs = st.tabs(
        [
            "Resumen",
            "Variables",
            "Calidad de datos",
            "Distribuciones y relaciones",
            "Pruebas avanzadas",
        ]
    )
    numeric = list(eda.df.select_dtypes(include=np.number).columns)
    with tabs[0]:
        st.dataframe(eda.information(), use_container_width=True)
        st.dataframe(eda.summary(), use_container_width=True)
        st.session_state.eda_reviewed = True
        st.write("Columnas constantes:", eda.constant_columns())
    with tabs[1]:
        if numeric:
            st.dataframe(eda.statistics(), use_container_width=True)
        column = st.selectbox("Analizar variable", eda.columns())
        st.dataframe(eda.frequencies(column), use_container_width=True)
        with st.expander("Categorías poco frecuentes y contingencia"):
            rare = st.slider("Frecuencia relativa mínima", 0.0, 1.0, 0.01)
            st.dataframe(eda.rare_categories(column, rare), use_container_width=True)
            choices = [c for c in eda.columns() if c != column]
            if choices:
                other = st.selectbox("Cruzar con", choices)
                if st.button("Crear tabla de contingencia"):
                    st.dataframe(eda.contingency(column, other))
        if column in numeric:
            st.write("Percentiles", eda.percentiles(column, [5, 25, 50, 75, 95]))
            group = st.selectbox(
                "Agrupar por", ["Sin agrupar"] + [c for c in eda.columns() if c != column]
            )
            if group != "Sin agrupar":
                st.dataframe(eda.grouped_statistics(group, column))
    with tabs[2]:
        st.dataframe(eda.missing_values(), use_container_width=True)
        st.write("Filas duplicadas", len(eda.duplicates()))
        st.dataframe(eda.duplicates().head(100))
        st.dataframe(eda.type_inconsistencies())
        figure(eda.chart("missing"), "missing")
        if numeric:
            outcols = st.multiselect("Variables para outliers", numeric, default=numeric[:1])
            method = st.selectbox("Detección", ["iqr", "zscore", "isolation"])
            threshold = st.number_input(
                "Umbral IQR/Z-score", 0.1, 10.0, 3.0 if method == "zscore" else 1.5
            )
            if st.button("Detectar outliers"):
                diagnostic = eda.outlier_diagnostic(outcols, method, threshold)
                st.write("Filas detectadas:", diagnostic["count"])
                st.dataframe(diagnostic["rows"])
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
        x_choices = (
            numeric
            if kind in {"histogram", "box", "violin", "kde", "qq", "outliers"}
            else eda.columns()
        )
        x = st.selectbox("Variable X", x_choices or eda.columns())
        y = st.selectbox(
            "Variable Y",
            ["Ninguna"] + eda.columns(),
            disabled=kind not in {"scatter", "line"},
            help="Y se utiliza en dispersión y líneas. Los histogramas muestran frecuencias de X.",
        )
        color = st.selectbox(
            "Color por variable",
            ["Ninguna"] + eda.columns(),
            help="Categorías: colores y símbolos. En dispersión, una variable numérica continua usa una escala con barra.",
        )
        corr = st.selectbox("Correlación", ["pearson", "spearman", "kendall"])
        options = {}
        with st.expander("Ajustes de visualización"):
            automatic = st.checkbox(
                "Actualizar al cambiar variables", value=True, key="eda_automatic"
            )
            if kind == "histogram":
                options["nbins"] = st.slider(
                    "Intervalos del histograma", 5, 150, 30, key="eda_bins"
                )
            if kind in {"histogram", "scatter", "line"}:
                options["log_x"] = st.checkbox("Escala X logarítmica", key="eda_log_x")
                if kind in {"scatter", "line"}:
                    options["log_y"] = st.checkbox("Escala Y logarítmica", key="eda_log_y")
                facet = st.selectbox(
                    "Separar en paneles", ["Ninguna"] + eda.columns(), key="eda_facet"
                )
                options["facet"] = None if facet == "Ninguna" else facet
        st.session_state.eda_selection = {
            "kind": kind,
            "x": x,
            "y": y,
            "color": color,
            "correlation": corr,
        }
        chart_request = (
            eda.dataset_version.content_hash,
            kind,
            x,
            y,
            color,
            corr,
            tuple(options.items()),
        )
        can_plot = kind not in {"scatter", "line"} or y != "Ninguna"
        if not can_plot:
            st.caption("Selecciona una variable Y para visualizar esta relación.")
        if st.button("Generar gráfico", type="primary", disabled=not can_plot) or (
            automatic
            and can_plot
            and st.session_state.get("eda_chart_request") != chart_request
            and kind not in {"pairplot", "kde", "qq"}
        ):
            st.session_state.graphs["eda"] = eda.chart(
                kind,
                x,
                None if y == "Ninguna" else y,
                None if color == "Ninguna" else color,
                method=corr,
                **options,
            )
            st.session_state.eda_chart_request = chart_request
        if "eda" in st.session_state.graphs:
            figure(st.session_state.graphs["eda"], "eda")
        if len(numeric) >= 2 and st.button("Calcular VIF"):
            st.dataframe(eda.vif(numeric).to_frame())
    with tabs[4], st.expander("Pruebas estadísticas y supuestos", expanded=False):
        st.info(
            "Para pruebas de grupos, cada columna debe contener una muestra independiente. En datos pareados, este diseño no corresponde. Las pruebas son orientativas."
        )
        test = st.selectbox(
            "Prueba", ["shapiro", "normaltest", "levene", "ttest", "anova", "chi2", "mannwhitney"]
        )
        selected = st.multiselect("Muestras / variables", eda.columns(), key="test_cols")
        alpha = st.number_input("Nivel alpha", 0.001, 0.2, 0.05)
        if st.button("Ejecutar prueba"):
            st.json(eda.statistical_test(test, selected, alpha))


def cleaning_page() -> None:
    """Provide cleaning page using validated inputs."""
    page_header("preparation")
    eda = EDAController(st.session_state.eda)
    st.info(
        "Estas transformaciones sirven para explorar. Para evaluar modelos, usa los datos originales y configura el pipeline en la pantalla de entrenamiento."
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
            ("Variables", len(eda.columns()), "Columnas disponibles"),
            ("Nulos restantes", eda.profile()["missing_cells"], "Celdas sin información"),
            ("Versión actual", f"v{eda.dataset_version.version}", "Historial restaurable"),
        ]
    )
    configuration, preview, history = st.columns([1, 1.8, 1])
    with configuration.container(border=True, key="atlas_card_clean_config"):
        st.subheader("Configuración de limpieza")
        name = st.selectbox("Operación", list(operations))
        selected = st.multiselect("Columnas", eda.columns(), default=eda.columns()[:1])
        op = operations[name]
        st.session_state.eda_selection = {"operation": op, "columns": selected}
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
                "data": eda.transform(op, selected, **params),
                "name": name,
                "columns": selected,
            }
    with preview.container(border=True, key="atlas_card_clean_preview"):
        st.subheader("Vista previa de los cambios")
        pending = st.session_state.pending
        if pending:
            impact = eda.transformation_impact(pending["data"])
            st.caption(
                f"Impacto previsto: {impact['rows_before']:,} → {impact['rows_after']:,} filas · {impact['missing_before']} → {impact['missing_after']} nulos · {len(impact['changed_columns'])} columnas afectadas. Sin aplicar todavía."
            )
            a, b = st.columns(2)
            a.subheader("Antes")
            a.dataframe(eda.preview())
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
                eda.apply(pending["data"], pending["name"], confirm=True)
                st.session_state.work = eda.df
                st.session_state.pending = None
                st.session_state.graphs = {}
                st.rerun()
        if not pending:
            st.caption("Previsualiza una operación para comparar antes y después.")
            st.dataframe(eda.preview(), use_container_width=True)
    with history.container(border=True, key="atlas_card_clean_history"):
        st.subheader("Historial y versiones")
        st.dataframe(eda.history, use_container_width=True)
        version = st.selectbox("Versión a restaurar (0 = original)", range(eda.version_count))
        if eda.version_count > 1 and st.button("Confirmar deshacer último cambio"):
            eda.restore(eda.version_count - 2, confirm=True)
            st.session_state.work = eda.df
            st.session_state.pending = None
            st.session_state.graphs = {}
            st.rerun()
        if st.button("Confirmar restauración"):
            eda.restore(version, confirm=True)
            st.session_state.work = eda.df
            st.session_state.pending = None
            st.session_state.graphs = {}
            st.rerun()
    downloads(eda)


def downloads(eda: DatasetController) -> None:
    """Provide downloads using validated inputs."""
    st.subheader("Exportar")
    fmt = st.selectbox("Formato de datos", ["csv", "xlsx", "json", "parquet"])
    st.download_button("Descargar dataset", eda.export(fmt), f"dataset.{fmt}")
    st.download_button("Reporte EDA HTML", eda.report(), "eda.html", "text/html")
    st.download_button(
        "Reporte de limpieza",
        html_report(
            "Atlas · Limpieza", {"Operaciones": eda.history, "Estado actual": eda.information()}
        ),
        "limpieza.html",
        "text/html",
    )
