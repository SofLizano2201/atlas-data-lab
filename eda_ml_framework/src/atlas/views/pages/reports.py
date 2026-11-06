"""Spanish Streamlit views: rendering and input collection only."""

import streamlit as st

from atlas.controllers.eda_controller import EDAController
from atlas.controllers.report_controller import SECTIONS, ReportController, export_frame
from atlas.views.components.layout import algorithm_cards, page_header
from atlas.views.components.ui import figure


def reports_page() -> None:
    """Provide reports page using validated inputs."""
    page_header("reports")
    run = st.session_state.get("autonomous_runs", {}).get(st.session_state.get("autonomous_active"))
    if run and run.report:
        with st.container(border=True, key="workflow_autonomous_report"):
            st.subheader("Informe del recorrido automático")
            st.caption(
                "Incluye decisiones, variables, experimentos, advertencias y gráficos interactivos. Se descarga como HTML y funciona sin conexión."
            )
            st.download_button(
                "Descargar todo lo que analizó ATLAS",
                run.report,
                "atlas_analisis_automatico.html",
                "text/html",
                type="primary",
            )
    eda = EDAController(st.session_state.eda)
    st.subheader("Tipos de reportes")
    st.session_state.setdefault("atlas_report_type", "eda")
    algorithm_cards(
        [
            ("eda", "Reporte EDA", "Estructura, estadísticas y calidad."),
            ("cleaning", "Limpieza", "Cambios y versiones de los datos."),
            ("clustering", "Clustering", "Experimentos de agrupamiento."),
            ("clasificacion", "Clasificación", "Métricas de clasificación."),
            ("regresion", "Regresión", "Errores de predicción numérica."),
            ("comparison", "Experimentos", "Registro de modelos y configuraciones."),
        ],
        "atlas_report_type",
        multiple=False,
    )
    kind = st.session_state.atlas_report_type
    presets = {
        "eda": ["overview", "statistics", "quality"],
        "cleaning": ["overview", "quality", "cleaning"],
    }
    experiments = [
        e
        for e in st.session_state.experiments
        if kind not in {"clustering", "clasificacion", "regresion"} or e.problem == kind
    ]
    configuration, preview, actions = st.tabs(
        ["1. Configurar el informe", "2. Historial y descargas", "3. Exportar datos y gráficos"]
    )
    with configuration.container(border=True, key="atlas_card_report_config"):
        st.subheader("Configuración del reporte")
        selected = st.multiselect(
            "Secciones del informe",
            list(SECTIONS),
            default=presets.get(kind, ["overview", "experiments"]),
            format_func=SECTIONS.get,
        )
        title = st.text_input("Título del reporte", "Atlas · Informe de análisis", max_chars=160)
        fmt = st.radio("Formato del informe", ["html", "pdf"], horizontal=True)
        if st.button("Generar reporte", type="primary", disabled=not selected):
            with st.spinner("Generando documento desde resultados calculados…"):
                ReportController.generate_for_session(
                    st.session_state, eda, selected, fmt, title, experiments
                )
            st.success(
                "Informe listo. Abre «2. Historial y descargas» para consultarlo y descargarlo."
            )
    with preview.container(border=True, key="atlas_card_report_preview"):
        st.subheader("Vista previa e historial")
        reports = st.session_state.get("reports", [])
        if reports:
            report = st.selectbox(
                "Historial de reportes",
                list(reversed(reports)),
                format_func=lambda r: f"{r.title} · {r.format.upper()} · {r.created[:19]}",
            )
            st.caption(
                f"Dataset {report.dataset_hash[:12]} · secciones: {', '.join(report.sections)}"
            )
            if report.dataset_hash != eda.dataset_version.content_hash:
                st.info(
                    "Este documento corresponde a otra versión o dataset. Se conserva como instantánea; genera un nuevo reporte para los datos activos."
                )
            st.download_button(
                "Descargar reporte generado",
                report.data,
                f"atlas_{report.id[:8]}.{report.format}",
                "application/pdf" if report.format == "pdf" else "text/html",
            )
        else:
            st.info("Elige las secciones y genera tu primer reporte para descargarlo.")
        if reports:
            if report.format == "html":
                import streamlit.components.v1 as components

                components.html(report.data.decode("utf-8"), height=440, scrolling=True)
            else:
                st.caption("Documento PDF generado. Descárgalo para abrir todas sus páginas.")
    with actions.container(border=True, key="atlas_card_report_actions"):
        st.subheader("Exportación local")
        st.caption(
            "Los documentos contienen resultados de esta sesión. No se incluyen cifras simuladas."
        )
    with actions, st.expander("Exportar tablas y gráficos calculados"):
        fmt_table = st.selectbox("Formato de tabla", ["csv", "xlsx", "json", "parquet"])
        st.download_button(
            "Descargar dataset de trabajo", eda.export(fmt_table), f"datos.{fmt_table}"
        )
        if not eda.history.empty:
            st.download_button(
                "Historial de limpieza CSV", export_frame(eda.history), "limpieza.csv"
            )
        charts = st.session_state.graphs
        if charts:
            chart_key = st.selectbox("Gráfico de esta sesión", list(charts))
            figure(charts[chart_key], f"report_{chart_key}")
        st.caption(
            "Las figuras de agentes también pueden descargarse en sus paneles de asistencia."
        )
