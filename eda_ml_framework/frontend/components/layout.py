"""Composición compartida de los mockups, con acciones Streamlit reales."""

from html import escape

import streamlit as st

from frontend.components.agent_panel import render_agent_panel
from frontend.components.sidebar import NAV_ITEMS, navigate

PAGES = {
    "home": (
        "ATLAS Dashboard",
        "BIENVENIDO A ATLAS",
        "Framework de análisis de datos",
        (
            "Explora, modela y genera conocimiento a partir de tus datos con un flujo de trabajo "
            "completo y asistido por inteligencia artificial."
        ),
        "EDAAgent",
    ),
    "data": (
        "Importación de datos",
        "IMPORTACIÓN DE DATOS",
        "Importación de datos",
        "Carga tus datos desde un archivo local o un ejemplo y prepáralos para el análisis.",
        "EDAAgent",
    ),
    "exploration": (
        "Exploración EDA",
        "CONOCE TUS DATOS",
        "Exploración de datos",
        "Comprende la estructura, las distribuciones y las relaciones de tus variables.",
        "EDAAgent",
    ),
    "preparation": (
        "Limpieza y transformación",
        "PREPARA TUS DATOS",
        "Limpieza y transformación",
        "Mejora la calidad de tus datos con cambios previsualizados y versiones restaurables.",
        "EDAAgent",
    ),
    "clustering": (
        "Clustering y agrupamiento",
        "CLUSTERING",
        "Clustering y agrupamiento",
        "Descubre segmentos y patrones. Explora y evalúa distintos métodos de agrupamiento.",
        "ClusteringAgent",
    ),
    "classification": (
        "Clasificación",
        "ANÁLISIS DE CLASIFICACIÓN",
        "Clasificación",
        "Entrena y evalúa modelos para predecir clases con validación y asistencia contextual.",
        "ClassificationAgent",
    ),
    "regression": (
        "Regresión",
        "ANÁLISIS DE REGRESIÓN",
        "Regresión",
        "Modela relaciones entre variables y realiza predicciones numéricas a partir de tus datos.",
        "RegressionAgent",
    ),
    "experiments": (
        "Comparador de modelos",
        "COMPARADOR DE MODELOS",
        "Comparador de modelos",
        "Compara algoritmos y configuraciones bajo la misma evaluación y elige según tu métrica.",
        "ModelComparisonAgent",
    ),
    "reports": (
        "Reportes y exportación",
        "REPORTES",
        "Reportes y exportación",
        "Genera informes de tus análisis, personaliza su contenido y exporta resultados calculados.",
        None,
    ),
}


def page_header(page: str) -> None:
    """Encabezado, búsqueda de navegación y banner común sin datos ficticios."""
    title, eyebrow, heading, description, agent = PAGES[page]
    with st.container(key="atlas_topbar"):
        left, right = st.columns([2, 1], vertical_alignment="center")
        left.markdown(
            f'<div class="atlas-page-title">{escape(title)}</div>', unsafe_allow_html=True
        )
        search_key = f"atlas_search_{page}"

        def select_destination() -> None:
            destination = st.session_state[search_key]
            if destination is not None:
                navigate(destination.route)
                st.session_state[search_key] = None

        right.selectbox(
            "Buscar en ATLAS",
            [None, *NAV_ITEMS],
            format_func=lambda item: "Buscar en ATLAS…" if item is None else item.label,
            key=search_key,
            label_visibility="collapsed",
            on_change=select_destination,
        )
    with st.container(key="atlas_page_hero"):
        st.markdown(
            f'<div class="atlas-hero-copy"><div class="atlas-banner-eyebrow">{escape(eyebrow)}</div>'
            f"<h1>{escape(heading)}</h1><p>{escape(description)}</p></div>",
            unsafe_allow_html=True,
        )
        if page == "home":
            a, b, c = st.columns(3)
            a.button(
                "Cargar dataset", type="primary", on_click=navigate, args=(NAV_ITEMS[1].route,)
            )
            b.button("Explorar datos", on_click=navigate, args=(NAV_ITEMS[2].route,))
            with c:
                render_agent_panel("EDAAgent", button_only=True, label="Usar agente ATLAS")
        elif agent:
            render_agent_panel(agent, button_only=True)
    if agent and (page != "home" or "eda" not in st.session_state):
        render_agent_panel(agent, show_button=False)


def metric_cards(items: list[tuple[str, object, str]]) -> None:
    """Indicadores en una cuadrícula adaptable con tarjetas de igual altura."""
    if not items:
        return
    cards = []
    for label, value, detail in items:
        label, value, detail = (escape(str(item), quote=True) for item in (label, value, detail))
        cards.append(
            f'<article class="atlas-metric-item" aria-label="{label}">'
            f'<p class="atlas-metric-label">{label}</p>'
            f'<div class="atlas-card-value" title="{value}">{value}</div>'
            f'<p class="atlas-metric-detail">{detail}</p></article>'
        )
    st.markdown(
        '<div class="atlas-metrics-row">' + "".join(cards) + "</div>", unsafe_allow_html=True
    )


def algorithm_cards(
    items: list[tuple[str, str, str]], state_key: str, *, multiple: bool = True
) -> None:
    """Accesos de selección conectados al control del catálogo completo."""

    def select(algorithm: str) -> None:
        st.session_state[state_key] = [algorithm] if multiple else algorithm

    with st.container(key="atlas_algorithm_cards"):
        for column, (algorithm, label, description) in zip(st.columns(len(items)), items):
            with column.container(border=True, key=f"atlas_card_algorithm_{algorithm}"):
                selected = st.session_state.get(state_key, [])
                active = algorithm in selected if multiple else algorithm == selected
                st.button(
                    label,
                    key=f"choose_{state_key}_{algorithm}",
                    on_click=select,
                    args=(algorithm,),
                    type="primary" if active else "secondary",
                    use_container_width=True,
                )
                st.caption(description)
