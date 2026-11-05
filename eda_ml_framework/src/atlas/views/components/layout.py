"""Spanish Streamlit views: rendering and input collection only."""

from html import escape

import streamlit as st

from atlas.controllers.workspace_controller import WorkspaceController
from atlas.views.components.agent_panel import render_agent_panel
from atlas.views.components.sidebar import NAV_ITEMS, navigate

PAGES = {
    "history": (
        "Experimentos",
        "RESULTADOS",
        "Historial de experimentos",
        "Consulta modelos, datos de origen y configuraciones guardadas en esta sesión.",
        "ModelComparisonAgent",
    ),
    "home": (
        "ATLAS Dashboard",
        "BIENVENIDO A ATLAS",
        "Framework de análisis de datos",
        "Explora, modela y genera conocimiento a partir de tus datos con un flujo de trabajo completo y asistido por inteligencia artificial.",
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
    """Provide page header using validated inputs."""
    notice = st.session_state.pop("assistant_navigation_notice", None)
    if notice:
        st.info(notice)
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
            f'<div class="atlas-hero-copy"><div class="atlas-banner-eyebrow">{escape(eyebrow)}</div><h1>{escape(heading)}</h1><p>{escape(description)}</p></div>',
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
    context = WorkspaceController.context(st.session_state, title)
    if context:
        quality_class = (
            "attention" if context["missing_cells"] or context["additional_duplicates"] else "clear"
        )
        st.markdown(
            '<section class="atlas-context-summary" aria-label="Contexto del trabajo">'
            '<div class="atlas-context-identity"><span class="atlas-context-eyebrow">DATASET ACTIVO</span>'
            f'<strong title="{escape(str(context["name"]), quote=True)}">{escape(str(context["name"]))}</strong>'
            f'<span class="atlas-context-version">Versión {context["version"]}</span></div>'
            '<div class="atlas-context-facts">'
            f"<div><strong>{context['rows']:,}</strong><span>Filas</span></div>"
            f"<div><strong>{context['columns']}</strong><span>Variables</span></div>"
            f"<div><strong>{context['experiments']}</strong><span>Experimentos</span></div></div>"
            f'<div class="atlas-context-quality {quality_class}"><span class="atlas-context-eyebrow">CALIDAD DE LOS DATOS</span>'
            f"<strong>{'Hay datos por revisar' if quality_class == 'attention' else 'Sin nulos ni duplicados'}</strong>"
            f"<span>{context['missing_cells']:,} nulos ({context['missing_percentage']:.1f}%) · {context['additional_duplicates']:,} duplicados</span></div></section>",
            unsafe_allow_html=True,
        )
        stages = WorkspaceController.progress(st.session_state)
        current = {
            "data": "Datos",
            "exploration": "Exploración",
            "preparation": "Preparación",
            "clustering": "Modelado",
            "classification": "Modelado",
            "regression": "Modelado",
            "experiments": "Evaluación",
            "history": "Evaluación",
            "reports": "Exportación",
        }.get(page)
        st.markdown(
            '<div class="atlas-workflow-heading">TU FLUJO DE TRABAJO <span>Puedes navegar en cualquier orden</span></div>'
            '<div class="atlas-workflow" aria-label="Estado del flujo">'
            + "".join(
                f'<div class="atlas-step {"current" if stage == current else "done" if done else "pending"}"'
                + (' aria-current="step"' if stage == current else "")
                + f'><span class="atlas-step-number">{index}</span><div><strong>{escape(stage)}</strong>'
                f"<small>{'Estás aquí' if stage == current else 'Completado' if done else 'Opcional' if stage == 'Preparación' else 'Pendiente'}</small></div></div>"
                for index, (stage, done) in enumerate(stages.items(), 1)
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        next_slug = {
            "data": "exploration",
            "exploration": "preparation",
            "preparation": "clustering",
            "clustering": "history",
            "classification": "history",
            "regression": "history",
            "history": "experiments",
            "experiments": "reports",
        }.get(page)
        modeling_has_results = page not in {"clustering", "classification", "regression"} or any(
            result.problem
            == {
                "clustering": "clustering",
                "classification": "clasificacion",
                "regression": "regresion",
            }.get(page)
            for result in st.session_state.get("experiments", [])
        )
        if next_slug and modeling_has_results:
            destination = next(item for item in NAV_ITEMS if item.slug == next_slug)
            st.button(
                f"Siguiente: {destination.label}",
                key=f"next_{page}",
                on_click=navigate,
                args=(destination.route,),
                type="secondary",
            )
    if agent and (page != "home" or "eda" not in st.session_state):
        render_agent_panel(agent, show_button=False)


def metric_cards(items: list[tuple[str, object, str]]) -> None:
    """Provide metric cards using validated inputs."""
    if not items:
        return
    cards = []
    for label, value, detail in items:
        label, value, detail = (escape(str(item), quote=True) for item in (label, value, detail))
        cards.append(
            f'<article class="atlas-metric-item" aria-label="{label}"><p class="atlas-metric-label">{label}</p><div class="atlas-card-value" title="{value}">{value}</div><p class="atlas-metric-detail">{detail}</p></article>'
        )
    st.markdown(
        '<div class="atlas-metrics-row">' + "".join(cards) + "</div>", unsafe_allow_html=True
    )


def algorithm_cards(
    items: list[tuple[str, str, str]], state_key: str, *, multiple: bool = True
) -> None:
    """Provide algorithm cards using validated inputs."""

    def select(algorithm: str) -> None:
        st.session_state[state_key] = [algorithm] if multiple else algorithm

    with st.container(key="atlas_algorithm_cards"):
        columns = st.columns(2 if len(items) == 4 else min(3, len(items)))
        for index, (algorithm, label, description) in enumerate(items):
            column = columns[index % len(columns)]
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
