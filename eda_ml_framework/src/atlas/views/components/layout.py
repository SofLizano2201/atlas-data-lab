"""Spanish Streamlit views: rendering and input collection only."""

from html import escape

import streamlit as st

from atlas.controllers.workspace_controller import WorkspaceController
from atlas.views.components.agent_panel import render_agent_panel
from atlas.views.components.sidebar import NAV_ITEMS, navigate

PAGES = {
    "lab": ("Laboratorio científico", "INVESTIGACIÓN REPRODUCIBLE", "Investiga con evidencia",
            "Formula una pregunta, controla las condiciones y revisa la incertidumbre antes de concluir.", "ModelComparisonAgent"),
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

PAGE_GUIDES = {
    "lab": ("De una pregunta a una investigación reproducible",
            ["Define y valida el diseño", "Confirma las evaluaciones", "Interpreta y reproduce"]),
    "home": (
        "Tu espacio para convertir datos en decisiones",
        ["Carga tus datos", "Explora o deja que ATLAS analice", "Revisa y comparte"],
    ),
    "data": (
        "El punto de partida de tu análisis",
        ["Elige un archivo o ejemplo", "Revisa su contenido", "Confirma para comenzar"],
    ),
    "exploration": (
        "Descubre qué cuentan tus datos",
        ["Mira el resumen", "Explora una variable", "Encuentra relaciones y problemas"],
    ),
    "preparation": (
        "Prepara tus datos con confianza",
        ["Elige un cambio", "Compara antes y después", "Confirma o restaura"],
    ),
    "clustering": (
        "Encuentra registros que tienen algo en común",
        ["Elige características", "Busca grupos", "Comprende sus diferencias"],
    ),
    "classification": (
        "Aprende a predecir una categoría",
        ["Define qué quieres predecir", "Crea un modelo", "Revisa aciertos y errores"],
    ),
    "regression": (
        "Estima cantidades y comprende sus errores",
        ["Elige una cantidad", "Crea un modelo", "Revisa las predicciones"],
    ),
    "history": (
        "Todo lo que probaste, en un solo lugar",
        ["Busca un análisis", "Revisa cómo se evaluó", "Consulta o exporta resultados"],
    ),
    "experiments": (
        "Elige con evidencia, no por intuición",
        ["Selecciona pruebas compatibles", "Elige qué importa", "Compara ventajas y límites"],
    ),
    "reports": (
        "Dale forma a tus hallazgos",
        ["Elige el contenido", "Genera el informe", "Descarga y conserva"],
    ),
}


def page_header(page: str) -> None:
    """Provide page header using validated inputs."""
    notice = st.session_state.pop("assistant_navigation_notice", None)
    if notice:
        st.info(notice)
    title, eyebrow, _heading, description, agent = PAGES[page]
    with st.container(key="atlas_topbar"):
        left, right, settings = st.columns([2, 1.3, 0.7], vertical_alignment="center")
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
        with settings.popover("Vista", use_container_width=True):
            st.caption("Personaliza el espacio sin cambiar tus datos.")
            st.toggle("Mostrar la guía de la sección", key="view_show_guide", value=False)
            st.toggle("Mostrar todo el contexto", key="view_show_context", value=False)
            st.toggle("Reducir movimiento", key="view_reduce_motion", value=False)
        if st.session_state.get("view_reduce_motion"):
            st.html(
                "<style>*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important;}</style>"
            )
    with st.container(key="atlas_page_hero"):
        purpose, steps = PAGE_GUIDES[page]
        st.markdown(
            f'<div class="atlas-hero-copy"><div class="atlas-banner-eyebrow">{escape(eyebrow)}</div><h1>{escape(purpose)}</h1><p>{escape(description)}</p></div>',
            unsafe_allow_html=True,
        )
        with st.expander(
            "Cómo aprovechar esta sección", expanded=st.session_state.get("view_show_guide", False)
        ):
            for index, step in enumerate(steps, 1):
                st.markdown(f"**{index}.** {step}")
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
        st.markdown(
            '<div class="atlas-workspace-ribbon" aria-label="Resumen del espacio de trabajo">'
            f"<strong>{escape(str(context['name']))}</strong><span>v{context['version']}</span>"
            f"<span>{context['rows']:,} filas</span><span>{context['columns']} variables</span>"
            f"<span>{context['experiments']} análisis guardados</span>"
            f"<span>{context['missing_cells']:,} nulos · {context['additional_duplicates']:,} duplicados</span></div>",
            unsafe_allow_html=True,
        )
        with st.expander(
            "Estado del dataset y próximos pasos",
            expanded=st.session_state.get("view_show_context", False),
        ):
            workspace_status(context, page)
    if agent and (page != "home" or "eda" not in st.session_state):
        render_agent_panel(agent, show_button=False)


def workspace_status(context, page):
    """Render factual dataset status and direct workflow navigation."""
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
    st.caption("Tu flujo de trabajo · puedes navegar en cualquier orden")
    destinations = {
        "Datos": "data",
        "Exploración": "exploration",
        "Preparación": "preparation",
        "Modelado": {"classification": "classification", "regression": "regression"}.get(
            page, "clustering"
        ),
        "Evaluación": "history",
        "Exportación": "reports",
    }
    with st.container(key="atlas_workflow_links"):
        for column, (stage, done) in zip(st.columns(6), stages.items()):
            slug = destinations[stage]
            destination = next(item for item in NAV_ITEMS if item.slug == slug)
            column.button(
                stage,
                key=f"flow_{page}_{slug}",
                on_click=navigate,
                args=(destination.route,),
                type="primary" if stage == current else "secondary",
                use_container_width=True,
            )
            column.caption(
                "Estás aquí"
                if stage == current
                else "Completado"
                if done
                else "Opcional"
                if stage == "Preparación"
                else "Pendiente"
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
