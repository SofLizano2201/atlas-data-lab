"""Sidebar Atlas: controles Streamlit reales, estado existente y SVG locales."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


@dataclass(frozen=True)
class NavItem:
    """Separar ruta existente de etiqueta e icono visibles."""

    route: str
    label: str
    icon: str
    slug: str


NAV_ITEMS = (
    NavItem("🏠 Inicio", "Inicio", "house", "home"),
    NavItem("📂 Datos", "Importación de datos", "database", "data"),
    NavItem("🔎 Exploración", "Exploración EDA", "search-check", "exploration"),
    NavItem("🧹 Preparación", "Limpieza y transformación", "sliders-horizontal", "preparation"),
    NavItem("🧩 Clustering", "Clustering", "network", "clustering"),
    NavItem("🎯 Clasificación", "Clasificación", "tags", "classification"),
    NavItem("📈 Regresión", "Regresión", "chart-no-axes-combined", "regression"),
    NavItem("⚖️ Experimentos", "Comparador de modelos", "flask-conical", "experiments"),
    NavItem("📤 Reportes", "Reportes", "file-chart-column", "reports"),
)

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "lucide"
SESSION_DETAIL = (
    "Los datos y resultados permanecen en memoria solo durante esta sesión. "
    "Exporta los datasets y experimentos antes de cerrar o reiniciar la aplicación."
)


def icon_svg(name: str) -> str:
    """Leer únicamente iconos locales del catálogo conocido."""
    if name not in {item.icon for item in NAV_ITEMS} | {"chevrons-left", "chevrons-right", "info"}:
        raise ValueError("Icono fuera del catálogo Atlas.")
    return (ASSET_ROOT / f"{name}.svg").read_text(encoding="utf-8")


def icon_mask(name: str) -> str:
    """URI SVG para la máscara CSS: color y tamaño se controlan con el tema."""
    return "data:image/svg+xml," + quote(icon_svg(name).replace("currentColor", "black"), safe="")


def navigate(route: str) -> None:
    """Cambiar la misma clave utilizada por el router y los accesos rápidos."""
    if route in {item.route for item in NAV_ITEMS}:
        st.session_state.navigation = route


def toggle_compact() -> None:
    """Alternar ancho sin reiniciar datos, modelos ni experimentos."""
    st.session_state.sidebar_compact = not st.session_state.get("sidebar_compact", False)


def sidebar_css(compact: bool) -> str:
    """Estilo de iconos por claves propias y ancho según estado de sesión."""
    rules = []
    for item in NAV_ITEMS:
        rules.append(
            f".st-key-atlas_nav_{item.slug} button::before {{"
            f'mask-image:url("{icon_mask(item.icon)}");'
            f'-webkit-mask-image:url("{icon_mask(item.icon)}");}}'
        )
    toggle_icon = "chevrons-right" if compact else "chevrons-left"
    rules.append(
        f'.st-key-atlas_sidebar_toggle button::before {{mask-image:url("{icon_mask(toggle_icon)}");'
        f'-webkit-mask-image:url("{icon_mask(toggle_icon)}");}}'
    )
    width = 72 if compact else 260
    rules.append(
        f'@media(min-width:769px){{[data-testid="stSidebar"]{{width:{width}px!important;'
        f"min-width:{width}px!important;max-width:{width}px!important;}}}}"
    )
    if compact:
        rules.append("""
        @media(min-width:769px) {
        .st-key-atlas_sidebar .atlas-wordmark,
        .st-key-atlas_sidebar .atlas-section-label,
        .st-key-atlas_sidebar .atlas-dataset-copy,
        .st-key-atlas_sidebar .atlas-session-copy {display:none;}
        .st-key-atlas_sidebar .atlas-brand {padding:0;justify-content:center;}
        .st-key-atlas_sidebar_header {padding-top:4px!important;min-height:92px;}
        [data-testid="stSidebarContent"] {padding-left:8px;padding-right:8px;}
        .st-key-atlas_sidebar_toggle {top:52px!important;left:50%;right:auto!important;transform:translateX(-50%);}
        .st-key-atlas_sidebar_nav button {justify-content:center!important;padding:0!important;gap:0!important;}
        .st-key-atlas_sidebar_nav button > div {flex:0!important;width:0!important;}
        .st-key-atlas_sidebar_nav button [data-testid="stMarkdownContainer"] {
            position:absolute;width:1px;height:1px;padding:0;overflow:hidden;
            clip:rect(0,0,0,0);white-space:nowrap;}
        .st-key-atlas_sidebar .atlas-dataset-card {padding:12px 0;justify-content:center;}
        .st-key-atlas_sidebar .atlas-session-note {padding:12px 0;justify-content:center;}
        }
        """)
    return "\n".join(rules)


def render_sidebar() -> tuple[str, DeltaGenerator]:
    """Construir navegación nativa y devolver el identificador de ruta existente."""
    # Reasignar desvincula la clave del antiguo radio y evita su limpieza automática.
    current = st.session_state.get("navigation", NAV_ITEMS[0].route)
    if current not in {item.route for item in NAV_ITEMS}:
        current = NAV_ITEMS[0].route
    st.session_state.navigation = current
    compact = st.session_state.get("sidebar_compact", False)
    st.html(f"<style>{sidebar_css(compact)}</style>")
    with st.sidebar, st.container(key="atlas_sidebar"):
        with st.container(key="atlas_sidebar_header"):
            st.markdown(
                '<div class="atlas-brand"><div class="atlas-brand-mark" aria-hidden="true">'
                + (ASSET_ROOT.parent / "atlas-mark.svg").read_text(encoding="utf-8")
                + '</div><div class="atlas-wordmark"><span class="atlas-product-name">ATLAS</span>'
                '<span class="atlas-product-subtitle">DATA LAB · EDA + ML</span></div></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="atlas_sidebar_toggle"):
                action = "Expandir menú" if compact else "Contraer menú"
                st.button(action, key="atlas_toggle", help=action, on_click=toggle_compact)
        st.markdown(
            '<div class="atlas-section-label">ESPACIO DE TRABAJO</div>', unsafe_allow_html=True
        )
        with st.container(key="atlas_sidebar_nav"):
            for item in NAV_ITEMS:
                active = current == item.route
                with st.container(key=f"atlas_nav_{item.slug}"):
                    st.button(
                        item.label,
                        key=f"atlas_page_{item.slug}",
                        type="primary" if active else "secondary",
                        help=f"{item.label} · sección actual" if active else item.label,
                        use_container_width=True,
                        on_click=navigate,
                        args=(item.route,),
                    )
        with st.container(key="atlas_sidebar_footer"):
            status = st.empty()
    return current, status


def render_sidebar_status(slot: DeltaGenerator) -> None:
    """Actualizar la tarjeta después de las acciones de la página en el mismo rerun."""
    name = st.session_state.get("dataset_name")
    title = escape(str(name) if name else "Carga un dataset para comenzar", quote=True)
    label = "DATASET ACTIVO" if name else "SIN DATASET"
    filename = escape(str(name) if name else "Sin datos activos")
    with slot.container():
        st.markdown(
            f'<div class="atlas-dataset-card" title="{title}" tabindex="0" aria-label="{label}: {title}">'
            '<span class="atlas-dataset-icon" aria-hidden="true">'
            + icon_svg("database")
            + f'</span><div class="atlas-dataset-copy"><span class="atlas-dataset-label">{label}</span>'
            f'<span class="atlas-dataset-name">{filename}</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="atlas-session-note" tabindex="0" title="{escape(SESSION_DETAIL, quote=True)}" '
            f'aria-label="{escape(SESSION_DETAIL, quote=True)}"><span aria-hidden="true">'
            + icon_svg("info")
            + '</span><span class="atlas-session-copy">Los datos se guardan solo durante esta sesión</span></div>',
            unsafe_allow_html=True,
        )
