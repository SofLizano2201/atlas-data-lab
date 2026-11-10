"""Spanish presentation of structured scientific findings."""

from html import escape

import streamlit as st

from atlas.controllers.methodological_controller import MethodologicalController
from atlas.views.components.sidebar import icon_svg


def confirm_correction():
    try:
        MethodologicalController.correct_target_selection(st.session_state, confirm=True)
        st.session_state.pop("methodological_pending_correction", None)
        st.session_state.pop("methodological_correction_error", None)
    except ValueError:
        st.session_state["methodological_correction_error"] = (
            "Selecciona al menos un predictor válido y revisa la configuración en el formulario."
        )


def render_methodological_feedback():
    findings = st.session_state.get("methodological_findings", [])
    st.html(
        '<div class="atlas-panel-section">'
        + icon_svg("search-check")
        + "<strong>Guardia metodológica</strong></div>"
    )
    st.caption("Verificación científica contextual")
    if st.session_state.get("methodological_correction_error"):
        st.info(st.session_state["methodological_correction_error"])
    if not findings:
        st.caption(
            "Sin hallazgos en las verificaciones disponibles. Esto no garantiza ausencia de riesgos."
        )
    acknowledged = st.session_state.setdefault("methodological_acknowledged", [])
    for finding in findings:
        key = st.session_state.get("methodological_key", "") + finding.finding_id
        if key in acknowledged:
            continue
        with st.container(border=True):
            label = {
                "blocking": "Es necesario corregir",
                "warning": "Conviene revisar",
                "suggestion": "Sugerencia",
            }[finding.severity]
            st.html(
                f'<div style="color:#331832;border-left:3px solid #694D75;padding-left:12px">'
                f"<strong>{label}: {escape(finding.title)}</strong></div>"
            )
            st.write(finding.recommended_action)
            with st.expander("Ver explicación"):
                st.write(finding.explanation)
                st.caption(
                    "La ejecución está bloqueada." if finding.blocking else "Puedes continuar."
                )
            if st.button("Revisar", key="review_" + key):
                st.session_state["methodological_review"] = finding.affected_component
                st.info("Revisa la selección en los controles del análisis actual.")
            if finding.finding_id == "target_leakage":
                if st.button("Aplicar corrección", key="correct_" + key):
                    st.session_state["methodological_pending_correction"] = key
                if st.session_state.get("methodological_pending_correction") == key:
                    st.caption(
                        "Se excluirá el objetivo de los predictores. No se modificarán filas ni valores."
                    )
                    st.button(
                        "Confirmar corrección",
                        key="confirm_guardian_" + key,
                        on_click=confirm_correction,
                    )
            if not finding.blocking and st.button("Continuar de todos modos", key="ack_" + key):
                acknowledged.append(key)
                st.rerun()
    with st.expander("Historial de verificaciones"):
        for _, version, items in reversed(st.session_state.get("methodological_history", [])):
            st.caption(f"Versión {version} · {len(items)} hallazgos")
            for item in items:
                st.write(item.title)
