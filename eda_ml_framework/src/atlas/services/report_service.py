"""Maintain session-owned report history using the existing report generator."""

from atlas.models.reporting import generate_report


class ReportService:
    @staticmethod
    def generate(state, model, sections, file_format, title, experiments=None):
        context = state.get("agent_context")
        report = generate_report(
            model,
            state["experiments"] if experiments is None else experiments,
            context.decisions if context else [],
            sections,
            file_format,
            title,
        )
        state.setdefault("reports", []).append(report)
        state["reports"] = state["reports"][-10:]
        from atlas.core.events import EventType, emit

        emit(
            state,
            EventType.REPORT_GENERATED,
            {"title": title, "format": file_format, "number": len(state["reports"])},
        )
        return report
