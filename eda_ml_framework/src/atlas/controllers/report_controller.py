"""Report and export coordination using existing model implementations."""

from atlas.models.experiment import export_frame, html_report
from atlas.models.reporting import SECTIONS, generate_report

__all__ = ["SECTIONS", "ReportController", "export_frame", "generate_report", "html_report"]


class ReportController:
    sections = SECTIONS
    generate = staticmethod(generate_report)
    export_frame = staticmethod(export_frame)
    html_report = staticmethod(html_report)

    @staticmethod
    def generate_for_session(state, model, sections, file_format, title, experiments=None):
        from atlas.services.report_service import ReportService

        return ReportService.generate(
            state, model._model, sections, file_format, title, experiments
        )
