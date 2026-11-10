"""Maintain session-owned report history using the existing report generator."""

from atlas.models.reporting import generate_report


class ReportService:
    @staticmethod
    def remember_lab_report(state, run, fmt):
        from uuid import uuid4

        from atlas.core.events import EventType, emit
        from atlas.models.reporting import ReportDocument
        from atlas.models.scientific_experiment import now

        report = ReportDocument(
            id=uuid4().hex,
            title=run["definition"].name,
            format=fmt,
            data=ReportService.lab_report(run, fmt),
            dataset_hash=run["definition"].dataset.content_hash,
            sections=["Laboratorio científico"],
            created=now(),
        )
        state.setdefault("reports", []).append(report)
        state["reports"] = state["reports"][-10:]
        emit(state, EventType.REPORT_GENERATED, {"investigation_id": run["id"], "format": fmt})
        return report

    @staticmethod
    def lab_results(run):
        import pandas as pd

        return pd.DataFrame(list(run["evaluations"].values())).drop(
            columns=["metrics", "configuration", "estimator_parameters", "warnings"], errors="ignore"
        )

    @staticmethod
    def lab_report(run, fmt="html"):
        import json

        import pandas as pd

        from atlas.models.experiment import html_report
        from atlas.models.reporting import _pdf
        from atlas.models.scientific_experiment import ScientificExperiment

        if fmt not in {"html", "pdf"}:
            raise ValueError("Selecciona HTML o PDF.")
        definition = run["definition"]
        tables = {
            "Pregunta e hipótesis": pd.DataFrame(
                [
                    {"Aspecto": "Pregunta", "Contenido": definition.question},
                    {"Aspecto": "Hipótesis", "Contenido": definition.hypothesis},
                    {"Aspecto": "Tipo", "Contenido": definition.hypothesis_type},
                    {"Aspecto": "Hipótesis nula", "Contenido": definition.null_hypothesis},
                    {
                        "Aspecto": "Hipótesis alternativa",
                        "Contenido": definition.alternative_hypothesis,
                    },
                ]
            ),
            "Dataset y variables": pd.DataFrame(
                [
                    {
                        "Identificador": definition.dataset.dataset_id,
                        "Versión": definition.dataset.version,
                        "Huella": definition.dataset.content_hash,
                        "Filas": len(run["dataset"]),
                        "Objetivo": definition.target or "Sin objetivo",
                        "Variables": ", ".join(definition.predictors),
                    }
                ]
            ),
            "Diseño y condiciones controladas": pd.DataFrame(
                [
                    {
                        "Factor": definition.factor,
                        "Protocolo": definition.validation,
                        "Métrica": definition.metric,
                        "Dirección": definition.direction,
                        "Semillas": str(definition.seeds),
                        "Folds": definition.cv if definition.task != "clustering" else "No aplica",
                        "Prueba": definition.test_size
                        if definition.task != "clustering"
                        else "Mismas filas",
                        "Justificación": definition.custom_justification
                        or "Una condición principal variable",
                    }
                ]
            ),
            "Algoritmos y configuraciones": pd.DataFrame(
                [
                    {
                        "Alternativa": a.name,
                        "Algoritmo": a.algorithm,
                        "Configuración": a.model_dump_json(),
                    }
                    for a in definition.alternatives
                ]
            ),
            "Resultados por evaluación": ReportService.lab_results(run),
            "Variabilidad descriptiva": ScientificExperiment.summary(run),
            "Diferencias pareadas": ScientificExperiment.paired_differences(run),
            "Estabilidad de clusters": ScientificExperiment.stability(run),
            "Incertidumbre": pd.DataFrame([run["uncertainty"]])
            if run.get("uncertainty")
            else pd.DataFrame(),
            "Interpretación, limitaciones y conclusión": pd.DataFrame(
                [{"Conclusión": ScientificExperiment.interpretation(run)}]
            ),
            "Advertencias": pd.DataFrame({"Advertencia": run["warnings"]}),
            "Reproducibilidad": pd.DataFrame(
                [
                    {
                        "ID": run["id"],
                        "Origen": run.get("parent_id"),
                        "Creación": run["created_at"],
                        "Inicio": run.get("started_at"),
                        "Fin": run.get("completed_at"),
                        "Estado": run["status"],
                        "Entorno": json.dumps(run["environment"], ensure_ascii=False),
                    }
                ]
            ),
        }
        note = "Conserva el paquete ZIP con el manifiesto y el dataset para reproducir esta investigación. Los resultados incompletos no permiten conclusiones definitivas."
        return (
            html_report(definition.name, tables, note).encode("utf-8")
            if fmt == "html"
            else _pdf(definition.name, tables, note)
        )

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
