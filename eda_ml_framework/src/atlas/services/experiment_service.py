"""One session-owned experiment store shared by manual and assisted workflows."""

from collections.abc import MutableMapping
from typing import Any


class ExperimentService:
    @staticmethod
    def remember_investigation(state, run):
        """Keep scientific investigations beside the shared experiment registry."""
        investigations = state.setdefault("scientific_investigations", {})
        if len(investigations) >= 10 and run["id"] not in investigations:
            raise ValueError(
                "Máximo de diez investigaciones por sesión. Exporta antes de crear otra."
            )
        investigations[run["id"]] = run
        state["active_investigation"] = run["id"]

    @staticmethod
    def export_investigation(run):
        """Portable data/configuration/results bundle; never load pickled code."""
        import json
        from io import BytesIO
        from zipfile import ZIP_DEFLATED, ZipFile

        from plotly.offline import get_plotlyjs

        from atlas.models.scientific_experiment import ScientificExperiment
        from atlas.services.report_service import ReportService

        buffer, dataset = BytesIO(), BytesIO()
        run["dataset"].to_parquet(dataset, index=True)
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(
                    ScientificExperiment.manifest(run),
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                ),
            )
            archive.writestr("dataset.parquet", dataset.getvalue())
            archive.writestr("results.csv", ReportService.lab_results(run).to_csv(index=False))
            archive.writestr("report.html", ReportService.lab_report(run, "html"))
            archive.writestr("plotly.min.js", get_plotlyjs())
            for name, chart in ScientificExperiment.charts(run).items():
                archive.writestr(
                    f"charts/{name}.html", chart.to_html(include_plotlyjs="../plotly.min.js")
                )
        return buffer.getvalue()

    @staticmethod
    def import_investigation(bundle):
        """Validate an exported manifest and snapshot; importing never executes training."""
        import json
        from io import BytesIO
        from zipfile import ZipFile

        import pandas as pd

        from atlas.models.scientific_experiment import (
            ScientificExperiment,
            ScientificExperimentDefinition,
        )

        if len(bundle) > 25 * 1024**2:
            raise ValueError("El paquete supera 25 MB.")
        with ZipFile(BytesIO(bundle)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 100 * 1024**2:
                raise ValueError("El contenido del paquete supera 100 MB.")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schema_version") != 1:
                raise ValueError("Versión de manifiesto no compatible.")
            definition = ScientificExperimentDefinition.model_validate(manifest["definition"])
            frame = pd.read_parquet(BytesIO(archive.read("dataset.parquet")))
        return ScientificExperiment.create(definition, frame, parent_id=manifest["id"])

    @staticmethod
    def record(experiments, models, result, engine):
        result.model_version = len(experiments) + 1
        result.config["model_version"] = result.model_version
        experiments.append(result)
        models[result.id] = engine

    @staticmethod
    def remember(state: MutableMapping[str, Any], result: Any, engine: Any) -> None:
        ExperimentService.record(state["experiments"], state["models"], result, engine)
        state["metrics"][result.id] = result.metrics
        from atlas.core.events import EventType, emit

        payload = {"experiment_id": result.id, "problem": result.problem}
        emit(
            state,
            EventType.CLUSTERING_COMPLETED
            if result.problem == "clustering"
            else EventType.TRAINING_COMPLETED,
            payload,
        )
        emit(state, EventType.MODEL_EVALUATED, payload)
        emit(state, EventType.EXPERIMENT_SAVED, payload)
