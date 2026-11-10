"""Real Wine/K-Means investigation and reproduction. No simulated results."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

from atlas.controllers.experiment_lab_controller import ExperimentLabController as Lab
from atlas.models import ExploratoryDataAnalysis
from atlas.models.scientific_experiment import (
    ExperimentalAlternative,
    ScientificExperiment,
    ScientificExperimentDefinition,
)
from atlas.services.experiment_service import ExperimentService
from atlas.services.report_service import ReportService


def main():
    parser = argparse.ArgumentParser(
        description="Demostración real: 12 ajustes K-Means incluidos los de reproducción."
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Autorizar los 12 ajustes sobre Wine."
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/scientific-lab-demo"))
    args = parser.parse_args()
    if not args.confirm:
        parser.error("Revisa el presupuesto y añade --confirm para entrenar.")
    frame = load_wine(as_frame=True).data
    state = {"eda": ExploratoryDataAnalysis(frame), "experiments": [], "models": {}, "metrics": {}}
    definition = ScientificExperimentDefinition(
        name="Wine · efecto de estandarizar K-Means",
        question="¿Estandarizar mejora la silueta de K-Means en Wine?",
        hypothesis="Estandarizar mejora la silueta bajo las condiciones definidas.",
        hypothesis_type="directional",
        template="preprocessing",
        factor="scaler",
        task="clustering",
        dataset=state["eda"].dataset_version,
        predictors=list(frame.columns),
        alternatives=[
            ExperimentalAlternative(
                name="A · Sin escalar",
                algorithm="kmeans",
                scaler="none",
                parameters={"n_clusters": 3, "n_init": 10},
            ),
            ExperimentalAlternative(
                name="B · Estandarizado",
                algorithm="kmeans",
                scaler="standard",
                parameters={"n_clusters": 3, "n_init": 10},
            ),
        ],
        metric="Silhouette",
        direction="max",
        seeds=[42, 43, 44],
        validation="same_rows_initializations",
    )
    run = Lab.prepare(state, definition)
    try:
        Lab.authorize(state, run["id"])
    except ValueError:
        pass
    else:
        raise AssertionError("La ejecución sin confirmación debe rechazarse.")
    Lab.execute(state, run["id"], confirm=True)
    reproduction = Lab.reproduce(state, run["id"])
    Lab.execute(state, reproduction["id"], confirm=True)
    assert run["status"] == reproduction["status"] == "completed"
    np.testing.assert_allclose(
        ScientificExperiment.summary(run)["mean"],
        ScientificExperiment.summary(reproduction)["mean"],
        rtol=1e-12,
    )
    pd.testing.assert_frame_equal(state["eda"].df, frame)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, item in [("original", run), ("reproduction", reproduction)]:
        (args.output / f"{name}_manifest.json").write_text(
            json.dumps(
                ScientificExperiment.manifest(item), ensure_ascii=False, allow_nan=False, indent=2
            ),
            encoding="utf-8",
        )
    (args.output / "report.html").write_bytes(ReportService.lab_report(run, "html"))
    ReportService.lab_results(run).to_csv(args.output / "results.csv", index=False)
    (args.output / "reproducible_experiment.zip").write_bytes(
        ExperimentService.export_investigation(run)
    )
    result = {
        "dataset": "sklearn Wine",
        "rows": len(frame),
        "columns": len(frame.columns),
        "dataset_fingerprint": definition.dataset.content_hash,
        "evaluations": 6,
        "reproduction_evaluations": 6,
        "status": run["status"],
        "reproduction_status": reproduction["status"],
        "reproduced": True,
        "source_unchanged": True,
        "confirmation_enforced": True,
        "summary": ScientificExperiment.summary(run).to_dict("records"),
        "stability": ScientificExperiment.stability(run).to_dict("records"),
        "interpretation": ScientificExperiment.interpretation(run),
    }
    (args.output / "demonstration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
