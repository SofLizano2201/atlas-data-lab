"""Defaults for the modular ATLAS assistant system."""

from atlas.core.session import AgentContext


def default_arguments(tool: str, context: AgentContext, problems: set[str]) -> dict:
    """Provide defaults for using validated inputs."""
    metadata = context.metadata()
    selection = context.selections.get("analysis_selection") or {}
    available_columns = context.dataset.columns
    suggested = metadata.get("feature_roles", {}).get("suggested", available_columns)
    numeric = [
        c["name"]
        for c in metadata["columns"]
        if c["name"] in available_columns
        and any(t in c["dtype"].lower() for t in ["int", "float", "double"])
    ]
    columns = [column for column in numeric if column in suggested][:4] or suggested[:1]
    ids = [e.id for e in context.experiments if e.problem in problems]
    if tool in {"eda.frequencies", "classification.class_distribution"}:
        return {"column": selection.get("target") or available_columns[-1]}
    if tool == "analysis.configure":
        problem = (
            "classification"
            if "clasificacion" in problems
            else "regression"
            if "regresion" in problems
            else "clustering"
        )
        target = (
            None if problem == "clustering" else selection.get("target") or available_columns[-1]
        )
        return {
            "problem": problem,
            "target": target,
            "predictors": [
                c
                for c in (selection.get("predictors") or suggested[:4])
                if c != target and c in suggested
            ],
        }
    if tool in {"eda.profile", "eda.correlation"}:
        return {}
    if tool == "eda.statistics":
        return {"columns": columns}
    if tool == "eda.outliers":
        return {"columns": columns, "method": "iqr", "threshold": 1.5}
    if tool == "cluster.elbow":
        return {"columns": columns, "max_k": 8}
    if tool == "eda.plot":
        return {"kind": "histogram" if numeric else "count", "x": columns[0]}
    if tool == "eda.test":
        return {"test": "shapiro", "columns": columns[:1]}
    if tool == "eda.preview":
        return {"operation": "median", "columns": columns}
    if tool == "eda.apply":
        return {"proposal_id": next(reversed(context.proposals), "")}
    if tool == "cluster.reduce":
        return {"columns": columns, "method": "pca", "dimensions": 2}
    if tool == "cluster.distances":
        return {"columns": columns, "metric": "euclidean"}
    if tool == "cluster.train":
        return {"columns": columns, "algorithm": "kmeans", "parameters": {"n_clusters": 3}}
    if tool.endswith(".train"):
        source = metadata.get("training_columns", available_columns)
        target = selection.get("target") or str(source[-1])
        return {
            "algorithm": "logistic" if tool.startswith("classification") else "ridge",
            "target": target,
            "predictors": [str(c) for c in source if c != target and c in suggested][:12],
        }
    if tool == "comparison.compare":
        eligible = [e for e in context.experiments if e.problem in problems]
        if not eligible:
            return {"experiment_ids": [], "metric": "RMSE", "direction": "min"}
        candidates = [
            item
            for item in eligible
            if any(
                other.id != item.id and (other.problem, other.cohort) == (item.problem, item.cohort)
                for other in eligible
            )
        ]
        anchor = candidates[-1] if candidates else eligible[-1]
        compatible = [
            e.id for e in eligible if (e.problem, e.cohort) == (anchor.problem, anchor.cohort)
        ]
        metric = (
            "Accuracy"
            if anchor.problem == "clasificacion"
            else "RMSE"
            if anchor.problem == "regresion"
            else "Silhouette"
            if "Silhouette" in anchor.metrics
            else "ruido"
        )
        return {
            "experiment_ids": compatible[-20:] if "clustering" in problems else compatible[-2:],
            "metric": metric,
            "direction": "max" if metric in {"Accuracy", "Silhouette"} else "min",
        }
    if tool == "model.plot":
        return {
            "experiment_id": ids[-1] if ids else "",
            "kind": "confusion" if "clasificacion" in problems else "actual_predicted",
        }
    return {"experiment_id": ids[-1] if ids else ""}
