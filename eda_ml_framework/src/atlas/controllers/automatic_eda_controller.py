"""Bounded orchestration of existing EDA methods, without new statistical formulas."""

from dataclasses import dataclass, field
from html import escape
from time import monotonic
from typing import Any

import pandas as pd

from atlas.core.events import EventType, emit
from atlas.models.experiment import html_report


@dataclass(frozen=True)
class EDALimits:
    max_seconds: float = 20.0
    max_numeric: int = 20
    max_categories: int = 12
    max_charts: int = 6
    max_pairs: int = 6
    max_chart_points: int = 1000
    max_kendall_rows: int = 3000

    def __post_init__(self):
        if (
            self.max_seconds < 0
            or self.max_numeric < 1
            or self.max_categories < 0
            or self.max_charts < 0
            or self.max_pairs < 0
            or self.max_chart_points < 1
            or self.max_kendall_rows < 1
        ):
            raise ValueError(
                "EDA limits must be nonnegative; numerical and row limits must be positive"
            )


@dataclass
class EDAExecution:
    key: str
    status: str = "running"
    operations: dict[str, dict[str, Any]] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, list[str]] = field(default_factory=dict)
    columns: list[dict[str, Any]] = field(default_factory=list)
    report: str = ""
    seconds: float = 0.0


class AutomaticEDAController:
    @staticmethod
    def report(state):
        """Attach agent-authored interpretation to the calculated report without recalculating."""
        execution = state.get("automatic_eda", {}).get(AutomaticEDAController.key(state["eda"]))
        if execution is None:
            return ""
        feedback = state.get("intelligence_feedback", {}).get("EDAAgent")
        if feedback is None:
            return execution.report
        section = f"<h2>Hallazgos y próximos pasos</h2><p>{escape(feedback.message)}</p>"
        for evidence in feedback.evidence:
            section += f"<p>{escape(evidence)}</p>"
        if feedback.candidates:
            section += "<h2>Posibles objetivos (no seleccionados automáticamente)</h2>"
            for candidate in feedback.candidates:
                section += f"<p>{escape(candidate['column'])}: {escape(candidate['objective'])}. {escape(candidate['limitation'])}</p>"
        return execution.report.replace("</body>", section + "</body>")

    @staticmethod
    def key(model):
        version = model.dataset_version
        return f"{version.dataset_id}:{version.version}:{version.content_hash}"

    @staticmethod
    def run(state, limits: EDALimits | None = None, progress=None, refresh=False):
        """Run fast then comprehensive passes synchronously, once per version by default."""
        model = state["eda"]
        key = AutomaticEDAController.key(model)
        cache = state.setdefault("automatic_eda", {})
        if key in cache and not refresh:
            emit(
                state,
                EventType.EDA_COMPLETED,
                {"run": key, "status": cache[key].status, "reused": True},
            )
            return cache[key]
        limits = limits or state.get("eda_limits") or EDALimits()
        execution = EDAExecution(key)
        cache[key] = execution
        while len(cache) > 2:
            cache.pop(next(iter(cache)))
        started = monotonic()
        if refresh:
            state["eda_refresh_count"] = state.get("eda_refresh_count", 0) + 1
        run_reference = {"run": key, "refresh": state.get("eda_refresh_count", 0)}
        emit(state, EventType.EDA_STARTED, run_reference)

        def skip(name, reason):
            execution.operations[name] = {"status": "skipped", "reason": reason}

        def execute(name, callback, fast=False, sampled=False):
            if not fast and monotonic() - started >= limits.max_seconds:
                skip(name, "Time budget reached; not executed")
                return None
            if state.get("eda_cancel_requested"):
                skip(name, "Cancelled before starting operation")
                return None
            if progress:
                progress(name)
            try:
                value = callback()
                execution.operations[name] = {
                    "status": "completed",
                    "value": value,
                    "sampled": sampled,
                }
                return value
            except (
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                ImportError,
                ArithmeticError,
            ) as exc:
                execution.operations[name] = {"status": "failed", "reason": str(exc)}
                return None

        # Fast pass: full-dataset quality and schema, never sampled.
        execution.profile = execute("profile", model.profile, True) or {}
        execution.roles = execute("feature_roles", model.feature_roles, True) or {}
        information = execute("schema", model.information, True)
        execute("missing_values", model.missing_values, True)
        execute("type_inconsistencies", model.type_inconsistencies, True)
        execute("memory_usage", model.memory_usage, True)
        if information is None or not execution.profile:
            execution.status = "failed"
            emit(state, EventType.EDA_FAILED, {"run": key})
            return execution
        frame = model.df
        numeric = list(frame.select_dtypes(include="number").columns)
        categorical = [column for column in frame if column not in numeric]
        execution.columns = [
            {
                "name": column,
                "dtype": str(information.loc[column, "tipo"]),
                "unique": int(information.loc[column, "unicos"]),
                "missing_percentage": float(information.loc[column, "faltantes_%"]),
            }
            for column in frame
        ]
        if numeric:
            execute("descriptive_statistics", lambda: model.statistics(numeric), True)
            bounded = numeric[: limits.max_numeric]
            execute(
                "percentiles",
                lambda: {
                    column: model.percentiles(column, [5, 25, 50, 75, 95]) for column in bounded
                },
            )
            execute("outliers_iqr", lambda: model.outlier_diagnostic(bounded, "iqr"))
            execute("outliers_zscore", lambda: model.outlier_diagnostic(bounded, "zscore"))
            if len(frame) <= 5000 and len(bounded) <= 10:
                execute(
                    "outliers_isolation", lambda: model.outlier_diagnostic(bounded, "isolation")
                )
            else:
                skip("outliers_isolation", "Limited to 5000 rows and 10 numerical columns")
            if len(numeric) >= 2 and len(numeric) <= limits.max_numeric:
                for method in ["pearson", "spearman", "kendall"]:
                    if method == "kendall" and len(frame) > limits.max_kendall_rows:
                        skip(
                            f"correlation_{method}",
                            "Full-data Kendall exceeds configured row limit",
                        )
                    else:
                        execute(
                            f"correlation_{method}", lambda method=method: model.correlation(method)
                        )
                if len(numeric) <= 10 and len(frame) <= 5000:
                    execute("multicollinearity_vif", lambda: model.vif(numeric))
                else:
                    skip(
                        "multicollinearity_vif", "VIF limited to 10 numerical columns and 5000 rows"
                    )
            else:
                skip("correlations", "Requires 2 numerical columns, within configured width limit")
            if len(numeric) > limits.max_numeric:
                skip(
                    "additional_numeric_diagnostics",
                    "Additional columns excluded from bounded diagnostics; descriptive statistics remain full-data",
                )
        else:
            skip("numeric_analysis", "No numerical columns")
        discrete = categorical + [
            column for column in numeric if information.loc[column, "unicos"] <= 20
        ]
        for column in discrete[: limits.max_categories]:
            if information.loc[column, "unicos"] > 100:
                skip(
                    f"frequencies:{column}",
                    "High cardinality; schema retains full-data unique count",
                )
                continue
            execute(f"frequencies:{column}", lambda column=column: model.frequencies(column))
            execute(
                f"rare_categories:{column}", lambda column=column: model.rare_categories(column)
            )
        if len(discrete) > limits.max_categories:
            skip("additional_categorical_analysis", "Configured categorical-column budget reached")
        low_cardinality = [
            column for column in categorical if information.loc[column, "unicos"] <= 20
        ]
        pairs = [(a, b) for i, a in enumerate(low_cardinality) for b in low_cardinality[i + 1 :]]
        for a, b in pairs[: limits.max_pairs]:
            execute(f"contingency:{a}:{b}", lambda a=a, b=b: model.contingency(a, b))
        for a, b in [(a, b) for a in low_cardinality[:2] for b in numeric[:3]][: limits.max_pairs]:
            execute(f"grouped_statistics:{a}:{b}", lambda a=a, b=b: model.grouped_statistics(a, b))
        if len(pairs) > limits.max_pairs:
            skip("additional_relationships", "Configured pair budget reached")
        # Charts use a labelled reproducible sample; metrics above always use full data.
        chart_model = model
        sampled = len(frame) > limits.max_chart_points
        if sampled:
            chart_model = model.visualization_sample(limits.max_chart_points)
        chart_plan = []
        if numeric:
            chart_plan.extend(
                [
                    (f"chart:{numeric[0]}", "histogram", numeric[0], None),
                    (f"chart:box:{numeric[0]}", "box", numeric[0], None),
                ]
            )
        if len(numeric) >= 2:
            chart_plan.extend(
                [
                    ("chart:scatter", "scatter", numeric[0], numeric[1]),
                    ("chart:correlation", "correlation", None, None),
                ]
            )
        if low_cardinality:
            chart_plan.append(
                (f"chart:count:{low_cardinality[0]}", "count", low_cardinality[0], None)
            )
        if execution.profile.get("missing_cells"):
            chart_plan.insert(0, ("chart:missing", "missing", None, None))
        for column in numeric[: limits.max_numeric]:
            for kind in ["histogram", "box", "violin", "kde", "qq"]:
                name = f"chart:{column}" if kind == "histogram" else f"chart:{kind}:{column}"
                if not any(item[0] == name for item in chart_plan):
                    chart_plan.append((name, kind, column, None))
        if 2 <= len(numeric) <= 6:
            chart_plan.append(("chart:pairplot", "pairplot", None, None))
        for index, (name, kind, x, y) in enumerate(chart_plan):
            if index >= limits.max_charts:
                skip(name, "Chart count budget reached; available manually")
            else:
                execute(
                    name, lambda kind=kind, x=x, y=y: chart_model.chart(kind, x, y), sampled=sampled
                )
        skip(
            "hypothesis_tests",
            "No analytical hypothesis or group assumptions supplied; tests remain available manually",
        )
        skip(
            "domain_validation",
            "No domain-specific rules supplied; unexpected values cannot be inferred safely",
        )
        if len(numeric or low_cardinality) > limits.max_charts:
            skip(
                "additional_charts",
                "Chart count budget reached; other variable statistics remain available",
            )
        tables = {
            name: output["value"]
            for name, output in execution.operations.items()
            if output["status"] == "completed" and isinstance(output["value"], pd.DataFrame)
        }
        tables["Dataset overview"] = pd.DataFrame([execution.profile])
        for name, output in execution.operations.items():
            if (
                output["status"] == "completed"
                and isinstance(output["value"], dict)
                and "count" in output["value"]
            ):
                tables[name + " preview"] = output["value"]["rows"]
                tables[name + " count"] = pd.DataFrame([{"count": output["value"]["count"]}])
        tables["Execution coverage"] = pd.DataFrame(
            [
                {
                    "operation": name,
                    "status": output["status"],
                    "sampled_chart": output.get("sampled", False),
                    "reason": output.get("reason", ""),
                }
                for name, output in execution.operations.items()
            ]
        )
        execution.report = html_report(
            "ATLAS · Exploración automática",
            tables,
            "Full-dataset statistics; chart samples and skipped operations are documented in the application. No hypothesis tests without a specified hypothesis.",
        )
        chart_markup = ""
        include_runtime = True
        for name, output in execution.operations.items():
            if (
                name.startswith("chart:")
                and output["status"] == "completed"
                and hasattr(output["value"], "to_plotly_json")
            ):
                scope = (
                    f"Muestra reproducible de hasta {limits.max_chart_points} registros solo para visualización."
                    if output.get("sampled")
                    else "Gráfico con todos los registros."
                )
                chart_markup += f"<h2>{escape(name)}</h2><p>{scope}</p>" + output["value"].to_html(
                    full_html=False, include_plotlyjs=include_runtime
                )
                include_runtime = False
        execution.report = execution.report.replace("</body>", chart_markup + "</body>")
        execution.seconds = monotonic() - started
        execution.status = (
            "partial"
            if any(
                output["status"] in {"failed", "skipped"}
                for name, output in execution.operations.items()
                if name not in {"hypothesis_tests", "domain_validation"}
            )
            else "completed"
        )
        emit(state, EventType.EDA_COMPLETED, {**run_reference, "status": execution.status})
        return execution
