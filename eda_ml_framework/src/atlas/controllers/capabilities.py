"""Validated workflow coordination using existing scientific models."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from atlas.core.schemas import (
    AnalysisResult,
    Contract,
    ExperimentResult,
    ModelResult,
    ToolCall,
    ToolResult,
)
from atlas.core.session import AgentContext, table_records
from atlas.models import (
    Classification,
    Clustering,
    ExploratoryDataAnalysis,
    Regression,
    SupervisedLearning,
)
from atlas.models.comparison import ExperimentComparison
from atlas.models.experiment import Experiment
from atlas.services.experiment_service import ExperimentService

from .configuration_controller import ConfigurationArgs, ConfigurationController


class EmptyArgs(Contract):
    """Define the EmptyArgs contract."""


class ColumnArgs(Contract):
    column: str


class ColumnsArgs(Contract):
    columns: list[str] = Field(min_length=1, max_length=30)


class OutlierArgs(ColumnsArgs):
    method: Literal["iqr", "zscore", "isolation"] = "iqr"
    threshold: float = Field(default=1.5, gt=0, le=10)


class ElbowArgs(ColumnsArgs):
    max_k: int = Field(default=8, ge=2, le=15)


class PlotArgs(Contract):
    kind: Literal[
        "histogram", "box", "violin", "scatter", "count", "correlation", "missing", "qq", "kde"
    ]
    x: str | None = None
    y: str | None = None


class CorrelationArgs(Contract):
    method: Literal["pearson", "spearman", "kendall"] = "pearson"


class TestArgs(ColumnsArgs):
    test: Literal["shapiro", "normaltest", "levene", "ttest", "anova", "chi2", "mannwhitney"]
    alpha: float = Field(default=0.05, gt=0, lt=1)


class TransformArgs(ColumnsArgs):
    operation: Literal[
        "mean",
        "median",
        "mode",
        "constant",
        "knn",
        "drop_rows",
        "drop_null_columns",
        "duplicates",
        "standard",
        "minmax",
        "robust",
        "normalizer",
        "onehot",
        "ordinal",
        "label",
        "log",
        "boxcox",
        "yeojohnson",
        "binning",
        "select",
        "drop_columns",
        "convert",
        "normalize_names",
        "clip",
        "winsor",
        "drop_outliers",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)


class ApplyArgs(Contract):
    proposal_id: str


class ReductionArgs(ColumnsArgs):
    method: Literal["pca", "tsne", "umap"] = "pca"
    dimensions: Literal[2, 3] = 2
    scaler: Literal["standard", "robust", "minmax", "none"] = "standard"


class DistanceArgs(ColumnsArgs):
    metric: Literal["euclidean", "cityblock", "cosine"] = "euclidean"


class ClusterArgs(ColumnsArgs):
    algorithm: str
    scaler: Literal["standard", "robust", "minmax", "none"] = "standard"
    parameters: dict[str, Any] = Field(default_factory=dict)


class TrainingArgs(Contract):
    algorithm: str
    target: str
    predictors: list[str] = Field(min_length=1, max_length=30)
    source: Literal["original", "working"] = "original"
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    cv: int = Field(default=3, ge=2, le=10)
    random_state: int = Field(default=42, ge=0, le=99999)
    scaler: Literal["standard", "minmax", "robust", "normalizer", "none"] = "standard"
    encoding: Literal["onehot", "ordinal"] = "onehot"
    parameters: dict[str, Any] = Field(default_factory=dict)
    search: dict[str, list[Any]] | None = None
    search_type: Literal["grid", "random"] = "grid"


class ExperimentArgs(Contract):
    experiment_id: str


class ModelPlotArgs(ExperimentArgs):
    kind: str


class ComparisonArgs(Contract):
    experiment_ids: list[str] = Field(min_length=2, max_length=20)
    metric: str
    direction: Literal["min", "max"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    arguments: type[Contract]
    confirmation: bool = False
    backend_method: Callable[..., Any] | tuple[Callable[..., Any], ...] | None = None
    output_schema: type[Contract] = AnalysisResult
    read_only: bool = True
    module: str = ""

    def schema(self) -> dict[str, Any]:
        """Provide schema using validated inputs."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.arguments.model_json_schema(),
            "confirmation": self.confirmation,
            "read_only": self.read_only,
            "module": self.module,
            "output_schema": self.output_schema.model_json_schema(),
        }


SPECS = {
    s.name: s
    for s in [
        ToolSpec("eda.profile", "Información, faltantes y duplicados reales", EmptyArgs),
        ToolSpec(
            "analysis.configure",
            "Proponer configuración del análisis; requiere confirmación",
            ConfigurationArgs,
            True,
        ),
        ToolSpec("eda.statistics", "Estadísticas de columnas numéricas", ColumnsArgs),
        ToolSpec("eda.frequencies", "Frecuencias calculadas de una variable", ColumnArgs),
        ToolSpec(
            "eda.outliers", "Diagnóstico de outliers con métodos existentes", OutlierArgs, True
        ),
        ToolSpec(
            "classification.class_distribution",
            "Distribución real de clases de la variable objetivo",
            ColumnArgs,
        ),
        ToolSpec("eda.correlation", "Matriz de correlación", CorrelationArgs),
        ToolSpec("eda.plot", "Figura calculada de los datos locales", PlotArgs),
        ToolSpec("eda.test", "Prueba estadística con supuestos", TestArgs),
        ToolSpec("eda.preview", "Proponer transformación sin modificar datos", TransformArgs),
        ToolSpec("eda.apply", "Aplicar propuesta local vigente; modifica datos", ApplyArgs, True),
        ToolSpec("cluster.reduce", "Representación PCA/t-SNE/UMAP", ReductionArgs, True),
        ToolSpec("cluster.distances", "Matriz de distancias", DistanceArgs, True),
        ToolSpec("cluster.train", "Entrenar algoritmo de clustering", ClusterArgs, True),
        ToolSpec(
            "cluster.elbow", "Método del codo existente; no demuestra un K óptimo", ElbowArgs, True
        ),
        ToolSpec(
            "classification.train", "Entrenar clasificador con pipeline y CV", TrainingArgs, True
        ),
        ToolSpec("regression.train", "Entrenar regresor con pipeline y CV", TrainingArgs, True),
        ToolSpec("model.metrics", "Consultar métricas de experimento permitido", ExperimentArgs),
        ToolSpec("model.plot", "Diagnóstico de un modelo ya entrenado", ModelPlotArgs),
        ToolSpec(
            "model.importance",
            "Importancia por permutación, solo interpretación",
            ExperimentArgs,
            True,
        ),
        ToolSpec(
            "comparison.compare",
            "Comparar cohortes compatibles según métrica y sentido",
            ComparisonArgs,
        ),
    ]
}
METHOD_REFERENCES = {
    "eda.outliers": ExploratoryDataAnalysis.outlier_diagnostic,
    "cluster.elbow": Clustering.elbow,
    "analysis.configure": ConfigurationController.apply,
    "eda.profile": ExploratoryDataAnalysis.profile,
    "eda.statistics": ExploratoryDataAnalysis.statistics,
    "eda.frequencies": ExploratoryDataAnalysis.frequencies,
    "classification.class_distribution": ExploratoryDataAnalysis.frequencies,
    "eda.correlation": ExploratoryDataAnalysis.correlation,
    "eda.plot": ExploratoryDataAnalysis.chart,
    "eda.test": ExploratoryDataAnalysis.statistical_test,
    "eda.preview": ExploratoryDataAnalysis.transform,
    "eda.apply": ExploratoryDataAnalysis.apply,
    "cluster.reduce": Clustering.reduce_dimensions,
    "cluster.distances": Clustering.distances,
    "cluster.train": Clustering.train,
    "classification.train": Classification.train,
    "regression.train": Regression.train,
    "model.metrics": Experiment.get_metrics,
    "model.plot": (Classification.plot_result, Regression.plot_result),
    "model.importance": SupervisedLearning.feature_importance,
    "comparison.compare": ExperimentComparison.rank,
}
SPECS = {
    name: replace(
        spec,
        backend_method=METHOD_REFERENCES[name],
        output_schema=ExperimentResult if name.endswith(".train") else AnalysisResult,
        read_only=name
        not in {
            "analysis.configure",
            "eda.apply",
            "cluster.train",
            "classification.train",
            "regression.train",
        },
        module=name.split(".")[0],
    )
    for name, spec in SPECS.items()
}


def analysis(
    context: AgentContext,
    summary: str,
    tables: dict[str, pd.DataFrame] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Provide analysis using validated inputs."""
    return AnalysisResult(
        summary=summary,
        dataset=context.dataset,
        tables={name: table_records(df) for name, df in (tables or {}).items()},
        **kwargs,
    )


def experiment_result(context: AgentContext, engine: Any, experiment: Any) -> ExperimentResult:
    """Provide experiment result using validated inputs."""
    ExperimentService.record(context.experiments, context.engines, experiment, engine)
    return ExperimentResult(
        experiment_id=experiment.id,
        dataset_id=experiment.dataset_id,
        model=ModelResult(
            model_id=experiment.id,
            algorithm=experiment.name,
            problem=experiment.problem,
            version=experiment.model_version,
            artifact_id=experiment.id,
        ),
        metrics={k: float(v) if np.isfinite(v) else None for k, v in experiment.metrics.items()},
        configuration=experiment.config,
        seconds=experiment.seconds,
        cohort=experiment.cohort,
    )


def find_experiment(context: AgentContext, experiment_id: str, problems: set[str]) -> Any:
    """Provide find experiment using validated inputs."""
    for experiment in context.experiments:
        if experiment.id == experiment_id and experiment.problem in problems:
            return experiment
    raise ValueError("Experimento inexistente o fuera de los permisos de este agente.")


def execute_tool(context: AgentContext, call: ToolCall, problems: set[str]) -> ToolResult:
    """Provide execute tool using validated inputs."""
    start = perf_counter()
    name = call.tool
    args = SPECS[name].arguments.model_validate(call.parameters)
    warnings = []
    eda = context.eda
    if name == "analysis.configure":
        if {
            "classification": "clasificacion",
            "regression": "regresion",
            "clustering": "clustering",
        }[args.problem] not in problems:
            raise ValueError("La configuración no pertenece a la especialidad activa.")
        configuration = ConfigurationController.apply(context, args)
        result = analysis(
            context,
            "Configuración confirmada y aplicada al formulario; no se ha entrenado un modelo.",
            {"configuracion": pd.DataFrame([configuration])},
        )
    elif name == "eda.profile":
        profile = eda.profile()
        result = analysis(
            context,
            "Diagnóstico calculado sobre la versión activa.",
            {"informacion": eda.information(), "faltantes": eda.missing_values()},
            metrics={
                "filas": profile["rows"],
                "columnas": profile["columns"],
                "duplicadas_adicionales": profile["additional_duplicates"],
                "celdas_nulas": profile["missing_cells"],
                "columnas_constantes": profile["constant_columns"],
                "numeric_columns": profile["numeric_columns"],
                "categorical_columns": profile["categorical_columns"],
            },
        )
    elif name in {"eda.frequencies", "classification.class_distribution"}:
        frequencies = eda.frequencies(args.column)
        if name == "classification.class_distribution" and len(frequencies) > 100:
            raise ValueError("Más de 100 clases: revisa si el objetivo es un identificador.")
        result = analysis(
            context,
            "Frecuencias calculadas, incluidos valores faltantes.",
            {"frecuencias": frequencies},
            metrics={
                f"group_{i}.{key}": float(row[key])
                for i, (_, row) in enumerate(frequencies.head(30).iterrows())
                for key in ["absoluta", "relativa"]
            },
        )
        if len(frequencies) > 30:
            warnings.append(
                "El resumen para el proveedor contiene las primeras 30 frecuencias, sin etiquetas de categorías."
            )
    elif name == "eda.outliers":
        diagnostic = eda.outlier_diagnostic(args.columns, args.method, args.threshold)
        result = analysis(
            context,
            "Filas señaladas por el método seleccionado; no se eliminaron observaciones.",
            {"outliers": diagnostic["rows"]},
            metrics={"outlier_rows": diagnostic["count"]},
        )
    elif name == "eda.statistics":
        statistics = eda.statistics(args.columns)
        result = analysis(
            context,
            "Estadística descriptiva numérica.",
            {"estadisticas": statistics},
            metrics={
                f"{column}.{metric}": float(statistics.loc[column, metric])
                if pd.notna(statistics.loc[column, metric])
                and np.isfinite(float(statistics.loc[column, metric]))
                else None
                for column in args.columns
                for metric in ["media", "mediana", "desviacion"]
            },
        )
    elif name == "eda.correlation":
        result = analysis(
            context,
            "Correlación no implica causalidad.",
            {"correlacion": eda.correlation(args.method)},
        )
    elif name == "eda.plot":
        fig = eda.chart(args.kind, args.x, args.y)
        result = analysis(
            context,
            "Figura calculada con los datos locales.",
            visualizations=[context.artifact(fig)],
        )
    elif name == "eda.test":
        calculated = eda.statistical_test(args.test, args.columns, args.alpha)
        warnings = calculated["advertencias"]
        result = analysis(
            context,
            calculated["interpretacion"],
            metrics={"estadistico": calculated["estadistico"], "p_value": calculated["p_value"]},
        )
    elif name == "eda.preview":
        preview = eda.transform(args.operation, args.columns, **args.parameters)
        proposal_id = context.propose(preview, args.operation)
        result = analysis(
            context,
            "Propuesta sin aplicar. Se requiere confirmación para modificar datos.",
            {"antes": eda.df, "despues": preview},
            proposal_id=proposal_id,
        )
    elif name == "eda.apply":
        proposal = context.proposals.get(args.proposal_id)
        if not proposal or proposal[0] != context.key:
            raise ValueError("La propuesta no existe o pertenece a otra versión del dataset.")
        eda.apply(proposal[1], proposal[2], confirm=True)
        context.proposals.clear()
        result = analysis(context, "Transformación aplicada y versionada.", {"datos": eda.df})
    elif name in {"cluster.reduce", "cluster.distances", "cluster.train", "cluster.elbow"}:
        engine = Clustering(eda.df, random_state=eda.random_state)
        if name == "cluster.elbow":
            result = analysis(
                context,
                "Inercia calculada para los K solicitados; no se infiere automáticamente un K óptimo.",
                {"elbow": engine.elbow(args.columns, args.max_k)},
            )
        elif name == "cluster.reduce":
            reduced = engine.reduce_dimensions(
                args.columns, args.method, args.dimensions, scaler=args.scaler
            )
            result = analysis(
                context,
                reduced["advertencia"],
                {"representacion": reduced["datos"]},
                metrics={"varianza_explicada": reduced["explained_variance_total"]}
                if reduced["explained_variance_total"] is not None
                else {},
                visualizations=[context.artifact(engine.plot_representation(reduced["datos"]))],
            )
        elif name == "cluster.distances":
            result = analysis(
                context,
                "Distancias en espacio preprocesado.",
                {"distancias": engine.distances(args.columns, args.metric)},
            )
        else:
            trained = engine.train(
                args.algorithm, args.columns, scaler=args.scaler, **args.parameters
            )
            result = experiment_result(context, engine, trained)
            warnings.append(
                "Las métricas excluyen ruido; no prueban que exista una partición verdadera."
            )
    elif name in {"classification.train", "regression.train"}:
        problem = "clasificacion" if name.startswith("classification") else "regresion"
        cls = Classification if problem == "clasificacion" else Regression
        frame = eda.original if args.source == "original" else eda.df
        engine = cls(frame, random_state=args.random_state)
        trained = engine.train(
            args.algorithm,
            args.target,
            args.predictors,
            test_size=args.test_size,
            cv=args.cv,
            scaler=args.scaler,
            encoding=args.encoding,
            parameters=args.parameters,
            search=args.search,
            search_type=args.search_type,
        )
        trained.config["data_source"] = args.source
        trained.config["source_version"] = (
            context.dataset.version if args.source == "working" else 0
        )
        result = experiment_result(context, engine, trained)
        warnings.append("El holdout no debe reutilizarse para seleccionar modelos o variables.")
        if args.source == "working":
            warnings.append("Transformaciones exploratorias previas pueden sesgar la evaluación.")
    elif name.startswith("model."):
        experiment = find_experiment(context, args.experiment_id, problems)
        if experiment.dataset_id != context.dataset.content_hash:
            warnings.append(
                "El modelo pertenece a una versión de datos diferente de la activa; las métricas y diagnósticos utilizan su partición original registrada."
            )
        if name == "model.metrics":
            result = analysis(
                context,
                f"Métricas registradas de {experiment.name}.",
                metrics={
                    k: float(v) if np.isfinite(v) else None for k, v in experiment.metrics.items()
                },
            )
        elif name == "model.plot":
            if experiment.problem == "clustering":
                raise ValueError("Para clustering use cluster.reduce o sus diagnósticos manuales.")
            allowed = (
                {"confusion", "confusion_normalized", "roc", "pr", "tree", "correctness"}
                if experiment.problem == "clasificacion"
                else {"actual_predicted", "residuals", "histogram", "qq"}
            )
            if args.kind not in allowed:
                raise ValueError("Gráfico incompatible con el tipo de experimento.")
            figure = context.engines[experiment.id].plot_result(experiment, args.kind)
            result = analysis(
                context,
                "Diagnóstico calculado del modelo registrado.",
                visualizations=[context.artifact(figure)],
            )
        else:
            result = analysis(
                context,
                "Importancia en prueba para interpretación, no selección.",
                {"importancia": context.engines[experiment.id].feature_importance(experiment)},
            )
        result = result.model_copy(
            update={
                "source_dataset_id": experiment.dataset_id,
                "source_experiment_id": experiment.id,
            }
        )
    elif name == "comparison.compare":
        if len(set(args.experiment_ids)) != len(args.experiment_ids):
            raise ValueError("No repita experimentos en la comparación.")
        experiments = [find_experiment(context, i, problems) for i in args.experiment_ids]
        table, best, best_value = ExperimentComparison.rank(
            experiments, args.metric, args.direction
        )
        result = analysis(
            context,
            f"Mejor experimento según {args.metric} ({args.direction}): {best}.",
            {"comparacion": table},
            metrics={"mejor_valor": best_value},
        )
        warnings.append(
            "La conclusión depende de cohorte, métrica y sentido; no es un ganador universal."
        )
    else:
        raise ValueError("Herramienta no implementada en el registro.")
    return ToolResult(
        call=call,
        status="success",
        result=result,
        warnings=warnings,
        seconds=perf_counter() - start,
    )


class CapabilityGateway:
    """Expose only schema-validated, registered scientific capabilities."""

    tools = SPECS

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    def execute(self, call: ToolCall, problems: set[str], *, confirmed: bool = False) -> ToolResult:
        if call.tool not in self.tools:
            raise ValueError("Unknown capability.")
        self.tools[call.tool].arguments.model_validate(call.parameters)
        if self.tools[call.tool].confirmation and not confirmed:
            raise ValueError("Esta capacidad requiere una confirmación local explícita.")
        return execute_tool(self.context, call, problems)
