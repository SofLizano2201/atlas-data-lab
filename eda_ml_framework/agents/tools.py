"""Registro cerrado de herramientas con parámetros y permisos por especialidad."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from contracts import AnalysisResult, ExperimentResult, ModelResult, ToolCall, ToolResult
from contracts.models import Contract
from framework import Clasificacion, Clustering, Regresion
from framework.results import compare

from .context import AgentContext, table_records


class EmptyArgs(Contract):
    """Sin parámetros admitidos."""


class ColumnsArgs(Contract):
    columns: list[str] = Field(min_length=1, max_length=30)


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

    def schema(self) -> dict[str, Any]:
        """Esquema público sin función ejecutable ni acceso libre a objetos."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.arguments.model_json_schema(),
            "confirmation": self.confirmation,
        }


SPECS = {
    s.name: s
    for s in [
        ToolSpec("eda.profile", "Información, faltantes y duplicados reales", EmptyArgs),
        ToolSpec("eda.statistics", "Estadísticas de columnas numéricas", ColumnsArgs),
        ToolSpec("eda.correlation", "Matriz de correlación", CorrelationArgs),
        ToolSpec("eda.plot", "Figura calculada de los datos locales", PlotArgs),
        ToolSpec("eda.test", "Prueba estadística con supuestos", TestArgs),
        ToolSpec("eda.preview", "Proponer transformación sin modificar datos", TransformArgs),
        ToolSpec("eda.apply", "Aplicar propuesta local vigente; modifica datos", ApplyArgs, True),
        ToolSpec("cluster.reduce", "Representación PCA/t-SNE/UMAP", ReductionArgs, True),
        ToolSpec("cluster.distances", "Matriz de distancias", DistanceArgs, True),
        ToolSpec("cluster.train", "Entrenar algoritmo de clustering", ClusterArgs, True),
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


def analysis(
    context: AgentContext,
    summary: str,
    tables: dict[str, pd.DataFrame] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Convertir salidas reales al contrato serializable compartido."""
    return AnalysisResult(
        summary=summary,
        dataset=context.dataset,
        tables={name: table_records(df) for name, df in (tables or {}).items()},
        **kwargs,
    )


def experiment_result(context: AgentContext, engine: Any, experiment: Any) -> ExperimentResult:
    """Compartir el mismo historial que utiliza el comparador manual."""
    context.experiments.append(experiment)
    experiment.model_version = len(context.experiments)
    experiment.config["model_version"] = experiment.model_version
    context.engines[experiment.id] = engine
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
    """Solo resultados de esta sesión y especialidad."""
    for experiment in context.experiments:
        if experiment.id == experiment_id and experiment.problem in problems:
            return experiment
    raise ValueError("Experimento inexistente o fuera de los permisos de este agente.")


def execute_tool(context: AgentContext, call: ToolCall, problems: set[str]) -> ToolResult:
    """Despacho explícito; ni getattr arbitrario, ni eval, ni ejecución de Python del LLM."""
    start = perf_counter()
    name = call.tool
    args = SPECS[name].arguments.model_validate(call.parameters)
    warnings = []
    eda = context.eda
    if name == "eda.profile":
        result = analysis(
            context,
            "Diagnóstico calculado sobre la versión activa.",
            {"informacion": eda.informacion(), "faltantes": eda.nulos()},
            metrics={
                "filas": len(eda.df),
                "columnas": len(eda.columnas()),
                "duplicadas_adicionales": int(eda.df.duplicated().sum()),
                "celdas_nulas": int(eda.df.isna().sum().sum()),
                "columnas_constantes": len(eda.constantes()),
            },
        )
    elif name == "eda.statistics":
        statistics = eda.estadisticas(args.columns)
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
            {"correlacion": eda.correlacion(args.method)},
        )
    elif name == "eda.plot":
        fig = eda.grafico(args.kind, args.x, args.y)
        result = analysis(
            context,
            "Figura calculada con los datos locales.",
            visualizations=[context.artifact(fig)],
        )
    elif name == "eda.test":
        calculated = eda.prueba(args.test, args.columns, args.alpha)
        warnings = calculated["advertencias"]
        result = analysis(
            context,
            calculated["interpretacion"],
            metrics={"estadistico": calculated["estadistico"], "p_value": calculated["p_value"]},
        )
    elif name == "eda.preview":
        preview = eda.transformar(args.operation, args.columns, **args.parameters)
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
        eda.aplicar(proposal[1], proposal[2], confirmar=True)
        context.proposals.clear()
        result = analysis(context, "Transformación aplicada y versionada.", {"datos": eda.df})
    elif name in {"cluster.reduce", "cluster.distances", "cluster.train"}:
        engine = Clustering(eda.df, random_state=eda.random_state)
        if name == "cluster.reduce":
            reduced = engine.reducir(args.columns, args.method, args.dimensions, scaler=args.scaler)
            result = analysis(
                context,
                reduced["advertencia"],
                {"representacion": reduced["datos"]},
                metrics={"varianza_explicada": float(np.sum(reduced["varianza"]))}
                if reduced["varianza"] is not None
                else {},
                visualizations=[
                    context.artifact(engine.visualizar_representacion(reduced["datos"]))
                ],
            )
        elif name == "cluster.distances":
            result = analysis(
                context,
                "Distancias en espacio preprocesado.",
                {"distancias": engine.distancias(args.columns, args.metric)},
            )
        else:
            trained = engine.entrenar(
                args.algorithm, args.columns, scaler=args.scaler, **args.parameters
            )
            result = experiment_result(context, engine, trained)
            warnings.append(
                "Las métricas excluyen ruido; no prueban que exista una partición verdadera."
            )
    elif name in {"classification.train", "regression.train"}:
        problem = "clasificacion" if name.startswith("classification") else "regresion"
        cls = Clasificacion if problem == "clasificacion" else Regresion
        frame = eda.original if args.source == "original" else eda.df
        engine = cls(frame, random_state=args.random_state)
        trained = engine.entrenar(
            args.algorithm,
            args.target,
            args.predictors,
            test_size=args.test_size,
            cv=args.cv,
            scaler=args.scaler,
            encoding=args.encoding,
            parametros=args.parameters,
            busqueda=args.search,
            tipo_busqueda=args.search_type,
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
                "El modelo pertenece a una versión de datos diferente de la activa; "
                "las métricas y diagnósticos utilizan su partición original registrada."
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
                if (experiment.problem == "clasificacion")
                else {"actual_predicted", "residuals", "histogram", "qq"}
            )
            if args.kind not in allowed:
                raise ValueError("Gráfico incompatible con el tipo de experimento.")
            figure = context.engines[experiment.id].grafico_resultado(experiment, args.kind)
            result = analysis(
                context,
                "Diagnóstico calculado del modelo registrado.",
                visualizations=[context.artifact(figure)],
            )
        else:
            result = analysis(
                context,
                "Importancia en prueba para interpretación, no selección.",
                {"importancia": context.engines[experiment.id].importancia(experiment)},
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
        table = compare(experiments)
        if args.metric not in table or not pd.api.types.is_numeric_dtype(table[args.metric]):
            raise ValueError("Métrica no disponible o no numérica.")
        valid = table[args.metric].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            raise ValueError("No hay valores válidos de esa métrica.")
        best = valid.idxmin() if args.direction == "min" else valid.idxmax()
        result = analysis(
            context,
            f"Mejor experimento según {args.metric} ({args.direction}): {best}.",
            {"comparacion": table},
            metrics={"mejor_valor": float(valid.loc[best])},
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
