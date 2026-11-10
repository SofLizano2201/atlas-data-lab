"""Controlled investigations built on ATLAS training, not a second ML pipeline."""

import json
import platform
from datetime import UTC, datetime
from hashlib import sha256
from itertools import combinations
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
from pydantic import Field, JsonValue, model_validator
from sklearn.metrics import adjusted_rand_score

from atlas.core.exceptions import FrameworkError
from atlas.core.schemas import Contract, DatasetVersion
from atlas.models import Classification, Clustering, ExploratoryDataAnalysis, Regression
from atlas.models.methodological_validation import MethodologicalFinding, MethodologicalValidation
from atlas.models.provenance import fingerprint, runtime_versions


def now():
    return datetime.now(UTC).isoformat()


class ExperimentalAlternative(Contract):
    name: str = Field(min_length=1, max_length=80)
    algorithm: str = Field(min_length=1, max_length=60)
    scaler: Literal["standard", "robust", "minmax", "none"] = "standard"
    encoding: Literal["onehot", "ordinal"] = "onehot"
    imputation: Literal["median", "mean", "most_frequent", "constant"] = "median"
    predictors: list[str] | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict, max_length=12)


class ScientificExperimentDefinition(Contract):
    name: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=5, max_length=600)
    hypothesis: str = Field(min_length=5, max_length=1000)
    hypothesis_type: Literal["exploratory", "directional", "comparative", "stability"] = (
        "comparative"
    )
    null_hypothesis: str = Field(
        default="Sin mejora en la cantidad de desempeño seleccionada.", max_length=600
    )
    alternative_hypothesis: str = Field(
        default="Mejora en la dirección especificada bajo este diseño.", max_length=600
    )
    template: Literal[
        "algorithms",
        "preprocessing",
        "features",
        "model_stability",
        "cluster_stability",
        "sensitivity",
        "custom",
    ]
    task: Literal["classification", "regression", "clustering"]
    dataset: DatasetVersion
    predictors: list[str] = Field(min_length=1, max_length=30)
    target: str | None = None
    alternatives: list[ExperimentalAlternative] = Field(min_length=1, max_length=4)
    factor: Literal["algorithm", "scaler", "features", "parameters", "seed", "custom"]
    custom_justification: str = Field(default="", max_length=1000)
    metric: str = Field(max_length=60)
    direction: Literal["min", "max"]
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44], min_length=1, max_length=5)
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    cv: int = Field(default=3, ge=2, le=5)
    validation: Literal["random_holdout_cv", "same_rows_initializations", "group", "time", "nested"]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    paired_bootstrap: bool = False
    independent_observations_confirmed: bool = False
    bootstrap_samples: int = Field(default=1000, ge=200, le=3000)
    practical_difference: float = Field(default=0.0, ge=0, le=1e12)

    @model_validator(mode="after")
    def unique_keys(self):
        if len(set(self.seeds)) != len(self.seeds) or any(not 0 <= s <= 99999 for s in self.seeds):
            raise ValueError("Usa semillas distintas entre 0 y 99999.")
        if len({a.name for a in self.alternatives}) != len(self.alternatives):
            raise ValueError("Las alternativas necesitan nombres distintos.")
        if len(set(self.predictors)) != len(self.predictors):
            raise ValueError("No repitas predictores.")
        return self

    @property
    def evaluations(self):
        return len(self.alternatives) * len(self.seeds)

    @property
    def fits(self):
        return self.evaluations * (1 if self.task == "clustering" else self.cv + 1)


TASK_MODELS = {"classification": Classification, "regression": Regression, "clustering": Clustering}
METRICS = {
    "classification": {"Accuracy": "max", "Balanced_accuracy": "max", "F1_macro": "max"},
    "regression": {"MAE": "min", "RMSE": "min", "R2": "max"},
    "clustering": {"Silhouette": "max", "Davies_Bouldin": "min", "Calinski_Harabasz": "max"},
}
LIMITATIONS = (
    "Comparación predictiva, no demuestra causalidad. Las repeticiones pueden compartir observaciones; "
    "su dispersión no es un intervalo de confianza ni una prueba de significación. "
    "Los folds no se tratan como observaciones independientes. "
    "La estabilidad no demuestra que existan grupos naturales. "
    "La reproducción puede variar entre versiones de software y hardware."
)


def parameter_options(task, algorithm):
    """Expose a small registered sensitivity surface, never arbitrary execution."""
    if task == "clustering":
        from atlas.models.clustering import ALGORITHMS

        parameters = ALGORITHMS[algorithm]().get_params()
    else:
        parameters = (
            TASK_MODELS[task](pd.DataFrame({"x": [0, 1]}))
            .create_estimator(algorithm, {})
            .get_params()
        )
    return [
        name
        for name in (
            "alpha",
            "C",
            "n_clusters",
            "n_components",
            "n_estimators",
            "max_depth",
            "n_neighbors",
            "eps",
            "min_samples",
        )
        if name in parameters
    ]


class ScientificExperiment:
    @staticmethod
    def compare_investigations(runs, *, rank=False):
        signatures, rows = [], []
        for run in runs:
            definition = run["definition"]
            signature = definition.model_dump(
                exclude={
                    "name",
                    "question",
                    "hypothesis",
                    "null_hypothesis",
                    "alternative_hypothesis",
                    "hypothesis_type",
                }
            )
            signature["dataset"] = definition.dataset.content_hash
            signatures.append(json.dumps(signature, sort_keys=True))
            for row in ScientificExperiment.summary(run).to_dict("records"):
                rows.append({"investigation": run["id"], "name": definition.name, **row})
        comparable = len(set(signatures)) == 1 and all(run["status"] == "completed" for run in runs)
        if rank and not comparable:
            raise FrameworkError(
                "Los diseños, datos o protocolos no son compatibles para una clasificación directa. Se permite una consulta descriptiva."
            )
        table = pd.DataFrame(rows)
        if rank and not table.empty:
            table = table.sort_values("mean", ascending=runs[0]["definition"].direction == "min")
        table.attrs["comparable"] = comparable
        return table

    @staticmethod
    def validate(definition, frame):
        findings = []

        def add(code, title, why, blocking=True):
            findings.append(
                MethodologicalFinding(
                    code,
                    "experiment",
                    "blocking" if blocking else "warning",
                    title,
                    why,
                    "Revisa el diseño antes de ejecutar.",
                    "experiment",
                )
            )

        if fingerprint(frame) != definition.dataset.content_hash:
            add(
                "dataset_version",
                "Los datos no corresponden a la versión del diseño",
                "Se requiere una nueva definición para usar datos modificados.",
            )
        if len(frame) > 5000 or definition.evaluations > 12 or definition.fits > 72:
            add(
                "budget",
                "El diseño supera el presupuesto interactivo",
                "Máximo: 5000 filas, 12 evaluaciones y 72 ajustes incluidos los folds.",
            )
        if (
            definition.metric not in METRICS[definition.task]
            or METRICS[definition.task].get(definition.metric) != definition.direction
        ):
            add(
                "metric",
                "La métrica o su dirección no corresponde a la tarea",
                "No se utiliza una puntuación universal entre tareas.",
            )
        expected_validation = (
            "same_rows_initializations" if definition.task == "clustering" else "random_holdout_cv"
        )
        if definition.validation != expected_validation:
            add(
                "protocol",
                "Este protocolo todavía no está disponible en el laboratorio",
                "No se sustituirá silenciosamente por una partición aleatoria.",
            )
        if definition.metadata.get("group_column") or definition.metadata.get("time_column"):
            add(
                "structured_data",
                "Los datos agrupados o temporales necesitan un protocolo específico",
                "El entrenamiento actual no permite coordinar estas particiones de forma válida.",
            )
        if definition.task != "clustering" and not definition.target:
            add(
                "target",
                "Selecciona una variable objetivo",
                "El aprendizaje supervisado requiere etiquetas.",
            )
        if definition.factor == "custom" and not definition.custom_justification.strip():
            add(
                "custom_reason",
                "Documenta las diferencias entre alternativas",
                "Una comparación personalizada requiere justificar los cambios simultáneos.",
            )
        signatures = []
        if definition.task == "clustering" and definition.target is not None:
            add(
                "clustering_target",
                "Agrupamiento no utiliza una variable objetivo",
                "Retira el objetivo del diseño y revisa las variables seleccionadas.",
            )
        if definition.template == "cluster_stability" and definition.task != "clustering":
            add(
                "stability_task",
                "La estabilidad de clusters requiere agrupamiento",
                "Selecciona la tarea correspondiente.",
            )
        model_class = TASK_MODELS[definition.task]
        eda = ExploratoryDataAnalysis(frame)
        legacy_task = {"classification": "clasificacion", "regression": "regresion"}.get(
            definition.task, definition.task
        )
        for alternative in definition.alternatives:
            try:
                json.dumps(alternative.parameters, allow_nan=False)
            except ValueError:
                add(
                    "nonfinite_parameters",
                    "Los parámetros contienen valores no finitos",
                    "Usa valores numéricos definidos.",
                )
            if (
                alternative.parameters.get("metric") == "precomputed"
                or alternative.parameters.get("affinity") == "precomputed"
            ):
                add(
                    "precomputed_distance",
                    "Este flujo no recibe matrices de distancia precomputadas",
                    "Selecciona una distancia calculada por el método a partir de las variables.",
                )
            if (
                alternative.algorithm == "agglomerative"
                and alternative.parameters.get("linkage", "ward") == "ward"
                and alternative.parameters.get("metric", "euclidean") != "euclidean"
            ):
                add(
                    "ward_distance",
                    "El enlace Ward requiere distancia euclidiana",
                    "Cambia el enlace o la distancia.",
                )
            if alternative.algorithm in {"ridge_cv", "lasso_cv", "elasticnet_cv"}:
                add(
                    "internal_search",
                    "La búsqueda interna supera el presupuesto descrito por este diseño",
                    "Selecciona el modelo base y utiliza la plantilla de sensibilidad para declarar cada alternativa.",
                )
            if definition.task == "clustering" and alternative.imputation != "median":
                add(
                    "clustering_imputation",
                    "Esta imputación no está expuesta por el entrenamiento de clustering",
                    "Usa la estrategia existente; no se cambiará silenciosamente la configuración registrada.",
                )
            if (
                definition.task == "clustering"
                and len(definition.seeds) > 1
                and alternative.algorithm in {"dbscan", "agglomerative", "birch", "optics"}
            ):
                add(
                    "deterministic:" + alternative.name,
                    "Cambiar la semilla puede no cambiar este algoritmo",
                    "La repetición mide inicializaciones, no estabilidad frente a perturbaciones del dataset.",
                    False,
                )
            features = (
                alternative.predictors
                if alternative.predictors is not None
                else definition.predictors
            )
            if (
                not features
                or len(features) > 30
                or len(set(features)) != len(features)
                or not set(features) <= set(frame)
            ):
                add(
                    "features:" + alternative.name,
                    "Hay predictores ausentes, repetidos o vacíos",
                    "Cada alternativa necesita variables disponibles en la versión guardada.",
                )
            if alternative.algorithm not in model_class.algorithms():
                add(
                    "algorithm:" + alternative.name,
                    "El algoritmo no pertenece a esta tarea",
                    "Selecciona un algoritmo registrado en ATLAS.",
                )
            if any(
                k in {"random_state", "n_jobs", "warm_start"} or "__" in k
                for k in alternative.parameters
            ):
                add(
                    "parameters:" + alternative.name,
                    "Un parámetro altera las condiciones controladas",
                    "Semillas, paralelismo y Pipeline se coordinan desde el laboratorio.",
                )
            from atlas.models.limits import validate_parameter_budget

            try:
                for key, value in alternative.parameters.items():
                    validate_parameter_budget(key, value)
                    if (
                        key == "n_init"
                        and value != "auto"
                        and (not isinstance(value, int) or not 1 <= value <= 50)
                    ):
                        raise FrameworkError("Máximo de 50 inicializaciones por ajuste.")
                    if key == "max_iter" and (not isinstance(value, int) or not 1 <= value <= 2000):
                        raise FrameworkError(
                            "Máximo de 2000 iteraciones por ajuste en el laboratorio."
                        )
                if definition.task != "clustering":
                    estimator = model_class(frame).create_estimator(
                        alternative.algorithm, dict(alternative.parameters)
                    )
                else:
                    from atlas.models.clustering import ALGORITHMS

                    if alternative.algorithm not in ALGORITHMS:
                        raise FrameworkError(
                            "El laboratorio admite los algoritmos de clustering registrados sin extensiones opcionales."
                        )
                    estimator = ALGORITHMS[alternative.algorithm](**alternative.parameters)
                if getattr(estimator, "_parameter_constraints", None) is not None:
                    estimator._validate_params()
            except (ValueError, TypeError, ImportError) as exc:
                add(
                    "estimator:" + alternative.name,
                    "La configuración del algoritmo no es válida",
                    str(exc),
                )
            if (
                definition.task == "clustering"
                and alternative.algorithm in {"agglomerative", "spectral", "optics"}
                and len(frame) > 2000
            ):
                add(
                    "quadratic_budget",
                    "El algoritmo supera el límite de 2000 filas",
                    "Reduce el dataset de forma explícita antes de diseñar el experimento.",
                )
            for seed in definition.seeds:
                findings += MethodologicalValidation.validate(
                    eda,
                    {
                        "target": definition.target,
                        "predictors": features,
                        "problem": legacy_task,
                        "algorithm": alternative.algorithm,
                        "scaler": alternative.scaler,
                        "test_size": definition.test_size,
                        "cv": definition.cv,
                        "random_state": seed,
                        **alternative.parameters,
                    },
                    metadata=definition.metadata,
                )
            signature = alternative.model_dump(exclude={"name"})
            signature["predictors"] = features
            signatures.append(signature)
        excluded = {
            "algorithm": {"algorithm", "parameters"},
            "scaler": {"scaler"},
            "features": {"predictors"},
            "parameters": {"parameters"},
        }.get(definition.factor, set())
        if definition.factor != "custom":
            controls = [
                {k: v for k, v in signature.items() if k not in excluded}
                for signature in signatures
            ]
            if any(control != controls[0] for control in controls[1:]):
                add(
                    "confounding",
                    "Cambian varias condiciones a la vez",
                    "Mantén constantes las condiciones ajenas al factor elegido.",
                )
        if (
            definition.factor == "algorithm"
            and len(signatures) > 1
            and len({s["algorithm"] for s in signatures}) != len(signatures)
        ):
            add(
                "algorithm_factor",
                "La comparación de algoritmos necesita algoritmos distintos",
                "Para variar parámetros del mismo algoritmo utiliza Sensibilidad.",
            )
        if definition.task == "clustering" and len({a.scaler for a in definition.alternatives}) > 1:
            add(
                "geometry",
                "La escala cambia la geometría de la evaluación",
                "Los índices se calculan en representaciones diferentes. Una silueta mayor no demuestra un agrupamiento más útil ni una escala universalmente superior.",
                False,
            )
        if len(signatures) > 1 and len({json.dumps(s, sort_keys=True) for s in signatures}) != len(
            signatures
        ):
            add(
                "duplicate_alternatives",
                "Hay alternativas idénticas",
                "No se ejecutan duplicados para aparentar comparaciones.",
            )
        if len(signatures) == 1 and len(definition.seeds) < 2:
            add(
                "no_comparison",
                "Faltan alternativas o repeticiones",
                "Una sola evaluación no permite comparar ni medir estabilidad.",
            )
        if definition.factor == "parameters" and len(signatures) > 1:
            keys = set().union(*(s["parameters"] for s in signatures))
            changed = [
                k
                for k in keys
                if len({json.dumps(s["parameters"].get(k), sort_keys=True) for s in signatures}) > 1
            ]
            if len(changed) != 1:
                add(
                    "parameter_factor",
                    "La sensibilidad requiere un único hiperparámetro variable",
                    "Especifica el mismo conjunto de parámetros y varía solo uno.",
                )
        if definition.paired_bootstrap and (
            definition.task == "clustering"
            or definition.metric not in {"Accuracy", "MAE"}
            or len(definition.seeds) != 1
            or len(definition.alternatives) != 2
            or not definition.independent_observations_confirmed
            or int(np.ceil(len(frame) * definition.test_size)) < 10
        ):
            add(
                "bootstrap_design",
                "El bootstrap pareado no corresponde a este diseño",
                "Requiere dos alternativas, una partición con al menos diez observaciones, Accuracy o MAE y observaciones independientes confirmadas.",
            )
        if definition.factor == "custom":
            add(
                "custom_scope",
                "La comparación cambia varias condiciones",
                "El resultado describe alternativas completas; no permite atribuir el cambio a un único factor.",
                False,
            )
        return list({(f.finding_id, f.explanation): f for f in findings}.values())

    @staticmethod
    def evaluate(definition, frame, alternative, seed):
        import warnings

        from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result, engine = ScientificExperiment._evaluate(definition, frame, alternative, seed)
        result.details["lab_warnings"] = [
            str(item.message)
            for item in captured
            if issubclass(
                item.category, (ConvergenceWarning, DataConversionWarning, RuntimeWarning)
            )
        ]
        return result, engine

    @staticmethod
    def _evaluate(definition, frame, alternative, seed):
        """Delegate every estimator fit and metric to the existing scientific models."""
        engine = TASK_MODELS[definition.task](frame, random_state=seed)
        features = (
            alternative.predictors if alternative.predictors is not None else definition.predictors
        )
        if definition.task == "clustering":
            result = engine.train(
                alternative.algorithm,
                features,
                scaler=alternative.scaler,
                encoding=alternative.encoding,
                **alternative.parameters,
            )
        else:
            result = engine.train(
                alternative.algorithm,
                definition.target,
                features,
                test_size=definition.test_size,
                cv=definition.cv,
                scaler=alternative.scaler,
                encoding=alternative.encoding,
                imputation=alternative.imputation,
                parameters=dict(alternative.parameters),
            )
        return result, engine

    @staticmethod
    def summary(run):
        table = pd.DataFrame([r for r in run["evaluations"].values() if r["status"] == "completed"])
        if table.empty:
            return pd.DataFrame()
        return (
            table.groupby("alternative", sort=False)["value"]
            .agg(
                count="count",
                mean="mean",
                std="std",
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )
            .reset_index()
        )

    @staticmethod
    def paired_differences(run):
        definition = run["definition"]
        if len(definition.alternatives) != 2:
            return pd.DataFrame()
        rows = [r for r in run["evaluations"].values() if r["status"] == "completed"]
        if not rows:
            return pd.DataFrame()
        table = pd.DataFrame(rows).pivot(index="seed", columns="alternative", values="value")
        a, b = [item.name for item in definition.alternatives]
        if a not in table or b not in table:
            return pd.DataFrame()
        # Positive always means B improves on A in the chosen performance quantity.
        difference = (table[b] - table[a]) * (1 if definition.direction == "max" else -1)
        return difference.dropna().rename("improvement").reset_index()

    @staticmethod
    def stability(run):
        if run["definition"].task != "clustering":
            return pd.DataFrame()
        rows = []
        for alternative in run["definition"].alternatives:
            results = [
                (key, result)
                for key, result in run["artifacts"].items()
                if run["evaluations"][key]["alternative"] == alternative.name
            ]
            for (key_a, a), (key_b, b) in combinations(results, 2):
                left, right = np.asarray(a.details["labels"]), np.asarray(b.details["labels"])
                if len(left) != len(right) or not a.predictions.index.equals(b.predictions.index):
                    raise FrameworkError(
                        "La estabilidad requiere las mismas observaciones alineadas."
                    )
                keep = (left != -1) & (right != -1)
                if (
                    keep.sum() < 3
                    or len(np.unique(left[keep])) < 2
                    or len(np.unique(right[keep])) < 2
                ):
                    continue
                rows.append(
                    {
                        "alternative": alternative.name,
                        "seed_a": run["evaluations"][key_a]["seed"],
                        "seed_b": run["evaluations"][key_b]["seed"],
                        "ari": float(adjusted_rand_score(left[keep], right[keep])),
                        "coverage": float(keep.mean()),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def bootstrap(run):
        from scipy.stats import bootstrap

        definition = run["definition"]
        if not definition.paired_bootstrap:
            return None
        if run["status"] != "completed" or len(run["artifacts"]) != 2:
            return None
        a, b = [run["artifacts"].get(f"{index}:{definition.seeds[0]}") for index in range(2)]
        if a is None or b is None or not a.predictions.index.equals(b.predictions.index):
            raise FrameworkError("El bootstrap requiere predicciones sobre las mismas filas.")
        ya, yb = a.details["y_test"], b.details["y_test"]
        if not ya.equals(yb) or len(ya) < 10:
            raise FrameworkError("Se necesitan al menos diez etiquetas de prueba alineadas.")
        if definition.metric == "Accuracy":
            delta = (b.predictions["prediccion"].to_numpy() == ya.to_numpy()).astype(float) - (
                a.predictions["prediccion"].to_numpy() == ya.to_numpy()
            ).astype(float)
        else:
            delta = np.abs(ya.to_numpy() - a.predictions["prediccion"].to_numpy()) - np.abs(
                ya.to_numpy() - b.predictions["prediccion"].to_numpy()
            )
        # Matched differences retain pairing; SciPy supplies the resampling
        # method and batch bounds memory. This is not a bootstrap of CV folds.
        result = bootstrap(
            (delta,),
            np.mean,
            method="percentile",
            paired=True,
            n_resamples=definition.bootstrap_samples,
            confidence_level=0.95,
            batch=100,
            random_state=np.random.default_rng(definition.seeds[0]),
        )
        low, high = result.confidence_interval
        return {
            "estimate": float(delta.mean()),
            "low": float(low),
            "high": float(high),
            "confidence": 0.95,
            "samples": definition.bootstrap_samples,
            "test_observations": len(delta),
            "method": "paired_percentile_bootstrap",
            "scope": "Incertidumbre condicional a estos modelos y esta partición; no incluye variación del entrenamiento ni corrige comparaciones múltiples."
            + (
                " El intervalo remuestreado es puntual: la ausencia de variación observada no demuestra certeza ni equivalencia entre los modelos."
                if low == high
                else ""
            ),
        }

    @staticmethod
    def interpretation(run):
        summary = ScientificExperiment.summary(run)
        if summary.empty:
            return "Todavía no hay evaluaciones válidas. Revisa el diseño y las operaciones pendientes."
        differences = ScientificExperiment.paired_differences(run)
        text = f"Se completaron {sum(summary['count'])} evaluaciones válidas. "
        if not differences.empty:
            values = differences.improvement
            tolerance = run["definition"].practical_difference
            if len(values) == 1:
                text += "Hay una sola comparación pareada; no permite afirmar consistencia entre repeticiones. "
            elif (values > tolerance).all():
                text += "La alternativa B mejoró en todas las repeticiones por encima del umbral práctico indicado; apoyo descriptivo dentro de este diseño. "
            elif (values < -tolerance).all():
                text += "La alternativa B empeoró en todas las repeticiones por encima del umbral práctico indicado; contradice la mejora esperada dentro de este diseño. "
            else:
                text += "Las diferencias no muestran una mejora uniforme por encima del umbral práctico; resultado inconcluso dentro de este diseño. "
        else:
            text += "Revisa la dispersión de cada alternativa y la estabilidad disponible; no se realiza una prueba de significación automática. "
        if run["status"] != "completed":
            text += "La ejecución está incompleta: las conclusiones son provisionales. "
        if run["definition"].task == "clustering":
            stability = ScientificExperiment.stability(run)
            if not stability.empty:
                text += f"El ARI entre soluciones de la misma alternativa varió entre {stability.ari.min():.3f} y {stability.ari.max():.3f}, con cobertura mínima de filas sin ruido de {stability.coverage.min():.1%}. "
            if len({a.scaler for a in run["definition"].alternatives}) > 1:
                text += "Los índices de clustering se calcularon en geometrías distintas: no establecen qué representación es más útil en el dominio. "
        return text + LIMITATIONS

    @staticmethod
    def charts(run):
        rows = pd.DataFrame([r for r in run["evaluations"].values() if r["status"] == "completed"])
        if rows.empty:
            return {}
        metric = run["definition"].metric
        labels = {
            "alternative": "Alternativa",
            "value": metric,
            "seed": "Semilla",
            "seconds": "Tiempo total (s)",
        }
        charts = {
            "performance": px.box(
                rows,
                x="alternative",
                y="value",
                color="alternative",
                points="all",
                hover_data=["seed"],
                labels=labels,
                title="Desempeño observado por repetición",
            ),
            "cost": px.scatter(
                rows,
                x="seconds",
                y="value",
                color="alternative",
                hover_data=["seed"],
                labels=labels,
                title="Desempeño y coste medido",
            ),
        }
        if run["definition"].template == "sensitivity":
            alternatives = run["definition"].alternatives
            parameters = set().union(*(a.parameters for a in alternatives))
            varied = [
                key
                for key in parameters
                if len({json.dumps(a.parameters.get(key), sort_keys=True) for a in alternatives})
                > 1
            ]
            if len(varied) == 1:
                parameter = varied[0]
                settings = {a.name: a.parameters.get(parameter) for a in alternatives}
                sensitivity = rows.assign(parameter=rows.alternative.map(settings))
                charts["sensitivity"] = px.scatter(
                    sensitivity,
                    x="parameter",
                    y="value",
                    color="alternative",
                    hover_data=["seed"],
                    labels={"parameter": parameter, **labels},
                    title="Sensibilidad al parámetro · valores realmente evaluados",
                )
        differences = ScientificExperiment.paired_differences(run)
        if not differences.empty:
            charts["differences"] = px.bar(
                differences,
                x="seed",
                y="improvement",
                labels={
                    "seed": "Semilla coordinada",
                    "improvement": f"Mejora B respecto a A · {metric}",
                },
                title="Diferencias pareadas observadas",
            )
        stability = ScientificExperiment.stability(run)
        if not stability.empty:
            charts["stability"] = px.scatter(
                stability,
                x="alternative",
                y="ari",
                color="alternative",
                hover_data=["seed_a", "seed_b", "coverage"],
                labels={"alternative": "Alternativa", "ari": "Índice Rand ajustado"},
                title="Estabilidad de etiquetas · filas alineadas",
            )
            charts["stability"].update_yaxes(range=[-1, 1])
        interval = run.get("uncertainty")
        if interval:
            charts["interval"] = px.scatter(
                pd.DataFrame([{"improvement": interval["estimate"], "comparison": "B frente a A"}]),
                x="improvement",
                y="comparison",
                labels={"improvement": f"Mejora · {metric}", "comparison": "Comparación"},
                title="Intervalo bootstrap pareado · 95%",
            )
            charts["interval"].add_scatter(
                x=[interval["low"], interval["high"]],
                y=["B frente a A", "B frente a A"],
                mode="lines+markers",
                name="Intervalo percentil 95%",
                hovertemplate="Límite: %{x:.4g}<extra>Bootstrap pareado</extra>",
            )
        return charts

    @staticmethod
    def manifest(run):
        return {
            "schema_version": 1,
            "id": run["id"],
            "parent_id": run.get("parent_id"),
            "definition": run["definition"].model_dump(mode="json"),
            "status": run["status"],
            "created_at": run["created_at"],
            "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at"),
            "environment": run["environment"],
            "evaluations": list(run["evaluations"].values()),
            "warnings": run["warnings"],
            "attempts": run.get("attempts", []),
            "summary": ScientificExperiment.summary(run).replace({np.nan: None}).to_dict("records"),
            "stability": ScientificExperiment.stability(run).to_dict("records"),
            "uncertainty": run.get("uncertainty"),
            "interpretation": ScientificExperiment.interpretation(run),
            "artifacts": {key: value.id for key, value in run["artifacts"].items()},
        }

    @staticmethod
    def create(definition, frame, parent_id=None):
        from importlib.metadata import version

        definition = definition.model_copy(deep=True)
        findings = ScientificExperiment.validate(definition, frame)
        MethodologicalValidation.enforce(findings)
        return {
            "id": uuid4().hex,
            "parent_id": parent_id,
            "definition": definition,
            "dataset": frame.copy(deep=True),
            "status": "ready",
            "created_at": now(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "implementation": "ATLAS scientific-lab-1",
                "libraries": runtime_versions()
                | {name: version(name) for name in ("plotly", "pydantic", "joblib")},
            },
            "evaluations": {},
            "artifacts": {},
            "attempts": [],
            "warnings": [f.explanation for f in findings],
            "design_hash": sha256(definition.model_dump_json().encode()).hexdigest(),
        }
