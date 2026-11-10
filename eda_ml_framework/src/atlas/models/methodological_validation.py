"""Local evidence-based policies; never fit estimators or mutate datasets."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from atlas.core.exceptions import FrameworkError


@dataclass(frozen=True)
class MethodologicalFinding:
    finding_id: str
    category: str
    severity: Literal["blocking", "warning", "suggestion"]
    title: str
    explanation: str
    recommended_action: str
    affected_component: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self):
        return self.severity == "blocking"


class MethodologicalValidation:
    @staticmethod
    def validate(eda, config=None, *, train=None, test=None, metadata=None, labels=None):
        config, metadata = config or {}, metadata or {}
        df, findings = eda.df, []

        def add(code, severity, title, why, action, component="data", **evidence):
            findings.append(
                MethodologicalFinding(
                    code, component, severity, title, why, action, component, evidence
                )
            )

        profile = eda.profile()
        if profile["missing_cells"]:
            add(
                "missing",
                "warning",
                "Hay información faltante",
                "Puede afectar la muestra y la interpretación.",
                "Revisa su distribución y elige un tratamiento justificado.",
                count=profile["missing_cells"],
            )
        constants = eda.constant_columns()
        if constants:
            add(
                "constants",
                "suggestion",
                "Hay variables constantes",
                "No aportan variación para distinguir observaciones.",
                "Revisa si deben participar en el análisis.",
                columns=constants,
            )
        if profile["additional_duplicates"]:
            add(
                "duplicates",
                "warning",
                "Hay registros repetidos",
                "Pueden representar repeticiones reales o duplicados de carga.",
                "Verifica su origen antes de eliminarlos.",
                count=profile["additional_duplicates"],
            )
        predictors = config.get("predictors", config.get("columns", []))
        selected = [c for c in predictors if c in df]
        numeric = df[selected].select_dtypes(include=np.number)
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            add(
                "infinite_features",
                "blocking",
                "Hay valores numéricos infinitos",
                "El modelo no puede utilizar estos valores.",
                "Revisa su origen antes de transformarlos.",
                "features",
            )
        if config.get("algorithm") in {
            "kmeans",
            "minibatch",
            "gaussian",
            "agglomerative",
            "spectral",
            "birch",
            "hierarchical",
            "kmedoids",
        }:
            k = config.get("n_clusters", config.get("n_components", 3))
            if (
                not isinstance(k, int)
                or not 2 <= k < len(df)
                or (selected and k > len(df[selected].drop_duplicates()))
            ):
                add(
                    "invalid_k",
                    "blocking",
                    "La cantidad de grupos no es válida",
                    "Debe ser compatible con las observaciones y su variedad.",
                    "Reduce la cantidad de grupos.",
                    "model",
                )
        target = config.get("target")
        if target and target in predictors:
            add(
                "target_leakage",
                "blocking",
                "El objetivo está entre los predictores",
                "Esto produciría una evaluación engañosa.",
                "Excluye el objetivo de los predictores.",
                "features",
            )
        if target and target not in df:
            add(
                "invalid_target",
                "blocking",
                "El objetivo no está disponible",
                "No existe en esta versión.",
                "Selecciona otro objetivo.",
                "target",
            )
        if target in df:
            y = df[target]
            problem = {"classification": "clasificacion", "regression": "regresion"}.get(
                config.get("problem"), config.get("problem")
            )
            if y.isna().any():
                add(
                    "missing_target",
                    "blocking",
                    "El objetivo contiene valores ausentes",
                    "Faltan etiquetas para evaluar esas filas.",
                    "Decide cómo tratarlas.",
                    "target",
                )
            if y.nunique() < 2:
                add(
                    "constant_target",
                    "blocking",
                    "El objetivo no tiene variación",
                    "No permite evaluar aprendizaje de diferencias.",
                    "Revisa el objetivo.",
                    "target",
                )
            if problem == "clasificacion":
                counts = y.value_counts()
                if len(counts) > 1 and counts.min() / counts.sum() < 0.1:
                    add(
                        "imbalance",
                        "warning",
                        "Hay clases poco frecuentes",
                        "Accuracy puede ocultar errores en la clase menos frecuente.",
                        "Revisa métricas por clase y la distribución de errores.",
                        "metrics",
                        counts=counts.to_dict(),
                    )
            if (
                problem == "regresion"
                and pd.api.types.is_numeric_dtype(y)
                and (y.abs() < 1e-8).any()
            ):
                add(
                    "mape_domain",
                    "warning",
                    "MAPE requiere precaución cerca de cero",
                    "Dividir por objetivos nulos o pequeños hace esta métrica indefinida o inestable.",
                    "Revisa MAE y RMSE junto con las unidades del objetivo.",
                    "metrics",
                )
            if problem in {"clasificacion", "regresion"} and not y.isna().any():
                from sklearn.model_selection import train_test_split
                from sklearn.utils.multiclass import type_of_target

                from .supervised import SupervisedLearning

                if problem == "clasificacion" and type_of_target(y) not in {"binary", "multiclass"}:
                    add(
                        "target_type",
                        "blocking",
                        "El objetivo no contiene clases discretas",
                        "La tarea seleccionada no corresponde a este tipo de objetivo.",
                        "Revisa la tarea.",
                        "target",
                    )
                elif problem == "regresion" and (
                    not pd.api.types.is_numeric_dtype(y) or not np.isfinite(y).all()
                ):
                    add(
                        "target_type",
                        "blocking",
                        "Regresión requiere un objetivo numérico finito",
                        "Este objetivo no se puede evaluar como una cantidad.",
                        "Revisa el objetivo.",
                        "target",
                    )
                elif "cv" in config:
                    try:
                        y_train, y_test = train_test_split(
                            y,
                            test_size=config.get("test_size", 0.2),
                            random_state=config.get("random_state", 42),
                            stratify=y if problem == "clasificacion" else None,
                        )
                        validator = SupervisedLearning(df, config.get("random_state", 42))
                        validator.problem = problem
                        validator._cv(y_train, config["cv"])
                        if len(y_test) < 2:
                            raise FrameworkError(
                                "La prueba necesita al menos dos observaciones para R²."
                            )
                    except (ValueError, FrameworkError) as exc:
                        add(
                            "invalid_validation",
                            "blocking",
                            "La validación no es compatible con los datos",
                            str(exc),
                            "Revisa el tamaño de prueba y la cantidad de folds.",
                            "validation",
                        )
        for column in metadata.get("target_derived_features", []) + metadata.get(
            "post_outcome_features", []
        ):
            if column in selected:
                add(
                    "unavailable:" + column,
                    "blocking",
                    "Un predictor contiene información no disponible al predecir",
                    "Los metadatos lo identifican como derivado del objetivo o posterior al resultado.",
                    "Excluye la variable o corrige los metadatos.",
                    "features",
                    column=column,
                )
        for operation in ("preprocessing", "feature_selection", "resampling"):
            if metadata.get(operation + "_fit_scope") == "full_dataset":
                add(
                    operation + "_leakage",
                    "blocking",
                    "Se utilizó información de evaluación",
                    "El ajuste con el conjunto completo contamina la evaluación independiente.",
                    "Repite el ajuste dentro de cada partición de entrenamiento.",
                    "preprocessing",
                    operation=operation,
                )
        if train is not None and test is not None:
            common = [c for c in train if c in test]
            overlap = set(pd.util.hash_pandas_object(train[common], index=False)) & set(
                pd.util.hash_pandas_object(test[common], index=False)
            )
            if overlap:
                add(
                    "split_duplicates",
                    "warning",
                    "Hay observaciones iguales entre particiones",
                    "La igualdad no prueba una duplicación accidental; puede sobreestimar la generalización.",
                    "Verifica su origen y considera separar por entidad.",
                    "validation",
                    count=len(overlap),
                )
            group = metadata.get("group_column")
            if group in common and set(train[group].dropna()) & set(test[group].dropna()):
                add(
                    "group_leakage",
                    "warning",
                    "Hay entidades compartidas entre particiones",
                    "Esta partición no evalúa generalización a entidades nuevas.",
                    "Confirma el objetivo y considera separar por grupos.",
                    "validation",
                )
            time = metadata.get("time_column")
            if (
                time in common
                and metadata.get("future_prediction")
                and train[time].max() >= test[time].min()
            ):
                add(
                    "temporal_leakage",
                    "blocking",
                    "La partición temporal utiliza información futura",
                    "El entrenamiento alcanza fechas del periodo de evaluación.",
                    "Separa cronológicamente antes de ajustar.",
                    "validation",
                )
        if (
            config.get("algorithm") in {"kmeans", "minibatch"}
            and config.get("metric", "euclidean") != "euclidean"
        ):
            add(
                "distance",
                "blocking",
                "La distancia no es compatible con K-Means",
                "Este método optimiza distancias euclidianas.",
                "Cambia la distancia o el método.",
                "model",
            )
        if labels is not None:
            valid = np.asarray(labels)[np.asarray(labels) != -1]
            if not 1 < len(np.unique(valid)) < len(valid):
                add(
                    "silhouette_domain",
                    "warning",
                    "No se puede calcular la silueta",
                    "Se necesitan dos grupos válidos y menos grupos que observaciones.",
                    "Revisa la solución antes de interpretarla.",
                    "metrics",
                )
        return findings

    @staticmethod
    def comparison(experiments):
        if len(experiments) < 2:
            return []
        signatures = {
            (
                e.problem,
                e.dataset_id,
                e.cohort,
                e.config.get("target"),
                e.config.get("cv"),
                e.details.get("scoring"),
            )
            for e in experiments
        }
        if len(signatures) == 1:
            return []
        return [
            MethodologicalFinding(
                "incomparable",
                "comparison",
                "blocking",
                "Los experimentos no son comparables directamente",
                "Difieren el conjunto de datos, el objetivo, la partición o el protocolo de evaluación.",
                "Selecciona experimentos compatibles; consulta los demás de forma descriptiva.",
                "comparison",
                {"experiment_ids": [e.id for e in experiments]},
            )
        ]

    @staticmethod
    def enforce(findings):
        blocked = [f for f in findings if f.blocking]
        if blocked:
            raise FrameworkError(" ".join(f.title + ". " + f.recommended_action for f in blocked))
