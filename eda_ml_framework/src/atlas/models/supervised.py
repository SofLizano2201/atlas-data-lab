"""Scientific model implementation, independent of agents and Streamlit."""

from __future__ import annotations

import json
from hashlib import sha256
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

from atlas.core.exceptions import FrameworkError

from .eda import ExploratoryDataAnalysis
from .experiment import Experiment, compare
from .limits import validate_parameter_budget
from .preprocessing import make_preprocessor, validate_features


class SupervisedLearning(ExploratoryDataAnalysis):
    """Share leakage-safe training, validation and interpretation workflows."""

    problem = "supervisado"

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)
        self.experiments: list[Experiment] = []

    def identify_problem(self, target: str) -> str:
        """Provide identify problem using validated inputs."""
        self._columns([target])
        kind = type_of_target(self._df[target].dropna())
        return "regresion" if kind == "continuous" else "clasificacion"

    def _cv(self, y: pd.Series, folds: int) -> Any:
        if folds < 2 or folds > 10 or len(y) < folds * 2:
            raise FrameworkError("CV requiere 2–10 folds y al menos 2 filas por fold.")
        if self.problem == "clasificacion":
            if y.value_counts().min() < folds:
                raise FrameworkError(
                    "Cada clase de entrenamiento requiere al menos cv observaciones."
                )
            return StratifiedKFold(folds, shuffle=True, random_state=self.random_state)
        return KFold(folds, shuffle=True, random_state=self.random_state)

    def train(
        self,
        algorithm: str,
        target: str,
        predictors: list[str],
        test_size: float = 0.2,
        cv: int = 5,
        scaler: str = "standard",
        encoding: str = "onehot",
        imputation: str = "median",
        parameters: dict[str, Any] | None = None,
        search: dict[str, list[Any]] | None = None,
        search_type: str = "grid",
        n_iter: int = 10,
    ) -> Experiment:
        """Fit the existing estimator workflow and record reproducible experiment results."""
        self._columns([target])
        self._columns(predictors)
        if target in predictors:
            raise FrameworkError("La variable objetivo no puede ser un predictor.")
        if self.problem not in {"clasificacion", "regresion"}:
            raise FrameworkError("Use Clasificacion o Regresion para entrenar.")
        if not 0.05 <= test_size <= 0.5 or len(self._df) < 10:
            raise FrameworkError("Use test_size entre .05 y .5 y al menos 10 filas.")
        if self._df[target].isna().any():
            raise FrameworkError("La variable objetivo contiene nulos; decida cómo tratarlos.")
        y = self._df[target]
        if self.problem == "regresion" and (
            not pd.api.types.is_numeric_dtype(y) or not np.isfinite(y).all()
        ):
            raise FrameworkError("Regresión requiere un objetivo numérico finito.")
        if self.problem == "clasificacion":
            if type_of_target(y) not in {"binary", "multiclass"} or y.nunique() < 2:
                raise FrameworkError("Clasificación requiere 2+ clases discretas.")
            if y.value_counts().min() < 2:
                counts = y.value_counts()
                rare = counts[counts < 2]
                examples = ", ".join(str(value)[:50] for value in rare.index[:5])
                raise FrameworkError(
                    f"No se puede aprender a predecir «{target}»: hay {len(rare)} categorías con un solo ejemplo (por ejemplo: {examples}). "
                    "Para separar datos de aprendizaje y de prueba, cada categoría necesita al menos dos registros; la validación puede necesitar más. "
                    "Cambiar de modelo no resuelve este problema. Si elegiste un precio, importe u otra cantidad, usa Regresión. "
                    "Si elegiste un ID, cambia la columna que quieres predecir. Si son categorías reales, reúne más ejemplos; no dupliques filas para cumplir el mínimo."
                )
            if y.nunique() > 100:
                raise FrameworkError(
                    "Más de 100 clases: revise si el objetivo es un identificador."
                )
        x = validate_features(self._df[predictors])
        category_count = sum(x[c].nunique() for c in x.select_dtypes(exclude=np.number))
        if category_count > 1000 and encoding == "onehot":
            raise FrameworkError("Demasiadas categorías; excluya identificadores o use ordinal.")
        width = category_count + len(predictors) if encoding == "onehot" else len(predictors)
        if width * len(x) > 10000000:
            raise FrameworkError(
                "La matriz densa superaría 10 millones de celdas; reduzca datos o categorías."
            )
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y if self.problem == "clasificacion" else None,
            )
        except ValueError as exc:
            raise FrameworkError(f"Partición incompatible con el tamaño o clases: {exc}") from exc
        if self.problem == "clasificacion" and set(y_train.unique()) != set(y_test.unique()):
            raise FrameworkError("La partición debe contener todas las clases en ambos conjuntos.")
        splitter = self._cv(y_train, cv)
        estimator = self.create_estimator(algorithm, parameters or {})
        if algorithm.startswith(("svm_", "svr_")) and len(x_train) > 10000:
            raise FrameworkError("SVM interactiva limitada a 10000 filas de entrenamiento.")
        for key, value in estimator.get_params().items():
            validate_parameter_budget(key, value)
        pipe = Pipeline(
            [
                ("preprocess", make_preprocessor(x, scaler, encoding, imputation)),
                ("model", estimator),
            ]
        )
        score = "f1_macro" if self.problem == "clasificacion" else "neg_root_mean_squared_error"
        if algorithm in {"ridge_cv", "lasso_cv", "elasticnet_cv"}:
            if search is not None:
                raise FrameworkError(
                    "Los modelos CV ya tienen su búsqueda; use el modelo base para otra."
                )
            search = {"alpha": list(np.logspace(-4, 2, 15))}
            if algorithm == "elasticnet_cv":
                search["l1_ratio"] = [0.1, 0.5, 0.9]
        fitted: Any = pipe
        if search:
            prefix = (
                "model__estimator__"
                if algorithm == "xgboost" and self.problem == "clasificacion"
                else "model__"
            )
            grid = {k if "__" in k else f"{prefix}{k}": v for k, v in search.items()}
            for key, candidates in grid.items():
                for value in candidates:
                    validate_parameter_budget(key, value)
                if any(callable(v) for v in candidates):
                    raise FrameworkError("No se aceptan funciones como candidatos de búsqueda.")
            combinations = int(np.prod([len(v) for v in grid.values()]))
            if combinations < 1 or combinations > 100 or (not 1 <= n_iter <= 100):
                raise FrameworkError("Búsqueda limitada a 100 combinaciones; listas no vacías.")
            if search_type == "grid":
                fitted = GridSearchCV(
                    pipe, grid, cv=splitter, scoring=score, n_jobs=1, error_score="raise"
                )
            elif search_type == "random":
                fitted = RandomizedSearchCV(
                    pipe,
                    grid,
                    n_iter=min(n_iter, combinations),
                    cv=splitter,
                    scoring=score,
                    random_state=self.random_state,
                    n_jobs=1,
                    error_score="raise",
                )
            else:
                raise FrameworkError("Use búsqueda grid o random.")
        start = perf_counter()
        fitted.fit(x_train, y_train)
        final = fitted.best_estimator_ if hasattr(fitted, "best_estimator_") else fitted
        pred = final.predict(x_test)
        values, details = self.evaluate(final, x_test, y_test, pred)
        if hasattr(fitted, "best_score_"):
            values["CV_seleccion"] = float(fitted.best_score_)
        else:
            scores = cross_val_score(
                clone(pipe),
                x_train,
                y_train,
                cv=splitter,
                scoring=score,
                n_jobs=1,
                error_score="raise",
            )
            values["CV_media"], values["CV_std"] = (float(scores.mean()), float(scores.std()))
            details["cv_scores"] = scores.tolist()
        config = {
            "target": target,
            "predictors": predictors,
            "test_size": test_size,
            "random_state": self.random_state,
            "cv": cv,
            "scaler": scaler,
            "encoding": encoding,
            "imputation": imputation,
            "parameters": parameters or {},
            "search": search,
            "search_type": search_type,
            "best_parameters": getattr(fitted, "best_params_", {}),
        }
        cohort = sha256(
            pd.util.hash_pandas_object(self._df, index=True).values.tobytes()
            + json.dumps(
                {k: config[k] for k in ["target", "predictors", "test_size", "random_state"]},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        details.update(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            cv=splitter,
            scoring=score,
        )
        predictions = x_test.copy()
        predictions["real"] = y_test
        predictions["prediccion"] = pred
        result = Experiment(
            algorithm,
            self.problem,
            values,
            perf_counter() - start,
            final,
            predictions,
            config,
            cohort,
            details,
        )
        self.experiments.append(result)
        result.record_provenance(self._df, len(self.experiments))
        return result

    def create_estimator(self, algorithm: str, parameters: dict[str, Any]) -> Any:
        """Build a registered estimator with validated resource bounds."""
        raise FrameworkError("Seleccione una clase concreta de aprendizaje supervisado.")

    def evaluate(
        self, model: Any, x: pd.DataFrame, y: pd.Series, pred: np.ndarray
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Calculate existing task metrics from the stored evaluation partition."""
        raise FrameworkError("La evaluación requiere una clase concreta.")

    def predict(self, result: Experiment, df: pd.DataFrame) -> pd.DataFrame:
        """Provide predict using validated inputs."""
        columns = result.config["predictors"]
        if any(c not in df for c in columns):
            raise FrameworkError("Faltan predictores requeridos por el modelo.")
        out = df.copy()
        out["prediccion"] = result.model.predict(validate_features(df[columns]))
        return out

    def feature_importance(self, result: Experiment) -> pd.DataFrame:
        """Calculate held-out permutation importance for interpretation, not feature selection."""
        if len(result.details["x_test"]) > 2000:
            raise FrameworkError("Importancia limitada a 2000 filas de prueba.")
        importance = permutation_importance(
            result.model,
            result.details["x_test"],
            result.details["y_test"],
            scoring=result.details["scoring"],
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=1,
        )
        return pd.DataFrame(
            {
                "variable": result.config["predictors"],
                "importancia": importance.importances_mean,
                "std": importance.importances_std,
            }
        )

    def coefficients(self, result: Experiment) -> pd.DataFrame:
        """Provide coefficients using validated inputs."""
        estimator = result.model.named_steps["model"]
        if not hasattr(estimator, "coef_"):
            raise FrameworkError("El modelo no ofrece coeficientes.")
        names = result.model.named_steps["preprocess"].get_feature_names_out()
        coef = np.atleast_2d(estimator.coef_)
        return pd.DataFrame(
            coef.T, index=names, columns=[f"coef_{i}" for i in range(coef.shape[0])]
        )

    def learning_curve(self, result: Experiment) -> pd.DataFrame:
        """Provide learning curve using validated inputs."""
        sizes, train, valid = learning_curve(
            clone(result.model),
            result.details["x_train"],
            result.details["y_train"],
            cv=result.details["cv"],
            scoring=result.details["scoring"],
            train_sizes=[0.5, 0.75, 1.0],
            shuffle=True,
            random_state=self.random_state,
            n_jobs=1,
            error_score="raise",
        )
        return pd.DataFrame(
            {"filas": sizes, "entrenamiento": train.mean(axis=1), "validacion": valid.mean(axis=1)}
        )

    def compare(self) -> pd.DataFrame:
        """Compare recorded experiments only when their problem and cohort agree."""
        return compare(self.experiments)
