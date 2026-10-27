"""Flujo compartido de entrenamiento, validación, búsqueda y persistencia."""

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

from .eda import AnalisisDatosExploratorio
from .exceptions import FrameworkError
from .limits import validate_parameter_budget
from .preprocessing import make_preprocessor, validate_features
from .results import Experiment, compare


class Supervisado(AnalisisDatosExploratorio):
    """EDA heredado y contrato único para predicción supervisada sin fuga de datos."""

    problem = "supervisado"

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)
        self.experimentos: list[Experiment] = []

    def identificar_problema(self, objetivo: str) -> str:
        """Sugerir tipo sin impedir la selección explícita del usuario."""
        self._columns([objetivo])
        kind = type_of_target(self._df[objetivo].dropna())
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

    def entrenar(
        self,
        algoritmo: str,
        objetivo: str,
        predictores: list[str],
        test_size: float = 0.2,
        cv: int = 5,
        scaler: str = "standard",
        encoding: str = "onehot",
        imputation: str = "median",
        parametros: dict[str, Any] | None = None,
        busqueda: dict[str, list[Any]] | None = None,
        tipo_busqueda: str = "grid",
        n_iter: int = 10,
    ) -> Experiment:
        """Separar antes de ajustar; CV y búsqueda incluyen todo el preprocesamiento."""
        self._columns([objetivo])
        self._columns(predictores)
        if objetivo in predictores:
            raise FrameworkError("La variable objetivo no puede ser un predictor.")
        if self.problem not in {"clasificacion", "regresion"}:
            raise FrameworkError("Use Clasificacion o Regresion para entrenar.")
        if not 0.05 <= test_size <= 0.5 or len(self._df) < 10:
            raise FrameworkError("Use test_size entre .05 y .5 y al menos 10 filas.")
        if self._df[objetivo].isna().any():
            raise FrameworkError("La variable objetivo contiene nulos; decida cómo tratarlos.")
        y = self._df[objetivo]
        if self.problem == "regresion" and (
            not pd.api.types.is_numeric_dtype(y) or not np.isfinite(y).all()
        ):
            raise FrameworkError("Regresión requiere un objetivo numérico finito.")
        if self.problem == "clasificacion":
            if type_of_target(y) not in {"binary", "multiclass"} or y.nunique() < 2:
                raise FrameworkError("Clasificación requiere 2+ clases discretas.")
            if y.value_counts().min() < 2:
                raise FrameworkError("Cada clase requiere al menos dos observaciones.")
            if y.nunique() > 100:
                raise FrameworkError(
                    "Más de 100 clases: revise si el objetivo es un identificador."
                )
        x = validate_features(self._df[predictores])
        category_count = sum(x[c].nunique() for c in x.select_dtypes(exclude=np.number))
        if category_count > 1000 and encoding == "onehot":
            raise FrameworkError("Demasiadas categorías; excluya identificadores o use ordinal.")
        width = category_count + len(predictores) if encoding == "onehot" else len(predictores)
        if width * len(x) > 10_000_000:
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
        estimator = self.crear_estimador(algoritmo, parametros or {})
        if algoritmo.startswith(("svm_", "svr_")) and len(x_train) > 10000:
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
        # Regularización CV: buscar en el pipeline completo, evitando ajustar escaladores fuera de folds.
        if algoritmo in {"ridge_cv", "lasso_cv", "elasticnet_cv"}:
            if busqueda is not None:
                raise FrameworkError(
                    "Los modelos CV ya tienen su búsqueda; use el modelo base para otra."
                )
            busqueda = {"alpha": list(np.logspace(-4, 2, 15))}
            if algoritmo == "elasticnet_cv":
                busqueda["l1_ratio"] = [0.1, 0.5, 0.9]
        fitted: Any = pipe
        if busqueda:
            prefix = (
                "model__estimator__"
                if algoritmo == "xgboost" and self.problem == "clasificacion"
                else "model__"
            )
            grid = {k if "__" in k else f"{prefix}{k}": v for k, v in busqueda.items()}
            for key, candidates in grid.items():
                for value in candidates:
                    validate_parameter_budget(key, value)
                if any(callable(v) for v in candidates):
                    raise FrameworkError("No se aceptan funciones como candidatos de búsqueda.")
            combinations = int(np.prod([len(v) for v in grid.values()]))
            if combinations < 1 or combinations > 100 or not 1 <= n_iter <= 100:
                raise FrameworkError("Búsqueda limitada a 100 combinaciones; listas no vacías.")
            if tipo_busqueda == "grid":
                fitted = GridSearchCV(
                    pipe, grid, cv=splitter, scoring=score, n_jobs=1, error_score="raise"
                )
            elif tipo_busqueda == "random":
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
        values, details = self.evaluar(final, x_test, y_test, pred)
        if hasattr(fitted, "best_score_"):
            # Este valor es de selección; no es una estimación no sesgada del modelo seleccionado.
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
            values["CV_media"], values["CV_std"] = float(scores.mean()), float(scores.std())
        config = {
            "objetivo": objetivo,
            "predictores": predictores,
            "test_size": test_size,
            "random_state": self.random_state,
            "cv": cv,
            "scaler": scaler,
            "encoding": encoding,
            "imputation": imputation,
            "parametros": parametros or {},
            "busqueda": busqueda,
            "tipo_busqueda": tipo_busqueda,
            "mejores_parametros": getattr(fitted, "best_params_", {}),
        }
        cohort = sha256(
            pd.util.hash_pandas_object(self._df, index=True).values.tobytes()
            + json.dumps(
                {k: config[k] for k in ["objetivo", "predictores", "test_size", "random_state"]},
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
            algoritmo,
            self.problem,
            values,
            perf_counter() - start,
            final,
            predictions,
            config,
            cohort,
            details,
        )
        self.experimentos.append(result)
        result.record_provenance(self._df, len(self.experimentos))
        return result

    def crear_estimador(self, algoritmo: str, parametros: dict[str, Any]) -> Any:
        """Contrato implementado por las clases especializadas."""
        raise FrameworkError("Seleccione una clase concreta de aprendizaje supervisado.")

    def evaluar(
        self, model: Any, x: pd.DataFrame, y: pd.Series, pred: np.ndarray
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Contrato implementado por clasificación y regresión."""
        raise FrameworkError("La evaluación requiere una clase concreta.")

    def predecir(self, resultado: Experiment, df: pd.DataFrame) -> pd.DataFrame:
        """Predicción de nuevos datos usando transformadores ya ajustados."""
        columns = resultado.config["predictores"]
        if any(c not in df for c in columns):
            raise FrameworkError("Faltan predictores requeridos por el modelo.")
        out = df.copy()
        out["prediccion"] = resultado.model.predict(validate_features(df[columns]))
        return out

    def importancia(self, resultado: Experiment) -> pd.DataFrame:
        """Importancia por permutación en prueba; interpretación, no selección de variables."""
        if len(resultado.details["x_test"]) > 2000:
            raise FrameworkError("Importancia limitada a 2000 filas de prueba.")
        importance = permutation_importance(
            resultado.model,
            resultado.details["x_test"],
            resultado.details["y_test"],
            scoring=resultado.details["scoring"],
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=1,
        )
        return pd.DataFrame(
            {
                "variable": resultado.config["predictores"],
                "importancia": importance.importances_mean,
                "std": importance.importances_std,
            }
        )

    def coeficientes(self, resultado: Experiment) -> pd.DataFrame:
        """Coeficientes en el espacio codificado y escalado."""
        estimator = resultado.model.named_steps["model"]
        if not hasattr(estimator, "coef_"):
            raise FrameworkError("El modelo no ofrece coeficientes.")
        names = resultado.model.named_steps["preprocess"].get_feature_names_out()
        coef = np.atleast_2d(estimator.coef_)
        return pd.DataFrame(
            coef.T, index=names, columns=[f"coef_{i}" for i in range(coef.shape[0])]
        )

    def curva_aprendizaje(self, resultado: Experiment) -> pd.DataFrame:
        """Aprendizaje usando exclusivamente la partición de entrenamiento."""
        sizes, train, valid = learning_curve(
            clone(resultado.model),
            resultado.details["x_train"],
            resultado.details["y_train"],
            cv=resultado.details["cv"],
            scoring=resultado.details["scoring"],
            train_sizes=[0.5, 0.75, 1.0],
            shuffle=True,
            random_state=self.random_state,
            n_jobs=1,
            error_score="raise",
        )
        return pd.DataFrame(
            {"filas": sizes, "entrenamiento": train.mean(axis=1), "validacion": valid.mean(axis=1)}
        )

    def comparar(self) -> pd.DataFrame:
        """Comparar ejecuciones almacenadas sin volver a entrenarlas."""
        return compare(self.experimentos)
