"""Preprocesamiento reproducible, ajustado dentro de cada partición."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from .exceptions import FrameworkError

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "normalizer": Normalizer,
}


def make_preprocessor(
    df: pd.DataFrame, scaler: str = "standard", encoding: str = "onehot", imputation: str = "median"
) -> ColumnTransformer:
    """Construir un transformador sin ajustarlo; categorías desconocidas son admisibles."""
    if scaler not in {*SCALERS, "none"} or encoding not in {"onehot", "ordinal"}:
        raise FrameworkError("Escalamiento o codificación no reconocido.")
    if imputation not in {"mean", "median", "most_frequent", "constant"}:
        raise FrameworkError("Estrategia de imputación inválida.")
    numeric = list(df.select_dtypes(include=np.number).columns)
    categorical = [c for c in df.columns if c not in numeric]
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy=imputation, keep_empty_features=True))
    ]
    if scaler != "none":
        steps.append(("scale", SCALERS[scaler]()))
    encoder = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if encoding == "onehot"
        else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transformers.append(
            (
                "category",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="(faltante)",
                                keep_empty_features=True,
                            ),
                        ),
                        ("encode", encoder),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=True)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizar categorías sin aprender estadísticas del dataset."""
    result = df.copy(deep=True)
    for c in result.columns:
        if pd.api.types.is_numeric_dtype(result[c]):
            if np.isinf(result[c].dropna().to_numpy(dtype=float)).any():
                raise FrameworkError(f"{c}: contiene infinitos; conviértalos antes de entrenar.")
        else:
            result[c] = result[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return result
