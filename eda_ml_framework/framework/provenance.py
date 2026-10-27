"""Identidad de datos y metadatos reproducibles sin dependencia de Streamlit."""

import json
from hashlib import sha256
from importlib.metadata import version

import pandas as pd


def fingerprint(df: pd.DataFrame) -> str:
    """Hash de valores, índice, nombres y tipos, independiente de la identidad del objeto."""
    schema = json.dumps([(str(c), str(t)) for c, t in df.dtypes.items()], ensure_ascii=False)
    try:
        content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    except TypeError as exc:
        raise ValueError(
            "El dataset debe ser tabular, sin listas u objetos anidados por celda."
        ) from exc
    return sha256(schema.encode() + content).hexdigest()


def runtime_versions() -> dict[str, str]:
    """Versiones de bibliotecas que intervienen en los cálculos."""
    return {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "scipy"]}
