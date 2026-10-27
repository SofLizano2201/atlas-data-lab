"""Presupuestos de parámetros compartidos por entrenamiento y búsquedas."""

from numbers import Integral
from typing import Any

from .exceptions import FrameworkError


def validate_parameter_budget(name: str, value: Any) -> None:
    """Acotar recursos incluso si los parámetros provienen de un plan confirmado."""
    if value is None:
        return
    leaf = name.split("__")[-1]
    limits = {"n_estimators": 1000, "max_iter": 10000, "max_depth": 50, "n_jobs": 4}
    if leaf in limits:
        if leaf == "max_iter" and value == -1:
            return  # Convención de SVC; el tamaño de SVM se limita por separado.
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or not 1 <= value <= limits[leaf]
        ):
            raise FrameworkError(f"{leaf}: candidatos/valor fuera del límite {limits[leaf]}.")
    if leaf == "hidden_layer_sizes":
        layers = [value] if isinstance(value, Integral) else value
        if (
            not isinstance(layers, (list, tuple))
            or not 1 <= len(layers) <= 3
            or any(
                not isinstance(n, Integral) or isinstance(n, bool) or not 1 <= n <= 256
                for n in layers
            )
        ):
            raise FrameworkError("MLP limitado a tres capas de hasta 256 neuronas cada una.")
