"""Backward-compatible academic API delegating to atlas.models."""

from atlas.models import (
    Classification,
    Clustering,
    ExploratoryDataAnalysis,
    Regression,
    SupervisedLearning,
    UnsupervisedLearning,
)

from ._compat import install_aliases

install_aliases()
AnalisisDatosExploratorio = ExploratoryDataAnalysis
NoSupervisado = UnsupervisedLearning
Supervisado = SupervisedLearning
Clasificacion = Classification
Regresion = Regression
