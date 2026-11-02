"""Public English scientific API; legacy academic aliases live in framework."""

from .classification import Classification
from .clustering import Clustering
from .eda import ExploratoryDataAnalysis
from .regression import Regression
from .supervised import SupervisedLearning
from .unsupervised import UnsupervisedLearning

__all__ = [
    "Classification",
    "Clustering",
    "ExploratoryDataAnalysis",
    "Regression",
    "SupervisedLearning",
    "UnsupervisedLearning",
]
