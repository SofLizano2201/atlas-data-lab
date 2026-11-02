"""Scientific model implementation, independent of agents and Streamlit."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Define the EncodedClassifier contract."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, x: Any, y: Any) -> "EncodedClassifier":
        """Provide fit using validated inputs."""
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(x, self.encoder_.transform(y))
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x: Any) -> np.ndarray:
        """Provide predict using validated inputs."""
        return self.encoder_.inverse_transform(self.estimator_.predict(x).astype(int))

    def predict_proba(self, x: Any) -> np.ndarray:
        """Provide predict proba using validated inputs."""
        return self.estimator_.predict_proba(x)
