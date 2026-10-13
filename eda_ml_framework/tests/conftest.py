"""Fixtures pequeñas y reproducibles sin red."""

import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_blobs, make_regression


@pytest.fixture
def mixed():
    return pd.DataFrame(
        {
            "numero": [1.0, 2.0, None, 4.0, 100.0],
            "grupo": ["a", "a", "b", None, "b"],
            "constante": [1] * 5,
        }
    )


@pytest.fixture
def iris():
    return load_iris(as_frame=True).frame


@pytest.fixture
def regression():
    x, y = make_regression(n_samples=80, n_features=3, noise=1, random_state=10)
    df = pd.DataFrame(x, columns=["a", "b", "c"])
    df["target"] = y
    return df


@pytest.fixture
def blobs():
    x, _ = make_blobs(n_samples=70, centers=3, n_features=3, random_state=42)
    return pd.DataFrame(x, columns=["a", "b", "c"])
