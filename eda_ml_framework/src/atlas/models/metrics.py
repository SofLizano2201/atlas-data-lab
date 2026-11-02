"""Scientific model implementation, independent of agents and Streamlit."""

import numpy as np
from sklearn import metrics as m


def regression_metrics(y: np.ndarray, pred: np.ndarray, p: int) -> dict[str, float]:
    """Provide regression metrics using validated inputs."""
    y, pred = (np.asarray(y, float), np.asarray(pred, float))
    error = np.abs(y - pred)
    mse = m.mean_squared_error(y, pred)
    r2 = m.r2_score(y, pred) if len(y) > 1 else np.nan
    denominator = np.abs(y) + np.abs(pred)
    return {
        "MAE": m.mean_absolute_error(y, pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2,
        "R2_ajustado": 1 - (1 - r2) * (len(y) - 1) / (len(y) - p - 1) if len(y) > p + 1 else np.nan,
        "MAPE": float(np.mean(error / np.abs(y))) if np.all(y != 0) else np.nan,
        "sMAPE": float(
            np.mean(
                np.divide(2 * error, denominator, out=np.zeros_like(error), where=denominator != 0)
            )
        ),
        "ER": float(error.sum() / np.abs(y).sum()) if np.abs(y).sum() else np.nan,
    }


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Provide classification metrics using validated inputs."""
    result = {
        "Accuracy": m.accuracy_score(y, pred),
        "Balanced_accuracy": m.balanced_accuracy_score(y, pred),
        "F1_macro": m.f1_score(y, pred, average="macro", zero_division=0),
        "F1_weighted": m.f1_score(y, pred, average="weighted", zero_division=0),
    }
    for name, function in [
        ("Precision", m.precision_score),
        ("Recall", m.recall_score),
        ("F1", m.f1_score),
    ]:
        result[name] = function(y, pred, average="macro", zero_division=0)
    return result


def clustering_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Provide clustering metrics using validated inputs."""
    keep = labels != -1
    clean, groups = (x[keep], labels[keep])
    result = {"clusters": float(len(np.unique(groups))), "ruido": float(np.mean(~keep))}
    if 1 < len(np.unique(groups)) < len(clean):
        result.update(
            Silhouette=m.silhouette_score(
                clean, groups, sample_size=min(2000, len(clean)), random_state=42
            ),
            Davies_Bouldin=m.davies_bouldin_score(clean, groups),
            Calinski_Harabasz=m.calinski_harabasz_score(clean, groups),
        )
    return result
