"""Regresión y equivalencia conceptual con la guía del profesor."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import ensemble, linear_model, neighbors, neural_network, svm, tree

from .exceptions import FrameworkError, OptionalDependencyError
from .metrics import regression_metrics
from .results import Experiment
from .supervisado import Supervisado
from .visualization import PALETTE, plot, style

REGRESSORS = {
    "linear_simple": linear_model.LinearRegression,
    "linear_multiple": linear_model.LinearRegression,
    "ridge": linear_model.Ridge,
    "ridge_cv": linear_model.Ridge,
    "lasso": linear_model.Lasso,
    "lasso_cv": linear_model.Lasso,
    "elasticnet": linear_model.ElasticNet,
    "elasticnet_cv": linear_model.ElasticNet,
    "svr_linear": svm.SVR,
    "svr_rbf": svm.SVR,
    "svr_poly": svm.SVR,
    "tree": tree.DecisionTreeRegressor,
    "forest": ensemble.RandomForestRegressor,
    "extra_trees": ensemble.ExtraTreesRegressor,
    "gradient_boosting": ensemble.GradientBoostingRegressor,
    "hist_gradient_boosting": ensemble.HistGradientBoostingRegressor,
    "knn": neighbors.KNeighborsRegressor,
    "mlp": neural_network.MLPRegressor,
}


class Regresion(Supervisado):
    """Modelos y errores de regresión sobre el ciclo supervisado heredado."""

    problem = "regresion"

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)

    @staticmethod
    def algoritmos() -> list[str]:
        """Catálogo de regresores, incluyendo integración opcional."""
        return [*REGRESSORS, "xgboost"]

    def entrenar(
        self, algoritmo: str, objetivo: str, predictores: list[str], **config: Any
    ) -> Experiment:
        """Regresión simple exige exactamente una variable numérica."""
        if algoritmo == "linear_simple":
            if len(predictores) != 1:
                raise FrameworkError("Regresión simple requiere exactamente un predictor numérico.")
            self._columns(predictores, True)
        return super().entrenar(algoritmo, objetivo, predictores, **config)

    def crear_estimador(self, algoritmo: str, parametros: dict[str, Any]) -> Any:
        """Instanciar sin entrenar; Gradient Boosting y XGBoost son diferentes."""
        if algoritmo == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise OptionalDependencyError("Instale pip install '.[xgboost]'.") from exc
            cls = XGBRegressor
            parametros = {"n_jobs": 1, **parametros}
        elif algoritmo in REGRESSORS:
            cls = REGRESSORS[algoritmo]
        else:
            raise FrameworkError("Regresor desconocido.")
        defaults = {}
        if "random_state" in cls().get_params():
            defaults["random_state"] = self.random_state
        if algoritmo.startswith("svr_"):
            defaults.update(kernel=algoritmo.split("_")[1], C=100, epsilon=0.1)
        if algoritmo == "lasso":
            defaults["alpha"] = 0.1
        if algoritmo.startswith(("lasso", "elasticnet")):
            defaults["max_iter"] = 5000
        if algoritmo == "mlp":
            defaults["max_iter"] = 1000
        return cls(**(defaults | parametros))

    def evaluar(
        self, model: Any, x: pd.DataFrame, y: pd.Series, pred: np.ndarray
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """R² ajustado solo como diagnóstico para regresión lineal con n > p+1."""
        p = model.named_steps["preprocess"].transform(x).shape[1]
        values = regression_metrics(y.to_numpy(), pred, p)
        if not isinstance(model.named_steps["model"], linear_model.LinearRegression):
            values.pop("R2_ajustado")
        return values, {"residuos": y.to_numpy() - pred}

    def grafico_resultado(self, resultado: Experiment, tipo: str = "actual_predicted") -> Any:
        """Valores reales, residuos, distribución y Q-Q."""
        data = resultado.predictions.assign(residuo=resultado.details["residuos"])
        if tipo == "actual_predicted":
            fig = px.scatter(data, x="real", y="prediccion", color_discrete_sequence=PALETTE)
            fig.update_traces(name="Observaciones", showlegend=True)
            low, high = (
                min(data.real.min(), data.prediccion.min()),
                max(data.real.max(), data.prediccion.max()),
            )
            fig.add_scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="Predicción perfecta · y = x",
                line={"color": PALETTE[1], "dash": "dash"},
            )
            style(fig, "Valores reales frente a predichos")
            fig.update_xaxes(title="Valor real", range=[low, high])
            fig.update_yaxes(title="Valor predicho", range=[low, high])
            return fig
        if tipo == "residuals":
            fig = style(
                px.scatter(data, x="prediccion", y="residuo", color_discrete_sequence=PALETTE),
                "Residuos frente a la predicción",
            )
            fig.add_hline(y=0, line_dash="dash", line_color=PALETTE[1])
            fig.update_xaxes(title="Valor predicho")
            fig.update_yaxes(title="Residuo · real − predicho")
            return fig
        return plot(data, "qq" if tipo == "qq" else "histogram", x="residuo")

    def comparar_lasso(self) -> pd.DataFrame:
        """Comparar Lasso y Lasso CV ya entrenados."""
        from .results import compare

        return compare([e for e in self.experimentos if e.name in {"lasso", "lasso_cv"}])

    def comparar_ridge(self) -> pd.DataFrame:
        """Comparar Ridge y Ridge CV sin reentrenamiento."""
        from .results import compare

        return compare([e for e in self.experimentos if e.name in {"ridge", "ridge_cv"}])

    def comparar_svm(self) -> pd.DataFrame:
        """Comparar kernels SVR almacenados."""
        from .results import compare

        return compare([e for e in self.experimentos if e.name.startswith("svr_")])
