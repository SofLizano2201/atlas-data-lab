"""Scientific model implementation, independent of agents and Streamlit."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import ensemble, linear_model, neighbors, neural_network, svm, tree

from atlas.core.exceptions import FrameworkError, OptionalDependencyError

from .experiment import Experiment
from .metrics import regression_metrics
from .supervised import SupervisedLearning
from .visualization import PALETTE, SEQUENTIAL, note, plot, style

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


class Regression(SupervisedLearning):
    """Fit regression estimators and preserve the professor's relative-error definition."""

    problem = "regresion"

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)

    @staticmethod
    def algorithms() -> list[str]:
        """Provide algorithms using validated inputs."""
        return [*REGRESSORS, "xgboost"]

    def train(
        self, algorithm: str, target: str, predictors: list[str], **config: Any
    ) -> Experiment:
        """Fit the existing estimator workflow and record reproducible experiment results."""
        if algorithm == "linear_simple":
            if len(predictors) != 1:
                raise FrameworkError("Regresión simple requiere exactamente un predictor numérico.")
            self._columns(predictors, True)
        return super().train(algorithm, target, predictors, **config)

    def create_estimator(self, algorithm: str, parameters: dict[str, Any]) -> Any:
        """Build a registered estimator with validated resource bounds."""
        if algorithm == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise OptionalDependencyError("Instale pip install '.[xgboost]'.") from exc
            cls = XGBRegressor
            parameters = {"n_jobs": 1, **parameters}
        elif algorithm in REGRESSORS:
            cls = REGRESSORS[algorithm]
        else:
            raise FrameworkError("Regresor desconocido.")
        defaults = {}
        if "random_state" in cls().get_params():
            defaults["random_state"] = self.random_state
        if algorithm.startswith("svr_"):
            defaults.update(kernel=algorithm.split("_")[1], C=100, epsilon=0.1)
        if algorithm == "lasso":
            defaults["alpha"] = 0.1
        if algorithm.startswith(("lasso", "elasticnet")):
            defaults["max_iter"] = 5000
        if algorithm == "mlp":
            defaults["max_iter"] = 1000
        return cls(**defaults | parameters)

    def evaluate(
        self, model: Any, x: pd.DataFrame, y: pd.Series, pred: np.ndarray
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Calculate existing task metrics from the stored evaluation partition."""
        p = model.named_steps["preprocess"].transform(x).shape[1]
        values = regression_metrics(y.to_numpy(), pred, p)
        if not isinstance(model.named_steps["model"], linear_model.LinearRegression):
            values.pop("R2_ajustado")
        return (values, {"residuos": y.to_numpy() - pred})

    def plot_result(self, result: Experiment, kind: str = "actual_predicted") -> Any:
        """Build an interpretable diagnostic from already calculated experiment results."""
        data = result.predictions.assign(residuo=result.details["residuos"])
        data["Registro"] = data.index.astype(str)
        data["Error absoluto"] = data.residuo.abs()
        shown = data.sample(min(len(data), 10000), random_state=self.random_state)
        hover = {"real": True, "prediccion": True, "residuo": True, "Registro": True}
        if kind == "actual_predicted":
            fig = px.scatter(
                shown,
                x="real",
                y="prediccion",
                color="Error absoluto",
                color_continuous_scale=SEQUENTIAL,
                hover_data=hover,
                labels={
                    "real": "Valor real",
                    "prediccion": "Predicción",
                    "residuo": "Error · real − predicción",
                },
            )
            fig.update_traces(name="Observaciones", showlegend=True)
            low, high = (
                min(data.real.min(), data["prediccion"].min()),
                max(data.real.max(), data["prediccion"].max()),
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
            return note(
                fig,
                f"{len(shown):,} de {len(data):,} observaciones de prueba. La línea representa predicciones perfectas; el color muestra la magnitud del error. No se modificaron las métricas.",
            )
        if kind == "residuals":
            fig = style(
                px.scatter(
                    shown,
                    x="prediccion",
                    y="residuo",
                    color="Error absoluto",
                    color_continuous_scale=SEQUENTIAL,
                    hover_data=hover,
                ),
                "¿Dónde se producen los errores de predicción?",
            )
            fig.add_hline(y=0, line_dash="dash", line_color=PALETTE[1])
            fig.update_xaxes(title="Valor predicho")
            fig.update_yaxes(title="Residuo · real − predicho")
            return note(
                fig,
                f"{len(shown):,} de {len(data):,} observaciones de prueba. Sobre cero: el valor real fue mayor que la predicción. Bajo cero: fue menor.",
            )
        fig = plot(data, "qq" if kind == "qq" else "histogram", x="residuo")
        if kind != "qq":
            fig.add_vline(x=0, line_dash="dash", line_color=PALETTE[1])
        return fig

    def compare_lasso(self) -> pd.DataFrame:
        """Provide compare lasso using validated inputs."""
        from .experiment import compare

        return compare([e for e in self.experiments if e.name in {"lasso", "lasso_cv"}])

    def compare_ridge(self) -> pd.DataFrame:
        """Provide compare ridge using validated inputs."""
        from .experiment import compare

        return compare([e for e in self.experiments if e.name in {"ridge", "ridge_cv"}])

    def compare_svm(self) -> pd.DataFrame:
        """Provide compare svm using validated inputs."""
        from .experiment import compare

        return compare([e for e in self.experiments if e.name.startswith("svr_")])
