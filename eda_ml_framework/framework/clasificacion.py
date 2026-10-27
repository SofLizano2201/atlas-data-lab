"""Clasificación binaria/multiclase y gráficos adaptados a capacidades del estimador."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import (
    ensemble,
    linear_model,
    metrics,
    naive_bayes,
    neighbors,
    neural_network,
    svm,
    tree,
)

from .estimators import EncodedClassifier
from .exceptions import FrameworkError, OptionalDependencyError
from .metrics import classification_metrics
from .results import Experiment
from .supervisado import Supervisado
from .visualization import PALETTE, SEQUENTIAL, categorical_colors, style

CLASSIFIERS = {
    "logistic": linear_model.LogisticRegression,
    "knn": neighbors.KNeighborsClassifier,
    "tree": tree.DecisionTreeClassifier,
    "forest": ensemble.RandomForestClassifier,
    "extra_trees": ensemble.ExtraTreesClassifier,
    "naive_bayes": naive_bayes.GaussianNB,
    "svm_linear": svm.SVC,
    "svm_rbf": svm.SVC,
    "svm_poly": svm.SVC,
    "adaboost": ensemble.AdaBoostClassifier,
    "gradient_boosting": ensemble.GradientBoostingClassifier,
    "hist_gradient_boosting": ensemble.HistGradientBoostingClassifier,
    "mlp": neural_network.MLPClassifier,
}


class Clasificacion(Supervisado):
    """Hereda todo el ciclo supervisado y especializa modelos y evaluación."""

    problem = "clasificacion"

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)

    @staticmethod
    def algoritmos() -> list[str]:
        """Modelos disponibles y opcional XGBoost."""
        return [*CLASSIFIERS, "xgboost"]

    def crear_estimador(self, algoritmo: str, parametros: dict[str, Any]) -> Any:
        """Construir estimador; parámetros desconocidos producen error explícito."""
        if algoritmo == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise OptionalDependencyError("Instale pip install '.[xgboost]'.") from exc
            cls = XGBClassifier
            # El adaptador conserva las categorías originales, incluso textos.
            parametros = {"n_jobs": 1, **parametros}
        elif algoritmo in CLASSIFIERS:
            cls = CLASSIFIERS[algoritmo]
        else:
            raise FrameworkError("Clasificador desconocido.")
        defaults = {}
        if "random_state" in cls().get_params():
            defaults["random_state"] = self.random_state
        if algoritmo.startswith("svm_"):
            defaults["kernel"] = algoritmo.split("_")[1]
        if algoritmo in {"logistic", "mlp"}:
            defaults["max_iter"] = 1000
        estimator = cls(**(defaults | parametros))
        return EncodedClassifier(estimator) if algoritmo == "xgboost" else estimator

    def evaluar(
        self, model: Any, x: pd.DataFrame, y: pd.Series, pred: np.ndarray
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """ROC usa decision_function cuando no existe predict_proba."""
        classes = model.classes_
        values = classification_metrics(y, pred)
        details: dict[str, Any] = {
            "classes": classes,
            "confusion": metrics.confusion_matrix(y, pred, labels=classes),
            "confusion_normalized": metrics.confusion_matrix(
                y, pred, labels=classes, normalize="true"
            ),
            "report": pd.DataFrame(
                metrics.classification_report(
                    y, pred, labels=classes, output_dict=True, zero_division=0
                )
            ).T,
        }
        probabilities = model.predict_proba(x) if hasattr(model, "predict_proba") else None
        scores = (
            probabilities
            if probabilities is not None
            else model.decision_function(x)
            if (hasattr(model, "decision_function"))
            else None
        )
        if (
            len(classes) > 2
            and probabilities is None
            and getattr(model.named_steps["model"], "decision_function_shape", None) == "ovo"
        ):
            scores = None  # Scores por pares no son probabilidades/puntuaciones OvR por clase.
        if probabilities is not None:
            values["Log_loss"] = metrics.log_loss(y, probabilities, labels=classes)
        if scores is not None and len(classes) == 2:
            scores = scores[:, 1] if scores.ndim == 2 else scores
            binary = (y.to_numpy() == classes[1]).astype(int)
            values["ROC_AUC"] = metrics.roc_auc_score(binary, scores)
            values["Average_precision"] = metrics.average_precision_score(binary, scores)
            details["roc"] = metrics.roc_curve(binary, scores)
            details["pr"] = metrics.precision_recall_curve(binary, scores)
            values["F1_binario"] = metrics.f1_score(y, pred, pos_label=classes[1], zero_division=0)
            values["Precision_binaria"] = metrics.precision_score(
                y, pred, pos_label=classes[1], zero_division=0
            )
            values["Recall_binario"] = metrics.recall_score(
                y, pred, pos_label=classes[1], zero_division=0
            )
        elif scores is not None:
            details["roc_multiclase"], details["pr_multiclase"] = {}, {}
            auc, ap = [], []
            for i, label in enumerate(classes):
                binary = (y.to_numpy() == label).astype(int)
                details["roc_multiclase"][str(label)] = metrics.roc_curve(binary, scores[:, i])
                details["pr_multiclase"][str(label)] = metrics.precision_recall_curve(
                    binary, scores[:, i]
                )
                auc.append(metrics.roc_auc_score(binary, scores[:, i]))
                ap.append(metrics.average_precision_score(binary, scores[:, i]))
            values["ROC_AUC_macro_ovr"] = float(np.mean(auc))
            values["Average_precision_macro"] = float(np.mean(ap))
        return values, details

    def grafico_resultado(self, resultado: Experiment, tipo: str = "confusion") -> Any:
        """Matrices, curvas por clase, árbol o aciertos/errores."""
        d = resultado.details
        if tipo in {"confusion", "confusion_normalized"}:
            labels = [str(c) for c in d["classes"]]
            fig = style(
                px.imshow(
                    d[tipo],
                    x=labels,
                    y=labels,
                    text_auto=".1%" if tipo == "confusion_normalized" else "d",
                    labels={
                        "x": "Clase predicha",
                        "y": "Clase real",
                        "color": "Proporción" if tipo == "confusion_normalized" else "Registros",
                    },
                    zmin=0,
                    zmax=1 if tipo == "confusion_normalized" else max(1, np.max(d[tipo])),
                    color_continuous_scale=SEQUENTIAL,
                ),
                "Matriz de confusión"
                + (
                    " · proporción por clase real"
                    if tipo == "confusion_normalized"
                    else " · registros"
                ),
            )
            if tipo == "confusion_normalized":
                fig.update_coloraxes(colorbar_tickformat=".0%")
            return fig
        if tipo in {"roc", "pr"}:
            curves = d.get(f"{tipo}_multiclase", {str(d["classes"][-1]): d.get(tipo)})
            if not any(v is not None for v in curves.values()):
                raise FrameworkError("El estimador no ofrece probabilidades ni puntuaciones.")
            fig = go.Figure()
            colors = categorical_colors(curves)
            for name, points in curves.items():
                if points is None:
                    continue
                a, b, _ = points
                fig.add_scatter(
                    x=a if tipo == "roc" else b,
                    y=b if tipo == "roc" else a,
                    name=name,
                    mode="lines",
                    line={"color": colors[str(name)]},
                )
            fig.update_layout(
                xaxis_title="Tasa de falsos positivos"
                if tipo == "roc"
                else "Recall · sensibilidad",
                yaxis_title="Sensibilidad · verdaderos positivos" if tipo == "roc" else "Precisión",
            )
            if tipo == "roc":
                fig.add_scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Referencia aleatoria",
                    line={"color": "#827887", "dash": "dash", "width": 1},
                )
            style(
                fig, "Curva ROC · por clase" if tipo == "roc" else "Precisión y recall · por clase"
            )
            fig.update_xaxes(range=[0, 1], tickformat=".0%")
            fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
            return fig
        if tipo == "tree":
            estimator = resultado.model.named_steps["model"]
            if not isinstance(estimator, tree.DecisionTreeClassifier):
                raise FrameworkError("Seleccione un árbol de decisión individual.")
            fig, ax = plt.subplots(figsize=(16, 8))
            tree.plot_tree(
                estimator,
                max_depth=3,
                feature_names=resultado.model.named_steps["preprocess"].get_feature_names_out(),
                class_names=list(map(str, d["classes"])),
                rounded=True,
                precision=2,
                fontsize=9,
                ax=ax,
            )
            ax.set_title("Árbol de decisión · vista hasta profundidad 3", pad=18)
            fig.tight_layout()
            return fig
        table = resultado.predictions.assign(
            resultado=lambda v: np.where(v.real == v.prediccion, "Correcta", "Incorrecta")
        )
        return style(
            px.histogram(
                table,
                x="resultado",
                color="resultado",
                color_discrete_map={"Correcta": PALETTE[0], "Incorrecta": PALETTE[1]},
                labels={"resultado": "Predicción"},
            ),
            "Predicciones correctas e incorrectas",
        )
