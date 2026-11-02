"""Validate and apply explicitly confirmed analysis selections without training models."""

from typing import Literal

from pydantic import Field

from atlas.core.schemas import Contract
from atlas.core.session import AgentContext
from atlas.models import Classification, Clustering, Regression


class ConfigurationArgs(Contract):
    problem: Literal["classification", "regression", "clustering"]
    predictors: list[str] = Field(min_length=1, max_length=30)
    target: str | None = None
    algorithm: str | None = None
    scaler: Literal["standard", "robust", "minmax", "normalizer", "none"] = "standard"
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    cv: int = Field(default=3, ge=2, le=10)
    random_state: int = Field(default=42, ge=0, le=99999)


class ConfigurationController:
    @staticmethod
    def apply(context: AgentContext, request: ConfigurationArgs) -> dict:
        frame = context.eda.df if request.problem == "clustering" else context.eda.original
        if len(set(request.predictors)) != len(request.predictors) or not set(
            request.predictors
        ) <= set(frame.columns):
            raise ValueError("Los predictores no existen o están repetidos.")
        if request.problem != "clustering" and (
            request.target not in frame or request.target in request.predictors
        ):
            raise ValueError("Selecciona un objetivo válido que no aparezca entre los predictores.")
        model_type = {
            "classification": Classification,
            "regression": Regression,
            "clustering": Clustering,
        }[request.problem]
        if request.algorithm and request.algorithm not in model_type.algorithms():
            raise ValueError("El algoritmo no pertenece al módulo activo.")
        if request.problem == "clustering" and request.scaler == "normalizer":
            raise ValueError(
                "Selecciona standard, robust, minmax o none para este formulario de clustering."
            )
        state = context.application_state
        if state is None:
            raise ValueError("La configuración necesita un estado de aplicación conectado.")
        legacy = {
            "classification": "clasificacion",
            "regression": "regresion",
            "clustering": "clustering",
        }[request.problem]
        if request.problem == "clustering":
            state["analysis_cluster_columns"] = list(request.predictors)
            state["analysis_cluster_scaler"] = request.scaler
            state["analysis_cluster_seed"] = request.random_state
            if request.algorithm:
                state["atlas_cluster_algorithm"] = request.algorithm
        else:
            state[f"analysis_target_{legacy}"] = request.target
            state[f"analysis_predictors_{legacy}"] = list(request.predictors)
            state[f"analysis_scaler_{legacy}"] = request.scaler
            state[f"analysis_size_{legacy}"] = request.test_size
            state[f"analysis_cv_{legacy}"] = request.cv
            state[f"analysis_seed_{legacy}"] = request.random_state
            if request.algorithm:
                state[f"atlas_algorithms_{legacy}"] = [request.algorithm]
        state["analysis_selection"] = request.model_dump()
        state["selected_variables"] = list(request.predictors)
        context.selections["analysis_selection"] = request.model_dump()
        return request.model_dump()
