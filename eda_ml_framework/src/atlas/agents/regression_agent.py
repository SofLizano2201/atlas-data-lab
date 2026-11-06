"""Regression agent for the modular ATLAS assistant system."""

from typing import Any

from atlas.core.schemas import (
    ToolResult,
)

from .base import BaseAgent


class RegressionAgent(BaseAgent):
    name = "RegressionAgent"
    objective = "Especialista en regresión. Distingue ajuste y generalización, analiza residuos y errores, evita leakage y advierte denominadores nulos y extrapolación."
    allowed_tools = frozenset(
        {
            "analysis.configure",
            "eda.profile",
            "regression.train",
            "model.metrics",
            "model.plot",
            "model.importance",
        }
    )
    problems = frozenset({"regresion"})

    def get_instructions(self) -> str:
        return "Check numeric target, regularization, error units and residual evidence. MAPE is unsuitable near zero. Never assert residual assumptions or overfitting without recorded evidence. Use existing regression methods."

    def get_allowed_tools(self) -> frozenset[str]:
        return self.allowed_tools

    def interpret_result(self, result: ToolResult) -> dict[str, Any]:
        interpretation = self._interpret(
            result,
            "regresión",
            "Mira cuánto se alejan las predicciones de los valores reales. Si los errores son demasiado grandes para tu objetivo, compara otro modelo con la misma separación de datos.",
        )
        metrics = result.result.metrics if result.result else {}
        if "MAPE" in metrics and metrics["MAPE"] is None:
            interpretation["recommendations"][0]["text"] = (
                "El error porcentual no se pudo calcular. Revisa si el objetivo contiene ceros y utiliza el error absoluto para saber cuánto se equivoca en sus unidades originales."
            )
        elif metrics.get("R2") is not None and metrics["R2"] < 0:
            interpretation["recommendations"][0]["text"] = (
                "El modelo no está dando un buen ajuste en la prueba. Revisa las variables elegidas y la separación de datos antes de probar otra alternativa."
            )
        return interpretation
