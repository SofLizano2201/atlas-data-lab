"""Comparison agent for the modular ATLAS assistant system."""

from typing import Any

from atlas.core.schemas import (
    AnalysisResult,
    ToolResult,
)

from .base import BaseAgent


class ModelComparisonAgent(BaseAgent):
    name = "ModelComparisonAgent"
    objective = "Especialista en comparación rigurosa. Compara solo cohortes y particiones compatibles. Exige métrica y sentido explícitos; considera tiempos y no declares un ganador universal."
    allowed_tools = frozenset({"comparison.compare", "model.metrics"})
    problems = frozenset({"clasificacion", "regresion", "clustering"})

    def get_instructions(self) -> str:
        return "Validate dataset versions, cohorts, validation procedures and task compatibility. Rank only on an explicit metric and direction using the existing comparator. Discuss runtime and uncertainty; never name a universal winner."

    def get_allowed_tools(self) -> frozenset[str]:
        return self.allowed_tools

    def interpret_result(self, result: ToolResult) -> dict[str, Any]:
        interpretation = self._interpret(
            result,
            "comparación",
            "El resultado favorece un modelo solo según la medida elegida. Revisa también su tiempo y los errores importantes para tu caso antes de elegirlo.",
        )
        if isinstance(result.result, AnalysisResult) and result.result.tables.get("seleccion"):
            selected = result.result.tables["seleccion"][0]
            model = {
                "logistic": "Regresión logística",
                "tree": "Árbol de decisión",
                "forest": "Random Forest",
                "ridge": "Ridge",
                "linear_multiple": "Regresión lineal",
            }.get(selected["modelo"], selected["modelo"])
            metric = {
                "Accuracy": "porcentaje global de aciertos",
                "F1_macro": "equilibrio entre aciertos y detección por categoría",
                "RMSE": "error de predicción",
                "MAE": "error absoluto",
                "segundos": "tiempo de entrenamiento",
            }.get(selected["metrica"], selected["metrica"])
            interpretation["recommendations"][0]["text"] = (
                f"Según {metric}, la alternativa favorecida es {model} (valor {selected['valor']:.4g}). Revisa también sus errores y su tiempo antes de elegirla; no es la mejor para todos los casos."
            )
        return interpretation
