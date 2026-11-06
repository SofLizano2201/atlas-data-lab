"""Classification agent for the modular ATLAS assistant system."""

from typing import Any

from atlas.core.schemas import (
    AnalysisResult,
    ToolResult,
)

from .base import BaseAgent


class ClassificationAgent(BaseAgent):
    name = "ClassificationAgent"
    objective = "Especialista en clasificación binaria y multiclase. Revisa desbalance y separación de objetivo. Usa entrenamiento original y pipeline, prioriza métricas según contexto."
    allowed_tools = frozenset(
        {
            "analysis.configure",
            "eda.profile",
            "classification.class_distribution",
            "classification.train",
            "model.metrics",
            "model.plot",
            "model.importance",
        }
    )
    problems = frozenset({"clasificacion"})

    def get_instructions(self) -> str:
        return "Check target, class distribution and cross-validation feasibility. Select metrics for the user's costs and imbalance. Never use holdout results for hyperparameter selection. Do not assume probability support."

    def get_allowed_tools(self) -> frozenset[str]:
        return self.allowed_tools

    def interpret_result(self, result: ToolResult) -> dict[str, Any]:
        interpretation = self._interpret(
            result,
            "clasificación",
            "Revisa qué categorías confunde el modelo, especialmente las que tienen pocos ejemplos. Decide qué error es más importante evitar antes de comparar modelos.",
        )
        if isinstance(result.result, AnalysisResult):
            rows = result.result.tables.get("frecuencias", [])
            if rows:
                interpretation["recommendations"][0]["text"] = (
                    "Este es el reparto de tus categorías. Ahora puedes preparar un primer modelo y comprobar después cómo responde para cada una, especialmente si tiene pocos ejemplos."
                )
            if len(rows) > 1 and any(row.get("relativa", 0) > 0.8 for row in rows):
                interpretation["recommendations"][0]["text"] = (
                    "Hay una clase dominante: una categoría tiene muchos más ejemplos. Revisa los aciertos de cada categoría; el porcentaje total de aciertos puede ocultar fallos en las menos frecuentes."
                )
        return interpretation
