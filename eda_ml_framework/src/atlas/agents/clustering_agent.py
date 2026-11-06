"""Clustering agent for the modular ATLAS assistant system."""

from typing import Any

from atlas.core.schemas import (
    AnalysisResult,
    ToolResult,
)

from .base import BaseAgent


class ClusteringAgent(BaseAgent):
    name = "ClusteringAgent"
    objective = "Especialista en agrupamiento. Excluye identificadores, justifica escalamiento y distancias. Distingue ruido de grupos y no interpretes t-SNE como geometría original."
    allowed_tools = frozenset(
        {
            "analysis.configure",
            "eda.profile",
            "cluster.reduce",
            "cluster.distances",
            "cluster.train",
            "cluster.elbow",
            "comparison.compare",
            "model.metrics",
        }
    )
    problems = frozenset({"clustering"})

    def get_instructions(self) -> str:
        return "Check selected variables, scaling, distance compatibility and noise. Use existing clustering diagnostics. Do not infer stability from a single silhouette value or confuse projections with original distances."

    def get_allowed_tools(self) -> frozenset[str]:
        return self.allowed_tools

    def interpret_result(self, result: ToolResult) -> dict[str, Any]:
        interpretation = self._interpret(
            result,
            "agrupamiento",
            "Revisa qué caracteriza a cada grupo y si tiene sentido para tu objetivo. Prueba otra configuración con las mismas variables antes de decidir.",
        )
        if result.result and result.result.metrics.get("ruido", 0) > 0.3:
            interpretation["recommendations"][0]["text"] = (
                "Muchos registros quedaron sin grupo. Antes de cambiar el algoritmo, revisa las variables elegidas y su escalamiento; después compara otra configuración."
            )
        if isinstance(result.result, AnalysisResult) and result.result.tables.get("seleccion"):
            selected = result.result.tables["seleccion"][0]
            experiment = next(
                (
                    item
                    for item in self.context.experiments
                    if item.id == selected.get("experimento")
                ),
                None,
            )
            if experiment:
                count = experiment.config.get("params", {}).get("n_clusters")
                grouping = f"{count} grupos" if count is not None else experiment.name
                interpretation["recommendations"][0]["text"] = (
                    f"Entre las alternativas comparables probadas, {grouping} destaca según {selected['metrica']} ({selected['valor']:.3g}). Revisa qué caracteriza a cada grupo antes de elegirlo. Esta comparación no demuestra estabilidad ni que sea la mejor opción para todos los usos."
                )
        elif result.call.tool == "cluster.train" and result.status == "success":
            interpretation["recommendations"][0]["text"] = (
                "El agrupamiento quedó guardado. Abre «Ver otras ideas» y compara las alternativas para saber qué cantidad de grupos presenta una separación más clara con estas variables."
            )
        return interpretation
