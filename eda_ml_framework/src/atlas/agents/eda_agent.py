"""Eda agent for the modular ATLAS assistant system."""

from typing import Any

from atlas.core.schemas import (
    ToolResult,
)

from .base import BaseAgent


class EDAAgent(BaseAgent):
    name = "EDAAgent"
    objective = "Especialista en exploración y preparación. Diagnostica calidad antes de transformar. Justifica pruebas y supuestos. Propón cambios reversibles y nunca apliques sin confirmación."
    allowed_tools = frozenset(
        {
            "eda.profile",
            "eda.frequencies",
            "eda.outliers",
            "eda.statistics",
            "eda.correlation",
            "eda.plot",
            "eda.test",
            "eda.preview",
            "eda.apply",
        }
    )

    def get_instructions(self) -> str:
        return "Prioritize data quality using actual missingness, duplicates and statistics. Choose tests only when compatible. Recommend the next registered analysis, never calculate statistics yourself."

    def get_allowed_tools(self) -> frozenset[str]:
        return self.allowed_tools

    def interpret_result(self, result: ToolResult) -> dict[str, Any]:
        interpretation = self._interpret(
            result,
            "calidad y exploración",
            "Elige una variable importante y observa cómo se distribuyen sus valores. Después revisa si los valores poco habituales son errores o casos válidos.",
        )
        metrics = result.result.metrics if result.result else {}
        if self.context.selections.get("workspace_page") == "preparation":
            options = self.preparation_options(self.context.metadata())
            if result.call.tool == "eda.preview":
                option = next((item for item in options if item[1] == result.call.parameters), None)
                interpretation["recommendations"][0]["text"] = (
                    (
                        option[4]
                        if option
                        else "Revisa la vista previa y comprueba que el cambio tenga sentido para tus datos."
                    )
                    + " Tus datos siguen intactos; aplica la propuesta solo después de revisar el antes y el después."
                )
                return interpretation
            if options and result.call.tool == "eda.profile":
                interpretation["recommendations"][0]["text"] = options[0][2] + ". " + options[0][4]
                return interpretation
        if metrics.get("celdas_nulas", 0) > 0:
            interpretation["recommendations"][0]["text"] = (
                f"Hay {metrics['celdas_nulas']:,} valores faltantes. Ve a Preparación, revisa las columnas afectadas y previsualiza cómo completar esos valores antes de confirmar."
            )
        elif metrics.get("duplicadas_adicionales", 0) > 0:
            interpretation["recommendations"][0]["text"] = (
                f"Hay {metrics['duplicadas_adicionales']:,} filas repetidas adicionales. Revísalas en Preparación y elimínalas solo si representan el mismo registro por error."
            )
        elif metrics.get("categorical_columns", 0) > metrics.get("numeric_columns", 0):
            interpretation["recommendations"][0]["text"] = (
                "Predominan columnas no numéricas: comienza por frecuencias y asociaciones compatibles, no por correlaciones numéricas indiscriminadas."
            )
        return interpretation
