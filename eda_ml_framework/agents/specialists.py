"""Cinco identidades especializadas sobre un executor común y el framework existente."""

from .base import BaseAgent


class EDAAgent(BaseAgent):
    name = "EDAAgent"
    objective = (
        "Especialista en exploración y preparación. Diagnostica calidad antes de transformar. "
        "Justifica pruebas y supuestos. Propón cambios reversibles y nunca apliques sin confirmación."
    )
    allowed_tools = frozenset(
        {
            "eda.profile",
            "eda.statistics",
            "eda.correlation",
            "eda.plot",
            "eda.test",
            "eda.preview",
            "eda.apply",
        }
    )


class ClusteringAgent(BaseAgent):
    name = "ClusteringAgent"
    objective = (
        "Especialista en agrupamiento. Excluye identificadores, justifica escalamiento y "
        "distancias. Distingue ruido de grupos y no interpretes t-SNE como geometría original."
    )
    allowed_tools = frozenset(
        {"eda.profile", "cluster.reduce", "cluster.distances", "cluster.train", "model.metrics"}
    )
    problems = frozenset({"clustering"})


class ClassificationAgent(BaseAgent):
    name = "ClassificationAgent"
    objective = (
        "Especialista en clasificación binaria y multiclase. Revisa desbalance y separación "
        "de objetivo. Usa entrenamiento original y pipeline, prioriza métricas según contexto."
    )
    allowed_tools = frozenset(
        {"eda.profile", "classification.train", "model.metrics", "model.plot", "model.importance"}
    )
    problems = frozenset({"clasificacion"})


class RegressionAgent(BaseAgent):
    name = "RegressionAgent"
    objective = (
        "Especialista en regresión. Distingue ajuste y generalización, analiza residuos y "
        "errores, evita leakage y advierte denominadores nulos y extrapolación."
    )
    allowed_tools = frozenset(
        {"eda.profile", "regression.train", "model.metrics", "model.plot", "model.importance"}
    )
    problems = frozenset({"regresion"})


class ModelComparisonAgent(BaseAgent):
    name = "ModelComparisonAgent"
    objective = (
        "Especialista en comparación rigurosa. Compara solo cohortes y particiones compatibles. "
        "Exige métrica y sentido explícitos; considera tiempos y no declares un ganador universal."
    )
    allowed_tools = frozenset({"comparison.compare", "model.metrics"})
    problems = frozenset({"clasificacion", "regresion", "clustering"})


AGENTS = {
    cls.name: cls
    for cls in [
        EDAAgent,
        ClusteringAgent,
        ClassificationAgent,
        RegressionAgent,
        ModelComparisonAgent,
    ]
}
