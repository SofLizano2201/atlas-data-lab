"""Contexto privado por sesión y capacidades locales con protección contra replay."""

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd

from contracts import AgentDecision, DatasetVersion
from framework import AnalisisDatosExploratorio
from framework.results import Experiment


def table_records(df: pd.DataFrame, limit: int = 200) -> list[dict[str, Any]]:
    """Tablas acotadas, JSON válido (NaN → null), índices visibles."""
    data = df.head(limit).copy()
    data.columns = [str(c) for c in data.columns]
    if data.index.name or not isinstance(data.index, pd.RangeIndex):
        data.insert(0, "indice_fila", data.index.astype(str))
    return json.loads(data.to_json(orient="records", date_format="iso"))


@dataclass
class AgentContext:
    """Los datasets y artefactos nunca se comparten entre sesiones ni se envían al LLM."""

    eda: AnalisisDatosExploratorio
    session_id: str = field(default_factory=lambda: uuid4().hex)
    experiments: list[Experiment] = field(default_factory=list)
    engines: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    proposals: dict[str, tuple[str, pd.DataFrame, str]] = field(default_factory=dict)
    pending: dict[str, Any] = field(default_factory=dict)
    decisions: list[AgentDecision] = field(default_factory=list)
    conversation: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def dataset(self) -> DatasetVersion:
        """Descripción de la versión actualmente activa."""
        return self.eda.dataset_version

    @property
    def key(self) -> str:
        """Versión concreta para invalidar planes tras limpieza o restauración."""
        d = self.dataset
        return f"{d.dataset_id}:{d.version}:{d.content_hash}"

    def artifact(self, value: Any) -> str:
        """Guardar un objeto sin introducirlo en contratos JSON ni contexto externo."""
        if len(self.artifacts) >= 100:
            raise ValueError(
                "Límite de 100 artefactos por sesión; exporte antes de iniciar otra sesión."
            )
        key = uuid4().hex
        self.artifacts[key] = value
        return key

    def propose(self, frame: pd.DataFrame, operation: str) -> str:
        """Conservar propuestas locales sin duplicarlas en el almacén de figuras."""
        if len(self.proposals) >= 5:
            raise ValueError(
                "Máximo de cinco propuestas pendientes; aplique una o reinicie el contexto."
            )
        memory = sum(p[1].memory_usage(deep=True).sum() for p in self.proposals.values())
        if memory + frame.memory_usage(deep=True).sum() > 128 * 1024**2:
            raise ValueError("Las propuestas excederían 128 MB; reduzca el dataset.")
        key = uuid4().hex
        self.proposals[key] = (self.key, frame.copy(deep=True), operation)
        return key

    def metadata(self) -> dict[str, Any]:
        """Solo esquema y conteos: nunca filas, categorías, índices o nombres de archivo."""
        df = self.eda.df
        return {
            "rows": len(df),
            "columns": [{"name": str(c)[:100], "dtype": str(df[c].dtype)} for c in df.columns[:80]],
            "column_count": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "experiments": [
                {"id": e.id, "problem": e.problem, "algorithm": e.name}
                for e in self.experiments[-20:]
            ],
        }
