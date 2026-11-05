"""Shared typed infrastructure and dependency contracts."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd

from atlas.core.schemas import AgentDecision, DatasetVersion
from atlas.models import ExploratoryDataAnalysis
from atlas.models.experiment import Experiment


def table_records(df: pd.DataFrame, limit: int = 200) -> list[dict[str, Any]]:
    """Provide table records using validated inputs."""
    data = df.head(limit).copy()
    data.columns = [str(c) for c in data.columns]
    if data.index.name or not isinstance(data.index, pd.RangeIndex):
        data.insert(0, "indice_fila", data.index.astype(str))
    return json.loads(data.to_json(orient="records", date_format="iso"))


@dataclass
class AgentContext:
    """Define the AgentContext contract."""

    eda: ExploratoryDataAnalysis
    session_id: str = field(default_factory=lambda: uuid4().hex)
    experiments: list[Experiment] = field(default_factory=list)
    engines: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    proposals: dict[str, tuple[str, pd.DataFrame, str]] = field(default_factory=dict)
    pending: dict[str, Any] = field(default_factory=dict)
    decisions: list[AgentDecision] = field(default_factory=list)
    conversation: dict[str, list[Any]] = field(default_factory=dict)
    event_evidence: dict[str, Any] = field(default_factory=dict)
    selections: dict[str, Any] = field(default_factory=dict)
    application_state: Any = None
    _metadata_key: str = field(default="", repr=False)
    _metadata_base: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def dataset(self) -> DatasetVersion:
        """Provide dataset using validated inputs."""
        return self.eda.dataset_version

    @property
    def key(self) -> str:
        """Provide key using validated inputs."""
        d = self.dataset
        return f"{d.dataset_id}:{d.version}:{d.content_hash}"

    def artifact(self, value: Any) -> str:
        """Provide artifact using validated inputs."""
        if len(self.artifacts) >= 100:
            raise ValueError(
                "Límite de 100 artefactos por sesión; exporte antes de iniciar otra sesión."
            )
        key = uuid4().hex
        self.artifacts[key] = value
        return key

    def propose(self, frame: pd.DataFrame, operation: str) -> str:
        """Provide propose using validated inputs."""
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
        """Provide metadata using validated inputs."""
        if getattr(self, "_metadata_key", "") != self.key:
            df = self.eda.df
            profile = self.eda.profile()
            self._metadata_base = {
                "rows": profile["rows"],
                "columns": [
                    {"name": str(c)[:100], "dtype": str(df[c].dtype)} for c in df.columns[:80]
                ],
                "column_count": profile["columns"],
                "missing_cells": profile["missing_cells"],
                "additional_duplicates": profile["additional_duplicates"],
                "constant_columns": profile["constant_columns"],
                "constant_names": self.eda.constant_columns()[:30],
                "missing_by_column": self.eda.missing_values().to_dict(orient="index"),
                "training_columns": self.eda.source_columns[:80],
                "feature_roles": self.eda.feature_roles(),
            }
            self._metadata_key = self.key
        return {
            **deepcopy(self._metadata_base),
            "active_selection": deepcopy(self.selections),
            "experiments": [
                {
                    "id": e.id,
                    "problem": e.problem,
                    "algorithm": e.name,
                    "dataset_id": e.dataset_id,
                    "cohort": e.cohort,
                    "seconds": e.seconds,
                    "metrics": e.metrics_for_context(),
                    "validation": {
                        key: e.config[key]
                        for key in ["target", "predictors", "test_size", "cv", "scaler"]
                        if key in e.config
                    },
                }
                for e in self.experiments[-20:]
            ],
        }
