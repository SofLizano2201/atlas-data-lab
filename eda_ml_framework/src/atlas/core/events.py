"""Session-owned event contracts and deduplication; no analytical policies."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any


class EventType(StrEnum):
    DATASET_UPLOADED = "DATASET_UPLOADED"
    DATASET_CONFIRMED = "DATASET_CONFIRMED"
    DATASET_CHANGED = "DATASET_CHANGED"
    EDA_STARTED = "EDA_STARTED"
    EDA_COMPLETED = "EDA_COMPLETED"
    EDA_FAILED = "EDA_FAILED"
    DATA_CLEANED = "DATA_CLEANED"
    FEATURES_CHANGED = "FEATURES_CHANGED"
    TARGET_SELECTED = "TARGET_SELECTED"
    MODEL_SELECTED = "MODEL_SELECTED"
    TRAINING_STARTED = "TRAINING_STARTED"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    MODEL_EVALUATED = "MODEL_EVALUATED"
    CLUSTERING_COMPLETED = "CLUSTERING_COMPLETED"
    EXPERIMENT_SAVED = "EXPERIMENT_SAVED"
    COMPARISON_UPDATED = "COMPARISON_UPDATED"
    REPORT_GENERATED = "REPORT_GENERATED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    WORKSPACE_ENTERED = "WORKSPACE_ENTERED"


@dataclass
class ApplicationEvent:
    id: str
    type: EventType
    dataset_id: str
    dataset_version: int
    content_hash: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "pending"


def emit(state, event_type: EventType, payload=None) -> ApplicationEvent | None:
    """Deduplicate identical facts within one dataset version and session."""
    if "eda" not in state:
        return None
    version = state["eda"].dataset_version
    payload = payload or {}
    signature = json.dumps(
        [event_type, version.dataset_id, version.version, version.content_hash, payload],
        sort_keys=True,
        default=str,
    )
    identifier = sha256(signature.encode()).hexdigest()
    events = state.setdefault("workflow_events", {})
    if identifier not in events:
        events[identifier] = ApplicationEvent(
            identifier,
            event_type,
            version.dataset_id,
            version.version,
            version.content_hash,
            payload,
        )
    # Preserve a bounded activity history; analytical caches have their own lifecycle.
    while len(events) > 300:
        events.pop(next(iter(events)))
    return events[identifier]
