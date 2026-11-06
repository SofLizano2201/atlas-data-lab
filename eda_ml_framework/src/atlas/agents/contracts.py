"""Contracts for the modular ATLAS assistant system."""

from dataclasses import dataclass

from pydantic import Field

from atlas.core.schemas import (
    AgentRequest,
    Contract,
)


class PlannedCall(Contract):
    """Define the PlannedCall contract."""

    tool: str
    parameters_json: str
    reason: str


class Plan(Contract):
    """Define the Plan contract."""

    calls: list[PlannedCall] = Field(max_length=3)
    explanation: str


class Explanation(Contract):
    """Define the Explanation contract."""

    explanation: str


@dataclass
class PendingPlan:
    request: AgentRequest
    dataset_key: str
    created: float
