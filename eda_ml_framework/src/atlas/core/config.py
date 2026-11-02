"""Local orchestration limits; credentials remain in environment variables."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    confirmation_ttl: float = 300
    max_pending_plans: int = 10
    conversation_limit: int = 30
    decision_limit: int = 500
