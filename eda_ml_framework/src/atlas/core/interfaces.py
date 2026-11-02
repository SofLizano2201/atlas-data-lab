"""Dependency contracts shared without importing views or provider implementations."""

from collections.abc import Mapping
from typing import Any, Protocol

from .schemas import ToolCall, ToolResult


class CapabilityAccess(Protocol):
    """A closed registry and validated execution boundary."""

    tools: Mapping[str, Any]

    def execute(
        self, call: ToolCall, problems: set[str], *, confirmed: bool = False
    ) -> ToolResult: ...
