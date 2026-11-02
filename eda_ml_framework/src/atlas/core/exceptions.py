"""Shared typed infrastructure and dependency contracts."""


class FrameworkError(ValueError):
    """Define the FrameworkError contract."""


class OptionalDependencyError(FrameworkError):
    """Define the OptionalDependencyError contract."""
