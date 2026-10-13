"""Errores públicos y comprensibles del framework."""


class FrameworkError(ValueError):
    """Configuración o datos incompatibles con una operación."""


class OptionalDependencyError(FrameworkError):
    """Una integración requiere una dependencia adicional."""
