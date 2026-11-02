"""Read legacy serialized configuration keys without storing duplicate scientific state."""

from typing import ClassVar


class CompatibleConfiguration(dict):
    """Store English keys and accept historical academic lookups at the compatibility boundary."""

    aliases: ClassVar[dict[str, str]] = {
        "objetivo": "target",
        "predictores": "predictors",
        "columnas": "columns",
        "parametros": "parameters",
        "busqueda": "search",
        "tipo_busqueda": "search_type",
        "mejores_parametros": "best_parameters",
    }

    def __init__(self, values=(), **kwargs):
        super().__init__()
        self.update(values, **kwargs)

    def __getitem__(self, key):
        return super().__getitem__(self.aliases.get(key, key))

    def __setitem__(self, key, value):
        super().__setitem__(self.aliases.get(key, key), value)

    def __contains__(self, key):
        return super().__contains__(self.aliases.get(key, key))

    def get(self, key, default=None):
        return super().get(self.aliases.get(key, key), default)

    def update(self, values=(), **kwargs):
        for key, value in dict(values, **kwargs).items():
            self[key] = value


def normalize_transform_parameters(parameters):
    """Accept old academic keywords while keeping the scientific implementation in English."""
    aliases = {
        "umbral": "threshold",
        "valor": "value",
        "tipo": "dtype",
        "limite": "limit",
        "metodo": "method",
    }
    return {aliases.get(key, key): value for key, value in parameters.items()}


def upgrade_session_types(state):
    """Rebind known pre-MVC in-memory objects without refitting or changing dataset values."""
    from atlas.core.schemas import AgentDecision, AgentResponse
    from atlas.core.session import AgentContext
    from atlas.models import (
        Classification,
        Clustering,
        ExploratoryDataAnalysis,
        Regression,
        SupervisedLearning,
        UnsupervisedLearning,
    )
    from atlas.models.experiment import Experiment

    known = {
        "AnalisisDatosExploratorio": ExploratoryDataAnalysis,
        "ExploratoryDataAnalysis": ExploratoryDataAnalysis,
        "NoSupervisado": UnsupervisedLearning,
        "UnsupervisedLearning": UnsupervisedLearning,
        "Supervisado": SupervisedLearning,
        "SupervisedLearning": SupervisedLearning,
        "Clustering": Clustering,
        "Clasificacion": Classification,
        "Classification": Classification,
        "Regresion": Regression,
        "Regression": Regression,
        "Experiment": Experiment,
    }
    changed = False

    def upgrade(value):
        nonlocal changed
        old_type = type(value)
        expected = known.get(old_type.__name__)
        if (
            expected is not None
            and old_type is not expected
            and old_type.__module__.startswith(("framework.", "atlas.models."))
        ):
            if "experimentos" in value.__dict__:
                value.__dict__["experiments"] = value.__dict__.pop("experimentos")
            value.__class__ = expected
            changed = True
        if isinstance(value, Experiment) and (
            not isinstance(value.config, CompatibleConfiguration)
        ):
            value.config = CompatibleConfiguration(value.config)

    if "eda" in state:
        upgrade(state["eda"])
    for value in state.get("experiments", []):
        upgrade(value)
    for value in state.get("models", {}).values():
        upgrade(value.__dict__.get("_model", value))
    context = state.get("agent_context")
    if (
        context is not None
        and type(context) is not AgentContext
        and type(context).__module__ in {"agents.context", "atlas.core.session"}
    ):
        context.__class__ = AgentContext
        context.selections = {}
        context.application_state = state
        changed = True
    if context is not None and changed:
        context.pending.clear()
        context.decisions[:] = [
            AgentDecision.model_validate_json(item.model_dump_json()) for item in context.decisions
        ]
        for name, history in context.conversation.items():
            context.conversation[name] = [
                AgentResponse.model_validate_json(item.model_dump_json()) for item in history
            ]
        for key in list(state):
            if key.startswith("agent_last_") and state[key] is not None:
                state[key] = AgentResponse.model_validate_json(state[key].model_dump_json())
