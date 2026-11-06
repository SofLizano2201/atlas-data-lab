"""Default dependency assembly; no scientific computation or AI policies."""


def build_gateway(context):
    from atlas.controllers.capabilities import CapabilityGateway

    return CapabilityGateway(context)


def build_agent(name, context):
    """Assemble the configured provider and gateway without initiating an LLM call."""
    from atlas.agents.factory import AgentFactory
    from atlas.agents.providers import NoProvider, ProviderError, configured_provider

    error = None
    try:
        provider = configured_provider()
    except ProviderError as exc:
        provider, error = (NoProvider(), str(exc))
    return (AgentFactory.get_agent(name, context, provider, gateway=build_gateway(context)), error)


def get_default_arguments(context, tool, problems):
    """Delegate suggested tool arguments to the agent defaults module."""
    from atlas.agents.defaults import default_arguments

    return default_arguments(tool, context, problems)


def build_offline_agent(name, context):
    """Construct event-driven assistance without probing any LLM configuration."""
    from atlas.agents.factory import AgentFactory
    from atlas.agents.providers import NoProvider

    return AgentFactory.get_agent(name, context, NoProvider(), gateway=build_gateway(context)), None
