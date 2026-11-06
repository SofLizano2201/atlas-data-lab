# Agents and provider configuration

The persistent intelligence panel now receives deduplicated workflow events and
reads actual results automatically. See [event behavior and verification](event_intelligence.md).
Automatic assistance uses NoProvider and cannot grant training or mutation permissions.
The explicit specialist dialog remains available for confirmed actions and optional conversation.

ATLAS 2.0 defaults to finite offline decision policies, not a required language
model. Open a specialist and select a guided goal; no API key or JSON editing is
necessary. Existing advanced tool calls and optional consented conversation remain
available. See [2.0 policies and optional Ollama setup](atlas_2_0.md).

Agents are organized by responsibility in `src/atlas/agents/`, with one file per
specialist. See [module map and extension guide](agents_structure.md). The
`agent.py` facade and root `agents/` package preserve legacy imports without
duplicating implementations.

## Local actions and conversation

Open the contextual assistant button on the relevant page. Quick actions call real
framework methods through an allowlisted gateway, without needing a language model.
They are labelled deterministic guidance. Conversation requires a configured model
and explicit consent to send the question and reduced metadata. No offline response
is presented as an LLM answer.

Training, searches, expensive diagnostics and configuration proposals require
confirmation. A proposal can update the actual form controls only after confirmation;
applying configuration never silently trains a model. Pending tickets are bound to
session and dataset version, expire, and cannot be reused. Closing the panel cancels
pending actions for that specialist. Changing the dataset clears dependent context.

Structured responses expose actual tool calls, parameters, status, results, metrics,
figure/table references, evidence, recommendations, warnings and proposed/executed
actions. Evidence refers to computed results; interpretations do not generate numbers.
The model cannot authorize itself or execute Python. Data content is untrusted input.

## Configuration

`.env.example` is a template; ATLAS does not automatically load it. Set environment
variables before starting Streamlit, or use an appropriate secret manager. Never
commit a key. In PowerShell:

```powershell
$env:ATLAS_LLM_PROVIDER = "responses_tools"
$env:ATLAS_LLM_MODEL = "your-enabled-model"
$env:ATLAS_LLM_BASE_URL = "https://api.openai.com/v1"
# Set ATLAS_LLM_API_KEY securely in your environment, then start Streamlit.
python -m streamlit run app.py
```

`responses_tools` uses native Responses API function calling, a closed function
allowlist and strict envelopes. Tool parameters are parsed as JSON and independently
validated against each registered input schema. Parallel tool calls are disabled.
The `responses` structured-output and `chat_completions` compatible transports remain
available. HTTPS is required except for explicit local test endpoints; redirects
are rejected, responses bounded and timeouts enforced. Invalid configuration leaves
the assistant unavailable without blocking manual analysis.

Contracts follow the official [function-calling guide](https://developers.openai.com/api/docs/guides/function-calling)
and [structured-output guide](https://developers.openai.com/api/docs/guides/structured-outputs).
Automated transport verification uses a local HTTP test server. An external paid
provider call has not been verified in this delivery; valid credentials and a model
supporting the selected transport are required.

## Polymorphic use and demonstration

```python
from sklearn.datasets import load_iris
from atlas.models import ExploratoryDataAnalysis
from atlas.core.session import AgentContext
from atlas.core.schemas import ToolCall
from atlas.agents.agent import AgentFactory, NoProvider

context = AgentContext(ExploratoryDataAnalysis(load_iris(as_frame=True).frame))
agent = AgentFactory.get_agent("eda", context, NoProvider())
response = agent.run(agent.request("Resumen", [ToolCall(tool="eda.profile")]))
assert response.status == "success"
```

Run `python tools/demo_agents.py --confirm-training` to exercise all five agents on
public Iris/Diabetes datasets, including real training, metrics and comparison.
The output `artifacts/mvc-demo.json` explicitly identifies local framework execution,
not external inference. Omit the flag to retain confirmation gates.

To extend providers or agents, follow [MVC extension rules](mvc.md). Provider,
capability gateway, logger and AgentConfig limits support dependency injection.
