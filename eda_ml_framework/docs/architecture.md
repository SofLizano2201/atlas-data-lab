# Architecture

Current event-driven integration: [automatic EDA and persistent intelligence](event_intelligence.md).
`core/events.py` carries typed session-owned facts. `WorkflowController` dispatches
them to `BaseAgent.on_event`. `AutomaticEDAController` orchestrates existing Model
calculations. Automatic feedback uses `NoProvider`; optional language-model setup
remains restricted to explicitly opened assistance. The legacy API remains compatible.

ATLAS 1.2 uses MVC within the existing application. The canonical source is
`src/atlas`, with thin legacy import adapters retained at the project root.
See [MVC responsibilities, contracts and extension](mvc.md) for the full design.

## Scientific hierarchy

```text
ExploratoryDataAnalysis
├── UnsupervisedLearning
│   └── Clustering
└── SupervisedLearning
    ├── Classification
    └── Regression
```

Models accept pandas DataFrames and can be used without Streamlit or a provider.
Inherited EDA, validated preprocessing, train-only pipelines and experiment results
remain shared. Controller subclasses mirror the model hierarchy with explicit
forwarding methods. Views consume controllers and structured results.

`core/schemas.py` defines AgentRequest, AgentResponse, ToolCall, ToolResult,
AnalysisResult, ExperimentResult, ModelResult, AgentDecision, DatasetVersion,
EvidenceRecord and Recommendation. Pydantic validates boundaries. SessionContext
and dependency contracts are separate from the frontend.

Agent logic resides in the modular `agents/` package: abstract BaseAgent, five
specialist modules, factory, contracts, providers and shared orchestration helpers.
See [the module map](agents_structure.md). The capability
gateway has a closed registry of validated operations referencing existing model
methods. It does not accept arbitrary code or arbitrary method names.

`services/` manages dataset activation, the common manual/agent experiment store and
report history. Models retain dataset hashes, versions, split indices, parameters,
library versions and random seeds. Dataset originals are copied defensively;
mutations and restoration require confirmation. Test data never fit preprocessing.

The optional assistant consumes only consented, bounded metadata and computed
metrics. No raw rows or tables are sent automatically. User questions and column
names can themselves be sensitive; consent remains necessary. Unconfigured AI is
reported as unavailable, while explicit local tools continue working.

Tests enforce model isolation, one canonical definition per agent, controller/provider
separation, inheritance, permissions and frontend integration. See
[verification](verification_mvc_1_2.md) and [machine-readable inventory](capabilities_mvc.json).
