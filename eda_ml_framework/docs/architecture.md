# Architecture

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

All agent logic resides in `agents/agent.py`: abstract BaseAgent, five specialists,
factory, plans, instructions, provider adapters and shared execution. The capability
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

Tests enforce model isolation, a single agent definition file, controller/provider
separation, inheritance, permissions and frontend integration. See
[verification](verification_mvc_1_2.md) and [machine-readable inventory](capabilities_mvc.json).
