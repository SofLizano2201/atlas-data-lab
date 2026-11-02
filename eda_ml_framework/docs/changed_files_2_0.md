# Changes introduced for ATLAS 2.0

The previous MVC migration was already present locally. This list identifies 2.0
work rather than attributing all uncommitted migration files to this redesign.

| Files | Change |
|---|---|
| `app.py` | Workspace stylesheet and experiment-history route |
| `src/atlas/agents/agent.py` | Explicit local policies, prerequisites, contextual prioritization, guided commands, optional Ollama transport selection and availability probe |
| `src/atlas/controllers/agent_controller.py` | Polymorphic guided-command/probe forwarding; history context |
| `src/atlas/controllers/workspace_controller.py` | Real context and operation-based progress |
| `src/atlas/controllers/eda_controller.py` | Preview impact forwarding |
| `src/atlas/controllers/clustering_controller.py` | Size/profile/inspection forwarding |
| `src/atlas/controllers/comparison_controller.py` | Cohort groups and comparison-chart forwarding |
| `src/atlas/models/eda.py` | Actual missing percentage and non-mutating transformation impact |
| `src/atlas/models/visualization.py` | Atlas series, contrast, histogram patterns/binning, log/facet validation, stable interaction state |
| `src/atlas/models/unsupervised.py` | Extended group colors with marker shapes |
| `src/atlas/models/clustering.py` | Existing-profile heatmaps, group sizes, bounded inspection; preprocessing honored by elbow/dendrogram |
| `src/atlas/models/classification.py` | Store real probabilities and visualize them when available |
| `src/atlas/models/supervised.py` | Retain actual fold scores, without new training |
| `src/atlas/models/comparison.py` | Compatible-cohort groups; ranking/runtime/cost/fold and binary ROC/PR overlays |
| `src/atlas/services/dataset_service.py` | Clear version-dependent view/prediction state while preserving report snapshots |
| `src/atlas/views/components/sidebar.py` | Ten destinations grouped by workflow |
| `src/atlas/views/components/layout.py` | Compact context/progress and next-navigation controls |
| `src/atlas/views/components/agent_panel.py` | Guided local goals, compact context, collapsed technical tools and evidence |
| `src/atlas/views/components/ui.py` | Real zoom, selection inspection, enlargement and existing downloads |
| `src/atlas/views/pages/experiments.py` | Saved experiment/configuration/results view |
| `src/atlas/views/pages/workspace.py` | Format-specific import, fewer previews, EDA controls, impact and undo |
| `src/atlas/views/pages/modeling.py` | Retained diagnostics, experiment-scoped predictions, cluster inspection and guarded comparison views |
| `src/atlas/views/styles/workspace.css` | Compact design tokens-based composition |
| `pyproject.toml`, `src/atlas/__init__.py` | Package version 2.0.0; no new runtime dependency |
| `.env.example`, project/root READMEs, docs | Offline defaults and reproduction/limitations documentation |
| `tests/test_offline_policies.py`, `tests/test_workspace_2.py`, `tests/test_providers.py` | Policies, permissions, graph/data contracts, end-to-end manual workflow, model probe |
| `tests/test_integration.py` | Canonical version_count assertions independent of legacy import ordering |
| `tools/audit_ui.py`, `tools/audit_capabilities.py`, `tools/demo_offline.py` | Repeatable inventories and actual offline decision demonstration |
| `tools/sidebar_browser_check.py` | Ten-screen/contextual-panel/mobile/chart capture verification |

Generated screenshots, wheel, demo evidence and galleries are local artifacts under
`artifacts/` and are excluded from Git. The scientific inventory JSON is versioned
documentation, not generated agent implementation.
