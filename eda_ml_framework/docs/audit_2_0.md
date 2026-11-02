# ATLAS 2.0 — initial functional audit

The existing application was inspected before editing: models, controller forwards,
21 capabilities, session lifecycle, all page modules, chart builders, providers,
dependencies and tests. The 1.2 capability inventory remains the scientific baseline.

| Workflow / visible actions | View | Controller → model | Initial finding |
|---|---|---|---|
| Navigation, collapse, search | sidebar/layout | session routing | Functional; ungrouped; comparison conflated with experiment history |
| Upload, parse, sheet selection, confirmation | workspace.data_page | DatasetController → EDA.load; DatasetService.activate | Functional; CSV options visible for unrelated formats; redundant previews |
| Summary, variables, frequencies, grouped statistics | workspace.eda_page | EDAController → EDA methods | Functional; dense tabs; limited contextual orientation |
| Charts, correlation, tests, outliers | workspace.eda_page | EDAController → chart/correlation/statistical_test/outlier_diagnostic | Functional; visualization preferences and interactive selection not exposed |
| Cleaning preview, apply, restore, export | workspace.cleaning_page | EDAController → transform/apply/restore/export | Functional; impact summary and direct undo missing |
| Representation, distances, clustering, elbow, dendrogram | modeling.clustering_page | ClusteringController → Clustering/UnsupervisedLearning | Functional; multiple expensive actions; some results disappear on reruns |
| Train, search, evaluate, predict, interpret | modeling.supervised_page | Classification/RegressionController → scientific model | Functional; defaults and result persistence present; auxiliary outputs ephemeral; prediction state shared across models |
| Comparison, direction, chart, export | modeling.comparison_page | ComparisonController → ExperimentComparison.rank | Functional; incompatible experiments raise understandable error but filtering precedes no guidance |
| HTML/PDF, sections, history, data/figure export | reports.reports_page | ReportController → reporting | Functional; scheduler explicitly unavailable, not a fake working control |
| Open, quick tools, chat, confirm/cancel | agent_panel | AgentController → BaseAgent → gateway → existing method | Partially functional: tools work offline, goal selection requires LLM or raw JSON; panel intrusive |
| Chart theme and downloads | ui.figure; model visualization | chart constructors in model | Functional; style exists, selection/enlargement/log/facets not exposed consistently |

Runtime correctness is established by tests and explicit verification, not this
initial classification alone. `ui_controls_2_0.json` enumerates view controls and
their owning functions. Dynamic controls are represented by source expressions;
the inventory is not a claim that every combinatorial parameter choice was executed.
Final test results and unresolved limits are recorded separately.

## Repair ledger and executed evidence

| Finding | Repair in appropriate layer | Verification |
|---|---|---|
| Offline assistance required raw tool parameters | Agent DecisionPolicy catalog, explicit goals/prerequisites, shared validated execution and contextual priorities | `test_offline_policies.py`; actual five-agent demo |
| Dense assistant competed with analytical forms | Guided actions first; objective, technical tools and evidence collapsed; no panel before explicit invocation | AppTest panel integration; actual panel screenshots |
| No distinct saved-experiment workspace | New history view, existing experiment store/downloads; separate compatible-cohort comparator | `test_workspace_2.py` manual end-to-end; ten-screen browser check |
| Dataset/workflow context unclear | WorkspaceController with real quality/count/version and operation-derived progress | Progress tests; native browser captures |
| Import showed unrelated settings/repeated previews | CSV settings conditional; Excel only for XLSX; single file preview and confirmation, collapsed sample preview | Existing four-format parser/round-trip tests and data-screen inspection; native upload-dialog automation not claimed |
| Cleaning impact/undo absent | Model transformation_impact, explicit confirmed undo and preserved audit/version history | Model mutation guard test; cleaning AppTest and existing restoration tests |
| Visualization options hidden or absent | Validated bins/log/facets, chart preferences, actual point selection, enlargement, stable zoom identity, ATLAS colors/patterns | Semantic figure tests; chart export gallery; EDA in-app capture |
| Light series hard to distinguish | Brand-derived palette, dark marker outlines, histogram patterns, extended cluster colors plus shapes | Figure assertions and inspected chart captures |
| Predictions could appear under another selected model | Session prediction keys scoped by experiment; cleared on dataset activation | Source fix; existing model prediction tests; session isolation contracts |
| Costly auxiliary outputs disappeared on reruns | Stored importance/coefficient/learning tables; persisted PCA/silhouette and cached summary figures | Source review; no duplicate training on rerun in end-to-end test |
| Elbow/dendrogram ignored selected preprocessing | Forward selected scaler/encoding to existing model preparation | Existing clustering/preprocessing suite; direct reviewed forwarding |
| Comparison errors occurred after selection | Expose cohorts first; model still validates compatibility; guarded runtime/cost/CV/ROC/PR views | Comparison tests, incompatible-cohort rejection, actual retained fold scores |
| Probability distributions absent | Store real estimator probabilities and guarded model visualization | Actual trained-classifier chart test; models without probabilities reject instead of fabricating |

All 159 static control declarations were inventoried, including shared/dynamic
controls. Main action families have automated backend/integration evidence. This
does not certify every possible parameter combination, native OS file chooser,
download dialog or arbitrary user dataset. No hidden scheduler, synthetic results
or placeholder agent was introduced. See [final verification](verification_2_0.md).
