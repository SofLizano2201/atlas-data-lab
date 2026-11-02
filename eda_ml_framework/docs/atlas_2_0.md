# ATLAS 2.0 — intelligent workspace

## Architecture

The existing MVC hierarchy is retained. Scientific calculations, preprocessing,
training and diagnostics remain in `src/atlas/models`. Controllers forward and
coordinate workflows; Spanish views render controls and results. The new
WorkspaceController exposes real session context and progress without calculating
scientific metrics. Services retain session isolation and versioned datasets.

All local decision policies and optional provider logic remain exclusively in
`src/atlas/agents/agent.py`. Five polymorphic specialists use the existing gateway;
there is no second scientific engine. No new runtime dependency is required.

## Offline assistance

The default panel offers finite, explicit objectives. DecisionPolicy maps each goal
to a registered method, a prerequisite and validated arguments. Shared checks cover
dataset presence, numeric features, sample size, chosen target and saved experiments.
Suggestions adapt to current selections, existing results and prior assistance.
The user must initiate assistance; no data changes or training occur during page
rendering. Sensitive requests still require expiring, session/version-bound consent.

Examples: quality diagnosis; distribution exploration; correlation/outlier diagnosis;
PCA and elbow diagnostics; initial K-Means/logistic/Ridge proposals; class balance;
recorded metrics/residuals; compatible model ranking. These are documented initial
policies, not claims of globally optimal model selection or causal knowledge.
The model methods validate scientific prerequisites again at execution time.

Structured evidence and recommendations identify the actual tool and dataset.
Unknown guided commands are rejected. Without a language model the application does
not claim to understand arbitrary natural language. Manual analysis remains independent.

## Optional local language model

Install and start Ollama yourself, download a suitable local model, then configure
the environment before starting Streamlit. A model download is not performed by ATLAS.

```powershell
$env:ATLAS_LLM_PROVIDER = "ollama"
$env:ATLAS_LLM_MODEL = "your-installed-model"
$env:ATLAS_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
python -m streamlit run app.py
```

No API key is required for this local adapter. It uses Ollama's documented
[local compatibility endpoint](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx).
The UI can explicitly check `/v1/models` without sending a dataset. Unavailable or
incompatible models produce a clear message; guided offline commands remain usable.
Structured model output is validated against the same allowlist and argument schemas.
Actual inference depends on the installed model's schema support and local resources.
There is no automatic paid/cloud fallback. Reduced context is sent only with consent.

## Navigation and interaction

Three sidebar groups separate workspace, machine learning and results. Experiments
and comparison are distinct views. Existing routes remain compatible. A compact bar
shows dataset/version/dimensions/quality/workflow/count. Progress derives from actual
state: confirming data, calculating EDA, applying transformations, successful evaluated
experiments and reports for the current content hash. Preparation remains optional.
Suggested next actions navigate only.

Import displays format-specific controls, one file preview and explicit activation.
EDA offers overview, variable inspection, quality, distributions/relationships and
collapsed statistical tests. Cleaning distinguishes diagnosis, preview/impact, apply,
undo and restoration. Models retain existing results across reruns; predictions are
scoped by experiment rather than shared between unrelated models.

## Charts

The model chart standard uses the ATLAS palette, warm surfaces, clear axes, Spanish
labels, stable category colors, marker shapes/line patterns and actual-data tooltips.
Histograms support validated bin counts; supported plots support positive-only log
scales and facets capped at six categories. EDA can update simple charts on variable
changes; expensive statistical views stay explicit. A chart menu exposes zoom, point
selection and enlargement. Selected coordinates are inspectable; HTML downloads are
self-contained. These display interactions never refit a trained model.

Clustering includes group-size bars, profile heatmaps, bounded group inspection and
persisted PCA/silhouette displays. Profiles retain original units and warn against
cross-variable magnitude comparisons. Classification adds actual probability
distributions only when the estimator provides probabilities. Comparison offers
ranking, recorded runtime, cost/performance, fold distributions and binary ROC/PR
overlays, guarded by cohort/metric compatibility. Fold variation is not claimed to
prove stability between seeds or new datasets. No synthetic confidence bands appear.

## Reproduction

From `eda_ml_framework`, install with `python -m pip install -r requirements.txt`.
Run `python -m streamlit run app.py`. In this workspace the virtual environment is
in the parent directory: `..\.venv\Scripts\python.exe -m streamlit run app.py`.

```powershell
python -m pytest -q
python -m ruff check .
python tools/audit_ui.py
python tools/audit_capabilities.py
python tools/demo_offline.py
```

The demo explicitly runs small, public-data training jobs. It exercises all five
agents and saves calculated evidence in `artifacts/offline-decisions.json`.
See [initial audit and repair ledger](audit_2_0.md) for control ownership and limits.

## Limits

The local policy engine is decision support, not a generative LLM. Optional Ollama
inference requires an actual installation; HTTP contract tests cannot certify every
model. Optional UMAP/K-Medoids/XGBoost depend on their packages. Statistical Matplotlib
views remain for specialized diagnostics. Data is memory-only; no email scheduler
exists. Raster mockups are adapted to native Streamlit and available local fonts.
Automated tests cover representative workflows and validation cases, not every
possible dataset/parameter combination or every browser's download dialog.
