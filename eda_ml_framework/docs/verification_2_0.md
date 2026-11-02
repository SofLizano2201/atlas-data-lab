# ATLAS 2.0 — verification on 2026-10-28

## Automated checks executed

- Baseline before changes: **182 passed**, seven warnings, 38.25 seconds.
- Final full suite: **203 passed**, seven warnings, **47.30 seconds**.
- Warnings: one empty-column mean diagnostic and six MLP convergence warnings from
  tests intentionally limited to 20 iterations. No new warning was hidden.
- Ruff lint and format checks passed. The final discovery included 146 Python files,
  including generated verification applications under ignored artifacts.
- `pip check`: no broken requirements; no new mandatory runtime dependency.
- Existing scientific hierarchy, train-only preprocessing, metrics, methods, optional
  dependency errors, permissions, schema validation and architectural-boundary tests
  remain in the suite.
- New policy tests exercise all five specialists offline: actual diagnosis,
  distribution/PCA/class tools, confirmation-gated training, missing prerequisites,
  unknown/malicious goals and compatible comparison.
- AppTest executes import of public example → EDA → cleaning preview/confirmation →
  two-model training/evaluation → history → comparison → report. Provider construction
  is patched to raise during this manual workflow: manual features remain independent.
  Rerunning does not add another experiment. Existing regression/clustering workflows,
  report formats, restoration and four-format file parser tests also pass.
- Real chart tests verify bins, log validation, facets, unchanged input frames, actual
  group sizes/profiles, probabilities and recorded CV distributions; incompatible
  cohorts are rejected. Native Plotly interaction remains a display operation.
- `tools/demo_offline.py` completed for all five agents on public Iris and Diabetes
  data, including an explicit alternative classifier. Seven successful responses
  were saved to `artifacts/offline-decisions.json`; no language-model inference is
  claimed for this deterministic demonstration.
- Inventories regenerated: **159 static UI control declarations**, **95 owned public
  methods/properties**, **21 existing authorized capabilities**. Static inventory is
  not a claim of exhaustive parameter-combination runtime coverage.

## Packaging and exports

Built `artifacts/dist/atlas_eda_ml-2.0.0-py3-none-any.whl` with isolated PEP 517.
Installed it into `artifacts/wheel-smoke-2.0` without dependencies. Isolated Python
verified canonical/compatibility imports came from that installation, checked the
2.0 version and packaged workspace CSS, and executed an offline EDA goal successfully.

`tools/verify_charts.py` verified 12 Plotly figures and three specialized statistical
figures. The gallery is `artifacts/charts/gallery.html`; actual scatter, histogram,
correlation/confusion, regression/residual, PCA and silhouette captures were inspected.
`tools/verify_pdf.py` reopened the generated PDF, checked text and all three page bounds.

## Browser and visual evidence

An isolated Edge/CDP session checked ten destinations, active navigation, contextual
agent panels, sample activation, sidebar expanded/compact states and a mobile layout.
Measured desktop sidebar widths: **260 / 72 px**. All ten visible navigation targets
were **46 px** tall. Mobile viewport and document width both **390 px**.
Captures and measurements are under `artifacts/atlas-2.0/`; this directory is excluded
from Git to avoid publishing session screenshots. Home, EDA, agent panel, mobile and
chart gallery captures were actually inspected. Mockup identity is preserved with a
shorter hero, grouped navigation, real context and native controls; raster-perfect
identity is not claimed. The mobile menu overlays content while open and is collapsible.

Final review instance: `http://127.0.0.1:8506/`. No user session on 8503/8504 was
deliberately stopped or replaced. No real/private dataset was used for QA screenshots.

## Explicit limits

No actual Ollama model installation/inference was available for this verification.
Loopback transport, catalog discovery, schema validation and unavailable-model
behavior are tested with an explicitly identified local HTTP fixture. This is not
presented as real LLM inference. Guided policies require neither that runtime nor API keys.

Optional UMAP/K-Medoids/XGBoost training was not exercised. Local model capabilities,
large-data limits, scientific assumptions and compatibility constraints still apply.
CV fold variation is not a multi-seed stability study. No paid provider, mail scheduler,
unrestricted natural-language interpreter, automatic destructive cleanup or fabricated
metric is supplied. Generic EDA category colors remain capped for readability; cluster
inspection supports an extended palette. Specialized Matplotlib views retain PNG export.

OS-native file dialogs, browser download dialogs and all arbitrary datasets/parameter
combinations are not exhaustively automated. File parsing and export bytes are tested;
main UI workflows and visible state transitions have integration and browser evidence.
