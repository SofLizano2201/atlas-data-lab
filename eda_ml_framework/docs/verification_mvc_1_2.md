# MVC 1.2 verification — 2026-10-28

## Executed checks

- Pre-refactor baseline: 155 tests passed. Existing scientific and interface tests
  remain in the suite; calculations were moved into `src/atlas/models`, not replaced.
- Final pytest run: **182 passed, 7 warnings in 47.54 seconds**. The warnings are
  the empty-column mean diagnostic and six intentionally short MLP convergence cases.
- Final Ruff lint and format checks passed (137 Python files discovered, including
  the generated chart verification application).
- `pip check`: no broken requirements.
- Built `artifacts/dist/atlas_eda_ml-1.2.0-py3-none-any.whl` through isolated PEP 517.
- Installed that wheel without dependencies into `artifacts/wheel-smoke-1.2`.
  An isolated Python invocation verified that canonical and compatibility imports
  originated from that installation, academic inheritance remained valid, and CSS
  and SVG resources were packaged.
- `tools/demo_agents.py --confirm-training`: all five specialists succeeded, using
  actual public Iris/Diabetes data, training and comparison. Six executions cover
  two classification models. Output: `artifacts/mvc-demo.json`; explicitly local
  framework tools, not claimed external LLM inference.
- `tools/audit_capabilities.py`: 91 owned public methods/properties and 21 registered
  capabilities with typed contracts and actual callable references.
- Fresh Streamlit instance on port 8504 started successfully. The user's 8503 session
  was not stopped or replaced.
- Edge/CDP check `tools/sidebar_browser_check.py --port 8504 --all-pages --agents`
  completed: nine screens, contextual panels, dataset activation and navigation
  without execution exceptions. Sidebar widths 260/72 px; navigation heights 46 px;
  mobile viewport/document widths both 390 px.
- Captures and measurements: `artifacts/atlas-mockups/`. Home, EDA, report and
  classification-panel screenshots were visually inspected. Existing palette,
  sidebar, hero, cards and native controls remain consistent. The mobile sidebar
  overlays content while open and can be collapsed; raster-perfect identity is not
  claimed. See `mockups_2026_10_27.md` for native-control differences.
- `tools/verify_charts.py`: 12 Plotly charts and three statistical figures verified;
  gallery generated at `artifacts/charts/gallery.html`.
- `tools/verify_pdf.py`: generated PDF reopened, text checked, and all three pages
  checked against their page bounds.

New tests cover abstract contracts, polymorphic dispatch, real method reuse,
structured evidence, provider-independent manual pages, configuration confirmation
updating actual Streamlit controls, permissions at the gateway boundary, session
scope/expiry, English configuration keys, legacy compatibility and architectural
dependency boundaries. Transport tests use a local HTTP server with strict function
calls, including rejection/error cases.

## Remaining limits

No external provider credentials were available, so paid remote inference was not
executed. Configure the variables in `agents.md` to validate your actual account,
model and provider. The interface states unavailability clearly; it never simulates
an LLM response. Optional UMAP, K-Medoids and XGBoost training was not part of this
verification. Known MLP convergence warnings in short tests and the empty-column
mean warning are reported, not hidden.

Data/session state remains memory-only. No email scheduler is configured. This
refactor is local and was not automatically committed or pushed to GitHub.
