# AGENTS.md — Statistics WebApp

## Run
```sh
# Test Finder (main app with workspace)
streamlit run apps/app_finder.py

# Other standalone apps:
streamlit run apps/app_tabulation.py
streamlit run apps/app_explorer.py
streamlit run apps/app_distributions.py
streamlit run apps/app_power.py
streamlit run apps/app_diagnostics.py
streamlit run apps/app_examples.py
```
No tests, no linter, no type checker. Verification = `python -c "import ast; ast.parse(open('f').read()); from f import func"`.

## Architecture
Multi-module with `apps/`, `features/`, `core/` split. Each `apps/*.py` is a standalone entry point. `core/` has shared utilities; `features/` has UI modules.

### Apps (entry points)
| Module | Lines | Export |
|---|---|---|
| `apps/app_finder.py` | ~30 | `main()` — finder + data workspace routing via `st.session_state.page` |
| `apps/app_tabulation.py` | ~30 | `main()` — tabulation standalone |
| `apps/app_explorer.py` | ~30 | `main()` — graph explorer standalone |
| `apps/app_distributions.py` | ~30 | `main()` — distributions standalone |
| `apps/app_power.py` | ~30 | `main()` — power calculator standalone |
| `apps/app_diagnostics.py` | ~30 | `main()` — diagnostics standalone |
| `apps/app_examples.py` | ~30 | `main()` — solved examples standalone |

### Features (UI modules)
| Module | Lines | Export |
|---|---|---|
| `features/widgets.py` | ~5570 | `render_test_widget(test_name, external_data=None)` — 40+ test widgets |
| `features/finder_ui.py` | ~370 | `render_test_finder()` — finder UI + "All Tests" section + Data Import button |
| `features/data_workspace.py` | ~470 | `render_data_workspace()` — two-column workspace: upload, format selection, test picker, results |
| `features/builtin_datasets.py` | ~1040 | `get_builtin_datasets()`, `load_builtin_dataset()` — 20 curated datasets |
| `features/graph_explorer.py` | ~8010 | `render_graph_explorer()` |
| `features/tabulation.py` | ~3420 | `render_tabulation()` |
| `features/distributions.py` | ~940 | `render_distributions()` |
| `features/power_calculator.py` | ~3440 | `render_power_calculator(params)` |
| `features/glossary.py` | ~372 | `render_glossary()` |
| `features/diagnostics.py` | ~300 | `render_diagnostics()` |
| `features/solved_examples.py` | ~200 | `render_solved_examples()` |

### Core (shared utilities)
| Module | Lines | Export |
|---|---|---|
| `core/data.py` | ~1100 | `rules` list, `TEST_TO_SS_TYPE` mapping |
| `core/matching.py` | 36 | `find_matching_tests()` |
| `core/post_hoc.py` | 350 | `render_post_hoc(groups, param_type, key)` — 8 methods |
| `core/utils.py` | ~450 | `data_source_toggle()`, `format_p_value()`, `cohens_d_*` helpers, `st_plot_with_download()` |

### Legacy (flat-root modules)
These are the original monolithic modules preserved for reference:
`app_legacy.py`, `widgets.py`, `data.py`, `matching.py`, `post_hoc.py`, `flowchart.py`, `glossary.py`, `graph_explorer.py`, `tabulation.py`, `distributions.py`, `power_calculator.py`, `utils.py`, `solved_examples.py`, `diagnostics.py`

## Dependencies (from imports, no requirements.txt)
`streamlit numpy pandas plotly scipy statsmodels scikit-learn`

## Gotchas
- **Slider keys** must be **unique across all widgets** in `features/widgets.py`. Duplicate labels cause `StreamlitDuplicateElementId`. Format: `key=f"{test_slug}_{label_slug}"`.
- **Slider types** must match: `min_value`, `max_value`, `value`, `step` all int or all float.
- **Plotly colors**: use `rgba(r,g,b,a)` — 8-digit hex (`#4C78A825`) is not supported.
- **scipy.special.gammaln** — not `scipy.stats.gammaln` (does not exist).
- **`st.rerun()`** required after setting `st.session_state` keys to take effect.
- **Random seed**: `np.random.seed(42)` in widgets/distributions; `np.random.default_rng(42)` in graph_explorer/tabulation.
- **No `.gitignore`** — `__pycache__/` appears in git status.
- **Theme**: `template="plotly_dark"` on every Plotly figure.
- **`icon=""` breaks** — Streamlit requires valid emoji or no `icon` param in `st.success/info/warning/error`.
- **`data_source_toggle()`** — each widget's `src = data_source_toggle(...)` is now wrapped: `if external_data and external_data.get("using_uploaded"): src = external_data else: src = data_source_toggle(...)` — this enables the Data Workspace to inject external data.

## Conventions
- All charts: Plotly, dark template, `use_container_width=True`.
- Num tables styled with `_apa_table()` helper.
- `st.info()` / `st.expander()` for educational guidance.
- Sub-navigation uses 2‑column `_tab_buttons()` helper.
- `render_post_hoc(groups, param_type="parametric"|"nonparametric", key="prefix")` at end of multi-group widgets.
- **Data Workspace**: two-column layout (left: data mgmt, right: results). Uses `st.session_state.ws_*` keys for persistence.
- **`_build_external_data(df, organization, col_config)`** converts raw data into `external_data` dict matching `data_source_toggle()` return format. Supports modes: `one_sample`, `two_sample`, `multi_sample`, `paired`, `repeated`, `correlation`, `categorical_two`, `categorical_one`.
