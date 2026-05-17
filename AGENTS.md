# AGENTS.md — Statistics WebApp

## Run
```sh
streamlit run app.py
```
No tests, no linter, no type checker. Verification = `python -c "import ast; ast.parse(open('f').read()); from f import func"`.

## Architecture
Flat modules in root (no package, no `__init__.py`). Entry: `app.py::main()` routes to one of four mode render-functions. Each module exports a single `render_*()` called by `app.py`.

| Module | Lines | Export |
|---|---|---|
| `app.py` | ~1390 | `main()` — entry, sidebar routing, power calculator |
| `widgets.py` | ~4760 | `render_test_widget(test_name)` — interactive plot + stats per test |
| `post_hoc.py` | 350 | `render_post_hoc(groups, param_type, key)` — 8 methods |
| `graph_explorer.py` | ~8010 | `render_graph_explorer()` |
| `tabulation.py` | ~3420 | `render_tabulation()` |
| `distributions.py` | ~940 | `render_distributions()` |
| `power_calculator.py` | ~3440 | `render_power_calculator(params)` |
| `data.py` | ~1100 | `rules` list, `TEST_TO_SS_TYPE` mapping |
| `matching.py` | 36 | `find_matching_tests()` |
| `flowchart.py` | 95 | `build_tree()`, `build_sunburst_chart()` |
| `glossary.py` | 372 | `render_glossary()` |

## Dependencies (from imports, no requirements.txt)
`streamlit numpy pandas plotly scipy statsmodels scikit-learn`

## Gotchas
- **Slider keys** must be **unique across all widgets** in `widgets.py`. Duplicate labels cause `StreamlitDuplicateElementId`. Format: `key=f"{test_slug}_{label_slug}"`.
- **Slider types** must match: `min_value`, `max_value`, `value`, `step` all int or all float.
- **Plotly colors**: use `rgba(r,g,b,a)` — 8-digit hex (`#4C78A825`) is not supported.
- **scipy.special.gammaln** — not `scipy.stats.gammaln` (does not exist).
- **`st.rerun()`** required after setting `st.session_state` keys to take effect.
- **Random seed**: `np.random.seed(42)` in widgets/distributions; `np.random.default_rng(42)` in graph_explorer/tabulation.
- **No `.gitignore`** — `__pycache__/` appears in git status.
- **Theme**: `template="plotly_dark"` on every Plotly figure.

## Conventions
- All charts: Plotly, dark template, `use_container_width=True`.
- Num tables styled with `_apa_table()` helper.
- `st.info()` / `st.expander()` for educational guidance.
- Sub-navigation uses 2‑column `_tab_buttons()` helper.
- `render_post_hoc(groups, param_type="parametric"|"nonparametric", key="prefix")` at end of multi-group widgets.
