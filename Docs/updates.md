# Updates.md — Statistics WebApp Changelog

## [2026-06-12] Phase 3: Cross-Linking, Widget Expansion, Diagnostic Expansion

### Overview
Cross-linked the test finder with diagnostics, expanded to 71 tests + 24 diagnostics, integrated solved examples into test detail pages, and fully refactored the monolithic widgets.py into a modular `features/widgets/` package.

### Architecture Changes

#### New Package: `features/widgets/`
Refactored the monolithic `features/widgets.py` (7015 lines) into a clean directory with 9 module files:

```
features/widgets/
├── __init__.py          # Registry: _widget_registry dict, register_widget(), get_widget()
├── parametric.py        # t-tests, ANOVA, z-test, equivalence, correlation, regression
├── nonparametric.py     # Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, sign, permutation
├── categorical.py       # Chi-square, Fisher's, McNemar, multinomial, Poisson GOF, Barnard, Boschloo, Stuart-Maxwell
├── agreement.py         # Cohen's Kappa, Fleiss' Kappa, Weighted Kappa, Gwet's AC1, Krippendorff's Alpha, ICC
├── survival.py          # Kaplan-Meier, Cox PH, Log-Rank
├── diagnostic.py        # Sensitivity, Specificity, ROC, PPV/NPV, LR/LR-, calibration, confusion matrix
├── spc.py               # Xbar-R, Xbar-S, I-MR, p, np, c, u, EWMA, CUSUM
└── regression.py        # Hosmer-Lemeshow, regularization path
```

Each test registers via `@register_widget("Test Name")` decorator. Total: **71 registered widgets**.

#### New Asset: `assets/test_definitions.json`
Moved all 71 test definitions from `core/data.py` into structured JSON. Each entry contains:
- `name`, `category`, `family`, `objective`, `variables`, `description`, `formula`, `assumptions`, `interpretation`, `decision_rule`, `examples`, `when_to_use`, `limitations`, `references`

#### Cross-Linking: Finder → Diagnostics
- `_TEST_DIAG_MAP` in `finder_ui.py` maps 54 tests → (diagnostic name, category, description)
- Rendered as "🔍 Recommended Assumption Checks" in the left column of each test detail page
- Query param support in `diagnostics.py` — `?diag=Shapiro-Wilk+Test` deep-links from finder

#### Solved Examples Integration
- `_SOLVED_EXAMPLE_MAP` maps 7 test names → `_solved_*()` functions from `solved_examples.py`
- Rendered as "📝 Step-by-Step Walkthrough" expander in detail pages
- Currently covers 7 one-sample tests (one-sample t, z, Wilcoxon, sign, proportion exact, proportion normal, binomial)

#### Layout Changes
- **Interpretation** moved from right column → left column (after Decision Rules, before Post-Hoc)
- **Recommended Assumption Checks** added to left column (bottom)
- Both changes in `render_test_detail_page()`

### New Tests Added (14 total)

| Test | Module | Type |
|------|--------|------|
| One-way Welch ANOVA | `parametric.py` | Parametric |
| Hotelling's T-Squared | `parametric.py` | Multivariate |
| Two-Sample Kolmogorov-Smirnov | `nonparametric.py` | Nonparametric |
| Jonckheere-Terpstra Test | `nonparametric.py` | Nonparametric |
| Page Test for Ordered Alternatives | `nonparametric.py` | Nonparametric |
| Mann-Kendall Trend Test | `nonparametric.py` | Nonparametric |
| G-Test (Goodness-of-Fit) | `categorical.py` | Categorical |
| Barnard's Exact Test | `categorical.py` | Categorical |
| Boschloo's Exact Test | `categorical.py` | Categorical |
| Stuart-Maxwell Test | `categorical.py` | Categorical |
| Gwet's AC1 | `agreement.py` | Agreement |
| Krippendorff's Alpha | `agreement.py` | Agreement |
| Intraclass Correlation Coefficient (ICC) | `agreement.py` | Agreement |
| Hosmer-Lemeshow Test | `regression.py` | Regression |

### New Diagnostics Added (2 total)

| Diagnostic | Category | Description |
|-----------|---|-----------|
| **Brown-Forsythe Test** | Homogeneity of Variance | Robust alternative to Levene's (uses median instead of mean) |
| **Rainbow Test** | Linearity / Specification Tests | Tests linearity assumption in regression models |

Total: **24 diagnostic functions** across **9 categories**.

### Migrated to Legacy
- `apps/app_examples.py` → `legacy/app_examples.py` (functionality integrated into finder as solved examples)

### Bug Fixes
- `barnard_exact` / `boschloo_exact`: Result objects expose `.statistic`, `.pvalue` (not tuples)
- `page_trend_test`: No `.statistic_z` — Z computed manually from L formula
- `stuart_maxwell`: `chi2_sm.cdf()` on numpy float → `scipy.stats.chi2.cdf()`

### Data Workspace Updates
- Categorical widgets now consume `external_data` with `categorical_one`/`categorical_two` format
- Dynamic adaptation to variable numbers of categories/cells from user data
- `_build_external_data()` helper converts raw DataFrames → `external_data` dict
- Supports modes: `one_sample`, `two_sample`, `multi_sample`, `paired`, `repeated`, `correlation`, `categorical_two`, `categorical_one`
- `_apa_table()` consolidated in `core/utils.py` (was duplicated across 4 feature files)

### Key Numbers
| Metric | Before | After |
|--------|--------|-------|
| Test definitions | 57 (in `core/data.py`) | 71 (in `assets/test_definitions.json`) |
| Registered widgets | 57 (in `features/widgets.py`) | 71 (in `features/widgets/`) |
| Diagnostics | 22 | 24 |
| Cross-link mappings | 0 | 54 (`_TEST_DIAG_MAP`) |
| Solved examples integrated | 0 | 7 (`_SOLVED_EXAMPLE_MAP`) |

---

## [2026-05-26] Phase 2: Multi-App Refactoring

### Overview
Refactored the monolithic `app.py` (2655 lines) into 7 independently deployable mini-apps with clean separation of concerns.

### Architecture Changes

#### Directory Structure Created
```
apps/                    # Independent app entry points
├── app_finder.py        # 🔍 Test Finder (original mode "Find my test")
├── app_explorer.py      # 📈 Graph Explorer (mode "Graph explorer")
├── app_tabulation.py    # 📋 Tabulation (mode "Tabulation")
├── app_power.py         # ⚡ Power Analysis (mode "Power calculator")
├── app_distributions.py # 🎲 Probability Distributions (mode "Distributions")
├── app_designs.py       # 📐 Study Designs (mode "Study Designs")
├── app_data_workspace.py# 🛠️ Data Workspace (standalone)
└── app_diagnostics.py   # 🔬 Data Screening (mode "Diagnostics")

legacy/                  # Archived monolithic modules
├── app_legacy.py        # Original monolithic app.py (2655 lines)
└── app_examples.py      # Solved Examples (functionality integrated into finder)

core/                    # Shared utilities imported by multiple apps
├── __init__.py
├── utils.py             # format_p_value(), cohens_d_ci(), omega_squared(), etc.
├── data.py              # rules list, TEST_TO_SS_TYPE mapping, FIELDS
├── matching.py          # find_matching_tests()
└── post_hoc.py          # render_post_hoc() (8 post-hoc methods)

features/                # Feature modules (1 per app or logical group)
├── widgets/             # Modular package (71 widgets across 9 files)
│   ├── __init__.py      # Registry: _widget_registry, register_widget(), get_widget()
│   ├── parametric.py    # t-tests, ANOVA, z-test, equivalence, correlation, regression
│   ├── nonparametric.py # Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, etc.
│   ├── categorical.py   # Chi-square, Fisher's, McNemar, Barnard, Boschloo, etc.
│   ├── agreement.py     # Kappa, ICC, Gwet's AC1, Krippendorff's Alpha
│   ├── survival.py      # Kaplan-Meier, Cox PH, Log-Rank
│   ├── diagnostic.py    # Sensitivity, Specificity, ROC, PPV/NPV
│   ├── spc.py           # Control charts (Xbar-R, I-MR, p, c, EWMA, CUSUM)
│   └── regression.py    # Hosmer-Lemeshow, regularization path
├── __init__.py
├── finder_ui.py         # Test Finder UI + Flowchart + cross-linking
├── data_workspace.py    # Data Workspace with AG Grid
├── power_ui.py          # Power Analysis UI (1300+ lines)
├── graph_explorer.py    # Graph Explorer UI
├── tabulation.py        # Tabulation UI
├── distributions.py     # Distributions UI
├── diagnostics.py       # Diagnostics UI (24 functions, query-param routing)
├── solved_examples.py   # Solved Examples UI
├── glossary.py          # Glossary UI
└── flowchart.py         # build_tree(), build_sunburst_chart()
```

#### File Extraction from `app.py`
- Lines 19-1314 → `features/power_ui.py` (Power Analysis UI)
- Lines 1376-1408 + 2433-2648 → `features/finder_ui.py` (Test Finder UI)
- Lines 1413-2432 → Removed (dead code: `if False:` block containing duplicate old Sample Size Estimation UI)

### Key Improvements

| Aspect | Before (Monolith) | After (Multi-App) |
|---|---|---|
| **Deployment** | Single app only | 7 apps independently deployable |
| **Import footprint** | `sklearn` always loads | Only loads for Graph Explorer |
| **Startup time** | Loads all 8000-line modules | Only loads what's needed |
| **Maintainability** | Single 2655-line `app.py` | Cleanly separated modules |
| **Archive** | No reference | `app_legacy.py` preserved as reference |

### How to Run Each App

```sh
# From project root, run any app independently:
streamlit run apps/app_finder.py          # Test Finder
streamlit run apps/app_explorer.py        # Graph Explorer
streamlit run apps/app_tabulation.py      # Tabulation
streamlit run apps/app_power.py           # Power Analysis
streamlit run apps/app_distributions.py   # Distributions
streamlit run apps/app_diagnostics.py     # Data Screening
streamlit run apps/app_designs.py         # Study Designs
streamlit run apps/app_data_workspace.py  # Data Workspace
```

### Import Fixes Applied
- `matching.py`: `from data` → `from core.data`
- `flowchart.py`: `from data` → `from core.data`
- `widgets.py`: `from post_hoc` → `from core.post_hoc`, `from utils` → `from core.utils`
- All `apps/app_*.py`: Use `sys.path.insert()` to enable running from `apps/` subdirectory

### Removed
- Dead `if False:` block in `app.py` (lines 1413-2432) — duplicate old Sample Size Estimation UI that was never executed

### Archived
- Original `app.py` → `app_legacy.py` (preserved for reference)
- `apps/app_examples.py` → `legacy/app_examples.py` (functionality integrated into finder solved examples)

### Later Refinements
- **Monolithic widgets.py broken up** → `features/widgets/` package with 9 modules and a registry pattern
- **Test definitions decoupled** → `assets/test_definitions.json` (71 entries)
- **Cross-linking system** → Finder ↔ Diagnostics via `_TEST_DIAG_MAP` and query params
- **Solved examples integrated** → Inline walkthroughs in test detail pages
- **14 new tests added** → Total now 71 (Widgets + SPC + Agreement + Regression)
- **2 new diagnostics added** → Total now 24 (9 categories)

---

## [2026-05-26] Phase 1: Professional Statistical Output

### Overview
Brought output quality in line with professional statistical software (R, JASP, SPSS) per APA 7th edition guidelines.

### New File: `utils.py`
Created utility module with reusable statistical and UI functions:

| Function | Description |
|---|---|
| `format_p_value(p)` | APA 7th edition p-value formatting: `p < .001`, `p = .035` (no leading zeros) |
| `cohens_d_one_sample_ci(d, n)` | 95% CI for one-sample/paired Cohen's d using non-central t-distribution |
| `cohens_d_independent_ci(d, n1, n2)` | 95% CI for independent-samples Cohen's d |
| `hedges_g(d, n1, n2)` | Small-sample bias correction for Cohen's d (important when n < 50) |
| `omega_squared_partial(...)` | Unbiased partial ω² for ANOVA (APA-preferred over biased partial η²) |
| `st_plot_with_download(fig, key)` | Plotly figure wrapper with PNG (300 DPI) and SVG download buttons |
| `interpret_cohens_d()` / `interpret_eta_squared()` | Text interpretation using Cohen's (1988) benchmarks |

### Modified: `widgets.py`

#### Updated Core Test Widgets
6 widgets comprehensively updated with full Phase 1 improvements:

| Test | New Features |
|---|---|
| **One-sample t-test** | APA p-values, Cohen's d [95% CI], Hedges' g, interpretation, plot export |
| **Independent t-test** | APA p-values, Cohen's d [95% CI], Hedges' g, interpretation, plot export |
| **Welch's t-test** | APA p-values, Cohen's d [95% CI], Hedges' g, interpretation, plot export |
| **Paired t-test** | APA p-values, Cohen's d_z [95% CI], Hedges' g, interpretation, plot export |
| **One-way ANOVA** | APA p-values, ω² (unbiased), interpretation, plot export, explanatory expander |
| **Two-way ANOVA** | APA p-values, **partial ω² for all 3 effects** (A, B, A×B), plot export, explanatory expander |

#### All Tests: APA P-Value Formatting
Applied `replaceAll` to standardize p-value display:
- **Before**: `st.latex(rf"\text{{p-value}} = {p:.5f}")` → `"p-value = 0.00000"`
- **After**: `st.latex(rf"\text{{{format_p_value(p)}}}")` → `"p < .001"` or `"p = .035"`

#### Statistical Detail Tables Enhanced
For t-tests:
- Added Cohen's d confidence interval column: `0.50 [0.12, 0.88]`
- Added Hedges' g (small-sample unbiased correction)
- Added qualitative interpretation column (Trivial/Small/Medium/Large)

For ANOVA:
- Added ω² (omega-squared) alongside η² (eta-squared)
- Added explanatory expanders documenting why ω² is preferred:
  > "η² is positively biased. ω² applies a correction recommended by APA 7th edition."
- Added interpretation column using Cohen's benchmarks (ω² ≈ 0.01=Small, 0.06=Medium, 0.14=Large)

#### Plot Export
Converted key `st.plotly_chart()` calls to `st_plot_with_download()`:
- Each plot now has 2 download buttons below it
- **PNG**: 2x scale = ~300 DPI for publication
- **SVG**: Vector format for editing in Illustrator/Inkscape
- Gracefully handles missing `kaleido` package with info message

### Before/After Comparison

| Feature | Before | After (APA 7th / Professional) |
|---|---|---|
| P-value display | `p-value = 0.00000` | `p < .001` (no leading zeros) |
| Cohen's d | Point estimate only: `0.500` | With CI: `0.50 [0.12, 0.88]` |
| Small sample N < 50 | No correction | Hedges' g corrects Cohen's d bias |
| ANOVA effect size | Only partial η² (biased) | Both η² and **partial ω²** (unbiased, APA preferred) |
| Plot export | Right-click → screenshot | 📥 Download PNG (300 DPI) or SVG |
| Interpretation | None | Trivial/Small/Medium/Large per Cohen (1988) |

### Dependencies
Optional for plot export:
```
pip install kaleido  # For static PNG/SVG export
pip install -U plotly  # Ensure latest Plotly for SVG export
```

---

## [2026-05-26] Aesthetic Prototype: Statistics Assistant Homepage

### New File: `statistics_assistant.py`

Created a clickable unified homepage prototype for aesthetic testing. This demonstrates the multi-app architecture **without refactoring** — purely for design/UX evaluation.

### What It Does

**7 large interactive app cards in 3 sections:

| Section | Mini-App |
|---|---|
| **Statistical Analysis** | 🔍 Test Finder, 📈 Graph Explorer |
| | 📋 Tabulation, ⚡ Power Analysis |
| **Reference & Learning** | 🎲 Probability Distributions |
| | 📚 Step-by-Step Solved Examples |
| **Data Quality** | 🔬 Data Screening & Diagnostics |

### Interactive Demo Features:
- Each card shows icon, title, description, and colored capability badges
- Clicking "🚀 Launch" navigates to a fake "loaded app" view
- Each fake app view shows:
  - Real-looking preview content matching that app's purpose
  - Architecture explanation comparing monolith vs multi-app
  - Back button to return home

### Design Choices Evaluated

| Aspect | Design Decision |
|---|---|
| **Layout** | ~~2-column~~ → **3-column grid** for large screens, with logical groupings |
| **Grouping** | Row 1: Core Analysis (Finder, Graphs, Tables); Row 2: Planning & Reference (Power, Distributions, Examples); Row 3: Data Quality (Diagnostics centered) |
| **Card styling** | Dark gradient cards with hover effects (border color change, shadow, lift) |
| **Badges** | Colored by capability type |
| **Primary apps** | Blue gradient buttons, **secondary** (reference/learning) |
| **Diagnostics** | Centered in middle column (1:3:1 ratio) for visual emphasis as final "check" step |

### Layout Change: 2-column → 3-column

**Before (2-column):**
```
[ Test Finder ]    [ Graph Explorer ]
[ Tabulation  ]    [ Power Analysis ]
[ Distributions]   [ Examples ]
[ Diagnostics (full width) ]
```

**After (3-column):**
```
[ Core Analysis ]
[ Finder ]  [ Graphs ]  [ Tables ]

[ Planning & Reference ]
[ Power ]   [ Distributions ]  [ Examples ]

[ Data Quality ]
          [ Diagnostics ]    (centered)
```

Better use of horizontal space on large monitors; logical workflow grouping:
1. **Core Analysis** → What you actually run (tests, visualizations, tables)
2. **Planning & Reference** → Up-front planning (power) + learning tools
3. **Data Quality** → Final check before/after analysis

### To Run

```sh
streamlit run statistics_assistant.py
```

### Purpose

This file is for **aesthetic testing only**. Clicking an app demonstrates the navigation experience and visual language. In the actual multi-app refactor:

- Each `Launch` button would go to a separate `streamlit run apps/app_finder.py` etc.
- Only the clicked mini-app's modules would be imported
- `sklearn` never loads unless Graph Explorer is explicitly launched
- Power calculator code never loads unless Power Analysis is launched

---

## [2026-05-26] Pure HTML/CSS Portal Homepage (No Streamlit)

### Problem
1. Streamlit's auto-theming conflicted with custom CSS
2. A portal/landing page shouldn't require Python/Streamlit to serve
3. Need clean separation: static homepage linking to deployed Streamlit apps

### Solution
Created **`index.html`** — a pure HTML/CSS landing page with beautiful, production-ready styling that works independently.

### Key Features

| Aspect | Details |
|---|---|
| **Pure static** | No Python, no Streamlit, no dependencies |
| **Modern dark theme** | Slate/indigo gradient background with glassmorphism cards |
| **Responsive** | 3-column → 2-column → 1-column based on screen width |
| **Interactive** | Hover effects, lift animation, glow borders, arrow movement on hover |
| **Deployment-ready** | Deploy this single HTML file; then link each card to your Streamlit apps |

### Layout Structure

```
[ Logo Badge: MULTI-APP PORTAL ]
[ Statistics Assistant ]
[ Choose your analysis path... ]

       8     71    83
    Mini Apps  Tests  Plots

--- Core Analysis ---
[ 🔍 Test Finder ]  [ 📈 Graph Explorer ]  [ 📋 Tabulation ]

--- Planning & Reference ---
[ ⚡ Power Analysis ]  [ 🎲 Distributions (secondary) ]  [ 📚 Examples (secondary) ]

--- Data Quality ---
               [ 🔬 Data Screening (centered horizontal card) ]

--- Architecture Comparison ---
[ ❌ Monolith ]  [ ✅ Multi-App ]  [ 📊 Improvements Table ]
```

### Deployment Instructions

Each app card is wrapped in `<a href="#">...`

**Change** (in index.html, 7 locations):
```html
<a href="#" style="text-decoration: none; color: inherit;">
```

**To** (point to your deployed Streamlit apps):
```html
<a href="https://yourdomain.com/finder" style="text-decoration: none; color: inherit;">
<a href="https://yourdomain.com/graphs" style="text-decoration: none; color: inherit;">
<a href="https://yourdomain.com/tables" style="text-decoration: none; color: inherit;">
<!-- etc. -->
```

### Visual Design Highlights

| Element | Style |
|---|---|
| Background | `linear-gradient(135deg, #0f172a → #0c1222)` |
| Cards | Dark glass gradient with `rgba()` overlay, subtle top border on hover |
| Hover state | `-4px` lift, indigo glow border, `-shadow-xl` with indigo tint |
| Tags | Pill-shaped with color coding by type (blue=parametric, emerald=nonparam, etc.) |
| Buttons | `linear-gradient(135deg, #6366f1 → #a855f7)` with scale/arrow animation |
| Secondary apps | Slightly dimmer cards + buttons for visual hierarchy |

### Removed Files
- `statistics_assistant.py` (Streamlit-based prototype — no longer needed)

### To View
Double-click `index.html` in a browser, or serve with any static web server:
```sh
python -m http.server 8080
# Then visit http://localhost:8080
```
