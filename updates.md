# Updates.md — Statistics WebApp Changelog

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

       7     40+    80+
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
