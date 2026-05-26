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
```
