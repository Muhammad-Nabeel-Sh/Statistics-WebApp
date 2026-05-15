# Statistical Test Finder & Tabulation WebApp

A comprehensive Streamlit-based platform for statistical test selection, interactive visualization, and tabulation/cross-tabulation analysis. Designed for researchers, students, and data scientists.

## Modes

### Test Finder
Rule-based engine matching study design inputs (objective, variables, groups, relation, distribution) to 30+ statistical tests with explanations, examples, and LaTeX formulas. Includes interactive widgets with real-time parameter sliders for most tests.

### Graph Explorer
Interactive plot generation with educational guidance:
- **Distribution Plots**: Histogram, KDE, Boxplot, Violin, Q-Q
- **Comparison Plots**: Grouped Bar, Error Bar, Paired Line, Boxplot Comparison, Violin Comparison
- **Correlation Plots**: Scatterplot, Correlation Heatmap, Bubble Plot, Monotonic vs Linear
- **Regression Plots**: Linear Regression, Multiple Regression Surface, Logistic Sigmoid

Each plot includes interpretation notes, when-to-use guidance, associated tests, and common mistakes.

### Tabulation & Cross Tabulation
Eight comprehensive sections:

1. **Descriptive Tabulation** — Frequency, Relative Frequency, Cumulative Frequency tables; Descriptive Statistics (mean, median, skewness, kurtosis, CI); Grouped Summary; Pivot Tables; Distribution Summary (percentiles)

2. **Cross-Tabulation** — 2×2 and RxC contingency tables with χ², Cramér's V, Odds Ratio; Proportion/Row%/Col% tables; Marginal Totals; Expected Frequency tables with observed vs expected comparison and cell-wise χ² contribution breakdown

3. **Diagnostic Accuracy** — Confusion matrix; Sensitivity, Specificity, PPV, NPV, LR+/LR−; Bayesian updating from pre-test to post-test probability

4. **Agreement Tables** — Observer Agreement Matrix; Cohen's Kappa with SE/z/p and Landis & Koch interpretation; ICC Reliability

5. **Regression Summary Tables** — Coefficient tables with forest plots; Odds Ratio tables; Model Fit (R², AIC, BIC); ANOVA decomposition; Residual diagnostics with Shapiro-Wilk test

6. **Effect Size Tables** — Cohen's d with overlapping distribution plot; η²/Eta-squared; Cramér's V; Odds Ratio with RR/ARR/RRR/NNT; Relative Risk

7. **Educational Modules** — Interactive Frequency Explorer; Cross-Tabulation Explorer; Expected Frequency Explorer (cell-wise χ² contributions); OR/RR/RRR/NNT Explorer; Conditional Probability Explorer; Bayesian Updating Table

8. **APA/Journal Export** — Generate publication-ready tables (Descriptive, Correlation Matrix, Regression, ANOVA, Contingency, Effect Sizes) with configurable decimals, significance stars, CI, and CSV download

### Sample Size Estimator
Power analysis and sample size estimation for 25+ analysis types including means, proportions, ANOVA, regression, survival, equivalence, ROC, kappa, cluster RCT, and simulation-based Monte Carlo power. Includes effect size converter, dropout/multiple-testing adjustments, and budget feasibility.

## Installation

```bash
pip install streamlit numpy pandas plotly scipy statsmodels scikit-learn
streamlit run app.py
```

## Project Structure

- `app.py` — Main application with UI, matching engine, and routing
- `data.py` — Test rule definitions and mappings
- `widgets.py` — Interactive test widgets with real-time controls
- `graph_explorer.py` — Graph Explorer section with all plot widgets
- `tabulation.py` — Tabulation & Cross Tabulation section (8 modules)
- `power_calculator.py` — Sample size and power calculator
- `flowchart.py` — Interactive decision tree for test selection
- `glossary.py` — Statistical terms glossary sidebar
- `matching.py` — Rule-based test matching logic
