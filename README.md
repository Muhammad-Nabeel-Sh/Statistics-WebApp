# Statistical Test Finder & Analysis WebApp

A comprehensive Streamlit-based platform featuring statistical test selection with interactive widgets, data import workspace, graph explorer, tabulation, and power analysis. Designed for researchers, students, and data scientists.

## Apps

The monolith has been refactored into **7 independent mini-apps**, each deployable separately:

| App | Command |
|-----|---------|
| **Test Finder + Data Workspace** | `streamlit run apps/app_finder.py` |
| **Graph Explorer** | `streamlit run apps/app_explorer.py` |
| **Tabulation & Cross Tabulation** | `streamlit run apps/app_tabulation.py` |
| **Distributions & CLT Simulator** | `streamlit run apps/app_distributions.py` |
| **Power & Sample Size Calculator** | `streamlit run apps/app_power.py` |
| **Diagnostic Test Analysis** | `streamlit run apps/app_diagnostics.py` |
| **Solved Examples** | `streamlit run apps/app_examples.py` |

## Features

### Test Finder
Rule-based engine matching study design inputs (objective, variables, groups, relation, distribution) to 40+ statistical tests with explanations, examples, and LaTeX formulas. Includes interactive widgets with real-time parameter sliders for each test.

### Data Workspace
Unified data import and analysis environment (two-column layout):
- **Upload CSV/Excel** or pick from **20 built-in open-source datasets** (Iris, Penguins, PlantGrowth, Tips, Cars, mtcars, Titanic, etc.)
- Auto-detect data structure: wide, long, correlation, or categorical formats
- Contingency table builder for categorical variables
- Compatible tests filtered by data format
- Runs selected test using uploaded data via `external_data` injection

### Graph Explorer
Interactive plot generation with educational guidance:
- Distribution, comparison, correlation, and regression plots
- Interpretation notes, when-to-use guidance, and common mistakes

### Tabulation & Cross Tabulation
Eight comprehensive sections: descriptive statistics, cross-tabulation with chi-square and Cramer's V, diagnostic accuracy (sensitivity/specificity/LR), agreement tables with Cohen's Kappa, regression summary tables, effect sizes (Cohen's d, eta-squared), interactive educational modules, and APA/journal export.

### Power & Sample Size Calculator
Power analysis for 25+ analysis types including means, proportions, ANOVA, regression, survival, equivalence, ROC, and cluster RCT. Includes effect size converter, dropout/multiple-testing adjustments, and Monte Carlo simulation.

## Installation

```bash
pip install streamlit numpy pandas plotly scipy statsmodels scikit-learn
streamlit run apps/app_finder.py
```

## Project Structure

```
apps/                     # Standalone entry points (one per app)
  app_finder.py           #   Finder + Data Workspace
  app_tabulation.py       #   Tabulation
  app_explorer.py         #   Graph Explorer
  app_distributions.py    #   Distributions
  app_power.py            #   Power calculator
  app_diagnostics.py      #   Diagnostics
  app_examples.py         #   Solved examples

features/                 # UI modules
  widgets.py              #   40+ interactive test widgets
  finder_ui.py            #   Test finder UI + all-tests browser
  data_workspace.py       #   Data import/workspace (two-column)
  builtin_datasets.py     #   20 curated open-source datasets
  graph_explorer.py       #   Graph explorer
  tabulation.py           #   Tabulation (8 sections)
  distributions.py        #   Distributions + CLT Simulator
  glossary.py             #   Statistical glossary
  power_calculator.py     #   Power analysis
  diagnostics.py          #   Diagnostic test analysis
  solved_examples.py      #   Educational examples

core/                     # Shared utilities
  data.py                 #   Test rule definitions
  matching.py             #   Rule-based test matching
  post_hoc.py             #   8 post-hoc methods
  utils.py                #   format_p_value, data_source_toggle, effect size helpers
```
