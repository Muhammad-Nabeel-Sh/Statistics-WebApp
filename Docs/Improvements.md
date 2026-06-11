# Comprehensive Analysis & Improvement Plan for Statistics WebApp

After a thorough review of the `apps`, `core`, and `features` directories of the Statistics WebApp, I have identified several key areas for improvement. While the application currently serves as an excellent educational and analytical tool, scaling it further requires architectural refactoring, UI/UX enhancements, and the addition of robust data engineering and advanced statistical features.

Here is a comprehensive breakdown of what can be improved or added to the application.

---

## 1. Architectural & Codebase Improvements

Currently, the application suffers from monolithic files and hard-coded data structures, which hinder maintainability and scalability.

❌ ~~**Refactor Monolithic Files:**~~ ✅ **COMPLETED (June 2026)**
    *   ~~`features/widgets.py` is over 7,000 lines and 250 KB. It currently acts as a massive container for every statistical test widget and rendering logic. This should be broken down into a `features/widgets/` directory with separate modules for each statistical family (e.g., `parametric.py`, `nonparametric.py`, `regression.py`, `categorical.py`).~~
    *   **Done**: `features/widgets.py` → `features/widgets/` package with 9 modules, registry pattern, decorator-based registration. Total: 71 registered widgets.
    *   ~~`features/graph_explorer.py` is similarly massive (~8000 lines). It should be split into individual plot types (e.g., `comparisons.py`, `distributions.py`... etc.).~~
    *   **Remaining**: Graph explorer (~7469 lines, 12 submodules) is still monolithic — left as-is for now.
*   ❌ ~~**Decouple Static Data from Code:**~~ ✅ **COMPLETED (June 2026)**
    *   ~~`core/data.py` contains over 1,000 lines of hardcoded dictionaries (`rules`) defining tests, formulas, explanations, and mappings. This data should be moved out of Python files and into structured JSON or YAML files (e.g., `assets/test_definitions.json`).~~
    *   **Done**: All 71 test definitions moved to `assets/test_definitions.json`. `core/data.py` remains for runtime utilities only (`rules` list, `TEST_TO_SS_TYPE` mapping, `FIELDS`).
*   ❌ ~~**Implement a Dynamic Registry Pattern:**~~ ✅ **COMPLETED (June 2026)**
    *   ~~Currently, the application likely relies on giant `if/elif` chains or static dictionary mapping to render specific tests.~~
    *   **Done**: `@register_widget("Test Name")` decorator in `features/widgets/__init__.py`. Widgets auto-register into `_widget_registry` dict. Access via `get_widget("Test Name")`.
*   **Introduce Strict Typing and Data Classes:** 
    *   The `external_data` dictionary passed between the Data Workspace and Widgets relies on loose string keys (`{"mode": "uploaded", "_format": "one_sample", ...}`). Introducing `Pydantic` models or Python `dataclasses` would enforce strict type-checking, reducing runtime `KeyError` issues and making the code self-documenting.
    *   **Status**: Not yet implemented. Partial mitigation: `_build_external_data()` helper standardizes dict creation, but still no formal validation.

---

## 2. Testing and Engineering Practices

As noted, the project currently lacks a testing framework, linters, and type checkers.

*   **Automated Testing Suite:** Introduce `pytest`. Given the mathematical nature of the app, it is critical to have unit tests that validate the outputs of the custom statistical functions (e.g., Cohen's d, exact binomials) against known benchmarks from `R` or `Minitab`.
*   **Dependency Management:** Add a `requirements.txt` or `pyproject.toml` (using Poetry or uv) to lock down dependency versions for `streamlit`, `pandas`, `scipy`, `statsmodels`, and `plotly`. This ensures the app doesn't break when a library updates.
*   **Linting & Formatting:** Implement `ruff` or `flake8` for linting, and `black` for formatting to maintain a consistent code style across the large codebase.

---

## 3. UI/UX and Interactivity Enhancements

While Streamlit provides a great rapid-prototyping interface, the app's UX can be optimized to feel more like a professional SaaS product.

*   **Interactive Data Tables (AG Grid):** 
    *   Currently, tables are rendered using simple `st.dataframe` or `pandas.Styler` (via `_apa_table`). Replacing these with `streamlit-aggrid` would allow users to sort, filter, paginate, and directly export the data from the tables themselves.
*   **Consolidate Educational Content:** 
    *   The app makes heavy use of `st.expander` and `st.info` for mathematical formulas and explanations. While educational, this can clutter the analytical workflow. A better approach would be to implement a dedicated "Learn" tab alongside the "Analysis" tab, or use side-panel drawers to display the theoretical context dynamically without interrupting the UI.
*   **Step-by-Step Wizard for Data Workspace:** 
    *   `features/data_workspace.py` relies on a long continuous column for data loading, structuring, summarizing, and testing. Implementing a multi-step wizard UI (using `st.stepper` or session state-based steps) would guide the user more intuitively: *Step 1: Upload -> Step 2: Clean/Map -> Step 3: Select Test -> Step 4: Results*.
*   **Global Theme and Custom CSS:** 
    *   Although `plotly_dark` is used for charts, injecting a cohesive global CSS file to style Streamlit containers, buttons, and metrics would give the app a more distinct, premium feel.

---

## 4. Data Management & Preprocessing (Data Workspace)

The current `data_workspace.py` is capable but rigid. It expects pristine data and strict mappings (Long vs. Wide).

*   **Data Wrangling & Cleaning Tools:** 
    *   Add a preprocessing step where users can handle missing values (drop rows, impute with mean/median), detect and remove outliers, and filter specific rows (e.g., "Age > 18").
*   **Dynamic Data Type Casting:** 
    *   The app currently infers data types automatically via `select_dtypes()`. It should allow users to explicitly cast a column (e.g., converting a numeric binary column like `[0, 1]` to a categorical column like `["Control", "Treatment"]`).
*   **Data Transformations:** 
    *   Allow users to apply mathematical transformations (Log, Square Root, Box-Cox, Standardization) directly in the workspace before passing the data to the statistical tests to help resolve normality assumption violations.

---

## 5. New Statistical Features & Additions

To elevate the app from a basic statistical calculator to a comprehensive data science platform:

*   **Advanced Predictive Modeling (Machine Learning):** 
    *   Add algorithms like Random Forests, Support Vector Machines (SVM), and Gradient Boosting. 
    *   Include clustering techniques like K-Means, Hierarchical Clustering, and DBSCAN (currently, only PCA/FA is covered in `factor_analysis.py`).
*   **Time Series Analysis:** 
    *   Introduce a new module for Time Series forecasting, including ARIMA, SARIMA, Exponential Smoothing, and Stationarity tests (Dickey-Fuller).
*   **Survival Analysis Dashboard:** 
    *   Expand the current Kaplan-Meier/Cox Regression implementations into a dedicated UI feature that handles censoring indicators, time-to-event curves, and hazard ratios comprehensively.
*   **Bayesian Statistics Alternative:** 
    *   For every frequentist test (e.g., t-test, ANOVA), offer a "Bayesian Alternative" toggle that calculates Bayes Factors using PyMC or specialized scipy integrations, showing the probability of the alternative hypothesis vs. the null.
*   **Advanced Diagnostics Dashboard:** 
    *   While `diagnostics.py` exists, integrating automatic Assumption Checks (Normality, Homoscedasticity) *before* running a test—and suggesting the non-parametric alternative if assumptions fail—would make the app incredibly smart and educational.

---

## 6. Reporting and Exporting

Currently, the app relies on Plotly's `st_plot_with_download` for PNG/SVG exports.

*   **Full Report Generation (PDF/HTML):** 
    *   Implement an "Export Full Report" button that aggregates the dataset summary, statistical test results, APA tables, and Plotly charts into a formatted PDF (using `pdfkit` or `WeasyPrint`) or an HTML file.
*   **Reproducible Code Export:** 
    *   Add a feature that generates and exports the actual `Python/Pandas/Scipy` script used to run the user's selected analysis. This teaches the user how to perform the analysis in code and ensures full reproducibility.
