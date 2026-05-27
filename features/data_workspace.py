import streamlit as st
import pandas as pd
import numpy as np
from features.builtin_datasets import (
    get_builtin_datasets,
    load_builtin_dataset,
    get_all_dataset_names,
)
from features.widgets import render_test_widget
from scipy.stats import shapiro, normaltest, kstest


def _render_normality_test_results(df, numeric_cols):
    """Run Shapiro-Wilk on each numeric column and display results."""
    from core.utils import _apa_table
    results = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) < 3:
            continue
        stat, p = shapiro(vals)
        skew = vals.skew()
        kurt = vals.kurtosis()
        verdict = "Normal" if p > 0.05 else "Non-normal"
        results.append({
            "Column": col,
            "n": len(vals),
            "W": f"{stat:.4f}",
            "p-value": f"{p:.4f}",
            "Skew": f"{skew:.3f}",
            "Kurtosis": f"{kurt:.3f}",
            "Verdict": verdict,
        })
    if results:
        _apa_table(pd.DataFrame(results), title="Normality Test (Shapiro-Wilk)")
        n_non = sum(1 for r in results if r["Verdict"] == "Non-normal")
        if n_non > 0:
            st.warning(f"{n_non} column(s) deviate from normality (p < .05). Consider non-parametric alternatives.")
        else:
            st.success("All columns appear normally distributed (p ≥ .05).")


def _build_external_data(df, organization, col_config):
    if organization == "Wide format (each group/condition in its own column)":
        cols = col_config["selected_cols"]
        n_cols = len(cols)

        if n_cols == 1:
            vals = df[cols[0]].dropna().values
            return {"mode": "uploaded", "data": {"values": vals}, "using_uploaded": True, "_format": "one_sample"}

        relation = col_config.get("relation")
        if relation == "Paired (same subjects, different conditions)":
            vals_list = [df[c].dropna().values for c in cols]
            return {"mode": "uploaded", "data": {"measurements": vals_list, "col_names": list(cols)}, "using_uploaded": True, "_format": "repeated"}

        if relation == "Correlation / Regression (two variables)":
            return {"mode": "uploaded", "data": {"x": df[cols[0]].dropna().values, "y": df[cols[1]].dropna().values, "col_names": list(cols)}, "using_uploaded": True, "_format": "correlation"}

        vals_list = [df[c].dropna().values for c in cols]
        if n_cols == 2:
            return {"mode": "uploaded", "data": {"group1": vals_list[0], "group2": vals_list[1], "group_names": list(cols)}, "using_uploaded": True, "_format": "two_sample"}
        else:
            return {"mode": "uploaded", "data": {"groups": vals_list, "group_names": list(cols)}, "using_uploaded": True, "_format": "multi_sample"}

    elif organization == "Long format (one value column + one group column)":
        value_col = col_config["value_col"]
        group_col = col_config["group_col"]
        grouped = df.groupby(group_col)[value_col]
        group_names = list(grouped.groups.keys())
        vals_list = [g.dropna().values for _, g in grouped]

        n_groups = len(vals_list)
        if n_groups == 1:
            return {"mode": "uploaded", "data": {"values": vals_list[0]}, "using_uploaded": True, "_format": "one_sample"}
        elif n_groups == 2:
            return {"mode": "uploaded", "data": {"group1": vals_list[0], "group2": vals_list[1], "group_names": [str(g) for g in group_names]}, "using_uploaded": True, "_format": "two_sample"}
        else:
            return {"mode": "uploaded", "data": {"groups": vals_list, "group_names": [str(g) for g in group_names]}, "using_uploaded": True, "_format": "multi_sample"}

    elif organization == "Correlation / Regression (X and Y variables)":
        return {"mode": "uploaded", "data": {"x": df[col_config["x_col"]].dropna().values, "y": df[col_config["y_col"]].dropna().values, "col_names": [col_config["x_col"], col_config["y_col"]]}, "using_uploaded": True, "_format": "correlation"}

    elif organization == "Two categorical variables (contingency table)":
        col_a = col_config["cat_col_a"]
        col_b = col_config["cat_col_b"]
        ct = pd.crosstab(df[col_a], df[col_b])
        return {
            "mode": "uploaded", "_format": "categorical_two",
            "data": {"contingency_table": ct, "col_a": col_a, "col_b": col_b,
                     "col_a_vals": list(ct.index), "col_b_vals": list(ct.columns)},
            "using_uploaded": True,
        }

    elif organization == "Single categorical variable (frequency table)":
        col = col_config["cat_col"]
        counts = df[col].value_counts().sort_index()
        return {
            "mode": "uploaded", "_format": "categorical_one",
            "data": {"categories": list(counts.index), "counts": counts.values,
                     "col": col},
            "using_uploaded": True,
        }

    return None


def _get_compatible_tests(external_data):
    fmt = external_data.get("_format", "")
    compat = {
        "one_sample": [
            "One-sample t-test",
            "One-sample z-test",
            "One-sample Wilcoxon Signed-Rank Test",
            "Sign Test (One-sample)",
            "One-sample Poisson Rate Test",
            "Runs Test for Randomness",
            "Binomial Test",
            "Poisson Goodness-of-Fit Test",
            "Multinomial Test",
        ],
        "two_sample": [
            "Student's t-test (Independent)",
            "Welch's t-test (Independent, Unequal Variances)",
            "Mann-Whitney U Test",
            "F-Test for Two Variances",
            "Equivalence Test (TOST) - Two Independent Samples",
            "Point-Biserial Correlation",
        ],
        "multi_sample": [
            "One-way ANOVA",
            "One-way Welch ANOVA",
            "Kruskal-Wallis Test",
            "Mood's Median Test",
            "Permutation MANOVA or Non-Parametric MANOVA",
            "Two-way ANOVA",
            "MANOVA",
        ],
        "paired": [
            "Paired t-test",
            "Wilcoxon Signed-Rank Test",
            "Sign Test (Paired)",
            "Bland-Altman Analysis",
        ],
        "repeated": [
            "Friedman Test",
            "Repeated Measures ANOVA (One-way)",
            "Cochran's Q Test",
        ],
        "correlation": [
            "Pearson Correlation",
            "Spearman Rank Correlation",
            "Kendall's Tau-b",
            "Simple Linear Regression",
            "Multiple Linear Regression",
            "Point-Biserial Correlation",
            "Logistic Regression",
            "Multinomial Logistic Regression",
            "Ordinal Logistic Regression",
            "Poisson Regression",
            "Negative Binomial Regression",
        ],
        "categorical_two": [
            "Chi-Square Test of Independence",
            "Chi-Square Test",
            "Cohen's Kappa (Agreement Analysis)",
            "McNemar's Test",
            "Fisher's Exact Test",
            "Sensitivity & Specificity Analysis",
            "ROC Curve Analysis",
            "Likelihood Ratio Analysis",
            "Weighted Kappa",
            "Fleiss' Kappa",
        ],
        "categorical_one": [
            "Chi-Square Goodness-of-Fit Test",
            "One-sample Proportion Test (Binomial Test)",
            "Binomial Test",
            "Poisson Goodness-of-Fit Test",
            "Multinomial Test",
        ],
    }
    return compat.get(fmt, [])


def render_data_workspace():
    """Main render function for the unified data workspace — two-column layout."""

    if "ws_df" not in st.session_state:
        st.session_state.ws_df = None
    if "ws_dataset_name" not in st.session_state:
        st.session_state.ws_dataset_name = None
    if "ws_external_data" not in st.session_state:
        st.session_state.ws_external_data = None
    if "ws_selected_test" not in st.session_state:
        st.session_state.ws_selected_test = ""

    # ──────────────────────────────────────
    # Top bar
    # ──────────────────────────────────────
    top_col1, top_col2 = st.columns([1, 10])
    with top_col1:
        if st.button("← Back", use_container_width=True, type="secondary"):
            for k in ["ws_df", "ws_dataset_name", "ws_external_data", "ws_selected_test"]:
                st.session_state[k] = None
            st.session_state.page = "finder"
            st.rerun()
    with top_col2:
        st.title("Data Workspace — Import & Analyze")

    left, right = st.columns([1.1, 2], gap="large")

    # ==========================================
    # LEFT COLUMN — Phase 1: Data source
    # ==========================================
    with left:
        st.subheader("1. Data Source")

        src_mode = st.radio(
            "Choose a data source",
            ["Upload", "Built-in"],
            horizontal=True,
            key="ws_src_mode",
        )

        df = None
        dataset_name = None

        if "Upload" in src_mode:
            uploaded = st.file_uploader(
                "Upload CSV or Excel",
                type=["csv", "xlsx", "xls"],
                key="workspace_file",
            )
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                    st.success(f"Loaded **{uploaded.name}** — {len(df)} rows, {len(df.columns)} cols")
                    dataset_name = uploaded.name
                except Exception as e:
                    st.error(f"Error: {e}")
                    df = None
        else:
            all_ds = get_all_dataset_names()
            selected_ds = st.selectbox(
                "Choose a built-in dataset",
                [""] + all_ds,
                key="workspace_ds",
            )
            if selected_ds:
                info = get_builtin_datasets().get(selected_ds)
                if info:
                    with st.expander("About this dataset", expanded=True):
                        st.markdown(f"**{info['source']}**")
                        st.markdown(info.get("description", ""))
                        st.caption(f"Compatible: {', '.join(info['test_types'][:4])}{'...' if len(info['test_types']) > 4 else ''}")
                df = load_builtin_dataset(selected_ds)
                dataset_name = selected_ds
                st.success(f"Loaded **{selected_ds}** — {len(df)} rows, {len(df.columns)} cols")

        if df is None:
            st.info("No data loaded yet. Choose a source above.")
            return

        st.session_state.ws_df = df
        st.session_state.ws_dataset_name = dataset_name

        numeric_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    # ==========================================
    # RIGHT COLUMN — Phase 2: Preview + Normality
    # ==========================================
    with right:
        st.subheader("Data Preview")
        with st.expander("Show data table", expanded=True):
            st.dataframe(df, use_container_width=True, height=min(350, 35 * (len(df) + 1)))

        if numeric_cols:
            with st.expander("Normality Test (Shapiro-Wilk)", expanded=False):
                _render_normality_test_results(df, numeric_cols)

            with st.expander("Descriptive Statistics", expanded=False):
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # ==========================================
    # LEFT COLUMN — Phase 3: Data structure + summary + test
    # ==========================================
    with left:
        has_numeric = len(numeric_cols) > 0
        has_categorical = len(cat_cols) > 0

        st.divider()
        st.subheader("2. Data Structure")

        org_options = []
        if has_numeric:
            org_options += [
                "Wide format (each group/condition in its own column)",
                "Long format (one value column + one group column)",
            ]
            if len(numeric_cols) >= 2:
                org_options.append("Correlation / Regression (X and Y variables)")
        if has_categorical:
            if len(cat_cols) >= 2:
                org_options.append("Two categorical variables (contingency table)")
            org_options.append("Single categorical variable (frequency table)")

        if not org_options:
            st.error("Dataset has no usable columns.")
            return

        organization = st.radio("Choose the structure:", org_options, key="ws_org")

        col_config = {}

        if "Wide format" in organization:
            selected_cols = st.multiselect(
                "Select columns (each = one group/condition):",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
                key="ws_wide_cols",
            )
            if len(selected_cols) < 1:
                st.warning("Select at least one column.")
                return
            col_config["selected_cols"] = selected_cols
            if len(selected_cols) >= 2:
                opts = [
                    "Independent groups (different subjects per column)",
                    "Paired (same subjects, different conditions)",
                ]
                if len(selected_cols) == 2:
                    opts.append("Correlation / Regression (two variables)")
                col_config["relation"] = st.radio("Relationship:", opts, key="ws_relation")

        elif "Long format" in organization:
            if not has_categorical:
                st.warning("No group column detected. Use 'Wide format' instead.")
                return
            val_col = st.selectbox("Value column (numeric):", numeric_cols, key="ws_val")
            grp_col = st.selectbox("Group column (labels):", cat_cols, key="ws_grp")
            col_config["value_col"] = val_col
            col_config["group_col"] = grp_col
            st.caption(f"**{df[grp_col].nunique()}** group(s): {', '.join(str(g) for g in df[grp_col].unique())}")

        elif "Correlation / Regression" in organization:
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns.")
                return
            x_col = st.selectbox("X (predictor):", numeric_cols, key="ws_x")
            y_opts = [c for c in numeric_cols if c != x_col]
            y_col = st.selectbox("Y (outcome):", y_opts or numeric_cols, index=0, key="ws_y")
            col_config["x_col"] = x_col
            col_config["y_col"] = y_col

        elif "Two categorical variables" in organization:
            opts = cat_cols if len(cat_cols) >= 2 else numeric_cols
            col_a = st.selectbox("First variable (rows):", opts, key="ws_cat_a")
            col_b_opts = [c for c in opts if c != col_a]
            col_b = st.selectbox("Second variable (columns):", col_b_opts or opts, key="ws_cat_b")
            col_config["cat_col_a"] = col_a
            col_config["cat_col_b"] = col_b

        elif "Single categorical variable" in organization:
            col_opts = cat_cols if has_categorical else numeric_cols
            col = st.selectbox("Categorical variable:", col_opts, key="ws_cat_one")
            col_config["cat_col"] = col

        # ── Build & show format ──
        external_data = _build_external_data(df, organization, col_config)

        if external_data is None:
            st.error("Could not determine data format. Try a different structure.")
            return

        fmt = external_data.get("_format", "")
        fmt_labels = {
            "one_sample": "One-sample",
            "two_sample": "Two-sample (independent)",
            "multi_sample": "Multi-sample (3+ groups)",
            "paired": "Paired / Dependent",
            "repeated": "Repeated measures (3+ conditions)",
            "correlation": "Correlation / Regression",
            "categorical_two": "Contingency table (two categorical variables)",
            "categorical_one": "Frequency table (single categorical variable)",
        }
        st.info(f"Format: **{fmt_labels.get(fmt, fmt)}**")
        st.session_state.ws_external_data = external_data

        # ── Data summary ──
        st.divider()
        st.subheader("3. Data Summary")

        if fmt == "one_sample":
            v = external_data["data"]["values"]
            st.metric("n", len(v))
            st.metric("Mean", f"{np.mean(v):.3f}")
            st.metric("SD", f"{np.std(v, ddof=1):.3f}")
        elif fmt in ("two_sample", "paired"):
            if fmt == "two_sample":
                g1, g2 = external_data["data"]["group1"], external_data["data"]["group2"]
                names = external_data["data"]["group_names"]
            else:
                g1, g2 = external_data["data"]["values1"], external_data["data"]["values2"]
                names = external_data["data"]["col_names"]
            st.table(pd.DataFrame({"Group": names, "n": [len(g1), len(g2)],
                                    "Mean": [f"{np.mean(g1):.3f}", f"{np.mean(g2):.3f}"],
                                    "SD": [f"{np.std(g1, ddof=1):.3f}", f"{np.std(g2, ddof=1):.3f}"]}))
        elif fmt == "multi_sample":
            groups = external_data["data"]["groups"]
            names = external_data["data"]["group_names"]
            st.table(pd.DataFrame({"Group": names, "n": [len(g) for g in groups],
                                    "Mean": [f"{np.mean(g):.3f}" for g in groups],
                                    "SD": [f"{np.std(g, ddof=1):.3f}" for g in groups]}))
        elif fmt == "repeated":
            col_names = external_data["data"].get("col_names", [])
            measurements = external_data["data"]["measurements"]
            st.table(pd.DataFrame({"Condition": col_names, "n": [len(m) for m in measurements],
                                    "Mean": [f"{np.mean(m):.3f}" for m in measurements],
                                    "SD": [f"{np.std(m, ddof=1):.3f}" for m in measurements]}))
        elif fmt == "correlation":
            x, y = external_data["data"]["x"], external_data["data"]["y"]
            cn = external_data["data"]["col_names"]
            n_numeric_summary = pd.DataFrame({"Variable": cn, "n": [len(x), len(y)],
                                              "Mean": [f"{np.mean(x):.3f}", f"{np.mean(y):.3f}"],
                                              "SD": [f"{np.std(x, ddof=1):.3f}", f"{np.std(y, ddof=1):.3f}"]})
            st.table(n_numeric_summary)
        elif fmt == "categorical_two":
            ct = external_data["data"]["contingency_table"]
            st.write(f"**{external_data['data']['col_a']}** (rows) **x** **{external_data['data']['col_b']}** (columns)")
            st.dataframe(ct, use_container_width=True)
            st.caption(f"Total count: {ct.values.sum()}")
        elif fmt == "categorical_one":
            st.write(f"Variable: **{external_data['data']['col']}**")
            freq_df = pd.DataFrame({"Category": external_data["data"]["categories"],
                                     "Count": external_data["data"]["counts"]})
            st.table(freq_df)
            st.caption(f"Total: {freq_df['Count'].sum()}")

        # ── Test selection ──
        st.divider()
        st.subheader("4. Select Test")

        compatible = _get_compatible_tests(external_data)
        if not compatible:
            st.warning("No tests compatible with this data format.")
            return

        selected_test = st.selectbox(
            f"Available tests ({len(compatible)}):",
            [""] + compatible,
            key="workspace_test",
        )

        if not selected_test:
            st.info("Select a test above to run the analysis.")
            return

        st.session_state.ws_selected_test = selected_test

        if dataset_name:
            st.caption(f"   **{dataset_name}**  \u2192  **{selected_test}**")
        else:
            st.caption(f"Uploaded data  \u2192  **{selected_test}**")

    # ==========================================
    # RIGHT COLUMN — Phase 4: Results
    # ==========================================
    with right:
        df_right = st.session_state.ws_df
        dataset_name_right = st.session_state.ws_dataset_name
        external_data_right = st.session_state.ws_external_data
        selected_test_right = st.session_state.ws_selected_test

        if selected_test_right and external_data_right:
            st.divider()
            st.subheader("Results")

            if dataset_name_right:
                st.info(f"**{dataset_name_right}**  \u2192  **{selected_test_right}**")
            else:
                st.info(f"Uploaded data  \u2192  **{selected_test_right}**")

            fmt_right = external_data_right.get("_format", "")
            cat_formats = ("categorical_two", "categorical_one")

            if fmt_right in cat_formats:
                st.warning(
                    "This test uses built-in controls (sliders/number inputs) for educational purposes. "
                    "Your data preview and summary above show the actual contingency table \u2014 "
                    "adjust the values in the test widget below to match if desired.",
                )

            with st.spinner(f"Running {selected_test_right}..."):
                try:
                    render_test_widget(selected_test_right, external_data=external_data_right)
                except Exception as e:
                    st.error(f"Error running test: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")
