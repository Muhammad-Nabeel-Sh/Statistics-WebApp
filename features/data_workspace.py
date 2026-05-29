import streamlit as st
import pandas as pd
import numpy as np
from features.builtin_datasets import (
    get_builtin_datasets,
    load_builtin_dataset,
    get_all_dataset_names,
)
from features.widgets import render_test_widget
from core.models import ExternalData, is_using_external
from scipy.stats import shapiro



def _render_normality_test_results(df, numeric_cols):
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
            st.success("All columns appear normally distributed (p \u2265 .05).")


def _render_data_editor(df):
    """Display editable data table using st.data_editor with typed column config."""
    from streamlit.column_config import NumberColumn, TextColumn, Column

    # Coerce bool-with-NA columns to object (pandas bool can't hold NaN)
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]) and df[c].isna().any():
            df[c] = df[c].astype(object)

    col_config = {}
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            col_config[c] = Column(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            col_config[c] = NumberColumn(c, format="%.4f" if "float" in str(df[c].dtype) else None)
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            col_config[c] = Column(c)
        else:
            col_config[c] = TextColumn(c)

    edited = st.data_editor(
        df,
        column_config=col_config,
        use_container_width=True,
        height=min(500, 35 * (len(df) + 1)),
        num_rows="dynamic",
        hide_index=True,
        key="ws_data_editor",
    )
    return edited


def _render_categorical_mapping(df):
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    if not cat_cols:
        st.info("No categorical columns to map.")
        return df

    st.markdown("### Categorical → Numeric Mapping")
    map_col = st.selectbox("Column to map", [""] + cat_cols, key="ws_map_col")
    if not map_col:
        return df

    strategy = st.radio("Mapping strategy", ["Manual mapping", "Ordinal encode (rank)", "One-hot encode"],
                        horizontal=True, key="ws_map_strategy")

    result_df = df.copy()

    if strategy == "Ordinal encode (rank)":
        new_col = map_col + "_rank"
        if new_col in result_df.columns:
            st.info(f"Column **{new_col}** already exists.")
            return df
        mapping = {cat: i for i, cat in enumerate(result_df[map_col].unique())}
        result_df[new_col] = result_df[map_col].map(mapping)
        mapping_df = pd.DataFrame({"Category": list(mapping.keys()), "Rank": list(mapping.values())})
        st.table(mapping_df)
        st.success(f"Created column: **{new_col}**")
        return result_df

    elif strategy == "One-hot encode":
        dummies = pd.get_dummies(result_df[map_col], prefix=map_col)
        new_cols = [c for c in dummies.columns if c not in result_df.columns]
        if not new_cols:
            st.info(f"Dummy columns for **{map_col}** already exist.")
            return df
        result_df = pd.concat([result_df, dummies[new_cols]], axis=1)
        st.success(f"Created {len(new_cols)} dummy columns from **{map_col}**")
        return result_df

    else:
        new_col = map_col + "_mapped"
        if new_col in result_df.columns:
            st.info(f"Column **{new_col}** already exists.")
            return df
        unique_vals = sorted(result_df[map_col].unique())
        mapping_vals = {}
        st.caption("Enter a numeric value for each category:")
        for i, cat in enumerate(unique_vals):
            mapping_vals[cat] = st.number_input(
                f"\"{cat}\" \u2192", value=float(i), key=f"ws_map_{map_col}_{cat}")
        if st.button("Apply Mapping", key="ws_map_apply", type="primary"):
            result_df[new_col] = result_df[map_col].map(mapping_vals)
            st.success(f"Created column: **{new_col}**")
            return result_df

    return df


def _build_external_data(df, organization, col_config):
    ext = ExternalData.simulated()
    if organization == "Wide format (each group/condition in its own column)":
        cols = col_config["selected_cols"]
        n_cols = len(cols)

        relation = col_config.get("relation")
        if relation == "Paired (same subjects, different conditions)":
            vals_list = [df[c].dropna().values for c in cols]
            ext = ExternalData.from_format("repeated", {"measurements": vals_list, "col_names": list(cols)})
        elif relation == "Correlation / Regression (two variables)":
            ext = ExternalData.from_format("correlation", {"x": df[cols[0]].dropna().values, "y": df[cols[1]].dropna().values, "col_names": list(cols)})
        elif n_cols == 1:
            vals = df[cols[0]].dropna().values
            ext = ExternalData.from_format("one_sample", {"values": vals})
        elif n_cols == 2:
            vals_list = [df[c].dropna().values for c in cols]
            ext = ExternalData.from_format("two_sample", {"group1": vals_list[0], "group2": vals_list[1], "group_names": list(cols)})
        else:
            vals_list = [df[c].dropna().values for c in cols]
            ext = ExternalData.from_format("multi_sample", {"groups": vals_list, "group_names": list(cols)})

    elif organization == "Long format (one value column + one group column)":
        value_col = col_config["value_col"]
        group_col = col_config["group_col"]
        grouped = df.groupby(group_col)[value_col]
        group_names = list(grouped.groups.keys())
        vals_list = [g.dropna().values for _, g in grouped]

        n_groups = len(vals_list)
        if n_groups == 1:
            ext = ExternalData.from_format("one_sample", {"values": vals_list[0]})
        elif n_groups == 2:
            ext = ExternalData.from_format("two_sample", {"group1": vals_list[0], "group2": vals_list[1], "group_names": [str(g) for g in group_names]})
        else:
            ext = ExternalData.from_format("multi_sample", {"groups": vals_list, "group_names": [str(g) for g in group_names]})

    elif organization == "Correlation / Regression (X and Y variables)":
        ext = ExternalData.from_format("correlation", {"x": df[col_config["x_col"]].dropna().values, "y": df[col_config["y_col"]].dropna().values, "col_names": [col_config["x_col"], col_config["y_col"]]})

    elif organization == "Two categorical variables (contingency table)":
        col_a = col_config["cat_col_a"]
        col_b = col_config["cat_col_b"]
        ct = pd.crosstab(df[col_a], df[col_b])
        ext = ExternalData.from_format("categorical_two", {"contingency_table": ct, "col_a": col_a, "col_b": col_b, "col_a_vals": list(ct.index), "col_b_vals": list(ct.columns)})

    elif organization == "Single categorical variable (frequency table)":
        col = col_config["cat_col"]
        counts = df[col].value_counts().sort_index()
        ext = ExternalData.from_format("categorical_one", {"categories": list(counts.index), "counts": counts.values, "col": col})

    return ext if ext.using_uploaded else None


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
    """Main render function for the unified data workspace — two-column layout with AG Grid."""

    if "ws_df" not in st.session_state:
        st.session_state.ws_df = None
    if "ws_dataset_name" not in st.session_state:
        st.session_state.ws_dataset_name = None
    if "ws_external_data" not in st.session_state:
        st.session_state.ws_external_data = None
    if "ws_selected_test" not in st.session_state:
        st.session_state.ws_selected_test = ""

    st.title("Data Workspace")
    st.caption("Import, explore, clean, and analyze data — with interactive editing and filtering.")

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

        new_df = None
        new_name = None

        if "Upload" in src_mode:
            uploaded = st.file_uploader(
                "Upload CSV or Excel",
                type=["csv", "xlsx", "xls"],
                key="workspace_file",
            )
            if uploaded is not None:
                try:
                    new_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                    new_name = uploaded.name
                except Exception as e:
                    st.error(f"Error: {e}")
                    new_df = None
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
                new_df = load_builtin_dataset(selected_ds)
                new_name = selected_ds

        # Build a source key to detect when user picks a different dataset
        new_source_key = f"{src_mode}:{new_name or ''}"
        prev_source_key = st.session_state.get("ws_source_key", "")

        if new_df is not None and new_source_key != prev_source_key:
            # Fresh data load — store and initialize
            st.session_state.ws_df = new_df
            st.session_state.ws_dataset_name = new_name
            st.session_state.ws_source_key = new_source_key
            st.success(f"Loaded **{new_name}** — {len(new_df)} rows, {len(new_df.columns)} cols")

        if st.session_state.ws_df is None:
            st.info("No data loaded yet. Choose a source above.")
            return

        # Always work from session state downstream
        df = st.session_state.ws_df
        dataset_name = st.session_state.ws_dataset_name

        numeric_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    # ==========================================
    # RIGHT COLUMN — Phase 2: Data editor + Cleaning
    # ==========================================
    with right:
        st.subheader("Data Preview")
        edited = _render_data_editor(df)

        if edited is not None:
            try:
                if not edited.equals(st.session_state.ws_df):
                    st.session_state.ws_df = edited
                    st.rerun()
            except Exception:
                pass

        with st.expander("Data Cleaning & Transformation", expanded=False):
            tc1, tc2 = st.columns(2)

            with tc1:
                st.markdown("**Filter Rows**")
                if numeric_cols:
                    filt_col = st.selectbox("Column", numeric_cols, key="ws_filt_col")
                    min_v = float(df[filt_col].min())
                    max_v = float(df[filt_col].max())
                    filt_range = st.slider("Range", min_v, max_v, (min_v, max_v), key="ws_filt_range")
                    if filt_range != (min_v, max_v):
                        df = df[(df[filt_col] >= filt_range[0]) & (df[filt_col] <= filt_range[1])]
                        st.caption(f"Filtered to {len(df)} rows")

            with tc2:
                st.markdown("**Drop Columns**")
                drop_cols = st.multiselect("Select columns to remove", df.columns.tolist(), key="ws_drop_cols")
                if drop_cols:
                    if st.button("Apply Drop", key="ws_drop_apply"):
                        df = df.drop(columns=drop_cols)
                        st.session_state.ws_df = df
                        st.rerun()

            st.markdown("**Handle Missing Values**")
            missing_method = st.selectbox("Method", ["Drop rows with any NA", "Drop rows with all NA",
                                                      "Fill with column mean (numeric)", "Fill with 0"], key="ws_na_method")
            if st.button("Apply", key="ws_na_apply"):
                if missing_method == "Drop rows with any NA":
                    prev = len(df)
                    df = df.dropna()
                    st.session_state.ws_df = df
                    st.toast(f"Dropped {prev - len(df)} rows with missing values.", icon="🗑️")
                    st.rerun()
                elif missing_method == "Drop rows with all NA":
                    prev = len(df)
                    df = df.dropna(how="all")
                    st.session_state.ws_df = df
                    st.toast(f"Dropped {prev - len(df)} fully-empty rows.", icon="🗑️")
                    st.rerun()
                elif "mean" in missing_method:
                    num_df = df.select_dtypes(include=["float64", "int64"])
                    df[num_df.columns] = df[num_df.columns].fillna(num_df.mean())
                    st.session_state.ws_df = df
                    st.toast("Filled numeric NAs with column means.", icon="📊")
                    st.rerun()
                else:
                    df = df.fillna(0)
                    st.session_state.ws_df = df
                    st.toast("Filled all NAs with 0.", icon="0️⃣")
                    st.rerun()

            st.markdown("**Computed Column**")
            col_name = st.text_input("New column name", placeholder="e.g. log_x", key="ws_comp_name")
            expression = st.text_input(
                "Expression (use `col` for column values)",
                placeholder="e.g. col * 2 + 1  or  np.log(col)",
                key="ws_comp_expr",
            )
            comp_col = st.selectbox("Source column", [""] + numeric_cols, key="ws_comp_source")
            if col_name and expression and comp_col:
                if st.button("Create Column", key="ws_comp_apply", type="primary"):
                    try:
                        col = df[comp_col].values
                        result = eval(expression, {"np": np, "pd": pd, "col": col, "df": df})
                        df[col_name] = result
                        st.session_state.ws_df = df
                        st.rerun()
                    except Exception as e:
                        st.error(f"Expression error: {e}")

        # Categorical mapping
        mapped = _render_categorical_mapping(df)
        if mapped is not None and isinstance(mapped, pd.DataFrame) and len(mapped.columns) > len(df.columns):
            st.session_state.ws_df = mapped
            st.rerun()

        # Normality + Descriptives
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

        # Build & show format
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

        # Data summary
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

        # Test selection
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
