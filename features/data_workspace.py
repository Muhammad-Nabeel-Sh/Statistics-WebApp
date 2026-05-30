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

    df = df.copy()
    for c in df.columns:
        # Coerce bool-with-NA columns to object (pandas bool can't hold NaN)
        if pd.api.types.is_bool_dtype(df[c]) and df[c].isna().any():
            df[c] = df[c].astype(object)
        # Coerce object columns containing numeric strings to proper numeric type
        elif pd.api.types.is_object_dtype(df[c]):
            try:
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().sum() > 0:
                    df[c] = coerced
            except (ValueError, TypeError):
                pass

    col_config = {}
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            col_config[c] = Column(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            is_int = pd.api.types.is_integer_dtype(df[c])
            col_config[c] = NumberColumn(c, format=None if is_int else "%.4f")
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


@st.dialog("Expression Gallery")
def _expression_gallery_dialog():
    st.markdown("Browse expressions to use in **Computed Column** or **Transform Column**. Copy one (Ctrl+C) and paste into the expression field.")
    st.markdown("""
| Category | Expression | Description |
|---|---|---|
| **Arithmetic** | `col * 2` | Double |
| | `col + 10` | Add constant |
| | `col / col.max()` | Scale 0–1 |
| | `col ** 2` | Square |
| | `col % 2` | Modulo (remainder) |
| | `1 / col` | Reciprocal |
| | `col + col.shift(1)` | Add lag-1 value |
| **Power & Root** | `col ** 3` | Cube |
| | `np.sqrt(col)` | Square root |
| | `np.cbrt(col)` | Cube root |
| | `np.square(col)` | Square (fast) |
| | `np.power(col, 1.5)` | Power 1.5 |
| **Log / Exp** | `np.log(col)` | Natural log |
| | `np.log10(col)` | Log base 10 |
| | `np.log2(col)` | Log base 2 |
| | `np.log1p(col)` | Log(1+col), stable for small values |
| | `np.exp(col)` | Exponential |
| | `np.exp2(col)` | 2^col |
| | `np.expm1(col)` | exp(col)-1, stable for small values |
| | `np.power(10, col)` | 10^col (antilog base 10) |
| **Trigonometry** | `np.sin(col)` | Sine (radians) |
| | `np.cos(col)` | Cosine (radians) |
| | `np.tan(col)` | Tangent (radians) |
| | `np.deg2rad(col)` | Degrees to radians |
| | `np.rad2deg(col)` | Radians to degrees |
| | `np.arcsin(col)` | Arc sine |
| | `np.arccos(col)` | Arc cosine |
| | `np.arctan(col)` | Arc tangent |
| **Rounding** | `np.round(col, 0)` | Round to integer |
| | `np.floor(col)` | Round down |
| | `np.ceil(col)` | Round up |
| | `np.trunc(col)` | Truncate toward zero |
| | `np.rint(col)` | Round to nearest even |
| | `col.round(2)` | 2 decimal places |
| **Cumulative / Lag** | `np.cumsum(col)` | Cumulative sum |
| | `col.cummax()` | Cumulative max |
| | `col.cummin()` | Cumulative min |
| | `col.cumprod()` | Cumulative product |
| | `col.pct_change()` | % change from previous row |
| | `col.diff()` | First difference (lag 1) |
| | `col.shift(1)` | Lag by 1 row |
| **Statistics** | `(col-col.mean())/col.std()` | Z-score (standardize) |
| | `col.rank()` | Rank (1=smallest) |
| | `col / col.sum()` | Proportion of total |
| | `col - col.mean()` | Deviation from mean |
| | `col.rolling(3).mean()` | Rolling mean (window 3) |
| | `col.expanding().mean()` | Expanding mean |
| | `col.corr(df['b'])` | Correlation with another column |
| **Conditional** | `np.where(col>0, col, 0)` | Keep positives |
| | `np.where(col>0, 1, -1)` | Sign indicator (1/-1) |
| | `np.clip(col, 0, 100)` | Clip to range |
| | `np.sign(col)` | Sign (-1 / 0 / 1) |
| | `pd.cut(col, 5, labels=False)` | Bin into 5 equal-width groups |
| | `pd.qcut(col, 4, labels=False)` | Quartile binning |
| **Multi-column** | `df['a'] + df['b']` | Sum two columns |
| | `df['a'] * df['b']` | Multiply two columns |
| | `df['a'] / df['b']` | Ratio |
| | `df['a'] - df['b']` | Difference |
| | `df[['a','b']].mean(axis=1)` | Row-wise mean |
| | `df[['a','b']].sum(axis=1)` | Row-wise sum |
| **Special** | `np.abs(col)` | Absolute value |
| | `np.radians(col)` | Degrees to radians (alt) |
| | `np.degrees(col)` | Radians to degrees (alt) |
""")
    if st.button("Close Gallery"):
        st.rerun()


def _display_ps_results(title, params_df, stat_name, stat_val, df_val, p_val, ci_lower, ci_upper, es_name, es_val, interpretation=""):
    """Display a nicely formatted result card for the Parameter Solver."""
    from scipy.stats import t as t_dist
    st.success(f"**{title}** — Results")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Parameters**")
        st.table(params_df)
    with col_b:
        results_data = {
            stat_name: f"{stat_val:.4f}",
            "df": f"{df_val:.4f}" if df_val == int(df_val) else f"{df_val:.2f}",
            "p-value": f"{p_val:.4f}" if p_val >= 0.0001 else "p < .0001",
            "95% CI": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            es_name: f"{es_val:.4f}",
        }
        st.markdown("**Results**")
        st.table(pd.DataFrame([results_data]).T.rename(columns={0: "Value"}))
    if interpretation:
        st.caption(interpretation)


def _render_summarized_data_inputs():
    """Show input forms for summary statistics with two modes: Data Generator and Parameter Solver."""
    st.markdown(":orange[**Enter summary statistics from published research:**]")

    approach = st.radio(
        "Mode",
        ["Data Generator (synthetic data)", "Parameter Solver (exact formulas)"],
        horizontal=True,
        key="ws_summary_approach",
    )

    if "Data Generator" in approach:
        # ======================================================================
        # MODE 1: Data Generator — generate synthetic data matching summary stats
        # ======================================================================
        summary_type = st.selectbox(
            "Summary type",
            ["", "One-sample (n, Mean, SD)", "Two independent samples (n, M, SD each)",
             "Paired (n, mean difference, SD of differences)",
             "One proportion (n, k)", "Two proportions (n1, k1, n2, k2)",
             "Correlation (n, r)"],
            key="ws_summary_type",
        )

        if not summary_type:
            st.info("Select a summary type above to enter parameters.")
            return False

        _rng = np.random.default_rng(42)

        if "One-sample (n" in summary_type:
            n = st.number_input("Sample size (n)", 3, 100000, 30, 1, key="ws_dg_ss_n")
            mean = st.number_input("Mean", value=0.0, key="ws_dg_ss_mean", format="%.4f")
            sd = st.number_input("SD", 0.001, 1000000.0, 1.0, key="ws_dg_ss_sd", format="%.4f")
            if st.button("Generate Data & Run Test", key="ws_dg_ss_gen", type="primary"):
                data = _rng.normal(0, 1, int(n))
                data = (data - data.mean()) / data.std() * float(sd) + float(mean)
                df = pd.DataFrame({"Value": data})
                ext = ExternalData.from_format("one_sample", {"values": data})
                st.session_state.ws_df = df
                st.session_state.ws_df_backup = df.copy()
                st.session_state.ws_external_data = ext
                st.session_state.ws_dataset_name = f"Summarized: One-sample (n={int(n)}, m={mean:.3g}, sd={sd:.3g})"
                st.session_state.ws_source_key = f"dg:one_sample:{int(n)}:{mean}:{sd}"
                st.session_state.ws_using_summarized = True
                st.rerun()

        elif "Two independent" in summary_type:
            col1, col2 = st.columns(2)
            with col1:
                n1 = st.number_input("n₁", 3, 100000, 25, 1, key="ws_dg_ts_n1")
                m1 = st.number_input("Mean₁", value=0.0, key="ws_dg_ts_m1", format="%.4f")
                s1 = st.number_input("SD₁", 0.001, 1000000.0, 1.0, key="ws_dg_ts_s1", format="%.4f")
                g1_name = st.text_input("Group₁ label", "Group 1", key="ws_dg_ts_g1")
            with col2:
                n2 = st.number_input("n₂", 3, 100000, 25, 1, key="ws_dg_ts_n2")
                m2 = st.number_input("Mean₂", value=1.0, key="ws_dg_ts_m2", format="%.4f")
                s2 = st.number_input("SD₂", 0.001, 1000000.0, 1.0, key="ws_dg_ts_s2", format="%.4f")
                g2_name = st.text_input("Group₂ label", "Group 2", key="ws_dg_ts_g2")
            if st.button("Generate Data & Run Test", key="ws_dg_ts_gen", type="primary"):
                g1 = _rng.normal(0, 1, int(n1))
                g1 = (g1 - g1.mean()) / g1.std() * float(s1) + float(m1)
                g2 = _rng.normal(0, 1, int(n2))
                g2 = (g2 - g2.mean()) / g2.std() * float(s2) + float(m2)
                df = pd.DataFrame({g1_name: np.append(g1, [np.nan] * (max(len(g1), len(g2)) - len(g1))),
                                   g2_name: np.append(g2, [np.nan] * (max(len(g1), len(g2)) - len(g2)))})
                df = df.dropna(how="all").reset_index(drop=True)
                ext = ExternalData.from_format("two_sample", {
                    "group1": g1, "group2": g2,
                    "group_names": [g1_name or "Group 1", g2_name or "Group 2"],
                })
                st.session_state.ws_df = df
                st.session_state.ws_df_backup = df.copy()
                st.session_state.ws_external_data = ext
                st.session_state.ws_dataset_name = f"Summarized: Two-sample ({g1_name}: n={int(n1)}, m={m1:.3g}, sd={s1:.3g} | {g2_name}: n={int(n2)}, m={m2:.3g}, sd={s2:.3g})"
                st.session_state.ws_source_key = f"dg:two_sample:{int(n1)}:{m1}:{s1}:{int(n2)}:{m2}:{s2}"
                st.session_state.ws_using_summarized = True
                st.rerun()

        elif "Paired" in summary_type:
            n = st.number_input("Sample size (n)", 3, 100000, 30, 1, key="ws_dg_pd_n")
            mean_diff = st.number_input("Mean difference", value=5.0, key="ws_dg_pd_mdiff", format="%.4f")
            sd_diff = st.number_input("SD of differences", 0.001, 1000000.0, 10.0, key="ws_dg_pd_sddiff", format="%.4f")
            pre_mean = st.number_input("Pre-test mean (for display)", value=50.0, key="ws_dg_pd_pre", format="%.4f")
            pre_sd = st.number_input("Pre-test SD (for display)", 0.001, 1000000.0, 15.0, key="ws_dg_pd_presd", format="%.4f")
            if st.button("Generate Data & Run Test", key="ws_dg_pd_gen", type="primary"):
                pre = _rng.normal(0, 1, int(n))
                pre = (pre - pre.mean()) / pre.std() * float(pre_sd) + float(pre_mean)
                diff = _rng.normal(0, 1, int(n))
                diff = (diff - diff.mean()) / diff.std() * float(sd_diff) + float(mean_diff)
                post = pre + diff
                df = pd.DataFrame({"Pre": pre, "Post": post})
                ext = ExternalData.from_format("paired", {"values1": pre, "values2": post})
                st.session_state.ws_df = df
                st.session_state.ws_df_backup = df.copy()
                st.session_state.ws_external_data = ext
                st.session_state.ws_dataset_name = f"Summarized: Paired (n={int(n)}, diff={mean_diff:.3g}, sd_diff={sd_diff:.3g})"
                st.session_state.ws_source_key = f"dg:paired:{int(n)}:{mean_diff}:{sd_diff}"
                st.session_state.ws_using_summarized = True
                st.rerun()

        elif "One proportion" in summary_type:
            n = st.number_input("Total observations (n)", 1, 100000, 50, 1, key="ws_dg_op_n")
            k = st.number_input("Number of successes (k)", 0, 100000, 20, 1, key="ws_dg_op_k")
            if k > n:
                st.error("k cannot exceed n.")
                return False
            prop = k / n
            st.info(f"Proportion = {prop:.4f} ({k}/{n})")
            if st.button("Generate Data & Run Test", key="ws_dg_op_gen", type="primary"):
                values = np.array([1] * int(k) + [0] * (int(n) - int(k)))
                _rng.shuffle(values)
                df = pd.DataFrame({"Outcome": values})
                ext = ExternalData.from_format("one_sample", {"values": values})
                st.session_state.ws_df = df
                st.session_state.ws_df_backup = df.copy()
                st.session_state.ws_external_data = ext
                st.session_state.ws_dataset_name = f"Summarized: One proportion (n={int(n)}, k={int(k)}, p={prop:.4f})"
                st.session_state.ws_source_key = f"dg:one_prop:{int(n)}:{int(k)}"
                st.session_state.ws_using_summarized = True
                st.rerun()

        elif "Two proportions" in summary_type:
            col1, col2 = st.columns(2)
            with col1:
                n1 = st.number_input("n₁", 1, 100000, 50, 1, key="ws_dg_tp_n1")
                k1 = st.number_input("Successes₁ (k₁)", 0, 100000, 20, 1, key="ws_dg_tp_k1")
            with col2:
                n2 = st.number_input("n₂", 1, 100000, 50, 1, key="ws_dg_tp_n2")
                k2 = st.number_input("Successes₂ (k₂)", 0, 100000, 10, 1, key="ws_dg_tp_k2")
            if k1 > n1 or k2 > n2:
                st.error("k cannot exceed n for either group.")
                return False
            p1, p2 = k1 / n1, k2 / n2
            st.info(f"p₁ = {p1:.4f} ({k1}/{n1})  |  p₂ = {p2:.4f} ({k2}/{n2})")
            if st.button("Generate Data & Run Test", key="ws_dg_tp_gen", type="primary"):
                ct = pd.DataFrame(
                    {"Success": [int(k1), int(k2)], "Failure": [int(n1) - int(k1), int(n2) - int(k2)]},
                    index=["Group 1", "Group 2"],
                )
                df = ct.copy()
                ext = ExternalData.from_format("categorical_two", {
                    "contingency_table": ct, "col_a": "Group", "col_b": "Outcome",
                    "col_a_vals": ["Group 1", "Group 2"], "col_b_vals": ["Success", "Failure"],
                })
                st.session_state.ws_df = df
                st.session_state.ws_df_backup = df.copy()
                st.session_state.ws_external_data = ext
                st.session_state.ws_dataset_name = f"Summarized: Two proportions (n₁={int(n1)}, p₁={p1:.3g} | n₂={int(n2)}, p₂={p2:.3g})"
                st.session_state.ws_source_key = f"dg:two_prop:{int(n1)}:{int(k1)}:{int(n2)}:{int(k2)}"
                st.session_state.ws_using_summarized = True
                st.rerun()

        elif "Correlation" in summary_type:
            n = st.number_input("Sample size (n)", 3, 100000, 50, 1, key="ws_dg_cr_n")
            r_val = st.slider("Pearson r", -1.0, 1.0, 0.5, 0.01, key="ws_dg_cr_r")
            x_label = st.text_input("X variable label", "X", key="ws_dg_cr_xlabel")
            y_label = st.text_input("Y variable label", "Y", key="ws_dg_cr_ylabel")
            if st.button("Generate Data & Run Test", key="ws_dg_cr_gen", type="primary"):
                x = _rng.normal(0, 1, int(n))
                y = float(r_val) * x + np.sqrt(1 - float(r_val) ** 2) * _rng.normal(0, 1, int(n))
                df = pd.DataFrame({x_label or "X": x, y_label or "Y": y})
                ext = ExternalData.from_format("correlation", {
                    "x": x, "y": y,
                    "col_names": [x_label or "X", y_label or "Y"],
                })
                st.session_state.ws_df = df
                st.session_state.ws_df_backup = df.copy()
                st.session_state.ws_external_data = ext
                st.session_state.ws_dataset_name = f"Summarized: Correlation (n={int(n)}, r={r_val:.3g})"
                st.session_state.ws_source_key = f"dg:correlation:{int(n)}:{r_val}"
                st.session_state.ws_using_summarized = True
                st.rerun()

    else:
        # ======================================================================
        # MODE 2: Parameter Solver — exact formula computation, no data needed
        # ======================================================================
        from scipy.stats import t as t_dist, norm as norm_dist

        ps_test = st.selectbox(
            "Test type",
            ["", "One-sample t-test", "Two independent samples t-test (Welch)",
             "Two independent samples t-test (pooled)", "Paired t-test",
             "One proportion z-test", "Two proportions z-test",
             "Correlation t-test"],
            key="ws_ps_test",
        )

        if not ps_test:
            st.info("Select a test type above to enter parameters and compute exact results.")
            return False

        # ---- One-sample t-test ----
        if ps_test == "One-sample t-test":
            n = st.number_input("Sample size (n)", 3, 100000, 30, 1, key="ws_ps_os_n")
            mean = st.number_input("Sample mean", value=0.0, key="ws_ps_os_mean", format="%.4f")
            sd = st.number_input("Sample SD", 0.001, 1000000.0, 1.0, key="ws_ps_os_sd", format="%.4f")
            mu0 = st.number_input("Null hypothesis μ₀", value=0.0, key="ws_ps_os_mu0", format="%.4f")
            if st.button("Compute", key="ws_ps_os_btn", type="primary"):
                se = float(sd) / np.sqrt(int(n))
                t_stat = (float(mean) - float(mu0)) / se
                df_val = int(n) - 1
                p_val = 2 * t_dist.sf(abs(t_stat), df_val)
                ci_lo = float(mean) - t_dist.ppf(0.975, df_val) * se
                ci_hi = float(mean) + t_dist.ppf(0.975, df_val) * se
                d_val = (float(mean) - float(mu0)) / float(sd)
                params_df = pd.DataFrame({"Parameter": ["n", "Mean", "SD", "μ₀"],
                                           "Value": [int(n), f"{mean:.4f}", f"{sd:.4f}", f"{mu0:.4f}"]})
                _display_ps_results(
                    "One-sample t-test", params_df,
                    "t", t_stat, df_val, p_val, ci_lo, ci_hi,
                    "Cohen's d", d_val,
                    f"H₀: μ = {mu0}  |  t({df_val}) = {t_stat:.4f}, p = {'{:.4f}'.format(p_val) if p_val >= 0.0001 else '< .0001'}"
                )

        # ---- Two independent t-test (Welch) ----
        elif ps_test == "Two independent samples t-test (Welch)":
            col1, col2 = st.columns(2)
            with col1:
                n1 = st.number_input("n₁", 3, 100000, 25, 1, key="ws_ps_tsw_n1")
                m1 = st.number_input("Mean₁", value=0.0, key="ws_ps_tsw_m1", format="%.4f")
                s1 = st.number_input("SD₁", 0.001, 1000000.0, 1.0, key="ws_ps_tsw_s1", format="%.4f")
                g1_name = st.text_input("Group₁ label", "Group 1", key="ws_ps_tsw_g1")
            with col2:
                n2 = st.number_input("n₂", 3, 100000, 25, 1, key="ws_ps_tsw_n2")
                m2 = st.number_input("Mean₂", value=1.0, key="ws_ps_tsw_m2", format="%.4f")
                s2 = st.number_input("SD₂", 0.001, 1000000.0, 1.0, key="ws_ps_tsw_s2", format="%.4f")
                g2_name = st.text_input("Group₂ label", "Group 2", key="ws_ps_tsw_g2")
            if st.button("Compute", key="ws_ps_tsw_btn", type="primary"):
                v1, v2 = float(s1)**2 / int(n1), float(s2)**2 / int(n2)
                t_stat = (float(m1) - float(m2)) / np.sqrt(v1 + v2)
                num = (v1 + v2)**2
                den = v1**2 / (int(n1) - 1) + v2**2 / (int(n2) - 1)
                df_val = num / den if den > 0 else 1
                p_val = 2 * t_dist.sf(abs(t_stat), df_val)
                se_diff = np.sqrt(v1 + v2)
                ci_lo = (float(m1) - float(m2)) - t_dist.ppf(0.975, df_val) * se_diff
                ci_hi = (float(m1) - float(m2)) + t_dist.ppf(0.975, df_val) * se_diff
                params_df = pd.DataFrame({"Group": [g1_name or "Group 1", g2_name or "Group 2"],
                                           "n": [int(n1), int(n2)], "Mean": [f"{m1:.4f}", f"{m2:.4f}"],
                                           "SD": [f"{s1:.4f}", f"{s2:.4f}"]})
                _display_ps_results(
                    "Two-sample t-test (Welch)", params_df,
                    "t", t_stat, df_val, p_val, ci_lo, ci_hi,
                    "Mean diff", float(m1) - float(m2),
                    f"Welch df = {df_val:.2f}  |  t({df_val:.2f}) = {t_stat:.4f}"
                )

        # ---- Two independent t-test (pooled) ----
        elif ps_test == "Two independent samples t-test (pooled)":
            col1, col2 = st.columns(2)
            with col1:
                n1 = st.number_input("n₁", 3, 100000, 25, 1, key="ws_ps_tsp_n1")
                m1 = st.number_input("Mean₁", value=0.0, key="ws_ps_tsp_m1", format="%.4f")
                s1 = st.number_input("SD₁", 0.001, 1000000.0, 1.0, key="ws_ps_tsp_s1", format="%.4f")
                g1_name = st.text_input("Group₁ label", "Group 1", key="ws_ps_tsp_g1")
            with col2:
                n2 = st.number_input("n₂", 3, 100000, 25, 1, key="ws_ps_tsp_n2")
                m2 = st.number_input("Mean₂", value=1.0, key="ws_ps_tsp_m2", format="%.4f")
                s2 = st.number_input("SD₂", 0.001, 1000000.0, 1.0, key="ws_ps_tsp_s2", format="%.4f")
                g2_name = st.text_input("Group₂ label", "Group 2", key="ws_ps_tsp_g2")
            if st.button("Compute", key="ws_ps_tsp_btn", type="primary"):
                sp2 = ((int(n1) - 1) * float(s1)**2 + (int(n2) - 1) * float(s2)**2) / (int(n1) + int(n2) - 2)
                sp = np.sqrt(sp2)
                se_diff = sp * np.sqrt(1/int(n1) + 1/int(n2))
                t_stat = (float(m1) - float(m2)) / se_diff
                df_val = int(n1) + int(n2) - 2
                p_val = 2 * t_dist.sf(abs(t_stat), df_val)
                ci_lo = (float(m1) - float(m2)) - t_dist.ppf(0.975, df_val) * se_diff
                ci_hi = (float(m1) - float(m2)) + t_dist.ppf(0.975, df_val) * se_diff
                d_val = (float(m1) - float(m2)) / sp
                params_df = pd.DataFrame({"Group": [g1_name or "Group 1", g2_name or "Group 2"],
                                           "n": [int(n1), int(n2)], "Mean": [f"{m1:.4f}", f"{m2:.4f}"],
                                           "SD": [f"{s1:.4f}", f"{s2:.4f}"]})
                _display_ps_results(
                    "Two-sample t-test (pooled)", params_df,
                    "t", t_stat, df_val, p_val, ci_lo, ci_hi,
                    "Cohen's d", d_val,
                    f"Pooled SD = {sp:.4f}  |  t({df_val}) = {t_stat:.4f}"
                )

        # ---- Paired t-test ----
        elif ps_test == "Paired t-test":
            n = st.number_input("Sample size (n)", 3, 100000, 30, 1, key="ws_ps_pd_n")
            mean_diff = st.number_input("Mean difference", value=5.0, key="ws_ps_pd_mdiff", format="%.4f")
            sd_diff = st.number_input("SD of differences", 0.001, 1000000.0, 10.0, key="ws_ps_pd_sddiff", format="%.4f")
            if st.button("Compute", key="ws_ps_pd_btn", type="primary"):
                se = float(sd_diff) / np.sqrt(int(n))
                t_stat = float(mean_diff) / se
                df_val = int(n) - 1
                p_val = 2 * t_dist.sf(abs(t_stat), df_val)
                ci_lo = float(mean_diff) - t_dist.ppf(0.975, df_val) * se
                ci_hi = float(mean_diff) + t_dist.ppf(0.975, df_val) * se
                d_val = float(mean_diff) / float(sd_diff)
                params_df = pd.DataFrame({"Parameter": ["n", "Mean diff", "SD diff"],
                                           "Value": [int(n), f"{mean_diff:.4f}", f"{sd_diff:.4f}"]})
                _display_ps_results(
                    "Paired t-test", params_df,
                    "t", t_stat, df_val, p_val, ci_lo, ci_hi,
                    "Cohen's d_z", d_val,
                    f"H₀: μ_diff = 0  |  t({df_val}) = {t_stat:.4f}"
                )

        # ---- One proportion z-test ----
        elif ps_test == "One proportion z-test":
            n = st.number_input("Total observations (n)", 1, 100000, 50, 1, key="ws_ps_op_n")
            k = st.number_input("Number of successes (k)", 0, 100000, 20, 1, key="ws_ps_op_k")
            p0 = st.number_input("Null hypothesis proportion (p₀)", 0.001, 0.999, 0.5, key="ws_ps_op_p0", format="%.4f")
            if k > n:
                st.error("k cannot exceed n.")
                return False
            p_hat = k / n
            st.info(f"Observed proportion = {p_hat:.4f} ({k}/{n})")
            if st.button("Compute", key="ws_ps_op_btn", type="primary"):
                se0 = np.sqrt(float(p0) * (1 - float(p0)) / int(n))
                z_stat = (float(p_hat) - float(p0)) / se0 if se0 > 0 else 0.0
                p_val = 2 * norm_dist.sf(abs(z_stat))
                se_obs = np.sqrt(p_hat * (1 - p_hat) / int(n)) if p_hat > 0 and p_hat < 1 else 0
                ci_lo = p_hat - norm_dist.ppf(0.975) * se_obs
                ci_hi = p_hat + norm_dist.ppf(0.975) * se_obs
                params_df = pd.DataFrame({"Parameter": ["n", "Successes (k)", "p̂", "p₀"],
                                           "Value": [int(n), int(k), f"{p_hat:.4f}", f"{p0:.4f}"]})
                _display_ps_results(
                    "One-proportion z-test", params_df,
                    "z", z_stat, 1, p_val, ci_lo, ci_hi,
                    "p̂ - p₀", p_hat - float(p0),
                    f"H₀: p = {p0}  |  z = {z_stat:.4f}, p = {'{:.4f}'.format(p_val) if p_val >= 0.0001 else '< .0001'}"
                )

        # ---- Two proportions z-test ----
        elif ps_test == "Two proportions z-test":
            col1, col2 = st.columns(2)
            with col1:
                n1 = st.number_input("n₁", 1, 100000, 50, 1, key="ws_ps_tp_n1")
                k1 = st.number_input("Successes₁ (k₁)", 0, 100000, 20, 1, key="ws_ps_tp_k1")
            with col2:
                n2 = st.number_input("n₂", 1, 100000, 50, 1, key="ws_ps_tp_n2")
                k2 = st.number_input("Successes₂ (k₂)", 0, 100000, 10, 1, key="ws_ps_tp_k2")
            if k1 > n1 or k2 > n2:
                st.error("k cannot exceed n for either group.")
                return False
            p1, p2 = k1 / n1, k2 / n2
            p_bar = (k1 + k2) / (n1 + n2)
            st.info(f"p₁ = {p1:.4f} ({k1}/{n1})  |  p₂ = {p2:.4f} ({k2}/{n2})  |  p̄ = {p_bar:.4f}")
            if st.button("Compute", key="ws_ps_tp_btn", type="primary"):
                se_pooled = np.sqrt(p_bar * (1 - p_bar) * (1/int(n1) + 1/int(n2)))
                z_stat = (float(p1) - float(p2)) / se_pooled if se_pooled > 0 else 0.0
                p_val = 2 * norm_dist.sf(abs(z_stat))
                se_diff = np.sqrt(float(p1)*(1-float(p1))/int(n1) + float(p2)*(1-float(p2))/int(n2))
                ci_lo = (float(p1) - float(p2)) - norm_dist.ppf(0.975) * se_diff
                ci_hi = (float(p1) - float(p2)) + norm_dist.ppf(0.975) * se_diff
                params_df = pd.DataFrame({"Group": ["Group 1", "Group 2"],
                                           "n": [int(n1), int(n2)], "Successes": [int(k1), int(k2)],
                                           "p̂": [f"{p1:.4f}", f"{p2:.4f}"]})
                _display_ps_results(
                    "Two-proportion z-test", params_df,
                    "z", z_stat, 1, p_val, ci_lo, ci_hi,
                    "p₁ - p₂", float(p1) - float(p2),
                    f"Pooled p̄ = {p_bar:.4f}  |  z = {z_stat:.4f}"
                )

        # ---- Correlation t-test ----
        elif ps_test == "Correlation t-test":
            n = st.number_input("Sample size (n)", 3, 100000, 50, 1, key="ws_ps_cr_n")
            r_val = st.slider("Pearson r", -1.0, 1.0, 0.5, 0.01, key="ws_ps_cr_r")
            if st.button("Compute", key="ws_ps_cr_btn", type="primary"):
                t_stat = float(r_val) * np.sqrt((int(n) - 2) / (1 - float(r_val)**2)) if float(r_val)**2 < 1 else 0.0
                df_val = int(n) - 2
                p_val = 2 * t_dist.sf(abs(t_stat), df_val)
                z_r = np.arctanh(float(r_val))
                se_z = 1 / np.sqrt(int(n) - 3)
                z_crit = norm_dist.ppf(0.975)
                ci_lo = np.tanh(z_r - z_crit * se_z)
                ci_hi = np.tanh(z_r + z_crit * se_z)
                params_df = pd.DataFrame({"Parameter": ["n", "r"],
                                           "Value": [int(n), f"{r_val:.4f}"]})
                _display_ps_results(
                    "Correlation t-test", params_df,
                    "t", t_stat, df_val, p_val, ci_lo, ci_hi,
                    "r", float(r_val),
                    f"H₀: ρ = 0  |  t({df_val}) = {t_stat:.4f}"
                )

    return True


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
    if "ws_using_summarized" not in st.session_state:
        st.session_state.ws_using_summarized = False

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
            ["Upload", "Built-in", "Summarized Data"],
            horizontal=True,
            key="ws_src_mode",
        )

        # Clear summary flags when switching away from Summarized Data
        if "Summarized" not in src_mode and st.session_state.get("ws_using_summarized"):
            st.session_state.ws_using_summarized = False
            st.session_state.ws_df = None
            st.session_state.ws_external_data = None
            st.session_state.ws_selected_test = ""

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
        elif "Built-in" in src_mode:
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
        else:
            # Summarized Data — show input forms
            _render_summarized_data_inputs()

        # Build a source key to detect when user picks a different dataset
        new_source_key = f"{src_mode}:{new_name or ''}"
        prev_source_key = st.session_state.get("ws_source_key", "")

        if new_df is not None and new_source_key != prev_source_key:
            # Fresh data load — store and initialize
            st.session_state.ws_using_summarized = False
            st.session_state.ws_df = new_df
            st.session_state.ws_df_backup = new_df.copy()
            st.session_state.ws_dataset_name = new_name
            st.session_state.ws_source_key = new_source_key
            st.success(f"Loaded **{new_name}** — {len(new_df)} rows, {len(new_df.columns)} cols")

        if st.session_state.ws_df is None:
            if "Summarized" not in src_mode:
                st.info("No data loaded yet. Choose a source above.")
            return

        # Always work from session state downstream
        df = st.session_state.ws_df
        dataset_name = st.session_state.ws_dataset_name

        numeric_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    # ==========================================
    # SIDEBAR — Data Cleaning & Transformation
    # ==========================================
    with st.sidebar:
        st.subheader("Data Cleaning")
        st.caption(f"Current: **{len(df)}** rows × **{len(df.columns)}** cols")

        undo_col, csv_col, remaining_cols = st.columns(3)
        with undo_col:
            if "ws_df_backup" in st.session_state and st.session_state.ws_df_backup is not None:
                if st.button("Undo", key="ws_undo", type="secondary"):
                    st.session_state.ws_df = st.session_state.ws_df_backup.copy()
                    st.rerun()
        with csv_col:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⤓ CSV", csv_data, f"{dataset_name or 'data'}.csv", "text/csv", key="ws_export_csv",
            )

        ws_using_summarized = st.session_state.get("ws_using_summarized", False)

        if ws_using_summarized:
            st.info("Data cleaning is disabled for summarized data. "
                    "Modify the summary parameters above to adjust.")
        else:
            st.markdown(":orange[**Filter Rows**]")
            if numeric_cols:
                filt_col = st.selectbox("Numeric column", numeric_cols, key="ws_filt_col")
                col_vals = df[filt_col].dropna()
                min_v = float(col_vals.min())
                max_v = float(col_vals.max())
                if min_v >= max_v:
                    st.caption(f"Constant value `{min_v}` \u2014 no filter needed.")
                else:
                    filt_range = st.slider("Range", min_v, max_v, (min_v, max_v), key="ws_filt_range")
                    if st.button("Apply Filter", key="ws_filt_apply", type="secondary"):
                        st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                        df = df[(df[filt_col] >= filt_range[0]) & (df[filt_col] <= filt_range[1])].reset_index(drop=True)
                        st.session_state.ws_df = df
                        st.rerun()
            if cat_cols:
                filt_cat_col = st.selectbox("Categorical column", cat_cols, key="ws_filt_cat_col")
                all_vals = df[filt_cat_col].dropna().unique().tolist()
                if all_vals:
                    selected_vals = st.multiselect("Keep values", all_vals, default=all_vals, key="ws_filt_cat_vals")
                    if len(selected_vals) < len(all_vals):
                        if st.button("Apply", key="ws_filt_cat_apply", type="primary"):
                            st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                            df = df[df[filt_cat_col].isin(selected_vals)].reset_index(drop=True)
                            st.session_state.ws_df = df
                            st.rerun()

            st.divider()
            st.markdown(":orange[**Drop Columns**]")
            drop_cols = st.multiselect("Select columns to remove", df.columns.tolist(), key="ws_drop_cols")
            if drop_cols:
                if st.button("Drop", key="ws_drop_apply"):
                    st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                    df = df.drop(columns=drop_cols)
                    st.session_state.ws_df = df
                    st.rerun()

            st.markdown(":orange[**Rename Column**]")
            rename_col = st.selectbox("Column to rename", [""] + df.columns.tolist(), key="ws_rename_col")
            if rename_col:
                new_name = st.text_input("New name", value=rename_col, key="ws_rename_new")
                if new_name and new_name != rename_col:
                    if st.button("Rename", key="ws_rename_apply", type="primary"):
                        st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                        df = df.rename(columns={rename_col: new_name})
                        st.session_state.ws_df = df
                        st.rerun()

            st.divider()
            st.markdown(":orange[**Type Conversion**]")
            type_col = st.selectbox("Column to convert", [""] + df.columns.tolist(), key="ws_type_col")
            if type_col:
                st.caption(f"Current type: `{str(df[type_col].dtype)}`")
                target_type = st.selectbox(
                    "Convert to", ["", "float", "int", "string", "bool", "datetime", "category"], key="ws_type_target",
                )
                if target_type and st.button("Convert", key="ws_type_apply", type="primary"):
                    st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                    try:
                        if target_type == "float":
                            df[type_col] = pd.to_numeric(df[type_col], errors="coerce")
                        elif target_type == "int":
                            df[type_col] = pd.to_numeric(df[type_col], errors="coerce").astype("Int64")
                        elif target_type == "string":
                            df[type_col] = df[type_col].astype(str)
                        elif target_type == "bool":
                            df[type_col] = df[type_col].astype(bool)
                        elif target_type == "datetime":
                            df[type_col] = pd.to_datetime(df[type_col], errors="coerce")
                        elif target_type == "category":
                            df[type_col] = df[type_col].astype("category")
                        st.session_state.ws_df = df
                        st.rerun()
                    except Exception as e:
                        st.error(f"Conversion failed: {e}")

            st.divider()
            st.markdown(":orange[**Duplicate Rows**]")
            dup_cols = st.multiselect(
                "Columns to check (empty = all)", df.columns.tolist(), key="ws_dup_cols",
            )
            n_dups = df.duplicated(subset=dup_cols or None).sum()
            st.caption(f"**{n_dups}** duplicate(s)")
            if n_dups > 0:
                if st.button("Drop Dups", key="ws_dup_drop", type="primary"):
                    st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                    df = df.drop_duplicates(subset=dup_cols or None).reset_index(drop=True)
                    st.session_state.ws_df = df
                    st.success(f"Removed {n_dups} duplicate(s).")
                    st.rerun()

            st.divider()
            st.markdown(":orange[**Missing Values**]")
            missing_method = st.selectbox(
                "Method",
                ["Drop NA rows", "Drop all-NA rows", "Fill with column mean", "Fill with 0"], key="ws_na_method",
            )
            if st.button("Apply", key="ws_na_apply"):
                st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                if "all-NA" in missing_method:
                    prev = len(df)
                    df = df.dropna(how="all").reset_index(drop=True)
                    st.session_state.ws_df = df
                    st.success(f"Dropped {prev - len(df)} rows.")
                    st.rerun()
                elif "NA" in missing_method:
                    prev = len(df)
                    df = df.dropna().reset_index(drop=True)
                    st.session_state.ws_df = df
                    st.success(f"Dropped {prev - len(df)} rows.")
                    st.rerun()
                elif "mean" in missing_method:
                    num_df = df.select_dtypes(include=["float64", "int64"])
                    df[num_df.columns] = df[num_df.columns].fillna(num_df.mean())
                    st.session_state.ws_df = df
                    st.success("Filled with means.")
                    st.rerun()
                else:
                    df = df.fillna(0)
                    st.session_state.ws_df = df
                    st.success("Filled with 0.")
                    st.rerun()

            st.divider()
            st.markdown(":orange[**Transform Column**]")
            transform_col = st.selectbox("Column to transform", [""] + numeric_cols, key="ws_transform_col")
            if transform_col:
                transform_expr = st.text_input(
                    "Expression (use `col`)", placeholder="np.log(col)", key="ws_transform_expr",
                )
                if transform_expr:
                    if st.button("Transform", key="ws_transform_apply", type="primary"):
                        st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                        try:
                            col_vals = df[transform_col].values
                            result = eval(transform_expr, {"np": np, "pd": pd, "col": col_vals})
                            df[transform_col] = result
                            st.session_state.ws_df = df
                            st.rerun()
                        except Exception as e:
                            st.error(f"Expression error: {e}")

            st.divider()
            st.markdown(":orange[**Computed Column**]")
            col_name = st.text_input("New column name", placeholder="log_x", key="ws_comp_name")
            expression = st.text_input(
                "Expression (use `col`)", placeholder="np.log(col)", key="ws_comp_expr",
            )
            comp_col = st.selectbox("Source column", [""] + numeric_cols, key="ws_comp_source")
            if st.button("Open Expression Gallery", key="ws_gallery_btn", use_container_width=True):
                _expression_gallery_dialog()
            if col_name and expression and comp_col:
                if st.button("Create", key="ws_comp_apply", type="primary"):
                    st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                    try:
                        col = df[comp_col].values
                        result = eval(expression, {"np": np, "pd": pd, "col": col, "df": df})
                        df[col_name] = result
                        st.session_state.ws_df = df
                        st.rerun()
                    except Exception as e:
                        st.error(f"Expression error: {e}")

            st.divider()
            mapped = _render_categorical_mapping(df)
            if mapped is not None and isinstance(mapped, pd.DataFrame) and len(mapped.columns) > len(df.columns):
                st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                st.session_state.ws_df = mapped
                st.rerun()

    # ==========================================
    # RIGHT COLUMN — Phase 2: Data preview, stats, aggregation
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

        # Normality + Descriptives
        if numeric_cols:
            with st.expander("Normality Test (Shapiro-Wilk)", expanded=False):
                _render_normality_test_results(df, numeric_cols)
            with st.expander("Descriptive Statistics", expanded=False):
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        # Aggregation
        if numeric_cols and cat_cols:
            with st.expander("Aggregation (Groupby-Summarize)", expanded=False):
                grp_col = st.selectbox("Group by (categorical)", cat_cols, key="ws_agg_grp")
                val_cols = st.multiselect(
                    "Aggregate columns",
                    [c for c in numeric_cols if c != grp_col],
                    default=[c for c in numeric_cols[:2] if c != grp_col],
                    key="ws_agg_vals",
                )
                agg_funcs = st.multiselect(
                    "Functions",
                    ["Mean", "Sum", "Count", "Median", "Min", "Max", "Std", "Var"],
                    default=["Mean", "Count"],
                    key="ws_agg_funcs",
                )
                if grp_col and val_cols and agg_funcs:
                    func_map = {
                        "Mean": "mean", "Sum": "sum", "Count": "count",
                        "Median": "median", "Min": "min", "Max": "max",
                        "Std": "std", "Var": "var",
                    }
                    agg_dict = {c: [func_map[f] for f in agg_funcs] for c in val_cols}
                    agg_result = df.groupby(grp_col).agg(agg_dict)
                    agg_result.columns = [f"{col}_{fn}" for col, fn in agg_result.columns]
                    agg_result = agg_result.reset_index()
                    st.dataframe(agg_result, use_container_width=True)
                    if st.button("Replace data with aggregated table", key="ws_agg_replace", type="primary"):
                        st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                        st.session_state.ws_df = agg_result
                        st.rerun()

        # Pivot / Reshape
        with st.expander("Pivot / Reshape", expanded=False):
            pivot_mode = st.radio("Direction", ["Long → Wide (pivot)", "Wide → Long (melt)"], horizontal=True, key="ws_pivot_mode")
            if pivot_mode == "Long → Wide (pivot)":
                pi_idx = st.selectbox("Index", df.columns.tolist(), key="ws_pi_idx")
                pi_col = st.selectbox("Pivot column", df.columns.tolist(), key="ws_pi_col")
                pi_val = st.selectbox("Values column", [c for c in df.columns if c not in (pi_idx, pi_col, "")] if pi_idx and pi_col else df.columns.tolist(), key="ws_pi_val")
                if pi_idx and pi_col and pi_val and st.button("Pivot", key="ws_pivot_apply", type="primary"):
                    st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                    try:
                        pivoted = df.pivot_table(index=pi_idx, columns=pi_col, values=pi_val, aggfunc="first").reset_index()
                        pivoted.columns = [str(c) for c in pivoted.columns]
                        st.session_state.ws_df = pivoted
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pivot failed: {e}")
            else:
                mi_id = st.multiselect("ID columns", df.columns.tolist(), default=[], key="ws_mi_id")
                mi_val = st.multiselect("Value columns", [c for c in df.columns if c not in mi_id], default=[c for c in df.columns[:2] if c not in mi_id], key="ws_mi_val")
                mi_var = st.text_input("Variable name", value="variable", key="ws_mi_var")
                mi_val_name = st.text_input("Value name", value="value", key="ws_mi_val_name")
                if mi_val and st.button("Melt", key="ws_melt_apply", type="primary"):
                    st.session_state.ws_df_backup = st.session_state.ws_df.copy()
                    try:
                        melted = df.melt(id_vars=mi_id or None, value_vars=mi_val, var_name=mi_var, value_name=mi_val_name)
                        st.session_state.ws_df = melted
                        st.rerun()
                    except Exception as e:
                        st.error(f"Melt failed: {e}")

    # ==========================================
    # LEFT COLUMN — Phase 3: Data structure + summary + test
    # ==========================================
    with left:
        has_numeric = len(numeric_cols) > 0
        has_categorical = len(cat_cols) > 0

        st.divider()
        st.subheader("2. Data Structure")

        using_summarized = st.session_state.get("ws_using_summarized", False)

        if using_summarized:
            summary_fmt_labels = {
                "one_sample": "One-sample", "two_sample": "Two-sample (independent)",
                "paired": "Paired / Dependent",
                "correlation": "Correlation / Regression",
                "categorical_two": "Contingency table (two categorical variables)",
                "categorical_one": "Frequency table (single categorical variable)",
            }
            ext = st.session_state.ws_external_data
            sfmt = ext.get("_format", "") if ext else ""
            st.info(f"Using Summarized Data — format: **{summary_fmt_labels.get(sfmt, sfmt)}**")
            organization = None
            col_config = {}
            external_data = ext
        else:
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

        fmt = external_data.get("_format", "")

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
            if using_summarized:
                st.caption(f"   **{dataset_name}**")
            else:
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
