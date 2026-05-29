import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.post_hoc import render_post_hoc
from core.utils import (
    format_p_value,
    cohens_d_one_sample_ci,
    cohens_d_independent_ci,
    hedges_g,
    omega_squared_partial,
    format_effect_size_with_ci,
    st_plot_with_download,
    interpret_cohens_d,
    interpret_eta_squared,
    data_source_toggle,
)
from features.widgets import register_test


@register_test("One-sample Proportion Test (Binomial Test)")
def render_one_sample_proportion_test_binomial_test(external_data=None):

    from scipy.stats import binomtest

    st.subheader("Interactive One-sample Proportion Test")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("prop_1samp", mode="categorical_one")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        counts = src["data"]["counts"]
        categories = src["data"]["categories"]
        successes = int(counts[0])
        n = int(np.sum(counts))
        observed_p = successes / n
        expected_p = st.number_input("Expected Proportion (H₀)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f", key="prop_1samp_expected")
        st.info(f"Using first category **'{categories[0]}'** as success ({successes}/{n} = {observed_p:.1%})")
    else:
        expected_p = st.number_input("Expected Proportion", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f")
        observed_p = st.slider("Observed Proportion", 0.0, 1.0, 0.7, 0.01)
        n = st.slider("Sample Size", 10, 500, 100)
        successes = int(observed_p * n)

    result = binomtest(
        successes,
        n,
        expected_p,
    )

    # =========================
    # STATS
    # =========================

    st.latex(rf"\hat{{p}} = {observed_p:.2f}")

    st.latex(rf"\text{{{format_p_value(result.pvalue)}}}")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Expected", "Observed"],
            y=[expected_p, observed_p],
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        yaxis=dict(range=[0, 1]),
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # DETAILED STATISTICS TABLE
    # =========================

    st.divider()
    st.subheader("Detailed Results")

    from scipy.stats import binomtest as binomtest2
    from scipy.stats import norm as norm_prop

    p_hat = successes / n
    se_prop = np.sqrt(expected_p * (1 - expected_p) / n)
    z_prop = (p_hat - expected_p) / se_prop if se_prop > 0 else 0
    ci_prop = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n)

    results_data = {
        "Metric": [
            "Observed Proportion",
            "Expected Proportion",
            "Difference",
            "95% CI of Proportion",
            "Number of Successes",
            "Sample Size (n)",
            "z-approximation",
            "Exact p-value",
        ],
        "Value": [
            f"{p_hat:.3f}",
            f"{expected_p:.3f}",
            f"{p_hat - expected_p:.3f}",
            f"{p_hat - ci_prop:.3f} to {p_hat + ci_prop:.3f}",
            f"{successes}",
            f"{n}",
            f"{z_prop:.3f}",
            f"{result.pvalue:.5f}",
        ],
    }
    st.table(pd.DataFrame(results_data))

    # =========================
    # ENHANCED CHART
    # =========================

    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(
            name="Expected",
            x=["Proportion"],
            y=[expected_p],
            marker_color="rgba(255, 99, 71, 0.7)",
            width=[0.3],
            offsetgroup=0,
        )
    )
    fig2.add_trace(
        go.Bar(
            name="Observed",
            x=["Proportion"],
            y=[p_hat],
            marker_color="rgba(54, 162, 235, 0.7)",
            width=[0.3],
            offsetgroup=1,
        )
    )

    fig2.add_hline(
        y=expected_p,
        line_dash="dash",
        line_color="red",
        annotation_text="Expected",
    )
    fig2.add_hline(
        y=p_hat,
        line_dash="dot",
        line_color="blue",
        annotation_text="Observed",
    )

    # Error bar for CI
    fig2.add_trace(
        go.Scatter(
            x=["Proportion"],
            y=[p_hat],
            error_y=dict(
                type="data",
                symmetric=True,
                array=[ci_prop],
                visible=True,
                color="blue",
            ),
            mode="markers",
            marker=dict(size=8, color="blue"),
            showlegend=False,
        )
    )

    fig2.update_layout(
        template="plotly_dark",
        height=400,
        yaxis=dict(range=[0, 1]),
        barmode="group",
        xaxis_title="",
        yaxis_title="Proportion",
    )

    st.plotly_chart(fig2, use_container_width=True)



@register_test("Binomial Test")
def render_binomial_test(external_data=None):

    from scipy.stats import binomtest

    st.subheader("Interactive Binomial Test (Exact)")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("binom_exact", mode="categorical_one")

    # =========================
    # DATA
    # =========================

    if src["using_uploaded"]:
        counts = src["data"]["counts"]
        if len(counts) >= 2:
            successes_bt = int(counts[0])
            n_bt = int(counts.sum())
        else:
            successes_bt = int(counts[0])
            n_bt = int(counts[0]) * 2
    else:
        expected_p_bt = st.number_input("Hypothesized Probability (p₀)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f", key="binom_exact_expected_p")
        successes_bt = st.slider("Number of Successes", 0, 100, 15, key="binom_exact_successes")
        n_bt = st.slider("Number of Trials (n)", 1, 200, 30, key="binom_exact_n")

    # =========================
    # TEST
    # =========================

    if src["using_uploaded"]:
        expected_p_bt = st.number_input("Hypothesized Probability (p₀)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f", key="binom_exact_expected_p_uploaded")

    result_bt = binomtest(successes_bt, n_bt, expected_p_bt)
    p_hat_bt = successes_bt / n_bt
    ci_bt = result_bt.proportion_ci()

    st.latex(rf"\hat{{p}} = {p_hat_bt:.3f}")
    st.latex(rf"\text{{{format_p_value(result_bt.pvalue)}}}")
    st.latex(rf"95\% \text{{ CI}}: ({ci_bt.low:.3f}, {ci_bt.high:.3f})")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Expected", "Observed"], y=[expected_p_bt, p_hat_bt],
                         marker_color=["rgba(255,99,71,0.7)", "rgba(54,162,235,0.7)"]))
    fig.add_hline(y=expected_p_bt, line_dash="dash", line_color="red", annotation_text=f"p₀ = {expected_p_bt}")
    fig.update_layout(template="plotly_dark", height=400, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # DETAILED RESULTS
    # =========================

    st.divider()
    st.subheader("Detailed Results")
    bt_results = {
        "Metric": ["Observed Proportion", "Hypothesized p₀", "Successes", "Trials (n)", "95% CI Lower", "95% CI Upper", "Exact p-value"],
        "Value": [f"{p_hat_bt:.4f}", f"{expected_p_bt:.4f}", f"{successes_bt}", f"{n_bt}", f"{ci_bt.low:.4f}", f"{ci_bt.high:.4f}", format_p_value(result_bt.pvalue)],
    }
    st.table(pd.DataFrame(bt_results))



@register_test("Multinomial Test")
def render_multinomial_test(external_data=None):

    from scipy.stats import chisquare

    st.subheader("Interactive Multinomial Test")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("multinomial", mode="categorical_one")

    # =========================
    # DATA
    # =========================

    if src["using_uploaded"]:
        categories_mt = list(src["data"]["categories"])
        observed_mt = np.array(src["data"]["counts"], dtype=float)
    else:
        n_cat_mt = st.slider("Number of Categories", 2, 8, 3, key="multinomial_ncat")
        observed_mt = []
        cat_labels_mt = []
        for i in range(n_cat_mt):
            val = st.slider(f"Category {chr(65+i)}", 0, 200, 20 + i * 10, key=f"multinomial_cat_{i}")
            observed_mt.append(val)
            cat_labels_mt.append(chr(65 + i))
        observed_mt = np.array(observed_mt, dtype=float)
        categories_mt = cat_labels_mt

    expected_mt = np.full_like(observed_mt, observed_mt.sum() / len(observed_mt))
    chi2_mt, p_mt = chisquare(observed_mt)

    n_mt = int(observed_mt.sum())

    st.latex(rf"\chi^2 = {chi2_mt:.3f}")
    st.latex(rf"\text{{df}} = {len(observed_mt) - 1}")
    st.latex(rf"\text{{{format_p_value(p_mt)}}}")

    st.info("The Multinomial Test extends the Binomial Test to multiple categories. The null hypothesis is that all categories are equally likely (uniform distribution).")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories_mt, y=observed_mt, name="Observed", marker_color="rgba(54,162,235,0.7)"))
    fig.add_trace(go.Scatter(x=categories_mt, y=expected_mt, mode="lines+markers", name="Expected (uniform)", line=dict(color="red", width=3, dash="dash")))
    fig.update_layout(template="plotly_dark", height=400, xaxis_title="Category", yaxis_title="Count", barmode="group")
    st.plotly_chart(fig, use_container_width=True)



@register_test("Chi-Square Goodness-of-Fit Test")
def render_chi_square_goodness_of_fit_test(external_data=None):

    from scipy.stats import chisquare

    st.subheader("Interactive Chi-Square Goodness-of-Fit Test")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("chisq_gof", mode="categorical_one")

    # =========================
    # DATA
    # =========================

    if src["using_uploaded"]:
        categories = list(src["data"]["categories"])
        observed = np.array(src["data"]["counts"], dtype=float)
    else:
        n_cat = st.slider("Number of Categories", 2, 10, 3, key="chisq_gof_ncat")
        observed = []
        cat_labels = []
        for i in range(n_cat):
            col_key = f"chisq_gof_cat_{i}"
            val = st.slider(f"Category {chr(65+i)}", 1, 200, 30 + i * 10, key=col_key)
            observed.append(val)
            cat_labels.append(chr(65 + i))
        observed = np.array(observed, dtype=float)
        categories = cat_labels

    expected = np.mean(observed)
    chi2, p = chisquare(observed)

    # =========================
    # STATS
    # =========================

    st.latex(rf"\chi^2 = {chi2:.3f}")
    st.latex(rf"\text{{{format_p_value(p)}}}")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=categories, y=observed, name="Observed")
    )
    fig.add_trace(
        go.Scatter(x=categories, y=[expected] * len(categories),
                   mode="lines", name="Expected")
    )
    fig.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")

    from scipy.stats import chi2 as chi2_dist_gof

    n_gof = np.sum(observed)
    k_gof = len(observed)
    df_gof = k_gof - 1
    cramer_v_gof = (
        np.sqrt(chi2 / (n_gof * (k_gof - 1))) if n_gof > 0 and k_gof > 1 else 0
    )

    metric_col = ["Category"] + categories + ["Expected (mean)", "χ²", "df", "p-value", "Cramer's V"]
    val_col = [""] + [f"{int(x)}" for x in observed] + [f"{expected:.1f}", f"{chi2:.3f}", f"{df_gof}", f"{p:.5f}", f"{cramer_v_gof:.4f}"]
    st.table(pd.DataFrame({"Metric": metric_col, "Value": val_col}))

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(name="Observed", x=categories, y=observed,
               marker_color="rgba(54, 162, 235, 0.7)")
    )
    fig2.add_trace(
        go.Bar(name="Expected", x=categories, y=[expected] * len(categories),
               marker_color="rgba(255, 99, 71, 0.7)")
    )
    fig2.update_layout(template="plotly_dark", height=400,
                       xaxis_title="Category", yaxis_title="Count", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)



@register_test("Poisson Goodness-of-Fit Test")
def render_poisson_goodness_of_fit_test(external_data=None):

    from scipy.stats import chisquare, poisson

    st.subheader("Interactive Poisson Goodness-of-Fit Test")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("poisson_gof", mode="categorical_one")

    # =========================
    # DATA
    # =========================

    if src["using_uploaded"]:
        categories = list(src["data"]["categories"])
        observed = np.array(src["data"]["counts"], dtype=float)
    else:
        n_cat = st.slider("Number of Bins", 3, 10, 5, key="poisson_gof_ncat")
        observed = []
        cat_labels = []
        for i in range(n_cat):
            val = st.slider(f"Bin {chr(65+i)}", 0, 100, 15 + i * 5, key=f"poisson_gof_cat_{i}")
            observed.append(val)
            cat_labels.append(chr(65 + i))
        observed = np.array(observed, dtype=float)
        categories = cat_labels

    # Estimate λ from data
    total_count = observed.sum()
    bin_centers = np.arange(len(observed))
    lam_est = np.sum(bin_centers * observed) / total_count if total_count > 0 else 1

    # Expected Poisson frequencies
    expected_pois = np.array([poisson.pmf(k, lam_est) for k in bin_centers]) * total_count
    # Ensure no zero expected values, then renormalize to match observed sum
    expected_pois = np.maximum(expected_pois, 0.5)
    expected_pois = expected_pois * (observed.sum() / expected_pois.sum())

    chi2_pois, p_pois = chisquare(observed, expected_pois)
    df_pois = len(observed) - 2

    # =========================
    # STATS
    # =========================

    st.latex(rf"\hat{{\lambda}} = {lam_est:.3f}")
    st.latex(rf"\chi^2 = {chi2_pois:.3f}")
    st.latex(rf"\text{{df}} = {df_pois}")
    st.latex(rf"\text{{{format_p_value(p_pois)}}}")

    var_mean_ratio = np.var(np.repeat(bin_centers, observed.astype(int))) / lam_est if lam_est > 0 else 1

    if p_pois < 0.05:
        st.warning(f"The data does NOT follow a Poisson distribution (p = {p_pois:.4f}). Variance/Mean ratio = {var_mean_ratio:.3f}.")
        if var_mean_ratio > 1.2:
            st.info("Variance > Mean indicates **over-dispersion**. Consider Negative Binomial regression.")
    else:
        st.success(f"The data is consistent with a Poisson distribution (p = {p_pois:.4f}).")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=observed, name="Observed", marker_color="rgba(54,162,235,0.7)"))
    fig.add_trace(go.Scatter(x=categories, y=expected_pois, mode="lines+markers", name="Poisson Expected", line=dict(color="red", width=3)))
    fig.update_layout(template="plotly_dark", height=400, xaxis_title="Category", yaxis_title="Count", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# Categorial Tests


@register_test("Chi-Square Test")
def render_chi_square_test(external_data=None):

    from scipy.stats import chi2_contingency

    st.subheader("Interactive Chi-Square Test of Independence")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("chisq_indep", mode="categorical_two")

    # =========================
    # DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        table = ct.values.astype(float)
        row_labels = list(ct.index)
        col_labels = list(ct.columns)
    else:
        n_rows = st.slider("Number of Rows", 2, 5, 2, key="chisq_nrows")
        n_cols = st.slider("Number of Columns", 2, 5, 2, key="chisq_ncols")
        table = np.zeros((n_rows, n_cols), dtype=float)
        for r in range(n_rows):
            for c in range(n_cols):
                table[r, c] = st.slider(
                    f"Row {r+1}, Col {c+1}", 0, 200, 20 + r * 10 + c * 5,
                    key=f"chisq_cell_{r}_{c}"
                )
        row_labels = [f"R{r+1}" for r in range(n_rows)]
        col_labels = [f"C{c+1}" for c in range(n_cols)]

    chi2, p, dof, expected = chi2_contingency(table)

    # =========================
    # STATS
    # =========================

    st.latex(rf"\chi^2 = {chi2:.3f}")
    st.latex(rf"\text{{Degrees of Freedom}} = {dof}")
    st.latex(rf"\text{{{format_p_value(p)}}}")

    # =========================
    # HEATMAP
    # =========================

    fig = go.Figure(
        data=go.Heatmap(z=table, text=table.astype(int), texttemplate="%{text}")
    )
    fig.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")

    from scipy.stats import chi2 as chi2_dist_cs

    n_cs = np.sum(table)
    cramer_v_cs = (
        np.sqrt(chi2 / (n_cs * min(table.shape[0] - 1, table.shape[1] - 1)))
        if n_cs > 0 else 0
    )

    st.write(f"**Cramer's V:** {cramer_v_cs:.4f}")

    cell_labels = [f"({r},{c})" for r in row_labels for c in col_labels]
    observed_flat = table.flatten()
    expected_flat = expected.flatten()
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(name="Observed", x=cell_labels, y=observed_flat,
               marker_color="rgba(54, 162, 235, 0.7)")
    )
    fig2.add_trace(
        go.Bar(name="Expected", x=cell_labels, y=expected_flat,
               marker_color="rgba(255, 99, 71, 0.7)")
    )
    fig2.update_layout(template="plotly_dark", height=400,
                       xaxis_title="Cell", yaxis_title="Count", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)



@register_test("Chi-Square Test of Independence")
def render_chi_square_test_of_independence(external_data=None):

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("chisq_indep_dup", mode="categorical_two")

    from scipy.stats import chi2_contingency

    st.subheader("Interactive Chi-Square Test of Independence")

    # =========================
    # DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        table = ct.values.astype(float)
        row_labels = list(ct.index)
        col_labels = list(ct.columns)
    else:
        a = st.slider(
            "Cell A", 1, 100, 30, key="chi_square_test_of_independence_cell_a"
        )
        b = st.slider(
            "Cell B", 1, 100, 20, key="chi_square_test_of_independence_cell_b"
        )
        c = st.slider(
            "Cell C", 1, 100, 10, key="chi_square_test_of_independence_cell_c"
        )
        d = st.slider(
            "Cell D", 1, 100, 40, key="chi_square_test_of_independence_cell_d"
        )
        table = np.array([[a, b], [c, d]])
        row_labels = ["Row 1", "Row 2"]
        col_labels = ["Col 1", "Col 2"]

    chi2, p, dof, expected = chi2_contingency(table)

    # =========================
    # STATISTICS
    # =========================

    st.latex(rf"\chi^2 = {chi2:.3f}")

    st.write(f"p-value = {p:.5f}")

    # =========================
    # HEATMAP
    # =========================

    fig = go.Figure(
        data=go.Heatmap(
            z=table,
            text=table,
            texttemplate="%{text}",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("Note: This is a simplified 2×2 widget. For larger contingency tables, use **Chi-Square Test** from the sidebar, which supports dynamic row/column counts and uploaded data.")



@register_test("McNemar's Test")
def render_mcnemar_s_test(external_data=None):

    from statsmodels.stats.contingency_tables import mcnemar

    st.subheader("Interactive McNemar's Test")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("mcnemar", mode="categorical_two")

    # =========================
    # DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        if ct.shape != (2, 2):
            st.error("McNemar's Test requires a 2×2 contingency table. "
                     "Please select two binary categorical variables.")
            return
        table = ct.values.astype(float)
        row_labels = list(ct.index)
        col_labels = list(ct.columns)
    else:
        st.caption("Enter paired binary outcomes (Before/After or Time 1/Time 2):")
        yes_yes = st.slider("Yes → Yes", 0, 100, 40, key="mcnemar_yy")
        yes_no = st.slider("Yes → No", 0, 100, 10, key="mcnemar_yn")
        no_yes = st.slider("No → Yes", 0, 100, 30, key="mcnemar_ny")
        no_no = st.slider("No → No", 0, 100, 20, key="mcnemar_nn")
        table = np.array([[yes_yes, yes_no], [no_yes, no_no]], dtype=float)
        row_labels = ["Before Yes", "Before No"]
        col_labels = ["After Yes", "After No"]

    result = mcnemar(table)

    # =========================
    # STATS
    # =========================

    st.latex(rf"\chi^2 = {result.statistic:.3f}")
    st.latex(rf"\text{{{format_p_value(result.pvalue)}}}")

    # =========================
    # HEATMAP
    # =========================

    fig = go.Figure(
        data=go.Heatmap(z=table, text=table.astype(int), texttemplate="%{text}",
                        x=col_labels, y=row_labels)
    )
    fig.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")

    b_mc = table[0, 1]
    c_mc = table[1, 0]
    odds_ratio_mc = b_mc / c_mc if c_mc > 0 else float("inf")

    results_data = {
        "Metric": ["χ²", "p-value", "b (Yes→No)", "c (No→Yes)", "Odds Ratio (b/c)"],
        "Value": [
            f"{result.statistic:.3f}", f"{result.pvalue:.5f}",
            f"{int(b_mc)}", f"{int(c_mc)}",
            f"{odds_ratio_mc:.3f}" if c_mc > 0 else "∞"
        ],
    }
    st.table(pd.DataFrame(results_data))

    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(name="Discordant Pairs",
               x=["b (Yes→No)", "c (No→Yes)"], y=[b_mc, c_mc],
               marker_color=["rgba(255, 99, 71, 0.7)", "rgba(54, 162, 235, 0.7)"])
    )
    fig2.update_layout(template="plotly_dark", height=400,
                       xaxis_title="Discordant Pair Type", yaxis_title="Count")
    st.plotly_chart(fig2, use_container_width=True)



@register_test("Cochran's Q Test")
def render_cochran_s_q_test(external_data=None):

    from statsmodels.stats.contingency_tables import cochrans_q

    st.subheader("Interactive Cochran's Q Test")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("cochran_q", mode="repeated")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        measurements = src["data"]["measurements"]
        data = np.column_stack(measurements)
    else:
        n_conds = st.slider("Number of Conditions", 2, 6, 3, key="cq_nconds")
        probs = []
        for i in range(n_conds):
            p = st.slider(f"Condition {i+1} Success Probability", 0.0, 1.0, 0.3 + i * 0.2, 0.01,
                          key=f"cq_prob_{i}")
            probs.append(p)
        subjects = st.slider("Subjects", 10, 300, 100, key="cq_subjects")

        np.random.seed(42)
        cols = [np.random.binomial(1, p, subjects) for p in probs]
        data = np.column_stack(cols)

    if data.shape[1] < 2:
        st.error("Cochran's Q requires at least 2 conditions.")
        return

    result = cochrans_q(data)

    # =========================
    # STATS
    # =========================

    st.latex(rf"Q = {result.statistic:.3f}")

    st.latex(rf"\text{{{format_p_value(result.pvalue)}}}")

    # =========================
    # PLOT
    # =========================

    means = data.mean(axis=0)
    k_cq = data.shape[1]
    cond_labels = [f"Condition {i+1}" for i in range(k_cq)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=cond_labels, y=means)
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        yaxis=dict(range=[0, 1]),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")

    df_cq = k_cq - 1

    metrics = ["Q", "df", "p-value"] + [f"Proportion {l}" for l in cond_labels]
    values = [f"{result.statistic:.3f}", f"{df_cq}", f"{result.pvalue:.5f}"] + [f"{means[i]:.3f}" for i in range(k_cq)]

    results_data = {"Metric": metrics, "Value": values}
    st.table(pd.DataFrame(results_data))
    fig2 = go.Figure()
    palette = ["rgba(54, 162, 235, 0.7)", "rgba(255, 99, 71, 0.7)", "rgba(75, 192, 192, 0.7)",
                "rgba(153, 102, 255, 0.7)", "rgba(255, 159, 64, 0.7)", "rgba(46, 204, 113, 0.7)"]
    fig2.add_trace(
        go.Bar(
            x=cond_labels,
            y=means,
            marker_color=[palette[i % len(palette)] for i in range(k_cq)],
            name="Proportion",
        )
    )
    np.random.seed(123)
    for j in range(data.shape[1]):
        jitter_x = np.random.normal(j, 0.05, size=data.shape[0])
        response_y = data[:, j]
        fig2.add_trace(
            go.Scatter(
                x=jitter_x,
                y=response_y,
                mode="markers",
                showlegend=False,
                marker=dict(color="white", size=3, opacity=0.3),
            )
        )
    fig2.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Condition",
        yaxis_title="Proportion",
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig2, use_container_width=True)



@register_test("Fisher's Exact Test")
def render_fisher_s_exact_test(external_data=None):

    from scipy.stats import fisher_exact

    st.subheader("Interactive Fisher's Exact Test")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("fisher_exact", mode="categorical_two")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        if ct.shape != (2, 2):
            st.error("Fisher's Exact Test requires a 2×2 contingency table. "
                     "Please select two binary categorical variables.")
            return
        a = ct.iloc[0, 0]
        b = ct.iloc[0, 1]
        c = ct.iloc[1, 0]
        d = ct.iloc[1, 1]
    else:
        a = st.slider("Cell A", 0, 50, 8, key="fisher_s_exact_test_cell_a")
        b = st.slider("Cell B", 0, 50, 2, key="fisher_s_exact_test_cell_b")
        c = st.slider("Cell C", 0, 50, 1, key="fisher_s_exact_test_cell_c")
        d = st.slider("Cell D", 0, 50, 9, key="fisher_s_exact_test_cell_d")

    table = np.array([[a, b], [c, d]])
    odds_ratio, p = fisher_exact(table)

    # =========================
    # STATS
    # =========================

    st.latex(rf"OR = {odds_ratio:.3f}")

    st.latex(rf"\text{{{format_p_value(p)}}}")

    # =========================
    # HEATMAP
    # =========================

    fig = go.Figure(
        data=go.Heatmap(
            z=table,
            text=table,
            texttemplate="%{text}",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")

    from scipy.stats import fisher_exact as fisher_exact2

    or_f, p_f = fisher_exact2(table)
    log_or = np.log(or_f) if or_f > 0 else 0
    se_log_or = np.sqrt(np.sum(1 / table[table > 0])) if np.all(table > 0) else 0
    ci_low_f = np.exp(log_or - 1.96 * se_log_or) if se_log_or > 0 else 0
    ci_high_f = np.exp(log_or + 1.96 * se_log_or) if se_log_or > 0 else float("inf")

    results_data = {
        "Metric": [
            "Odds Ratio",
            "95% CI (OR)",
            "p-value",
            "",
            "Cell A",
            "Cell B",
            "Cell C",
            "Cell D",
        ],
        "Value": [
            f"{or_f:.3f}",
            f"[{ci_low_f:.3f}, {ci_high_f:.3f}]",
            f"{p_f:.5f}",
            "",
            f"{table[0, 0]}",
            f"{table[0, 1]}",
            f"{table[1, 0]}",
            f"{table[1, 1]}",
        ],
    }
    st.table(pd.DataFrame(results_data))

    fig2 = go.Figure(
        data=go.Heatmap(
            z=table,
            text=table,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig2.update_layout(template="plotly_dark", height=400)
    fig2.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=f"OR = {or_f:.3f}, 95% CI: [{ci_low_f:.3f}, {ci_high_f:.3f}]",
        showarrow=False,
        font=dict(size=14),
    )
    st.plotly_chart(fig2, use_container_width=True)

# Parametric Two Sample Tests

