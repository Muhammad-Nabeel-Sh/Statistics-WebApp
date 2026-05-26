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
)


def render_latex(formula_text):
    """Render LaTeX formulas from text with $$ delimiters."""
    import re

    last_end = 0
    for match in re.finditer(r"\$\$(.*?)\$\$", formula_text, re.DOTALL):
        # Text before this match
        text_before = formula_text[last_end : match.start()]
        if text_before.strip():
            st.markdown(text_before)

        # The LaTeX block (without $$ delimiters)
        latex_code = match.group(1).strip()
        st.latex(latex_code)

        last_end = match.end()

    # Text after the last match
    text_after = formula_text[last_end:]
    if text_after.strip():
        st.markdown(text_after)


def render_test_widget(test_name):
    """Render interactive widget for specific statistical test."""

    # One Sample Tests

    if test_name == "One-sample t-test":

        from scipy.stats import ttest_1samp

        st.subheader("Interactive One-sample t-test")

        # =========================
        # CONTROLS
        # =========================

        population_mean = st.slider(
            "Reference Mean",
            -10.0,
            10.0,
            0.0,
            0.1,
        )

        sample_mean_shift = st.slider(
            "Sample Mean Shift",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        sd = st.slider(
            "Standard Deviation",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        sample = np.random.normal(
            population_mean + sample_mean_shift,
            sd,
            80,
        )

        t, p = ttest_1samp(sample, population_mean)

        # =========================
        # STATS
        # =========================

        st.latex(rf"t = {t:.3f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=sample,
                nbinsx=20,
            )
        )

        fig.add_vline(
            x=population_mean,
            line_dash="dash",
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st_plot_with_download(fig, key="ttest_1samp_hist", height=550)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import t as t_dist
        from scipy.stats import sem

        n = len(sample)
        sample_mean = np.mean(sample)
        sample_sd = np.std(sample, ddof=1)
        se = sem(sample)
        ci = se * t_dist.ppf(0.975, n - 1)
        cohens_d = (sample_mean - population_mean) / sample_sd
        d_lower, d_upper = cohens_d_one_sample_ci(cohens_d, n)
        hedges = hedges_g(cohens_d, n)
        d_interp = interpret_cohens_d(cohens_d)

        results_data = {
            "Metric": [
                "Sample Mean",
                "Reference Mean",
                "Mean Difference",
                "95% CI of Diff",
                "t-statistic",
                "df",
                "p-value",
                "Cohen's d [95% CI]",
                "Hedges' g (unbiased)",
                "Interpretation",
            ],
            "Value": [
                f"{sample_mean:.3f}",
                f"{population_mean:.3f}",
                f"{sample_mean - population_mean:.3f}",
                f"[{sample_mean - population_mean - ci:.3f}, {sample_mean - population_mean + ci:.3f}]",
                f"{t:.3f}",
                f"{n - 1}",
                format_p_value(p),
                format_effect_size_with_ci(cohens_d, d_lower, d_upper),
                f"{hedges:.3f}",
                d_interp,
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Histogram(
                x=sample,
                nbinsx=25,
                histnorm="probability density",
                name="Sample",
                marker=dict(color="rgba(0, 123, 255, 0.5)"),
            )
        )

        x_dense = np.linspace(sample.min(), sample.max(), 200)
        from scipy.stats import norm

        y_dense = norm.pdf(x_dense, sample_mean, sample_sd)
        fig2.add_trace(
            go.Scatter(
                x=x_dense,
                y=y_dense,
                mode="lines",
                name="Normal fit",
                line=dict(color="red", width=2),
            )
        )

        fig2.add_vline(
            x=population_mean,
            line_dash="dash",
            line_color="green",
            annotation_text="Reference Mean",
        )
        fig2.add_vline(
            x=sample_mean,
            line_dash="dot",
            line_color="blue",
            annotation_text="Sample Mean",
        )
        fig2.add_vline(
            x=sample_mean - ci,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
        )
        fig2.add_vline(
            x=sample_mean + ci,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
        )

        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Value",
            yaxis_title="Density",
        )

        st_plot_with_download(fig2, key="ttest_1samp_density", height=400)

    elif test_name == "One-sample z-test":

        from statsmodels.stats.weightstats import ztest

        st.subheader("Interactive One-sample z-test")

        # =========================
        # CONTROLS
        # =========================

        population_mean = st.slider(
            "Population Mean",
            -10.0,
            10.0,
            0.0,
            0.1,
        )

        shift = st.slider(
            "Sample Mean Shift",
            -5.0,
            5.0,
            1.0,
            0.1,
            key="sample_mean_shift_1",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        sample = np.random.normal(
            population_mean + shift,
            1,
            300,
        )

        z, p = ztest(sample, value=population_mean)

        # =========================
        # STATS
        # =========================

        st.latex(rf"z = {z:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Histogram(x=sample))

        fig.add_vline(
            x=population_mean,
            line_dash="dash",
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        n_z = len(sample)
        sample_mean_z = np.mean(sample)
        se_z = 1 / np.sqrt(n_z)
        ci_z = 1.96 * se_z

        results_data = {
            "Metric": [
                "Sample Mean",
                "Population Mean (μ₀)",
                "z-statistic",
                "p-value",
                "SE (σ/√n)",
                "n",
                "95% CI of Mean",
                "Known σ",
            ],
            "Value": [
                f"{sample_mean_z:.3f}",
                f"{population_mean:.3f}",
                f"{z:.3f}",
                f"{p:.5f}",
                f"{se_z:.4f}",
                f"{n_z}",
                f"{sample_mean_z - ci_z:.2f} to {sample_mean_z + ci_z:.2f}",
                "1.0",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Histogram(
                x=sample,
                nbinsx=25,
                histnorm="probability density",
                name="Sample",
                marker=dict(color="rgba(0, 123, 255, 0.5)"),
            )
        )

        x_dense = np.linspace(sample.min(), sample.max(), 200)
        from scipy.stats import norm as norm2

        y_dense = norm2.pdf(x_dense, sample_mean_z, 1 / np.sqrt(n_z) * np.sqrt(n_z))
        y_dense = norm2.pdf(x_dense, sample_mean_z, np.std(sample, ddof=1))
        fig2.add_trace(
            go.Scatter(
                x=x_dense,
                y=y_dense,
                mode="lines",
                name="Normal fit",
                line=dict(color="red", width=2),
            )
        )

        fig2.add_vline(
            x=population_mean,
            line_dash="dash",
            line_color="green",
            annotation_text="H₀ Mean",
        )
        fig2.add_vline(
            x=sample_mean_z,
            line_dash="dot",
            line_color="blue",
            annotation_text="Sample Mean",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Value",
            yaxis_title="Density",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "One-sample Proportion Test (Binomial Test)":

        from scipy.stats import binomtest

        st.subheader("Interactive One-sample Proportion Test")

        # =========================
        # CONTROLS
        # =========================

        expected_p = st.slider(
            "Expected Proportion",
            0.0,
            1.0,
            0.5,
            0.01,
        )

        observed_p = st.slider(
            "Observed Proportion",
            0.0,
            1.0,
            0.7,
            0.01,
        )

        n = st.slider(
            "Sample Size",
            10,
            500,
            100,
        )

        # =========================
        # TEST
        # =========================

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

    elif test_name == "One-sample Wilcoxon Signed-Rank Test":

        from scipy.stats import wilcoxon

        st.subheader("Interactive One-sample Wilcoxon Signed-Rank Test")

        # =========================
        # CONTROLS
        # =========================

        median_shift = st.slider(
            "Median Shift",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        sample = np.random.exponential(1, 80)

        sample = sample + median_shift

        stat, p = wilcoxon(sample)

        # =========================
        # STATS
        # =========================

        st.latex(rf"W = {stat:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=sample))

        fig.add_hline(y=0)

        fig.update_layout(
            template="plotly_dark",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        n_1w = len(sample)
        median_1w = np.median(sample)
        hypothesized_median = 0
        median_diff_1w = median_1w - hypothesized_median
        from scipy.stats import wilcoxon as wilcoxon_1samp

        T_1w, p_1w = wilcoxon_1samp(sample)
        r_rb_1w = 1 - 2 * T_1w / (n_1w * (n_1w + 1) / 2)

        results_data = {
            "Metric": [
                "Median",
                "Hypothesized Median",
                "W-statistic",
                "p-value",
                "Sample Size (n)",
                "Median Difference",
                "Rank-biserial r",
                "",
            ],
            "Value": [
                f"{median_1w:.3f}",
                f"{hypothesized_median}",
                f"{T_1w:.3f}",
                f"{p_1w:.5f}",
                f"{n_1w}",
                f"{median_diff_1w:.3f}",
                f"{r_rb_1w:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(go.Box(y=sample, name="Sample", boxmean="sd"))

        jitter_x = np.random.normal(0, 0.06, n_1w)
        fig2.add_trace(
            go.Scatter(
                x=jitter_x,
                y=sample,
                mode="markers",
                name="Data points",
                marker=dict(color="rgba(0, 123, 255, 0.5)", size=5),
            )
        )

        fig2.add_hline(
            y=hypothesized_median,
            line_dash="dash",
            line_color="red",
            annotation_text="H₀: median = 0",
        )

        ci_1w = (
            1.58
            * (np.percentile(sample, 75) - np.percentile(sample, 25))
            / np.sqrt(n_1w)
        )
        fig2.add_hline(
            y=median_1w + ci_1w, line_dash="dot", line_color="gray", opacity=0.5
        )
        fig2.add_hline(
            y=median_1w - ci_1w, line_dash="dot", line_color="gray", opacity=0.5
        )

        fig2.update_layout(
            template="plotly_dark",
            height=450,
            xaxis=dict(showticklabels=False),
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Chi-Square Goodness-of-Fit Test":

        from scipy.stats import chisquare

        st.subheader("Interactive Chi-Square Goodness-of-Fit Test")

        # =========================
        # CONTROLS
        # =========================

        obs1 = st.slider("Observed Category A", 1, 100, 40)
        obs2 = st.slider("Observed Category B", 1, 100, 30)
        obs3 = st.slider("Observed Category C", 1, 100, 20)

        # =========================
        # DATA
        # =========================

        observed = np.array([obs1, obs2, obs3])

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
            go.Bar(
                x=["A", "B", "C"],
                y=observed,
                name="Observed",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=["A", "B", "C"],
                y=[expected] * 3,
                mode="lines",
                name="Expected",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import chi2 as chi2_dist_gof

        n_gof = np.sum(observed)
        k_gof = len(observed)
        df_gof = k_gof - 1
        expected_val = expected
        cramer_v_gof = (
            np.sqrt(chi2 / (n_gof * (k_gof - 1))) if n_gof > 0 and k_gof > 1 else 0
        )

        results_data = {
            "Metric": [
                "Observed A",
                "Observed B",
                "Observed C",
                "Expected (mean)",
                "χ²",
                "df",
                "p-value",
                "Cramer's V",
            ],
            "Value": [
                f"{observed[0]}",
                f"{observed[1]}",
                f"{observed[2]}",
                f"{expected_val:.1f}",
                f"{chi2:.3f}",
                f"{df_gof}",
                f"{p:.5f}",
                f"{cramer_v_gof:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        categories_gof = ["A", "B", "C"]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                name="Observed",
                x=categories_gof,
                y=observed,
                marker_color="rgba(54, 162, 235, 0.7)",
            )
        )
        fig2.add_trace(
            go.Bar(
                name="Expected",
                x=categories_gof,
                y=[expected_val] * 3,
                marker_color="rgba(255, 99, 71, 0.7)",
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Category",
            yaxis_title="Count",
            barmode="group",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Categorial Tests
    elif test_name == "Chi-Square Test":

        from scipy.stats import chi2_contingency

        st.subheader("Interactive Chi-Square Test of Independence")

        # =========================
        # CONTROLS
        # =========================

        a = st.slider("Cell A", 1, 100, 40, key="chi_square_test_cell_a")
        b = st.slider("Cell B", 1, 100, 20, key="chi_square_test_cell_b")
        c = st.slider("Cell C", 1, 100, 10, key="chi_square_test_cell_c")
        d = st.slider("Cell D", 1, 100, 30, key="chi_square_test_cell_d")

        # =========================
        # TABLE
        # =========================

        table = np.array(
            [
                [a, b],
                [c, d],
            ]
        )

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

        from scipy.stats import chi2 as chi2_dist_cs

        n_cs = np.sum(table)
        cramer_v_cs = (
            np.sqrt(chi2 / (n_cs * min(table.shape[0] - 1, table.shape[1] - 1)))
            if n_cs > 0
            else 0
        )

        results_data = {
            "Metric": [
                "χ²",
                "df",
                "p-value",
                "Cramer's V",
                "Cell A (R1,C1)",
                "Cell B (R1,C2)",
                "Cell C (R2,C1)",
                "Cell D (R2,C2)",
            ],
            "Value": [
                f"{chi2:.3f}",
                f"{dof}",
                f"{p:.5f}",
                f"{cramer_v_cs:.4f}",
                f"{table[0, 0]}",
                f"{table[0, 1]}",
                f"{table[1, 0]}",
                f"{table[1, 1]}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        cells_cs = ["(R1,C1)", "(R1,C2)", "(R2,C1)", "(R2,C2)"]
        observed_flat = table.flatten()
        expected_flat = expected.flatten()
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                name="Observed",
                x=cells_cs,
                y=observed_flat,
                marker_color="rgba(54, 162, 235, 0.7)",
            )
        )
        fig2.add_trace(
            go.Bar(
                name="Expected",
                x=cells_cs,
                y=expected_flat,
                marker_color="rgba(255, 99, 71, 0.7)",
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Cell",
            yaxis_title="Count",
            barmode="group",
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "McNemar's Test":

        from statsmodels.stats.contingency_tables import mcnemar

        st.subheader("Interactive McNemar's Test")

        # =========================
        # CONTROLS
        # =========================

        yes_yes = st.slider("Yes → Yes", 0, 100, 40)

        yes_no = st.slider("Yes → No", 0, 100, 10)

        no_yes = st.slider("No → Yes", 0, 100, 30)

        no_no = st.slider("No → No", 0, 100, 20)

        # =========================
        # TABLE
        # =========================

        table = np.array(
            [
                [yes_yes, yes_no],
                [no_yes, no_no],
            ]
        )

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

        b_mc = table[0, 1]
        c_mc = table[1, 0]
        odds_ratio_mc = b_mc / c_mc if c_mc > 0 else float("inf")

        results_data = {
            "Metric": [
                "χ²",
                "p-value",
                "b (Yes→No)",
                "c (No→Yes)",
                "Odds Ratio (b/c)",
                "",
                "",
                "",
            ],
            "Value": [
                f"{result.statistic:.3f}",
                f"{result.pvalue:.5f}",
                f"{b_mc}",
                f"{c_mc}",
                f"{odds_ratio_mc:.3f}" if c_mc > 0 else "∞",
                "",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                name="Discordant Pairs",
                x=["b (Yes→No)", "c (No→Yes)"],
                y=[b_mc, c_mc],
                marker_color=["rgba(255, 99, 71, 0.7)", "rgba(54, 162, 235, 0.7)"],
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Discordant Pair Type",
            yaxis_title="Count",
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Cochran's Q Test":

        from statsmodels.stats.contingency_tables import cochrans_q

        st.subheader("Interactive Cochran's Q Test")

        # =========================
        # CONTROLS
        # =========================

        prob1 = st.slider(
            "Condition 1 Success Probability",
            0.0,
            1.0,
            0.3,
            0.01,
        )

        prob2 = st.slider(
            "Condition 2 Success Probability",
            0.0,
            1.0,
            0.5,
            0.01,
        )

        prob3 = st.slider(
            "Condition 3 Success Probability",
            0.0,
            1.0,
            0.7,
            0.01,
        )

        subjects = st.slider(
            "Subjects",
            10,
            300,
            100,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        c1 = np.random.binomial(1, prob1, subjects)

        c2 = np.random.binomial(1, prob2, subjects)

        c3 = np.random.binomial(1, prob3, subjects)

        data = np.column_stack([c1, c2, c3])

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

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Condition 1", "Condition 2", "Condition 3"],
                y=means,
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            yaxis=dict(range=[0, 1]),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        k_cq = data.shape[1]
        df_cq = k_cq - 1
        proportions = means

        results_data = {
            "Metric": [
                "Q",
                "df",
                "p-value",
                "Proportion C1",
                "Proportion C2",
                "Proportion C3",
                "",
                "",
            ],
            "Value": [
                f"{result.statistic:.3f}",
                f"{df_cq}",
                f"{result.pvalue:.5f}",
                f"{proportions[0]:.3f}",
                f"{proportions[1]:.3f}",
                f"{proportions[2]:.3f}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        cond_names = ["Condition 1", "Condition 2", "Condition 3"]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=cond_names,
                y=proportions,
                marker_color=[
                    "rgba(54, 162, 235, 0.7)",
                    "rgba(255, 99, 71, 0.7)",
                    "rgba(75, 192, 192, 0.7)",
                ],
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

    elif test_name == "Fisher's Exact Test":

        from scipy.stats import fisher_exact

        st.subheader("Interactive Fisher's Exact Test")

        # =========================
        # CONTROLS
        # =========================

        a = st.slider("Cell A", 0, 50, 8, key="fisher_s_exact_test_cell_a")

        b = st.slider("Cell B", 0, 50, 2, key="fisher_s_exact_test_cell_b")

        c = st.slider("Cell C", 0, 50, 1, key="fisher_s_exact_test_cell_c")

        d = st.slider("Cell D", 0, 50, 9, key="fisher_s_exact_test_cell_d")

        # =========================
        # TABLE
        # =========================

        table = np.array(
            [
                [a, b],
                [c, d],
            ]
        )

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

    elif test_name == "Student's t-test (Independent)":

        from scipy.stats import ttest_ind

        st.subheader("Interactive Independent t-test")

        # =========================
        # CONTROLS
        # =========================

        mean_diff = st.slider(
            "Mean Difference",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        sd = st.slider(
            "Shared Standard Deviation",
            0.5,
            5.0,
            1.5,
            0.1,
        )

        n = st.slider(
            "Sample Size per Group",
            10,
            300,
            50,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        group1 = np.random.normal(0, sd, n)

        group2 = np.random.normal(mean_diff, sd, n)

        t, p = ttest_ind(group1, group2)

        # =========================
        # STATS
        # =========================

        st.latex(rf"t = {t:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=group1, name="Group 1"))
        fig.add_trace(go.Box(y=group2, name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st_plot_with_download(fig, key="ttest_indep_box", height=550)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import t as t_dist_student

        n1_s, n2_s = len(group1), len(group2)
        m1_s, m2_s = np.mean(group1), np.mean(group2)
        sd1_s, sd2_s = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_sd = np.sqrt(
            ((n1_s - 1) * sd1_s**2 + (n2_s - 1) * sd2_s**2) / (n1_s + n2_s - 2)
        )
        mean_diff_s = m2_s - m1_s
        se_diff_s = pooled_sd * np.sqrt(1 / n1_s + 1 / n2_s)
        ci_diff_s = se_diff_s * t_dist_student.ppf(0.975, n1_s + n2_s - 2)
        cohens_d_s = mean_diff_s / pooled_sd
        d_lower_s, d_upper_s = cohens_d_independent_ci(cohens_d_s, n1_s, n2_s)
        hedges_g_s = hedges_g(cohens_d_s, n1_s, n2_s)
        d_interp_s = interpret_cohens_d(cohens_d_s)

        results_data = {
            "Metric": [
                "Mean G1 (SD)",
                "Mean G2 (SD)",
                "Mean Difference",
                "95% CI of Diff",
                "Pooled SD",
                "t-statistic",
                "df",
                "p-value",
                "Cohen's d [95% CI]",
                "Hedges' g (unbiased)",
                "Interpretation",
            ],
            "Value": [
                f"{m1_s:.2f} ({sd1_s:.2f})",
                f"{m2_s:.2f} ({sd2_s:.2f})",
                f"{mean_diff_s:.3f}",
                f"[{mean_diff_s - ci_diff_s:.3f}, {mean_diff_s + ci_diff_s:.3f}]",
                f"{pooled_sd:.3f}",
                f"{t:.3f}",
                f"{n1_s + n2_s - 2}",
                format_p_value(p),
                format_effect_size_with_ci(cohens_d_s, d_lower_s, d_upper_s),
                f"{hedges_g_s:.3f}",
                d_interp_s,
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(zip([group1, group2], ["Group 1", "Group 2"])):
            fig2.add_trace(
                go.Violin(
                    y=g,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st_plot_with_download(fig2, key="ttest_indep_violin", height=550)

    elif test_name == "Welch's t-test (Independent, Unequal Variances)":

        from scipy.stats import ttest_ind

        st.subheader("Interactive Welch's t-test")

        # =========================
        # CONTROLS
        # =========================

        mean_diff = st.slider("Mean Difference", 0.0, 10.0, 2.0, 0.1, key="mean_difference_1")

        sd1 = st.slider("Group 1 SD", 0.5, 8.0, 1.0, 0.1)

        sd2 = st.slider("Group 2 SD", 0.5, 8.0, 3.0, 0.1)

        n = st.slider(
            "Sample Size",
            10,
            300,
            50,
            key="welch_s_t_test_independent_unequal_variances_sample_size",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        g1 = np.random.normal(0, sd1, n)

        g2 = np.random.normal(mean_diff, sd2, n)

        t, p = ttest_ind(g1, g2, equal_var=False)

        # =========================
        # STATS
        # =========================

        st.latex(rf"t = {t:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Violin(y=g1, name="Group 1"))

        fig.add_trace(go.Violin(y=g2, name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st_plot_with_download(fig, key="welch_ttest_violin", height=550)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import t as t_dist_welch

        n1_w, n2_w = len(g1), len(g2)
        m1_w, m2_w = np.mean(g1), np.mean(g2)
        sd1_w, sd2_w = np.std(g1, ddof=1), np.std(g2, ddof=1)
        mean_diff_w = m2_w - m1_w
        se_w = np.sqrt(sd1_w**2 / n1_w + sd2_w**2 / n2_w)

        welch_df_num = (sd1_w**2 / n1_w + sd2_w**2 / n2_w) ** 2
        welch_df_den = (sd1_w**2 / n1_w) ** 2 / (n1_w - 1) + (
            sd2_w**2 / n2_w
        ) ** 2 / (n2_w - 1)
        welch_df = welch_df_num / welch_df_den

        ci_diff_w = se_w * t_dist_welch.ppf(0.975, welch_df)

        pooled_sd_w = np.sqrt((sd1_w**2 + sd2_w**2) / 2)
        cohens_d_w = mean_diff_w / pooled_sd_w
        d_lower_w, d_upper_w = cohens_d_independent_ci(cohens_d_w, n1_w, n2_w)
        hedges_g_w = hedges_g(cohens_d_w, n1_w, n2_w)
        d_interp_w = interpret_cohens_d(cohens_d_w)

        results_data = {
            "Metric": [
                "Mean G1 (SD)",
                "Mean G2 (SD)",
                "Mean Difference",
                "95% CI of Diff",
                "t-statistic",
                "Welch df",
                "p-value",
                "Cohen's d [95% CI]",
                "Hedges' g (unbiased)",
                "Interpretation",
            ],
            "Value": [
                f"{m1_w:.2f} ({sd1_w:.2f})",
                f"{m2_w:.2f} ({sd2_w:.2f})",
                f"{mean_diff_w:.3f}",
                f"[{mean_diff_w - ci_diff_w:.3f}, {mean_diff_w + ci_diff_w:.3f}]",
                f"{t:.3f}",
                f"{welch_df:.1f}",
                format_p_value(p),
                format_effect_size_with_ci(cohens_d_w, d_lower_w, d_upper_w),
                f"{hedges_g_w:.3f}",
                d_interp_w,
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(zip([g1, g2], ["Group 1", "Group 2"])):
            fig2.add_trace(
                go.Violin(
                    y=g,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st_plot_with_download(fig2, key="welch_ttest_strip", height=550)

    elif test_name == "Paired t-test":

        from scipy.stats import ttest_rel

        st.subheader("Interactive Paired t-test")

        # =========================
        # CONTROLS
        # =========================

        effect = st.slider(
            "Treatment Effect",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        n = st.slider(
            "Number of Subjects", 10, 200, 40, key="paired_t_test_number_of_subjects"
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        before = np.random.normal(10, noise, n)

        after = before + effect + np.random.normal(0, noise, n)

        t, p = ttest_rel(before, after)

        # =========================
        # STATS
        # =========================

        st.latex(rf"t = {t:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i in range(n):

            fig.add_trace(
                go.Scatter(
                    x=["Before", "After"],
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
        )

        st_plot_with_download(fig, key="paired_ttest_spaghetti", height=600)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import t as t_dist_paired

        mean_pre = np.mean(before)
        mean_post = np.mean(after)
        sd_pre = np.std(before, ddof=1)
        sd_post = np.std(after, ddof=1)
        diffs = after - before
        mean_diff_p = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        n_p = len(before)
        se_diff = sd_diff / np.sqrt(n_p)
        ci_diff_paired = se_diff * t_dist_paired.ppf(0.975, n_p - 1)
        cohens_dz = mean_diff_p / sd_diff
        dz_lower, dz_upper = cohens_d_one_sample_ci(cohens_dz, n_p)
        hedges_g_paired = hedges_g(cohens_dz, n_p)
        dz_interp = interpret_cohens_d(cohens_dz)

        results_data = {
            "Metric": [
                "Mean Pre (SD)",
                "Mean Post (SD)",
                "Mean Difference",
                "95% CI of Diff",
                "t-statistic",
                "df",
                "p-value",
                "Cohen's d_z [95% CI]",
                "Hedges' g (unbiased)",
                "Interpretation",
            ],
            "Value": [
                f"{mean_pre:.2f} ({sd_pre:.2f})",
                f"{mean_post:.2f} ({sd_post:.2f})",
                f"{mean_diff_p:.3f}",
                f"[{mean_diff_p - ci_diff_paired:.3f}, {mean_diff_p + ci_diff_paired:.3f}]",
                f"{t:.3f}",
                f"{n_p - 1}",
                format_p_value(p),
                format_effect_size_with_ci(cohens_dz, dz_lower, dz_upper),
                f"{hedges_g_paired:.3f}",
                dz_interp,
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i in range(n_p):
            fig2.add_trace(
                go.Scatter(
                    x=["Before", "After"],
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=["Before", "After"],
                y=[mean_pre, mean_post],
                mode="lines+markers",
                name="Mean change",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=12),
            )
        )

        fig2.add_hline(
            y=mean_pre,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"Pre Mean = {mean_pre:.2f}",
        )

        fig2.add_hrect(
            y0=mean_diff_p - ci_diff_paired,
            y1=mean_diff_p + ci_diff_paired,
            x0=-0.5,
            x1=1.5,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            name="95% CI of difference",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time Point",
            yaxis_title="Value",
        )

        st_plot_with_download(fig2, key="paired_ttest_profile", height=600)

    # Parametric Multiple Group Tests

    elif test_name == "One-way ANOVA":

        from scipy.stats import f_oneway

        st.subheader("Interactive One-way ANOVA")

        # =========================
        # CONTROLS
        # =========================

        mean_shift = st.slider(
            "Group Separation",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        noise = st.slider(
            "Within-group Variability",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        g1 = np.random.normal(0, noise, 60)
        g2 = np.random.normal(mean_shift, noise, 60)
        g3 = np.random.normal(mean_shift * 2, noise, 60)

        F, p = f_oneway(g1, g2, g3)

        # =========================
        # STATS
        # =========================

        st.latex(rf"F = {F:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=g1, name="Group 1"))
        fig.add_trace(go.Box(y=g2, name="Group 2"))
        fig.add_trace(go.Box(y=g3, name="Group 3"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st_plot_with_download(fig, key="oneway_anova_box", height=550)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import f as f_dist_1w

        groups_1w = [g1, g2, g3]
        means_1w = [np.mean(g) for g in groups_1w]
        sds_1w = [np.std(g, ddof=1) for g in groups_1w]
        n_1w = [len(g) for g in groups_1w]
        n_total_1w = sum(n_1w)
        k_1w = len(groups_1w)
        grand_mean_1w = np.mean(np.concatenate(groups_1w))

        ss_between = sum(
            n_i * (m_i - grand_mean_1w) ** 2 for n_i, m_i in zip(n_1w, means_1w)
        )
        ss_within = sum((n_i - 1) * sd_i**2 for n_i, sd_i in zip(n_1w, sds_1w))
        df_between = k_1w - 1
        df_within = n_total_1w - k_1w
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        F_1w = ms_between / ms_within
        p_1w = 1 - f_dist_1w.cdf(F_1w, df_between, df_within)
        eta_sq = ss_between / (ss_between + ss_within)
        omega_sq = (ss_between - df_between * ms_within) / (
            ss_between + ss_within + ms_within
        )
        eta_interp = interpret_eta_squared(eta_sq)

        results_data = {
            "Metric": [
                "Mean G1 (SD)",
                "Mean G2 (SD)",
                "Mean G3 (SD)",
                "F",
                f"df ({df_between}, {df_within})",
                "p-value",
                "η²",
                "ω² (unbiased)",
                "Interpretation",
            ],
            "Value": [
                f"{means_1w[0]:.2f} ({sds_1w[0]:.2f})",
                f"{means_1w[1]:.2f} ({sds_1w[1]:.2f})",
                f"{means_1w[2]:.2f} ({sds_1w[2]:.2f})",
                f"{F_1w:.3f}",
                f"{df_between}, {df_within}",
                format_p_value(p_1w),
                f"{eta_sq:.4f}",
                f"{max(0, omega_sq):.4f}",
                eta_interp,
            ],
        }
        st.table(pd.DataFrame(results_data))

        with st.expander("Note: ω² (Omega-squared) is preferred over η²"):
            st.markdown("""
            **η² (Eta-squared)** measures the proportion of variance explained, but it is **positively biased** — it tends to overestimate the true population effect size, especially in small samples.

            **ω² (Omega-squared)** applies a correction that makes it **unbiased**. It is:
            - Always smaller than η² (except when the true effect is exactly 0)
            - Recommended by the APA 7th edition for reporting ANOVA results
            - Less likely to capitalize on chance in small samples

            **Interpretation (Cohen, 1988):**
            - Small: ω² ≈ 0.01 (1% of variance explained)
            - Medium: ω² ≈ 0.06 (6% of variance explained)
            - Large: ω² ≈ 0.14 (14%+ of variance explained)
            """)

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(
            zip(groups_1w, ["Group 1", "Group 2", "Group 3"])
        ):
            fig2.add_trace(
                go.Violin(
                    y=g,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.add_hline(
            y=grand_mean_1w,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Grand Mean = {grand_mean_1w:.2f}",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st_plot_with_download(fig2, key="oneway_anova_violin", height=550)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc([g1, g2, g3], param_type="parametric", key="anova_ph")

    elif test_name == "Two-way ANOVA":

        from scipy.stats import f_oneway
        from scipy.stats import f as f_dist

        st.subheader("Interactive Two-way ANOVA")

        # =========================
        # CONTROLS
        # =========================

        effect_A = st.slider("Effect of Factor A (Group)", 0.0, 10.0, 2.0, 0.1, key="tw_effect_A")

        effect_B = st.slider("Effect of Factor B (Sex)", 0.0, 10.0, 1.0, 0.1, key="tw_effect_B")

        interaction = st.slider("Interaction (A × B)", -5.0, 5.0, 0.0, 0.1, key="tw_interaction")

        noise = st.slider("Within-group Variability", 0.1, 5.0, 1.0, 0.1, key="tw_noise")

        # =========================
        # DATA
        # =========================

        np.random.seed(42)
        n_per_cell = 30

        A1B1 = np.random.normal(0, noise, n_per_cell)
        A1B2 = np.random.normal(effect_B, noise, n_per_cell)
        A2B1 = np.random.normal(effect_A, noise, n_per_cell)
        A2B2 = np.random.normal(effect_A + effect_B + interaction, noise, n_per_cell)

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=A1B1, name="A1 (Control), B1 (Male)"))
        fig.add_trace(go.Box(y=A1B2, name="A1 (Control), B2 (Female)"))
        fig.add_trace(go.Box(y=A2B1, name="A2 (Drug), B1 (Male)"))
        fig.add_trace(go.Box(y=A2B2, name="A2 (Drug), B2 (Female)"))

        fig.update_layout(template="plotly_dark", height=550)

        st_plot_with_download(fig, key="twoway_anova_box", height=550)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from statsmodels.formula.api import ols as ols_tw
        import statsmodels.api as sm_tw

        df_tw = pd.DataFrame(
            {
                "y": np.concatenate([A1B1, A1B2, A2B1, A2B2]),
                "A": np.repeat(["A1", "A1", "A2", "A2"], n_per_cell),
                "B": np.repeat(["B1", "B2", "B1", "B2"], n_per_cell),
            }
        )
        model_tw = ols_tw("y ~ C(A) * C(B)", data=df_tw).fit()
        anova_tw = sm_tw.stats.anova_lm(model_tw, typ=2)

        F_A = anova_tw.loc["C(A)", "F"]
        p_A = anova_tw.loc["C(A)", "PR(>F)"]
        F_B = anova_tw.loc["C(B)", "F"]
        p_B = anova_tw.loc["C(B)", "PR(>F)"]
        F_AB = anova_tw.loc["C(A):C(B)", "F"]
        p_AB = anova_tw.loc["C(A):C(B)", "PR(>F)"]

        ss_A = anova_tw.loc["C(A)", "sum_sq"]
        ss_B = anova_tw.loc["C(B)", "sum_sq"]
        ss_AB = anova_tw.loc["C(A):C(B)", "sum_sq"]
        ss_resid_tw = anova_tw.loc["Residual", "sum_sq"]
        df_A = anova_tw.loc["C(A)", "df"]
        df_B = anova_tw.loc["C(B)", "df"]
        df_AB = anova_tw.loc["C(A):C(B)", "df"]
        ms_resid_tw = anova_tw.loc["Residual", "sum_sq"] / anova_tw.loc["Residual", "df"]
        n_total_tw = len(df_tw)

        partial_eta_A = ss_A / (ss_A + ss_resid_tw)
        partial_eta_B = ss_B / (ss_B + ss_resid_tw)
        partial_eta_AB = ss_AB / (ss_AB + ss_resid_tw)

        partial_omega_A = omega_squared_partial(ss_A, int(df_A), ss_resid_tw, ms_resid_tw, n_total_tw)
        partial_omega_B = omega_squared_partial(ss_B, int(df_B), ss_resid_tw, ms_resid_tw, n_total_tw)
        partial_omega_AB = omega_squared_partial(ss_AB, int(df_AB), ss_resid_tw, ms_resid_tw, n_total_tw)

        results_data = {
            "Metric": [
                "Mean A1B1",
                "Mean A1B2",
                "Mean A2B1",
                "Mean A2B2",
                "F_A",
                "p_A",
                "Partial η²_A",
                "Partial ω²_A",
                "F_B",
                "p_B",
                "Partial η²_B",
                "Partial ω²_B",
                "F_AB",
                "p_AB",
                "Partial η²_AB",
                "Partial ω²_AB",
            ],
            "Value": [
                f"{np.mean(A1B1):.3f}",
                f"{np.mean(A1B2):.3f}",
                f"{np.mean(A2B1):.3f}",
                f"{np.mean(A2B2):.3f}",
                f"{F_A:.3f}",
                format_p_value(p_A),
                f"{partial_eta_A:.4f}",
                f"{partial_omega_A:.4f}",
                f"{F_B:.3f}",
                format_p_value(p_B),
                f"{partial_eta_B:.4f}",
                f"{partial_omega_B:.4f}",
                f"{F_AB:.3f}",
                format_p_value(p_AB),
                f"{partial_eta_AB:.4f}",
                f"{partial_omega_AB:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        with st.expander("Note: Partial ω² (Omega-squared) vs η² (Eta-squared)"):
            st.markdown("""
            **Partial η²** is the most commonly reported effect size for ANOVA, but it is **biased upward** in small samples.

            **Partial ω²** is an **unbiased estimator** that:
            - Corrects for positive bias in η²
            - Is always smaller than η²
            - Is preferred by APA 7th edition when sample sizes are small

            **Interpretation (Cohen, 1988):**
            - Small: ω² ≈ 0.01
            - Medium: ω² ≈ 0.06
            - Large: ω² ≈ 0.14
            """)

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        cell_means = {
            "A1": {"B1": np.mean(A1B1), "B2": np.mean(A1B2)},
            "A2": {"B1": np.mean(A2B1), "B2": np.mean(A2B2)},
        }
        for level_B, color, dash in [("B1", "blue", "solid"), ("B2", "red", "dash")]:
            means = [cell_means[a][level_B] for a in ["A1", "A2"]]
            fig2.add_trace(
                go.Scatter(
                    x=["A1", "A2"],
                    y=means,
                    mode="lines+markers",
                    name=f"Factor B: {level_B}",
                    line=dict(color=color, width=3, dash=dash),
                    marker=dict(size=10),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Factor A",
            yaxis_title="Cell Mean",
        )

        st_plot_with_download(fig2, key="twoway_anova_interaction", height=500)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc([A1B1, A1B2, A2B1, A2B2], param_type="parametric", key="tw_anova_ph")

    elif test_name == "ANCOVA":

        from scipy.stats import linregress

        st.subheader("Interactive ANCOVA")

        # =========================
        # CONTROLS
        # =========================

        treatment_effect = st.slider("Treatment Effect", 0.0, 10.0, 3.0, 0.1, key="treatment_effect_1")

        covariate_strength = st.slider("Covariate Strength (β)", 0.0, 3.0, 1.0, 0.1)

        noise = st.slider("Noise", 0.1, 5.0, 1.0, 0.1, key="ancova_noise")

        # =========================
        # DATA
        # =========================

        np.random.seed(42)
        n = 40

        covariate = np.random.normal(50, 10, n)

        control = covariate * covariate_strength + np.random.normal(0, noise, n)

        treatment = (
            covariate * covariate_strength
            + treatment_effect
            + np.random.normal(0, noise, n)
        )

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=covariate,
                y=control,
                mode="markers",
                name="Control",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=covariate,
                y=treatment,
                mode="markers",
                name="Treatment",
            )
        )

        # Regression lines
        slope_c, intercept_c, _, _, _ = linregress(covariate, control)
        slope_t, intercept_t, _, _, _ = linregress(covariate, treatment)

        x_line = np.linspace(covariate.min(), covariate.max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept_c + slope_c * x_line,
                mode="lines",
                name="Control (adjusted)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept_t + slope_t * x_line,
                mode="lines",
                name="Treatment (adjusted)",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Covariate (Baseline)",
            yaxis_title="Outcome",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from statsmodels.formula.api import ols as ols_anc

        df_anc = pd.DataFrame(
            {
                "outcome": np.concatenate([control, treatment]),
                "group": np.repeat(["Control", "Treatment"], n),
                "cov": np.tile(covariate, 2),
            }
        )
        model_anc = ols_anc("outcome ~ C(group) + cov", data=df_anc).fit()
        beta_cov = model_anc.params["cov"]
        grand_mean_cov = np.mean(covariate)
        adj_control = model_anc.params["Intercept"] + beta_cov * grand_mean_cov
        adj_treatment = (
            model_anc.params["Intercept"]
            + model_anc.params["C(group)[T.Treatment]"]
            + beta_cov * grand_mean_cov
        )
        F_group = model_anc.fvalue
        p_group = model_anc.f_pvalue
        ss_resid = model_anc.ssr
        ss_expl = model_anc.ess
        partial_eta_anc = ss_expl / (ss_expl + ss_resid)

        results_data = {
            "Metric": [
                "Adj. Mean Control",
                "Adj. Mean Treatment",
                "F (group)",
                "p-value",
                "Covariate β",
                "Partial η²",
                "",
                "",
            ],
            "Value": [
                f"{adj_control:.3f}",
                f"{adj_treatment:.3f}",
                f"{F_group:.3f}",
                f"{p_group:.5f}",
                f"{beta_cov:.3f}",
                f"{partial_eta_anc:.4f}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=covariate,
                y=control,
                mode="markers",
                name="Control",
                marker=dict(color="rgba(0, 123, 255, 0.6)", size=6),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=covariate,
                y=treatment,
                mode="markers",
                name="Treatment",
                marker=dict(color="rgba(255, 65, 54, 0.6)", size=6),
            )
        )

        slope_c, intercept_c, _, _, _ = linregress(covariate, control)
        slope_t, intercept_t, _, _, _ = linregress(covariate, treatment)
        x_line = np.linspace(covariate.min(), covariate.max(), 100)
        fig2.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept_c + slope_c * x_line,
                mode="lines",
                name="Control (adjusted)",
                line=dict(color="blue", width=2, dash="dash"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept_t + slope_t * x_line,
                mode="lines",
                name="Treatment (adjusted)",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig2.add_vline(
            x=grand_mean_cov, line_dash="dot", line_color="gray", opacity=0.5
        )
        fig2.add_annotation(
            x=grand_mean_cov,
            y=adj_control,
            text=f"Adj. Control: {adj_control:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=-30,
            bgcolor="blue",
        )
        fig2.add_annotation(
            x=grand_mean_cov,
            y=adj_treatment,
            text=f"Adj. Treatment: {adj_treatment:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=30,
            bgcolor="red",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Covariate (Baseline)",
            yaxis_title="Outcome",
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc([control, treatment], param_type="parametric", key="ancova_ph")

    elif test_name == "Repeated Measures ANOVA":

        st.subheader("Interactive Repeated Measures ANOVA")

        # =========================
        # CONTROLS
        # =========================

        trend = st.slider(
            "Time Trend",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
            key="noise_2",
        )

        subjects = st.slider(
            "Subjects",
            5,
            100,
            20,
            key="subjects_1",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        timepoints = 4

        data = []

        for s in range(subjects):

            baseline = np.random.normal(10, noise)

            vals = [
                baseline + trend * t + np.random.normal(0, noise)
                for t in range(timepoints)
            ]

            data.append(vals)

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for vals in data:

            fig.add_trace(
                go.Scatter(
                    x=[1, 2, 3, 4],
                    y=vals,
                    mode="lines+markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time",
            yaxis_title="Measurement",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        time_means = [
            np.mean([data[s][t] for s in range(subjects)]) for t in range(timepoints)
        ]

        df_rm = pd.DataFrame(
            {
                "subject": np.tile(range(subjects), timepoints),
                "time": np.repeat(range(timepoints), subjects),
                "y": np.array(data).T.flatten(),
            }
        )

        from statsmodels.stats.anova import AnovaRM

        rm_result = AnovaRM(df_rm, "y", "subject", within=["time"]).fit()
        rm_table = rm_result.anova_table
        F_rm = rm_table.loc["time", "F Value"]
        p_rm = rm_table.loc["time", "Pr > F"]
        df_num_rm = int(rm_table.loc["time", "Num DF"])
        df_den_rm = int(rm_table.loc["time", "Den DF"])
        partial_eta_sq_rm = (
            rm_table.loc["time", "F Value"]
            * df_num_rm
            / (rm_table.loc["time", "F Value"] * df_num_rm + df_den_rm)
        )

        results_data = {
            "Metric": [
                "Mean T1",
                "Mean T2",
                "Mean T3",
                "Mean T4",
                "F",
                f"df ({df_num_rm}, {df_den_rm})",
                "p-value",
                "Partial η²",
            ],
            "Value": [
                f"{time_means[0]:.3f}",
                f"{time_means[1]:.3f}",
                f"{time_means[2]:.3f}",
                f"{time_means[3]:.3f}",
                f"{F_rm:.3f}",
                f"{df_num_rm}, {df_den_rm}",
                f"{p_rm:.5f}",
                f"{partial_eta_sq_rm:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for vals in data:
            fig2.add_trace(
                go.Scatter(
                    x=[1, 2, 3, 4],
                    y=vals,
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4],
                y=time_means,
                mode="lines+markers",
                name="Mean trend",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=10),
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time",
            yaxis_title="Measurement",
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Post-Hoc Tests (Across Timepoints)")
        time_groups = [np.array([row[t] for row in data]) for t in range(timepoints)]
        render_post_hoc(time_groups, param_type="parametric", key="rm_anova_ph")

    elif test_name == "MANOVA":

        st.subheader("Interactive MANOVA")

        # =========================
        # CONTROLS
        # =========================

        separation = st.slider(
            "Group Separation",
            0.0,
            10.0,
            3.0,
            0.1,
            key="group_separation_1",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        n = 100

        g1 = np.random.multivariate_normal(
            [0, 0, 0],
            np.eye(3),
            n,
        )

        g2 = np.random.multivariate_normal(
            [separation, separation, separation],
            np.eye(3),
            n,
        )

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=g1[:, 0],
                y=g1[:, 1],
                z=g1[:, 2],
                mode="markers",
                name="Group 1",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=g2[:, 0],
                y=g2[:, 1],
                z=g2[:, 2],
                mode="markers",
                name="Group 2",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=700,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        n_m = len(g1)
        y_all = np.vstack([g1, g2])
        grand_mean = np.mean(y_all, axis=0)
        mean1 = np.mean(g1, axis=0)
        mean2 = np.mean(g2, axis=0)

        H_m = n_m * np.outer(mean1 - grand_mean, mean1 - grand_mean) + n_m * np.outer(
            mean2 - grand_mean, mean2 - grand_mean
        )
        E_m = (n_m - 1) * np.cov(g1, rowvar=False) + (n_m - 1) * np.cov(
            g2, rowvar=False
        )
        wilks_lambda = np.linalg.det(E_m) / np.linalg.det(E_m + H_m)

        p_manova = 3
        k_manova = 2
        df1_m = p_manova
        df2_m = 2 * n_m - p_manova - 1
        F_m = ((1 - wilks_lambda) / wilks_lambda) * (df2_m / df1_m)

        from scipy.stats import f as f_dist_m

        p_m = 1 - f_dist_m.cdf(F_m, df1_m, df2_m)

        results_data = {
            "Metric": [
                "Wilks' Λ",
                "F-approx",
                f"df ({df1_m}, {df2_m:.0f})",
                "p-value",
                "Number of DVs",
                "Number of Groups",
                "",
                "",
            ],
            "Value": [
                f"{wilks_lambda:.4f}",
                f"{F_m:.3f}",
                f"{df1_m}, {df2_m:.0f}",
                f"{p_m:.5f}",
                f"{p_manova}",
                f"{k_manova}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        fig2 = go.Figure()
        for idx, (g, name, color) in enumerate(
            zip([g1, g2], ["Group 1", "Group 2"], ["blue", "red"])
        ):
            fig2.add_trace(
                go.Scatter(
                    x=g[:, 0],
                    y=g[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=6, opacity=0.5),
                )
            )
            centroid = np.mean(g[:, :2], axis=0)
            fig2.add_trace(
                go.Scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color=color, size=15, symbol="x"),
                )
            )
        fig2.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc([g1[:, 0], g1[:, 1], g1[:, 2]], param_type="parametric", key="manova_ph")

    # Non-parametric Two Sample Tests

    elif test_name == "Wilcoxon Signed-Rank Test":

        from scipy.stats import wilcoxon

        st.subheader("Interactive Wilcoxon Signed-Rank Test")

        # =========================
        # CONTROLS
        # =========================

        median_shift = st.slider(
            "Median Shift",
            -5.0,
            5.0,
            1.0,
            0.1,
            key="median_shift_1",
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
            key="noise_3",
        )

        n = st.slider(
            "Sample Size",
            10,
            200,
            40,
            key="sample_size_2",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        before = np.random.exponential(1, n)

        after = before + median_shift + np.random.normal(0, noise, n)

        stat, p = wilcoxon(before, after)

        # =========================
        # STATS
        # =========================

        st.latex(rf"W = {stat:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i in range(n):

            fig.add_trace(
                go.Scatter(
                    x=["Before", "After"],
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        median_pre = np.median(before)
        median_post = np.median(after)
        median_diff = median_post - median_pre
        from scipy.stats import norm as norm_w

        z_w = -norm_w.ppf(p / 2)
        r_ws = z_w / np.sqrt(n) if n > 0 else 0

        results_data = {
            "Metric": [
                "Median (Pre)",
                "Median (Post)",
                "Median Difference",
                "W-statistic",
                "z",
                "p-value",
                "Rank-biserial r",
                "",
            ],
            "Value": [
                f"{median_pre:.3f}",
                f"{median_post:.3f}",
                f"{median_diff:.3f}",
                f"{stat:.3f}",
                f"{z_w:.3f}",
                f"{p:.5f}",
                f"{r_ws:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i in range(n):
            fig2.add_trace(
                go.Scatter(
                    x=["Before", "After"],
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=["Before", "After"],
                y=[median_pre, median_post],
                mode="lines+markers",
                name="Median change",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=12),
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time Point",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Mann-Whitney U Test":

        from scipy.stats import mannwhitneyu

        st.subheader("Interactive Mann-Whitney U Test")

        # =========================
        # CONTROLS
        # =========================

        location_shift = st.slider(
            "Distribution Shift",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        spread = st.slider(
            "Distribution Spread",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        n = st.slider(
            "Sample Size",
            10,
            300,
            60,
            key="sample_size_3",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        g1 = np.random.exponential(spread, n)

        g2 = np.random.exponential(spread, n) + location_shift

        u, p = mannwhitneyu(g1, g2)

        # =========================
        # STATS
        # =========================

        st.latex(rf"U = {u:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Violin(y=g1, name="Group 1"))
        fig.add_trace(go.Violin(y=g2, name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        medians_mw = [np.median(g) for g in [g1, g2]]
        iqr_mw = [
            f"{np.percentile(g, 25):.2f}–{np.percentile(g, 75):.2f}" for g in [g1, g2]
        ]
        from scipy.stats import norm as norm_mw

        z_mw = -norm_mw.ppf(p / 2)
        r_rb_mw = 1 - 2 * min(u, n * n - u) / (n * n)

        results_data = {
            "Metric": [
                "Median G1 (IQR)",
                "Median G2 (IQR)",
                "U-statistic",
                "z",
                "p-value",
                "Rank-biserial r",
                "",
                "",
            ],
            "Value": [
                f"{medians_mw[0]:.3f} ({iqr_mw[0]})",
                f"{medians_mw[1]:.3f} ({iqr_mw[1]})",
                f"{u:.3f}",
                f"{z_mw:.3f}",
                f"{p:.5f}",
                f"{r_rb_mw:.4f}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(zip([g1, g2], ["Group 1", "Group 2"])):
            fig2.add_trace(
                go.Violin(y=g, name=name, box_visible=True, meanline_visible=True)
            )
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Non-parametric Multiple Group Tests

    elif test_name == "Kruskal-Wallis Test":

        from scipy.stats import kruskal

        st.subheader("Interactive Kruskal-Wallis Test")

        # =========================
        # CONTROLS
        # =========================

        shift = st.slider(
            "Group Separation",
            0.0,
            10.0,
            2.0,
            0.1,
            key="group_separation_2",
        )

        spread = st.slider(
            "Distribution Spread",
            0.1,
            5.0,
            1.0,
            0.1,
            key="distribution_spread_1",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        g1 = np.random.gamma(2, spread, 60)

        g2 = np.random.gamma(2, spread, 60) + shift

        g3 = np.random.gamma(2, spread, 60) + shift * 2

        H, p = kruskal(g1, g2, g3)

        # =========================
        # STATS
        # =========================

        st.latex(rf"H = {H:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=g1, name="Group 1"))
        fig.add_trace(go.Box(y=g2, name="Group 2"))
        fig.add_trace(go.Box(y=g3, name="Group 3"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        n_kw = [len(g1), len(g2), len(g3)]
        medians_kw = [np.median(g) for g in [g1, g2, g3]]
        iqr_kw = [
            f"{np.percentile(g, 25):.2f}–{np.percentile(g, 75):.2f}"
            for g in [g1, g2, g3]
        ]
        n_total_kw = sum(n_kw)
        eps_sq = H / (n_total_kw - 1) if n_total_kw > 1 else 0

        results_data = {
            "Metric": [
                "Median G1 (IQR)",
                "Median G2 (IQR)",
                "Median G3 (IQR)",
                "H",
                "df",
                "p-value",
                "ε²",
                "",
            ],
            "Value": [
                f"{medians_kw[0]:.3f} ({iqr_kw[0]})",
                f"{medians_kw[1]:.3f} ({iqr_kw[1]})",
                f"{medians_kw[2]:.3f} ({iqr_kw[2]})",
                f"{H:.3f}",
                "2",
                f"{p:.5f}",
                f"{eps_sq:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(
            zip([g1, g2, g3], ["Group 1", "Group 2", "Group 3"])
        ):
            fig2.add_trace(go.Box(y=g, name=name, boxmean="sd"))
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc([g1, g2, g3], param_type="nonparametric", key="kw_ph")

    elif test_name == "Friedman Test":

        from scipy.stats import friedmanchisquare

        st.subheader("Interactive Friedman Test")

        # =========================
        # CONTROLS
        # =========================

        trend = st.slider(
            "Repeated Trend",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
            key="noise_4",
        )

        subjects = st.slider(
            "Subjects",
            5,
            100,
            20,
            key="subjects_2",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        t1 = np.random.exponential(1, subjects)

        t2 = t1 + trend + np.random.normal(0, noise, subjects)

        t3 = t2 + trend + np.random.normal(0, noise, subjects)

        stat, p = friedmanchisquare(t1, t2, t3)

        # =========================
        # STATS
        # =========================

        st.latex(rf"\chi^2 = {stat:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i in range(subjects):

            fig.add_trace(
                go.Scatter(
                    x=["T1", "T2", "T3"],
                    y=[t1[i], t2[i], t3[i]],
                    mode="lines+markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        medians_f = [np.median(t1), np.median(t2), np.median(t3)]
        kendall_w = stat / (subjects * (3 - 1))

        results_data = {
            "Metric": [
                "Median T1",
                "Median T2",
                "Median T3",
                "χ²",
                "df",
                "p-value",
                "Kendall's W",
                "",
            ],
            "Value": [
                f"{medians_f[0]:.3f}",
                f"{medians_f[1]:.3f}",
                f"{medians_f[2]:.3f}",
                f"{stat:.3f}",
                "2",
                f"{p:.5f}",
                f"{kendall_w:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i in range(subjects):
            fig2.add_trace(
                go.Scatter(
                    x=["T1", "T2", "T3"],
                    y=[t1[i], t2[i], t3[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=["T1", "T2", "T3"],
                y=medians_f,
                mode="lines+markers",
                name="Median trend",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=10),
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time Point",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc([t1, t2, t3], param_type="nonparametric", key="friedman_ph")

    elif test_name == "Permutation MANOVA or Non-Parametric MANOVA":

        st.subheader("Interactive Permutation MANOVA")

        # =========================
        # CONTROLS
        # =========================

        separation = st.slider(
            "Cluster Separation",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        dispersion = st.slider(
            "Cluster Dispersion",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        n = 150

        g1 = np.random.multivariate_normal(
            [0, 0],
            np.eye(2) * dispersion,
            n,
        )

        g2 = np.random.multivariate_normal(
            [separation, separation],
            np.eye(2) * dispersion,
            n,
        )

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=g1[:, 0],
                y=g1[:, 1],
                mode="markers",
                name="Group 1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=g2[:, 0],
                y=g2[:, 1],
                mode="markers",
                name="Group 2",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        n_pm = len(g1)
        y_all_pm = np.vstack([g1, g2])
        grand_mean_pm = np.mean(y_all_pm, axis=0)

        SSt_pm = np.sum((y_all_pm - grand_mean_pm) ** 2)
        mean1_pm = np.mean(g1, axis=0)
        mean2_pm = np.mean(g2, axis=0)
        SSb_pm = n_pm * np.sum((mean1_pm - grand_mean_pm) ** 2) + n_pm * np.sum(
            (mean2_pm - grand_mean_pm) ** 2
        )
        SSw_pm = SSt_pm - SSb_pm

        k_pm = 2
        N_pm = 2 * n_pm
        pseudo_F = (SSb_pm / (k_pm - 1)) / (SSw_pm / (N_pm - k_pm))
        R2_pm = SSb_pm / SSt_pm

        n_perms = 199
        pseudo_Fs = np.zeros(n_perms)
        combined = y_all_pm.copy()
        for perm in range(n_perms):
            np.random.shuffle(combined)
            perm_g1 = combined[:n_pm]
            perm_g2 = combined[n_pm:]
            perm_mean1 = np.mean(perm_g1, axis=0)
            perm_mean2 = np.mean(perm_g2, axis=0)
            perm_grand = np.mean(combined, axis=0)
            perm_SSb = n_pm * np.sum((perm_mean1 - perm_grand) ** 2) + n_pm * np.sum(
                (perm_mean2 - perm_grand) ** 2
            )
            perm_SSw = np.sum((combined - perm_grand) ** 2) - perm_SSb
            pseudo_Fs[perm] = (perm_SSb / (k_pm - 1)) / (perm_SSw / (N_pm - k_pm))

        p_pm = (np.sum(pseudo_Fs >= pseudo_F) + 1) / (n_perms + 1)

        results_data = {
            "Metric": ["Pseudo-F", "R²", "Permutations", "p-value"],
            "Value": [
                f"{pseudo_F:.3f}",
                f"{R2_pm:.4f}",
                f"{n_perms + 1}",
                f"{p_pm:.5f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        fig2 = go.Figure()
        for idx, (g, name, color) in enumerate(
            zip([g1, g2], ["Group 1", "Group 2"], ["blue", "red"])
        ):
            fig2.add_trace(
                go.Scatter(
                    x=g[:, 0],
                    y=g[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=6, opacity=0.5),
                )
            )
            mean_pos = np.mean(g, axis=0)
            cov_pos = np.cov(g, rowvar=False)
            theta = np.linspace(0, 2 * np.pi, 100)
            eigvals, eigvecs = np.linalg.eigh(cov_pos)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            ellipse = (
                np.column_stack([np.cos(theta), np.sin(theta)])
                @ (np.diag(np.sqrt(eigvals) * 2))
                @ eigvecs.T
                + mean_pos
            )
            fig2.add_trace(
                go.Scatter(
                    x=ellipse[:, 0],
                    y=ellipse[:, 1],
                    mode="lines",
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dash"),
                )
            )
        fig2.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc([g1[:50, 0], g1[50:100, 0], g2[:50, 0]], param_type="nonparametric", key="permanova_ph")

    # Correlation and Association Tests

    elif test_name == "Pearson Correlation":
        st.subheader("Interactive Pearson Correlation")

        # =========================
        # CONTROLS
        # =========================

        col1, col2 = st.columns(2)

        with col1:
            r = st.slider("Correlation Coefficient (r)", -1.0, 1.0, 0.5, 0.01)

        with col2:
            n = st.slider("Sample Size (n)", 3, 100, 30)

        # =========================
        # LATEX
        # =========================

        st.latex(rf"""
            r = {r:.2f}
            """)

        # =========================
        # PLOTLY FIGURE
        # =========================

        x = np.random.normal(size=n)
        y = r * x + np.sqrt(1 - r**2) * np.random.normal(size=n)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data Points",
            )
        )

        fig.update_layout(
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="X",
            yaxis_title="Y",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Spearman Rank Correlation":

        from scipy.stats import spearmanr

        st.subheader("Interactive Spearman Rank Correlation")

        # =========================
        # CONTROLS
        # =========================
        direction = st.selectbox(
            "Correlation Direction",
            [
                "Positive",
                "Negative",
            ],
        )

        curve_strength = st.slider(
            "Monotonic Strength",
            0.1,
            3.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
            key="noise_5",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        x = np.linspace(0, 10, 300)

        direction_multiplier = 1 if direction == "Positive" else -1

        y = direction_multiplier * (x**curve_strength) + np.random.normal(0, noise, 300)

        rho, _ = spearmanr(x, y)

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"\rho = {rho:.3f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Ranked Data",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Ranked X",
            yaxis_title="Ranked Y",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Kendall's Tau-b":

        from scipy.stats import kendalltau

        st.subheader("Interactive Kendall's Tau-b")

        # =========================
        # CONTROLS
        # =========================

        strength = st.slider("Association Strength", 0.0, 1.0, 0.5, 0.05)

        n = st.slider("Sample Size", 10, 200, 60, key="kendall_s_tau_b_sample_size")

        noise = st.slider("Noise", 0.1, 5.0, 1.0, 0.1, key="kendall_s_tau_b_noise")

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        x = np.random.normal(0, 1, n)
        y = strength * x + np.random.normal(0, noise, n)

        tau, p = kendalltau(x, y)

        # =========================
        # STATS
        # =========================

        st.latex(rf"\tau_b = {tau:.3f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=y, mode="markers"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="X",
            yaxis_title="Y",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Chi-Square Test of Independence":

        from scipy.stats import chi2_contingency

        st.subheader("Interactive Chi-Square Test of Independence")

        # =========================
        # CONTROLS
        # =========================

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

        # =========================
        # TABLE
        # =========================

        table = np.array([[a, b], [c, d]])

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

    elif test_name == "Point-Biserial Correlation":

        from scipy.stats import pointbiserialr

        st.subheader("Interactive Point-Biserial Correlation")

        # =========================
        # CONTROLS
        # =========================

        group_difference = st.slider(
            "Group Mean Difference",
            0.0,
            10.0,
            3.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
            key="noise_7",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        group = np.random.binomial(1, 0.5, 300)

        y = group * group_difference + np.random.normal(0, noise, 300)

        r, p = pointbiserialr(group, y)

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"r_{{pb}} = {r:.3f}")

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=y[group == 0],
                name="Group 0",
            )
        )

        fig.add_trace(
            go.Box(
                y=y[group == 1],
                name="Group 1",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            yaxis_title="Continuous Variable",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Regression Tests

    elif test_name == "Logistic Regression":

        st.subheader("Interactive Logistic Regression")

        # =========================
        # CONTROLS
        # =========================

        col1, col2 = st.columns(2)

        with col1:
            beta0 = st.slider(
                "Intercept (β₀)",
                -10.0,
                10.0,
                0.0,
                0.1,
                key="logistic_regression_intercept",
            )

        with col2:
            beta1 = st.slider(
                "Slope (β₁)", -5.0, 5.0, 1.0, 0.1, key="logistic_regression_slope"
            )

        # =========================
        # DATA
        # =========================

        x = np.linspace(-10, 10, 1000)

        logit = beta0 + beta1 * x

        p = 1 / (1 + np.exp(-logit))

        # =========================
        # LATEX
        # =========================

        st.latex(rf"""
            p = \dfrac{{1}}{{1 + e^{{-({beta0:.2f} + {beta1:.2f}x)}}}}
            """)

        # =========================
        # PLOTLY FIGURE
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=p,
                mode="lines",
                name="Sigmoid Curve",
            )
        )

        fig.update_layout(
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Predictor (x)",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Simple Linear Regression":
        st.subheader("Interactive Simple Linear Regression")

        # =========================
        # CONTROLS
        # =========================

        col1, col2 = st.columns(2)

        with col1:
            beta0 = st.slider(
                "Intercept (β₀)",
                -20.0,
                20.0,
                0.0,
                0.1,
                key="simple_linear_regression_intercept",
            )

        with col2:
            beta1 = st.slider(
                "Slope (β₁)",
                -10.0,
                10.0,
                1.0,
                0.1,
                key="simple_linear_regression_slope",
            )

        # =========================
        # DATA
        # =========================

        x = np.linspace(0, 10, 500)

        y = beta0 + beta1 * x

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"y = {beta0:.2f} + ({beta1:.2f})x")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Regression Line",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="x",
            yaxis_title="y",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Multiple Linear Regression":
        st.subheader("Interactive Multiple Linear Regression")

        # =========================
        # CONTROLS
        # =========================

        beta0 = st.slider(
            "β₀", -20.0, 20.0, 0.0, 0.1, key="multiple_linear_regression_beta0"
        )

        beta1 = st.slider("β₁ (x₁ coefficient)", -10.0, 10.0, 1.0, 0.1)

        beta2 = st.slider("β₂ (x₂ coefficient)", -10.0, 10.0, 1.0, 0.1)

        # =========================
        # GRID
        # =========================

        x1 = np.linspace(-10, 10, 50)
        x2 = np.linspace(-10, 10, 50)

        X1, X2 = np.meshgrid(x1, x2)

        Y = beta0 + beta1 * X1 + beta2 * X2

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"y = {beta0:.2f} + ({beta1:.2f})x_1 + ({beta2:.2f})x_2")

        # =========================
        # SURFACE PLOT
        # =========================

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X1,
                    y=X2,
                    z=Y,
                )
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            height=700,
            scene=dict(
                xaxis_title="x₁",
                yaxis_title="x₂",
                zaxis_title="y",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Multinomial Logistic Regression":
        st.subheader("Interactive Multinomial Logistic Regression")

        # =========================
        # CONTROLS
        # =========================

        beta1 = st.slider("Class A coefficient", -5.0, 5.0, 1.0, 0.1)

        beta2 = st.slider("Class B coefficient", -5.0, 5.0, -1.0, 0.1)

        # =========================
        # DATA
        # =========================

        x = np.linspace(-10, 10, 500)

        score1 = np.exp(beta1 * x)
        score2 = np.exp(beta2 * x)
        score3 = np.exp(0)

        total = score1 + score2 + score3

        p1 = score1 / total
        p2 = score2 / total
        p3 = score3 / total

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=p1, mode="lines", name="Class A"))
        fig.add_trace(go.Scatter(x=x, y=p2, mode="lines", name="Class B"))
        fig.add_trace(go.Scatter(x=x, y=p3, mode="lines", name="Reference"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            yaxis=dict(range=[0, 1]),
            xaxis_title="Predictor",
            yaxis_title="Class Probability",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Ordinal Logistic Regression":
        st.subheader("Interactive Ordinal Logistic Regression")

        # =========================
        # CONTROLS
        # =========================

        beta = st.slider("β coefficient", -5.0, 5.0, 1.0, 0.1)

        threshold1 = st.slider("Threshold θ₁", -5.0, 5.0, -1.0, 0.1)

        threshold2 = st.slider("Threshold θ₂", -5.0, 5.0, 1.0, 0.1)

        # =========================
        # DATA
        # =========================

        x = np.linspace(-10, 10, 500)

        cum1 = 1 / (1 + np.exp(-(threshold1 - beta * x)))
        cum2 = 1 / (1 + np.exp(-(threshold2 - beta * x)))

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=cum1, mode="lines", name="P(Y ≤ 1)"))

        fig.add_trace(go.Scatter(x=x, y=cum2, mode="lines", name="P(Y ≤ 2)"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            yaxis=dict(range=[0, 1]),
            xaxis_title="Predictor",
            yaxis_title="Cumulative Probability",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Poisson Regression":
        st.subheader("Interactive Poisson Regression")

        # =========================
        # CONTROLS
        # =========================

        beta0 = st.slider("β₀", -3.0, 3.0, 0.5, 0.1, key="poisson_regression_beta0")

        beta1 = st.slider("β₁", -1.0, 1.0, 0.2, 0.05)

        # =========================
        # DATA
        # =========================

        x = np.linspace(0, 20, 500)

        lam = np.exp(beta0 + beta1 * x)

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"\lambda = e^{{{beta0:.2f} + ({beta1:.2f})x}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=lam,
                mode="lines",
                name="Expected Count",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Predictor",
            yaxis_title="Expected Count (λ)",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Cox Proportional Hazards Regression":

        st.subheader("Interactive Cox Regression")

        # =========================
        # CONTROLS
        # =========================

        hazard_ratio = st.slider("True Hazard Ratio (exp(β))", 0.5, 4.0, 2.0, 0.1)

        n_subjects = st.slider(
            "Number of Subjects",
            20,
            500,
            100,
            key="cox_proportional_hazards_regression_number_of_subjects",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        group = np.random.binomial(1, 0.5, n_subjects)

        log_hr = np.log(hazard_ratio)
        baseline_hazard = 0.05

        survival_times = -np.log(np.random.uniform(size=n_subjects)) / (
            baseline_hazard * np.exp(log_hr * group)
        )

        censor_times = np.random.uniform(5, 20, n_subjects)

        observed = (survival_times <= censor_times).astype(int)
        times = np.minimum(survival_times, censor_times)

        # =========================
        # SURVIVAL CURVES (theoretical exponential)
        # =========================

        fig = go.Figure()
        t_grid = np.linspace(0, max(times) * 1.1, 200)

        for grp, label, color in [(0, "Control", "blue"), (1, "Treatment", "red")]:
            hr = 1 if grp == 0 else hazard_ratio
            # Exponential survival: S(t) = exp(-h0 * t * exp(beta * X))
            surv = np.exp(-baseline_hazard * hr * t_grid)
            fig.add_trace(
                go.Scatter(
                    x=t_grid,
                    y=surv,
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Time",
            yaxis_title="Survival Probability",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Kaplan-Meier Survival Analysis":

        from scipy.stats import chi2 as _chi2

        st.subheader("Interactive Kaplan-Meier Survival Analysis")

        # =========================
        # CONTROLS
        # =========================
        n = st.slider("Patients per Group", 10, 200, 50, key="km_n_pat")
        hr = st.slider(
            "Hazard Ratio (Trt/Control)", 0.2, 2.0, 0.5, 0.05, key="km_hr_ratio"
        )
        cens = st.slider("Censoring Rate", 0.0, 0.5, 0.2, 0.05, key="km_cens_rate")
        show_ci = st.toggle("Show 95% CI", True, key="km_ci_toggle")

        # =========================
        # DATA
        # =========================

        np.random.seed(42)
        t_ctrl = np.random.exponential(12, n)
        t_trt = np.random.exponential(12 / hr, n)
        all_t = np.concatenate([t_ctrl, t_trt])
        groups = np.array([0] * n + [1] * n)

        max_fup = 30
        fup = np.random.uniform(0, max_fup, 2 * n)
        if cens > 0.01:
            early = np.random.exponential(8 / max(cens, 0.05), 2 * n)
            fup = np.minimum(fup, early)
        obs = np.minimum(all_t, fup)
        event = (all_t <= fup).astype(int)

        # =========================
        # HELPERS
        # =========================

        def _km_est(tt, ee):
            df = pd.DataFrame({"t": tt, "e": ee}).sort_values("t")
            ut = sorted(df["t"].unique())
            s = 1.0
            ot, os = [0], [1.0]
            for ti in ut:
                nr = (df["t"] >= ti).sum()
                ne = df.loc[df["t"] == ti, "e"].sum()
                if nr > 0:
                    s *= 1 - ne / nr
                ot.extend([ti, ti])
                os.extend([os[-1], s])
            return np.array(ot), np.array(os)

        def _km_ci(tt, ee):
            df = pd.DataFrame({"t": tt, "e": ee}).sort_values("t")
            ut = sorted(df["t"].unique())
            s, cv = 1.0, 0.0
            ot, os, olo, ohi = [0], [1.0], [1.0], [1.0]
            for ti in ut:
                nr = (df["t"] >= ti).sum()
                ne = df.loc[df["t"] == ti, "e"].sum()
                if nr > 0 and ne > 0:
                    s *= 1 - ne / nr
                    cv += ne / (nr * (nr - ne))
                elif nr > 0:
                    s *= 1 - ne / nr
                se = s * np.sqrt(cv) if cv > 0 else 0
                lo = np.clip(s - 1.96 * se, 0, 1)
                hi = np.clip(s + 1.96 * se, 0, 1)
                ot.extend([ti, ti])
                os.extend([os[-1], s])
                olo.extend([olo[-1], lo])
                ohi.extend([ohi[-1], hi])
            return np.array(ot), np.array(os), np.array(olo), np.array(ohi)

        def _logrank(tt, ee, gg):
            sort_idx = np.argsort(tt)
            tt_s = tt[sort_idx]
            ee_s = ee[sort_idx]
            gg_s = gg[sort_idx]
            ut = sorted(set(tt_s))
            o1 = e1 = v1 = 0.0
            for ti in ut:
                mask_t = tt_s == ti
                ne = ee_s[mask_t].sum()
                if ne == 0:
                    continue
                at_risk = tt_s >= ti
                n1 = (gg_s[at_risk] == 0).sum()
                n2 = (gg_s[at_risk] == 1).sum()
                nr = n1 + n2
                if nr < 2:
                    continue
                d1 = (gg_s[mask_t & (ee_s == 1)] == 0).sum()
                o1 += d1
                e1 += ne * n1 / nr
                v1 += ne * n1 * n2 * (nr - ne) / (nr * nr * (nr - 1))
            if v1 <= 0:
                return 0.0, 1.0
            chi2 = (o1 - e1) ** 2 / v1
            return chi2, 1 - _chi2.cdf(chi2, 1)

        # =========================
        # COMPUTE
        # =========================

        t1, s1 = _km_est(obs[groups == 0], event[groups == 0])
        t2, s2 = _km_est(obs[groups == 1], event[groups == 1])
        chi2, p_val = _logrank(obs, event, groups)

        below1 = np.where(s1 <= 0.5)[0]
        below2 = np.where(s2 <= 0.5)[0]
        med1 = t1[below1[0]] if len(below1) > 0 else None
        med2 = t2[below2[0]] if len(below2) > 0 else None

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()
        colors = {"Control": "#4C78A8", "Treatment": "#E45756"}
        fill_colors = {
            "Control": "rgba(76,120,168,0.15)",
            "Treatment": "rgba(228,87,86,0.15)",
        }

        for grp, name in [(0, "Control"), (1, "Treatment")]:
            t_, s_, lo_, hi_ = _km_ci(obs[groups == grp], event[groups == grp])
            if show_ci:
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([t_, t_[::-1]]),
                        y=np.concatenate([hi_, lo_[::-1]]),
                        fill="toself",
                        fillcolor=fill_colors[name],
                        line=dict(width=0),
                        name=f"{name} 95% CI",
                        showlegend=True,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=t_,
                    y=s_,
                    mode="lines",
                    line=dict(color=colors[name], width=2.5),
                    name=name,
                    hovertemplate="Time=%{x:.1f}<br>Survival=%{y:.3f}<extra></extra>",
                    legendgroup=name,
                )
            )
            mask = groups == grp
            cens_t = obs[mask & (event == 0)]
            if len(cens_t) > 0:
                idxs = np.searchsorted(t_, cens_t, side="right") - 1
                cs = s_[np.maximum(0, idxs)]
                fig.add_trace(
                    go.Scatter(
                        x=cens_t,
                        y=cs,
                        mode="markers",
                        marker=dict(color=colors[name], symbol="line-ns", size=7),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        events_pct = int(event.mean() * 100)
        fig.update_xaxes(tickformat=".2f")
        fig.update_yaxes(tickformat=".2f")
        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Time",
            yaxis_title="Survival Probability",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # METRICS
        # =========================

        m1 = st.columns(4)
        m1[0].metric(
            "Control Median Survival",
            f"{med1:.1f}" if med1 is not None else "Not reached",
        )
        m1[1].metric(
            "Treatment Median Survival",
            f"{med2:.1f}" if med2 is not None else "Not reached",
        )
        m1[2].metric("Log-Rank χ²", f"{chi2:.3f}")
        m1[3].metric(
            "Log-Rank p-value", f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
        )

        # =========================
        # INTERPRETATION
        # =========================

        with st.expander("Interpretation & Guidance", expanded=True):
            st.markdown(f"""
            - **Events observed**: {int(event.sum())} / {len(event)} ({events_pct}% of patients experienced the event)
            - **Censored**: {(~event.astype(bool)).sum()} patients ({100 - events_pct}%) had their event time censored
            """)
            col_i, col_w, col_t, col_m = st.columns(4)
            with col_i:
                st.info(
                    "**Interpretation**\n\n"
                    "- Step down = event\n"
                    "- Tick marks = censored\n"
                    "- Lower curve = worse survival\n"
                    "- HR < 1 favors treatment"
                )
            with col_w:
                st.success(
                    "**When To Use**\n\n"
                    "- Time-to-event data\n"
                    "- Censored observations\n"
                    "- Compare survival curves\n"
                    "- Estimate median survival"
                )
            with col_t:
                st.warning(
                    "**Associated**\n\n"
                    "- Log-Rank Test\n"
                    "- Cox PH Regression\n"
                    "- Nelson-Aalen Plot\n"
                    "- Hazard Ratio"
                )
            with col_m:
                st.error(
                    "**Caution**\n\n"
                    "KM beyond last event is "
                    "unstable. Always show "
                    "number at risk. "
                    "CI widens at late times."
                )

    elif test_name == "Log-Rank Test":

        st.subheader("Interactive Log-Rank Test")

        # =========================
        # CONTROLS
        # =========================

        sep = st.slider("Survival Separation", 0.0, 5.0, 1.5, 0.1)

        n = st.slider("Subjects per Group", 10, 200, 50)

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        t1 = np.random.exponential(10, n)
        t2 = np.random.exponential(10, n) / (1 + sep * 0.3)

        censor = np.random.exponential(15, n)
        censor2 = np.random.exponential(15, n)

        obs1 = (t1 <= censor).astype(int)
        obs2 = (t2 <= censor2).astype(int)
        t1_obs = np.minimum(t1, censor)
        t2_obs = np.minimum(t2, censor2)

        fig = go.Figure()
        t_grid = np.linspace(0, max(max(t1_obs), max(t2_obs)) * 1.1, 200)

        # Exponential fit approximations
        rate1 = 1 / np.mean(t1_obs[obs1 == 1]) if obs1.sum() > 0 else 0.1
        rate2 = 1 / np.mean(t2_obs[obs2 == 1]) if obs2.sum() > 0 else 0.1

        surv1 = np.exp(-rate1 * t_grid)
        surv2 = np.exp(-rate2 * t_grid)

        fig.add_trace(go.Scatter(x=t_grid, y=surv1, mode="lines", name="Group 1"))
        fig.add_trace(go.Scatter(x=t_grid, y=surv2, mode="lines", name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Time",
            yaxis_title="Survival Probability",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Sensitivity & Specificity Analysis":
        st.subheader("Interactive Diagnostic Accuracy Calculator")

        # =========================
        # CONTROLS
        # =========================
        col1, col2 = st.columns(2)
        with col1:
            tp = st.number_input("True Positives (TP)", min_value=0, value=80)
            fn = st.number_input("False Negatives (FN)", min_value=0, value=20)
        with col2:
            fp = st.number_input("False Positives (FP)", min_value=0, value=10)
            tn = st.number_input("True Negatives (TN)", min_value=0, value=90)

        # =========================
        # CALCULATIONS
        # =========================
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        LR_positive = (
            sensitivity / (1 - specificity) if (1 - specificity) > 0 else float("inf")
        )
        LR_negative = (
            (1 - sensitivity) / specificity if specificity > 0 else float("inf")
        )
        F1_score = (
            2 * (ppv * sensitivity) / (ppv + sensitivity)
            if (ppv + sensitivity) > 0
            else 0
        )
        DOR = LR_positive / LR_negative if LR_negative > 0 else float("inf")

        # =========================
        # STATS
        # =========================
        cols = st.columns(3)
        cols[0].metric("Sensitivity", f"{sensitivity:.1%}")
        cols[1].metric("Specificity", f"{specificity:.1%}")
        cols[2].metric("Accuracy", f"{accuracy:.1%}")

        cols2 = st.columns(3)
        cols2[0].metric("Pos. Pred. Value (PPV)", f"{ppv:.1%}")
        cols2[1].metric("Neg. Pred. Value (NPV)", f"{npv:.1%}")
        cols2[2].metric("F1 Score", f"{F1_score:.2f}")

        cols3 = st.columns(3)
        cols3[0].metric("LR+", f"{LR_positive:.2f}")
        cols3[1].metric("LR-", f"{LR_negative:.2f}")
        cols3[2].metric("Diagnostic Odds Ratio (DOR)", f"{DOR:.2f}")

        matrix = np.array(
            [
                [tp, fp],
                [fn, tn],
            ]
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Negative", "Positive"],
                y=["Negative", "Positive"],
                text=matrix,
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "ROC Curve Analysis":
        st.subheader("Interactive ROC Curve Analysis")

        # =========================
        # CONTROLS
        # =========================
        separation = st.slider(
            "Diagnostic Power (Group Separation)", 0.0, 5.0, 1.5, 0.1
        )

        # =========================
        # DATA
        # =========================
        np.random.seed(42)
        n = 500
        scores_healthy = np.random.normal(0, 1, n)
        scores_disease = np.random.normal(separation, 1, n)

        y_true = np.concatenate([np.zeros(n), np.ones(n)])
        y_scores = np.concatenate([scores_healthy, scores_disease])

        from sklearn.metrics import roc_curve, auc

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # =========================
        # STATS
        # =========================
        st.metric("Area Under Curve (AUC)", f"{roc_auc:.3f}")

        # =========================
        # PLOT
        # =========================
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC curve (AUC = {roc_auc:.2f})",
                fill="tozeroy",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                name="Random Guess",
            )
        )

        fig.update_layout(
            title="Receiver Operating Characteristic (ROC)",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity)",
            template="plotly_dark",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Likelihood Ratio Analysis":
        st.subheader("Interactive Likelihood Ratio Analysis")

        # =========================
        # CONTROLS
        # =========================
        col1, col2 = st.columns(2)
        with col1:
            sens = st.slider("Sensitivity", 0.0, 1.0, 0.8)
        with col2:
            spec = st.slider("Specificity", 0.0, 1.0, 0.9)

        # =========================
        # CALCULATIONS
        # =========================
        lr_pos = sens / (1 - spec) if spec < 1 else float("inf")
        lr_neg = (1 - sens) / spec if spec > 0 else float("inf")

        # =========================
        # STATS
        # =========================
        c1, c2 = st.columns(2)
        c1.metric("LR+", f"{lr_pos:.2f}")
        c2.metric("LR-", f"{lr_neg:.2f}")

        st.info("""
        - **LR+ > 10**: Strong evidence to rule in disease.
        - **LR- < 0.1**: Strong evidence to rule out disease.
        """)

    elif test_name == "Cohen's Kappa (Agreement Analysis)":
        st.subheader("Interactive Agreement Analysis (Cohen's Kappa)")

        # =========================
        # CONTROLS
        # =========================
        st.write("Enter agreement counts between two raters:")
        c1, c2 = st.columns(2)
        with c1:
            yy = st.number_input("Both say YES", min_value=0, value=40)
            yn = st.number_input(
                "Rater 1 says YES, Rater 2 says NO", min_value=0, value=10
            )
        with c2:
            ny = st.number_input(
                "Rater 1 says NO, Rater 2 says YES", min_value=0, value=5
            )
            nn = st.number_input("Both say NO", min_value=0, value=45)

        # =========================
        # CALCULATIONS
        # =========================
        total = yy + yn + ny + nn
        if total > 0:
            po = (yy + nn) / total
            pe = ((yy + yn) * (yy + ny) + (ny + nn) * (yn + nn)) / (total * total)
            kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        else:
            kappa = 0

        # =========================
        # STATS
        # =========================
        st.metric("Cohen's Kappa (κ)", f"{kappa:.3f}")

        if kappa > 0.8:
            interpretation = "Almost Perfect Agreement"
        elif kappa > 0.6:
            interpretation = "Substantial Agreement"
        elif kappa > 0.4:
            interpretation = "Moderate Agreement"
        elif kappa > 0.2:
            interpretation = "Fair Agreement"
        else:
            interpretation = "Slight/Poor Agreement"

        st.success(f"Interpretation: {interpretation}")

    elif test_name == "Bland-Altman Analysis":

        st.subheader("Interactive Bland-Altman Analysis")

        # =========================
        # CONTROLS
        # =========================

        bias = st.slider("Bias (Mean Difference)", -5.0, 5.0, 0.2, 0.1)

        agreement_sd = st.slider("SD of Differences", 0.1, 5.0, 1.0, 0.1)

        n = st.slider(
            "Sample Size", 10, 200, 50, key="bland_altman_analysis_sample_size"
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        true_val = np.random.uniform(10, 50, n)
        diff = np.random.normal(bias, agreement_sd, n)
        method1 = true_val - diff / 2
        method2 = true_val + diff / 2

        mean_pair = (method1 + method2) / 2
        diff_pair = method1 - method2

        mean_diff = np.mean(diff_pair)
        sd_diff = np.std(diff_pair, ddof=1)
        upper_loa = mean_diff + 1.96 * sd_diff
        lower_loa = mean_diff - 1.96 * sd_diff

        # =========================
        # STATS
        # =========================

        cols = st.columns(3)
        cols[0].metric("Mean Difference (Bias)", f"{mean_diff:.3f}")
        cols[1].metric("Upper LoA (+1.96 SD)", f"{upper_loa:.3f}")
        cols[2].metric("Lower LoA (−1.96 SD)", f"{lower_loa:.3f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=mean_pair,
                y=diff_pair,
                mode="markers",
                name="Differences",
            )
        )

        fig.add_hline(y=mean_diff, line_dash="solid", annotation_text="Bias")

        fig.add_hline(y=upper_loa, line_dash="dash", annotation_text="+1.96 SD")

        fig.add_hline(y=lower_loa, line_dash="dash", annotation_text="−1.96 SD")

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Mean of Two Measurements",
            yaxis_title="Difference (Method 1 − Method 2)",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Interactive widget coming soon for this test.")
