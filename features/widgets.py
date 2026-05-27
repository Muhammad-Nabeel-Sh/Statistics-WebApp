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


def render_test_widget(test_name, external_data=None):
    """Render interactive widget for specific statistical test.
    
    Parameters
    ----------
    test_name : str
    external_data : dict or None
        If provided (with "using_uploaded": True), used instead of data_source_toggle.
        Same format as data_source_toggle() return value.
    """

    # One Sample Tests

    if test_name == "One-sample t-test":

        from scipy.stats import ttest_1samp

        st.subheader("Interactive One-sample t-test")

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("ttest_1samp", mode="one_sample")

        # =========================
        # CONTROLS
        # =========================

        population_mean = st.slider(
            "Reference Mean (H₀: μ = μ₀)",
            -10.0,
            10.0,
            0.0,
            0.1,
        )

        if src["using_uploaded"]:
            sample = src["data"]["values"]
        else:
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

        st.info("""
        **z-test** assumes the population standard deviation is **known**. In practice,
        it is rarely used since we almost never know the population σ.
        
        When σ is unknown, use the **One-sample t-test** instead.
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("ztest_1samp", mode="one_sample")

        # =========================
        # CONTROLS
        # =========================

        population_mean = st.slider(
            "Population Mean (μ₀)",
            -10.0,
            10.0,
            0.0,
            0.1,
        )

        known_sigma = st.slider(
            "Known Population σ",
            0.1,
            10.0,
            1.0,
            0.1,
        )

        if src["using_uploaded"]:
            sample = src["data"]["values"]
        else:
            shift = st.slider(
                "Sample Mean Shift",
                -5.0,
                5.0,
                1.0,
                0.1,
                key="sample_mean_shift_1",
            )

            np.random.seed(42)
            sample = np.random.normal(
                population_mean + shift,
                known_sigma,
                300,
            )

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        n_z = len(sample)
        sample_mean_z = np.mean(sample)
        sample_sd_z = np.std(sample, ddof=1)

        summary_data = {
            "Metric": ["n", "Sample Mean", "Sample SD", "Known σ"],
            "Value": [
                n_z,
                f"{sample_mean_z:.3f}",
                f"{sample_sd_z:.3f}",
                f"{known_sigma:.3f}",
            ],
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # TEST
        # =========================

        se_z = known_sigma / np.sqrt(n_z)
        z = (sample_mean_z - population_mean) / se_z
        from scipy.stats import norm
        p_two_sided = 2 * (1 - norm.cdf(abs(z)))

        ci_low = sample_mean_z - 1.96 * se_z
        ci_high = sample_mean_z + 1.96 * se_z

        st.latex(rf"z = {z:.3f}")
        st.latex(rf"\text{{{format_p_value(p_two_sided)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Histogram(x=sample, name="Sample", nbinsx=25))

        fig.add_vline(
            x=population_mean,
            line_dash="dash",
            line_color="green",
            annotation_text="H₀ Mean",
        )
        fig.add_vline(
            x=sample_mean_z,
            line_dash="dot",
            line_color="blue",
            annotation_text="Sample Mean",
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            title="Histogram with Mean Annotations",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        results_data = {
            "Metric": [
                "Sample Mean",
                "Population Mean (μ₀)",
                "Mean Difference",
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
                f"{sample_mean_z - population_mean:.3f}",
                f"{z:.3f}",
                format_p_value(p_two_sided),
                f"{se_z:.4f}",
                f"{n_z}",
                f"[{ci_low:.2f}, {ci_high:.2f}]",
                f"{known_sigma:.1f}",
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

        y_dense = norm2.pdf(x_dense, sample_mean_z, sample_sd_z)
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
            title="Density Plot with Normal Fit",
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

        st.info("""
        **One-sample Wilcoxon Signed-Rank Test** is a nonparametric alternative to the one-sample t-test.
        It tests whether the median of a sample differs from a hypothesized value (typically 0).
        
        - Use when data is **not normally distributed**
        - Tests for **median** (not mean)
        - Also used for paired differences: H₀: median difference = 0
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("wilcoxon_1samp", mode="one_sample")

        # =========================
        # CONTROLS
        # =========================

        hypothesized_median = st.slider(
            "Hypothesized Median (H₀)",
            -10.0,
            10.0,
            0.0,
            0.5,
        )

        if src["using_uploaded"]:
            sample_raw = src["data"]["values"]
            sample = sample_raw - hypothesized_median
        else:
            median_shift = st.slider(
                "Median Shift from H₀",
                -5.0,
                5.0,
                1.0,
                0.1,
            )

            np.random.seed(42)
            sample_raw = np.random.exponential(1, 80) + hypothesized_median + median_shift
            sample = sample_raw - hypothesized_median

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        n_1w = len(sample)
        median_1w = np.median(sample_raw)
        median_diff_1w = median_1w - hypothesized_median
        mean_1w = np.mean(sample_raw)
        sd_1w = np.std(sample_raw, ddof=1)

        summary_data = {
            "Metric": ["n", "Median", "Mean", "SD", "Hypothesized Median", "Median Difference"],
            "Value": [
                f"{n_1w}",
                f"{median_1w:.3f}",
                f"{mean_1w:.3f}",
                f"{sd_1w:.3f}",
                f"{hypothesized_median}",
                f"{median_diff_1w:.3f}",
            ],
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # TEST
        # =========================

        T_1w, p_1w = wilcoxon(sample)
        r_rb_1w = 1 - 2 * T_1w / (n_1w * (n_1w + 1) / 2)

        st.latex(rf"W = {T_1w:.3f}")
        st.latex(rf"\text{{{format_p_value(p_1w)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=sample_raw, name="Sample", boxmean="sd", marker_color="#4C78A8"))

        fig.add_hline(
            y=hypothesized_median,
            line_dash="dash",
            line_color="red",
            annotation_text="H₀ Median",
        )
        fig.add_hline(
            y=median_1w,
            line_dash="dot",
            line_color="blue",
            annotation_text="Observed Median",
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            title="Boxplot with Median Annotations",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        results_data = {
            "Metric": [
                "Median",
                "Hypothesized Median",
                "W-statistic",
                "p-value",
                "Sample Size (n)",
                "Median Difference",
                "Rank-biserial r",
            ],
            "Value": [
                f"{median_1w:.3f}",
                f"{hypothesized_median}",
                f"{T_1w:.3f}",
                format_p_value(p_1w),
                f"{n_1w}",
                f"{median_diff_1w:.3f}",
                f"{r_rb_1w:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Box(y=sample_raw, name="Sample", boxmean="sd", marker_color="#4C78A8")
        )

        jitter_x = np.random.normal(0, 0.06, n_1w)
        fig2.add_trace(
            go.Scatter(
                x=jitter_x,
                y=sample_raw,
                mode="markers",
                name="Data points",
                marker=dict(color="rgba(0, 123, 255, 0.5)", size=5),
            )
        )

        fig2.add_hline(
            y=hypothesized_median,
            line_dash="dash",
            line_color="red",
            annotation_text=f"H₀: median = {hypothesized_median}",
        )

        ci_1w = (
            1.58
            * (np.percentile(sample_raw, 75) - np.percentile(sample_raw, 25))
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
            title="Boxplot with Individual Data Points",
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

        from scipy.stats import ttest_ind, t as t_dist

        st.subheader("Interactive Independent t-test")

        st.info("""
        **This widget shows BOTH Student's t-test (equal variance assumed) and Welch's t-test (unequal variance)**
        side-by-side for easy comparison with statistical software like Minitab.
        
        - **Student's**: Used when you assume equal variances (pooled variance)
        - **Welch's**: Used when variances may be unequal (adjusted degrees of freedom)
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("ttest_indep_enhanced", mode="two_sample")

        # =========================
        # CONTROLS
        # =========================

        if src["using_uploaded"]:
            group1 = src["data"]["group1"]
            group2 = src["data"]["group2"]
            g1_name, g2_name = src["data"]["group_names"]
        else:
            mean_diff = st.slider(
                "Mean Difference",
                0.0,
                10.0,
                2.0,
                0.1,
            )

            sd1 = st.slider(
                "Group 1 SD",
                0.5,
                5.0,
                1.5,
                0.1,
            )

            sd2 = st.slider(
                "Group 2 SD",
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

            np.random.seed(42)
            group1 = np.random.normal(0, sd1, n)
            group2 = np.random.normal(mean_diff, sd2, n)
            g1_name = "Group 1"
            g2_name = "Group 2"

        # =========================
        # DATA SUMMARY
        # =========================

        n1, n2 = len(group1), len(group2)
        m1, m2 = np.mean(group1), np.mean(group2)
        sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        se1, se2 = sd1 / np.sqrt(n1), sd2 / np.sqrt(n2)

        st.divider()
        st.subheader("Data Summary")

        summary_data = {
            "Group": [g1_name, g2_name],
            "n": [n1, n2],
            "Mean": [f"{m1:.3f}", f"{m2:.3f}"],
            "SD": [f"{sd1:.3f}", f"{sd2:.3f}"],
            "SE Mean": [f"{se1:.3f}", f"{se2:.3f}"],
        }
        st.table(pd.DataFrame(summary_data))

        st.info(f"""
        **About group ordering:**
        - t-statistic = ({g1_name} Mean) - ({g2_name} Mean) = {m1:.3f} - {m2:.3f} = {m1 - m2:.3f}
        - A negative t means {g2_name} > {g1_name}
        - Software like Minitab may show the opposite sign depending on group order
        """)

        # =========================
        # BOTH TESTS
        # =========================

        t_student, p_student = ttest_ind(group1, group2, equal_var=True)
        t_welch, p_welch = ttest_ind(group1, group2, equal_var=False)

        df_student = n1 + n2 - 2
        diff_student = m1 - m2

        se_student = np.sqrt(
            ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
        ) * np.sqrt(1 / n1 + 1 / n2)

        se_welch = np.sqrt(sd1**2 / n1 + sd2**2 / n2)

        welch_df_num = (sd1**2 / n1 + sd2**2 / n2) ** 2
        welch_df_den = (sd1**2 / n1) ** 2 / (n1 - 1) + (sd2**2 / n2) ** 2 / (n2 - 1)
        df_welch = welch_df_num / welch_df_den

        ci_low_student = diff_student - t_dist.ppf(0.975, df_student) * se_student
        ci_high_student = diff_student + t_dist.ppf(0.975, df_student) * se_student

        ci_low_welch = diff_student - t_dist.ppf(0.975, df_welch) * se_welch
        ci_high_welch = diff_student + t_dist.ppf(0.975, df_welch) * se_welch

        st.divider()
        st.subheader("Test Results (Side-by-Side Comparison)")

        results_data = {
            "Metric": [
                "Difference (G1 - G2)",
                "SE of Difference",
                "t-statistic",
                "df",
                "p-value",
                "95% CI Lower",
                "95% CI Upper",
                "",
            ],
            "Student's (equal var)": [
                f"{diff_student:.4f}",
                f"{se_student:.4f}",
                f"{t_student:.4f}",
                f"{df_student:.0f}",
                format_p_value(p_student),
                f"{ci_low_student:.4f}",
                f"{ci_high_student:.4f}",
                "",
            ],
            "Welch's (unequal var)": [
                f"{diff_student:.4f}",
                f"{se_welch:.4f}",
                f"{t_welch:.4f}",
                f"{df_welch:.2f}",
                format_p_value(p_welch),
                f"{ci_low_welch:.4f}",
                f"{ci_high_welch:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        if p_student < 0.05 and p_welch < 0.05:
            st.success("✅ Both tests agree: Significant difference between groups")
        elif p_student < 0.05 or p_welch < 0.05:
            st.warning("⚠️ Tests disagree! Check variance assumptions:")
            st.write(f"- Student's p: {p_student:.4f}")
            st.write(f"- Welch's p: {p_welch:.4f}")
            st.write(f"- SD ratio (G2/G1): {sd2/sd1:.2f}")
        else:
            st.info("ℹ️ No significant difference detected by either test")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Box(y=group1, name=g1_name, boxmean="sd", marker_color="#4C78A8")
        )
        fig.add_trace(
            go.Box(y=group2, name=g2_name, boxmean="sd", marker_color="#E45756")
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            yaxis_title="Value",
            title=f"Boxplots: {g1_name} vs {g2_name}",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Welch's t-test (Independent, Unequal Variances)":

        from scipy.stats import ttest_ind, t as t_dist_welch

        st.subheader("Interactive Welch's t-test")

        st.info("""
        **Welch's t-test** is used when you cannot assume equal variances between the two groups.
        It uses the Welch-Satterthwaite approximation to adjust the degrees of freedom.
        
        *Note: The enhanced "Student's t-test" widget now shows BOTH tests side-by-side for easy comparison.*
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("welch_ttest", mode="two_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            g1 = src["data"]["group1"]
            g2 = src["data"]["group2"]
            g1_name, g2_name = src["data"]["group_names"]
        else:
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

            np.random.seed(42)
            g1 = np.random.normal(0, sd1, n)
            g2 = np.random.normal(mean_diff, sd2, n)
            g1_name = "Group 1"
            g2_name = "Group 2"

        # =========================
        # DATA SUMMARY
        # =========================

        n1_w, n2_w = len(g1), len(g2)
        m1_w, m2_w = np.mean(g1), np.mean(g2)
        sd1_w, sd2_w = np.std(g1, ddof=1), np.std(g2, ddof=1)

        st.divider()
        st.subheader("Data Summary")

        summary_data = {
            "Group": [g1_name, g2_name],
            "n": [n1_w, n2_w],
            "Mean": [f"{m1_w:.3f}", f"{m2_w:.3f}"],
            "SD": [f"{sd1_w:.3f}", f"{sd2_w:.3f}"],
        }
        st.table(pd.DataFrame(summary_data))

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

        fig.add_trace(go.Violin(y=g1, name=g1_name))
        fig.add_trace(go.Violin(y=g2, name=g2_name))

        fig.update_layout(
            template="plotly_dark",
            height=550,
            title=f"Violin Plots: {g1_name} vs {g2_name}",
        )

        st_plot_with_download(fig, key="welch_ttest_violin", height=550)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        mean_diff_w = m2_w - m1_w
        se_w = np.sqrt(sd1_w**2 / n1_w + sd2_w**2 / n2_w)

        welch_df_num = (sd1_w**2 / n1_w + sd2_w**2 / n2_w) ** 2
        welch_df_den = (sd1_w**2 / n1_w) ** 2 / (n1_w - 1) + (
            sd2_w**2 / n2_w
        ) ** 2 / (n2_w - 1)
        df_welch = welch_df_num / welch_df_den

        ci_diff_w = se_w * t_dist_welch.ppf(0.975, df_welch)

        pooled_sd_w = np.sqrt((sd1_w**2 + sd2_w**2) / 2)
        cohens_d_w = mean_diff_w / pooled_sd_w
        d_lower_w, d_upper_w = cohens_d_independent_ci(cohens_d_w, n1_w, n2_w)
        hedges_g_w = hedges_g(cohens_d_w, n1_w, n2_w)
        d_interp_w = interpret_cohens_d(cohens_d_w)

        results_data = {
            "Metric": [
                f"Mean {g1_name} (SD)",
                f"Mean {g2_name} (SD)",
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
                f"{df_welch:.1f}",
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

        for i, (g, name) in enumerate(zip([g1, g2], [g1_name, g2_name])):
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("ttest_paired", mode="paired")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            before = src["data"]["values1"]
            after = src["data"]["values2"]
            col_names = src["data"].get("col_names", ["Before", "After"])
        else:
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

            np.random.seed(42)
            before = np.random.normal(10, noise, n)
            after = before + effect + np.random.normal(0, noise, n)
            col_names = ["Before", "After"]

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

        for i in range(len(before)):

            fig.add_trace(
                go.Scatter(
                    x=col_names,
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
                f"Mean {col_names[0]} (SD)",
                f"Mean {col_names[1]} (SD)",
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
                    x=col_names,
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=col_names,
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
            annotation_text=f"{col_names[0]} Mean = {mean_pre:.2f}",
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

        from scipy.stats import f_oneway, f as f_dist_1w

        st.subheader("Interactive One-way ANOVA")

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("oneway_anova", mode="multi_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            groups_1w = src["data"]["groups"]
            group_names_1w = src["data"]["group_names"]
        else:
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

            np.random.seed(42)
            g1 = np.random.normal(0, noise, 60)
            g2 = np.random.normal(mean_shift, noise, 60)
            g3 = np.random.normal(mean_shift * 2, noise, 60)
            groups_1w = [g1, g2, g3]
            group_names_1w = ["Group 1", "Group 2", "Group 3"]

        F, p = f_oneway(*groups_1w)

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        means_1w = [np.mean(g) for g in groups_1w]
        sds_1w = [np.std(g, ddof=1) for g in groups_1w]
        n_1w = [len(g) for g in groups_1w]

        summary_data = {
            "Group": group_names_1w,
            "n": n_1w,
            "Mean": [f"{m:.3f}" for m in means_1w],
            "SD": [f"{s:.3f}" for s in sds_1w],
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # STATS
        # =========================

        st.latex(rf"F = {F:.3f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i, (g, name) in enumerate(zip(groups_1w, group_names_1w)):
            fig.add_trace(go.Box(y=g, name=name))

        fig.update_layout(
            template="plotly_dark",
            height=550,
            title="Boxplots by Group",
        )

        st_plot_with_download(fig, key="oneway_anova_box", height=550)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

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

        metric_list = [f"Mean {name} (SD)" for name in group_names_1w] + [
            "F",
            f"df ({df_between}, {df_within})",
            "p-value",
            "η²",
            "ω² (unbiased)",
            "Interpretation",
            "",
        ]
        value_list = [
            f"{m:.2f} ({s:.2f})" for m, s in zip(means_1w, sds_1w)
        ] + [
            f"{F_1w:.3f}",
            f"{df_between}, {df_within}",
            format_p_value(p_1w),
            f"{eta_sq:.4f}",
            f"{max(0, omega_sq):.4f}",
            eta_interp,
            "",
        ]

        results_data = {
            "Metric": metric_list,
            "Value": value_list,
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

        for i, (g, name) in enumerate(zip(groups_1w, group_names_1w)):
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
            title="Violin Plots with Data Points",
        )

        st_plot_with_download(fig2, key="oneway_anova_violin", height=550)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc(groups_1w, param_type="parametric", key="anova_ph")

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

        st.info("""
        **Wilcoxon Signed-Rank Test (paired)** is the nonparametric alternative to the **Paired t-test**.
        
        Use when:
        - Comparing **two related/dependent groups** (e.g., before/after measurements)
        - Data is **not normally distributed**
        - Tests for differences in **medians** (not means)
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("wilcoxon_signedrank_paired", mode="paired")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            before = src["data"]["values1"]
            after = src["data"]["values2"]
            col_names = src["data"].get("col_names", ["Before", "After"])
        else:
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

            np.random.seed(42)
            before = np.random.exponential(1, n)
            after = before + median_shift + np.random.normal(0, noise, n)
            col_names = ["Before", "After"]

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        n = len(before)
        median_pre = np.median(before)
        median_post = np.median(after)
        mean_pre = np.mean(before)
        mean_post = np.mean(after)
        median_diff = median_post - median_pre

        summary_data = {
            "Measure": [col_names[0], col_names[1], "Difference"],
            "n": [n, n, n],
            "Median": [
                f"{median_pre:.3f}",
                f"{median_post:.3f}",
                f"{median_diff:.3f}",
            ],
            "Mean": [
                f"{mean_pre:.3f}",
                f"{mean_post:.3f}",
                f"{mean_post - mean_pre:.3f}",
            ],
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # TEST
        # =========================

        stat, p = wilcoxon(before, after)

        from scipy.stats import norm as norm_w

        z_w = -norm_w.ppf(p / 2) if p > 0 else 0
        r_ws = z_w / np.sqrt(n) if n > 0 else 0

        st.latex(rf"W = {stat:.3f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i in range(n):
            fig.add_trace(
                go.Scatter(
                    x=col_names,
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(150, 150, 150, 0.3)", width=1),
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            title="Spaghetti Plot: Individual Changes",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        results_data = {
            "Metric": [
                f"Median ({col_names[0]})",
                f"Median ({col_names[1]})",
                "Median Difference",
                "W-statistic",
                "z (approx)",
                "p-value",
                "Rank-biserial r (effect size)",
            ],
            "Value": [
                f"{median_pre:.3f}",
                f"{median_post:.3f}",
                f"{median_diff:.3f}",
                f"{stat:.3f}",
                f"{z_w:.3f}",
                format_p_value(p),
                f"{r_ws:.4f}",
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
                    x=col_names,
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=col_names,
                y=[median_pre, median_post],
                mode="lines+markers",
                name="Median change",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=12),
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Time Point",
            yaxis_title="Value",
            title="Profile Plot with Median Trend",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Mann-Whitney U Test":

        from scipy.stats import mannwhitneyu

        st.subheader("Interactive Mann-Whitney U Test")

        st.info("""
        **Mann-Whitney U Test** (also called Wilcoxon rank-sum test) is the **nonparametric alternative** 
        to the **Student's t-test (Independent)**.
        
        Use when:
        - Comparing **two independent groups**
        - Data is **not normally distributed**
        - Tests for **stochastic dominance** (is one group systematically larger/smaller?)
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("mannwhitney_2samp", mode="two_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            g1 = src["data"]["group1"]
            g2 = src["data"]["group2"]
            group_names = src["data"]["group_names"]
        else:
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

            np.random.seed(42)
            g1 = np.random.exponential(spread, n)
            g2 = np.random.exponential(spread, n) + location_shift
            group_names = ["Group 1", "Group 2"]

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        n1, n2 = len(g1), len(g2)
        medians_mw = [np.median(g1), np.median(g2)]
        means_mw = [np.mean(g1), np.mean(g2)]
        iqr_mw = [
            f"{np.percentile(g, 25):.2f}–{np.percentile(g, 75):.2f}" for g in [g1, g2]
        ]

        summary_data = {
            "Group": group_names,
            "n": [n1, n2],
            "Median": [f"{m:.3f}" for m in medians_mw],
            "Mean": [f"{m:.3f}" for m in means_mw],
            "IQR": iqr_mw,
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # TEST
        # =========================

        u, p = mannwhitneyu(g1, g2)

        from scipy.stats import norm as norm_mw

        z_mw = -norm_mw.ppf(p / 2) if p > 0 else 0
        n_total = n1 * n2
        r_rb_mw = 1 - 2 * min(u, n_total - u) / n_total if n_total > 0 else 0

        st.latex(rf"U = {u:.3f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Violin(y=g1, name=group_names[0], box_visible=True, meanline_visible=True))
        fig.add_trace(go.Violin(y=g2, name=group_names[1], box_visible=True, meanline_visible=True))

        fig.update_layout(
            template="plotly_dark",
            height=450,
            title=f"Violin Plots: {group_names[0]} vs {group_names[1]}",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        results_data = {
            "Metric": [
                f"Median {group_names[0]} (IQR)",
                f"Median {group_names[1]} (IQR)",
                "U-statistic",
                "z (normal approx)",
                "p-value",
                "Rank-biserial r (effect size)",
            ],
            "Value": [
                f"{medians_mw[0]:.3f} ({iqr_mw[0]})",
                f"{medians_mw[1]:.3f} ({iqr_mw[1]})",
                f"{u:.3f}",
                f"{z_mw:.3f}",
                format_p_value(p),
                f"{r_rb_mw:.4f}",
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

        st.info("""
        **Kruskal-Wallis Test** is the **nonparametric alternative** to **One-way ANOVA**.
        
        Use when:
        - Comparing **three or more independent groups**
        - Data is **not normally distributed**
        - Tests for differences in **medians** across groups (stochastic dominance)
        - Also called "one-way ANOVA on ranks"
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("kruskal_wallis", mode="multi_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            groups_kw = src["data"]["groups"]
            group_names = src["data"]["group_names"]
        else:
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

            np.random.seed(42)
            g1 = np.random.gamma(2, spread, 60)
            g2 = np.random.gamma(2, spread, 60) + shift
            g3 = np.random.gamma(2, spread, 60) + shift * 2
            groups_kw = [g1, g2, g3]
            group_names = ["Group 1", "Group 2", "Group 3"]

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        n_kw = [len(g) for g in groups_kw]
        medians_kw = [np.median(g) for g in groups_kw]
        means_kw = [np.mean(g) for g in groups_kw]
        iqr_kw = [
            f"{np.percentile(g, 25):.2f}–{np.percentile(g, 75):.2f}" for g in groups_kw
        ]

        summary_data = {
            "Group": group_names,
            "n": n_kw,
            "Median": [f"{m:.3f}" for m in medians_kw],
            "Mean": [f"{m:.3f}" for m in means_kw],
            "IQR": iqr_kw,
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # TEST
        # =========================

        H, p = kruskal(*groups_kw)

        n_total_kw = sum(n_kw)
        k_kw = len(groups_kw)
        df_kw = k_kw - 1
        eps_sq = H / (n_total_kw - 1) if n_total_kw > 1 else 0

        st.latex(rf"H = {H:.3f}")
        st.latex(rf"df = {df_kw}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i, (g, name) in enumerate(zip(groups_kw, group_names)):
            fig.add_trace(go.Box(y=g, name=name, boxmean="sd"))

        fig.update_layout(
            template="plotly_dark",
            height=450,
            title="Boxplots by Group",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        metric_list = [f"Median {name} (IQR)" for name in group_names] + [
            "H-statistic",
            "df",
            "p-value",
            "ε² (epsilon-squared, effect size)",
        ]
        value_list = [
            f"{m:.3f} ({iqr})" for m, iqr in zip(medians_kw, iqr_kw)
        ] + [
            f"{H:.3f}",
            f"{df_kw}",
            format_p_value(p),
            f"{eps_sq:.4f}",
        ]

        results_data = {
            "Metric": metric_list,
            "Value": value_list,
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(zip(groups_kw, group_names)):
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
            height=500,
            xaxis_title="Group",
            yaxis_title="Value",
            title="Boxplots with Individual Data Points",
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc(groups_kw, param_type="nonparametric", key="kw_ph")

    elif test_name == "Friedman Test":

        from scipy.stats import friedmanchisquare

        st.subheader("Interactive Friedman Test")

        st.info("""
        **Friedman Test** is the **nonparametric alternative** to **Repeated Measures ANOVA**.
        
        Use when:
        - Comparing **three or more measurements** from the **same subjects** (repeated measures)
        - Data is **not normally distributed** or measurements are ordinal
        - Tests for differences across conditions/time points
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("friedman", mode="repeated")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            measurements = src["data"]["measurements"]
            time_names = src["data"]["col_names"]
            n_subj = len(measurements[0])
        else:
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

            n_subj = st.slider(
                "Subjects",
                5,
                100,
                20,
                key="subjects_2",
            )

            np.random.seed(42)

            t1 = np.random.exponential(1, n_subj)
            t2 = t1 + trend + np.random.normal(0, noise, n_subj)
            t3 = t2 + trend + np.random.normal(0, noise, n_subj)
            measurements = [t1, t2, t3]
            time_names = ["T1", "T2", "T3"]

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        medians_f = [np.median(m) for m in measurements]
        means_f = [np.mean(m) for m in measurements]
        iqr_f = [
            f"{np.percentile(m, 25):.2f}–{np.percentile(m, 75):.2f}" for m in measurements
        ]

        summary_data = {
            "Time/Condition": time_names,
            "Median": [f"{m:.3f}" for m in medians_f],
            "Mean": [f"{m:.3f}" for m in means_f],
            "IQR": iqr_f,
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # TEST
        # =========================

        stat, p = friedmanchisquare(*measurements)

        k_f = len(measurements)
        df_f = k_f - 1
        kendall_w = stat / (n_subj * (k_f - 1)) if n_subj > 0 and k_f > 1 else 0

        st.latex(rf"\chi^2 = {stat:.3f}")
        st.latex(rf"df = {df_f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i in range(n_subj):
            fig.add_trace(
                go.Scatter(
                    x=time_names,
                    y=[measurements[j][i] for j in range(len(measurements))],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=time_names,
                y=medians_f,
                mode="lines+markers",
                name="Median trend",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=10),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Time/Condition",
            yaxis_title="Value",
            title=f"Individual Subject Trajectories (n={n_subj})",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        metric_list = [f"Median {name}" for name in time_names] + [
            "χ² (Friedman chi-square)",
            "df",
            "p-value",
            "Kendall's W (effect size, concordance)",
        ]
        value_list = [f"{m:.3f}" for m in medians_f] + [
            f"{stat:.3f}",
            f"{df_f}",
            format_p_value(p),
            f"{kendall_w:.4f}",
        ]

        results_data = {
            "Metric": metric_list,
            "Value": value_list,
        }
        st.table(pd.DataFrame(results_data))

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc(measurements, param_type="nonparametric", key="friedman_ph")

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

        st.info("""
        **Pearson Correlation (r)** measures the **linear relationship** between two continuous variables.
        
        - r = +1: perfect positive linear relationship
        - r = -1: perfect negative linear relationship  
        - r = 0: no linear relationship
        
        Assumptions: normality, linearity, homoscedasticity
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("pearson", mode="correlation")

        # =========================
        # CONTROLS / DATA
        # =========================

        np.random.seed(42)

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
            n = len(x)
            from scipy.stats import pearsonr
            r, p = pearsonr(x, y)
        else:
            col1, col2 = st.columns(2)

            with col1:
                r = st.slider("Correlation Coefficient (r)", -1.0, 1.0, 0.5, 0.01)

            with col2:
                n = st.slider("Sample Size (n)", 3, 100, 30)

            x = np.random.normal(size=n)
            y = r * x + np.sqrt(1 - r**2) * np.random.normal(size=n)
            col_names = ["X", "Y"]
            from scipy.stats import pearsonr
            _, p = pearsonr(x, y)

        # =========================
        # STATS
        # =========================

        st.latex(rf"""
            r = {r:.2f}
            """)

        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # REGRESSION LINE
        # =========================

        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(x, y)
        x_line = np.array([min(x), max(x)])
        y_line = intercept + slope * x_line

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data Points",
                marker=dict(color="rgba(100, 150, 255, 0.7)", size=8),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Regression: y = {intercept:.2f} + {slope:.2f}x",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title=col_names[0],
            yaxis_title=col_names[1],
            title=f"r = {r:.3f}, n = {n}",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Summary Statistics")

        summary_data = {
            "Variable": col_names,
            "n": [n, n],
            "Mean": [f"{np.mean(x):.3f}", f"{np.mean(y):.3f}"],
            "SD": [f"{np.std(x, ddof=1):.3f}", f"{np.std(y, ddof=1):.3f}"],
        }
        st.table(pd.DataFrame(summary_data))

    elif test_name == "Spearman Rank Correlation":

        from scipy.stats import spearmanr

        st.subheader("Interactive Spearman Rank Correlation")

        st.info("""
        **Spearman's ρ (rho)** is the **nonparametric alternative** to Pearson's r.
        
        Use when:
        - Relationship is **monotonic** but not necessarily linear
        - Data is **ordinal** or **not normally distributed**
        - Measures rank-based association
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("spearman", mode="correlation")

        # =========================
        # CONTROLS / DATA
        # =========================

        np.random.seed(42)

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
        else:
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

            x = np.linspace(0, 10, 300)
            direction_multiplier = 1 if direction == "Positive" else -1
            y = direction_multiplier * (x**curve_strength) + np.random.normal(0, noise, 300)
            col_names = ["X", "Y"]

        # =========================
        # TEST
        # =========================

        rho, p = spearmanr(x, y)

        st.latex(rf"\rho = {rho:.3f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data",
                marker=dict(color="rgba(100, 200, 150, 0.6)", size=6),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title=col_names[0],
            yaxis_title=col_names[1],
            title=f"Spearman ρ = {rho:.3f}",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Summary Statistics")

        from scipy.stats import pearsonr
        r_pearson, _ = pearsonr(x, y)

        summary_data = {
            "Metric": [
                "Sample Size (n)",
                "Spearman's ρ",
                "Pearson's r (for comparison)",
                "p-value",
            ],
            "Value": [
                f"{len(x)}",
                f"{rho:.4f}",
                f"{r_pearson:.4f}",
                format_p_value(p),
            ],
        }
        st.table(pd.DataFrame(summary_data))

    elif test_name == "Kendall's Tau-b":

        from scipy.stats import kendalltau

        st.subheader("Interactive Kendall's Tau-b")

        st.info("""
        **Kendall's τ-b (tau-b)** is another **nonparametric** measure of rank correlation.
        
        - Based on **concordant/discordant pairs**
        - Good for **small samples** or when there are **ties**
        - τ-b corrects for ties (unlike the simpler τ-a)
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("kendall", mode="correlation")

        # =========================
        # CONTROLS / DATA
        # =========================

        np.random.seed(42)

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
        else:
            strength = st.slider("Association Strength", 0.0, 1.0, 0.5, 0.05)

            n = st.slider("Sample Size", 10, 200, 60, key="kendall_s_tau_b_sample_size")

            noise = st.slider("Noise", 0.1, 5.0, 1.0, 0.1, key="kendall_s_tau_b_noise")

            x = np.random.normal(0, 1, n)
            y = strength * x + np.random.normal(0, noise, n)
            col_names = ["X", "Y"]

        # =========================
        # TEST
        # =========================

        tau, p = kendalltau(x, y)

        st.latex(rf"\tau_b = {tau:.3f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data",
                marker=dict(color="rgba(200, 150, 100, 0.6)", size=8),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title=col_names[0],
            yaxis_title=col_names[1],
            title=f"Kendall's τ-b = {tau:.3f}",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # COMPARISON
        # =========================

        st.divider()
        st.subheader("Comparison with Other Correlation Measures")

        from scipy.stats import pearsonr, spearmanr
        r, _ = pearsonr(x, y)
        rho, _ = spearmanr(x, y)

        comp_data = {
            "Measure": ["Pearson's r", "Spearman's ρ", "Kendall's τ-b"],
            "Value": [f"{r:.4f}", f"{rho:.4f}", f"{tau:.4f}"],
            "Type": ["Linear (parametric)", "Monotonic (nonparametric)", "Concordant pairs (nonparametric)"],
        }
        st.table(pd.DataFrame(comp_data))

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

        from scipy.stats import pointbiserialr, ttest_ind

        st.subheader("Interactive Point-Biserial Correlation")

        st.info("""
        **Point-Biserial Correlation (r_pb)** measures the relationship between a
        **binary/dichotomous variable** and a **continuous variable**.
        
        - Mathematically equivalent to **Pearson's r** with one binary variable
        - Also directly related to the **independent samples t-test**
        - r_pb² = proportion of variance explained by group membership
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("pointbiserial", mode="two_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        np.random.seed(42)

        if src["using_uploaded"]:
            g0 = src["data"]["group1"]
            g1 = src["data"]["group2"]
            group_names = src["data"]["group_names"]
            
            group = np.concatenate([np.zeros(len(g0)), np.ones(len(g1))])
            y = np.concatenate([g0, g1])
        else:
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

            group = np.random.binomial(1, 0.5, 300)
            y = group * group_difference + np.random.normal(0, noise, 300)
            
            g0 = y[group == 0]
            g1 = y[group == 1]
            group_names = ["Group 0", "Group 1"]

        # =========================
        # TEST
        # =========================

        group_for_pb = np.concatenate([np.zeros(len(g0)), np.ones(len(g1))])
        y_for_pb = np.concatenate([g0, g1])
        r, p = pointbiserialr(group_for_pb, y_for_pb)

        st.latex(rf"r_{{pb}} = {r:.3f}")
        st.latex(rf"\text{{{format_p_value(p)}}}")
        st.latex(rf"R^2 = {r**2:.4f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=g0,
                name=group_names[0],
                boxmean="sd",
            )
        )

        fig.add_trace(
            go.Box(
                y=g1,
                name=group_names[1],
                boxmean="sd",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            yaxis_title="Continuous Variable",
            title=f"Point-Biserial r = {r:.3f}",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # SUMMARY & COMPARISON
        # =========================

        st.divider()
        st.subheader("Group Statistics & Comparison")

        n0, n1 = len(g0), len(g1)
        mean0, mean1 = np.mean(g0), np.mean(g1)
        sd0, sd1 = np.std(g0, ddof=1), np.std(g1, ddof=1)
        
        t_ind, p_ind = ttest_ind(g0, g1)
        
        summary_data = {
            "Group": group_names,
            "n": [n0, n1],
            "Mean": [f"{mean0:.3f}", f"{mean1:.3f}"],
            "SD": [f"{sd0:.3f}", f"{sd1:.3f}"],
        }
        st.table(pd.DataFrame(summary_data))

        st.info(f"""
        **Relationship with t-test:**
        - t({n0+n1-2}) = {t_ind:.3f}, {format_p_value(p_ind)}
        - r_pb² = {(r**2):.4f} ({(r**2)*100:.1f}% of variance explained by group membership)
        """)

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

        st.info("""
        **Simple Linear Regression** models the linear relationship between:
        - A **predictor variable (X)**
        - An **outcome variable (Y)**
        
        Model: Y = β₀ + β₁X + ε
        
        - β₀: intercept (value of Y when X=0)
        - β₁: slope (change in Y per unit change in X)
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("regression", mode="correlation")

        # =========================
        # CONTROLS / DATA
        # =========================

        np.random.seed(42)

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
            n = len(x)
        else:
            col1, col2, col3 = st.columns(3)

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

            with col3:
                noise = st.slider(
                    "Noise (σ)",
                    0.0,
                    20.0,
                    3.0,
                    0.1,
                    key="slr_noise",
                )

            n = st.slider("Sample Size (n)", 10, 500, 100, key="slr_n")

            x = np.linspace(0, 10, n)
            y_true = beta0 + beta1 * x
            y = y_true + np.random.normal(0, noise, n)
            col_names = ["X", "Y"]

        # =========================
        # FIT REGRESSION
        # =========================

        from scipy.stats import linregress, pearsonr
        result = linregress(x, y)
        slope, intercept, r, p, se_slope = result
        
        r, _ = pearsonr(x, y)
        r2 = r ** 2
        
        y_pred = intercept + slope * x
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"\hat{{y}} = {intercept:.3f} + ({slope:.3f})x")
        st.latex(rf"R^2 = {r2:.4f}")
        st.latex(rf"F_{{1,{n-2}}} \text{{ p = {format_p_value(p)}}}")

        # =========================
        # PLOT - SCATTER WITH REGRESSION LINE
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Observed Data",
                marker=dict(color="rgba(100, 150, 255, 0.6)", size=8),
            )
        )

        sorted_idx = np.argsort(x)
        fig.add_trace(
            go.Scatter(
                x=x[sorted_idx],
                y=y_pred[sorted_idx],
                mode="lines",
                name=f"Regression: ŷ = {intercept:.2f} + {slope:.2f}x",
                line=dict(color="red", width=3),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title=col_names[0],
            yaxis_title=col_names[1],
            title=f"Simple Linear Regression: R² = {r2:.4f}",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED RESULTS TABLE
        # =========================

        st.divider()
        st.subheader("Regression Coefficients")

        from scipy import stats
        t_intercept = intercept / (np.std(residuals) * np.sqrt(1/n + np.mean(x)**2 / np.sum((x - np.mean(x))**2))) if n > 2 else np.nan

        coeff_data = {
            "Term": ["Intercept (β₀)", "Slope (β₁)"],
            "Estimate": [f"{intercept:.4f}", f"{slope:.4f}"],
            "SE": ["(not computed)", f"{se_slope:.4f}"],
            "t": ["-", f"{slope/se_slope:.3f}" if se_slope > 0 else "-"],
            "p-value": ["-", format_p_value(p)],
        }
        st.table(pd.DataFrame(coeff_data))

        # =========================
        # MODEL FIT STATISTICS
        # =========================

        st.divider()
        st.subheader("Model Fit")

        fit_data = {
            "Metric": [
                "Sample Size (n)",
                "Pearson r",
                "R-squared (R²)",
                "Adjusted R²",
                "RMSE (Root MSE)",
            ],
            "Value": [
                f"{n}",
                f"{r:.4f}",
                f"{r2:.4f}",
                f"{1 - (1 - r2) * (n - 1) / (n - 2):.4f}" if n > 2 else "-",
                f"{rmse:.4f}",
            ],
        }
        st.table(pd.DataFrame(fit_data))

        # =========================
        # RESIDUAL PLOT
        # =========================

        st.divider()
        st.subheader("Residual Plot (Check Homoscedasticity)")

        fig_resid = go.Figure()

        fig_resid.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(color="rgba(255, 150, 100, 0.6)", size=6),
            )
        )

        fig_resid.add_hline(
            y=0,
            line=dict(color="red", dash="dash"),
            name="Zero Line",
        )

        fig_resid.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Predicted Values (ŷ)",
            yaxis_title="Residuals (y - ŷ)",
            title="Residuals vs Fitted Values",
        )

        st.plotly_chart(fig_resid, use_container_width=True)

        st.info("""
        **How to interpret the residual plot:**
        - ✅ Good: Random scatter around 0, no pattern
        - ❌ Funnel shape = Heteroscedasticity (non-constant variance)
        - ❌ Curved pattern = Nonlinear relationship (consider adding polynomial term)
        """)

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
