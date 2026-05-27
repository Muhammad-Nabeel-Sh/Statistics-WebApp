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
            expected_p = st.slider("Expected Proportion (H₀)", 0.0, 1.0, 0.5, 0.01, key="prop_1samp_expected")
            st.info(f"Using first category **'{categories[0]}'** as success ({successes}/{n} = {observed_p:.1%})")
        else:
            expected_p = st.slider("Expected Proportion", 0.0, 1.0, 0.5, 0.01)
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

    elif test_name == "Binomial Test":

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
            expected_p_bt = st.slider("Hypothesized Probability (p₀)", 0.0, 1.0, 0.5, 0.01, key="binom_exact_expected_p")
            successes_bt = st.slider("Number of Successes", 0, 100, 15, key="binom_exact_successes")
            n_bt = st.slider("Number of Trials (n)", 1, 200, 30, key="binom_exact_n")

        # =========================
        # TEST
        # =========================

        if src["using_uploaded"]:
            expected_p_bt = st.slider("Hypothesized Probability (p₀)", 0.0, 1.0, 0.5, 0.01, key="binom_exact_expected_p_uploaded")

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

    elif test_name == "Sign Test (One-sample)":

        from scipy.stats import binomtest

        st.subheader("Interactive Sign Test (One-sample)")

        st.info("""
        **Sign Test (One-sample)** tests whether the median of a sample differs from a hypothesized value.
        It examines only the signs (positive/negative) of differences, ignoring magnitudes.
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("signtest_1samp", mode="one_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        hypothesized_median = st.slider("Hypothesized Median (H₀)", -10.0, 10.0, 0.0, 0.5)

        if src["using_uploaded"]:
            values = np.array(src["data"]["values"])
            diffs = values - hypothesized_median
        else:
            median_shift = st.slider("Median Shift from H₀", -5.0, 5.0, 1.0, 0.1, key="signtest_1samp_shift")
            n = st.slider("Sample Size", 10, 200, 40, key="signtest_1samp_n")
            np.random.seed(42)
            values = np.random.exponential(1, n) + hypothesized_median + median_shift
            diffs = values - hypothesized_median

        # =========================
        # TEST
        # =========================

        signs = np.sign(diffs)
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        n_total = n_pos + n_neg

        if n_total > 0:
            p_sig = 2 * binomtest(max(n_pos, n_neg), n_total, 0.5).pvalue
        else:
            p_sig = 1.0

        st.latex(rf"\text{{Positives}} = {n_pos},\ \text{{Negatives}} = {n_neg}")
        st.latex(rf"\text{{{format_p_value(p_sig)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()
        fig.add_trace(go.Box(y=values, name="Sample", boxmean="sd", marker_color="#4C78A8"))
        fig.add_hline(y=hypothesized_median, line_dash="dash", line_color="red", annotation_text="H₀ Median")
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED RESULTS
        # =========================

        st.divider()
        st.subheader("Detailed Results")
        results_data = {
            "Metric": ["Sample Median", "Hypothesized Median", "n (non-zero diffs)", "Positives", "Negatives", "p-value"],
            "Value": [f"{np.median(values):.3f}", f"{hypothesized_median}", f"{n_total}", f"{n_pos}", f"{n_neg}", format_p_value(p_sig)],
        }
        st.table(pd.DataFrame(results_data))

    elif test_name == "Multinomial Test":

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

    elif test_name == "Chi-Square Goodness-of-Fit Test":

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

    elif test_name == "Poisson Goodness-of-Fit Test":

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
    elif test_name == "Chi-Square Test":

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

    elif test_name == "McNemar's Test":

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

    elif test_name == "Cochran's Q Test":

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

    elif test_name == "Fisher's Exact Test":

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

    elif test_name == "One-way Welch ANOVA":

        st.subheader("Interactive One-way Welch ANOVA")

        st.info("""
        **Welch's ANOVA** is an alternative to **One-way ANOVA** that **does not assume equal variances**.
        It uses Welch's F-statistic with adjusted degrees of freedom (Welch, 1951).
        Use when group variances are heterogeneous.
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("welch_anova", mode="multi_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            groups_w = src["data"]["groups"]
            group_names_w = src["data"]["group_names"]
        else:
            mean_shift = st.slider("Group Separation", 0.0, 10.0, 2.0, 0.1, key="welch_mean_shift")
            noise_base = st.slider("Base Noise", 0.1, 5.0, 1.0, 0.1, key="welch_noise_base")
            noise_scale = st.slider("Noise Ratio (group 3 vs group 1)", 1.0, 5.0, 3.0, 0.1, key="welch_noise_scale")

            np.random.seed(42)
            g1_w = np.random.normal(0, noise_base, 60)
            g2_w = np.random.normal(mean_shift, noise_base * 1.5, 60)
            g3_w = np.random.normal(mean_shift * 2, noise_base * noise_scale, 60)
            groups_w = [g1_w, g2_w, g3_w]
            group_names_w = ["Group 1 (low var)", "Group 2 (med var)", "Group 3 (high var)"]

        # =========================
        # STATISTICAL COMPUTATION
        # =========================

        from scipy.stats import f as f_dist_w

        k_w = len(groups_w)
        n_w = np.array([len(g) for g in groups_w])
        means_w = np.array([np.mean(g) for g in groups_w])
        vars_w = np.array([np.var(g, ddof=1) for g in groups_w])

        weights = n_w / vars_w
        x_tilde = np.sum(weights * means_w) / np.sum(weights)

        numer_w = np.sum(weights * (means_w - x_tilde) ** 2) / (k_w - 1)
        denom_w = 1 + (2 * (k_w - 2) / (k_w ** 2 - 1)) * np.sum(((1 - weights / np.sum(weights)) ** 2) / (n_w - 1))
        F_welch = numer_w / denom_w

        df1_w = k_w - 1
        df2_w = 1 / ((3 / (k_w ** 2 - 1)) * np.sum(((1 - weights / np.sum(weights)) ** 2) / (n_w - 1)))
        p_w = 1 - f_dist_w.cdf(F_welch, df1_w, df2_w)

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        sds_w = [np.std(g, ddof=1) for g in groups_w]
        summary_data = {
            "Group": group_names_w,
            "n": [int(n) for n in n_w],
            "Mean": [f"{m:.3f}" for m in means_w],
            "SD": [f"{s:.3f}" for s in sds_w],
            "Variance": [f"{v:.3f}" for v in vars_w],
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # STATS
        # =========================

        st.latex(rf"F_{{Welch}} = {F_welch:.3f}")
        st.latex(rf"df = ({df1_w:.0f}, {df2_w:.2f})")
        st.latex(rf"\text{{{format_p_value(p_w)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()
        for i, (g, name) in enumerate(zip(groups_w, group_names_w)):
            fig.add_trace(go.Box(y=g, name=name))
        fig.update_layout(template="plotly_dark", height=550, title="Boxplots by Group")
        st_plot_with_download(fig, key="welch_anova_box", height=550)

        # =========================
        # DETAILED RESULTS
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        metric_list = [f"Mean {name} (SD)" for name in group_names_w] + [
            "F (Welch)",
            f"df1",
            f"df2",
            "p-value",
        ]
        value_list = [f"{m:.2f} ({s:.2f})" for m, s in zip(means_w, sds_w)] + [
            f"{F_welch:.3f}",
            f"{df1_w:.0f}",
            f"{df2_w:.2f}",
            format_p_value(p_w),
        ]
        st.table(pd.DataFrame({"Metric": metric_list, "Value": value_list}))

        st.divider()
        st.subheader("Post-Hoc Tests")
        render_post_hoc(groups_w, param_type="parametric", key="welch_anova_ph")

    elif test_name == "Two-way ANOVA":

        from scipy.stats import f_oneway
        from scipy.stats import f as f_dist

        st.subheader("Interactive Two-way ANOVA")

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("twoway_anova", mode="multi_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            groups_tw = src["data"]["groups"]
            group_names_tw = src["data"]["group_names"]
            n_per_cell = len(groups_tw[0])
            A1B1 = groups_tw[0]
            A1B2 = groups_tw[1] if len(groups_tw) > 1 else np.array([])
            A2B1 = groups_tw[2] if len(groups_tw) > 2 else np.array([])
            A2B2 = groups_tw[3] if len(groups_tw) > 3 else np.array([])
        else:
            effect_A = st.slider("Effect of Factor A (Group)", 0.0, 10.0, 2.0, 0.1, key="tw_effect_A")

            effect_B = st.slider("Effect of Factor B (Sex)", 0.0, 10.0, 1.0, 0.1, key="tw_effect_B")

            interaction = st.slider("Interaction (A × B)", -5.0, 5.0, 0.0, 0.1, key="tw_interaction")

            noise = st.slider("Within-group Variability", 0.1, 5.0, 1.0, 0.1, key="tw_noise")

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

        if src["using_uploaded"]:
            names_tw = group_names_tw
        else:
            names_tw = ["A1 (Control), B1 (Male)", "A1 (Control), B2 (Female)", "A2 (Drug), B1 (Male)", "A2 (Drug), B2 (Female)"]
        fig.add_trace(go.Box(y=A1B1, name=names_tw[0]))
        fig.add_trace(go.Box(y=A1B2, name=names_tw[1] if len(names_tw) > 1 else ""))
        fig.add_trace(go.Box(y=A2B1, name=names_tw[2] if len(names_tw) > 2 else ""))
        fig.add_trace(go.Box(y=A2B2, name=names_tw[3] if len(names_tw) > 3 else ""))

        fig.update_layout(template="plotly_dark", height=550)

        st_plot_with_download(fig, key="twoway_anova_box", height=550)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from statsmodels.formula.api import ols as ols_tw
        import statsmodels.api as sm_tw

        if src["using_uploaded"] and len(groups_tw) == 4:
            parts = [n.replace("_", " ").replace("-", " ").split() for n in group_names_tw]
            if all(len(p) >= 2 for p in parts):
                a1_label, a2_label = parts[0][0], parts[2][0]
                b1_label, b2_label = parts[0][1], parts[1][1]
                a_labels = [a1_label, a1_label, a2_label, a2_label]
                b_labels = [b1_label, b2_label, b1_label, b2_label]
            else:
                a_labels = ["A1", "A1", "A2", "A2"]
                b_labels = ["B1", "B2", "B1", "B2"]
            df_tw = pd.DataFrame({
                "y": np.concatenate([A1B1, A1B2, A2B1, A2B2]),
                "A": np.repeat(a_labels, n_per_cell),
                "B": np.repeat(b_labels, n_per_cell),
            })
        else:
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("ancova", mode="multi_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            groups_anc = src["data"]["groups"]
            group_names_anc = src["data"]["group_names"]
            n_groups_anc = len(groups_anc)
            if n_groups_anc < 2:
                st.error("ANCOVA needs at least 2 groups. Using simulated data.")
                src = {"using_uploaded": False, "data": None}
            else:
                treatment_effect = st.slider("Treatment Effect", 0.0, 10.0, 3.0, 0.1, key="treatment_effect_anc_uploaded")
                st.info("Workspace data provides group comparison. ANCOVA also needs a covariate — using a random simulated covariate for illustration.")
                np.random.seed(42)
                n = min(len(g) for g in groups_anc)
                covariate = np.random.normal(50, 10, n)
                control = groups_anc[0][:n]
                treatment = groups_anc[1][:n]
                group_names_anc_short = group_names_anc[:2]
        else:
            treatment_effect = st.slider("Treatment Effect", 0.0, 10.0, 3.0, 0.1, key="treatment_effect_1")
            covariate_strength = st.slider("Covariate Strength (β)", 0.0, 3.0, 1.0, 0.1)
            noise = st.slider("Noise", 0.1, 5.0, 1.0, 0.1, key="ancova_noise")

            np.random.seed(42)
            n = 40

            covariate = np.random.normal(50, 10, n)

            control = covariate * covariate_strength + np.random.normal(0, noise, n)

            treatment = (
                covariate * covariate_strength
                + treatment_effect
                + np.random.normal(0, noise, n)
            )
            group_names_anc_short = ["Control", "Treatment"]

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

        gname0 = group_names_anc_short[0] if group_names_anc_short else "Group 0"
        gname1 = group_names_anc_short[1] if len(group_names_anc_short) > 1 else "Group 1"
        df_anc = pd.DataFrame(
            {
                "outcome": np.concatenate([control, treatment]),
                "group": np.repeat([gname0, gname1], n),
                "cov": np.tile(covariate, 2),
            }
        )
        model_anc = ols_anc("outcome ~ C(group) + cov", data=df_anc).fit()
        beta_cov = model_anc.params["cov"]
        grand_mean_cov = np.mean(covariate)
        adj_control = model_anc.params["Intercept"] + beta_cov * grand_mean_cov
        trt_param = [k for k in model_anc.params.keys() if "C(group)" in k and "T." in k]
        adj_treatment = (
            model_anc.params["Intercept"]
            + (model_anc.params[trt_param[0]] if trt_param else 0)
            + beta_cov * grand_mean_cov
        )
        F_group = model_anc.fvalue
        p_group = model_anc.f_pvalue
        ss_resid = model_anc.ssr
        ss_expl = model_anc.ess
        partial_eta_anc = ss_expl / (ss_expl + ss_resid)

        results_data = {
            "Metric": [
                f"Adj. Mean {gname0}",
                f"Adj. Mean {gname1}",
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("rm_anova", mode="repeated")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            measurements_rm = src["data"]["measurements"]
            col_names_rm = src["data"]["col_names"]
            subjects = len(measurements_rm[0])
            timepoints = len(measurements_rm)
            data = [np.array([measurements_rm[t][s] for t in range(timepoints)]) for s in range(subjects)]
        else:
            trend = st.slider(
                "Time Trend",
                -5.0,
                5.0,
                1.0,
                0.1,
                key="rm_trend",
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("manova", mode="multi_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            groups_m = src["data"]["groups"]
            group_names_m = src["data"]["group_names"]
        else:
            separation = st.slider(
                "Group Separation",
                0.0,
                10.0,
                3.0,
                0.1,
                key="group_separation_1",
            )

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

        if src["using_uploaded"]:
            manova_groups = groups_m
            manova_names = group_names_m
            n_dims = manova_groups[0].shape[1] if manova_groups[0].ndim > 1 else 1
        else:
            manova_groups = [g1, g2]
            manova_names = ["Group 1", "Group 2"]
            n_dims = 3

        fig = go.Figure()

        if n_dims >= 3 and not src["using_uploaded"]:
            for i, (g, name) in enumerate(zip(manova_groups, manova_names)):
                fig.add_trace(
                    go.Scatter3d(
                        x=g[:, 0], y=g[:, 1], z=g[:, 2],
                        mode="markers", name=name,
                    )
                )
            fig.update_layout(template="plotly_dark", height=700)
        else:
            colors = ["blue", "red", "green", "orange", "purple", "brown"]
            for i, (g, name) in enumerate(zip(manova_groups, manova_names)):
                c = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=g[:, 0] if g.ndim > 1 else np.arange(len(g)),
                        y=g[:, 1] if g.ndim > 1 else g,
                        mode="markers", name=name,
                        marker=dict(color=c, size=6, opacity=0.5),
                    )
                )
            fig.update_layout(template="plotly_dark", height=600)

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        k_m = len(manova_groups)
        n_m = len(manova_groups[0])
        p_m_dv = manova_groups[0].shape[1] if manova_groups[0].ndim > 1 else 1

        if k_m >= 2 and p_m_dv >= 2:
            y_all = np.vstack(manova_groups)
            grand_mean = np.mean(y_all, axis=0)
            H_m = sum(
                n_m * np.outer(np.mean(g, axis=0) - grand_mean, np.mean(g, axis=0) - grand_mean)
                for g in manova_groups
            )
            E_m = sum((n_m - 1) * np.cov(g, rowvar=False) for g in manova_groups)
            wilks_lambda = np.linalg.det(E_m) / np.linalg.det(E_m + H_m)

            df1_m = p_m_dv
            df2_m = k_m * n_m - p_m_dv - 1
            F_m = ((1 - wilks_lambda) / wilks_lambda) * (df2_m / df1_m)

            from scipy.stats import f as f_dist_m

            p_m_val = 1 - f_dist_m.cdf(F_m, df1_m, df2_m)
        else:
            wilks_lambda = F_m = p_m_val = df1_m = df2_m = None

        results_data = {
            "Metric": [
                "Wilks' Λ",
                "F-approx",
                f"df ({df1_m}, {df2_m:.0f})" if df1_m else "df",
                "p-value",
                "Number of DVs",
                "Number of Groups",
                "",
                "",
            ],
            "Value": [
                f"{wilks_lambda:.4f}" if wilks_lambda else "N/A",
                f"{F_m:.3f}" if F_m else "N/A",
                f"{df1_m}, {df2_m:.0f}" if df1_m else "N/A",
                f"{p_m_val:.5f}" if p_m_val else "N/A",
                f"{p_m_dv}",
                f"{k_m}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        if p_m_dv >= 2:
            fig2 = go.Figure()
            colors = ["blue", "red", "green", "orange", "purple", "brown"]
            for i, (g, name) in enumerate(zip(manova_groups, manova_names)):
                c = colors[i % len(colors)]
                fig2.add_trace(
                    go.Scatter(
                        x=g[:, 0],
                        y=g[:, 1],
                        mode="markers",
                        name=name,
                        marker=dict(color=c, size=6, opacity=0.5),
                    )
                )
                centroid = np.mean(g[:, :2], axis=0)
                fig2.add_trace(
                    go.Scatter(
                        x=[centroid[0]],
                        y=[centroid[1]],
                        mode="markers",
                        showlegend=False,
                        marker=dict(color=c, size=15, symbol="x"),
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
        if n_m >= 3:
            render_post_hoc([g[:, 0] for g in manova_groups], param_type="parametric", key="manova_ph")

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

    elif test_name == "Sign Test (Paired)":

        st.subheader("Interactive Sign Test (Paired)")

        st.info("""
        **Sign Test (Paired)** compares two related measurements by examining the signs of their differences.
        It asks: Is one measurement typically greater than the other?
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("signtest_paired", mode="paired")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            values1 = np.array(src["data"]["values1"])
            values2 = np.array(src["data"]["values2"])
        else:
            shift = st.slider("Median Shift", -5.0, 5.0, 1.0, 0.1, key="signtest_paired_shift")
            noise = st.slider("Noise", 0.1, 5.0, 1.0, 0.1, key="signtest_paired_noise")
            n = st.slider("Sample Size", 10, 200, 40, key="signtest_paired_n")
            np.random.seed(42)
            values1 = np.random.exponential(1, n)
            values2 = values1 + shift + np.random.normal(0, noise, n)

        # =========================
        # TEST
        # =========================

        diffs = values2 - values1
        signs = np.sign(diffs)
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        n_total = n_pos + n_neg

        if n_total > 0:
            from scipy.stats import binomtest
            p_sig = 2 * binomtest(max(n_pos, n_neg), n_total, 0.5).pvalue
        else:
            p_sig = 1.0

        st.latex(rf"\text{{Positives}} = {n_pos},\ \text{{Negatives}} = {n_neg}")
        st.latex(rf"\text{{{format_p_value(p_sig)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()
        fig.add_trace(go.Box(y=diffs, name="Differences", boxmean="sd", marker_color="#4C78A8"))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="No difference")
        fig.update_layout(template="plotly_dark", height=450, title="Paired Differences")
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED RESULTS
        # =========================

        st.divider()
        st.subheader("Detailed Results")
        results_data = {
            "Metric": ["Median Difference", "n (non-zero diffs)", "Positives", "Negatives", "p-value"],
            "Value": [f"{np.median(diffs):.3f}", f"{n_total}", f"{n_pos}", f"{n_neg}", format_p_value(p_sig)],
        }
        st.table(pd.DataFrame(results_data))

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

    elif test_name == "F-Test for Two Variances":

        from scipy.stats import f as f_dist

        st.subheader("Interactive F-Test for Two Variances")

        st.info("""
        **F-Test for Two Variances** compares the variances of two independent samples.
        It tests H₀: σ₁² = σ₂² using the ratio of sample variances F = s₁² / s₂².
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("ftest_2var", mode="two_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            group1 = np.array(src["data"]["group1"])
            group2 = np.array(src["data"]["group2"])
            g1_name, g2_name = src["data"]["group_names"]
        else:
            var1 = st.slider("Group 1 Variance (σ₁²)", 0.1, 10.0, 2.0, 0.1, key="ftest_var1")
            var2 = st.slider("Group 2 Variance (σ₂²)", 0.1, 10.0, 1.0, 0.1, key="ftest_var2")
            n1 = st.slider("Group 1 Sample Size", 5, 200, 30, key="ftest_n1")
            n2 = st.slider("Group 2 Sample Size", 5, 200, 30, key="ftest_n2")
            np.random.seed(42)
            group1 = np.random.normal(0, np.sqrt(var1), n1)
            group2 = np.random.normal(0, np.sqrt(var2), n2)
            g1_name, g2_name = "Group 1", "Group 2"

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        s1_sq = np.var(group1, ddof=1)
        s2_sq = np.var(group2, ddof=1)
        n1 = len(group1)
        n2 = len(group2)

        summary_data = {
            "Metric": [f"{g1_name}", f"{g2_name}"],
            "n": [n1, n2],
            "Variance": [f"{s1_sq:.4f}", f"{s2_sq:.4f}"],
            "SD": [f"{np.sqrt(s1_sq):.4f}", f"{np.sqrt(s2_sq):.4f}"],
        }
        st.table(pd.DataFrame(summary_data))

        # =========================
        # TEST
        # =========================

        if s1_sq >= s2_sq:
            f_stat = s1_sq / s2_sq if s2_sq > 0 else float("inf")
            df1 = n1 - 1
            df2 = n2 - 1
        else:
            f_stat = s2_sq / s1_sq if s1_sq > 0 else float("inf")
            df1 = n2 - 1
            df2 = n1 - 1

        if np.isfinite(f_stat):
            p_f = 2 * (1 - f_dist.cdf(f_stat, df1, df2))
        else:
            p_f = 0.0

        st.latex(rf"F = {f_stat:.4f}")
        st.latex(rf"\text{{{format_p_value(p_f)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()
        fig.add_trace(go.Box(y=group1, name=g1_name, boxmean="sd", marker_color="#4C78A8"))
        fig.add_trace(go.Box(y=group2, name=g2_name, boxmean="sd", marker_color="#F58518"))
        fig.update_layout(template="plotly_dark", height=450, title="Group Comparison (Variance)")
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED RESULTS
        # =========================

        st.divider()
        st.subheader("Detailed Results")
        results_data = {
            "Metric": [
                f"Variance ({g1_name})", f"Variance ({g2_name})",
                "F-statistic", "df1", "df2", "p-value",
                "Ratio (larger/smaller)",
            ],
            "Value": [
                f"{s1_sq:.4f}", f"{s2_sq:.4f}",
                f"{f_stat:.4f}", f"{df1}", f"{df2}",
                format_p_value(p_f),
                f"{max(s1_sq, s2_sq) / min(s1_sq, s2_sq) if min(s1_sq, s2_sq) > 0 else float('inf'):.2f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

    elif test_name == "Equivalence Test (TOST) - Two Independent Samples":

        from scipy.stats import t as t_tost, nct

        st.subheader("Interactive Equivalence Test (TOST)")
        st.info("""
        **Two One-Sided Tests (TOST)** determines if two means are *practically equivalent* within
        a pre-specified equivalence bound ±Δ. Unlike a traditional t-test, TOST reverses the burden:
        H₀ assumes a meaningful difference exists, and you seek evidence for equivalence.
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("tost", mode="two_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        delta_eq = st.slider("Equivalence Bound (±Δ)", 0.1, 5.0, 1.0, 0.1, key="tost_delta",
                             help="Maximum practically insignificant difference between means")

        if src["using_uploaded"]:
            g1 = np.array(src["data"]["group1"])
            g2 = np.array(src["data"]["group2"])
            g1n, g2n = src["data"]["group_names"]
        else:
            mean1 = st.slider("Group 1 Mean", -5.0, 5.0, 0.0, 0.1, key="tost_m1")
            mean2 = st.slider("Group 2 Mean", -5.0, 5.0, 0.5, 0.1, key="tost_m2")
            sd_t = st.slider("Common SD", 0.1, 5.0, 1.5, 0.1, key="tost_sd")
            n_t1 = st.slider("Group 1 Size", 5, 200, 30, key="tost_n1")
            n_t2 = st.slider("Group 2 Size", 5, 200, 30, key="tost_n2")
            np.random.seed(42)
            g1 = np.random.normal(mean1, sd_t, n_t1)
            g2 = np.random.normal(mean2, sd_t, n_t2)
            g1n, g2n = "Group 1", "Group 2"

        n1, n2 = len(g1), len(g2)
        m1, m2 = np.mean(g1), np.mean(g2)
        s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
        sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        se = sp * np.sqrt(1 / n1 + 1 / n2)
        diff = m1 - m2
        df_t = n1 + n2 - 2

        # TOST: two one-sided tests
        t_lower = (diff + delta_eq) / se
        t_upper = (diff - delta_eq) / se
        p_lower = t_tost.cdf(t_lower, df_t)
        p_upper = 1 - t_tost.cdf(t_upper, df_t)
        p_tost = max(p_lower, p_upper)

        # 90% CI for equivalence testing
        t_crit = t_tost.ppf(0.95, df_t)
        ci_low = diff - t_crit * se
        ci_high = diff + t_crit * se
        within_bounds = ci_low >= -delta_eq and ci_high <= delta_eq

        # Regular t-test for reference
        from scipy.stats import ttest_ind
        t_ref, p_ref = ttest_ind(g1, g2)

        # =========================
        # SUMMARY
        # =========================
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Diff", f"{diff:.3f}")
        c2.metric("SE", f"{se:.3f}")
        c3.metric("90% CI", f"[{ci_low:.3f}, {ci_high:.3f}]")
        c4.metric("Equivalence Δ", f"±{delta_eq}")

        st.latex(rf"t_{{\text{{lower}}}} = {t_lower:.3f},\; p_{{\text{{lower}}}} = {p_lower:.4f}")
        st.latex(rf"t_{{\text{{upper}}}} = {t_upper:.3f},\; p_{{\text{{upper}}}} = {p_upper:.4f}")
        st.latex(rf"\text{{TOST }} p = {p_tost:.4f}")

        if within_bounds:
            st.success(f"The 90% CI [{ci_low:.3f}, {ci_high:.3f}] falls entirely within [–{delta_eq:.1f}, {delta_eq:.1f}] → **Equivalence concluded** (p = {p_tost:.4f})")
        else:
            st.warning(f"The 90% CI [{ci_low:.3f}, {ci_high:.3f}] extends outside [–{delta_eq:.1f}, {delta_eq:.1f}] → Cannot conclude equivalence (p = {p_tost:.4f})")

        with st.expander("Comparison with standard t-test"):
            st.markdown(f"""
            - Standard t-test: p = {p_ref:.4f} ({'significant' if p_ref < 0.05 else 'not significant'})
            - TOST p = {p_tost:.4f} ({'equivalence shown' if p_tost < 0.05 else 'equivalence not shown'})

            A **significant** t-test means the means are *different*.
            A **significant** TOST means the means are *practically equivalent*.
            Both can be true (or false) depending on the data and Δ.
            """)

        fig_tost = go.Figure()
        fig_tost.add_trace(go.Scatter(x=[m1], y=[1], mode="markers", marker=dict(size=12, color="#4C78A8"),
                                       name=g1n, showlegend=True))
        fig_tost.add_trace(go.Scatter(x=[m2], y=[1], mode="markers", marker=dict(size=12, color="#F58518"),
                                       name=g2n, showlegend=True))
        fig_tost.add_hline(y=1, line_color="gray", opacity=0.3)
        fig_tost.add_vline(x=-delta_eq, line_dash="dash", line_color="green", annotation_text=f"−Δ ({-delta_eq})")
        fig_tost.add_vline(x=delta_eq, line_dash="dash", line_color="green", annotation_text=f"+Δ ({delta_eq})")
        fig_tost.add_vrect(x0=ci_low, x1=ci_high, fillcolor="rgba(0,255,0,0.1)", line_width=0,
                           annotation_text="90% CI")
        fig_tost.update_layout(template="plotly_dark", height=300,
                                xaxis_title="Mean Difference", yaxis_title="",
                                showlegend=True, margin=dict(t=10, b=10))
        st.plotly_chart(fig_tost, use_container_width=True)

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

    elif test_name == "Mood's Median Test":

        st.subheader("Interactive Mood's Median Test")

        st.info("""
        **Mood's Median Test** is a simple nonparametric alternative to **One-way ANOVA**.
        It tests whether multiple independent samples come from populations with the **same median**.
        Works by: (1) computing the grand median, (2) counting values above/below per group,
        (3) performing a Chi-square test of independence on the resulting contingency table.
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("moods_median", mode="multi_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            groups_mm = src["data"]["groups"]
            group_names_mm = src["data"]["group_names"]
        else:
            shift = st.slider("Group Separation", 0.0, 10.0, 2.0, 0.1, key="mm_shift")
            spread = st.slider("Distribution Spread", 0.1, 5.0, 1.0, 0.1, key="mm_spread")

            np.random.seed(42)
            g1_mm = np.random.exponential(spread, 60)
            g2_mm = np.random.exponential(spread, 60) + shift
            g3_mm = np.random.exponential(spread, 60) + shift * 2
            groups_mm = [g1_mm, g2_mm, g3_mm]
            group_names_mm = ["Group 1", "Group 2", "Group 3"]

        # =========================
        # STATISTICAL COMPUTATION
        # =========================

        from scipy.stats import chi2_contingency

        all_vals_mm = np.concatenate(groups_mm)
        grand_median_mm = np.median(all_vals_mm)

        above = np.array([np.sum(g > grand_median_mm) for g in groups_mm])
        below = np.array([np.sum(g <= grand_median_mm) for g in groups_mm])

        ct_mm = np.column_stack([above, below])
        chi2_mm, p_mm, dof_mm, expected_mm = chi2_contingency(ct_mm)

        # Effect size: phi coefficient
        n_total_mm = len(all_vals_mm)
        phi_mm = np.sqrt(chi2_mm / n_total_mm)

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        medians_mm = [np.median(g) for g in groups_mm]
        ns_mm = [len(g) for g in groups_mm]

        summary_data = {
            "Group": group_names_mm,
            "n": ns_mm,
            "Median": [f"{m:.3f}" for m in medians_mm],
            "Above GM": above,
            "Below/Eq GM": below,
        }
        st.table(pd.DataFrame(summary_data))

        st.markdown(f"**Grand Median:** {grand_median_mm:.3f}")

        # =========================
        # CONTINGENCY TABLE
        # =========================

        st.subheader("Contingency Table (Above/Below Grand Median)")
        ct_df = pd.DataFrame(
            ct_mm,
            index=group_names_mm,
            columns=[f"Above ({grand_median_mm:.2f})", f"Below/Equal ({grand_median_mm:.2f})"],
        )
        st.table(ct_df)

        # =========================
        # STATS
        # =========================

        st.latex(rf"\chi^2 = {chi2_mm:.3f}")
        st.latex(rf"df = {dof_mm}")
        st.latex(rf"\text{{{format_p_value(p_mm)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()
        for i, (g, name) in enumerate(zip(groups_mm, group_names_mm)):
            fig.add_trace(go.Box(y=g, name=name, boxmean="sd"))
        fig.add_hline(
            y=grand_median_mm,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Grand Median = {grand_median_mm:.2f}",
        )
        fig.update_layout(
            template="plotly_dark",
            height=450,
            title="Boxplots with Grand Median (dashed line)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED RESULTS
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        metric_list = [f"Median {name}" for name in group_names_mm] + [
            "χ²",
            "df",
            "p-value",
            "φ (effect size)",
        ]
        value_list = [f"{m:.3f}" for m in medians_mm] + [
            f"{chi2_mm:.3f}",
            f"{dof_mm}",
            format_p_value(p_mm),
            f"{phi_mm:.4f}",
        ]
        st.table(pd.DataFrame({"Metric": metric_list, "Value": value_list}))

        st.divider()
        st.subheader("Pairwise Median Tests")
        render_post_hoc(groups_mm, param_type="nonparametric", key="mm_ph")

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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("permanova", mode="multi_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            groups_pm = src["data"]["groups"]
            group_names_pm = src["data"]["group_names"]
        else:
            separation = st.slider(
                "Cluster Separation",
                0.0,
                10.0,
                2.0,
                0.1,
                key="pm_separation",
            )

            dispersion = st.slider(
                "Cluster Dispersion",
                0.1,
                5.0,
                1.0,
                0.1,
                key="pm_dispersion",
            )

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

        if src["using_uploaded"]:
            pm_groups = groups_pm
            pm_names = group_names_pm
        else:
            pm_groups = [g1, g2]
            pm_names = ["Group 1", "Group 2"]

        pm_colors = ["blue", "red", "green", "orange", "purple", "brown"]
        fig = go.Figure()
        for i, (g, name) in enumerate(zip(pm_groups, pm_names)):
            c = pm_colors[i % len(pm_colors)]
            fig.add_trace(
                go.Scatter(
                    x=g[:, 0] if g.ndim > 1 else np.arange(len(g)),
                    y=g[:, 1] if g.ndim > 1 else g,
                    mode="markers",
                    name=name,
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

        n_pm = len(pm_groups[0])
        k_pm = len(pm_groups)
        N_pm = k_pm * n_pm

        y_all_pm = np.vstack(pm_groups)
        grand_mean_pm = np.mean(y_all_pm, axis=0)

        SSt_pm = np.sum((y_all_pm - grand_mean_pm) ** 2)
        SSb_pm = sum(
            n_pm * np.sum((np.mean(g, axis=0) - grand_mean_pm) ** 2)
            for g in pm_groups
        )
        SSw_pm = SSt_pm - SSb_pm

        pseudo_F = (SSb_pm / (k_pm - 1)) / (SSw_pm / (N_pm - k_pm))
        R2_pm = SSb_pm / SSt_pm

        n_perms = 199
        pseudo_Fs = np.zeros(n_perms)
        combined = y_all_pm.copy()
        for perm in range(n_perms):
            np.random.shuffle(combined)
            perm_groups = [combined[i * n_pm:(i + 1) * n_pm] for i in range(k_pm)]
            perm_grand = np.mean(combined, axis=0)
            perm_SSb = sum(
                n_pm * np.sum((np.mean(pg, axis=0) - perm_grand) ** 2)
                for pg in perm_groups
            )
            perm_SSw = np.sum((combined - perm_grand) ** 2) - perm_SSb
            pseudo_Fs[perm] = (perm_SSb / (k_pm - 1)) / (perm_SSw / (N_pm - k_pm))

        p_pm = (np.sum(pseudo_Fs >= pseudo_F) + 1) / (n_perms + 1)

        results_data = {
            "Metric": ["Pseudo-F", "R²", "Permutations", "p-value", "Number of Groups"],
            "Value": [
                f"{pseudo_F:.3f}",
                f"{R2_pm:.4f}",
                f"{n_perms + 1}",
                f"{p_pm:.5f}",
                f"{k_pm}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        fig2 = go.Figure()
        for i, (g, name) in enumerate(zip(pm_groups, pm_names)):
            c = pm_colors[i % len(pm_colors)]
            fig2.add_trace(
                go.Scatter(
                    x=g[:, 0] if g.ndim > 1 else np.arange(len(g)),
                    y=g[:, 1] if g.ndim > 1 else g,
                    mode="markers",
                    name=name,
                    marker=dict(color=c, size=6, opacity=0.5),
                )
            )
            if g.ndim >= 2 and g.shape[1] >= 2:
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
                        line=dict(color=c, width=2, dash="dash"),
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
        if n_pm >= 6:
            ph_groups = [g[:n_pm // k_pm, 0] if g.ndim > 1 else g[:n_pm // k_pm] for g in pm_groups] if pm_groups[0].ndim > 1 else pm_groups
            render_post_hoc(ph_groups, param_type="nonparametric", key="permanova_ph")

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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("log_reg", mode="correlation")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
            from sklearn.linear_model import LogisticRegression as LR
            x_reshaped = x.reshape(-1, 1)
            log_model = LR(C=1e6, solver="lbfgs", max_iter=1000)
            log_model.fit(x_reshaped, y)
            coef_ = log_model.coef_[0][0]
            intercept_ = log_model.intercept_[0]
            y_pred_prob = log_model.predict_proba(x_reshaped)[:, 1]
            y_pred_class = log_model.predict(x_reshaped)

            from sklearn.metrics import log_loss
            ll_null = log_loss(y, np.full_like(y, y.mean()))
            ll_model = log_loss(y, y_pred_prob)
            pseudo_r2 = 1 - ll_model / ll_null
            n_log = len(y)
            st.latex(rf"\text{{Logistic Regression with uploaded data}}")
            st.latex(rf"\text{{Coefficient (β₁)}} = {coef_:.4f}")
            st.latex(rf"\text{{Intercept (β₀)}} = {intercept_:.4f}")
            st.latex(rf"\text{{Odds Ratio (OR)}} = {np.exp(coef_):.4f}")
            st.latex(rf"\text{{McFadden R²}} = {pseudo_r2:.4f}")

            x_sorted = np.linspace(x.min(), x.max(), 500)
            logit_line = intercept_ + coef_ * x_sorted
            p_line = 1 / (1 + np.exp(-logit_line))

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="markers", name="Observed",
                    marker=dict(size=6, color="rgba(100,150,255,0.6)")
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_sorted, y=p_line, mode="lines", name="Sigmoid Fit",
                    line=dict(color="red", width=3)
                )
            )
            fig.update_layout(
                height=500, template="plotly_dark",
                xaxis_title=col_names[0] if col_names else "Predictor",
                yaxis_title="Probability",
                yaxis=dict(range=[-0.05, 1.05]),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Model Summary")
            results_data = {
                "Metric": ["Sample Size", "Coefficient (β₁)", "Intercept (β₀)",
                           "Odds Ratio (OR)", "McFadden R²"],
                "Value": [f"{n_log}", f"{coef_:.4f}", f"{intercept_:.4f}",
                          f"{np.exp(coef_):.4f}", f"{pseudo_r2:.4f}"],
            }
            st.table(pd.DataFrame(results_data))

        else:
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("mlr", mode="correlation")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]

            from scipy.stats import linregress
            slr_result = linregress(x, y)
            slope, intercept, r, p, se_slope = slr_result
            r2 = r ** 2
            y_pred = intercept + slope * x
            n = len(x)

            st.latex(rf"\hat{{y}} = {intercept:.3f} + ({slope:.3f}) \cdot x")
            st.latex(rf"R^2 = {r2:.4f}")
            st.info("Workspace data provides 2-variable regression. For full multiple regression with multiple predictors, upload a dataset with additional variables.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed", marker=dict(color="rgba(100,150,255,0.6)", size=8)))
            sorted_idx = np.argsort(x)
            fig.add_trace(go.Scatter(x=x[sorted_idx], y=y_pred[sorted_idx], mode="lines", name=f"ŷ = {intercept:.2f} + {slope:.2f}x", line=dict(color="red", width=3)))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "X", yaxis_title=col_names[1] if col_names else "Y", title=f"Bivariate Regression: R² = {r2:.4f}")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Regression Coefficients")
            coeff_data = {
                "Term": ["Intercept (β₀)", "Slope (β₁)"],
                "Estimate": [f"{intercept:.4f}", f"{slope:.4f}"],
                "SE": ["(not computed)", f"{se_slope:.4f}"],
                "t": ["-", f"{slope/se_slope:.3f}" if se_slope > 0 else "-"],
                "p-value": ["-", format_p_value(p)],
            }
            st.table(pd.DataFrame(coeff_data))

        else:
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("multinomial_log", mode="correlation")

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
            st.info("""
            **Multinomial Logistic Regression with uploaded data**
            This is a simplified widget. The uploaded data is used to show the relationship
            between a predictor and a multi-category outcome. For a full analysis,
            consider using a dedicated statistical package.
            """)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(size=6, color="rgba(100,150,255,0.6)")))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "Predictor", yaxis_title="Category")
            st.plotly_chart(fig, use_container_width=True)
        else:
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("ordinal_log", mode="correlation")

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
            st.info("""
            **Ordinal Logistic Regression with uploaded data**
            This is a simplified widget. The uploaded data is used to show the relationship
            between a predictor and an ordinal outcome. For a full analysis,
            consider using a dedicated statistical package.
            """)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(size=6, color="rgba(100,150,255,0.6)")))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "Predictor", yaxis_title="Ordinal Outcome")
            st.plotly_chart(fig, use_container_width=True)
        else:
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("poisson_reg", mode="correlation")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
            from scipy.optimize import minimize as _minimize

            def _pois_nll(beta, x, y):
                lam = np.exp(beta[0] + beta[1] * x)
                return np.sum(lam - y * np.log(lam + 1e-10))

            res = _minimize(_pois_nll, [0.5, 0.1], args=(x, y), method="Nelder-Mead")
            beta0_hat, beta1_hat = res.x
            lam_fit = np.exp(beta0_hat + beta1_hat * x)
            n_pois = len(x)

            st.latex(rf"\hat{{\lambda}} = e^{{{beta0_hat:.3f} + ({beta1_hat:.3f})x}}")
            st.info(f"Fitted Poisson regression on uploaded data (n={n_pois}). y should be count data.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed", marker=dict(color="rgba(100,150,255,0.6)", size=6)))
            sorted_idx = np.argsort(x)
            fig.add_trace(go.Scatter(x=x[sorted_idx], y=lam_fit[sorted_idx], mode="lines", name="Expected Count", line=dict(color="red", width=3)))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "Predictor", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Model Summary")
            st.table(pd.DataFrame({
                "Metric": ["Sample Size", "β₀ (Intercept)", "β₁ (Slope)", "AIC"],
                "Value": [f"{n_pois}", f"{beta0_hat:.4f}", f"{beta1_hat:.4f}", f"{2 * res.fun + 4:.2f}"],
            }))
        else:
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

    elif test_name == "Negative Binomial Regression":

        st.subheader("Interactive Negative Binomial Regression")

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("neg_binom_reg", mode="correlation")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            x = src["data"]["x"]
            y = src["data"]["y"]
            col_names = src["data"]["col_names"]
            from scipy.optimize import minimize as _nb_min

            def _nb_nll(params, x, y):
                beta0, beta1, alpha = params
                mu = np.exp(beta0 + beta1 * x)
                theta = 1 / alpha
                nll = -np.sum(
                    np.random.gamma(theta, 1, len(y)) * 0
                    + y * np.log(mu + 1e-10)
                    - (y + theta) * np.log(mu + theta + 1e-10)
                    + np.random.gamma(1, 1, 1) * 0
                )
                from scipy.special import gammaln
                nll = -np.sum(
                    gammaln(y + theta)
                    - gammaln(theta)
                    - gammaln(y + 1)
                    + theta * np.log(theta + 1e-10)
                    + y * np.log(mu + 1e-10)
                    - (y + theta) * np.log(mu + theta + 1e-10)
                )
                return nll

            res_nb = _nb_min(_nb_nll, [0.5, 0.1, 0.5], args=(x, y), method="Nelder-Mead")
            beta0_nb, beta1_nb, alpha_nb = res_nb.x
            n_nb = len(x)

            st.latex(rf"\hat{{\mu}} = e^{{{beta0_nb:.3f} + ({beta1_nb:.3f})x}}")
            st.latex(rf"\hat{{\alpha}} = {alpha_nb:.4f} \quad (\text{{dispersion parameter}})")
            st.info(f"Fitted Negative Binomial regression on uploaded data (n={n_nb}). Overdispersion parameter α = {alpha_nb:.4f} (α=0 would be Poisson).")

            mu_fit = np.exp(beta0_nb + beta1_nb * x)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed", marker=dict(color="rgba(100,150,255,0.6)", size=6)))
            sorted_idx = np.argsort(x)
            fig.add_trace(go.Scatter(x=x[sorted_idx], y=mu_fit[sorted_idx], mode="lines", name="Expected Count", line=dict(color="red", width=3)))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "Predictor", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Model Summary")
            st.table(pd.DataFrame({
                "Metric": ["Sample Size", "β₀ (Intercept)", "β₁ (Slope)", "α (Dispersion)", "AIC"],
                "Value": [f"{n_nb}", f"{beta0_nb:.4f}", f"{beta1_nb:.4f}", f"{alpha_nb:.4f}", f"{2 * res_nb.fun + 6:.2f}"],
            }))
        else:
            beta0 = st.slider("β₀", -3.0, 3.0, 0.5, 0.1, key="negbinom_beta0")
            beta1 = st.slider("β₁", -1.0, 1.0, 0.2, 0.05, key="negbinom_beta1")
            alpha_disp = st.slider("Dispersion (α)", 0.05, 5.0, 1.0, 0.05, key="negbinom_alpha")

            x = np.linspace(0, 20, 500)
            mu = np.exp(beta0 + beta1 * x)

            st.latex(rf"\mu = e^{{{beta0:.2f} + ({beta1:.2f})x}}")
            st.latex(rf"\text{{Var}}(Y) = \mu + \alpha\mu^2")
            st.info("Negative Binomial extends Poisson by adding a dispersion parameter α. When α → 0, it reduces to Poisson.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=mu, mode="lines", name="Expected Count"))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title="Predictor", yaxis_title="Expected Count (μ)")
            st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Cox Proportional Hazards Regression":

        st.subheader("Interactive Cox Regression")

        # =========================
        # DATA SOURCE TOGGLE (survival special)
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            with st.expander("📁 Optional: Use Your Own Data", expanded=False):
                st.markdown("Upload a CSV with time, event, and predictor columns.")
                source = st.radio(
                    "Data Source",
                    ["Simulated (sliders, for learning)", "Upload CSV/Excel (your data)"],
                    key="cox_datasource",
                    index=0,
                    label_visibility="collapsed",
                )
            if "Simulated" in source:
                src = {"using_uploaded": False, "data": None}
            else:
                cox_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="cox_file")
                if cox_file is not None:
                    cox_df = pd.read_csv(cox_file) if cox_file.name.endswith(".csv") else pd.read_excel(cox_file)
                    st.success(f"Loaded {len(cox_df)} rows")
                    num_cols = list(cox_df.select_dtypes(include=["int64", "float64"]).columns)
                    if len(num_cols) >= 2:
                        time_col = st.selectbox("Time column", num_cols, key="cox_time")
                        event_col = st.selectbox("Event column (0/1)", num_cols, key="cox_event", index=min(1, len(num_cols)-1))
                        pred_col = st.selectbox("Predictor column", [c for c in num_cols if c not in [time_col, event_col]] or num_cols, key="cox_pred")
                        src = {"using_uploaded": True, "data": {"time": cox_df[time_col].dropna().values, "event": cox_df[event_col].dropna().values, "predictor": cox_df[pred_col].dropna().values, "time_col": time_col, "event_col": event_col, "pred_col": pred_col}}
                    else:
                        st.error("Need at least 2 numeric columns")
                        src = {"using_uploaded": False, "data": None}
                else:
                    src = {"using_uploaded": False, "data": None}

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            cox_times = src["data"]["time"]
            cox_event = src["data"]["event"]
            cox_pred = src["data"]["predictor"]

            from scipy.stats import chi2 as _cox_chi2

            def _cox_partial_likelihood(beta, t, e, x):
                order = np.argsort(t)
                t, e, x = t[order], e[order], x[order]
                risk = np.exp(beta * x)
                total_risk = np.cumsum(risk[::-1])[::-1]
                log_pl = np.sum(e * (beta * x - np.log(total_risk + 1e-10)))
                return -log_pl

            from scipy.optimize import minimize_scalar
            cox_res = minimize_scalar(lambda b: _cox_partial_likelihood(b, cox_times, cox_event, cox_pred))
            cox_beta = cox_res.x
            cox_hr = np.exp(cox_beta)
            cox_n = len(cox_times)
            cox_events = int(cox_event.sum())

            null_ll = _cox_partial_likelihood(0, cox_times, cox_event, cox_pred)
            alt_ll = cox_res.fun
            cox_lr = 2 * (null_ll - alt_ll)
            cox_p = 1 - _cox_chi2.cdf(cox_lr, 1)

            st.latex(rf"\hat{{\beta}} = {cox_beta:.4f}")
            st.latex(rf"\text{{Hazard Ratio}} = e^{{\hat{{\beta}}}} = {cox_hr:.4f}")
            st.latex(rf"\text{{Likelihood Ratio }} \chi^2 = {cox_lr:.3f}, \; p = {format_p_value(cox_p)}")

            fig = go.Figure()
            t_grid = np.linspace(0, np.percentile(cox_times[cox_event == 1], 95) * 1.2, 200)
            low_pred = np.percentile(cox_pred, 25)
            high_pred = np.percentile(cox_pred, 75)
            base_haz = cox_events / np.sum(cox_times)
            for val, lbl, clr in [(low_pred, "Low Predictor", "blue"), (high_pred, "High Predictor", "red")]:
                surv = np.exp(-base_haz * np.exp(cox_beta * val) * t_grid)
                fig.add_trace(go.Scatter(x=t_grid, y=surv, mode="lines", name=lbl, line=dict(color=clr)))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Survival Probability")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Model Summary")
            st.table(pd.DataFrame({
                "Metric": ["N", "Events", "β", "HR", "LR χ²", "p-value"],
                "Value": [f"{cox_n}", f"{cox_events}", f"{cox_beta:.4f}", f"{cox_hr:.4f}", f"{cox_lr:.3f}", format_p_value(cox_p)],
            }))
        else:
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
        # DATA SOURCE TOGGLE (survival special)
        # =========================

        if external_data and external_data.get("using_uploaded") and external_data.get("_format") == "raw":
            src = external_data
        else:
            with st.expander("📁 Optional: Use Your Own Data", expanded=False):
                st.markdown("Upload a CSV with time, event, and group columns.")
                source = st.radio(
                    "Data Source",
                    ["Simulated (sliders, for learning)", "Upload CSV/Excel (your data)"],
                    key="km_datasource",
                    index=0,
                    label_visibility="collapsed",
                )
            if "Simulated" in source:
                src = {"using_uploaded": False, "data": None}
            else:
                km_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="km_file")
                if km_file is not None:
                    km_df = pd.read_csv(km_file) if km_file.name.endswith(".csv") else pd.read_excel(km_file)
                    st.success(f"Loaded {len(km_df)} rows")
                    num_cols = list(km_df.select_dtypes(include=["int64", "float64"]).columns)
                    if len(num_cols) >= 2:
                        time_col = st.selectbox("Time column", num_cols, key="km_time")
                        event_col = st.selectbox("Event column (0/1)", num_cols, key="km_event", index=min(1, len(num_cols)-1))
                        group_options = list(km_df.select_dtypes(include=["object", "category", "int64"]).columns) + num_cols
                        group_col = st.selectbox("Group column", group_options, key="km_group")
                        km_data_df = km_df[[time_col, event_col, group_col]].dropna()
                        src = {"using_uploaded": True, "data": {"df": km_data_df, "time_col": time_col, "event_col": event_col, "group_col": group_col}}
                    else:
                        st.error("Need at least 2 numeric columns")
                        src = {"using_uploaded": False, "data": None}
                else:
                    src = {"using_uploaded": False, "data": None}

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            km_data = src["data"]
            km_df_in = km_data["df"]
            km_times = km_df_in[km_data["time_col"]].values
            km_event = km_df_in[km_data["event_col"]].values.astype(int)
            km_group_raw = km_df_in[km_data["group_col"]].values
            km_group_names = sorted(set(str(v) for v in km_group_raw))
            if len(km_group_names) > 2:
                km_group_names = km_group_names[:2]
            km_group = np.array([0 if str(v) == km_group_names[0] else 1 for v in km_group_raw])
            show_ci = st.toggle("Show 95% CI", True, key="km_ci_toggle_uploaded")
        else:
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
            km_times = obs
            km_event = event
            km_group = groups
            km_group_names = ["Control", "Treatment"]

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

        t1, s1 = _km_est(km_times[km_group == 0], km_event[km_group == 0])
        t2, s2 = _km_est(km_times[km_group == 1], km_event[km_group == 1])
        chi2, p_val = _logrank(km_times, km_event, km_group)

        below1 = np.where(s1 <= 0.5)[0]
        below2 = np.where(s2 <= 0.5)[0]
        med1 = t1[below1[0]] if len(below1) > 0 else None
        med2 = t2[below2[0]] if len(below2) > 0 else None

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()
        gname0 = km_group_names[0] if len(km_group_names) > 0 else "Group 0"
        gname1 = km_group_names[1] if len(km_group_names) > 1 else "Group 1"
        colors = {gname0: "#4C78A8", gname1: "#E45756"}
        fill_colors = {
            gname0: "rgba(76,120,168,0.15)",
            gname1: "rgba(228,87,86,0.15)",
        }

        for grp, name in [(0, gname0), (1, gname1)]:
            t_, s_, lo_, hi_ = _km_ci(km_times[km_group == grp], km_event[km_group == grp])
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
            mask = km_group == grp
            cens_t = km_times[mask & (km_event == 0)]
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

        events_pct = int(km_event.mean() * 100)
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
            f"{gname0} Median Survival",
            f"{med1:.1f}" if med1 is not None else "Not reached",
        )
        m1[1].metric(
            f"{gname1} Median Survival",
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
            - **Events observed**: {int(km_event.sum())} / {len(km_event)} ({events_pct}% of patients experienced the event)
            - **Censored**: {(~km_event.astype(bool)).sum()} patients ({100 - events_pct}%) had their event time censored
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
        # DATA SOURCE TOGGLE (survival special)
        # =========================

        if external_data and external_data.get("using_uploaded") and external_data.get("_format") == "raw":
            src = external_data
        else:
            with st.expander("📁 Optional: Use Your Own Data", expanded=False):
                st.markdown("Upload a CSV with time, event, and group columns.")
                source = st.radio(
                    "Data Source",
                    ["Simulated (sliders, for learning)", "Upload CSV/Excel (your data)"],
                    key="lr_datasource",
                    index=0,
                    label_visibility="collapsed",
                )
            if "Simulated" in source:
                src = {"using_uploaded": False, "data": None}
            else:
                lr_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="lr_file")
                if lr_file is not None:
                    lr_df = pd.read_csv(lr_file) if lr_file.name.endswith(".csv") else pd.read_excel(lr_file)
                    st.success(f"Loaded {len(lr_df)} rows")
                    num_cols = list(lr_df.select_dtypes(include=["int64", "float64"]).columns)
                    if len(num_cols) >= 2:
                        time_col = st.selectbox("Time column", num_cols, key="lr_time")
                        event_col = st.selectbox("Event column (0/1)", num_cols, key="lr_event", index=min(1, len(num_cols)-1))
                        group_options = list(lr_df.select_dtypes(include=["object", "category", "int64"]).columns) + num_cols
                        group_col = st.selectbox("Group column", group_options, key="lr_group")
                        src = {"using_uploaded": True, "data": {"df": lr_df[[time_col, event_col, group_col]].dropna(), "time_col": time_col, "event_col": event_col, "group_col": group_col}}
                    else:
                        st.error("Need at least 2 numeric columns")
                        src = {"using_uploaded": False, "data": None}
                else:
                    src = {"using_uploaded": False, "data": None}

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            lr_data = src["data"]
            lr_df_in = lr_data["df"]
            lr_times = lr_df_in[lr_data["time_col"]].values
            lr_event = lr_df_in[lr_data["event_col"]].values.astype(int)
            lr_group_raw = lr_df_in[lr_data["group_col"]].values
            lr_group_names = sorted(set(str(v) for v in lr_group_raw))
            if len(lr_group_names) > 2:
                lr_group_names = lr_group_names[:2]
            lr_group = np.array([0 if str(v) == lr_group_names[0] else 1 for v in lr_group_raw])

            from scipy.stats import chi2 as _lr_chi2
            def _logrank_lr(tt, ee, gg):
                sort_idx = np.argsort(tt)
                tt_s, ee_s, gg_s = tt[sort_idx], ee[sort_idx], gg[sort_idx]
                ut = sorted(set(tt_s))
                o1 = e1 = v1 = 0.0
                for ti in ut:
                    mask_t = tt_s == ti
                    ne = ee_s[mask_t].sum()
                    if ne == 0: continue
                    at_risk = tt_s >= ti
                    n1, n2 = (gg_s[at_risk] == 0).sum(), (gg_s[at_risk] == 1).sum()
                    nr = n1 + n2
                    if nr < 2: continue
                    d1 = (gg_s[mask_t & (ee_s == 1)] == 0).sum()
                    o1 += d1; e1 += ne * n1 / nr
                    v1 += ne * n1 * n2 * (nr - ne) / (nr * nr * (nr - 1))
                if v1 <= 0: return 0.0, 1.0
                chi2 = (o1 - e1) ** 2 / v1
                return chi2, 1 - _lr_chi2.cdf(chi2, 1)

            lr_chi2, lr_p = _logrank_lr(lr_times, lr_event, lr_group)
            st.latex(rf"\text{{Log-Rank }} \chi^2 = {lr_chi2:.3f}")
            st.latex(rf"\text{{{format_p_value(lr_p)}}}")

            fig = go.Figure()
            t_grid = np.linspace(0, np.percentile(lr_times, 95) * 1.2, 200)
            for grp, name in [(0, lr_group_names[0]), (1, lr_group_names[1])]:
                mask_g = lr_group == grp
                tt_grp = lr_times[mask_g]
                ee_grp = lr_event[mask_g]
                rate = 1 / np.mean(tt_grp[ee_grp == 1]) if ee_grp.sum() > 0 else 0.1
                surv = np.exp(-rate * t_grid)
                fig.add_trace(go.Scatter(x=t_grid, y=surv, mode="lines", name=name))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Survival Probability")
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Log-Rank Test Results")
            st.table(pd.DataFrame({
                "Metric": ["N", "Events", "χ²", "p-value"],
                "Value": [f"{len(lr_times)}", f"{int(lr_event.sum())}", f"{lr_chi2:.3f}", format_p_value(lr_p)],
            }))
        else:
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("sens_spec", mode="categorical_two")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            ct = src["data"]["contingency_table"]
            if ct.shape != (2, 2):
                st.error("Sensitivity & Specificity requires a 2×2 contingency table. "
                         "Please select two binary categorical variables.")
                return
            tp = ct.iloc[0, 0]
            fn = ct.iloc[1, 0]
            fp = ct.iloc[0, 1]
            tn = ct.iloc[1, 1]
        else:
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("roc_curve", mode="categorical_two")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            ct = src["data"]["contingency_table"]
            if ct.shape != (2, 2):
                st.error("ROC Curve Analysis requires a 2×2 contingency table. "
                         "Please select two binary categorical variables.")
                return
            tp = ct.iloc[0, 0]
            fn = ct.iloc[1, 0]
            fp = ct.iloc[0, 1]
            tn = ct.iloc[1, 1]
            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr = np.array([0.0, fpr_val, 1.0])
            tpr = np.array([0.0, tpr_val, 1.0])
            roc_auc = np.trapz(tpr, fpr)
            st.info(f"ROC point from uploaded data: (1−Specificity={fpr_val:.3f}, Sensitivity={tpr_val:.3f})")
        else:
            separation = st.slider("Diagnostic Power (Group Separation)", 0.0, 5.0, 1.5, 0.1)
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("likelihood_ratio", mode="categorical_two")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            ct = src["data"]["contingency_table"]
            if ct.shape != (2, 2):
                st.error("Likelihood Ratio Analysis requires a 2×2 contingency table. "
                         "Please select two binary categorical variables.")
                return
            tp = ct.iloc[0, 0]
            fn = ct.iloc[1, 0]
            fp = ct.iloc[0, 1]
            tn = ct.iloc[1, 1]
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
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
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("cohen_kappa", mode="categorical_two")

        # =========================
        # DATA
        # =========================

        if src["using_uploaded"]:
            ct = src["data"]["contingency_table"]
            if ct.shape != (2, 2):
                st.error("Cohen's Kappa requires a 2×2 contingency table. "
                         "Please select two binary categorical variables.")
                return
            yy = ct.iloc[0, 0]
            yn = ct.iloc[0, 1]
            ny = ct.iloc[1, 0]
            nn = ct.iloc[1, 1]
        else:
            st.write("Enter agreement counts between two raters:")
            c1, c2 = st.columns(2)
            with c1:
                yy = st.number_input("Both say YES", min_value=0, value=40, key="kappa_yy")
                yn = st.number_input(
                    "Rater 1 says YES, Rater 2 says NO", min_value=0, value=10, key="kappa_yn"
                )
            with c2:
                ny = st.number_input(
                    "Rater 1 says NO, Rater 2 says YES", min_value=0, value=5, key="kappa_ny"
                )
                nn = st.number_input("Both say NO", min_value=0, value=45, key="kappa_nn")

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

    elif test_name == "Fleiss' Kappa":

        st.subheader("Interactive Fleiss' Kappa — Multi-Rater Agreement")
        st.info("""
        **Fleiss' Kappa** measures agreement among **3+ raters** assigning nominal ratings
        to the same subjects. Unlike Cohen's Kappa (which handles 2 raters), Fleiss' Kappa
        generalizes to any number of raters.
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("fleiss_kappa", mode="categorical_two")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            ct = src["data"]["contingency_table"]
            st.warning("Fleiss' Kappa requires per-subject rater-by-category data "
                       "(subjects × raters, not a contingency table). Using contingency table as "
                       "the agreement matrix for demonstration.")
            table_fk = ct.values.astype(float)
            n_subjects_fk, n_cat_fk = table_fk.shape
            n_raters_fk = st.slider("Number of Raters", 3, 10, 3, key="fk_raters",
                                    help="Assuming each subject is rated by this many raters")
        else:
            n_subjects_fk = st.slider("Number of Subjects", 10, 200, 30, key="fk_n")
            n_raters_fk = st.slider("Number of Raters", 3, 10, 3, key="fk_raters_sim")
            n_cat_fk = st.slider("Number of Categories", 2, 5, 3, key="fk_cat")
            agreement_p = st.slider("Agreement Probability", 0.0, 1.0, 0.7, 0.01, key="fk_agree",
                                    help="How often raters agree on the same category")
            np.random.seed(42)
            true_cat = np.random.randint(0, n_cat_fk, n_subjects_fk)
            table_fk = np.zeros((n_subjects_fk, n_cat_fk), dtype=float)
            for i in range(n_subjects_fk):
                ratings = np.random.choice(n_cat_fk, n_raters_fk, p=[
                    agreement_p + (1 - agreement_p) / n_cat_fk if j == true_cat[i]
                    else (1 - agreement_p) / n_cat_fk for j in range(n_cat_fk)
                ])
                for r in ratings:
                    table_fk[i, r] += 1

        # Fleiss' Kappa computation
        n = table_fk.shape[0]
        k = table_fk.shape[1]
        N = n_raters_fk * n

        p_j = table_fk.sum(axis=0) / N
        P_i = (table_fk ** 2).sum(axis=1) - n_raters_fk
        P_i = P_i / (n_raters_fk * (n_raters_fk - 1))
        P_bar = P_i.mean()
        P_e = (p_j ** 2).sum()
        fleiss_kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0.0

        # Variance approximation
        var_kappa = 2 / (n * n_raters_fk * (n_raters_fk - 1)) * (
            (P_bar - (2 * n_raters_fk - 1) * P_e / (n_raters_fk - 1)) /
            (1 - P_e) ** 2
        ) if P_e < 1 and (1 - P_e) > 0 else 0
        se_kappa = np.sqrt(max(var_kappa, 0))
        z_fk = fleiss_kappa / se_kappa if se_kappa > 0 else 0
        from scipy.stats import norm as norm_fk
        p_fk = 2 * (1 - norm_fk.cdf(abs(z_fk)))

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Fleiss' κ", f"{fleiss_kappa:.4f}")
        c2.metric("SE", f"{se_kappa:.4f}")
        c3.metric("p-value", format_p_value(p_fk))

        if fleiss_kappa > 0.8:
            fk_interp = "Almost Perfect Agreement"
        elif fleiss_kappa > 0.6:
            fk_interp = "Substantial Agreement"
        elif fleiss_kappa > 0.4:
            fk_interp = "Moderate Agreement"
        elif fleiss_kappa > 0.2:
            fk_interp = "Fair Agreement"
        else:
            fk_interp = "Slight/Poor Agreement"
        st.success(f"**Interpretation:** {fk_interp}")

        with st.expander("Category Statistics"):
            cat_df = pd.DataFrame({
                "Category": [f"Cat {j+1}" for j in range(k)],
                "p_j (marginal)": [f"{pj:.4f}" for pj in p_j],
            })
            st.dataframe(cat_df, use_container_width=True)

        st.caption(f"Based on {n} subjects × {int(n_raters_fk)} raters, {k} categories")

    elif test_name == "Weighted Kappa":

        from scipy.stats import norm as norm_wk

        st.subheader("Interactive Weighted Kappa — Ordinal Agreement")
        st.info("""
        **Weighted Kappa** extends Cohen's Kappa to **ordinal** categories by penalizing
        disagreements proportionally to their severity (e.g., a 2-category disagreement
        is worse than a 1-category disagreement). Uses quadratic weights.
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("weighted_kappa", mode="categorical_two")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            ct = src["data"]["contingency_table"]
            if ct.shape[0] != ct.shape[1]:
                st.error("Weighted Kappa requires a square contingency table "
                         "(same categories for both raters).")
                return
            table_wk = ct.values.astype(float)
            n_cat_wk = table_wk.shape[0]
            cat_labels_wk = list(ct.index) if hasattr(ct, 'index') else [f"Cat {i+1}" for i in range(n_cat_wk)]
        else:
            n_cat_wk = st.slider("Number of Ordinal Categories", 2, 7, 4, key="wk_ncat")
            table_wk = np.zeros((n_cat_wk, n_cat_wk), dtype=float)
            for r in range(n_cat_wk):
                for c in range(n_cat_wk):
                    default_val = max(5 - abs(r - c) * 3, 1)
                    table_wk[r, c] = st.number_input(
                        f"Rater1={r+1}, Rater2={c+1}", min_value=0, value=int(default_val),
                        key=f"wk_cell_{r}_{c}"
                    )
            cat_labels_wk = [f"Cat {i+1}" for i in range(n_cat_wk)]

        # Quadratic weights
        n_lev = n_cat_wk
        weights = np.zeros((n_lev, n_lev))
        for i in range(n_lev):
            for j in range(n_lev):
                weights[i, j] = 1 - (i - j) ** 2 / (n_lev - 1) ** 2

        total_wk = table_wk.sum()
        prop_wk = table_wk / total_wk if total_wk > 0 else table_wk

        # Observed weighted agreement
        po_wk = (weights * prop_wk).sum()

        # Expected agreement (under independence)
        row_marg = prop_wk.sum(axis=1)
        col_marg = prop_wk.sum(axis=0)
        pe_wk = (weights * np.outer(row_marg, col_marg)).sum()

        wk = (po_wk - pe_wk) / (1 - pe_wk) if pe_wk < 1 else 0.0

        # Variance (Fleiss, Cohen & Everitt 1969 approximation)
        var_wk = 0
        if pe_wk < 1 and total_wk > 0:
            var_wk = (po_wk * (1 - po_wk)) / (total_wk * (1 - pe_wk) ** 2)
        se_wk = np.sqrt(max(var_wk, 0))
        z_wk = wk / se_wk if se_wk > 0 else 0
        p_wk = 2 * (1 - norm_wk.cdf(abs(z_wk)))

        # Unweighted kappa for comparison
        po_unw = np.trace(prop_wk)
        pe_unw = (row_marg * col_marg).sum()
        unw_kappa = (po_unw - pe_unw) / (1 - pe_unw) if pe_unw < 1 else 0.0

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Weighted κ (quadratic)", f"{wk:.4f}")
        c2.metric("Unweighted κ (comparison)", f"{unw_kappa:.4f}")
        c3.metric("p-value", format_p_value(p_wk))

        if wk > 0.8:
            wk_interp = "Almost Perfect Agreement"
        elif wk > 0.6:
            wk_interp = "Substantial Agreement"
        elif wk > 0.4:
            wk_interp = "Moderate Agreement"
        elif wk > 0.2:
            wk_interp = "Fair Agreement"
        else:
            wk_interp = "Slight/Poor Agreement"
        st.success(f"**Interpretation:** {wk_interp}")

        with st.expander("Weight Matrix (quadratic)"):
            wt_df = pd.DataFrame(weights, index=cat_labels_wk, columns=cat_labels_wk)
            st.dataframe(wt_df.style.format("{:.3f}"), use_container_width=True)

        st.caption(f"Total observations: {int(total_wk)}")

    elif test_name == "Bland-Altman Analysis":

        st.subheader("Interactive Bland-Altman Analysis")

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("bland_altman", mode="paired")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            method1 = np.array(src["data"]["values1"])
            method2 = np.array(src["data"]["values2"])
        else:
            bias = st.slider("Bias (Mean Difference)", -5.0, 5.0, 0.2, 0.1)
            agreement_sd = st.slider("SD of Differences", 0.1, 5.0, 1.0, 0.1)
            n = st.slider("Sample Size", 10, 200, 50, key="bland_altman_analysis_sample_size")

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

    elif test_name == "Runs Test for Randomness":

        st.subheader("Interactive Runs Test for Randomness")

        st.info("""
        **Runs Test for Randomness** tests whether a sequence of values is **random**.
        A "run" is a consecutive sequence of values above (or below) the median.
        - **Too few runs** → clustered pattern (positive autocorrelation)
        - **Too many runs** → alternating pattern (negative autocorrelation)
        - Used in time-series analysis, quality control, and residual diagnostics.
        """)

        # =========================
        # DATA SOURCE TOGGLE
        # =========================

        if external_data and external_data.get("using_uploaded"):
            src = external_data
        else:
            src = data_source_toggle("runs_test", mode="one_sample")

        # =========================
        # CONTROLS / DATA
        # =========================

        if src["using_uploaded"]:
            values_rt = src["data"]["values"]
        else:
            pattern = st.selectbox("Sequence Pattern", ["Random", "Trend", "Cyclical", "Clustered"], key="rt_pattern")
            n_rt = st.slider("Sample Size", 10, 300, 100, key="rt_n")

            np.random.seed(42)
            if pattern == "Random":
                values_rt = np.random.normal(0, 1, n_rt)
            elif pattern == "Trend":
                values_rt = np.linspace(-2, 2, n_rt) + np.random.normal(0, 0.3, n_rt)
            elif pattern == "Cyclical":
                x = np.linspace(0, 4 * np.pi, n_rt)
                values_rt = np.sin(x) + np.random.normal(0, 0.2, n_rt)
            else:  # Clustered
                block1 = np.random.normal(-1, 0.3, n_rt // 3)
                block2 = np.random.normal(1, 0.3, n_rt // 3)
                block3 = np.random.normal(-1, 0.3, n_rt - 2 * (n_rt // 3))
                values_rt = np.concatenate([block1, block2, block3])

        # =========================
        # STATISTICAL COMPUTATION
        # =========================

        from scipy.stats import norm as norm_rt

        median_rt = np.median(values_rt)
        binary_rt = (values_rt > median_rt).astype(int)

        n1_rt = np.sum(binary_rt == 1)
        n2_rt = np.sum(binary_rt == 0)

        runs_rt = 1 + np.sum(binary_rt[1:] != binary_rt[:-1])

        n_runs = n1_rt + n2_rt
        expected_runs = 1 + (2 * n1_rt * n2_rt) / n_runs
        var_runs = (2 * n1_rt * n2_rt * (2 * n1_rt * n2_rt - n_runs)) / (n_runs ** 2 * (n_runs - 1))
        z_rt = (runs_rt - expected_runs) / np.sqrt(var_runs)
        p_rt = 2 * (1 - norm_rt.cdf(abs(z_rt)))

        # =========================
        # DATA SUMMARY
        # =========================

        st.divider()
        st.subheader("Data Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("N", f"{n_runs}")
        col2.metric("Runs (observed)", f"{runs_rt}")
        col3.metric("Runs (expected)", f"{expected_runs:.2f}")
        col4.metric("Median", f"{median_rt:.3f}")

        # =========================
        # STATS
        # =========================

        st.latex(rf"z = {z_rt:.3f}")
        st.latex(rf"\text{{{format_p_value(p_rt)}}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        colors = ["#1f77b4" if b == 1 else "#ff7f0e" for b in binary_rt]
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(values_rt)),
                y=values_rt,
                mode="lines+markers",
                marker=dict(color=colors, size=6),
                line=dict(color="gray", width=1),
                name="Sequence",
            )
        )
        fig.add_hline(
            y=median_rt,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median = {median_rt:.2f}",
        )

        # Highlight runs
        run_starts = [0] + list(np.where(binary_rt[1:] != binary_rt[:-1])[0] + 1)
        for rs in run_starts:
            fig.add_vline(x=rs - 0.5, line_dash="dot", line_color="green", line_width=1, opacity=0.5)

        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Position in Sequence",
            yaxis_title="Value",
            title=f"Runs Test: {runs_rt} runs observed vs {expected_runs:.1f} expected",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")
        results_data = {
            "Metric": ["Median", "Runs (observed)", "Runs (expected)", "z", "p-value", "n₁ (above)", "n₂ (below/equal)"],
            "Value": [
                f"{median_rt:.3f}",
                f"{runs_rt}",
                f"{expected_runs:.2f}",
                f"{z_rt:.3f}",
                format_p_value(p_rt),
                f"{n1_rt}",
                f"{n2_rt}",
            ],
        }
        st.table(pd.DataFrame(results_data))

    else:
        st.info("Interactive widget coming soon for this test.")
