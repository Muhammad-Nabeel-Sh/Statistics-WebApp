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


@register_test("One-sample t-test")
def render_one_sample_t_test(external_data=None):

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

    population_mean = st.number_input(
        "Reference Mean (H₀: μ = μ₀)",
        value=0.0,
        step=0.1,
        format="%.2f",
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



@register_test("One-sample z-test")
def render_one_sample_z_test(external_data=None):

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

    population_mean = st.number_input(
        "Population Mean (μ₀)",
        value=0.0,
        step=0.1,
        format="%.2f",
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



@register_test("Student's t-test (Independent)")
def render_student_s_t_test_independent(external_data=None):

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



@register_test("Welch's t-test (Independent, Unequal Variances)")
def render_welch_s_t_test_independent_unequal_variances(external_data=None):

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



@register_test("Paired t-test")
def render_paired_t_test(external_data=None):

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



@register_test("One-way ANOVA")
def render_one_way_anova(external_data=None):

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



@register_test("One-way Welch ANOVA")
def render_one_way_welch_anova(external_data=None):

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



@register_test("Two-way ANOVA")
def render_two_way_anova(external_data=None):

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



@register_test("ANCOVA")
def render_ancova(external_data=None):

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



@register_test("Repeated Measures ANOVA")
def render_repeated_measures_anova(external_data=None):

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



@register_test("MANOVA")
def render_manova(external_data=None):

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



@register_test("F-Test for Two Variances")
def render_f_test_for_two_variances(external_data=None):

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



@register_test("Equivalence Test (TOST) - Two Independent Samples")
def render_equivalence_test_tost_two_independent_samples(external_data=None):

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

@register_test("Yuen's Trimmed t-test")
def render_yuen_trimmed_t_test(external_data=None):

    from scipy import stats as scipy_stats

    st.subheader("Interactive Yuen's Trimmed t-test")

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        from core.utils import data_source_toggle
        src = data_source_toggle("yuen_ttest", mode="two_sample")

    trim_prop = st.slider(
        "Trim Proportion (per tail)",
        min_value=0.0,
        max_value=0.45,
        value=0.2,
        step=0.05,
        key="yuen_trim",
    )

    if src["using_uploaded"]:
        g1 = src["data"]["group1"]
        g2 = src["data"]["group2"]
    else:
        np.random.seed(42)
        mu1 = st.slider("Group 1 Mean", -5.0, 5.0, 0.0, 0.1, key="yuen_m1")
        mu2 = st.slider("Group 2 Mean", -5.0, 5.0, 1.0, 0.1, key="yuen_m2")
        sd1 = st.slider("Group 1 SD", 0.1, 5.0, 1.0, 0.1, key="yuen_sd1")
        sd2 = st.slider("Group 2 SD", 0.1, 5.0, 1.0, 0.1, key="yuen_sd2")
        n = st.slider("Sample Size per Group", 10, 200, 50, key="yuen_n")

        g1 = np.random.normal(mu1, sd1, n)
        g2 = np.random.normal(mu2, sd2, n)
        outlier_frac = st.slider("Outlier Fraction", 0.0, 0.2, 0.05, 0.01, key="yuen_outliers")
        n_out1 = int(len(g1) * outlier_frac)
        n_out2 = int(len(g2) * outlier_frac)
        if n_out1 > 0:
            g1[:n_out1] = np.random.normal(mu1 + 5 * sd1, sd1 * 0.5, n_out1)
        if n_out2 > 0:
            g2[:n_out2] = np.random.normal(mu2 + 5 * sd2, sd2 * 0.5, n_out2)

    def _yuen_ttest(x, y, tr):
        n1, n2 = len(x), len(y)
        n1t = int(np.floor(n1 * tr))
        n2t = int(np.floor(n2 * tr))
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        if n1t > 0:
            x_trimmed = x_sorted[n1t:-n1t]
        else:
            x_trimmed = x_sorted
        if n2t > 0:
            y_trimmed = y_sorted[n2t:-n2t]
        else:
            y_trimmed = y_sorted
        h1 = len(x_trimmed)
        h2 = len(y_trimmed)
        tx = np.mean(x_trimmed)
        ty = np.mean(y_trimmed)

        x_winsor = x_sorted.copy()
        if n1t > 0:
            x_winsor[:n1t] = x_sorted[n1t]
            x_winsor[-n1t:] = x_sorted[-n1t - 1]
        y_winsor = y_sorted.copy()
        if n2t > 0:
            y_winsor[:n2t] = y_sorted[n2t]
            y_winsor[-n2t:] = y_sorted[-n2t - 1]

        sw1_sq = np.var(x_winsor, ddof=1)
        sw2_sq = np.var(y_winsor, ddof=1)

        d1 = (n1 - 1) * sw1_sq / (h1 * (h1 - 1))
        d2 = (n2 - 1) * sw2_sq / (h2 * (h2 - 1))
        se = np.sqrt(d1 + d2)
        tyuen = (tx - ty) / se if se > 0 else 0

        num = (d1 + d2) ** 2
        den = d1 ** 2 / (h1 - 1) + d2 ** 2 / (h2 - 1)
        df = num / den if den > 0 else 1
        p = 2 * (1 - scipy_stats.t.cdf(abs(tyuen), df)) if df > 0 else 1.0
        return tyuen, p, df, tx, ty, h1, h2

    t_y, p_y, df_y, tx, ty, h1, h2 = _yuen_ttest(
        np.asarray(g1), np.asarray(g2), trim_prop
    )

    st.divider()
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Group 1 Trimmed Mean", f"{tx:.3f}", delta=None)
    col2.metric("Group 2 Trimmed Mean", f"{ty:.3f}", delta=None)
    col3.metric("Difference", f"{tx - ty:.3f}")

    st.latex(rf"t_{{\text{{Yuen}}}} = {t_y:.3f}")
    st.latex(rf"df = {df_y:.1f}")
    st.latex(rf"\text{{{format_p_value(p_y)}}}")

    st.info(
        f"Trimmed sample sizes: n₁ = {h1}, n₂ = {h2} "
        f"(trimmed {trim_prop:.0%} from each tail)"
    )

    fig_yuen = go.Figure()
    fig_yuen.add_trace(go.Box(y=g1, name="Group 1", boxmean=True,
                               marker_color="#4C78A8"))
    fig_yuen.add_trace(go.Box(y=g2, name="Group 2", boxmean=True,
                               marker_color="#F58518"))
    fig_yuen.add_hline(y=tx, line_dash="dash", line_color="#4C78A8",
                        annotation_text=f"Trimmed mean 1 = {tx:.2f}")
    fig_yuen.add_hline(y=ty, line_dash="dash", line_color="#F58518",
                        annotation_text=f"Trimmed mean 2 = {ty:.2f}")
    fig_yuen.update_layout(template="plotly_dark", height=400,
                            yaxis_title="Value",
                            title="Group Distributions with Trimmed Means")
    st.plotly_chart(fig_yuen, use_container_width=True)

    render_post_hoc(
        [np.asarray(g1), np.asarray(g2)],
        param_type="parametric",
        key="yuen_ph",
    )



@register_test("Hotelling's T-Squared")
def render_hotelling_t_squared(external_data=None):
    from scipy.stats import f

    st.subheader("Interactive Hotelling's T-Squared")
    st.markdown("Tests whether the mean vectors of two groups differ across multiple dependent variables.")

    n = st.slider("Sample Size per Group", 10, 100, 30, key="hot_n", label_visibility="collapsed")
    p = st.selectbox("Number of Dependent Variables", [2, 3, 4, 5], index=0, key="hot_p")
    effect = st.slider("Effect Size (Mahalanobis D)", 0.0, 3.0, 0.5, 0.1, key="hot_d")

    np.random.seed(42)
    mean_diff = np.array([effect] + [0] * (p - 1))
    cov = np.eye(p)
    group1 = np.random.multivariate_normal(np.zeros(p), cov, n)
    group2 = np.random.multivariate_normal(mean_diff, cov, n)

    mean1 = group1.mean(axis=0)
    mean2 = group2.mean(axis=0)
    diff = mean1 - mean2
    S1 = np.cov(group1, rowvar=False)
    S2 = np.cov(group2, rowvar=False)
    Sp = ((n - 1) * S1 + (n - 1) * S2) / (2 * n - 2)

    try:
        Sp_inv = np.linalg.inv(Sp)
        T2 = (n * n / (2 * n)) * diff @ Sp_inv @ diff
        F_stat = (2 * n - p - 1) / (p * (2 * n - 2)) * T2
        p_val = 1 - f.cdf(F_stat, p, 2 * n - p - 1)
    except np.linalg.LinAlgError:
        st.error("Covariance matrix is singular. Try a larger sample size.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("T-Squared", f"{T2:.3f}")
    c2.metric("F(df1={}, df2={})".format(p, 2 * n - p - 1), f"{F_stat:.3f}")
    c3.metric("p-value", f"{p_val:.4f}")

    st.markdown("**Group Means:**")
    comp = pd.DataFrame({"Variable": [f"V{i+1}" for i in range(p)], "Group 1": mean1, "Group 2": mean2})
    st.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns if c != "Variable"}), use_container_width=True)

    fig = go.Figure()
    for g, name, color in [(group1, "Group 1", "#4C78A8"), (group2, "Group 2", "#E45756")]:
        if p >= 2:
            fig.add_trace(go.Scatter(x=g[:, 0], y=g[:, 1], mode="markers", name=name,
                                     marker=dict(color=color, size=6, opacity=0.7),
                                     text=[f"V1={v1:.1f}, V2={v2:.1f}" for v1, v2 in g[:, :2]]))
            fig.add_trace(go.Scatter(x=[g[:, 0].mean()], y=[g[:, 1].mean()], mode="markers",
                                     marker=dict(color=color, size=14, symbol="x"), showlegend=False))
    fig.update_layout(template="plotly_dark", height=400,
                      xaxis_title="Variable 1", yaxis_title="Variable 2",
                      title="First Two Variables with Group Means")
    st.plotly_chart(fig, use_container_width=True)

# Non-parametric Multiple Group Tests

