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


@register_test("One-sample Wilcoxon Signed-Rank Test")
def render_one_sample_wilcoxon_signed_rank_test(external_data=None):

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

    hypothesized_median = st.number_input(
        "Hypothesized Median (H₀)",
        value=0.0,
        step=0.5,
        format="%.2f",
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



@register_test("Sign Test (One-sample)")
def render_sign_test_one_sample(external_data=None):

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

    hypothesized_median = st.number_input("Hypothesized Median (H₀)", value=0.0, step=0.5, format="%.2f")

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



@register_test("Wilcoxon Signed-Rank Test")
def render_wilcoxon_signed_rank_test(external_data=None):

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



@register_test("Sign Test (Paired)")
def render_sign_test_paired(external_data=None):

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



@register_test("Mann-Whitney U Test")
def render_mann_whitney_u_test(external_data=None):

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



@register_test("Kruskal-Wallis Test")
def render_kruskal_wallis_test(external_data=None):

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



@register_test("Mood's Median Test")
def render_mood_s_median_test(external_data=None):

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



@register_test("Friedman Test")
def render_friedman_test(external_data=None):

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



@register_test("Permutation MANOVA or Non-Parametric MANOVA")
def render_permutation_manova_or_non_parametric_manova(external_data=None):

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

