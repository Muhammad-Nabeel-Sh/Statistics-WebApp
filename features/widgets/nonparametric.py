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

@register_test("Brunner-Munzel Test")
def render_brunner_munzel_test(external_data=None):

    from scipy.stats import mannwhitneyu, rankdata

    st.subheader("Interactive Brunner-Munzel Test")

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        from core.utils import data_source_toggle
        src = data_source_toggle("brunner_munzel", mode="two_sample")

    if src["using_uploaded"]:
        g1 = src["data"]["group1"]
        g2 = src["data"]["group2"]
    else:
        np.random.seed(42)
        mu1 = st.slider("Group 1 Location", -5.0, 5.0, 0.0, 0.1, key="bm_m1")
        mu2 = st.slider("Group 2 Location", -5.0, 5.0, 1.0, 0.1, key="bm_m2")
        sd1 = st.slider("Group 1 Spread", 0.1, 5.0, 1.0, 0.1, key="bm_sd1")
        sd2 = st.slider("Group 2 Spread", 0.1, 5.0, 2.0, 0.1, key="bm_sd2")
        n = st.slider("Sample Size per Group", 10, 200, 50, key="bm_n")
        dist = st.selectbox("Distribution", ["Normal", "Exponential", "Uniform"],
                            key="bm_dist")
        if dist == "Normal":
            g1 = np.random.normal(mu1, sd1, n)
            g2 = np.random.normal(mu2, sd2, n)
        elif dist == "Exponential":
            g1 = np.random.exponential(sd1, n) + mu1
            g2 = np.random.exponential(sd2, n) + mu2
        else:
            g1 = np.random.uniform(mu1 - sd1, mu1 + sd1, n)
            g2 = np.random.uniform(mu2 - sd2, mu2 + sd2, n)

    def _brunner_munzel(x, y):
        n1, n2 = len(x), len(y)
        combined = np.concatenate([x, y])
        R = rankdata(combined)
        R1 = R[:n1]
        R2 = R[n1:]
        R1_bar = np.mean(R1)
        R2_bar = np.mean(R2)

        p_hat = (R2_bar - (n2 + 1) / 2) / n1

        S_sq = np.var(R, ddof=1)
        S1_sq = np.var(R1, ddof=1)
        S2_sq = np.var(R2, ddof=1)

        sigma_sq = (n1 * n2) * (S_sq / (n1 + n2) -
                                 (S1_sq / n1 * (n1 - 1) + S2_sq / n2 * (n2 - 1)) /
                                 ((n1 + n2) * (n1 + n2 - 1)))

        sigma_sq = max(sigma_sq, 1e-10)
        W = (p_hat - 0.5) / np.sqrt(sigma_sq) * np.sqrt(n1 * n2)

        num = (S1_sq / n1 + S2_sq / n2) ** 2
        den = (S1_sq / n1) ** 2 / (n1 - 1) + (S2_sq / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else 1

        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(W), df))
        return W, p, df, p_hat

    W_bm, p_bm, df_bm, p_hat_bm = _brunner_munzel(
        np.asarray(g1), np.asarray(g2)
    )

    mw_stat, mw_p = mannwhitneyu(g1, g2, alternative="two-sided")

    st.divider()
    st.subheader("Results")

    col1, col2 = st.columns(2)
    col1.metric("Brunner-Munzel W", f"{W_bm:.3f}")
    col2.metric("Relative Effect p̂", f"{p_hat_bm:.3f}")

    st.latex(rf"W = {W_bm:.3f}")
    st.latex(rf"df = {df_bm:.1f}")
    st.latex(rf"\text{{{format_p_value(p_bm)}}}")
    st.latex(rf"\hat{{p}} = {p_hat_bm:.3f} \quad "
             rf"(\text{{interpretation: }} P(\text{{Group 2 > Group 1}}) = {p_hat_bm:.3f})")

    st.info(
        f"**Comparison with Mann-Whitney U:** U = {mw_stat:.1f}, "
        f"p = {format_p_value(mw_p)}."
    )

    fig_bm = go.Figure()
    fig_bm.add_trace(go.Violin(y=g1, name="Group 1", box_visible=True,
                                meanline_visible=True, fillcolor="#4C78A8",
                                opacity=0.6, line_color="#4C78A8"))
    fig_bm.add_trace(go.Violin(y=g2, name="Group 2", box_visible=True,
                                meanline_visible=True, fillcolor="#F58518",
                                opacity=0.6, line_color="#F58518"))
    fig_bm.update_layout(template="plotly_dark", height=400,
                          yaxis_title="Value",
                          title="Group Distributions (Brunner-Munzel Test)")
    st.plotly_chart(fig_bm, use_container_width=True)

    render_post_hoc(
        [np.asarray(g1), np.asarray(g2)],
        param_type="nonparametric",
        key="brunner_munzel_ph",
    )




@register_test("Two-Sample Kolmogorov-Smirnov Test")
def render_two_sample_kolmogorov_smirnov_test(external_data=None):

    from scipy.stats import ks_2samp

    st.subheader("Interactive Two-Sample Kolmogorov-Smirnov Test")

    st.info("""
    **Two-Sample Kolmogorov-Smirnov Test** compares the empirical distributions of two samples.
    It is sensitive to differences in location, shape, and spread.
    H₀: Both samples come from the same continuous distribution.
    """)

    np.random.seed(42)

    n_ks = st.slider("Sample Size", 10, 200, 50, key="ks2_n")
    dist1 = st.selectbox("Distribution 1", ["Normal", "Uniform", "Exponential", "t (df=3)"], key="ks2_d1")
    dist2 = st.selectbox("Distribution 2", ["Normal", "Uniform", "Exponential", "t (df=3)"], key="ks2_d2")
    shift = st.slider("Shift (distribution 2)", -2.0, 2.0, 0.5, 0.1, key="ks2_shift")
    scale_diff = st.slider("Scale Difference", 0.5, 2.0, 1.0, 0.1, key="ks2_scale")

    def _sample(dist, n, loc=0, scale=1):
        if dist == "Normal":
            return np.random.normal(loc, scale, n)
        elif dist == "Uniform":
            return np.random.uniform(loc - scale, loc + scale, n)
        elif dist == "Exponential":
            return np.random.exponential(scale, n) + loc
        else:
            from scipy.stats import t as t_dist
            return t_dist.rvs(df=3, loc=loc, scale=scale, size=n)

    s1 = _sample(dist1, n_ks, loc=0, scale=1)
    s2 = _sample(dist2, n_ks, loc=shift, scale=scale_diff)

    D_ks, p_ks = ks_2samp(s1, s2)

    col1, col2 = st.columns(2)
    col1.metric("D Statistic", f"{D_ks:.4f}")
    col2.metric("p-value", f"{p_ks:.4f}")

    st.subheader("Distribution Comparison")

    bins = np.histogram_bin_edges(np.concatenate([s1, s2]), bins="auto")
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=s1, name=dist1, opacity=0.6, nbinsx=len(bins),
                                marker_color="#4C78A8", histnorm="probability density"))
    fig.add_trace(go.Histogram(x=s2, name=dist2, opacity=0.6, nbinsx=len(bins),
                                marker_color="#F58518", histnorm="probability density"))

    from scipy.stats import norm as norm_ks

    x_grid = np.linspace(min(s1.min(), s2.min()), max(s1.max(), s2.max()), 200)
    ecdf1 = np.array([np.mean(s1 <= x) for x in x_grid])
    ecdf2 = np.array([np.mean(s2 <= x) for x in x_grid])

    fig.add_trace(go.Scatter(x=x_grid, y=ecdf1, mode="lines",
                              name=f"{dist1} ECDF", line=dict(color="#4C78A8", dash="dot")))
    fig.add_trace(go.Scatter(x=x_grid, y=ecdf2, mode="lines",
                              name=f"{dist2} ECDF", line=dict(color="#F58518", dash="dot")))

    fig.update_layout(template="plotly_dark", height=500,
                      xaxis_title="Value", yaxis_title="Density / ECDF",
                      title="Histograms with Empirical CDF Overlay")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")
    st.table(pd.DataFrame({
        "Metric": ["D-statistic", "p-value", "Sample Size Dist 1", "Sample Size Dist 2"],
        "Value": [f"{D_ks:.4f}", format_p_value(p_ks), f"{n_ks}", f"{n_ks}"],
    }))


@register_test("Jonckheere-Terpstra Test")
def render_jonckheere_terpstra_test(external_data=None):

    from scipy.stats import mannwhitneyu
    from scipy.stats import norm as norm_jt

    st.subheader("Interactive Jonckheere-Terpstra Test")

    st.info("""
    **Jonckheere-Terpstra Test** tests for an ordered trend across k independent groups.
    H₀: median₁ = median₂ = … = medianₖ
    H₁: median₁ ≤ median₂ ≤ … ≤ medianₖ (at least one strict inequality)
    It is more powerful than Kruskal-Wallis when a monotonic trend is expected.
    """)

    np.random.seed(42)

    group_sizes = st.slider("Per-Group Sample Size", 10, 50, 20, key="jt_n")
    n_groups = st.slider("Number of Groups", 3, 5, 3, key="jt_k")
    trend = st.slider("Trend Strength", 0.0, 3.0, 0.5, 0.1, key="jt_trend")

    groups_jt = []
    means = [i * trend for i in range(n_groups)]
    for m in means:
        groups_jt.append(np.random.exponential(1, group_sizes) + m)

    J = 0
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            u_stat, _ = mannwhitneyu(groups_jt[i], groups_jt[j], alternative="less")
            J += u_stat

    n_total_jt = n_groups * group_sizes
    n_per_jt = group_sizes
    E_J = (n_total_jt ** 2 - sum(n_per_jt ** 2 for _ in range(n_groups))) / 4
    Var_J = (n_total_jt ** 2 * (2 * n_total_jt + 3) - sum(
        n_per_jt ** 2 * (2 * n_per_jt + 3) for _ in range(n_groups))) / 72
    z_jt = (J - E_J) / np.sqrt(Var_J) if Var_J > 0 else 0
    p_jt = 2 * (1 - norm_jt.cdf(abs(z_jt)))

    col1, col2, col3 = st.columns(3)
    col1.metric("J Statistic", f"{J:.1f}")
    col2.metric("z-score", f"{z_jt:.3f}")
    col3.metric("p-value", f"{p_jt:.4f}")

    fig = go.Figure()
    for i, g in enumerate(groups_jt):
        fig.add_trace(go.Box(y=g, name=f"Group {i + 1} (μ≈{means[i]:.1f})",
                              marker_color=f"rgba({50 + i * 40}, {100 + i * 30}, 200, 0.7)"))
    fig.update_layout(template="plotly_dark", height=450,
                      title="Box Plots by Group (ordered by increasing mean)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")
    st.table(pd.DataFrame({
        "Metric": ["J Statistic", "E[J]", "Var[J]", "z-score", "p-value", "Groups", "Per-group n"],
        "Value": [f"{J:.1f}", f"{E_J:.1f}", f"{Var_J:.1f}", f"{z_jt:.3f}",
                  format_p_value(p_jt), f"{n_groups}", f"{group_sizes}"],
    }))


@register_test("Page Test")
def render_page_test(external_data=None):

    from scipy.stats import page_trend_test

    st.subheader("Interactive Page Test")

    st.info("""
    **Page's L Test** is a nonparametric test for monotonic trend across **repeated measures** conditions.
    H₀: All conditions have the same median
    H₁: median₁ < median₂ < … < medianₖ (ordered alternative)
    """)

    np.random.seed(42)

    n_subj = st.slider("Number of Subjects", 10, 50, 20, key="page_n")
    n_cond = st.slider("Number of Conditions", 3, 6, 4, key="page_k")
    trend_st = st.slider("Trend Strength", 0.0, 3.0, 0.5, 0.1, key="page_trend")

    data_page = np.zeros((n_subj, n_cond))
    for i in range(n_subj):
        base = np.random.exponential(1)
        for j in range(n_cond):
            data_page[i, j] = base + j * trend_st + np.random.normal(0, 0.5)

    res = page_trend_test(data_page)
    L_stat = res.statistic
    p_page = res.pvalue
    # Standardized Z for Page's L
    n_p, k_p = data_page.shape
    z_page = (12 * L_stat - 3 * n_p * k_p * (k_p + 1) ** 2) / np.sqrt(n_p * k_p * (k_p + 1) * (k_p ** 2 - 1))

    col1, col2, col3 = st.columns(3)
    col1.metric("L Statistic", f"{L_stat:.1f}")
    col2.metric("Z Statistic", f"{z_page:.3f}")
    col3.metric("p-value", f"{p_page:.4f}")

    fig = go.Figure()
    cond_labels = [f"C{j + 1}" for j in range(n_cond)]

    for i in range(n_subj):
        fig.add_trace(go.Scatter(x=cond_labels, y=data_page[i], mode="lines+markers",
                                  showlegend=False,
                                  line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                                  marker=dict(size=3)))

    means_page = data_page.mean(axis=0)
    fig.add_trace(go.Scatter(x=cond_labels, y=means_page, mode="lines+markers",
                              name="Mean Trend", line=dict(color="red", width=3),
                              marker=dict(color="red", size=10)))

    fig.update_layout(template="plotly_dark", height=500,
                      xaxis_title="Condition", yaxis_title="Value",
                      title=f"Subject Trajectories (n={n_subj}) with Mean Trend")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")
    st.table(pd.DataFrame({
        "Metric": ["L Statistic", "Z Statistic", "p-value", "Subjects", "Conditions"],
        "Value": [f"{L_stat:.1f}", f"{z_page:.3f}", format_p_value(p_page),
                  f"{n_subj}", f"{n_cond}"],
    }))


@register_test("Mann-Kendall Trend Test")
def render_mann_kendall_trend_test(external_data=None):

    from scipy.stats import norm as norm_mk
    from scipy.special import gammaln

    st.subheader("Interactive Mann-Kendall Trend Test")

    st.info("""
    **Mann-Kendall Trend Test** detects monotonic trends in time series data.
    H₀: No monotonic trend (independent and identically distributed)
    H₁: Presence of a monotonic upward or downward trend
    Sen's slope estimates the magnitude of the trend.
    """)

    np.random.seed(42)

    n_years = st.slider("Number of Time Points", 10, 100, 30, key="mk_n")
    slope_val = st.slider("Annual Change (Slope)", -1.0, 1.0, 0.05, 0.01, key="mk_slope")
    noise = st.slider("Noise Level", 0.1, 2.0, 0.5, 0.1, key="mk_noise")
    seasonality = st.selectbox("Seasonality", ["None", "Monthly", "Quarterly"], key="mk_season")

    t = np.arange(n_years)
    seasonal_effect = np.zeros(n_years)
    if seasonality == "Monthly":
        seasonal_effect = 0.5 * np.sin(2 * np.pi * t / 12)
    elif seasonality == "Quarterly":
        seasonal_effect = 0.3 * np.sin(2 * np.pi * t / 4)

    values_mk = slope_val * t + seasonal_effect + np.random.normal(0, noise, n_years)

    S = 0
    n_mk = len(values_mk)
    for i in range(n_mk):
        for j in range(i + 1, n_mk):
            S += np.sign(values_mk[j] - values_mk[i])

    Var_S = n_mk * (n_mk - 1) * (2 * n_mk + 5) / 18
    z_mk = (S - np.sign(S)) / np.sqrt(Var_S) if Var_S > 0 else 0
    p_mk = 2 * (1 - norm_mk.cdf(abs(z_mk)))

    slopes_arr = []
    for i in range(n_mk):
        for j in range(i + 1, n_mk):
            if t[j] != t[i]:
                slopes_arr.append((values_mk[j] - values_mk[i]) / (t[j] - t[i]))
    sens_slope = np.median(slopes_arr) if slopes_arr else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("S Statistic", f"{S:.0f}")
    col2.metric("Z", f"{z_mk:.3f}")
    col3.metric("p-value", f"{p_mk:.4f}")
    col4.metric("Sen's Slope", f"{sens_slope:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=values_mk, mode="markers+lines",
                              name="Observed", marker=dict(color="#4C78A8", size=5),
                              line=dict(color="#4C78A8", width=1)))
    trend_line = sens_slope * t + (np.median(values_mk) - sens_slope * np.median(t))
    fig.add_trace(go.Scatter(x=t, y=trend_line, mode="lines",
                              name=f"Sen's Slope = {sens_slope:.3f}",
                              line=dict(color="red", width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", height=450,
                      xaxis_title="Time", yaxis_title="Value",
                      title="Time Series with Sen's Slope Trend Line")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")
    st.table(pd.DataFrame({
        "Metric": ["S Statistic", "Var(S)", "z-score", "p-value",
                   "Sen's Slope", "Number of Time Points"],
        "Value": [f"{S:.0f}", f"{Var_S:.1f}", f"{z_mk:.3f}",
                  format_p_value(p_mk), f"{sens_slope:.4f}", f"{n_mk}"],
    }))


# Correlation and Association Tests

