import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import scipy.optimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .shared import _rng, _gen_corr, _gen_reg

def bland_altman_widget():
    st.markdown("## Bland-Altman Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="ba_n")
        bias = st.slider(
            "Systematic Bias (Mean Diff)",
            -2.0,
            2.0,
            0.2,
            0.1,
            key="ba_bias",
            help="Average difference between methods",
        )
        proportional_bias = st.slider(
            "Proportional Bias",
            -1.0,
            1.0,
            0.0,
            0.05,
            key="ba_prop",
            help="Bias that changes with measurement magnitude",
        )
        limits_factor = st.slider(
            "Limits of Agreement Multiplier",
            1.0,
            3.0,
            1.96,
            0.01,
            key="ba_loa",
            help="1.96 approx 95% limits",
        )
    np.random.seed(42)
    true_val = np.random.uniform(0, 20, n)
    m1 = true_val + np.random.normal(0, 0.5, n)
    m2 = (
        true_val
        + bias
        + proportional_bias * (true_val - 10)
        + np.random.normal(0, 0.5, n)
    )
    mean = (m1 + m2) / 2
    diff = m1 - m2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    upper_loa = mean_diff + limits_factor * std_diff
    lower_loa = mean_diff - limits_factor * std_diff
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mean,
            y=diff,
            mode="markers",
            name="Differences",
            marker=dict(color="#4C78A8", size=7, opacity=0.7),
            hovertemplate="Mean = %{x:.2f}<br>Difference = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=mean_diff,
        line=dict(color="#E45756", width=2),
        annotation_text=f"Mean Diff = {mean_diff:.3f}",
    )
    fig.add_hline(
        y=upper_loa,
        line=dict(color="rgba(200,200,200,0.7)", dash="dash"),
        annotation_text=f"+{limits_factor}SD = {upper_loa:.3f}",
    )
    fig.add_hline(
        y=lower_loa,
        line=dict(color="rgba(200,200,200,0.7)", dash="dash"),
        annotation_text=f"-{limits_factor}SD = {lower_loa:.3f}",
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="Mean of Two Measurements",
        yaxis_title="Difference (Method 1 - Method 2)",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- y = 0 line = perfect agreement\n"
                "- Mean diff = systematic bias\n"
                "- Dashed lines = limits of agreement\n"
                "- Fan-shape = proportional bias\n"
                "- 95% of points should be within limits"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare measurement methods\n"
                "- Assess test-retest reliability\n"
                "- Medical device validation\n"
                "- Laboratory method comparison"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Paired t-test (fixed bias)\n"
                "- Intraclass Correlation (ICC)\n"
                "- Deming regression\n"
                "- Passing-Bablok regression"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Bland-Altman assesses agreement, not correlation. "
                "High r does NOT mean good agreement - two methods "
                "can be perfectly correlated but have a large bias. "
                "Always use Bland-Altman for method comparison."
            )



def kappa_widget():
    st.markdown("## Cohen's Kappa Agreement Matrix")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="kap_n")
        n_cats = st.selectbox(
            "Number of Categories", [2, 3, 4], index=0, key="kap_cats"
        )
        agreement = st.slider(
            "Agreement Rate",
            0.0,
            1.0,
            0.7,
            0.05,
            key="kap_agree",
            help="Proportion of perfect agreement between raters",
        )
        marginal_bias = st.slider(
            "Marginal Bias",
            0.0,
            0.5,
            0.1,
            0.01,
            key="kap_bias",
            help="How much raters prefer different categories",
        )
    k = int(n_cats)
    np.random.seed(42)
    true_labels = np.random.choice(k, n)
    rater1 = true_labels.copy()
    rater2 = true_labels.copy()
    disagree_mask = np.random.rand(n) > agreement
    for i in np.where(disagree_mask)[0]:
        other = [c for c in range(k) if c != true_labels[i]]
        rater1[i] = np.random.choice(other)
        rater2[i] = np.random.choice(other)
    if marginal_bias > 0:
        shift = int(marginal_bias * n)
        if shift > 0:
            rater2[:shift] = (rater2[:shift] + 1) % k
    cm = np.zeros((k, k), dtype=int)
    for i in range(n):
        cm[rater1[i], rater2[i]] += 1
    n_total = cm.sum()
    p_o = np.trace(cm) / n_total
    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    p_e = np.sum(row_marg * col_marg) / n_total**2
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    labels = [f"Cat {i + 1}" for i in range(k)]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=14),
            colorscale="Blues",
            showscale=False,
            hovertemplate="Rater1: %{y}<br>Rater2: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Rater 2",
        yaxis_title="Rater 1",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Cohen's kappa", f"{kappa:.3f}")
    m2.metric("Observed Agreement", f"{p_o:.1%}")
    m3.metric("Chance Agreement", f"{p_e:.1%}")
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Diagonal = perfect agreement\n"
                "- Off-diagonal = disagreement\n"
                "- kappa < 0 = worse than chance\n"
                "- kappa 0.4-0.6 = moderate\n"
                "- kappa > 0.8 = near perfect"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Inter-rater reliability\n"
                "- Diagnostic agreement studies\n"
                "- Psychiatric diagnostic assessment\n"
                "- Image/scan rating agreement"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Weighted Kappa (ordinal)\n"
                "- Fleiss' Kappa (3+ raters)\n"
                "- McNemar's test (2x2 bias)\n"
                "- ICC (continuous)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Kappa is prevalence-dependent - rare categories "
                "produce low kappa even with high agreement. "
                "Also, kappa penalizes raters differently: two "
                "raters can have 80% agreement but kappa = 0.5. "
                "Always report agreement rate alongside kappa."
            )



def icc_widget():
    st.markdown("## ICC Visualization")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Subjects", 5, 50, 20, key="icc_n")
        n_raters = st.selectbox(
            "Number of Raters/Occasions", [2, 3, 4], index=0, key="icc_r"
        )
        icc_true = st.slider(
            "True ICC",
            0.0,
            1.0,
            0.6,
            0.05,
            key="icc_true",
            help="Proportion of variance due to between-subject differences",
        )
        show_subject = st.toggle("Connect Subject Lines", True, key="icc_lines")
    np.random.seed(42)
    n_s, n_r = int(n), int(n_raters)
    subject_effects = np.random.normal(0, np.sqrt(icc_true), n_s)
    error_sd = np.sqrt(1 - icc_true)
    data = np.zeros((n_s, n_r))
    for j in range(n_r):
        data[:, j] = subject_effects + np.random.normal(0, error_sd, n_s)
    grand_mean = np.mean(data)
    ss_between = np.sum(n_r * (np.mean(data, axis=1) - grand_mean) ** 2)
    ss_within = np.sum((data - np.mean(data, axis=1, keepdims=True)) ** 2)
    ms_between = ss_between / (n_s - 1)
    ms_within = ss_within / (n_s * (n_r - 1))
    icc_est = (
        (ms_between - ms_within) / (ms_between + (n_r - 1) * ms_within)
        if (ms_between + (n_r - 1) * ms_within) > 0
        else 0
    )
    fig = go.Figure()
    for i in range(n_s):
        ys = data[i]
        xs = [f"Rater {j + 1}" for j in range(n_r)]
        if show_subject:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=dict(color="rgba(200,200,200,0.3)", width=1),
                    marker=dict(size=5, color=px.colors.qualitative.Plotly[i % 10]),
                    showlegend=False,
                    hovertemplate=f"Subject {i+1}<br>Rater=%{{x}}<br>Value=%{{y:.2f}}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=7, color=px.colors.qualitative.Plotly[i % 10]),
                    name=f"S{i + 1}",
                    hovertemplate=f"Subject {i+1}<br>Rater=%{{x}}<br>Value=%{{y:.2f}}<extra></extra>",
                )
            )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="Rater",
        yaxis_title="Measurement",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    col_est, _ = st.columns(2)
    col_est.metric("Estimated ICC", f"{icc_est:.3f}", delta=f"Target = {icc_true}")
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each color = one subject\n"
                "- Lines horizontal = high ICC\n"
                "- Lines crossing = low ICC\n"
                "- Tight clustering within subject\n"
                "- Wide spread within subject = error"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Test-retest reliability\n"
                "- Inter-rater reliability (continuous)\n"
                "- Intra-rater reliability\n"
                "- Longitudinal measurement stability"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- ICC(1,1) - single rater absolute\n"
                "- ICC(2,1) - consistency\n"
                "- ICC(3,k) - average random raters\n"
                "- Bland-Altman (agreement)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "High ICC can mask systematic bias (raters "
                "consistently disagreeing). ICC measures "
                "relative consistency, not absolute agreement. "
                "Combine with Bland-Altman for full assessment."
            )


# --- MULTIVARIATE PLOTS ---




GRAPHS = {
    "Bland-Altman Plot": bland_altman_widget,
    "Cohen's Kappa Agreement Matrix": kappa_widget,
    "ICC Visualization": icc_widget
}