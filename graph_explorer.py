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

# =========================
# DATA GENERATION HELPERS
# =========================

_rng = np.random.default_rng(42)


def _gen_corr(n, r, noise=0.2, heteroscedastic=False, outlier=False):
    x = np.random.normal(0, 1, n)
    y = r * x + np.sqrt(1 - r**2) * np.random.normal(0, 1, n)
    if heteroscedastic:
        y = y * (1 + 0.5 * np.abs(x))
    if outlier:
        x[-1] = 4
        y[-1] = -4 if r > 0 else 4
    return x, y


def _gen_reg(n, beta=1.0, noise=1.0):
    x = np.random.uniform(0, 10, n)
    y = beta * x + np.random.normal(0, noise, n)
    return x, y


# =========================
# WIDGET FUNCTIONS
# =========================

# --- DISTRIBUTION PLOTS ---


def histogram_widget():
    st.markdown("## Histogram")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 1000, 200, key="hist_n")
        skew = st.slider(
            "Skewness",
            -3.0,
            3.0,
            0.0,
            0.1,
            key="hist_skew",
            help="Negative = left skew, Positive = right skew, 0 = symmetric",
        )
        bins = st.slider("Bin Count", 5, 100, 30, key="hist_bins")
        show_kde = st.toggle("Show KDE Overlay", True, key="hist_kde")
        dist_type = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Uniform"], key="hist_dist"
        )
    np.random.seed(42 + int(skew * 10))
    if dist_type == "Normal":
        data = np.random.normal(skew * 0.5, 1, n)
    elif dist_type == "Skewed":
        data = np.random.gamma(2 if skew >= 0 else 3, 1, n) - (2 if skew >= 0 else 3)
        if skew < 0:
            data = -data
    else:
        data = np.random.uniform(-1.5, 1.5, n)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data, nbinsx=bins, name="Frequency", marker_color="#4C78A8", opacity=0.75
        )
    )
    if show_kde:
        kde_x = np.linspace(min(data), max(data), 300)
        kde_y = stats.gaussian_kde(data)(kde_x)
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y * n * (max(data) - min(data)) / bins,
                mode="lines",
                name="KDE",
                line=dict(color="#E45756", width=2),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Bell shape = normal distribution\n"
                "- Tail on right = positive skew\n"
                "- Tail on left = negative skew\n"
                "- Gaps = potential outliers/modes\n"
                "- KDE = smooth density estimate"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Explore data distribution shape\n"
                "- Check for normality or skewness\n"
                "- Identify outliers and gaps\n"
                "- Compare to theoretical distributions"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Shapiro-Wilk (normality)\n"
                "- Kolmogorov-Smirnov\n"
                "- Anderson-Darling\n"
                "- t-test, ANOVA (assumptions)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Bin width dramatically changes appearance. "
                "Too few bins hide detail, too many create noise. "
                "Always try multiple bin widths before interpreting shape."
            )


def kde_widget():
    st.markdown("## Density / KDE Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 1000, 300, key="kde_n")
        bw = st.select_slider(
            "Bandwidth",
            [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            value=0.5,
            key="kde_bw",
            help="Smaller = more detail (risk overfitting), Larger = smoother",
        )
        multimodal = st.toggle("Show Multimodal", False, key="kde_multi")
        dist_type = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Bimodal"], key="kde_dist"
        )
    np.random.seed(42)
    if dist_type == "Normal":
        data = np.random.normal(0, 1, n)
    elif dist_type == "Skewed":
        data = np.random.gamma(2, 1, n)
    else:
        data = np.concatenate(
            [np.random.normal(-2, 0.7, n // 2), np.random.normal(2, 0.7, n // 2)]
        )
    if multimodal and dist_type != "Bimodal":
        data = np.concatenate([data, np.random.normal(3, 0.5, n // 2)])
    kde_x = np.linspace(min(data) - 1, max(data) + 1, 500)
    try:
        kde = stats.gaussian_kde(data, bw_method=bw)
        kde_y = kde(kde_x)
    except Exception:
        kde_y = np.zeros_like(kde_x)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde_y,
            fill="tozeroy",
            mode="lines",
            name="Density",
            line=dict(color="#4C78A8", width=3),
            hovertemplate="x = %{x:.2f}<br>density = %{y:.4f}<extra></extra>",
        )
    )
    rug_x = data
    rug_y = np.full_like(rug_x, -0.02)
    fig.add_trace(
        go.Scatter(
            x=rug_x,
            y=rug_y,
            mode="markers",
            name="Data",
            marker=dict(color="#E45756", size=3, symbol="line-ns-open"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title="Density",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Peak = most common value region\n"
                "- Spread = variance of data\n"
                "- Multiple peaks = multimodal\n"
                "- Bandwidth controls smoothness"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare distributions smoothly\n"
                "- Identify modes and shape\n"
                "- Assess normality visually\n"
                "- Overlay multiple groups"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- t-test / ANOVA\n"
                "- Kruskal-Wallis\n"
                "- Kolmogorov-Smirnov\n"
                "- Permutation tests"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "KDE can oversmooth bimodal data with large bandwidth, "
                "hiding real structure. Conversely, small bandwidth "
                "can create spurious peaks from noise."
            )


def boxplot_widget():
    st.markdown("## Boxplot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 10, 200, 50, key="box_n")
        n_groups = st.selectbox("Number of Groups", [1, 2, 3, 4], index=2, key="box_ng")
        show_outliers = st.toggle("Show Outliers", True, key="box_out")
        spread = st.slider(
            "Variance Spread",
            0.2,
            3.0,
            1.0,
            0.1,
            key="box_spread",
            help="Higher = more variability between groups",
        )
    np.random.seed(42)
    groups = []
    names = []
    for i in range(int(n_groups)):
        d = np.random.normal(i * spread, 0.5 + i * 0.15, n)
        if not show_outliers:
            d = np.clip(d, -3 + i * spread, 3 + i * spread)
        groups.append(d)
        names.append(f"Group {chr(65 + i)}")
    fig = go.Figure()
    for i, (d, name) in enumerate(zip(groups, names)):
        fig.add_trace(
            go.Box(
                y=d,
                name=name,
                marker_color=px.colors.qualitative.Plotly[i],
                boxpoints="outliers" if show_outliers else False,
                hovertemplate=f"{name}<br>y = %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title="Value",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Center line = median\n"
                "- Box = IQR (middle 50%)\n"
                "- Whiskers = 1.5xIQR range\n"
                "- Dots = potential outliers\n"
                "- Non-overlapping = group diff"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare groups at a glance\n"
                "- Identify outliers visually\n"
                "- Assess symmetry & spread\n"
                "- Before parametric tests"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- t-test (2 groups)\n"
                "- ANOVA (3+ groups)\n"
                "- Mann-Whitney U\n"
                "- Kruskal-Wallis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Overlapping boxes do NOT guarantee non-significance. "
                "Median outside the other's box is a rough heuristic, "
                "not a formal test. Always verify with proper testing."
            )


def violin_widget():
    st.markdown("## Violin Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 10, 200, 80, key="vio_n")
        n_groups = st.selectbox("Number of Groups", [1, 2, 3], index=2, key="vio_ng")
        show_box = st.toggle("Show Inner Boxplot", True, key="vio_box")
        separation = st.slider("Group Separation", 0.0, 3.0, 1.0, 0.1, key="vio_sep")
    np.random.seed(42)
    groups, names = [], []
    for i in range(int(n_groups)):
        d = np.random.normal(i * separation, 0.5, n)
        groups.append(d)
        names.append(f"Group {chr(65 + i)}")
    fig = go.Figure()
    for i, (d, name) in enumerate(zip(groups, names)):
        fig.add_trace(
            go.Violin(
                y=d,
                name=name,
                box_visible=show_box,
                meanline_visible=True,
                line_color=px.colors.qualitative.Plotly[i],
                fillcolor=px.colors.qualitative.Plotly[i],
                opacity=0.6,
                hovertemplate=f"{name}<br>y = %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title="Value",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Width = density of observations\n"
                "- Narrow sections = sparse data\n"
                "- Wide sections = dense data\n"
                "- Symmetry = normal distribution\n"
                "- Boxplot inside = summary stats"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Replace boxplot for distribution shape\n"
                "- Detect multimodality in groups\n"
                "- Compare shape + spread + central tendency\n"
                "- Assess normality assumptions"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- ANOVA\n"
                "- Kruskal-Wallis\n"
                "- Welch's t-test\n"
                "- Permutation test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Violin plots can be misleading with very small samples "
                "(n < 15) because the KDE over-interprets sparse data. "
                "Use boxplots or stripcharts for small samples."
            )


def qq_widget():
    st.markdown("## Q-Q Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 500, 100, key="qq_n")
        dist_type = st.selectbox(
            "Data Distribution",
            ["Normal", "Heavy-tailed (t)", "Skewed", "Uniform"],
            key="qq_dist",
        )
        tail_param = st.slider(
            "Tail Heaviness / Skew",
            0.5,
            5.0,
            2.0,
            0.1,
            key="qq_tail",
            help="t dist: lower df = heavier tails. Skew: higher = more skew.",
        )
    np.random.seed(42)
    if dist_type == "Normal":
        data = np.random.normal(0, 1, n)
    elif dist_type == "Heavy-tailed (t)":
        data = np.random.standard_t(tail_param, n)
    elif dist_type == "Skewed":
        data = np.random.gamma(tail_param, 1, n)
    else:
        data = np.random.uniform(-1.5, 1.5, n)
    os = np.sort(data)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, n))
    r, _ = stats.pearsonr(theoretical, os)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=os,
            mode="markers",
            name="Observed",
            marker=dict(color="#4C78A8", size=5),
            hovertemplate="Theoretical: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>",
        )
    )
    line_x = np.linspace(min(theoretical), max(theoretical), 10)
    line_y = np.linspace(min(os), max(os), 10)
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="y=x (Normal)",
            line=dict(color="#E45756", dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Observed Quantiles",
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"r = {r:.3f}",
                showarrow=False,
                font=dict(color="white", size=14),
            )
        ],
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Points on diagonal = normal\n"
                "- S-curve = heavy/light tails\n"
                "- Curve above line = right skew\n"
                "- Curve below line = left skew\n"
                "- r near 1 approx normal"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Check normality assumption\n"
                "- Identify distribution shape\n"
                "- Detect outliers in tails\n"
                "- Before t-test, ANOVA, regression"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Shapiro-Wilk\n"
                "- Kolmogorov-Smirnov\n"
                "- Anderson-Darling\n"
                "- D'Agostino-Pearson"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Small samples (n < 30) produce noisy Q-Q plots that "
                "can look non-normal even when data are normal. "
                "Always complement with a formal test for small n."
            )


# --- COMPARISON PLOTS ---


def grouped_bar_widget():
    st.markdown("## Grouped Bar Chart")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_groups = st.selectbox("Number of Groups", [2, 3, 4], index=1, key="gb_n")
        n_cats = st.selectbox("Number of Categories", [2, 3, 4], index=2, key="gb_cats")
        effect = st.slider(
            "Group Difference Effect",
            0.0,
            3.0,
            1.0,
            0.1,
            key="gb_eff",
            help="How much groups differ from each other",
        )
        show_error = st.toggle("Show Error Bars", True, key="gb_err")
    np.random.seed(42)
    n_g, n_c = int(n_groups), int(n_cats)
    means = np.zeros((n_c, n_g))
    for c_i in range(n_c):
        for g_i in range(n_g):
            means[c_i, g_i] = g_i * effect + c_i * 0.3
    data = means + np.random.normal(0, 0.3, (n_c, n_g))
    errors = np.abs(np.random.normal(0.2, 0.1, (n_c, n_g)))
    groups = [f"Group {chr(65 + i)}" for i in range(n_g)]
    categories = [f"Cat {i + 1}" for i in range(n_c)]
    fig = go.Figure()
    for g_i in range(n_g):
        fig.add_trace(
            go.Bar(
                name=groups[g_i],
                x=categories,
                y=data[:, g_i],
                error_y=dict(type="data", array=errors[:, g_i]) if show_error else None,
                marker_color=px.colors.qualitative.Plotly[g_i],
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        barmode="group",
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Bar height = group mean\n"
                "- Error bars = variability\n"
                "- Non-overlapping = likely sig.\n"
                "- Pattern consistency across cats"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare groups across categories\n"
                "- Show means with uncertainty\n"
                "- Present descriptive results\n"
                "- Visualize interaction effects"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Two-way ANOVA\n"
                "- Welch's t-test\n"
                "- Mixed-effects models\n"
                "- Post-hoc comparisons"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Bars should always start at zero to avoid misleading "
                "visual amplification of differences. Truncated y-axes "
                "exaggerate small effects."
            )


def error_bar_widget():
    st.markdown("## Error Bar Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 5, 100, 30, key="eb_n")
        n_groups = st.selectbox("Number of Groups", [2, 3, 4, 5], index=1, key="eb_ng")
        ci_width = st.slider(
            "Confidence Level",
            0.8,
            0.99,
            0.95,
            0.01,
            key="eb_ci",
            help="Width of confidence interval",
        )
        effect = st.slider("Effect Size (Cohen's d)", 0.0, 2.0, 0.5, 0.1, key="eb_eff")
    np.random.seed(42)
    n_g = int(n_groups)
    means_arr = [i * effect * 0.5 for i in range(n_g)]
    z = stats.norm.ppf(1 - (1 - ci_width) / 2)
    fig = go.Figure()
    for i in range(n_g):
        d = np.random.normal(means_arr[i], 0.3, n)
        sem = np.std(d, ddof=1) / np.sqrt(n)
        fig.add_trace(
            go.Scatter(
                x=[f"Group {chr(65 + i)}"],
                y=[np.mean(d)],
                error_y=dict(type="data", array=[z * sem], thickness=1.5, width=10),
                mode="markers",
                marker=dict(size=12, color=px.colors.qualitative.Plotly[i]),
                name=f"Group {chr(65 + i)}",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title="Mean +/- CI",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Dot = group mean\n"
                "- Bar = confidence interval\n"
                "- Non-overlapping CIs approx p < .05\n"
                "- Wide CI = imprecise estimate"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Show precision of estimates\n"
                "- Compare group means visually\n"
                "- Present meta-analysis results\n"
                "- Forest plots"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- t-test\n"
                "- ANOVA\n"
                "- Welch's t-test\n"
                "- Linear regression"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Non-overlapping 95% CIs are more conservative than "
                "p < 0.05 - they approximately correspond to p < 0.01. "
                "Overlapping CIs do NOT guarantee non-significance."
            )


def paired_line_widget():
    st.markdown("## Paired Line Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Pairs", 5, 100, 20, key="pl_n")
        effect = st.slider(
            "Pre-Post Effect",
            -2.0,
            2.0,
            0.8,
            0.1,
            key="pl_eff",
            help="How much values change from pre to post",
        )
        noise = st.slider("Within-Pair Noise", 0.1, 2.0, 0.5, 0.1, key="pl_noise")
        show_means = st.toggle("Show Mean Line", True, key="pl_mean")
    np.random.seed(42)
    pre = np.random.normal(0, 1, n)
    post = pre + effect + np.random.normal(0, noise, n)
    fig = go.Figure()
    for i in range(n):
        fig.add_trace(
            go.Scatter(
                x=["Pre", "Post"],
                y=[pre[i], post[i]],
                mode="lines+markers",
                line=dict(color="rgba(200,200,200,0.3)", width=1),
                marker=dict(size=4, color="rgba(200,200,200,0.5)"),
                showlegend=False,
                hovertemplate=f"Subject {i+1}<br>" + "%{x}: %{y:.2f}<extra></extra>",
            )
        )
    if show_means:
        fig.add_trace(
            go.Scatter(
                x=["Pre", "Post"],
                y=[np.mean(pre), np.mean(post)],
                mode="lines+markers",
                name="Mean Change",
                line=dict(color="#E45756", width=4),
                marker=dict(size=10, color="#E45756", symbol="diamond"),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title="Value",
        showlegend=True,
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each line = one subject\n"
                "- Upward = increase over time\n"
                "- Downward = decrease over time\n"
                "- Thick line = group average\n"
                "- Consistent direction = effect"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Pre-post intervention studies\n"
                "- Within-subject designs\n"
                "- Repeated measures (2 time points)\n"
                "- Crossover trials"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Paired t-test\n"
                "- Wilcoxon Signed-Rank\n"
                "- Repeated measures ANOVA\n"
                "- Linear mixed models"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Ignoring the paired nature of the data. Using an "
                "independent t-test on paired data inflates Type II "
                "error and ignores within-subject correlation."
            )


def boxplot_comp_widget():
    st.markdown("## Boxplot Comparison")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 10, 200, 50, key="bpc_n")
        n_groups = st.selectbox("Number of Groups", [2, 3, 4], index=1, key="bpc_ng")
        effect = st.slider("Between-Group Effect", 0.0, 3.0, 0.8, 0.1, key="bpc_eff")
        var_ratio = st.slider(
            "Variance Ratio (group 1 : last)",
            0.2,
            5.0,
            1.0,
            0.1,
            key="bpc_var",
            help="Heterogeneity of variance across groups",
        )
    np.random.seed(42)
    n_g = int(n_groups)
    groups, names = [], []
    for i in range(n_g):
        sd = 0.5 * (1 + (var_ratio - 1) * i / max(n_g - 1, 1))
        d = np.random.normal(i * effect, sd, n)
        groups.append(d)
        names.append(f"Group {chr(65 + i)}")
    fig = go.Figure()
    for i, (d, name) in enumerate(zip(groups, names)):
        fig.add_trace(
            go.Box(
                y=d,
                name=name,
                marker_color=px.colors.qualitative.Plotly[i],
                boxpoints="outliers",
                hovertemplate=f"{name}<br>y = %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title="Value",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Compare medians across groups\n"
                "- Box overlap approx group similarity\n"
                "- Box size = group variability\n"
                "- Whisker length = tail behavior"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare multiple groups robustly\n"
                "- Check equal variance assumption\n"
                "- Identify group-level outliers\n"
                "- Explore pilot data"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- One-way ANOVA\n"
                "- Welch's ANOVA\n"
                "- Kruskal-Wallis\n"
                "- Levene's test (variance)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Boxplots hide distribution shape entirely - "
                "different distributions can produce identical boxplots "
                "(Anscombe's quartet). Always check with a violin plot too."
            )


def violin_comp_widget():
    st.markdown("## Violin Comparison")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 10, 200, 60, key="vic_n")
        n_groups = st.selectbox("Number of Groups", [2, 3], index=1, key="vic_ng")
        effect = st.slider("Group Effect", 0.0, 3.0, 1.0, 0.1, key="vic_eff")
        shape_diff = st.toggle(
            "Show Shape Difference",
            False,
            key="vic_shape",
            help="Make groups have different distribution shapes",
        )
        split = st.toggle("Split Violin (Side-by-side)", False, key="vic_split")
    np.random.seed(42)
    n_g = int(n_groups)
    groups = []
    for i in range(n_g):
        if shape_diff and i == 1:
            d = np.random.gamma(2, 0.5, n) + i * effect
        elif shape_diff and i == 2:
            d = np.concatenate(
                [
                    np.random.normal(i * effect - 1, 0.3, n // 2),
                    np.random.normal(i * effect + 1, 0.3, n // 2),
                ]
            )
        else:
            d = np.random.normal(i * effect, 0.5, n)
        groups.append(d)
    names_a = [f"Group {chr(65 + i)}" for i in range(n_g)]
    fig = go.Figure()
    for i, d in enumerate(groups):
        side = (
            "positive"
            if (split and i == 0)
            else "negative" if (split and i == 1) else "both"
        )
        fig.add_trace(
            go.Violin(
                y=d,
                name=names_a[i],
                box_visible=True,
                meanline_visible=True,
                line_color=px.colors.qualitative.Plotly[i],
                fillcolor=px.colors.qualitative.Plotly[i],
                opacity=0.6,
                side=side,
                hovertemplate=f"{names_a[i]}<br>y = %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        yaxis_title="Value",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Width = density of data\n"
                "- Shift = group difference\n"
                "- Shape difference = distributional diff\n"
                "- Box inside shows median & IQR"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Replace boxplot for detailed comparison\n"
                "- Detect shape differences between groups\n"
                "- Assess normality & equal variance\n"
                "- Visualize 2-group comparisons"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Independent t-test\n"
                "- Mann-Whitney U\n"
                "- Welch's t-test\n"
                "- Fligner-Killeen (variance)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Violin width is often misinterpreted as frequency. "
                "It represents density - a wide violin does not mean "
                "more data, it means data are more spread out."
            )


# --- CORRELATION PLOTS ---


def scatter_widget():
    st.markdown("## Scatterplot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 500, 100, key="scat_n")
        r = st.slider(
            "Correlation (r)",
            -1.0,
            1.0,
            0.5,
            0.05,
            key="scat_r",
            help="Strength and direction of relationship",
        )
        noise = st.slider(
            "Noise",
            0.05,
            1.0,
            0.2,
            0.05,
            key="scat_noise",
            help="Higher = more scatter around the line",
        )
        add_outlier = st.toggle("Add Outlier", False, key="scat_out")
        show_line = st.toggle("Show Regression Line", True, key="scat_line")
    np.random.seed(42)
    x, y = _gen_corr(n, r, noise, outlier=add_outlier)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker=dict(color="#4C78A8", size=6, opacity=0.7),
            hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>",
        )
    )
    if show_line:
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        line_x = np.linspace(min(x), max(x), 100)
        line_y = slope * line_x + intercept
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name=f"r = {r_val:.3f}",
                line=dict(color="#E45756", width=2),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Upward slope = positive r\n"
                "- Downward slope = negative r\n"
                "- Tight cluster = strong correlation\n"
                "- Wide scatter = weak correlation\n"
                "- Single outlier can change r"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Assess relationship direction & strength\n"
                "- Detect outliers and non-linearity\n"
                "- Check homoscedasticity\n"
                "- Explore two continuous variables"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Pearson correlation\n"
                "- Spearman correlation\n"
                "- Linear regression\n"
                "- Correlation test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Correlation does NOT imply causation. "
                "A strong r could be spurious, confounded, "
                "or driven by outliers. Always visualize first."
            )


def heatmap_widget():
    st.markdown("## Correlation Heatmap")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_vars = st.selectbox(
            "Number of Variables", [4, 5, 6, 7, 8], index=1, key="hm_nvars"
        )
        strength = st.slider(
            "Average Correlation Strength",
            0.0,
            1.0,
            0.5,
            0.05,
            key="hm_str",
            help="How strongly variables are correlated on average",
        )
        show_annot = st.toggle("Show Correlation Values", True, key="hm_annot")
        cluster = st.toggle(
            "Show Clustered Order",
            False,
            key="hm_cluster",
            help="Group similar correlations together",
        )
    np.random.seed(42)
    n = int(n_vars)
    cov = np.full((n, n), strength * 0.4)
    np.fill_diagonal(cov, 1.0)
    for i in range(n):
        for j in range(n):
            cov[i, j] += np.random.uniform(-0.2, 0.2)
            cov[j, i] = cov[i, j]
    np.fill_diagonal(cov, 1.0)
    eigvals = np.linalg.eigvalsh(cov)
    if min(eigvals) <= 0:
        cov += np.eye(n) * (abs(min(eigvals)) + 0.01)
    data = np.random.multivariate_normal(np.zeros(n), cov, 200)
    corr = np.corrcoef(data.T)
    labels = [f"V{i + 1}" for i in range(n)]
    if cluster:
        from scipy.cluster.hierarchy import linkage, leaves_list

        try:
            link = linkage(corr, method="average")
            order = leaves_list(link)
            corr = corr[order][:, order]
            labels = [labels[i] for i in order]
        except Exception:
            pass
    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=labels,
            y=labels,
            text=np.round(corr, 2) if show_annot else None,
            texttemplate="%{text}" if show_annot else None,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="",
        yaxis_title="",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Red = positive correlation\n"
                "- Blue = negative correlation\n"
                "- Dark = strong relationship\n"
                "- White/light = weak relationship\n"
                "- Diagonal = variable with itself (r=1)"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Check multicollinearity\n"
                "- Explore many variable relationships\n"
                "- Identify variable clusters\n"
                "- Before regression / factor analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Pearson correlation matrix\n"
                "- Spearman correlation matrix\n"
                "- Variance Inflation Factor (VIF)\n"
                "- Factor analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Visual patterns can be misleading when variables "
                "have different scales. Always use standardized "
                "(rank or z-score) data for correlation heatmaps."
            )


def bubble_widget():
    st.markdown("## Bubble Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Points", 10, 200, 50, key="bub_n")
        r = st.slider("X-Y Correlation", -1.0, 1.0, 0.3, 0.05, key="bub_r")
        size_effect = st.slider(
            "Size-Value Correlation",
            0.0,
            1.0,
            0.6,
            0.05,
            key="bub_size",
            help="How strongly bubble size relates to y-value",
        )
        show_color = st.toggle("Color by Third Variable", True, key="bub_color")
    np.random.seed(42)
    n_actual = int(n)
    x, y_base = _gen_corr(n_actual, r, 0.3)
    size = np.random.uniform(5, 40, n_actual) * (
        1 + size_effect * (y_base - np.mean(y_base))
    )
    size = np.clip(size, 5, 80)
    color = np.random.uniform(0, 100, n_actual) if show_color else None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_base,
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                colorscale="Viridis",
                showscale=show_color,
                opacity=0.7,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<br>size = %{marker.size:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Position = X-Y relationship\n"
                "- Bubble size = third variable\n"
                "- Color = fourth variable\n"
                "- Larger bubbles draw attention"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Show 3-4 variables at once\n"
                "- Highlight weighted importance\n"
                "- Population/economic data\n"
                "- Risk-benefit visualizations"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Weighted correlation\n"
                "- Multiple regression\n"
                "- Weighted least squares\n"
                "- Meta-analysis (forest plots)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Bubble area (not diameter) should encode the third "
                "variable. Using diameter exaggerates differences "
                "and misleads the viewer."
            )


def monotonic_widget():
    st.markdown("## Monotonic vs Linear Correlation")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 300, 100, key="mono_n")
        rel_type = st.selectbox(
            "Relationship Type",
            ["Linear", "Quadratic (U-shape)", "Exponential", "Sine (Periodic)"],
            key="mono_type",
        )
        noise = st.slider("Noise Level", 0.0, 2.0, 0.3, 0.05, key="mono_noise")
        show_pearson = st.toggle("Show Pearson r", True, key="mono_pearson")
        show_spearman = st.toggle("Show Spearman rho", True, key="mono_spearman")
    np.random.seed(42)
    x = np.random.uniform(-3, 3, n)
    if rel_type == "Linear":
        y = x + np.random.normal(0, noise, n)
    elif rel_type == "Quadratic (U-shape)":
        y = x**2 + np.random.normal(0, noise, n)
    elif rel_type == "Exponential":
        y = np.exp(x / 2) + np.random.normal(0, noise, n)
    else:
        y = np.sin(x * 2) + np.random.normal(0, noise, n)
    r_p, _ = stats.pearsonr(x, y)
    r_s, _ = stats.spearmanr(x, y)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker=dict(color="#4C78A8", size=5, opacity=0.7),
            hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>",
        )
    )
    title = ""
    if show_pearson:
        title += f"Pearson r = {r_p:.3f}"
    if show_spearman:
        if title:
            title += " | "
        title += f"Spearman rho = {r_s:.3f}"
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="X",
        yaxis_title="Y",
        title=title if title else None,
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Pearson = linear relationship\n"
                "- Spearman = monotonic relationship\n"
                "- Big diff (r vs rho) = non-linear\n"
                "- r near 0 but rho large = monotonic"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Test if relationship is linear\n"
                "- Choose Pearson vs Spearman\n"
                "- Detect non-linear patterns\n"
                "- Understand correlation choice"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Pearson correlation\n"
                "- Spearman correlation\n"
                "- Kendall's tau\n"
                "- Distance correlation"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Pearson r near 0 does NOT mean no relationship - "
                "it only means no linear relationship. Always "
                "visualize: a U-shape can have r approx 0."
            )


# --- REGRESSION PLOTS ---


def linear_reg_widget():
    st.markdown("## Linear Regression Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="lr_n")
        slope = st.slider("True Slope (beta1)", -3.0, 3.0, 1.0, 0.1, key="lr_slope")
        noise = st.slider("Error (sigma)", 0.1, 5.0, 1.0, 0.1, key="lr_noise")
        show_ci = st.toggle("Show Confidence Band", True, key="lr_ci")
        show_resid = st.toggle("Show Residuals", False, key="lr_resid")
    np.random.seed(42)
    x, y = _gen_reg(n, slope, noise)
    slope_est, intercept, r_val, p_val, se = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 200)
    y_line = slope_est * x_line + intercept
    fig = go.Figure()
    if show_resid:
        y_pred = slope_est * x + intercept
        for xi, yi, ypi in zip(x, y, y_pred):
            fig.add_trace(
                go.Scatter(
                    x=[xi, xi],
                    y=[yi, ypi],
                    mode="lines",
                    line=dict(color="rgba(200,200,200,0.3)", width=1),
                    showlegend=False,
                )
            )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker=dict(color="#4C78A8", size=6, opacity=0.7),
            hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>",
        )
    )
    if show_ci:
        t_val = stats.t.ppf(0.975, n - 2)
        pred_se = np.sqrt(
            se**2
            * (1 + 1 / n + (x_line - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        )
        ci = t_val * pred_se
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line + ci,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line - ci,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(228, 87, 86, 0.15)",
                name="95% CI",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"beta1={slope_est:.2f}, R-squared={r_val**2:.3f}",
            line=dict(color="#E45756", width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Line = best fit (minimizes residuals)\n"
                "- Band = 95% CI for prediction\n"
                "- Slope = change in Y per unit X\n"
                "- R-squared = prop of variance explained\n"
                "- p-value = test if slope != 0"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Model continuous outcome\n"
                "- Estimate effect size (beta1)\n"
                "- Predict Y from X\n"
                "- Test linear relationship"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- F-test (overall model)\n"
                "- t-test (coefficient)\n"
                "- Pearson correlation\n"
                "- ANOVA (nested models)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Extrapolating beyond the observed x-range. "
                "Linear relationships may not hold outside "
                "the data - never predict beyond your data."
            )


def multiple_reg_widget():
    st.markdown("## Multiple Regression Surface")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="mr_n")
        b1 = st.slider("Coefficient beta1", -2.0, 2.0, 0.7, 0.1, key="mr_b1")
        b2 = st.slider("Coefficient beta2", -2.0, 2.0, 0.5, 0.1, key="mr_b2")
        noise = st.slider("Noise", 0.1, 3.0, 0.5, 0.1, key="mr_noise")
        show_interaction = st.toggle(
            "Show Interaction",
            False,
            key="mr_interact",
            help="Include beta12 x X1 x X2 term",
        )
    np.random.seed(42)
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    if show_interaction:
        y = b1 * x1 + b2 * x2 + 0.5 * x1 * x2 + np.random.normal(0, noise, n)
    else:
        y = b1 * x1 + b2 * x2 + np.random.normal(0, noise, n)
    grid = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(grid, grid)
    if show_interaction:
        Y_pred = b1 * X1 + b2 * X2 + 0.5 * X1 * X2
    else:
        Y_pred = b1 * X1 + b2 * X2
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x1,
            y=x2,
            z=y,
            mode="markers",
            name="Data",
            marker=dict(size=4, color="#4C78A8", opacity=0.7),
            hovertemplate="X1=%{x:.2f}<br>X2=%{y:.2f}<br>Y=%{z:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Surface(
            x=grid,
            y=grid,
            z=Y_pred,
            name="Predicted",
            colorscale="Reds",
            opacity=0.5,
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        scene=dict(
            xaxis_title="X1",
            yaxis_title="X2",
            zaxis_title="Y",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Plane = predicted Y from X1, X2\n"
                "- Slope along X1 = beta1 (holding X2 constant)\n"
                "- Slope along X2 = beta2 (holding X1 constant)\n"
                "- Twisted surface = interaction"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Model multiple predictors\n"
                "- Control for confounders\n"
                "- Test interaction effects\n"
                "- Understand partial effects"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- F-test (overall model)\n"
                "- Partial F-test (nested models)\n"
                "- t-test (individual coefficients)\n"
                "- VIF (multicollinearity)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Coefficients are sensitive to predictor scaling. "
                "Standardize (z-score) predictors before comparing "
                "coefficient magnitudes. Unstandardized betas "
                "depend on the unit of X."
            )


def logistic_widget():
    st.markdown("## Logistic Regression Sigmoid Curve")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="logit_n")
        separation = st.slider(
            "Class Separation",
            0.5,
            5.0,
            2.0,
            0.1,
            key="logit_sep",
            help="How far apart the two classes are on X",
        )
        threshold = st.slider(
            "Decision Threshold", 0.1, 0.9, 0.5, 0.05, key="logit_thresh"
        )
        show_data = st.toggle("Show Data Points", True, key="logit_data")
        show_prob = st.toggle("Show Probability Curve", True, key="logit_prob")
    np.random.seed(42)
    x0 = np.random.normal(-separation / 2, 1, n // 2)
    x1 = np.random.normal(separation / 2, 1, n - n // 2)
    x = np.concatenate([x0, x1])
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1000)
    model.fit(x.reshape(-1, 1), y)
    x_grid = np.linspace(min(x) - 1, max(x) + 1, 300)
    y_prob = model.predict_proba(x_grid.reshape(-1, 1))[:, 1]
    fig = go.Figure()
    if show_prob:
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_prob,
                mode="lines",
                name="P(Y=1)",
                line=dict(color="#4C78A8", width=3),
                hovertemplate="X = %{x:.2f}<br>P(Y=1) = %{y:.3f}<extra></extra>",
            )
        )
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#E45756",
            annotation_text=f"Threshold = {threshold}",
        )
    if show_data:
        jitter = np.random.uniform(-0.05, 0.05, n)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y + jitter,
                mode="markers",
                marker=dict(
                    color=y, colorscale="RdBu", size=6, showscale=False, opacity=0.7
                ),
                name="Data",
                hovertemplate="X = %{x:.2f}<br>Y = %{y:.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Predictor (X)",
        yaxis_title="Probability",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- S-curve = logistic probability\n"
                "- Steep curve = strong predictor\n"
                "- Shallow curve = weak predictor\n"
                "- Threshold = classification cutoff\n"
                "- Above threshold -> predict class 1"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Binary outcome prediction\n"
                "- Estimate odds ratios\n"
                "- Medical diagnosis models\n"
                "- Risk factor analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Likelihood ratio test\n"
                "- Wald test (coefficients)\n"
                "- Hosmer-Lemeshow (calibration)\n"
                "- ROC-AUC (discrimination)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Changing the decision threshold changes "
                "sensitivity and specificity. Default 0.5 "
                "is not always optimal - adjust based on "
                "the cost of false positives vs false negatives."
            )


def multinomial_widget():
    st.markdown("## Multinomial Decision Boundaries")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Samples per Class", 10, 100, 30, key="multi_n")
        n_classes = st.selectbox("Number of Classes", [3, 4], index=0, key="multi_k")
        separation = st.slider("Class Separation", 0.5, 4.0, 2.0, 0.1, key="multi_sep")
        show_boundaries = st.toggle("Show Decision Regions", True, key="multi_bound")
    np.random.seed(42)
    k = int(n_classes)
    centers = [
        [np.cos(2 * np.pi * i / k) * separation, np.sin(2 * np.pi * i / k) * separation]
        for i in range(k)
    ]
    X_list2, y_list2 = [], []
    for i, center in enumerate(centers):
        X_list2.append(np.random.normal(center, 0.5, (n, 2)))
        y_list2.append(np.full(n, i))
    X = np.vstack(X_list2)
    y = np.concatenate(y_list2)
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    if show_boundaries:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=10, solver="lbfgs")
        model.fit(X, y)
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
        )
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                showscale=False,
                colorscale=[[i / (k - 1), colors[i]] for i in range(k)],
                opacity=0.3,
                name="Decision Regions",
                hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<extra></extra>",
            )
        )
    for i in range(k):
        mask = y == i
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=f"Class {i}",
                marker=dict(color=colors[i], size=6, opacity=0.8),
                hovertemplate=f"Class {i}<br>x1=%{{x:.2f}}<br>x2=%{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Colored regions = decision zones\n"
                "- Boundaries = where model is uncertain\n"
                "- Overlap = classification difficulty\n"
                "- Filled area = multinomial probability"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Multi-class classification\n"
                "- Understand decision boundaries\n"
                "- Compare classifier geometries\n"
                "- Feature space exploration"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Multinomial logistic regression\n"
                "- MANOVA\n"
                "- Discriminant analysis\n"
                "- Classification metrics"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Linear decision boundaries (logistic regression) "
                "cannot separate non-linear class patterns. "
                "If classes are interleaved, consider non-linear "
                "methods (kernels, trees, neural nets)."
            )


def ordinal_logit_widget():
    st.markdown("## Ordinal Logistic Probability Curves")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="ord_n")
        n_levels = st.selectbox(
            "Number of Ordinal Levels", [3, 4, 5], index=1, key="ord_k"
        )
        effect = st.slider("Predictor Effect", 0.0, 3.0, 1.0, 0.1, key="ord_eff")
        show_cumulative = st.toggle(
            "Show Cumulative Probabilities", False, key="ord_cum"
        )
    k = int(n_levels)
    np.random.seed(42)
    x = np.random.uniform(-3, 3, n)
    thresholds = np.sort(np.random.uniform(-2, 2, k - 1))
    logit = effect * x
    cum_probs = []
    for thresh in thresholds:
        cum_probs.append(1 / (1 + np.exp(-(logit - thresh))))
    cum_probs.append(np.ones(n))
    x_grid = np.linspace(-3, 3, 200)
    logit_grid = effect * x_grid
    cum_grid = []
    for thresh in thresholds:
        cum_grid.append(1 / (1 + np.exp(-(logit_grid - thresh))))
    cum_grid.append(np.ones(200))
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly[:k]
    if show_cumulative:
        for i in range(k - 1):
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=cum_grid[i],
                    mode="lines",
                    name=f"P(Y <= {i+1})",
                    line=dict(dash="dash", color=colors[i]),
                    hovertemplate="X = %{x:.2f}<br>P(Y <= %s) = %{y:.3f}<extra></extra>"
                    % (i + 1),
                )
            )
    for i in range(k):
        prob_i = (
            cum_grid[i]
            if i == 0
            else [cum_grid[i][j] - cum_grid[i - 1][j] for j in range(200)]
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=prob_i,
                mode="lines",
                name=f"P(Y = {i+1})",
                line=dict(color=colors[i], width=2.5),
                fill="tozeroy" if not show_cumulative else None,
                hovertemplate=f"X = %{{x:.2f}}<br>P(Y={i+1}) = %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Predictor",
        yaxis_title="Probability",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each curve = probability of one level\n"
                "- Curves shift with predictor value\n"
                "- Non-parallel = proportional odds violation\n"
                "- Steep transition = strong predictor"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Ordered categorical outcomes\n"
                "- Likert scale responses\n"
                "- Disease severity staging\n"
                "- Patient-reported outcomes"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Proportional odds test\n"
                "- Brant test (parallel regression)\n"
                "- Likelihood ratio test\n"
                "- Score test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "The proportional odds assumption (parallel curves) "
                "must hold. If curves cross or are non-parallel, "
                "a generalized ordered logit or multinomial model "
                "is needed."
            )


def poisson_widget():
    st.markdown("## Poisson Count Regression")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 300, 100, key="pois_n")
        base_rate = st.slider(
            "Base Rate (intercept)",
            0.5,
            5.0,
            2.0,
            0.1,
            key="pois_base",
            help="Expected count when X = 0",
        )
        effect = st.slider(
            "Effect (log-rate ratio)",
            0.0,
            2.0,
            0.5,
            0.05,
            key="pois_eff",
            help="Multiplicative effect per unit X",
        )
        show_mean = st.toggle("Show Mean Curve", True, key="pois_mean")
        show_overdisp = st.toggle(
            "Add Overdispersion",
            False,
            key="pois_over",
            help="Extra-Poisson variability",
        )
    np.random.seed(42)
    x = np.random.uniform(0, 5, n)
    log_lambda = np.log(base_rate) + effect * x
    if show_overdisp:
        y = np.random.negative_binomial(np.exp(log_lambda) * 2, 0.5, n)
    else:
        y = np.random.poisson(np.exp(log_lambda), n)
    x_grid = np.linspace(0, 5, 100)
    log_lambda_grid = np.log(base_rate) + effect * x_grid
    y_grid = np.exp(log_lambda_grid)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observed",
            marker=dict(color="#4C78A8", size=5, opacity=0.6),
            hovertemplate="X = %{x:.2f}<br>Count = %{y:.0f}<extra></extra>",
        )
    )
    if show_mean:
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name="Predicted Mean",
                line=dict(color="#E45756", width=3),
                hovertemplate="X = %{x:.2f}<br>Mean = %{y:.1f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Predictor",
        yaxis_title="Count",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Y-axis = count (0, 1, 2, ...)\n"
                "- Curve = predicted mean count\n"
                "- Spread increases with mean\n"
                "- Clustering at zero = zero-inflation"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Modeling count outcomes\n"
                "- Event frequencies\n"
                "- Rare disease incidence\n"
                "- Hospital readmissions"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Likelihood ratio test\n"
                "- Wald test\n"
                "- Deviance goodness-of-fit\n"
                "- Dispersion test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Poisson assumes mean = variance. If variance > mean "
                "(overdispersion), use Negative Binomial. Many zeros? "
                "Consider zero-inflated or hurdle models."
            )


# --- DIAGNOSTIC ACCURACY PLOTS ---


def confusion_widget():
    st.markdown("## Confusion Matrix Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 1000, 200, key="cm_n")
        prevalence = st.slider(
            "Prevalence (True Class 1 Rate)",
            0.05,
            0.95,
            0.3,
            0.05,
            key="cm_prev",
            help="Proportion of actual positives",
        )
        sensitivity = st.slider(
            "Sensitivity (True Positive Rate)", 0.5, 1.0, 0.85, 0.01, key="cm_sens"
        )
        specificity = st.slider(
            "Specificity (True Negative Rate)", 0.5, 1.0, 0.90, 0.01, key="cm_spec"
        )
    np.random.seed(42)
    n_pos = int(n * prevalence)
    n_neg = n - n_pos
    tp = int(n_pos * sensitivity)
    fn = n_pos - tp
    tn = int(n_neg * specificity)
    fp = n_neg - tn
    cm = [[tn, fp], [fn, tp]]
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / n
    labels = ["Predicted Negative", "Predicted Positive"]
    true_labels = ["True Negative", "True Positive"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=true_labels,
            text=[[f"{cm[0][0]}", f"{cm[0][1]}"], [f"{cm[1][0]}", f"{cm[1][1]}"]],
            texttemplate="%{text}",
            textfont=dict(size=16),
            colorscale="Blues",
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.1%}")
    m2.metric("Sensitivity", f"{sensitivity:.1%}")
    m3.metric("Specificity", f"{specificity:.1%}")
    m4.metric("Prevalence", f"{prevalence:.1%}")
    m1.metric("PPV", f"{ppv:.1%}")
    m2.metric("NPV", f"{npv:.1%}")
    m3.metric("TP", tp)
    m4.metric("FP", fp)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Diagonal = correct predictions\n"
                "- Off-diagonal = errors\n"
                "- Top-right (FP) = Type I error\n"
                "- Bottom-left (FN) = Type II error\n"
                "- PPV depends on prevalence"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Evaluate binary classifiers\n"
                "- Compare diagnostic tests\n"
                "- Understand error types\n"
                "- Choose operating threshold"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- McNemar's test (paired comparison)\n"
                "- Cohen's Kappa\n"
                "- ROC-AUC\n"
                "- Diagnostic likelihood ratios"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Accuracy is misleading with imbalanced classes. "
                "A test with 95% accuracy on 5% prevalence "
                "could be useless (always predict negative). "
                "Always report PPV, NPV, and prevalence."
            )


def roc_widget():
    st.markdown("## ROC Curve Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="roc_n")
        auc = st.slider(
            "AUC (Area Under Curve)",
            0.5,
            1.0,
            0.85,
            0.01,
            key="roc_auc",
            help="0.5 = random, 1.0 = perfect",
        )
        show_threshold = st.toggle("Show Optimal Threshold", True, key="roc_thresh")
        show_chance = st.toggle("Show Chance Line", True, key="roc_chance")
    np.random.seed(42)
    n_pos = n // 2
    n_neg = n - n_pos
    d_prime = stats.norm.ppf(auc) * np.sqrt(2) if auc < 0.99 else 5
    scores_pos = np.random.normal(d_prime / 2, 1, n_pos)
    scores_neg = np.random.normal(-d_prime / 2, 1, n_neg)
    scores = np.concatenate([scores_neg, scores_pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    thresholds = np.sort(scores)
    tpr, fpr = [], []
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn_ = np.sum((pred == 0) & (labels == 1))
        tn_ = np.sum((pred == 0) & (labels == 0))
        tpr.append(tp / (tp + fn_) if (tp + fn_) > 0 else 0)
        fpr.append(fp / (fp + tn_) if (fp + tn_) > 0 else 0)
    tpr, fpr = np.array(tpr), np.array(fpr)
    idx = np.argsort(fpr)
    fpr_s, tpr_s = fpr[idx], tpr[idx]
    auc_actual = np.trapz(tpr_s, fpr_s)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr_s,
            y=tpr_s,
            mode="lines",
            name=f"ROC (AUC = {auc_actual:.3f})",
            line=dict(color="#4C78A8", width=3),
            hovertemplate="FPR = %{x:.3f}<br>TPR = %{y:.3f}<extra></extra>",
        )
    )
    if show_chance:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line=dict(color="rgba(200,200,200,0.5)", dash="dash"),
            )
        )
    if show_threshold:
        youden = tpr_s - fpr_s
        best_idx = np.argmax(youden)
        fig.add_trace(
            go.Scatter(
                x=[fpr_s[best_idx]],
                y=[tpr_s[best_idx]],
                mode="markers",
                name="Optimal Threshold",
                marker=dict(color="#E45756", size=12, symbol="star"),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Curve closer to top-left = better\n"
                "- AUC = probability correct ranking\n"
                "- AUC 0.5 = guessing\n"
                "- AUC 0.8+ = good discrimination\n"
                "- Star = Youden's optimal threshold"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare diagnostic tests\n"
                "- Assess model discrimination\n"
                "- Choose optimal threshold\n"
                "- Meta-analysis of test accuracy"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- DeLong test (compare AUCs)\n"
                "- Hanley-McNeil test\n"
                "- Bootstrap AUC comparison\n"
                "- Sensitivity analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "AUC ignores calibration - a model can have "
                "high AUC but poorly calibrated probabilities. "
                "Always check calibration (calibration plot) "
                "alongside ROC analysis."
            )


def pr_curve_widget():
    st.markdown("## Precision-Recall Curve")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 200, key="pr_n")
        prevalence = st.slider(
            "Prevalence (Class Imbalance)",
            0.02,
            0.5,
            0.1,
            0.01,
            key="pr_prev",
            help="Lower = more imbalanced",
        )
        sep = st.slider("Class Separation", 0.5, 3.0, 1.5, 0.1, key="pr_sep")
        show_baseline = st.toggle("Show Baseline (Prevalence)", True, key="pr_baseline")
    np.random.seed(42)
    n_pos = max(1, int(n * prevalence))
    n_neg = n - n_pos
    scores_pos = np.random.normal(sep / 2, 1, n_pos)
    scores_neg = np.random.normal(-sep / 2, 1, n_neg)
    scores = np.concatenate([scores_neg, scores_pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    thresholds = np.sort(scores)
    prec, rec = [], []
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn_ = np.sum((pred == 0) & (labels == 1))
        prec.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        rec.append(tp / (tp + fn_) if (tp + fn_) > 0 else 1.0)
    prec, rec = np.array(prec)[::-1], np.array(rec)[::-1]
    baseline = n_pos / n
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rec,
            y=prec,
            mode="lines",
            name="PR Curve",
            line=dict(color="#4C78A8", width=3),
            fill="tozeroy",
            hovertemplate="Recall = %{x:.3f}<br>Precision = %{y:.3f}<extra></extra>",
        )
    )
    if show_baseline:
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="rgba(200,200,200,0.5)",
            annotation_text=f"Baseline (Prevalence = {baseline:.2%})",
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Higher curve = better\n"
                "- Baseline = always-predict-positive\n"
                "- PR better than ROC for imbalanced\n"
                "- AP = area under PR curve"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Imbalanced classification\n"
                "- Rare disease detection\n"
                "- Fraud/anomaly detection\n"
                "- When PPV matters more"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Average Precision (AP)\n"
                "- F1 score (harmonic of P,R)\n"
                "- F-beta score (weighted F1)\n"
                "- Bootstrap PR comparison"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "PR curves from small samples are noisy and "
                "can show high precision simply by chance. "
                "Always use confidence bands (bootstrap) "
                "when sample size is limited."
            )


def threshold_widget():
    st.markdown("## Sensitivity vs Specificity Threshold Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 200, key="thresh_n")
        sep = st.slider("Class Separation", 0.5, 3.0, 1.5, 0.1, key="thresh_sep")
        prevalence = st.slider("Prevalence", 0.05, 0.95, 0.3, 0.05, key="thresh_prev")
        cost_fp = st.slider(
            "Cost of FP (relative to FN)",
            0.1,
            10.0,
            1.0,
            0.1,
            key="thresh_cost",
            help="Higher = penalize false positives more",
        )
    np.random.seed(42)
    n_pos = int(n * prevalence)
    n_neg = n - n_pos
    scores_pos = np.random.normal(sep / 2, 1, n_pos)
    scores_neg = np.random.normal(-sep / 2, 1, n_neg)
    scores = np.concatenate([scores_neg, scores_pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    thresholds = np.linspace(min(scores), max(scores), 100)
    sens, spec, costs = [], [], []
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn_ = np.sum((pred == 0) & (labels == 1))
        tn_ = np.sum((pred == 0) & (labels == 0))
        sens.append(tp / max(tp + fn_, 1))
        spec.append(tn_ / max(tn_ + fp, 1))
        costs.append(fp * cost_fp + fn_ * 1.0)
    opt_idx = np.argmin(costs)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=sens,
            mode="lines",
            name="Sensitivity",
            line=dict(color="#4C78A8", width=2),
            hovertemplate="Threshold = %{x:.2f}<br>Sensitivity = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=spec,
            mode="lines",
            name="Specificity",
            line=dict(color="#E45756", width=2),
            hovertemplate="Threshold = %{x:.2f}<br>Specificity = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=thresholds[opt_idx],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Optimal = {thresholds[opt_idx]:.2f}",
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Threshold",
        yaxis_title="Rate",
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Blue = sensitivity (catch positives)\n"
                "- Red = specificity (avoid false alarms)\n"
                "- Tradeoff: increase one = decrease other\n"
                "- Green = optimal based on costs"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Choose diagnostic cutoff\n"
                "- Balance sensitivity vs specificity\n"
                "- Incorporate cost of errors\n"
                "- Laboratory test thresholds"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- ROC analysis\n"
                "- Youden's index\n"
                "- Cost-benefit analysis\n"
                "- Decision curve analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Youden's index (max sensitivity + specificity - 1) "
                "treats FP and FN equally. In medicine, FN is often "
                "more costly (missed diagnosis). Adjust threshold "
                "based on clinical consequences, not statistics."
            )


def calibration_widget():
    st.markdown("## Calibration Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 100, 2000, 500, key="cal_n")
        calibration = st.slider(
            "Calibration Slope",
            0.0,
            2.0,
            1.0,
            0.05,
            key="cal_slope",
            help="1.0 = perfect, < 1 = overconfident, > 1 = underconfident",
        )
        noise = st.slider("Calibration Noise", 0.0, 0.3, 0.05, 0.01, key="cal_noise")
        n_bins = st.slider("Number of Bins", 5, 20, 10, key="cal_bins")
    np.random.seed(42)
    true_probs = np.random.uniform(0.05, 0.95, n)
    pred_probs = true_probs**calibration
    pred_probs = np.clip(pred_probs + np.random.normal(0, noise, n), 0.01, 0.99)
    y = (np.random.uniform(0, 1, n) < true_probs).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_mean_pred, bin_mean_obs = [], []
    for i in range(n_bins):
        mask = (pred_probs >= bins[i]) & (pred_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_mean_pred.append(np.mean(pred_probs[mask]))
            bin_mean_obs.append(np.mean(y[mask]))
        else:
            bin_mean_pred.append(bin_centers[i])
            bin_mean_obs.append(bin_centers[i])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bin_mean_pred,
            y=bin_mean_obs,
            mode="markers+lines",
            name="Model",
            marker=dict(color="#4C78A8", size=10),
            line=dict(color="#4C78A8", width=2),
            hovertemplate="Predicted = %{x:.3f}<br>Observed = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="rgba(200,200,200,0.5)", dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Proportion",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Points on diagonal = perfectly calibrated\n"
                "- Above diagonal = underestimated probability\n"
                "- Below diagonal = overestimated probability\n"
                "- Slope < 1 = overconfident (common)"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Assess probability accuracy\n"
                "- Check model reliability\n"
                "- Compare risk prediction models\n"
                "- Before clinical deployment"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Hosmer-Lemeshow test\n"
                "- Brier score\n"
                "- Spiegelhalter z-test\n"
                "- Calibration intercept & slope"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Hosmer-Lemeshow test is sensitive to binning "
                "choices and sample size. Large samples will "
                "reject even well-calibrated models. Use "
                "calibration plots + intercept/slope instead."
            )


# --- AGREEMENT PLOTS ---


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


def pca_widget():
    st.markdown("## PCA Scatter Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="pca_n")
        n_groups = st.selectbox(
            "Number of Groups", [2, 3, 4], index=1, key="pca_groups"
        )
        separation = st.slider("Group Separation", 0.0, 5.0, 2.0, 0.1, key="pca_sep")
        n_features = st.selectbox(
            "Number of Features", [5, 10, 20], index=0, key="pca_feat"
        )
        show_ellipse = st.toggle("Show Confidence Ellipses", True, key="pca_ellipse")
    k, n_f = int(n_groups), int(n_features)
    np.random.seed(42)
    X_list, y_list = [], []
    for i in range(k):
        center = np.random.uniform(-separation, separation, n_f) * (i / max(k - 1, 1))
        X_list.append(np.random.normal(center, 1, (n // k, n_f)))
        y_list.append(np.full(n // k, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    var_expl = pca.explained_variance_ratio_
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    for i in range(k):
        mask = y == i
        fig.add_trace(
            go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode="markers",
                name=f"Group {i}",
                marker=dict(color=colors[i], size=5, opacity=0.7),
                hovertemplate=f"Group {i}<br>PC1 = %{{x:.2f}}<br>PC2 = %{{y:.2f}}<extra></extra>",
            )
        )
        if show_ellipse and mask.sum() > 2:
            x_c, y_c = np.mean(X_pca[mask, 0]), np.mean(X_pca[mask, 1])
            angle = np.linspace(0, 2 * np.pi, 50)
            cov_ = np.cov(X_pca[mask].T)
            try:
                eigvals2, eigvecs = np.linalg.eigh(cov_)
                order = eigvals2.argsort()[::-1]
                eigvals2, eigvecs = eigvals2[order], eigvecs[:, order]
                theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
                a, b = 2 * np.sqrt(eigvals2[0]), 2 * np.sqrt(eigvals2[1])
                x_e = (
                    x_c
                    + a * np.cos(theta) * np.cos(angle)
                    - b * np.sin(theta) * np.sin(angle)
                )
                y_e = (
                    y_c
                    + a * np.sin(theta) * np.cos(angle)
                    + b * np.cos(theta) * np.sin(angle)
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_e,
                        y=y_e,
                        mode="lines",
                        line=dict(color=colors[i], width=1.5, dash="dash"),
                        showlegend=False,
                    )
                )
            except Exception:
                pass
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title=f"PC1 ({var_expl[0]:.1%} variance)",
        yaxis_title=f"PC2 ({var_expl[1]:.1%} variance)",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- PCs = directions of max variance\n"
                "- Close points = similar profiles\n"
                "- Separated groups = distinct clusters\n"
                "- Ellipses = 95% confidence region\n"
                "- Axis labels show % variance explained"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Reduce dimensionality\n"
                "- Visualize high-dim data\n"
                "- Check for natural clusters\n"
                "- Exploratory data analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- MANOVA (on PCs or raw)\n"
                "- Factor analysis\n"
                "- K-means clustering\n"
                "- PERMANOVA"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "PCA assumes linear relationships. Non-linear "
                "structure (e.g., U-shape, spiral) will NOT "
                "be captured. Use t-SNE or UMAP for non-linear "
                "dimensionality reduction."
            )


def manova_widget():
    st.markdown("## MANOVA Group Clouds")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Samples per Group", 10, 100, 30, key="man_n")
        n_groups = st.selectbox("Number of Groups", [2, 3], index=1, key="man_groups")
        n_dims = st.selectbox("Number of DVs", [2, 3], index=0, key="man_dims")
        effect = st.slider("Group Separation Effect", 0.0, 5.0, 2.0, 0.1, key="man_eff")
        show_centroids = st.toggle("Show Group Centroids", True, key="man_cent")
    k = int(n_groups)
    d = int(n_dims)
    np.random.seed(42)
    groups, names, colors = [], [], []
    for i in range(k):
        center = np.full(d, i * effect)
        cov = np.eye(d) * 0.5
        groups.append(np.random.multivariate_normal(center, cov, n))
        names.append(f"Group {chr(65 + i)}")
        colors.append(px.colors.qualitative.Plotly[i])
    fig = go.Figure()
    if d == 2:
        for i, (g, name, color) in enumerate(zip(groups, names, colors)):
            fig.add_trace(
                go.Scatter(
                    x=g[:, 0],
                    y=g[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=5, opacity=0.7),
                    hovertemplate=f"{name}<br>DV1 = %{{x:.2f}}<br>DV2 = %{{y:.2f}}<extra></extra>",
                )
            )
            if show_centroids:
                fig.add_trace(
                    go.Scatter(
                        x=[np.mean(g[:, 0])],
                        y=[np.mean(g[:, 1])],
                        mode="markers",
                        marker=dict(color=color, size=15, symbol="x"),
                        showlegend=False,
                    )
                )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="closest",
            xaxis_title="DV1",
            yaxis_title="DV2",
        )
    else:
        for i, (g, name, color) in enumerate(zip(groups, names, colors)):
            fig.add_trace(
                go.Scatter3d(
                    x=g[:, 0],
                    y=g[:, 1],
                    z=g[:, 2],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=4, opacity=0.7),
                    hovertemplate=f"{name}<br>DV1=%{{x:.2f}}<br>DV2=%{{y:.2f}}<br>DV3=%{{z:.2f}}<extra></extra>",
                )
            )
            if show_centroids:
                fig.add_trace(
                    go.Scatter3d(
                        x=[np.mean(g[:, 0])],
                        y=[np.mean(g[:, 1])],
                        z=[np.mean(g[:, 2])],
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="diamond"),
                        showlegend=False,
                    )
                )
        fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
            scene=dict(
                xaxis_title="DV1",
                yaxis_title="DV2",
                zaxis_title="DV3",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            ),
        )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each color = one group\n"
                "- Distance between = group difference\n"
                "- Overlap = no significant difference\n"
                "- X marks = group centroid (mean)\n"
                "- Ellipsoid shape = covariance"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare groups on multiple DVs\n"
                "- Control for correlated outcomes\n"
                "- Multivariate experimental design\n"
                "- Protect against inflated Type I error"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Pillai's Trace\n"
                "- Wilks' Lambda\n"
                "- Hotelling-Lawley Trace\n"
                "- Roy's Largest Root"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "MANOVA requires multivariate normality and "
                "homogeneity of covariance matrices. Violations "
                "inflate Type I error. When assumptions fail, "
                "consider PERMANOVA or non-parametric alternatives."
            )


def cluster_widget():
    st.markdown("## Cluster Visualization")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 30, 500, 150, key="clust_n")
        k_true = st.selectbox(
            "True Number of Clusters", [2, 3, 4, 5], index=1, key="clust_k"
        )
        separation = st.slider(
            "Cluster Separation", 0.5, 5.0, 2.0, 0.1, key="clust_sep"
        )
        n_features = st.selectbox(
            "Project from Features", [2, 5, 10], index=0, key="clust_feat"
        )
        show_centers = st.toggle("Show Cluster Centers", True, key="clust_center")
    k, n_f = int(k_true), int(n_features)
    np.random.seed(42)
    if n_f == 2:
        centers = [
            [
                np.cos(2 * np.pi * i / k) * separation,
                np.sin(2 * np.pi * i / k) * separation,
            ]
            for i in range(k)
        ]
        X = np.vstack(
            [np.random.normal(centers[i], 0.5, (n // k, 2)) for i in range(k)]
        )
    else:
        X_list = []
        for i in range(k):
            center = np.random.uniform(-separation, separation, n_f) * (
                i / max(k - 1, 1)
            )
            X_list.append(np.random.normal(center, 1, (n // k, n_f)))
        X = np.vstack(X_list)
        X = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    centers2 = kmeans.cluster_centers_
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    for i in range(k):
        mask = labels == i
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=f"Cluster {i}",
                marker=dict(color=colors[i], size=5, opacity=0.6),
                hovertemplate=f"Cluster {i}<br>x = %{{x:.2f}}<br>y = %{{y:.2f}}<extra></extra>",
            )
        )
    if show_centers:
        fig.add_trace(
            go.Scatter(
                x=centers2[:, 0],
                y=centers2[:, 1],
                mode="markers",
                marker=dict(
                    color="black",
                    size=12,
                    symbol="x",
                    line=dict(color="white", width=1),
                ),
                name="Centers",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="PC1 / Feature 1",
        yaxis_title="PC2 / Feature 2",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each color = discovered cluster\n"
                "- X marks = cluster center\n"
                "- Tight clusters = well-separated\n"
                "- Overlap = ambiguous assignment\n"
                "- K-means assumes spherical clusters"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Discover natural groupings\n"
                "- Segment patients/populations\n"
                "- Pattern recognition\n"
                "- Exploratory data mining"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Silhouette score\n"
                "- Elbow method (WCSS)\n"
                "- Gap statistic\n"
                "- Davies-Bouldin index"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "K-means requires specifying k beforehand and "
                "assumes spherical, equally-sized clusters. "
                "Elongated or irregular clusters will be "
                "incorrectly split. Use DBSCAN or GMM for "
                "complex shapes."
            )


def scatter3d_widget():
    st.markdown("## 3D Scatter Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="s3d_n")
        n_groups = st.selectbox(
            "Number of Groups", [1, 2, 3], index=1, key="sd3_groups"
        )
        correlation = st.slider(
            "Variable Correlation",
            -1.0,
            1.0,
            0.5,
            0.05,
            key="s3d_corr",
            help="Correlation among the 3 dimensions",
        )
        spread = st.slider("Data Spread", 0.2, 3.0, 1.0, 0.1, key="s3d_spread")
        rotate = st.toggle("Auto-Rotate", True, key="s3d_rotate")
    k = int(n_groups)
    np.random.seed(42)
    cov_mat = np.array(
        [
            [1, correlation, correlation],
            [correlation, 1, correlation],
            [correlation, correlation, 1],
        ]
    )
    eigvals = np.linalg.eigvalsh(cov_mat)
    if min(eigvals) <= 0:
        cov_mat += np.eye(3) * (abs(min(eigvals)) + 0.01)
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    for i in range(k):
        offset = np.full(3, i * spread * 1.5)
        data = np.random.multivariate_normal(offset, cov_mat * spread, n // k)
        fig.add_trace(
            go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode="markers",
                name=f"Group {chr(65 + i)}" if k > 1 else "Data",
                marker=dict(
                    color=colors[i] if k > 1 else "#4C78A8", size=4, opacity=0.7
                ),
                hovertemplate=f"{'Group ' + chr(65 + i) if k > 1 else 'Point'}"
                f"<br>X=%{{x:.2f}}<br>Y=%{{y:.2f}}<br>Z=%{{z:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each axis = one variable\n"
                "- Position in 3D space = multi-dim profile\n"
                "- Clusters = groups with similar profiles\n"
                "- Rotation reveals different patterns\n"
                "- Elliptical shape = correlated variables"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Explore 3-variable relationships\n"
                "- Identify 3D clusters\n"
                "- Present multivariate patterns\n"
                "- Interactive data exploration"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- MANOVA\n"
                "- Multivariate regression\n"
                "- Canonical correlation\n"
                "- 3D PCA visualization"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "3D plots can obscure patterns depending on "
                "viewing angle. Always rotate and view from "
                "multiple perspectives. Pre-projected 2D views "
                "(PCA) often reveal structure more clearly."
            )


def pp_widget():
    st.markdown("## Normal P-P Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="pp_n")
        dist_type = st.selectbox(
            "Distribution",
            ["Normal", "Skewed", "Heavy-tailed", "Uniform"],
            key="pp_dist",
        )
    np.random.seed(42)
    if dist_type == "Normal":
        data = np.random.normal(0, 1, n)
    elif dist_type == "Skewed":
        data = np.random.gamma(2, 1, n)
    elif dist_type == "Heavy-tailed":
        data = np.random.standard_t(3, n)
    else:
        data = np.random.uniform(-2, 2, n)
    data_sorted = np.sort(data)
    emp_p = (np.arange(1, n + 1) - 0.5) / n
    theo_p = stats.norm.cdf((data_sorted - np.mean(data)) / np.std(data))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theo_p,
            y=emp_p,
            mode="markers",
            marker=dict(color="#4C78A8", size=4),
            name="Observed",
            hovertemplate="Theoretical P=%{x:.3f}<br>Empirical P=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Ideal (Normal)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Theoretical Cumulative Probability",
        yaxis_title="Empirical Cumulative Probability",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Points on diagonal = normal\n"
                "- S-curve = heavy tails\n"
                "- Points above = right skew\n"
                "- Points below = left skew"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Assess normality assumption\n"
                "- Complement to Q-Q plot\n"
                "- Sensitive to center deviations\n"
                "- Visual goodness-of-fit check"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Shapiro-Wilk test\n"
                "- Kolmogorov-Smirnov\n"
                "- Anderson-Darling\n"
                "- D'Agostino-Pearson"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "P-P plots are less sensitive to tail "
                "departures than Q-Q plots. Use both "
                "for a complete normality assessment."
            )


def raincloud_widget():
    st.markdown("## Raincloud Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Samples per Group", 10, 200, 50, key="rain_n")
        n_groups = st.selectbox("Number of Groups", [2, 3, 4], index=0, key="rain_ng")
        separation = st.slider("Group Separation", 0.0, 3.0, 1.0, 0.1, key="rain_sep")
    np.random.seed(42)
    k = int(n_groups)
    colors = px.colors.qualitative.Plotly[:k]
    means = np.linspace(-separation * (k - 1) / 2, separation * (k - 1) / 2, k)
    data = [np.random.normal(m, 1, n) for m in means]
    fig = go.Figure()
    for i, d in enumerate(data):
        x_jitter = np.random.uniform(i - 0.2, i + 0.2, n)
        fig.add_trace(
            go.Scatter(
                x=x_jitter,
                y=d,
                mode="markers",
                marker=dict(color=colors[i], size=4, opacity=0.4),
                legendgroup=f"g{i}",
                name=f"Group {i+1}",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Box(
                x0=i,
                y=d,
                name=f"Group {i+1}",
                marker_color=colors[i],
                line=dict(color=colors[i], width=2),
                fillcolor="rgba(0,0,0,0)",
                boxpoints=False,
                width=0.15,
                legendgroup=f"g{i}",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Violin(
                x0=i,
                y=d,
                side="positive",
                line=dict(color=colors[i], width=2),
                fillcolor=colors[i],
                opacity=0.3,
                points=False,
                width=0.6,
                legendgroup=f"g{i}",
                name=f"Group {i+1}",
                showlegend=True,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(k)),
            ticktext=[f"Group {i+1}" for i in range(k)],
        ),
        yaxis_title="Value",
        hovermode="closest",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Points = raw data (cloud)\n"
                "- Box = median + IQR\n"
                "- Half-violin = density shape\n"
                "- Combines all three views"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Replace boxplot for more detail\n"
                "- Show both distribution and raw data\n"
                "- Modern publication-ready graphics\n"
                "- Small to moderate sample sizes"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Independent t-test\n"
                "- Mann-Whitney U\n"
                "- Welch's t-test\n"
                "- Permutation tests"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Jitter width is arbitrary and only "
                "shows density not exact x-position. "
                "Set random seed for reproducibility."
            )


def residuals_fitted_widget():
    st.markdown("## Residuals vs Fitted Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="resid_n")
        noise = st.slider("Noise Level", 0.1, 3.0, 1.0, 0.1, key="resid_noise")
        pattern = st.selectbox(
            "Pattern",
            ["Linear (OK)", "Heteroscedastic", "Non-linear", "Outlier"],
            key="resid_pattern",
        )
    np.random.seed(42)
    x = np.random.uniform(0, 10, n)
    if pattern == "Linear (OK)":
        y = 2 + 1.5 * x + np.random.normal(0, noise, n)
    elif pattern == "Heteroscedastic":
        y = 2 + 1.5 * x + np.random.normal(0, noise * (0.5 + 0.5 * x / 10), n)
    elif pattern == "Non-linear":
        y = 2 + 1.5 * x + 0.5 * x**2 + np.random.normal(0, noise, n)
    else:
        y = 2 + 1.5 * x + np.random.normal(0, noise, n)
        y[-1] += 15
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    fitted = model.predict(x.reshape(-1, 1))
    residuals = y - fitted
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            marker=dict(color="#4C78A8", size=5),
            name="Residuals",
            hovertemplate="Fitted=%{x:.2f}<br>Residual=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="red", dash="dash"), opacity=0.7)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Random scatter around 0 = OK\n"
                "- Fan shape = heteroscedasticity\n"
                "- U-shape = non-linearity\n"
                "- Isolated points = outliers"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- After fitting linear regression\n"
                "- Check homoscedasticity assumption\n"
                "- Check linearity assumption\n"
                "- Identify influential points"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Breusch-Pagan test\n"
                "- Goldfeld-Quandt test\n"
                "- RESET test\n"
                "- Cook's distance"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Patterned residuals indicate model "
                "misspecification — do NOT interpret "
                "coefficients until residuals are "
                "well-behaved."
            )


def poly_reg_widget():
    st.markdown("## Polynomial Regression Fit")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 300, 80, key="poly_reg_n")
        degree = st.slider("Polynomial Degree", 1, 10, 1, key="poly_reg_deg")
        noise = st.slider("Noise Level", 0.1, 3.0, 0.5, 0.1, key="poly_reg_noise")
        true_fn = st.selectbox(
            "True Relationship",
            ["Linear", "Quadratic", "Cubic", "Sine"],
            key="poly_reg_fn",
        )
    np.random.seed(42)
    x = np.sort(np.random.uniform(-3, 3, n))
    if true_fn == "Linear":
        y_true = 2 + 1.5 * x
    elif true_fn == "Quadratic":
        y_true = 1 + x + 0.5 * x**2
    elif true_fn == "Cubic":
        y_true = 1 + x + 0.5 * x**2 - 0.2 * x**3
    else:
        y_true = 2 * np.sin(x)
    y = y_true + np.random.normal(0, noise, n)
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression

    poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly.fit(x.reshape(-1, 1), y)
    x_smooth = np.linspace(min(x), max(x), 300)
    y_pred = poly.predict(x_smooth.reshape(-1, 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="#4C78A8", size=5, opacity=0.6),
            name="Data",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_pred,
            mode="lines",
            line=dict(color="#E45756", width=3),
            name=f"Degree {degree}",
            hovertemplate="x=%{x:.2f}<br>Pred=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_true,
            mode="lines",
            line=dict(color="gray", width=2, dash="dot"),
            name="True Function",
            hovertemplate="x=%{x:.2f}<br>True=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Higher degree = more flexible\n"
                "- Degree 1 = straight line\n"
                "- Degree 2 = one bend\n"
                "- Degree 10 can overfit wildly"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Model non-linear relationships\n"
                "- Test for curvature in data\n"
                "- Understand bias-variance tradeoff\n"
                "- Teaching overfitting concepts"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- F-test for nested models\n"
                "- Cross-validation MSE\n"
                "- AIC / BIC comparison\n"
                "- ANOVA model comparison"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "High-degree polynomials overfit "
                "near boundaries. Never extrapolate "
                "beyond data range. Use splines "
                "for better behavior."
            )


def reg_path_widget():
    st.markdown("## Regularization Path")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 100, key="regpath_n")
        n_features = st.slider("Number of Features", 5, 30, 10, key="regpath_k")
        reg_type = st.selectbox(
            "Regularization", ["Lasso (L1)", "Ridge (L2)"], key="regpath_type"
        )
    np.random.seed(42)
    true_coefs = np.zeros(n_features)
    true_coefs[:5] = [3, -2, 1.5, -1, 0.5]
    np.random.shuffle(true_coefs)
    X = np.random.normal(0, 1, (n, n_features))
    y = X @ true_coefs + np.random.normal(0, 1, n)
    from sklearn.linear_model import Lasso, Ridge

    alphas = np.logspace(-2, 2, 100)
    if reg_type == "Lasso (L1)":
        coefs = np.array(
            [Lasso(alpha=a, max_iter=10000).fit(X, y).coef_ for a in alphas]
        )
    else:
        coefs = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
    fig = go.Figure()
    for i in range(n_features):
        fig.add_trace(
            go.Scatter(
                x=np.log10(alphas),
                y=coefs[:, i],
                mode="lines",
                line=dict(width=1.5),
                name=f"Feature {i+1}",
                hovertemplate="log10(α)=%{x:.2f}<br>Coeff=%{y:.3f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"))
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="log10(Alpha)",
        yaxis_title="Coefficient Value",
        hovermode="closest",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each line = one coefficient\n"
                "- Left (low α) = unregularized\n"
                "- Right (high α) = strong shrinkage\n"
                "- Lasso forces coefficients to zero"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- High-dimensional data\n"
                "- Feature selection (Lasso)\n"
                "- Combat multicollinearity (Ridge)\n"
                "- Bias-variance tradeoff analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Cross-validated MSE\n"
                "- Regularization path stability\n"
                "- Bayesian information criterion\n"
                "- Bootstrap coefficient stability"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Lasso selects at most n features. "
                "With p > n, Ridge may generalize "
                "better. Always standardize predictors "
                "before regularization."
            )


def splom_widget():
    st.markdown("## Scatterplot Matrix (SPLOM)")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 30, 500, 150, key="splom_n")
        n_vars = st.selectbox(
            "Number of Variables", [3, 4, 5, 6], index=1, key="splom_k"
        )
        corr = st.slider("Correlation Strength", 0.0, 0.95, 0.5, 0.05, key="splom_corr")
    np.random.seed(42)
    k = int(n_vars)
    cov = np.full((k, k), corr)
    for i in range(k):
        cov[i, i] = 1.0
    data = np.random.multivariate_normal(np.zeros(k), cov, n)
    col_names = [f"Var {i+1}" for i in range(k)]
    df = pd.DataFrame(data, columns=col_names)
    fig = px.scatter_matrix(df, dimensions=col_names, opacity=0.5)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        template="plotly_dark", height=600, margin=dict(l=10, r=10, t=30, b=10)
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each panel = bivariate scatter\n"
                "- Diagonal = each variable vs itself\n"
                "- Tight ellipse = strong correlation\n"
                "- Row/col patterns = multivariate structure"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Explore many variable pairs at once\n"
                "- Detect multicollinearity patterns\n"
                "- Identify multivariate outliers\n"
                "- EDA for high-dimensional data"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Pearson correlation matrix\n"
                "- Variance Inflation Factor (VIF)\n"
                "- Bartlett's sphericity test\n"
                "- MANOVA assumptions check"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "SPLOMs become unreadable with >10 "
                "variables. Use correlation heatmap "
                "or PCA for higher dimensions."
            )


def parallel_coords_widget():
    st.markdown("## Parallel Coordinates Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="parcoords_n")
        n_dims = st.selectbox(
            "Number of Dimensions", [4, 5, 6, 7, 8], index=1, key="parcoords_k"
        )
        n_clusters = st.selectbox(
            "Number of Clusters", [2, 3, 4], index=0, key="parcoords_clust"
        )
    np.random.seed(42)
    k = int(n_dims)
    c = int(n_clusters)
    cluster_centers = np.random.uniform(-3, 3, (c, k))
    data_list = []
    labels = []
    for i in range(c):
        n_per = n // c
        data_list.append(np.random.normal(cluster_centers[i], 0.6, (n_per, k)))
        labels.extend([i] * n_per)
    X = np.vstack(data_list)[:n]
    labels = np.array(labels[:n])
    col_names = [f"Dim {i+1}" for i in range(k)]
    df = pd.DataFrame(X, columns=col_names)
    df["Cluster"] = labels
    fig = px.parallel_coordinates(
        df, color="Cluster", dimensions=col_names, color_continuous_scale="Viridis"
    )
    fig.update_layout(
        template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10)
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each vertical axis = one variable\n"
                "- Each line = one observation\n"
                "- Crossing lines = negative correlation\n"
                "- Parallel lines = positive correlation"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Visualize high-dimensional data\n"
                "- Identify variable relationships\n"
                "- Find multivariate patterns\n"
                "- Complement to PCA"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- MANOVA\n"
                "- Canonical correlation\n"
                "- Discriminant analysis\n"
                "- Cluster validation"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Axis order affects interpretation. "
                "Reorder axes to highlight patterns. "
                "Too many observations creates clutter "
                "— consider sampling."
            )


def stem_leaf_widget():
    st.markdown("## Stem-and-Leaf Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 200, 60, key="stemn")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Bimodal", "Uniform"], key="stemd"
        )
    np.random.seed(42)
    if dist == "Normal":
        d = np.random.normal(50, 15, n)
    elif dist == "Skewed":
        d = np.random.gamma(2, 20, n) + 10
    elif dist == "Bimodal":
        d = np.concatenate(
            [np.random.normal(30, 8, n // 2), np.random.normal(70, 8, n // 2)]
        )
    else:
        d = np.random.uniform(10, 90, n)
    d = np.round(d).astype(int)
    d = d[(d >= 0)]
    stems = d // 10
    leaves = d % 10
    us = np.sort(np.unique(stems))
    stem_vals = [str(s) for s in us]
    leaf_vals = [" ".join(str(l) for l in np.sort(leaves[stems == s])) for s in us]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Stem", "Leaf"],
                    font=dict(size=14),
                    align="left",
                    fill_color="#1E1E1E",
                ),
                cells=dict(
                    values=[stem_vals, leaf_vals],
                    font=dict(family="monospace", size=13),
                    align="left",
                    height=24,
                    fill_color="#2D2D2D",
                ),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(26 * len(us) + 60, 200),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Stem = leading digit(s)\n"
                "- Leaf = trailing digit\n"
                "- Row length = frequency\n"
                "- Preserves exact values"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Small datasets (< 200)\n"
                "- Quick distribution view\n"
                "- Paper-and-pencil stats\n"
                "- Classroom teaching tool"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Kolmogorov-Smirnov\n"
                "- Shapiro-Wilk\n"
                "- Anderson-Darling\n"
                "- Visual shape assessment"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Choose stem unit carefully — "
                "too few stems loses detail, "
                "too many creates sparse rows. "
                "Aim for 10-20 stems."
            )


def freq_poly_widget():
    st.markdown("## Frequency Polygon")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="fp_n")
        bins = st.slider("Number of Bins", 5, 30, 10, key="fp_bins")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Bimodal"], key="fp_dist"
        )
    np.random.seed(42)
    if dist == "Normal":
        d = np.random.normal(0, 1, n)
    elif dist == "Skewed":
        d = np.random.gamma(2, 1, n)
    else:
        d = np.concatenate(
            [np.random.normal(-1.5, 0.7, n // 2), np.random.normal(1.5, 0.7, n // 2)]
        )
    counts, edges = np.histogram(d, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=np.diff(edges),
            marker=dict(color="#4C78A8", opacity=0.4),
            name="Histogram",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=counts,
            mode="lines+markers",
            line=dict(color="#E45756", width=3),
            marker=dict(size=8),
            name="Frequency Polygon",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Value",
        yaxis_title="Frequency",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Points at bin midpoints\n"
                "- Line = distribution shape\n"
                "- Area under polygon = N\n"
                "- Smoother than histogram"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Overlay multiple distributions\n"
                "- Compare group shapes\n"
                "- Cumulative frequency curve\n"
                "- Smooth frequency display"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Chi-square GOF test\n"
                "- Kolmogorov-Smirnov\n"
                "- Anderson-Darling\n"
                "- Distribution fit test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Bin count changes polygon shape. "
                "Fewer bins = smoother but less "
                "detail. Always try multiple "
                "bin widths."
            )


def _swarm_positions(values, size=0.3):
    n = len(values)
    idx = np.argsort(values)
    sv = values[idx]
    pos = np.zeros((n, 2))
    placed = []
    for i, v in enumerate(sv):
        for r in range(50):
            for s in [1, -1] if r > 0 else [1]:
                x = s * r * size * 0.3
                ok = True
                for px, py in placed:
                    if abs(v - py) < size * 0.5 and abs(x - px) < size * 0.4:
                        ok = False
                        break
                if ok:
                    pos[i] = [x, v]
                    placed.append((x, v))
                    break
    return pos[np.argsort(idx)]


def beeswarm_widget():
    st.markdown("## Beeswarm / Swarm Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Samples per Group", 10, 150, 40, key="bsw_n")
        k = st.selectbox("Number of Groups", [2, 3, 4], index=0, key="bsw_k")
        sep = st.slider("Group Separation", 0.0, 3.0, 1.0, 0.1, key="bsw_sep")
    np.random.seed(42)
    kg = int(k)
    colors = px.colors.qualitative.Plotly[:kg]
    means = np.linspace(-sep * (kg - 1) / 2, sep * (kg - 1) / 2, kg)
    fig = go.Figure()
    for i in range(kg):
        d = np.random.normal(means[i], 1, n)
        pos = _swarm_positions(d)
        fig.add_trace(
            go.Scatter(
                x=pos[:, 0] + i,
                y=pos[:, 1],
                mode="markers",
                marker=dict(color=colors[i], size=5, opacity=0.7),
                name=f"Group {i+1}",
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(kg)),
            ticktext=[f"Group {i+1}" for i in range(kg)],
        ),
        yaxis_title="Value",
        hovermode="closest",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each dot = one observation\n"
                "- No overlap = exact value\n"
                "- Density = vertical stacking\n"
                "- Curved edge = distribution"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Small to moderate n\n"
                "- Show every data point\n"
                "- Avoid jitter ambiguity\n"
                "- Publication-ready plots"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Independent t-test\n"
                "- Mann-Whitney U\n"
                "- Welch's t-test\n"
                "- Permutation test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Cluttered with n > 200. "
                "Use violin or boxplot for "
                "larger samples."
            )


def polar_density_widget():
    st.markdown("## Polar Density Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="pold_n")
        modes = st.selectbox(
            "Number of Modes",
            ["1 (Uniform)", "2 (Bimodal)", "3 (Trimodal)"],
            key="pold_m",
        )
        bw = st.slider("Bandwidth", 0.1, 1.0, 0.3, 0.05, key="pold_bw")
    np.random.seed(42)
    if modes == "1 (Uniform)":
        theta = np.random.uniform(0, 2 * np.pi, n)
    elif modes == "2 (Bimodal)":
        theta = np.concatenate(
            [np.random.vonmises(0, 2, n // 2), np.random.vonmises(np.pi, 2, n // 2)]
        )
    else:
        theta = np.concatenate(
            [
                np.random.vonmises(0, 3, n // 3),
                np.random.vonmises(2 * np.pi / 3, 3, n // 3),
                np.random.vonmises(4 * np.pi / 3, 3, n // 3),
            ]
        )
    theta = theta % (2 * np.pi)
    kde_x = np.linspace(0, 2 * np.pi, 200)
    kde = stats.gaussian_kde(theta, bw_method=bw)
    kde_y = kde(kde_x)
    kde_y = kde_y / max(kde_y) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=kde_y,
            theta=np.degrees(kde_x),
            mode="lines",
            line=dict(color="#4C78A8", width=3),
            fill="toself",
            name="Density",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=np.full(n, 5),
            theta=np.degrees(theta),
            mode="markers",
            marker=dict(color="#E45756", size=2, opacity=0.3),
            name="Data",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=30, b=40),
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 90, 180, 270],
                ticktext=["0°", "90°", "180°", "270°"],
            )
        ),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Angle = direction\n"
                "- Radius = density\n"
                "- Peaks = preferred direction\n"
                "- Troughs = avoided direction"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Circular / directional data\n"
                "- Wind direction analysis\n"
                "- Seasonal pattern data\n"
                "- Animal movement bearings"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Rayleigh test\n"
                "- V-test\n"
                "- Watson-Williams test\n"
                "- Circular ANOVA"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Circular data wraps at 0°/360° "
                "— linear KDE gives wrong "
                "density at boundary. Use "
                "von Mises-based KDE."
            )


def pdf_plot_widget():
    st.markdown("## Probability Density Function Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        dist = st.selectbox(
            "Distribution",
            [
                "Normal",
                "t (Student)",
                "F",
                "Chi-square",
                "Exponential",
                "Beta",
                "Gamma",
            ],
            key="pdf_dist",
        )
        n_range = st.slider("x Range (—n to +n)", 1, 10, 4, key="pdf_n")
    np.random.seed(42)
    x = np.linspace(-n_range, n_range, 500)
    x_pos = np.linspace(0.001, 2 * n_range, 500)
    if dist == "Normal":
        mu = st.slider("Mean", -3.0, 3.0, 0.0, 0.1, key="pdf_mu")
        sigma = st.slider("Std Dev", 0.1, 3.0, 1.0, 0.1, key="pdf_sigma")
        y = stats.norm.pdf(x, mu, sigma)
    elif dist == "t (Student)":
        df = st.slider("Degrees of Freedom", 1, 50, 10, key="pdf_t_df")
        y = stats.t.pdf(x, df)
    elif dist == "F":
        df1 = st.slider("DF Numerator", 1, 30, 5, key="pdf_f1")
        df2 = st.slider("DF Denominator", 1, 50, 20, key="pdf_f2")
        y = stats.f.pdf(x_pos, df1, df2)
        x = x_pos
    elif dist == "Chi-square":
        df = st.slider("Degrees of Freedom", 1, 30, 5, key="pdf_cdf")
        y = stats.chi2.pdf(x_pos, df)
        x = x_pos
    elif dist == "Exponential":
        rate = st.slider("Rate (lambda)", 0.1, 5.0, 1.0, 0.1, key="pdf_exp")
        y = stats.expon.pdf(x_pos, scale=1 / rate)
        x = x_pos
    elif dist == "Beta":
        a = st.slider("Alpha", 0.1, 10.0, 2.0, 0.1, key="pdf_ba")
        b = st.slider("Beta", 0.1, 10.0, 2.0, 0.1, key="pdf_bb")
        y = stats.beta.pdf(np.linspace(0.001, 0.999, 500), a, b)
        x = np.linspace(0.001, 0.999, 500)
    else:
        shape = st.slider("Shape (k)", 0.1, 5.0, 2.0, 0.1, key="pdf_gk")
        scale = st.slider("Scale (theta)", 0.1, 5.0, 1.0, 0.1, key="pdf_gs")
        y = stats.gamma.pdf(x_pos, a=shape, scale=scale)
        x = x_pos
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#4C78A8", width=3),
            name=dist,
            hovertemplate="x=%{x:.2f}<br>PDF=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="x",
        yaxis_title="Density",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Area under curve = 1\n"
                "- Height = relative likelihood\n"
                "- Peak = most probable region\n"
                "- Spread = parameter variance"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Understand distribution shapes\n"
                "- Compare theoretical PDFs\n"
                "- Learn parameter effects\n"
                "- Check data-model fit"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- GOF (KS, AD, CVM tests)\n"
                "- Q-Q plot assessment\n"
                "- Parameter estimation\n"
                "- MLE / Bayesian inference"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "PDF > 1 is normal — PDF is "
                "a density, not a probability. "
                "Only the integral over a "
                "range gives probability."
            )


def pareto_widget():
    st.markdown("## Pareto Chart")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 1000, 200, key="pareto_n")
        n_cat = st.selectbox(
            "Number of Categories", [5, 6, 7, 8, 10], index=2, key="pareto_k"
        )
    np.random.seed(42)
    k = int(n_cat)
    probs = np.random.dirichlet(np.ones(k) * 0.5)
    counts = np.random.multinomial(n, probs)
    cats = [f"Category {i+1}" for i in range(k)]
    order = np.argsort(-counts)
    cats_sorted = [cats[i] for i in order]
    counts_sorted = counts[order]
    cum_pct = np.cumsum(counts_sorted) / n * 100
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cats_sorted,
            y=counts_sorted,
            name="Frequency",
            marker=dict(color="#4C78A8"),
            hovertemplate="%{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cats_sorted,
            y=cum_pct,
            mode="lines+markers",
            name="Cumulative %",
            yaxis="y2",
            line=dict(color="#E45756", width=3),
            marker=dict(size=8),
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=80, line=dict(color="gray", dash="dash"), opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Category",
        yaxis_title="Frequency",
        yaxis2=dict(
            overlaying="y",
            side="right",
            range=[0, 110],
            title="Cumulative %",
            tickformat=".0f",
        ),
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Bars sorted descending\n"
                "- Line = cumulative percent\n"
                "- 80% line = Pareto principle\n"
                "- Top few = majority of effect"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Quality control analysis\n"
                "- Identify vital few vs many\n"
                "- Prioritize improvements\n"
                "- Resource allocation"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Chi-square GOF\n"
                "- Lorenz curve\n"
                "- Gini coefficient\n"
                "- Concentration indices"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "80/20 is a guideline not a "
                "law. Actual split depends on "
                "your data. Always check the "
                "actual cumulative curve."
            )


def dot_plot_widget():
    st.markdown("## Dot Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="dot_n")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Bimodal"], key="dot_dist"
        )
        jitter = st.slider("Jitter Amount", 0.0, 0.5, 0.2, 0.05, key="dot_jitter")
    np.random.seed(42)
    if dist == "Normal":
        d = np.random.normal(0, 1, n)
    elif dist == "Skewed":
        d = np.random.gamma(2, 1, n)
    else:
        d = np.concatenate(
            [np.random.normal(-1.5, 0.6, n // 2), np.random.normal(1.5, 0.6, n // 2)]
        )
    y_jitter = np.random.uniform(-jitter, jitter, n)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d,
            y=y_jitter,
            mode="markers",
            marker=dict(color="#4C78A8", size=5, opacity=0.6),
            name="Data",
            hovertemplate="Value=%{x:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Value",
        yaxis=dict(visible=False),
        hovermode="x",
        showlegend=False,
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each dot = one observation\n"
                "- Horizontal spread = distribution\n"
                "- Stacked dots = multiple values\n"
                "- Gaps = empty regions"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Small datasets (n < 100)\n"
                "- Show exact distribution\n"
                "- Identify clusters and gaps\n"
                "- Complement to boxplot"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- One-sample t-test\n"
                "- Wilcoxon signed-rank\n"
                "- Sign test\n"
                "- Shapiro-Wilk normality"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Random jitter can mislead. "
                "Use fixed seed for "
                "reproducibility. Beeswarm "
                "is more accurate."
            )


def time_series_widget():
    st.markdown("## Time Series Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Time Points", 20, 300, 100, key="ts_n")
        trend = st.selectbox(
            "Trend",
            ["None (stationary)", "Linear", "Seasonal", "Trend + Seasonal"],
            key="ts_trend",
        )
        noise = st.slider("Noise Level", 0.0, 3.0, 0.5, 0.1, key="ts_noise")
    np.random.seed(42)
    t = np.arange(n)
    y = np.random.normal(0, noise, n)
    if trend == "Linear":
        y += 0.05 * t
    elif trend == "Seasonal":
        y += 2 * np.sin(2 * np.pi * t / 12)
    elif trend == "Trend + Seasonal":
        y += 0.03 * t + 2 * np.sin(2 * np.pi * t / 12)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines+markers",
            line=dict(color="#4C78A8", width=2),
            marker=dict(size=3),
            name="Series",
            hovertemplate="t=%{x}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- X-axis = time order\n"
                "- Points connected by line\n"
                "- Trend = long-term direction\n"
                "- Seasonality = repeating pattern"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Longitudinal / repeated data\n"
                "- Trend analysis / forecasting\n"
                "- Seasonal pattern detection\n"
                "- Intervention effect analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Augmented Dickey-Fuller\n"
                "- Ljung-Box test\n"
                "- Durbin-Watson test\n"
                "- Granger causality"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Don't connect points across "
                "missing time gaps. Always "
                "check for autocorrelation "
                "before modeling."
            )


def pie_chart_widget():
    st.markdown("## Pie Chart")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        k = st.selectbox("Number of Slices", [3, 4, 5, 6], index=1, key="pie_k")
        pattern = st.selectbox(
            "Distribution", ["Equal", "Dominant", "Gradual"], key="pie_pattern"
        )
        explode = st.toggle("Explode Largest Slice", False, key="pie_explode")
    np.random.seed(42)
    kg = int(k)
    if pattern == "Equal":
        vals = np.ones(kg)
    elif pattern == "Dominant":
        vals = np.array([5] + [1] * (kg - 1))
    else:
        vals = np.arange(kg, 0, -1) + 1
    labels = [f"Category {i+1}" for i in range(kg)]
    colors = px.colors.qualitative.Plotly[:kg]
    pull = [0.1 if explode and i == 0 else 0 for i in range(kg)]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=vals,
                pull=pull,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{value} (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10)
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each slice = proportion\n"
                "- Area encodes percentage\n"
                "- Full circle = 100%\n"
                "- Best for few categories"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Simple part-to-whole\n"
                "- Few categories (2-5)\n"
                "- Rough visual comparison\n"
                "- Non-technical audience"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Chi-square GOF\n"
                "- Binomial test\n"
                "- Proportion tests\n"
                "- Confidence intervals"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                ">5 slices is hard to read. "
                "3D pies distort proportions. "
                "Bar charts or treemaps "
                "are often better."
            )


def area_graph_widget():
    st.markdown("## Area Graph")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Time Points", 10, 100, 40, key="area_n")
        k = st.selectbox("Number of Series", [2, 3, 4], index=1, key="area_k")
        style = st.selectbox("Style", ["Stacked", "Overlaid"], key="area_style")
    np.random.seed(42)
    kg = int(k)
    t = np.arange(n)
    colors = px.colors.qualitative.Plotly[:kg]
    fig = go.Figure()
    if style == "Stacked":
        bases = np.zeros(n)
        for i in range(kg):
            y = np.abs(np.random.normal(10 * (i + 1), 2, n) + 0.1 * t * (i + 1))
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y + bases,
                    mode="lines",
                    line=dict(width=2),
                    name=f"Series {i+1}",
                    stackgroup="one",
                    fillcolor=colors[i],
                )
            )
            bases += y
    else:
        for i in range(kg):
            y = np.abs(np.random.normal(10 * (i + 1), 2, n) + 0.05 * t * (i + 1))
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(width=2),
                    name=f"Series {i+1}",
                    fillcolor=colors[i],
                    opacity=0.3,
                )
            )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Filled area = magnitude\n"
                "- Stacked = total + composition\n"
                "- Overlaid = compare shapes\n"
                "- Slope = rate of change"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Magnitude over time\n"
                "- Compare series contributions\n"
                "- Cumulative trends\n"
                "- Composition changes"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Time series decomposition\n"
                "- Change point detection\n"
                "- Trend analysis\n"
                "- Intervention analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                ">4 stacked series becomes "
                "unreadable. Overlaid areas "
                "need transparency."
            )


def contour_widget():
    st.markdown("## Contour Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 100, 1000, 300, key="cont_n")
        bw = st.slider("Bandwidth", 0.1, 1.0, 0.3, 0.05, key="cont_bw")
        dist = st.selectbox(
            "Distribution",
            ["Bivariate Normal", "Two Clusters", "Donut"],
            key="cont_dist",
        )
    np.random.seed(42)
    if dist == "Bivariate Normal":
        x = np.random.normal(0, 1, n)
        y = np.random.normal(0, 1, n)
    elif dist == "Two Clusters":
        x = np.concatenate(
            [np.random.normal(-2, 0.8, n // 2), np.random.normal(2, 0.8, n // 2)]
        )
        y = np.concatenate(
            [np.random.normal(0, 0.8, n // 2), np.random.normal(0, 0.8, n // 2)]
        )
    else:
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.normal(2, 0.4, n)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
    kde = stats.gaussian_kde(np.vstack([x, y]), bw_method=bw)
    xi, yi = np.meshgrid(
        np.linspace(min(x) - 1, max(x) + 1, 50), np.linspace(min(y) - 1, max(y) + 1, 50)
    )
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xi[0],
            y=yi[:, 0],
            z=zi,
            colorscale="Viridis",
            contours=dict(coloring="heatmap"),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>density=%{z:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="white", size=2, opacity=0.3),
            name="Data",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Lines = constant density\n"
                "- Closer lines = steeper gradient\n"
                "- Peaks = dense regions\n"
                "- Color = density intensity"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Visualize 2D distribution\n"
                "- Identify density peaks\n"
                "- Replace scatter for large n\n"
                "- Topographic data display"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Bivariate normality test\n"
                "- Hotelling's T-squared\n"
                "- Multivariate outlier test\n"
                "- Kernel density estimation"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Bandwidth changes contours "
                "dramatically. Too low = spikes, "
                "too high = oversmoothed. "
                "Use cross-validation."
            )


def stacked_bar_widget():
    st.markdown("## Stacked Bar Chart")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_groups = st.selectbox("Number of Groups", [2, 3, 4], index=1, key="stbar_ng")
        n_cat = st.selectbox("Number of Categories", [2, 3, 4], index=2, key="stbar_nc")
        style = st.selectbox("Style", ["Stacked", "100% Stacked"], key="stbar_style")
    np.random.seed(42)
    ng = int(n_groups)
    nc = int(n_cat)
    data = np.random.randint(5, 30, (ng, nc))
    groups = [f"Group {i+1}" for i in range(ng)]
    cats = [f"Cat {i+1}" for i in range(nc)]
    colors = px.colors.qualitative.Plotly[:nc]
    fig = go.Figure()
    if style == "Stacked":
        for i in range(nc):
            fig.add_trace(
                go.Bar(
                    name=cats[i],
                    x=groups,
                    y=data[:, i],
                    marker_color=colors[i],
                    hovertemplate="%{y}<extra></extra>",
                )
            )
    else:
        totals = data.sum(axis=1)
        pcts = data / totals[:, None] * 100
        for i in range(nc):
            fig.add_trace(
                go.Bar(
                    name=cats[i],
                    x=groups,
                    y=pcts[:, i],
                    marker_color=colors[i],
                    hovertemplate="%{y:.1f}%<extra></extra>",
                )
            )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Group",
        yaxis_title="Count" if style == "Stacked" else "Percentage",
        barmode="stack",
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Total bar = group total\n"
                "- Segment = category contribution\n"
                "- 100% = proportions not counts\n"
                "- Compare composition across groups"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Show group composition\n"
                "- Compare totals and parts\n"
                "- Survey response breakdowns\n"
                "- Budget allocation view"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Chi-square independence\n"
                "- Fisher's exact test\n"
                "- G-test\n"
                "- Correspondence analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Non-100% bars are hard to "
                "compare across different "
                "totals. Normalize to 100% "
                "for composition comparison."
            )


def pop_pyramid_widget():
    st.markdown("## Population Pyramid")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Total Population (K)", 10, 1000, 100, key="pyram_n")
        skew = st.slider("Age Skew", -2.0, 2.0, 0.0, 0.1, key="pyram_skew")
        dev = st.slider("Sex Deviation", -1.0, 1.0, 0.0, 0.1, key="pyram_dev")
    np.random.seed(42)
    age_groups = [
        "0-4",
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ]
    k = len(age_groups)
    ages = np.arange(k)
    base = np.exp(-0.1 * ages + skew * 0.05 * ages)
    base = base / base.sum() * n
    male = base * (0.5 + dev * 0.1) + np.random.uniform(-1, 1, k) * 0.5
    female = base * (0.5 - dev * 0.1) + np.random.uniform(-1, 1, k) * 0.5
    male = np.maximum(male, 0)
    female = np.maximum(female, 0)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=-male,
            y=age_groups,
            orientation="h",
            name="Male",
            marker_color="#4C78A8",
            hovertemplate="%{x:.1f}K<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=female,
            y=age_groups,
            orientation="h",
            name="Female",
            marker_color="#E45756",
            hovertemplate="%{x:.1f}K<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Population (K)",
        yaxis_title="Age Group",
        barmode="overlay",
        hovermode="y unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Left = male, Right = female\n"
                "- Wide base = high birth rate\n"
                "- Narrow top = lower life exp\n"
                "- Bulges = baby boom cohorts"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Demographic analysis\n"
                "- Population structure\n"
                "- Age-sex distribution\n"
                "- Policy and planning"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Chi-square independence\n"
                "- Age standardization\n"
                "- Dependency ratio\n"
                "- Life table analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Different scales on left/right "
                "axes mislead. Always use "
                "the same scale on both "
                "sides for fair comparison."
            )


def growth_curve_widget():
    st.markdown("## Growth Curve Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Time Points", 10, 100, 30, key="gc_n")
        noise = st.slider("Noise Level", 0.0, 2.0, 0.3, 0.05, key="gc_noise")
        model = st.selectbox(
            "Growth Model", ["Logistic", "Gompertz", "Exponential"], key="gc_model"
        )
    np.random.seed(42)
    t = np.linspace(0, 20, n)
    if model == "Logistic":
        L, k, t0 = 10, 0.5, 10
        y_true = L / (1 + np.exp(-k * (t - t0)))
    elif model == "Gompertz":
        L, k, t0 = 10, 0.3, 5
        y_true = L * np.exp(-np.exp(-k * (t - t0)))
    else:
        r, y0 = 0.2, 0.5
        y_true = y0 * np.exp(r * t)
    y = y_true + np.random.normal(0, noise, n)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="markers",
            marker=dict(color="#4C78A8", size=5, opacity=0.6),
            name="Observed",
            hovertemplate="t=%{x:.1f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_true,
            mode="lines",
            line=dict(color="#E45756", width=3),
            name="True Growth",
            hovertemplate="t=%{x:.1f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Size / Population",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- S-curve = logistic growth\n"
                "- Asymptote = carrying capacity\n"
                "- Steepest = max growth rate\n"
                "- Lag → log → stationary phases"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Population growth modeling\n"
                "- Epidemic curve analysis\n"
                "- Learning curve analysis\n"
                "- Biological growth processes"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Non-linear regression F-test\n"
                "- Model comparison (AIC/BIC)\n"
                "- Residual diagnostics\n"
                "- Bootstrap parameter CIs"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Extrapolating beyond observed "
                "data is risky. Asymptote "
                "depends strongly on model "
                "choice."
            )


def forest_plot_widget():
    st.markdown("## Forest Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_studies = st.slider("Number of Studies", 5, 30, 10, key="forest_ns")
        eff = st.slider(
            "Overall Effect (log OR)", -1.0, 1.0, 0.3, 0.05, key="forest_eff"
        )
        hetero = st.slider("Heterogeneity", 0.0, 1.0, 0.3, 0.05, key="forest_hetero")
    np.random.seed(42)
    k = int(n_studies)
    ses = np.random.uniform(0.1, 0.5, k)
    log_ors = np.random.normal(eff, hetero, k)
    lower = log_ors - 1.96 * ses
    upper = log_ors + 1.96 * ses
    ors = np.exp(log_ors)
    or_lower = np.exp(lower)
    or_upper = np.exp(upper)
    studies = [f"Study {i+1}" for i in range(k)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ors,
            y=studies,
            mode="markers",
            marker=dict(color="#4C78A8", size=10),
            error_x=dict(
                type="data",
                symmetric=False,
                array=or_upper - ors,
                arrayminus=ors - or_lower,
                width=5,
                color="#4C78A8",
            ),
            hovertemplate="%{y}<br>OR=%{x:.2f} "
            "[%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>",
            customdata=np.column_stack([or_lower, or_upper]),
        )
    )
    fig.add_vline(x=1, line=dict(color="gray", dash="dash"), opacity=0.5)
    fig.add_vline(
        x=np.exp(eff), line=dict(color="#E45756", width=3, dash="dot"), opacity=0.7
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(30 * k, 200),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Odds Ratio (log scale)",
        xaxis=dict(type="log"),
        yaxis_title="",
        hovermode="y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each row = one study\n"
                "- Dot = effect size (OR)\n"
                "- Line = 95% confidence interval\n"
                "- Red = pooled estimate (meta)"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Meta-analysis reporting\n"
                "- Systematic review synthesis\n"
                "- Compare across multiple studies\n"
                "- Identify heterogeneous results"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Cochran's Q test\n"
                "- Higgins I² statistic\n"
                "- Egger's test (pub bias)\n"
                "- Meta-regression"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Pooled estimate (diamond) is "
                "the overall effect, NOT "
                "the average of individual "
                "study ORs."
            )


# =========================
# SURVIVAL ANALYSIS HELPERS
# =========================


def _gen_surv_data(n, hr, cens_frac):
    np.random.seed(42)
    t_ctrl = np.random.exponential(12, n)
    t_trt = np.random.exponential(12 / hr, n)
    times = np.concatenate([t_ctrl, t_trt])
    groups = np.array([0] * n + [1] * n)
    cens = np.random.uniform(0, 25, 2 * n)
    obs = np.minimum(times, cens)
    event = (times <= cens).astype(int)
    return obs, event, groups


def _km(t, e):
    df = pd.DataFrame({"t": t, "e": e}).sort_values("t")
    ts = sorted(df["t"].unique())
    s = 1.0
    ot, os = [0], [1.0]
    for ti in ts:
        nr = (df["t"] >= ti).sum()
        ne = df.loc[df["t"] == ti, "e"].sum()
        if nr > 0:
            s *= 1 - ne / nr
        ot.extend([ti, ti])
        os.extend([os[-1], s])
    return np.array(ot), np.array(os)


def _na(t, e):
    df = pd.DataFrame({"t": t, "e": e}).sort_values("t")
    h = 0.0
    ot, oh = [0], [0.0]
    for ti in sorted(df["t"].unique()):
        nr = (df["t"] >= ti).sum()
        ne = df.loc[df["t"] == ti, "e"].sum()
        if nr > 0:
            h += ne / nr
        ot.extend([ti, ti])
        oh.extend([oh[-1], h])
    return np.array(ot), np.array(oh)


def kaplan_meier_widget():
    st.markdown("## Kaplan-Meier Curve")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Patients per Group", 20, 200, 50, key="km_n")
        hr = st.slider("Hazard Ratio (Trt/Control)", 0.2, 1.5, 0.5, 0.05, key="km_hr")
        cens = st.slider("Censoring Rate", 0.0, 0.5, 0.2, 0.05, key="km_cens")
        show_ci = st.toggle("Show 95% CI", True, key="km_ci")
    t, e, g = _gen_surv_data(n, hr, cens)
    fig = go.Figure()
    for grp, name, color in [(0, "Control", "#4C78A8"), (1, "Treatment", "#E45756")]:
        mask = g == grp
        tt, ss = _km(t[mask], e[mask])
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=ss,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=name,
                hovertemplate="Time=%{x:.1f}<br>Survival=%{y:.3f}<extra></extra>",
            )
        )
        cens_t = t[mask & (e == 0)]
        if len(cens_t) > 0:
            cens_s = []
            for ct in cens_t:
                idx = np.searchsorted(tt, ct, side="right") - 1
                cens_s.append(ss[max(0, idx)])
            fig.add_trace(
                go.Scatter(
                    x=cens_t,
                    y=cens_s,
                    mode="markers",
                    marker=dict(color=color, symbol="line-ns", size=8),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Survival Probability",
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Step down = event occurred\n"
                "- Tick marks = censored\n"
                "- Lower curve = worse survival\n"
                "- Gap between groups = treatment effect"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Time-to-event analysis\n"
                "- Clinical trial comparison\n"
                "- Estimate median survival\n"
                "- Treatment efficacy assessment"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Log-rank test\n"
                "- Wilcoxon-Gehan test\n"
                "- Peto-Peto test\n"
                "- Cox proportional hazards"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "KM curves beyond last event "
                "are unstable. Always show "
                "number at risk below "
                "the x-axis."
            )


def nelson_aalen_widget():
    st.markdown("## Nelson-Aalen Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Patients per Group", 20, 200, 50, key="na_n")
        hr = st.slider("Hazard Ratio", 0.2, 1.5, 0.5, 0.05, key="na_hr")
        cens = st.slider("Censoring Rate", 0.0, 0.5, 0.2, 0.05, key="na_cens")
    t, e, g = _gen_surv_data(n, hr, cens)
    fig = go.Figure()
    for grp, name, color in [(0, "Control", "#4C78A8"), (1, "Treatment", "#E45756")]:
        mask = g == grp
        tt, hh = _na(t[mask], e[mask])
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=hh,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=name,
                hovertemplate="Time=%{x:.1f}<br>Cum Hazard=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Cumulative Hazard",
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Step up = event increment\n"
                "- Steeper = higher hazard\n"
                "- Slope = instantaneous hazard\n"
                "- Gap = constant HR assumption"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Non-parametric hazard estimate\n"
                "- Compare hazard between groups\n"
                "- Check proportional hazards\n"
                "- Complement to KM curve"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Log-rank test\n"
                "- Cox PH model\n"
                "- Schoenfeld residuals\n"
                "- Cumulative hazard comparison"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Nelson-Aalen is a cumulative "
                "estimate, not the hazard "
                "rate itself. Slope gives "
                "hazard, not the absolute "
                "value."
            )


def hazard_function_widget():
    st.markdown("## Hazard Function Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="hf_n")
        base_hazard = st.selectbox(
            "Baseline Shape",
            ["Constant", "Increasing", "Decreasing", "Bathtub"],
            key="hf_shape",
        )
        bw = st.slider("Smoothing Bandwidth", 1.0, 10.0, 3.0, 0.5, key="hf_bw")
    np.random.seed(42)
    t = np.linspace(0, 50, 500)
    if base_hazard == "Constant":
        h_true = np.full_like(t, 0.05)
    elif base_hazard == "Increasing":
        h_true = 0.01 + 0.003 * t
    elif base_hazard == "Decreasing":
        h_true = 0.08 - 0.001 * t
    else:
        h_true = 0.01 + 0.002 * np.abs(t - 25)
    h_true = np.maximum(h_true, 0.001)
    surv_true = np.exp(-np.cumsum(h_true) * (t[1] - t[0]))
    event_times = []
    for i in range(n):
        u = np.random.uniform()
        idx = np.searchsorted(surv_true, u, side="left")
        if idx < len(t):
            event_times.append(t[idx])
        else:
            event_times.append(50)
    event_times = np.array(event_times)
    k = stats.gaussian_kde(event_times, bw_method=bw / 50)
    h_est = k(t) / np.maximum(np.array([(event_times >= ti).mean() for ti in t]), 0.001)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=h_true,
            mode="lines",
            line=dict(color="gray", width=2, dash="dot"),
            name="True Hazard",
            hovertemplate="Time=%{x:.1f}<br>Hazard=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=h_est,
            mode="lines",
            line=dict(color="#4C78A8", width=2.5),
            name="Estimated Hazard",
            hovertemplate="Time=%{x:.1f}<br>Hazard=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Hazard Rate",
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Hazard = instantaneous risk\n"
                "- Increasing = wearing out\n"
                "- Decreasing = early failures\n"
                "- Bathtub = both phases"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Model time-to-failure\n"
                "- Understand risk over time\n"
                "- Compare population hazard shapes\n"
                "- Reliability engineering"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Weibull distribution fit\n"
                "- Exponential GOF test\n"
                "- Cox-Snell residuals\n"
                "- Hazard shape tests"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Hazard is not a probability — "
                "it can exceed 1. It is a "
                "rate (events per time unit)."
                "Do not confuse with risk."
            )


def cumulative_hazard_widget():
    st.markdown("## Cumulative Hazard Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Patients per Group", 20, 200, 50, key="ch_n")
        hr = st.slider("Hazard Ratio (Trt/Control)", 0.2, 1.5, 0.5, 0.05, key="ch_hr")
        cens = st.slider("Censoring Rate", 0.0, 0.5, 0.2, 0.05, key="ch_cens")
        log_scale = st.toggle("Log Scale", False, key="ch_log")
    t, e, g = _gen_surv_data(n, hr, cens)
    fig = go.Figure()
    for grp, name, color in [(0, "Control", "#4C78A8"), (1, "Treatment", "#E45756")]:
        mask = g == grp
        tt, hh = _na(t[mask], e[mask])
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=hh,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=name,
                hovertemplate="Time=%{x:.1f}<br>Cum H=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Cumulative Hazard" + (" (log)" if log_scale else ""),
        yaxis_type="log" if log_scale else "linear",
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Log scale → parallel = PH\n"
                "- Upward curve = increasing hazard\n"
                "- Downward curve = decreasing\n"
                "- Straight line = constant hazard"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Check proportional hazards\n"
                "- Estimate cumulative risk\n"
                "- Model diagnostic tool\n"
                "- Complement to KM curves"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Cox PH model\n"
                "- Schoenfeld residuals\n"
                "- Log-cumulative hazard plot\n"
                "- PH assumption check"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Log-cumulative hazard lines "
                "must be parallel for PH. "
                "Crossing lines = non-PH. "
                "Use stratified Cox or AFT."
            )


def cox_ph_widget():
    st.markdown("## Cox PH Effect Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_cov = st.selectbox("Number of Covariates", [4, 6, 8], index=0, key="cox_n")
        n = st.slider("Sample Size", 100, 1000, 300, key="cox_sample")
        hr_range = st.slider("HR Range", 1.0, 5.0, 3.0, 0.5, key="cox_hr")
    np.random.seed(42)
    k = int(n_cov)
    cov_names = [
        f"Treatment",
        "Age (10yr)",
        "Biomarker",
        "Comorbidity",
        "BMI (>30)",
        "Smoking",
        "Stage III+",
        "Surgery",
    ][:k]
    log_hrs = np.random.uniform(-np.log(hr_range), np.log(hr_range), k)
    ses = np.random.uniform(0.15, 0.4, k)
    hr = np.exp(log_hrs)
    lower = np.exp(log_hrs - 1.96 * ses)
    upper = np.exp(log_hrs + 1.96 * ses)
    p_vals = np.random.uniform(0.001, 0.2, k)
    sig = p_vals < 0.05
    order = np.argsort(hr)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hr[order],
            y=[cov_names[i] for i in order],
            mode="markers",
            marker=dict(
                color=["#E45756" if s else "#4C78A8" for s in sig[order]], size=12
            ),
            error_x=dict(
                type="data",
                symmetric=False,
                array=upper[order] - hr[order],
                arrayminus=hr[order] - lower[order],
                width=5,
                color="gray",
            ),
            hovertemplate="%{y}<br>HR=%{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=1, line=dict(color="gray", dash="dash"), opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        height=max(40 * k, 200),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Hazard Ratio (log scale)",
        xaxis=dict(type="log"),
        yaxis_title="",
        hovermode="y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Dot = estimated HR\n"
                "- Line = 95% CI\n"
                "- HR > 1 = increased risk\n"
                "- Red = significant (p < .05)"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Multivariable survival analysis\n"
                "- Adjust for confounders\n"
                "- Identify risk factors\n"
                "- Compare effect sizes"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Wald test per covariate\n"
                "- Likelihood ratio test\n"
                "- Proportional hazards test\n"
                "- Schoenfeld residual test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "HR > 1 = higher hazard (worse). "
                "PH assumption must hold. "
                "HR is not risk ratio — "
                "it is relative hazard rate."
            )


def surv_heatmap_widget():
    st.markdown("## Survival Probability Heatmap")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_time = st.slider("Time Points", 10, 50, 20, key="shm_nt")
        n_cov = st.slider("Covariate Values", 10, 50, 20, key="shm_nc")
        base_haz = st.slider(
            "Baseline Hazard Rate", 0.01, 0.2, 0.05, 0.01, key="shm_haz"
        )
        hr_effect = st.slider("HR per Covariate Unit", 0.5, 3.0, 1.5, 0.1, key="shm_hr")
    np.random.seed(42)
    times = np.linspace(0, 50, n_time)
    cov_vals = np.linspace(-2, 2, n_cov)
    S = np.zeros((n_cov, n_time))
    for i, c in enumerate(cov_vals):
        hr = hr_effect**c
        S[i, :] = np.exp(-base_haz * hr * times)
    fig = go.Figure(
        data=go.Heatmap(
            x=times,
            y=np.round(cov_vals, 1),
            z=S,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Survival"),
            hovertemplate="Time=%{x:.1f}<br>Covariate=%{y}<br>Survival=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Covariate Value",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Yellow = high survival\n"
                "- Purple = low survival\n"
                "- Row = covariate effect on survival\n"
                "- Column = time effect"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Visualize covariate effect\n"
                "- Model-predicted survival\n"
                "- Identify risk strata\n"
                "- Treatment effect surface"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Cox PH model predictions\n"
                "- Parametric survival models\n"
                "- Time-dependent covariates\n"
                "- Interaction testing"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Assumes proportional hazards "
                "and exponential baseline. "
                "Real survival may have "
                "time-varying effects."
            )


# =========================
# META-ANALYSIS HELPERS
# =========================


def _gen_meta_data(k, eff, het):
    np.random.seed(42)
    se = np.random.uniform(0.05, 0.5, k)
    y = np.random.normal(eff, np.sqrt(se**2 + het**2))
    return y, se


def _gen_meta_bias(k, eff, het, bias):
    np.random.seed(42)
    se = np.random.uniform(0.05, 0.5, k)
    y = np.random.normal(eff, np.sqrt(se**2 + het**2))
    if bias > 0:
        for i in range(k):
            if se[i] > 0.15:
                p = 2 * (1 - stats.norm.cdf(abs(y[i] / se[i])))
                if p > 0.05 and np.random.random() < bias:
                    y[i] = np.nan
                    se[i] = np.nan
    m = ~np.isnan(y)
    return y[m], se[m]


def funnel_widget():
    st.markdown("## Funnel Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Studies", 10, 60, 25, key="funnel_n")
        bias = st.slider("Publication Bias", 0.0, 1.0, 0.0, 0.1, key="funnel_bias")
        het = st.slider("Heterogeneity", 0.0, 0.5, 0.1, 0.05, key="funnel_het")
    eff, se = _gen_meta_bias(int(n), 0.3, het, bias)
    pooled = np.sum(eff / se**2) / np.sum(1 / se**2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eff,
            y=se,
            mode="markers",
            marker=dict(color="#4C78A8", size=7),
            name="Studies",
            hovertemplate="Effect=%{x:.3f}<br>SE=%{y:.4f}<extra></extra>",
        )
    )
    y_grid = np.linspace(min(se), max(se), 50)
    fig.add_trace(
        go.Scatter(
            x=pooled - 1.96 * y_grid,
            y=y_grid,
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pooled + 1.96 * y_grid,
            y=y_grid,
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.08)",
            hoverinfo="skip",
        )
    )
    fig.add_vline(x=pooled, line=dict(color="#E45756", width=2), opacity=0.7)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Effect Size",
        yaxis_title="Standard Error",
        yaxis=dict(autorange="reversed"),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Symmetric funnel = no bias\n"
                "- Missing left = suppressed negative\n"
                "- Missing right = suppressed positive\n"
                "- Gap = publication bias"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Assess publication bias\n"
                "- Meta-analysis quality check\n"
                "- Detect small-study effects\n"
                "- Sensitivity analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Egger's regression test\n"
                "- Begg's rank test\n"
                "- Trim-and-fill method\n"
                "- Selection models"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Funnel asymmetry is NOT always "
                "publication bias. It can also "
                "indicate heterogeneity or "
                "chance, especially with few studies."
            )


def galbraith_widget():
    st.markdown("## Galbraith (Radial) Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Studies", 10, 60, 25, key="gal_n")
        het = st.slider("Heterogeneity", 0.0, 0.5, 0.1, 0.05, key="gal_het")
    eff, se = _gen_meta_data(int(n), 0.3, het)
    prec = 1 / se
    z = eff / se
    pooled = np.sum(eff / se**2) / np.sum(1 / se**2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prec,
            y=z,
            mode="markers",
            marker=dict(color="#4C78A8", size=7),
            name="Studies",
            hovertemplate="Precision=%{x:.2f}<br>z=%{y:.2f}<extra></extra>",
        )
    )
    x_line = np.linspace(0, max(prec) * 1.05, 100)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=pooled * x_line,
            mode="lines",
            line=dict(color="#E45756", width=2.5),
            name=f"Pooled ({pooled:.3f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=pooled * x_line + 2,
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=pooled * x_line - 2,
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.08)",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Precision (1/SE)",
        yaxis_title="z-score (Effect/SE)",
        hovermode="closest",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Slope = pooled effect size\n"
                "- Scatter around line = heterogeneity\n"
                "- Outside gray band = outlier\n"
                "- Intercept ≈ 0 = no bias"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Identify outlier studies\n"
                "- Assess heterogeneity visually\n"
                "- Complement to funnel plot\n"
                "- Detect effect modification"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Cochran's Q test\n"
                "- I² heterogeneity statistic\n"
                "- H statistic\n"
                "- Subgroup meta-analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Radial plot uses the same data "
                "as the funnel — points on both "
                "axes are correlated. Use it "
                "with the funnel for full picture."
            )


def baujat_widget():
    st.markdown("## Baujat Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Studies", 10, 50, 20, key="baujat_n")
        het = st.slider("Heterogeneity", 0.0, 1.0, 0.3, 0.05, key="baujat_het")
        outlier = st.slider(
            "Outlier Study Effect", 0.0, 3.0, 0.0, 0.5, key="baujat_out"
        )
    eff, se = _gen_meta_data(int(n), 0.3, het)
    if outlier > 0:
        eff[-1] += outlier
    w = 1 / se**2
    pooled = np.sum(w * eff) / np.sum(w)
    q_contrib = w * (eff - pooled) ** 2
    loo_effects = np.array(
        [
            np.sum(np.delete(w, i) * np.delete(eff, i)) / np.sum(np.delete(w, i))
            for i in range(len(eff))
        ]
    )
    influence = np.abs(loo_effects - pooled)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=q_contrib,
            y=influence,
            mode="markers+text",
            marker=dict(color="#4C78A8", size=10),
            text=[str(i + 1) for i in range(len(eff))],
            textposition="top center",
            name="Studies",
            hovertemplate="Study %{text}<br>Q contrib=%{x:.2f}<br>Influence=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=np.median(influence) * 2, line=dict(color="gray", dash="dash"), opacity=0.5
    )
    fig.add_vline(
        x=np.median(q_contrib) * 2, line=dict(color="gray", dash="dash"), opacity=0.5
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Contribution to Heterogeneity (Q)",
        yaxis_title="Influence on Pooled Effect",
        hovermode="closest",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Upper-right = high influence + heterogeneity\n"
                "- Lower-right = heterogeneous but not influential\n"
                "- Upper-left = influential but not heterogeneous\n"
                "- Lower-left = well-fitting studies"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Identify influential studies\n"
                "- Detect heterogeneity sources\n"
                "- Sensitivity analysis\n"
                "- Meta-analysis diagnostics"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Leave-one-out analysis\n"
                "- DFBETAS / Cook's distance\n"
                "- Q-test for heterogeneity\n"
                "- I² statistic per subgroup"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Baujat plot is descriptive, not "
                "inferential. Cut-off lines are "
                "arbitrary — use with caution "
                "when k is small (< 10)."
            )


def loo_widget():
    st.markdown("## Leave-One-Out Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Studies", 5, 30, 15, key="loo_n")
        het = st.slider("Heterogeneity", 0.0, 0.5, 0.1, 0.05, key="loo_het")
    eff, se = _gen_meta_data(int(n), 0.3, het)
    w = 1 / se**2
    pooled_all = np.sum(w * eff) / np.sum(w)
    loo_pooled = []
    loo_lower = []
    loo_upper = []
    for i in range(len(eff)):
        w_i = np.delete(w, i)
        e_i = np.delete(eff, i)
        p = np.sum(w_i * e_i) / np.sum(w_i)
        se_p = np.sqrt(1 / np.sum(w_i))
        loo_pooled.append(p)
        loo_lower.append(p - 1.96 * se_p)
        loo_upper.append(p + 1.96 * se_p)
    loo_pooled = np.array(loo_pooled)
    loo_lower = np.array(loo_lower)
    loo_upper = np.array(loo_upper)
    study_labels = [f"Omit {i+1}" for i in range(len(eff))]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=loo_pooled,
            y=study_labels,
            mode="markers",
            marker=dict(color="#4C78A8", size=10),
            error_x=dict(
                type="data",
                symmetric=False,
                array=loo_upper - loo_pooled,
                arrayminus=loo_pooled - loo_lower,
                width=5,
                color="#4C78A8",
            ),
            name="Leave-One-Out",
            hovertemplate="%{y}<br>Pooled=%{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=pooled_all, line=dict(color="#E45756", width=3, dash="dot"), opacity=0.7
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(30 * len(eff), 200),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Pooled Effect Size",
        yaxis_title="",
        hovermode="y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Dot = pooled effect without that study\n"
                "- Line = 95% CI without that study\n"
                "- Red line = overall pooled effect\n"
                "- Dot far from red = influential study"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Identify influential studies\n"
                "- Sensitivity analysis\n"
                "- Check result robustness\n"
                "- Meta-analysis diagnostics"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Baujat plot\n"
                "- DFBETAS statistic\n"
                "- Cook's distance\n"
                "- Cumulative meta-analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "If removing a study changes "
                "the conclusion, that study "
                "drives the result. Report "
                "both with and without "
                "influential studies."
            )


# =========================
# RAINCLOUD PLOT
# =========================


def raincloud_widget():
    st.markdown("## Raincloud Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 20, 200, 80, key="rc_n")
        n_groups = st.selectbox("Number of Groups", [2, 3, 4], index=1, key="rc_ng")
        effect = st.slider("Group Separation", 0.0, 3.0, 1.0, 0.1, key="rc_eff")
        show_box = st.toggle("Show Boxplot", True, key="rc_box")
    np.random.seed(42)
    ng = int(n_groups)
    groups = [np.random.normal(i * effect, 0.8, n) for i in range(ng)]
    names = [f"Group {chr(65+i)}" for i in range(ng)]
    fig = go.Figure()
    for i, (d, name) in enumerate(zip(groups, names)):
        color = px.colors.qualitative.Plotly[i]
        kde_x = np.linspace(d.min() - 0.5, d.max() + 0.5, 200)
        kde_y = stats.gaussian_kde(d)(kde_x)
        kde_y = kde_y / kde_y.max() * 0.35
        fig.add_trace(
            go.Scatter(
                x=kde_x + i,
                y=kde_y,
                mode="lines",
                fill="tozeroy",
                name=f"{name} density",
                line=dict(color=color),
                showlegend=False,
                hovertemplate="%{x:.2f}<extra></extra>",
            )
        )
        jitter = np.random.uniform(-0.1, 0.1, len(d))
        fig.add_trace(
            go.Scatter(
                x=np.full_like(d, i) + jitter,
                y=d,
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.5),
                showlegend=False,
                hovertemplate="%{y:.2f}<extra></extra>",
            )
        )
        if show_box:
            fig.add_trace(
                go.Box(
                    y=d,
                    name=name,
                    marker_color=color,
                    boxmean="sd",
                    width=0.15,
                    showlegend=False,
                )
            )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(tickmode="array", tickvals=list(range(ng)), ticktext=names),
        yaxis_title="Value",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Cloud = density shape\n"
                "- Rain = individual data points\n"
                "- Box = median & IQR\n"
                "- Combines all three views"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Replace boxplot for richer view\n"
                "- Show distribution + raw data\n"
                "- Modern alternative to violin\n"
                "- Best for medium n (20-200)"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Independent t-test\n"
                "- Mann-Whitney U\n"
                "- Welch's t-test\n"
                "- One-way ANOVA"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Rainclouds need sufficient n to show meaningful "
                "density. With n < 15, the KDE cloud is unreliable "
                "and raw dots alone are preferable."
            )


# =========================
# RIDGELINE PLOT
# =========================


def ridgeline_widget():
    st.markdown("## Ridgeline Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 20, 300, 100, key="rl_n")
        n_groups = st.selectbox("Number of Groups", [3, 5, 7], index=1, key="rl_ng")
        separation = st.slider("Group Separation", 0.5, 4.0, 2.0, 0.1, key="rl_sep")
        overlap = st.slider(
            "Vertical Overlap",
            0.0,
            1.0,
            0.5,
            0.05,
            key="rl_overlap",
            help="Lower = more separation between densities",
        )
    np.random.seed(42)
    ng = int(n_groups)
    groups = [
        np.random.normal(i * separation * overlap, 1 + i * 0.1, n) for i in range(ng)
    ]
    names = [f"Group {chr(65+i)}" for i in range(ng)]
    fig = go.Figure()
    x_min = min(g.min() for g in groups) - 1
    x_max = max(g.max() for g in groups) + 1
    x_grid = np.linspace(x_min, x_max, 300)
    for i, (d, name) in enumerate(zip(reversed(groups), reversed(names))):
        kde = stats.gaussian_kde(d)
        y_dens = kde(x_grid)
        y_dens = y_dens / y_dens.max() * 0.8
        idx = ng - 1 - i
        base = idx
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_dens + base,
                mode="lines",
                fill="tozeroy",
                name=name,
                line=dict(color=px.colors.qualitative.Plotly[idx % 10], width=1.5),
                hovertemplate=f"{name}<br>x = %{{x:.2f}}<br>density = %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Value",
        yaxis=dict(showticklabels=False),
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each ridge = group density\n"
                "- Peak shift = group difference\n"
                "- Width = group variance\n"
                "- Overlap = group similarity"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare many groups' distributions\n"
                "- Show distribution change over time\n"
                "- Replace overlaid KDEs (less cluttered)\n"
                "- Visualize group-level patterns"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Kruskal-Wallis\n"
                "- One-way ANOVA\n"
                "- Fligner-Killeen (variance)\n"
                "- Kolmogorov-Smirnov (pairwise)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Ridgelines can obscure fine details when many "
                "groups are plotted. Limit to 5-7 groups and "
                "use consistent bandwidth across groups."
            )


# =========================
# SANKEY DIAGRAM
# =========================


def sankey_widget():
    st.markdown("## Sankey Diagram")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        flow_type = st.selectbox(
            "Flow Type",
            ["Treatment Pathway", "Pre→Post Change", "Diagnostic Flow"],
            key="sk_type",
        )
        n = st.slider("Sample Size", 50, 1000, 200, key="sk_n")
        show_labels = st.toggle("Show Labels", True, key="sk_labels")
    np.random.seed(42)
    if flow_type == "Treatment Pathway":
        labels = [
            "Screened",
            "Eligible",
            "Randomized",
            "Treatment",
            "Control",
            "Responded",
            "No Response",
            "Responded",
            "No Response",
        ]
        source = [0, 1, 2, 2, 3, 3, 4, 4]
        target = [1, 2, 3, 4, 5, 6, 7, 8]
        probs = [0.6, 0.8, 0.5, 0.5, 0.6, 0.4, 0.4, 0.6]
    elif flow_type == "Pre→Post Change":
        labels = [
            "Pre: Yes",
            "Pre: No",
            "Post: Yes",
            "Post: No",
            "Post: Yes",
            "Post: No",
        ]
        source = [0, 0, 1, 1]
        target = [2, 3, 4, 5]
        probs = [0.7, 0.3, 0.2, 0.8]
    else:
        labels = [
            "Symptom Present",
            "Test +",
            "Test -",
            "Disease +",
            "Disease -",
            "Disease +",
            "Disease -",
        ]
        source = [0, 0, 1, 1, 2, 2]
        target = [1, 2, 3, 4, 5, 6]
        probs = [0.8, 0.2, 0.7, 0.3, 0.1, 0.9]
    values = [int(n * p) for p in probs]
    for v in range(1, len(values)):
        values[v] = min(
            values[v],
            values[(v - 1) // 2 if flow_type != "Diagnostic Flow" else v // 2],
        )
    colors_list = px.colors.qualitative.Plotly * 3
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=labels if show_labels else [""] * len(labels),
                    color=colors_list[: len(labels)],
                    pad=15,
                    thickness=20,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                    color=[colors_list[s % len(colors_list)] for s in source],
                ),
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10)
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Width = flow volume\n"
                "- Colors = categories\n"
                "- Nodes = decision points\n"
                "- Thin paths = small subgroups"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Visualize categorical flow\n"
                "- Show attrition in trials\n"
                "- Display diagnostic pathways\n"
                "- Pre-post transition patterns"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- McNemar's test (pre→post)\n"
                "- Chi-square (flow patterns)\n"
                "- Logistic regression\n"
                "- Markov transition models"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Too many nodes create unreadable diagrams. "
                "Limit to 10-15 nodes and group small "
                "categories into 'Other'."
            )


# =========================
# CLEVELAND DOT PLOT
# =========================


def cleveland_dot_widget():
    st.markdown("## Cleveland Dot Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_cats = st.selectbox(
            "Number of Categories", [5, 10, 15, 20], index=1, key="cdp_cats"
        )
        sort_by = st.radio(
            "Sort By", ["Value", "Alphabetical"], horizontal=True, key="cdp_sort"
        )
        show_ci = st.toggle("Show 95% CI", True, key="cdp_ci")
        error = st.slider(
            "Variability",
            0.05,
            0.5,
            0.15,
            0.05,
            key="cdp_err",
            help="How much noise/error around each estimate",
        )
    np.random.seed(42)
    nc = int(n_cats)
    values = np.random.uniform(10, 90, nc)
    errors = np.random.uniform(error * 5, error * 15, nc)
    cats = [f"Item {chr(65+i)}" for i in range(nc)]
    df = pd.DataFrame({"Category": cats, "Value": values, "Error": errors})
    if sort_by == "Value":
        df = df.sort_values("Value")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Value"],
            y=df["Category"],
            mode="markers",
            marker=dict(color="#4C78A8", size=10),
            error_x=dict(type="data", array=df["Error"] * 1.96, visible=show_ci),
            hovertemplate="%{y}: %{x:.1f} ± %{error_x.array:.1f}<extra></extra>",
        )
    )
    fig.add_vline(x=np.mean(values), line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Value",
        yaxis_title="",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Dot = point estimate\n"
                "- Bar = 95% confidence interval\n"
                "- Dashed line = overall mean\n"
                "- Dot position = relative ranking"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare many categories\n"
                "- Replace bar charts (avoids zero-baseline issue)\n"
                "- Show rankings with uncertainty\n"
                "- Forest-plot style summaries"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- One-sample t-test vs mean\n"
                "- Confidence interval analysis\n"
                "- Effect size comparison\n"
                "- Meta-analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Cleveland dot plots with too many categories "
                "(> 25) become cluttered. Use hierarchical "
                "grouping or filtering for large sets."
            )


# =========================
# HEXBIN PLOT
# =========================


def hexbin_widget():
    st.markdown("## Hexbin Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 100, 5000, 1000, key="hx_n")
        r = st.slider("Correlation (r)", -1.0, 1.0, 0.3, 0.05, key="hx_r")
        gridsize = st.slider(
            "Hexagon Size",
            10,
            50,
            25,
            5,
            key="hx_grid",
            help="Smaller = more bins, larger = smoother",
        )
        colorscale = st.selectbox(
            "Color Scale",
            ["Viridis", "Plasma", "Inferno", "Magma", "Blues"],
            index=0,
            key="hx_cs",
        )
    np.random.seed(42)
    x = np.random.normal(0, 1, n)
    y = r * x + np.sqrt(1 - r**2) * np.random.normal(0, 1, n)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram2d(
            x=x,
            y=y,
            nbinsx=gridsize,
            nbinsy=gridsize,
            colorscale=colorscale,
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Hexagon color = point density\n"
                "- Bright = many overlapping points\n"
                "- Dark/sparse = few points\n"
                "- Shape reveals correlation pattern"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Large n with overplotting\n"
                "- Dense scatter data (n > 500)\n"
                "- Alternative to scatterplots\n"
                "- Visualize 2D density"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Pearson correlation\n"
                "- Spearman correlation\n"
                "- Linear regression\n"
                "- 2D Kolmogorov-Smirnov"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Hexbin plots hide individual outliers. "
                "Always complement with a standard scatterplot "
                "if outlier detection is important."
            )


# =========================
# VENN / EULER DIAGRAM
# =========================


def venn_widget():
    st.markdown("## Venn / Euler Diagram")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_sets = st.selectbox("Number of Sets", [2, 3], index=1, key="venn_n")
        size_a = st.slider("Size of Set A", 10, 200, 100, key="venn_a")
        size_b = st.slider("Size of Set B", 10, 200, 80, key="venn_b")
        overlap_ab = st.slider(
            "A ∩ B", 0, min(size_a, size_b), min(size_a, size_b) // 3, key="venn_ab"
        )
        size_c = 0
        overlap_ac = 0
        overlap_bc = 0
        overlap_abc = 0
        if n_sets == 3:
            size_c = st.slider("Size of Set C", 10, 200, 60, key="venn_c")
            overlap_ac = st.slider(
                "A ∩ C", 0, min(size_a, size_c), min(size_a, size_c) // 4, key="venn_ac"
            )
            overlap_bc = st.slider(
                "B ∩ C", 0, min(size_b, size_c), min(size_b, size_c) // 4, key="venn_bc"
            )
            overlap_abc = st.slider(
                "A ∩ B ∩ C",
                0,
                min(overlap_ab, overlap_ac, overlap_bc),
                min(overlap_ab, overlap_ac, overlap_bc) // 2,
                key="venn_abc",
            )
    sets = []
    labels = []
    a_only = max(0, size_a - overlap_ab)
    b_only = max(0, size_b - overlap_ab)
    if n_sets == 2:
        sets = [a_only, overlap_ab, b_only]
        labels = [f"A only<br>{a_only}", f"A∩B<br>{overlap_ab}", f"B only<br>{b_only}"]
        set_labels = ["Set A", "Set B"]
    else:
        ac_only = max(0, overlap_ac - overlap_abc)
        bc_only = max(0, overlap_bc - overlap_abc)
        abc = max(0, overlap_abc)
        a_only = max(0, size_a - overlap_ab - overlap_ac + overlap_abc)
        b_only = max(0, size_b - overlap_ab - overlap_bc + overlap_abc)
        c_only = max(0, size_c - overlap_ac - overlap_bc + overlap_abc)
        ab_only = max(0, overlap_ab - abc)
        labels = [
            f"A<br>{a_only}",
            f"B<br>{b_only}",
            f"C<br>{c_only}",
            f"A∩B<br>{ab_only}",
            f"A∩C<br>{ac_only}",
            f"B∩C<br>{bc_only}",
            f"A∩B∩C<br>{abc}",
        ]
        sets = [a_only, b_only, c_only, ab_only, ac_only, bc_only, abc]
        set_labels = ["Set A", "Set B", "Set C"]

    def _circle_overlap_area(d, r1, r2):
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return math.pi * min(r1, r2) ** 2
        return (
            r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
            + r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
            - 0.5
            * math.sqrt(
                max(0, (r1 + r2 + d) * (r1 + r2 - d) * (r1 - r2 + d) * (-r1 + r2 + d))
            )
        )

    fig = go.Figure()
    if n_sets == 2:
        total = size_a + size_b
        base_r = 2.0
        r_a = math.sqrt(size_a / total) * base_r
        r_b = math.sqrt(size_b / total) * base_r
        max_possible = math.pi * min(r_a, r_b) ** 2
        area_ratio = overlap_ab / min(size_a, size_b) if min(size_a, size_b) > 0 else 0
        target_area = area_ratio * max_possible
        if overlap_ab <= 0:
            d = r_a + r_b + 0.3
        elif area_ratio >= 0.99:
            d = abs(r_a - r_b) * 0.3
        else:
            f = lambda dd: _circle_overlap_area(dd, r_a, r_b) - target_area
            d = scipy.optimize.brentq(
                f, max(0.01, abs(r_a - r_b) * 1.001), (r_a + r_b) * 0.999
            )
        cx_a, cy_a = -d / 2, 0
        cx_b, cy_b = d / 2, 0
        fig.add_shape(
            type="circle",
            x0=cx_a - r_a,
            y0=cy_a - r_a,
            x1=cx_a + r_a,
            y1=cy_a + r_a,
            fillcolor="rgba(76, 120, 168, 0.35)",
            line=dict(color="#4C78A8", width=2),
        )
        fig.add_shape(
            type="circle",
            x0=cx_b - r_b,
            y0=cy_b - r_b,
            x1=cx_b + r_b,
            y1=cy_b + r_b,
            fillcolor="rgba(228, 87, 86, 0.35)",
            line=dict(color="#E45756", width=2),
        )
        a_txt, b_txt, o_txt = str(a_only), str(b_only), str(overlap_ab)
        fig.add_annotation(
            x=cx_a - r_a * 0.6,
            y=0,
            text=a_txt,
            showarrow=False,
            font=dict(size=16, color="white"),
        )
        fig.add_annotation(
            x=0, y=0, text=o_txt, showarrow=False, font=dict(size=16, color="white")
        )
        fig.add_annotation(
            x=cx_b + r_b * 0.6,
            y=0,
            text=b_txt,
            showarrow=False,
            font=dict(size=16, color="white"),
        )
        fig.add_annotation(
            x=cx_a,
            y=cy_a + r_a + 0.3,
            text="Set A",
            showarrow=False,
            font=dict(size=14, color="#4C78A8"),
        )
        fig.add_annotation(
            x=cx_b,
            y=cy_b + r_b + 0.3,
            text="Set B",
            showarrow=False,
            font=dict(size=14, color="#E45756"),
        )
        padding_x = d / 2 + max(r_a, r_b) + 0.4
        padding_y = max(r_a, r_b) + 0.4
        fig.update_xaxes(range=[-padding_x, padding_x], visible=False)
        fig.update_yaxes(range=[-padding_y, padding_y], visible=False)
    else:
        total = size_a + size_b + size_c
        base_r = 2.0
        r_a = math.sqrt(size_a / total) * base_r if size_a > 0 else 0.5
        r_b = math.sqrt(size_b / total) * base_r if size_b > 0 else 0.5
        r_c = math.sqrt(size_c / total) * base_r if size_c > 0 else 0.5

        def _ideal_sep(r1, r2, overlap, s1, s2):
            if overlap <= 0:
                return r1 + r2 + 0.5
            max_possible = math.pi * min(r1, r2) ** 2
            ratio = overlap / min(s1, s2) if min(s1, s2) > 0 else 0
            target = min(ratio, 0.99) * max_possible
            lo, hi = abs(r1 - r2) * 1.001, (r1 + r2) * 0.999
            try:
                return scipy.optimize.brentq(
                    lambda dd: _circle_overlap_area(dd, r1, r2) - target, lo, hi
                )
            except (ValueError, RuntimeError):
                return (r1 + r2) * max(0.1, 1 - ratio * 0.6)

        s_ab = _ideal_sep(r_a, r_b, overlap_ab, size_a, size_b)
        s_ac = _ideal_sep(r_a, r_c, overlap_ac, size_a, size_c)
        s_bc = _ideal_sep(r_b, r_c, overlap_bc, size_b, size_c)
        s = max(0.5, (s_ab + s_ac + s_bc) / 3)
        h = s * math.sqrt(3) / 2
        cx_a, cy_a = 0, h * 2 / 3
        cx_b, cy_b = -s / 2, -h / 3
        cx_c, cy_c = s / 2, -h / 3
        colors = [
            ("rgba(76, 120, 168, 0.30)", "#4C78A8"),
            ("rgba(228, 87, 86, 0.30)", "#E45756"),
            ("rgba(0, 204, 150, 0.30)", "#00CC96"),
        ]
        for cx, cy, r, (fc, lc) in zip(
            [cx_a, cx_b, cx_c], [cy_a, cy_b, cy_c], [r_a, r_b, r_c], colors
        ):
            fig.add_shape(
                type="circle",
                x0=cx - r,
                y0=cy - r,
                x1=cx + r,
                y1=cy + r,
                fillcolor=fc,
                line=dict(color=lc, width=2),
            )

        def _region_pos(cx, cy, r, dx, dy):
            mag = math.hypot(dx, dy)
            if mag == 0:
                return cx, cy
            return cx + dx / mag * r * 0.55, cy + dy / mag * r * 0.55

        centers = {
            "A": (cx_a, cy_a, r_a, "#4C78A8"),
            "B": (cx_b, cy_b, r_b, "#E45756"),
            "C": (cx_c, cy_c, r_c, "#00CC96"),
        }
        only_map = {
            "A only": a_only,
            "B only": b_only,
            "C only": c_only,
            "A∩B only": ab_only,
            "A∩C only": ac_only,
            "B∩C only": bc_only,
            "A∩B∩C": abc,
        }
        ann_color = {
            "A only": "#4C78A8",
            "B only": "#E45756",
            "C only": "#00CC96",
            "A∩B only": "white",
            "A∩C only": "white",
            "B∩C only": "white",
            "A∩B∩C": "white",
        }
        ann_positions = {
            "A only": (cx_a, cy_a, -1, 0),
            "B only": (cx_b, cy_b, 1, 0),
            "C only": (cx_c, cy_c, 0, -1),
            "A∩B only": ((cx_a + cx_b) / 2, (cy_a + cy_b) / 2, 0, 0),
            "A∩C only": ((cx_a + cx_c) / 2, (cy_a + cy_c) / 2, 0, 0),
            "B∩C only": ((cx_b + cx_c) / 2, (cy_b + cy_c) / 2, 0, 0),
            "A∩B∩C": ((cx_a + cx_b + cx_c) / 3, (cy_a + cy_b + cy_c) / 3, 0, 0),
        }
        for key, (bx, by, ddx, ddy) in ann_positions.items():
            val = only_map[key]
            if key in ("A only", "B only", "C only"):
                ck = key[0]
                _, _, cr, _ = centers[ck]
                px, py = _region_pos(bx, by, cr, ddx, ddy)
            else:
                px, py = bx, by
            fig.add_annotation(
                x=px,
                y=py,
                text=str(val),
                showarrow=False,
                font=dict(size=15, color=ann_color[key]),
            )
        for ck, (cx, cy, cr, cc) in centers.items():
            fig.add_annotation(
                x=cx,
                y=cy + cr + 0.3,
                text=f"Set {ck}",
                showarrow=False,
                font=dict(size=14, color=cc),
            )
        x_ext = max(cx_a + r_a, abs(cx_b) + r_b, abs(cx_c) + r_c) + 0.4
        y_ext = max(cy_a + r_a, abs(cy_b) + r_b, abs(cy_c) + r_c) + 0.4
        padding_3 = max(x_ext, y_ext)
        fig.update_xaxes(range=[-padding_3, padding_3], visible=False)
        fig.update_yaxes(range=[-padding_3, padding_3], visible=False)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showticklabels=False, zeroline=False),
        yaxis=dict(showticklabels=False, zeroline=False),
        title=f"Venn Diagram: {', '.join(set_labels)}",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    clean_labels = [l.split("<br>")[0] for l in labels]
    venn_df = pd.DataFrame(
        {
            "Region": clean_labels,
            "Count": sets,
            "Proportion": [
                f"{s / sum(sets) * 100:.1f}%" if sum(sets) > 0 else "0%" for s in sets
            ],
        }
    )
    _apa_table_ge(venn_df, "Set Overlap Summary")
    if n_sets == 2:
        jaccard = (
            overlap_ab / (size_a + size_b - overlap_ab)
            if (size_a + size_b - overlap_ab) > 0
            else 0
        )
    else:
        union = (
            size_a
            + size_b
            + size_c
            - overlap_ab
            - overlap_ac
            - overlap_bc
            + overlap_abc
        )
        jaccard = overlap_abc / union if union > 0 else 0
    st.metric("Jaccard Similarity", f"{jaccard:.4f}")
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Circle size = set size\n"
                "- Overlap = intersection\n"
                "- No overlap = disjoint sets\n"
                "- Jaccard = overlap / union"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Show gene set overlaps\n"
                "- Display diagnostic test agreement\n"
                "- Visualize survey response overlap\n"
                "- Illustrate inclusion/exclusion criteria"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Fisher's exact test\n"
                "- Cohen's kappa\n"
                "- Jaccard/Tanimoto coefficient\n"
                "- Overlap coefficient"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Venn diagrams assume all intersections exist. "
                "With more than 3 sets, Euler diagrams (area-proportional) "
                "are preferred."
            )


def _apa_table_ge(df, title):
    st.markdown(f"**{title}**")
    st.dataframe(df, use_container_width=True)


# =========================
# POST-HOC ANALYSIS WIDGETS
# =========================

def _gen_ph_data(n_groups, n_per_group, effect_size_label):
    _rng_ph = np.random.default_rng(137)
    effect_map = {"None": 0.0, "Small": 0.2, "Medium": 0.5, "Large": 0.8}
    em = effect_map[effect_size_label]
    bm = np.array([0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5][:n_groups]) * em
    bm -= bm.mean()
    gd = []
    for i in range(n_groups):
        gd.append(bm[i] + _rng_ph.standard_normal(n_per_group))
    gm = [g.mean() for g in gd]
    pairs = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            x, y = gd[i], gd[j]
            m1, m2 = gm[i], gm[j]
            s1, s2 = x.std(ddof=1), y.std(ddof=1)
            sp = math.sqrt(((n_per_group - 1) * s1**2 + (n_per_group - 1) * s2**2) / (2 * n_per_group - 2))
            se = sp * math.sqrt(2 / n_per_group)
            md = m1 - m2
            d = md / sp
            t_stat = md / se
            p_val = 2 * stats.t.sf(abs(t_stat), 2 * n_per_group - 2)
            ci_lo = md - 1.96 * se
            ci_hi = md + 1.96 * se
            pairs.append({
                "i": i, "j": j,
                "pair": f"G{i+1} vs G{j+1}",
                "pair_long": f"Group {i+1} vs Group {j+1}",
                "mean_i": m1, "mean_j": m2,
                "md": md, "se": se, "d": d,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "t": t_stat, "p": p_val,
            })
    m = len(pairs)
    for idx in range(m):
        pairs[idx]["p_bonf"] = min(pairs[idx]["p"] * m, 1.0)
    sidx = np.argsort([p["p"] for p in pairs])
    for rk, idx in enumerate(sidx):
        pairs[idx]["p_holm"] = min(pairs[idx]["p"] * (m - rk), 1.0)
    pmat = np.eye(n_groups)
    dmat = np.eye(n_groups)
    for p in pairs:
        pmat[p["i"], p["j"]] = pmat[p["j"], p["i"]] = p["p"]
        dmat[p["i"], p["j"]] = dmat[p["j"], p["i"]] = p["d"]
    return {
        "gd": gd, "gm": gm, "pairs": pairs, "pmat": pmat, "dmat": dmat,
        "ng": n_groups, "np": n_per_group,
        "labels": [f"Group {i+1}" for i in range(n_groups)],
    }


def _assign_cld(means, pmat, alpha):
    n = len(means)
    order = np.argsort(-np.array(means))
    non_sig = pmat > alpha
    letters = ["" for _ in range(n)]
    next_l = 97
    for idx in order:
        friends = [j for j in range(n) if non_sig[idx, j] or idx == j]
        fl = set()
        for j in friends:
            for ch in letters[j]:
                fl.add(ch)
        assigned = False
        for ch in sorted(fl):
            ok = True
            for j in range(n):
                if ch in letters[j] and not non_sig[idx, j]:
                    ok = False
                    break
            if ok:
                letters[idx] += ch
                assigned = True
        if not assigned:
            letters[idx] = chr(next_l)
            next_l += 1
    return letters


def ph_ci_comparison_widget():
    st.markdown("## Confidence Interval Comparison Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        ng = st.selectbox("Number of Groups", [3, 4, 5, 6], 1, key="phge_ci_ng")
        npg = st.slider("Observations per Group", 10, 100, 30, 5, key="phge_ci_npg")
        eff = st.selectbox("Effect Size", ["None", "Small", "Medium", "Large"], 2, key="phge_ci_eff")
        alpha = st.slider("α", 0.001, 0.10, 0.05, 0.001, key="phge_ci_a")
    data = _gen_ph_data(ng, npg, eff)
    fig = go.Figure()
    pairs_sorted = sorted(data["pairs"], key=lambda p: p["md"])
    for p in pairs_sorted:
        col = "#00CC96" if p["p"] <= alpha else "#E45756"
        fig.add_trace(go.Scatter(
            x=[p["md"]], y=[p["pair"]], mode="markers",
            marker=dict(size=11, color=col, symbol="diamond"),
            showlegend=False,
            error_x=dict(type="data", symmetric=False,
                         array=[[p["ci_hi"] - p["md"]]],
                         arrayminus=[[p["md"] - p["ci_lo"]]],
                         color=col, thickness=2, width=8),
            hovertemplate="%{y}<br>MD = %{x:.3f}<br>95%CI [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>p = %{customdata[2]:.4f}<extra></extra>",
            customdata=[[p["ci_lo"], p["ci_hi"], p["p"]]],
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(template="plotly_dark", height=400,
                      title="Pairwise Confidence Intervals",
                      xaxis_title="Mean Difference",
                      margin=dict(l=10, r=10, t=30, b=10))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        ci, cw, ct, cm = st.columns(4)
        with ci:
            st.info("**Interpretation**\n\n- Diamond = mean difference\n- Bar = 95% CI\n- Green = p ≤ α")
        with cw:
            st.success("**When To Use**\n\n- After significant omnibus test\n- Visualizing pairwise group differences")
        with ct:
            st.warning("**Associated Tests**\n\n- Tukey HSD\n- Bonferroni t-test\n- Dunnett")
        with cm:
            st.error("**Common Mistake**\n\n- CI overlap ≠ non-significant\n- Use adjusted CIs for post-hoc")


def ph_forest_plot_widget():
    st.markdown("## Mean Difference Forest Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        ng = st.selectbox("Number of Groups", [3, 4, 5, 6], 1, key="phge_fp_ng")
        npg = st.slider("Observations per Group", 10, 100, 30, 5, key="phge_fp_npg")
        eff = st.selectbox("Effect Size", ["None", "Small", "Medium", "Large"], 2, key="phge_fp_eff")
        alpha = st.slider("α", 0.001, 0.10, 0.05, 0.001, key="phge_fp_a")
        correction = st.selectbox("Correction", ["Unadjusted", "Bonferroni", "Holm"], key="phge_fp_corr")
        sort = st.toggle("Sort by effect size", True, key="phge_fp_sort")
    data = _gen_ph_data(ng, npg, eff)
    pkey = "p" if correction == "Unadjusted" else ({"Bonferroni": "p_bonf", "Holm": "p_holm"})[correction]
    pairs_sorted = sorted(data["pairs"], key=lambda p: abs(p["md"]), reverse=sort)
    fig = go.Figure()
    for p in pairs_sorted:
        col = "#00CC96" if p[pkey] <= alpha else "#E45756"
        ypos = p["pair"] if not sort else p["pair"]
        fig.add_trace(go.Scatter(
            x=[p["md"]], y=[ypos], mode="markers+text",
            marker=dict(size=10, color=col),
            text=[f"{p['md']:.2f}"],
            textposition="middle right",
            showlegend=False,
            error_x=dict(type="data", symmetric=False,
                         array=[[p["ci_hi"] - p["md"]]],
                         arrayminus=[[p["md"] - p["ci_lo"]]],
                         color=col, thickness=2, width=8),
            hovertemplate=f"%{{y}}<br>MD = %{{x:.3f}}<br>CI [%{{customdata[0]:.3f}}, %{{customdata[1]:.3f}}]<br>p({correction}) = %{{customdata[2]:.4f}}<extra></extra>",
            customdata=[[p["ci_lo"], p["ci_hi"], p[pkey]]],
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(template="plotly_dark", height=350,
                      title=f"Forest Plot — {correction} p-values",
                      xaxis_title="Mean Difference",
                      margin=dict(l=10, r=60, t=30, b=10))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        ci, cw, ct, cm = st.columns(4)
        with ci:
            st.info("**Interpretation**\n\n- Points = mean difference\n- Lines = 95% CI\n- Labels = MD value\n- Green = significant")
        with cw:
            st.success("**When To Use**\n\n- Tukey/Games-Howell results\n- Meta-analytic summaries")
        with ct:
            st.warning("**Associated Tests**\n\n- Tukey HSD\n- Games-Howell\n- Dunnett")
        with cm:
            st.error("**Common Mistake**\n\n- Comparing across different outcome scales\n- Ignoring multiplicity correction")


def ph_cld_widget():
    st.markdown("## Compact Letter Display (CLD)")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        ng = st.selectbox("Number of Groups", [3, 4, 5, 6], 1, key="phge_cld_ng")
        npg = st.slider("Observations per Group", 10, 100, 30, 5, key="phge_cld_npg")
        eff = st.selectbox("Effect Size", ["None", "Small", "Medium", "Large"], 2, key="phge_cld_eff")
        alpha = st.slider("α", 0.001, 0.10, 0.05, 0.001, key="phge_cld_a")
        use_adjusted = st.toggle("Use adjusted p-values (Holm)", True, key="phge_cld_adj")
    data = _gen_ph_data(ng, npg, eff)
    pmat = data["pmat"].copy()
    if use_adjusted:
        for p in data["pairs"]:
            pmat[p["i"], p["j"]] = pmat[p["j"], p["i"]] = p["p_holm"]
    letters = _assign_cld(data["gm"], pmat, alpha)
    df_cld = pd.DataFrame({"Group": data["labels"], "Mean": data["gm"], "Letter": letters})
    df_cld = df_cld.sort_values("Mean", ascending=False).reset_index(drop=True)
    colors = px.colors.qualitative.Plotly[:ng]
    fig = go.Figure()
    for idx, row in df_cld.iterrows():
        gi = int(row["Group"].split()[-1]) - 1
        fig.add_trace(go.Bar(
            x=[row["Mean"]], y=[row["Group"]],
            orientation="h",
            marker=dict(color=colors[idx % len(colors)]),
            text=row["Letter"],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{y}<br>Mean = %{x:.3f}<br>Letter = %{text}<extra></extra>",
        ))
    fig.update_layout(template="plotly_dark", height=350,
                      title=f"Compact Letter Display (α = {alpha})",
                      xaxis_title="Mean",
                      margin=dict(l=10, r=50, t=30, b=10))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**CLD Table**")
        st.dataframe(df_cld.style.apply(
            lambda x: [f"background-color: {colors[int(x['Group'].split()[-1]) - 1]}; color: white" if c == "Group" else "" for c in df_cld.columns],
            axis=1
        ), use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        ci, cw, ct, cm = st.columns(4)
        with ci:
            st.info("**Interpretation**\n\n- Same letter = not significantly different\n- Different letter = significantly different\n- Two letters = intermediate group")
        with cw:
            st.success("**When To Use**\n\n- Journal publication tables\n- Agricultural/biological research")
        with ct:
            st.warning("**Associated Tests**\n\n- Tukey HSD\n- Duncan's MRT\n- Fisher LSD")
        with cm:
            st.error("**Common Mistake**\n\n- Letters depend on chosen α\n- CLD can be ambiguous with many groups")


def ph_significance_heatmap_widget():
    st.markdown("## Significance Heatmap")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        ng = st.selectbox("Number of Groups", [3, 4, 5, 6], 1, key="phge_sh_ng")
        npg = st.slider("Observations per Group", 10, 100, 30, 5, key="phge_sh_npg")
        eff = st.selectbox("Effect Size", ["None", "Small", "Medium", "Large"], 2, key="phge_sh_eff")
        alpha = st.slider("α", 0.001, 0.10, 0.05, 0.001, key="phge_sh_a")
        display = st.selectbox("Display", ["p-values (adjusted)", "Cohen's d", "Significance (binary)"], key="phge_sh_disp")
        correction = st.selectbox("Correction", ["Unadjusted", "Bonferroni", "Holm"], key="phge_sh_corr")
    data = _gen_ph_data(ng, npg, eff)
    pmat = data["pmat"].copy()
    if correction == "Bonferroni":
        for p in data["pairs"]:
            pmat[p["i"], p["j"]] = pmat[p["j"], p["i"]] = p["p_bonf"]
    elif correction == "Holm":
        for p in data["pairs"]:
            pmat[p["i"], p["j"]] = pmat[p["j"], p["i"]] = p["p_holm"]
    np.fill_diagonal(pmat, np.nan)
    np.fill_diagonal(data["dmat"], np.nan)
    if display == "p-values (adjusted)":
        z = pmat
        cs = [[0, "#00CC96"], [alpha, "#FFD700"], [1, "#E45756"]]
        tt = "%{z:.4f}"
        title = f"Post-Hoc p-values ({correction})"
        zmn, zmx = 0, 1
    elif display == "Cohen's d":
        z = data["dmat"]
        vmx = max(np.nanmax(np.abs(z)), 0.01)
        cs = "RdBu_r"
        tt = "%{z:.2f}"
        title = "Pairwise Cohen's d"
        zmn, zmx = -vmx, vmx
    else:
        z = (pmat <= alpha).astype(float)
        np.fill_diagonal(z, np.nan)
        cs = [[0, "#E45756"], [1, "#00CC96"]]
        tt = "%{z}"
        title = f"Significance Binary (α = {alpha})"
        zmn, zmx = 0, 1
    fig = go.Figure(data=go.Heatmap(
        z=z, x=data["labels"], y=data["labels"],
        texttemplate=tt, colorscale=cs,
        zmin=zmn, zmax=zmx,
        hovertemplate="%{x} vs %{y}<br>%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(template="plotly_dark", height=400,
                      title=title, margin=dict(l=10, r=10, t=30, b=10))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        ci, cw, ct, cm = st.columns(4)
        with ci:
            st.info("**Interpretation**\n\n- Green = significant (p ≤ α)\n- Red = not significant\n- Yellow = borderline")
        with cw:
            st.success("**When To Use**\n\n- Quick visual scan of pairwise results\n- Supplementing CLD tables")
        with ct:
            st.warning("**Associated Tests**\n\n- All post-hoc procedures")
        with cm:
            st.error("**Common Mistake**\n\n- Using unadjusted p-values for inference after multiple tests")


def ph_network_widget():
    st.markdown("## Pairwise Network Graph")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        ng = st.selectbox("Number of Groups", [3, 4, 5, 6, 7, 8], 2, key="phge_nw_ng")
        npg = st.slider("Observations per Group", 10, 100, 30, 5, key="phge_nw_npg")
        eff = st.selectbox("Effect Size", ["None", "Small", "Medium", "Large"], 2, key="phge_nw_eff")
        alpha = st.slider("α", 0.001, 0.10, 0.05, 0.001, key="phge_nw_a")
        layout = st.selectbox("Layout", ["Circle", "Spring (by mean)"], key="phge_nw_layout")
    data = _gen_ph_data(ng, npg, eff)
    angles = np.linspace(0, 2 * np.pi, ng, endpoint=False)
    if layout == "Spring (by mean)":
        gmin, gmax = min(data["gm"]), max(data["gm"])
        rng_s = gmax - gmin if gmax != gmin else 1
        radii = 0.3 + 0.7 * (data["gm"] - gmin) / rng_s
    else:
        radii = np.ones(ng)
    nx = radii * np.cos(angles)
    ny = radii * np.sin(angles)
    edge_x, edge_y = [], []
    edge_colors = []
    for p in data["pairs"]:
        if p["p_holm"] <= alpha:
            edge_x += [nx[p["i"]], nx[p["j"]], None]
            edge_y += [ny[p["i"]], ny[p["j"]], None]
            edge_colors.append("#00CC96")
    fig = go.Figure()
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#00CC96", width=2),
            showlegend=False,
            hovertemplate="Significant difference<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers+text",
        marker=dict(size=30, color=px.colors.qualitative.Plotly[:ng], line=dict(color="white", width=2)),
        text=data["labels"],
        textfont=dict(size=10, color="white"),
        textposition="middle center",
        showlegend=False,
        hovertemplate="%{text}<br>Mean = %{customdata:.3f}<extra></extra>",
        customdata=data["gm"],
    ))
    sig_count = sum(1 for p in data["pairs"] if p["p_holm"] <= alpha)
    fig.update_layout(
        template="plotly_dark", height=450,
        title=f"Pairwise Network — {sig_count} significant edges (Holm, α = {alpha})",
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="x"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        ci, cw, ct, cm = st.columns(4)
        with ci:
            st.info("**Interpretation**\n\n- Nodes = groups\n- Green edges = significant difference\n- Isolated node = not sig different from any")
        with cw:
            st.success("**When To Use**\n\n- Complex multi-group comparisons\n- Presentation/communication to non-statisticians")
        with ct:
            st.warning("**Associated Tests**\n\n- Any post-hoc procedure")
        with cm:
            st.error("**Common Mistake**\n\n- Edge thickness ≠ effect size in this simple layout\n- Network layout can be misleading")


def ph_estimation_widget():
    st.markdown("## Estimation Plot (Gardner-Altman)")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        npg = st.slider("Observations per Group", 10, 80, 30, 5, key="phge_est_npg")
        eff = st.selectbox("Effect Size", ["None", "Small", "Medium", "Large"], 2, key="phge_est_eff")
        alpha = st.slider("α", 0.001, 0.10, 0.05, 0.001, key="phge_est_a")
        label_a = st.text_input("Group A label", "Control", key="phge_est_la")
        label_b = st.text_input("Group B label", "Treatment", key="phge_est_lb")
    data = _gen_ph_data(2, npg, eff)
    pair = data["pairs"][0]
    g0, g1 = data["gd"][0], data["gd"][1]
    m0, m1 = data["gm"][0], data["gm"][1]
    se0 = g0.std(ddof=1) / math.sqrt(npg)
    se1 = g1.std(ddof=1) / math.sqrt(npg)
    xs0 = np.random.default_rng(42).uniform(-0.2, 0.2, npg)
    xs1 = np.random.default_rng(43).uniform(-0.2, 0.2, npg)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs0, y=g0, mode="markers",
        marker=dict(color="#4C78A8", size=6, opacity=0.7),
        name=label_a, showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=1 + xs1, y=g1, mode="markers",
        marker=dict(color="#E45756", size=6, opacity=0.7),
        name=label_b, showlegend=False,
    ))
    # Mean ± CI bars
    for xc, mc, sec, col in [(0, m0, se0, "#4C78A8"), (1, m1, se1, "#E45756")]:
        fig.add_trace(go.Scatter(
            x=[xc - 0.3, xc + 0.3], y=[mc, mc], mode="lines",
            line=dict(color=col, width=3), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[xc, xc], y=[mc - 1.96 * sec, mc + 1.96 * sec], mode="lines",
            line=dict(color=col, width=2), showlegend=False,
        ))
    # Mean difference on separate axis
    md = pair["md"]
    md_lo, md_hi = pair["ci_lo"], pair["ci_hi"]
    fig.add_trace(go.Scatter(
        x=[2], y=[md], mode="markers",
        marker=dict(size=14, color="#00CC96", symbol="diamond"),
        showlegend=False,
        error_y=dict(type="data", symmetric=False,
                     array=[[md_hi - md]], arrayminus=[[md - md_lo]],
                     color="#00CC96", thickness=2, width=8),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template="plotly_dark", height=400,
        title=f"Estimation Plot: MD = {md:.3f} [{md_lo:.3f}, {md_hi:.3f}]",
        xaxis=dict(tickvals=[0, 1, 2], ticktext=[label_a, label_b, "Mean Diff"], range=[-0.5, 2.5]),
        yaxis_title="Value",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        ci, cw, ct, cm = st.columns(4)
        with ci:
            st.info("**Interpretation**\n\n- Raw data plotted on left\n- Mean ± CI shown as bar + whisker\n- Right diamond = mean difference with CI\n- Connecting lines show paired change")
        with cw:
            st.success("**When To Use**\n\n- Two-group comparisons\n- Showing both raw data and effect size\n- Modern reporting standards")
        with ct:
            st.warning("**Associated Tests**\n\n- Independent t-test\n- Welch's t-test\n- Mann-Whitney U")
        with cm:
            st.error("**Common Mistake**\n\n- Not reporting the CI alongside the point estimate\n- Misinterpreting the Gardner-Altman scale")


def ph_raincloud_widget():
    st.markdown("## Raincloud Post Hoc Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        ng = st.selectbox("Number of Groups", [2, 3, 4], 1, key="phge_rc_ng")
        npg = st.slider("Observations per Group", 15, 100, 40, 5, key="phge_rc_npg")
        eff = st.selectbox("Effect Size", ["None", "Small", "Medium", "Large"], 2, key="phge_rc_eff")
        show_box = st.toggle("Show Boxplot", True, key="phge_rc_box")
        show_violin = st.toggle("Show Density (cloud)", True, key="phge_rc_vln")
        show_points = st.toggle("Show Raw Points", True, key="phge_rc_pts")
    data = _gen_ph_data(ng, npg, eff)
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly[:ng]
    for gi in range(ng):
        vals = data["gd"][gi]
        # Violin (raincloud = half-violin on top / positive side)
        if show_violin:
            fig.add_trace(go.Violin(
                y=vals, x0=gi, side="positive", line_color=colors[gi],
                fillcolor=colors[gi], opacity=0.4, name=data["labels"][gi],
                points=False, showlegend=False,
                bandwidth=npg ** -0.2,
            ))
        # Boxplot
        if show_box:
            fig.add_trace(go.Box(
                y=vals, x0=gi, fillcolor=colors[gi], line=dict(color=colors[gi]),
                opacity=0.6, width=0.12, name=data["labels"][gi],
                boxpoints=False, showlegend=False,
            ))
        # Raw points (beeswarm jitter)
        if show_points:
            x_jit = np.random.default_rng(42 + gi).uniform(-0.2, 0.2, npg)
            fig.add_trace(go.Scatter(
                x=np.full(npg, gi) + x_jit, y=vals,
                mode="markers", marker=dict(color=colors[gi], size=5, opacity=0.7),
                name=data["labels"][gi], showlegend=False,
            ))
    fig.update_layout(
        template="plotly_dark", height=400,
        title="Raincloud Plot",
        xaxis=dict(tickvals=list(range(ng)), ticktext=data["labels"]),
        yaxis_title="Value",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        ci, cw, ct, cm = st.columns(4)
        with ci:
            st.info("**Interpretation**\n\n- Cloud = density distribution\n- Box = median + IQR\n- Points = raw data\n- Rain = vertical data distribution")
        with cw:
            st.success("**When To Use**\n\n- Comparing distributions across groups\n- Publication-quality visualizations")
        with ct:
            st.warning("**Associated Tests**\n\n- One-way ANOVA\n- Kruskal-Wallis\n- All post-hoc methods")
        with cm:
            st.error("**Common Mistake**\n\n- Hiding raw data behind summary statistics\n- Overcrowding with too many groups")


# =========================
# GRAPH REGISTRY
# =========================

graphs = {
    "Histogram": {
        "category": "Distribution Plots",
        "description": "Visualize the frequency distribution of a single continuous variable using bins.",
        "when_to_use": "Exploring data distribution, checking normality, identifying outliers.",
        "interpretation": "Bell shape = normal; tail on right = positive skew; gaps = potential outliers.",
        "common_mistakes": "Bin width dramatically changes appearance. Always try multiple bin widths.",
        "associated_tests": [
            "Shapiro-Wilk",
            "Kolmogorov-Smirnov",
            "Anderson-Darling",
            "t-test (assumptions)",
        ],
        "widget_function": histogram_widget,
    },
    "Density / KDE Plot": {
        "category": "Distribution Plots",
        "description": "A smooth, continuous estimate of the probability density function.",
        "when_to_use": "Smooth distribution comparison, identifying modes, assessing shape.",
        "interpretation": "Peak = most common region; multiple peaks = multimodal; bandwidth controls smoothness.",
        "common_mistakes": "KDE can oversmooth bimodal data with large bandwidth, hiding real structure.",
        "associated_tests": ["t-test", "ANOVA", "Kruskal-Wallis", "Kolmogorov-Smirnov"],
        "widget_function": kde_widget,
    },
    "Boxplot": {
        "category": "Distribution Plots",
        "description": "Displays median, quartiles, range, and potential outliers via a box-and-whisker diagram.",
        "when_to_use": "Comparing groups at a glance, identifying outliers, assessing spread.",
        "interpretation": "Center line = median; box = IQR; whiskers = 1.5xIQR; dots = outliers.",
        "common_mistakes": "Overlapping boxes do NOT guarantee non-significance.",
        "associated_tests": ["t-test", "ANOVA", "Mann-Whitney U", "Kruskal-Wallis"],
        "widget_function": boxplot_widget,
    },
    "Violin Plot": {
        "category": "Distribution Plots",
        "description": "Combines boxplot with kernel density estimate to show distribution shape.",
        "when_to_use": "Replace boxplot for shape details, detect multimodality, compare distributions.",
        "interpretation": "Width = density; narrow = sparse; symmetry = normal distribution.",
        "common_mistakes": "Misleading with n < 15 - KDE over-interprets sparse data.",
        "associated_tests": [
            "ANOVA",
            "Kruskal-Wallis",
            "Welch's t-test",
            "Permutation test",
        ],
        "widget_function": violin_widget,
    },
    "Q-Q Plot": {
        "category": "Distribution Plots",
        "description": "Plots observed quantiles against theoretical normal quantiles to assess normality.",
        "when_to_use": "Checking normality assumption, identifying distribution shape, detecting tail outliers.",
        "interpretation": "Points on diagonal = normal; S-curve = heavy tails; curve above = right skew.",
        "common_mistakes": "Small samples (n < 30) produce noisy Q-Q plots even when data are normal.",
        "associated_tests": [
            "Shapiro-Wilk",
            "Kolmogorov-Smirnov",
            "Anderson-Darling",
            "D'Agostino-Pearson",
        ],
        "widget_function": qq_widget,
    },
    "Grouped Bar Chart": {
        "category": "Comparison Plots",
        "description": "Disaggregates group means across categories with side-by-side bars.",
        "when_to_use": "Comparing groups across categories, showing means with uncertainty bars.",
        "interpretation": "Bar height = mean; error bars = variability; non-overlapping approx significant.",
        "common_mistakes": "Bars should always start at zero to avoid misleading visual differences.",
        "associated_tests": [
            "Two-way ANOVA",
            "Welch's t-test",
            "Mixed-effects models",
            "Post-hoc comparisons",
        ],
        "widget_function": grouped_bar_widget,
    },
    "Error Bar Plot": {
        "category": "Comparison Plots",
        "description": "Shows group means with confidence intervals to indicate estimate precision.",
        "when_to_use": "Showing precision of estimates, comparing means visually, forest plots.",
        "interpretation": "Dot = mean; bar = CI; non-overlapping approx p < .05 (conservative).",
        "common_mistakes": "Overlapping CIs do NOT guarantee non-significance.",
        "associated_tests": ["t-test", "ANOVA", "Welch's t-test", "Linear regression"],
        "widget_function": error_bar_widget,
    },
    "Paired Line Plot": {
        "category": "Comparison Plots",
        "description": "Connects paired observations (e.g., pre-post) to show within-subject change.",
        "when_to_use": "Pre-post studies, within-subject designs, crossover trials.",
        "interpretation": "Each line = one subject; direction = increase or decrease; thick line = mean change.",
        "common_mistakes": "Using independent t-test on paired data inflates Type II error.",
        "associated_tests": [
            "Paired t-test",
            "Wilcoxon Signed-Rank",
            "Repeated measures ANOVA",
            "Linear mixed models",
        ],
        "widget_function": paired_line_widget,
    },
    "Boxplot Comparison": {
        "category": "Comparison Plots",
        "description": "Side-by-side boxplots for comparing multiple groups robustly.",
        "when_to_use": "Comparing groups robustly, checking equal variance, identifying group outliers.",
        "interpretation": "Compare medians; box overlap approx group similarity; box size = variability.",
        "common_mistakes": "Different distributions can produce identical boxplots (Anscombe's quartet).",
        "associated_tests": [
            "One-way ANOVA",
            "Welch's ANOVA",
            "Kruskal-Wallis",
            "Levene's test",
        ],
        "widget_function": boxplot_comp_widget,
    },
    "Violin Comparison": {
        "category": "Comparison Plots",
        "description": "Side-by-side violins showing distribution shape, spread, and central tendency across groups.",
        "when_to_use": "Detailed group comparison, detecting shape differences, assessing normality assumptions.",
        "interpretation": "Width = density; shift = group difference; shape = distributional difference.",
        "common_mistakes": "Violin width is often misinterpreted as frequency - it represents density.",
        "associated_tests": [
            "Independent t-test",
            "Mann-Whitney U",
            "Welch's t-test",
            "Fligner-Killeen",
        ],
        "widget_function": violin_comp_widget,
    },
    "Scatterplot": {
        "category": "Correlation Plots",
        "description": "Plots pairs of (X, Y) values to reveal the form, direction, and strength of relationships.",
        "when_to_use": "Assess relationship direction and strength, detect outliers and non-linearity.",
        "interpretation": "Upward slope = positive r; downward = negative r; tight cluster = strong correlation.",
        "common_mistakes": "Correlation does NOT imply causation.",
        "associated_tests": [
            "Pearson correlation",
            "Spearman correlation",
            "Linear regression",
            "Correlation test",
        ],
        "widget_function": scatter_widget,
    },
    "Correlation Heatmap": {
        "category": "Correlation Plots",
        "description": "A 2D grid colored by correlation values to visualize relationships among many variables.",
        "when_to_use": "Checking multicollinearity, exploring many variable relationships, identifying clusters.",
        "interpretation": "Red = positive; blue = negative; dark = strong; diagonal = variable with itself (r=1).",
        "common_mistakes": "Visual patterns can be misleading when variables have different scales.",
        "associated_tests": [
            "Pearson correlation matrix",
            "Spearman correlation matrix",
            "VIF",
            "Factor analysis",
        ],
        "widget_function": heatmap_widget,
    },
    "Bubble Plot": {
        "category": "Correlation Plots",
        "description": "A scatterplot variant where point size encodes a third variable and color a fourth.",
        "when_to_use": "Showing 3-4 variables at once, highlighting weighted importance, population data.",
        "interpretation": "Position = X-Y relationship; bubble size = third variable; color = fourth variable.",
        "common_mistakes": "Bubble area (not diameter) should encode the third variable.",
        "associated_tests": [
            "Weighted correlation",
            "Multiple regression",
            "Weighted least squares",
            "Meta-analysis",
        ],
        "widget_function": bubble_widget,
    },
    "Monotonic vs Linear Correlation": {
        "category": "Correlation Plots",
        "description": "Compares Pearson (linear) and Spearman (monotonic) correlation on non-linear relationships.",
        "when_to_use": "Testing if relationship is linear, choosing Pearson vs Spearman, detecting non-linear patterns.",
        "interpretation": "Pearson = linear; Spearman = monotonic; big diff (r vs rho) = non-linear.",
        "common_mistakes": "Pearson r near 0 does NOT mean no relationship - only no linear relationship.",
        "associated_tests": [
            "Pearson correlation",
            "Spearman correlation",
            "Kendall's tau",
            "Distance correlation",
        ],
        "widget_function": monotonic_widget,
    },
    "Linear Regression Plot": {
        "category": "Regression Plots",
        "description": "Fits a line to (X,Y) data showing the best-fit slope, confidence band, and residuals.",
        "when_to_use": "Modeling continuous outcomes, estimating effect size, predicting Y from X.",
        "interpretation": "Line = best fit; band = 95% CI; slope = change in Y per unit X; R-squared = variance explained.",
        "common_mistakes": "Extrapolating beyond the observed x-range - relationships may not hold outside data.",
        "associated_tests": [
            "F-test (overall)",
            "t-test (coefficient)",
            "Pearson correlation",
            "ANOVA (nested)",
        ],
        "widget_function": linear_reg_widget,
    },
    "Multiple Regression Surface": {
        "category": "Regression Plots",
        "description": "A 3D surface plot showing the predicted outcome from two continuous predictors.",
        "when_to_use": "Modeling multiple predictors, controlling for confounders, testing interaction effects.",
        "interpretation": "Plane = predicted Y; slope along X1 = beta1 holding X2 constant; twist = interaction.",
        "common_mistakes": "Coefficients are sensitive to predictor scaling - standardize before comparing.",
        "associated_tests": [
            "F-test (overall)",
            "Partial F-test",
            "t-test (coefficients)",
            "VIF",
        ],
        "widget_function": multiple_reg_widget,
    },
    "Logistic Regression Sigmoid Curve": {
        "category": "Regression Plots",
        "description": "The S-shaped probability curve showing how a binary outcome changes with a predictor.",
        "when_to_use": "Binary outcome prediction, estimating odds ratios, medical diagnosis models.",
        "interpretation": "S-curve = logistic probability; steep = strong predictor; threshold = classification cutoff.",
        "common_mistakes": "Default 0.5 threshold is not always optimal - adjust based on error costs.",
        "associated_tests": [
            "Likelihood ratio test",
            "Wald test",
            "Hosmer-Lemeshow",
            "ROC-AUC",
        ],
        "widget_function": logistic_widget,
    },
    "Multinomial Decision Boundaries": {
        "category": "Regression Plots",
        "description": "Decision regions for multi-class classification in 2D feature space.",
        "when_to_use": "Multi-class classification, understanding decision boundaries, feature space exploration.",
        "interpretation": "Colored regions = decision zones; boundaries = uncertain areas; overlap = difficulty.",
        "common_mistakes": "Linear boundaries cannot separate non-linear class patterns.",
        "associated_tests": [
            "Multinomial logistic",
            "MANOVA",
            "Discriminant analysis",
            "Classification metrics",
        ],
        "widget_function": multinomial_widget,
    },
    "Ordinal Logistic Probability Curves": {
        "category": "Regression Plots",
        "description": "Shows how the probability of each ordinal level changes across a continuous predictor.",
        "when_to_use": "Ordered categorical outcomes, Likert scales, disease severity staging.",
        "interpretation": "Each curve = probability of one level; non-parallel = proportional odds violation.",
        "common_mistakes": "Proportional odds assumption must hold - non-parallel curves require alternative models.",
        "associated_tests": [
            "Proportional odds test",
            "Brant test",
            "Likelihood ratio test",
            "Score test",
        ],
        "widget_function": ordinal_logit_widget,
    },
    "Poisson Count Regression": {
        "category": "Regression Plots",
        "description": "Models count outcomes with a log-linear mean curve showing predicted counts.",
        "when_to_use": "Modeling count outcomes, event frequencies, rare disease incidence.",
        "interpretation": "Y-axis = count; curve = predicted mean; spread increases with mean.",
        "common_mistakes": "Poisson assumes mean = variance - overdispersion needs Negative Binomial.",
        "associated_tests": [
            "Likelihood ratio test",
            "Wald test",
            "Deviance GOF",
            "Dispersion test",
        ],
        "widget_function": poisson_widget,
    },
    "Confusion Matrix Explorer": {
        "category": "Diagnostic Accuracy Plots",
        "description": "A 2x2 heatmap showing true/false positives and negatives with derived metrics.",
        "when_to_use": "Evaluating binary classifiers, comparing diagnostic tests, understanding error types.",
        "interpretation": "Diagonal = correct; off-diagonal = errors; PPV depends on prevalence.",
        "common_mistakes": "Accuracy is misleading with imbalanced classes - always report PPV and NPV.",
        "associated_tests": [
            "McNemar's test",
            "Cohen's Kappa",
            "ROC-AUC",
            "Diagnostic likelihood ratios",
        ],
        "widget_function": confusion_widget,
    },
    "ROC Curve Explorer": {
        "category": "Diagnostic Accuracy Plots",
        "description": "Plots TPR vs FPR across thresholds with AUC summarizing discriminative ability.",
        "when_to_use": "Comparing diagnostic tests, assessing model discrimination, choosing optimal threshold.",
        "interpretation": "Top-left = better; AUC = probability correct ranking; 0.5 = guessing; 0.8+ = good.",
        "common_mistakes": "AUC ignores calibration - high AUC can have poorly calibrated probabilities.",
        "associated_tests": [
            "DeLong test",
            "Hanley-McNeil test",
            "Bootstrap AUC",
            "Sensitivity analysis",
        ],
        "widget_function": roc_widget,
    },
    "Precision-Recall Curve": {
        "category": "Diagnostic Accuracy Plots",
        "description": "Plots precision vs recall across thresholds, better for imbalanced classification than ROC.",
        "when_to_use": "Imbalanced classification, rare disease detection, when PPV matters more.",
        "interpretation": "Higher curve = better; baseline = always-predict-positive; PR better than ROC for imbalance.",
        "common_mistakes": "PR curves from small samples are noisy - use bootstrap confidence bands.",
        "associated_tests": [
            "Average Precision",
            "F1 score",
            "F-beta score",
            "Bootstrap PR comparison",
        ],
        "widget_function": pr_curve_widget,
    },
    "Sensitivity-Specificity Threshold Explorer": {
        "category": "Diagnostic Accuracy Plots",
        "description": "Shows how sensitivity and specificity trade off across decision thresholds with cost optimization.",
        "when_to_use": "Choosing diagnostic cutoff, balancing sensitivity vs specificity, incorporating error costs.",
        "interpretation": "Blue = sensitivity; red = specificity; tradeoff: increase one = decrease other.",
        "common_mistakes": "Youden's index treats FP and FN equally - in medicine FN is often more costly.",
        "associated_tests": [
            "ROC analysis",
            "Youden's index",
            "Cost-benefit analysis",
            "Decision curve analysis",
        ],
        "widget_function": threshold_widget,
    },
    "Calibration Plot": {
        "category": "Diagnostic Accuracy Plots",
        "description": "Plots observed proportions against predicted probabilities to assess model calibration.",
        "when_to_use": "Assessing probability accuracy, checking model reliability, comparing risk prediction models.",
        "interpretation": "On diagonal = perfectly calibrated; above = underestimated; below = overestimated.",
        "common_mistakes": "Hosmer-Lemeshow is sensitive to binning - use calibration plots + intercept/slope instead.",
        "associated_tests": [
            "Hosmer-Lemeshow test",
            "Brier score",
            "Spiegelhalter z-test",
            "Calibration slope",
        ],
        "widget_function": calibration_widget,
    },
    "Bland-Altman Plot": {
        "category": "Agreement Plots",
        "description": "Plots the difference between two measurements against their mean to assess agreement.",
        "when_to_use": "Comparing measurement methods, assessing test-retest reliability, medical device validation.",
        "interpretation": "y=0 = perfect agreement; mean diff = bias; dashed lines = limits of agreement.",
        "common_mistakes": "High r does NOT mean good agreement - two methods can be correlated but biased.",
        "associated_tests": [
            "Paired t-test",
            "ICC",
            "Deming regression",
            "Passing-Bablok regression",
        ],
        "widget_function": bland_altman_widget,
    },
    "Cohen's Kappa Agreement Matrix": {
        "category": "Agreement Plots",
        "description": "A heatmap of inter-rater classifications with Cohen's Kappa statistic for chance-corrected agreement.",
        "when_to_use": "Inter-rater reliability, diagnostic agreement studies, psychiatric assessment.",
        "interpretation": "Diagonal = perfect agreement; kappa < 0 = worse than chance; 0.4-0.6 = moderate; >0.8 = near perfect.",
        "common_mistakes": "Kappa is prevalence-dependent - rare categories produce low kappa even with high agreement.",
        "associated_tests": [
            "Weighted Kappa",
            "Fleiss' Kappa",
            "McNemar's test",
            "ICC",
        ],
        "widget_function": kappa_widget,
    },
    "ICC Visualization": {
        "category": "Agreement Plots",
        "description": "Shows subject-level measurements across raters/occasions with estimated Intraclass Correlation.",
        "when_to_use": "Test-retest reliability, inter-rater reliability (continuous), intra-rater reliability.",
        "interpretation": "Each color = one subject; horizontal lines = high ICC; crossing lines = low ICC.",
        "common_mistakes": "High ICC can mask systematic bias - combine with Bland-Altman for full assessment.",
        "associated_tests": ["ICC(1,1)", "ICC(2,1)", "ICC(3,k)", "Bland-Altman"],
        "widget_function": icc_widget,
    },
    "PCA Scatter Plot": {
        "category": "Multivariate Plots",
        "description": "Projects high-dimensional data onto the first two principal components to reveal structure.",
        "when_to_use": "Reducing dimensionality, visualizing high-dim data, checking for natural clusters.",
        "interpretation": "PCs = directions of max variance; close points = similar profiles; axis labels show % variance.",
        "common_mistakes": "PCA assumes linear relationships - non-linear structure is not captured.",
        "associated_tests": [
            "MANOVA",
            "Factor analysis",
            "K-means clustering",
            "PERMANOVA",
        ],
        "widget_function": pca_widget,
    },
    "MANOVA Group Clouds": {
        "category": "Multivariate Plots",
        "description": "Visualizes multivariate group differences in 2D or 3D space with centroid markers.",
        "when_to_use": "Comparing groups on multiple DVs, controlling for correlated outcomes.",
        "interpretation": "Each color = one group; distance = group difference; overlap = non-significant.",
        "common_mistakes": "Requires multivariate normality and homogeneity of covariance matrices.",
        "associated_tests": [
            "Pillai's Trace",
            "Wilks' Lambda",
            "Hotelling-Lawley Trace",
            "Roy's Largest Root",
        ],
        "widget_function": manova_widget,
    },
    "Cluster Visualization": {
        "category": "Multivariate Plots",
        "description": "Applies K-means clustering and visualizes discovered clusters with centroids.",
        "when_to_use": "Discovering natural groupings, segmenting populations, pattern recognition.",
        "interpretation": "Each color = discovered cluster; X marks = cluster center; tight clusters = well-separated.",
        "common_mistakes": "K-means assumes spherical, equally-sized clusters - elongated clusters are split incorrectly.",
        "associated_tests": [
            "Silhouette score",
            "Elbow method",
            "Gap statistic",
            "Davies-Bouldin index",
        ],
        "widget_function": cluster_widget,
    },
    "3D Scatter Explorer": {
        "category": "Multivariate Plots",
        "description": "Interactive 3D scatter plot for exploring three-variable relationships with group coloring.",
        "when_to_use": "Exploring 3-variable relationships, identifying 3D clusters, interactive data exploration.",
        "interpretation": "Each axis = one variable; position = multi-dim profile; clusters = groups with similar profiles.",
        "common_mistakes": "3D plots can obscure patterns depending on viewing angle - rotate to see all perspectives.",
        "associated_tests": [
            "MANOVA",
            "Multivariate regression",
            "Canonical correlation",
            "3D PCA",
        ],
        "widget_function": scatter3d_widget,
    },
    "Normal P-P Plot": {
        "category": "Distribution Plots",
        "description": "Plots empirical cumulative probabilities against theoretical normal probabilities to assess normality.",
        "when_to_use": "Assessing normality, complementing Q-Q plots, checking distribution center fit.",
        "interpretation": "Points on diagonal = normal; S-curve = heavy tails; above diagonal = right skew.",
        "common_mistakes": "P-P plots are less sensitive to tail departures than Q-Q plots — use both together.",
        "associated_tests": [
            "Shapiro-Wilk",
            "Kolmogorov-Smirnov",
            "Anderson-Darling",
            "D'Agostino-Pearson",
        ],
        "widget_function": pp_widget,
    },
    "Raincloud Plot": {
        "category": "Comparison Plots",
        "description": "Combines jittered raw data, boxplot, and half-violin (KDE) for a comprehensive group comparison.",
        "when_to_use": "Modern group comparisons, replacing boxplots, showing distributions + raw data simultaneously.",
        "interpretation": "Points = raw data; box = median + IQR; half-violin = density shape; wider = more density.",
        "common_mistakes": "Jitter width is arbitrary — it shows density not exact x-position. Set seed for reproducibility.",
        "associated_tests": [
            "Independent t-test",
            "Mann-Whitney U",
            "Welch's t-test",
            "Permutation tests",
        ],
        "widget_function": raincloud_widget,
    },
    "Residuals vs Fitted Plot": {
        "category": "Regression Plots",
        "description": "Scatter plot of residuals against fitted values to detect model misspecification.",
        "when_to_use": "After fitting linear regression, checking homoscedasticity, linearity, and outliers.",
        "interpretation": "Random scatter around 0 = OK; fan shape = heteroscedasticity; U-shape = non-linearity.",
        "common_mistakes": "Patterned residuals indicate model misspecification — fix model before interpreting coefficients.",
        "associated_tests": [
            "Breusch-Pagan",
            "Goldfeld-Quandt",
            "RESET test",
            "Cook's distance",
        ],
        "widget_function": residuals_fitted_widget,
    },
    "Polynomial Regression Fit": {
        "category": "Regression Plots",
        "description": "Fits a polynomial of adjustable degree to data, comparing against the true generating function.",
        "when_to_use": "Teaching bias-variance tradeoff, modeling non-linear relationships, testing curvature.",
        "interpretation": "Higher degree = more flexible; degree 1 = straight line; high degrees overfit at boundaries.",
        "common_mistakes": "Never extrapolate polynomial fits — they diverge wildly outside the observed data range.",
        "associated_tests": [
            "F-test (nested)",
            "Cross-validation MSE",
            "AIC / BIC",
            "ANOVA model comparison",
        ],
        "widget_function": poly_reg_widget,
    },
    "Regularization Path": {
        "category": "Regression Plots",
        "description": "Shows how Lasso (L1) or Ridge (L2) regression coefficients change as regularization strength increases.",
        "when_to_use": "High-dimensional data, feature selection, understanding bias-variance tradeoff interactively.",
        "interpretation": "Each line = one coefficient; left = no shrinkage; right = heavy shrinkage; Lasso zeros out features.",
        "common_mistakes": "Lasso selects at most n features. With p > n, Ridge may generalize better. Standardize predictors first.",
        "associated_tests": [
            "Cross-validated MSE",
            "Regularization path stability",
            "BIC",
            "Bootstrap stability",
        ],
        "widget_function": reg_path_widget,
    },
    "Scatterplot Matrix (SPLOM)": {
        "category": "Correlation Plots",
        "description": "A grid of bivariate scatterplots for every pair of variables with adjustable correlation structure.",
        "when_to_use": "Exploring multivariate relationships, detecting multicollinearity, identifying multivariate outliers.",
        "interpretation": "Each panel = one pair; tight ellipse = strong correlation; diagonal = variable vs itself.",
        "common_mistakes": "SPLOMs become unreadable with >10 variables — use correlation heatmap or PCA instead.",
        "associated_tests": [
            "Pearson correlation",
            "VIF",
            "Bartlett's sphericity",
            "MANOVA assumptions",
        ],
        "widget_function": splom_widget,
    },
    "Parallel Coordinates Plot": {
        "category": "Multivariate Plots",
        "description": "Each observation is a line crossing parallel axes, revealing multivariate patterns and clusters.",
        "when_to_use": "Visualizing high-dimensional data, identifying clusters, exploring variable relationships.",
        "interpretation": "Each axis = one variable; crossing lines = negative correlation; parallel lines = positive correlation.",
        "common_mistakes": "Axis order affects interpretation — reorder to highlight patterns. Too many lines creates visual clutter.",
        "associated_tests": [
            "MANOVA",
            "Canonical correlation",
            "Discriminant analysis",
            "Cluster validation",
        ],
        "widget_function": parallel_coords_widget,
    },
    "Stem-and-Leaf Plot": {
        "category": "Distribution Plots",
        "description": "A text-based display showing the leading digit (stem) and trailing digit (leaf) of each data point.",
        "when_to_use": "Small datasets, quick manual distribution check, preserving exact values.",
        "interpretation": "Stem = leading digit; leaf = trailing digit; row length = frequency of stem.",
        "common_mistakes": "Choose stem unit carefully — too few stems loses detail, too many creates sparse rows.",
        "associated_tests": [
            "Kolmogorov-Smirnov",
            "Shapiro-Wilk",
            "Anderson-Darling",
            "Visual shape assessment",
        ],
        "widget_function": stem_leaf_widget,
    },
    "Frequency Polygon": {
        "category": "Distribution Plots",
        "description": "A line graph connecting the midpoints of histogram bins to show distribution shape smoothly.",
        "when_to_use": "Overlaying multiple distributions, comparing group shapes, cumulative frequency display.",
        "interpretation": "Points at bin midpoints; line shows distribution shape; area under polygon = total N.",
        "common_mistakes": "Bin count changes polygon shape significantly — always try multiple bin widths.",
        "associated_tests": [
            "Chi-square GOF",
            "Kolmogorov-Smirnov",
            "Anderson-Darling",
            "Distribution fit test",
        ],
        "widget_function": freq_poly_widget,
    },
    "Beeswarm / Swarm Plot": {
        "category": "Comparison Plots",
        "description": "Non-overlapping points showing every observation with exact position and distribution shape.",
        "when_to_use": "Small to moderate n, showing every data point clearly, publication-ready group plots.",
        "interpretation": "Each dot = one observation; no overlap = exact value visible; density by vertical stacking.",
        "common_mistakes": "Cluttered with n > 200 — use violin or boxplot for larger samples.",
        "associated_tests": [
            "Independent t-test",
            "Mann-Whitney U",
            "Welch's t-test",
            "Permutation test",
        ],
        "widget_function": beeswarm_widget,
    },
    "Polar Density Plot": {
        "category": "Distribution Plots",
        "description": "A circular density plot showing the distribution of directional / angular data.",
        "when_to_use": "Circular or directional data, wind direction analysis, seasonal patterns, animal movement.",
        "interpretation": "Angle = direction; radius = density; peaks = preferred direction; troughs = avoided direction.",
        "common_mistakes": "Circular data wraps at 0°/360° — linear KDE gives wrong boundary density.",
        "associated_tests": [
            "Rayleigh test",
            "V-test",
            "Watson-Williams test",
            "Circular ANOVA",
        ],
        "widget_function": polar_density_widget,
    },
    "Probability Density Function Plot": {
        "category": "Distribution Plots",
        "description": "Interactive theoretical PDFs for Normal, t, F, Chi-square, Exponential, Beta, and Gamma distributions.",
        "when_to_use": "Understanding distribution shapes, comparing theoretical PDFs, learning parameter effects.",
        "interpretation": "Area under curve = 1; height = relative likelihood; peak = most probable region.",
        "common_mistakes": "PDF can exceed 1 — it is a density not a probability. Only the integral over a range gives probability.",
        "associated_tests": [
            "GOF (KS, AD, CVM)",
            "Q-Q plot",
            "Parameter estimation",
            "MLE / Bayesian inference",
        ],
        "widget_function": pdf_plot_widget,
    },
    "Pareto Chart": {
        "category": "Distribution Plots",
        "description": "A bar chart sorted by frequency with a cumulative percentage line to highlight the vital few.",
        "when_to_use": "Quality control, identifying vital few vs trivial many, prioritizing improvements.",
        "interpretation": "Bars sorted descending; line = cumulative percent; 80% line = Pareto principle.",
        "common_mistakes": "80/20 is a guideline not a law — actual split depends on your data.",
        "associated_tests": [
            "Chi-square GOF",
            "Lorenz curve",
            "Gini coefficient",
            "Concentration indices",
        ],
        "widget_function": pareto_widget,
    },
    "Dot Plot": {
        "category": "Distribution Plots",
        "description": "Each observation shown as a dot along a single axis with jitter for overlapping values.",
        "when_to_use": "Small datasets, showing exact distribution, identifying clusters and gaps.",
        "interpretation": "Each dot = one observation; horizontal spread = distribution; stacked = multiple values.",
        "common_mistakes": "Random jitter can mislead — use fixed seed for reproducibility.",
        "associated_tests": [
            "One-sample t-test",
            "Wilcoxon signed-rank",
            "Sign test",
            "Shapiro-Wilk",
        ],
        "widget_function": dot_plot_widget,
    },
    "Time Series Plot": {
        "category": "Comparison Plots",
        "description": "Data points plotted in time order to reveal trends, seasonality, and patterns.",
        "when_to_use": "Longitudinal data, trend analysis, seasonal pattern detection, intervention effects.",
        "interpretation": "X-axis = time order; trend = long-term direction; seasonality = repeating pattern.",
        "common_mistakes": "Don't connect points across missing time gaps. Check for autocorrelation before modeling.",
        "associated_tests": [
            "ADF test",
            "Ljung-Box test",
            "Durbin-Watson test",
            "Granger causality",
        ],
        "widget_function": time_series_widget,
    },
    "Pie Chart": {
        "category": "Comparison Plots",
        "description": "A circular chart divided into slices proportional to the quantities they represent.",
        "when_to_use": "Simple part-to-whole displays, few categories, rough visual comparison, non-technical audiences.",
        "interpretation": "Each slice = proportion; area encodes percentage; full circle = 100%.",
        "common_mistakes": "More than 5 slices is hard to read. 3D pies distort proportions. Bar charts are often better.",
        "associated_tests": [
            "Chi-square GOF",
            "Binomial test",
            "Proportion tests",
            "Confidence intervals",
        ],
        "widget_function": pie_chart_widget,
    },
    "Area Graph": {
        "category": "Comparison Plots",
        "description": "A line chart with the area below the line filled, showing magnitude over time or categories.",
        "when_to_use": "Showing magnitude over time, comparing series contributions, cumulative trends, composition changes.",
        "interpretation": "Filled area = magnitude; stacked = total + composition; overlaid = shape comparison.",
        "common_mistakes": "More than 4 stacked series becomes unreadable. Overlaid areas need transparency.",
        "associated_tests": [
            "Time series decomposition",
            "Change point detection",
            "Trend analysis",
            "Intervention analysis",
        ],
        "widget_function": area_graph_widget,
    },
    "Contour Plot": {
        "category": "Multivariate Plots",
        "description": "2D density contours showing the distribution of bivariate data with color-filled regions.",
        "when_to_use": "Visualizing 2D distributions, identifying density peaks, replacing scatterplots for large n.",
        "interpretation": "Lines = constant density; closer lines = steeper gradient; peaks = dense regions.",
        "common_mistakes": "Bandwidth changes contours dramatically — use cross-validation to choose optimally.",
        "associated_tests": [
            "Bivariate normality test",
            "Hotelling's T-squared",
            "Multivariate outlier test",
            "KDE",
        ],
        "widget_function": contour_widget,
    },
    "Stacked Bar Chart": {
        "category": "Comparison Plots",
        "description": "Bars divided into sub-segments showing both total magnitude and category composition per group.",
        "when_to_use": "Showing group composition, comparing totals across groups, survey response breakdowns.",
        "interpretation": "Total bar = group total; segment = category contribution; 100% = proportions.",
        "common_mistakes": "Non-100% bars are hard to compare across different totals — normalize for fair comparison.",
        "associated_tests": [
            "Chi-square independence",
            "Fisher's exact test",
            "G-test",
            "Correspondence analysis",
        ],
        "widget_function": stacked_bar_widget,
    },
    "Population Pyramid": {
        "category": "Distribution Plots",
        "description": "Two back-to-back horizontal bar charts showing the age-sex distribution of a population.",
        "when_to_use": "Demographic analysis, population structure visualization, age-sex distribution, policy planning.",
        "interpretation": "Left = male, right = female; wide base = high birth rate; narrow top = lower life expectancy.",
        "common_mistakes": "Different scales on left/right axes mislead — always use the same scale on both sides.",
        "associated_tests": [
            "Chi-square independence",
            "Age standardization",
            "Dependency ratio",
            "Life table analysis",
        ],
        "widget_function": pop_pyramid_widget,
    },
    "Growth Curve Plot": {
        "category": "Regression Plots",
        "description": "Non-linear growth models (Logistic, Gompertz, Exponential) fitted to time-series data.",
        "when_to_use": "Population growth modeling, epidemic curves, learning curves, biological growth processes.",
        "interpretation": "S-curve = logistic; asymptote = carrying capacity; steepest point = max growth rate.",
        "common_mistakes": "Extrapolating beyond observed data is risky — the asymptote depends strongly on model choice.",
        "associated_tests": [
            "Non-linear regression F-test",
            "AIC/BIC comparison",
            "Residual diagnostics",
            "Bootstrap CIs",
        ],
        "widget_function": growth_curve_widget,
    },
    "Forest Plot": {
        "category": "Meta-Analysis Visualizations",
        "description": "Effect sizes with confidence intervals for multiple studies, used in meta-analysis.",
        "when_to_use": "Meta-analysis reporting, systematic review synthesis, comparing across multiple studies.",
        "interpretation": "Each row = one study; dot = effect size; line = 95% CI; red line = pooled estimate.",
        "common_mistakes": "Pooled estimate is the overall effect, NOT the average of individual study effect sizes.",
        "associated_tests": [
            "Cochran's Q test",
            "Higgins I² statistic",
            "Egger's test",
            "Meta-regression",
        ],
        "widget_function": forest_plot_widget,
    },
    "Funnel Plot": {
        "category": "Meta-Analysis Visualizations",
        "description": "Scatter plot of effect size vs precision to detect publication bias via funnel asymmetry.",
        "when_to_use": "Assessing publication bias, meta-analysis quality check, detecting small-study effects.",
        "interpretation": "Symmetric funnel = no bias; missing side = suppressed results; gap = publication bias.",
        "common_mistakes": "Funnel asymmetry is NOT always publication bias — heterogeneity or chance can cause it.",
        "associated_tests": [
            "Egger's regression test",
            "Begg's rank test",
            "Trim-and-fill",
            "Selection models",
        ],
        "widget_function": funnel_widget,
    },
    "Galbraith (Radial) Plot": {
        "category": "Meta-Analysis Visualizations",
        "description": "Plots precision vs z-score to identify outlier studies and assess heterogeneity.",
        "when_to_use": "Identifying outliers, assessing heterogeneity visually, complementing funnel plots.",
        "interpretation": "Slope = pooled effect; scatter around line = heterogeneity; outside band = outlier.",
        "common_mistakes": "Radial and funnel plot use correlated axes — interpret them together, not separately.",
        "associated_tests": [
            "Cochran's Q test",
            "I² heterogeneity",
            "H statistic",
            "Subgroup analysis",
        ],
        "widget_function": galbraith_widget,
    },
    "Baujat Plot": {
        "category": "Meta-Analysis Visualizations",
        "description": "Plots each study's contribution to heterogeneity vs its influence on the pooled effect.",
        "when_to_use": "Identifying influential studies, detecting heterogeneity sources, sensitivity analysis.",
        "interpretation": "Upper-right = high influence + heterogeneity; lower-left = well-fitting studies.",
        "common_mistakes": "Baujat plot is descriptive, not inferential — cut-offs are arbitrary, especially with k < 10.",
        "associated_tests": [
            "Leave-one-out analysis",
            "DFBETAS",
            "Cook's distance",
            "Q-test for heterogeneity",
        ],
        "widget_function": baujat_widget,
    },
    "Leave-One-Out Plot": {
        "category": "Meta-Analysis Visualizations",
        "description": "Shows how the pooled effect changes when each study is removed one at a time.",
        "when_to_use": "Identifying influential studies, sensitivity analysis, checking result robustness.",
        "interpretation": "Dot far from red line = influential study; stable estimates = robust meta-analysis.",
        "common_mistakes": "If removing a study changes the conclusion, that study drives the result — report both.",
        "associated_tests": [
            "Baujat plot",
            "DFBETAS",
            "Cook's distance",
            "Cumulative meta-analysis",
        ],
        "widget_function": loo_widget,
    },
    "Kaplan-Meier Curve": {
        "category": "Survival Analysis Plots",
        "description": "Step-function estimate of survival probability over time with censoring marks.",
        "when_to_use": "Time-to-event analysis, clinical trial comparison, estimating median survival.",
        "interpretation": "Steps down at events; tick marks = censored; lower curve = worse survival.",
        "common_mistakes": "KM curves beyond last event are unstable — always show number at risk.",
        "associated_tests": [
            "Log-rank test",
            "Wilcoxon-Gehan test",
            "Peto-Peto test",
            "Cox PH model",
        ],
        "widget_function": kaplan_meier_widget,
    },
    "Nelson-Aalen Plot": {
        "category": "Survival Analysis Plots",
        "description": "Non-parametric cumulative hazard estimate showing event accumulation over time.",
        "when_to_use": "Non-parametric hazard estimate, comparing hazard between groups, checking PH assumption.",
        "interpretation": "Stepped line = cumulative hazard; steeper = higher hazard; gap = treatment effect.",
        "common_mistakes": "Nelson-Aalen is cumulative hazard, not hazard rate — slope gives the rate.",
        "associated_tests": [
            "Log-rank test",
            "Cox PH model",
            "Schoenfeld residuals",
            "Cumulative hazard comparison",
        ],
        "widget_function": nelson_aalen_widget,
    },
    "Hazard Function Plot": {
        "category": "Survival Analysis Plots",
        "description": "Smoothed instantaneous hazard rate over time showing risk dynamics.",
        "when_to_use": "Modeling time-to-failure, understanding risk over time, reliability engineering.",
        "interpretation": "Hazard = instantaneous risk; increasing = wearing out; decreasing = early failures.",
        "common_mistakes": "Hazard is not a probability — it can exceed 1. It is a rate (events per time unit).",
        "associated_tests": [
            "Weibull distribution fit",
            "Exponential GOF",
            "Cox-Snell residuals",
            "Hazard shape tests",
        ],
        "widget_function": hazard_function_widget,
    },
    "Cumulative Hazard Plot": {
        "category": "Survival Analysis Plots",
        "description": "Cumulative hazard with optional log scale to check proportional hazards assumption.",
        "when_to_use": "Checking proportional hazards, estimating cumulative risk, model diagnostics.",
        "interpretation": "Log scale → parallel lines = PH holds; upward = increasing hazard; straight = constant.",
        "common_mistakes": "Log-cumulative hazard lines must be parallel for PH — crossing = non-PH.",
        "associated_tests": [
            "Cox PH model",
            "Schoenfeld residuals",
            "Log-cumulative hazard plot",
            "PH assumption check",
        ],
        "widget_function": cumulative_hazard_widget,
    },
    "Cox PH Effect Plot": {
        "category": "Survival Analysis Plots",
        "description": "Forest-style plot of hazard ratios with 95% CIs for multiple covariates.",
        "when_to_use": "Multivariable survival analysis, adjusting for confounders, identifying risk factors.",
        "interpretation": "Dot = estimated HR; line = 95% CI; HR > 1 = increased risk; red = significant.",
        "common_mistakes": "HR > 1 = higher hazard (worse). PH assumption must hold for valid interpretation.",
        "associated_tests": [
            "Wald test",
            "Likelihood ratio test",
            "Proportional hazards test",
            "Schoenfeld residuals",
        ],
        "widget_function": cox_ph_widget,
    },
    "Survival Probability Heatmap": {
        "category": "Survival Analysis Plots",
        "description": "Heatmap of predicted survival probabilities across time and a continuous covariate.",
        "when_to_use": "Visualizing covariate effects on survival, model predictions, identifying risk strata.",
        "interpretation": "Yellow = high survival; purple = low survival; row = covariate effect; column = time.",
        "common_mistakes": "Assumes proportional hazards and exponential baseline — real data may differ.",
        "associated_tests": [
            "Cox PH predictions",
            "Parametric survival models",
            "Time-dependent covariates",
            "Interactions",
        ],
        "widget_function": surv_heatmap_widget,
    },
    "Raincloud Plot": {
        "category": "Distribution Plots",
        "description": "Combines half-violin density cloud, jittered raw data (rain), and boxplot in one view.",
        "when_to_use": "Replace boxplots for richer distribution view; show raw data + density + summary simultaneously.",
        "interpretation": "Cloud = density shape; rain = individual points; box = median & IQR.",
        "common_mistakes": "KDE cloud unreliable with n < 15; raw dots alone are preferable for small samples.",
        "associated_tests": [
            "Independent t-test",
            "Mann-Whitney U",
            "Welch's t-test",
            "One-way ANOVA",
        ],
        "widget_function": raincloud_widget,
    },
    "Ridgeline Plot": {
        "category": "Distribution Plots",
        "description": "Stacked density plots for comparing distributions across multiple groups without occlusion.",
        "when_to_use": "Comparing many groups' distributions, showing change over time/conditions.",
        "interpretation": "Each ridge = group density; peak shift = group difference; width = variance.",
        "common_mistakes": "Can obscure details with >7 groups; use consistent bandwidth across groups.",
        "associated_tests": [
            "Kruskal-Wallis",
            "One-way ANOVA",
            "Fligner-Killeen",
            "Kolmogorov-Smirnov",
        ],
        "widget_function": ridgeline_widget,
    },
    "Sankey Diagram": {
        "category": "Comparison Plots",
        "description": "Flow diagram showing categorical transitions between states (pre→post, diagnostic pathways).",
        "when_to_use": "Showing subject flow, treatment pathways, pre-post transitions, diagnostic algorithms.",
        "interpretation": "Width = flow volume; nodes = decision points; thin paths = small subgroups.",
        "common_mistakes": "Too many nodes (>15) creates unreadable diagrams; group small categories into 'Other'.",
        "associated_tests": [
            "McNemar's test",
            "Chi-square",
            "Logistic regression",
            "Markov transition models",
        ],
        "widget_function": sankey_widget,
    },
    "Cleveland Dot Plot": {
        "category": "Comparison Plots",
        "description": "Dot-and-CI plot for comparing many categories without zero-baseline bar chart issues.",
        "when_to_use": "Comparing many categories, showing rankings with uncertainty, forest-plot style summaries.",
        "interpretation": "Dot = point estimate; bar = 95% CI; dashed line = overall mean; position = ranking.",
        "common_mistakes": "More than 25 categories becomes cluttered; use hierarchical grouping for large sets.",
        "associated_tests": [
            "One-sample t-test vs mean",
            "CI analysis",
            "Effect size comparison",
            "Meta-analysis",
        ],
        "widget_function": cleveland_dot_widget,
    },
    "Hexbin Plot": {
        "category": "Correlation Plots",
        "description": "2D hexagonal binning heatmap for visualizing dense scatter data without overplotting.",
        "when_to_use": "Large-n scatter data (n > 500), dense point clouds, 2D density visualization.",
        "interpretation": "Hexagon color = point density; bright = many overlapping points; shape reveals correlation.",
        "common_mistakes": "Hides individual outliers; always complement with a standard scatterplot for outlier detection.",
        "associated_tests": [
            "Pearson correlation",
            "Spearman correlation",
            "Linear regression",
            "2D Kolmogorov-Smirnov",
        ],
        "widget_function": hexbin_widget,
    },
    "Venn / Euler Diagram": {
        "category": "Comparison Plots",
        "description": "Set overlap visualization showing intersections between 2-3 categorical sets.",
        "when_to_use": "Gene set overlaps, diagnostic test agreement, survey response overlap, inclusion criteria.",
        "interpretation": "Circle size = set size; overlap = intersection; Jaccard = overlap/union similarity.",
        "common_mistakes": "Euler diagrams (area-proportional) preferred over Venn for more than 3 sets.",
        "associated_tests": [
            "Fisher's exact test",
            "Cohen's kappa",
            "Jaccard coefficient",
            "Overlap coefficient",
        ],
        "widget_function": venn_widget,
    },
    "Confidence Interval Comparison Plot": {
        "category": "Post-Hoc Plots",
        "description": "Dumbbell-style plot showing mean difference and 95% CI for each pairwise comparison with significance coloring.",
        "when_to_use": "After significant omnibus test, visualizing pairwise group differences with uncertainty.",
        "interpretation": "Diamond = mean difference; bar = CI; green = significant at α; dashed line at zero.",
        "common_mistakes": "Using unadjusted CIs for post-hoc inference; interpreting overlapping CIs as non-significant.",
        "associated_tests": ["Tukey HSD", "Bonferroni t-test", "Games-Howell", "Dunnett"],
        "widget_function": ph_ci_comparison_widget,
    },
    "Mean Difference Forest Plot": {
        "category": "Post-Hoc Plots",
        "description": "Classic forest plot of pairwise contrasts with unadjusted or multiplicity-corrected p-values and sortable by effect size.",
        "when_to_use": "Tukey/Games-Howell results, meta-analytic summaries, publication-quality pairwise comparisons.",
        "interpretation": "Points = mean difference; lines = 95% CI; labels show MD values; green = significant after correction.",
        "common_mistakes": "Comparing across different outcome scales; ignoring multiplicity correction when reporting.",
        "associated_tests": ["Tukey HSD", "Games-Howell", "Dunnett", "Bonferroni"],
        "widget_function": ph_forest_plot_widget,
    },
    "Compact Letter Display (CLD)": {
        "category": "Post-Hoc Plots",
        "description": "Groups sharing the same letter are not significantly different. Standard in agricultural and biological sciences.",
        "when_to_use": "Journal publication tables, summarizing complex pairwise comparisons in a compact format.",
        "interpretation": "Same letter = not significantly different; different letter = significantly different; two letters = intermediate.",
        "common_mistakes": "CLD depends on chosen α; ambiguous with many groups; letters can be deceptive with unbalanced designs.",
        "associated_tests": ["Tukey HSD", "Duncan's MRT", "Fisher LSD", "REGWR"],
        "widget_function": ph_cld_widget,
    },
    "Significance Heatmap": {
        "category": "Post-Hoc Plots",
        "description": "Color-coded matrix of adjusted p-values, Cohen's d, or binary significance for quick visual scanning of pairwise results.",
        "when_to_use": "Quick visual scan of all pairwise results, supplementing CLD tables with continuous information.",
        "interpretation": "Green = significant (p ≤ α); red = not significant; yellow = borderline; RdBu for Cohen's d shows direction.",
        "common_mistakes": "Using unadjusted p-values for inference after multiple tests; heatmap colors can be misleading with poor scaling.",
        "associated_tests": ["All post-hoc procedures"],
        "widget_function": ph_significance_heatmap_widget,
    },
    "Pairwise Network Graph": {
        "category": "Post-Hoc Plots",
        "description": "Interactive network where nodes = groups and edges = significant differences after Holm correction. Circle or spring layout.",
        "when_to_use": "Complex multi-group comparisons, presentation to non-statisticians, exploratory data analysis.",
        "interpretation": "Nodes = groups; green edges = significant difference; isolated node = not significantly different from any group.",
        "common_mistakes": "Edge thickness does not encode effect size; network layout can visually exaggerate small differences.",
        "associated_tests": ["Any post-hoc procedure"],
        "widget_function": ph_network_widget,
    },
    "Estimation Plot (Gardner-Altman)": {
        "category": "Post-Hoc Plots",
        "description": "Modern two-group visualization showing raw data paired with bootstrap-style mean difference and confidence interval.",
        "when_to_use": "Two-group comparisons where both raw data and effect size should be shown; modern reporting standards.",
        "interpretation": "Left: raw data with mean ± CI; right: mean difference diamond with CI; dashed line at zero.",
        "common_mistakes": "Not reporting the CI alongside the point estimate; misinterpreting the difference-axis scale.",
        "associated_tests": ["Independent t-test", "Welch's t-test", "Mann-Whitney U"],
        "widget_function": ph_estimation_widget,
    },
    "Raincloud Post Hoc Plot": {
        "category": "Post-Hoc Plots",
        "description": "Information-rich multi-group visualization combining half-violin density, boxplot, and raw data points.",
        "when_to_use": "Comparing distributions across groups, publication-quality visualizations that show all data features.",
        "interpretation": "Cloud = density distribution; box = median + IQR; points = raw data; rain = vertical data scatter.",
        "common_mistakes": "Hiding raw data behind summary statistics; overcrowding with too many groups or too few observations.",
        "associated_tests": ["One-way ANOVA", "Kruskal-Wallis", "All post-hoc methods"],
        "widget_function": ph_raincloud_widget,
    },
}

CATEGORIES = [
    "Distribution Plots",
    "Comparison Plots",
    "Correlation Plots",
    "Regression Plots",
    "Diagnostic Accuracy Plots",
    "Agreement Plots",
    "Multivariate Plots",
    "Survival Analysis Plots",
    "Meta-Analysis Visualizations",
    "Post-Hoc Plots",
]


def render_graph_explorer():
    st.title("Interactive Graph Explorer")
    st.write(
        "Explore statistical graphs interactively. Adjust controls to build visual intuition."
    )

    graph_category = st.sidebar.radio(
        "Graph Category",
        CATEGORIES,
        key="graph_category_radio",
    )

    category_graphs = {
        k: v for k, v in graphs.items() if v["category"] == graph_category
    }

    st.header(f"{graph_category}", divider="orange")

    for name, info in category_graphs.items():
        with st.expander(f"**{name}**", expanded=True):
            info["widget_function"]()
