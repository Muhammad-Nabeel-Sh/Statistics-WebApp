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




GRAPHS = {
    "Histogram": histogram_widget,
    "Density / KDE Plot": kde_widget,
    "Boxplot": boxplot_widget,
    "Violin Plot": violin_widget,
    "Q-Q Plot": qq_widget,
    "Normal P-P Plot": pp_widget,
    "Raincloud Plot": raincloud_widget,
    "Stem-and-Leaf Plot": stem_leaf_widget,
    "Frequency Polygon": freq_poly_widget,
    "Polar Density Plot": polar_density_widget,
    "Probability Density Function Plot": pdf_plot_widget,
    "Pareto Chart": pareto_widget,
    "Dot Plot": dot_plot_widget,
    "Population Pyramid": pop_pyramid_widget,
    "Ridgeline Plot": ridgeline_widget
}