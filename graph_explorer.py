import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
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
        skew = st.slider("Skewness", -3.0, 3.0, 0.0, 0.1, key="hist_skew",
                         help="Negative = left skew, Positive = right skew, 0 = symmetric")
        bins = st.slider("Bin Count", 5, 100, 30, key="hist_bins")
        show_kde = st.toggle("Show KDE Overlay", True, key="hist_kde")
        dist_type = st.selectbox("Distribution", ["Normal", "Skewed", "Uniform"], key="hist_dist")
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
    fig.add_trace(go.Histogram(x=data, nbinsx=bins, name="Frequency",
                               marker_color="#4C78A8", opacity=0.75))
    if show_kde:
        kde_x = np.linspace(min(data), max(data), 300)
        kde_y = stats.gaussian_kde(data)(kde_x)
        fig.add_trace(go.Scatter(x=kde_x, y=kde_y * n * (max(data) - min(data)) / bins,
                                 mode="lines", name="KDE", line=dict(color="#E45756", width=2)))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", legend=dict(orientation="h", y=1.1))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Bell shape = normal distribution\n"
                    "- Tail on right = positive skew\n"
                    "- Tail on left = negative skew\n"
                    "- Gaps = potential outliers/modes\n"
                    "- KDE = smooth density estimate")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Explore data distribution shape\n"
                       "- Check for normality or skewness\n"
                       "- Identify outliers and gaps\n"
                       "- Compare to theoretical distributions")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Shapiro-Wilk (normality)\n"
                       "- Kolmogorov-Smirnov\n"
                       "- Anderson-Darling\n"
                       "- t-test, ANOVA (assumptions)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Bin width dramatically changes appearance. "
                     "Too few bins hide detail, too many create noise. "
                     "Always try multiple bin widths before interpreting shape.")


def kde_widget():
    st.markdown("## Density / KDE Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 1000, 300, key="kde_n")
        bw = st.select_slider("Bandwidth", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0], value=0.5, key="kde_bw",
                              help="Smaller = more detail (risk overfitting), Larger = smoother")
        multimodal = st.toggle("Show Multimodal", False, key="kde_multi")
        dist_type = st.selectbox("Distribution", ["Normal", "Skewed", "Bimodal"], key="kde_dist")
    np.random.seed(42)
    if dist_type == "Normal":
        data = np.random.normal(0, 1, n)
    elif dist_type == "Skewed":
        data = np.random.gamma(2, 1, n)
    else:
        data = np.concatenate([np.random.normal(-2, 0.7, n // 2), np.random.normal(2, 0.7, n // 2)])
    if multimodal and dist_type != "Bimodal":
        data = np.concatenate([data, np.random.normal(3, 0.5, n // 2)])
    kde_x = np.linspace(min(data) - 1, max(data) + 1, 500)
    try:
        kde = stats.gaussian_kde(data, bw_method=bw)
        kde_y = kde(kde_x)
    except Exception:
        kde_y = np.zeros_like(kde_x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, fill="tozeroy", mode="lines",
                             name="Density", line=dict(color="#4C78A8", width=3),
                             hovertemplate="x = %{x:.2f}<br>density = %{y:.4f}<extra></extra>"))
    rug_x = data
    rug_y = np.full_like(rug_x, -0.02)
    fig.add_trace(go.Scatter(x=rug_x, y=rug_y, mode="markers", name="Data",
                             marker=dict(color="#E45756", size=3, symbol="line-ns-open")))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", yaxis_title="Density")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Peak = most common value region\n"
                    "- Spread = variance of data\n"
                    "- Multiple peaks = multimodal\n"
                    "- Bandwidth controls smoothness")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Compare distributions smoothly\n"
                       "- Identify modes and shape\n"
                       "- Assess normality visually\n"
                       "- Overlay multiple groups")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- t-test / ANOVA\n"
                       "- Kruskal-Wallis\n"
                       "- Kolmogorov-Smirnov\n"
                       "- Permutation tests")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "KDE can oversmooth bimodal data with large bandwidth, "
                     "hiding real structure. Conversely, small bandwidth "
                     "can create spurious peaks from noise.")


def boxplot_widget():
    st.markdown("## Boxplot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 10, 200, 50, key="box_n")
        n_groups = st.selectbox("Number of Groups", [1, 2, 3, 4], index=2, key="box_ng")
        show_outliers = st.toggle("Show Outliers", True, key="box_out")
        spread = st.slider("Variance Spread", 0.2, 3.0, 1.0, 0.1, key="box_spread",
                           help="Higher = more variability between groups")
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
        fig.add_trace(go.Box(y=d, name=name, marker_color=px.colors.qualitative.Plotly[i],
                             boxpoints="outliers" if show_outliers else False,
                             hovertemplate=f"{name}<br>y = %{{y:.2f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", yaxis_title="Value")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Center line = median\n"
                    "- Box = IQR (middle 50%)\n"
                    "- Whiskers = 1.5xIQR range\n"
                    "- Dots = potential outliers\n"
                    "- Non-overlapping = group diff")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Compare groups at a glance\n"
                       "- Identify outliers visually\n"
                       "- Assess symmetry & spread\n"
                       "- Before parametric tests")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- t-test (2 groups)\n"
                       "- ANOVA (3+ groups)\n"
                       "- Mann-Whitney U\n"
                       "- Kruskal-Wallis")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Overlapping boxes do NOT guarantee non-significance. "
                     "Median outside the other's box is a rough heuristic, "
                     "not a formal test. Always verify with proper testing.")


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
        fig.add_trace(go.Violin(y=d, name=name, box_visible=show_box, meanline_visible=True,
                                line_color=px.colors.qualitative.Plotly[i],
                                fillcolor=px.colors.qualitative.Plotly[i], opacity=0.6,
                                hovertemplate=f"{name}<br>y = %{{y:.2f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", yaxis_title="Value")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Width = density of observations\n"
                    "- Narrow sections = sparse data\n"
                    "- Wide sections = dense data\n"
                    "- Symmetry = normal distribution\n"
                    "- Boxplot inside = summary stats")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Replace boxplot for distribution shape\n"
                       "- Detect multimodality in groups\n"
                       "- Compare shape + spread + central tendency\n"
                       "- Assess normality assumptions")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- ANOVA\n"
                       "- Kruskal-Wallis\n"
                       "- Welch's t-test\n"
                       "- Permutation test")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Violin plots can be misleading with very small samples "
                     "(n < 15) because the KDE over-interprets sparse data. "
                     "Use boxplots or stripcharts for small samples.")


def qq_widget():
    st.markdown("## Q-Q Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 500, 100, key="qq_n")
        dist_type = st.selectbox("Data Distribution", ["Normal", "Heavy-tailed (t)", "Skewed", "Uniform"],
                                 key="qq_dist")
        tail_param = st.slider("Tail Heaviness / Skew", 0.5, 5.0, 2.0, 0.1, key="qq_tail",
                               help="t dist: lower df = heavier tails. Skew: higher = more skew.")
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
    fig.add_trace(go.Scatter(x=theoretical, y=os, mode="markers", name="Observed",
                             marker=dict(color="#4C78A8", size=5),
                             hovertemplate="Theoretical: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>"))
    line_x = np.linspace(min(theoretical), max(theoretical), 10)
    line_y = np.linspace(min(os), max(os), 10)
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="y=x (Normal)",
                             line=dict(color="#E45756", dash="dash")))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Theoretical Quantiles", yaxis_title="Observed Quantiles",
                      annotations=[dict(x=0.05, y=0.95, xref="paper", yref="paper",
                                        text=f"r = {r:.3f}", showarrow=False,
                                        font=dict(color="white", size=14))])
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Points on diagonal = normal\n"
                    "- S-curve = heavy/light tails\n"
                    "- Curve above line = right skew\n"
                    "- Curve below line = left skew\n"
                    "- r near 1 approx normal")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Check normality assumption\n"
                       "- Identify distribution shape\n"
                       "- Detect outliers in tails\n"
                       "- Before t-test, ANOVA, regression")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Shapiro-Wilk\n"
                       "- Kolmogorov-Smirnov\n"
                       "- Anderson-Darling\n"
                       "- D'Agostino-Pearson")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Small samples (n < 30) produce noisy Q-Q plots that "
                     "can look non-normal even when data are normal. "
                     "Always complement with a formal test for small n.")


# --- COMPARISON PLOTS ---


def grouped_bar_widget():
    st.markdown("## Grouped Bar Chart")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_groups = st.selectbox("Number of Groups", [2, 3, 4], index=1, key="gb_n")
        n_cats = st.selectbox("Number of Categories", [2, 3, 4], index=2, key="gb_cats")
        effect = st.slider("Group Difference Effect", 0.0, 3.0, 1.0, 0.1, key="gb_eff",
                           help="How much groups differ from each other")
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
        fig.add_trace(go.Bar(name=groups[g_i], x=categories, y=data[:, g_i],
                             error_y=dict(type="data", array=errors[:, g_i]) if show_error else None,
                             marker_color=px.colors.qualitative.Plotly[g_i]))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      barmode="group", hovermode="x unified")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Bar height = group mean\n"
                    "- Error bars = variability\n"
                    "- Non-overlapping = likely sig.\n"
                    "- Pattern consistency across cats")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Compare groups across categories\n"
                       "- Show means with uncertainty\n"
                       "- Present descriptive results\n"
                       "- Visualize interaction effects")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Two-way ANOVA\n"
                       "- Welch\'s t-test\n"
                       "- Mixed-effects models\n"
                       "- Post-hoc comparisons")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Bars should always start at zero to avoid misleading "
                     "visual amplification of differences. Truncated y-axes "
                     "exaggerate small effects.")


def error_bar_widget():
    st.markdown("## Error Bar Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 5, 100, 30, key="eb_n")
        n_groups = st.selectbox("Number of Groups", [2, 3, 4, 5], index=1, key="eb_ng")
        ci_width = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01, key="eb_ci",
                             help="Width of confidence interval")
        effect = st.slider("Effect Size (Cohen\'s d)", 0.0, 2.0, 0.5, 0.1, key="eb_eff")
    np.random.seed(42)
    n_g = int(n_groups)
    means_arr = [i * effect * 0.5 for i in range(n_g)]
    z = stats.norm.ppf(1 - (1 - ci_width) / 2)
    fig = go.Figure()
    for i in range(n_g):
        d = np.random.normal(means_arr[i], 0.3, n)
        sem = np.std(d, ddof=1) / np.sqrt(n)
        fig.add_trace(go.Scatter(x=[f"Group {chr(65 + i)}"], y=[np.mean(d)],
                                 error_y=dict(type="data", array=[z * sem], thickness=1.5, width=10),
                                 mode="markers", marker=dict(size=12, color=px.colors.qualitative.Plotly[i]),
                                 name=f"Group {chr(65 + i)}"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", yaxis_title="Mean +/- CI")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Dot = group mean\n"
                    "- Bar = confidence interval\n"
                    "- Non-overlapping CIs approx p < .05\n"
                    "- Wide CI = imprecise estimate")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Show precision of estimates\n"
                       "- Compare group means visually\n"
                       "- Present meta-analysis results\n"
                       "- Forest plots")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- t-test\n"
                       "- ANOVA\n"
                       "- Welch\'s t-test\n"
                       "- Linear regression")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Non-overlapping 95% CIs are more conservative than "
                     "p < 0.05 - they approximately correspond to p < 0.01. "
                     "Overlapping CIs do NOT guarantee non-significance.")


def paired_line_widget():
    st.markdown("## Paired Line Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Pairs", 5, 100, 20, key="pl_n")
        effect = st.slider("Pre-Post Effect", -2.0, 2.0, 0.8, 0.1, key="pl_eff",
                           help="How much values change from pre to post")
        noise = st.slider("Within-Pair Noise", 0.1, 2.0, 0.5, 0.1, key="pl_noise")
        show_means = st.toggle("Show Mean Line", True, key="pl_mean")
    np.random.seed(42)
    pre = np.random.normal(0, 1, n)
    post = pre + effect + np.random.normal(0, noise, n)
    fig = go.Figure()
    for i in range(n):
        fig.add_trace(go.Scatter(x=["Pre", "Post"], y=[pre[i], post[i]], mode="lines+markers",
                                 line=dict(color="rgba(200,200,200,0.3)", width=1),
                                 marker=dict(size=4, color="rgba(200,200,200,0.5)"),
                                 showlegend=False, hovertemplate=f"Subject {i+1}<br>" + "%{x}: %{y:.2f}<extra></extra>"))
    if show_means:
        fig.add_trace(go.Scatter(x=["Pre", "Post"], y=[np.mean(pre), np.mean(post)],
                                 mode="lines+markers", name="Mean Change",
                                 line=dict(color="#E45756", width=4),
                                 marker=dict(size=10, color="#E45756", symbol="diamond")))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", yaxis_title="Value", showlegend=True)
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each line = one subject\n"
                    "- Upward = increase over time\n"
                    "- Downward = decrease over time\n"
                    "- Thick line = group average\n"
                    "- Consistent direction = effect")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Pre-post intervention studies\n"
                       "- Within-subject designs\n"
                       "- Repeated measures (2 time points)\n"
                       "- Crossover trials")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Paired t-test\n"
                       "- Wilcoxon Signed-Rank\n"
                       "- Repeated measures ANOVA\n"
                       "- Linear mixed models")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Ignoring the paired nature of the data. Using an "
                     "independent t-test on paired data inflates Type II "
                     "error and ignores within-subject correlation.")


def boxplot_comp_widget():
    st.markdown("## Boxplot Comparison")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 10, 200, 50, key="bpc_n")
        n_groups = st.selectbox("Number of Groups", [2, 3, 4], index=1, key="bpc_ng")
        effect = st.slider("Between-Group Effect", 0.0, 3.0, 0.8, 0.1, key="bpc_eff")
        var_ratio = st.slider("Variance Ratio (group 1 : last)", 0.2, 5.0, 1.0, 0.1, key="bpc_var",
                              help="Heterogeneity of variance across groups")
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
        fig.add_trace(go.Box(y=d, name=name, marker_color=px.colors.qualitative.Plotly[i],
                             boxpoints="outliers", hovertemplate=f"{name}<br>y = %{{y:.2f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", yaxis_title="Value")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Compare medians across groups\n"
                    "- Box overlap approx group similarity\n"
                    "- Box size = group variability\n"
                    "- Whisker length = tail behavior")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Compare multiple groups robustly\n"
                       "- Check equal variance assumption\n"
                       "- Identify group-level outliers\n"
                       "- Explore pilot data")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- One-way ANOVA\n"
                       "- Welch\'s ANOVA\n"
                       "- Kruskal-Wallis\n"
                       "- Levene\'s test (variance)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Boxplots hide distribution shape entirely - "
                     "different distributions can produce identical boxplots "
                     "(Anscombe\'s quartet). Always check with a violin plot too.")


def violin_comp_widget():
    st.markdown("## Violin Comparison")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size per Group", 10, 200, 60, key="vic_n")
        n_groups = st.selectbox("Number of Groups", [2, 3], index=1, key="vic_ng")
        effect = st.slider("Group Effect", 0.0, 3.0, 1.0, 0.1, key="vic_eff")
        shape_diff = st.toggle("Show Shape Difference", False, key="vic_shape",
                               help="Make groups have different distribution shapes")
        split = st.toggle("Split Violin (Side-by-side)", False, key="vic_split")
    np.random.seed(42)
    n_g = int(n_groups)
    groups = []
    for i in range(n_g):
        if shape_diff and i == 1:
            d = np.random.gamma(2, 0.5, n) + i * effect
        elif shape_diff and i == 2:
            d = np.concatenate([np.random.normal(i * effect - 1, 0.3, n // 2),
                                np.random.normal(i * effect + 1, 0.3, n // 2)])
        else:
            d = np.random.normal(i * effect, 0.5, n)
        groups.append(d)
    names_a = [f"Group {chr(65 + i)}" for i in range(n_g)]
    fig = go.Figure()
    for i, d in enumerate(groups):
        side = "positive" if (split and i == 0) else "negative" if (split and i == 1) else "both"
        fig.add_trace(go.Violin(y=d, name=names_a[i], box_visible=True, meanline_visible=True,
                                line_color=px.colors.qualitative.Plotly[i],
                                fillcolor=px.colors.qualitative.Plotly[i], opacity=0.6,
                                side=side,
                                hovertemplate=f"{names_a[i]}<br>y = %{{y:.2f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", yaxis_title="Value")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Width = density of data\n"
                    "- Shift = group difference\n"
                    "- Shape difference = distributional diff\n"
                    "- Box inside shows median & IQR")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Replace boxplot for detailed comparison\n"
                       "- Detect shape differences between groups\n"
                       "- Assess normality & equal variance\n"
                       "- Visualize 2-group comparisons")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Independent t-test\n"
                       "- Mann-Whitney U\n"
                       "- Welch\'s t-test\n"
                       "- Fligner-Killeen (variance)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Violin width is often misinterpreted as frequency. "
                     "It represents density - a wide violin does not mean "
                     "more data, it means data are more spread out.")


# --- CORRELATION PLOTS ---


def scatter_widget():
    st.markdown("## Scatterplot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 500, 100, key="scat_n")
        r = st.slider("Correlation (r)", -1.0, 1.0, 0.5, 0.05, key="scat_r",
                       help="Strength and direction of relationship")
        noise = st.slider("Noise", 0.05, 1.0, 0.2, 0.05, key="scat_noise",
                          help="Higher = more scatter around the line")
        add_outlier = st.toggle("Add Outlier", False, key="scat_out")
        show_line = st.toggle("Show Regression Line", True, key="scat_line")
    np.random.seed(42)
    x, y = _gen_corr(n, r, noise, outlier=add_outlier)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                             marker=dict(color="#4C78A8", size=6, opacity=0.7),
                             hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>"))
    if show_line:
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        line_x = np.linspace(min(x), max(x), 100)
        line_y = slope * line_x + intercept
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name=f"r = {r_val:.3f}",
                                 line=dict(color="#E45756", width=2)))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", xaxis_title="X", yaxis_title="Y")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Upward slope = positive r\n"
                    "- Downward slope = negative r\n"
                    "- Tight cluster = strong correlation\n"
                    "- Wide scatter = weak correlation\n"
                    "- Single outlier can change r")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Assess relationship direction & strength\n"
                       "- Detect outliers and non-linearity\n"
                       "- Check homoscedasticity\n"
                       "- Explore two continuous variables")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Pearson correlation\n"
                       "- Spearman correlation\n"
                       "- Linear regression\n"
                       "- Correlation test")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Correlation does NOT imply causation. "
                     "A strong r could be spurious, confounded, "
                     "or driven by outliers. Always visualize first.")


def heatmap_widget():
    st.markdown("## Correlation Heatmap")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n_vars = st.selectbox("Number of Variables", [4, 5, 6, 7, 8], index=1, key="hm_nvars")
        strength = st.slider("Average Correlation Strength", 0.0, 1.0, 0.5, 0.05, key="hm_str",
                             help="How strongly variables are correlated on average")
        show_annot = st.toggle("Show Correlation Values", True, key="hm_annot")
        cluster = st.toggle("Show Clustered Order", False, key="hm_cluster",
                            help="Group similar correlations together")
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
    fig = go.Figure(data=go.Heatmap(z=corr, x=labels, y=labels,
                                     text=np.round(corr, 2) if show_annot else None,
                                     texttemplate="%{text}" if show_annot else None,
                                     colorscale="RdBu_r", zmin=-1, zmax=1,
                                     hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="", yaxis_title="")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Red = positive correlation\n"
                    "- Blue = negative correlation\n"
                    "- Dark = strong relationship\n"
                    "- White/light = weak relationship\n"
                    "- Diagonal = variable with itself (r=1)")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Check multicollinearity\n"
                       "- Explore many variable relationships\n"
                       "- Identify variable clusters\n"
                       "- Before regression / factor analysis")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Pearson correlation matrix\n"
                       "- Spearman correlation matrix\n"
                       "- Variance Inflation Factor (VIF)\n"
                       "- Factor analysis")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Visual patterns can be misleading when variables "
                     "have different scales. Always use standardized "
                     "(rank or z-score) data for correlation heatmaps.")


def bubble_widget():
    st.markdown("## Bubble Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Points", 10, 200, 50, key="bub_n")
        r = st.slider("X-Y Correlation", -1.0, 1.0, 0.3, 0.05, key="bub_r")
        size_effect = st.slider("Size-Value Correlation", 0.0, 1.0, 0.6, 0.05, key="bub_size",
                                help="How strongly bubble size relates to y-value")
        show_color = st.toggle("Color by Third Variable", True, key="bub_color")
    np.random.seed(42)
    n_actual = int(n)
    x, y_base = _gen_corr(n_actual, r, 0.3)
    size = np.random.uniform(5, 40, n_actual) * (1 + size_effect * (y_base - np.mean(y_base)))
    size = np.clip(size, 5, 80)
    color = np.random.uniform(0, 100, n_actual) if show_color else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_base, mode="markers",
                             marker=dict(size=size, color=color, colorscale="Viridis",
                                         showscale=show_color, opacity=0.7,
                                         line=dict(color="white", width=0.5)),
                             hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<br>size = %{marker.size:.0f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", xaxis_title="X", yaxis_title="Y")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Position = X-Y relationship\n"
                    "- Bubble size = third variable\n"
                    "- Color = fourth variable\n"
                    "- Larger bubbles draw attention")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Show 3-4 variables at once\n"
                       "- Highlight weighted importance\n"
                       "- Population/economic data\n"
                       "- Risk-benefit visualizations")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Weighted correlation\n"
                       "- Multiple regression\n"
                       "- Weighted least squares\n"
                       "- Meta-analysis (forest plots)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Bubble area (not diameter) should encode the third "
                     "variable. Using diameter exaggerates differences "
                     "and misleads the viewer.")


def monotonic_widget():
    st.markdown("## Monotonic vs Linear Correlation")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 300, 100, key="mono_n")
        rel_type = st.selectbox("Relationship Type",
                                ["Linear", "Quadratic (U-shape)", "Exponential", "Sine (Periodic)"],
                                key="mono_type")
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
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                             marker=dict(color="#4C78A8", size=5, opacity=0.7),
                             hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>"))
    title = ""
    if show_pearson:
        title += f"Pearson r = {r_p:.3f}"
    if show_spearman:
        if title:
            title += " | "
        title += f"Spearman rho = {r_s:.3f}"
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", xaxis_title="X", yaxis_title="Y",
                      title=title if title else None)
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Pearson = linear relationship\n"
                    "- Spearman = monotonic relationship\n"
                    "- Big diff (r vs rho) = non-linear\n"
                    "- r near 0 but rho large = monotonic")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Test if relationship is linear\n"
                       "- Choose Pearson vs Spearman\n"
                       "- Detect non-linear patterns\n"
                       "- Understand correlation choice")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Pearson correlation\n"
                       "- Spearman correlation\n"
                       "- Kendall\'s tau\n"
                       "- Distance correlation")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Pearson r near 0 does NOT mean no relationship - "
                     "it only means no linear relationship. Always "
                     "visualize: a U-shape can have r approx 0.")


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
            fig.add_trace(go.Scatter(x=[xi, xi], y=[yi, ypi], mode="lines",
                                     line=dict(color="rgba(200,200,200,0.3)", width=1),
                                     showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                             marker=dict(color="#4C78A8", size=6, opacity=0.7),
                             hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>"))
    if show_ci:
        t_val = stats.t.ppf(0.975, n - 2)
        pred_se = np.sqrt(se**2 * (1 + 1 / n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2)))
        ci = t_val * pred_se
        fig.add_trace(go.Scatter(x=x_line, y=y_line + ci, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_line, y=y_line - ci, mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(228, 87, 86, 0.15)",
                                 name="95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                             name=f"beta1={slope_est:.2f}, R-squared={r_val**2:.3f}",
                             line=dict(color="#E45756", width=2)))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", xaxis_title="X", yaxis_title="Y")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Line = best fit (minimizes residuals)\n"
                    "- Band = 95% CI for prediction\n"
                    "- Slope = change in Y per unit X\n"
                    "- R-squared = prop of variance explained\n"
                    "- p-value = test if slope != 0")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Model continuous outcome\n"
                       "- Estimate effect size (beta1)\n"
                       "- Predict Y from X\n"
                       "- Test linear relationship")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- F-test (overall model)\n"
                       "- t-test (coefficient)\n"
                       "- Pearson correlation\n"
                       "- ANOVA (nested models)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Extrapolating beyond the observed x-range. "
                     "Linear relationships may not hold outside "
                     "the data - never predict beyond your data.")


def multiple_reg_widget():
    st.markdown("## Multiple Regression Surface")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="mr_n")
        b1 = st.slider("Coefficient beta1", -2.0, 2.0, 0.7, 0.1, key="mr_b1")
        b2 = st.slider("Coefficient beta2", -2.0, 2.0, 0.5, 0.1, key="mr_b2")
        noise = st.slider("Noise", 0.1, 3.0, 0.5, 0.1, key="mr_noise")
        show_interaction = st.toggle("Show Interaction", False, key="mr_interact",
                                     help="Include beta12 x X1 x X2 term")
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
    fig.add_trace(go.Scatter3d(x=x1, y=x2, z=y, mode="markers", name="Data",
                               marker=dict(size=4, color="#4C78A8", opacity=0.7),
                               hovertemplate="X1=%{x:.2f}<br>X2=%{y:.2f}<br>Y=%{z:.2f}<extra></extra>"))
    fig.add_trace(go.Surface(x=grid, y=grid, z=Y_pred, name="Predicted",
                             colorscale="Reds", opacity=0.5, showscale=False))
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10),
                      scene=dict(xaxis_title="X1", yaxis_title="X2", zaxis_title="Y",
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Plane = predicted Y from X1, X2\n"
                    "- Slope along X1 = beta1 (holding X2 constant)\n"
                    "- Slope along X2 = beta2 (holding X1 constant)\n"
                    "- Twisted surface = interaction")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Model multiple predictors\n"
                       "- Control for confounders\n"
                       "- Test interaction effects\n"
                       "- Understand partial effects")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- F-test (overall model)\n"
                       "- Partial F-test (nested models)\n"
                       "- t-test (individual coefficients)\n"
                       "- VIF (multicollinearity)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Coefficients are sensitive to predictor scaling. "
                     "Standardize (z-score) predictors before comparing "
                     "coefficient magnitudes. Unstandardized betas "
                     "depend on the unit of X.")


def logistic_widget():
    st.markdown("## Logistic Regression Sigmoid Curve")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="logit_n")
        separation = st.slider("Class Separation", 0.5, 5.0, 2.0, 0.1, key="logit_sep",
                               help="How far apart the two classes are on X")
        threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05, key="logit_thresh")
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
        fig.add_trace(go.Scatter(x=x_grid, y=y_prob, mode="lines", name="P(Y=1)",
                                 line=dict(color="#4C78A8", width=3),
                                 hovertemplate="X = %{x:.2f}<br>P(Y=1) = %{y:.3f}<extra></extra>"))
        fig.add_hline(y=threshold, line_dash="dash", line_color="#E45756",
                      annotation_text=f"Threshold = {threshold}")
    if show_data:
        jitter = np.random.uniform(-0.05, 0.05, n)
        fig.add_trace(go.Scatter(x=x, y=y + jitter, mode="markers",
                                 marker=dict(color=y, colorscale="RdBu", size=6,
                                             showscale=False, opacity=0.7),
                                 name="Data",
                                 hovertemplate="X = %{x:.2f}<br>Y = %{y:.0f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", xaxis_title="Predictor (X)", yaxis_title="Probability")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- S-curve = logistic probability\n"
                    "- Steep curve = strong predictor\n"
                    "- Shallow curve = weak predictor\n"
                    "- Threshold = classification cutoff\n"
                    "- Above threshold -> predict class 1")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Binary outcome prediction\n"
                       "- Estimate odds ratios\n"
                       "- Medical diagnosis models\n"
                       "- Risk factor analysis")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Likelihood ratio test\n"
                       "- Wald test (coefficients)\n"
                       "- Hosmer-Lemeshow (calibration)\n"
                       "- ROC-AUC (discrimination)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Changing the decision threshold changes "
                     "sensitivity and specificity. Default 0.5 "
                     "is not always optimal - adjust based on "
                     "the cost of false positives vs false negatives.")


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
    centers = [[np.cos(2 * np.pi * i / k) * separation, np.sin(2 * np.pi * i / k) * separation] for i in range(k)]
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
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                             np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, showscale=False,
                                 colorscale=[[i/(k-1), colors[i]] for i in range(k)],
                                 opacity=0.3, name="Decision Regions",
                                 hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<extra></extra>"))
    for i in range(k):
        mask = y == i
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers",
                                 name=f"Class {i}", marker=dict(color=colors[i], size=6, opacity=0.8),
                                 hovertemplate=f"Class {i}<br>x1=%{{x:.2f}}<br>x2=%{{y:.2f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", xaxis_title="Feature 1", yaxis_title="Feature 2")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Colored regions = decision zones\n"
                    "- Boundaries = where model is uncertain\n"
                    "- Overlap = classification difficulty\n"
                    "- Filled area = multinomial probability")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Multi-class classification\n"
                       "- Understand decision boundaries\n"
                       "- Compare classifier geometries\n"
                       "- Feature space exploration")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Multinomial logistic regression\n"
                       "- MANOVA\n"
                       "- Discriminant analysis\n"
                       "- Classification metrics")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Linear decision boundaries (logistic regression) "
                     "cannot separate non-linear class patterns. "
                     "If classes are interleaved, consider non-linear "
                     "methods (kernels, trees, neural nets).")


def ordinal_logit_widget():
    st.markdown("## Ordinal Logistic Probability Curves")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="ord_n")
        n_levels = st.selectbox("Number of Ordinal Levels", [3, 4, 5], index=1, key="ord_k")
        effect = st.slider("Predictor Effect", 0.0, 3.0, 1.0, 0.1, key="ord_eff")
        show_cumulative = st.toggle("Show Cumulative Probabilities", False, key="ord_cum")
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
            fig.add_trace(go.Scatter(x=x_grid, y=cum_grid[i], mode="lines",
                                     name=f"P(Y <= {i+1})", line=dict(dash="dash", color=colors[i]),
                                     hovertemplate="X = %{x:.2f}<br>P(Y <= %s) = %{y:.3f}<extra></extra>" % (i+1)))
    for i in range(k):
        prob_i = cum_grid[i] if i == 0 else [cum_grid[i][j] - cum_grid[i-1][j] for j in range(200)]
        fig.add_trace(go.Scatter(x=x_grid, y=prob_i, mode="lines", name=f"P(Y = {i+1})",
                                 line=dict(color=colors[i], width=2.5),
                                 fill="tozeroy" if not show_cumulative else None,
                                 hovertemplate=f"X = %{{x:.2f}}<br>P(Y={i+1}) = %{{y:.3f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", xaxis_title="Predictor", yaxis_title="Probability")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each curve = probability of one level\n"
                    "- Curves shift with predictor value\n"
                    "- Non-parallel = proportional odds violation\n"
                    "- Steep transition = strong predictor")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Ordered categorical outcomes\n"
                       "- Likert scale responses\n"
                       "- Disease severity staging\n"
                       "- Patient-reported outcomes")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Proportional odds test\n"
                       "- Brant test (parallel regression)\n"
                       "- Likelihood ratio test\n"
                       "- Score test")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "The proportional odds assumption (parallel curves) "
                     "must hold. If curves cross or are non-parallel, "
                     "a generalized ordered logit or multinomial model "
                     "is needed.")


def poisson_widget():
    st.markdown("## Poisson Count Regression")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 300, 100, key="pois_n")
        base_rate = st.slider("Base Rate (intercept)", 0.5, 5.0, 2.0, 0.1, key="pois_base",
                              help="Expected count when X = 0")
        effect = st.slider("Effect (log-rate ratio)", 0.0, 2.0, 0.5, 0.05, key="pois_eff",
                           help="Multiplicative effect per unit X")
        show_mean = st.toggle("Show Mean Curve", True, key="pois_mean")
        show_overdisp = st.toggle("Add Overdispersion", False, key="pois_over",
                                  help="Extra-Poisson variability")
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
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed",
                             marker=dict(color="#4C78A8", size=5, opacity=0.6),
                             hovertemplate="X = %{x:.2f}<br>Count = %{y:.0f}<extra></extra>"))
    if show_mean:
        fig.add_trace(go.Scatter(x=x_grid, y=y_grid, mode="lines", name="Predicted Mean",
                                 line=dict(color="#E45756", width=3),
                                 hovertemplate="X = %{x:.2f}<br>Mean = %{y:.1f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", xaxis_title="Predictor", yaxis_title="Count")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Y-axis = count (0, 1, 2, ...)\n"
                    "- Curve = predicted mean count\n"
                    "- Spread increases with mean\n"
                    "- Clustering at zero = zero-inflation")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Modeling count outcomes\n"
                       "- Event frequencies\n"
                       "- Rare disease incidence\n"
                       "- Hospital readmissions")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Likelihood ratio test\n"
                       "- Wald test\n"
                       "- Deviance goodness-of-fit\n"
                       "- Dispersion test")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Poisson assumes mean = variance. If variance > mean "
                     "(overdispersion), use Negative Binomial. Many zeros? "
                     "Consider zero-inflated or hurdle models.")


# --- DIAGNOSTIC ACCURACY PLOTS ---


def confusion_widget():
    st.markdown("## Confusion Matrix Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 1000, 200, key="cm_n")
        prevalence = st.slider("Prevalence (True Class 1 Rate)", 0.05, 0.95, 0.3, 0.05, key="cm_prev",
                               help="Proportion of actual positives")
        sensitivity = st.slider("Sensitivity (True Positive Rate)", 0.5, 1.0, 0.85, 0.01, key="cm_sens")
        specificity = st.slider("Specificity (True Negative Rate)", 0.5, 1.0, 0.90, 0.01, key="cm_spec")
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
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=true_labels,
                                     text=[[f"{cm[0][0]}", f"{cm[0][1]}"],
                                           [f"{cm[1][0]}", f"{cm[1][1]}"]],
                                     texttemplate="%{text}", textfont=dict(size=16),
                                     colorscale="Blues", showscale=False,
                                     hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Predicted", yaxis_title="Actual")
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
            st.info("**Interpretation**\n\n"
                    "- Diagonal = correct predictions\n"
                    "- Off-diagonal = errors\n"
                    "- Top-right (FP) = Type I error\n"
                    "- Bottom-left (FN) = Type II error\n"
                    "- PPV depends on prevalence")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Evaluate binary classifiers\n"
                       "- Compare diagnostic tests\n"
                       "- Understand error types\n"
                       "- Choose operating threshold")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- McNemar\'s test (paired comparison)\n"
                       "- Cohen\'s Kappa\n"
                       "- ROC-AUC\n"
                       "- Diagnostic likelihood ratios")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Accuracy is misleading with imbalanced classes. "
                     "A test with 95% accuracy on 5% prevalence "
                     "could be useless (always predict negative). "
                     "Always report PPV, NPV, and prevalence.")


def roc_widget():
    st.markdown("## ROC Curve Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="roc_n")
        auc = st.slider("AUC (Area Under Curve)", 0.5, 1.0, 0.85, 0.01, key="roc_auc",
                        help="0.5 = random, 1.0 = perfect")
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
    fig.add_trace(go.Scatter(x=fpr_s, y=tpr_s, mode="lines", name=f"ROC (AUC = {auc_actual:.3f})",
                             line=dict(color="#4C78A8", width=3),
                             hovertemplate="FPR = %{x:.3f}<br>TPR = %{y:.3f}<extra></extra>"))
    if show_chance:
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                 line=dict(color="rgba(200,200,200,0.5)", dash="dash")))
    if show_threshold:
        youden = tpr_s - fpr_s
        best_idx = np.argmax(youden)
        fig.add_trace(go.Scatter(x=[fpr_s[best_idx]], y=[tpr_s[best_idx]],
                                 mode="markers", name="Optimal Threshold",
                                 marker=dict(color="#E45756", size=12, symbol="star")))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", xaxis_title="False Positive Rate (1 - Specificity)",
                      yaxis_title="True Positive Rate (Sensitivity)",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Curve closer to top-left = better\n"
                    "- AUC = probability correct ranking\n"
                    "- AUC 0.5 = guessing\n"
                    "- AUC 0.8+ = good discrimination\n"
                    "- Star = Youden\'s optimal threshold")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Compare diagnostic tests\n"
                       "- Assess model discrimination\n"
                       "- Choose optimal threshold\n"
                       "- Meta-analysis of test accuracy")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- DeLong test (compare AUCs)\n"
                       "- Hanley-McNeil test\n"
                       "- Bootstrap AUC comparison\n"
                       "- Sensitivity analysis")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "AUC ignores calibration - a model can have "
                     "high AUC but poorly calibrated probabilities. "
                     "Always check calibration (calibration plot) "
                     "alongside ROC analysis.")


def pr_curve_widget():
    st.markdown("## Precision-Recall Curve")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 200, key="pr_n")
        prevalence = st.slider("Prevalence (Class Imbalance)", 0.02, 0.5, 0.1, 0.01, key="pr_prev",
                               help="Lower = more imbalanced")
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
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR Curve",
                             line=dict(color="#4C78A8", width=3), fill="tozeroy",
                             hovertemplate="Recall = %{x:.3f}<br>Precision = %{y:.3f}<extra></extra>"))
    if show_baseline:
        fig.add_hline(y=baseline, line_dash="dash", line_color="rgba(200,200,200,0.5)",
                      annotation_text=f"Baseline (Prevalence = {baseline:.2%})")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", xaxis_title="Recall (Sensitivity)",
                      yaxis_title="Precision (PPV)",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Higher curve = better\n"
                    "- Baseline = always-predict-positive\n"
                    "- PR better than ROC for imbalanced\n"
                    "- AP = area under PR curve")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Imbalanced classification\n"
                       "- Rare disease detection\n"
                       "- Fraud/anomaly detection\n"
                       "- When PPV matters more")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Average Precision (AP)\n"
                       "- F1 score (harmonic of P,R)\n"
                       "- F-beta score (weighted F1)\n"
                       "- Bootstrap PR comparison")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "PR curves from small samples are noisy and "
                     "can show high precision simply by chance. "
                     "Always use confidence bands (bootstrap) "
                     "when sample size is limited.")


def threshold_widget():
    st.markdown("## Sensitivity vs Specificity Threshold Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 200, key="thresh_n")
        sep = st.slider("Class Separation", 0.5, 3.0, 1.5, 0.1, key="thresh_sep")
        prevalence = st.slider("Prevalence", 0.05, 0.95, 0.3, 0.05, key="thresh_prev")
        cost_fp = st.slider("Cost of FP (relative to FN)", 0.1, 10.0, 1.0, 0.1, key="thresh_cost",
                            help="Higher = penalize false positives more")
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
    fig.add_trace(go.Scatter(x=thresholds, y=sens, mode="lines", name="Sensitivity",
                             line=dict(color="#4C78A8", width=2),
                             hovertemplate="Threshold = %{x:.2f}<br>Sensitivity = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=thresholds, y=spec, mode="lines", name="Specificity",
                             line=dict(color="#E45756", width=2),
                             hovertemplate="Threshold = %{x:.2f}<br>Specificity = %{y:.3f}<extra></extra>"))
    fig.add_vline(x=thresholds[opt_idx], line_dash="dash", line_color="green",
                  annotation_text=f"Optimal = {thresholds[opt_idx]:.2f}")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", xaxis_title="Threshold", yaxis_title="Rate",
                      yaxis=dict(range=[0, 1]))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Blue = sensitivity (catch positives)\n"
                    "- Red = specificity (avoid false alarms)\n"
                    "- Tradeoff: increase one = decrease other\n"
                    "- Green = optimal based on costs")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Choose diagnostic cutoff\n"
                       "- Balance sensitivity vs specificity\n"
                       "- Incorporate cost of errors\n"
                       "- Laboratory test thresholds")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- ROC analysis\n"
                       "- Youden\'s index\n"
                       "- Cost-benefit analysis\n"
                       "- Decision curve analysis")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Youden\'s index (max sensitivity + specificity - 1) "
                     "treats FP and FN equally. In medicine, FN is often "
                     "more costly (missed diagnosis). Adjust threshold "
                     "based on clinical consequences, not statistics.")


def calibration_widget():
    st.markdown("## Calibration Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 100, 2000, 500, key="cal_n")
        calibration = st.slider("Calibration Slope", 0.0, 2.0, 1.0, 0.05, key="cal_slope",
                                help="1.0 = perfect, < 1 = overconfident, > 1 = underconfident")
        noise = st.slider("Calibration Noise", 0.0, 0.3, 0.05, 0.01, key="cal_noise")
        n_bins = st.slider("Number of Bins", 5, 20, 10, key="cal_bins")
    np.random.seed(42)
    true_probs = np.random.uniform(0.05, 0.95, n)
    pred_probs = true_probs ** calibration
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
    fig.add_trace(go.Scatter(x=bin_mean_pred, y=bin_mean_obs, mode="markers+lines",
                             name="Model", marker=dict(color="#4C78A8", size=10),
                             line=dict(color="#4C78A8", width=2),
                             hovertemplate="Predicted = %{x:.3f}<br>Observed = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration",
                             line=dict(color="rgba(200,200,200,0.5)", dash="dash")))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified", xaxis_title="Predicted Probability",
                      yaxis_title="Observed Proportion",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Points on diagonal = perfectly calibrated\n"
                    "- Above diagonal = underestimated probability\n"
                    "- Below diagonal = overestimated probability\n"
                    "- Slope < 1 = overconfident (common)")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Assess probability accuracy\n"
                       "- Check model reliability\n"
                       "- Compare risk prediction models\n"
                       "- Before clinical deployment")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Hosmer-Lemeshow test\n"
                       "- Brier score\n"
                       "- Spiegelhalter z-test\n"
                       "- Calibration intercept & slope")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Hosmer-Lemeshow test is sensitive to binning "
                     "choices and sample size. Large samples will "
                     "reject even well-calibrated models. Use "
                     "calibration plots + intercept/slope instead.")


# --- AGREEMENT PLOTS ---


def bland_altman_widget():
    st.markdown("## Bland-Altman Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="ba_n")
        bias = st.slider("Systematic Bias (Mean Diff)", -2.0, 2.0, 0.2, 0.1, key="ba_bias",
                         help="Average difference between methods")
        proportional_bias = st.slider("Proportional Bias", -1.0, 1.0, 0.0, 0.05, key="ba_prop",
                                      help="Bias that changes with measurement magnitude")
        limits_factor = st.slider("Limits of Agreement Multiplier", 1.0, 3.0, 1.96, 0.01, key="ba_loa",
                                  help="1.96 approx 95% limits")
    np.random.seed(42)
    true_val = np.random.uniform(0, 20, n)
    m1 = true_val + np.random.normal(0, 0.5, n)
    m2 = true_val + bias + proportional_bias * (true_val - 10) + np.random.normal(0, 0.5, n)
    mean = (m1 + m2) / 2
    diff = m1 - m2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    upper_loa = mean_diff + limits_factor * std_diff
    lower_loa = mean_diff - limits_factor * std_diff
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean, y=diff, mode="markers", name="Differences",
                             marker=dict(color="#4C78A8", size=7, opacity=0.7),
                             hovertemplate="Mean = %{x:.2f}<br>Difference = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=mean_diff, line=dict(color="#E45756", width=2),
                  annotation_text=f"Mean Diff = {mean_diff:.3f}")
    fig.add_hline(y=upper_loa, line=dict(color="rgba(200,200,200,0.7)", dash="dash"),
                  annotation_text=f"+{limits_factor}SD = {upper_loa:.3f}")
    fig.add_hline(y=lower_loa, line=dict(color="rgba(200,200,200,0.7)", dash="dash"),
                  annotation_text=f"-{limits_factor}SD = {lower_loa:.3f}")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", xaxis_title="Mean of Two Measurements",
                      yaxis_title="Difference (Method 1 - Method 2)")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- y = 0 line = perfect agreement\n"
                    "- Mean diff = systematic bias\n"
                    "- Dashed lines = limits of agreement\n"
                    "- Fan-shape = proportional bias\n"
                    "- 95% of points should be within limits")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Compare measurement methods\n"
                       "- Assess test-retest reliability\n"
                       "- Medical device validation\n"
                       "- Laboratory method comparison")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Paired t-test (fixed bias)\n"
                       "- Intraclass Correlation (ICC)\n"
                       "- Deming regression\n"
                       "- Passing-Bablok regression")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Bland-Altman assesses agreement, not correlation. "
                     "High r does NOT mean good agreement - two methods "
                     "can be perfectly correlated but have a large bias. "
                     "Always use Bland-Altman for method comparison.")


def kappa_widget():
    st.markdown("## Cohen\'s Kappa Agreement Matrix")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="kap_n")
        n_cats = st.selectbox("Number of Categories", [2, 3, 4], index=0, key="kap_cats")
        agreement = st.slider("Agreement Rate", 0.0, 1.0, 0.7, 0.05, key="kap_agree",
                              help="Proportion of perfect agreement between raters")
        marginal_bias = st.slider("Marginal Bias", 0.0, 0.5, 0.1, 0.01, key="kap_bias",
                                  help="How much raters prefer different categories")
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
    p_e = np.sum(row_marg * col_marg) / n_total ** 2
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    labels = [f"Cat {i + 1}" for i in range(k)]
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels,
                                     text=cm, texttemplate="%{text}",
                                     textfont=dict(size=14),
                                     colorscale="Blues", showscale=False,
                                     hovertemplate="Rater1: %{y}<br>Rater2: %{x}<br>Count: %{z}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Rater 2", yaxis_title="Rater 1")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Cohen\'s kappa", f"{kappa:.3f}")
    m2.metric("Observed Agreement", f"{p_o:.1%}")
    m3.metric("Chance Agreement", f"{p_e:.1%}")
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Diagonal = perfect agreement\n"
                    "- Off-diagonal = disagreement\n"
                    "- kappa < 0 = worse than chance\n"
                    "- kappa 0.4-0.6 = moderate\n"
                    "- kappa > 0.8 = near perfect")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Inter-rater reliability\n"
                       "- Diagnostic agreement studies\n"
                       "- Psychiatric diagnostic assessment\n"
                       "- Image/scan rating agreement")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Weighted Kappa (ordinal)\n"
                       "- Fleiss\' Kappa (3+ raters)\n"
                       "- McNemar\'s test (2x2 bias)\n"
                       "- ICC (continuous)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Kappa is prevalence-dependent - rare categories "
                     "produce low kappa even with high agreement. "
                     "Also, kappa penalizes raters differently: two "
                     "raters can have 80% agreement but kappa = 0.5. "
                     "Always report agreement rate alongside kappa.")


def icc_widget():
    st.markdown("## ICC Visualization")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Number of Subjects", 5, 50, 20, key="icc_n")
        n_raters = st.selectbox("Number of Raters/Occasions", [2, 3, 4], index=0, key="icc_r")
        icc_true = st.slider("True ICC", 0.0, 1.0, 0.6, 0.05, key="icc_true",
                             help="Proportion of variance due to between-subject differences")
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
    icc_est = (ms_between - ms_within) / (ms_between + (n_r - 1) * ms_within) if (ms_between + (n_r - 1) * ms_within) > 0 else 0
    fig = go.Figure()
    for i in range(n_s):
        ys = data[i]
        xs = [f"Rater {j + 1}" for j in range(n_r)]
        if show_subject:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                     line=dict(color="rgba(200,200,200,0.3)", width=1),
                                     marker=dict(size=5, color=px.colors.qualitative.Plotly[i % 10]),
                                     showlegend=False,
                                     hovertemplate=f"Subject {i+1}<br>Rater=%{{x}}<br>Value=%{{y:.2f}}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                                     marker=dict(size=7, color=px.colors.qualitative.Plotly[i % 10]),
                                     name=f"S{i + 1}",
                                     hovertemplate=f"Subject {i+1}<br>Rater=%{{x}}<br>Value=%{{y:.2f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", xaxis_title="Rater", yaxis_title="Measurement")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    col_est, _ = st.columns(2)
    col_est.metric("Estimated ICC", f"{icc_est:.3f}", delta=f"Target = {icc_true}")
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each color = one subject\n"
                    "- Lines horizontal = high ICC\n"
                    "- Lines crossing = low ICC\n"
                    "- Tight clustering within subject\n"
                    "- Wide spread within subject = error")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Test-retest reliability\n"
                       "- Inter-rater reliability (continuous)\n"
                       "- Intra-rater reliability\n"
                       "- Longitudinal measurement stability")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- ICC(1,1) - single rater absolute\n"
                       "- ICC(2,1) - consistency\n"
                       "- ICC(3,k) - average random raters\n"
                       "- Bland-Altman (agreement)")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "High ICC can mask systematic bias (raters "
                     "consistently disagreeing). ICC measures "
                     "relative consistency, not absolute agreement. "
                     "Combine with Bland-Altman for full assessment.")



# --- MULTIVARIATE PLOTS ---


def pca_widget():
    st.markdown("## PCA Scatter Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="pca_n")
        n_groups = st.selectbox("Number of Groups", [2, 3, 4], index=1, key="pca_groups")
        separation = st.slider("Group Separation", 0.0, 5.0, 2.0, 0.1, key="pca_sep")
        n_features = st.selectbox("Number of Features", [5, 10, 20], index=0, key="pca_feat")
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
        fig.add_trace(go.Scatter(x=X_pca[mask, 0], y=X_pca[mask, 1], mode="markers",
                                 name=f"Group {i}", marker=dict(color=colors[i], size=5, opacity=0.7),
                                 hovertemplate=f"Group {i}<br>PC1 = %{{x:.2f}}<br>PC2 = %{{y:.2f}}<extra></extra>"))
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
                x_e = x_c + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
                y_e = y_c + a * np.sin(theta) * np.cos(angle) + b * np.cos(theta) * np.sin(angle)
                fig.add_trace(go.Scatter(x=x_e, y=y_e, mode="lines",
                                         line=dict(color=colors[i], width=1.5, dash="dash"),
                                         showlegend=False))
            except Exception:
                pass
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest",
                      xaxis_title=f"PC1 ({var_expl[0]:.1%} variance)",
                      yaxis_title=f"PC2 ({var_expl[1]:.1%} variance)")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- PCs = directions of max variance\n"
                    "- Close points = similar profiles\n"
                    "- Separated groups = distinct clusters\n"
                    "- Ellipses = 95% confidence region\n"
                    "- Axis labels show % variance explained")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Reduce dimensionality\n"
                       "- Visualize high-dim data\n"
                       "- Check for natural clusters\n"
                       "- Exploratory data analysis")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- MANOVA (on PCs or raw)\n"
                       "- Factor analysis\n"
                       "- K-means clustering\n"
                       "- PERMANOVA")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "PCA assumes linear relationships. Non-linear "
                     "structure (e.g., U-shape, spiral) will NOT "
                     "be captured. Use t-SNE or UMAP for non-linear "
                     "dimensionality reduction.")


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
            fig.add_trace(go.Scatter(x=g[:, 0], y=g[:, 1], mode="markers", name=name,
                                     marker=dict(color=color, size=5, opacity=0.7),
                                     hovertemplate=f"{name}<br>DV1 = %{{x:.2f}}<br>DV2 = %{{y:.2f}}<extra></extra>"))
            if show_centroids:
                fig.add_trace(go.Scatter(x=[np.mean(g[:, 0])], y=[np.mean(g[:, 1])],
                                         mode="markers", marker=dict(color=color, size=15, symbol="x"),
                                         showlegend=False))
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                          hovermode="closest", xaxis_title="DV1", yaxis_title="DV2")
    else:
        for i, (g, name, color) in enumerate(zip(groups, names, colors)):
            fig.add_trace(go.Scatter3d(x=g[:, 0], y=g[:, 1], z=g[:, 2], mode="markers", name=name,
                                       marker=dict(color=color, size=4, opacity=0.7),
                                       hovertemplate=f"{name}<br>DV1=%{{x:.2f}}<br>DV2=%{{y:.2f}}<br>DV3=%{{z:.2f}}<extra></extra>"))
            if show_centroids:
                fig.add_trace(go.Scatter3d(x=[np.mean(g[:, 0])], y=[np.mean(g[:, 1])], z=[np.mean(g[:, 2])],
                                           mode="markers", marker=dict(color=color, size=8, symbol="diamond"),
                                           showlegend=False))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10),
                          scene=dict(xaxis_title="DV1", yaxis_title="DV2", zaxis_title="DV3",
                                     camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each color = one group\n"
                    "- Distance between = group difference\n"
                    "- Overlap = no significant difference\n"
                    "- X marks = group centroid (mean)\n"
                    "- Ellipsoid shape = covariance")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Compare groups on multiple DVs\n"
                       "- Control for correlated outcomes\n"
                       "- Multivariate experimental design\n"
                       "- Protect against inflated Type I error")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Pillai's Trace\n"
                       "- Wilks' Lambda\n"
                       "- Hotelling-Lawley Trace\n"
                       "- Roy's Largest Root")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "MANOVA requires multivariate normality and "
                     "homogeneity of covariance matrices. Violations "
                     "inflate Type I error. When assumptions fail, "
                     "consider PERMANOVA or non-parametric alternatives.")


def cluster_widget():
    st.markdown("## Cluster Visualization")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 30, 500, 150, key="clust_n")
        k_true = st.selectbox("True Number of Clusters", [2, 3, 4, 5], index=1, key="clust_k")
        separation = st.slider("Cluster Separation", 0.5, 5.0, 2.0, 0.1, key="clust_sep")
        n_features = st.selectbox("Project from Features", [2, 5, 10], index=0, key="clust_feat")
        show_centers = st.toggle("Show Cluster Centers", True, key="clust_center")
    k, n_f = int(k_true), int(n_features)
    np.random.seed(42)
    if n_f == 2:
        centers = [[np.cos(2 * np.pi * i / k) * separation, np.sin(2 * np.pi * i / k) * separation] for i in range(k)]
        X = np.vstack([np.random.normal(centers[i], 0.5, (n // k, 2)) for i in range(k)])
    else:
        X_list = []
        for i in range(k):
            center = np.random.uniform(-separation, separation, n_f) * (i / max(k - 1, 1))
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
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers",
                                 name=f"Cluster {i}", marker=dict(color=colors[i], size=5, opacity=0.6),
                                 hovertemplate=f"Cluster {i}<br>x = %{{x:.2f}}<br>y = %{{y:.2f}}<extra></extra>"))
    if show_centers:
        fig.add_trace(go.Scatter(x=centers2[:, 0], y=centers2[:, 1], mode="markers",
                                 marker=dict(color="black", size=12, symbol="x", line=dict(color="white", width=1)),
                                 name="Centers"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="closest", xaxis_title="PC1 / Feature 1", yaxis_title="PC2 / Feature 2")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each color = discovered cluster\n"
                    "- X marks = cluster center\n"
                    "- Tight clusters = well-separated\n"
                    "- Overlap = ambiguous assignment\n"
                    "- K-means assumes spherical clusters")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Discover natural groupings\n"
                       "- Segment patients/populations\n"
                       "- Pattern recognition\n"
                       "- Exploratory data mining")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Silhouette score\n"
                       "- Elbow method (WCSS)\n"
                       "- Gap statistic\n"
                       "- Davies-Bouldin index")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "K-means requires specifying k beforehand and "
                     "assumes spherical, equally-sized clusters. "
                     "Elongated or irregular clusters will be "
                     "incorrectly split. Use DBSCAN or GMM for "
                     "complex shapes.")


def scatter3d_widget():
    st.markdown("## 3D Scatter Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="s3d_n")
        n_groups = st.selectbox("Number of Groups", [1, 2, 3], index=1, key="sd3_groups")
        correlation = st.slider("Variable Correlation", -1.0, 1.0, 0.5, 0.05, key="s3d_corr",
                                help="Correlation among the 3 dimensions")
        spread = st.slider("Data Spread", 0.2, 3.0, 1.0, 0.1, key="s3d_spread")
        rotate = st.toggle("Auto-Rotate", True, key="s3d_rotate")
    k = int(n_groups)
    np.random.seed(42)
    cov_mat = np.array([[1, correlation, correlation],
                        [correlation, 1, correlation],
                        [correlation, correlation, 1]])
    eigvals = np.linalg.eigvalsh(cov_mat)
    if min(eigvals) <= 0:
        cov_mat += np.eye(3) * (abs(min(eigvals)) + 0.01)
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    for i in range(k):
        offset = np.full(3, i * spread * 1.5)
        data = np.random.multivariate_normal(offset, cov_mat * spread, n // k)
        fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode="markers",
                                   name=f"Group {chr(65 + i)}" if k > 1 else "Data",
                                   marker=dict(color=colors[i] if k > 1 else "#4C78A8",
                                               size=4, opacity=0.7),
                                   hovertemplate=f"{'Group ' + chr(65 + i) if k > 1 else 'Point'}"
                                                 f"<br>X=%{{x:.2f}}<br>Y=%{{y:.2f}}<br>Z=%{{z:.2f}}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10),
                      scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each axis = one variable\n"
                    "- Position in 3D space = multi-dim profile\n"
                    "- Clusters = groups with similar profiles\n"
                    "- Rotation reveals different patterns\n"
                    "- Elliptical shape = correlated variables")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Explore 3-variable relationships\n"
                       "- Identify 3D clusters\n"
                       "- Present multivariate patterns\n"
                       "- Interactive data exploration")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- MANOVA\n"
                       "- Multivariate regression\n"
                       "- Canonical correlation\n"
                       "- 3D PCA visualization")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "3D plots can obscure patterns depending on "
                     "viewing angle. Always rotate and view from "
                     "multiple perspectives. Pre-projected 2D views "
                     "(PCA) often reveal structure more clearly.")


def pp_widget():
    st.markdown("## Normal P-P Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="pp_n")
        dist_type = st.selectbox("Distribution", ["Normal", "Skewed", "Heavy-tailed", "Uniform"], key="pp_dist")
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
    fig.add_trace(go.Scatter(x=theo_p, y=emp_p, mode="markers",
                             marker=dict(color="#4C78A8", size=4),
                             name="Observed",
                             hovertemplate="Theoretical P=%{x:.3f}<br>Empirical P=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="red", dash="dash"),
                             name="Ideal (Normal)"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Theoretical Cumulative Probability",
                      yaxis_title="Empirical Cumulative Probability",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Points on diagonal = normal\n"
                    "- S-curve = heavy tails\n"
                    "- Points above = right skew\n"
                    "- Points below = left skew")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Assess normality assumption\n"
                       "- Complement to Q-Q plot\n"
                       "- Sensitive to center deviations\n"
                       "- Visual goodness-of-fit check")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Shapiro-Wilk test\n"
                       "- Kolmogorov-Smirnov\n"
                       "- Anderson-Darling\n"
                       "- D'Agostino-Pearson")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "P-P plots are less sensitive to tail "
                     "departures than Q-Q plots. Use both "
                     "for a complete normality assessment.")


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
        fig.add_trace(go.Scatter(x=x_jitter, y=d, mode="markers",
                                 marker=dict(color=colors[i], size=4, opacity=0.4),
                                 legendgroup=f"g{i}", name=f"Group {i+1}", showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Box(x0=i, y=d, name=f"Group {i+1}",
                             marker_color=colors[i], line=dict(color=colors[i], width=2),
                             fillcolor="rgba(0,0,0,0)", boxpoints=False, width=0.15,
                             legendgroup=f"g{i}", showlegend=False))
        fig.add_trace(go.Violin(x0=i, y=d, side="positive",
                                line=dict(color=colors[i], width=2),
                                fillcolor=colors[i], opacity=0.3,
                                points=False, width=0.6,
                                legendgroup=f"g{i}", name=f"Group {i+1}",
                                showlegend=True))
    fig.update_layout(template="plotly_dark", height=450,
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(tickmode="array", tickvals=list(range(k)),
                                 ticktext=[f"Group {i+1}" for i in range(k)]),
                      yaxis_title="Value", hovermode="closest")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Points = raw data (cloud)\n"
                    "- Box = median + IQR\n"
                    "- Half-violin = density shape\n"
                    "- Combines all three views")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Replace boxplot for more detail\n"
                       "- Show both distribution and raw data\n"
                       "- Modern publication-ready graphics\n"
                       "- Small to moderate sample sizes")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Independent t-test\n"
                       "- Mann-Whitney U\n"
                       "- Welch's t-test\n"
                       "- Permutation tests")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Jitter width is arbitrary and only "
                     "shows density not exact x-position. "
                     "Set random seed for reproducibility.")


def residuals_fitted_widget():
    st.markdown("## Residuals vs Fitted Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="resid_n")
        noise = st.slider("Noise Level", 0.1, 3.0, 1.0, 0.1, key="resid_noise")
        pattern = st.selectbox("Pattern", ["Linear (OK)", "Heteroscedastic", "Non-linear", "Outlier"], key="resid_pattern")
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
    fig.add_trace(go.Scatter(x=fitted, y=residuals, mode="markers",
                             marker=dict(color="#4C78A8", size=5),
                             name="Residuals",
                             hovertemplate="Fitted=%{x:.2f}<br>Residual=%{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="red", dash="dash"), opacity=0.7)
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Fitted Values", yaxis_title="Residuals")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Random scatter around 0 = OK\n"
                    "- Fan shape = heteroscedasticity\n"
                    "- U-shape = non-linearity\n"
                    "- Isolated points = outliers")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- After fitting linear regression\n"
                       "- Check homoscedasticity assumption\n"
                       "- Check linearity assumption\n"
                       "- Identify influential points")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Breusch-Pagan test\n"
                       "- Goldfeld-Quandt test\n"
                       "- RESET test\n"
                       "- Cook's distance")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Patterned residuals indicate model "
                     "misspecification — do NOT interpret "
                     "coefficients until residuals are "
                     "well-behaved.")


def poly_reg_widget():
    st.markdown("## Polynomial Regression Fit")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 300, 80, key="poly_reg_n")
        degree = st.slider("Polynomial Degree", 1, 10, 1, key="poly_reg_deg")
        noise = st.slider("Noise Level", 0.1, 3.0, 0.5, 0.1, key="poly_reg_noise")
        true_fn = st.selectbox("True Relationship", ["Linear", "Quadratic", "Cubic", "Sine"], key="poly_reg_fn")
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
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                             marker=dict(color="#4C78A8", size=5, opacity=0.6),
                             name="Data",
                             hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x_smooth, y=y_pred, mode="lines",
                             line=dict(color="#E45756", width=3),
                             name=f"Degree {degree}",
                             hovertemplate="x=%{x:.2f}<br>Pred=%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x_smooth, y=y_true, mode="lines",
                             line=dict(color="gray", width=2, dash="dot"),
                             name="True Function",
                             hovertemplate="x=%{x:.2f}<br>True=%{y:.2f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="X", yaxis_title="Y")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Higher degree = more flexible\n"
                    "- Degree 1 = straight line\n"
                    "- Degree 2 = one bend\n"
                    "- Degree 10 can overfit wildly")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Model non-linear relationships\n"
                       "- Test for curvature in data\n"
                       "- Understand bias-variance tradeoff\n"
                       "- Teaching overfitting concepts")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- F-test for nested models\n"
                       "- Cross-validation MSE\n"
                       "- AIC / BIC comparison\n"
                       "- ANOVA model comparison")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "High-degree polynomials overfit "
                     "near boundaries. Never extrapolate "
                     "beyond data range. Use splines "
                     "for better behavior.")


def reg_path_widget():
    st.markdown("## Regularization Path")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 100, key="regpath_n")
        n_features = st.slider("Number of Features", 5, 30, 10, key="regpath_k")
        reg_type = st.selectbox("Regularization", ["Lasso (L1)", "Ridge (L2)"], key="regpath_type")
    np.random.seed(42)
    true_coefs = np.zeros(n_features)
    true_coefs[:5] = [3, -2, 1.5, -1, 0.5]
    np.random.shuffle(true_coefs)
    X = np.random.normal(0, 1, (n, n_features))
    y = X @ true_coefs + np.random.normal(0, 1, n)
    from sklearn.linear_model import Lasso, Ridge
    alphas = np.logspace(-2, 2, 100)
    if reg_type == "Lasso (L1)":
        coefs = np.array([Lasso(alpha=a, max_iter=10000).fit(X, y).coef_ for a in alphas])
    else:
        coefs = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
    fig = go.Figure()
    for i in range(n_features):
        fig.add_trace(go.Scatter(x=np.log10(alphas), y=coefs[:, i], mode="lines",
                                 line=dict(width=1.5),
                                 name=f"Feature {i+1}",
                                 hovertemplate="log10(α)=%{x:.2f}<br>Coeff=%{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="log10(Alpha)", yaxis_title="Coefficient Value",
                      hovermode="closest")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each line = one coefficient\n"
                    "- Left (low α) = unregularized\n"
                    "- Right (high α) = strong shrinkage\n"
                    "- Lasso forces coefficients to zero")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- High-dimensional data\n"
                       "- Feature selection (Lasso)\n"
                       "- Combat multicollinearity (Ridge)\n"
                       "- Bias-variance tradeoff analysis")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Cross-validated MSE\n"
                       "- Regularization path stability\n"
                       "- Bayesian information criterion\n"
                       "- Bootstrap coefficient stability")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Lasso selects at most n features. "
                     "With p > n, Ridge may generalize "
                     "better. Always standardize predictors "
                     "before regularization.")


def splom_widget():
    st.markdown("## Scatterplot Matrix (SPLOM)")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 30, 500, 150, key="splom_n")
        n_vars = st.selectbox("Number of Variables", [3, 4, 5, 6], index=1, key="splom_k")
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
    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10, r=10, t=30, b=10))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each panel = bivariate scatter\n"
                    "- Diagonal = each variable vs itself\n"
                    "- Tight ellipse = strong correlation\n"
                    "- Row/col patterns = multivariate structure")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Explore many variable pairs at once\n"
                       "- Detect multicollinearity patterns\n"
                       "- Identify multivariate outliers\n"
                       "- EDA for high-dimensional data")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- Pearson correlation matrix\n"
                       "- Variance Inflation Factor (VIF)\n"
                       "- Bartlett's sphericity test\n"
                       "- MANOVA assumptions check")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "SPLOMs become unreadable with >10 "
                     "variables. Use correlation heatmap "
                     "or PCA for higher dimensions.")


def parallel_coords_widget():
    st.markdown("## Parallel Coordinates Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="parcoords_n")
        n_dims = st.selectbox("Number of Dimensions", [4, 5, 6, 7, 8], index=1, key="parcoords_k")
        n_clusters = st.selectbox("Number of Clusters", [2, 3, 4], index=0, key="parcoords_clust")
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
    fig = px.parallel_coordinates(df, color="Cluster", dimensions=col_names,
                                  color_continuous_scale="Viridis")
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10))
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info("**Interpretation**\n\n"
                    "- Each vertical axis = one variable\n"
                    "- Each line = one observation\n"
                    "- Crossing lines = negative correlation\n"
                    "- Parallel lines = positive correlation")
        with col_w:
            st.success("**When To Use**\n\n"
                       "- Visualize high-dimensional data\n"
                       "- Identify variable relationships\n"
                       "- Find multivariate patterns\n"
                       "- Complement to PCA")
        with col_t:
            st.warning("**Associated Tests**\n\n"
                       "- MANOVA\n"
                       "- Canonical correlation\n"
                       "- Discriminant analysis\n"
                       "- Cluster validation")
        with col_m:
            st.error("**Common Mistake**\n\n"
                     "Axis order affects interpretation. "
                     "Reorder axes to highlight patterns. "
                     "Too many observations creates clutter "
                     "— consider sampling.")


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
        "associated_tests": ["Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling", "t-test (assumptions)"],
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
        "associated_tests": ["ANOVA", "Kruskal-Wallis", "Welch's t-test", "Permutation test"],
        "widget_function": violin_widget,
    },
    "Q-Q Plot": {
        "category": "Distribution Plots",
        "description": "Plots observed quantiles against theoretical normal quantiles to assess normality.",
        "when_to_use": "Checking normality assumption, identifying distribution shape, detecting tail outliers.",
        "interpretation": "Points on diagonal = normal; S-curve = heavy tails; curve above = right skew.",
        "common_mistakes": "Small samples (n < 30) produce noisy Q-Q plots even when data are normal.",
        "associated_tests": ["Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling", "D'Agostino-Pearson"],
        "widget_function": qq_widget,
    },
    "Grouped Bar Chart": {
        "category": "Comparison Plots",
        "description": "Disaggregates group means across categories with side-by-side bars.",
        "when_to_use": "Comparing groups across categories, showing means with uncertainty bars.",
        "interpretation": "Bar height = mean; error bars = variability; non-overlapping approx significant.",
        "common_mistakes": "Bars should always start at zero to avoid misleading visual differences.",
        "associated_tests": ["Two-way ANOVA", "Welch's t-test", "Mixed-effects models", "Post-hoc comparisons"],
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
        "associated_tests": ["Paired t-test", "Wilcoxon Signed-Rank", "Repeated measures ANOVA", "Linear mixed models"],
        "widget_function": paired_line_widget,
    },
    "Boxplot Comparison": {
        "category": "Comparison Plots",
        "description": "Side-by-side boxplots for comparing multiple groups robustly.",
        "when_to_use": "Comparing groups robustly, checking equal variance, identifying group outliers.",
        "interpretation": "Compare medians; box overlap approx group similarity; box size = variability.",
        "common_mistakes": "Different distributions can produce identical boxplots (Anscombe's quartet).",
        "associated_tests": ["One-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis", "Levene's test"],
        "widget_function": boxplot_comp_widget,
    },
    "Violin Comparison": {
        "category": "Comparison Plots",
        "description": "Side-by-side violins showing distribution shape, spread, and central tendency across groups.",
        "when_to_use": "Detailed group comparison, detecting shape differences, assessing normality assumptions.",
        "interpretation": "Width = density; shift = group difference; shape = distributional difference.",
        "common_mistakes": "Violin width is often misinterpreted as frequency - it represents density.",
        "associated_tests": ["Independent t-test", "Mann-Whitney U", "Welch's t-test", "Fligner-Killeen"],
        "widget_function": violin_comp_widget,
    },
    "Scatterplot": {
        "category": "Correlation Plots",
        "description": "Plots pairs of (X, Y) values to reveal the form, direction, and strength of relationships.",
        "when_to_use": "Assess relationship direction and strength, detect outliers and non-linearity.",
        "interpretation": "Upward slope = positive r; downward = negative r; tight cluster = strong correlation.",
        "common_mistakes": "Correlation does NOT imply causation.",
        "associated_tests": ["Pearson correlation", "Spearman correlation", "Linear regression", "Correlation test"],
        "widget_function": scatter_widget,
    },
    "Correlation Heatmap": {
        "category": "Correlation Plots",
        "description": "A 2D grid colored by correlation values to visualize relationships among many variables.",
        "when_to_use": "Checking multicollinearity, exploring many variable relationships, identifying clusters.",
        "interpretation": "Red = positive; blue = negative; dark = strong; diagonal = variable with itself (r=1).",
        "common_mistakes": "Visual patterns can be misleading when variables have different scales.",
        "associated_tests": ["Pearson correlation matrix", "Spearman correlation matrix", "VIF", "Factor analysis"],
        "widget_function": heatmap_widget,
    },
    "Bubble Plot": {
        "category": "Correlation Plots",
        "description": "A scatterplot variant where point size encodes a third variable and color a fourth.",
        "when_to_use": "Showing 3-4 variables at once, highlighting weighted importance, population data.",
        "interpretation": "Position = X-Y relationship; bubble size = third variable; color = fourth variable.",
        "common_mistakes": "Bubble area (not diameter) should encode the third variable.",
        "associated_tests": ["Weighted correlation", "Multiple regression", "Weighted least squares", "Meta-analysis"],
        "widget_function": bubble_widget,
    },
    "Monotonic vs Linear Correlation": {
        "category": "Correlation Plots",
        "description": "Compares Pearson (linear) and Spearman (monotonic) correlation on non-linear relationships.",
        "when_to_use": "Testing if relationship is linear, choosing Pearson vs Spearman, detecting non-linear patterns.",
        "interpretation": "Pearson = linear; Spearman = monotonic; big diff (r vs rho) = non-linear.",
        "common_mistakes": "Pearson r near 0 does NOT mean no relationship - only no linear relationship.",
        "associated_tests": ["Pearson correlation", "Spearman correlation", "Kendall's tau", "Distance correlation"],
        "widget_function": monotonic_widget,
    },
    "Linear Regression Plot": {
        "category": "Regression Plots",
        "description": "Fits a line to (X,Y) data showing the best-fit slope, confidence band, and residuals.",
        "when_to_use": "Modeling continuous outcomes, estimating effect size, predicting Y from X.",
        "interpretation": "Line = best fit; band = 95% CI; slope = change in Y per unit X; R-squared = variance explained.",
        "common_mistakes": "Extrapolating beyond the observed x-range - relationships may not hold outside data.",
        "associated_tests": ["F-test (overall)", "t-test (coefficient)", "Pearson correlation", "ANOVA (nested)"],
        "widget_function": linear_reg_widget,
    },
    "Multiple Regression Surface": {
        "category": "Regression Plots",
        "description": "A 3D surface plot showing the predicted outcome from two continuous predictors.",
        "when_to_use": "Modeling multiple predictors, controlling for confounders, testing interaction effects.",
        "interpretation": "Plane = predicted Y; slope along X1 = beta1 holding X2 constant; twist = interaction.",
        "common_mistakes": "Coefficients are sensitive to predictor scaling - standardize before comparing.",
        "associated_tests": ["F-test (overall)", "Partial F-test", "t-test (coefficients)", "VIF"],
        "widget_function": multiple_reg_widget,
    },
    "Logistic Regression Sigmoid Curve": {
        "category": "Regression Plots",
        "description": "The S-shaped probability curve showing how a binary outcome changes with a predictor.",
        "when_to_use": "Binary outcome prediction, estimating odds ratios, medical diagnosis models.",
        "interpretation": "S-curve = logistic probability; steep = strong predictor; threshold = classification cutoff.",
        "common_mistakes": "Default 0.5 threshold is not always optimal - adjust based on error costs.",
        "associated_tests": ["Likelihood ratio test", "Wald test", "Hosmer-Lemeshow", "ROC-AUC"],
        "widget_function": logistic_widget,
    },
    "Multinomial Decision Boundaries": {
        "category": "Regression Plots",
        "description": "Decision regions for multi-class classification in 2D feature space.",
        "when_to_use": "Multi-class classification, understanding decision boundaries, feature space exploration.",
        "interpretation": "Colored regions = decision zones; boundaries = uncertain areas; overlap = difficulty.",
        "common_mistakes": "Linear boundaries cannot separate non-linear class patterns.",
        "associated_tests": ["Multinomial logistic", "MANOVA", "Discriminant analysis", "Classification metrics"],
        "widget_function": multinomial_widget,
    },
    "Ordinal Logistic Probability Curves": {
        "category": "Regression Plots",
        "description": "Shows how the probability of each ordinal level changes across a continuous predictor.",
        "when_to_use": "Ordered categorical outcomes, Likert scales, disease severity staging.",
        "interpretation": "Each curve = probability of one level; non-parallel = proportional odds violation.",
        "common_mistakes": "Proportional odds assumption must hold - non-parallel curves require alternative models.",
        "associated_tests": ["Proportional odds test", "Brant test", "Likelihood ratio test", "Score test"],
        "widget_function": ordinal_logit_widget,
    },
    "Poisson Count Regression": {
        "category": "Regression Plots",
        "description": "Models count outcomes with a log-linear mean curve showing predicted counts.",
        "when_to_use": "Modeling count outcomes, event frequencies, rare disease incidence.",
        "interpretation": "Y-axis = count; curve = predicted mean; spread increases with mean.",
        "common_mistakes": "Poisson assumes mean = variance - overdispersion needs Negative Binomial.",
        "associated_tests": ["Likelihood ratio test", "Wald test", "Deviance GOF", "Dispersion test"],
        "widget_function": poisson_widget,
    },
    "Confusion Matrix Explorer": {
        "category": "Diagnostic Accuracy Plots",
        "description": "A 2x2 heatmap showing true/false positives and negatives with derived metrics.",
        "when_to_use": "Evaluating binary classifiers, comparing diagnostic tests, understanding error types.",
        "interpretation": "Diagonal = correct; off-diagonal = errors; PPV depends on prevalence.",
        "common_mistakes": "Accuracy is misleading with imbalanced classes - always report PPV and NPV.",
        "associated_tests": ["McNemar's test", "Cohen's Kappa", "ROC-AUC", "Diagnostic likelihood ratios"],
        "widget_function": confusion_widget,
    },
    "ROC Curve Explorer": {
        "category": "Diagnostic Accuracy Plots",
        "description": "Plots TPR vs FPR across thresholds with AUC summarizing discriminative ability.",
        "when_to_use": "Comparing diagnostic tests, assessing model discrimination, choosing optimal threshold.",
        "interpretation": "Top-left = better; AUC = probability correct ranking; 0.5 = guessing; 0.8+ = good.",
        "common_mistakes": "AUC ignores calibration - high AUC can have poorly calibrated probabilities.",
        "associated_tests": ["DeLong test", "Hanley-McNeil test", "Bootstrap AUC", "Sensitivity analysis"],
        "widget_function": roc_widget,
    },
    "Precision-Recall Curve": {
        "category": "Diagnostic Accuracy Plots",
        "description": "Plots precision vs recall across thresholds, better for imbalanced classification than ROC.",
        "when_to_use": "Imbalanced classification, rare disease detection, when PPV matters more.",
        "interpretation": "Higher curve = better; baseline = always-predict-positive; PR better than ROC for imbalance.",
        "common_mistakes": "PR curves from small samples are noisy - use bootstrap confidence bands.",
        "associated_tests": ["Average Precision", "F1 score", "F-beta score", "Bootstrap PR comparison"],
        "widget_function": pr_curve_widget,
    },
    "Sensitivity-Specificity Threshold Explorer": {
        "category": "Diagnostic Accuracy Plots",
        "description": "Shows how sensitivity and specificity trade off across decision thresholds with cost optimization.",
        "when_to_use": "Choosing diagnostic cutoff, balancing sensitivity vs specificity, incorporating error costs.",
        "interpretation": "Blue = sensitivity; red = specificity; tradeoff: increase one = decrease other.",
        "common_mistakes": "Youden's index treats FP and FN equally - in medicine FN is often more costly.",
        "associated_tests": ["ROC analysis", "Youden's index", "Cost-benefit analysis", "Decision curve analysis"],
        "widget_function": threshold_widget,
    },
    "Calibration Plot": {
        "category": "Diagnostic Accuracy Plots",
        "description": "Plots observed proportions against predicted probabilities to assess model calibration.",
        "when_to_use": "Assessing probability accuracy, checking model reliability, comparing risk prediction models.",
        "interpretation": "On diagonal = perfectly calibrated; above = underestimated; below = overestimated.",
        "common_mistakes": "Hosmer-Lemeshow is sensitive to binning - use calibration plots + intercept/slope instead.",
        "associated_tests": ["Hosmer-Lemeshow test", "Brier score", "Spiegelhalter z-test", "Calibration slope"],
        "widget_function": calibration_widget,
    },
    "Bland-Altman Plot": {
        "category": "Agreement Plots",
        "description": "Plots the difference between two measurements against their mean to assess agreement.",
        "when_to_use": "Comparing measurement methods, assessing test-retest reliability, medical device validation.",
        "interpretation": "y=0 = perfect agreement; mean diff = bias; dashed lines = limits of agreement.",
        "common_mistakes": "High r does NOT mean good agreement - two methods can be correlated but biased.",
        "associated_tests": ["Paired t-test", "ICC", "Deming regression", "Passing-Bablok regression"],
        "widget_function": bland_altman_widget,
    },
    "Cohen's Kappa Agreement Matrix": {
        "category": "Agreement Plots",
        "description": "A heatmap of inter-rater classifications with Cohen's Kappa statistic for chance-corrected agreement.",
        "when_to_use": "Inter-rater reliability, diagnostic agreement studies, psychiatric assessment.",
        "interpretation": "Diagonal = perfect agreement; kappa < 0 = worse than chance; 0.4-0.6 = moderate; >0.8 = near perfect.",
        "common_mistakes": "Kappa is prevalence-dependent - rare categories produce low kappa even with high agreement.",
        "associated_tests": ["Weighted Kappa", "Fleiss' Kappa", "McNemar's test", "ICC"],
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
        "associated_tests": ["MANOVA", "Factor analysis", "K-means clustering", "PERMANOVA"],
        "widget_function": pca_widget,
    },
    "MANOVA Group Clouds": {
        "category": "Multivariate Plots",
        "description": "Visualizes multivariate group differences in 2D or 3D space with centroid markers.",
        "when_to_use": "Comparing groups on multiple DVs, controlling for correlated outcomes.",
        "interpretation": "Each color = one group; distance = group difference; overlap = non-significant.",
        "common_mistakes": "Requires multivariate normality and homogeneity of covariance matrices.",
        "associated_tests": ["Pillai's Trace", "Wilks' Lambda", "Hotelling-Lawley Trace", "Roy's Largest Root"],
        "widget_function": manova_widget,
    },
    "Cluster Visualization": {
        "category": "Multivariate Plots",
        "description": "Applies K-means clustering and visualizes discovered clusters with centroids.",
        "when_to_use": "Discovering natural groupings, segmenting populations, pattern recognition.",
        "interpretation": "Each color = discovered cluster; X marks = cluster center; tight clusters = well-separated.",
        "common_mistakes": "K-means assumes spherical, equally-sized clusters - elongated clusters are split incorrectly.",
        "associated_tests": ["Silhouette score", "Elbow method", "Gap statistic", "Davies-Bouldin index"],
        "widget_function": cluster_widget,
    },
    "3D Scatter Explorer": {
        "category": "Multivariate Plots",
        "description": "Interactive 3D scatter plot for exploring three-variable relationships with group coloring.",
        "when_to_use": "Exploring 3-variable relationships, identifying 3D clusters, interactive data exploration.",
        "interpretation": "Each axis = one variable; position = multi-dim profile; clusters = groups with similar profiles.",
        "common_mistakes": "3D plots can obscure patterns depending on viewing angle - rotate to see all perspectives.",
        "associated_tests": ["MANOVA", "Multivariate regression", "Canonical correlation", "3D PCA"],
        "widget_function": scatter3d_widget,
    },
    "Normal P-P Plot": {
        "category": "Distribution Plots",
        "description": "Plots empirical cumulative probabilities against theoretical normal probabilities to assess normality.",
        "when_to_use": "Assessing normality, complementing Q-Q plots, checking distribution center fit.",
        "interpretation": "Points on diagonal = normal; S-curve = heavy tails; above diagonal = right skew.",
        "common_mistakes": "P-P plots are less sensitive to tail departures than Q-Q plots — use both together.",
        "associated_tests": ["Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling", "D'Agostino-Pearson"],
        "widget_function": pp_widget,
    },
    "Raincloud Plot": {
        "category": "Comparison Plots",
        "description": "Combines jittered raw data, boxplot, and half-violin (KDE) for a comprehensive group comparison.",
        "when_to_use": "Modern group comparisons, replacing boxplots, showing distributions + raw data simultaneously.",
        "interpretation": "Points = raw data; box = median + IQR; half-violin = density shape; wider = more density.",
        "common_mistakes": "Jitter width is arbitrary — it shows density not exact x-position. Set seed for reproducibility.",
        "associated_tests": ["Independent t-test", "Mann-Whitney U", "Welch's t-test", "Permutation tests"],
        "widget_function": raincloud_widget,
    },
    "Residuals vs Fitted Plot": {
        "category": "Regression Plots",
        "description": "Scatter plot of residuals against fitted values to detect model misspecification.",
        "when_to_use": "After fitting linear regression, checking homoscedasticity, linearity, and outliers.",
        "interpretation": "Random scatter around 0 = OK; fan shape = heteroscedasticity; U-shape = non-linearity.",
        "common_mistakes": "Patterned residuals indicate model misspecification — fix model before interpreting coefficients.",
        "associated_tests": ["Breusch-Pagan", "Goldfeld-Quandt", "RESET test", "Cook's distance"],
        "widget_function": residuals_fitted_widget,
    },
    "Polynomial Regression Fit": {
        "category": "Regression Plots",
        "description": "Fits a polynomial of adjustable degree to data, comparing against the true generating function.",
        "when_to_use": "Teaching bias-variance tradeoff, modeling non-linear relationships, testing curvature.",
        "interpretation": "Higher degree = more flexible; degree 1 = straight line; high degrees overfit at boundaries.",
        "common_mistakes": "Never extrapolate polynomial fits — they diverge wildly outside the observed data range.",
        "associated_tests": ["F-test (nested)", "Cross-validation MSE", "AIC / BIC", "ANOVA model comparison"],
        "widget_function": poly_reg_widget,
    },
    "Regularization Path": {
        "category": "Regression Plots",
        "description": "Shows how Lasso (L1) or Ridge (L2) regression coefficients change as regularization strength increases.",
        "when_to_use": "High-dimensional data, feature selection, understanding bias-variance tradeoff interactively.",
        "interpretation": "Each line = one coefficient; left = no shrinkage; right = heavy shrinkage; Lasso zeros out features.",
        "common_mistakes": "Lasso selects at most n features. With p > n, Ridge may generalize better. Standardize predictors first.",
        "associated_tests": ["Cross-validated MSE", "Regularization path stability", "BIC", "Bootstrap stability"],
        "widget_function": reg_path_widget,
    },
    "Scatterplot Matrix (SPLOM)": {
        "category": "Correlation Plots",
        "description": "A grid of bivariate scatterplots for every pair of variables with adjustable correlation structure.",
        "when_to_use": "Exploring multivariate relationships, detecting multicollinearity, identifying multivariate outliers.",
        "interpretation": "Each panel = one pair; tight ellipse = strong correlation; diagonal = variable vs itself.",
        "common_mistakes": "SPLOMs become unreadable with >10 variables — use correlation heatmap or PCA instead.",
        "associated_tests": ["Pearson correlation", "VIF", "Bartlett's sphericity", "MANOVA assumptions"],
        "widget_function": splom_widget,
    },
    "Parallel Coordinates Plot": {
        "category": "Multivariate Plots",
        "description": "Each observation is a line crossing parallel axes, revealing multivariate patterns and clusters.",
        "when_to_use": "Visualizing high-dimensional data, identifying clusters, exploring variable relationships.",
        "interpretation": "Each axis = one variable; crossing lines = negative correlation; parallel lines = positive correlation.",
        "common_mistakes": "Axis order affects interpretation — reorder to highlight patterns. Too many lines creates visual clutter.",
        "associated_tests": ["MANOVA", "Canonical correlation", "Discriminant analysis", "Cluster validation"],
        "widget_function": parallel_coords_widget,
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
]


def render_graph_explorer():
    st.title(":bar_chart: Interactive Graph Explorer")
    st.write("Explore statistical graphs interactively. Adjust controls to build visual intuition.")

    graph_category = st.sidebar.radio(
        "Graph Category",
        CATEGORIES,
        key="graph_category_radio",
    )

    category_graphs = {k: v for k, v in graphs.items() if v["category"] == graph_category}

    st.header(f"{graph_category}", divider="orange")

    for name, info in category_graphs.items():
        with st.expander(f"**{name}**", expanded=True):
            info["widget_function"]()
