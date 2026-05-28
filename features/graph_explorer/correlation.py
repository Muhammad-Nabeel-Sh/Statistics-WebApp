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




GRAPHS = {
    "Scatterplot": scatter_widget,
    "Correlation Heatmap": heatmap_widget,
    "Bubble Plot": bubble_widget,
    "Monotonic vs Linear Correlation": monotonic_widget,
    "Scatterplot Matrix (SPLOM)": splom_widget,
    "Hexbin Plot": hexbin_widget
}