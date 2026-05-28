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

from .shared import _rng, _gen_corr, _gen_reg, _swarm_positions, _apa_table_ge, _circle_overlap_area, _ideal_sep, _region_pos

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




GRAPHS = {
    "Grouped Bar Chart": grouped_bar_widget,
    "Error Bar Plot": error_bar_widget,
    "Paired Line Plot": paired_line_widget,
    "Boxplot Comparison": boxplot_comp_widget,
    "Violin Comparison": violin_comp_widget,
    "Beeswarm / Swarm Plot": beeswarm_widget,
    "Time Series Plot": time_series_widget,
    "Pie Chart": pie_chart_widget,
    "Area Graph": area_graph_widget,
    "Stacked Bar Chart": stacked_bar_widget,
    "Sankey Diagram": sankey_widget,
    "Cleveland Dot Plot": cleveland_dot_widget,
    "Venn / Euler Diagram": venn_widget
}