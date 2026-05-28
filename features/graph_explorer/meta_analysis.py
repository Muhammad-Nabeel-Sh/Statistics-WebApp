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

from .shared import _rng, _gen_corr, _gen_reg, _gen_meta_data, _gen_meta_bias

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




GRAPHS = {
    "Forest Plot": forest_plot_widget,
    "Funnel Plot": funnel_widget,
    "Galbraith (Radial) Plot": galbraith_widget,
    "Baujat Plot": baujat_widget,
    "Leave-One-Out Plot": loo_widget
}