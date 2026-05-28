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

from .shared import _rng, _gen_corr, _gen_reg, _gen_surv_data, _km, _na

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




GRAPHS = {
    "Kaplan-Meier Curve": kaplan_meier_widget,
    "Nelson-Aalen Plot": nelson_aalen_widget,
    "Hazard Function Plot": hazard_function_widget,
    "Cumulative Hazard Plot": cumulative_hazard_widget,
    "Cox PH Effect Plot": cox_ph_widget,
    "Survival Probability Heatmap": surv_heatmap_widget
}