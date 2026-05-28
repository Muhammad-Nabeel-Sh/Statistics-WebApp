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


@register_test("Kaplan-Meier Survival Analysis")
def render_kaplan_meier_survival_analysis(external_data=None):

    from scipy.stats import chi2 as _chi2

    st.subheader("Interactive Kaplan-Meier Survival Analysis")

    # =========================
    # DATA SOURCE TOGGLE (survival special)
    # =========================

    if external_data and external_data.get("using_uploaded") and external_data.get("_format") == "raw":
        src = external_data
    else:
        with st.expander("📁 Optional: Use Your Own Data", expanded=False):
            st.markdown("Upload a CSV with time, event, and group columns.")
            source = st.radio(
                "Data Source",
                ["Simulated (sliders, for learning)", "Upload CSV/Excel (your data)"],
                key="km_datasource",
                index=0,
                label_visibility="collapsed",
            )
        if "Simulated" in source:
            src = {"using_uploaded": False, "data": None}
        else:
            km_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="km_file")
            if km_file is not None:
                km_df = pd.read_csv(km_file) if km_file.name.endswith(".csv") else pd.read_excel(km_file)
                st.success(f"Loaded {len(km_df)} rows")
                num_cols = list(km_df.select_dtypes(include=["int64", "float64"]).columns)
                if len(num_cols) >= 2:
                    time_col = st.selectbox("Time column", num_cols, key="km_time")
                    event_col = st.selectbox("Event column (0/1)", num_cols, key="km_event", index=min(1, len(num_cols)-1))
                    group_options = list(km_df.select_dtypes(include=["object", "category", "int64"]).columns) + num_cols
                    group_col = st.selectbox("Group column", group_options, key="km_group")
                    km_data_df = km_df[[time_col, event_col, group_col]].dropna()
                    src = {"using_uploaded": True, "data": {"df": km_data_df, "time_col": time_col, "event_col": event_col, "group_col": group_col}}
                else:
                    st.error("Need at least 2 numeric columns")
                    src = {"using_uploaded": False, "data": None}
            else:
                src = {"using_uploaded": False, "data": None}

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        km_data = src["data"]
        km_df_in = km_data["df"]
        km_times = km_df_in[km_data["time_col"]].values
        km_event = km_df_in[km_data["event_col"]].values.astype(int)
        km_group_raw = km_df_in[km_data["group_col"]].values
        km_group_names = sorted(set(str(v) for v in km_group_raw))
        if len(km_group_names) > 2:
            km_group_names = km_group_names[:2]
        km_group = np.array([0 if str(v) == km_group_names[0] else 1 for v in km_group_raw])
        show_ci = st.toggle("Show 95% CI", True, key="km_ci_toggle_uploaded")
    else:
        n = st.slider("Patients per Group", 10, 200, 50, key="km_n_pat")
        hr = st.slider(
            "Hazard Ratio (Trt/Control)", 0.2, 2.0, 0.5, 0.05, key="km_hr_ratio"
        )
        cens = st.slider("Censoring Rate", 0.0, 0.5, 0.2, 0.05, key="km_cens_rate")
        show_ci = st.toggle("Show 95% CI", True, key="km_ci_toggle")

        # =========================
        # DATA
        # =========================

        np.random.seed(42)
        t_ctrl = np.random.exponential(12, n)
        t_trt = np.random.exponential(12 / hr, n)
        all_t = np.concatenate([t_ctrl, t_trt])
        groups = np.array([0] * n + [1] * n)

        max_fup = 30
        fup = np.random.uniform(0, max_fup, 2 * n)
        if cens > 0.01:
            early = np.random.exponential(8 / max(cens, 0.05), 2 * n)
            fup = np.minimum(fup, early)
        obs = np.minimum(all_t, fup)
        event = (all_t <= fup).astype(int)
        km_times = obs
        km_event = event
        km_group = groups
        km_group_names = ["Control", "Treatment"]

    # =========================
    # HELPERS
    # =========================

    def _km_est(tt, ee):
        df = pd.DataFrame({"t": tt, "e": ee}).sort_values("t")
        ut = sorted(df["t"].unique())
        s = 1.0
        ot, os = [0], [1.0]
        for ti in ut:
            nr = (df["t"] >= ti).sum()
            ne = df.loc[df["t"] == ti, "e"].sum()
            if nr > 0:
                s *= 1 - ne / nr
            ot.extend([ti, ti])
            os.extend([os[-1], s])
        return np.array(ot), np.array(os)

    def _km_ci(tt, ee):
        df = pd.DataFrame({"t": tt, "e": ee}).sort_values("t")
        ut = sorted(df["t"].unique())
        s, cv = 1.0, 0.0
        ot, os, olo, ohi = [0], [1.0], [1.0], [1.0]
        for ti in ut:
            nr = (df["t"] >= ti).sum()
            ne = df.loc[df["t"] == ti, "e"].sum()
            if nr > 0 and ne > 0:
                s *= 1 - ne / nr
                cv += ne / (nr * (nr - ne))
            elif nr > 0:
                s *= 1 - ne / nr
            se = s * np.sqrt(cv) if cv > 0 else 0
            lo = np.clip(s - 1.96 * se, 0, 1)
            hi = np.clip(s + 1.96 * se, 0, 1)
            ot.extend([ti, ti])
            os.extend([os[-1], s])
            olo.extend([olo[-1], lo])
            ohi.extend([ohi[-1], hi])
        return np.array(ot), np.array(os), np.array(olo), np.array(ohi)

    def _logrank(tt, ee, gg):
        sort_idx = np.argsort(tt)
        tt_s = tt[sort_idx]
        ee_s = ee[sort_idx]
        gg_s = gg[sort_idx]
        ut = sorted(set(tt_s))
        o1 = e1 = v1 = 0.0
        for ti in ut:
            mask_t = tt_s == ti
            ne = ee_s[mask_t].sum()
            if ne == 0:
                continue
            at_risk = tt_s >= ti
            n1 = (gg_s[at_risk] == 0).sum()
            n2 = (gg_s[at_risk] == 1).sum()
            nr = n1 + n2
            if nr < 2:
                continue
            d1 = (gg_s[mask_t & (ee_s == 1)] == 0).sum()
            o1 += d1
            e1 += ne * n1 / nr
            v1 += ne * n1 * n2 * (nr - ne) / (nr * nr * (nr - 1))
        if v1 <= 0:
            return 0.0, 1.0
        chi2 = (o1 - e1) ** 2 / v1
        return chi2, 1 - _chi2.cdf(chi2, 1)

    # =========================
    # COMPUTE
    # =========================

    t1, s1 = _km_est(km_times[km_group == 0], km_event[km_group == 0])
    t2, s2 = _km_est(km_times[km_group == 1], km_event[km_group == 1])
    chi2, p_val = _logrank(km_times, km_event, km_group)

    below1 = np.where(s1 <= 0.5)[0]
    below2 = np.where(s2 <= 0.5)[0]
    med1 = t1[below1[0]] if len(below1) > 0 else None
    med2 = t2[below2[0]] if len(below2) > 0 else None

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()
    gname0 = km_group_names[0] if len(km_group_names) > 0 else "Group 0"
    gname1 = km_group_names[1] if len(km_group_names) > 1 else "Group 1"
    colors = {gname0: "#4C78A8", gname1: "#E45756"}
    fill_colors = {
        gname0: "rgba(76,120,168,0.15)",
        gname1: "rgba(228,87,86,0.15)",
    }

    for grp, name in [(0, gname0), (1, gname1)]:
        t_, s_, lo_, hi_ = _km_ci(km_times[km_group == grp], km_event[km_group == grp])
        if show_ci:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([t_, t_[::-1]]),
                    y=np.concatenate([hi_, lo_[::-1]]),
                    fill="toself",
                    fillcolor=fill_colors[name],
                    line=dict(width=0),
                    name=f"{name} 95% CI",
                    showlegend=True,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=t_,
                y=s_,
                mode="lines",
                line=dict(color=colors[name], width=2.5),
                name=name,
                hovertemplate="Time=%{x:.1f}<br>Survival=%{y:.3f}<extra></extra>",
                legendgroup=name,
            )
        )
        mask = km_group == grp
        cens_t = km_times[mask & (km_event == 0)]
        if len(cens_t) > 0:
            idxs = np.searchsorted(t_, cens_t, side="right") - 1
            cs = s_[np.maximum(0, idxs)]
            fig.add_trace(
                go.Scatter(
                    x=cens_t,
                    y=cs,
                    mode="markers",
                    marker=dict(color=colors[name], symbol="line-ns", size=7),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    events_pct = int(km_event.mean() * 100)
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Survival Probability",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # METRICS
    # =========================

    m1 = st.columns(4)
    m1[0].metric(
        f"{gname0} Median Survival",
        f"{med1:.1f}" if med1 is not None else "Not reached",
    )
    m1[1].metric(
        f"{gname1} Median Survival",
        f"{med2:.1f}" if med2 is not None else "Not reached",
    )
    m1[2].metric("Log-Rank χ²", f"{chi2:.3f}")
    m1[3].metric(
        "Log-Rank p-value", f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    )

    # =========================
    # INTERPRETATION
    # =========================

    with st.expander("Interpretation & Guidance", expanded=True):
        st.markdown(f"""
        - **Events observed**: {int(km_event.sum())} / {len(km_event)} ({events_pct}% of patients experienced the event)
        - **Censored**: {(~km_event.astype(bool)).sum()} patients ({100 - events_pct}%) had their event time censored
        """)
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Step down = event\n"
                "- Tick marks = censored\n"
                "- Lower curve = worse survival\n"
                "- HR < 1 favors treatment"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Time-to-event data\n"
                "- Censored observations\n"
                "- Compare survival curves\n"
                "- Estimate median survival"
            )
        with col_t:
            st.warning(
                "**Associated**\n\n"
                "- Log-Rank Test\n"
                "- Cox PH Regression\n"
                "- Nelson-Aalen Plot\n"
                "- Hazard Ratio"
            )
        with col_m:
            st.error(
                "**Caution**\n\n"
                "KM beyond last event is "
                "unstable. Always show "
                "number at risk. "
                "CI widens at late times."
            )



@register_test("Log-Rank Test")
def render_log_rank_test(external_data=None):

    st.subheader("Interactive Log-Rank Test")

    # =========================
    # DATA SOURCE TOGGLE (survival special)
    # =========================

    if external_data and external_data.get("using_uploaded") and external_data.get("_format") == "raw":
        src = external_data
    else:
        with st.expander("📁 Optional: Use Your Own Data", expanded=False):
            st.markdown("Upload a CSV with time, event, and group columns.")
            source = st.radio(
                "Data Source",
                ["Simulated (sliders, for learning)", "Upload CSV/Excel (your data)"],
                key="lr_datasource",
                index=0,
                label_visibility="collapsed",
            )
        if "Simulated" in source:
            src = {"using_uploaded": False, "data": None}
        else:
            lr_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="lr_file")
            if lr_file is not None:
                lr_df = pd.read_csv(lr_file) if lr_file.name.endswith(".csv") else pd.read_excel(lr_file)
                st.success(f"Loaded {len(lr_df)} rows")
                num_cols = list(lr_df.select_dtypes(include=["int64", "float64"]).columns)
                if len(num_cols) >= 2:
                    time_col = st.selectbox("Time column", num_cols, key="lr_time")
                    event_col = st.selectbox("Event column (0/1)", num_cols, key="lr_event", index=min(1, len(num_cols)-1))
                    group_options = list(lr_df.select_dtypes(include=["object", "category", "int64"]).columns) + num_cols
                    group_col = st.selectbox("Group column", group_options, key="lr_group")
                    src = {"using_uploaded": True, "data": {"df": lr_df[[time_col, event_col, group_col]].dropna(), "time_col": time_col, "event_col": event_col, "group_col": group_col}}
                else:
                    st.error("Need at least 2 numeric columns")
                    src = {"using_uploaded": False, "data": None}
            else:
                src = {"using_uploaded": False, "data": None}

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        lr_data = src["data"]
        lr_df_in = lr_data["df"]
        lr_times = lr_df_in[lr_data["time_col"]].values
        lr_event = lr_df_in[lr_data["event_col"]].values.astype(int)
        lr_group_raw = lr_df_in[lr_data["group_col"]].values
        lr_group_names = sorted(set(str(v) for v in lr_group_raw))
        if len(lr_group_names) > 2:
            lr_group_names = lr_group_names[:2]
        lr_group = np.array([0 if str(v) == lr_group_names[0] else 1 for v in lr_group_raw])

        from scipy.stats import chi2 as _lr_chi2
        def _logrank_lr(tt, ee, gg):
            sort_idx = np.argsort(tt)
            tt_s, ee_s, gg_s = tt[sort_idx], ee[sort_idx], gg[sort_idx]
            ut = sorted(set(tt_s))
            o1 = e1 = v1 = 0.0
            for ti in ut:
                mask_t = tt_s == ti
                ne = ee_s[mask_t].sum()
                if ne == 0: continue
                at_risk = tt_s >= ti
                n1, n2 = (gg_s[at_risk] == 0).sum(), (gg_s[at_risk] == 1).sum()
                nr = n1 + n2
                if nr < 2: continue
                d1 = (gg_s[mask_t & (ee_s == 1)] == 0).sum()
                o1 += d1; e1 += ne * n1 / nr
                v1 += ne * n1 * n2 * (nr - ne) / (nr * nr * (nr - 1))
            if v1 <= 0: return 0.0, 1.0
            chi2 = (o1 - e1) ** 2 / v1
            return chi2, 1 - _lr_chi2.cdf(chi2, 1)

        lr_chi2, lr_p = _logrank_lr(lr_times, lr_event, lr_group)
        st.latex(rf"\text{{Log-Rank }} \chi^2 = {lr_chi2:.3f}")
        st.latex(rf"\text{{{format_p_value(lr_p)}}}")

        fig = go.Figure()
        t_grid = np.linspace(0, np.percentile(lr_times, 95) * 1.2, 200)
        for grp, name in [(0, lr_group_names[0]), (1, lr_group_names[1])]:
            mask_g = lr_group == grp
            tt_grp = lr_times[mask_g]
            ee_grp = lr_event[mask_g]
            rate = 1 / np.mean(tt_grp[ee_grp == 1]) if ee_grp.sum() > 0 else 0.1
            surv = np.exp(-rate * t_grid)
            fig.add_trace(go.Scatter(x=t_grid, y=surv, mode="lines", name=name))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Survival Probability")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Log-Rank Test Results")
        st.table(pd.DataFrame({
            "Metric": ["N", "Events", "χ²", "p-value"],
            "Value": [f"{len(lr_times)}", f"{int(lr_event.sum())}", f"{lr_chi2:.3f}", format_p_value(lr_p)],
        }))
    else:
        sep = st.slider("Survival Separation", 0.0, 5.0, 1.5, 0.1)

        n = st.slider("Subjects per Group", 10, 200, 50)

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        t1 = np.random.exponential(10, n)
        t2 = np.random.exponential(10, n) / (1 + sep * 0.3)

        censor = np.random.exponential(15, n)
        censor2 = np.random.exponential(15, n)

        obs1 = (t1 <= censor).astype(int)
        obs2 = (t2 <= censor2).astype(int)
        t1_obs = np.minimum(t1, censor)
        t2_obs = np.minimum(t2, censor2)

        fig = go.Figure()
        t_grid = np.linspace(0, max(max(t1_obs), max(t2_obs)) * 1.1, 200)

        # Exponential fit approximations
        rate1 = 1 / np.mean(t1_obs[obs1 == 1]) if obs1.sum() > 0 else 0.1
        rate2 = 1 / np.mean(t2_obs[obs2 == 1]) if obs2.sum() > 0 else 0.1

        surv1 = np.exp(-rate1 * t_grid)
        surv2 = np.exp(-rate2 * t_grid)

        fig.add_trace(go.Scatter(x=t_grid, y=surv1, mode="lines", name="Group 1"))
        fig.add_trace(go.Scatter(x=t_grid, y=surv2, mode="lines", name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Time",
            yaxis_title="Survival Probability",
        )

        st.plotly_chart(fig, use_container_width=True)

