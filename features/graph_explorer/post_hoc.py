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

from .shared import _rng, _gen_corr, _gen_reg, _gen_ph_data, _assign_cld

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



GRAPHS = {
    "Confidence Interval Comparison Plot": ph_ci_comparison_widget,
    "Mean Difference Forest Plot": ph_forest_plot_widget,
    "Compact Letter Display (CLD)": ph_cld_widget,
    "Significance Heatmap": ph_significance_heatmap_widget,
    "Pairwise Network Graph": ph_network_widget,
    "Estimation Plot (Gardner-Altman)": ph_estimation_widget,
    "Raincloud Post Hoc Plot": ph_raincloud_widget
}