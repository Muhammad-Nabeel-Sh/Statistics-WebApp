import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
from core.utils import _apa_table

_rng = np.random.default_rng(42)


# SPC constants for n=2..15 (Xbar-R, Xbar-S, I-MR)
_SPC_CONSTANTS = {
    2:  {"A2": 1.880, "D3": 0, "D4": 3.267, "d2": 1.128, "B3": 0, "B4": 3.267, "c4": 0.7979},
    3:  {"A2": 1.023, "D3": 0, "D4": 2.575, "d2": 1.693, "B3": 0, "B4": 2.568, "c4": 0.8862},
    4:  {"A2": 0.729, "D3": 0, "D4": 2.282, "d2": 2.059, "B3": 0, "B4": 2.266, "c4": 0.9213},
    5:  {"A2": 0.577, "D3": 0, "D4": 2.115, "d2": 2.326, "B3": 0, "B4": 2.089, "c4": 0.9400},
    6:  {"A2": 0.483, "D3": 0, "D4": 2.004, "d2": 2.534, "B3": 0.030, "B4": 1.970, "c4": 0.9515},
    7:  {"A2": 0.419, "D3": 0.076, "D4": 1.924, "d2": 2.704, "B3": 0.118, "B4": 1.882, "c4": 0.9594},
    8:  {"A2": 0.373, "D3": 0.136, "D4": 1.864, "d2": 2.847, "B3": 0.185, "B4": 1.815, "c4": 0.9650},
    9:  {"A2": 0.337, "D3": 0.184, "D4": 1.816, "d2": 2.970, "B3": 0.239, "B4": 1.761, "c4": 0.9693},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "d2": 3.078, "B3": 0.284, "B4": 1.716, "c4": 0.9727},
    11: {"A2": 0.285, "D3": 0.256, "D4": 1.744, "d2": 3.173, "B3": 0.322, "B4": 1.678, "c4": 0.9754},
    12: {"A2": 0.266, "D3": 0.284, "D4": 1.716, "d2": 3.258, "B3": 0.354, "B4": 1.646, "c4": 0.9776},
    13: {"A2": 0.249, "D3": 0.308, "D4": 1.692, "d2": 3.336, "B3": 0.382, "B4": 1.618, "c4": 0.9794},
    14: {"A2": 0.235, "D3": 0.329, "D4": 1.671, "d2": 3.407, "B3": 0.406, "B4": 1.594, "c4": 0.9810},
    15: {"A2": 0.223, "D3": 0.348, "D4": 1.652, "d2": 3.472, "B3": 0.428, "B4": 1.572, "c4": 0.9823},
}
_E2 = 2.660  # E2 = 3/d2 for n=2 (I-MR)


def _get_constants(n):
    clamped = max(2, min(15, n))
    return _SPC_CONSTANTS[clamped]


def _check_shewhart_rules(values, center, sigma):
    rules = []
    k = len(values)
    if k < 2:
        return rules

    ucl = center + 3 * sigma
    lcl = center - 3 * sigma
    u2 = center + 2 * sigma
    l2 = center - 2 * sigma
    u1 = center + 1 * sigma
    l1 = center - 1 * sigma

    def side(v):
        if v > center:
            return 1
        elif v < center:
            return -1
        return 0

    sides_arr = np.array([side(v) for v in values])

    # Rule 1: Beyond 3σ
    for i, v in enumerate(values):
        if v > ucl or v < lcl:
            rules.append(f"🔴 **Rule 1** — Point {i+1} is beyond the 3σ control limit (value = {v:.3f})")

    # Rule 2: 2 of 3 consecutive beyond 2σ (same side)
    for i in range(k - 2):
        trio = values[i:i+3]
        beyond = [abs(v - center) > 2 * sigma for v in trio]
        if sum(beyond) >= 2:
            sides_beyond = [s for s, b in zip(sides_arr[i:i+3], beyond) if b and s != 0]
            if sides_beyond and len(set(sides_beyond)) == 1:
                rules.append(f"⚠️ **Rule 2** — 2 of 3 points (indices {i+1}-{i+3}) beyond 2σ on same side")

    # Rule 3: 4 of 5 beyond 1σ (same side)
    for i in range(k - 4):
        five = values[i:i+5]
        beyond = [abs(v - center) > sigma for v in five]
        if sum(beyond) >= 4:
            sides_beyond = [s for s, b in zip(sides_arr[i:i+5], beyond) if b and s != 0]
            if sides_beyond and len(set(sides_beyond)) == 1:
                rules.append(f"⚠️ **Rule 3** — 4 of 5 points (indices {i+1}-{i+5}) beyond 1σ on same side")

    # Rule 4: 8+ consecutive on same side of center
    run_start = 0
    current_side = sides_arr[0]
    for i in range(1, k):
        if sides_arr[i] == 0:
            continue
        if sides_arr[i] == current_side:
            if i - run_start + 1 >= 8:
                dir_label = "above" if current_side == 1 else "below"
                rules.append(f"🔴 **Rule 4** — {i - run_start + 1} consecutive points (indices {run_start+1}-{i+1}) on same side ({dir_label}) of center")
                run_start = i
        else:
            run_start = i
            current_side = sides_arr[i]

    # Rule 5: 6+ consecutive trending
    for i in range(k - 5):
        six = values[i:i+6]
        diffs = np.diff(six)
        if all(d > 0 for d in diffs):
            rules.append(f"⚠️ **Rule 5** — 6 consecutive points trending upward (indices {i+1}-{i+6})")
        elif all(d < 0 for d in diffs):
            rules.append(f"⚠️ **Rule 5** — 6 consecutive points trending downward (indices {i+1}-{i+6})")

    # Rule 6: 14+ consecutive alternating up/down
    for i in range(k - 13):
        fourteen = values[i:i+14]
        diffs = np.sign(np.diff(fourteen))
        if all(diffs[j] != diffs[j+1] for j in range(len(diffs)-1)):
            rules.append(f"🔴 **Rule 6** — 14 consecutive points alternating up/down (indices {i+1}-{i+14}) — possible over-control")

    # Rule 7: 15+ consecutive within 1σ (either side)
    for i in range(k - 14):
        fifteen = values[i:i+15]
        if all(abs(v - center) < sigma for v in fifteen):
            rules.append(f"🔴 **Rule 7** — 15 consecutive points within 1σ (indices {i+1}-{i+15}) — possible data tampering or reduced variability")

    # Rule 8: 8+ consecutive beyond 1σ (either side)
    for i in range(k - 7):
        eight = values[i:i+8]
        if all(abs(v - center) > sigma for v in eight):
            rules.append(f"🔴 **Rule 8** — 8 consecutive points beyond 1σ (indices {i+1}-{i+8}) — possible mixture or stratification")

    return rules


def _render_p_chart():
    """Proportion (p) chart — fraction defective per subgroup."""
    st.subheader("p-Chart — Proportion Defective")
    st.info("""
    **p-Chart** monitors the proportion of defective items in each subgroup.
    Use when subgroup sizes may vary. Control limits vary with each subgroup size.
    """)

    rng = np.random.default_rng(42)

    with st.sidebar:
        st.markdown("##### :orange[Process Parameters]")
        p_bar = st.slider("Process Defect Rate (p̄)", 0.01, 0.50, 0.08, 0.01, key="spc_p_mean",
                          help="Average proportion defective when in control")
        n_subgroups_p = st.slider("Number of Subgroups", 15, 100, 25, 5, key="spc_p_k")
        var_n = st.checkbox("Varying Subgroup Sizes", True, key="spc_p_varn")
        if var_n:
            n_range = st.slider("Subgroup Size Range", 20, 200, (50, 150), key="spc_p_nrange")
            ns = rng.integers(n_range[0], n_range[1] + 1, n_subgroups_p)
        else:
            n_fixed = st.slider("Subgroup Size (n)", 20, 500, 100, 10, key="spc_p_nfixed")
            ns = np.full(n_subgroups_p, n_fixed, dtype=int)

        shift_type_p = st.selectbox("Shift Type", ["None", "Defect Rate Increase", "Defect Rate Decrease"], key="spc_p_shift")
        if shift_type_p == "Defect Rate Increase":
            shift_p = st.slider("Rate Increase (pct points)", 0.02, 0.30, 0.10, 0.01, key="spc_p_shift_up")
            p_after = p_bar + shift_p
        elif shift_type_p == "Defect Rate Decrease":
            shift_p = st.slider("Rate Decrease (pct points)", 0.02, 0.30, 0.05, 0.01, key="spc_p_shift_down")
            p_after = max(0.001, p_bar - shift_p)
        else:
            p_after = p_bar

        st.markdown("---")
        st.markdown("##### :orange[Chart Settings]")
        sigma_limits_p = st.slider("Control Limit Width (σ)", 1, 4, 3, 1, key="spc_p_sigma")
        show_rules_p = st.checkbox("Show Shewhart Rules", True, key="spc_p_rules")

    # Generate data
    shift_point_p = n_subgroups_p // 2
    defects = np.zeros(n_subgroups_p, dtype=int)
    for i in range(n_subgroups_p):
        p_i = p_after if i >= shift_point_p and shift_type_p != "None" else p_bar
        defects[i] = rng.binomial(ns[i], p_i)

    pi = defects / ns
    p_bar_calc = defects.sum() / ns.sum()
    z_p = sigma_limits_p

    # Control limits (vary with n if sizes vary)
    se_p = np.sqrt(p_bar_calc * (1 - p_bar_calc) / ns)
    ucl_p = p_bar_calc + z_p * se_p
    lcl_p = np.maximum(0, p_bar_calc - z_p * se_p)

    # Fixed limits for display
    n_avg = np.mean(ns)
    se_avg = np.sqrt(p_bar_calc * (1 - p_bar_calc) / n_avg)
    ucl_p_fixed = p_bar_calc + z_p * se_avg
    lcl_p_fixed = max(0, p_bar_calc - z_p * se_avg)

    # Plot
    labels = [str(i + 1) for i in range(n_subgroups_p)]

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=labels, y=pi, mode="lines+markers",
        marker=dict(color="#4C78A8", size=7), line=dict(color="#4C78A8", width=1.5),
        name="p (fraction defective)",
    ))
    fig_p.add_hline(y=p_bar_calc, line_dash="solid", line_color="#54A24B",
                    annotation_text=f"p̄ = {p_bar_calc:.4f}")
    fig_p.add_hline(y=ucl_p_fixed, line_dash="dash", line_color="#E45756",
                    annotation_text=f"UCL (avg n) = {ucl_p_fixed:.4f}")
    fig_p.add_hline(y=lcl_p_fixed, line_dash="dash", line_color="#E45756",
                    annotation_text=f"LCL (avg n) = {lcl_p_fixed:.4f}")

    # Varying limits as area
    if var_n:
        fig_p.add_trace(go.Scatter(
            x=labels + labels[::-1],
            y=ucl_p.tolist() + lcl_p[::-1].tolist(),
            fill="toself", fillcolor="rgba(228, 87, 86, 0.08)",
            line=dict(color="rgba(228, 87, 86, 0)"), showlegend=False,
            name="Varying limits",
        ))

    # OOC points
    for i in range(n_subgroups_p):
        if pi[i] > ucl_p[i] or pi[i] < lcl_p[i]:
            fig_p.add_trace(go.Scatter(
                x=[labels[i]], y=[pi[i]], mode="markers",
                marker=dict(color="#E45756", size=12, symbol="x"),
                showlegend=False,
            ))

    if shift_type_p != "None":
        fig_p.add_vline(x=shift_point_p - 0.5, line_dash="dot", line_color="white", opacity=0.5,
                        annotation_text="Shift introduced")

    fig_p.update_layout(
        template="plotly_dark", height=450,
        xaxis_title="Subgroup", yaxis_title="Proportion Defective",
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_p, use_container_width=True, key="spc_p_chart")

    # Summary
    st.subheader("Process Summary", divider="gray")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average p̄", f"{p_bar_calc:.4f}")
    c2.metric("Total Defects", f"{defects.sum()}")
    c3.metric("Total Items", f"{ns.sum()}")
    c4.metric("Subgroups", str(n_subgroups_p))

    # Rules
    if show_rules_p:
        st.subheader("Shewhart Rule Violations", divider="gray")
        rules_p = _check_shewhart_rules(pi, p_bar_calc, np.std(pi, ddof=1) if len(pi) > 1 else 0)
        if rules_p:
            for r in rules_p:
                st.markdown(r)
        else:
            st.success("✅ No violations detected.")

    # Data table
    with st.expander("View Raw Data"):
        pdf = pd.DataFrame({
            "Subgroup": range(1, n_subgroups_p + 1),
            "n": ns, "Defects": defects,
            "Proportion (p)": [f"{v:.4f}" for v in pi],
            "UCL": [f"{v:.4f}" for v in ucl_p],
            "LCL": [f"{v:.4f}" for v in lcl_p],
        })
        st.dataframe(pdf, use_container_width=True)

    with st.expander("About the p-Chart"):
        st.markdown("""
        | Element | Formula |
        |---------|---------|
        | p̄ (center) | Total defects / Total items |
        | UCL | p̄ + k·√[p̄(1−p̄)/nᵢ] — varies with each subgroup size nᵢ |
        | LCL | max(0, p̄ − k·√[p̄(1−p̄)/nᵢ]) |
        | Assumptions | Binomial distribution, constant defect probability within subgroups |
        """)


def _render_np_chart():
    """Number defective (np) chart — constant subgroup size."""
    st.subheader("np-Chart — Number Defective")
    st.info("""
    **np-Chart** monitors the count of defective items per subgroup.
    Requires constant subgroup size. Use when the count of defectives
    is easier to track than proportion.
    """)

    rng = np.random.default_rng(42)

    with st.sidebar:
        st.markdown("##### :orange[Process Parameters]")
        p_bar_np = st.slider("Process Defect Rate (p̄)", 0.01, 0.50, 0.08, 0.01, key="spc_np_mean")
        n_np = st.slider("Subgroup Size (n)", 30, 500, 100, 10, key="spc_np_n")
        k_np = st.slider("Number of Subgroups", 15, 100, 25, 5, key="spc_np_k")
        shift_np = st.selectbox("Shift Type", ["None", "Defect Rate Increase", "Defect Rate Decrease"], key="spc_np_shift")
        if shift_np == "Defect Rate Increase":
            delta_np = st.slider("Rate Increase (pct points)", 0.02, 0.30, 0.10, 0.01, key="spc_np_up")
            p_after_np = p_bar_np + delta_np
        elif shift_np == "Defect Rate Decrease":
            delta_np = st.slider("Rate Decrease (pct points)", 0.02, 0.30, 0.05, 0.01, key="spc_np_down")
            p_after_np = max(0.001, p_bar_np - delta_np)
        else:
            p_after_np = p_bar_np

        st.markdown("---")
        st.markdown("##### :orange[Chart Settings]")
        sigma_np = st.slider("Control Limit Width (σ)", 1, 4, 3, 1, key="spc_np_sigma")
        show_rules_np = st.checkbox("Show Shewhart Rules", True, key="spc_np_rules")

    shift_pt_np = k_np // 2
    np_counts = np.zeros(k_np, dtype=int)
    for i in range(k_np):
        p_i = p_after_np if i >= shift_pt_np and shift_np != "None" else p_bar_np
        np_counts[i] = rng.binomial(n_np, p_i)

    p_bar_np_calc = np_counts.sum() / (k_np * n_np)
    np_bar = n_np * p_bar_np_calc
    z_np = sigma_np
    sigma_np_chart = np.sqrt(n_np * p_bar_np_calc * (1 - p_bar_np_calc))
    ucl_np = np_bar + z_np * sigma_np_chart
    lcl_np = max(0, np_bar - z_np * sigma_np_chart)

    labels_np = [str(i + 1) for i in range(k_np)]
    fig_np = go.Figure()
    fig_np.add_trace(go.Scatter(
        x=labels_np, y=np_counts, mode="lines+markers",
        marker=dict(color="#4C78A8", size=7), line=dict(color="#4C78A8", width=1.5),
        name="np (defect count)",
    ))
    fig_np.add_hline(y=np_bar, line_dash="solid", line_color="#54A24B",
                     annotation_text=f"np̄ = {np_bar:.1f}")
    fig_np.add_hline(y=ucl_np, line_dash="dash", line_color="#E45756",
                     annotation_text=f"UCL = {ucl_np:.1f}")
    fig_np.add_hline(y=lcl_np, line_dash="dash", line_color="#E45756",
                     annotation_text=f"LCL = {lcl_np:.1f}")
    for i in range(k_np):
        if np_counts[i] > ucl_np or np_counts[i] < lcl_np:
            fig_np.add_trace(go.Scatter(
                x=[labels_np[i]], y=[np_counts[i]], mode="markers",
                marker=dict(color="#E45756", size=12, symbol="x"), showlegend=False,
            ))
    if shift_np != "None":
        fig_np.add_vline(x=shift_pt_np - 0.5, line_dash="dot", line_color="white", opacity=0.5,
                         annotation_text="Shift")
    fig_np.update_layout(template="plotly_dark", height=400,
                          xaxis_title="Subgroup", yaxis_title="Number Defective",
                          margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_np, use_container_width=True, key="spc_np_chart")

    c1, c2, c3 = st.columns(3)
    c1.metric("np̄", f"{np_bar:.2f}")
    c2.metric("p̄", f"{p_bar_np_calc:.4f}")
    c3.metric("Subgroups", str(k_np))

    if show_rules_np:
        st.subheader("Shewhart Rule Violations", divider="gray")
        rules_np = _check_shewhart_rules(np_counts.astype(float), np_bar, sigma_np_chart)
        if rules_np:
            for r in rules_np:
                st.markdown(r)
        else:
            st.success("✅ No violations detected.")

    with st.expander("About the np-Chart"):
        st.markdown("""
        | Element | Formula |
        |---------|---------|
        | np̄ (center) | n·p̄ = total defects / number of subgroups |
        | UCL | np̄ + k·√[n·p̄(1−p̄)] |
        | LCL | max(0, np̄ − k·√[n·p̄(1−p̄)]) |
        | Constant n? | Required — use p-chart if subgroup sizes vary |
        """)


def _render_c_chart():
    """Count (c) chart — defects per unit (constant opportunity)."""
    st.subheader("c-Chart — Count of Defects per Unit")
    st.info("""
    **c-Chart** monitors the number of defects per unit when the area of
    opportunity is constant. Examples: surface defects per windshield,
    errors per page, complaints per day.
    """)

    rng = np.random.default_rng(42)

    with st.sidebar:
        st.markdown("##### :orange[Process Parameters]")
        c_bar = st.slider("Average Defects per Unit (c̄)", 1.0, 30.0, 5.0, 0.5, key="spc_c_mean")
        k_c = st.slider("Number of Subgroups", 15, 100, 25, 5, key="spc_c_k")
        shift_c = st.selectbox("Shift Type", ["None", "Defect Count Increase", "Defect Count Decrease"], key="spc_c_shift")
        if shift_c == "Defect Count Increase":
            c_after = c_bar + st.slider("Increase in c", 1.0, 20.0, 5.0, 0.5, key="spc_c_up")
        elif shift_c == "Defect Count Decrease":
            c_after = max(0.5, c_bar - st.slider("Decrease in c", 1.0, 20.0, 3.0, 0.5, key="spc_c_down"))
        else:
            c_after = c_bar

        st.markdown("---")
        st.markdown("##### :orange[Chart Settings]")
        sigma_c = st.slider("Control Limit Width (σ)", 1, 4, 3, 1, key="spc_c_sigma")
        show_rules_c = st.checkbox("Show Shewhart Rules", True, key="spc_c_rules")

    shift_pt_c = k_c // 2
    counts_c = np.zeros(k_c, dtype=int)
    for i in range(k_c):
        mu_c = c_after if i >= shift_pt_c and shift_c != "None" else c_bar
        counts_c[i] = rng.poisson(mu_c)

    c_bar_calc = counts_c.mean()
    z_c = sigma_c
    sigma_c_chart = np.sqrt(c_bar_calc)
    ucl_c = c_bar_calc + z_c * sigma_c_chart
    lcl_c = max(0, c_bar_calc - z_c * sigma_c_chart)

    labels_c = [str(i + 1) for i in range(k_c)]
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=labels_c, y=counts_c, mode="lines+markers",
        marker=dict(color="#4C78A8", size=7), line=dict(color="#4C78A8", width=1.5),
        name="c (defect count)",
    ))
    fig_c.add_hline(y=c_bar_calc, line_dash="solid", line_color="#54A24B",
                    annotation_text=f"c̄ = {c_bar_calc:.2f}")
    fig_c.add_hline(y=ucl_c, line_dash="dash", line_color="#E45756",
                    annotation_text=f"UCL = {ucl_c:.2f}")
    fig_c.add_hline(y=lcl_c, line_dash="dash", line_color="#E45756",
                    annotation_text=f"LCL = {lcl_c:.2f}")
    for i in range(k_c):
        if counts_c[i] > ucl_c or counts_c[i] < lcl_c:
            fig_c.add_trace(go.Scatter(
                x=[labels_c[i]], y=[counts_c[i]], mode="markers",
                marker=dict(color="#E45756", size=12, symbol="x"), showlegend=False,
            ))
    if shift_c != "None":
        fig_c.add_vline(x=shift_pt_c - 0.5, line_dash="dot", line_color="white", opacity=0.5,
                        annotation_text="Shift")
    fig_c.update_layout(template="plotly_dark", height=400,
                         xaxis_title="Subgroup", yaxis_title="Defect Count",
                         margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_c, use_container_width=True, key="spc_c_chart")

    c1, c2 = st.columns(2)
    c1.metric("c̄", f"{c_bar_calc:.2f}")
    c2.metric("Subgroups", str(k_c))

    if show_rules_c:
        st.subheader("Shewhart Rule Violations", divider="gray")
        rules_c = _check_shewhart_rules(counts_c.astype(float), c_bar_calc, sigma_c_chart)
        if rules_c:
            for r in rules_c:
                st.markdown(r)
        else:
            st.success("✅ No violations detected.")

    with st.expander("About the c-Chart"):
        st.markdown("""
        | Element | Formula |
        |---------|---------|
        | c̄ (center) | Average defect count per unit |
        | UCL | c̄ + k·√c̄ |
        | LCL | max(0, c̄ − k·√c̄) |
        | Assumptions | Poisson distribution, constant opportunity area |
        """)


def _render_u_chart():
    """Defects per unit (u) chart — varying opportunity."""
    st.subheader("u-Chart — Defects per Unit")
    st.info("""
    **u-Chart** monitors defects per unit when the area of opportunity varies.
    Examples: defects per batch of different sizes, errors per document of varying length.
    """)

    rng = np.random.default_rng(42)

    with st.sidebar:
        st.markdown("##### :orange[Process Parameters]")
        u_bar = st.slider("Average Defect Rate (ū)", 0.5, 15.0, 3.0, 0.5, key="spc_u_mean")
        k_u = st.slider("Number of Subgroups", 15, 100, 25, 5, key="spc_u_k")
        var_opp = st.checkbox("Varying Opportunity", True, key="spc_u_var")
        if var_opp:
            opp_range = st.slider("Opportunity Range", 10, 200, (30, 100), key="spc_u_range")
            opps = rng.integers(opp_range[0], opp_range[1] + 1, k_u)
        else:
            opp_fixed = st.slider("Opportunity per Unit", 20, 200, 50, 10, key="spc_u_fixed")
            opps = np.full(k_u, opp_fixed, dtype=int)

        shift_u = st.selectbox("Shift Type", ["None", "Defect Rate Increase", "Defect Rate Decrease"], key="spc_u_shift")
        if shift_u == "Defect Rate Increase":
            u_after = u_bar + st.slider("Increase in ū", 0.5, 10.0, 3.0, 0.5, key="spc_u_up")
        elif shift_u == "Defect Rate Decrease":
            u_after = max(0.1, u_bar - st.slider("Decrease in ū", 0.5, 10.0, 2.0, 0.5, key="spc_u_down"))
        else:
            u_after = u_bar

        st.markdown("---")
        st.markdown("##### :orange[Chart Settings]")
        sigma_u = st.slider("Control Limit Width (σ)", 1, 4, 3, 1, key="spc_u_sigma")
        show_rules_u = st.checkbox("Show Shewhart Rules", True, key="spc_u_rules")

    shift_pt_u = k_u // 2
    defects_u = np.zeros(k_u, dtype=int)
    for i in range(k_u):
        rate = u_after if i >= shift_pt_u and shift_u != "None" else u_bar
        defects_u[i] = rng.poisson(rate * opps[i])

    ui = defects_u / opps
    u_bar_calc = defects_u.sum() / opps.sum()
    z_u = sigma_u
    se_u = np.sqrt(u_bar_calc / opps)
    ucl_u = u_bar_calc + z_u * se_u
    lcl_u = np.maximum(0, u_bar_calc - z_u * se_u)

    # Fixed limits display
    opp_avg = opps.mean()
    se_u_avg = np.sqrt(u_bar_calc / opp_avg)
    ucl_u_fixed = u_bar_calc + z_u * se_u_avg
    lcl_u_fixed = max(0, u_bar_calc - z_u * se_u_avg)

    labels_u = [str(i + 1) for i in range(k_u)]
    fig_u = go.Figure()
    fig_u.add_trace(go.Scatter(
        x=labels_u, y=ui, mode="lines+markers",
        marker=dict(color="#4C78A8", size=7), line=dict(color="#4C78A8", width=1.5),
        name="u (defects/unit)",
    ))
    fig_u.add_hline(y=u_bar_calc, line_dash="solid", line_color="#54A24B",
                    annotation_text=f"ū = {u_bar_calc:.4f}")
    fig_u.add_hline(y=ucl_u_fixed, line_dash="dash", line_color="#E45756",
                    annotation_text=f"UCL (avg opp) = {ucl_u_fixed:.4f}")
    fig_u.add_hline(y=lcl_u_fixed, line_dash="dash", line_color="#E45756",
                    annotation_text=f"LCL (avg opp) = {lcl_u_fixed:.4f}")
    if var_opp:
        fig_u.add_trace(go.Scatter(
            x=labels_u + labels_u[::-1],
            y=ucl_u.tolist() + lcl_u[::-1].tolist(),
            fill="toself", fillcolor="rgba(228, 87, 86, 0.08)",
            line=dict(color="rgba(228, 87, 86, 0)"), showlegend=False,
        ))
    for i in range(k_u):
        if ui[i] > ucl_u[i] or ui[i] < lcl_u[i]:
            fig_u.add_trace(go.Scatter(
                x=[labels_u[i]], y=[ui[i]], mode="markers",
                marker=dict(color="#E45756", size=12, symbol="x"), showlegend=False,
            ))
    if shift_u != "None":
        fig_u.add_vline(x=shift_pt_u - 0.5, line_dash="dot", line_color="white", opacity=0.5,
                        annotation_text="Shift")
    fig_u.update_layout(template="plotly_dark", height=400,
                         xaxis_title="Subgroup", yaxis_title="Defects per Unit",
                         margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_u, use_container_width=True, key="spc_u_chart")

    c1, c2, c3 = st.columns(3)
    c1.metric("ū", f"{u_bar_calc:.4f}")
    c2.metric("Total Defects", f"{defects_u.sum()}")
    c3.metric("Subgroups", str(k_u))

    if show_rules_u:
        st.subheader("Shewhart Rule Violations", divider="gray")
        rules_u = _check_shewhart_rules(ui, u_bar_calc, np.std(ui, ddof=1) if len(ui) > 1 else 0)
        if rules_u:
            for r in rules_u:
                st.markdown(r)
        else:
            st.success("✅ No violations detected.")

    with st.expander("About the u-Chart"):
        st.markdown("""
        | Element | Formula |
        |---------|---------|
        | ū (center) | Total defects / Total opportunity |
        | UCL | ū + k·√(ū / nᵢ) — varies with opportunity nᵢ |
        | LCL | max(0, ū − k·√(ū / nᵢ)) |
        | Assumptions | Poisson distribution, defects are rare and independent |
        """)


def _render_cusum_chart():
    st.subheader("CUSUM Chart — Cumulative Sum Control")
    st.info("""
    **CUSUM** accumulates deviations from a target value to detect **small persistent shifts**
    (0.5–1.5σ) much faster than X̄ or I-MR charts. Two one-sided statistics track shifts
    upward (S⁺) and downward (S⁻).
    """)

    rng = np.random.default_rng(42)

    with st.sidebar:
        st.markdown("##### :orange[Process Parameters]")
        target = st.number_input("Target Mean (μ₀)", 90.0, 110.0, 100.0, 0.5, key="cusum_target")
        sigma_c = st.slider("Process Sigma (σ)", 0.5, 5.0, 2.0, 0.1, key="cusum_sigma")
        n_c = st.slider("Number of Observations", 20, 200, 60, 5, key="cusum_n")

        shift_cusum = st.selectbox("Shift Type", ["None", "Mean Shift", "Gradual Drift"], key="cusum_shift")
        if shift_cusum == "Mean Shift":
            shift_mag_c = st.slider("Shift Magnitude (σ units)", 0.0, 3.0, 1.0, 0.1, key="cusum_shift_mag")
        elif shift_cusum == "Gradual Drift":
            drift_c = st.slider("Drift at end (σ units)", 0.0, 4.0, 2.0, 0.1, key="cusum_drift")
        else:
            shift_mag_c = 0

        st.markdown("---")
        st.markdown("##### :orange[CUSUM Parameters]")
        k_c = st.slider("Reference Value K (σ units)", 0.0, 2.0, 0.5, 0.05, key="cusum_k",
                        help="Typically 0.5σ. Smaller K = more sensitive to small shifts.")
        h_c = st.slider("Decision Interval H (σ units)", 1.0, 8.0, 5.0, 0.5, key="cusum_h",
                        help="Typically 4–5σ. Smaller H = faster detection, more false alarms.")
        show_cusum_rules = st.checkbox("Show OOC Signals", True, key="cusum_rules")

    # Generate data
    shift_pt_c = n_c // 2
    raw_c = rng.normal(target, sigma_c, n_c)
    if shift_cusum == "Mean Shift":
        raw_c[shift_pt_c:] += shift_mag_c * sigma_c
    elif shift_cusum == "Gradual Drift":
        drift_vec = np.linspace(0, drift_c * sigma_c, n_c - shift_pt_c)
        raw_c[shift_pt_c:] += drift_vec

    # Compute CUSUM statistics
    K = k_c * sigma_c
    H = h_c * sigma_c
    S_plus = np.zeros(n_c)
    S_minus = np.zeros(n_c)
    for i in range(n_c):
        if i == 0:
            S_plus[i] = max(0, (raw_c[i] - target) - K)
            S_minus[i] = max(0, -(raw_c[i] - target) - K)
        else:
            S_plus[i] = max(0, S_plus[i-1] + (raw_c[i] - target) - K)
            S_minus[i] = max(0, S_minus[i-1] - (raw_c[i] - target) - K)

    # Signal detection
    signals_plus = S_plus > H
    signals_minus = S_minus > H

    # Plot
    labels_c = [str(i + 1) for i in range(n_c)]
    fig_cusum = make_subplots(rows=2, cols=1, subplot_titles=["CUSUM S⁺ (upward shift)", "CUSUM S⁻ (downward shift)"],
                               vertical_spacing=0.12)

    # S+ chart
    fig_cusum.add_trace(go.Scatter(x=labels_c, y=S_plus, mode="lines+markers",
                                     marker=dict(color="#4C78A8", size=5), line=dict(color="#4C78A8", width=1.5),
                                     name="S⁺"), row=1, col=1)
    fig_cusum.add_hline(y=H, line_dash="dash", line_color="#E45756",
                         annotation_text=f"H = {H:.2f}", row=1, col=1)
    for i in range(n_c):
        if signals_plus[i]:
            fig_cusum.add_trace(go.Scatter(x=[labels_c[i]], y=[S_plus[i]], mode="markers",
                                             marker=dict(color="#E45756", size=10, symbol="x"), showlegend=False),
                                 row=1, col=1)

    # S- chart
    fig_cusum.add_trace(go.Scatter(x=labels_c, y=S_minus, mode="lines+markers",
                                     marker=dict(color="#54A24B", size=5), line=dict(color="#54A24B", width=1.5),
                                     name="S⁻"), row=2, col=1)
    fig_cusum.add_hline(y=H, line_dash="dash", line_color="#E45756",
                         annotation_text=f"H = {H:.2f}", row=2, col=1)
    for i in range(n_c):
        if signals_minus[i]:
            fig_cusum.add_trace(go.Scatter(x=[labels_c[i]], y=[S_minus[i]], mode="markers",
                                             marker=dict(color="#E45756", size=10, symbol="x"), showlegend=False),
                                 row=2, col=1)

    # Shift reference
    if shift_cusum != "None":
        fig_cusum.add_vline(x=shift_pt_c - 0.5, line_dash="dot", line_color="white", opacity=0.5,
                             annotation_text="Shift", row=1, col=1)
        fig_cusum.add_vline(x=shift_pt_c - 0.5, line_dash="dot", line_color="white", opacity=0.3, row=2, col=1)

    fig_cusum.update_layout(template="plotly_dark", height=500, showlegend=False,
                             margin=dict(l=10, r=10, t=30, b=10))
    fig_cusum.update_xaxes(title_text="Observation", row=2, col=1)
    st.plotly_chart(fig_cusum, use_container_width=True, key="spc_cusum_chart")

    # Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target (μ₀)", f"{target:.2f}")
    c2.metric("K", f"{K:.3f} ({k_c:.2f}σ)")
    c3.metric("H", f"{H:.3f} ({h_c:.2f}σ)")
    c4.metric("Mean (data)", f"{raw_c.mean():.3f}")

    if show_cusum_rules:
        st.subheader("Out-of-Control Signals", divider="gray")
        ooc_list = []
        for i in range(n_c):
            if signals_plus[i]:
                ooc_list.append(f"🔴 Observation {i+1}: S⁺ = {S_plus[i]:.3f} exceeds H = {H:.3f} (upward shift)")
            if signals_minus[i]:
                ooc_list.append(f"🔴 Observation {i+1}: S⁻ = {S_minus[i]:.3f} exceeds H = {H:.3f} (downward shift)")
        if ooc_list:
            for msg in ooc_list:
                st.markdown(msg)
        else:
            st.success("✅ No OOC signals detected. Process is stable around target.")

    with st.expander("About CUSUM"):
        st.markdown("""
        | Element | Description |
        |---------|-------------|
        | **S⁺** | max(0, S⁺₍ᵢ₋₁₎ + (xᵢ − μ₀) − K) — accumulates *positive* deviations |
        | **S⁻** | max(0, S⁻₍ᵢ₋₁₎ − (xᵢ − μ₀) − K) — accumulates *negative* deviations |
        | **K** | Reference value — only deviations > K are accumulated |
        | **H** | Decision interval — when S > H, signal an out-of-control condition |
        | **ARL₀** | For K=0.5, H=5: ~465 (in-control ARL, matching 3σ X̄ chart) |
        | **ARL₁** | For a 1σ shift: ~10 vs ~43 for X̄ chart (4.3× faster detection) |
        """)


def _render_ewma_chart():
    st.subheader("EWMA Chart — Exponentially Weighted Moving Average")
    st.info("""
    **EWMA** applies exponentially decreasing weights to past observations, making it
    sensitive to **small shifts** while filtering out noise. The smoothing parameter
    λ controls the memory: low λ = smoother, high λ = more responsive.
    """)

    rng = np.random.default_rng(42)

    with st.sidebar:
        st.markdown("##### :orange[Process Parameters]")
        target_e = st.number_input("Target Mean (μ₀)", 90.0, 110.0, 100.0, 0.5, key="ewma_target")
        sigma_e = st.slider("Process Sigma (σ)", 0.5, 5.0, 2.0, 0.1, key="ewma_sigma")
        n_e = st.slider("Number of Observations", 15, 200, 50, 5, key="ewma_n")

        shift_ewma = st.selectbox("Shift Type", ["None", "Mean Shift", "Gradual Drift", "Transient Spike"], key="ewma_shift")
        if shift_ewma == "Mean Shift":
            shift_mag_e = st.slider("Shift Magnitude (σ units)", 0.0, 3.0, 1.0, 0.1, key="ewma_shift_mag")
        elif shift_ewma == "Gradual Drift":
            drift_e = st.slider("Drift at end (σ units)", 0.0, 4.0, 2.0, 0.1, key="ewma_drift")
        elif shift_ewma == "Transient Spike":
            spike_mag = st.slider("Spike Magnitude (σ units)", 1.0, 6.0, 3.0, 0.5, key="ewma_spike")
            spike_pos = st.slider("Spike Position", 0.2, 0.8, 0.4, 0.05, key="ewma_spike_pos")
        else:
            shift_mag_e = 0

        st.markdown("---")
        st.markdown("##### :orange[EWMA Parameters]")
        lam = st.slider("Smoothing Constant λ", 0.02, 1.0, 0.2, 0.01, key="ewma_lambda",
                        help="Lower λ = smoother, more weight on past. Typical: 0.05–0.3.")
        L_e = st.slider("Control Limit Width L", 1.0, 4.0, 3.0, 0.1, key="ewma_L",
                        help="Typically 3. Wider limits = fewer false alarms.")
        show_ewma_rules = st.checkbox("Show OOC Signals", True, key="ewma_rules")

    # Generate data
    shift_pt_e = n_e // 2
    raw_e = rng.normal(target_e, sigma_e, n_e)
    if shift_ewma == "Mean Shift":
        raw_e[shift_pt_e:] += shift_mag_e * sigma_e
    elif shift_ewma == "Gradual Drift":
        drift_vec_e = np.linspace(0, drift_e * sigma_e, n_e - shift_pt_e)
        raw_e[shift_pt_e:] += drift_vec_e
    elif shift_ewma == "Transient Spike":
        spike_idx = int(n_e * spike_pos)
        raw_e[spike_idx] += spike_mag * sigma_e

    # Compute EWMA
    z = np.zeros(n_e)
    z[0] = target_e
    for i in range(1, n_e):
        z[i] = lam * raw_e[i] + (1 - lam) * z[i-1]

    # Control limits (time-varying, converging to steady-state)
    factor = np.sqrt(lam / (2 - lam))
    cl_e = target_e
    steady_state_sigma = sigma_e * factor
    ucl_e_steady = cl_e + L_e * steady_state_sigma
    lcl_e_steady = cl_e - L_e * steady_state_sigma

    ucl_e = np.full(n_e, ucl_e_steady)
    lcl_e = np.full(n_e, lcl_e_steady)
    # Time-varying limits
    for i in range(1, n_e):
        w = np.sqrt((1 - (1 - lam) ** (2 * i)) * lam / (2 - lam))
        ucl_e[i] = cl_e + L_e * sigma_e * w
        lcl_e[i] = cl_e - L_e * sigma_e * w

    # Signal detection
    signals_e = (z > ucl_e) | (z < lcl_e)

    # Plot
    labels_e = [str(i + 1) for i in range(n_e)]
    fig_ewma = go.Figure()

    # Raw data
    fig_ewma.add_trace(go.Scatter(x=labels_e, y=raw_e, mode="markers",
                                    marker=dict(color="gray", size=4, opacity=0.4),
                                    name="Raw observations"))

    # EWMA line
    fig_ewma.add_trace(go.Scatter(x=labels_e, y=z, mode="lines+markers",
                                    marker=dict(color="#4C78A8", size=6),
                                    line=dict(color="#4C78A8", width=2),
                                    name=f"EWMA (λ={lam})"))

    # Center line
    fig_ewma.add_hline(y=cl_e, line_dash="solid", line_color="#54A24B",
                        annotation_text=f"μ₀ = {cl_e:.2f}")

    # Steady-state limits
    fig_ewma.add_hline(y=ucl_e_steady, line_dash="dash", line_color="#E45756",
                        annotation_text=f"UCL (ss) = {ucl_e_steady:.3f}")
    fig_ewma.add_hline(y=lcl_e_steady, line_dash="dash", line_color="#E45756",
                        annotation_text=f"LCL (ss) = {lcl_e_steady:.3f}")

    # Time-varying limits as ribbon
    fig_ewma.add_trace(go.Scatter(
        x=labels_e + labels_e[::-1],
        y=ucl_e.tolist() + lcl_e[::-1].tolist(),
        fill="toself", fillcolor="rgba(228, 87, 86, 0.06)",
        line=dict(color="rgba(228, 87, 86, 0)"), showlegend=False,
        name="Time-varying limits",
    ))

    # OOC points
    for i in range(n_e):
        if signals_e[i]:
            fig_ewma.add_trace(go.Scatter(
                x=[labels_e[i]], y=[z[i]], mode="markers",
                marker=dict(color="#E45756", size=12, symbol="x"), showlegend=False,
            ))

    if shift_ewma != "None" and shift_ewma != "Transient Spike":
        fig_ewma.add_vline(x=shift_pt_e - 0.5, line_dash="dot", line_color="white", opacity=0.5,
                            annotation_text="Shift")
    elif shift_ewma == "Transient Spike":
        fig_ewma.add_vline(x=spike_idx - 0.5, line_dash="dot", line_color="#F58518", opacity=0.7,
                            annotation_text="Spike")

    fig_ewma.update_layout(template="plotly_dark", height=450,
                            xaxis_title="Observation", yaxis_title="Value",
                            margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_ewma, use_container_width=True, key="spc_ewma_chart")

    # Summary
    st.subheader("Process Summary", divider="gray")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("λ", f"{lam:.3f}")
    c2.metric("L", f"{L_e:.2f}")
    c3.metric("EWMA mean", f"{z[-1]:.3f}")
    c4.metric("Data mean", f"{raw_e.mean():.3f}")

    if show_ewma_rules:
        st.subheader("Out-of-Control Signals", divider="gray")
        ooc_e = []
        for i in range(n_e):
            if signals_e[i]:
                side = "above" if z[i] > ucl_e[i] else "below"
                ooc_e.append(f"🔴 Observation {i+1}: EWMA = {z[i]:.3f} ({side} limit)")
        if ooc_e:
            for msg in ooc_e:
                st.markdown(msg)
        else:
            st.success("✅ No OOC signals detected.")

    with st.expander("About EWMA"):
        st.markdown(f"""
        | Element | Formula |
        |---------|---------|
        | **EWMA** | Zᵢ = λ·xᵢ + (1−λ)·Zᵢ₋₁ |
        | **Starting value** | Z₀ = μ₀ (target) |
        | **Steady-state σ** | σ_Z = σ·√(λ/(2−λ)) |
        | **Time-varying σ** | σ_Zᵢ = σ·√((1−(1−λ)^{{2i}})·λ/(2−λ)) |
        | **Effective memory** | ~1/λ observations ("age" of oldest non-negligible weight) |

        **Typical λ choices:**
        - λ = 0.05 – very smooth, detects shifts < 0.5σ
        - λ = 0.20 – balanced, general purpose
        - λ = 0.40 – more responsive, closer to X̄ chart
        """)

def render_control_charts():
    st.title("Statistical Process Control (SPC) Charts")
    st.markdown("""
    Interactive control charts for monitoring process stability and capability.
    Adjust parameters to see how control charts detect out-of-control conditions
    and how capability indices quantify process performance.
    """)

    # ── SIDEBAR CONTROLS ──
    with st.sidebar:
        st.markdown("##### :orange[Chart Family]")
        chart_family = st.radio(
            "Family",
            ["Variable (X̄-R, X̄-S, I-MR)", "Attribute (p, np, c, u)",
         "Advanced (CUSUM, EWMA)"],
            key="spc_family",
        )

        st.markdown("---")

        if chart_family.startswith("Attribute"):
            attr_chart = st.selectbox(
                "Chart Type",
                ["p-Chart (proportion defective)", "np-Chart (number defective)",
                 "c-Chart (defect count)", "u-Chart (defects per unit)"],
                key="spc_attr_type",
            )
            st.markdown("---")
            st.markdown("Attribute charts monitor **counts or proportions** of defects/defectives.")
        elif chart_family.startswith("Advanced"):
            adv_chart = st.selectbox(
                "Chart Type",
                ["CUSUM", "EWMA"],
                key="spc_adv_type",
            )
            st.markdown("---")
            st.markdown("Advanced charts detect **small shifts** faster than traditional Shewhart charts.")

    # Route to chart renderers
    if chart_family.startswith("Attribute"):
        if attr_chart.startswith("p-Chart"):
            _render_p_chart()
        elif attr_chart.startswith("np-Chart"):
            _render_np_chart()
        elif attr_chart.startswith("c-Chart"):
            _render_c_chart()
        else:
            _render_u_chart()
        return

    if chart_family.startswith("Advanced"):
        if adv_chart == "CUSUM":
            _render_cusum_chart()
        else:
            _render_ewma_chart()
        return

    # ── VARIABLE CHARTS (existing code) ──
    with st.sidebar:
        st.markdown("##### :orange[Data Source]")
        data_src = st.radio(
            "Source",
            ["Simulated Process", "Upload CSV"],
            key="spc_data_src",
        )

        if data_src == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV (one column of measurements, no header)",
                type=["csv"],
                key="spc_upload",
            )
            if uploaded_file is None:
                st.info("Upload a CSV file with one column of numeric measurements.")
                st.stop()
            raw_data = pd.read_csv(uploaded_file, header=None).iloc[:, 0].dropna().values
            if len(raw_data) < 10:
                st.error("Need at least 10 measurements.")
                st.stop()

            chart_type = st.selectbox(
                "Chart Type",
                ["X̄-R (subgroup)", "X̄-S (subgroup)", "I-MR (individuals)"],
                key="spc_chart_type_file",
            )
            if "subgroup" in chart_type:
                n_subgroup = st.slider("Subgroup Size (n)", 2, 15, 5, 1, key="spc_n_file")
                n_subgroups = len(raw_data) // n_subgroup
                base_data = raw_data[:n_subgroups * n_subgroup].reshape(n_subgroups, n_subgroup)
            else:
                base_data = raw_data.reshape(-1, 1)
                n_subgroup = 1
                n_subgroups = len(base_data)
            sigma_limits = st.slider("Control Limit Width (σ)", 1, 4, 3, 1, key="spc_sigma_file")
            show_rules = st.checkbox("Show Shewhart Rules", True, key="spc_rules_file")
            use_phase12 = st.checkbox("Phase 1/2 (Trial Limits)", False, key="spc_phase12_file",
                                      help="Mark Phase 1 baseline period, identify OOC subgroups, recalculate revised limits")
            n_phase1 = n_subgroups // 2
            if use_phase12:
                n_phase1 = st.slider("Phase 1 Subgroups", 5, n_subgroups - 1, max(5, n_subgroups // 2), 1,
                                     key="spc_phase1_n_file",
                                     help="Number of subgroups used for initial (trial) control limits")
            use_spec = st.checkbox("Show Specification Limits", False, key="spc_spec_file")
            lsl, usl = None, None
            if use_spec:
                c1, c2 = st.columns(2)
                data_min, data_max = float(np.min(raw_data)), float(np.max(raw_data))
                with c1:
                    lsl = st.number_input("LSL", data_min - 10, float(np.mean(raw_data)), data_min * 0.9, 0.5, key="spc_lsl_file")
                with c2:
                    usl = st.number_input("USL", float(np.mean(raw_data)), data_max + 10, data_max * 1.1, 0.5, key="spc_usl_file")
        else:
            st.markdown("##### :orange[Process Parameters]")
            process_mean = st.slider("Process Mean (μ)", 90.0, 110.0, 100.0, 0.5, key="spc_mean")
            process_sigma = st.slider("Process Std (σ)", 0.5, 5.0, 2.0, 0.1, key="spc_sigma")
            shift_type = st.selectbox("Shift Type", ["None", "Mean Shift", "Variance Increase", "Gradual Drift"], key="spc_shift_type")
            if shift_type == "Mean Shift":
                shift_magnitude = st.slider("Mean Shift (σ units)", 0.0, 3.0, 1.0, 0.1, key="spc_shift_mag")
            elif shift_type == "Variance Increase":
                var_increase = st.slider("Variance Multiplier", 1.0, 4.0, 2.0, 0.1, key="spc_var_mult")
            elif shift_type == "Gradual Drift":
                drift_end = st.slider("Drift at end (σ units)", 0.0, 4.0, 2.0, 0.1, key="spc_drift")
            else:
                shift_magnitude, var_increase, drift_end = 0, 1.0, 0

            st.markdown("---")
            st.markdown("##### :orange[Chart Settings]")
            chart_type = st.selectbox(
                "Chart Type",
                ["X̄-R", "X̄-S", "I-MR (individuals)"],
                key="spc_chart_type",
            )
            mult = 1 if chart_type == "I-MR (individuals)" else None
            if chart_type == "I-MR (individuals)":
                n_individuals = st.slider("Number of Observations", 15, 200, 50, 5, key="spc_n_ind")
                n_subgroup = 1
                n_subgroups = n_individuals
            else:
                n_subgroup = st.slider("Subgroup Size (n)", 2, 15, 5, 1, key="spc_n")
                n_subgroups = st.slider("Number of Subgroups", 10, 100, 25, 5, key="spc_k")
            sigma_limits = st.slider("Control Limit Width (σ)", 1, 4, 3, 1, key="spc_sigma_limits")
            show_rules = st.checkbox("Show Shewhart Rules", True, key="spc_rules")
            use_phase12 = st.checkbox("Phase 1/2 (Trial Limits)", False, key="spc_phase12",
                                      help="Mark Phase 1 baseline period, identify OOC subgroups, recalculate revised limits")
            n_phase1 = n_subgroups // 2
            if use_phase12:
                n_phase1 = st.slider("Phase 1 Subgroups", 5, n_subgroups - 1, max(5, n_subgroups // 2), 1,
                                     key="spc_phase1_n",
                                     help="Number of subgroups used for initial (trial) control limits")

            st.markdown("---")
            st.markdown("##### :orange[Capability Settings]")
            use_spec = st.checkbox("Show Specification Limits", True, key="spc_spec")
            if use_spec:
                c1, c2 = st.columns(2)
                with c1:
                    lsl = st.number_input("LSL", 80.0, 99.0, 94.0, 0.5, key="spc_lsl")
                with c2:
                    usl = st.number_input("USL", 101.0, 120.0, 106.0, 0.5, key="spc_usl")
            else:
                lsl, usl = None, None

    # ── GENERATE PROCESS DATA ──
    if data_src != "Upload CSV":
        rng = np.random.default_rng(42)
        if chart_type == "I-MR (individuals)":
            raw_individuals = rng.normal(process_mean, process_sigma, n_individuals)
            shift_point = n_individuals // 2
            if shift_type == "Mean Shift":
                raw_individuals[shift_point:] += shift_magnitude * process_sigma
            elif shift_type == "Variance Increase":
                raw_individuals[shift_point:] = rng.normal(
                    process_mean, process_sigma * np.sqrt(var_increase), n_individuals - shift_point
                )
            elif shift_type == "Gradual Drift":
                drift = np.linspace(0, drift_end * process_sigma, n_individuals - shift_point)
                raw_individuals[shift_point:] += drift
            base_data = raw_individuals.reshape(-1, 1)
        else:
            base_data = rng.normal(process_mean, process_sigma, (n_subgroups, n_subgroup))
            shift_point = n_subgroups // 2
            if shift_type == "Mean Shift":
                base_data[shift_point:] = rng.normal(
                    process_mean + shift_magnitude * process_sigma,
                    process_sigma,
                    (n_subgroups - shift_point, n_subgroup),
                )
            elif shift_type == "Variance Increase":
                base_data[shift_point:] = rng.normal(
                    process_mean,
                    process_sigma * np.sqrt(var_increase),
                    (n_subgroups - shift_point, n_subgroup),
                )
            elif shift_type == "Gradual Drift":
                for i in range(shift_point, n_subgroups):
                    drift = drift_end * process_sigma * (i - shift_point) / (n_subgroups - shift_point)
                    base_data[i] = rng.normal(process_mean + drift, process_sigma, n_subgroup)

    is_imr = "I-MR" in chart_type
    use_s_chart = "X̄-S" in chart_type
    use_r_chart = "X̄-R" in chart_type

    if is_imr:
        # Individual chart
        individuals = base_data.flatten()
        k = len(individuals)
        moving_ranges = np.abs(np.diff(individuals))
        mr_bar = np.mean(moving_ranges)
        grand_mean = np.mean(individuals)

        const = _get_constants(2)
        d2 = const["d2"]
        sigma_within = mr_bar / d2

        z_mult = sigma_limits
        cl_x = grand_mean
        ucl_x = grand_mean + _E2 * mr_bar * (z_mult / 3)
        lcl_x = grand_mean - _E2 * mr_bar * (z_mult / 3)

        cl_r = mr_bar
        ucl_r = const["D4"] * mr_bar
        lcl_r = const["D3"] * mr_bar

        # Compute stats for charting
        top_stat = individuals
        bottom_stat = np.concatenate([[np.nan], moving_ranges])

        sigma_top = sigma_within
        sigma_bottom = np.std(moving_ranges, ddof=1) if len(moving_ranges) > 1 else 0

        xbar_rules = []
        mr_rules = []
        if show_rules:
            xbar_rules = _check_shewhart_rules(individuals, cl_x, sigma_top)
            mr_rules = _check_shewhart_rules(moving_ranges, cl_r, sigma_bottom)

        center_bottom = cl_r
        ucl_bottom = ucl_r
        lcl_bottom = lcl_r

        top_label = "Individual Value (X)"
        bottom_label = "Moving Range (mR)"
        title_suffix = "I-MR Chart"
        all_data = individuals

    elif use_s_chart:
        subgroup_means = np.mean(base_data, axis=1)
        subgroup_stds = np.std(base_data, axis=1, ddof=1)
        n_s = n_subgroup
        const = _get_constants(n_s)
        grand_mean = np.mean(subgroup_means)
        mean_std = np.mean(subgroup_stds)
        c4 = const["c4"]
        sigma_within = mean_std / c4
        B3 = const["B3"]
        B4 = const["B4"]

        z_mult = sigma_limits
        cl_x = grand_mean
        ucl_x = grand_mean + z_mult * sigma_within / np.sqrt(n_s)
        lcl_x = grand_mean - z_mult * sigma_within / np.sqrt(n_s)

        cl_s = mean_std
        ucl_s = B4 * mean_std
        lcl_s = B3 * mean_std

        top_stat = subgroup_means
        bottom_stat = subgroup_stds
        sigma_top = sigma_within / np.sqrt(n_s)
        sigma_bottom = np.std(subgroup_stds, ddof=1) if len(subgroup_stds) > 1 else 0

        xbar_rules = []
        s_rules = []
        if show_rules:
            xbar_rules = _check_shewhart_rules(subgroup_means, cl_x, sigma_top)
            s_rules = _check_shewhart_rules(subgroup_stds, cl_s, sigma_bottom)

        center_bottom = cl_s
        ucl_bottom = ucl_s
        lcl_bottom = lcl_s

        top_label = "Subgroup Mean (X̄)"
        bottom_label = "Subgroup Std (S)"
        title_suffix = f"X̄-S Chart (n={n_s})"
        all_data = base_data.flatten()

    else:
        # X̄-R (default)
        subgroup_means = np.mean(base_data, axis=1)
        subgroup_ranges = np.ptp(base_data, axis=1)
        n_s = n_subgroup
        const = _get_constants(n_s)
        grand_mean = np.mean(subgroup_means)
        mean_range = np.mean(subgroup_ranges)
        A2 = const["A2"]
        d2 = const["d2"]
        D3 = const["D3"]
        D4 = const["D4"]
        sigma_within = mean_range / d2

        z_mult = sigma_limits
        cl_x = grand_mean
        ucl_x = grand_mean + (z_mult / 3) * A2 * mean_range
        lcl_x = grand_mean - (z_mult / 3) * A2 * mean_range

        cl_r = mean_range
        ucl_r = D4 * mean_range
        lcl_r = D3 * mean_range

        top_stat = subgroup_means
        bottom_stat = subgroup_ranges
        sigma_top = sigma_within / np.sqrt(n_s)
        sigma_bottom = np.std(subgroup_ranges, ddof=1) if len(subgroup_ranges) > 1 else 0

        xbar_rules = []
        r_rules = []
        if show_rules:
            xbar_rules = _check_shewhart_rules(subgroup_means, cl_x, sigma_top)
            r_rules = _check_shewhart_rules(subgroup_ranges, cl_r, sigma_bottom)

        center_bottom = cl_r
        ucl_bottom = ucl_r
        lcl_bottom = lcl_r

        top_label = "Subgroup Mean (X̄)"
        bottom_label = "Subgroup Range (R)"
        title_suffix = f"X̄-R Chart (n={n_s})"
        all_data = base_data.flatten()

    all_rules = xbar_rules + mr_rules if is_imr else xbar_rules + (r_rules if use_r_chart else s_rules)
    n_points = len(top_stat)

    # ── PHASE 1/2 (TRIAL LIMITS) ──
    trial_limits = False
    revised_ucl_x = revised_lcl_x = revised_cl_x = None
    revised_ucl_b = revised_lcl_b = revised_cl_b = None
    phase1_end = n_phase1 if use_phase12 else 0

    if use_phase12 and n_phase1 >= 5:
        trial_limits = True
        phase1_mask = np.arange(n_points) < n_phase1
        phase1_stat = top_stat[phase1_mask]
        phase1_bot = bottom_stat[phase1_mask]

        # Trial limits (Phase 1)
        trial_cl_x = np.mean(phase1_stat)
        trial_cl_b = np.mean(phase1_bot)
        trial_sigma_top = np.std(phase1_stat, ddof=1) if len(phase1_stat) > 1 else sigma_top
        trial_sigma_bot = np.std(phase1_bot, ddof=1) if len(phase1_bot) > 1 else sigma_bottom

        z_trial = sigma_limits
        trial_ucl_x = trial_cl_x + z_trial * trial_sigma_top
        trial_lcl_x = trial_cl_x - z_trial * trial_sigma_top
        trial_ucl_b = trial_cl_b + z_trial * trial_sigma_bot
        trial_lcl_b = max(0, trial_cl_b - z_trial * trial_sigma_bot)

        # Identify OOC points in Phase 1
        ooc_phase1_idx = []
        for i in range(n_phase1):
            if top_stat[i] > trial_ucl_x or top_stat[i] < trial_lcl_x:
                ooc_phase1_idx.append(i)
            if bottom_stat[i] > trial_ucl_b or bottom_stat[i] < trial_lcl_b:
                if i not in ooc_phase1_idx:
                    ooc_phase1_idx.append(i)

        # Recalculate button (session state toggles)
        recalc_key = "spc_recalc_limits"
        if recalc_key not in st.session_state:
            st.session_state[recalc_key] = False

        st.subheader("Phase 1 / Phase 2 Analysis", divider="orange")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"**Phase 1:** {n_phase1} subgroups (trial limits)  •  "
                        f"**Phase 2:** {n_points - n_phase1} subgroups (monitoring)")
            if ooc_phase1_idx:
                st.warning(f"⚠️ {len(ooc_phase1_idx)} OOC subgroup(s) in Phase 1: "
                           f"{', '.join(str(i+1) for i in ooc_phase1_idx)}")
            else:
                st.success("✅ No OOC points in Phase 1 — trial limits are stable.")
        with c2:
            if ooc_phase1_idx and not st.session_state[recalc_key]:
                if st.button("🔄 Recalculate Limits", key="spc_recalc_btn",
                             help="Remove OOC Phase 1 subgroups and recalculate control limits"):
                    st.session_state[recalc_key] = True
                    st.rerun()

        if st.session_state[recalc_key]:
            # Remove OOC Phase 1 points and recalculate
            keep_mask = np.ones(n_phase1, dtype=bool)
            for idx in ooc_phase1_idx:
                keep_mask[idx] = False

            clean_phase1_stat = top_stat[:n_phase1][keep_mask]
            clean_phase1_bot = bottom_stat[:n_phase1][keep_mask]

            if len(clean_phase1_stat) >= 5:
                revised_cl_x = np.mean(clean_phase1_stat)
                revised_cl_b = np.mean(clean_phase1_bot)
                revised_sigma_top = np.std(clean_phase1_stat, ddof=1) if len(clean_phase1_stat) > 1 else trial_sigma_top
                revised_sigma_bot = np.std(clean_phase1_bot, ddof=1) if len(clean_phase1_bot) > 1 else trial_sigma_bot
                revised_ucl_x = revised_cl_x + z_trial * revised_sigma_top
                revised_lcl_x = revised_cl_x - z_trial * revised_sigma_top
                revised_ucl_b = revised_cl_b + z_trial * revised_sigma_bot
                revised_lcl_b = max(0, revised_cl_b - z_trial * revised_sigma_bot)

                st.success(f"✅ **Revised limits** (removed {len(ooc_phase1_idx)} OOC subgroups, "
                           f"{len(clean_phase1_stat)} remaining in Phase 1)")
            else:
                st.error("Too few subgroups remain in Phase 1 after removal. "
                         "Increase Phase 1 size or review the trial limits.")

        # Override chart limits with Phase 1 trial or revised limits
        if st.session_state[recalc_key] and revised_ucl_x is not None:
            ucl_x, lcl_x, cl_x = revised_ucl_x, revised_lcl_x, revised_cl_x
            ucl_bottom, lcl_bottom, center_bottom = revised_ucl_b, revised_lcl_b, revised_cl_b
        else:
            ucl_x, lcl_x, cl_x = trial_ucl_x, trial_lcl_x, trial_cl_x
            ucl_bottom, lcl_bottom, center_bottom = trial_ucl_b, trial_lcl_b, trial_cl_b

    # ── PLOTS ──
    st.subheader("Control Charts", divider="orange")

    labels = [f"{i+1}" for i in range(n_points)]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[top_label, bottom_label],
        vertical_spacing=0.12,
        shared_xaxes=not is_imr,
    )

    # Top chart
    fig.add_trace(
        go.Scatter(x=labels, y=top_stat, mode="lines+markers",
                   marker=dict(color="#4C78A8", size=7), line=dict(color="#4C78A8", width=1.5),
                   name=top_label),
        row=1, col=1,
    )
    fig.add_hline(y=cl_x, line_dash="solid", line_color="#54A24B",
                  annotation_text=f"CL = {cl_x:.3f}", row=1, col=1)
    fig.add_hline(y=ucl_x, line_dash="dash", line_color="#E45756",
                  annotation_text=f"UCL = {ucl_x:.3f}", row=1, col=1)
    fig.add_hline(y=lcl_x, line_dash="dash", line_color="#E45756",
                  annotation_text=f"LCL = {lcl_x:.3f}", row=1, col=1)
    # Zone lines
    k_mult = 1 if z_mult >= 1 else 0
    for z in range(1, z_mult):
        color = "#B279A2"
        opacity = 0.4 if z == z_mult - 1 else 0.2
        fig.add_hline(y=cl_x + z * sigma_top, line_dash="dot", line_color=color, opacity=opacity, row=1, col=1)
        fig.add_hline(y=cl_x - z * sigma_top, line_dash="dot", line_color=color, opacity=opacity, row=1, col=1)

    for i, v in enumerate(top_stat):
        if v > ucl_x or v < lcl_x:
            fig.add_trace(
                go.Scatter(x=[labels[i]], y=[v], mode="markers",
                           marker=dict(color="#E45756", size=12, symbol="x"), name=f"OOC ({i+1})"),
                row=1, col=1,
            )

    # Bottom chart
    fig.add_trace(
        go.Scatter(x=labels, y=bottom_stat, mode="lines+markers",
                   marker=dict(color="#54A24B", size=7), line=dict(color="#54A24B", width=1.5),
                   name=bottom_label),
        row=2, col=1,
    )
    fig.add_hline(y=center_bottom, line_dash="solid", line_color="#4C78A8",
                  annotation_text=f"CL = {center_bottom:.3f}", row=2, col=1)
    fig.add_hline(y=ucl_bottom, line_dash="dash", line_color="#E45756",
                  annotation_text=f"UCL = {ucl_bottom:.3f}", row=2, col=1)
    if lcl_bottom > 0:
        fig.add_hline(y=lcl_bottom, line_dash="dash", line_color="#E45756",
                      annotation_text=f"LCL = {lcl_bottom:.3f}", row=2, col=1)

    for i, v in enumerate(bottom_stat):
        if np.isnan(v):
            continue
        if v > ucl_bottom or (lcl_bottom > 0 and v < lcl_bottom):
            fig.add_trace(
                go.Scatter(x=[labels[i]], y=[v], mode="markers",
                           marker=dict(color="#E45756", size=12, symbol="x"), name=f"OOC Bot ({i+1})"),
                row=2, col=1,
            )

    # Shift reference line
    if data_src != "Upload CSV" and shift_type != "None":
        sp = shift_point if not is_imr else shift_point
        fig.add_vline(x=sp - 0.5, line_dash="dot", line_color="white", opacity=0.5,
                      annotation_text="Shift introduced", row=1, col=1)
        fig.add_vline(x=sp - 0.5, line_dash="dot", line_color="white", opacity=0.3, row=2, col=1)

    # Phase 1/2 boundary
    if trial_limits:
        p1_pos = phase1_end - 0.5
        fig.add_vline(x=p1_pos, line_dash="dot", line_color="#F58518", opacity=0.8, line_width=2,
                      annotation_text="Phase 1 End", row=1, col=1)
        fig.add_vline(x=p1_pos, line_dash="dot", line_color="#F58518", opacity=0.5, line_width=2, row=2, col=1)
        # Shade Phase 1 region
        fig.add_vrect(x0=-0.5, x1=p1_pos, fillcolor="rgba(76, 120, 168, 0.06)", line_width=0, row=1, col=1)
        fig.add_vrect(x0=-0.5, x1=p1_pos, fillcolor="rgba(76, 120, 168, 0.06)", line_width=0, row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=600, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(title_text="Subgroup" if not is_imr else "Observation", row=2, col=1)
    fig.update_yaxes(title_text=top_label, row=1, col=1)
    fig.update_yaxes(title_text=bottom_label, row=2, col=1)

    config = {
        "toImageButtonOptions": {"format": "png", "filename": "spc_charts", "height": 600, "width": 1000},
        "displaylogo": False, "scrollZoom": True,
    }
    st.plotly_chart(fig, use_container_width=True, config=config, key="spc_main_chart")

    # ── SUMMARY ──
    st.subheader("Process Summary", divider="gray")
    if is_imr:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean (X̄̄)", f"{grand_mean:.4f}")
        c2.metric("Mean Moving Range (mR̄)", f"{mr_bar:.4f}")
        c3.metric("Observations", str(k))
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Grand Mean (X̄̄)", f"{grand_mean:.4f}")
        c2.metric("σ (within)", f"{sigma_within:.4f}")
        c3.metric(f"{'S̄' if use_s_chart else 'R̄'}", f"{center_bottom:.4f}")
        c4.metric("Subgroups", str(n_subgroups))
        c5.metric("Subgroup Size", str(n_subgroup if not is_imr else 1))

    # ── CAPABILITY ──
    if use_spec and lsl is not None and usl is not None and usl > lsl:
        st.subheader("Process Capability", divider="orange")
        sigma_est = sigma_within if not is_imr else sigma_within
        overall_sigma = np.std(all_data, ddof=1)

        cp = (usl - lsl) / (6 * sigma_est)
        cpk = min((usl - grand_mean) / (3 * sigma_est), (grand_mean - lsl) / (3 * sigma_est))
        pp = (usl - lsl) / (6 * overall_sigma)
        ppk = min((usl - grand_mean) / (3 * overall_sigma), (grand_mean - lsl) / (3 * overall_sigma))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cp", f"{cp:.4f}")
        c2.metric("Cpk", f"{cpk:.4f}")
        c3.metric("Pp", f"{pp:.4f}")
        c4.metric("Ppk", f"{ppk:.4f}")

        def _cap_rating(val):
            if val >= 2.0:
                return "World-class"
            elif val >= 1.67:
                return "Excellent"
            elif val >= 1.33:
                return "Good"
            elif val >= 1.0:
                return "Adequate"
            elif val >= 0.67:
                return "Poor"
            return "Unacceptable"

        st.info(
            f"**Cp = {cp:.4f}** — {_cap_rating(cp)}. "
            f"**Cpk = {cpk:.4f}** — {_cap_rating(cpk)}. "
            + ("Process is centered within specs." if abs(cp - cpk) < 0.1
               else f"Process needs centering adjustment (Cp − Cpk = {cp - cpk:.4f}).")
        )

        fig_cap = go.Figure()
        fig_cap.add_trace(go.Histogram(
            x=all_data, nbinsx=min(40, len(all_data) // 5),
            marker_color="#4C78A8", opacity=0.7, histnorm="probability density",
            name="Process Data",
        ))
        x_fit = np.linspace(min(all_data) - sigma_est, max(all_data) + sigma_est, 300)
        y_fit = sp_stats.norm.pdf(x_fit, grand_mean, sigma_est)
        fig_cap.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines",
            line=dict(color="#54A24B", width=2.5), name="Normal Fit (within)",
        ))
        fig_cap.add_vline(x=lsl, line_dash="dash", line_color="#E45756",
                          annotation_text=f"LSL = {lsl}")
        fig_cap.add_vline(x=usl, line_dash="dash", line_color="#E45756",
                          annotation_text=f"USL = {usl}")
        fig_cap.add_vline(x=grand_mean, line_dash="solid", line_color="#B279A2",
                          annotation_text=f"Mean = {grand_mean:.2f}")
        fig_cap.update_layout(
            template="plotly_dark", height=350,
            title="Process Capability Histogram",
            xaxis_title="Measurement", yaxis_title="Density",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        config_cap = {
            "toImageButtonOptions": {"format": "png", "filename": "spc_capability", "height": 350, "width": 800},
            "displaylogo": False,
        }
        st.plotly_chart(fig_cap, use_container_width=True, config=config_cap, key="spc_cap_chart")

        pct_below = sp_stats.norm.cdf(lsl, grand_mean, sigma_est) * 100
        pct_above = (1 - sp_stats.norm.cdf(usl, grand_mean, sigma_est)) * 100

        # Cpm — Taguchi capability (penalizes deviation from target)
        target_cpm = (lsl + usl) / 2
        cpm = (usl - lsl) / (6 * np.sqrt(sigma_est**2 + (grand_mean - target_cpm)**2))

        # Confidence intervals for Cp (using chi-square approximation)
        n_cap = len(all_data)
        alpha_ci = 0.05
        cp_lower = cp * np.sqrt(sp_stats.chi2.ppf(alpha_ci / 2, n_cap - 1) / (n_cap - 1))
        cp_upper = cp * np.sqrt(sp_stats.chi2.ppf(1 - alpha_ci / 2, n_cap - 1) / (n_cap - 1))

        # DPMO / Sigma Level conversion
        total_defect_prob = pct_below + pct_above
        dpmo = total_defect_prob * 10000
        from scipy.stats import norm as norm_sigma
        sigma_level_short = norm_sigma.ppf(1 - total_defect_prob / 2 / 100) + 1.5 if total_defect_prob > 0 else 7.5
        sigma_level_long = norm_sigma.ppf(1 - total_defect_prob / 2 / 100) if total_defect_prob > 0 else 6.0
        cap_df = pd.DataFrame({
            "Metric": ["Cp", "Cpk", "Pp", "Ppk", "Cpm (Taguchi)",
                       "Cp 90% CI", "DPMO", "Sigma Level (ST)", "Sigma Level (LT)",
                       "% below LSL", "% above USL", "Total % out-of-spec"],
            "Value": [
                f"{cp:.4f}", f"{cpk:.4f}", f"{pp:.4f}", f"{ppk:.4f}", f"{cpm:.4f}",
                f"[{cp_lower:.4f}, {cp_upper:.4f}]",
                f"{dpmo:.0f}",
                f"{sigma_level_short:.2f}",
                f"{sigma_level_long:.2f}",
                f"{pct_below:.2f}%", f"{pct_above:.2f}%", f"{pct_below + pct_above:.2f}%",
            ],
        })
        _apa_table(cap_df, title="Capability Indices")

        # Non-normal capability (Box-Cox transform)
        with st.expander("Non-Normal Capability (Box-Cox Transform)", expanded=False):
            st.markdown("If your data is **not normally distributed**, the standard capability "
                        "indices may be misleading. Box-Cox transformation can help.")
            from scipy.stats import boxcox
            try:
                bc_data, bc_lambda = boxcox(all_data - min(all_data) + 0.001)
                bc_mean = np.mean(bc_data)
                bc_sigma = np.std(bc_data, ddof=1)
                # Transform spec limits
                bc_lsl = boxcox(lsl - min(all_data) + 0.001, bc_lambda) if lsl > min(all_data) else None
                bc_usl = boxcox(usl - min(all_data) + 0.001, bc_lambda) if usl > min(all_data) else None
                if bc_lsl is not None and bc_usl is not None:
                    bc_cp = (bc_usl - bc_lsl) / (6 * bc_sigma)
                    bc_cpk = min((bc_usl - bc_mean) / (3 * bc_sigma), (bc_mean - bc_lsl) / (3 * bc_sigma))
                    st.metric("Box-Cox λ", f"{bc_lambda:.4f}")
                    c1, c2 = st.columns(2)
                    c1.metric("Transformed Cp", f"{bc_cp:.4f}")
                    c2.metric("Transformed Cpk", f"{bc_cpk:.4f}")
                else:
                    st.info("Box-Cox transformation could not be applied (limits outside data range).")
            except Exception:
                st.info("Box-Cox transformation not applicable to this data.")

    # ── ADDITIONAL DIAGNOSTICS ──
    if not is_imr and not use_phase12:
        with st.expander("Run Chart (Before Control Limits)", expanded=False):
            st.markdown("A **run chart** plots the data in time order before establishing "
                        "control limits. It helps identify trends, shifts, and unusual patterns "
                        "that would invalidate the control limits if present during Phase 1.")
            run_fig = go.Figure()
            run_fig.add_trace(go.Scatter(
                x=labels, y=top_stat, mode="lines+markers",
                marker=dict(color="#4C78A8", size=6), line=dict(color="#4C78A8", width=1.5),
                name="Data",
            ))
            run_fig.add_hline(y=np.median(top_stat), line_dash="dash", line_color="#54A24B",
                              annotation_text=f"Median = {np.median(top_stat):.3f}")
            run_fig.update_layout(template="plotly_dark", height=300,
                                   xaxis_title="Subgroup", yaxis_title=top_label,
                                   margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(run_fig, use_container_width=True, key="spc_run_chart")

    with st.expander("Autocorrelation Check", expanded=False):
        st.markdown("**Autocorrelation** measures how correlated each point is with its "
                    "predecessors. SPC charts assume independent observations — high "
                    "autocorrelation inflates false alarms on I-MR charts.")
        lag_max = min(20, len(top_stat) // 4)
        acf = np.array([1] + [np.corrcoef(top_stat[:-i], top_stat[i:])[0, 1]
                               for i in range(1, lag_max + 1)])
        acf_fig = go.Figure()
        acf_fig.add_trace(go.Bar(
            x=list(range(lag_max + 1)), y=acf,
            marker_color="#4C78A8", name="ACF",
        ))
        # Significance bounds
        bound = 1.96 / np.sqrt(len(top_stat))
        acf_fig.add_hline(y=bound, line_dash="dash", line_color="#E45756",
                           annotation_text=f"±{bound:.3f}")
        acf_fig.add_hline(y=-bound, line_dash="dash", line_color="#E45756")
        acf_fig.update_layout(template="plotly_dark", height=300,
                               xaxis_title="Lag", yaxis_title="Autocorrelation",
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(acf_fig, use_container_width=True, key="spc_acf_chart")
        if abs(acf[1]) > bound:
            st.warning(f"⚠️ Lag-1 autocorrelation ({acf[1]:.3f}) exceeds significance bound "
                       f"(±{bound:.3f}). Consider using a CUSUM or EWMA chart instead of I-MR.")
        else:
            st.success(f"✅ Lag-1 autocorrelation ({acf[1]:.3f}) within expected bounds — "
                       "data appears independent.")

    with st.expander("Average Run Length (ARL) Reference", expanded=False):
        st.markdown("**ARL₀** (in-control) and **ARL₁** (out-of-control) for this chart configuration:")
        shift_sizes = np.arange(0, 3.5, 0.5)
        z_val = sigma_limits
        arl_data = []
        for shift in shift_sizes:
            if shift == 0:
                arl0 = 1 / (2 * (1 - sp_stats.norm.cdf(z_val)))
                arl_data.append(["0 (in control)", f"{arl0:.0f}", "—", "—"])
            else:
                arl_shewhart = 1 / (1 - sp_stats.norm.cdf(z_val - shift) + sp_stats.norm.cdf(-z_val - shift))
                # ARL for CUSUM with K=0.5, H=5 (approximate)
                arl_cusum = None
                arl_ewma = None
                arl_data.append([f"{shift:.1f}σ", f"{arl_shewhart:.0f}", "—", "—"])
        arl_df = pd.DataFrame(arl_data, columns=["Shift", "Shewhart (3σ) ARL", "CUSUM (K=0.5, H=5)", "EWMA (λ=0.2)"])
        st.dataframe(arl_df, use_container_width=True)
        st.caption("CUSUM and EWMA ARL values shown where available. "
                    "CUSUM detects 1σ shifts ~4× faster than X̄ charts.")

    # ── SHEWHART RULES ──
    if show_rules:
        st.subheader("Shewhart Rule Violations", divider="gray")
        if all_rules:
            for rule in all_rules:
                st.markdown(rule)
        else:
            st.success("✅ No Shewhart rule violations detected in the current process data.")

    # ── DATA TABLE ──
    with st.expander("View Raw Data", expanded=False):
        if is_imr:
            data_df = pd.DataFrame({
                "Observation": range(1, k + 1),
                "Value": individuals,
            })
            data_df["Moving Range"] = np.concatenate([[np.nan], moving_ranges])
            st.dataframe(data_df, use_container_width=True)
        elif use_s_chart:
            data_df = pd.DataFrame(
                base_data,
                index=[f"Subgroup {i+1}" for i in range(n_subgroups)],
                columns=[f"Obs {j+1}" for j in range(n_subgroup)],
            )
            data_df["Mean"] = subgroup_means
            data_df["Std"] = subgroup_stds
            st.dataframe(data_df, use_container_width=True)
        else:
            data_df = pd.DataFrame(
                base_data,
                index=[f"Subgroup {i+1}" for i in range(n_subgroups)],
                columns=[f"Obs {j+1}" for j in range(n_subgroup)],
            )
            data_df["Mean"] = subgroup_means
            data_df["Range"] = subgroup_ranges if not use_s_chart else subgroup_stds
            st.dataframe(data_df, use_container_width=True)

    # ── EDUCATIONAL CONTENT ──
    st.subheader("Interpreting Control Charts", divider="orange")
    with st.expander("How to Read Control Charts", expanded=True):
        st.markdown("""
        | Element | What It Tells You |
        |---------|------------------|
        | **Center Line (CL)** | The average (of subgroup means, ranges, individuals, etc.). Serves as the process baseline. |
        | **Upper Control Limit (UCL)** | +kσ from center. A point above indicates a likely special cause. |
        | **Lower Control Limit (LCL)** | -kσ from center. A point below indicates a likely special cause. |
        | **Zone lines** | 1σ and 2σ lines help detect subtle patterns (Shewhart rules 2-4). |

        **The eight Western Electric rules:**
        1. **1 point beyond 3σ** — immediate signal of a special cause
        2. **2 of 3 beyond 2σ** — early warning of a shift
        3. **4 of 5 beyond 1σ** — shift or trend developing
        4. **8 consecutive on same side** — sustained shift in process level
        5. **6 consecutive trending** — drift or tool wear
        6. **14 alternating** — over-control or oscillation
        7. **15 within 1σ** — reduced variability (possible data tampering)
        8. **8 beyond 1σ** — stratification or mixture of distributions
        """)

    with st.expander("Understanding Capability Indices", expanded=False):
        st.markdown("""
        | Index | Formula | What It Measures |
        |-------|---------|-----------------|
        | **Cp** | (USL − LSL) / (6σ) | **Potential** capability — compares spec width to process spread, ignoring centering |
        | **Cpk** | min(USL − μ, μ − LSL) / (3σ) | **Actual** capability — penalizes off-center processes |
        | **Pp** | (USL − LSL) / (6σₒᵥₑᵣₐₗₗ) | **Overall** performance — uses total variation including between-subgroup variation |
        | **Ppk** | min(USL − μ, μ − LSL) / (3σₒᵥₑᵣₐₗₗ) | **Overall** performance with centering adjustment |

        **Industry standards:**
        - **Cp/Cpk ≥ 1.33** → Capable (4σ quality)
        - **Cp/Cpk ≥ 1.67** → Good (5σ quality)
        - **Cp/Cpk ≥ 2.0** → Excellent (6σ quality)
        - **Cp/Cpk < 1.0** → Process is NOT capable — too much variation relative to specifications
        """)

    with st.expander("Common SPC Mistakes", expanded=False):
        st.markdown("""
        - **Ignoring the R/S chart**: The X̄ chart depends on the bottom chart for its control limits. If R/S is out of control,
        X̄ limits are unreliable.
        - **Using wrong chart type**: Use X̄-R for n≤8, X̄-S for n≥9, I-MR for individuals data (e.g., lab samples).
        - **Changing limits too often**: Control limits should be based on a stable baseline period.
        - **Confusing specification limits with control limits**: Spec limits (USL/LSL) define customer requirements.
        Control limits (UCL/LCL) define process behavior. They are completely different concepts.
        - **Over-analyzing patterns**: Not every pattern signals a special cause. The Western Electric rules have an expected
        false alarm rate.
        """)
