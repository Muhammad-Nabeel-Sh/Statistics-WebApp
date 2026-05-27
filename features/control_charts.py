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


def render_control_charts():
    st.title("Statistical Process Control (SPC) Charts")
    st.markdown("""
    Interactive control charts for monitoring process stability and capability.
    Adjust parameters to see how control charts detect out-of-control conditions
    and how capability indices quantify process performance.
    """)

    # ── SIDEBAR CONTROLS ──
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
            sigma_limits = st.slider("Control Limit Width (σ)", 1, 4, 3, 1, key="spc_sigma")
            show_rules = st.checkbox("Show Shewhart Rules", True, key="spc_rules")

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

    # ── PLOTS ──
    st.subheader("Control Charts", divider="orange")

    n_points = len(top_stat)
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
        cap_df = pd.DataFrame({
            "Metric": ["Cp", "Cpk", "Pp", "Ppk", "% below LSL", "% above USL", "Total % out-of-spec"],
            "Value": [
                f"{cp:.4f}", f"{cpk:.4f}", f"{pp:.4f}", f"{ppk:.4f}",
                f"{pct_below:.2f}%", f"{pct_above:.2f}%", f"{pct_below + pct_above:.2f}%",
            ],
        })
        _apa_table(cap_df, title="Capability Indices")

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
