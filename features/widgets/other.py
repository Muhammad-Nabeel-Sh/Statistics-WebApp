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


@register_test("Runs Test for Randomness")
def render_runs_test_for_randomness(external_data=None):

    st.subheader("Interactive Runs Test for Randomness")

    st.info("""
    **Runs Test for Randomness** tests whether a sequence of values is **random**.
    A "run" is a consecutive sequence of values above (or below) the median.
    - **Too few runs** → clustered pattern (positive autocorrelation)
    - **Too many runs** → alternating pattern (negative autocorrelation)
    - Used in time-series analysis, quality control, and residual diagnostics.
    """)

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("runs_test", mode="one_sample")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        values_rt = src["data"]["values"]
    else:
        pattern = st.selectbox("Sequence Pattern", ["Random", "Trend", "Cyclical", "Clustered"], key="rt_pattern")
        n_rt = st.slider("Sample Size", 10, 300, 100, key="rt_n")

        np.random.seed(42)
        if pattern == "Random":
            values_rt = np.random.normal(0, 1, n_rt)
        elif pattern == "Trend":
            values_rt = np.linspace(-2, 2, n_rt) + np.random.normal(0, 0.3, n_rt)
        elif pattern == "Cyclical":
            x = np.linspace(0, 4 * np.pi, n_rt)
            values_rt = np.sin(x) + np.random.normal(0, 0.2, n_rt)
        else:  # Clustered
            block1 = np.random.normal(-1, 0.3, n_rt // 3)
            block2 = np.random.normal(1, 0.3, n_rt // 3)
            block3 = np.random.normal(-1, 0.3, n_rt - 2 * (n_rt // 3))
            values_rt = np.concatenate([block1, block2, block3])

    # =========================
    # STATISTICAL COMPUTATION
    # =========================

    from scipy.stats import norm as norm_rt

    median_rt = np.median(values_rt)
    binary_rt = (values_rt > median_rt).astype(int)

    n1_rt = np.sum(binary_rt == 1)
    n2_rt = np.sum(binary_rt == 0)

    runs_rt = 1 + np.sum(binary_rt[1:] != binary_rt[:-1])

    n_runs = n1_rt + n2_rt
    expected_runs = 1 + (2 * n1_rt * n2_rt) / n_runs
    var_runs = (2 * n1_rt * n2_rt * (2 * n1_rt * n2_rt - n_runs)) / (n_runs ** 2 * (n_runs - 1))
    z_rt = (runs_rt - expected_runs) / np.sqrt(var_runs)
    p_rt = 2 * (1 - norm_rt.cdf(abs(z_rt)))

    # =========================
    # DATA SUMMARY
    # =========================

    st.divider()
    st.subheader("Data Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N", f"{n_runs}")
    col2.metric("Runs (observed)", f"{runs_rt}")
    col3.metric("Runs (expected)", f"{expected_runs:.2f}")
    col4.metric("Median", f"{median_rt:.3f}")

    # =========================
    # STATS
    # =========================

    st.latex(rf"z = {z_rt:.3f}")
    st.latex(rf"\text{{{format_p_value(p_rt)}}}")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()

    colors = ["#1f77b4" if b == 1 else "#ff7f0e" for b in binary_rt]
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(values_rt)),
            y=values_rt,
            mode="lines+markers",
            marker=dict(color=colors, size=6),
            line=dict(color="gray", width=1),
            name="Sequence",
        )
    )
    fig.add_hline(
        y=median_rt,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median = {median_rt:.2f}",
    )

    # Highlight runs
    run_starts = [0] + list(np.where(binary_rt[1:] != binary_rt[:-1])[0] + 1)
    for rs in run_starts:
        fig.add_vline(x=rs - 0.5, line_dash="dot", line_color="green", line_width=1, opacity=0.5)

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Position in Sequence",
        yaxis_title="Value",
        title=f"Runs Test: {runs_rt} runs observed vs {expected_runs:.1f} expected",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Results")
    results_data = {
        "Metric": ["Median", "Runs (observed)", "Runs (expected)", "z", "p-value", "n₁ (above)", "n₂ (below/equal)"],
        "Value": [
            f"{median_rt:.3f}",
            f"{runs_rt}",
            f"{expected_runs:.2f}",
            f"{z_rt:.3f}",
            format_p_value(p_rt),
            f"{n1_rt}",
            f"{n2_rt}",
        ],
    }
    st.table(pd.DataFrame(results_data))

