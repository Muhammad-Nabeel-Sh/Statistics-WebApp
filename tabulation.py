import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import math

_rng = np.random.default_rng(42)


def _tab_buttons(options, key, default=None):
    """Render navigation buttons in 2 columns, return selected option."""
    state_key = f"_tab_sel_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default or options[0]
    n = len(options)
    cols = st.columns(2)
    for i, opt in enumerate(options):
        with cols[i % 2]:
            if st.button(
                opt,
                key=f"_tab_btn_{key}_{i}",
                use_container_width=True,
                type="primary" if st.session_state[state_key] == opt else "secondary",
            ):
                st.session_state[state_key] = opt
                st.rerun()
    return st.session_state[state_key]


def _apa_table(df, title="Table"):
    """Render a DataFrame as an APA-style formatted table."""
    st.markdown(f"**{title}**")
    styled = df.style.set_table_attributes(
        'style="border-collapse: collapse; width: 100%;"'
    )
    styled = styled.set_properties(
        **{
            "border": "1px solid #555",
            "padding": "6px",
            "text-align": "center",
            "font-size": "14px",
        }
    )
    styled = styled.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1e1e1e"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("border", "1px solid #555"),
                    ("padding", "8px"),
                    ("text-align", "center"),
                ],
            }
        ]
    )
    st.dataframe(styled, use_container_width=True)
    return styled


def _heatmap_fig(
    data, x_labels, y_labels, title="", colorscale="Viridis", text_template="%{text}"
):
    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            text=np.round(data, 2) if isinstance(data, np.ndarray) else data,
            texttemplate=text_template,
            colorscale=colorscale,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
    )
    return fig


# =====================
# 1. DESCRIPTIVE TABULATION
# =====================


def descriptive_tabulation():
    st.header("Descriptive Tabulation")
    st.info("""
    Descriptive tables summarize the distribution and central tendency of your data.
    Use the controls to generate frequency, relative frequency, cumulative frequency,
    and descriptive statistics tables.
    """)

    tab = _tab_buttons(
        [
            "Frequency Tables",
            "Relative Frequency Tables",
            "Cumulative Frequency Tables",
            "Descriptive Statistics Tables",
            "Grouped Summary Tables",
            "Pivot Tables",
            "Distribution Summary Tables",
        ],
        "desc_tab",
    )

    if tab == "Frequency Tables":
        _freq_table_widget()
    elif tab == "Relative Frequency Tables":
        _rel_freq_table_widget()
    elif tab == "Cumulative Frequency Tables":
        _cum_freq_table_widget()
    elif tab == "Descriptive Statistics Tables":
        _desc_stats_table_widget()
    elif tab == "Grouped Summary Tables":
        _grouped_summary_widget()
    elif tab == "Pivot Tables":
        _pivot_table_widget()
    elif tab == "Distribution Summary Tables":
        _dist_summary_widget()


def _generate_sample_data(n, dist="Normal", seed=42):
    np.random.seed(seed)
    if dist == "Normal":
        return np.random.normal(50, 15, n)
    elif dist == "Skewed":
        return np.random.gamma(2, 15, n) + 10
    elif dist == "Uniform":
        return np.random.uniform(10, 90, n)
    elif dist == "Bimodal":
        return np.concatenate(
            [np.random.normal(30, 8, n // 2), np.random.normal(70, 8, n - n // 2)]
        )
    return np.random.normal(50, 15, n)


def _freq_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="ft_n")
        bins = st.slider("Number of Bins", 3, 20, 8, key="ft_bins")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Uniform", "Bimodal"], key="ft_dist"
        )
    data = _generate_sample_data(n, dist)
    counts, edges = np.histogram(data, bins=bins)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    df = pd.DataFrame(
        {
            "Class Interval": labels,
            "Frequency": counts,
            "Cumulative Freq": np.cumsum(counts),
        }
    )
    with c2:
        _apa_table(df, "Frequency Distribution Table")
        fig = go.Figure(data=[go.Bar(x=labels, y=counts, marker_color="#4C78A8")])
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=80),
            xaxis_title="Class",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig, use_container_width=True)
    with st.expander(" Educational Notes - Frequency Tables", expanded=True):
        st.info("""
        **Absolute Frequency** = raw count of observations in each category/interval.
        - Sum of all frequencies = total sample size (n)
        - Frequency distributions reveal shape, central tendency, and spread
        - Choose bin width carefully: too few bins hide patterns, too many create noise
        """)


def _rel_freq_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="rf_n")
        bins = st.slider("Number of Bins", 3, 20, 8, key="rf_bins")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Uniform", "Bimodal"], key="rf_dist"
        )
    data = _generate_sample_data(n, dist)
    counts, edges = np.histogram(data, bins=bins)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    rel_freq = counts / n
    df = pd.DataFrame(
        {
            "Class Interval": labels,
            "Frequency": counts,
            "Relative Freq": [f"{v:.4f}" for v in rel_freq],
            "Percentage": [f"{v*100:.2f}%" for v in rel_freq],
        }
    )
    with c2:
        _apa_table(df, "Relative Frequency Distribution Table")
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=rel_freq * 100,
                    marker_color="#E45756",
                    hovertemplate="%{x}<br>Percentage: %{y:.1f}%<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=80),
            xaxis_title="Class",
            yaxis_title="Percentage (%)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with st.expander(" Educational Notes - Relative Frequency", expanded=True):
        st.info("""
        **Relative Frequency** = frequency ÷ total n
        - Always sums to 1.0 (or 100%)
        - Allows comparison across different sample sizes
        - Directly estimates the probability P(X in interval)
        - Percentage = relative frequency × 100
        """)


def _cum_freq_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="cf_n")
        bins = st.slider("Number of Bins", 3, 20, 8, key="cf_bins")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Uniform", "Bimodal"], key="cf_dist"
        )
    data = _generate_sample_data(n, dist)
    counts, edges = np.histogram(data, bins=bins)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    cum_freq = np.cumsum(counts)
    rel_cum = cum_freq / n
    df = pd.DataFrame(
        {
            "Class Interval": labels,
            "Frequency": counts,
            "Cumulative Freq": cum_freq,
            "Cumulative %": [f"{v*100:.2f}%" for v in rel_cum],
        }
    )
    with c2:
        _apa_table(df, "Cumulative Frequency Distribution Table")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=cum_freq,
                mode="lines+markers",
                name="Cumulative Freq",
                marker_color="#4C78A8",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=rel_cum * 100,
                mode="lines+markers",
                name="Cumulative %",
                marker_color="#E45756",
                yaxis="y2",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=80),
            xaxis_title="Class",
            yaxis_title="Cumulative Frequency",
            yaxis2=dict(overlaying="y", side="right", title="Cumulative %"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    with st.expander(" Educational Notes - Cumulative Frequency", expanded=True):
        st.info("""
        **Cumulative Frequency** = running total of frequencies
        - Always ends at total sample size (n) or 100%
        - Useful for percentiles: the pth percentile is the value below which p% fall
        - The median is at the 50th percentile (50% cumulative frequency)
        - The ogive (cumulative frequency curve) helps estimate percentiles visually
        """)


def _desc_stats_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Sample Size", 10, 1000, 200, key="ds_n")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Uniform", "Bimodal"], key="ds_dist"
        )
        show_outliers = st.toggle("Add Outliers", False, key="ds_out")
    data = _generate_sample_data(n, dist)
    if show_outliers:
        data = np.append(data, [data.mean() + 5 * data.std()] * 3)
    mean_v = np.mean(data)
    median_v = np.median(data)
    std_v = np.std(data, ddof=1)
    se_v = std_v / np.sqrt(len(data))
    skew_v = stats.skew(data)
    kurt_v = stats.kurtosis(data, fisher=True)
    q1, q3 = np.percentile(data, [25, 75])
    ci = se_v * stats.t.ppf(0.975, len(data) - 1)
    df = pd.DataFrame(
        {
            "Statistic": [
                "n",
                "Mean",
                "Median",
                "Std Dev",
                "SE",
                "Skewness",
                "Kurtosis",
                "Q1 (25th)",
                "Q3 (75th)",
                "IQR",
                "Min",
                "Max",
                "95% CI Lower",
                "95% CI Upper",
            ],
            "Value": [
                f"{len(data)}",
                f"{mean_v:.3f}",
                f"{median_v:.3f}",
                f"{std_v:.3f}",
                f"{se_v:.3f}",
                f"{skew_v:.3f}",
                f"{kurt_v:.3f}",
                f"{q1:.3f}",
                f"{q3:.3f}",
                f"{q3 - q1:.3f}",
                f"{data.min():.3f}",
                f"{data.max():.3f}",
                f"{mean_v - ci:.3f}",
                f"{mean_v + ci:.3f}",
            ],
        }
    )
    with c2:
        _apa_table(df, "Descriptive Statistics")
        fig = go.Figure()
        fig.add_trace(go.Box(y=data, name="Data", boxmean="sd", marker_color="#4C78A8"))
        fig.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    with st.expander(" Educational Notes - Descriptive Statistics", expanded=True):
        st.info("""
        **Key Interpretations:**
        - **Mean vs Median**: If mean > median, data is right-skewed (positive skew).
        - **Skewness**: ~0 = symmetric, >0 = right tail, <0 = left tail
        - **Kurtosis**: ~0 = normal tails, >0 = heavy tails (more outliers), <0 = light tails
        - **SE (Standard Error)**: SD / √n — measures precision of the sample mean
        - **95% CI**: Range where the true population mean likely falls
        """)


def _grouped_summary_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Per Group", 10, 200, 50, key="gs_n")
        n_groups = st.selectbox("Groups", [2, 3, 4, 5], index=1, key="gs_ng")
        effect = st.slider("Between-Group Effect", 0.0, 5.0, 1.0, 0.1, key="gs_eff")
    np.random.seed(42)
    ng = int(n_groups)
    group_data = {}
    for i in range(ng):
        group_data[f"Group {chr(65+i)}"] = np.random.normal(i * effect, 1.5, n)
    stats_list = []
    for name, vals in group_data.items():
        stats_list.append(
            {
                "Group": name,
                "n": len(vals),
                "Mean": f"{np.mean(vals):.2f}",
                "SD": f"{np.std(vals, ddof=1):.2f}",
                "SE": f"{np.std(vals, ddof=1)/np.sqrt(len(vals)):.2f}",
                "Median": f"{np.median(vals):.2f}",
                "IQR": f"{np.percentile(vals,75)-np.percentile(vals,25):.2f}",
            }
        )
    df = pd.DataFrame(stats_list)
    with c2:
        _apa_table(df, "Grouped Summary Statistics")
        fig = go.Figure()
        for i, (name, vals) in enumerate(group_data.items()):
            fig.add_trace(
                go.Box(y=vals, name=name, marker_color=px.colors.qualitative.Plotly[i])
            )
        fig.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)


def _pivot_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="pt_n")
        row_cats = st.selectbox("Row Categories", [2, 3, 4], index=1, key="pt_rows")
        col_cats = st.selectbox("Column Categories", [2, 3, 4], index=1, key="pt_cols")
        agg_func = st.selectbox(
            "Aggregation", ["Count", "Sum", "Mean", "Row %", "Col %"], key="pt_agg"
        )
    np.random.seed(42)
    rows, cols = int(row_cats), int(col_cats)
    data = np.random.randint(5, 50, (rows, cols))
    row_names = [f"Row {chr(65+i)}" for i in range(rows)]
    col_names = [f"Col {i+1}" for i in range(cols)]
    if agg_func == "Count":
        display = data
    elif agg_func == "Sum":
        display = data
    elif agg_func == "Mean":
        display = data / n * 100
    elif agg_func == "Row %":
        display = (data / data.sum(axis=1, keepdims=True) * 100).round(1)
    elif agg_func == "Col %":
        display = (data / data.sum(axis=0, keepdims=True) * 100).round(1)
    df = pd.DataFrame(display, index=row_names, columns=col_names)
    df["Total"] = df.sum(axis=1)
    if agg_func in ["Row %", "Col %"]:
        df["Total"] = 100.0
    with c2:
        _apa_table(df, f"Pivot Table ({agg_func})")
        fig = _heatmap_fig(display, col_names, row_names, colorscale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    with st.expander(" Educational Notes - Pivot Tables", expanded=True):
        st.info("""
        **Pivot tables** cross-classify data by two or more variables.
        - **Count**: raw frequencies in each combination
        - **Row %**: percentages within each row (sums to 100% across columns)
        - **Col %**: percentages within each column (sums to 100% down rows)
        - Useful for identifying patterns and associations between categorical variables
        """)


def _dist_summary_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Sample Size", 20, 1000, 200, key="dss_n")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Uniform", "Bimodal"], key="dss_dist"
        )
    data = _generate_sample_data(n, dist)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    p_vals = np.percentile(data, percentiles)
    df = pd.DataFrame(
        {
            "Percentile": [f"{p}th" for p in percentiles],
            "Value": [f"{v:.2f}" for v in p_vals],
        }
    )
    with c2:
        _apa_table(df, "Distribution Summary (Percentiles)")
        fig = go.Figure()
        fig.add_trace(go.Box(y=data, name="Data", boxmean="sd", marker_color="#4C78A8"))
        for p in [25, 50, 75]:
            fig.add_hline(
                y=np.percentile(data, p),
                line_dash="dash",
                annotation_text=f"{p}th %ile={np.percentile(data,p):.1f}",
            )
        fig.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    with st.expander(" Educational Notes - Distribution Summary", expanded=True):
        st.info("""
        **Percentiles** divide the data into 100 equal parts.
        - **1st percentile**: only 1% of data falls below this value
        - **Median (50th)**: the middle value
        - **IQR**: Q3 - Q1 (75th - 25th percentile)
        - **Interpercentile ranges** are robust to outliers
        - Normal distribution: 68-95-99.7 rule (1, 2, 3 SDs)
        """)


# =====================
# 2. CROSS-TABULATION
# =====================


def cross_tabulation():
    st.header("Cross-Tabulation")
    st.info("""
    Cross-tabulation (contingency tables) is fundamental to analyzing relationships
    between categorical variables. These tables form the basis for Chi-square tests,
    Fisher's exact test, and McNemar's test.
    """)

    tab = _tab_buttons(
        [
            "2×2 Contingency Tables",
            "RxC Contingency Tables",
            "Proportion Tables",
            "Row Percentage Tables",
            "Column Percentage Tables",
            "Expected Frequency Tables",
        ],
        "cross_tab",
    )

    if tab == "2×2 Contingency Tables":
        _twobytwo_widget()
    elif tab == "RxC Contingency Tables":
        _rxc_widget()
    elif tab == "Proportion Tables":
        _proportion_table_widget()
    elif tab == "Row Percentage Tables":
        _row_pct_widget()
    elif tab == "Column Percentage Tables":
        _col_pct_widget()
    elif tab == "Expected Frequency Tables":
        _expected_freq_widget()


def _twobytwo_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("##### Cell Counts")
        a = st.slider("Cell a (Group 1, Success)", 0, 200, 40, key="2x2_a")
        b = st.slider("Cell b (Group 1, Failure)", 0, 200, 20, key="2x2_b")
        c = st.slider("Cell c (Group 2, Success)", 0, 200, 10, key="2x2_c")
        d = st.slider("Cell d (Group 2, Failure)", 0, 200, 30, key="2x2_d")
        show_expected = st.toggle("Show Expected Frequencies", True, key="2x2_exp")
        show_heat = st.toggle("Show Heatmap", True, key="2x2_heat")
    table = np.array([[a, b], [c, d]])
    n_total = a + b + c + d
    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)
    df_display = pd.DataFrame(
        table, index=["Group 1", "Group 2"], columns=["Success", "Failure"]
    )
    df_display["Total"] = row_totals
    df_display.loc["Total"] = list(col_totals) + [n_total]
    with c2:
        _apa_table(df_display, "2×2 Contingency Table")
        if show_expected:
            _, _, _, expected = chi2_contingency(table)
            exp_df = pd.DataFrame(
                expected, index=["Group 1", "Group 2"], columns=["Success", "Failure"]
            )
            st.caption("Expected Frequencies (under independence)")
            st.dataframe(exp_df.style.format("{:.1f}"), use_container_width=True)
        if show_heat:
            fig = _heatmap_fig(
                table,
                ["Success", "Failure"],
                ["Group 1", "Group 2"],
                colorscale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)
    colm1, colm2, colm3, colm4 = st.columns(4)
    chi2_val, p_val, dof, _ = chi2_contingency(table)
    with colm1:
        st.metric("χ²", f"{chi2_val:.3f}")
    with colm2:
        st.metric("p-value", f"{p_val:.4f}")
    with colm3:
        cramer = np.sqrt(chi2_val / (n_total * 1)) if n_total > 0 else 0
        st.metric("Cramér's V", f"{cramer:.4f}")
    with colm4:
        if b > 0 and c > 0:
            or_val = (a * d) / (b * c)
            st.metric("Odds Ratio", f"{or_val:.3f}")
        else:
            st.metric("Odds Ratio", "∞")
    with st.expander(" Observed vs Expected Comparison", expanded=True):
        obs_flat = table.flatten()
        exp_flat = expected.flatten()
        cells = ["(G1,S)", "(G1,F)", "(G2,S)", "(G2,F)"]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(name="Observed", x=cells, y=obs_flat, marker_color="#4C78A8")
        )
        fig2.add_trace(
            go.Bar(name="Expected", x=cells, y=exp_flat, marker_color="#E45756")
        )
        fig2.update_layout(
            template="plotly_dark",
            height=300,
            barmode="group",
            xaxis_title="Cell",
            yaxis_title="Count",
        )
        st.plotly_chart(fig2, use_container_width=True)
        deviations = obs_flat - exp_flat
        contrib = deviations**2 / exp_flat
        comp_df = pd.DataFrame(
            {
                "Cell": cells,
                "Observed": obs_flat,
                "Expected": np.round(exp_flat, 1),
                "Deviation": np.round(deviations, 1),
                "χ² Contrib": np.round(contrib, 3),
            }
        )
        st.dataframe(comp_df, use_container_width=True)
        st.info(f"""
        **Chi-square statistic = Σ(O-E)²/E = {chi2_val:.3f}
        Large deviations contribute heavily to χ².
        Cell with largest contribution: {cells[np.argmax(contrib)]} ({contrib.max():.3f})
        """)
    with st.expander(" Educational Notes - 2×2 Tables", expanded=True):
        st.info("""
        **2×2 Contingency Tables** are the foundation of categorical data analysis.
        - **Odds Ratio** = (a×d)/(b×c) — measures association strength
        - **OR = 1**: no association | **OR > 1**: positive association | **OR < 1**: negative
        - **Cramér's V**: standardized measure of association (0 to 1)
        - **Chi-square test**: compares observed vs expected frequencies under independence
        - Large deviations between O and E → larger χ² → more likely significant
        """)


def _rxc_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        rows = st.selectbox("Number of Rows", [2, 3, 4, 5], index=1, key="rxc_r")
        cols = st.selectbox("Number of Columns", [2, 3, 4, 5], index=1, key="rxc_c")
        effect = st.slider(
            "Association Strength",
            0.0,
            3.0,
            1.0,
            0.1,
            key="rxc_eff",
            help="Higher = stronger pattern away from independence",
        )
        show_heat = st.toggle("Show Heatmap", True, key="rxc_heat")
    np.random.seed(42)
    r, c = int(rows), int(cols)
    base = np.random.randint(5, 30, (r, c))
    if effect > 0:
        for i in range(r):
            for j in range(c):
                base[i, j] = max(1, int(base[i, j] + effect * (i - j)))
    with c2:
        row_names = [f"Row {chr(65+i)}" for i in range(r)]
        col_names = [f"Col {j+1}" for j in range(c)]
        df = pd.DataFrame(base, index=row_names, columns=col_names)
        df["Total"] = df.sum(axis=1)
        df.loc["Total"] = list(df.sum(axis=0))
        _apa_table(df, f"{r}×{c} Contingency Table")
        if show_heat:
            fig = _heatmap_fig(base, col_names, row_names, colorscale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
    chi2_val, p_val, dof, expected = chi2_contingency(base)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("χ²", f"{chi2_val:.3f}")
    with col2:
        st.metric("p-value", f"{p_val:.4f}")
    with col3:
        cramer = (
            np.sqrt(chi2_val / (base.sum() * min(r - 1, c - 1)))
            if base.sum() > 0
            else 0
        )
        st.metric("Cramér's V", f"{cramer:.4f}")
    with st.expander(" Expected Frequencies", expanded=True):
        exp_df = pd.DataFrame(np.round(expected, 1), index=row_names, columns=col_names)
        st.dataframe(exp_df, use_container_width=True)


def _proportion_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        rows = st.selectbox("Number of Rows", [2, 3, 4], index=1, key="prop_r")
        cols = st.selectbox("Number of Columns", [2, 3, 4], index=1, key="prop_c")
    np.random.seed(42)
    r, c = int(rows), int(cols)
    table = np.random.randint(5, 50, (r, c))
    total = table.sum()
    props = table / total
    row_names = [f"Group {chr(65+i)}" for i in range(r)]
    col_names = [f"Level {j+1}" for j in range(c)]
    prop_df = pd.DataFrame(np.round(props, 4), index=row_names, columns=col_names)
    with c2:
        _apa_table(prop_df, "Proportion Table (cell / grand total)")
        fig = _heatmap_fig(
            props, col_names, row_names, colorscale="Blues", text_template="%{text:.3f}"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Proportions represent each cell as a fraction of the grand total. They sum to 1.0."
    )


def _row_pct_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        rows = st.selectbox("Number of Rows", [2, 3, 4], index=1, key="rpct_r")
        cols = st.selectbox("Number of Columns", [2, 3, 4], index=1, key="rpct_c")
    np.random.seed(42)
    r, c = int(rows), int(cols)
    table = np.random.randint(5, 50, (r, c))
    row_pct = (table / table.sum(axis=1, keepdims=True) * 100).round(1)
    row_names = [f"Group {chr(65+i)}" for i in range(r)]
    col_names = [f"Level {j+1}" for j in range(c)]
    df = pd.DataFrame(row_pct, index=row_names, columns=col_names)
    df["Total"] = 100.0
    with c2:
        _apa_table(df, "Row Percentage Table")
        fig = _heatmap_fig(
            row_pct,
            col_names,
            row_names,
            colorscale="Reds",
            text_template="%{text:.1f}%",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**Row Percentages**: Each row sums to 100%. Compares distribution across columns WITHIN each row group."
    )


def _col_pct_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        rows = st.selectbox("Number of Rows", [2, 3, 4], index=1, key="cpct_r")
        cols = st.selectbox("Number of Columns", [2, 3, 4], index=1, key="cpct_c")
    np.random.seed(42)
    r, c = int(rows), int(cols)
    table = np.random.randint(5, 50, (r, c))
    col_pct = (table / table.sum(axis=0, keepdims=True) * 100).round(1)
    row_names = [f"Group {chr(65+i)}" for i in range(r)]
    col_names = [f"Level {j+1}" for j in range(c)]
    df = pd.DataFrame(col_pct, index=row_names, columns=col_names)
    df.loc["Total"] = 100.0
    with c2:
        _apa_table(df, "Column Percentage Table")
        fig = _heatmap_fig(
            col_pct,
            col_names,
            row_names,
            colorscale="Greens",
            text_template="%{text:.1f}%",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**Column Percentages**: Each column sums to 100%. Compares distribution across rows WITHIN each column level."
    )


def _expected_freq_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        rows = st.selectbox("Number of Rows", [2, 3, 4], index=0, key="exp_r")
        cols = st.selectbox("Number of Columns", [2, 3, 4], index=0, key="exp_c")
        show_deviations = st.toggle("Show Deviations (O - E)", True, key="exp_dev")
    np.random.seed(42)
    r, c = int(rows), int(cols)
    observed = np.random.randint(5, 50, (r, c))
    chi2_val, p_val, dof, expected = chi2_contingency(observed)
    row_names = [f"Row {chr(65+i)}" for i in range(r)]
    col_names = [f"Col {j+1}" for j in range(c)]
    tab_view = st.radio(
        "View",
        ["Observed", "Expected", "Side-by-Side"],
        horizontal=True,
        key="exp_view",
    )
    with c2:
        if tab_view == "Observed":
            df = pd.DataFrame(observed, index=row_names, columns=col_names)
            _apa_table(df, "Observed Frequencies")
            fig = _heatmap_fig(observed, col_names, row_names, colorscale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
        elif tab_view == "Expected":
            exp_df = pd.DataFrame(
                np.round(expected, 1), index=row_names, columns=col_names
            )
            _apa_table(exp_df, "Expected Frequencies (under independence)")
            fig = _heatmap_fig(expected, col_names, row_names, colorscale="Blues")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = go.Figure()
            obs_flat = observed.flatten()
            exp_flat = expected.flatten()
            cells = [f"({i},{j})" for i in row_names for j in col_names]
            fig.add_trace(
                go.Bar(name="Observed", x=cells, y=obs_flat, marker_color="#4C78A8")
            )
            fig.add_trace(
                go.Bar(name="Expected", x=cells, y=exp_flat, marker_color="#E45756")
            )
            fig.update_layout(
                template="plotly_dark",
                height=350,
                barmode="group",
                xaxis_title="Cell",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)
    if show_deviations:
        deviations = observed - expected
        chi2_contrib = deviations**2 / expected
        comp = pd.DataFrame(
            {
                "Cell": [f"{r}×{c}" for r in row_names for c in col_names],
                "Observed": observed.flatten(),
                "Expected": np.round(expected.flatten(), 1),
                "Deviation": np.round(deviations.flatten(), 1),
                "χ² Contrib": np.round(chi2_contrib.flatten(), 3),
            }
        )
        st.dataframe(comp, use_container_width=True)
        st.info(f"""
        **Observed vs Expected Comparison**
        - χ² = Σ(O-E)²/E = {chi2_val:.3f}
        - p-value = {p_val:.4f}
        - **Larger deviations produce larger χ² values.**
        - Cell contributions > 4 contribute substantially to significance.
        """)
    with st.expander(" Educational Notes - Expected Frequency Explorer", expanded=True):
        st.info("""
        **Expected Frequencies** = (Row Total × Column Total) ÷ Grand Total
        
        This is one of the most educational tools for understanding Chi-square.
        - Expected frequencies represent what we'd see if variables were independent
        - **Observed - Expected** deviations reveal association patterns
        - The Chi-square statistic sums (O-E)²/E across all cells
        - A large deviation in any cell contributes heavily to significance
        - Visually comparing O vs E helps intuitively grasp association strength
        """)


# =====================
# 3. DIAGNOSTIC ACCURACY TABLES
# =====================


def diagnostic_accuracy():
    st.header("Diagnostic Accuracy Tables")
    st.info("""
    Diagnostic accuracy tables evaluate how well a test identifies a condition.
    The confusion matrix forms the basis for sensitivity, specificity, predictive values,
    and likelihood ratios.
    """)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("##### Confusion Matrix")
        tp = st.slider("True Positives (TP)", 0, 200, 80, key="diag_tp")
        fp = st.slider("False Positives (FP)", 0, 200, 20, key="diag_fp")
        fn = st.slider("False Negatives (FN)", 0, 200, 10, key="diag_fn")
        tn = st.slider("True Negatives (TN)", 0, 200, 90, key="diag_tn")
        show_heat = st.toggle("Show Heatmap", True, key="diag_heat")

    n_total = tp + fp + fn + tn
    prevalence = (tp + fn) / n_total if n_total > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / n_total if n_total > 0 else 0
    lr_plus = sensitivity / (1 - specificity) if specificity < 1 else float("inf")
    lr_minus = (1 - sensitivity) / specificity if specificity > 0 else float("inf")

    matrix = np.array([[tp, fp], [fn, tn]])
    with c2:
        df = pd.DataFrame(
            matrix, index=["Test +", "Test -"], columns=["Condition +", "Condition -"]
        )
        _apa_table(df, "Confusion Matrix")
        if show_heat:
            fig = _heatmap_fig(
                matrix,
                ["Condition +", "Condition -"],
                ["Test +", "Test -"],
                colorscale="RdBu_r",
            )
            st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Sensitivity",
            f"{sensitivity:.3f}",
            help="TP / (TP + FN) — ability to detect condition",
        )
    with col2:
        st.metric(
            "Specificity",
            f"{specificity:.3f}",
            help="TN / (TN + FP) — ability to rule out condition",
        )
    with col3:
        st.metric(
            "PPV",
            f"{ppv:.3f}",
            help="TP / (TP + FP) — probability condition given + test",
        )
    with col4:
        st.metric(
            "NPV",
            f"{npv:.3f}",
            help="TN / (TN + FN) — probability no condition given - test",
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prevalence", f"{prevalence:.3f}")
    with col2:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col3:
        st.metric(
            "LR+",
            f"{lr_plus:.3f}" if lr_plus != float("inf") else "∞",
            help="Sensitivity / (1 - Specificity)",
        )
    with col4:
        st.metric(
            "LR-",
            f"{lr_minus:.3f}" if lr_minus != float("inf") else "∞",
            help="(1 - Sensitivity) / Specificity",
        )

    diag_df = pd.DataFrame(
        {
            "Metric": [
                "Sensitivity",
                "Specificity",
                "PPV",
                "NPV",
                "Prevalence",
                "Accuracy",
                "LR+",
                "LR-",
            ],
            "Formula": [
                "TP/(TP+FN)",
                "TN/(TN+FP)",
                "TP/(TP+FP)",
                "TN/(TN+FN)",
                "(TP+FN)/N",
                "(TP+TN)/N",
                "Se/(1-Sp)",
                "(1-Se)/Sp",
            ],
            "Value": [
                f"{sensitivity:.4f}",
                f"{specificity:.4f}",
                f"{ppv:.4f}",
                f"{npv:.4f}",
                f"{prevalence:.4f}",
                f"{accuracy:.4f}",
                f"{lr_plus:.3f}" if lr_plus != float("inf") else "∞",
                f"{lr_minus:.3f}" if lr_minus != float("inf") else "∞",
            ],
        }
    )
    st.dataframe(diag_df, use_container_width=True)

    with st.expander(" Bayesian Updating Table", expanded=True):
        st.markdown("##### Post-Test Probabilities")
        pre_test_odds = (
            prevalence / (1 - prevalence) if prevalence < 1 else float("inf")
        )
        post_test_odds_pos = (
            pre_test_odds * lr_plus if lr_plus != float("inf") else float("inf")
        )
        post_test_prob_pos = (
            post_test_odds_pos / (1 + post_test_odds_pos)
            if post_test_odds_pos != float("inf")
            else 1.0
        )
        post_test_odds_neg = pre_test_odds * lr_minus if lr_minus != float("inf") else 0
        post_test_prob_neg = post_test_odds_neg / (1 + post_test_odds_neg)
        bayes_df = pd.DataFrame(
            {
                "Step": ["Pre-test (Prevalence)", "Post-test if +", "Post-test if -"],
                "Probability": [
                    f"{prevalence:.4f}",
                    f"{post_test_prob_pos:.4f}",
                    f"{post_test_prob_neg:.4f}",
                ],
                "Odds": [
                    f"{pre_test_odds:.4f}" if pre_test_odds != float("inf") else "∞",
                    (
                        f"{post_test_odds_pos:.4f}"
                        if post_test_odds_pos != float("inf")
                        else "∞"
                    ),
                    f"{post_test_odds_neg:.4f}",
                ],
            }
        )
        st.dataframe(bayes_df, use_container_width=True)
        st.info(f"""
        **Bayesian Updating:**
        - Pre-test probability (prevalence) = {prevalence:.3f}
        - LR+ = {lr_plus:.2f}: a positive test multiplies odds by {lr_plus:.2f}×
        - Post-test P(disease|+) = {post_test_prob_pos:.3f}
        """)
    with st.expander(" Educational Notes - Diagnostic Accuracy", expanded=True):
        st.info("""
        **Key Diagnostic Concepts:**
        - **Sensitivity**: P(+|Disease) — "how well does it catch positives?"
        - **Specificity**: P(-|No Disease) — "how well does it rule out?"
        - **PPV**: P(Disease|+) — depends heavily on prevalence
        - **NPV**: P(No Disease|-) — also prevalence-dependent
        - **LR+ > 10** or **LR- < 0.1**: convincing diagnostic evidence
        - **Bayesian thinking**: LR converts pre-test to post-test probability
        """)


# =====================
# 4. AGREEMENT TABLES
# =====================


def agreement_tables():
    st.header("Agreement Tables")
    st.info("""
    Agreement tables measure how well two raters, methods, or instruments agree.
    Cohen's Kappa corrects for chance agreement.
    """)

    tab = _tab_buttons(
        [
            "Observer Agreement Matrix",
            "Cohen's Kappa Tables",
            "Reliability Summary Tables",
        ],
        "agree_tab",
    )

    if tab == "Observer Agreement Matrix":
        _observer_agreement_widget()
    elif tab == "Cohen's Kappa Tables":
        _kappa_widget()
    elif tab == "Reliability Summary Tables":
        _reliability_widget()


def _observer_agreement_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n_cats = st.selectbox("Number of Categories", [2, 3, 4], index=0, key="oa_cats")
        strength = st.slider(
            "Agreement Strength",
            0.0,
            1.0,
            0.7,
            0.05,
            key="oa_str",
            help="Proportion of observations on the diagonal",
        )
    np.random.seed(42)
    nc = int(n_cats)
    n = 200
    base = np.random.randint(5, 30, (nc, nc))
    for i in range(nc):
        base[i, i] = int(base[i, i] * (1 + strength * 3))
    base = np.maximum(base, 1)
    row_names = [f"Rater A: Cat {chr(65+i)}" for i in range(nc)]
    col_names = [f"Rater B: Cat {chr(65+i)}" for i in range(nc)]
    df = pd.DataFrame(base, index=row_names, columns=col_names)
    df["Total"] = df.sum(axis=1)
    df.loc["Total"] = list(df.sum(axis=0))
    with c2:
        _apa_table(df, "Observer Agreement Matrix")
        fig = _heatmap_fig(base, col_names, row_names, colorscale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    total = base.sum()
    agreement = np.trace(base) / total
    chance_agree = sum(base.sum(axis=0) * base.sum(axis=1)) / total**2
    kappa = (agreement - chance_agree) / (1 - chance_agree) if chance_agree < 1 else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Observed Agreement", f"{agreement:.4f}")
    with col2:
        st.metric("Chance Agreement", f"{chance_agree:.4f}")
    with col3:
        st.metric("Cohen's Kappa", f"{kappa:.4f}")


def _kappa_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        a_diag = st.slider("Both Yes (a)", 0, 100, 40, key="kap_a")
        b = st.slider("Rater1 Yes, Rater2 No (b)", 0, 50, 10, key="kap_b")
        c = st.slider("Rater1 No, Rater2 Yes (c)", 0, 50, 15, key="kap_c")
        d_diag = st.slider("Both No (d)", 0, 100, 35, key="kap_d")
    table = np.array([[a_diag, b], [c, d_diag]])
    n_tot = a_diag + b + c + d_diag
    p_o = (a_diag + d_diag) / n_tot if n_tot > 0 else 0
    p1 = (a_diag + b) * (a_diag + c) / n_tot**2 if n_tot > 0 else 0
    p2 = (c + d_diag) * (b + d_diag) / n_tot**2 if n_tot > 0 else 0
    p_e = p1 + p2
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    se_kappa = (
        math.sqrt(p_o * (1 - p_o) / (n_tot * (1 - p_e) ** 2))
        if n_tot > 0 and p_e < 1
        else 0
    )
    z_kappa = kappa / se_kappa if se_kappa > 0 else 0
    p_kappa = 2 * (1 - stats.norm.cdf(abs(z_kappa)))
    df = pd.DataFrame(
        table,
        index=["Rater1: Yes", "Rater1: No"],
        columns=["Rater2: Yes", "Rater2: No"],
    )
    with c2:
        _apa_table(df, "Agreement Table (2×2)")
        fig = _heatmap_fig(
            table,
            ["Rater2: Yes", "Rater2: No"],
            ["Rater1: Yes", "Rater1: No"],
            colorscale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observed Agreement", f"{p_o:.4f}")
    with col2:
        st.metric("Chance Agreement", f"{p_e:.4f}")
    with col3:
        st.metric("Cohen's Kappa", f"{kappa:.4f}")
    with col4:
        kappa_label = (
            "Almost Perfect"
            if kappa > 0.8
            else (
                "Substantial"
                if kappa > 0.6
                else "Moderate" if kappa > 0.4 else "Fair" if kappa > 0.2 else "Slight"
            )
        )
        st.metric("Interpretation", kappa_label)
    st.caption(f"SE = {se_kappa:.4f}, z = {z_kappa:.3f}, p = {p_kappa:.4f}")
    with st.expander(" Educational Notes - Cohen's Kappa", expanded=True):
        st.info("""
        **Cohen's Kappa** = (observed agreement - chance agreement) / (1 - chance agreement)
        - **κ = 1**: perfect agreement
        - **κ = 0**: agreement same as chance
        - **κ < 0**: worse than chance (systematic disagreement)
        - Landis & Koch benchmarks: <0 Slight, 0-0.2 Fair, 0.2-0.4 Fair,
          0.4-0.6 Moderate, 0.6-0.8 Substantial, 0.8-1.0 Almost Perfect
        - Kappa is prevalence-dependent: high prevalence reduces kappa
        """)


def _reliability_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        icc_type = st.selectbox(
            "ICC Type",
            [
                "ICC(1,1) - Single rater, absolute",
                "ICC(2,1) - Random rater, absolute",
                "ICC(3,1) - Fixed rater, consistency",
            ],
            key="icc_type",
        )
        n_subjects = st.slider("Subjects", 10, 100, 30, key="icc_n")
        n_raters = st.selectbox("Raters", [2, 3, 4], index=0, key="icc_r")
        reliability = st.slider(
            "True Reliability",
            0.0,
            1.0,
            0.6,
            0.05,
            key="icc_rel",
            help="How consistent the measurements are",
        )
    np.random.seed(42)
    ns, nr = n_subjects, int(n_raters)
    true_scores = np.random.normal(0, 1, ns)
    noise = np.sqrt(1 - reliability) / reliability if reliability > 0 else 10
    ratings = np.array(
        [true_scores + np.random.normal(0, noise, ns) for _ in range(nr)]
    ).T
    rating_df = pd.DataFrame(ratings, columns=[f"Rater {chr(65+i)}" for i in range(nr)])
    rating_df["Subject"] = [f"S{i+1}" for i in range(ns)]
    rating_df = rating_df[["Subject"] + [f"Rater {chr(65+i)}" for i in range(nr)]]
    with c2:
        _apa_table(
            rating_df.head(10), f"Reliability Data (showing first 10 of {ns} subjects)"
        )
    ms_between = np.var(ratings.mean(axis=1), ddof=1) * ns
    ms_within = np.mean(np.var(ratings, axis=1, ddof=1))
    icc_est = (
        ms_between / (ms_between + ms_within) if (ms_between + ms_within) > 0 else 0
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ICC Estimate", f"{icc_est:.4f}")
    with col2:
        st.metric("Between-Subject Var", f"{ms_between:.4f}")
    with col3:
        st.metric("Within-Subject Var", f"{ms_within:.4f}")
    with st.expander(" Educational Notes - Reliability", expanded=True):
        st.info("""
        **Intraclass Correlation Coefficient (ICC)** measures reliability.
        - **ICC near 1**: measurements are highly consistent
        - **ICC near 0**: measurements are mostly noise
        - ICC(1,1): absolute agreement, single rater
        - ICC(2,1): absolute agreement, random raters
        - ICC(3,1): consistency, fixed raters
        - Benchmarks: <0.5 poor, 0.5-0.75 moderate, 0.75-0.9 good, >0.9 excellent
        """)


# =====================
# 5. REGRESSION SUMMARY TABLES
# =====================


def regression_tables():
    st.header("Regression Summary Tables")
    st.info("""
    Regression summary tables present model coefficients, model fit statistics,
    and ANOVA decomposition for regression analyses.
    """)

    tab = _tab_buttons(
        [
            "Coefficient Tables",
            "Odds Ratio Tables",
            "Model Fit Tables",
            "ANOVA Tables",
            "Residual Summary Tables",
        ],
        "reg_tab",
    )

    if tab == "Coefficient Tables":
        _coeff_table_widget()
    elif tab == "Odds Ratio Tables":
        _odds_ratio_table_widget()
    elif tab == "Model Fit Tables":
        _model_fit_widget()
    elif tab == "ANOVA Tables":
        _anova_table_widget()
    elif tab == "Residual Summary Tables":
        _residual_table_widget()


def _coeff_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n_predictors = st.selectbox(
            "Number of Predictors", [1, 2, 3, 4], index=1, key="coef_np"
        )
        r_sq = st.slider("R² (model fit)", 0.0, 0.99, 0.45, 0.05, key="coef_r2")
        n = st.slider("Sample Size", 20, 500, 100, key="coef_n")
    np.random.seed(42)
    npred = int(n_predictors)
    names = ["Intercept"] + [f"X{i+1}" for i in range(npred)]
    beta = np.random.uniform(-2, 2, npred + 1)
    se = np.random.uniform(0.05, 0.5, npred + 1)
    t_vals = beta / se
    p_vals = 2 * (1 - stats.t.cdf(abs(t_vals), n - npred - 1))
    stars = [
        "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        for p in p_vals
    ]
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    df = pd.DataFrame(
        {
            "Predictor": names,
            "β": np.round(beta, 4),
            "SE": np.round(se, 4),
            "t": np.round(t_vals, 3),
            "p-value": np.round(p_vals, 4),
            "95% CI": [f"[{l:.4f}, {h:.4f}]" for l, h in zip(ci_low, ci_high)],
            "": stars,
        }
    )
    with c2:
        _apa_table(df, "Regression Coefficient Table")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=beta,
                y=names,
                mode="markers",
                marker=dict(size=10, color="#4C78A8"),
                error_x=dict(type="data", array=1.96 * se),
                hovertemplate="%{y}: %{x:.3f} [%{error_x.array:.3f}]<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Coefficient (95% CI)",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Signif. codes: *** p < 0.001, ** p < 0.01, * p < 0.05")


def _odds_ratio_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n_pred = st.selectbox(
            "Number of Predictors", [1, 2, 3, 4], index=1, key="or_np"
        )
        n = st.slider("Sample Size", 50, 1000, 200, key="or_n")
    np.random.seed(42)
    npred = int(n_pred)
    names = [f"X{i+1}" for i in range(npred)]
    or_vals = np.random.uniform(0.3, 3.0, npred)
    log_or = np.log(or_vals)
    se_log = np.random.uniform(0.1, 0.5, npred)
    ci_low = np.exp(log_or - 1.96 * se_log)
    ci_high = np.exp(log_or + 1.96 * se_log)
    z_vals = log_or / se_log
    p_vals = 2 * (1 - stats.norm.cdf(abs(z_vals)))
    df = pd.DataFrame(
        {
            "Predictor": names,
            "OR": np.round(or_vals, 3),
            "95% CI": [f"[{l:.3f}, {h:.3f}]" for l, h in zip(ci_low, ci_high)],
            "z": np.round(z_vals, 3),
            "p-value": np.round(p_vals, 4),
            "": [
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                for p in p_vals
            ],
        }
    )
    with c2:
        _apa_table(df, "Odds Ratio Table (Logistic Regression)")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=or_vals,
                y=names,
                mode="markers",
                marker=dict(size=10, color="#E45756"),
                error_x=dict(
                    type="data",
                    array=[or_vals[i] - ci_low[i] for i in range(npred)],
                    symmetric=False,
                ),
                hovertemplate="%{y}: OR=%{x:.2f}<extra></extra>",
            )
        )
        fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Odds Ratio (95% CI)",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.caption("OR = 1: no effect | OR > 1: increased odds | OR < 1: decreased odds")


def _model_fit_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n_predictors = st.slider("Number of Predictors", 1, 10, 3, key="mf_np")
        n = st.slider("Sample Size", 20, 1000, 100, key="mf_n")
        r2 = st.slider("R²", 0.0, 0.99, 0.45, 0.01, key="mf_r2")
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_predictors - 1)
    f_val = (r2 / n_predictors) / ((1 - r2) / (n - n_predictors - 1)) if r2 < 1 else 999
    p_f = 1 - stats.f.cdf(f_val, n_predictors, n - n_predictors - 1)
    aic = n * np.log(1 - r2) + 2 * (n_predictors + 1) if r2 < 1 else -999
    bic = n * np.log(1 - r2) + (n_predictors + 1) * np.log(n) if r2 < 1 else -999
    rmse = np.sqrt(1 - r2) * np.std(np.random.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "Metric": [
                "R²",
                "Adjusted R²",
                "F-statistic",
                "df (model)",
                "df (residual)",
                "p-value (F)",
                "AIC",
                "BIC",
                "RMSE",
            ],
            "Value": [
                f"{r2:.4f}",
                f"{r2_adj:.4f}",
                f"{f_val:.3f}",
                f"{n_predictors}",
                f"{n - n_predictors - 1}",
                f"{p_f:.4e}",
                f"{aic:.2f}",
                f"{bic:.2f}",
                f"{rmse:.4f}",
            ],
        }
    )
    with c2:
        _apa_table(df, "Model Fit Summary")
        fig = go.Figure(
            data=go.Indicator(
                mode="gauge+number+delta",
                value=r2 * 100,
                title={"text": "R² (%)"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4C78A8"},
                    "steps": [
                        {"range": [0, 25], "color": "rgba(255,0,0,0.2)"},
                        {"range": [25, 50], "color": "rgba(255,255,0,0.2)"},
                        {"range": [50, 75], "color": "rgba(0,255,0,0.2)"},
                        {"range": [75, 100], "color": "rgba(0,128,0,0.2)"},
                    ],
                },
            )
        )
        fig.update_layout(
            template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)


def _anova_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n_groups = st.selectbox(
            "Number of Groups", [2, 3, 4, 5], index=2, key="anova_ng"
        )
        effect = st.slider(
            "Effect Size (Cohen's f)", 0.0, 1.0, 0.4, 0.05, key="anova_f"
        )
        n_per = st.slider("Per Group", 10, 100, 30, key="anova_n")
    np.random.seed(42)
    ng = int(n_groups)
    n_total = ng * n_per
    grand_mean = 0
    group_means = [effect * i for i in range(ng)]
    data = [np.random.normal(m, 1, n_per) for m in group_means]
    all_data = np.concatenate(data)
    ss_between = sum(n_per * (m - np.mean(all_data)) ** 2 for m in group_means)
    ss_within = sum(np.sum((d - m) ** 2) for d, m in zip(data, group_means))
    ss_total = ss_between + ss_within
    df_between = ng - 1
    df_within = n_total - ng
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_val = ms_between / ms_within
    p_val = 1 - stats.f.cdf(f_val, df_between, df_within)
    eta_sq = ss_between / ss_total
    anova_df = pd.DataFrame(
        {
            "Source": ["Between Groups", "Within Groups", "Total"],
            "SS": [f"{ss_between:.3f}", f"{ss_within:.3f}", f"{ss_total:.3f}"],
            "df": [f"{df_between}", f"{df_within}", f"{df_between + df_within}"],
            "MS": [f"{ms_between:.3f}", f"{ms_within:.3f}", ""],
            "F": [f"{f_val:.3f}", "", ""],
            "p-value": [f"{p_val:.4f}", "", ""],
            "η²": [f"{eta_sq:.4f}", "", ""],
        }
    )
    with c2:
        _apa_table(anova_df, "ANOVA Table")
    with st.expander(" Educational Notes - ANOVA Table", expanded=True):
        st.info("""
        **ANOVA decomposes total variance:**
        - **SS Between**: variance explained by group differences
        - **SS Within**: residual/error variance within groups
        - **F = MS_between / MS_within**: larger F → more group separation
        - **η² = SS_between / SS_total**: proportion of variance explained
        - η² benchmarks: 0.01 small, 0.06 medium, 0.14 large
        """)


def _residual_table_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Sample Size", 20, 200, 50, key="res_n")
        dist = st.selectbox(
            "Error Distribution",
            ["Normal", "Heavy-tailed (t)", "Skewed"],
            key="res_dist",
        )
    np.random.seed(42)
    x = np.random.uniform(0, 10, n)
    if dist == "Normal":
        errors = np.random.normal(0, 1, n)
    elif dist == "Heavy-tailed (t)":
        errors = np.random.standard_t(3, n)
    else:
        errors = np.random.gamma(2, 1, n) - 2
    y = 2 + 0.5 * x + errors
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_pred = intercept + slope * x
    residuals = y - y_pred
    standardized = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)
    res_df = pd.DataFrame(
        {
            "Observation": range(1, n + 1),
            "Observed": np.round(y, 3),
            "Predicted": np.round(y_pred, 3),
            "Residual": np.round(residuals, 3),
            "Standardized": np.round(standardized, 3),
        }
    )
    with c2:
        _apa_table(res_df.head(15), f"Residual Summary (first 15 of {n})")
    fig, axes = None, None
    colr1, colr2 = st.columns(2)
    with colr1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(color="#4C78A8", size=6),
                hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
            )
        )
        fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig1.update_layout(
            template="plotly_dark",
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Predicted",
            yaxis_title="Residuals",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with colr2:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=residuals, nbinsx=20, marker_color="#4C78A8"))
        fig2.update_layout(
            template="plotly_dark",
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Residual",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig2, use_container_width=True)
    shapiro_stat, shapiro_p = stats.shapiro(
        residuals[:5000] if len(residuals) > 5000 else residuals
    )
    st.info(
        f"Shapiro-Wilk normality test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}"
        + (
            " → Residuals appear normal"
            if shapiro_p > 0.05
            else " → Residuals deviate from normality"
        )
    )


# =====================
# 6. EFFECT SIZE TABLES
# =====================


def effect_size_tables():
    st.header("Effect Size Tables")
    st.info("""
    Effect sizes measure the magnitude of an effect, independent of sample size.
    They allow comparison across studies and are essential for meta-analysis.
    """)

    tab = _tab_buttons(
        ["Cohen's d", "η² (Eta-squared)", "Cramér's V", "Odds Ratios", "Relative Risk"],
        "es_tab",
    )

    if tab == "Cohen's d":
        _cohens_d_widget()
    elif tab == "η² (Eta-squared)":
        _eta_sq_widget()
    elif tab == "Cramér's V":
        _cramer_v_widget()
    elif tab == "Odds Ratios":
        _odds_ratio_es_widget()
    elif tab == "Relative Risk":
        _relative_risk_widget()


def _cohens_d_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        m1 = st.slider("Mean Group 1", -10.0, 10.0, 0.0, 0.1, key="cd_m1")
        m2 = st.slider("Mean Group 2", -10.0, 10.0, 1.5, 0.1, key="cd_m2")
        sd1 = st.slider("SD Group 1", 0.1, 5.0, 1.0, 0.1, key="cd_sd1")
        sd2 = st.slider("SD Group 2", 0.1, 5.0, 1.0, 0.1, key="cd_sd2")
        n1 = st.slider("n Group 1", 5, 200, 30, key="cd_n1")
        n2 = st.slider("n Group 2", 5, 200, 30, key="cd_n2")
    pooled_sd = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    d = abs(m1 - m2) / pooled_sd if pooled_sd > 0 else 0
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d
    label = (
        "Large"
        if d >= 0.8
        else "Medium" if d >= 0.5 else "Small" if d >= 0.2 else "Negligible"
    )
    df = pd.DataFrame(
        {
            "Metric": [
                "Cohen's d",
                "Pooled SD",
                "SE(d)",
                "95% CI Lower",
                "95% CI Upper",
                "Interpretation",
            ],
            "Value": [
                f"{d:.4f}",
                f"{pooled_sd:.4f}",
                f"{se_d:.4f}",
                f"{ci_low:.4f}",
                f"{ci_high:.4f}",
                label,
            ],
        }
    )
    with c2:
        _apa_table(df, "Cohen's d Effect Size")
        fig = go.Figure()
        x_range = np.linspace(-4, 4, 200)
        y1 = stats.norm.pdf(x_range, 0, 1)
        y2 = stats.norm.pdf(x_range, d, 1)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y1,
                mode="lines",
                name="Group 1",
                fill="tozeroy",
                line=dict(color="#4C78A8"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y2,
                mode="lines",
                name="Group 2",
                fill="tozeroy",
                line=dict(color="#E45756"),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Value",
            yaxis_title="Density",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"d = {d:.3f}: {label} effect. Benchmarks: 0.2=Small, 0.5=Medium, 0.8=Large"
    )


def _eta_sq_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        n_groups = st.selectbox("Number of Groups", [2, 3, 4, 5], index=2, key="es_ng")
        f_val = st.slider("Cohen's f", 0.0, 1.0, 0.4, 0.01, key="es_f")
        n = st.slider("Per Group", 10, 100, 30, key="es_n")
    eta_sq = f_val**2 / (1 + f_val**2)
    eta_sq_partial = f_val**2 / (1 + f_val**2)
    omega_sq = (
        (f_val**2 - (1 / (int(n_groups) * n))) / (1 + f_val**2)
        if int(n_groups) * n > 0
        else 0
    )
    label = (
        "Large"
        if eta_sq >= 0.14
        else "Medium" if eta_sq >= 0.06 else "Small" if eta_sq >= 0.01 else "Negligible"
    )
    df = pd.DataFrame(
        {
            "Metric": ["η²", "Partial η²", "ω²", "Cohen's f", "Interpretation"],
            "Value": [
                f"{eta_sq:.4f}",
                f"{eta_sq_partial:.4f}",
                f"{omega_sq:.4f}",
                f"{f_val:.4f}",
                label,
            ],
        }
    )
    with c2:
        _apa_table(df, "Eta-Squared Effect Size")
    st.caption("η² benchmarks: 0.01=Small, 0.06=Medium, 0.14=Large. η² = f²/(1+f²)")


def _cramer_v_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        rows = st.selectbox("Rows", [2, 3, 4, 5], index=0, key="cv_r")
        cols = st.selectbox("Columns", [2, 3, 4, 5], index=0, key="cv_c")
        strength = st.slider("Association Strength", 0.0, 1.0, 0.3, 0.05, key="cv_str")
    np.random.seed(42)
    r, c = int(rows), int(cols)
    table = np.random.randint(5, 30, (r, c))
    chi2_val = strength * min(r - 1, c - 1) * table.sum() * 0.1
    chi2_val = max(chi2_val, 0.01)
    n_total = table.sum()
    v = (
        math.sqrt(chi2_val / (n_total * min(r - 1, c - 1)))
        if n_total > 0 and min(r - 1, c - 1) > 0
        else 0
    )
    df_rows = (r - 1) * (c - 1)
    p_val = 1 - stats.chi2.cdf(chi2_val, df_rows)
    label = (
        "Large"
        if v >= 0.5
        else "Medium" if v >= 0.3 else "Small" if v >= 0.1 else "Negligible"
    )
    df = pd.DataFrame(
        {
            "Metric": ["Cramér's V", "χ²", "df", "p-value", "Interpretation"],
            "Value": [
                f"{v:.4f}",
                f"{chi2_val:.3f}",
                f"{df_rows}",
                f"{p_val:.4f}",
                label,
            ],
        }
    )
    with c2:
        _apa_table(df, "Cramér's V Effect Size")
    st.caption(
        "Cramér's V benchmarks: 0.1=Small, 0.3=Medium, 0.5=Large. V = √(χ²/(N × min(r-1,c-1)))"
    )


def _odds_ratio_es_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("##### 2×2 Table")
        a = st.slider("Exposed + Outcome +", 0, 200, 40, key="ore_a")
        b = st.slider("Exposed + Outcome -", 0, 200, 20, key="ore_b")
        c = st.slider("Exposed - Outcome +", 0, 200, 10, key="ore_c")
        d = st.slider("Exposed - Outcome -", 0, 200, 30, key="ore_d")
    if b > 0 and c > 0:
        or_val = (a * d) / (b * c)
    else:
        or_val = float("inf")
    log_or = math.log(or_val) if or_val != float("inf") and or_val > 0 else 0
    se_log = (
        math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        if all(x > 0 for x in [a, b, c, d])
        else 0
    )
    ci_low = math.exp(log_or - 1.96 * se_log) if se_log > 0 else 0
    ci_high = math.exp(log_or + 1.96 * se_log) if se_log > 0 else float("inf")
    rr_exp = a / (a + b) if (a + b) > 0 else 0
    rr_nexp = c / (c + d) if (c + d) > 0 else 0
    rr = rr_exp / rr_nexp if rr_nexp > 0 else float("inf")
    arr_val = rr_exp - rr_nexp
    rrr_val = arr_val / rr_nexp if rr_nexp > 0 else float("inf")
    nnt = 1 / abs(arr_val) if abs(arr_val) > 0 else float("inf")
    df = pd.DataFrame(
        {
            "Metric": [
                "Odds Ratio",
                "95% CI (OR)",
                "Relative Risk",
                "ARR",
                "RRR",
                "NNT",
            ],
            "Value": [
                f"{or_val:.4f}" if or_val != float("inf") else "∞",
                f"[{ci_low:.4f}, {ci_high:.4f}]",
                f"{rr:.4f}" if rr != float("inf") else "∞",
                f"{arr_val:.4f}",
                f"{rrr_val:.4f}" if rrr_val != float("inf") else "∞",
                f"{nnt:.1f}" if nnt != float("inf") else "∞",
            ],
        }
    )
    with c2:
        _apa_table(df, "Odds Ratio & Risk Measures")
    with st.expander(
        " Educational Notes - Odds Ratio / Risk Ratio Explorer", expanded=True
    ):
        st.info(f"""
        **Key Measures from 2×2 Table:**
        - **OR = {or_val:.3f}**: odds of outcome in exposed ÷ odds in unexposed
        - **RR = {rr:.3f}**: risk of outcome in exposed ÷ risk in unexposed
        - **ARR = {arr_val:.3f}**: absolute risk reduction (exposed - unexposed)
        - **RRR = {rrr_val:.3f}**: relative risk reduction (ARR / risk in unexposed)
        - **NNT = {nnt:.1f}**: number needed to treat (1/ARR)
        - **OR ≈ RR** when outcome is rare (<10%)
        """)


def _relative_risk_widget():
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("##### 2×2 Table for RR")
        a_rr = st.slider("Exposed + Event +", 0, 200, 30, key="rr_a")
        b_rr = st.slider("Exposed + Event -", 0, 200, 70, key="rr_b")
        c_rr = st.slider("Unexposed + Event +", 0, 200, 15, key="rr_c")
        d_rr = st.slider("Unexposed + Event -", 0, 200, 85, key="rr_d")
    risk_exp = a_rr / (a_rr + b_rr) if (a_rr + b_rr) > 0 else 0
    risk_unexp = c_rr / (c_rr + d_rr) if (c_rr + d_rr) > 0 else 0
    rr_val = risk_exp / risk_unexp if risk_unexp > 0 else float("inf")
    se_log_rr = (
        math.sqrt(1 / a_rr - 1 / (a_rr + b_rr) + 1 / c_rr - 1 / (c_rr + d_rr))
        if all(x > 0 for x in [a_rr, b_rr, c_rr, d_rr])
        else 0
    )
    log_rr = math.log(rr_val) if rr_val != float("inf") and rr_val > 0 else 0
    ci_low_rr = math.exp(log_rr - 1.96 * se_log_rr) if se_log_rr > 0 else 0
    ci_high_rr = math.exp(log_rr + 1.96 * se_log_rr) if se_log_rr > 0 else float("inf")
    rd_val = risk_exp - risk_unexp
    nnt_val = 1 / abs(rd_val) if abs(rd_val) > 0 else float("inf")
    df = pd.DataFrame(
        {
            "Metric": [
                "Risk Exposed",
                "Risk Unexposed",
                "Relative Risk",
                "95% CI (RR)",
                "Risk Difference",
                "NNT",
            ],
            "Value": [
                f"{risk_exp:.4f}",
                f"{risk_unexp:.4f}",
                f"{rr_val:.4f}" if rr_val != float("inf") else "∞",
                f"[{ci_low_rr:.4f}, {ci_high_rr:.4f}]",
                f"{rd_val:.4f}",
                f"{nnt_val:.1f}" if nnt_val != float("inf") else "∞",
            ],
        }
    )
    with c2:
        _apa_table(df, "Relative Risk & Risk Difference")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(name="Exposed", x=["Risk"], y=[risk_exp], marker_color="#E45756")
        )
        fig.add_trace(
            go.Bar(name="Unexposed", x=["Risk"], y=[risk_unexp], marker_color="#4C78A8")
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            barmode="group",
            yaxis=dict(range=[0, 1]),
            yaxis_title="Risk",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"RR = {rr_val:.3f}: "
        + (
            "Increased risk"
            if rr_val > 1
            else "Decreased risk" if rr_val < 1 else "No effect"
        )
        + f" | NNT = {nnt_val:.1f}"
    )


# =====================
# 7. EDUCATIONAL MODULES
# =====================


def educational_modules():
    st.header(" Educational Modules")
    st.info("""
    Interactive educational widgets designed to build intuition about core statistical concepts.
    These modules are the most pedagogically valuable tools on this platform.
    """)

    module = _tab_buttons(
        [
            "Frequency Table Explorer",
            "Cross-Tabulation Explorer",
            "Expected Frequency Explorer",
            "Odds Ratio / Risk Ratio Explorer",
            "Conditional Probability Explorer",
            "Bayesian Updating Table",
        ],
        "edu_mod",
    )

    if module == "Frequency Table Explorer":
        _freq_explorer()
    elif module == "Cross-Tabulation Explorer":
        _cross_tab_explorer()
    elif module == "Expected Frequency Explorer":
        _expected_freq_explorer()
    elif module == "Odds Ratio / Risk Ratio Explorer":
        _odds_risk_explorer()
    elif module == "Conditional Probability Explorer":
        _cond_prob_explorer()
    elif module == "Bayesian Updating Table":
        _bayesian_updater()


def _freq_explorer():
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="fe_n")
        bins = st.slider("Bins", 3, 15, 6, key="fe_bins")
        dist = st.selectbox(
            "Distribution", ["Normal", "Skewed", "Uniform", "Bimodal"], key="fe_dist"
        )
        show_freq = st.toggle("Show Absolute Freq", True, key="fe_abs")
        show_rel = st.toggle("Show Relative Freq", True, key="fe_rel")
        show_cum = st.toggle("Show Cumulative Freq", True, key="fe_cum")
    data = _generate_sample_data(n, dist)
    counts, edges = np.histogram(data, bins=bins)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
    rel = counts / n
    cum = np.cumsum(counts)
    fig = go.Figure()
    if show_freq:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=counts,
                name="Absolute",
                marker_color="#4C78A8",
                hovertemplate="%{x}<br>Freq: %{y}<extra></extra>",
            )
        )
    if show_rel:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=rel * 100,
                mode="lines+markers",
                name="Relative %",
                marker_color="#E45756",
                yaxis="y2",
            )
        )
    if show_cum:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=cum,
                mode="lines+markers",
                name="Cumulative",
                marker_color="#00CC96",
                yaxis="y3",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=10, b=80),
        xaxis_title="Class",
        yaxis_title="Frequency",
        yaxis2=dict(overlaying="y", side="right", title="Relative %", range=[0, 105]),
        yaxis3=dict(overlaying="y", side="right", title="Cumulative", position=0.95),
        hovermode="x unified",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    multi = pd.DataFrame(
        {
            "Interval": labels,
            "Frequency": counts,
            "Relative": [f"{v*100:.1f}%" for v in rel],
            "Cumulative": cum,
            "Cum %": [f"{v*100:.1f}%" for v in cum / n],
        }
    )
    _apa_table(multi, "Complete Frequency Distribution")
    with st.expander(" Learn - Frequency Table Explorer", expanded=True):
        st.info("""
        **Three Types of Frequency:**
        1. **Absolute Frequency** — raw count per interval
        2. **Relative Frequency** — proportion of total (always sums to 1.0 or 100%)
        3. **Cumulative Frequency** — running total (always ends at n or 100%)
        
        The **histogram** shows absolute frequency. The **overlay curves** show
        relative and cumulative distributions. Notice how the cumulative curve
        always rises and the relative curve always sums to 100%.
        """)


def _cross_tab_explorer():
    c1, c2 = st.columns([1, 2])
    with c1:
        r = st.selectbox("Rows", [2, 3], index=0, key="cte_r")
        c = st.selectbox("Columns", [2, 3], index=0, key="cte_c")
        show_expected = st.toggle("Show Expected Frequencies", True, key="cte_exp")
        show_conditional = st.toggle(
            "Show Conditional Probabilities", True, key="cte_cond"
        )
    np.random.seed(42)
    rows, cols = int(r), int(c)
    observed = np.random.randint(5, 50, (rows, cols))
    if rows == 2 and cols == 2:
        observed = np.array([[40, 20], [10, 30]])
    _, _, _, expected = chi2_contingency(observed)
    row_tot = observed.sum(axis=1, keepdims=True)
    col_tot = observed.sum(axis=0, keepdims=True)
    grand_tot = observed.sum()
    row_names = [f"Row {chr(65+i)}" for i in range(rows)]
    col_names = [f"Col {j+1}" for j in range(cols)]
    with c2:
        st.markdown("##### Observed Frequencies")
        df_obs = pd.DataFrame(observed, index=row_names, columns=col_names)
        st.dataframe(df_obs, use_container_width=True)
        if show_expected:
            st.markdown("##### Expected Frequencies (under independence)")
            df_exp = pd.DataFrame(
                np.round(expected, 1), index=row_names, columns=col_names
            )
            st.dataframe(df_exp, use_container_width=True)
    if show_conditional:
        st.markdown("##### Conditional Probabilities")
        p_row_given_col = observed / col_tot
        p_col_given_row = observed / row_tot
        colc1, colc2 = st.columns(2)
        with colc1:
            st.caption("P(Row | Column) — sums to 1 down each column")
            st.dataframe(
                pd.DataFrame(
                    np.round(p_row_given_col, 3), index=row_names, columns=col_names
                ),
                use_container_width=True,
            )
        with colc2:
            st.caption("P(Column | Row) — sums to 1 across each row")
            st.dataframe(
                pd.DataFrame(
                    np.round(p_col_given_row, 3), index=row_names, columns=col_names
                ),
                use_container_width=True,
            )
    chi2_val, p_val, _, _ = chi2_contingency(observed)
    with st.expander(" Learn - Cross-Tabulation Explorer", expanded=True):
        st.info(f"""
        **Key Concepts:**
        - **Marginal Totals**: row sums and column sums
        - **Conditional Probability**: P(Column|Row) = cell / row total
        - **Independence**: P(A∩B) = P(A) × P(B), so expected = (row×col)/total
        - **Association**: large differences between observed and expected indicate association
        - **χ² = {chi2_val:.3f}**, p = {p_val:.4f}
        """)


def _expected_freq_explorer():
    st.markdown("### Expected Frequency Explorer")
    st.info("""
    This is one of the most valuable educational tools. It shows observed counts,
    expected counts under independence, and cell-wise Chi-square contributions.
    This is rarely taught visually — and it builds deep intuition for how Chi-square works.
    """)
    c1, c2 = st.columns([1, 2])
    with c1:
        rows = st.selectbox("Rows", [2, 3], index=0, key="efe_r")
        cols = st.selectbox("Columns", [2, 3], index=0, key="efe_c")
        st.markdown("##### Adjust cell values to see how χ² changes")
        if int(rows) == 2 and int(cols) == 2:
            a = st.slider("Cell (R1,C1)", 1, 100, 40, key="efe_a")
            b = st.slider("Cell (R1,C2)", 1, 100, 20, key="efe_b")
            c_sl = st.slider("Cell (R2,C1)", 1, 100, 10, key="efe_c2")
            d = st.slider("Cell (R2,C2)", 1, 100, 30, key="efe_d")
            observed = np.array([[a, b], [c_sl, d]])
        else:
            observed = np.random.randint(5, 50, (int(rows), int(cols)))
            observed = np.maximum(observed, 1)
    chi2_val, p_val, dof, expected = chi2_contingency(observed)
    deviations = observed - expected
    chi2_contrib = deviations**2 / expected
    row_names = [f"Row {chr(65+i)}" for i in range(observed.shape[0])]
    col_names = [f"Col {j+1}" for j in range(observed.shape[1])]
    view = st.radio(
        "View",
        ["Observed vs Expected", "Chi-square Contributions", "Heatmap"],
        horizontal=True,
        key="efe_view",
    )
    with c2:
        if view == "Observed vs Expected":
            fig = go.Figure()
            obs_flat = observed.flatten()
            exp_flat = expected.flatten()
            cells = [f"{r}×{c}" for r in row_names for c in col_names]
            fig.add_trace(
                go.Bar(
                    name="Observed",
                    x=cells,
                    y=obs_flat,
                    marker_color="#4C78A8",
                    hovertemplate="%{x}<br>Observed: %{y}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Expected",
                    x=cells,
                    y=exp_flat,
                    marker_color="#E45756",
                    hovertemplate="%{x}<br>Expected: %{y:.1f}<extra></extra>",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                height=350,
                barmode="group",
                xaxis_title="Cell",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif view == "Chi-square Contributions":
            fig = _heatmap_fig(
                chi2_contrib,
                col_names,
                row_names,
                colorscale="Reds",
                text_template="%{text:.3f}",
            )
            fig.update_layout(title="Cell-wise χ² Contribution (O-E)²/E", height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Cells with contributions > 4 contribute substantially to the total χ²"
            )
        else:
            fig = _heatmap_fig(observed, col_names, row_names, colorscale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
    comp = pd.DataFrame(
        {
            "Cell": [f"{r}×{c}" for r in row_names for c in col_names],
            "Observed": observed.flatten(),
            "Expected": np.round(expected.flatten(), 1),
            "Deviation": np.round(deviations.flatten(), 1),
            "χ² Contrib": np.round(chi2_contrib.flatten(), 3),
        }
    )
    _apa_table(comp, "Detailed Comparison: Observed vs Expected")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total χ²", f"{chi2_val:.3f}")
    with col2:
        st.metric("p-value", f"{p_val:.4f}")
    with col3:
        cramer_v = (
            math.sqrt(
                chi2_val
                / (observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1))
            )
            if observed.sum() > 0
            and min(observed.shape[0] - 1, observed.shape[1] - 1) > 0
            else 0
        )
        st.metric("Cramér's V", f"{cramer_v:.4f}")
    with st.expander(" How Chi-Square Works - Visual Explanation", expanded=True):
        st.info("""
        **Chi-square Statistic = Σ (O-E)²/E**
        1. **Expected (E)**: what we'd see if variables were independent — E = (row×col)/total
        2. **Deviation (O-E)**: how different reality is from independence
        3. **Squared deviation (O-E)²**: penalizes large differences more
        4. **Divide by E**: standardizes — a deviation of 10 matters more when E=5 than when E=100
        5. **Sum across all cells**: total χ²
        
        **The key insight**: large deviations in cells with small expected frequencies
        contribute MOST to χ². This is why Fisher's exact test is preferred for small samples.
        """)


def _odds_risk_explorer():
    st.markdown("### Odds Ratio / Risk Ratio Explorer")
    st.info("""
    Especially valuable for medical/clinical interpretation. 
    Adjust the 2×2 table interactively to see how OR, RR, ARR, RRR, and NNT change.
    """)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("##### 2×2 Table")
        a = st.slider("Exposed + Outcome +", 1, 200, 40, key="ore2_a")
        b = st.slider("Exposed + Outcome -", 0, 200, 20, key="ore2_b")
        c = st.slider("Unexposed + Outcome +", 0, 200, 10, key="ore2_c")
        d = st.slider("Unexposed + Outcome -", 1, 200, 30, key="ore2_d")
    n_exp = a + b
    n_unexp = c + d
    risk_exp = a / n_exp if n_exp > 0 else 0
    risk_unexp = c / n_unexp if n_unexp > 0 else 0
    or_val = (a * d) / (b * c) if b > 0 and c > 0 else float("inf")
    rr_val = risk_exp / risk_unexp if risk_unexp > 0 else float("inf")
    arr_val = risk_unexp - risk_exp
    rrr_val = arr_val / risk_unexp if risk_unexp > 0 else float("inf")
    nnt_val = 1 / abs(arr_val) if abs(arr_val) > 0 else float("inf")
    table = np.array([[a, b], [c, d]])
    _, p_fisher = fisher_exact(table)
    with c2:
        df = pd.DataFrame(
            table, index=["Exposed", "Unexposed"], columns=["Outcome +", "Outcome -"]
        )
        _apa_table(df, "2×2 Contingency Table")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Odds Ratio (OR)", f"{or_val:.3f}" if or_val != float("inf") else "∞")
    with col2:
        st.metric(
            "Relative Risk (RR)", f"{rr_val:.3f}" if rr_val != float("inf") else "∞"
        )
    with col3:
        st.metric(
            "ARR",
            f"{arr_val:.4f}",
            help="Absolute Risk Reduction = Risk(unexp) - Risk(exp)",
        )
    with col4:
        st.metric(
            "RRR",
            f"{rrr_val:.4f}" if rrr_val != float("inf") else "∞",
            help="Relative Risk Reduction = ARR / Risk(unexp)",
        )
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "NNT",
            f"{nnt_val:.1f}" if nnt_val != float("inf") else "∞",
            help="Number Needed to Treat = 1/|ARR|",
        )
    with col2:
        st.metric("Fisher's p", f"{p_fisher:.4f}")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Exposed (Risk)",
            x=["Risk"],
            y=[risk_exp * 100],
            marker_color="#E45756",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Unexposed (Risk)",
            x=["Risk"],
            y=[risk_unexp * 100],
            marker_color="#4C78A8",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=250,
        barmode="group",
        yaxis=dict(range=[0, 100]),
        yaxis_title="Risk (%)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander(" Interpreting OR, RR, ARR, RRR, NNT", expanded=True):
        st.info(f"""
        **OR = {or_val:.3f}** — odds of outcome in exposed ÷ odds in unexposed
        **RR = {rr_val:.3f}** — risk in exposed ÷ risk in unexposed
        **ARR = {arr_val:.3f}** — absolute difference in risk (unexposed - exposed)
        **RRR = {rrr_val:.3f}** — proportion of risk reduced by exposure/treatment
        **NNT = {nnt_val:.1f}** — number of patients who need to be treated for one to benefit
        
        **When to use OR vs RR:**
        - OR used in case-control studies and logistic regression
        - RR preferred in cohort studies and RCTs
        - OR ≈ RR when outcome is rare (<10%)
        - OR exaggerates the effect when outcome is common
        """)


def _cond_prob_explorer():
    st.markdown("### Conditional Probability Explorer")
    st.info("""
    Learn how P(A|B) — the probability of A given B — is calculated
    directly from contingency tables. This is fundamental to understanding
    diagnostic tests, Bayesian reasoning, and association.
    """)
    c1, c2 = st.columns([1, 2])
    with c1:
        a = st.slider("Cell (A∩B)", 0, 100, 30, key="cp_a")
        b = st.slider("Cell (A∩¬B)", 0, 100, 10, key="cp_b")
        c_sl = st.slider("Cell (¬A∩B)", 0, 100, 20, key="cp_c")
        d_sl = st.slider("Cell (¬A∩¬B)", 0, 100, 40, key="cp_d")
        direction = st.radio(
            "Conditional Probability Direction",
            ["P(A | B)", "P(B | A)", "P(A | ¬B)", "P(¬A | B)"],
            horizontal=True,
            key="cp_dir",
        )
    total = a + b + c_sl + d_sl
    table = np.array([[a, b], [c_sl, d_sl]])
    if direction == "P(A | B)":
        num, den = a, a + c_sl
        label = "P(A | B) = P(A∩B) / P(B)"
    elif direction == "P(B | A)":
        num, den = a, a + b
        label = "P(B | A) = P(A∩B) / P(A)"
    elif direction == "P(A | ¬B)":
        num, den = b, b + d_sl
        label = "P(A | ¬B) = P(A∩¬B) / P(¬B)"
    else:
        num, den = c_sl, a + c_sl
        label = "P(¬A | B) = P(¬A∩B) / P(B)"
    prob = num / den if den > 0 else 0
    p_a = (a + b) / total if total > 0 else 0
    p_b = (a + c_sl) / total if total > 0 else 0
    p_a_and_b = a / total if total > 0 else 0
    p_a_given_b = a / (a + c_sl) if (a + c_sl) > 0 else 0
    with c2:
        df = pd.DataFrame(
            table, index=["A present", "A absent"], columns=["B present", "B absent"]
        )
        _apa_table(df, "Contingency Table")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P(A)", f"{p_a:.4f}")
    with col2:
        st.metric("P(B)", f"{p_b:.4f}")
    with col3:
        st.metric("P(A∩B)", f"{p_a_and_b:.4f}")
    st.metric(label, f"{prob:.4f} = {num}/{den}")
    if direction == "P(A | B)":
        st.info(f"""
        **Interpretation**: The probability that A is present GIVEN that B is present is {prob:.3f}.
        - If P(A|B) > P(A), then A and B are positively associated
        - If P(A|B) = P(A), then A and B are independent
        - If P(A|B) < P(A), then A and B are negatively associated
        - Here: P(A|B) = {p_a_given_b:.3f} vs P(A) = {p_a:.3f} → 
          {'positive association' if p_a_given_b > p_a else 'negative association' if p_a_given_b < p_a else 'independence'}
        """)
    with st.expander(" Learn - Conditional Probability", expanded=True):
        st.info("""
        **Conditional Probability**: P(A|B) = P(A∩B) / P(B)
        
        Think of it as: "Among all cases where B is true, what proportion also have A?"
        - The denominator is always the condition (the "given" event)
        - The numerator is the intersection of both events
        - This is the foundation of Bayes' theorem
        - Diagnostic testing: P(Disease|Positive) = P(Positive∩Disease) / P(Positive)
        """)


def _bayesian_updater():
    st.markdown("### Bayesian Updating Table")
    st.info("""
    Bayes' theorem bridges diagnostics, clinical reasoning, and statistical inference.
    Input prevalence (pre-test probability), sensitivity, and specificity to compute
    posterior probabilities.
    """)
    c1, c2 = st.columns([1, 2])
    with c1:
        prevalence = st.slider(
            "Prevalence (Pre-test P(Disease))",
            0.001,
            0.5,
            0.1,
            0.001,
            format="%.3f",
            key="bayes_prev",
            help="Probability of disease before testing",
        )
        sensitivity = st.slider(
            "Sensitivity (P(+|Disease))",
            0.5,
            0.999,
            0.85,
            0.001,
            format="%.3f",
            key="bayes_sens",
            help="True positive rate",
        )
        specificity = st.slider(
            "Specificity (P(-|No Disease))",
            0.5,
            0.999,
            0.90,
            0.001,
            format="%.3f",
            key="bayes_spec",
            help="True negative rate",
        )
        n_pop = st.slider(
            "Population Size (for visualization)", 100, 10000, 1000, 100, key="bayes_n"
        )
    p_d = prevalence
    p_nd = 1 - prevalence
    p_pos_given_d = sensitivity
    p_neg_given_nd = specificity
    p_neg_given_d = 1 - sensitivity
    p_pos_given_nd = 1 - specificity
    p_pos = p_pos_given_d * p_d + p_pos_given_nd * p_nd
    p_neg = p_neg_given_d * p_d + p_neg_given_nd * p_nd
    p_d_given_pos = (p_pos_given_d * p_d) / p_pos if p_pos > 0 else 0
    p_d_given_neg = (p_neg_given_d * p_d) / p_neg if p_neg > 0 else 0
    pre_odds = p_d / p_nd if p_nd > 0 else float("inf")
    lr_plus = sensitivity / (1 - specificity) if specificity < 1 else float("inf")
    lr_minus = (1 - sensitivity) / specificity if specificity > 0 else float("inf")
    post_odds_pos = pre_odds * lr_plus if lr_plus != float("inf") else float("inf")
    post_prob_pos = (
        post_odds_pos / (1 + post_odds_pos) if post_odds_pos != float("inf") else 1.0
    )
    post_odds_neg = pre_odds * lr_minus
    post_prob_neg = post_odds_neg / (1 + post_odds_neg)
    # Population visualization
    tp = int(n_pop * p_d * sensitivity)
    fn = int(n_pop * p_d * (1 - sensitivity))
    fp = int(n_pop * (1 - p_d) * (1 - specificity))
    tn = int(n_pop * (1 - p_d) * specificity)
    total_calc = tp + fn + fp + tn
    scale = n_pop / total_calc if total_calc > 0 else 1
    tp, fn, fp, tn = [int(x * scale) for x in [tp, fn, fp, tn]]
    with c2:
        bayes_df = pd.DataFrame(
            {
                "Step": [
                    "Pre-test probability",
                    "LR+",
                    "LR-",
                    "Post-test prob (if +)",
                    "Post-test prob (if -)",
                ],
                "Value": [
                    f"{prevalence:.4f}",
                    f"{lr_plus:.3f}" if lr_plus != float("inf") else "∞",
                    f"{lr_minus:.3f}" if lr_minus != float("inf") else "∞",
                    f"{post_prob_pos:.4f}",
                    f"{post_prob_neg:.4f}",
                ],
            }
        )
        _apa_table(bayes_df, "Bayesian Updating")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pre-test Prob", f"{prevalence:.3f}")
    with col2:
        st.metric("Post-test (+)", f"{post_prob_pos:.3f}")
    with col3:
        st.metric("Post-test (-)", f"{post_prob_neg:.3f}")
    with col4:
        st.metric("LR+", f"{lr_plus:.2f}" if lr_plus != float("inf") else "∞")
    pop_df = pd.DataFrame(
        {"": ["Test +", "Test -"], "Disease +": [tp, fn], "Disease -": [fp, tn]}
    )
    st.dataframe(pop_df, use_container_width=True)
    st.caption(
        f"Population simulation (n ≈ {n_pop}): TP={tp}, FN={fn}, FP={fp}, TN={tn}"
    )
    fig = go.Figure()
    categories = ["Pre-test", "Post-test (+)", "Post-test (-)"]
    values = [prevalence * 100, post_prob_pos * 100, post_prob_neg * 100]
    colors = ["#4C78A8", "#E45756", "#00CC96"]
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        yaxis=dict(range=[0, 100], title="Probability (%)"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander(" Learn - Bayesian Reasoning", expanded=True):
        st.info(f"""
        **Bayes' Theorem**: P(D|+) = P(+|D) × P(D) / P(+)
        
        **Interpretation:**
        - Pre-test probability (prevalence) = {prevalence:.1%}
        - If test is POSITIVE → probability rises to {post_prob_pos:.1%}
        - If test is NEGATIVE → probability falls to {post_prob_neg:.1%}
        - LR+ = {lr_plus:.2f}: {lr_plus:.1f}× more likely in diseased vs non-diseased
        - **Key insight**: even with good sensitivity/specificity, low prevalence means
          many positives are false positives (the "base rate fallacy")
        """)


# =====================
# 8. APA/JOURNAL EXPORT
# =====================


def apa_export():
    st.header(" APA/Journal Format Export")
    st.info("""
    Generate publication-ready statistical tables formatted for APA journals.
    Select the type of table and configure the formatting options.
    """)

    table_type = st.selectbox(
        "Table Type",
        [
            "Descriptive Statistics",
            "Correlation Matrix",
            "Regression Results",
            "ANOVA Summary",
            "Contingency Table",
            "Effect Size Summary",
        ],
        key="apa_type",
    )

    st.markdown("##### Formatting Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        decimal_places = st.selectbox(
            "Decimal Places", [2, 3, 4], index=1, key="apa_dec"
        )
    with col2:
        include_stars = st.toggle("Significance Stars", True, key="apa_stars")
    with col3:
        include_ci = st.toggle("Include 95% CI", True, key="apa_ci")

    if st.button("Generate APA Table", type="primary", use_container_width=True):
        np.random.seed(42)
        if table_type == "Descriptive Statistics":
            data = _generate_sample_data(100, "Normal")
            df = pd.DataFrame(
                {
                    "Variable": ["Score"],
                    "M": [f"{np.mean(data):.{decimal_places}f}"],
                    "SD": [f"{np.std(data, ddof=1):.{decimal_places}f}"],
                    "SE": [f"{np.std(data, ddof=1)/np.sqrt(100):.{decimal_places}f}"],
                    "95% CI": [
                        f"[{np.mean(data)-1.96*np.std(data, ddof=1)/np.sqrt(100):.{decimal_places}f}, "
                        f"{np.mean(data)+1.96*np.std(data, ddof=1)/np.sqrt(100):.{decimal_places}f}]"
                    ],
                    "Skew": [f"{stats.skew(data):.{decimal_places}f}"],
                    "Kurtosis": [
                        f"{stats.kurtosis(data, fisher=True):.{decimal_places}f}"
                    ],
                }
            )
        elif table_type == "Correlation Matrix":
            n_vars = 4
            data = np.random.multivariate_normal(
                [0] * n_vars, np.eye(n_vars) * 0.5 + 0.5, 100
            )
            corr = np.corrcoef(data.T)
            df = pd.DataFrame(
                corr,
                columns=[f"V{i+1}" for i in range(n_vars)],
                index=[f"V{i+1}" for i in range(n_vars)],
            )
            df = df.map(lambda x: f"{x:.{decimal_places}f}")
        elif table_type == "Regression Results":
            n_pred = 3
            beta = np.random.uniform(-1, 1, n_pred + 1)
            se = np.random.uniform(0.05, 0.3, n_pred + 1)
            t_vals = beta / se
            p_vals = 2 * (1 - stats.t.cdf(abs(t_vals), 96))
            stars = [
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                for p in p_vals
            ]
            data_dict = {
                "Predictor": ["Intercept"] + [f"X{i+1}" for i in range(n_pred)],
                "β": [f"{b:.{decimal_places}f}" for b in beta],
                "SE": [f"{s:.{decimal_places}f}" for s in se],
                "t": [f"{t:.{decimal_places}f}" for t in t_vals],
            }
            if include_stars:
                data_dict[""] = stars
            if include_ci:
                data_dict["95% CI"] = [
                    f"[{b-1.96*s:.{decimal_places}f}, {b+1.96*s:.{decimal_places}f}]"
                    for b, s in zip(beta, se)
                ]
            df = pd.DataFrame(data_dict)
        elif table_type == "ANOVA Summary":
            df = pd.DataFrame(
                {
                    "Source": ["Between Groups", "Within Groups", "Total"],
                    "SS": [
                        f"{np.random.uniform(10, 50):.{decimal_places}f}",
                        f"{np.random.uniform(20, 80):.{decimal_places}f}",
                        "",
                    ],
                    "df": ["2", "57", "59"],
                    "MS": [
                        f"{np.random.uniform(5, 25):.{decimal_places}f}",
                        f"{np.random.uniform(0.5, 2):.{decimal_places}f}",
                        "",
                    ],
                    "F": [f"{np.random.uniform(3, 15):.{decimal_places}f}", "", ""],
                    "p": [
                        f"{np.random.uniform(0.001, 0.05):.{decimal_places}f}",
                        "",
                        "",
                    ],
                    "η²": [f"{np.random.uniform(0.1, 0.5):.{decimal_places}f}", "", ""],
                }
            )
        elif table_type == "Contingency Table":
            table = np.random.randint(5, 50, (2, 3))
            df = pd.DataFrame(
                table, index=["Group 1", "Group 2"], columns=["Cat A", "Cat B", "Cat C"]
            )
        elif table_type == "Effect Size Summary":
            df = pd.DataFrame(
                {
                    "Analysis": ["Group Comparison", "Association", "ANOVA"],
                    "Effect Size": ["Cohen's d", "Cramér's V", "η²"],
                    "Value": [
                        f"{np.random.uniform(0.2, 0.8):.{decimal_places}f}",
                        f"{np.random.uniform(0.1, 0.5):.{decimal_places}f}",
                        f"{np.random.uniform(0.06, 0.14):.{decimal_places}f}",
                    ],
                    "95% CI": [
                        f"[{np.random.uniform(0.1, 0.3):.{decimal_places}f}, "
                        f"{np.random.uniform(0.5, 0.9):.{decimal_places}f}]",
                        "",
                        "",
                    ],
                    "Interpretation": ["Medium", "Medium", "Medium"],
                }
            )
        _apa_table(df, f"APA-Style {table_type}")
        st.code(df.to_string(index=False), language="text")
        st.download_button(
            "Download as CSV",
            df.to_csv(index=False),
            file_name=f"apa_{table_type.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
    else:
        st.info("Configure your table options above and click 'Generate APA Table'.")


# =====================
# MAIN RENDER FUNCTION
# =====================


def render_tabulation():
    st.title(" Tabulation & Cross Tabulation")

    section = st.sidebar.radio(
        "Section",
        [
            "Descriptive Tabulation",
            "Cross-Tabulation",
            "Diagnostic Accuracy Tables",
            "Agreement Tables",
            "Regression Summary Tables",
            "Effect Size Tables",
            "Educational Modules",
            "APA/Journal Export",
        ],
        key="tab_section",
    )

    if section == "Descriptive Tabulation":
        descriptive_tabulation()
    elif section == "Cross-Tabulation":
        cross_tabulation()
    elif section == "Diagnostic Accuracy Tables":
        diagnostic_accuracy()
    elif section == "Agreement Tables":
        agreement_tables()
    elif section == "Regression Summary Tables":
        regression_tables()
    elif section == "Effect Size Tables":
        effect_size_tables()
    elif section == "Educational Modules":
        educational_modules()
    elif section == "APA/Journal Export":
        apa_export()
