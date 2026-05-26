import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as sp_stats
from scipy.special import gammaln, erfinv
from plotly.subplots import make_subplots

_rng = np.random.default_rng(42)


# =========================
# HELPERS
# =========================


def _step_header(step_num, title):
    st.markdown(f"**Step {step_num}: {title}**")


def _step_formula(formula):
    st.latex(formula)


def _step_result(text):
    st.markdown(f"> {text}")


def _section(title):
    st.subheader(title, divider="orange")


def _interpret_card(title, body):
    with st.expander(f"{title}", expanded=False):
        st.markdown(body)


def _apa_table(df, title="Table"):
    st.markdown(f"**{title}**")
    if isinstance(df, pd.DataFrame):
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
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.table(df)


def _qq_plot(data, title="Q-Q Plot"):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    n = len(data)
    theoretical = np.sort(sp_stats.norm.ppf(np.linspace(0.005, 0.995, n)))
    sample = np.sort(data)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sample,
            mode="markers",
            marker=dict(color="#4C78A8", size=5),
            name="Observed",
        )
    )
    min_val = min(theoretical.min(), sample.min())
    max_val = max(theoretical.max(), sample.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#E45756", dash="dash"),
            name="Reference",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _histogram_with_normal(data, title="Histogram with Normal Curve"):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    mu, sigma = np.mean(data), np.std(data, ddof=1)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=min(30, len(data) // 5),
            marker_color="#4C78A8",
            opacity=0.7,
            name="Observed",
            histnorm="probability density",
        )
    )
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=sp_stats.norm.pdf(x_range, mu, sigma),
            mode="lines",
            line=dict(color="#E45756", width=2.5),
            name="Normal Fit",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Value",
        yaxis_title="Density",
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _boxplot_groups(data_dict, title="Group Comparison"):
    fig = go.Figure()
    for i, (name, vals) in enumerate(data_dict.items()):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        fig.add_trace(
            go.Box(
                y=vals,
                name=name,
                marker_color=px.colors.qualitative.Plotly[i % 10],
                boxmean=True,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _residuals_plot(fitted, residuals, title="Residuals vs Fitted"):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            marker=dict(color="#4C78A8", size=6),
            name="Residuals",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#E45756")
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# =========================
# NORMALITY TESTS
# =========================


def _diag_shapiro_wilk():
    _section("Shapiro-Wilk Test for Normality")

    st.markdown("""
    **Objective:** Tests whether a sample comes from a normally distributed population.
    The null hypothesis is that the data follow a normal distribution.
    """)

    st.latex(
        r"W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}"
    )

    st.markdown("""
    Where $x_{(i)}$ are the ordered sample values, $a_i$ are the Shapiro-Wilk coefficients
    (derived from expected values of order statistics of a normal distribution), and
    $\\bar{x}$ is the sample mean.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 8, 200, 50, key="sw_n")
    with c2:
        dist = st.selectbox(
            "Distribution",
            [
                "Normal",
                "Skewed (Right)",
                "Skewed (Left)",
                "Bimodal",
                "Uniform",
                "t (df=3)",
                "Exponential",
            ],
            key="sw_dist",
        )
    with c3:
        noise = st.slider("Noise Level", 0.0, 2.0, 1.0, 0.1, key="sw_noise")

    _rng = np.random.default_rng(42)
    if dist == "Normal":
        data = _rng.normal(0, noise, n)
    elif dist == "Skewed (Right)":
        data = _rng.gamma(2, noise, n)
    elif dist == "Skewed (Left)":
        data = -_rng.gamma(2, noise, n)
    elif dist == "Bimodal":
        half = n // 2
        data = np.concatenate(
            [
                _rng.normal(-1.5, noise * 0.6, half),
                _rng.normal(1.5, noise * 0.6, n - half),
            ]
        )
    elif dist == "Uniform":
        data = _rng.uniform(-1.73 * noise, 1.73 * noise, n)
    elif dist == "t (df=3)":
        data = _rng.standard_t(3, n) * noise
    else:
        data = _rng.exponential(noise, n)

    stat, p = sp_stats.shapiro(data)
    stat_val = float(stat)
    p_val = float(p)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("W Statistic", f"{stat_val:.5f}")
    col_b.metric("p-value", f"{p_val:.5f}")
    col_c.metric("n", str(n))

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="sw_alpha"
    )
    conclusion = (
        "Reject H₀ — data are not normal"
        if p_val < alpha
        else "Fail to reject H₀ — no evidence against normality"
    )
    st.success(f"**Conclusion:** {conclusion} (p = {p_val:.5f}, α = {alpha:.3f})")

    tab1, tab2 = st.tabs(["Histogram", "Q-Q Plot"])
    with tab1:
        st.plotly_chart(
            _histogram_with_normal(data, "Histogram with Normal Curve"),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(_qq_plot(data, "Q-Q Plot"), use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Sort the data")
        sorted_data = np.sort(data)
        _step_formula(
            rf"x_{{(1)}}, x_{{(2)}}, \dots, x_{{(n)}} = {np.array2string(sorted_data, precision=3, suppress_small=True)}"
        )

        _step_header(2, "Compute sample mean")
        xbar = np.mean(data)
        _step_formula(r"\bar{x} = " + f"{xbar:.4f}")

        _step_header(3, "Compute sum of squared deviations")
        ss = np.sum((data - xbar) ** 2)
        _step_formula(r"\sum (x_i - \bar{x})^2 = " + f"{ss:.4f}")

        _step_header(4, "Shapiro-Wilk statistic")
        _step_formula(rf"W = {stat_val:.5f}")

        _step_header(5, "Compute p-value")
        _step_formula(rf"p = {p_val:.5f}")

        _step_header(6, "Decision")
        _decision_op = "<" if p_val < alpha else r"\geq"
        _step_formula(
            rf"\text{{Since }}p = {p_val:.5f} {_decision_op} \alpha = {alpha:.3f},\ \text{{we }}"
            + (r"\text{{reject }}" if p_val < alpha else r"\text{{fail to reject }}")
            + r" \text{the null hypothesis.}"
        )

    _interpret_card(
        "Interpretation",
        f"""
    The Shapiro-Wilk test evaluates normality by comparing the order statistics of the sample
    to expected order statistics of a normal distribution. The W statistic ranges from 0 to 1,
    with values close to 1 indicating normality.

    - **W = {stat_val:.4f}**: This value is {'close to 1, suggesting normality' if stat_val > 0.9 else 'moderately below 1' if stat_val > 0.8 else 'well below 1, indicating departure from normality'}.
    - **p = {p_val:.4f}**: {'Statistically significant' if p_val < 0.05 else 'Not statistically significant'} at the conventional 0.05 level.
    - The Shapiro-Wilk test is recommended for sample sizes up to ~2000 and is generally more powerful than the K-S test for detecting departures from normality.
    """,
    )


def _diag_kolmogorov_smirnov():
    _section("Kolmogorov-Smirnov Test (One-Sample Normality)")

    st.markdown("""
    **Objective:** Tests whether a sample comes from a specified distribution (here, the normal distribution).
    The K-S test compares the empirical cumulative distribution function (ECDF) of the sample
    with the theoretical CDF of the normal distribution.
    """)

    st.latex(r"D_n = \sup_x |F_n(x) - F_0(x)|")

    st.markdown("""
    Where $F_n(x)$ is the ECDF of the sample and $F_0(x)$ is the CDF of the normal distribution.
    The test statistic $D_n$ is the maximum absolute difference between the two CDFs.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 5, 500, 80, key="ks_n")
    with c2:
        dist = st.selectbox(
            "Distribution",
            [
                "Normal",
                "Skewed (Right)",
                "Skewed (Left)",
                "Bimodal",
                "Uniform",
                "t (df=3)",
                "Exponential",
            ],
            key="ks_dist",
        )
    with c3:
        noise = st.slider("Noise Level", 0.0, 2.0, 1.0, 0.1, key="ks_noise")

    _rng = np.random.default_rng(42)
    if dist == "Normal":
        data = _rng.normal(0, noise, n)
    elif dist == "Skewed (Right)":
        data = _rng.gamma(2, noise, n)
    elif dist == "Skewed (Left)":
        data = -_rng.gamma(2, noise, n)
    elif dist == "Bimodal":
        half = n // 2
        data = np.concatenate(
            [
                _rng.normal(-1.5, noise * 0.6, half),
                _rng.normal(1.5, noise * 0.6, n - half),
            ]
        )
    elif dist == "Uniform":
        data = _rng.uniform(-1.73 * noise, 1.73 * noise, n)
    elif dist == "t (df=3)":
        data = _rng.standard_t(3, n) * noise
    else:
        data = _rng.exponential(noise, n)

    mu_est, sigma_est = np.mean(data), np.std(data, ddof=1)
    stat, p = sp_stats.kstest(data, "norm", args=(mu_est, sigma_est))
    stat_val = float(stat)
    p_val = float(p)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("D Statistic", f"{stat_val:.5f}")
    col_b.metric("p-value", f"{p_val:.5f}")
    col_c.metric("n", str(n))

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="ks_alpha"
    )
    conclusion = (
        "Reject H₀ — data are not normal"
        if p_val < alpha
        else "Fail to reject H₀ — no evidence against normality"
    )
    st.success(f"**Conclusion:** {conclusion} (p = {p_val:.5f}, α = {alpha:.3f})")

    sorted_data = np.sort(data)
    ecdf = np.arange(1, n + 1) / n
    theo_cdf = sp_stats.norm.cdf(sorted_data, mu_est, sigma_est)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_data,
            y=ecdf,
            mode="lines+markers",
            line=dict(shape="hv", color="#4C78A8"),
            marker=dict(size=4, color="#4C78A8"),
            name="ECDF",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sorted_data,
            y=theo_cdf,
            mode="lines",
            line=dict(color="#E45756", dash="dash"),
            name="Normal CDF",
        )
    )
    max_idx = np.argmax(np.abs(ecdf - theo_cdf))
    fig.add_trace(
        go.Scatter(
            x=[sorted_data[max_idx], sorted_data[max_idx]],
            y=[ecdf[max_idx], theo_cdf[max_idx]],
            mode="lines",
            line=dict(color="#54A24B", width=2.5),
            name=f"D = {stat_val:.4f}",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Value",
        yaxis_title="Cumulative Probability",
        title="ECDF vs Normal CDF",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Sort the data and compute ECDF")
        _step_result(
            f"Sorted: {np.array2string(sorted_data[:10], precision=3, suppress_small=True)}..."
        )
        _step_header(2, "Compute theoretical CDF under normal assumption")
        _step_formula(
            rf"\hat{{\mu}} = {mu_est:.4f},\quad \hat{{\sigma}} = {sigma_est:.4f}"
        )
        _step_header(3, "Find maximum absolute difference")
        diffs = np.abs(ecdf - theo_cdf)
        max_diff_idx = np.argmax(diffs)
        _step_formula(
            rf"D_n = \max |F_n(x) - F_0(x)| = {diffs[max_diff_idx]:.5f}\ \text{{at }}x = {sorted_data[max_diff_idx]:.4f}"
        )
        _step_header(4, "Compute p-value")
        _step_formula(rf"p = {p_val:.5f}")
        _step_header(5, "Decision")
        _ks_op = "<" if p_val < alpha else r"\geq"
        _step_formula(
            rf"\text{{Since }}p = {p_val:.5f} {_ks_op} \alpha = {alpha:.3f},\ \text{{we }}"
            + (r"\text{{reject }}" if p_val < alpha else r"\text{{fail to reject }}")
            + r" \text{H}_0."
        )

    _interpret_card(
        "Interpretation",
        f"""
    The Kolmogorov-Smirnov test is a nonparametric test that compares the entire empirical
    distribution to a theoretical distribution. It is most sensitive to differences in the
    center of the distribution and less sensitive to differences in the tails.

    - **D = {stat_val:.4f}**: The maximum vertical distance between ECDF and normal CDF.
    - **p = {p_val:.4f}**: {'Statistically significant' if p_val < 0.05 else 'Not statistically significant'}.
    - Note: When parameters are estimated from the data (as done here), the K-S test is conservative
      (Lilliefors correction is more appropriate). Consider using the Shapiro-Wilk or Anderson-Darling
      test as alternatives.
    """,
    )


def _diag_anderson_darling():
    _section("Anderson-Darling Test for Normality")

    st.markdown("""
    **Objective:** Tests whether a sample comes from a normal distribution. The Anderson-Darling
    test is a modification of the K-S test that gives more weight to the tails of the distribution,
    making it more powerful for detecting tail departures.
    """)

    st.latex(
        r"A^2 = -n - \frac{1}{n}\sum_{i=1}^{n} (2i-1)\left[\ln F_0(x_{(i)}) + \ln(1 - F_0(x_{(n+1-i)}))\right]"
    )

    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("Sample Size", 5, 500, 80, key="ad_n")
    with c2:
        dist = st.selectbox(
            "Distribution",
            [
                "Normal",
                "Skewed (Right)",
                "Bimodal",
                "t (df=3)",
                "Exponential",
            ],
            key="ad_dist",
        )

    _rng = np.random.default_rng(42)
    if dist == "Normal":
        data = _rng.normal(0, 1, n)
    elif dist == "Skewed (Right)":
        data = _rng.gamma(2, 1, n)
    elif dist == "Bimodal":
        half = n // 2
        data = np.concatenate(
            [_rng.normal(-1.5, 0.6, half), _rng.normal(1.5, 0.6, n - half)]
        )
    elif dist == "t (df=3)":
        data = _rng.standard_t(3, n)
    else:
        data = _rng.exponential(1, n)

    mu_est, sigma_est = np.mean(data), np.std(data, ddof=1)
    result = sp_stats.anderson(data, dist="norm")
    stat_val = float(result.statistic)
    crit_vals = result.critical_values
    sig_levels = result.significance_level

    col_a, col_b = st.columns(2)
    col_a.metric("A² Statistic", f"{stat_val:.5f}")
    col_b.metric("n", str(n))

    crit_df = pd.DataFrame(
        {
            "Significance Level": [f"{s}%" for s in sig_levels],
            "Critical Value": [f"{c:.4f}" for c in crit_vals],
            "Significant": ["Yes" if stat_val > c else "No" for c in crit_vals],
        }
    )
    _apa_table(crit_df, "Critical Values Table")

    sig_str = f"A² = {stat_val:.4f}"
    for sl, cv in zip(sig_levels, crit_vals):
        if stat_val > cv:
            sig_str += f" > {cv:.4f} (α = {sl}%)"
            break
    else:
        sig_str += f" ≤ {crit_vals[0]:.4f} (not significant at α = 1%)"
    st.success(f"**Conclusion:** {sig_str}")

    tab1, tab2 = st.tabs(["Histogram", "Q-Q Plot"])
    with tab1:
        st.plotly_chart(
            _histogram_with_normal(data, "Histogram with Normal Curve"),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(_qq_plot(data, "Q-Q Plot"), use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        sorted_data = np.sort(data)
        _step_header(1, "Sort the data")
        _step_result(
            f"$x_{{(1)}}, x_{{(2)}}, \\dots, x_{{(n)}}$ = {np.array2string(sorted_data[:10], precision=3, suppress_small=True)}... (n = {n})"
        )

        _step_header(2, "Standardize using estimated parameters")
        _step_result(
            f"$\\hat{{\\mu}} = {mu_est:.4f}$, $\\hat{{\\sigma}} = {sigma_est:.4f}$"
        )
        z_scores = (sorted_data - mu_est) / sigma_est
        _step_result(
            f"First 5 z-scores: {np.array2string(z_scores[:5], precision=3, suppress_small=True)}..."
        )

        _step_header(3, "Compute theoretical CDF values")
        cdf_vals = sp_stats.norm.cdf(sorted_data, mu_est, sigma_est)
        _step_result(
            f"First 5 CDF values: {np.array2string(cdf_vals[:5], precision=4, suppress_small=True)}..."
        )

        _step_header(4, "Compute Anderson-Darling statistic")
        _step_formula(
            r"A^2 = -n - \frac{1}{n}\sum_{i=1}^{n} (2i-1)\left[\ln F_0(x_{(i)}) + \ln(1 - F_0(x_{(n+1-i)}))\right]"
        )
        _step_formula(rf"A^2 = {stat_val:.5f}")

        _step_header(5, "Compare against critical values")
        for sl, cv in zip(sig_levels, crit_vals):
            verdict = "Reject H₀" if stat_val > cv else "Fail to reject H₀"
            _step_formula(
                rf"\text{{At }}\alpha = {sl}\%: A^2 = {stat_val:.4f}\ \text{{vs critical}} = {cv:.4f} \rightarrow \text{{{verdict}}}"
            )

    _interpret_card(
        "Interpretation",
        f"""
    The Anderson-Darling test is an ECDF-based test that emphasizes tail differences. It is
    generally more powerful than the K-S test and is recommended as a good omnibus test for normality.

    - **A² = {stat_val:.4f}**: Larger values indicate greater departure from normality.
    - Compare against critical values in the table above at the desired significance level.
    - The test is particularly sensitive to outliers and heavy-tailed distributions, making it
      an excellent choice for data screening.
    """,
    )


def _diag_jarque_bera():
    _section("Jarque-Bera Test for Normality")

    st.markdown("""
    **Objective:** Tests whether the sample skewness and kurtosis match those of a normal distribution.
    The normal distribution has skewness = 0 and excess kurtosis = 0.
    """)

    st.latex(r"JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)")

    st.markdown("""
    Where $S$ is the sample skewness, $K$ is the sample kurtosis, and $n$ is the sample size.
    Under the null hypothesis of normality, JB follows a $\\chi^2$ distribution with 2 degrees of freedom.
    """)

    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("Sample Size", 10, 1000, 100, key="jb_n")
    with c2:
        dist = st.selectbox(
            "Distribution",
            [
                "Normal",
                "Skewed (Right)",
                "t (df=5)",
                "Uniform",
                "Exponential",
                "Bimodal",
            ],
            key="jb_dist",
        )

    _rng = np.random.default_rng(42)
    if dist == "Normal":
        data = _rng.normal(0, 1, n)
    elif dist == "Skewed (Right)":
        data = _rng.gamma(2, 1, n)
    elif dist == "t (df=5)":
        data = _rng.standard_t(5, n)
    elif dist == "Uniform":
        data = _rng.uniform(-1.73, 1.73, n)
    elif dist == "Exponential":
        data = _rng.exponential(1, n)
    else:
        half = n // 2
        data = np.concatenate(
            [_rng.normal(-1.5, 0.6, half), _rng.normal(1.5, 0.6, n - half)]
        )

    skewness = float(sp_stats.skew(data))
    kurt = float(sp_stats.kurtosis(data, fisher=True))
    n_f = float(n)
    jb_stat = n_f / 6.0 * (skewness**2 + (kurt) ** 2 / 4.0)
    p_val = 1.0 - float(sp_stats.chi2.cdf(jb_stat, 2))

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Skewness", f"{skewness:.4f}")
    col_b.metric("Excess Kurtosis", f"{kurt:.4f}")
    col_c.metric("JB Statistic", f"{jb_stat:.4f}")
    col_d.metric("p-value", f"{p_val:.5f}")

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="jb_alpha"
    )
    conclusion = (
        "Reject H₀ — data are not normal"
        if p_val < alpha
        else "Fail to reject H₀ — no evidence against normality"
    )
    st.success(f"**Conclusion:** {conclusion}")

    st.plotly_chart(
        _histogram_with_normal(data, "Histogram with Normal Curve"),
        use_container_width=True,
    )

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Compute sample skewness")
        _step_formula(
            r"S = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3 = "
            + f"{skewness:.4f}"
        )
        _step_header(2, "Compute sample excess kurtosis")
        _step_formula(
            r"K = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4 - 3 = "
            + f"{kurt:.4f}"
        )
        _step_header(3, "Compute Jarque-Bera statistic")
        _step_formula(
            rf"JB = \frac{{{n_f}}}{{6}}\left({skewness:.4f}^2 + \frac{{({kurt:.4f})^2}}{{4}}\right) = {jb_stat:.4f}"
        )
        _step_header(4, "Compute p-value")
        _step_formula(rf"p = P(\chi^2_{{2}} > {jb_stat:.4f}) = {p_val:.5f}")

    _interpret_card(
        "Interpretation",
        f"""
    The Jarque-Bera test is a goodness-of-fit test that checks if the sample skewness and kurtosis
    match those of a normal distribution (skewness = 0, excess kurtosis = 0).

    - **Skewness = {skewness:.4f}**: {'Close to 0 (symmetric)' if abs(skewness) < 0.5 else 'Moderately skewed' if abs(skewness) < 1 else 'Highly skewed'}.
    - **Excess Kurtosis = {kurt:.4f}**: {'Close to 0 (mesokurtic)' if abs(kurt) < 0.5 else 'Leptokurtic (heavy-tailed)' if kurt > 0.5 else 'Platykurtic (light-tailed)'}.
    - **JB = {jb_stat:.4f}, p = {p_val:.4f}**: {'Significant departure from normality' if p_val < 0.05 else 'No significant departure from normality'}.
    """,
    )


# =========================
# HOMOGENEITY OF VARIANCE TESTS
# =========================


def _diag_levene():
    _section("Levene's Test for Homogeneity of Variance")

    st.markdown("""
    **Objective:** Tests whether $k$ groups have equal variances. Levene's test is robust to
    departures from normality, making it suitable for data that may not be normally distributed.
    The null hypothesis is that all group variances are equal.
    """)

    st.latex(
        r"W = \frac{(N - k)}{(k - 1)} \frac{\sum_{i=1}^{k} n_i (Z_{i\cdot} - Z_{\cdot\cdot})^2}{\sum_{i=1}^{k}\sum_{j=1}^{n_i} (Z_{ij} - Z_{i\cdot})^2}"
    )

    st.markdown("""
    Where $Z_{ij} = |Y_{ij} - \\tilde{Y}_{i\\cdot}|$ (absolute deviation from group median)
    and $N$ is the total sample size across $k$ groups.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        k = st.slider("Number of Groups", 2, 6, 3, key="lev_k")
    with c2:
        n_per = st.slider("Observations per Group", 5, 100, 30, key="lev_n")
    with c3:
        unequal_var = st.checkbox("Unequal Variances", value=False, key="lev_unequal")
    with c4:
        normal_dist = st.checkbox("Normal Distribution", value=True, key="lev_normal")

    _rng = np.random.default_rng(42)
    base_var = 1.0
    groups = {}
    for i in range(k):
        var_i = base_var * (1 + i * 1.5) if unequal_var else base_var
        if normal_dist:
            groups[f"Group {i+1}"] = _rng.normal(i * 0.5, np.sqrt(var_i), n_per)
        else:
            groups[f"Group {i+1}"] = (
                _rng.gamma(2, np.sqrt(var_i) / 1.4, n_per) + i * 0.5
            )

    df_list = [groups[g] for g in groups]
    stat, p = sp_stats.levene(*df_list)
    stat_val = float(stat)
    p_val = float(p)

    col_a, col_b = st.columns(2)
    col_a.metric("Levene Statistic", f"{stat_val:.5f}")
    col_b.metric("p-value", f"{p_val:.5f}")

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="lev_alpha"
    )
    conclusion = (
        "Reject H₀ — variances are not equal"
        if p_val < alpha
        else "Fail to reject H₀ — variances are equal"
    )
    st.success(f"**Conclusion:** {conclusion}")

    st.plotly_chart(
        _boxplot_groups(groups, "Group Comparison — Levene's Test"),
        use_container_width=True,
    )

    var_df = pd.DataFrame(
        {
            "Group": list(groups.keys()),
            "n": [len(groups[g]) for g in groups],
            "Mean": [f"{np.mean(groups[g]):.3f}" for g in groups],
            "Median": [f"{np.median(groups[g]):.3f}" for g in groups],
            "Variance": [f"{np.var(groups[g], ddof=1):.4f}" for g in groups],
            "SD": [f"{np.std(groups[g], ddof=1):.4f}" for g in groups],
        }
    )
    _apa_table(var_df, "Group Descriptive Statistics")

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Compute group medians")
        medians = {g: np.median(groups[g]) for g in groups}
        for g in groups:
            _step_result(f"{g}: median = {medians[g]:.4f}")

        _step_header(2, "Compute absolute deviations from group medians")
        N = k * n_per
        Z_vals = {}
        for g in groups:
            Z_vals[g] = np.abs(groups[g] - medians[g])
            _step_result(f"{g}: mean absolute deviation = {np.mean(Z_vals[g]):.4f}")

        _step_header(3, "Compute group means of absolute deviations")
        Z_means = {g: np.mean(Z_vals[g]) for g in groups}
        grand_mean = np.mean([Z_vals[g] for g in groups])
        _step_result(f"Grand mean of absolute deviations = {grand_mean:.4f}")

        _step_header(4, "Compute between-group and within-group sums of squares")
        ss_between = sum(n_per * (Z_means[g] - grand_mean) ** 2 for g in groups)
        ss_within = sum(np.sum((Z_vals[g] - Z_means[g]) ** 2) for g in groups)
        _step_formula(
            rf"\text{{SS}}_{{\text{{between}}}} = {ss_between:.4f}\quad \text{{(df = {k - 1})}}"
        )
        _step_formula(
            rf"\text{{SS}}_{{\text{{within}}}} = {ss_within:.4f}\quad \text{{(df = {N - k})}}"
        )

        _step_header(5, "Compute Levene's W statistic")
        _step_formula(
            rf"W = \frac{{{N - k}}}{{{k - 1}}} \cdot \frac{{{ss_between:.4f}}}{{{ss_within:.4f}}} = {stat_val:.5f}"
        )
        _step_result(f"$p = {p_val:.5f}$")

    _interpret_card(
        "Interpretation",
        f"""
    Levene's test assesses whether groups have equal variances. It uses absolute deviations from
    group medians, making it robust to non-normality.

    - **W = {stat_val:.4f}, p = {p_val:.4f}**: {'Variances differ significantly across groups' if p_val < alpha else 'No significant evidence of unequal variances'}.
    - Levene's test is preferred over Bartlett's test when data may not be normally distributed.
    - If significant, consider using Welch's ANOVA or non-parametric alternatives that do not assume equal variances.
    """,
    )


def _diag_bartlett():
    _section("Bartlett's Test for Homogeneity of Variance")

    st.markdown("""
    **Objective:** Tests whether $k$ groups have equal variances. Bartlett's test is sensitive to
    departures from normality — if the data are not normally distributed, the test may be
    significant due to non-normality rather than unequal variances.
    """)

    st.latex(
        r"T = \frac{(N - k)\ln s_p^2 - \sum_{i=1}^{k} (n_i - 1)\ln s_i^2}{1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k}\frac{1}{n_i-1} - \frac{1}{N-k}\right)}"
    )

    st.markdown("""
    Where $s_i^2$ is the variance of group $i$, $s_p^2$ is the pooled variance,
    $n_i$ is the size of group $i$, $N$ is the total sample size, and $k$ is the number of groups.
    Under H₀, $T \\sim \\chi^2_{k-1}$.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.slider("Number of Groups", 2, 6, 3, key="bart_k")
    with c2:
        n_per = st.slider("Observations per Group", 5, 100, 30, key="bart_n")
    with c3:
        unequal_var = st.checkbox("Unequal Variances", value=False, key="bart_unequal")

    _rng = np.random.default_rng(42)
    base_var = 1.0
    groups = {}
    for i in range(k):
        var_i = base_var * (1 + i * 1.5) if unequal_var else base_var
        groups[f"Group {i+1}"] = _rng.normal(i * 0.5, np.sqrt(var_i), n_per)

    df_list = [groups[g] for g in groups]
    stat, p = sp_stats.bartlett(*df_list)
    stat_val = float(stat)
    p_val = float(p)

    col_a, col_b = st.columns(2)
    col_a.metric("Bartlett Statistic", f"{stat_val:.5f}")
    col_b.metric("p-value", f"{p_val:.5f}")

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="bart_alpha"
    )
    conclusion = (
        "Reject H₀ — variances are not equal"
        if p_val < alpha
        else "Fail to reject H₀ — variances are equal"
    )
    st.success(f"**Conclusion:** {conclusion}")

    st.plotly_chart(
        _boxplot_groups(groups, "Group Comparison — Bartlett's Test"),
        use_container_width=True,
    )

    var_df = pd.DataFrame(
        {
            "Group": list(groups.keys()),
            "n": [len(groups[g]) for g in groups],
            "Variance": [f"{np.var(groups[g], ddof=1):.4f}" for g in groups],
            "log(Var)": [f"{np.log(np.var(groups[g], ddof=1)):.4f}" for g in groups],
        }
    )
    _apa_table(var_df, "Group Variances")

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Compute group variances")
        group_vars = {g: np.var(groups[g], ddof=1) for g in groups}
        group_ns = {g: len(groups[g]) for g in groups}
        N = sum(group_ns.values())
        for g in groups:
            _step_formula(
                rf"\text{{{g}}}: s^2 = {group_vars[g]:.4f},\ n = {group_ns[g]}"
            )

        _step_header(2, "Compute pooled variance")
        pooled_var = sum((group_ns[g] - 1) * group_vars[g] for g in groups) / (N - k)
        _step_formula(
            rf"s_p^2 = \frac{{\sum (n_i - 1)s_i^2}}{{N - k}} = {pooled_var:.4f}"
        )

        _step_header(3, "Compute the numerator of Bartlett's statistic")
        num = (N - k) * np.log(pooled_var) - sum(
            (group_ns[g] - 1) * np.log(group_vars[g]) for g in groups
        )
        _step_formula(
            rf"\text{{Numerator}} = (N - k)\ln(s_p^2) - \sum (n_i - 1)\ln(s_i^2) = {num:.4f}"
        )

        _step_header(4, "Compute the correction factor")
        denom = 1 + (1 / (3 * (k - 1))) * (
            sum(1 / (group_ns[g] - 1) for g in groups) - 1 / (N - k)
        )
        _step_result(f"Correction factor = {denom:.4f}")

        _step_header(5, "Compute Bartlett's T statistic")
        _step_formula(rf"T = \frac{{{num:.4f}}}{{{denom:.4f}}} = {stat_val:.5f}")
        _step_result(f"$p = P(\\chi^2_{{{k - 1}}} > {stat_val:.4f}) = {p_val:.5f}$")

    _interpret_card(
        "Interpretation",
        f"""
    Bartlett's test compares group variances using a chi-square statistic. It is parametric and
    assumes normal distributions.

    - **T = {stat_val:.4f}, p = {p_val:.4f}**: {'Variances differ significantly' if p_val < alpha else 'No significant evidence of unequal variances'}.
    - Bartlett's test is **sensitive to non-normality**. If the data are not normally distributed,
      use Levene's test instead.
    - If significant, consider using the Welch correction or non-parametric methods.
    """,
    )


def _diag_fligner_killeen():
    _section("Fligner-Killeen Test for Homogeneity of Variance")

    st.markdown("""
    **Objective:** A non-parametric test for homogeneity of variances across groups.
    It is robust to non-normality and is a good alternative to both Levene's and Bartlett's tests.
    """)

    st.latex(r"\chi^2 = \frac{\sum_{i=1}^{k} n_i(\bar{R}_i - \bar{R})^2}{s_R^2}")

    st.markdown("""
    Where $\\bar{R}_i$ is the mean rank of absolute deviations in group $i$, and $s_R^2$ is the
    variance of the ranks. Under H₀, $\\chi^2 \\sim \\chi^2_{k-1}$.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.slider("Number of Groups", 2, 6, 3, key="fk_k")
    with c2:
        n_per = st.slider("Observations per Group", 5, 100, 30, key="fk_n")
    with c3:
        unequal_var = st.checkbox("Unequal Variances", value=False, key="fk_unequal")

    _rng = np.random.default_rng(42)
    base_var = 1.0
    groups = {}
    for i in range(k):
        var_i = base_var * (1 + i * 1.5) if unequal_var else base_var
        groups[f"Group {i+1}"] = _rng.normal(i * 0.5, np.sqrt(var_i), n_per)

    df_list = [groups[g] for g in groups]
    stat, p = sp_stats.fligner(*df_list)
    stat_val = float(stat)
    p_val = float(p)

    col_a, col_b = st.columns(2)
    col_a.metric("Fligner-Killeen χ²", f"{stat_val:.5f}")
    col_b.metric("p-value", f"{p_val:.5f}")

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="fk_alpha"
    )
    conclusion = (
        "Reject H₀ — variances are not equal"
        if p_val < alpha
        else "Fail to reject H₀ — variances are equal"
    )
    st.success(f"**Conclusion:** {conclusion}")

    st.plotly_chart(
        _boxplot_groups(groups, "Group Comparison — Fligner-Killeen"),
        use_container_width=True,
    )

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Compute group medians and absolute deviations")
        medians_fk = {g: np.median(groups[g]) for g in groups}
        all_devs = []
        group_labels = []
        for g in groups:
            devs = np.abs(groups[g] - medians_fk[g])
            all_devs.extend(devs.tolist())
            group_labels.extend([g] * len(devs))
        _step_result(
            f"Computed absolute deviations from group medians (n = {len(all_devs)})"
        )

        _step_header(2, "Rank the absolute deviations")
        ranks = sp_stats.rankdata(all_devs)
        _step_result(f"Mean rank of all deviations = {np.mean(ranks):.2f}")

        _step_header(3, "Compute mean rank per group")
        rank_dict = {g: [] for g in groups}
        for lbl, r in zip(group_labels, ranks):
            rank_dict[lbl].append(r)
        for g in groups:
            _step_result(f"{g}: mean rank = {np.mean(rank_dict[g]):.4f}")

        _step_header(4, "Compute Fligner-Killeen chi-square statistic")
        _step_formula(rf"\chi^2 = {stat_val:.5f},\quad \text{{df}} = {k - 1}")
        _step_formula(rf"p = {p_val:.5f}")

    _interpret_card(
        "Interpretation",
        f"""
    Fligner-Killeen is a non-parametric test for equal variances that uses ranks of absolute
    deviations from group medians.

    - **χ² = {stat_val:.4f}, p = {p_val:.4f}**: {'Significant — variances differ' if p_val < alpha else 'Not significant — variances are equal'}.
    - It is the most robust test for homogeneity of variances, making no normality assumption.
    - Recommended as the default test when screening data for ANOVA assumptions.
    """,
    )


# =========================
# AUTOCORRELATION TEST
# =========================


def _diag_durbin_watson():
    _section("Durbin-Watson Test for Autocorrelation")

    st.markdown("""
    **Objective:** Tests whether the residuals from a regression model exhibit first-order
    autocorrelation. The null hypothesis is that $\\rho = 0$ (no autocorrelation).
    """)

    st.latex(r"d = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}")

    st.markdown("""
    Where $e_t$ are the residuals at time $t$. The DW statistic ranges from 0 to 4:
    - $d \\approx 2$: No autocorrelation
    - $d < 2$: Positive autocorrelation
    - $d > 2$: Negative autocorrelation
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Number of Observations", 10, 200, 50, key="dw_n")
    with c2:
        rho = st.slider(
            "Autocorrelation (ρ)",
            -0.95,
            0.95,
            0.0,
            0.05,
            key="dw_rho",
            help="True autocorrelation in the error term",
        )
    with c3:
        noise = st.slider("Noise SD", 0.1, 5.0, 1.0, 0.1, key="dw_noise")

    _rng = np.random.default_rng(42)
    e = np.zeros(n)
    e[0] = _rng.normal(0, noise)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + _rng.normal(0, noise * np.sqrt(1 - rho**2))

    x = np.linspace(0, 10, n)
    y = 2 + 0.5 * x + e

    dw_num = np.sum((e[1:] - e[:-1]) ** 2)
    dw_den = np.sum(e**2)
    dw = dw_num / dw_den if dw_den > 0 else np.nan

    col_a, col_b = st.columns(2)
    col_a.metric("DW Statistic", f"{dw:.4f}")
    autocorr_type = (
        "Positive" if dw < 1.5 else "Negative" if dw > 2.5 else "No significant"
    )
    col_b.metric("Autocorrelation", autocorr_type)

    cl = 1.5
    cu = 2.5
    st.markdown(f"""
    - $d = {dw:.4f}$
    - Critical bounds (approximate): $d_L = {cl:.2f}$, $d_U = {cu:.2f}$
    - $d < d_L$ → Positive autocorrelation | $d > 4 - d_L$ → Negative autocorrelation
    """)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Time Series Plot", "Residuals")
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=y,
            mode="lines+markers",
            marker=dict(color="#4C78A8", size=4),
            line=dict(color="#4C78A8"),
            name="Observed",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=e,
            mode="lines+markers",
            marker=dict(color="#E45756", size=4),
            line=dict(color="#E45756"),
            name="Residuals",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=2)
    fig.update_layout(
        template="plotly_dark", height=400, margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=e[:-1],
            y=e[1:],
            mode="markers",
            marker=dict(color="#4C78A8", size=6),
            name=f"ρ = {rho:.2f}",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[e.min(), e.max()],
            y=[e.min(), e.max()],
            mode="lines",
            line=dict(color="#E45756", dash="dash"),
            name="Reference (ρ=0)",
        )
    )
    fig2.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="e(t-1)",
        yaxis_title="e(t)",
        title="Residual Autocorrelation Plot",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Obtain residuals from regression")
        _step_result(
            f"First 5 residuals: {np.array2string(e[:5], precision=4, suppress_small=True)}..."
        )
        _step_header(2, "Compute sum of squared differences")
        _step_formula(rf"\sum_{{t=2}}^{{n}} (e_t - e_{{t-1}})^2 = {dw_num:.4f}")
        _step_header(3, "Compute sum of squared residuals")
        _step_formula(rf"\sum_{{t=1}}^{{n}} e_t^2 = {dw_den:.4f}")
        _step_header(4, "Compute Durbin-Watson statistic")
        _step_formula(rf"d = \frac{{{dw_num:.4f}}}{{{dw_den:.4f}}} = {dw:.4f}")

    _interpret_card(
        "Interpretation",
        f"""
    The Durbin-Watson test checks for first-order autocorrelation in regression residuals.

    - **d = {dw:.4f}**: {'Close to 2 — no autocorrelation' if abs(dw - 2) < 0.3 else 'Indicates ' + autocorr_type.lower() + ' autocorrelation'}.
    - Values toward 0 indicate positive autocorrelation (adjacent residuals are similar).
    - Values toward 4 indicate negative autocorrelation (adjacent residuals differ).
    - Autocorrelated residuals violate the independence assumption of OLS regression,
      leading to biased standard errors and invalid inference.
    """,
    )


# =========================
# HETEROSCEDASTICITY TESTS
# =========================


def _diag_breusch_pagan():
    _section("Breusch-Pagan Test for Heteroscedasticity")

    st.markdown("""
    **Objective:** Tests whether the variance of regression residuals is constant (homoscedastic).
    The null hypothesis is that the error variances are all equal (homoscedasticity).
    """)

    st.latex(r"BP = n \cdot R^2_{\text{aux}} \sim \chi^2_{k}")

    st.markdown("""
    Where $R^2_{\\text{aux}}$ is the $R^2$ from a regression of the squared residuals on the
    independent variables. Under H₀, BP follows a $\\chi^2$ distribution with $k$ degrees of
    freedom (where $k$ is the number of predictors).
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 20, 300, 80, key="bp_n")
    with c2:
        het_strength = st.slider(
            "Heteroscedasticity Strength",
            0.0,
            3.0,
            0.0,
            0.1,
            key="bp_het",
            help="0 = homoscedastic, larger = more heteroscedastic",
        )
    with c3:
        error_dist = st.selectbox(
            "Error Distribution", ["Normal", "t (df=4)", "Skewed"], key="bp_err"
        )

    _rng = np.random.default_rng(42)
    x = _rng.uniform(1, 10, n)
    if error_dist == "Normal":
        base_err = _rng.normal(0, 1, n)
    elif error_dist == "t (df=4)":
        base_err = _rng.standard_t(4, n)
    else:
        base_err = _rng.gamma(2, 0.7, n) - 1.4
    sd_mult = 1.0 + het_strength * (x - x.mean()) / x.std()
    sd_mult = np.clip(sd_mult, 0.2, None)
    e = base_err * sd_mult
    y = 2 + 0.8 * x + e

    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    n_f = float(n)
    resid2 = residuals**2
    X_aux = np.column_stack([np.ones(n), x])
    beta_aux = np.linalg.lstsq(X_aux, resid2, rcond=None)[0]
    fitted_aux = X_aux @ beta_aux
    ss_res_aux = np.sum((resid2 - fitted_aux) ** 2)
    ss_tot_aux = np.sum((resid2 - np.mean(resid2)) ** 2)
    r2_aux = 1.0 - ss_res_aux / ss_tot_aux if ss_tot_aux > 0 else 0.0
    bp_stat = n_f * r2_aux
    p_val = 1.0 - float(sp_stats.chi2.cdf(bp_stat, 1))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("BP Statistic", f"{bp_stat:.4f}")
    col_b.metric("p-value", f"{p_val:.5f}")
    col_c.metric("R² (auxiliary)", f"{r2_aux:.4f}")

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="bp_alpha"
    )
    conclusion = (
        "Reject H₀ — heteroscedasticity present"
        if p_val < alpha
        else "Fail to reject H₀ — homoscedastic"
    )
    st.success(f"**Conclusion:** {conclusion}")

    tab1, tab2 = st.tabs(["Residuals vs Fitted", "Scale-Location Plot"])
    with tab1:
        st.plotly_chart(
            _residuals_plot(X @ beta, residuals, "Residuals vs Fitted Values"),
            use_container_width=True,
        )
    with tab2:
        std_resid = residuals / np.std(residuals)
        sqrt_abs_std = np.sqrt(np.abs(std_resid))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=X @ beta,
                y=sqrt_abs_std,
                mode="markers",
                marker=dict(color="#4C78A8", size=6),
            )
        )
        z = np.polyfit(X @ beta, sqrt_abs_std, 1)
        p_line = np.poly1d(z)
        x_sorted = np.sort(X @ beta)
        fig.add_trace(
            go.Scatter(
                x=x_sorted,
                y=p_line(x_sorted),
                mode="lines",
                line=dict(color="#E45756", dash="dash"),
                name="Trend",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Fitted Values",
            yaxis_title="√|Standardized Residuals|",
            title="Scale-Location Plot",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Fit regression model and obtain residuals")
        _step_result(
            f"First 5 residuals: {np.array2string(residuals[:5], precision=4, suppress_small=True)}..."
        )
        _step_header(2, "Square the residuals")
        _step_result(
            f"First 5 squared residuals: {np.array2string(resid2[:5], precision=4, suppress_small=True)}..."
        )
        _step_header(3, "Regress squared residuals on predictors")
        _step_formula(rf"\text{{Auxiliary }}R^2 = {r2_aux:.4f}")
        _step_header(4, "Compute BP statistic")
        _step_formula(rf"\text{{BP}} = {n_f:.0f} \times {r2_aux:.4f} = {bp_stat:.4f}")
        _step_header(5, "Compute p-value")
        _step_formula(rf"p = P(\chi^2_{{1}} > {bp_stat:.4f}) = {p_val:.5f}")

    _interpret_card(
        "Interpretation",
        f"""
    The Breusch-Pagan test detects heteroscedasticity by testing whether the squared residuals
    can be predicted by the independent variables.

    - **BP = {bp_stat:.4f}, p = {p_val:.4f}**: {'Evidence of heteroscedasticity' if p_val < 0.05 else 'No evidence of heteroscedasticity'}.
    - Heteroscedasticity does not bias coefficient estimates but inflates standard errors,
      leading to unreliable hypothesis tests.
    - If detected, consider using heteroscedasticity-consistent (robust) standard errors,
      weighted least squares, or variance-stabilizing transformations (e.g., log).
    """,
    )


def _diag_white():
    _section("White's Test for Heteroscedasticity")

    st.markdown("""
    **Objective:** A more general test for heteroscedasticity than Breusch-Pagan. White's test
    regresses squared residuals on the original predictors, their squares, and cross-products,
    making it sensitive to a wider range of heteroscedasticity patterns.
    """)

    st.latex(r"W = n \cdot R^2_{\text{aux}} \sim \chi^2_{p}")

    st.markdown("""
    Where $p$ is the number of terms in the auxiliary regression (including squares and
    interactions). This tests both pure heteroscedasticity and model misspecification.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 30, 300, 100, key="white_n")
    with c2:
        het_strength = st.slider(
            "Heteroscedasticity Strength", 0.0, 3.0, 0.5, 0.1, key="white_het"
        )
    with c3:
        pattern = st.selectbox(
            "Heteroscedasticity Pattern",
            ["Linear", "Quadratic", "Exponential"],
            key="white_pattern",
        )

    _rng = np.random.default_rng(42)
    x = _rng.uniform(1, 10, n)
    if pattern == "Linear":
        sd_mult = 1.0 + het_strength * (x - x.mean()) / x.std()
    elif pattern == "Quadratic":
        sd_mult = 1.0 + het_strength * ((x - x.mean()) / x.std()) ** 2
    else:
        sd_mult = np.exp(het_strength * (x - x.mean()) / x.std() / 2)
    sd_mult = np.clip(sd_mult, 0.2, None)
    e = _rng.normal(0, 1, n) * sd_mult
    y = 2 + 0.8 * x + e

    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta
    n_f = float(n)
    resid2 = residuals**2
    x2 = x**2
    X_aux = np.column_stack([np.ones(n), x, x2, x * x2])
    beta_aux = np.linalg.lstsq(X_aux, resid2, rcond=None)[0]
    fitted_aux = X_aux @ beta_aux
    ss_res_aux = np.sum((resid2 - fitted_aux) ** 2)
    ss_tot_aux = np.sum((resid2 - np.mean(resid2)) ** 2)
    r2_aux = 1.0 - ss_res_aux / ss_tot_aux if ss_tot_aux > 0 else 0.0
    p_aux = X_aux.shape[1] - 1
    white_stat = n_f * r2_aux
    p_val = 1.0 - float(sp_stats.chi2.cdf(white_stat, p_aux))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("White Statistic", f"{white_stat:.4f}")
    col_b.metric("p-value", f"{p_val:.5f}")
    col_c.metric("R² (auxiliary)", f"{r2_aux:.4f}")

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="white_alpha"
    )
    conclusion = (
        "Reject H₀ — heteroscedasticity present"
        if p_val < alpha
        else "Fail to reject H₀ — homoscedastic"
    )
    st.success(f"**Conclusion:** {conclusion}")

    st.plotly_chart(
        _residuals_plot(X @ beta, residuals, "Residuals vs Fitted — White's Test"),
        use_container_width=True,
    )

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Fit regression model and obtain residuals")
        _step_result(f"$\\hat{{y}} = {beta[0]:.4f} + {beta[1]:.4f}x$")
        _step_result(
            f"First 5 residuals: {np.array2string(residuals[:5], precision=4, suppress_small=True)}..."
        )

        _step_header(2, "Square the residuals")
        _step_result(
            f"First 5 squared residuals: {np.array2string(resid2[:5], precision=4, suppress_small=True)}..."
        )

        _step_header(
            3, "Construct auxiliary regression with squares and cross-products"
        )
        _step_result(
            f"Predictors in auxiliary model: x, x², x·x² (p = {p_aux} terms excluding intercept)"
        )

        _step_header(4, "Compute R² from auxiliary regression")
        _step_formula(rf"\text{{Auxiliary }}R^2 = {r2_aux:.4f}")

        _step_header(5, "Compute White's statistic")
        _step_formula(rf"W = {n_f:.0f} \times {r2_aux:.4f} = {white_stat:.4f}")
        _step_formula(rf"p = P(\chi^2_{{{p_aux}}} > {white_stat:.4f}) = {p_val:.5f}")

    _interpret_card(
        "Interpretation",
        f"""
    White's test is a general test for heteroscedasticity that does not assume a specific form.

    - **W = {white_stat:.4f}, p = {p_val:.4f}**: {'Heteroscedasticity detected' if p_val < 0.05 else 'No heteroscedasticity detected'}.
    - Unlike the Breusch-Pagan test, White's test includes squared and cross-product terms,
      making it sensitive to nonlinear heteroscedasticity patterns.
    - White's test uses more degrees of freedom, which can reduce power in small samples.
    - A significant result may also indicate model misspecification (not just heteroscedasticity).
    """,
    )


# =========================
# OUTLIER DETECTION TESTS
# =========================


def _diag_grubbs():
    _section("Grubbs' Test for Outliers")

    st.markdown("""
    **Objective:** Tests for a single outlier in a univariate dataset assuming normality.
    The null hypothesis is that there are no outliers in the data.
    """)

    st.latex(r"G = \frac{\max|x_i - \bar{x}|}{s}")

    st.markdown("""
    Where $\\bar{x}$ is the sample mean and $s$ is the sample standard deviation.
    The test identifies the most extreme value as a potential outlier.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = st.slider("Sample Size", 6, 100, 30, key="grubbs_n")
    with c2:
        outlier_mag = st.slider(
            "Outlier Magnitude (SDs)",
            2.0,
            6.0,
            4.0,
            0.1,
            key="grubbs_mag",
            help="How many SDs away from the mean the outlier is placed",
        )
    with c3:
        outlier_side = st.selectbox(
            "Outlier Side", ["Two-sided", "Upper", "Lower"], key="grubbs_side"
        )
    with c4:
        include_outlier = st.checkbox(
            "Include Outlier", value=True, key="grubbs_include"
        )

    _rng = np.random.default_rng(42)
    data = _rng.normal(0, 1, n)
    if include_outlier:
        if outlier_side == "Upper":
            data[-1] = outlier_mag
        elif outlier_side == "Lower":
            data[-1] = -outlier_mag
        else:
            data[-1] = outlier_mag * (1 if _rng.uniform() > 0.5 else -1)

    xbar = np.mean(data)
    s = np.std(data, ddof=1)
    deviations = np.abs(data - xbar)
    max_dev_idx = np.argmax(deviations)
    g_stat = deviations[max_dev_idx] / s if s > 0 else 0.0
    g_stat = float(g_stat)
    n_f = float(n)
    t_crit = float(sp_stats.t.ppf(1 - 0.025 / n_f, n_f - 2))
    g_crit = (n_f - 1) / np.sqrt(n_f) * np.sqrt(t_crit**2 / (n_f - 2 + t_crit**2))
    p_val = 2.0 * (
        1.0
        - float(
            sp_stats.t.cdf(
                np.sqrt((n_f - 2) * g_stat**2 / (n_f - 1 - g_stat**2)), n_f - 2
            )
        )
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("G Statistic", f"{g_stat:.4f}")
    col_b.metric("Critical Value", f"{g_crit:.4f}")
    col_c.metric("p-value", f"{p_val:.5f}")

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="grubbs_alpha"
    )
    is_outlier = g_stat > g_crit
    conclusion = (
        f"Outlier detected at x = {data[max_dev_idx]:.4f}"
        if is_outlier
        else "No outlier detected"
    )
    st.success(
        f"**Conclusion:** {conclusion} (G = {g_stat:.4f}, critical = {g_crit:.4f})"
    )

    fig = go.Figure()
    colors = ["#E45756" if i == max_dev_idx else "#4C78A8" for i in range(n)]
    sizes = [10 if i == max_dev_idx else 5 for i in range(n)]
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=data,
            mode="markers",
            marker=dict(color=colors, size=sizes),
            text=[f"Outlier candidate" if i == max_dev_idx else "" for i in range(n)],
        )
    )
    fig.add_hline(
        y=xbar,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Mean = {xbar:.3f}",
    )
    fig.add_hline(
        y=xbar + 2 * s,
        line_dash="dot",
        line_color="#54A24B",
        annotation_text="Mean + 2SD",
    )
    fig.add_hline(
        y=xbar - 2 * s,
        line_dash="dot",
        line_color="#54A24B",
        annotation_text="Mean - 2SD",
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Index",
        yaxis_title="Value",
        title="Grubbs' Test — Data with Outlier Highlighted",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Compute sample mean and SD")
        _step_formula(rf"\bar{{x}} = {xbar:.4f},\quad s = {s:.4f}")
        _step_header(2, "Find the most extreme value")
        _step_formula(
            rf"\max |x_i - \bar{{x}}|\ \text{{occurs at }}x = {data[max_dev_idx]:.4f}"
        )
        _step_header(3, "Compute G statistic")
        _step_formula(
            rf"G = \frac{{|{data[max_dev_idx]:.4f} - {xbar:.4f}|}}{{{s:.4f}}} = {g_stat:.4f}"
        )
        _step_header(4, "Compute critical value")
        _step_formula(rf"G_{{\text{{crit}}}} = {g_crit:.4f}")
        _step_header(5, "Decision")
        _grubbs_op = ">" if is_outlier else r"\leq"
        _step_formula(
            rf"\text{{Since }}G = {g_stat:.4f} {_grubbs_op} {g_crit:.4f},\ \text{{we }}"
            + (r"\text{{reject  }}" if is_outlier else r"\text{{fail to reject  }}")
            + r" \text{H}_0."
        )

    _interpret_card(
        "Interpretation",
        f"""
    Grubbs' test detects one outlier at a time in univariate data assumed to be normally distributed.

    - **G = {g_stat:.4f}**: {'Exceeds' if g_stat > g_crit else 'Within'} the critical value of {g_crit:.4f}.
    - The suspected outlier is at index {max_dev_idx} with value {data[max_dev_idx]:.4f}.
    - Note: Grubbs' test is designed for a **single** outlier. For multiple outliers,
      use Rosner's test or the generalized extreme Studentized deviate (ESD) test.
    - The test assumes normality; verify this assumption before applying.
    """,
    )


def _diag_rosner():
    _section("Rosner's Test (Generalized ESD) for Multiple Outliers")

    st.markdown("""
    **Objective:** Tests for up to $r$ outliers in a univariate dataset. Rosner's test (Generalized
    Extreme Studentized Deviate) iteratively removes the most extreme value and recomputes the
    test statistic, making it suitable for detecting multiple outliers.
    """)

    st.latex(r"R_i = \frac{\max|x_i - \bar{x}^{(i)}|}{s^{(i)}}")

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="rosner_n")
    with c2:
        max_outliers = st.slider("Max Outliers to Test (r)", 1, 10, 3, key="rosner_r")
    with c3:
        outlier_mag = st.slider(
            "Outlier Magnitude (SDs)", 2.0, 6.0, 4.0, 0.1, key="rosner_mag"
        )

    n_outliers = st.slider(
        "Number of True Outliers", 0, min(10, n // 5), 2, key="rosner_n_out"
    )

    _rng = np.random.default_rng(42)
    data = _rng.normal(0, 1, n)
    for i in range(n_outliers):
        data[-(i + 1)] = outlier_mag * (1 if i % 2 == 0 else -1)

    results = []
    work = data.copy()
    n_work = n
    for i in range(min(max_outliers, n_work - 1)):
        xbar_i = np.mean(work)
        s_i = np.std(work, ddof=1)
        devs = np.abs(work - xbar_i)
        max_idx = np.argmax(devs)
        r_i = devs[max_idx] / s_i if s_i > 0 else 0.0
        n_i = float(n_work)
        t_alpha = float(sp_stats.t.ppf(1 - 0.025 / (n_i - i), n_i - i - 1))
        lambda_i = (
            (n_i - i - 1) * t_alpha / np.sqrt((n_i - i - 1 + t_alpha**2) * (n_i - i))
        )
        results.append(
            {
                "Iteration": i + 1,
                "Test Statistic Rᵢ": f"{float(r_i):.4f}",
                "Critical Value λᵢ": f"{float(lambda_i):.4f}",
                "Suspect Value": f"{work[max_idx]:.4f}",
                "Significant": "Yes" if float(r_i) > float(lambda_i) else "No",
            }
        )
        work = np.delete(work, max_idx)
        n_work -= 1

    results_df = pd.DataFrame(results)
    _apa_table(results_df, "Rosner's Test — Iterative Results")

    n_sig = sum(1 for r in results if r["Significant"] == "Yes")
    st.success(
        f"**Conclusion:** {n_sig} outlier{'s' if n_sig != 1 else ''} detected out of {max_outliers} tested."
    )

    fig = go.Figure()
    colors = ["#E45756" if i >= n - n_outliers else "#4C78A8" for i in range(n)]
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=data,
            mode="markers",
            marker=dict(color=colors, size=7),
        )
    )
    fig.add_hline(y=np.mean(data), line_dash="dash", line_color="white")
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Index",
        yaxis_title="Value",
        title="Data with Outliers Highlighted",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Start with full dataset")
        _step_result(
            f"Initial n = {n}, mean(data) = {np.mean(data):.4f}, SD = {np.std(data, ddof=1):.4f}"
        )

        _step_header(2, "Iteratively remove extreme values")
        work = data.copy()
        n_work = n
        for i, r in enumerate(results):
            r_val = float(r["Test Statistic Rᵢ"])
            lam = float(r["Critical Value λᵢ"])
            suspect = float(r["Suspect Value"])
            sig = r["Significant"]
            _step_result(
                f"Iteration {i+1}: R = {r_val:.4f}, λ = {lam:.4f}, suspect = {suspect:.4f}, significant = {sig}"
            )
            max_dev_idx = np.argmax(np.abs(work - np.mean(work)))
            work = np.delete(work, max_dev_idx)
            n_work -= 1

        _step_header(3, "Summarize findings")
        _step_result(
            f"{n_sig} outlier(s) detected out of {max_outliers} tested iterations"
        )

    _interpret_card(
        "Interpretation",
        f"""
    Rosner's test sequentially removes extreme values and tests them using the generalized ESD procedure.

    - Found {n_sig} outlier(s) out of {max_outliers} tested.
    - The test is appropriate when the number of outliers is unknown but bounded.
    - It assumes the bulk of the data (after removing outliers) is approximately normal.
    - Recommended over Grubbs' test when multiple outliers may be present (masking effect).
    """,
    )


def _diag_mahalanobis():
    _section("Mahalanobis Distance for Multivariate Outliers")

    st.markdown("""
    **Objective:** Detects multivariate outliers by measuring the distance of each observation
    from the centroid of the data, accounting for the correlation structure between variables.
    """)

    st.latex(
        r"D^2_i = (\mathbf{x}_i - \bar{\mathbf{x}})^\top \mathbf{S}^{-1} (\mathbf{x}_i - \bar{\mathbf{x}})"
    )

    st.markdown("""
    Where $\\mathbf{x}_i$ is the observation vector, $\\bar{\\mathbf{x}}$ is the mean vector,
    and $\\mathbf{S}$ is the covariance matrix. Under multivariate normality, $D^2_i \\sim \\chi^2_p$
    where $p$ is the number of variables.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 20, 300, 100, key="maha_n")
    with c2:
        p = st.slider("Number of Variables", 2, 6, 3, key="maha_p")
    with c3:
        n_outliers = st.slider("Number of Outliers", 0, 10, 3, key="maha_n_out")

    _rng = np.random.default_rng(42)
    mean_vec = np.zeros(p)
    cov_mat = np.eye(p)
    for i in range(p):
        for j in range(p):
            cov_mat[i, j] = 0.5 ** abs(i - j)
    data = _rng.multivariate_normal(mean_vec, cov_mat, n)
    for i in range(n_outliers):
        data[i] = data[i] + _rng.choice([-1, 1], p) * 3.0

    centroid = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
    centered = data - centroid
    md2 = np.sum(centered @ inv_cov * centered, axis=1)
    p_val_md = 1.0 - sp_stats.chi2.cdf(md2, p)
    alpha_bonf = 0.05 / n
    flags = p_val_md < alpha_bonf

    sig_count = int(np.sum(flags))
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Outliers Detected", str(sig_count))
    col_b.metric("Total Observations", str(n))
    col_c.metric("Threshold (Bonferroni)", f"{alpha_bonf:.5f}")

    md_df = pd.DataFrame(
        {
            "Obs": list(range(1, n + 1)),
            "Mahalanobis D²": [f"{d:.3f}" for d in md2],
            "p-value": [f"{p:.5f}" for p in p_val_md],
            "Outlier?": ["Yes" if f else "No" for f in flags],
        }
    )
    _apa_table(md_df.head(20), "Mahalanobis Distance (first 20 observations)")

    fig = go.Figure()
    colors = ["#E45756" if flags[i] else "#4C78A8" for i in range(n)]
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=md2,
            mode="markers",
            marker=dict(color=colors, size=7),
            text=[f"D²={md2[i]:.2f}" for i in range(n)],
        )
    )
    chi2_crit = float(sp_stats.chi2.ppf(1 - alpha_bonf, p))
    fig.add_hline(
        y=chi2_crit,
        line_dash="dash",
        line_color="#E45756",
        annotation_text=f"χ²(p) threshold = {chi2_crit:.2f}",
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Observation Index",
        yaxis_title="Mahalanobis D²",
        title="Mahalanobis Distance with Outlier Threshold",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    if p >= 2:
        fig2 = go.Figure()
        colors2 = ["#E45756" if flags[i] else "#4C78A8" for i in range(n)]
        fig2.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode="markers",
                marker=dict(color=colors2, size=7),
                text=[f"D²={md2[i]:.2f}" for i in range(n)],
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=450,
            xaxis_title="Variable 1",
            yaxis_title="Variable 2",
            title="Scatter Plot with Outliers Highlighted",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Compute the centroid (mean vector)")
        _step_result(
            f"Centroid = {np.array2string(centroid, precision=4, suppress_small=True)}"
        )

        _step_header(2, "Compute the covariance matrix")
        _step_result(f"Covariance matrix ({p}×{p}):")
        st.latex(
            r"\mathbf{S} = " + np.array2string(cov, precision=4, suppress_small=True)
        )

        _step_header(3, "Invert the covariance matrix")
        _step_formula(
            rf"\text{{Condition number of }}\mathbf{{S}} = {np.linalg.cond(cov):.2f}"
        )

        _step_header(4, "Compute D² for each observation")
        _step_formula(
            r"D^2_i = (\mathbf{x}_i - \bar{\mathbf{x}})^\top \mathbf{S}^{-1} (\mathbf{x}_i - \bar{\mathbf{x}})"
        )
        _step_formula(
            rf"\text{{Range of }}D^2\text{{ values: }}[{md2.min():.3f},\ {md2.max():.3f}]"
        )

        _step_header(5, "Determine outlier threshold using Bonferroni correction")
        chi2_crit = float(sp_stats.chi2.ppf(1 - alpha_bonf, p))
        _step_formula(
            rf"\text{{Bonferroni }}\alpha = {alpha_bonf:.5f},\ \chi^2(p)\ \text{{threshold}} = {chi2_crit:.2f}"
        )
        _step_result(f"Observations exceeding threshold: {sig_count} of {n}")

    _interpret_card(
        "Interpretation",
        f"""
    Mahalanobis distance measures how many standard deviations away an observation is from the
    centroid of the data, accounting for the correlation between variables.

    - **{sig_count}** outlier(s) detected using Bonferroni-corrected χ² threshold.
    - Observations with large Mahalanobis distances are multivariate outliers, meaning their
      combination of values across all variables is unusual.
    - The method assumes multivariate normality. If violated, consider robust Mahalanobis
      distance using the Minimum Covariance Determinant (MCD) estimator.
    """,
    )


def _diag_iqr():
    _section("IQR-Based Outlier Detection")

    st.markdown("""
    **Objective:** Detect outliers using the interquartile range (IQR) method. This is a
    non-parametric approach that does not assume any particular distribution.
    """)

    st.latex(r"\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}")
    st.latex(r"\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}")
    st.latex(
        r"\text{Extreme Fence} = Q_1 - 3.0 \times \text{IQR} \quad \text{or} \quad Q_3 + 3.0 \times \text{IQR}"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 10, 500, 100, key="iqr_n")
    with c2:
        dist = st.selectbox(
            "Distribution",
            [
                "Normal",
                "Skewed (Right)",
                "Exponential",
                "Uniform",
                "Cauchy",
            ],
            key="iqr_dist",
        )
    with c3:
        iqr_mult = st.slider(
            "IQR Multiplier",
            1.0,
            3.5,
            1.5,
            0.1,
            key="iqr_mult",
            help="Standard multiplier is 1.5 (mild outliers), 3.0 (extreme outliers)",
        )

    _rng = np.random.default_rng(42)
    if dist == "Normal":
        data = _rng.normal(0, 1, n)
    elif dist == "Skewed (Right)":
        data = _rng.gamma(2, 1, n)
    elif dist == "Exponential":
        data = _rng.exponential(1, n)
    elif dist == "Uniform":
        data = _rng.uniform(-1.73, 1.73, n)
    else:
        data = _rng.standard_cauchy(n)
        data = data[np.isfinite(data)][:n]

    q1 = float(np.percentile(data, 25))
    q3 = float(np.percentile(data, 75))
    iqr = q3 - q1
    lower_fence = q1 - iqr_mult * iqr
    upper_fence = q3 + iqr_mult * iqr
    outlier_mask = (data < lower_fence) | (data > upper_fence)
    n_out = int(np.sum(outlier_mask))

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Q₁", f"{q1:.4f}")
    col_b.metric("Q₃", f"{q3:.4f}")
    col_c.metric("IQR", f"{iqr:.4f}")
    col_d.metric("Outliers", str(n_out))

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=data,
            name="Data",
            marker_color="#4C78A8",
            boxmean=True,
            jitter=0.3,
            pointpos=-1.8,
        )
    )
    fig.add_hline(
        y=lower_fence,
        line_dash="dash",
        line_color="#E45756",
        annotation_text=f"Lower Fence = {lower_fence:.2f}",
    )
    fig.add_hline(
        y=upper_fence,
        line_dash="dash",
        line_color="#E45756",
        annotation_text=f"Upper Fence = {upper_fence:.2f}",
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title=f"Boxplot — IQR Outlier Detection (multiplier = {iqr_mult})",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    if n_out > 0:
        outlier_vals = data[outlier_mask]
        outlier_df = pd.DataFrame(
            {
                "Index": [int(i) for i in np.where(outlier_mask)[0]],
                "Value": [f"{v:.4f}" for v in outlier_vals],
            }
        )
        _apa_table(outlier_df, f"Detected Outliers ({n_out} total)")
    else:
        st.info("No outliers detected with the current multiplier.")

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Minimum",
                "Q₁",
                "Median",
                "Q₃",
                "Maximum",
                "IQR",
                "Lower Fence",
                "Upper Fence",
            ],
            "Value": [
                f"{data.min():.4f}",
                f"{q1:.4f}",
                f"{np.median(data):.4f}",
                f"{q3:.4f}",
                f"{data.max():.4f}",
                f"{iqr:.4f}",
                f"{lower_fence:.4f}",
                f"{upper_fence:.4f}",
            ],
        }
    )
    _apa_table(summary_df, "Five-Number Summary")

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Sort the data and find quartiles")
        sorted_data = np.sort(data)
        _step_result(
            f"Sorted data (first 5): {np.array2string(sorted_data[:5], precision=3, suppress_small=True)}..."
        )

        _step_header(2, "Compute Q₁, Q₃, and IQR")
        _step_result(f"Q₁ = {q1:.4f} (25th percentile)")
        _step_result(f"Q₃ = {q3:.4f} (75th percentile)")
        _step_formula(rf"\text{{IQR}} = Q_3 - Q_1 = {q3:.4f} - {q1:.4f} = {iqr:.4f}")

        _step_header(3, "Compute fences")
        _step_formula(
            rf"\text{{Lower Fence}} = {q1:.4f} - {iqr_mult} \times {iqr:.4f} = {lower_fence:.4f}"
        )
        _step_formula(
            rf"\text{{Upper Fence}} = {q3:.4f} + {iqr_mult} \times {iqr:.4f} = {upper_fence:.4f}"
        )

        _step_header(4, "Identify outliers")
        _step_result(
            f"Values below {lower_fence:.4f} or above {upper_fence:.4f} are flagged as outliers"
        )
        _step_result(f"Total outliers detected: {n_out}")

    _interpret_card(
        "Interpretation",
        f"""
    The IQR method identifies outliers as values that fall below Q₁ − k × IQR or above Q₃ + k × IQR,
    where k is typically 1.5 (mild outliers) or 3.0 (extreme outliers).

    - **{n_out}** outlier(s) detected with multiplier {iqr_mult}.
    - This method is robust because it depends on quartiles rather than mean and SD.
    - The IQR method tends to flag more points as outliers in skewed distributions.
    - For normally distributed data, approximately 0.7% of points fall outside the 1.5×IQR fences.
    """,
    )


# =========================
# MULTICOLLINEARITY DIAGNOSTICS
# =========================


def _diag_vif():
    _section("Variance Inflation Factor (VIF) — Multicollinearity")

    st.markdown("""
    **Objective:** Detects multicollinearity among predictor variables in regression.
    VIF quantifies how much the variance of a regression coefficient is inflated due to
    correlation with other predictors. VIF > 5 or > 10 indicates problematic multicollinearity.
    """)

    st.latex(r"\text{VIF}_j = \frac{1}{1 - R_j^2}")

    st.markdown("""
    Where $R_j^2$ is the $R^2$ from regressing predictor $j$ on all other predictors.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="vif_n")
    with c2:
        n_pred = st.slider("Number of Predictors", 2, 8, 4, key="vif_n_pred")
    with c3:
        corr_level = st.slider(
            "Correlation Among Predictors",
            0.0,
            0.98,
            0.5,
            0.05,
            key="vif_corr",
            help="Higher = stronger multicollinearity",
        )

    _rng = np.random.default_rng(42)
    preds = np.zeros((n, n_pred))
    preds[:, 0] = _rng.normal(0, 1, n)
    for j in range(1, n_pred):
        shared = _rng.normal(0, 1, n)
        unique = _rng.normal(0, 1, n)
        preds[:, j] = np.sqrt(corr_level) * shared + np.sqrt(1 - corr_level) * unique

    vifs = []
    for j in range(n_pred):
        y_j = preds[:, j]
        X_j = np.column_stack(
            [np.ones(n)] + [preds[:, i] for i in range(n_pred) if i != j]
        )
        beta_j = np.linalg.lstsq(X_j, y_j, rcond=None)[0]
        residuals_j = y_j - X_j @ beta_j
        ss_res = np.sum(residuals_j**2)
        ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
        r2_j = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif_j = 1.0 / (1.0 - r2_j) if r2_j < 0.9999 else 999.0
        vifs.append(
            {
                "Predictor": f"X{j+1}",
                "R²_j": f"{r2_j:.4f}",
                "VIF": f"{vif_j:.3f}" if vif_j < 100 else ">100",
                "Tolerance": f"{1-r2_j:.4f}",
                "Severe Collinearity": (
                    "Yes" if vif_j > 10 else "Moderate" if vif_j > 5 else "No"
                ),
            }
        )

    vif_df = pd.DataFrame(vifs)
    _apa_table(vif_df, "VIF Diagnostic Table")

    max_vif = max(float(r["VIF"]) if r["VIF"] != ">100" else 999.0 for r in vifs)
    n_severe = sum(1 for r in vifs if r["Severe Collinearity"] == "Yes")
    if n_severe > 0:
        st.error(
            f"⚠️ {n_severe} predictor(s) show severe multicollinearity (VIF > 10)."
        )
    else:
        n_mod = sum(1 for r in vifs if r["Severe Collinearity"] == "Moderate")
        if n_mod > 0:
            st.warning(
                f"⚠️ {n_mod} predictor(s) show moderate multicollinearity (VIF > 5)."
            )
        else:
            st.success("✅ No concerning multicollinearity detected.")

    corr_mat = np.corrcoef(preds, rowvar=False)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_mat,
            x=[f"X{i+1}" for i in range(n_pred)],
            y=[f"X{i+1}" for i in range(n_pred)],
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr_mat, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title="Correlation Matrix of Predictors",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        for j in range(min(2, n_pred)):
            _step_header(j + 1, f"Regress X{j+1} on remaining predictors")
            r2_j = float(vifs[j]["R²_j"])
            vif_j_val = float(vifs[j]["VIF"]) if vifs[j]["VIF"] != ">100" else 999.0
            _step_formula(rf"R^2 = {r2_j:.4f}")
            _step_formula(
                rf"\text{{VIF}}_{{{j+1}}} = \frac{{1}}{{1 - {r2_j:.4f}}} = {vif_j_val:.3f}"
            )
            _step_formula(rf"\text{{Tolerance}} = 1 - R^2 = {1-r2_j:.4f}")

    _interpret_card(
        "Interpretation",
        f"""
    VIF measures how much the variance of a coefficient is inflated due to collinearity.

    - **VIF = 1**: No correlation with other predictors.
    - **VIF > 5**: Moderate multicollinearity (investigate further).
    - **VIF > 10**: Severe multicollinearity (requires remediation).
    - Tolerance = 1/VIF, the proportion of variance not shared with other predictors.
    - Remedies: remove correlated predictors, use ridge regression, PCA, or collect more data.
    """,
    )


def _diag_condition_index():
    _section("Condition Index — Multicollinearity Diagnostic")

    st.markdown("""
    **Objective:** Assess multicollinearity by examining the condition indices derived from
    singular value decomposition (SVD) of the design matrix. Condition indices > 30 indicate
    moderate multicollinearity; > 100 indicates severe multicollinearity.
    """)

    st.latex(r"\kappa_j = \frac{\lambda_{\max}}{\lambda_j}")

    st.markdown("""
    Where $\\lambda_j$ is the $j$-th eigenvalue of $\\mathbf{X}^\\top\\mathbf{X}$.
    The condition number $\\kappa_{\\max} = \\lambda_{\\max} / \\lambda_{\\min}$ is the most
    commonly reported index.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="ci_n")
    with c2:
        n_pred = st.slider("Number of Predictors", 2, 8, 4, key="ci_n_pred")
    with c3:
        corr_level = st.slider(
            "Correlation Among Predictors", 0.0, 0.98, 0.7, 0.05, key="ci_corr"
        )

    _rng = np.random.default_rng(42)
    preds = np.zeros((n, n_pred))
    preds[:, 0] = _rng.normal(0, 1, n)
    for j in range(1, n_pred):
        shared = _rng.normal(0, 1, n)
        unique = _rng.normal(0, 1, n)
        preds[:, j] = np.sqrt(corr_level) * shared + np.sqrt(1 - corr_level) * unique

    X_design = np.column_stack([np.ones(n)] + [preds[:, i] for i in range(n_pred)])
    _, s, _ = np.linalg.svd(X_design, full_matrices=False)
    max_s = s[0]
    condition_indices = [max_s / s[j] for j in range(len(s))]

    ci_df = pd.DataFrame(
        {
            "Dimension": list(range(1, len(s) + 1)),
            "Singular Value": [f"{sv:.4f}" for sv in s],
            "Condition Index": [f"{ci:.2f}" for ci in condition_indices],
            "Assessment": [
                (
                    "Good"
                    if ci < 10
                    else "Moderate" if ci < 30 else "Strong" if ci < 100 else "Severe"
                )
                for ci in condition_indices
            ],
        }
    )
    _apa_table(ci_df, "Condition Index Diagnostics")

    max_ci = condition_indices[-1]
    if max_ci > 100:
        st.error(
            f"⚠️ Condition number = {max_ci:.2f} — Severe multicollinearity detected."
        )
    elif max_ci > 30:
        st.warning(
            f"⚠️ Condition number = {max_ci:.2f} — Moderate-to-strong multicollinearity."
        )
    else:
        st.success(
            f"✅ Condition number = {max_ci:.2f} — No concerning multicollinearity."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(s) + 1)),
            y=condition_indices,
            marker_color="#4C78A8",
            text=[f"{ci:.1f}" for ci in condition_indices],
        )
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color="#F5A623", annotation_text="Moderate (30)"
    )
    fig.add_hline(
        y=100, line_dash="dash", line_color="#E45756", annotation_text="Severe (100)"
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Dimension",
        yaxis_title="Condition Index",
        title="Condition Indices by Dimension",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Construct the design matrix")
        _step_result(
            f"Design matrix X: {n} observations × {n_pred + 1} columns (intercept + {n_pred} predictors)"
        )

        _step_header(2, "Perform singular value decomposition (SVD)")
        _step_result(
            f"Singular values: {np.array2string(s, precision=4, suppress_small=True)}"
        )

        _step_header(3, "Compute condition indices")
        _step_formula(r"\kappa_j = \frac{\lambda_{\max}}{\lambda_j}")
        max_s = s[0]
        for j, sv in enumerate(s):
            ci = max_s / sv
            _step_formula(
                rf"\kappa_{{{j+1}}} = \frac{{{max_s:.4f}}}{{{sv:.4f}}} = {ci:.2f}"
            )

        _step_header(4, "Interpret the maximum condition number")
        _step_formula(
            rf"\text{{Condition number }}\kappa_{{\text{{max}}}} = \kappa_{{{len(s)}}} = {max_ci:.2f}"
        )
        if max_ci > 100:
            _step_formula(r"\kappa > 100:\ \text{Severe multicollinearity}")
        elif max_ci > 30:
            _step_formula(
                r"30 < \kappa \leq 100:\ \text{Moderate-to-strong multicollinearity}"
            )
        else:
            _step_formula(r"\kappa \leq 30:\ \text{No concerning multicollinearity}")

    _interpret_card(
        "Interpretation",
        f"""
    Condition indices are derived from the singular value decomposition of the design matrix.

    - **Condition Index < 10**: No collinearity concern.
    - **10 ≤ CI < 30**: Moderate collinearity.
    - **30 ≤ CI < 100**: Strong collinearity, investigate further.
    - **CI ≥ 100**: Severe collinearity requiring remediation.
    - The condition number {max_ci:.2f} indicates {ci_df.iloc[-1]['Assessment'].lower()} multicollinearity.
    - Use alongside VIF for a comprehensive multicollinearity assessment.
    """,
    )


# =========================
# INFLUENCE DIAGNOSTICS
# =========================


def _diag_cooks_distance():
    _section("Cook's Distance — Influence Diagnostics")

    st.markdown("""
    **Objective:** Identify influential observations in a regression model. Cook's distance
    measures the change in all fitted values when an observation is deleted.
    $$D_i = \\frac{\\sum_{j=1}^{n} (\\hat{Y}_j - \\hat{Y}_{j(i)})^2}{p \\cdot MSE}$$
    Where $\\hat{Y}_{j(i)}$ is the fitted value when observation $i$ is omitted,
    $p$ is the number of parameters, and $MSE$ is the mean squared error.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="cook_n")
    with c2:
        influence_type = st.selectbox(
            "Influence Type",
            [
                "No influential points",
                "One high-leverage point",
                "One outlier in Y",
                "High leverage + outlier",
            ],
            key="cook_type",
        )
    with c3:
        influence_strength = st.slider(
            "Influence Strength", 1.0, 5.0, 3.0, 0.5, key="cook_strength"
        )

    _rng = np.random.default_rng(42)
    x = _rng.uniform(1, 10, n)
    y = 2 + 0.8 * x + _rng.normal(0, 1, n)

    if influence_type == "One high-leverage point":
        x[-1] = 15 + influence_strength
        y[-1] = 2 + 0.8 * x[-1] + _rng.normal(0, 1)
    elif influence_type == "One outlier in Y":
        y[-1] = y[-1] + influence_strength * 3
    elif influence_type == "High leverage + outlier":
        x[-1] = 15 + influence_strength
        y[-1] = 2 + 0.8 * 5 + _rng.normal(0, 1) - influence_strength * 2

    X = np.column_stack([np.ones(n), x])
    beta_all = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted_all = X @ beta_all
    residuals = y - fitted_all
    mse = np.sum(residuals**2) / (n - 2)

    cooks_d = np.zeros(n)
    hat = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(hat)
    for i in range(n):
        cooks_d[i] = residuals[i] ** 2 * h[i] / (2 * mse * (1 - h[i]) ** 2)

    threshold = 4.0 / n
    influential = cooks_d > threshold
    n_inf = int(np.sum(influential))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Max Cook's D", f"{float(cooks_d.max()):.4f}")
    col_b.metric("Threshold (4/n)", f"{threshold:.4f}")
    col_c.metric("Influential Points", str(n_inf))

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Scatter with Regression", "Cook's Distance")
    )
    x_range = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    y_pred = beta_all[0] + beta_all[1] * x_range
    colors = ["#E45756" if influential[i] else "#4C78A8" for i in range(n)]
    sizes = [10 if influential[i] else 5 for i in range(n)]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color=colors, size=sizes),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_pred,
            mode="lines",
            line=dict(color="#F5A623", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=np.arange(n),
            y=cooks_d,
            marker_color=["#E45756" if influential[i] else "#4C78A8" for i in range(n)],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#E45756",
        annotation_text=f"4/n = {threshold:.3f}",
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_dark", height=400, margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    cooks_df = pd.DataFrame(
        {
            "Obs": list(range(1, n + 1)),
            "Cook's D": [f"{d:.4f}" for d in cooks_d],
            "Leverage (h)": [f"{h_i:.4f}" for h_i in h],
            "Standardized Residual": [f"{r / np.sqrt(mse):.3f}" for r in residuals],
            "Influential?": ["Yes" if f else "No" for f in influential],
        }
    )
    _apa_table(cooks_df.head(20), "Cook's Distance (first 20 observations)")

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Fit the full regression model")
        _step_formula(rf"\hat{{y}} = {beta_all[0]:.4f} + {beta_all[1]:.4f}x")
        _step_formula(rf"\text{{MSE}} = {mse:.4f},\quad \text{{df}} = {n - 2}")

        _step_header(2, "Compute leverage values (hat matrix diagonal)")
        _step_formula(
            rf"\text{{Range of }}h_i:\ [{h.min():.4f},\ {h.max():.4f}],\ \text{{mean }}\bar{{h}} = {h.mean():.4f}"
        )

        _step_header(3, "Compute Cook's distance for each observation")
        _step_formula(
            r"D_i = \frac{e_i^2}{p \cdot \text{MSE}} \cdot \frac{h_i}{(1 - h_i)^2}"
        )
        max_d_idx = int(np.argmax(cooks_d))
        _step_formula(
            rf"\text{{Observation }}{max_d_idx + 1}\ \text{{has highest }}D = {float(cooks_d[max_d_idx]):.4f}"
        )

        _step_header(4, "Compare to threshold")
        _step_formula(rf"\text{{Threshold}} = 4/n = 4/{n} = {threshold:.4f}")
        _step_result(f"**{n_inf}** influential point(s) with Dᵢ > {threshold:.4f}")

    _interpret_card(
        "Interpretation",
        f"""
    Cook's distance measures the combined influence of each observation on all fitted values.

    - **Dᵢ > 4/n**: Conventionally considered influential (n = {n}).
    - **Max D = {float(cooks_d.max()):.4f}**: {'Exceeds' if float(cooks_d.max()) > threshold else 'Below'} the threshold.
    - **{n_inf}** influential point(s) identified.
    - Influential points should be examined carefully — they may indicate data errors,
      a non-representative sample, or important subpopulations.
    - Large Cook's D can result from high leverage, a large residual, or both.
    """,
    )


def _diag_dffits():
    _section("DFFITS — Influence Diagnostics")

    st.markdown("""
    **Objective:** DFFITS measures the difference in fitted values when an observation is deleted,
    scaled by the standard error. It combines leverage and studentized residual information.
    """)

    st.latex(r"\text{DFFITS}_i = t_i \sqrt{\frac{h_i}{1 - h_i}}")

    st.markdown("""
    Where $t_i$ is the internally studentized residual and $h_i$ is the leverage (hat value).
    $|\\text{DFFITS}_i| > 2\\sqrt{(k+1)/(n-k-1)}$ is considered influential.
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="dffits_n")
    with c2:
        influence_type = st.selectbox(
            "Influence Type",
            [
                "No influential points",
                "One high-leverage point",
                "One outlier in Y",
                "High leverage + outlier",
            ],
            key="dffits_type",
        )
    with c3:
        influence_strength = st.slider(
            "Influence Strength", 1.0, 5.0, 3.0, 0.5, key="dffits_strength"
        )

    _rng = np.random.default_rng(42)
    x = _rng.uniform(1, 10, n)
    y = 2 + 0.8 * x + _rng.normal(0, 1, n)

    if influence_type == "One high-leverage point":
        x[-1] = 15 + influence_strength
    elif influence_type == "One outlier in Y":
        y[-1] = y[-1] + influence_strength * 3
    elif influence_type == "High leverage + outlier":
        x[-1] = 15 + influence_strength
        y[-1] = 2 + 0.8 * 5 + _rng.normal(0, 1) - influence_strength * 2

    X = np.column_stack([np.ones(n), x])
    hat = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(hat)
    beta_all = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted_all = X @ beta_all
    residuals = y - fitted_all
    mse = np.sum(residuals**2) / (n - 2)

    dffits_vals = np.zeros(n)
    for i in range(n):
        studentized = residuals[i] / (np.sqrt(mse * (1 - h[i])))
        dffits_vals[i] = (
            float(studentized * np.sqrt(h[i] / (1 - h[i]))) if h[i] < 0.999 else 0.0
        )

    k = 1
    threshold_dffits = 2 * np.sqrt((k + 1) / (n - k - 1))
    influential_dffits = np.abs(dffits_vals) > threshold_dffits
    n_inf_dffits = int(np.sum(influential_dffits))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Max |DFFITS|", f"{float(np.max(np.abs(dffits_vals))):.4f}")
    col_b.metric("Threshold", f"{threshold_dffits:.4f}")
    col_c.metric("Influential Points", str(n_inf_dffits))

    fig = go.Figure()
    colors_dffits = [
        "#E45756" if influential_dffits[i] else "#4C78A8" for i in range(n)
    ]
    sizes_dffits = [10 if influential_dffits[i] else 5 for i in range(n)]
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=dffits_vals,
            mode="markers",
            marker=dict(color=colors_dffits, size=sizes_dffits),
            text=[f"Obs {i+1}: DFFITS={dffits_vals[i]:.3f}" for i in range(n)],
        )
    )
    fig.add_hline(
        y=threshold_dffits,
        line_dash="dash",
        line_color="#E45756",
        annotation_text=f"+ threshold",
    )
    fig.add_hline(
        y=-threshold_dffits,
        line_dash="dash",
        line_color="#E45756",
        annotation_text=f"- threshold",
    )
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Observation Index",
        yaxis_title="DFFITS",
        title="DFFITS Values with Threshold",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    dffits_df = pd.DataFrame(
        {
            "Obs": list(range(1, n + 1)),
            "DFFITS": [f"{d:.4f}" for d in dffits_vals],
            "Leverage (h)": [f"{h_i:.4f}" for h_i in h],
            "Residual": [f"{r:.3f}" for r in residuals],
            "Influential?": ["Yes" if f else "No" for f in influential_dffits],
        }
    )
    _apa_table(dffits_df.head(20), "DFFITS (first 20 observations)")

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Fit the full regression model")
        _step_formula(rf"\hat{{y}} = {beta_all[0]:.4f} + {beta_all[1]:.4f}x")
        _step_formula(rf"\text{{MSE}} = {mse:.4f},\quad k = {k}\ \text{{predictor(s)}}")

        _step_header(2, "Compute leverage and studentized residuals")
        _step_formula(rf"\text{{Range of }}h_i:\ [{h.min():.4f},\ {h.max():.4f}]")
        studentized = np.array(
            [residuals[i] / (np.sqrt(mse * (1 - h[i]))) for i in range(n)]
        )
        _step_formula(
            rf"\text{{Range of studentized residuals: }}[{studentized.min():.3f},\ {studentized.max():.3f}]"
        )

        _step_header(3, "Compute DFFITS for each observation")
        _step_formula(r"\text{DFFITS}_i = t_i \sqrt{\frac{h_i}{1 - h_i}}")
        max_df_idx = int(np.argmax(np.abs(dffits_vals)))
        _step_formula(
            rf"\text{{Observation }}{max_df_idx + 1}\ \text{{has highest }}|\text{{DFFITS}}| = {float(np.abs(dffits_vals[max_df_idx])):.4f}"
        )

        _step_header(4, "Compare to threshold")
        _step_formula(
            rf"\text{{Threshold}} = 2\sqrt{{(k+1)/(n-k-1)}} = 2\sqrt{{({k+1})/({n - k - 1})}} = {threshold_dffits:.4f}"
        )
        _step_result(
            f"**{n_inf_dffits}** influential point(s) with |DFFITS| > {threshold_dffits:.4f}"
        )

    _interpret_card(
        "Interpretation",
        f"""
    DFFITS measures the standardized change in predicted values when an observation is removed.

    - **|DFFITS| > 2√((k+1)/(n-k-1))**
    - **Max |DFFITS| = {float(np.max(np.abs(dffits_vals))):.4f}**: {'Exceeds' if float(np.max(np.abs(dffits_vals))) > threshold_dffits else 'Within'} threshold.
    - **{n_inf_dffits}** influential point(s) identified.
    - DFFITS is similar to Cook's D but uses the studentized residual and a different scaling.
    - Observations with high DFFITS should be examined for data errors or special characteristics.
    """,
    )


def _diag_leverage():
    _section("Leverage (Hat Values) — Influence Diagnostics")

    st.markdown("""
    **Objective:** Identify observations with unusual predictor values (high leverage).
    High-leverage points have the potential to strongly influence the regression, even if their
    residual is small.
    """)

    st.latex(r"h_i = \mathbf{x}_i^\top (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}_i")

    st.markdown("""
    Where $\\mathbf{x}_i$ is the vector of predictor values for observation $i$.
    $$\\bar{h} = \\frac{k+1}{n} \\quad \\text{(average leverage)}$$
    $$\\text{High leverage: } h_i > 2\\bar{h} \\quad \\text{(or } > 3\\bar{h} \\text{ for conservative cutoff)}$$
    """)

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="lev_n")
    with c2:
        n_pred = st.slider("Number of Predictors", 1, 4, 2, key="lev_n_pred")
    with c3:
        leverage_type = st.selectbox(
            "Leverage Type",
            [
                "Normal data",
                "One extreme X",
                "Several extreme X",
                "Cluster of remote points",
            ],
            key="lev_type",
        )

    _rng = np.random.default_rng(42)
    X_mat = np.zeros((n, n_pred))
    for j in range(n_pred):
        X_mat[:, j] = _rng.uniform(1, 10, n)

    if leverage_type == "One extreme X":
        X_mat[-1, :] = 20
    elif leverage_type == "Several extreme X":
        for i in range(min(3, n)):
            X_mat[i, :] = 18 + _rng.uniform(0, 2, n_pred)
    elif leverage_type == "Cluster of remote points":
        for i in range(min(5, n)):
            X_mat[i, :] = _rng.uniform(15, 20, n_pred)

    X_design = np.column_stack([np.ones(n)] + [X_mat[:, j] for j in range(n_pred)])
    hat_mat = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
    h_vals = np.diag(hat_mat)
    avg_h = (n_pred + 1) / n
    high_lev = h_vals > 2 * avg_h
    n_high = int(np.sum(high_lev))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Average Leverage (h̄)", f"{avg_h:.4f}")
    col_b.metric("Max Leverage", f"{float(h_vals.max()):.4f}")
    col_c.metric("High Leverage Points", str(n_high))

    fig = go.Figure()
    colors_lev = ["#E45756" if high_lev[i] else "#4C78A8" for i in range(n)]
    sizes_lev = [10 if high_lev[i] else 5 for i in range(n)]
    fig.add_trace(
        go.Scatter(
            x=np.arange(n),
            y=h_vals,
            mode="markers",
            marker=dict(color=colors_lev, size=sizes_lev),
            text=[f"Obs {i+1}: h={h_vals[i]:.4f}" for i in range(n)],
        )
    )
    fig.add_hline(
        y=avg_h,
        line_dash="dot",
        line_color="#54A24B",
        annotation_text=f"Average = {avg_h:.4f}",
    )
    fig.add_hline(
        y=2 * avg_h,
        line_dash="dash",
        line_color="#E45756",
        annotation_text=f"2×Average = {2*avg_h:.4f}",
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Observation Index",
        yaxis_title="Leverage (h)",
        title="Leverage Values with High-Leverage Threshold",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    lev_df = pd.DataFrame(
        {
            "Obs": list(range(1, n + 1)),
            "Leverage (h)": [f"{h_i:.4f}" for h_i in h_vals],
            "High Leverage?": ["Yes" if f else "No" for f in high_lev],
        }
    )
    _apa_table(lev_df.head(20), "Leverage Values (first 20 observations)")

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Construct the design matrix")
        _step_result(
            f"Design matrix X: {n} observations × {n_pred + 1} columns (intercept + {n_pred} predictors)"
        )

        _step_header(2, "Compute the hat matrix")
        _step_formula(
            r"\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top"
        )
        _step_result(f"H is an {n}×{n} matrix; extracting diagonal elements")

        _step_header(3, "Extract leverage values from the diagonal")
        _step_formula(
            rf"\text{{Range of }}h_i:\ [{h_vals.min():.4f},\ {h_vals.max():.4f}]"
        )

        _step_header(4, "Compute average leverage")
        _step_formula(rf"\bar{{h}} = \frac{{{n_pred} + 1}}{{{n}}} = {avg_h:.4f}")
        _step_formula(
            rf"\text{{High-leverage threshold }} (2\bar{{h}}) = {2 * avg_h:.4f}"
        )
        _step_formula(
            rf"\text{{Conservative threshold }} (3\bar{{h}}) = {3 * avg_h:.4f}"
        )

        _step_header(5, "Identify high-leverage points")
        _step_result(f"**{n_high}** observation(s) exceed hᵢ > 2h̄")

    _interpret_card(
        "Interpretation",
        f"""
    Leverage measures how far an observation's predictor values are from the mean of all predictors.

    - **Average leverage h̄ = {avg_h:.4f}** = (k+1)/n where k = {n_pred} predictors.
    - **hᵢ > 2h̄ ({2*avg_h:.4f})**: Conventionally considered high leverage.
    - **{n_high}** high-leverage point(s) identified.
    - High-leverage points may not be influential if their y-value aligns with the regression line.
    - Combine leverage with residual analysis (Cook's D, DFFITS) for a complete influence assessment.
    - For data quality screening, investigate high-leverage observations for coding errors.
    """,
    )


# =========================
# ADDITIONAL NORMALITY TEST
# =========================


def _diag_dagostino():
    _section("D'Agostino-Pearson K² Test for Normality")

    st.markdown("""
    **Objective:** An omnibus test that combines skewness and kurtosis to assess normality.
    It is a more powerful alternative to the Jarque-Bera test.
    """)

    st.latex(r"K^2 = Z_s^2 + Z_k^2")

    st.markdown("""
    Where $Z_s$ and $Z_k$ are normal approximations of the skewness and kurtosis coefficients.
    Under H₀, $K^2 \\sim \\chi^2_2$.
    """)

    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("Sample Size", 10, 500, 80, key="dag_n")
    with c2:
        dist = st.selectbox(
            "Distribution",
            [
                "Normal",
                "Skewed (Right)",
                "t (df=5)",
                "Exponential",
                "Bimodal",
            ],
            key="dag_dist",
        )

    _rng = np.random.default_rng(42)
    if dist == "Normal":
        data = _rng.normal(0, 1, n)
    elif dist == "Skewed (Right)":
        data = _rng.gamma(2, 1, n)
    elif dist == "t (df=5)":
        data = _rng.standard_t(5, n)
    elif dist == "Exponential":
        data = _rng.exponential(1, n)
    else:
        half = n // 2
        data = np.concatenate(
            [_rng.normal(-1.5, 0.6, half), _rng.normal(1.5, 0.6, n - half)]
        )

    skew = float(sp_stats.skew(data))
    kur = float(sp_stats.kurtosis(data, fisher=True))

    n_f = float(n)
    s = skew
    k = kur

    z_skew = (n_f + 1) * (n_f + 3) / (6 * (n_f - 2)) * s**2
    z_skew = np.sqrt(2 * (n_f + 1) * (n_f + 3) / (6 * (n_f - 2))) * s
    k2_stat = z_skew**2 + (
        (k * np.sqrt((n_f + 1) * (n_f + 3) / (24 * n_f * (n_f - 2)))) ** 2
    )

    p_val_dag = 1.0 - float(sp_stats.chi2.cdf(k2_stat * 0.5, 2))

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Skewness Z", f"{z_skew:.4f}")
    col_b.metric(
        "Kurtosis Z",
        f"{k * np.sqrt((n_f + 1) * (n_f + 3) / (24 * n_f * (n_f - 2))):.4f}",
    )
    col_c.metric("K² Statistic", f"{k2_stat:.4f}")
    col_d.metric("p-value", f"{p_val_dag:.5f}")

    alpha = st.slider(
        "Significance Level (α)", 0.001, 0.10, 0.05, 0.001, key="dag_alpha"
    )
    conclusion = (
        "Reject H₀ — not normal" if p_val_dag < alpha else "Fail to reject H₀ — normal"
    )
    st.success(f"**Conclusion:** {conclusion}")

    st.plotly_chart(
        _histogram_with_normal(data, "Histogram with Normal Curve"),
        use_container_width=True,
    )

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Compute sample skewness")
        _step_formula(
            r"S = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3 = "
            + f"{skew:.4f}"
        )
        _step_result(
            f"Skewness = {skew:.4f} ({'symmetric' if abs(skew) < 0.5 else 'skewed'})"
        )

        _step_header(2, "Compute sample excess kurtosis")
        _step_formula(
            r"K = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4 - 3 = "
            + f"{kur:.4f}"
        )
        _step_result(
            f"Excess kurtosis = {kur:.4f} ({'mesokurtic' if abs(kur) < 0.5 else 'leptokurtic' if kur > 0.5 else 'platykurtic'})"
        )

        _step_header(3, "Transform skewness to a standard normal deviate Zs")
        z_skew_val = float(z_skew)
        _step_formula(rf"Z_s = {z_skew_val:.4f}")

        _step_header(4, "Transform kurtosis to a standard normal deviate Zk")
        z_kurt_val = float(k * np.sqrt((n_f + 1) * (n_f + 3) / (24 * n_f * (n_f - 2))))
        _step_formula(rf"Z_k = {z_kurt_val:.4f}")

        _step_header(5, "Compute K² statistic")
        _step_formula(
            rf"K^2 = Z_s^2 + Z_k^2 = {z_skew_val:.4f}^2 + {z_kurt_val:.4f}^2 = {k2_stat:.4f}"
        )
        _step_formula(rf"p = P(\chi^2_{{2}} > {k2_stat:.4f}) = {p_val_dag:.5f}")

    _interpret_card(
        "Interpretation",
        f"""
    The D'Agostino-Pearson K² test combines information from both skewness and kurtosis.

    - **K² = {k2_stat:.4f}, p = {p_val_dag:.4f}**: {'Significant departure from normality' if p_val_dag < 0.05 else 'No significant departure from normality'}.
    - Skewness Z = {z_skew:.4f}, Kurtosis Z = {k * np.sqrt((n_f + 1) * (n_f + 3) / (24 * n_f * (n_f - 2))):.4f}
    - This test is sensitive to both asymmetric and heavy/light-tailed departures from normality.
    - It is generally more powerful than the Jarque-Bera test, especially for moderate sample sizes.
    """,
    )


# =========================
# COCHRAN'S C TEST
# =========================


def _diag_cochran_c():
    _section("Cochran's C Test for Variance Homogeneity")

    st.markdown("""
    **Objective:** Tests whether the largest variance among groups is significantly larger than
    the others. Suitable when all groups have equal sample sizes.
    """)

    st.latex(r"C = \frac{\max(s_i^2)}{\sum_{i=1}^{k} s_i^2}")

    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.slider("Number of Groups", 2, 8, 4, key="coch_k")
    with c2:
        n_per = st.slider("Observations per Group", 5, 60, 20, key="coch_n")
    with c3:
        unequal_var = st.checkbox("Unequal Variances", value=False, key="coch_unequal")

    _rng = np.random.default_rng(42)
    base_var = 1.0
    groups = {}
    for i in range(k):
        var_i = base_var * (1 + i * 2.0) if unequal_var else base_var
        groups[f"Group {i+1}"] = _rng.normal(i * 0.5, np.sqrt(var_i), n_per)

    variances = [np.var(groups[g], ddof=1) for g in groups]
    c_stat = max(variances) / sum(variances)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("C Statistic", f"{float(c_stat):.4f}")
    col_b.metric("Max Variance", f"{max(variances):.4f}")
    col_c.metric("n per Group", str(n_per))

    st.plotly_chart(
        _boxplot_groups(groups, "Group Comparison — Cochran's C"),
        use_container_width=True,
    )

    var_df = pd.DataFrame(
        {
            "Group": list(groups.keys()),
            "Variance": [f"{v:.4f}" for v in variances],
        }
    )
    _apa_table(var_df, "Group Variances")

    with st.expander("Step-by-Step Calculation", expanded=False):
        _step_header(1, "Compute each group variance")
        for g in groups:
            v = np.var(groups[g], ddof=1)
            _step_formula(rf"\text{{{g}}}:\ s^2 = {v:.4f}")

        _step_header(2, "Identify the maximum variance")
        max_var = max(variances)
        max_group = list(groups.keys())[np.argmax(variances)]
        _step_formula(
            rf"\text{{Maximum variance: {max_group} with }}s^2 = {max_var:.4f}"
        )

        _step_header(3, "Compute sum of all variances")
        total_var_sum = sum(variances)
        _step_formula(rf"\sum s^2 = {total_var_sum:.4f}")

        _step_header(4, "Compute Cochran's C statistic")
        _step_formula(
            rf"C = \frac{{{max_var:.4f}}}{{{total_var_sum:.4f}}} = {float(c_stat):.5f}"
        )

    _interpret_card(
        "Interpretation",
        f"""
    Cochran's C test detects if one group has a significantly larger variance than others.

    - **C = {float(c_stat):.4f}**: Ratio of largest variance to sum of all variances.
    - Larger values indicate that one group dominates the variance structure.
    - The test requires equal sample sizes across groups.
    - If significant, the group with the largest variance may be problematic for ANOVA assumptions.
    """,
    )


# =========================
# MAIN RENDER FUNCTION
# =========================

DIAG_CATEGORIES = {
    "Normality Tests": [
        "Shapiro-Wilk Test",
        "Kolmogorov-Smirnov Test",
        "Anderson-Darling Test",
        "Jarque-Bera Test",
        "D'Agostino-Pearson K² Test",
    ],
    "Homogeneity of Variance Tests": [
        "Levene's Test",
        "Bartlett's Test",
        "Fligner-Killeen Test",
        "Cochran's C Test",
    ],
    "Autocorrelation Tests": [
        "Durbin-Watson Test",
    ],
    "Heteroscedasticity Tests": [
        "Breusch-Pagan Test",
        "White's Test",
    ],
    "Outlier Detection Tests": [
        "Grubbs' Test",
        "Rosner's Test",
        "Mahalanobis Distance",
        "IQR-Based Outlier Detection",
    ],
    "Multicollinearity Diagnostics": [
        "Variance Inflation Factor (VIF)",
        "Condition Index",
    ],
    "Influence Diagnostics": [
        "Cook's Distance",
        "DFFITS",
        "Leverage / Hat Values",
    ],
}

_CATEGORY_LIST = list(DIAG_CATEGORIES.keys())

_DIAG_ROUTER = {
    "Shapiro-Wilk Test": _diag_shapiro_wilk,
    "Kolmogorov-Smirnov Test": _diag_kolmogorov_smirnov,
    "Anderson-Darling Test": _diag_anderson_darling,
    "Jarque-Bera Test": _diag_jarque_bera,
    "D'Agostino-Pearson K² Test": _diag_dagostino,
    "Levene's Test": _diag_levene,
    "Bartlett's Test": _diag_bartlett,
    "Fligner-Killeen Test": _diag_fligner_killeen,
    "Cochran's C Test": _diag_cochran_c,
    "Durbin-Watson Test": _diag_durbin_watson,
    "Breusch-Pagan Test": _diag_breusch_pagan,
    "White's Test": _diag_white,
    "Grubbs' Test": _diag_grubbs,
    "Rosner's Test": _diag_rosner,
    "Mahalanobis Distance": _diag_mahalanobis,
    "IQR-Based Outlier Detection": _diag_iqr,
    "Variance Inflation Factor (VIF)": _diag_vif,
    "Condition Index": _diag_condition_index,
    "Cook's Distance": _diag_cooks_distance,
    "DFFITS": _diag_dffits,
    "Leverage / Hat Values": _diag_leverage,
}


def render_diagnostics():
    st.title("Data Screening & Diagnostics")
    st.markdown("""
    Comprehensive suite of diagnostic tests for evaluating regression assumptions,
    detecting outliers, assessing multicollinearity, and checking data quality.
    """)

    with st.sidebar:
        st.markdown("##### :orange[Diagnostic Category]")
        cat = st.radio(
            "Category",
            _CATEGORY_LIST,
            key="diag_category",
            label_visibility="collapsed",
        )

    cat_descriptions = {
        "Normality Tests": "Normality tests assess whether a sample plausibly came from a normal distribution. "
            "Many parametric procedures (t-tests, ANOVA, linear regression) assume normality of residuals. "
            "These tests vary in sensitivity — the Shapiro-Wilk and Anderson-Darling tests are generally preferred "
            "for their power across a wide range of non-normal alternatives.",
        "Homogeneity of Variance Tests": "These tests check whether multiple groups share a common variance, "
            "a core assumption of ANOVA and pooled-variance t-tests. Levene's and Fligner-Killeen are robust "
            "to non-normality; Bartlett's is more powerful under normality but sensitive to departures from it. "
            "Cochran's C is designed for balanced designs with equal group sizes.",
        "Autocorrelation Tests": "Autocorrelation tests detect correlation between consecutive residuals in "
            "time-ordered data. The Durbin-Watson test targets first-order autocorrelation in regression residuals. "
            "Violations inflate or deflate standard errors and distort hypothesis tests.",
        "Heteroscedasticity Tests": "Heteroscedasticity occurs when residual variance changes across levels of "
            "the predictors, violating the OLS constant-variance assumption. The Breusch-Pagan test detects "
            "linear forms of heteroscedasticity; White's test is more general, capturing non-linear patterns "
            "and potential model misspecification.",
        "Outlier Detection Tests": "Outlier tests identify observations that deviate markedly from the rest "
            "of the data. Grubbs' test detects a single outlier; Rosner's (Generalized ESD) handles multiple "
            "outliers. Mahalanobis distance flags multivariate outliers by accounting for correlations among "
            "variables. The IQR method provides a simple distribution-free heuristic.",
        "Multicollinearity Diagnostics": "Multicollinearity arises when predictors in a regression are highly "
            "correlated, inflating coefficient variances and making estimates unstable. VIF quantifies how much "
            "each coefficient's variance is inflated. The condition index — derived from singular value decomposition "
            "of the design matrix — detects the severity and dimensionality of collinearity.",
        "Influence Diagnostics": "Influence diagnostics identify observations that disproportionately affect "
            "regression estimates. Cook's distance combines leverage and residual magnitude into a single influence "
            "measure. DFFITS captures the standardized change in fitted values when an observation is removed. "
            "Leverage (hat values) flags points with extreme predictor values regardless of their residual.",
    }
    st.markdown(f"_{cat_descriptions[cat]}_")

    tests_in_cat = DIAG_CATEGORIES[cat]
    diag_choice = st.selectbox(
        f"**{cat}** — Select a test",
        tests_in_cat,
        key="diag_selector",
    )

    render_fn = _DIAG_ROUTER.get(diag_choice)
    if render_fn:
        render_fn()
    else:
        st.info("Select a test from the dropdown above.")
