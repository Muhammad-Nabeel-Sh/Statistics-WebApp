import streamlit as st
import numpy as np
from scipy.stats import norm, t, nct


def format_p_value(p, decimals=3):
    """Format p-value according to APA guidelines.

    APA 7th edition:
    - p < .001 for p < 0.001
    - Report to 2-3 decimal places otherwise
    - No leading zero (e.g., .05 not 0.05)
    """
    if p < 0.001:
        return "p < .001"
    elif p < 0.01:
        return f"p = {p:.3f}".replace("0.", ".")
    else:
        rounded = round(p, decimals)
        if rounded == 1.0:
            return "p > .999"
        return f"p = {rounded:.{decimals}f}".replace("0.", ".")


def cohens_d_one_sample_ci(d, n, conf_level=0.95):
    """Calculate 95% CI for Cohen's d (one-sample/paired).

    Uses the non-central t-distribution method.
    Formula based on: Algina & Keselman (2003)
    """
    df = n - 1
    t_stat = d * np.sqrt(n)

    alpha = 1 - conf_level

    def nct_search(t_stat, df, target):
        from scipy.optimize import root_scalar

        def objective(ncp):
            return nct.cdf(t_stat, df, ncp) - target

        try:
            result = root_scalar(
                objective, bracket=[-20, 20], method="bisect"
            )
            return result.root
        except Exception:
            se = np.sqrt((n / (n - 2)) * (1 + d**2 * n / (n - 2)) - d**2)
            z = norm.ppf(1 - alpha / 2)
            return d - z * se if target < 0.5 else d + z * se

    try:
        ncp_lower = nct_search(t_stat, df, 1 - alpha / 2)
        ncp_upper = nct_search(t_stat, df, alpha / 2)
        d_lower = ncp_lower / np.sqrt(n)
        d_upper = ncp_upper / np.sqrt(n)
    except Exception:
        se = np.sqrt(1 / n + d**2 / (2 * n))
        z = norm.ppf(1 - alpha / 2)
        d_lower = d - z * se
        d_upper = d + z * se

    return d_lower, d_upper


def cohens_d_independent_ci(d, n1, n2, conf_level=0.95):
    """Calculate 95% CI for Cohen's d (independent samples).

    Uses the non-central t-distribution approximation.
    """
    df = n1 + n2 - 2
    n_tilde = (n1 * n2) / (n1 + n2)
    t_stat = d * np.sqrt(n_tilde)

    alpha = 1 - conf_level

    try:
        from scipy.optimize import root_scalar

        def objective_lower(ncp):
            return nct.cdf(t_stat, df, ncp) - (1 - alpha / 2)

        def objective_upper(ncp):
            return nct.cdf(t_stat, df, ncp) - (alpha / 2)

        ncp_lower = root_scalar(
            objective_lower, bracket=[-20, 20], method="bisect"
        ).root
        ncp_upper = root_scalar(
            objective_upper, bracket=[-20, 20], method="bisect"
        ).root

        d_lower = ncp_lower / np.sqrt(n_tilde)
        d_upper = ncp_upper / np.sqrt(n_tilde)
    except Exception:
        a = n1 + n2 - 2
        b = (n1 + n2) / (n1 * n2)
        c = d**2 / (2 * (n1 + n2))
        se = np.sqrt(b + c)
        z = norm.ppf(1 - alpha / 2)
        d_lower = d - z * se
        d_upper = d + z * se

    return d_lower, d_upper


def hedges_g(d, n1, n2=None):
    """Convert Cohen's d to Hedges' g (small-sample bias correction).

    For n1 + n2 < 50, Cohen's d overestimates the population effect size.
    Hedges' g applies a correction factor.

    Formula: g = d * (1 - 3 / (4*(n1+n2) - 9)) for independent samples
             g = d * (1 - 3 / (4*(n1) - 5)) for paired/one-sample
    """
    if n2 is None:
        df = n1 - 1
        correction = 1 - 3 / (4 * df - 1) if df > 1 else 1
    else:
        df = n1 + n2 - 2
        correction = 1 - 3 / (4 * df - 1) if df > 1 else 1
    return d * correction


def omega_squared_partial(ss_effect, df_effect, ss_resid, ms_resid, n_total):
    """Calculate partial omega-squared (unbiased effect size for ANOVA).

    Formula: ω²_p = (SS_effect - df_effect * MS_resid) /
                   (SS_effect + (N - df_effect) * MS_resid)

    Partial omega-squared is preferred over partial eta-squared because
    it is less biased in small samples.
    """
    numerator = ss_effect - df_effect * ms_resid
    denominator = ss_effect + (n_total - df_effect) * ms_resid

    if denominator <= 0:
        return 0.0

    omega = numerator / denominator
    return max(0.0, omega)


def format_effect_size_with_ci(est, lower, upper, decimals=2):
    """Format effect size with CI as: 0.50 [0.25, 0.75]"""
    return f"{est:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def st_plot_with_download(fig, key, use_container_width=True, height=None):
    """Display a Plotly figure with download buttons for PNG and SVG.

    Args:
        fig: Plotly figure object
        key: Unique key for streamlit widgets (avoids duplicate IDs)
        use_container_width: Passed to st.plotly_chart
        height: Figure height for download (default: from fig.layout.height)

    Returns:
        The st.plotly_chart result
    """
    chart = st.plotly_chart(fig, use_container_width=use_container_width)

    col1, col2, col3 = st.columns([1, 1, 1])

    png_height = height or fig.layout.height or 500
    png_width = 800 if use_container_width else (fig.layout.width or 700)

    try:
        fig_bytes = fig.to_image(
            format="png", width=png_width, height=png_height, scale=2
        )
        with col1:
            st.download_button(
                label="📥 PNG (300 DPI)",
                data=fig_bytes,
                file_name=f"figure_{key}.png",
                mime="image/png",
                key=f"dl_png_{key}",
                use_container_width=True,
            )
    except Exception as e_png:
        with col1:
            st.info("ℹ️ Install kaleido for PNG export: `pip install kaleido`")

    try:
        import plotly.io as pio
        svg_content = pio.to_image(fig, format="svg")
        with col2:
            st.download_button(
                label="📥 SVG (Vector)",
                data=svg_content,
                file_name=f"figure_{key}.svg",
                mime="image/svg+xml",
                key=f"dl_svg_{key}",
                use_container_width=True,
            )
    except Exception as e_svg:
        with col2:
            st.info("ℹ️ SVG export: try `pip install -U plotly kaleido`")

    with col3:
        with st.expander("💡 Export Tips"):
            st.markdown("""
            - **PNG**: Best for presentations, emails, Word
            - **SVG**: Best for publications (Illustrator/Inkscape)
            - **Built-in camera**: Quick PNG via Plotly's JS renderer
            """)

    return chart


def interpret_cohens_d(d):
    """Text interpretation of Cohen's d effect size.

    Cohen (1988) benchmarks:
    - |d| < 0.2: Trivial / Very small
    - 0.2 ≤ |d| < 0.5: Small
    - 0.5 ≤ |d| < 0.8: Medium
    - |d| ≥ 0.8: Large
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Trivial"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def interpret_eta_squared(eta2):
    """Text interpretation of eta-squared/omega-squared.

    Cohen (1988) for ANOVA:
    - η² < 0.01: Small
    - 0.01 ≤ η² < 0.06: Medium
    - η² ≥ 0.14: Large
    """
    if eta2 < 0.01:
        return "Small"
    elif eta2 < 0.06:
        return "Medium"
    else:
        return "Large"


def interpret_r(r):
    """Text interpretation of correlation coefficient r.

    Cohen (1988):
    - |r| < 0.1: Trivial
    - 0.1 ≤ |r| < 0.3: Small
    - 0.3 ≤ |r| < 0.5: Medium
    - |r| ≥ 0.5: Large
    """
    abs_r = abs(r)
    if abs_r < 0.1:
        return "Trivial"
    elif abs_r < 0.3:
        return "Small"
    elif abs_r < 0.5:
        return "Medium"
    else:
        return "Large"
