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


def data_source_toggle(key_prefix, mode="one_sample"):
    """
    Elegant dual-mode data selector: Simulated (default) vs Uploaded.
    
    This keeps the educational app intact while adding optional real-data capability
    without bloating the codebase.
    
    Parameters
    ----------
    key_prefix : str
        Unique prefix for widget keys (e.g., "ttest_2samp")
    mode : str
        "one_sample", "two_sample", "multi_sample", "paired", "repeated", or "correlation"
        - "repeated": for 3+ measurements on same subjects (Friedman test, etc.)
    
    Returns
    -------
    dict
        - 'mode': 'simulated' or 'uploaded'
        - 'data': None if simulated (use existing sliders), or dict with data if uploaded
        - 'using_uploaded': bool
    """
    with st.expander("📁 Optional: Use Your Own Data", expanded=False):
        st.markdown("""
        **Educational mode is always the default.** Use this to run the test on your own data.
        """)
        
        source = st.radio(
            "Data Source",
            ["Simulated (sliders, for learning)", "Upload CSV/Excel (your data)"],
            key=f"{key_prefix}_datasource",
            index=0,
            label_visibility="collapsed",
        )
    
    if "Simulated" in source:
        return {"mode": "simulated", "data": None, "using_uploaded": False}
    
    uploaded_file = st.file_uploader(
        "Upload your data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key=f"{key_prefix}_file",
    )
    
    if uploaded_file is None:
        st.info("Upload a file to use your own data. The test will use simulated data instead.")
        return {"mode": "simulated", "data": None, "using_uploaded": False}
    
    import pandas as pd
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        data = {"df": df}
        
        if mode == "one_sample":
            value_col = st.selectbox(
                "Select Value Column",
                df.select_dtypes(include=["int64", "float64"]).columns,
                key=f"{key_prefix}_value",
            )
            data["values"] = df[value_col].dropna().values
            
        elif mode == "two_sample":
            layout = st.columns(2)
            with layout[0]:
                group_col = st.selectbox(
                    "Group Column (2 categories)",
                    df.select_dtypes(include=["object", "category", "int64", "bool"]).columns,
                    key=f"{key_prefix}_group",
                )
            with layout[1]:
                value_col = st.selectbox(
                    "Value Column",
                    df.select_dtypes(include=["int64", "float64"]).columns,
                    key=f"{key_prefix}_value",
                )
            
            groups = df.groupby(group_col)[value_col]
            group_names = list(groups.groups.keys())
            if len(group_names) >= 2:
                data["group1"] = groups.get_group(group_names[0]).dropna().values
                data["group2"] = groups.get_group(group_names[1]).dropna().values
                data["group_names"] = group_names[:2]
            else:
                st.error(f"Need at least 2 groups in '{group_col}'. Found: {group_names}")
                return {"mode": "simulated", "data": None, "using_uploaded": False}
                
        elif mode == "multi_sample":
            layout = st.columns(2)
            with layout[0]:
                group_col = st.selectbox(
                    "Group Column",
                    df.select_dtypes(include=["object", "category", "int64", "bool"]).columns,
                    key=f"{key_prefix}_group",
                )
            with layout[1]:
                value_col = st.selectbox(
                    "Value Column",
                    df.select_dtypes(include=["int64", "float64"]).columns,
                    key=f"{key_prefix}_value",
                )
            
            groups = df.groupby(group_col)[value_col]
            data["groups"] = [g.dropna().values for _, g in groups]
            data["group_names"] = list(groups.groups.keys())
            
        elif mode == "paired":
            col_options = list(df.select_dtypes(include=["int64", "float64"]).columns)
            layout = st.columns(2)
            with layout[0]:
                col1 = st.selectbox(
                    "Measurement 1 (e.g., Before)",
                    col_options,
                    key=f"{key_prefix}_col1",
                )
            with layout[1]:
                col2 = st.selectbox(
                    "Measurement 2 (e.g., After)",
                    col_options,
                    key=f"{key_prefix}_col2",
                    index=min(1, len(col_options)-1),
                )
            
            paired_df = df[[col1, col2]].dropna()
            data["values1"] = paired_df[col1].values
            data["values2"] = paired_df[col2].values
            data["col_names"] = [col1, col2]
            
        elif mode == "correlation":
            col_options = list(df.select_dtypes(include=["int64", "float64"]).columns)
            layout = st.columns(2)
            with layout[0]:
                x_col = st.selectbox(
                    "X Variable",
                    col_options,
                    key=f"{key_prefix}_x",
                )
            with layout[1]:
                y_col = st.selectbox(
                    "Y Variable",
                    col_options,
                    key=f"{key_prefix}_y",
                    index=min(1, len(col_options)-1),
                )
            
            corr_df = df[[x_col, y_col]].dropna()
            data["x"] = corr_df[x_col].values
            data["y"] = corr_df[y_col].values
            data["col_names"] = [x_col, y_col]
            
        elif mode == "repeated":
            col_options = list(df.select_dtypes(include=["int64", "float64"]).columns)
            selected_cols = st.multiselect(
                "Select measurement columns (3+ time points/conditions for same subjects)",
                col_options,
                default=col_options[:min(3, len(col_options))],
                key=f"{key_prefix}_repeated_cols",
            )
            
            if len(selected_cols) < 3:
                st.error(f"Friedman Test requires at least 3 repeated measurements. You selected {len(selected_cols)}.")
                return {"mode": "simulated", "data": None, "using_uploaded": False}
            
            repeated_df = df[selected_cols].dropna()
            data["measurements"] = [repeated_df[col].values for col in selected_cols]
            data["col_names"] = selected_cols
        
        return {"mode": "uploaded", "data": data, "using_uploaded": True}
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return {"mode": "simulated", "data": None, "using_uploaded": False}
