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


@register_test("Pearson Correlation")
def render_pearson_correlation(external_data=None):
    st.subheader("Interactive Pearson Correlation")

    st.info("""
    **Pearson Correlation (r)** measures the **linear relationship** between two continuous variables.
    
    - r = +1: perfect positive linear relationship
    - r = -1: perfect negative linear relationship  
    - r = 0: no linear relationship
    
    Assumptions: normality, linearity, homoscedasticity
    """)

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("pearson", mode="correlation")

    # =========================
    # CONTROLS / DATA
    # =========================

    np.random.seed(42)

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
        n = len(x)
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
    else:
        col1, col2 = st.columns(2)

        with col1:
            r = st.slider("Correlation Coefficient (r)", -1.0, 1.0, 0.5, 0.01)

        with col2:
            n = st.slider("Sample Size (n)", 3, 100, 30)

        x = np.random.normal(size=n)
        y = r * x + np.sqrt(1 - r**2) * np.random.normal(size=n)
        col_names = ["X", "Y"]
        from scipy.stats import pearsonr
        _, p = pearsonr(x, y)

    # =========================
    # STATS
    # =========================

    st.latex(rf"""
        r = {r:.2f}
        """)

    st.latex(rf"\text{{{format_p_value(p)}}}")

    # =========================
    # REGRESSION LINE
    # =========================

    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(x, y)
    x_line = np.array([min(x), max(x)])
    y_line = intercept + slope * x_line

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data Points",
            marker=dict(color="rgba(100, 150, 255, 0.7)", size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Regression: y = {intercept:.2f} + {slope:.2f}x",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title=col_names[0],
        yaxis_title=col_names[1],
        title=f"r = {r:.3f}, n = {n}",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # DATA SUMMARY
    # =========================

    st.divider()
    st.subheader("Summary Statistics")

    summary_data = {
        "Variable": col_names,
        "n": [n, n],
        "Mean": [f"{np.mean(x):.3f}", f"{np.mean(y):.3f}"],
        "SD": [f"{np.std(x, ddof=1):.3f}", f"{np.std(y, ddof=1):.3f}"],
    }
    st.table(pd.DataFrame(summary_data))



@register_test("Spearman Rank Correlation")
def render_spearman_rank_correlation(external_data=None):

    from scipy.stats import spearmanr

    st.subheader("Interactive Spearman Rank Correlation")

    st.info("""
    **Spearman's ρ (rho)** is the **nonparametric alternative** to Pearson's r.
    
    Use when:
    - Relationship is **monotonic** but not necessarily linear
    - Data is **ordinal** or **not normally distributed**
    - Measures rank-based association
    """)

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("spearman", mode="correlation")

    # =========================
    # CONTROLS / DATA
    # =========================

    np.random.seed(42)

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
    else:
        direction = st.selectbox(
            "Correlation Direction",
            [
                "Positive",
                "Negative",
            ],
        )

        curve_strength = st.slider(
            "Monotonic Strength",
            0.1,
            3.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
            key="noise_5",
        )

        x = np.linspace(0, 10, 300)
        direction_multiplier = 1 if direction == "Positive" else -1
        y = direction_multiplier * (x**curve_strength) + np.random.normal(0, noise, 300)
        col_names = ["X", "Y"]

    # =========================
    # TEST
    # =========================

    rho, p = spearmanr(x, y)

    st.latex(rf"\rho = {rho:.3f}")
    st.latex(rf"\text{{{format_p_value(p)}}}")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker=dict(color="rgba(100, 200, 150, 0.6)", size=6),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title=col_names[0],
        yaxis_title=col_names[1],
        title=f"Spearman ρ = {rho:.3f}",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # DATA SUMMARY
    # =========================

    st.divider()
    st.subheader("Summary Statistics")

    from scipy.stats import pearsonr
    r_pearson, _ = pearsonr(x, y)

    summary_data = {
        "Metric": [
            "Sample Size (n)",
            "Spearman's ρ",
            "Pearson's r (for comparison)",
            "p-value",
        ],
        "Value": [
            f"{len(x)}",
            f"{rho:.4f}",
            f"{r_pearson:.4f}",
            format_p_value(p),
        ],
    }
    st.table(pd.DataFrame(summary_data))



@register_test("Kendall's Tau-b")
def render_kendall_s_tau_b(external_data=None):

    from scipy.stats import kendalltau

    st.subheader("Interactive Kendall's Tau-b")

    st.info("""
    **Kendall's τ-b (tau-b)** is another **nonparametric** measure of rank correlation.
    
    - Based on **concordant/discordant pairs**
    - Good for **small samples** or when there are **ties**
    - τ-b corrects for ties (unlike the simpler τ-a)
    """)

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("kendall", mode="correlation")

    # =========================
    # CONTROLS / DATA
    # =========================

    np.random.seed(42)

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
    else:
        strength = st.slider("Association Strength", 0.0, 1.0, 0.5, 0.05)

        n = st.slider("Sample Size", 10, 200, 60, key="kendall_s_tau_b_sample_size")

        noise = st.slider("Noise", 0.1, 5.0, 1.0, 0.1, key="kendall_s_tau_b_noise")

        x = np.random.normal(0, 1, n)
        y = strength * x + np.random.normal(0, noise, n)
        col_names = ["X", "Y"]

    # =========================
    # TEST
    # =========================

    tau, p = kendalltau(x, y)

    st.latex(rf"\tau_b = {tau:.3f}")
    st.latex(rf"\text{{{format_p_value(p)}}}")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker=dict(color="rgba(200, 150, 100, 0.6)", size=8),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title=col_names[0],
        yaxis_title=col_names[1],
        title=f"Kendall's τ-b = {tau:.3f}",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # COMPARISON
    # =========================

    st.divider()
    st.subheader("Comparison with Other Correlation Measures")

    from scipy.stats import pearsonr, spearmanr
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)

    comp_data = {
        "Measure": ["Pearson's r", "Spearman's ρ", "Kendall's τ-b"],
        "Value": [f"{r:.4f}", f"{rho:.4f}", f"{tau:.4f}"],
        "Type": ["Linear (parametric)", "Monotonic (nonparametric)", "Concordant pairs (nonparametric)"],
    }
    st.table(pd.DataFrame(comp_data))



@register_test("Point-Biserial Correlation")
def render_point_biserial_correlation(external_data=None):

    from scipy.stats import pointbiserialr, ttest_ind

    st.subheader("Interactive Point-Biserial Correlation")

    st.info("""
    **Point-Biserial Correlation (r_pb)** measures the relationship between a
    **binary/dichotomous variable** and a **continuous variable**.
    
    - Mathematically equivalent to **Pearson's r** with one binary variable
    - Also directly related to the **independent samples t-test**
    - r_pb² = proportion of variance explained by group membership
    """)

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("pointbiserial", mode="two_sample")

    # =========================
    # CONTROLS / DATA
    # =========================

    np.random.seed(42)

    if src["using_uploaded"]:
        g0 = src["data"]["group1"]
        g1 = src["data"]["group2"]
        group_names = src["data"]["group_names"]
        
        group = np.concatenate([np.zeros(len(g0)), np.ones(len(g1))])
        y = np.concatenate([g0, g1])
    else:
        group_difference = st.slider(
            "Group Mean Difference",
            0.0,
            10.0,
            3.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
            key="noise_7",
        )

        group = np.random.binomial(1, 0.5, 300)
        y = group * group_difference + np.random.normal(0, noise, 300)
        
        g0 = y[group == 0]
        g1 = y[group == 1]
        group_names = ["Group 0", "Group 1"]

    # =========================
    # TEST
    # =========================

    group_for_pb = np.concatenate([np.zeros(len(g0)), np.ones(len(g1))])
    y_for_pb = np.concatenate([g0, g1])
    r, p = pointbiserialr(group_for_pb, y_for_pb)

    st.latex(rf"r_{{pb}} = {r:.3f}")
    st.latex(rf"\text{{{format_p_value(p)}}}")
    st.latex(rf"R^2 = {r**2:.4f}")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=g0,
            name=group_names[0],
            boxmean="sd",
        )
    )

    fig.add_trace(
        go.Box(
            y=g1,
            name=group_names[1],
            boxmean="sd",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        yaxis_title="Continuous Variable",
        title=f"Point-Biserial r = {r:.3f}",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # SUMMARY & COMPARISON
    # =========================

    st.divider()
    st.subheader("Group Statistics & Comparison")

    n0, n1 = len(g0), len(g1)
    mean0, mean1 = np.mean(g0), np.mean(g1)
    sd0, sd1 = np.std(g0, ddof=1), np.std(g1, ddof=1)
    
    t_ind, p_ind = ttest_ind(g0, g1)
    
    summary_data = {
        "Group": group_names,
        "n": [n0, n1],
        "Mean": [f"{mean0:.3f}", f"{mean1:.3f}"],
        "SD": [f"{sd0:.3f}", f"{sd1:.3f}"],
    }
    st.table(pd.DataFrame(summary_data))

    st.info(f"""
    **Relationship with t-test:**
    - t({n0+n1-2}) = {t_ind:.3f}, {format_p_value(p_ind)}
    - r_pb² = {(r**2):.4f} ({(r**2)*100:.1f}% of variance explained by group membership)
    """)

# Regression Tests



@register_test("Logistic Regression")
def render_logistic_regression(external_data=None):

    st.subheader("Interactive Logistic Regression")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("log_reg", mode="correlation")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
        from sklearn.linear_model import LogisticRegression as LR
        x_reshaped = x.reshape(-1, 1)
        log_model = LR(C=1e6, solver="lbfgs", max_iter=1000)
        log_model.fit(x_reshaped, y)
        coef_ = log_model.coef_[0][0]
        intercept_ = log_model.intercept_[0]
        y_pred_prob = log_model.predict_proba(x_reshaped)[:, 1]
        y_pred_class = log_model.predict(x_reshaped)

        from sklearn.metrics import log_loss
        ll_null = log_loss(y, np.full_like(y, y.mean()))
        ll_model = log_loss(y, y_pred_prob)
        pseudo_r2 = 1 - ll_model / ll_null
        n_log = len(y)
        st.latex(rf"\text{{Logistic Regression with uploaded data}}")
        st.latex(rf"\text{{Coefficient (β₁)}} = {coef_:.4f}")
        st.latex(rf"\text{{Intercept (β₀)}} = {intercept_:.4f}")
        st.latex(rf"\text{{Odds Ratio (OR)}} = {np.exp(coef_):.4f}")
        st.latex(rf"\text{{McFadden R²}} = {pseudo_r2:.4f}")

        x_sorted = np.linspace(x.min(), x.max(), 500)
        logit_line = intercept_ + coef_ * x_sorted
        p_line = 1 / (1 + np.exp(-logit_line))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers", name="Observed",
                marker=dict(size=6, color="rgba(100,150,255,0.6)")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_sorted, y=p_line, mode="lines", name="Sigmoid Fit",
                line=dict(color="red", width=3)
            )
        )
        fig.update_layout(
            height=500, template="plotly_dark",
            xaxis_title=col_names[0] if col_names else "Predictor",
            yaxis_title="Probability",
            yaxis=dict(range=[-0.05, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Model Summary")
        results_data = {
            "Metric": ["Sample Size", "Coefficient (β₁)", "Intercept (β₀)",
                       "Odds Ratio (OR)", "McFadden R²"],
            "Value": [f"{n_log}", f"{coef_:.4f}", f"{intercept_:.4f}",
                      f"{np.exp(coef_):.4f}", f"{pseudo_r2:.4f}"],
        }
        st.table(pd.DataFrame(results_data))

    else:
        col1, col2 = st.columns(2)

        with col1:
            beta0 = st.slider(
                "Intercept (β₀)",
                -10.0,
                10.0,
                0.0,
                0.1,
                key="logistic_regression_intercept",
            )

        with col2:
            beta1 = st.slider(
                "Slope (β₁)", -5.0, 5.0, 1.0, 0.1, key="logistic_regression_slope"
            )

        # =========================
        # DATA
        # =========================

        x = np.linspace(-10, 10, 1000)

        logit = beta0 + beta1 * x

        p = 1 / (1 + np.exp(-logit))

        # =========================
        # LATEX
        # =========================

        st.latex(rf"""
            p = \dfrac{{1}}{{1 + e^{{-({beta0:.2f} + {beta1:.2f}x)}}}}
            """)

        # =========================
        # PLOTLY FIGURE
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=p,
                mode="lines",
                name="Sigmoid Curve",
            )
        )

        fig.update_layout(
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Predictor (x)",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
        )

        st.plotly_chart(fig, use_container_width=True)



@register_test("Simple Linear Regression")
def render_simple_linear_regression(external_data=None):
    st.subheader("Interactive Simple Linear Regression")

    st.info("""
    **Simple Linear Regression** models the linear relationship between:
    - A **predictor variable (X)**
    - An **outcome variable (Y)**
    
    Model: Y = β₀ + β₁X + ε
    
    - β₀: intercept (value of Y when X=0)
    - β₁: slope (change in Y per unit change in X)
    """)

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("regression", mode="correlation")

    # =========================
    # CONTROLS / DATA
    # =========================

    np.random.seed(42)

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
        n = len(x)
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            beta0 = st.slider(
                "Intercept (β₀)",
                -20.0,
                20.0,
                0.0,
                0.1,
                key="simple_linear_regression_intercept",
            )

        with col2:
            beta1 = st.slider(
                "Slope (β₁)",
                -10.0,
                10.0,
                1.0,
                0.1,
                key="simple_linear_regression_slope",
            )

        with col3:
            noise = st.slider(
                "Noise (σ)",
                0.0,
                20.0,
                3.0,
                0.1,
                key="slr_noise",
            )

        n = st.slider("Sample Size (n)", 10, 500, 100, key="slr_n")

        x = np.linspace(0, 10, n)
        y_true = beta0 + beta1 * x
        y = y_true + np.random.normal(0, noise, n)
        col_names = ["X", "Y"]

    # =========================
    # FIT REGRESSION
    # =========================

    from scipy.stats import linregress, pearsonr
    result = linregress(x, y)
    slope, intercept, r, p, se_slope = result
    
    r, _ = pearsonr(x, y)
    r2 = r ** 2
    
    y_pred = intercept + slope * x
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)

    # =========================
    # EQUATION
    # =========================

    st.latex(rf"\hat{{y}} = {intercept:.3f} + ({slope:.3f})x")
    st.latex(rf"R^2 = {r2:.4f}")
    st.latex(rf"F_{{1,{n-2}}} \text{{ p = {format_p_value(p)}}}")

    # =========================
    # PLOT - SCATTER WITH REGRESSION LINE
    # =========================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observed Data",
            marker=dict(color="rgba(100, 150, 255, 0.6)", size=8),
        )
    )

    sorted_idx = np.argsort(x)
    fig.add_trace(
        go.Scatter(
            x=x[sorted_idx],
            y=y_pred[sorted_idx],
            mode="lines",
            name=f"Regression: ŷ = {intercept:.2f} + {slope:.2f}x",
            line=dict(color="red", width=3),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title=col_names[0],
        yaxis_title=col_names[1],
        title=f"Simple Linear Regression: R² = {r2:.4f}",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # DETAILED RESULTS TABLE
    # =========================

    st.divider()
    st.subheader("Regression Coefficients")

    from scipy import stats
    t_intercept = intercept / (np.std(residuals) * np.sqrt(1/n + np.mean(x)**2 / np.sum((x - np.mean(x))**2))) if n > 2 else np.nan

    coeff_data = {
        "Term": ["Intercept (β₀)", "Slope (β₁)"],
        "Estimate": [f"{intercept:.4f}", f"{slope:.4f}"],
        "SE": ["(not computed)", f"{se_slope:.4f}"],
        "t": ["-", f"{slope/se_slope:.3f}" if se_slope > 0 else "-"],
        "p-value": ["-", format_p_value(p)],
    }
    st.table(pd.DataFrame(coeff_data))

    # =========================
    # MODEL FIT STATISTICS
    # =========================

    st.divider()
    st.subheader("Model Fit")

    fit_data = {
        "Metric": [
            "Sample Size (n)",
            "Pearson r",
            "R-squared (R²)",
            "Adjusted R²",
            "RMSE (Root MSE)",
        ],
        "Value": [
            f"{n}",
            f"{r:.4f}",
            f"{r2:.4f}",
            f"{1 - (1 - r2) * (n - 1) / (n - 2):.4f}" if n > 2 else "-",
            f"{rmse:.4f}",
        ],
    }
    st.table(pd.DataFrame(fit_data))

    # =========================
    # RESIDUAL PLOT
    # =========================

    st.divider()
    st.subheader("Residual Plot (Check Homoscedasticity)")

    fig_resid = go.Figure()

    fig_resid.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            name="Residuals",
            marker=dict(color="rgba(255, 150, 100, 0.6)", size=6),
        )
    )

    fig_resid.add_hline(
        y=0,
        line=dict(color="red", dash="dash"),
        name="Zero Line",
    )

    fig_resid.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Predicted Values (ŷ)",
        yaxis_title="Residuals (y - ŷ)",
        title="Residuals vs Fitted Values",
    )

    st.plotly_chart(fig_resid, use_container_width=True)

    st.info("""
    **How to interpret the residual plot:**
    - ✅ Good: Random scatter around 0, no pattern
    - ❌ Funnel shape = Heteroscedasticity (non-constant variance)
    - ❌ Curved pattern = Nonlinear relationship (consider adding polynomial term)
    """)



@register_test("Multiple Linear Regression")
def render_multiple_linear_regression(external_data=None):
    st.subheader("Interactive Multiple Linear Regression")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("mlr", mode="correlation")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]

        from scipy.stats import linregress
        slr_result = linregress(x, y)
        slope, intercept, r, p, se_slope = slr_result
        r2 = r ** 2
        y_pred = intercept + slope * x
        n = len(x)

        st.latex(rf"\hat{{y}} = {intercept:.3f} + ({slope:.3f}) \cdot x")
        st.latex(rf"R^2 = {r2:.4f}")
        st.info("Workspace data provides 2-variable regression. For full multiple regression with multiple predictors, upload a dataset with additional variables.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed", marker=dict(color="rgba(100,150,255,0.6)", size=8)))
        sorted_idx = np.argsort(x)
        fig.add_trace(go.Scatter(x=x[sorted_idx], y=y_pred[sorted_idx], mode="lines", name=f"ŷ = {intercept:.2f} + {slope:.2f}x", line=dict(color="red", width=3)))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "X", yaxis_title=col_names[1] if col_names else "Y", title=f"Bivariate Regression: R² = {r2:.4f}")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Regression Coefficients")
        coeff_data = {
            "Term": ["Intercept (β₀)", "Slope (β₁)"],
            "Estimate": [f"{intercept:.4f}", f"{slope:.4f}"],
            "SE": ["(not computed)", f"{se_slope:.4f}"],
            "t": ["-", f"{slope/se_slope:.3f}" if se_slope > 0 else "-"],
            "p-value": ["-", format_p_value(p)],
        }
        st.table(pd.DataFrame(coeff_data))

    else:
        beta0 = st.slider(
            "β₀", -20.0, 20.0, 0.0, 0.1, key="multiple_linear_regression_beta0"
        )

        beta1 = st.slider("β₁ (x₁ coefficient)", -10.0, 10.0, 1.0, 0.1)

        beta2 = st.slider("β₂ (x₂ coefficient)", -10.0, 10.0, 1.0, 0.1)

        # =========================
        # GRID
        # =========================

        x1 = np.linspace(-10, 10, 50)
        x2 = np.linspace(-10, 10, 50)

        X1, X2 = np.meshgrid(x1, x2)

        Y = beta0 + beta1 * X1 + beta2 * X2

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"y = {beta0:.2f} + ({beta1:.2f})x_1 + ({beta2:.2f})x_2")

        # =========================
        # SURFACE PLOT
        # =========================

        fig = go.Figure(
            data=[
                go.Surface(
                    x=X1,
                    y=X2,
                    z=Y,
                )
            ]
        )

        fig.update_layout(
            template="plotly_dark",
            height=700,
            scene=dict(
                xaxis_title="x₁",
                yaxis_title="x₂",
                zaxis_title="y",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)



@register_test("Multinomial Logistic Regression")
def render_multinomial_logistic_regression(external_data=None):
    st.subheader("Interactive Multinomial Logistic Regression")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("multinomial_log", mode="correlation")

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
        st.info("""
        **Multinomial Logistic Regression with uploaded data**
        This is a simplified widget. The uploaded data is used to show the relationship
        between a predictor and a multi-category outcome. For a full analysis,
        consider using a dedicated statistical package.
        """)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(size=6, color="rgba(100,150,255,0.6)")))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "Predictor", yaxis_title="Category")
        st.plotly_chart(fig, use_container_width=True)
    else:
        beta1 = st.slider("Class A coefficient", -5.0, 5.0, 1.0, 0.1)

        beta2 = st.slider("Class B coefficient", -5.0, 5.0, -1.0, 0.1)

        # =========================
        # DATA
        # =========================

        x = np.linspace(-10, 10, 500)

        score1 = np.exp(beta1 * x)
        score2 = np.exp(beta2 * x)
        score3 = np.exp(0)

        total = score1 + score2 + score3

        p1 = score1 / total
        p2 = score2 / total
        p3 = score3 / total

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=p1, mode="lines", name="Class A"))
        fig.add_trace(go.Scatter(x=x, y=p2, mode="lines", name="Class B"))
        fig.add_trace(go.Scatter(x=x, y=p3, mode="lines", name="Reference"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            yaxis=dict(range=[0, 1]),
            xaxis_title="Predictor",
            yaxis_title="Class Probability",
        )

        st.plotly_chart(fig, use_container_width=True)



@register_test("Ordinal Logistic Regression")
def render_ordinal_logistic_regression(external_data=None):
    st.subheader("Interactive Ordinal Logistic Regression")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("ordinal_log", mode="correlation")

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
        st.info("""
        **Ordinal Logistic Regression with uploaded data**
        This is a simplified widget. The uploaded data is used to show the relationship
        between a predictor and an ordinal outcome. For a full analysis,
        consider using a dedicated statistical package.
        """)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data", marker=dict(size=6, color="rgba(100,150,255,0.6)")))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "Predictor", yaxis_title="Ordinal Outcome")
        st.plotly_chart(fig, use_container_width=True)
    else:
        beta = st.slider("β coefficient", -5.0, 5.0, 1.0, 0.1)

        threshold1 = st.slider("Threshold θ₁", -5.0, 5.0, -1.0, 0.1)

        threshold2 = st.slider("Threshold θ₂", -5.0, 5.0, 1.0, 0.1)

        # =========================
        # DATA
        # =========================

        x = np.linspace(-10, 10, 500)

        cum1 = 1 / (1 + np.exp(-(threshold1 - beta * x)))
        cum2 = 1 / (1 + np.exp(-(threshold2 - beta * x)))

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=cum1, mode="lines", name="P(Y ≤ 1)"))

        fig.add_trace(go.Scatter(x=x, y=cum2, mode="lines", name="P(Y ≤ 2)"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            yaxis=dict(range=[0, 1]),
            xaxis_title="Predictor",
            yaxis_title="Cumulative Probability",
        )

        st.plotly_chart(fig, use_container_width=True)



@register_test("Poisson Regression")
def render_poisson_regression(external_data=None):
    st.subheader("Interactive Poisson Regression")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("poisson_reg", mode="correlation")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
        from scipy.optimize import minimize as _minimize

        def _pois_nll(beta, x, y):
            lam = np.exp(beta[0] + beta[1] * x)
            return np.sum(lam - y * np.log(lam + 1e-10))

        res = _minimize(_pois_nll, [0.5, 0.1], args=(x, y), method="Nelder-Mead")
        beta0_hat, beta1_hat = res.x
        lam_fit = np.exp(beta0_hat + beta1_hat * x)
        n_pois = len(x)

        st.latex(rf"\hat{{\lambda}} = e^{{{beta0_hat:.3f} + ({beta1_hat:.3f})x}}")
        st.info(f"Fitted Poisson regression on uploaded data (n={n_pois}). y should be count data.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed", marker=dict(color="rgba(100,150,255,0.6)", size=6)))
        sorted_idx = np.argsort(x)
        fig.add_trace(go.Scatter(x=x[sorted_idx], y=lam_fit[sorted_idx], mode="lines", name="Expected Count", line=dict(color="red", width=3)))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "Predictor", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Model Summary")
        st.table(pd.DataFrame({
            "Metric": ["Sample Size", "β₀ (Intercept)", "β₁ (Slope)", "AIC"],
            "Value": [f"{n_pois}", f"{beta0_hat:.4f}", f"{beta1_hat:.4f}", f"{2 * res.fun + 4:.2f}"],
        }))
    else:
        beta0 = st.slider("β₀", -3.0, 3.0, 0.5, 0.1, key="poisson_regression_beta0")

        beta1 = st.slider("β₁", -1.0, 1.0, 0.2, 0.05)

        # =========================
        # DATA
        # =========================

        x = np.linspace(0, 20, 500)

        lam = np.exp(beta0 + beta1 * x)

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"\lambda = e^{{{beta0:.2f} + ({beta1:.2f})x}}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=lam,
                mode="lines",
                name="Expected Count",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Predictor",
            yaxis_title="Expected Count (λ)",
        )

        st.plotly_chart(fig, use_container_width=True)



@register_test("Negative Binomial Regression")
def render_negative_binomial_regression(external_data=None):

    st.subheader("Interactive Negative Binomial Regression")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("neg_binom_reg", mode="correlation")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        x = src["data"]["x"]
        y = src["data"]["y"]
        col_names = src["data"]["col_names"]
        from scipy.optimize import minimize as _nb_min

        def _nb_nll(params, x, y):
            beta0, beta1, alpha = params
            mu = np.exp(beta0 + beta1 * x)
            theta = 1 / alpha
            nll = -np.sum(
                np.random.gamma(theta, 1, len(y)) * 0
                + y * np.log(mu + 1e-10)
                - (y + theta) * np.log(mu + theta + 1e-10)
                + np.random.gamma(1, 1, 1) * 0
            )
            from scipy.special import gammaln
            nll = -np.sum(
                gammaln(y + theta)
                - gammaln(theta)
                - gammaln(y + 1)
                + theta * np.log(theta + 1e-10)
                + y * np.log(mu + 1e-10)
                - (y + theta) * np.log(mu + theta + 1e-10)
            )
            return nll

        res_nb = _nb_min(_nb_nll, [0.5, 0.1, 0.5], args=(x, y), method="Nelder-Mead")
        beta0_nb, beta1_nb, alpha_nb = res_nb.x
        n_nb = len(x)

        st.latex(rf"\hat{{\mu}} = e^{{{beta0_nb:.3f} + ({beta1_nb:.3f})x}}")
        st.latex(rf"\hat{{\alpha}} = {alpha_nb:.4f} \quad (\text{{dispersion parameter}})")
        st.info(f"Fitted Negative Binomial regression on uploaded data (n={n_nb}). Overdispersion parameter α = {alpha_nb:.4f} (α=0 would be Poisson).")

        mu_fit = np.exp(beta0_nb + beta1_nb * x)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observed", marker=dict(color="rgba(100,150,255,0.6)", size=6)))
        sorted_idx = np.argsort(x)
        fig.add_trace(go.Scatter(x=x[sorted_idx], y=mu_fit[sorted_idx], mode="lines", name="Expected Count", line=dict(color="red", width=3)))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title=col_names[0] if col_names else "Predictor", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Model Summary")
        st.table(pd.DataFrame({
            "Metric": ["Sample Size", "β₀ (Intercept)", "β₁ (Slope)", "α (Dispersion)", "AIC"],
            "Value": [f"{n_nb}", f"{beta0_nb:.4f}", f"{beta1_nb:.4f}", f"{alpha_nb:.4f}", f"{2 * res_nb.fun + 6:.2f}"],
        }))
    else:
        beta0 = st.slider("β₀", -3.0, 3.0, 0.5, 0.1, key="negbinom_beta0")
        beta1 = st.slider("β₁", -1.0, 1.0, 0.2, 0.05, key="negbinom_beta1")
        alpha_disp = st.slider("Dispersion (α)", 0.05, 5.0, 1.0, 0.05, key="negbinom_alpha")

        x = np.linspace(0, 20, 500)
        mu = np.exp(beta0 + beta1 * x)

        st.latex(rf"\mu = e^{{{beta0:.2f} + ({beta1:.2f})x}}")
        st.latex(rf"\text{{Var}}(Y) = \mu + \alpha\mu^2")
        st.info("Negative Binomial extends Poisson by adding a dispersion parameter α. When α → 0, it reduces to Poisson.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=mu, mode="lines", name="Expected Count"))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title="Predictor", yaxis_title="Expected Count (μ)")
        st.plotly_chart(fig, use_container_width=True)



@register_test("Cox Proportional Hazards Regression")
def render_cox_proportional_hazards_regression(external_data=None):

    st.subheader("Interactive Cox Regression")

    # =========================
    # DATA SOURCE TOGGLE (survival special)
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        with st.expander("📁 Optional: Use Your Own Data", expanded=False):
            st.markdown("Upload a CSV with time, event, and predictor columns.")
            source = st.radio(
                "Data Source",
                ["Simulated (sliders, for learning)", "Upload CSV/Excel (your data)"],
                key="cox_datasource",
                index=0,
                label_visibility="collapsed",
            )
        if "Simulated" in source:
            src = {"using_uploaded": False, "data": None}
        else:
            cox_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="cox_file")
            if cox_file is not None:
                cox_df = pd.read_csv(cox_file) if cox_file.name.endswith(".csv") else pd.read_excel(cox_file)
                st.success(f"Loaded {len(cox_df)} rows")
                num_cols = list(cox_df.select_dtypes(include=["int64", "float64"]).columns)
                if len(num_cols) >= 2:
                    time_col = st.selectbox("Time column", num_cols, key="cox_time")
                    event_col = st.selectbox("Event column (0/1)", num_cols, key="cox_event", index=min(1, len(num_cols)-1))
                    pred_col = st.selectbox("Predictor column", [c for c in num_cols if c not in [time_col, event_col]] or num_cols, key="cox_pred")
                    src = {"using_uploaded": True, "data": {"time": cox_df[time_col].dropna().values, "event": cox_df[event_col].dropna().values, "predictor": cox_df[pred_col].dropna().values, "time_col": time_col, "event_col": event_col, "pred_col": pred_col}}
                else:
                    st.error("Need at least 2 numeric columns")
                    src = {"using_uploaded": False, "data": None}
            else:
                src = {"using_uploaded": False, "data": None}

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        cox_times = src["data"]["time"]
        cox_event = src["data"]["event"]
        cox_pred = src["data"]["predictor"]

        from scipy.stats import chi2 as _cox_chi2

        def _cox_partial_likelihood(beta, t, e, x):
            order = np.argsort(t)
            t, e, x = t[order], e[order], x[order]
            risk = np.exp(beta * x)
            total_risk = np.cumsum(risk[::-1])[::-1]
            log_pl = np.sum(e * (beta * x - np.log(total_risk + 1e-10)))
            return -log_pl

        from scipy.optimize import minimize_scalar
        cox_res = minimize_scalar(lambda b: _cox_partial_likelihood(b, cox_times, cox_event, cox_pred))
        cox_beta = cox_res.x
        cox_hr = np.exp(cox_beta)
        cox_n = len(cox_times)
        cox_events = int(cox_event.sum())

        null_ll = _cox_partial_likelihood(0, cox_times, cox_event, cox_pred)
        alt_ll = cox_res.fun
        cox_lr = 2 * (null_ll - alt_ll)
        cox_p = 1 - _cox_chi2.cdf(cox_lr, 1)

        st.latex(rf"\hat{{\beta}} = {cox_beta:.4f}")
        st.latex(rf"\text{{Hazard Ratio}} = e^{{\hat{{\beta}}}} = {cox_hr:.4f}")
        st.latex(rf"\text{{Likelihood Ratio }} \chi^2 = {cox_lr:.3f}, \; p = {format_p_value(cox_p)}")

        fig = go.Figure()
        t_grid = np.linspace(0, np.percentile(cox_times[cox_event == 1], 95) * 1.2, 200)
        low_pred = np.percentile(cox_pred, 25)
        high_pred = np.percentile(cox_pred, 75)
        base_haz = cox_events / np.sum(cox_times)
        for val, lbl, clr in [(low_pred, "Low Predictor", "blue"), (high_pred, "High Predictor", "red")]:
            surv = np.exp(-base_haz * np.exp(cox_beta * val) * t_grid)
            fig.add_trace(go.Scatter(x=t_grid, y=surv, mode="lines", name=lbl, line=dict(color=clr)))
        fig.update_layout(template="plotly_dark", height=500, xaxis_title="Time", yaxis_title="Survival Probability")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Model Summary")
        st.table(pd.DataFrame({
            "Metric": ["N", "Events", "β", "HR", "LR χ²", "p-value"],
            "Value": [f"{cox_n}", f"{cox_events}", f"{cox_beta:.4f}", f"{cox_hr:.4f}", f"{cox_lr:.3f}", format_p_value(cox_p)],
        }))
    else:
        hazard_ratio = st.slider("True Hazard Ratio (exp(β))", 0.5, 4.0, 2.0, 0.1)

        n_subjects = st.slider(
            "Number of Subjects",
            20,
            500,
            100,
            key="cox_proportional_hazards_regression_number_of_subjects",
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        group = np.random.binomial(1, 0.5, n_subjects)

        log_hr = np.log(hazard_ratio)
        baseline_hazard = 0.05

        survival_times = -np.log(np.random.uniform(size=n_subjects)) / (
            baseline_hazard * np.exp(log_hr * group)
        )

        censor_times = np.random.uniform(5, 20, n_subjects)

        observed = (survival_times <= censor_times).astype(int)
        times = np.minimum(survival_times, censor_times)

        # =========================
        # SURVIVAL CURVES (theoretical exponential)
        # =========================

        fig = go.Figure()
        t_grid = np.linspace(0, max(times) * 1.1, 200)

        for grp, label, color in [(0, "Control", "blue"), (1, "Treatment", "red")]:
            hr = 1 if grp == 0 else hazard_ratio
            # Exponential survival: S(t) = exp(-h0 * t * exp(beta * X))
            surv = np.exp(-baseline_hazard * hr * t_grid)
            fig.add_trace(
                go.Scatter(
                    x=t_grid,
                    y=surv,
                    mode="lines",
                    name=label,
                    line=dict(color=color),
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Time",
            yaxis_title="Survival Probability",
        )

        st.plotly_chart(fig, use_container_width=True)

