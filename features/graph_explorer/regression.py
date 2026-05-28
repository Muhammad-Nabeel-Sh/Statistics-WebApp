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

from .shared import _rng, _gen_corr, _gen_reg

def linear_reg_widget():
    st.markdown("## Linear Regression Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 10, 200, 50, key="lr_n")
        slope = st.slider("True Slope (beta1)", -3.0, 3.0, 1.0, 0.1, key="lr_slope")
        noise = st.slider("Error (sigma)", 0.1, 5.0, 1.0, 0.1, key="lr_noise")
        show_ci = st.toggle("Show Confidence Band", True, key="lr_ci")
        show_resid = st.toggle("Show Residuals", False, key="lr_resid")
    np.random.seed(42)
    x, y = _gen_reg(n, slope, noise)
    slope_est, intercept, r_val, p_val, se = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 200)
    y_line = slope_est * x_line + intercept
    fig = go.Figure()
    if show_resid:
        y_pred = slope_est * x + intercept
        for xi, yi, ypi in zip(x, y, y_pred):
            fig.add_trace(
                go.Scatter(
                    x=[xi, xi],
                    y=[yi, ypi],
                    mode="lines",
                    line=dict(color="rgba(200,200,200,0.3)", width=1),
                    showlegend=False,
                )
            )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data",
            marker=dict(color="#4C78A8", size=6, opacity=0.7),
            hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<extra></extra>",
        )
    )
    if show_ci:
        t_val = stats.t.ppf(0.975, n - 2)
        pred_se = np.sqrt(
            se**2
            * (1 + 1 / n + (x_line - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
        )
        ci = t_val * pred_se
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line + ci,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line - ci,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(228, 87, 86, 0.15)",
                name="95% CI",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"beta1={slope_est:.2f}, R-squared={r_val**2:.3f}",
            line=dict(color="#E45756", width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Line = best fit (minimizes residuals)\n"
                "- Band = 95% CI for prediction\n"
                "- Slope = change in Y per unit X\n"
                "- R-squared = prop of variance explained\n"
                "- p-value = test if slope != 0"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Model continuous outcome\n"
                "- Estimate effect size (beta1)\n"
                "- Predict Y from X\n"
                "- Test linear relationship"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- F-test (overall model)\n"
                "- t-test (coefficient)\n"
                "- Pearson correlation\n"
                "- ANOVA (nested models)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Extrapolating beyond the observed x-range. "
                "Linear relationships may not hold outside "
                "the data - never predict beyond your data."
            )



def multiple_reg_widget():
    st.markdown("## Multiple Regression Surface")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="mr_n")
        b1 = st.slider("Coefficient beta1", -2.0, 2.0, 0.7, 0.1, key="mr_b1")
        b2 = st.slider("Coefficient beta2", -2.0, 2.0, 0.5, 0.1, key="mr_b2")
        noise = st.slider("Noise", 0.1, 3.0, 0.5, 0.1, key="mr_noise")
        show_interaction = st.toggle(
            "Show Interaction",
            False,
            key="mr_interact",
            help="Include beta12 x X1 x X2 term",
        )
    np.random.seed(42)
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    if show_interaction:
        y = b1 * x1 + b2 * x2 + 0.5 * x1 * x2 + np.random.normal(0, noise, n)
    else:
        y = b1 * x1 + b2 * x2 + np.random.normal(0, noise, n)
    grid = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(grid, grid)
    if show_interaction:
        Y_pred = b1 * X1 + b2 * X2 + 0.5 * X1 * X2
    else:
        Y_pred = b1 * X1 + b2 * X2
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x1,
            y=x2,
            z=y,
            mode="markers",
            name="Data",
            marker=dict(size=4, color="#4C78A8", opacity=0.7),
            hovertemplate="X1=%{x:.2f}<br>X2=%{y:.2f}<br>Y=%{z:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Surface(
            x=grid,
            y=grid,
            z=Y_pred,
            name="Predicted",
            colorscale="Reds",
            opacity=0.5,
            showscale=False,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        scene=dict(
            xaxis_title="X1",
            yaxis_title="X2",
            zaxis_title="Y",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Plane = predicted Y from X1, X2\n"
                "- Slope along X1 = beta1 (holding X2 constant)\n"
                "- Slope along X2 = beta2 (holding X1 constant)\n"
                "- Twisted surface = interaction"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Model multiple predictors\n"
                "- Control for confounders\n"
                "- Test interaction effects\n"
                "- Understand partial effects"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- F-test (overall model)\n"
                "- Partial F-test (nested models)\n"
                "- t-test (individual coefficients)\n"
                "- VIF (multicollinearity)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Coefficients are sensitive to predictor scaling. "
                "Standardize (z-score) predictors before comparing "
                "coefficient magnitudes. Unstandardized betas "
                "depend on the unit of X."
            )



def logistic_widget():
    st.markdown("## Logistic Regression Sigmoid Curve")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="logit_n")
        separation = st.slider(
            "Class Separation",
            0.5,
            5.0,
            2.0,
            0.1,
            key="logit_sep",
            help="How far apart the two classes are on X",
        )
        threshold = st.slider(
            "Decision Threshold", 0.1, 0.9, 0.5, 0.05, key="logit_thresh"
        )
        show_data = st.toggle("Show Data Points", True, key="logit_data")
        show_prob = st.toggle("Show Probability Curve", True, key="logit_prob")
    np.random.seed(42)
    x0 = np.random.normal(-separation / 2, 1, n // 2)
    x1 = np.random.normal(separation / 2, 1, n - n // 2)
    x = np.concatenate([x0, x1])
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)])
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1000)
    model.fit(x.reshape(-1, 1), y)
    x_grid = np.linspace(min(x) - 1, max(x) + 1, 300)
    y_prob = model.predict_proba(x_grid.reshape(-1, 1))[:, 1]
    fig = go.Figure()
    if show_prob:
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_prob,
                mode="lines",
                name="P(Y=1)",
                line=dict(color="#4C78A8", width=3),
                hovertemplate="X = %{x:.2f}<br>P(Y=1) = %{y:.3f}<extra></extra>",
            )
        )
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#E45756",
            annotation_text=f"Threshold = {threshold}",
        )
    if show_data:
        jitter = np.random.uniform(-0.05, 0.05, n)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y + jitter,
                mode="markers",
                marker=dict(
                    color=y, colorscale="RdBu", size=6, showscale=False, opacity=0.7
                ),
                name="Data",
                hovertemplate="X = %{x:.2f}<br>Y = %{y:.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Predictor (X)",
        yaxis_title="Probability",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- S-curve = logistic probability\n"
                "- Steep curve = strong predictor\n"
                "- Shallow curve = weak predictor\n"
                "- Threshold = classification cutoff\n"
                "- Above threshold -> predict class 1"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Binary outcome prediction\n"
                "- Estimate odds ratios\n"
                "- Medical diagnosis models\n"
                "- Risk factor analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Likelihood ratio test\n"
                "- Wald test (coefficients)\n"
                "- Hosmer-Lemeshow (calibration)\n"
                "- ROC-AUC (discrimination)"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Changing the decision threshold changes "
                "sensitivity and specificity. Default 0.5 "
                "is not always optimal - adjust based on "
                "the cost of false positives vs false negatives."
            )



def multinomial_widget():
    st.markdown("## Multinomial Decision Boundaries")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Samples per Class", 10, 100, 30, key="multi_n")
        n_classes = st.selectbox("Number of Classes", [3, 4], index=0, key="multi_k")
        separation = st.slider("Class Separation", 0.5, 4.0, 2.0, 0.1, key="multi_sep")
        show_boundaries = st.toggle("Show Decision Regions", True, key="multi_bound")
    np.random.seed(42)
    k = int(n_classes)
    centers = [
        [np.cos(2 * np.pi * i / k) * separation, np.sin(2 * np.pi * i / k) * separation]
        for i in range(k)
    ]
    X_list2, y_list2 = [], []
    for i, center in enumerate(centers):
        X_list2.append(np.random.normal(center, 0.5, (n, 2)))
        y_list2.append(np.full(n, i))
    X = np.vstack(X_list2)
    y = np.concatenate(y_list2)
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    if show_boundaries:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=10, solver="lbfgs")
        model.fit(X, y)
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
        )
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                showscale=False,
                colorscale=[[i / (k - 1), colors[i]] for i in range(k)],
                opacity=0.3,
                name="Decision Regions",
                hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<extra></extra>",
            )
        )
    for i in range(k):
        mask = y == i
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=f"Class {i}",
                marker=dict(color=colors[i], size=6, opacity=0.8),
                hovertemplate=f"Class {i}<br>x1=%{{x:.2f}}<br>x2=%{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Colored regions = decision zones\n"
                "- Boundaries = where model is uncertain\n"
                "- Overlap = classification difficulty\n"
                "- Filled area = multinomial probability"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Multi-class classification\n"
                "- Understand decision boundaries\n"
                "- Compare classifier geometries\n"
                "- Feature space exploration"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Multinomial logistic regression\n"
                "- MANOVA\n"
                "- Discriminant analysis\n"
                "- Classification metrics"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Linear decision boundaries (logistic regression) "
                "cannot separate non-linear class patterns. "
                "If classes are interleaved, consider non-linear "
                "methods (kernels, trees, neural nets)."
            )



def ordinal_logit_widget():
    st.markdown("## Ordinal Logistic Probability Curves")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="ord_n")
        n_levels = st.selectbox(
            "Number of Ordinal Levels", [3, 4, 5], index=1, key="ord_k"
        )
        effect = st.slider("Predictor Effect", 0.0, 3.0, 1.0, 0.1, key="ord_eff")
        show_cumulative = st.toggle(
            "Show Cumulative Probabilities", False, key="ord_cum"
        )
    k = int(n_levels)
    np.random.seed(42)
    x = np.random.uniform(-3, 3, n)
    thresholds = np.sort(np.random.uniform(-2, 2, k - 1))
    logit = effect * x
    cum_probs = []
    for thresh in thresholds:
        cum_probs.append(1 / (1 + np.exp(-(logit - thresh))))
    cum_probs.append(np.ones(n))
    x_grid = np.linspace(-3, 3, 200)
    logit_grid = effect * x_grid
    cum_grid = []
    for thresh in thresholds:
        cum_grid.append(1 / (1 + np.exp(-(logit_grid - thresh))))
    cum_grid.append(np.ones(200))
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly[:k]
    if show_cumulative:
        for i in range(k - 1):
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=cum_grid[i],
                    mode="lines",
                    name=f"P(Y <= {i+1})",
                    line=dict(dash="dash", color=colors[i]),
                    hovertemplate=f"X = %{{x:.2f}}<br>P(Y <= {i+1}) = %{{y:.3f}}<extra></extra>",
                )
            )
    for i in range(k):
        prob_i = (
            cum_grid[i]
            if i == 0
            else [cum_grid[i][j] - cum_grid[i - 1][j] for j in range(200)]
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=prob_i,
                mode="lines",
                name=f"P(Y = {i+1})",
                line=dict(color=colors[i], width=2.5),
                fill="tozeroy" if not show_cumulative else None,
                hovertemplate=f"X = %{{x:.2f}}<br>P(Y={i+1}) = %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Predictor",
        yaxis_title="Probability",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each curve = probability of one level\n"
                "- Curves shift with predictor value\n"
                "- Non-parallel = proportional odds violation\n"
                "- Steep transition = strong predictor"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Ordered categorical outcomes\n"
                "- Likert scale responses\n"
                "- Disease severity staging\n"
                "- Patient-reported outcomes"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Proportional odds test\n"
                "- Brant test (parallel regression)\n"
                "- Likelihood ratio test\n"
                "- Score test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "The proportional odds assumption (parallel curves) "
                "must hold. If curves cross or are non-parallel, "
                "a generalized ordered logit or multinomial model "
                "is needed."
            )



def poisson_widget():
    st.markdown("## Poisson Count Regression")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 300, 100, key="pois_n")
        base_rate = st.slider(
            "Base Rate (intercept)",
            0.5,
            5.0,
            2.0,
            0.1,
            key="pois_base",
            help="Expected count when X = 0",
        )
        effect = st.slider(
            "Effect (log-rate ratio)",
            0.0,
            2.0,
            0.5,
            0.05,
            key="pois_eff",
            help="Multiplicative effect per unit X",
        )
        show_mean = st.toggle("Show Mean Curve", True, key="pois_mean")
        show_overdisp = st.toggle(
            "Add Overdispersion",
            False,
            key="pois_over",
            help="Extra-Poisson variability",
        )
    np.random.seed(42)
    x = np.random.uniform(0, 5, n)
    log_lambda = np.log(base_rate) + effect * x
    if show_overdisp:
        y = np.random.negative_binomial(np.exp(log_lambda) * 2, 0.5, n)
    else:
        y = np.random.poisson(np.exp(log_lambda), n)
    x_grid = np.linspace(0, 5, 100)
    log_lambda_grid = np.log(base_rate) + effect * x_grid
    y_grid = np.exp(log_lambda_grid)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observed",
            marker=dict(color="#4C78A8", size=5, opacity=0.6),
            hovertemplate="X = %{x:.2f}<br>Count = %{y:.0f}<extra></extra>",
        )
    )
    if show_mean:
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name="Predicted Mean",
                line=dict(color="#E45756", width=3),
                hovertemplate="X = %{x:.2f}<br>Mean = %{y:.1f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Predictor",
        yaxis_title="Count",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Y-axis = count (0, 1, 2, ...)\n"
                "- Curve = predicted mean count\n"
                "- Spread increases with mean\n"
                "- Clustering at zero = zero-inflation"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Modeling count outcomes\n"
                "- Event frequencies\n"
                "- Rare disease incidence\n"
                "- Hospital readmissions"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Likelihood ratio test\n"
                "- Wald test\n"
                "- Deviance goodness-of-fit\n"
                "- Dispersion test"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Poisson assumes mean = variance. If variance > mean "
                "(overdispersion), use Negative Binomial. Many zeros? "
                "Consider zero-inflated or hurdle models."
            )


# --- DIAGNOSTIC ACCURACY PLOTS ---



def residuals_fitted_widget():
    st.markdown("## Residuals vs Fitted Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="resid_n")
        noise = st.slider("Noise Level", 0.1, 3.0, 1.0, 0.1, key="resid_noise")
        pattern = st.selectbox(
            "Pattern",
            ["Linear (OK)", "Heteroscedastic", "Non-linear", "Outlier"],
            key="resid_pattern",
        )
    np.random.seed(42)
    x = np.random.uniform(0, 10, n)
    if pattern == "Linear (OK)":
        y = 2 + 1.5 * x + np.random.normal(0, noise, n)
    elif pattern == "Heteroscedastic":
        y = 2 + 1.5 * x + np.random.normal(0, noise * (0.5 + 0.5 * x / 10), n)
    elif pattern == "Non-linear":
        y = 2 + 1.5 * x + 0.5 * x**2 + np.random.normal(0, noise, n)
    else:
        y = 2 + 1.5 * x + np.random.normal(0, noise, n)
        y[-1] += 15
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    fitted = model.predict(x.reshape(-1, 1))
    residuals = y - fitted
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            marker=dict(color="#4C78A8", size=5),
            name="Residuals",
            hovertemplate="Fitted=%{x:.2f}<br>Residual=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="red", dash="dash"), opacity=0.7)
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Random scatter around 0 = OK\n"
                "- Fan shape = heteroscedasticity\n"
                "- U-shape = non-linearity\n"
                "- Isolated points = outliers"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- After fitting linear regression\n"
                "- Check homoscedasticity assumption\n"
                "- Check linearity assumption\n"
                "- Identify influential points"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Breusch-Pagan test\n"
                "- Goldfeld-Quandt test\n"
                "- RESET test\n"
                "- Cook's distance"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Patterned residuals indicate model "
                "misspecification — do NOT interpret "
                "coefficients until residuals are "
                "well-behaved."
            )



def poly_reg_widget():
    st.markdown("## Polynomial Regression Fit")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 300, 80, key="poly_reg_n")
        degree = st.slider("Polynomial Degree", 1, 10, 1, key="poly_reg_deg")
        noise = st.slider("Noise Level", 0.1, 3.0, 0.5, 0.1, key="poly_reg_noise")
        true_fn = st.selectbox(
            "True Relationship",
            ["Linear", "Quadratic", "Cubic", "Sine"],
            key="poly_reg_fn",
        )
    np.random.seed(42)
    x = np.sort(np.random.uniform(-3, 3, n))
    if true_fn == "Linear":
        y_true = 2 + 1.5 * x
    elif true_fn == "Quadratic":
        y_true = 1 + x + 0.5 * x**2
    elif true_fn == "Cubic":
        y_true = 1 + x + 0.5 * x**2 - 0.2 * x**3
    else:
        y_true = 2 * np.sin(x)
    y = y_true + np.random.normal(0, noise, n)
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression

    poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly.fit(x.reshape(-1, 1), y)
    x_smooth = np.linspace(min(x), max(x), 300)
    y_pred = poly.predict(x_smooth.reshape(-1, 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="#4C78A8", size=5, opacity=0.6),
            name="Data",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_pred,
            mode="lines",
            line=dict(color="#E45756", width=3),
            name=f"Degree {degree}",
            hovertemplate="x=%{x:.2f}<br>Pred=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_true,
            mode="lines",
            line=dict(color="gray", width=2, dash="dot"),
            name="True Function",
            hovertemplate="x=%{x:.2f}<br>True=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Higher degree = more flexible\n"
                "- Degree 1 = straight line\n"
                "- Degree 2 = one bend\n"
                "- Degree 10 can overfit wildly"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Model non-linear relationships\n"
                "- Test for curvature in data\n"
                "- Understand bias-variance tradeoff\n"
                "- Teaching overfitting concepts"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- F-test for nested models\n"
                "- Cross-validation MSE\n"
                "- AIC / BIC comparison\n"
                "- ANOVA model comparison"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "High-degree polynomials overfit "
                "near boundaries. Never extrapolate "
                "beyond data range. Use splines "
                "for better behavior."
            )



def reg_path_widget():
    st.markdown("## Regularization Path")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 100, key="regpath_n")
        n_features = st.slider("Number of Features", 5, 30, 10, key="regpath_k")
        reg_type = st.selectbox(
            "Regularization", ["Lasso (L1)", "Ridge (L2)"], key="regpath_type"
        )
    np.random.seed(42)
    true_coefs = np.zeros(n_features)
    true_coefs[:5] = [3, -2, 1.5, -1, 0.5]
    np.random.shuffle(true_coefs)
    X = np.random.normal(0, 1, (n, n_features))
    y = X @ true_coefs + np.random.normal(0, 1, n)
    from sklearn.linear_model import Lasso, Ridge

    alphas = np.logspace(-2, 2, 100)
    if reg_type == "Lasso (L1)":
        coefs = np.array(
            [Lasso(alpha=a, max_iter=10000).fit(X, y).coef_ for a in alphas]
        )
    else:
        coefs = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
    fig = go.Figure()
    for i in range(n_features):
        fig.add_trace(
            go.Scatter(
                x=np.log10(alphas),
                y=coefs[:, i],
                mode="lines",
                line=dict(width=1.5),
                name=f"Feature {i+1}",
                hovertemplate="log10(α)=%{x:.2f}<br>Coeff=%{y:.3f}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dash"))
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="log10(Alpha)",
        yaxis_title="Coefficient Value",
        hovermode="closest",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each line = one coefficient\n"
                "- Left (low α) = unregularized\n"
                "- Right (high α) = strong shrinkage\n"
                "- Lasso forces coefficients to zero"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- High-dimensional data\n"
                "- Feature selection (Lasso)\n"
                "- Combat multicollinearity (Ridge)\n"
                "- Bias-variance tradeoff analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Cross-validated MSE\n"
                "- Regularization path stability\n"
                "- Bayesian information criterion\n"
                "- Bootstrap coefficient stability"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Lasso selects at most n features. "
                "With p > n, Ridge may generalize "
                "better. Always standardize predictors "
                "before regularization."
            )



def growth_curve_widget():
    st.markdown("## Growth Curve Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Time Points", 10, 100, 30, key="gc_n")
        noise = st.slider("Noise Level", 0.0, 2.0, 0.3, 0.05, key="gc_noise")
        model = st.selectbox(
            "Growth Model", ["Logistic", "Gompertz", "Exponential"], key="gc_model"
        )
    np.random.seed(42)
    t = np.linspace(0, 20, n)
    if model == "Logistic":
        L, k, t0 = 10, 0.5, 10
        y_true = L / (1 + np.exp(-k * (t - t0)))
    elif model == "Gompertz":
        L, k, t0 = 10, 0.3, 5
        y_true = L * np.exp(-np.exp(-k * (t - t0)))
    else:
        r, y0 = 0.2, 0.5
        y_true = y0 * np.exp(r * t)
    y = y_true + np.random.normal(0, noise, n)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="markers",
            marker=dict(color="#4C78A8", size=5, opacity=0.6),
            name="Observed",
            hovertemplate="t=%{x:.1f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_true,
            mode="lines",
            line=dict(color="#E45756", width=3),
            name="True Growth",
            hovertemplate="t=%{x:.1f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time",
        yaxis_title="Size / Population",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- S-curve = logistic growth\n"
                "- Asymptote = carrying capacity\n"
                "- Steepest = max growth rate\n"
                "- Lag → log → stationary phases"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Population growth modeling\n"
                "- Epidemic curve analysis\n"
                "- Learning curve analysis\n"
                "- Biological growth processes"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Non-linear regression F-test\n"
                "- Model comparison (AIC/BIC)\n"
                "- Residual diagnostics\n"
                "- Bootstrap parameter CIs"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Extrapolating beyond observed "
                "data is risky. Asymptote "
                "depends strongly on model "
                "choice."
            )




GRAPHS = {
    "Linear Regression Plot": linear_reg_widget,
    "Multiple Regression Surface": multiple_reg_widget,
    "Logistic Regression Sigmoid Curve": logistic_widget,
    "Multinomial Decision Boundaries": multinomial_widget,
    "Ordinal Logistic Probability Curves": ordinal_logit_widget,
    "Poisson Count Regression": poisson_widget,
    "Residuals vs Fitted Plot": residuals_fitted_widget,
    "Polynomial Regression Fit": poly_reg_widget,
    "Regularization Path": reg_path_widget,
    "Growth Curve Plot": growth_curve_widget
}