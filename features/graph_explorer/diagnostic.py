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

def confusion_widget():
    st.markdown("## Confusion Matrix Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 1000, 200, key="cm_n")
        prevalence = st.slider(
            "Prevalence (True Class 1 Rate)",
            0.05,
            0.95,
            0.3,
            0.05,
            key="cm_prev",
            help="Proportion of actual positives",
        )
        sensitivity = st.slider(
            "Sensitivity (True Positive Rate)", 0.5, 1.0, 0.85, 0.01, key="cm_sens"
        )
        specificity = st.slider(
            "Specificity (True Negative Rate)", 0.5, 1.0, 0.90, 0.01, key="cm_spec"
        )
    np.random.seed(42)
    n_pos = int(n * prevalence)
    n_neg = n - n_pos
    tp = int(n_pos * sensitivity)
    fn = n_pos - tp
    tn = int(n_neg * specificity)
    fp = n_neg - tn
    cm = [[tn, fp], [fn, tp]]
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / n
    labels = ["Predicted Negative", "Predicted Positive"]
    true_labels = ["True Negative", "True Positive"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=true_labels,
            text=[[f"{cm[0][0]}", f"{cm[0][1]}"], [f"{cm[1][0]}", f"{cm[1][1]}"]],
            texttemplate="%{text}",
            textfont=dict(size=16),
            colorscale="Blues",
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.1%}")
    m2.metric("Sensitivity", f"{sensitivity:.1%}")
    m3.metric("Specificity", f"{specificity:.1%}")
    m4.metric("Prevalence", f"{prevalence:.1%}")
    m1.metric("PPV", f"{ppv:.1%}")
    m2.metric("NPV", f"{npv:.1%}")
    m3.metric("TP", tp)
    m4.metric("FP", fp)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Diagonal = correct predictions\n"
                "- Off-diagonal = errors\n"
                "- Top-right (FP) = Type I error\n"
                "- Bottom-left (FN) = Type II error\n"
                "- PPV depends on prevalence"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Evaluate binary classifiers\n"
                "- Compare diagnostic tests\n"
                "- Understand error types\n"
                "- Choose operating threshold"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- McNemar's test (paired comparison)\n"
                "- Cohen's Kappa\n"
                "- ROC-AUC\n"
                "- Diagnostic likelihood ratios"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Accuracy is misleading with imbalanced classes. "
                "A test with 95% accuracy on 5% prevalence "
                "could be useless (always predict negative). "
                "Always report PPV, NPV, and prevalence."
            )



def roc_widget():
    st.markdown("## ROC Curve Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="roc_n")
        auc = st.slider(
            "AUC (Area Under Curve)",
            0.5,
            1.0,
            0.85,
            0.01,
            key="roc_auc",
            help="0.5 = random, 1.0 = perfect",
        )
        show_threshold = st.toggle("Show Optimal Threshold", True, key="roc_thresh")
        show_chance = st.toggle("Show Chance Line", True, key="roc_chance")
    np.random.seed(42)
    n_pos = n // 2
    n_neg = n - n_pos
    d_prime = stats.norm.ppf(auc) * np.sqrt(2) if auc < 0.99 else 5
    scores_pos = np.random.normal(d_prime / 2, 1, n_pos)
    scores_neg = np.random.normal(-d_prime / 2, 1, n_neg)
    scores = np.concatenate([scores_neg, scores_pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    thresholds = np.sort(scores)
    tpr, fpr = [], []
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn_ = np.sum((pred == 0) & (labels == 1))
        tn_ = np.sum((pred == 0) & (labels == 0))
        tpr.append(tp / (tp + fn_) if (tp + fn_) > 0 else 0)
        fpr.append(fp / (fp + tn_) if (fp + tn_) > 0 else 0)
    tpr, fpr = np.array(tpr), np.array(fpr)
    idx = np.argsort(fpr)
    fpr_s, tpr_s = fpr[idx], tpr[idx]
    auc_actual = np.trapz(tpr_s, fpr_s)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr_s,
            y=tpr_s,
            mode="lines",
            name=f"ROC (AUC = {auc_actual:.3f})",
            line=dict(color="#4C78A8", width=3),
            hovertemplate="FPR = %{x:.3f}<br>TPR = %{y:.3f}<extra></extra>",
        )
    )
    if show_chance:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance",
                line=dict(color="rgba(200,200,200,0.5)", dash="dash"),
            )
        )
    if show_threshold:
        youden = tpr_s - fpr_s
        best_idx = np.argmax(youden)
        fig.add_trace(
            go.Scatter(
                x=[fpr_s[best_idx]],
                y=[tpr_s[best_idx]],
                mode="markers",
                name="Optimal Threshold",
                marker=dict(color="#E45756", size=12, symbol="star"),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Curve closer to top-left = better\n"
                "- AUC = probability correct ranking\n"
                "- AUC 0.5 = guessing\n"
                "- AUC 0.8+ = good discrimination\n"
                "- Star = Youden's optimal threshold"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare diagnostic tests\n"
                "- Assess model discrimination\n"
                "- Choose optimal threshold\n"
                "- Meta-analysis of test accuracy"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- DeLong test (compare AUCs)\n"
                "- Hanley-McNeil test\n"
                "- Bootstrap AUC comparison\n"
                "- Sensitivity analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "AUC ignores calibration - a model can have "
                "high AUC but poorly calibrated probabilities. "
                "Always check calibration (calibration plot) "
                "alongside ROC analysis."
            )



def pr_curve_widget():
    st.markdown("## Precision-Recall Curve")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 200, key="pr_n")
        prevalence = st.slider(
            "Prevalence (Class Imbalance)",
            0.02,
            0.5,
            0.1,
            0.01,
            key="pr_prev",
            help="Lower = more imbalanced",
        )
        sep = st.slider("Class Separation", 0.5, 3.0, 1.5, 0.1, key="pr_sep")
        show_baseline = st.toggle("Show Baseline (Prevalence)", True, key="pr_baseline")
    np.random.seed(42)
    n_pos = max(1, int(n * prevalence))
    n_neg = n - n_pos
    scores_pos = np.random.normal(sep / 2, 1, n_pos)
    scores_neg = np.random.normal(-sep / 2, 1, n_neg)
    scores = np.concatenate([scores_neg, scores_pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    thresholds = np.sort(scores)
    prec, rec = [], []
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn_ = np.sum((pred == 0) & (labels == 1))
        prec.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        rec.append(tp / (tp + fn_) if (tp + fn_) > 0 else 1.0)
    prec, rec = np.array(prec)[::-1], np.array(rec)[::-1]
    baseline = n_pos / n
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rec,
            y=prec,
            mode="lines",
            name="PR Curve",
            line=dict(color="#4C78A8", width=3),
            fill="tozeroy",
            hovertemplate="Recall = %{x:.3f}<br>Precision = %{y:.3f}<extra></extra>",
        )
    )
    if show_baseline:
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="rgba(200,200,200,0.5)",
            annotation_text=f"Baseline (Prevalence = {baseline:.2%})",
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Higher curve = better\n"
                "- Baseline = always-predict-positive\n"
                "- PR better than ROC for imbalanced\n"
                "- AP = area under PR curve"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Imbalanced classification\n"
                "- Rare disease detection\n"
                "- Fraud/anomaly detection\n"
                "- When PPV matters more"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Average Precision (AP)\n"
                "- F1 score (harmonic of P,R)\n"
                "- F-beta score (weighted F1)\n"
                "- Bootstrap PR comparison"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "PR curves from small samples are noisy and "
                "can show high precision simply by chance. "
                "Always use confidence bands (bootstrap) "
                "when sample size is limited."
            )



def threshold_widget():
    st.markdown("## Sensitivity vs Specificity Threshold Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 200, key="thresh_n")
        sep = st.slider("Class Separation", 0.5, 3.0, 1.5, 0.1, key="thresh_sep")
        prevalence = st.slider("Prevalence", 0.05, 0.95, 0.3, 0.05, key="thresh_prev")
        cost_fp = st.slider(
            "Cost of FP (relative to FN)",
            0.1,
            10.0,
            1.0,
            0.1,
            key="thresh_cost",
            help="Higher = penalize false positives more",
        )
    np.random.seed(42)
    n_pos = int(n * prevalence)
    n_neg = n - n_pos
    scores_pos = np.random.normal(sep / 2, 1, n_pos)
    scores_neg = np.random.normal(-sep / 2, 1, n_neg)
    scores = np.concatenate([scores_neg, scores_pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    thresholds = np.linspace(min(scores), max(scores), 100)
    sens, spec, costs = [], [], []
    for thresh in thresholds:
        pred = (scores >= thresh).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn_ = np.sum((pred == 0) & (labels == 1))
        tn_ = np.sum((pred == 0) & (labels == 0))
        sens.append(tp / max(tp + fn_, 1))
        spec.append(tn_ / max(tn_ + fp, 1))
        costs.append(fp * cost_fp + fn_ * 1.0)
    opt_idx = np.argmin(costs)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=sens,
            mode="lines",
            name="Sensitivity",
            line=dict(color="#4C78A8", width=2),
            hovertemplate="Threshold = %{x:.2f}<br>Sensitivity = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=spec,
            mode="lines",
            name="Specificity",
            line=dict(color="#E45756", width=2),
            hovertemplate="Threshold = %{x:.2f}<br>Specificity = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=thresholds[opt_idx],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Optimal = {thresholds[opt_idx]:.2f}",
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Threshold",
        yaxis_title="Rate",
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Blue = sensitivity (catch positives)\n"
                "- Red = specificity (avoid false alarms)\n"
                "- Tradeoff: increase one = decrease other\n"
                "- Green = optimal based on costs"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Choose diagnostic cutoff\n"
                "- Balance sensitivity vs specificity\n"
                "- Incorporate cost of errors\n"
                "- Laboratory test thresholds"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- ROC analysis\n"
                "- Youden's index\n"
                "- Cost-benefit analysis\n"
                "- Decision curve analysis"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Youden's index (max sensitivity + specificity - 1) "
                "treats FP and FN equally. In medicine, FN is often "
                "more costly (missed diagnosis). Adjust threshold "
                "based on clinical consequences, not statistics."
            )



def calibration_widget():
    st.markdown("## Calibration Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 100, 2000, 500, key="cal_n")
        calibration = st.slider(
            "Calibration Slope",
            0.0,
            2.0,
            1.0,
            0.05,
            key="cal_slope",
            help="1.0 = perfect, < 1 = overconfident, > 1 = underconfident",
        )
        noise = st.slider("Calibration Noise", 0.0, 0.3, 0.05, 0.01, key="cal_noise")
        n_bins = st.slider("Number of Bins", 5, 20, 10, key="cal_bins")
    np.random.seed(42)
    true_probs = np.random.uniform(0.05, 0.95, n)
    pred_probs = true_probs**calibration
    pred_probs = np.clip(pred_probs + np.random.normal(0, noise, n), 0.01, 0.99)
    y = (np.random.uniform(0, 1, n) < true_probs).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_mean_pred, bin_mean_obs = [], []
    for i in range(n_bins):
        mask = (pred_probs >= bins[i]) & (pred_probs < bins[i + 1])
        if mask.sum() > 0:
            bin_mean_pred.append(np.mean(pred_probs[mask]))
            bin_mean_obs.append(np.mean(y[mask]))
        else:
            bin_mean_pred.append(bin_centers[i])
            bin_mean_obs.append(bin_centers[i])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bin_mean_pred,
            y=bin_mean_obs,
            mode="markers+lines",
            name="Model",
            marker=dict(color="#4C78A8", size=10),
            line=dict(color="#4C78A8", width=2),
            hovertemplate="Predicted = %{x:.3f}<br>Observed = %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="rgba(200,200,200,0.5)", dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Proportion",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Points on diagonal = perfectly calibrated\n"
                "- Above diagonal = underestimated probability\n"
                "- Below diagonal = overestimated probability\n"
                "- Slope < 1 = overconfident (common)"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Assess probability accuracy\n"
                "- Check model reliability\n"
                "- Compare risk prediction models\n"
                "- Before clinical deployment"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Hosmer-Lemeshow test\n"
                "- Brier score\n"
                "- Spiegelhalter z-test\n"
                "- Calibration intercept & slope"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Hosmer-Lemeshow test is sensitive to binning "
                "choices and sample size. Large samples will "
                "reject even well-calibrated models. Use "
                "calibration plots + intercept/slope instead."
            )


# --- AGREEMENT PLOTS ---




GRAPHS = {
    "Confusion Matrix Explorer": confusion_widget,
    "ROC Curve Explorer": roc_widget,
    "Precision-Recall Curve": pr_curve_widget,
    "Sensitivity-Specificity Threshold Explorer": threshold_widget,
    "Calibration Plot": calibration_widget
}