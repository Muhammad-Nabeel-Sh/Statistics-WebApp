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


@register_test("Sensitivity & Specificity Analysis")
def render_sensitivity_specificity_analysis(external_data=None):
    st.subheader("Interactive Diagnostic Accuracy Calculator")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("sens_spec", mode="categorical_two")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        if ct.shape != (2, 2):
            st.error("Sensitivity & Specificity requires a 2×2 contingency table. "
                     "Please select two binary categorical variables.")
            return
        tp = ct.iloc[0, 0]
        fn = ct.iloc[1, 0]
        fp = ct.iloc[0, 1]
        tn = ct.iloc[1, 1]
    else:
        col1, col2 = st.columns(2)
        with col1:
            tp = st.number_input("True Positives (TP)", min_value=0, value=80)
            fn = st.number_input("False Negatives (FN)", min_value=0, value=20)
        with col2:
            fp = st.number_input("False Positives (FP)", min_value=0, value=10)
            tn = st.number_input("True Negatives (TN)", min_value=0, value=90)

    # =========================
    # CALCULATIONS
    # =========================
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    LR_positive = (
        sensitivity / (1 - specificity) if (1 - specificity) > 0 else float("inf")
    )
    LR_negative = (
        (1 - sensitivity) / specificity if specificity > 0 else float("inf")
    )
    F1_score = (
        2 * (ppv * sensitivity) / (ppv + sensitivity)
        if (ppv + sensitivity) > 0
        else 0
    )
    DOR = LR_positive / LR_negative if LR_negative > 0 else float("inf")

    # =========================
    # STATS
    # =========================
    cols = st.columns(3)
    cols[0].metric("Sensitivity", f"{sensitivity:.1%}")
    cols[1].metric("Specificity", f"{specificity:.1%}")
    cols[2].metric("Accuracy", f"{accuracy:.1%}")

    cols2 = st.columns(3)
    cols2[0].metric("Pos. Pred. Value (PPV)", f"{ppv:.1%}")
    cols2[1].metric("Neg. Pred. Value (NPV)", f"{npv:.1%}")
    cols2[2].metric("F1 Score", f"{F1_score:.2f}")

    cols3 = st.columns(3)
    cols3[0].metric("LR+", f"{LR_positive:.2f}")
    cols3[1].metric("LR-", f"{LR_negative:.2f}")
    cols3[2].metric("Diagnostic Odds Ratio (DOR)", f"{DOR:.2f}")

    matrix = np.array(
        [
            [tp, fp],
            [fn, tn],
        ]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Negative", "Positive"],
            y=["Negative", "Positive"],
            text=matrix,
            texttemplate="%{text}",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)



@register_test("ROC Curve Analysis")
def render_roc_curve_analysis(external_data=None):
    st.subheader("Interactive ROC Curve Analysis")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("roc_curve", mode="categorical_two")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        if ct.shape != (2, 2):
            st.error("ROC Curve Analysis requires a 2×2 contingency table. "
                     "Please select two binary categorical variables.")
            return
        tp = ct.iloc[0, 0]
        fn = ct.iloc[1, 0]
        fp = ct.iloc[0, 1]
        tn = ct.iloc[1, 1]
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr = np.array([0.0, fpr_val, 1.0])
        tpr = np.array([0.0, tpr_val, 1.0])
        roc_auc = np.trapz(tpr, fpr)
        st.info(f"ROC point from uploaded data: (1−Specificity={fpr_val:.3f}, Sensitivity={tpr_val:.3f})")
    else:
        separation = st.slider("Diagnostic Power (Group Separation)", 0.0, 5.0, 1.5, 0.1)
        np.random.seed(42)
        n = 500
        scores_healthy = np.random.normal(0, 1, n)
        scores_disease = np.random.normal(separation, 1, n)
        y_true = np.concatenate([np.zeros(n), np.ones(n)])
        y_scores = np.concatenate([scores_healthy, scores_disease])
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

    # =========================
    # STATS
    # =========================
    st.metric("Area Under Curve (AUC)", f"{roc_auc:.3f}")

    # =========================
    # PLOT
    # =========================
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC curve (AUC = {roc_auc:.2f})",
            fill="tozeroy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random Guess",
        )
    )

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC)",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)



@register_test("Likelihood Ratio Analysis")
def render_likelihood_ratio_analysis(external_data=None):
    st.subheader("Interactive Likelihood Ratio Analysis")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("likelihood_ratio", mode="categorical_two")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        if ct.shape != (2, 2):
            st.error("Likelihood Ratio Analysis requires a 2×2 contingency table. "
                     "Please select two binary categorical variables.")
            return
        tp = ct.iloc[0, 0]
        fn = ct.iloc[1, 0]
        fp = ct.iloc[0, 1]
        tn = ct.iloc[1, 1]
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        col1, col2 = st.columns(2)
        with col1:
            sens = st.slider("Sensitivity", 0.0, 1.0, 0.8)
        with col2:
            spec = st.slider("Specificity", 0.0, 1.0, 0.9)

    # =========================
    # CALCULATIONS
    # =========================
    lr_pos = sens / (1 - spec) if spec < 1 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")

    # =========================
    # STATS
    # =========================
    c1, c2 = st.columns(2)
    c1.metric("LR+", f"{lr_pos:.2f}")
    c2.metric("LR-", f"{lr_neg:.2f}")

    st.info("""
    - **LR+ > 10**: Strong evidence to rule in disease.
    - **LR- < 0.1**: Strong evidence to rule out disease.
    """)

