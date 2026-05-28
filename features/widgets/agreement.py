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


@register_test("Cohen's Kappa (Agreement Analysis)")
def render_cohen_s_kappa_agreement_analysis(external_data=None):
    st.subheader("Interactive Agreement Analysis (Cohen's Kappa)")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("cohen_kappa", mode="categorical_two")

    # =========================
    # DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        if ct.shape != (2, 2):
            st.error("Cohen's Kappa requires a 2×2 contingency table. "
                     "Please select two binary categorical variables.")
            return
        yy = ct.iloc[0, 0]
        yn = ct.iloc[0, 1]
        ny = ct.iloc[1, 0]
        nn = ct.iloc[1, 1]
    else:
        st.write("Enter agreement counts between two raters:")
        c1, c2 = st.columns(2)
        with c1:
            yy = st.number_input("Both say YES", min_value=0, value=40, key="kappa_yy")
            yn = st.number_input(
                "Rater 1 says YES, Rater 2 says NO", min_value=0, value=10, key="kappa_yn"
            )
        with c2:
            ny = st.number_input(
                "Rater 1 says NO, Rater 2 says YES", min_value=0, value=5, key="kappa_ny"
            )
            nn = st.number_input("Both say NO", min_value=0, value=45, key="kappa_nn")

    # =========================
    # CALCULATIONS
    # =========================
    total = yy + yn + ny + nn
    if total > 0:
        po = (yy + nn) / total
        pe = ((yy + yn) * (yy + ny) + (ny + nn) * (yn + nn)) / (total * total)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    else:
        kappa = 0

    # =========================
    # STATS
    # =========================
    st.metric("Cohen's Kappa (κ)", f"{kappa:.3f}")

    if kappa > 0.8:
        interpretation = "Almost Perfect Agreement"
    elif kappa > 0.6:
        interpretation = "Substantial Agreement"
    elif kappa > 0.4:
        interpretation = "Moderate Agreement"
    elif kappa > 0.2:
        interpretation = "Fair Agreement"
    else:
        interpretation = "Slight/Poor Agreement"

    st.success(f"Interpretation: {interpretation}")



@register_test("Fleiss' Kappa")
def render_fleiss_kappa(external_data=None):

    st.subheader("Interactive Fleiss' Kappa — Multi-Rater Agreement")
    st.info("""
    **Fleiss' Kappa** measures agreement among **3+ raters** assigning nominal ratings
    to the same subjects. Unlike Cohen's Kappa (which handles 2 raters), Fleiss' Kappa
    generalizes to any number of raters.
    """)

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("fleiss_kappa", mode="categorical_two")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        st.warning("Fleiss' Kappa requires per-subject rater-by-category data "
                   "(subjects × raters, not a contingency table). Using contingency table as "
                   "the agreement matrix for demonstration.")
        table_fk = ct.values.astype(float)
        n_subjects_fk, n_cat_fk = table_fk.shape
        n_raters_fk = st.slider("Number of Raters", 3, 10, 3, key="fk_raters",
                                help="Assuming each subject is rated by this many raters")
    else:
        n_subjects_fk = st.slider("Number of Subjects", 10, 200, 30, key="fk_n")
        n_raters_fk = st.slider("Number of Raters", 3, 10, 3, key="fk_raters_sim")
        n_cat_fk = st.slider("Number of Categories", 2, 5, 3, key="fk_cat")
        agreement_p = st.slider("Agreement Probability", 0.0, 1.0, 0.7, 0.01, key="fk_agree",
                                help="How often raters agree on the same category")
        np.random.seed(42)
        true_cat = np.random.randint(0, n_cat_fk, n_subjects_fk)
        table_fk = np.zeros((n_subjects_fk, n_cat_fk), dtype=float)
        for i in range(n_subjects_fk):
            ratings = np.random.choice(n_cat_fk, n_raters_fk, p=[
                agreement_p + (1 - agreement_p) / n_cat_fk if j == true_cat[i]
                else (1 - agreement_p) / n_cat_fk for j in range(n_cat_fk)
            ])
            for r in ratings:
                table_fk[i, r] += 1

    # Fleiss' Kappa computation
    n = table_fk.shape[0]
    k = table_fk.shape[1]
    N = n_raters_fk * n

    p_j = table_fk.sum(axis=0) / N
    P_i = (table_fk ** 2).sum(axis=1) - n_raters_fk
    P_i = P_i / (n_raters_fk * (n_raters_fk - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    fleiss_kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0.0

    # Variance approximation
    var_kappa = 2 / (n * n_raters_fk * (n_raters_fk - 1)) * (
        (P_bar - (2 * n_raters_fk - 1) * P_e / (n_raters_fk - 1)) /
        (1 - P_e) ** 2
    ) if P_e < 1 and (1 - P_e) > 0 else 0
    se_kappa = np.sqrt(max(var_kappa, 0))
    z_fk = fleiss_kappa / se_kappa if se_kappa > 0 else 0
    from scipy.stats import norm as norm_fk
    p_fk = 2 * (1 - norm_fk.cdf(abs(z_fk)))

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Fleiss' κ", f"{fleiss_kappa:.4f}")
    c2.metric("SE", f"{se_kappa:.4f}")
    c3.metric("p-value", format_p_value(p_fk))

    if fleiss_kappa > 0.8:
        fk_interp = "Almost Perfect Agreement"
    elif fleiss_kappa > 0.6:
        fk_interp = "Substantial Agreement"
    elif fleiss_kappa > 0.4:
        fk_interp = "Moderate Agreement"
    elif fleiss_kappa > 0.2:
        fk_interp = "Fair Agreement"
    else:
        fk_interp = "Slight/Poor Agreement"
    st.success(f"**Interpretation:** {fk_interp}")

    with st.expander("Category Statistics"):
        cat_df = pd.DataFrame({
            "Category": [f"Cat {j+1}" for j in range(k)],
            "p_j (marginal)": [f"{pj:.4f}" for pj in p_j],
        })
        st.dataframe(cat_df, use_container_width=True)

    st.caption(f"Based on {n} subjects × {int(n_raters_fk)} raters, {k} categories")



@register_test("Weighted Kappa")
def render_weighted_kappa(external_data=None):

    from scipy.stats import norm as norm_wk

    st.subheader("Interactive Weighted Kappa — Ordinal Agreement")
    st.info("""
    **Weighted Kappa** extends Cohen's Kappa to **ordinal** categories by penalizing
    disagreements proportionally to their severity (e.g., a 2-category disagreement
    is worse than a 1-category disagreement). Uses quadratic weights.
    """)

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("weighted_kappa", mode="categorical_two")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        ct = src["data"]["contingency_table"]
        if ct.shape[0] != ct.shape[1]:
            st.error("Weighted Kappa requires a square contingency table "
                     "(same categories for both raters).")
            return
        table_wk = ct.values.astype(float)
        n_cat_wk = table_wk.shape[0]
        cat_labels_wk = list(ct.index) if hasattr(ct, 'index') else [f"Cat {i+1}" for i in range(n_cat_wk)]
    else:
        n_cat_wk = st.slider("Number of Ordinal Categories", 2, 7, 4, key="wk_ncat")
        table_wk = np.zeros((n_cat_wk, n_cat_wk), dtype=float)
        for r in range(n_cat_wk):
            for c in range(n_cat_wk):
                default_val = max(5 - abs(r - c) * 3, 1)
                table_wk[r, c] = st.number_input(
                    f"Rater1={r+1}, Rater2={c+1}", min_value=0, value=int(default_val),
                    key=f"wk_cell_{r}_{c}"
                )
        cat_labels_wk = [f"Cat {i+1}" for i in range(n_cat_wk)]

    # Quadratic weights
    n_lev = n_cat_wk
    weights = np.zeros((n_lev, n_lev))
    for i in range(n_lev):
        for j in range(n_lev):
            weights[i, j] = 1 - (i - j) ** 2 / (n_lev - 1) ** 2

    total_wk = table_wk.sum()
    prop_wk = table_wk / total_wk if total_wk > 0 else table_wk

    # Observed weighted agreement
    po_wk = (weights * prop_wk).sum()

    # Expected agreement (under independence)
    row_marg = prop_wk.sum(axis=1)
    col_marg = prop_wk.sum(axis=0)
    pe_wk = (weights * np.outer(row_marg, col_marg)).sum()

    wk = (po_wk - pe_wk) / (1 - pe_wk) if pe_wk < 1 else 0.0

    # Variance (Fleiss, Cohen & Everitt 1969 approximation)
    var_wk = 0
    if pe_wk < 1 and total_wk > 0:
        var_wk = (po_wk * (1 - po_wk)) / (total_wk * (1 - pe_wk) ** 2)
    se_wk = np.sqrt(max(var_wk, 0))
    z_wk = wk / se_wk if se_wk > 0 else 0
    p_wk = 2 * (1 - norm_wk.cdf(abs(z_wk)))

    # Unweighted kappa for comparison
    po_unw = np.trace(prop_wk)
    pe_unw = (row_marg * col_marg).sum()
    unw_kappa = (po_unw - pe_unw) / (1 - pe_unw) if pe_unw < 1 else 0.0

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Weighted κ (quadratic)", f"{wk:.4f}")
    c2.metric("Unweighted κ (comparison)", f"{unw_kappa:.4f}")
    c3.metric("p-value", format_p_value(p_wk))

    if wk > 0.8:
        wk_interp = "Almost Perfect Agreement"
    elif wk > 0.6:
        wk_interp = "Substantial Agreement"
    elif wk > 0.4:
        wk_interp = "Moderate Agreement"
    elif wk > 0.2:
        wk_interp = "Fair Agreement"
    else:
        wk_interp = "Slight/Poor Agreement"
    st.success(f"**Interpretation:** {wk_interp}")

    with st.expander("Weight Matrix (quadratic)"):
        wt_df = pd.DataFrame(weights, index=cat_labels_wk, columns=cat_labels_wk)
        st.dataframe(wt_df.style.format("{:.3f}"), use_container_width=True)

    st.caption(f"Total observations: {int(total_wk)}")



@register_test("Bland-Altman Analysis")
def render_bland_altman_analysis(external_data=None):

    st.subheader("Interactive Bland-Altman Analysis")

    # =========================
    # DATA SOURCE TOGGLE
    # =========================

    if external_data and external_data.get("using_uploaded"):
        src = external_data
    else:
        src = data_source_toggle("bland_altman", mode="paired")

    # =========================
    # CONTROLS / DATA
    # =========================

    if src["using_uploaded"]:
        method1 = np.array(src["data"]["values1"])
        method2 = np.array(src["data"]["values2"])
    else:
        bias = st.slider("Bias (Mean Difference)", -5.0, 5.0, 0.2, 0.1)
        agreement_sd = st.slider("SD of Differences", 0.1, 5.0, 1.0, 0.1)
        n = st.slider("Sample Size", 10, 200, 50, key="bland_altman_analysis_sample_size")

        np.random.seed(42)
        true_val = np.random.uniform(10, 50, n)
        diff = np.random.normal(bias, agreement_sd, n)
        method1 = true_val - diff / 2
        method2 = true_val + diff / 2

    mean_pair = (method1 + method2) / 2
    diff_pair = method1 - method2

    mean_diff = np.mean(diff_pair)
    sd_diff = np.std(diff_pair, ddof=1)
    upper_loa = mean_diff + 1.96 * sd_diff
    lower_loa = mean_diff - 1.96 * sd_diff

    # =========================
    # STATS
    # =========================

    cols = st.columns(3)
    cols[0].metric("Mean Difference (Bias)", f"{mean_diff:.3f}")
    cols[1].metric("Upper LoA (+1.96 SD)", f"{upper_loa:.3f}")
    cols[2].metric("Lower LoA (−1.96 SD)", f"{lower_loa:.3f}")

    # =========================
    # PLOT
    # =========================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mean_pair,
            y=diff_pair,
            mode="markers",
            name="Differences",
        )
    )

    fig.add_hline(y=mean_diff, line_dash="solid", annotation_text="Bias")

    fig.add_hline(y=upper_loa, line_dash="dash", annotation_text="+1.96 SD")

    fig.add_hline(y=lower_loa, line_dash="dash", annotation_text="−1.96 SD")

    fig.update_layout(
        template="plotly_dark",
        height=550,
        xaxis_title="Mean of Two Measurements",
        yaxis_title="Difference (Method 1 − Method 2)",
    )

    st.plotly_chart(fig, use_container_width=True)

