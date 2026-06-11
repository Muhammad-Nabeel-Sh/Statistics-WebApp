import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats as scipy_stats
from core.utils import _apa_table, format_p_value

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "study_designs.json"
)

BIAS_LABELS = {
    "selection_bias": "Selection Bias",
    "information_bias": "Information Bias",
    "confounding": "Confounding",
    "performance_bias": "Performance Bias",
    "detection_bias": "Detection Bias",
    "attrition_bias": "Attrition Bias",
    "reporting_bias": "Reporting Bias",
    "lead_time_bias": "Lead-Time Bias",
    "length_time_bias": "Length-Time Bias",
}

BIAS_DESCRIPTIONS = {
    "selection_bias": "Systematic differences between comparison groups in who is selected or who participates",
    "information_bias": "Systematic differences in how exposure or outcome information is collected",
    "confounding": "A third variable associated with both exposure and outcome that distorts the true relationship",
    "performance_bias": "Systematic differences in care, procedures, or co-interventions between groups",
    "detection_bias": "Systematic differences in how outcomes are assessed between groups",
    "attrition_bias": "Systematic differences in who drops out or is lost to follow-up between groups",
    "reporting_bias": "Selective reporting of outcomes, analyses, or subgroups based on results",
    "lead_time_bias": "Earlier detection creates artificial improvements in survival or outcomes",
    "length_time_bias": "Overrepresentation of less aggressive, slower-progressing disease in screening-detected cases",
}

RISK_ORDER = {"Very High": 0, "High": 1, "Moderate": 2, "Low": 3, "N/A": 4}
RISK_COLORS = {
    "Very High": "rgba(228,87,86,0.85)",
    "High": "rgba(228,87,86,0.55)",
    "Moderate": "rgba(241,197,55,0.65)",
    "Low": "rgba(76,120,168,0.55)",
    "N/A": "rgba(128,128,128,0.25)",
}

SESSION_KEY = "design_section"
DESIGN_KEY = "selected_design"


def _load_designs():
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _section_button(label, key_prefix):
    if st.sidebar.button(
        label,
        key=f"{key_prefix}_btn_{label.replace(' ', '_').replace('/', '_')}",
        use_container_width=True,
        type="primary" if st.session_state.get(SESSION_KEY) == label else "secondary",
    ):
        st.session_state[SESSION_KEY] = label
        st.session_state[DESIGN_KEY] = None
        st.rerun()


# ── Outcome Measures ──────────────────────────────────────────────

_OUTCOME_MEASURES = {
    "Randomized Controlled Trial (RCT)": {
        "title": "Risk Ratio & Risk Difference",
        "formulas": [
            r"RR = \frac{a/(a+b)}{c/(c+d)}",
            r"RD = \frac{a}{a+b} - \frac{c}{c+d}",
            r"NNT = \frac{1}{|RD|}",
        ],
        "description": "For a binary outcome, the effect is summarized as the risk ratio (RR), risk difference (RD), and number needed to treat (NNT). Adjust sliders to see how cell counts affect the estimates.",
        "widget": "binary_2x2",
    },
    "Prospective Cohort Study": {
        "title": "Cumulative Incidence & Relative Risk",
        "formulas": [
            r"CI_e = \frac{a}{a+b} \quad CI_u = \frac{c}{c+d}",
            r"RR = \frac{CI_e}{CI_u}",
            r"AR = CI_e - CI_u",
        ],
        "description": "Cumulative incidence in exposed (CIₑ) vs unexposed (CIᵤ), relative risk (RR), and attributable risk (AR).",
        "widget": "binary_2x2",
    },
    "Retrospective Cohort Study": {
        "title": "Relative Risk & Odds Ratio",
        "formulas": [
            r"RR = \frac{a/(a+b)}{c/(c+d)}",
            r"OR = \frac{a/c}{b/d} = \frac{ad}{bc}",
        ],
        "description": "Both RR and OR can be computed in a retrospective cohort when the sampling frame is known.",
        "widget": "binary_2x2",
    },
    "Case-Control Study": {
        "title": "Odds Ratio",
        "formulas": [
            r"OR = \frac{a/c}{b/d} = \frac{ad}{bc}",
        ],
        "description": "In a case-control study, the odds ratio is the only valid measure of association. RR cannot be computed because cases and controls are sampled separately.",
        "widget": "binary_or",
    },
    "Nested Case-Control Study": {
        "title": "Conditional Odds Ratio",
        "formulas": [
            r"OR_{MLE} = \frac{ad}{bc}",
            r"OR_{MH} = \frac{\sum a_i d_i / N_i}{\sum b_i c_i / N_i}",
        ],
        "description": "Odds ratio estimated from matched sets within a cohort. The Mantel-Haenszel estimator combines information across matched strata.",
        "widget": "binary_or",
    },
    "Cross-Sectional Study": {
        "title": "Prevalence & Prevalence Ratio",
        "formulas": [
            r"P = \frac{\text{cases}}{\text{total population}}",
            r"PR = \frac{P_{exposed}}{P_{unexposed}}",
            r"OR = \frac{ad}{bc}",
        ],
        "description": "Prevalence (P), prevalence ratio (PR), and odds ratio (OR) are the typical measures in cross-sectional designs.",
        "widget": "binary_2x2",
    },
    "Case Series / Case Report": {
        "title": "Proportion & Confidence Interval",
        "formulas": [
            r"p = \frac{k}{n}",
            r"CI_{\text{Wilson}} = \frac{p + \frac{z^2}{2n} \pm z\sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}",
        ],
        "description": "A case series reports the proportion of participants with a given outcome. The Wilson confidence interval is recommended for binomial proportions.",
        "widget": "single_proportion",
    },
    "Crossover Trial": {
        "title": "Mean Difference (Paired)",
        "formulas": [
            r"\bar{d} = \frac{1}{n}\sum_{i=1}^n (X_{i,A} - X_{i,B})",
            r"t = \frac{\bar{d}}{s_d / \sqrt{n}}",
            r"d_z = \frac{\bar{d}}{s_d}",
        ],
        "description": "Each participant serves as their own control. The paired t-test evaluates whether the mean within-person difference differs from zero.",
        "widget": "paired_continuous",
    },
    "Factorial Design": {
        "title": "Main Effects & Interaction",
        "formulas": [
            r"A_{\text{effect}} = \frac{1}{2}(\bar{Y}_{A+} - \bar{Y}_{A-})",
            r"AB_{\text{interaction}} = \frac{1}{2}(\bar{Y}_{A+B+} - \bar{Y}_{A+B-} - \bar{Y}_{A-B+} + \bar{Y}_{A-B-})",
        ],
        "description": "Factorial designs estimate main effects of each factor and their interaction. A 2×2 layout with sliders shows the four cell means.",
        "widget": "factorial_2x2",
    },
    "Diagnostic Accuracy Study": {
        "title": "Sensitivity & Specificity",
        "formulas": [
            r"Se = \frac{TP}{TP + FN} \quad Sp = \frac{TN}{TN + FP}",
            r"PPV = \frac{TP}{TP + FP} \quad NPV = \frac{TN}{TN + FN}",
            r"LR+ = \frac{Se}{1-Sp} \quad LR- = \frac{1-Se}{Sp}",
        ],
        "description": "Sensitivity and specificity measure intrinsic test accuracy. PPV and NPV depend on disease prevalence. Likelihood ratios combine both dimensions.",
        "widget": "diagnostic",
    },
    "Non-inferiority / Equivalence Trial": {
        "title": "Non-inferiority Margin & CI",
        "formulas": [
            r"\Delta = p_{\text{new}} - p_{\text{control}}",
            r"H_0: \Delta \leq -\delta \quad H_1: \Delta > -\delta",
        ],
        "description": "Non-inferiority is declared when the lower bound of the CI for the difference exceeds the pre-specified margin −δ. Equivalence requires the entire CI to lie within ±δ.",
        "widget": "non_inferiority",
    },
    "Single-Arm Trial": {
        "title": "Response Rate",
        "formulas": [
            r"p = \frac{\text{responders}}{n}",
            r"CI_{\text{Wilson}} = \frac{p + \frac{z^2}{2n} \pm z\sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}",
        ],
        "description": "The primary endpoint is typically the proportion of patients achieving a pre-defined response, reported with a Wilson confidence interval.",
        "widget": "single_proportion",
    },
    "Cluster Randomized Trial (CRT)": {
        "title": "Intracluster Correlation & Design Effect",
        "formulas": [
            r"ICC = \frac{\sigma^2_b}{\sigma^2_b + \sigma^2_w}",
            r"DE = 1 + (m-1) \times ICC",
            r"n_{\text{eff}} = \frac{n}{DE}",
        ],
        "description": "The intracluster correlation coefficient (ICC) measures within-cluster similarity. The design effect (DE) inflates the required sample size to account for clustering.",
        "widget": "icc_calculator",
    },
    "Stepped Wedge Design": {
        "title": "Time-Adjusted Treatment Effect",
        "formulas": [
            r"\log\left(\frac{p_{ij}}{1-p_{ij}}\right) = \beta_0 + \beta_1 \cdot X_{ij} + \theta_t + u_i",
        ],
        "description": "In a stepped wedge, the treatment effect is estimated from a mixed model adjusting for time period (θₜ) and cluster random effects (uᵢ). The widget below shows a simplified binary outcome analysis.",
        "widget": "binary_2x2",
    },
    "Systematic Review": {
        "title": "Summary Effect Size",
        "formulas": [
            r"\bar{\theta} = \frac{\sum w_i \theta_i}{\sum w_i}",
            r"w_i = \frac{1}{v_i + \tau^2}",
        ],
        "description": "Individual study effects (θᵢ) are weighted by precision. Under a random-effects model, τ² accounts for between-study heterogeneity.",
        "widget": "meta_forest",
    },
    "Meta-Analysis": {
        "title": "Pooled Effect & Heterogeneity",
        "formulas": [
            r"\bar{\theta}_{FE} = \frac{\sum w_i \theta_i}{\sum w_i}, \quad w_i = \frac{1}{v_i}",
            r"\bar{\theta}_{RE} = \frac{\sum w_i^* \theta_i}{\sum w_i^*}, \quad w_i^* = \frac{1}{v_i + \hat{\tau}^2}",
            r"I^2 = \frac{Q - (k-1)}{Q} \times 100\%",
        ],
        "description": "Fixed-effect (FE) and random-effects (RE) pooled estimates. I² quantifies the proportion of total variation due to heterogeneity.",
        "widget": "meta_forest",
    },
    "Narrative Review": {
        "title": "Narrative Synthesis",
        "formulas": [],
        "description": "A narrative review synthesizes findings qualitatively rather than quantitatively. Outcome measures vary by topic and are reported descriptively. No single formula applies across all narrative reviews.",
        "widget": "descriptive_only",
    },
    "Scoping Review": {
        "title": "Evidence Mapping",
        "formulas": [],
        "description": "Scoping reviews map the literature without quantitative synthesis. Outcome measures are reported as counts of studies, thematic summaries, and knowledge gaps. No single formula applies.",
        "widget": "descriptive_only",
    },
    "Pilot / Feasibility Study": {
        "title": "Feasibility Metrics",
        "formulas": [
            r"\text{Recruitment Rate} = \frac{\text{enrolled}}{\text{eligible} \times \text{time}}",
            r"\text{Retention Rate} = \frac{\text{completed}}{\text{enrolled}}",
            r"\text{Fidelity} = \frac{\text{sessions delivered}}{\text{sessions planned}}",
        ],
        "description": "Pilot studies estimate feasibility parameters — recruitment, retention, protocol fidelity — to inform the design of a definitive trial.",
        "widget": "pilot_metrics",
    },
    "Ecological Study": {
        "title": "Standardized Mortality/Morbidity Ratio",
        "formulas": [
            r"SMR = \frac{O}{E}",
            r"CI_{\text{Byar}} = \left( O \left(1 - \frac{1}{9O} - \frac{z}{3\sqrt{O}}\right)^3, (O+1)\left(1 - \frac{1}{9(O+1)} + \frac{z}{3\sqrt{O+1}}\right)^3 \right)",
        ],
        "description": "The standardized mortality ratio (SMR) compares observed events (O) to expected events (E) based on a reference population rate.",
        "widget": "ecological_smr",
    },
    "Critical Review": {
        "title": "Critical Appraisal",
        "formulas": [],
        "description": "Critical reviews evaluate the quality and strength of existing evidence. Outcome measures are not pooled; instead, studies are assessed for bias, relevance, and methodological rigor.",
        "widget": "descriptive_only",
    },
    "State-of-the-Art Review": {
        "title": "Current State Synthesis",
        "formulas": [],
        "description": "State-of-the-art reviews summarize the current knowledge landscape. Quantitative outcome measures are not typically the focus; the review highlights trends, controversies, and future directions.",
        "widget": "descriptive_only",
    },
}


def _render_binary_2x2(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    a = st.slider("Exposed + Outcome", 0, 200, 40, key=f"{key_prefix}_a")
    b = st.slider("Exposed + No Outcome", 0, 200, 60, key=f"{key_prefix}_b")
    c = st.slider("Unexposed + Outcome", 0, 200, 20, key=f"{key_prefix}_c")
    d = st.slider("Unexposed + No Outcome", 0, 200, 80, key=f"{key_prefix}_d")

    p1 = a / (a + b) if (a + b) > 0 else 0
    p2 = c / (c + d) if (c + d) > 0 else 0
    rr = p1 / p2 if p2 > 0 else float("inf")
    rd = p1 - p2
    or_val = (a * d) / (b * c) if (b * c) > 0 else float("inf")
    nnt = abs(1 / rd) if rd != 0 else float("inf")

    tbl = np.array([[a, b], [c, d]])

    fig = go.Figure(data=go.Heatmap(z=tbl, text=tbl, texttemplate="%{text}", x=["Outcome+", "Outcome-"], y=["Exposed+", "Exposed-"]))
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Measure": ["Risk (Exposed)", "Risk (Unexposed)", "Risk Ratio (RR)", "Risk Difference (RD)", "Odds Ratio (OR)", "NNT"],
        "Value": [f"{p1:.3f}", f"{p2:.3f}", f"{rr:.3f}", f"{rd:+.3f}", f"{or_val:.3f}", f"{nnt:.1f}" if nnt != float("inf") else "∞"],
    }
    st.table(pd.DataFrame(data))


def _render_binary_or(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    a = st.slider("Cases + Exposed", 0, 200, 30, key=f"{key_prefix}_or_a")
    b = st.slider("Cases + Unexposed", 0, 200, 10, key=f"{key_prefix}_or_b")
    c = st.slider("Controls + Exposed", 0, 200, 20, key=f"{key_prefix}_or_c")
    d = st.slider("Controls + Unexposed", 0, 200, 40, key=f"{key_prefix}_or_d")

    or_val = (a * d) / (b * c) if (b * c) > 0 else float("inf")
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_low = np.exp(np.log(or_val) - 1.96 * se_log_or)
    ci_high = np.exp(np.log(or_val) + 1.96 * se_log_or)

    tbl = np.array([[a, b], [c, d]])
    fig = go.Figure(data=go.Heatmap(z=tbl, text=tbl, texttemplate="%{text}", x=["Exposed+", "Exposed-"], y=["Cases", "Controls"]))
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Measure": ["Odds Ratio (OR)", "95% CI (OR)", "log(OR)", "SE(log OR)"],
        "Value": [f"{or_val:.3f}", f"[{ci_low:.3f}, {ci_high:.3f}]", f"{np.log(or_val):.3f}", f"{se_log_or:.3f}"],
    }
    st.table(pd.DataFrame(data))


def _render_diagnostic(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    tp = st.slider("True Positives (TP)", 0, 200, 80, key=f"{key_prefix}_tp")
    fp = st.slider("False Positives (FP)", 0, 200, 20, key=f"{key_prefix}_fp")
    fn = st.slider("False Negatives (FN)", 0, 200, 10, key=f"{key_prefix}_fn")
    tn = st.slider("True Negatives (TN)", 0, 200, 90, key=f"{key_prefix}_tn")

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")

    tbl = np.array([[tp, fp], [fn, tn]])
    fig = go.Figure(data=go.Heatmap(z=tbl, text=tbl, texttemplate="%{text}", x=["Predicted+", "Predicted-"], y=["True+", "True-"]))
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Measure": ["Sensitivity", "Specificity", "PPV", "NPV", "LR+", "LR-"],
        "Value": [f"{sens:.3f}", f"{spec:.3f}", f"{ppv:.3f}", f"{npv:.3f}", f"{lr_pos:.3f}" if lr_pos != float("inf") else "∞", f"{lr_neg:.3f}" if lr_neg != float("inf") else "∞"],
    }
    st.table(pd.DataFrame(data))


def _render_single_proportion(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    n = st.slider("Total (n)", 1, 500, 100, key=f"{key_prefix}_n")
    k = st.slider("Events (k)", 0, n, 30, key=f"{key_prefix}_k")

    p = k / n
    z = 1.96
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    ci_low = (center - margin) / denom
    ci_high = (center + margin) / denom

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Proportion"], y=[p], text=[f"{p:.1%}"], textposition="outside",
                         marker_color="rgba(54, 162, 235, 0.7)"))
    fig.add_hline(y=ci_low, line_dash="dash", line_color="gray", annotation_text=f"CI lower {ci_low:.3f}")
    fig.add_hline(y=ci_high, line_dash="dash", line_color="gray", annotation_text=f"CI upper {ci_high:.3f}")
    fig.update_layout(template="plotly_dark", height=350, yaxis_range=[0, 1], yaxis_title="Proportion")
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Measure": ["Proportion (p)", "Wilson CI Lower", "Wilson CI Upper"],
        "Value": [f"{p:.4f}", f"{ci_low:.4f}", f"{ci_high:.4f}"],
    }
    st.table(pd.DataFrame(data))


def _render_paired_continuous(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    mean_diff = st.slider("Mean Difference (d̄)", -50.0, 50.0, 5.0, step=0.5, key=f"{key_prefix}_md")
    sd_diff = st.slider("SD of Differences (sₙ)", 1.0, 50.0, 15.0, step=0.5, key=f"{key_prefix}_sd")
    n = st.slider("Sample Size (n)", 3, 200, 30, key=f"{key_prefix}_n")

    se = sd_diff / np.sqrt(n)
    t_stat = mean_diff / se
    p_val = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))
    d_z = mean_diff / sd_diff
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se

    x = np.linspace(-4, 4, 200)
    y = scipy_stats.t.pdf(x, df=n - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="t-distribution", line=dict(color="rgba(54, 162, 235, 0.7)")))
    fig.add_vline(x=t_stat, line_dash="dash", line_color="red", annotation_text=f"t = {t_stat:.2f}")
    fig.update_layout(template="plotly_dark", height=300, xaxis_title="t", yaxis_title="Density")
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Measure": ["Mean Difference", "SE of Difference", "t-statistic", "p-value", "Cohen's d_z", "95% CI (diff)"],
        "Value": [f"{mean_diff:.3f}", f"{se:.3f}", f"{t_stat:.3f}", f"{p_val:.5f}", f"{d_z:.3f}", f"[{ci_low:.3f}, {ci_high:.3f}]"],
    }
    st.table(pd.DataFrame(data))


def _render_non_inferiority(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    p_new = st.slider("Proportion (New)", 0.0, 1.0, 0.75, step=0.01, key=f"{key_prefix}_pnew")
    p_ctrl = st.slider("Proportion (Control)", 0.0, 1.0, 0.70, step=0.01, key=f"{key_prefix}_pctrl")
    n_new = st.slider("Sample Size (New)", 10, 500, 200, key=f"{key_prefix}_nnew")
    n_ctrl = st.slider("Sample Size (Control)", 10, 500, 200, key=f"{key_prefix}_nctrl")
    margin = st.slider("Non-inferiority Margin (δ)", 0.01, 0.20, 0.10, step=0.01, key=f"{key_prefix}_margin")

    diff = p_new - p_ctrl
    p_pool = (p_new * n_new + p_ctrl * n_ctrl) / (n_new + n_ctrl)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_new + 1 / n_ctrl))
    z_stat = (diff + margin) / se
    p_val = 1 - scipy_stats.norm.cdf(z_stat)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se

    ni_concluded = ci_low > -margin

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[diff], y=[0], mode="markers+text", text=[f"Δ = {diff:.3f}"], textposition="top center",
                             marker=dict(size=14, color="green" if ni_concluded else "red")))
    fig.add_vline(x=0, line_dash="solid", line_color="white", annotation_text="No diff")
    fig.add_vline(x=-margin, line_dash="dash", line_color="orange", annotation_text=f"-δ = {-margin:.2f}")
    fig.add_vline(x=margin, line_dash="dash", line_color="orange", annotation_text=f"+δ = {margin:.2f}")
    fig.add_hrect(y0=-0.5, y1=0.5, x0=ci_low, x1=ci_high, fillcolor="rgba(0,255,0,0.1)" if ni_concluded else "rgba(255,0,0,0.1)", line_width=0)
    fig.update_layout(template="plotly_dark", height=300, xaxis_title="Risk Difference", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    verdict = "✅ Non-inferiority demonstrated" if ni_concluded else "❌ Non-inferiority NOT demonstrated"
    st.markdown(f"**Verdict:** {verdict}")

    data = {
        "Measure": ["Difference (New − Control)", "95% CI", "z-statistic", "p-value (one-sided)", "Margin (δ)"],
        "Value": [f"{diff:+.4f}", f"[{ci_low:.4f}, {ci_high:.4f}]", f"{z_stat:.3f}", f"{p_val:.5f}", f"{margin:.3f}"],
    }
    st.table(pd.DataFrame(data))


def _render_factorial_2x2(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    st.markdown("**Cell Means (outcome)**")
    y11 = st.slider("Factor A+, B+", 0.0, 100.0, 65.0, step=0.5, key=f"{key_prefix}_y11")
    y10 = st.slider("Factor A+, B−", 0.0, 100.0, 55.0, step=0.5, key=f"{key_prefix}_y10")
    y01 = st.slider("Factor A−, B+", 0.0, 100.0, 50.0, step=0.5, key=f"{key_prefix}_y01")
    y00 = st.slider("Factor A−, B−", 0.0, 100.0, 45.0, step=0.5, key=f"{key_prefix}_y00")

    a_eff = ((y11 + y10) - (y01 + y00)) / 2
    b_eff = ((y11 + y01) - (y10 + y00)) / 2
    ab_int = ((y11 - y10) - (y01 - y00)) / 2

    tbl = np.array([[y11, y10], [y01, y00]])
    fig = go.Figure(data=go.Heatmap(z=tbl, text=tbl.round(1), texttemplate="%{text}",
                                     x=["B+", "B−"], y=["A+", "A−"]))
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Effect": ["Main Effect A", "Main Effect B", "A × B Interaction"],
        "Estimate": [f"{a_eff:+.3f}", f"{b_eff:+.3f}", f"{ab_int:+.3f}"],
    }
    st.table(pd.DataFrame(data))


def _render_icc_calculator(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    icc = st.slider("ICC", 0.0, 0.5, 0.05, step=0.005, key=f"{key_prefix}_icc")
    m = st.slider("Cluster Size (m)", 2, 100, 20, key=f"{key_prefix}_m")
    n_indiv = st.slider("Total Individuals (n)", 10, 5000, 400, key=f"{key_prefix}_nindiv")

    de = 1 + (m - 1) * icc
    neff = n_indiv / de

    fig = go.Figure()
    icc_range = np.linspace(0.001, 0.5, 100)
    de_range = 1 + (m - 1) * icc_range
    fig.add_trace(go.Scatter(x=icc_range, y=de_range, mode="lines", name="Design Effect",
                             line=dict(color="rgba(54, 162, 235, 0.8)")))
    fig.add_vline(x=icc, line_dash="dash", line_color="red", annotation_text=f"ICC = {icc:.3f}")
    fig.add_hline(y=de, line_dash="dash", line_color="orange", annotation_text=f"DE = {de:.2f}")
    fig.update_layout(template="plotly_dark", height=300, xaxis_title="ICC", yaxis_title="Design Effect (DE)")
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Measure": ["ICC", "Cluster Size (m)", "Design Effect (DE)", "Nominal n", "Effective n"],
        "Value": [f"{icc:.4f}", f"{m}", f"{de:.3f}", f"{n_indiv}", f"{neff:.1f}"],
    }
    st.table(pd.DataFrame(data))


def _render_meta_forest(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    n_studies = st.slider("Number of Studies", 3, 10, 5, key=f"{key_prefix}_ns")

    studies = []
    for i in range(n_studies):
        col1, col2 = st.columns(2)
        with col1:
            a_i = st.slider(f"Events (Study {i+1})", 0, 100, max(1, 15 + i * 3), key=f"{key_prefix}_ev{i}")
            n_i = st.slider(f"Total (Study {i+1})", 1, 200, 50 + i * 5, key=f"{key_prefix}_tot{i}")
        with col2:
            c_i = st.slider(f"Controls Events (Study {i+1})", 0, 100, max(1, 10 + i * 2), key=f"{key_prefix}_cev{i}")
            d_i = st.slider(f"Controls Total (Study {i+1})", 1, 200, 50 + i * 5, key=f"{key_prefix}_ctot{i}")
        if a_i > n_i: a_i = n_i
        if c_i > d_i: c_i = d_i
        studies.append((a_i, n_i, c_i, d_i))

    or_vals = []
    ci_lows = []
    ci_highs = []
    weights = []
    labels = []
    for i, (a_i, n_i, c_i, d_i) in enumerate(studies):
        b_i = n_i - a_i
        d_i_ = d_i - c_i
        or_i = (a_i * d_i_) / (b_i * c_i) if (b_i * c_i) > 0 else float("inf")
        se_i = np.sqrt(1 / a_i + 1 / b_i + 1 / c_i + 1 / d_i_) if all(x > 0 for x in [a_i, b_i, c_i, d_i_]) else 1
        or_vals.append(or_i)
        ci_lows.append(np.exp(np.log(or_i) - 1.96 * se_i) if or_i > 0 and or_i != float("inf") else 0)
        ci_highs.append(np.exp(np.log(or_i) + 1.96 * se_i) if or_i > 0 and or_i != float("inf") else float("inf"))
        weights.append(1 / (se_i**2) if se_i > 0 else 0)
        labels.append(f"Study {i+1}")

    fe_num = sum(w * np.log(max(or_v, 0.001)) for w, or_v in zip(weights, or_vals))
    fe_den = sum(weights)
    fe_log_or = fe_num / fe_den if fe_den > 0 else 0
    fe_se = np.sqrt(1 / fe_den) if fe_den > 0 else 0
    fe_or = np.exp(fe_log_or)
    fe_ci_low = np.exp(fe_log_or - 1.96 * fe_se)
    fe_ci_high = np.exp(fe_log_or + 1.96 * fe_se)

    labels_plot = labels + ["FE Summary"]
    ors_plot = or_vals + [fe_or]
    cis_low_plot = ci_lows + [fe_ci_low]
    cis_high_plot = ci_highs + [fe_ci_high]

    fig = go.Figure()
    for i in range(len(labels_plot)):
        color = "rgba(76, 120, 168, 0.8)" if i < len(labels) else "rgba(228, 87, 86, 0.9)"
        size = 8 if i < len(labels) else 12
        fig.add_trace(go.Scatter(
            x=[ors_plot[i]], y=[labels_plot[i]],
            mode="markers+text" if i >= len(labels) else "markers",
            text=[f"{ors_plot[i]:.2f}"] if i >= len(labels) else None,
            textposition="middle right",
            marker=dict(size=size, color=color),
            error_x=dict(type="data", symmetric=False,
                         array=[cis_high_plot[i] - ors_plot[i]],
                         arrayminus=[ors_plot[i] - cis_low_plot[i]],
                         color=color, thickness=2),
            showlegend=False,
        ))
    fig.add_vline(x=1, line_dash="dot", line_color="gray")
    fig.update_layout(template="plotly_dark", height=80 + 40 * len(labels_plot),
                      xaxis_title="Odds Ratio (log scale)", xaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Measure": ["FE Pooled OR", "95% CI (FE)", "I²"],
        "Value": [f"{fe_or:.3f}", f"[{fe_ci_low:.3f}, {fe_ci_high:.3f}]", "N/A"],
    }
    st.table(pd.DataFrame(data))


def _render_pilot_metrics(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    eligible = st.slider("Eligible Patients", 10, 500, 100, key=f"{key_prefix}_elig")
    enrolled = st.slider("Enrolled", 0, eligible, 60, key=f"{key_prefix}_enr")
    completed = st.slider("Completed Follow-up", 0, enrolled, 45, key=f"{key_prefix}_comp")

    rec_rate = enrolled / eligible if eligible > 0 else 0
    ret_rate = completed / enrolled if enrolled > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Recruitment Rate", x=["Recruitment"], y=[rec_rate], text=[f"{rec_rate:.0%}"], textposition="outside",
                         marker_color="rgba(54, 162, 235, 0.7)"))
    fig.add_trace(go.Bar(name="Retention Rate", x=["Retention"], y=[ret_rate], text=[f"{ret_rate:.0%}"], textposition="outside",
                         marker_color="rgba(76, 175, 80, 0.7)"))
    fig.update_layout(template="plotly_dark", height=350, yaxis_range=[0, 1], yaxis_title="Rate")
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Metric": ["Eligible", "Enrolled", "Completed", "Recruitment Rate", "Retention Rate"],
        "Value": [f"{eligible}", f"{enrolled}", f"{completed}", f"{rec_rate:.2%}", f"{ret_rate:.2%}"],
    }
    st.table(pd.DataFrame(data))


def _render_ecological_smr(title, formulas, description, key_prefix):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.caption(description)

    observed = st.slider("Observed Events (O)", 1, 500, 120, key=f"{key_prefix}_obs")
    expected = st.slider("Expected Events (E)", 1, 500, 100, key=f"{key_prefix}_exp")

    smr = observed / expected

    byar_low = observed * (1 - 1 / (9 * observed) - 1.96 / (3 * np.sqrt(observed))) ** 3 if observed > 0 else 0
    byar_high = (observed + 1) * (1 - 1 / (9 * (observed + 1)) + 1.96 / (3 * np.sqrt(observed + 1))) ** 3

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["SMR"], y=[smr], text=[f"{smr:.2f}"], textposition="outside",
                         marker_color="rgba(54, 162, 235, 0.7)"))
    fig.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="SMR = 1 (null)")
    fig.add_hrect(y0=byar_low, y1=byar_high, fillcolor="rgba(0,255,0,0.05)", line_width=0)
    fig.update_layout(template="plotly_dark", height=350, yaxis_title="SMR")
    st.plotly_chart(fig, use_container_width=True)

    data = {
        "Measure": ["SMR", "Byar 95% CI Lower", "Byar 95% CI Upper"],
        "Value": [f"{smr:.3f}", f"{byar_low:.3f}", f"{byar_high:.3f}"],
    }
    st.table(pd.DataFrame(data))


def _render_descriptive_only(title, formulas, description, key_prefix=None):

    st.markdown(f"**{title}**")
    for f in formulas:
        st.latex(f)
    st.info(description)


def _render_outcome_measures(design_name):
    om = _OUTCOME_MEASURES.get(design_name)
    if om is None:
        return

    st.markdown("### Outcome Measures")
    widget = om["widget"]
    key_prefix = f"om_{design_name.lower().replace(' ', '_').replace('/', '_').replace('-', '_')[:40]}"

    if widget == "binary_2x2":
        _render_binary_2x2(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "binary_or":
        _render_binary_or(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "diagnostic":
        _render_diagnostic(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "single_proportion":
        _render_single_proportion(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "paired_continuous":
        _render_paired_continuous(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "non_inferiority":
        _render_non_inferiority(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "factorial_2x2":
        _render_factorial_2x2(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "icc_calculator":
        _render_icc_calculator(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "meta_forest":
        _render_meta_forest(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "pilot_metrics":
        _render_pilot_metrics(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "ecological_smr":
        _render_ecological_smr(om["title"], om["formulas"], om["description"], key_prefix)
    elif widget == "descriptive_only":
        _render_descriptive_only(om["title"], om["formulas"], om["description"], key_prefix)

    st.divider()


def _init_state():
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = "Design Wizard"
    if DESIGN_KEY not in st.session_state:
        st.session_state[DESIGN_KEY] = None


def render_study_designs():
    _init_state()
    designs = _load_designs()

    with st.sidebar:
        st.title("Study Designs")

        st.subheader("Interactive Tools")
        _section_button("Design Wizard", "tool")
        _section_button("Bias Matrix", "tool")
        _section_button("Bias Checklist", "tool")

        st.divider()
        st.subheader("Design Reference")

        for level in range(1, 6):
            level_designs = [d for d in designs if d["evidence_pyramid"] == level]
            if not level_designs:
                continue
            level_labels = {
                1: "I — RCT / Experimental",
                2: "II — Quasi-Experimental",
                3: "III — Observational (Analytic)",
                4: "IV — Observational (Descriptive)",
                5: "V — Case Reports / Expert Opinion",
            }
            st.markdown(f"**Level {level_labels[level]}**")
            for d in level_designs:
                label = f"{d['icon']} {d['name']}"
                btn_key = f"design_{d['name'].replace(' ', '_').replace('/', '_').replace(',', '')}"
                if st.sidebar.button(
                    label,
                    key=btn_key,
                    use_container_width=True,
                    type=(
                        "primary"
                        if st.session_state.get(DESIGN_KEY) == d["name"]
                        else "secondary"
                    ),
                ):
                    st.session_state[SESSION_KEY] = "design_detail"
                    st.session_state[DESIGN_KEY] = d["name"]
                    st.rerun()
            st.sidebar.markdown("<br>", unsafe_allow_html=True)

    section = st.session_state[SESSION_KEY]

    if section == "Design Wizard":
        _render_wizard(designs)
    elif section == "Bias Matrix":
        _render_bias_matrix(designs)
    elif section == "Bias Checklist":
        _render_bias_checklist(designs)
    elif section == "design_detail":
        selected = st.session_state.get(DESIGN_KEY)
        design = next((d for d in designs if d["name"] == selected), None)
        if design:
            _render_design_page(design)
        else:
            st.warning("Design not found. Please select a design from the sidebar.")


def _render_wizard(designs):
    st.title("Interactive Study Design Wizard")
    st.info("""
    Answer a few questions to narrow down which study design is most appropriate 
    for your research question. This tool is for guidance only — consult a 
    biostatistician or epidemiologist for final design decisions.
    """)

    st.markdown("### Your Research Parameters")

    c1, c2 = st.columns(2)
    with c1:
        has_intervention = st.radio(
            "Do you assign an intervention?",
            ["Yes — I control the exposure", "No — I observe exposures as they occur"],
            key="wiz_intervention",
        )
    with c2:
        randomization = st.radio(
            "Can you randomize?",
            ["Yes", "No", "Not applicable (no intervention)"],
            key="wiz_random",
        )

    c3, c4 = st.columns(2)
    with c3:
        timeline = st.radio(
            "When is the outcome measured?",
            [
                "After the intervention (prospective)",
                "Looking back (retrospective)",
                "At the same time (cross-sectional)",
            ],
            key="wiz_timeline",
        )
    with c4:
        comparison = st.radio(
            "Do you have a comparison group?",
            [
                "Yes — a separate control/comparison group",
                "No — all participants receive the same treatment",
                "Each participant serves as their own control",
            ],
            key="wiz_comparison",
        )

    c5, c6 = st.columns(2)
    with c5:
        outcome_rarity = st.radio(
            "Is the outcome rare?",
            ["No — common outcome (>10%)", "Yes — rare outcome (<5%)", "Not sure"],
            key="wiz_rarity",
        )
    with c6:
        cluster = st.radio(
            "Is the intervention delivered at the group level?",
            ["No — individual level", "Yes — group/school/clinic level"],
            key="wiz_cluster",
        )

    st.divider()

    scores = {}
    for d in designs:
        score = 0
        n_criteria = 0

        if has_intervention.startswith("Yes"):
            if randomization == "Yes":
                if cluster == "No — individual level":
                    if timeline == "After the intervention (prospective)":
                        score += 3 if "RCT" in d["name"] else 0
                    else:
                        score += 1 if "RCT" in d["name"] else 0
                else:
                    score += 2 if "Cluster" in d["name"] else 0
                n_criteria += 1
            else:
                if (
                    "Case" in d["name"]
                    or "Cohort" in d["name"]
                    or "Stepped" in d["name"]
                ):
                    score += 1
                n_criteria += 1
        else:
            if comparison == "Yes — a separate control/comparison group":
                if timeline == "Looking back (retrospective)":
                    score += 2 if "Case-Control" in d["name"] else 0
                elif timeline == "After the intervention (prospective)":
                    score += 2 if "Cohort" in d["name"] else 0
                n_criteria += 1

            if outcome_rarity.startswith("Yes"):
                score += 2 if "Case-Control" in d["name"] else 0
                n_criteria += 1
            elif outcome_rarity.startswith("No"):
                score += (
                    2 if "Cohort" in d["name"] or "Cross-Sectional" in d["name"] else 0
                )
                n_criteria += 1

            if timeline == "At the same time (cross-sectional)":
                score += 3 if "Cross-Sectional" in d["name"] else 0
                n_criteria += 1

        if comparison == "Each participant serves as their own control":
            score += 3 if "Crossover" in d["name"] else 0
            n_criteria += 1
        elif comparison == "No — all participants receive the same treatment":
            score += 3 if "Single-Arm" in d["name"] else 0
            n_criteria += 1

        scores[d["name"]] = score / max(n_criteria, 1)

    top_n = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]
    top_designs = [(name, score) for name, score in top_n if score > 0]

    if not top_designs:
        st.warning(
            "No designs closely match your parameters. Consider consulting a research methodologist."
        )
        st.markdown("#### All Available Designs")
        for d in designs:
            st.markdown(
                f"- {d['icon']} **{d['name']}** (Evidence Level {d['evidence_pyramid']})"
            )
        return

    st.markdown("### Recommended Designs")
    st.markdown("Based on your answers, the following designs may be most appropriate:")

    cols = st.columns(len(top_designs))
    for i, (name, score) in enumerate(top_designs):
        design = next(d for d in designs if d["name"] == name)
        with cols[i]:
            st.markdown(f"### {design['icon']}")
            st.markdown(f"**{name}**")
            st.markdown(f"Evidence Level: **{design['evidence_pyramid']}**")
            st.markdown(f"Match: {score:.0%}")
            aliases = design.get("aliases", [])
            if aliases:
                st.caption(f"Also known as: {', '.join(aliases[:3])}")
            if st.button(
                "View Details",
                key=f"wiz_goto_{design['name']}",
                use_container_width=True,
            ):
                st.session_state[SESSION_KEY] = "design_detail"
                st.session_state[DESIGN_KEY] = name
                st.rerun()

    st.divider()
    st.markdown("#### Understanding Your Results")
    st.markdown("""
    The wizard scores designs based on **matching criteria**, not statistical validity. 
    A high match score means the design fits your structural parameters, not that it is 
    automatically the best choice. Consider:

    - **Feasibility**: Do you have the resources, time, and access to participants?
    - **Ethics**: Is randomization ethical? Is withholding the intervention acceptable?
    - **Bias profile**: What are the key vulnerabilities of the recommended design?
    - **Sample size**: Do you have enough participants for adequate statistical power?
    """)


def _render_bias_matrix(designs):
    st.title("Bias Vulnerability Matrix")
    st.info("""
    This matrix shows which biases pose the greatest threat to each study design. 
    Darker red cells indicate higher vulnerability. Use this to identify which biases 
    you need to address in your study design and analysis.
    """)

    rows = []
    for d in designs:
        bp = d["sections"]["bias_profile"]
        row = {"Design": f"{d['icon']} {d['name']}"}
        for bk in BIAS_LABELS:
            entry = bp.get(bk, {"risk": "N/A"})
            row[BIAS_LABELS[bk]] = entry["risk"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Design")

    risk_order = ["Very High", "High", "Moderate", "Low", "N/A"]
    risk_values = {r: i for i, r in enumerate(risk_order)}

    z = df.map(lambda x: risk_values.get(x, 4))
    bias_cols = list(BIAS_LABELS.values())

    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=bias_cols,
            y=df.index.tolist(),
            text=df.values,
            texttemplate="%{text}",
            textfont={"size": 10, "color": "white"},
            colorscale=[
                [0, "rgba(228,87,86,0.85)"],
                [0.25, "rgba(228,87,86,0.55)"],
                [0.5, "rgba(241,197,55,0.65)"],
                [0.75, "rgba(76,120,168,0.55)"],
                [1, "rgba(128,128,128,0.25)"],
            ],
            zmin=0,
            zmax=4,
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=60 + 40 * len(designs),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis={"side": "top", "tickangle": -30},
        yaxis={"tickfont": {"size": 11}},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Legend")
    c = st.columns(4)
    for i, risk in enumerate(risk_order):
        with c[i % 4]:
            st.markdown(
                f"<div style='background-color:{RISK_COLORS[risk]}; padding:6px 12px; "
                f"border-radius:4px; text-align:center; color:white; font-weight:600;'>{risk}</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    with st.expander("Bias Definitions", expanded=True):
        for bk, label in BIAS_LABELS.items():
            st.markdown(f":orange[**{label}**]: {BIAS_DESCRIPTIONS.get(bk, '')}")


def _render_bias_checklist(designs):
    st.title("Bias Checklist Tool")
    st.info("""
    Select a study design and check which biases you have addressed. 
    The tool will summarize your remaining vulnerabilities and suggest mitigation strategies.
    """)

    design_names = [d["name"] for d in designs]
    selected_name = st.selectbox("Select a study design", design_names, key="bc_design")
    design = next((d for d in designs if d["name"] == selected_name), None)
    if design is None:
        st.warning("Design not found. Please select a valid design.")
        return

    st.markdown(f"### {design['icon']} {design['name']} — Bias Assessment")

    bp = design["sections"]["bias_profile"]
    responses = {}

    st.markdown(
        "For each bias, indicate whether you have addressed it in your study design:"
    )

    for bk in BIAS_LABELS:
        label = BIAS_LABELS[bk]
        entry = bp.get(bk, {"risk": "N/A", "detail": ""})
        risk = entry["risk"]
        detail = entry.get("detail", "")

        if risk == "N/A":
            continue

        color = RISK_COLORS.get(risk, "gray")
        with st.container():
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.markdown(
                    f"<div style='border-left:4px solid {color}; padding-left:10px; margin:4px 0;'>"
                    f"<b>{label}</b><br><span style='font-size:0.85em;'>{detail}</span></div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                risk_label = st.markdown(
                    f"<div style='background:{color}; padding:2px 8px; border-radius:4px; "
                    f"text-align:center; color:white; font-weight:600; font-size:0.85em;'>{risk}</div>",
                    unsafe_allow_html=True,
                )
            with cols[2]:
                responses[bk] = st.selectbox(
                    "Status",
                    [
                        "Not Addressed",
                        "Partially",
                        "Fully Addressed",
                        "N/A for my study",
                    ],
                    key=f"bc_{bk}",
                    label_visibility="collapsed",
                )

    st.divider()

    not_addressed = [
        BIAS_LABELS[bk] for bk, v in responses.items() if v == "Not Addressed"
    ]
    partial = [BIAS_LABELS[bk] for bk, v in responses.items() if v == "Partially"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Fully Addressed",
            sum(1 for v in responses.values() if v == "Fully Addressed"),
        )
    with c2:
        st.metric("Partially Addressed", len(partial))
    with c3:
        st.metric("Not Addressed", len(not_addressed))

    if not_addressed:
        st.warning(f"**Vulnerabilities to address**: {', '.join(not_addressed)}")
        st.markdown("**Suggestions:**")
        suggestions = {
            "Selection Bias": "Use random or consecutive sampling; clearly define eligibility criteria; compare participants vs non-participants.",
            "Information Bias": "Use validated measurement instruments; blind assessors; standardize data collection protocols.",
            "Confounding": "Identify confounders via DAGs; use multivariable regression, propensity scores, or instrumental variables; consider matching.",
            "Performance Bias": "Standardize protocols across groups; blind participants and providers when possible; monitor co-interventions.",
            "Detection Bias": "Blind outcome assessors; use objective outcomes when possible; pre-specify outcome definitions.",
            "Attrition Bias": "Minimize loss to follow-up; track reasons for dropout; use multiple imputation or inverse probability weighting.",
            "Reporting Bias": "Pre-register your study; publish a protocol; report all pre-specified outcomes regardless of significance.",
            "Lead-Time Bias": "Use date of outcome (not date of diagnosis) as the endpoint; consider randomized designs for screening evaluation.",
            "Length-Time Bias": "Use incident (not prevalent) cases; avoid comparing screening-detected vs clinically detected cases without adjustment.",
        }
        for bias in not_addressed:
            if bias in suggestions:
                st.markdown(f"- **{bias}**: {suggestions[bias]}")

    if not not_addressed and not partial:
        st.success(
            "All biases have been addressed. Your study design is robust against the identified threats to validity."
        )
    elif not not_addressed and partial:
        st.info(
            "Most biases are addressed, but some need further attention. Review the suggestions above."
        )


def _render_design_page(design):
    s = design["sections"]

    with st.container():
        evidence_labels = {
            1: "I — High (RCT / Experimental)",
            2: "II — Moderate-High (Quasi-Experimental)",
            3: "III — Moderate (Observational — Analytic)",
            4: "IV — Low-Moderate (Observational — Descriptive)",
            5: "V — Low (Case Reports / Expert Opinion)",
        }
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>"
            f"<span style='font-size:2.5rem;'>{design['icon']}</span>"
            f"<span style='font-size:1.8rem; font-weight:700;'>{design['name']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        aliases = design.get("aliases", [])
        if aliases:
            st.caption(f"Also known as: {', '.join(aliases)}")

        ev_level = design["evidence_pyramid"]
        ev_colors = {
            1: "rgba(76,120,168,0.8)",
            2: "rgba(76,120,168,0.6)",
            3: "rgba(241,197,55,0.7)",
            4: "rgba(228,87,86,0.5)",
            5: "rgba(228,87,86,0.7)",
        }
        st.markdown(
            f"<div style='background:{ev_colors[ev_level]}; padding:6px 14px; border-radius:6px; "
            f"display:inline-block; margin:6px 0 16px; font-weight:600;'>"
            f"Evidence Level {ev_level}</div>",
            unsafe_allow_html=True,
        )

    # Left column — overview, causal, assumptions
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("### What Is This Design?")
        st.markdown(s["definition"])

        st.markdown("### When to Use")
        st.info(s["when_to_use"])

        st.markdown("### Causal Inference & Evidence Strength")
        st.markdown(s["causal_inference"])

        st.markdown("### Assumptions & Prerequisites")
        for i, assumption in enumerate(s["assumptions"], 1):
            st.markdown(f"{i}. {assumption}")

        _render_outcome_measures(design["name"])

        st.markdown("### Statistical Considerations")
        st.markdown(s.get("statistical_considerations", ""))

        st.markdown("### Software Notes & Data Structure")
        with st.expander("Implementation details", expanded=False):
            st.markdown(s.get("software_notes", ""))

    with c_right:
        st.markdown("### Worked Example")
        with st.expander("View worked example", expanded=True):
            st.markdown(s["worked_example"])

        st.markdown("### Reporting Guideline")
        rg = s.get("reporting_guideline", {})
        if rg:
            name = rg.get("name", "")
            url = rg.get("url", "")
            key_items = rg.get("key_items", [])
            st.markdown(f"**{name}**")
            if url:
                st.markdown(f"[{url}]({url})")
            if key_items:
                st.markdown("**Key reporting items:**")
                for item in key_items:
                    st.markdown(f"- {item}")

        st.markdown("### Common Pitfalls")
        for pitfall in s.get("pitfalls", []):
            st.warning(pitfall)

        st.markdown("### Further Reading")
        for ref in s.get("further_reading", []):
            st.markdown(f"- {ref}")

    # Bias profile — full width (excluded from two-column)
    st.markdown("### Bias Profile")
    bp = s["bias_profile"]
    bias_rows = []
    for bk in BIAS_LABELS:
        entry = bp.get(bk, {"risk": "N/A", "detail": ""})
        risk = entry["risk"]
        detail = entry.get("detail", "")
        color = RISK_COLORS.get(risk, "gray")
        bias_rows.append(
            {
                "Bias": BIAS_LABELS[bk],
                "Risk": risk,
                "Detail": detail,
            }
        )
    bias_df = pd.DataFrame(bias_rows)
    col_map = {"Bias": None, "Risk": None, "Detail": None}
    _apa_table(bias_df, hide_index=True)

    # Navigation footer
    st.divider()
    if st.button("Back to Design Wizard", use_container_width=True):
        st.session_state[SESSION_KEY] = "Design Wizard"
        st.session_state[DESIGN_KEY] = None
        st.rerun()
