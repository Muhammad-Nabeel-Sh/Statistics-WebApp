import streamlit as st
import numpy as np
from data import TEST_TO_SS_TYPE, rules, FIELDS
from matching import find_matching_tests
from widgets import render_latex, render_test_widget
from power_calculator import render_power_calculator
from flowchart import build_tree


# =========================
# STREAMLIT UI
# =========================
def main():

    st.set_page_config(page_title="Statistical Test Finder", layout="wide")

    # =========================
    # SIDEBAR GLOSSARY
    # =========================
    with st.sidebar:
        st.header(":orange[🕮 &ensp; Statistical Glossary]")

        with st.expander("Variable Types"):
            st.markdown("""
            :orange[**Binary / Dichotomous**]: Two mutually exclusive and exhaustive categories with no order or magnitude (e.g., Male/Female, Pass/Fail, Alive/Dead). Only frequencies, proportions, and mode are meaningful.

            :orange[**Categorical / Nominal**]: Categories without any inherent order or numerical meaning. Arithmetic operations are invalid; only frequency counts, proportions, and mode are appropriate (e.g., Eye color, Blood type, Department).

            :orange[**Ordinal**]: Categories with a meaningful order but unequal or unknown intervals between them. The median and percentiles are appropriate; the mean is not (e.g., Likert scales, Pain severity, Education level).

            :orange[**Discrete (Count)**]: Whole-number values representing countable quantities, often with a true zero. May follow a Poisson or negative binomial distribution (e.g., Number of teeth, Number of hospital visits).

            :orange[**Continuous (Scale)**]: Values that can theoretically take any magnitude within a range, measured on a continuum. Parametric tests are appropriate when normally distributed (e.g., Height, Blood pressure, Serum cholesterol, Temperature).

            :orange[**Dependent (Outcome)**]: The variable being measured, tested, or predicted — the "effect" or "response" in a research question (Y variable in regression).

            :orange[**Independent (Predictor)**]: The variable being manipulated, exposed, or used to predict — the "cause" or "exposure" (X variable in regression).
            """)

        with st.expander("Descriptive Statistics"):
            st.markdown("""
            :orange[**Mean**]: The arithmetic average (sum divided by n). Sensitive to outliers and skewness. Best used for symmetric, normally distributed data. In a normal distribution, mean ≈ median.

            :orange[**Median**]: The 50th percentile — the middle value when data are sorted. Robust to outliers and skewness. Preferred measure of central tendency for ordinal or non-normal data.

            :orange[**Mode**]: The most frequently occurring value(s). The only measure of central tendency applicable to nominal/categorical data. A dataset may have multiple modes (bimodal, multimodal).

            :orange[**Standard Deviation (SD)**]: The average distance of individual values from the mean, in the original units. In a normal distribution, ~68% of data falls within ±1 SD, ~95% within ±2 SD.

            :orange[**Variance**]: The average of squared deviations from the mean (SD²). Used in ANOVA and other methods because variances are additive, though the squared units make direct interpretation less intuitive.

            :orange[**Interquartile Range (IQR)**]: The range covering the middle 50% of values (Q3 − Q1). Robust to outliers. Typically paired with the median for non-normal data; visualized as the box in box plots.

            :orange[**Standard Error (SE)**]: The standard deviation of a sampling distribution (e.g., the SD of sample means). It measures the precision of an estimate and decreases as sample size increases (SE = SD/√n). Distinct from standard deviation, which measures dispersion in the sample itself.
            """)

        with st.expander("Sampling & Research Design"):
            st.markdown("""
            :orange[**Random Sampling**]: Every member of the population has an equal and independent chance of selection. Minimizes selection bias and enables generalization to the target population.

            :orange[**Stratified Sampling**]: Dividing the population into relevant subgroups (strata), then sampling from each. Ensures adequate representation of all subgroups, especially small ones.

            :orange[**Blinding**]: Concealing group allocation to reduce bias. Single-blind: participants are unaware. Double-blind: both participants and investigators are unaware. Essential for minimizing performance and detection bias.

            :orange[**Randomization**]: Assigning participants to groups by a chance mechanism (e.g., random number generator). Balances known and unknown confounders between groups, enabling causal inference.

            :orange[**Randomized Controlled Trial (RCT)**]: The gold standard for intervention studies, where participants are randomly allocated to treatment or control. Minimizes confounding and allows strong causal conclusions.

            :orange[**Cohort Study**]: A longitudinal design that follows a group over time, comparing outcomes between exposed and unexposed individuals. Can establish temporality (cause precedes effect) but susceptible to loss to follow-up.

            :orange[**Case-Control Study**]: A retrospective design comparing individuals with a condition (cases) to those without (controls), looking back at past exposures. Efficient for rare diseases but prone to recall bias.

            :orange[**Cross-sectional Study**]: A "snapshot" measuring exposure and outcome simultaneously at one time point. Useful for estimating prevalence but cannot establish causality due to lack of temporality.

            :orange[**Placebo Effect**]: Improvement in symptoms due to the belief in receiving treatment rather than the treatment's pharmacological action. Controlled for by using a placebo arm in RCTs.

            :orange[**Intention-to-Treat (ITT)**]: Analyzing participants according to the group they were originally randomized to, regardless of whether they completed the treatment or crossed over. ITT preserves the benefits of randomization and estimates real-world treatment effectiveness.

            :orange[**Selection Bias / Information Bias / Publication Bias**]: Three major sources of systematic error in medical research. Selection bias occurs when study groups differ systematically (e.g., non-random allocation). Information bias occurs when measurement errors differ between groups. Publication bias occurs when studies with positive results are more likely to be published than null findings.

            :orange[**Confounding by Indication**]: A specific confounding in observational studies where the reason a treatment was prescribed (indication) is itself associated with the outcome. For example, sicker patients receive more aggressive treatment, making it appear harmful if disease severity is unmeasured.
            """)

        with st.expander("Hypothesis Testing"):
            st.markdown("""
            :orange[**Null Hypothesis (H₀)**]: The default assumption of no effect, no difference, or no association. It is the hypothesis directly tested; rejecting H₀ provides evidence for the alternative.

            :orange[**Alternative Hypothesis (H₁)**]: The hypothesis that a real effect, difference, or association exists. It is supported only if sufficient evidence against H₀ is found.

            :orange[**Alpha (α)**]: The pre-specified significance threshold (conventionally 0.05). The maximum acceptable probability of a Type I error. If p < α, the result is declared statistically significant.

            :orange[**Type I Error (α)**]: Falsely rejecting a true null hypothesis — concluding an effect exists when it does not (a "false positive"). The risk is set by alpha.

            :orange[**Type II Error (β)**]: Failing to reject a false null hypothesis — concluding no effect when one exists (a "false negative"). Power (1 − β) is the probability of avoiding this error.
            
            :orange[**Confidence Interval (CI)**]: A range of values computed from the sample such that if the study were repeated many times, a given percentage (usually 95%) of such intervals would contain the true population parameter.

            :orange[**Degrees of Freedom (df)**]: The number of independent values in a calculation that are free to vary. In statistical tests, df determines which reference distribution (t, F, χ²) is used to compute the p-value.

            :orange[**One-tailed vs Two-tailed Test**]: A one-tailed test specifies the direction of the effect (e.g., treatment increases outcome), allocating the entire alpha to one tail. A two-tailed test tests for any difference (increase or decrease), splitting alpha across both tails. Two-tailed tests are standard in medical research.

            :orange[**Multiple Comparison Correction**]: Adjustments to significance thresholds when testing multiple hypotheses to control inflated Type I error risk. Bonferroni divides α by the number of tests (most conservative). False Discovery Rate (FDR) controls the expected proportion of false positives among rejected hypotheses (less conservative, more powerful).
            """)

        with st.expander("Distribution & Assumptions"):
            st.markdown("""
            :orange[**Parametric Tests**]: Tests that assume data follow a specific distribution (usually normal). Generally higher statistical power when assumptions are met, but require normality, independence, and often equal variances.

            :orange[**Non-parametric Tests**]: Distribution-free tests that rank data rather than using raw values. Do not assume normality. Less powerful when parametric assumptions hold, but more robust when they are violated. Appropriate for ordinal data.

            :orange[**Normality**]: The condition where data follows a symmetrical, bell-shaped distribution around the mean. Assessed via Q-Q plots, histograms, or tests (Shapiro-Wilk, Kolmogorov-Smirnov).

            :orange[**Skewness**]: A measure of asymmetry. Positive (right) skew has a long tail on the right; negative (left) skew has a long tail on the left. Skewed data may require transformation or non-parametric tests.

            :orange[**Homogeneity of Variance**]: The assumption that compared groups have equal variances. Assessed using Levene's test. Violations inflate Type I error; Welch's correction or non-parametric alternatives may be needed.

            :orange[**Outlier**]: An extreme observation lying far from the main body of data (typically > 1.5 × IQR beyond Q1/Q3 or > 3 SD from the mean). Should be investigated rather than automatically removed.

            :orange[**Central Limit Theorem (CLT)**]: The sampling distribution of the mean approaches a normal distribution as sample size increases, regardless of the population distribution's shape. This justifies parametric tests (t-tests, ANOVA) even with non-normal raw data when the sample is sufficiently large (typically n > 30).

            :orange[**Sphericity**]: The assumption that the variances of the differences between all pairs of repeated measures are equal. Critical for repeated measures ANOVA; violation inflates the F-statistic. Corrected using Greenhouse-Geisser or Huynh-Feldt adjustments.
            """)

        with st.expander("Effect Size & Power"):
            st.markdown("""
            :orange[**Effect Size**]: A standardized measure of the magnitude of a finding, independent of sample size. Essential for interpreting practical significance and for conducting power analyses. Unlike p-values, effect sizes are not inflated by large samples.

            :orange[**Statistical Power (1 − β)**]: The probability that a test will correctly reject a false null hypothesis (detect a true effect). Depends on effect size, sample size, alpha, and test type. Conventional target: 80% power.

            :orange[**Cohen's d**]: Standardized difference between two group means in pooled SD units. Benchmarks: 0.2 (small), 0.5 (medium), 0.8 (large). Commonly used for t-tests and meta-analysis.

            :orange[**Cohen's f**]: Effect size measure for ANOVA. Benchmarks: 0.10 (small), 0.25 (medium), 0.40 (large). Related to η² by f = √(η²/(1−η²)).

            :orange[**Cohen's f²**]: Effect size for multiple regression. f² = R²/(1−R²). Benchmarks: 0.02 (small), 0.15 (medium), 0.35 (large).

            :orange[**Cohen's w**]: Effect size for chi-square tests. Benchmarks: 0.10 (small), 0.30 (medium), 0.50 (large). Represents the discrepancy between observed and expected proportions.

            :orange[**Eta-squared (η²)**]: The proportion of total variance in the outcome attributable to a factor in ANOVA. Ranges from 0 to 1. Partial η² is used when multiple factors are present.

            :orange[**Cramer's V**]: An association measure for two nominal variables, derived from the chi-square statistic. Ranges from 0 (no association) to 1 (perfect association), adjusted for table dimensions.

            :orange[**Sample Size Estimation**]: The process of determining the minimum number of participants needed to detect a meaningful effect with adequate power. Depends on α, power, effect size, and test type. Use the "Sample Size Estimation" objective in this tool to compute required N for your study design.

            :orange[**Allocation Ratio**]: The ratio of sample sizes between two groups (n₂/n₁). An equal ratio (1.0) maximizes power for a given total N. Unequal ratios may be used when one group is easier or cheaper to recruit.

            :orange[**Sensitivity Analysis**]: Varying assumptions (effect size, alpha, power) to see how sample size changes. Helps researchers understand the robustness of their sample size estimate and plan for different scenarios.

            :orange[**Fisher's z-Transformation**]: A transformation used in sample size calculations for correlation: z = arctanh(r). Stabilizes the variance of the correlation coefficient, making it suitable for power analysis.
            """)

        with st.expander("Correlation & Regression"):
            st.markdown("""
            :orange[**Correlation**]: Quantifies the strength and direction of association between two variables, ranging from −1 to +1. Does not imply causation.

            :orange[**Pearson's r**]: Measures the linear relationship between two continuous variables. Assumes normality and linearity. Benchmarks: ±0.1 (small), ±0.3 (medium), ±0.5 (large).

            :orange[**Spearman's ρ**]: A rank-based measure of monotonic association. Does not assume linearity or normality. Appropriate for ordinal data or when Pearson assumptions are violated.

            :orange[**Regression**]: Models the relationship between a dependent variable and one or more predictors to estimate effects or make predictions. Includes linear, logistic, Poisson, and other generalized linear models.

            :orange[**R-squared (R²)**]: The coefficient of determination — the proportion of variance in the outcome explained by the predictors (0 to 1 or 0% to 100%).

            :orange[**Adjusted R²**]: R² penalized for the number of predictors, allowing fair comparison between models with different numbers of variables. Prevents overfitting by adding a penalty for each additional predictor.

            :orange[**Residuals**]: The differences between observed and model-predicted values. Analyzing residuals (vs. fitted plots, Q-Q plots) is essential for checking regression assumptions.
            """)

        with st.expander("Reliability & Validity"):
            st.markdown("""
            :orange[**Reliability**]: The consistency or reproducibility of a measurement under consistent conditions. A necessary but not sufficient condition for validity — a measure can be reliable without being valid.

            :orange[**Validity**]: The degree to which a tool accurately measures what it claims to measure. A valid measurement must first be reliable.

            :orange[**Face Validity**]: The subjective judgment that a test "looks like" it measures the intended construct. The weakest form of validity but important for participant and stakeholder acceptance.

            :orange[**Content Validity**]: The extent to which a measurement covers all relevant facets of the construct. Assessed by expert judgment (e.g., a depression scale covering all DSM criteria).

            :orange[**Construct Validity**]: Whether a test truly measures the theoretical construct it claims to. Established through convergent validity (agreement with related measures) and discriminant validity (distinction from unrelated measures).

            :orange[**Criterion Validity**]: How well a measure predicts an outcome based on an established gold standard. Concurrent validity compares with an existing measure simultaneously; predictive validity forecasts future outcomes.

            :orange[**Cronbach's Alpha**]: A measure of internal consistency (0 to 1). Values ≥ 0.7 are generally acceptable for research instruments. Reflects the average inter-item correlation among scale items.

            :orange[**Intraclass Correlation (ICC)**]: A reliability coefficient for quantitative ratings, assessing how strongly measurements from the same group resemble each other. Used for test-retest, inter-rater, and intra-rater reliability.

            :orange[**Inter-rater Reliability**]: The degree of agreement between different raters or observers assessing the same phenomenon. Quantified using Cohen's Kappa (categorical data) or ICC (continuous data).
            """)

        with st.expander("Clinical/Biostatistics"):
            st.markdown("""
            :orange[**Sensitivity (True Positive Rate)**]: The proportion of individuals with the condition who correctly test positive. A highly sensitive test rules out disease when negative (mnemonic: SNout).

            :orange[**Specificity (True Negative Rate)**]: The proportion of individuals without the condition who correctly test negative. A highly specific test rules in disease when positive (mnemonic: SPin).

            :orange[**Positive Predictive Value (PPV)**]: The probability that a positive test truly indicates the condition. Depends on disease prevalence — the same test has lower PPV in low-prevalence populations.

            :orange[**Negative Predictive Value (NPV)**]: The probability that a negative test result correctly rules out the condition. Higher in low-prevalence populations.

            :orange[**Likelihood Ratio (LR+/LR−)**]: The ratio of the probability of a given test result in a person with the condition to the probability in a person without. LR+ > 10 provides strong evidence to rule in disease; LR− < 0.1 provides strong evidence to rule out disease. Used with Bayes' theorem to update pre-test to post-test probability.

            :orange[**Odds Ratio (OR)**]: The ratio of the odds of an event in one group to the odds in another. Commonly used in case-control studies and logistic regression. OR > 1 indicates increased odds; OR < 1 indicates decreased odds.

            :orange[**Relative Risk (RR)**]: The ratio of the risk (cumulative incidence) in the exposed group to the risk in the unexposed group. Used in cohort studies and RCTs. RR > 1 indicates increased risk.

            :orange[**Number Needed to Treat (NNT)**]: The number of patients who must receive a treatment for one additional patient to benefit (avoid one adverse outcome). Calculated as 1/ARR. Lower NNT means more effective treatment.

            :orange[**Number Needed to Harm (NNH)**]: The number of patients who need to be treated for one additional patient to experience a harmful adverse outcome. Calculated as 1/ARI. Higher NNH means safer treatment.

            :orange[**Forest Plot**]: A graphical display in meta-analysis showing individual study effect sizes (squares) and confidence intervals (horizontal lines), along with a pooled summary estimate (diamond). The size of each square reflects the study's weight in the analysis.

            :orange[**Absolute Risk Reduction (ARR)**]: The absolute difference in event rates between control and treatment groups (CER − TER). Unlike RRR, ARR reflects the actual clinical benefit and is not inflated by low baseline risk.

            :orange[**Relative Risk Reduction (RRR)**]: The proportional reduction in risk between treatment and control (ARR/CER). RRR can be misleadingly large when the baseline risk is very low (e.g., reducing 0.01% to 0.005% is a 50% RRR but only 0.005% ARR).

            :orange[**Standardized Mean Difference (SMD)**]: An effect size expressing the difference between two group means in standard deviation units. Used in meta-analyses when studies measure the same outcome using different scales. Cohen's d and Hedges' g are common variants.
            """)

        with st.expander("Survival Analysis"):
            st.markdown("""
            :orange[**Censoring**]: Occurs when the exact survival time is unknown — typically because the event (e.g., death) has not occurred by study end (right-censoring), the subject withdrew, or was lost to follow-up. A defining feature of survival analysis.

            :orange[**Kaplan-Meier Curve**]: A non-parametric estimator of the survival function, plotting the probability of surviving past successive time points. Accounts for censored data. The curve steps down at each event time.

            :orange[**Hazard Ratio (HR)**]: The ratio of the instantaneous risk (hazard) of the event in one group to another at any given time. Under the Cox proportional hazards model, HR is assumed constant over time. HR = 2 means the event rate is twice as high.
            """)

        with st.expander("Advanced Modeling"):
            st.markdown("""
            :orange[**Multicollinearity**]: A condition in regression where independent variables are highly correlated, making it difficult to isolate their individual effects. Inflates standard errors and destabilizes coefficient estimates. Detected using Variance Inflation Factor (VIF > 5–10 signals concern).

            :orange[**Interaction Effect (Effect Modification)**]: Occurs when the effect of one variable on the outcome differs across levels of another variable (e.g., a drug works differently in men vs. women). Tested by including a product term in the model.

            :orange[**Confounding Variable**]: A variable that distorts the association between an exposure and outcome because it is associated with both. If unmeasured or uncontrolled, it can bias the estimated effect — either exaggerating or masking the true relationship.
            """)

    st.title("🔬 Statistical Test Finder")

    st.write(
        "Select your study characteristics to identify the appropriate statistical test."
    )

    # Initialize results and open_tests in session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "open_tests" not in st.session_state:
        st.session_state.open_tests = set()

    # Create columns: Left for inputs, Right for results with a large gap as a gutter
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        # =========================
        # RESEARCH Objective
        # =========================
        st.subheader("1. Research Objective")

        obj_opts = [
            "Comparison",
            "Association/Correlation",
            "Prediction",
            "Diagnostic Accuracy",
            "Survival Analysis",
            "Sample Size Estimation",
        ]
        default_obj_idx = obj_opts.index(
            st.session_state.pop("ss_pending_obj", "Comparison")
        )
        Objective = st.selectbox("What is your goal?", obj_opts, index=default_obj_idx)

        # =========================
        # SAMPLE SIZE ESTIMATION UI
        # =========================
        if Objective == "Sample Size Estimation":
            st.subheader("2. Analysis Type & Parameters")

            ss_at_opts = [
                "One-sample Mean (t/z-test)",
                "Two Independent Means (t-test)",
                "Paired Means (t-test)",
                "One-sample Proportion",
                "Two Proportions",
                "One-way ANOVA",
                "Correlation (Pearson)",
                "Multiple Linear Regression",
                "Logistic Regression",
                "Chi-Square Test",
                "Mann-Whitney / Wilcoxon (Non-parametric)",
                "Log-Rank Test (Survival)",
                "Cox Regression",
                "Equivalence / Non-Inferiority",
                "Repeated Measures ANOVA",
                "Two-way / Factorial ANOVA",
                "ROC / AUC Analysis",
                "Cohen's Kappa / ICC Agreement",
                "Cluster-RCT / Multilevel",
                "Precision-based (CI Width)",
                "Pilot / Feasibility Study",
                "Wilcoxon Signed-Rank (paired)",
                "Kruskal-Wallis Test",
                "Friedman Test",
                "McNemar's Test",
                "Fisher's Exact Test",
                "MANOVA (Multivariate ANOVA)",
                "Binomial Exact Test",
                "Simulation-based Power (Monte Carlo)",
            ]
            default_at_idx = ss_at_opts.index(
                st.session_state.pop("ss_pending_at", ss_at_opts[0])
            )
            analysis_type = st.selectbox(
                "Type of Analysis", ss_at_opts, index=default_at_idx
            )

            st.markdown("##### :orange[Common Parameters]")
            col_a, col_b = st.columns(2)
            with col_a:
                alpha_ss = st.slider(
                    "Significance Level (α)",
                    0.001,
                    0.10,
                    0.05,
                    0.001,
                    format="%.3f",
                )
            with col_b:
                power_ss = st.slider(
                    "Power (1 − β)",
                    0.50,
                    0.99,
                    0.80,
                    0.01,
                    format="%.2f",
                )
            tails_ss = st.radio(
                "Test Direction",
                ["Two-tailed", "One-tailed"],
                horizontal=True,
            )

            st.markdown("##### :orange[Test-Specific Parameters]")
            ss_params = {}

            if analysis_type == "One-sample Mean (t/z-test)":
                c1, c2 = st.columns(2)
                with c1:
                    mean_diff = st.number_input(
                        "Expected Mean Difference (μ − μ₀)",
                        0.0,
                        100.0,
                        1.0,
                        0.1,
                    )
                with c2:
                    std_dev_1s = st.number_input(
                        "Standard Deviation (σ)",
                        0.1,
                        100.0,
                        2.0,
                        0.1,
                    )
                d_1s = mean_diff / std_dev_1s if std_dev_1s > 0 else 0
                st.caption(
                    f"Cohen's d = {d_1s:.3f} — Small: 0.20 | Medium: 0.50 | Large: 0.80"
                )
                ss_params = {"type": "one_mean", "effect_size": d_1s}

            elif analysis_type == "Two Independent Means (t-test)":
                c1, c2, c3 = st.columns(3)
                with c1:
                    m1 = st.number_input("Mean of Group 1", 0.0, 100.0, 0.0, 0.1)
                with c2:
                    m2 = st.number_input("Mean of Group 2", 0.0, 100.0, 1.0, 0.1)
                with c3:
                    sd_2s = st.number_input("Pooled SD", 0.1, 100.0, 1.0, 0.1)
                ratio_2s = st.number_input(
                    "Allocation Ratio (n₂/n₁)",
                    0.1,
                    10.0,
                    1.0,
                    0.1,
                )
                d_2s = abs(m1 - m2) / sd_2s if sd_2s > 0 else 0
                st.caption(
                    f"Cohen's d = {d_2s:.3f} — Small: 0.20 | Medium: 0.50 | Large: 0.80"
                )
                ss_params = {
                    "type": "two_means",
                    "effect_size": d_2s,
                    "ratio": ratio_2s,
                }

            elif analysis_type == "Paired Means (t-test)":
                c1, c2 = st.columns(2)
                with c1:
                    pdiff = st.number_input(
                        "Expected Mean Difference",
                        0.0,
                        100.0,
                        1.0,
                        0.1,
                    )
                with c2:
                    sddiff = st.number_input(
                        "SD of Differences",
                        0.1,
                        100.0,
                        1.5,
                        0.1,
                    )
                d_pd = pdiff / sddiff if sddiff > 0 else 0
                st.caption(
                    f"Cohen's d_z = {d_pd:.3f} — Small: 0.20 | Medium: 0.50 | Large: 0.80"
                )
                ss_params = {"type": "paired", "effect_size": d_pd}

            elif analysis_type == "One-sample Proportion":
                c1, c2 = st.columns(2)
                with c1:
                    p0 = st.number_input("Null Proportion (p₀)", 0.01, 0.99, 0.5, 0.01)
                with c2:
                    p1 = st.number_input(
                        "Expected Proportion (p₁)",
                        0.01,
                        0.99,
                        0.7,
                        0.01,
                    )
                ss_params = {
                    "type": "one_prop",
                    "prop_null": p0,
                    "prop_alt": p1,
                }

            elif analysis_type == "Two Proportions":
                c1, c2, c3 = st.columns(3)
                with c1:
                    prop1 = st.number_input(
                        "Proportion in Group 1", 0.01, 0.99, 0.3, 0.01
                    )
                with c2:
                    prop2 = st.number_input(
                        "Proportion in Group 2", 0.01, 0.99, 0.5, 0.01
                    )
                with c3:
                    ratio_prop = st.number_input(
                        "Allocation Ratio (n₂/n₁)",
                        0.1,
                        10.0,
                        1.0,
                        0.1,
                    )
                ss_params = {
                    "type": "two_prop",
                    "p1": prop1,
                    "p2": prop2,
                    "ratio": ratio_prop,
                }

            elif analysis_type == "One-way ANOVA":
                c1, c2 = st.columns(2)
                with c1:
                    k_anova = st.number_input("Number of Groups", 3, 20, 3, 1)
                with c2:
                    f_anova = st.number_input(
                        "Cohen's f (effect size)",
                        0.01,
                        2.0,
                        0.25,
                        0.01,
                    )
                    st.caption("Small: 0.10 | Medium: 0.25 | Large: 0.40")
                ss_params = {"type": "anova", "k": int(k_anova), "effect_size": f_anova}

            elif analysis_type == "Correlation (Pearson)":
                r_val = st.number_input(
                    "Expected Correlation (r)",
                    0.01,
                    0.99,
                    0.3,
                    0.01,
                )
                st.caption("Small: 0.10 | Medium: 0.30 | Large: 0.50")
                ss_params = {"type": "correlation", "effect_size": r_val}

            elif analysis_type == "Multiple Linear Regression":
                c1, c2 = st.columns(2)
                with c1:
                    k_reg = st.number_input("Number of Predictors", 1, 50, 3, 1)
                with c2:
                    r2_reg = st.number_input("Expected R²", 0.01, 0.99, 0.15, 0.01)
                f2_reg = r2_reg / (1 - r2_reg) if r2_reg < 1 else 0
                st.caption(
                    f"Cohen's f² = {f2_reg:.3f} — Small: 0.02 | Medium: 0.15 | Large: 0.35"
                )
                ss_params = {
                    "type": "regression",
                    "k": int(k_reg),
                    "effect_size": f2_reg,
                }

            elif analysis_type == "Logistic Regression":
                c1, c2 = st.columns(2)
                with c1:
                    k_log = st.number_input("Number of Predictors", 1, 50, 3, 1)
                with c2:
                    ev_rate = st.number_input(
                        "Baseline Event Rate",
                        0.01,
                        0.99,
                        0.3,
                        0.01,
                    )
                or_val = st.number_input("Odds Ratio to Detect", 1.1, 10.0, 2.0, 0.1)
                ss_params = {
                    "type": "logistic",
                    "k": int(k_log),
                    "event_rate": ev_rate,
                    "or": or_val,
                }

            elif analysis_type == "Chi-Square Test":
                c1, c2 = st.columns(2)
                with c1:
                    df_cs = st.number_input("Degrees of Freedom", 1, 50, 2, 1)
                with c2:
                    w_cs = st.number_input(
                        "Cohen's w (effect size)",
                        0.01,
                        2.0,
                        0.3,
                        0.01,
                    )
                    st.caption("Small: 0.10 | Medium: 0.30 | Large: 0.50")
                ss_params = {"type": "chisq", "df": int(df_cs), "effect_size": w_cs}

            elif analysis_type == "Mann-Whitney / Wilcoxon (Non-parametric)":
                c1, c2 = st.columns(2)
                with c1:
                    P_val = st.number_input(
                        "P(X>Y) probability", 0.51, 0.99, 0.65, 0.01
                    )
                    st.caption("Small: ~0.56 | Medium: ~0.64 | Large: ~0.71")
                with c2:
                    are_val = st.number_input("ARE", 0.5, 1.5, 0.955, 0.001)
                ratio_mw = st.number_input(
                    "Allocation Ratio (n₂/n₁)", 0.1, 10.0, 1.0, 0.1
                )
                st.caption(
                    "ARE = 0.955 at normality, lower for heavy-tailed distributions"
                )
                ss_params = {
                    "type": "mannwhitney",
                    "effect_size": P_val,
                    "ratio": ratio_mw,
                    "are": are_val,
                }

            elif analysis_type == "Log-Rank Test (Survival)":
                c1, c2 = st.columns(2)
                with c1:
                    hr_val = st.number_input("Hazard Ratio", 1.1, 10.0, 2.0, 0.1)
                with c2:
                    ratio_lr = st.number_input(
                        "Allocation Ratio (n₂/n₁)", 0.1, 10.0, 1.0, 0.1
                    )
                c1, c2 = st.columns(2)
                with c1:
                    med_val = st.number_input(
                        "Median Survival Control (months)", 1, 120, 12, 1
                    )
                with c2:
                    dur_val = st.number_input(
                        "Total Study Duration (months)", 1, 240, 36, 1
                    )
                ss_params = {
                    "type": "logrank",
                    "hr": hr_val,
                    "ratio": ratio_lr,
                    "median_survival": med_val,
                    "study_duration": dur_val,
                }

            elif analysis_type == "Cox Regression":
                c1, c2 = st.columns(2)
                with c1:
                    hr_val = st.number_input("Hazard Ratio", 1.1, 10.0, 2.0, 0.1)
                with c2:
                    k_val = st.number_input("Number of Predictors", 1, 50, 3, 1)
                c1, c2 = st.columns(2)
                with c1:
                    sd_val = st.number_input("SD of Predictor", 0.1, 10.0, 1.0, 0.1)
                with c2:
                    r2_val = st.number_input(
                        "R-squared with other covariates", 0.0, 0.99, 0.0, 0.01
                    )
                ev_val = st.number_input("Event Rate", 0.01, 0.99, 0.5, 0.01)
                ss_params = {
                    "type": "cox",
                    "hr": hr_val,
                    "k": int(k_val),
                    "sd_x": sd_val,
                    "r2_x": r2_val,
                    "event_rate": ev_val,
                }

            elif analysis_type == "Equivalence / Non-Inferiority":
                equiv_param_type = st.radio(
                    "Parameter type", ["Mean", "Proportion"], horizontal=True
                )
                c1, c2 = st.columns(2)
                with c1:
                    margin = st.number_input("Margin (delta)", 0.001, 10.0, 1.0, 0.001)
                with c2:
                    d_exp = st.number_input(
                        "Expected Difference", -10.0, 10.0, 0.0, 0.01
                    )
                c1, c2 = st.columns(2)
                p1_eq = 0.5
                p2_eq = 0.5
                with c1:
                    if equiv_param_type == "Mean":
                        sd_val = st.number_input("SD", 0.1, 100.0, 1.0, 0.1)
                    else:
                        p1_eq = st.number_input(
                            "Expected proportion (Group 1)", 0.01, 0.99, 0.2, 0.01
                        )
                        sd_val = 1.0
                with c2:
                    ratio_eq = st.number_input(
                        "Allocation Ratio (n₂/n₁)", 0.1, 10.0, 1.0, 0.1
                    )
                if equiv_param_type == "Proportion":
                    p2_eq = st.number_input(
                        "Expected proportion (Group 2)", 0.01, 0.99, 0.2, 0.01
                    )
                ss_params = {
                    "type": "equiv",
                    "margin": margin,
                    "expected_diff": d_exp,
                    "sd": sd_val,
                    "ratio": ratio_eq,
                    "equiv_param_type": equiv_param_type,
                    "p1_eq": p1_eq,
                    "p2_eq": p2_eq,
                }

            elif analysis_type == "Repeated Measures ANOVA":
                c1, c2 = st.columns(2)
                with c1:
                    f_val = st.number_input("Cohen's f", 0.01, 2.0, 0.25, 0.01)
                    st.caption("Small: 0.10 | Medium: 0.25 | Large: 0.40")
                with c2:
                    k_val = st.number_input("Number of Groups", 2, 20, 2, 1)
                c1, c2 = st.columns(2)
                with c1:
                    m_val = st.number_input("Number of Measurements", 2, 20, 3, 1)
                with c2:
                    rho_val = st.number_input(
                        "Correlation between measurements", 0.0, 0.99, 0.5, 0.01
                    )
                eps_val = st.number_input(
                    "Sphericity correction epsilon", 0.1, 1.0, 0.75, 0.01
                )
                ss_params = {
                    "type": "rm_anova",
                    "effect_size": f_val,
                    "k": int(k_val),
                    "m": int(m_val),
                    "rho": rho_val,
                    "epsilon": eps_val,
                }

            elif analysis_type == "Two-way / Factorial ANOVA":
                c1, c2 = st.columns(2)
                with c1:
                    r_val = st.number_input("Rows (Factor A levels)", 2, 10, 2, 1)
                with c2:
                    c_val = st.number_input("Columns (Factor B levels)", 2, 10, 2, 1)
                c1, c2, c3 = st.columns(3)
                with c1:
                    f_a = st.number_input(
                        "Cohen's f for Factor A", 0.01, 2.0, 0.25, 0.01
                    )
                    st.caption("Small: 0.10 | Medium: 0.25 | Large: 0.40")
                with c2:
                    f_b = st.number_input(
                        "Cohen's f for Factor B", 0.01, 2.0, 0.25, 0.01
                    )
                with c3:
                    f_ab = st.number_input(
                        "Cohen's f for interaction", 0.01, 2.0, 0.25, 0.01
                    )
                focus = st.radio(
                    "Effect of interest",
                    ["Main Effect A", "Main Effect B", "Interaction"],
                    horizontal=True,
                )
                ss_params = {
                    "type": "twoway_anova",
                    "f_a": f_a,
                    "f_b": f_b,
                    "f_ab": f_ab,
                    "rows": int(r_val),
                    "cols": int(c_val),
                    "focus": focus,
                }

            elif analysis_type == "ROC / AUC Analysis":
                c1, c2 = st.columns(2)
                with c1:
                    auc_val = st.number_input("Expected AUC", 0.5, 0.99, 0.7, 0.01)
                with c2:
                    st.number_input("Null AUC", 0.5, 0.5, 0.5, disabled=True)
                ratio_roc = st.number_input(
                    "Ratio of controls to cases", 0.1, 10.0, 1.0, 0.1
                )
                ss_params = {
                    "type": "roc_auc",
                    "auc": auc_val,
                    "null_auc": 0.5,
                    "ratio": ratio_roc,
                }

            elif analysis_type == "Cohen's Kappa / ICC Agreement":
                atype = st.radio("Type", ["Cohen's Kappa", "ICC"], horizontal=True)
                c1, c2 = st.columns(2)
                with c1:
                    kappa_val = st.number_input("Expected Kappa", 0.01, 0.99, 0.6, 0.01)
                with c2:
                    null_kap = st.number_input("Null Kappa", 0.0, 0.5, 0.0, 0.01)
                c1, c2 = st.columns(2)
                with c1:
                    raters = st.number_input("Number of Raters", 2, 10, 2, 1)
                with c2:
                    cats = st.number_input("Number of Categories", 2, 10, 2, 1)
                ss_params = {
                    "type": "kappa",
                    "kappa": kappa_val,
                    "null_kappa": null_kap,
                    "raters": int(raters),
                    "categories": int(cats),
                    "agreement_type": atype,
                }

            elif analysis_type == "Cluster-RCT / Multilevel":
                c1, c2 = st.columns(2)
                with c1:
                    d_val = st.number_input("Effect size d", 0.1, 5.0, 0.5, 0.01)
                    st.caption("Small: 0.20 | Medium: 0.50 | Large: 0.80")
                with c2:
                    icc_val = st.number_input(
                        "ICC", 0.001, 0.5, 0.05, 0.001, format="%.3f"
                    )
                c1, c2 = st.columns(2)
                with c1:
                    m_val = st.number_input("Cluster size (m)", 2, 1000, 30, 1)
                with c2:
                    ratio_cl = st.number_input(
                        "Allocation Ratio (n₂/n₁)", 0.1, 10.0, 1.0, 0.1
                    )
                ss_params = {
                    "type": "cluster_rct",
                    "effect_size": d_val,
                    "icc": icc_val,
                    "cluster_size": int(m_val),
                    "ratio": ratio_cl,
                }

            elif analysis_type == "Precision-based (CI Width)":
                ptype = st.radio(
                    "Type of parameter", ["Mean", "Proportion"], horizontal=True
                )
                c1, c2 = st.columns(2)
                with c1:
                    hw_val = st.number_input(
                        "Desired half-width of CI", 0.01, 100.0, 5.0, 0.01
                    )
                with c2:
                    cl_val = st.number_input("Confidence Level %", 80, 99, 95, 1)
                if ptype == "Mean":
                    sd_val = st.number_input("SD", 0.1, 100.0, 10.0, 0.1)
                    prop_val = 0.5
                else:
                    sd_val = 1.0
                    prop_val = st.number_input(
                        "Expected Proportion", 0.01, 0.99, 0.5, 0.01
                    )
                ss_params = {
                    "type": "precision",
                    "half_width": hw_val,
                    "conf_level": cl_val,
                    "param_type": ptype,
                    "sd": sd_val,
                    "prop": prop_val,
                }

            elif analysis_type == "Pilot / Feasibility Study":
                method = st.radio(
                    "Method",
                    ["Rule of thumb", "Precision-based", "Fraction of main study"],
                    horizontal=True,
                )
                if method == "Rule of thumb":
                    npg_val = st.number_input("Participants per group", 5, 100, 12, 1)
                    ss_params = {
                        "type": "pilot",
                        "method": method,
                        "n_per_group": int(npg_val),
                    }
                elif method == "Precision-based":
                    c1, c2 = st.columns(2)
                    with c1:
                        hw_val = st.number_input(
                            "Desired half-width of CI", 0.01, 100.0, 5.0, 0.01
                        )
                    with c2:
                        cl_val = st.number_input("Confidence Level %", 80, 99, 95, 1)
                    sd_val = st.number_input("SD", 0.1, 100.0, 10.0, 0.1)
                    ss_params = {
                        "type": "pilot",
                        "method": method,
                        "half_width": hw_val,
                        "conf_level": cl_val,
                        "param_type": "Mean",
                        "sd": sd_val,
                        "prop": 0.5,
                    }
                else:
                    main_n = st.number_input("Expected main study N", 10, 10000, 100, 1)
                    fraction = st.number_input("Fraction", 0.05, 0.5, 0.1, 0.01)
                    ss_params = {
                        "type": "pilot",
                        "method": method,
                        "fraction": fraction,
                        "main_n": int(main_n),
                    }

            elif analysis_type == "Wilcoxon Signed-Rank (paired)":
                c1, c2 = st.columns(2)
                with c1:
                    pr_pos = st.number_input(
                        "Pr(positive difference)", 0.51, 0.99, 0.65, 0.01
                    )
                    st.caption("Small: ~0.56 | Medium: ~0.64 | Large: ~0.71")
                with c2:
                    are_wsr = st.number_input(
                        "ARE vs paired t-test", 0.5, 1.5, 0.955, 0.001
                    )
                st.caption(
                    "ARE = 0.955 at normality, lower for heavy-tailed distributions"
                )
                ss_params = {
                    "type": "wilcoxon_sr",
                    "effect_size": pr_pos,
                    "are": are_wsr,
                }

            elif analysis_type == "Kruskal-Wallis Test":
                c1, c2 = st.columns(2)
                with c1:
                    k_kw = st.number_input("Number of Groups", 3, 20, 3, 1)
                with c2:
                    f_kw = st.number_input(
                        "Cohen's f (effect size)", 0.01, 2.0, 0.25, 0.01
                    )
                    st.caption("Small: 0.10 | Medium: 0.25 | Large: 0.40")
                are_kw = st.number_input(
                    "ARE vs ANOVA (asymptotic relative efficiency)",
                    0.15, 1.5, 0.955, 0.001,
                    help="ARE = 0.955 at normality, lower for heavy-tailed distributions. Inflates N by 1/ARE.",
                )
                st.caption(f"Effective inflation = {1/are_kw:.2f}× (N_multiplier = {1/are_kw:.3f})")
                ss_params = {"type": "kruskal", "k": int(k_kw), "effect_size": f_kw, "are": are_kw}

            elif analysis_type == "Friedman Test":
                c1, c2 = st.columns(2)
                with c1:
                    k_fr = st.number_input("Number of Groups", 2, 20, 3, 1)
                with c2:
                    m_fr = st.number_input("Number of Measurements", 2, 20, 3, 1)
                c1, c2 = st.columns(2)
                with c1:
                    w_fr = st.number_input("Kendall's W", 0.01, 0.99, 0.3, 0.01)
                    st.caption("Small: 0.10 | Medium: 0.30 | Large: 0.50")
                with c2:
                    are_fr = st.number_input("ARE vs RM-ANOVA", 0.5, 1.5, 0.955, 0.001)
                ss_params = {
                    "type": "friedman",
                    "k": int(k_fr),
                    "m": int(m_fr),
                    "w": w_fr,
                    "are": are_fr,
                }

            elif analysis_type == "McNemar's Test":
                c1, c2 = st.columns(2)
                with c1:
                    p_b = st.number_input("Discordant prop (b)", 0.01, 0.99, 0.2, 0.01)
                with c2:
                    p_c = st.number_input("Discordant prop (c)", 0.01, 0.99, 0.4, 0.01)
                ss_params = {"type": "mcnemar", "p_b": p_b, "p_c": p_c}

            elif analysis_type == "Fisher's Exact Test":
                c1, c2, c3 = st.columns(3)
                with c1:
                    p1_fish = st.number_input(
                        "Proportion Group 1", 0.01, 0.99, 0.3, 0.01
                    )
                with c2:
                    p2_fish = st.number_input(
                        "Proportion Group 2", 0.01, 0.99, 0.5, 0.01
                    )
                with c3:
                    ratio_fish = st.number_input(
                        "Allocation Ratio (n₂/n₁)", 0.1, 10.0, 1.0, 0.1
                    )
                are_fish = st.number_input(
                    "ARE vs z-test (asymptotic relative efficiency)",
                    0.5, 1.0, 0.833, 0.001,
                    help="ARE ≈ 0.833 is the standard adjustment for Fisher's exact vs z-test. Lower values increase N.",
                )
                st.caption(f"Effective inflation = {1/are_fish:.2f}×")
                ss_params = {
                    "type": "fisher",
                    "p1": p1_fish,
                    "p2": p2_fish,
                    "ratio": ratio_fish,
                    "are": are_fish,
                }

            elif analysis_type == "MANOVA (Multivariate ANOVA)":
                c1, c2 = st.columns(2)
                with c1:
                    k_man = st.number_input("Number of Groups", 2, 20, 3, 1)
                with c2:
                    dv_man = st.number_input("Number of DVs", 2, 20, 3, 1)
                manova_test = st.selectbox(
                    "Test statistic",
                    ["Pillai's Trace", "Wilks' Lambda", "Hotelling-Lawley Trace", "Roy's Largest Root"],
                    help="Pillai: most robust, recommended. Wilks: traditional. Hotelling: more power when assumptions met. Roy: most powerful when one dimension dominates.",
                )
                c1, c2 = st.columns(2)
                with c1:
                    f2_man = st.number_input(
                        "Effect size f²(V)", 0.01, 2.0, 0.0625, 0.001, format="%.4f"
                    )
                    st.caption("Small: 0.01 | Medium: 0.0625 | Large: 0.16")
                with c2:
                    corr_man = st.number_input(
                        "Correlation among DVs", 0.0, 0.99, 0.5, 0.01
                    )
                ss_params = {
                    "type": "manova",
                    "k": int(k_man),
                    "dv": int(dv_man),
                    "f2": f2_man,
                    "rho": corr_man,
                    "manova_test": manova_test,
                }

            elif analysis_type == "Binomial Exact Test":
                c1, c2 = st.columns(2)
                with c1:
                    p0_bin = st.number_input(
                        "Null proportion (π₀)", 0.01, 0.99, 0.5, 0.01
                    )
                with c2:
                    p1_bin = st.number_input(
                        "Expected proportion (π₁)", 0.01, 0.99, 0.7, 0.01
                    )
                ss_params = {"type": "binomial", "p0": p0_bin, "p1": p1_bin}

            elif analysis_type == "Simulation-based Power (Monte Carlo)":
                sim_test = st.selectbox(
                    "Statistical test to simulate",
                    ["Independent t-test (pooled)", "Welch's t-test", "Mann-Whitney U test", "Two-proportion z-test"],
                )
                n_sim = st.number_input("Number of simulations", 100, 10000, 1000, 100, help="Higher = more precise but slower.")
                if sim_test in ("Independent t-test (pooled)", "Welch's t-test", "Mann-Whitney U test"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        mu1_s = st.number_input("Mean of Group 1", -100.0, 100.0, 0.0, 0.1)
                    with c2:
                        mu2_s = st.number_input("Mean of Group 2", -100.0, 100.0, 0.5, 0.1)
                    with c3:
                        sd_s = st.number_input("SD (both groups)", 0.1, 100.0, 1.0, 0.1)
                    n_per_s = st.number_input("N per group", 5, 5000, 50, 5)
                    dist_type = st.radio(
                        "Distribution shape",
                        ["Normal", "Skewed (Exponential)", "Heavy-tailed (Uniform)"],
                        horizontal=True,
                        help="Normal = standard normal. Exponential = skewed right. Uniform = light tails.",
                    )
                    ss_params = {
                        "type": "simulation",
                        "sim_test": sim_test,
                        "n_sim": int(n_sim),
                        "mu1": mu1_s, "mu2": mu2_s,
                        "sd": sd_s,
                        "n_per": int(n_per_s),
                        "dist": dist_type,
                    }
                else:
                    p1_s = st.number_input("Proportion in Group 1", 0.01, 0.99, 0.3, 0.01)
                    p2_s = st.number_input("Proportion in Group 2", 0.01, 0.99, 0.5, 0.01)
                    n_per_s = st.number_input("N per group", 5, 5000, 100, 5)
                    ss_params = {
                        "type": "simulation",
                        "sim_test": sim_test,
                        "n_sim": int(n_sim),
                        "p1_s": p1_s, "p2_s": p2_s,
                        "n_per": int(n_per_s),
                    }

            # Apply effect size converter value if present
            conv_es = st.session_state.pop("converted_es", None)
            conv_type = st.session_state.pop("converted_type", None)
            if conv_es is not None and conv_type is not None:
                atype_key = ss_params.get("type", "")
                if conv_type == "d" and atype_key in (
                    "one_mean",
                    "two_means",
                    "paired",
                    "cluster_rct",
                ):
                    ss_params["effect_size"] = conv_es
                elif conv_type == "r" and atype_key == "correlation":
                    ss_params["effect_size"] = conv_es
                elif conv_type == "f" and atype_key in (
                    "anova",
                    "rm_anova",
                    "twoway_anova",
                    "kruskal",
                ):
                    ss_params["effect_size"] = conv_es
                elif conv_type == "f2" and atype_key == "regression":
                    ss_params["effect_size"] = conv_es
                elif conv_type == "or" and atype_key == "logistic":
                    ss_params["or"] = conv_es
                elif conv_type == "w" and atype_key == "chisq":
                    ss_params["effect_size"] = conv_es
                elif conv_type == "d" and atype_key == "wilcoxon_sr":
                    from scipy.stats import norm

                    p_conv = 0.5 + conv_es / (2 * np.sqrt(3))
                    ss_params["effect_size"] = max(0.51, min(0.99, p_conv))

            # =========================
            # STUDY ADJUSTMENTS
            # =========================
            with st.expander("⚙️ Study Adjustments"):
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    adjust_attrition = st.checkbox(
                        "Adjust for dropout rate", value=False
                    )
                with col_d2:
                    dropout_rate = (
                        st.slider(
                            "Expected dropout rate",
                            0.0,
                            0.5,
                            0.1,
                            0.01,
                            disabled=not adjust_attrition,
                        )
                        if adjust_attrition
                        else 0.0
                    )

                adjust_multiple = st.checkbox(
                    "Multiple testing correction"
                )
                if adjust_multiple:
                    mc_method = st.selectbox(
                        "Correction method",
                        ["Bonferroni", "Holm-Bonferroni", "Benjamini-Hochberg (FDR)"],
                        help="Bonferroni: α/m (most conservative). Holm: sequential Bonferroni. BH-FDR: controls false discovery rate (less conservative).",
                    )
                    num_tests = st.number_input(
                        "Number of tests/comparisons",
                        1,
                        100,
                        1,
                        1,
                    )
                else:
                    mc_method = "None"
                    num_tests = 1

                show_budget = st.checkbox("Show budget / feasibility estimates")
                if show_budget:
                    c1, c2 = st.columns(2)
                    with c1:
                        cost_per = st.number_input(
                            "Cost per participant ($)", 0.0, 100000.0, 100.0, 10.0
                        )
                    with c2:
                        recruitment_rate = st.number_input(
                            "Recruitment rate (per month)", 0.0, 1000.0, 10.0, 1.0
                        )
                else:
                    cost_per = 0.0
                    recruitment_rate = 0.0

            ss_params["dropout_rate"] = dropout_rate if adjust_attrition else 0.0
            ss_params["num_tests"] = num_tests if adjust_multiple else 1
            ss_params["mc_method"] = mc_method
            ss_params["cost_per"] = cost_per if show_budget else 0.0
            ss_params["recruitment_rate"] = recruitment_rate if show_budget else 0.0

            # =========================
            # EFFECT SIZE CONVERTER
            # =========================
            with st.expander("📐 Effect Size Converter"):
                st.caption(
                    "Convert between common effect size measures. Click Apply to use the converted value."
                )
                conv_tab = st.radio(
                    "Conversion",
                    [
                        "Means → d",
                        "d ↔ r",
                        "d ↔ OR",
                        "η² ↔ f",
                        "R² ↔ f²",
                        "2×2 Table → w/OR",
                        "P(X>Y) ↔ d / Cliff's δ",
                    ],
                    horizontal=True,
                    label_visibility="collapsed",
                )
                import math as cmath

                if conv_tab == "Means → d":
                    c1, c2 = st.columns(2)
                    with c1:
                        m1_c = st.number_input(
                            "Mean 1", 0.0, 100.0, 0.0, 0.1, key="conv_m1"
                        )
                        m2_c = st.number_input(
                            "Mean 2", 0.0, 100.0, 1.0, 0.1, key="conv_m2"
                        )
                    with c2:
                        sd_c = st.number_input(
                            "Pooled SD", 0.1, 100.0, 1.0, 0.1, key="conv_sd"
                        )
                    d_c = abs(m1_c - m2_c) / sd_c if sd_c > 0 else 0
                    st.metric("Cohen's d", f"{d_c:.4f}")
                    if st.button("Apply d to current test", key="apply_d_means"):
                        st.session_state.converted_es = d_c
                        st.session_state.converted_type = "d"
                        st.rerun()
                elif conv_tab == "d ↔ r":
                    c1, c2 = st.columns(2)
                    with c1:
                        d_c = st.number_input(
                            "Cohen's d", 0.01, 10.0, 0.5, 0.01, key="conv_dr_d"
                        )
                    r_c = d_c / cmath.sqrt(d_c**2 + 4)
                    c2.metric("Correlation r", f"{r_c:.4f}")
                    if st.button("Apply r to Correlation test", key="apply_dr"):
                        st.session_state.converted_es = r_c
                        st.session_state.converted_type = "r"
                        st.rerun()
                elif conv_tab == "d ↔ OR":
                    c1, c2 = st.columns(2)
                    with c1:
                        d_c = st.number_input(
                            "Cohen's d", 0.01, 10.0, 0.5, 0.01, key="conv_do_d"
                        )
                    or_c = cmath.exp(d_c * cmath.pi / cmath.sqrt(3))
                    c2.metric("Odds Ratio", f"{or_c:.4f}")
                    if st.button("Apply OR to Logistic Regression", key="apply_do"):
                        st.session_state.converted_es = or_c
                        st.session_state.converted_type = "or"
                        st.rerun()
                elif conv_tab == "η² ↔ f":
                    c1, c2 = st.columns(2)
                    with c1:
                        eta2 = st.number_input(
                            "η²", 0.001, 0.99, 0.06, 0.001, key="conv_eta"
                        )
                    f_c = cmath.sqrt(eta2 / (1 - eta2))
                    c2.metric("Cohen's f", f"{f_c:.4f}")
                    if st.button("Apply f to ANOVA tests", key="apply_eta"):
                        st.session_state.converted_es = f_c
                        st.session_state.converted_type = "f"
                        st.rerun()
                elif conv_tab == "R² ↔ f²":
                    c1, c2 = st.columns(2)
                    with c1:
                        r2_c = st.number_input(
                            "R²", 0.001, 0.99, 0.15, 0.001, key="conv_r2"
                        )
                    f2_c = r2_c / (1 - r2_c) if r2_c < 1 else 0
                    c2.metric("Cohen's f²", f"{f2_c:.4f}")
                    if st.button("Apply f² to Regression test", key="apply_r2"):
                        st.session_state.converted_es = f2_c
                        st.session_state.converted_type = "f2"
                        st.rerun()
                elif conv_tab == "2×2 Table → w/OR":
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        a_t = st.number_input("Cell a", 0, 1000, 30, 1, key="conv_a")
                    with c2:
                        b_t = st.number_input("Cell b", 0, 1000, 20, 1, key="conv_b")
                    with c3:
                        c_t = st.number_input("Cell c", 0, 1000, 20, 1, key="conv_c")
                    with c4:
                        d_t = st.number_input("Cell d", 0, 1000, 30, 1, key="conv_d")
                    n_t = a_t + b_t + c_t + d_t
                    if n_t > 0:
                        p_exp = (a_t + c_t) / n_t
                        p_nexp = (b_t + d_t) / n_t
                        prop_diff = (
                            abs(a_t / (a_t + b_t) - c_t / (c_t + d_t))
                            if (a_t + b_t) > 0 and (c_t + d_t) > 0
                            else 0
                        )
                        or_t = (
                            (a_t * d_t) / (b_t * c_t) if b_t > 0 and c_t > 0 else None
                        )
                        chi2_t = (
                            n_t
                            * (abs(a_t * d_t - b_t * c_t) - n_t / 2) ** 2
                            / ((a_t + b_t) * (c_t + d_t) * (a_t + c_t) * (b_t + d_t))
                            if all(
                                x > 0
                                for x in [a_t + b_t, c_t + d_t, a_t + c_t, b_t + d_t]
                            )
                            else 0
                        )
                        w_t = cmath.sqrt(chi2_t / n_t) if n_t > 0 else 0
                        c1, c2 = st.columns(2)
                        c1.metric("Cohen's w", f"{w_t:.4f}")
                        if or_t:
                            c2.metric("Odds Ratio", f"{or_t:.4f}")
                        if st.button("Apply w to Chi-Square test", key="apply_2x2"):
                            st.session_state.converted_es = w_t
                            st.session_state.converted_type = "w"
                            st.rerun()
                elif conv_tab == "P(X>Y) ↔ d / Cliff's δ":
                    conv_dir = st.radio(
                        "Direction",
                        ["P(X>Y) → d / Cliff's δ", "Cliff's δ → d / P(X>Y)"],
                        horizontal=True,
                    )
                    if conv_dir == "P(X>Y) → d / Cliff's δ":
                        p_xy = st.number_input(
                            "P(X>Y) probability (common language effect size)",
                            0.51, 0.99, 0.65, 0.01,
                            help="Probability that a random observation from Group 1 exceeds one from Group 2.",
                        )
                        d_np = np.sqrt(3) * (p_xy - 0.5) * 2
                        cliff_d = 2 * p_xy - 1
                        c1, c2 = st.columns(2)
                        c1.metric("Cohen's d (approx)", f"{d_np:.4f}")
                        c2.metric("Cliff's δ / Glass r_b", f"{cliff_d:.4f}")
                        if st.button("Apply d to Mann-Whitney/Wilcoxon", key="apply_pxy_d"):
                            st.session_state.converted_es = d_np
                            st.session_state.converted_type = "d"
                            st.rerun()
                    else:
                        cliff_in = st.number_input(
                            "Cliff's δ (or Glass rank-biserial r)",
                            -1.0, 1.0, 0.3, 0.01,
                        )
                        p_xy_out = (cliff_in + 1) / 2
                        d_np_out = np.sqrt(3) * cliff_in
                        c1, c2 = st.columns(2)
                        c1.metric("P(X>Y)", f"{p_xy_out:.4f}")
                        c2.metric("Cohen's d (approx)", f"{d_np_out:.4f}")
                        if st.button("Apply d to Mann-Whitney/Wilcoxon", key="apply_cliff_d"):
                            st.session_state.converted_es = abs(d_np_out)
                            st.session_state.converted_type = "d"
                            st.rerun()

        else:
            # =========================
            # VARIABLES (existing)
            # =========================
            st.subheader("2. Variables")
            st.markdown("##### :green[Dependent Variable]")
            Dependent_Variable = st.selectbox(
                """Outcome / Target Variable / Y variable / Response Variable / Predicted Variable / Disease / Event / Output / Measured Variable / Result / Effect / Endpoint""",
                [
                    "Binary/Dichotomous",
                    "Categorical",
                    "Ordinal",
                    "Discrete",
                    "Continuous",
                    "Multiple Continuous",
                    "Time-to-event",
                ],
            )
            st.markdown("##### :red[Independent Variable]")
            Independent_Variable = st.selectbox(
                """Predictor / Explanatory Variable / X variable / Grouping variable / Exposure / Intervention / Treatment / Risk Factor / Input / Covariate / Control Variable""",
                [
                    "Binary/Dichotomous",
                    "Categorical",
                    "Ordinal",
                    "Discrete",
                    "Continuous",
                    "Multiple Continuous",
                    "None",
                ],
            )

            # =========================
            # DESIGN (existing)
            # =========================
            st.subheader("3. Experimental Design")

            Groups = st.selectbox(
                "Number of Groups",
                [
                    "1",
                    "2",
                    "More than 2",
                    "any",
                ],
            )

            Relation = st.selectbox(
                "Relationship Type",
                [
                    "Independent",
                    "Dependent",
                    "any",
                ],
            )

            Distribution = st.selectbox(
                "Distribution",
                [
                    "Normal",
                    "Non-normal",
                    "any",
                ],
            )

        # =========================
        # USER INPUT OR SAMPLE SIZE
        # =========================
        if Objective != "Sample Size Estimation":
            user_input = {
                "Objective": Objective,
                "Dependent_Variable": Dependent_Variable,
                "Independent_Variable": Independent_Variable,
                "Groups": Groups,
                "Relation": Relation,
                "Distribution": Distribution,
            }

            if st.button("Find My Test", use_container_width=True):
                st.session_state.results = find_matching_tests(user_input)
                st.session_state.open_tests = set()
                st.session_state.power_params = None
        else:
            if st.button(
                "Calculate Sample Size",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.power_params = {
                    "analysis_type": analysis_type,
                    "alpha": alpha_ss,
                    "power": power_ss,
                    "tails": tails_ss,
                    **ss_params,
                }
                st.session_state.results = None

    with col_right:
        # SAMPLE SIZE ESTIMATION RESULTS
        if Objective == "Sample Size Estimation" and st.session_state.get(
            "power_params"
        ):
            render_power_calculator(st.session_state.power_params)

        # TEST FINDER RESULTS
        elif st.session_state.results is not None:
            if st.session_state.results:
                st.success("Recommended Statistical Test(s):")

                for test in st.session_state.results:
                    rule = next((r for r in rules if r["name"] == test), None)
                    if rule:
                        is_open = test in st.session_state.open_tests
                        btn_label = f"▶ {test}" if not is_open else f"▼ {test}"

                        if st.button(
                            btn_label, key=f"btn_{test}", use_container_width=True
                        ):
                            if is_open:
                                st.session_state.open_tests.remove(test)
                            else:
                                st.session_state.open_tests.add(test)
                            st.rerun()

                        if test in st.session_state.open_tests:
                            if "Explanation" in rule:
                                st.markdown("## Explanation:")
                                st.markdown(rule["Explanation"])
                            if "Example" in rule:
                                st.markdown("## Example:")
                                st.markdown(rule["Example"])
                            if "Formula" in rule:
                                st.markdown("## Formula:")
                                render_latex(rule["Formula"])
                            render_test_widget(test)
                            ss_type = TEST_TO_SS_TYPE.get(test)
                            if ss_type:
                                if st.button(
                                    f"📐 Estimate sample size for this test",
                                    key=f"ss_link_{test}",
                                ):
                                    st.session_state.ss_pending_obj = (
                                        "Sample Size Estimation"
                                    )
                                    st.session_state.ss_pending_at = ss_type
                                    st.session_state.results = None
                                    st.session_state.power_params = None
                                    st.rerun()
                            st.markdown("---")

            else:
                st.error(
                    "No matching statistical test found. Try adjusting your selections."
                )
        else:
            if Objective == "Sample Size Estimation":
                st.info("Select your parameters and click 'Calculate Sample Size'.")
            else:
                st.info("Results will appear here once you click 'Find My Test'.")

    # =========================
    # FLOWCHART MODE
    # =========================

    if Objective != "Sample Size Estimation":
        st.divider()
        st.header("🌳 Interactive Statistical Flowchart")
        st.write(
            "Expand the branches below to navigate statistical test selection visually."
        )
        build_tree(rules, FIELDS, user_input)

    # =========================
    # FOOTER
    # =========================
    st.markdown("---")

    footer_html = """
<div style="padding: 20px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.3); margin-bottom: 20px; text-align: center;">
    <h3 style="margin-top: 0; color: #4CAF50;">Developed By</h3>
    <p style="font-size: 1.2em; margin-bottom: 5px;"><strong>Dr. Muhammad Nabeel Shaesha</strong></p>
    <p style="margin: 0; opacity: 0.8;">Teaching Assistant at the Prosthodontics Department, PUA</p>
    <p style="margin: 0; opacity: 0.8;">Currently enrolled in Masters of Prosthodontics and Implantology Program, PUA</p>
    <div style="margin-top: 20px;">
        <p style="font-size: 0.9em; opacity: 0.7; margin-bottom: 10px;">Built with the help of:</p>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;">
            <div style="border: 2px solid #CA6180; padding: 5px 15px; border-radius: 5px; font-weight: bold; color: #CA6180;">
                Gemma 4
            </div>
            <div style="border: 2px solid #4B9DA9; padding: 5px 15px; border-radius: 5px; font-weight: bold; color: #4B9DA9;">
                OpenCode
            </div>
            <div style="border: 2px solid #8E24AA; padding: 5px 15px; border-radius: 5px; font-weight: bold; color: #8E24AA;">
                GeminiCLI
            </div>
            <div style="border: 2px solid #10a37f; padding: 5px 15px; border-radius: 5px; font-weight: bold; color: #10a37f;">
                ChatGPT
            </div>
        </div>
        <p style="font-size: 0.9em; opacity: 0.7; margin-bottom: 10px;"><br /> Acknowledgment to my professors who taught me biosatistics and research methodology</p>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 10px;">
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Inas Karawia
            </div>
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Maha Adel
            </div>
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Hamida Abu Bakr
            </div>
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Hadeya Abdel Hamid
            </div>
            <div style="border: 2px solid #4285F4; padding: 2px 15px; border-radius: 5px; color: #4285F4;">
                Dr Nancy Bedwany
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; opacity: 0.6; font-size: 0.8em;">
    <p><strong>⚠️ Disclaimer</strong></p>
    <p>This tool is intended for <strong>educational and informational purposes only</strong>. 
    While it follows standard statistical guidelines, it does not account for all possible 
    complexities in study design (e.g., nesting, interaction effects, or specific data anomalies). 
    Recommendations should be verified by a qualified biostatistician or through standard 
    statistical literature before being used for clinical or formal research purposes.</p>
    <p>© 2026 Statistical Test Finder. Built with Streamlit.</p>
</div>
"""
    st.markdown(footer_html, unsafe_allow_html=True)






# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
