import streamlit as st


def render_glossary():
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

    with st.expander("Data Visualization"):
        st.markdown("""
        :orange[**Histogram**]: A bar plot showing the frequency distribution of a continuous variable by dividing its range into equal-width bins. The shape reveals skewness, modality, and outliers. Sensitive to bin width — too few bins obscure detail, too many create noise.

        :orange[**Box Plot (Box-and-Whisker)**]: Displays the median, first/third quartiles (box = IQR), and whiskers extending to the furthest points within 1.5×IQR of the hinges. Points beyond whiskers are plotted individually as potential outliers. Excellent for comparing distributions across groups.

        :orange[**Violin Plot**]: Combines a box plot with a rotated kernel density estimate on each side. Shows the full distribution shape (unlike a box plot alone) while still displaying quartiles and median. Useful for comparing distribution shapes and multimodality across groups.

        :orange[**Q-Q Plot (Quantile-Quantile)**]: Plots the quantiles of the sample against the quantiles of a theoretical distribution (usually normal). Points following the diagonal line indicate the sample matches the theoretical distribution. Systematic deviations reveal skewness (curvature), heavy tails (S-shape), or outliers.

        :orange[**Scatter Plot**]: Plots pairs of (X, Y) values as individual points. Reveals the form (linear, non-linear), direction (positive, negative), strength, and outliers of the relationship between two continuous variables. Often enhanced with a smoothing line (LOESS).

        :orange[**Scatter Plot Matrix (SPLOM)**]: A grid of scatter plots for every pair of variables in a multivariate dataset. The diagonal often shows density plots or variable names. Useful for detecting pairwise patterns, collinearity, and outliers across many variables simultaneously.

        :orange[**Bar Chart**]: Displays aggregate values (counts, proportions, means) as rectangular bars. Grouped bar charts compare subgroups side by side; stacked bar charts show composition. Bars should start at zero to avoid misleading visual comparisons.

        :orange[**Forest Plot**]: A graphical display used in meta-analysis showing individual study effect sizes (squares proportional to weight) with 95% confidence intervals (horizontal lines), plus a pooled summary estimate (diamond). Heterogeneity stats (I², Q) are typically displayed alongside.

        :orange[**Kaplan–Meier Curve**]: A step-function plot of survival probability over time, accounting for censoring. Vertical drops occur at event times. Typically shown with a 95% confidence band and a risk table below. Groups are compared using the log-rank test.

        :orange[**Heatmap**]: A 2D grid colored by the value of a third variable (e.g., correlation matrix entries, gene expression levels). Paired with hierarchical dendrograms for clustering. Essential for visualizing high-dimensional patterns and correlation structures.

        :orange[**Receiver Operating Characteristic (ROC) Curve**]: Plots the true positive rate (sensitivity) against the false positive rate (1 − specificity) across all classification thresholds. The area under the curve (AUC) summarizes discriminative ability — 0.5 = random, 1.0 = perfect.

        :orange[**Calibration Plot (Reliability Diagram)**]: Plots observed event proportions against predicted probabilities, binned by risk deciles. Points near the diagonal indicate well-calibrated predictions. Systematic deviations (over/under-confidence) signal poor model calibration even with good discrimination.
        """)

    with st.expander("Study Designs"):
        st.markdown("""
        :orange[**Randomized Controlled Trial (RCT)**]: The gold standard for causal inference. Participants are randomly allocated to intervention or control groups, balancing known and unknown confounders. Provides the strongest evidence for treatment efficacy when properly blinded and analyzed by intention-to-treat.

        :orange[**Cohort Study**]: A longitudinal design that follows a group over time, comparing outcomes between exposed and unexposed individuals. Can establish temporality (cause precedes effect) but susceptible to loss to follow-up.

        :orange[**Case-Control Study**]: A retrospective design comparing individuals with a condition (cases) to those without (controls), looking back at past exposures. Efficient for rare diseases but prone to recall bias.

        :orange[**Cross-sectional Study**]: A "snapshot" measuring exposure and outcome simultaneously at one time point. Useful for estimating prevalence but cannot establish causality due to lack of temporality.
        
        :orange[**Crossover Trial**]: Each participant receives multiple treatments in a random sequence, serving as their own control. Advantages: eliminates between-subject variability, requires smaller sample size. Requirements: condition must be chronic and stable; adequate washout period needed to avoid carryover effects.

        :orange[**Factorial Design**]: Tests two or more interventions simultaneously by randomizing participants to all possible combinations (e.g., 2×2 factorial: A alone, B alone, both, neither). Efficient for studying interactions between treatments. Requires analysis by factorial ANOVA or appropriate interaction model.

        :orange[**Cluster-RCT**]: Groups (clusters — clinics, schools, communities) are randomized rather than individuals. Used when contamination between individuals is likely or the intervention is delivered at the group level. Requires larger sample sizes due to the design effect (inflated by ICC × cluster size).

        :orange[**Stepped-Wedge Cluster Trial**]: A cluster-randomized design where all clusters start in the control condition and cross over to the intervention at randomly assigned time points. Useful when it is unethical or logistically infeasible to withhold the intervention from some clusters permanently.

        :orange[**N-of-1 Trial (Single-Case Design)**]: A within-participant design where an individual receives alternating periods of treatment and placebo in random order, with repeated outcome measurement. Provides evidence for personalized treatment efficacy. Meta-analysis of multiple N-of-1 trials can inform population-level decisions.

        :orange[**Adaptive Trial Design**]: A trial with pre-specified rules for modifying aspects (sample size, allocation ratio, treatment arms) based on interim data without compromising validity. Examples: group-sequential (early stopping), response-adaptive randomization, seamless Phase II/III designs.

        :orange[**Pragmatic Trial**]: A trial designed to test an intervention under real-world conditions rather than idealized settings (explanatory trials). Features: broad eligibility criteria, flexible protocols, usual-care comparators, and patient-centered outcomes. Results are more generalizable but often less precise.

        :orange[**Equivalence / Non-Inferiority Trial**]: A trial designed to show that a new treatment is neither worse (non-inferiority) nor meaningfully different (equivalence) compared to a reference treatment. Uses a pre-specified margin (Δ). Requires a larger sample size than a superiority trial for the same power.

        :orange[**Case-Crossover Design**]: A within-subject design comparing exposure during a hazard period (just before an event) to exposure during control periods in the same individual. Controls for time-invariant confounders. Used for transient exposures with acute outcomes (e.g., triggers of myocardial infarction).

        :orange[**Instrumental Variable (IV) Design**]: A method to estimate causal effects when unmeasured confounding exists. Uses an instrument (Z) that affects the treatment (X) only through the exposure and has no direct path to the outcome (Y). Common in econometrics and observational epidemiology.

        :orange[**Difference-in-Differences (DiD)**]: A quasi-experimental design comparing the change in outcome over time between an exposed group and an unexposed group. Removes bias from time-invariant confounders. Requires the parallel trends assumption: absent exposure, both groups would have followed the same trajectory.
        """)

    with st.expander("Sampling & Research Design"):
        st.markdown("""
        :orange[**Random Sampling**]: Every member of the population has an equal and independent chance of selection. Minimizes selection bias and enables generalization to the target population.

        :orange[**Stratified Sampling**]: Dividing the population into relevant subgroups (strata), then sampling from each. Ensures adequate representation of all subgroups, especially small ones.

        :orange[**Blinding**]: Concealing group allocation to reduce bias. Single-blind: participants are unaware. Double-blind: both participants and investigators are unaware. Essential for minimizing performance and detection bias.

        :orange[**Randomization**]: Assigning participants to groups by a chance mechanism (e.g., random number generator). Balances known and unknown confounders between groups, enabling causal inference.

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

    with st.expander("Bayesian Statistics"):
        st.markdown("""
        :orange[**Prior Distribution**]: The initial probability distribution representing uncertainty about a parameter before observing any data. May be informed (based on previous studies or expert opinion) or uninformative/vague (e.g., flat or weakly regularizing priors). In Bayesian analysis, the prior is updated with observed data to produce the posterior.

        :orange[**Posterior Distribution**]: The updated probability distribution of a parameter after combining the prior with observed data via Bayes' theorem. Represents all current knowledge about the parameter, with uncertainty quantified directly by its spread (e.g., credible intervals).

        :orange[**Bayes Factor (BF)**]: The ratio of the marginal likelihood under the alternative hypothesis to that under the null hypothesis. BF₁₀ > 1 supports H₁; BF₁₀ < 1 supports H₀. Benchmarks (Jeffreys): BF 1–3 (anecdotal), 3–10 (moderate), 10–30 (strong), > 100 (decisive). Unlike p-values, Bayes factors can provide evidence *for* the null.

        :orange[**Credible Interval (CrI)**]: The Bayesian analogue of a confidence interval. A 95% credible interval contains the true parameter value with 95% posterior probability — a direct probability statement that frequentist CIs do not provide. Calculated as the highest posterior density (HPD) or equal-tailed interval.

        :orange[**Markov Chain Monte Carlo (MCMC)**]: A computational method for drawing samples from the posterior distribution when it cannot be derived analytically. Algorithms include Metropolis-Hastings, Gibbs sampling, and Hamiltonian Monte Carlo (HMC, used by Stan). Convergence is assessed using trace plots and the R-hat statistic (< 1.01 indicates convergence).

        :orange[**Highest Posterior Density (HPD) Interval**]: The narrowest interval containing a given posterior probability (e.g., 95%). Every point inside the HPD has higher density than every point outside. For symmetric unimodal posteriors, the HPD equals the equal-tailed interval; for skewed posteriors, the HPD is narrower.

        :orange[**Informative vs Weakly Informative Prior**]: An informative prior substantially concentrates mass in a specific region based on prior evidence (e.g., an odds ratio prior centered on 1 with narrow spread). A weakly informative prior provides mild regularization without dominating the data — it keeps estimates in a plausible range while allowing the likelihood to drive inference.

        :orange[**Conjugate Prior**]: A prior distribution that, when combined with a given likelihood, yields a posterior in the same distribution family. Simplifies computation (closed-form updates). Common examples: Beta prior for Binomial likelihood (Beta posterior), Normal prior for Normal likelihood (Normal posterior).

        :orange[**Posterior Predictive Check**]: A Bayesian model diagnosis where data simulated from the posterior predictive distribution are compared to the observed data. Systematic discrepancies suggest model misfit (e.g., the model cannot reproduce key features of the data).
        """)

    with st.expander("Missing Data"):
        st.markdown("""
        :orange[**MCAR (Missing Completely at Random)**]: Missingness is independent of both observed and unobserved data — the missing values are a simple random subsample of the full data. The strictest assumption; rarely plausible in practice. Under MCAR, complete-case analysis yields unbiased but less precise estimates.

        :orange[**MAR (Missing at Random)**]: Missingness depends only on observed variables, not on the missing values themselves after conditioning on what is observed. The default assumption for most modern missing data methods (e.g., multiple imputation, FIML). More plausible than MCAR in most real-world settings.

        :orange[**MNAR (Missing Not at Random)**]: Missingness depends on the missing values themselves, even after accounting for observed data. Also called non-ignorable missingness. Cannot be verified from the data alone — requires sensitivity analysis or external information (e.g., pattern-mixture models, selection models).

        :orange[**Complete-Case Analysis (Listwise Deletion)**]: Excluding any observation with one or more missing values. Simple but wasteful — reduces sample size and introduces bias unless data are MCAR. Should generally be avoided except when missingness is minimal (< 5%) and known to be MCAR.

        :orange[**Available-Case Analysis (Pairwise Deletion)**]: Using all available non-missing values for each calculation (e.g., computing each correlation from all pairs with complete data). Preserves more information than listwise deletion but can produce inconsistent sample sizes and non-positive-definite covariance matrices.

        :orange[**Multiple Imputation (MI)**]: Creating M complete datasets by imputing missing values from a predictive model that incorporates uncertainty (e.g., MICE), analyzing each separately, and pooling results using Rubin's rules. The pooled variance accounts for both within-imputation and between-imputation variability. Produces valid inference under MAR. Standard practice: M = 20–100.

        :orange[**MICE (Multiple Imputation by Chained Equations)**]: A flexible MI approach that specifies a separate conditional model for each variable with missing values (e.g., logistic for binary, linear for continuous). Imputations are generated iteratively, cycling through each variable until convergence. Handles mixed variable types naturally.

        :orange[**Full Information Maximum Likelihood (FIML)**]: A model-based approach that uses all available data to estimate parameters by maximizing the likelihood for each observation using its observed variables only. More efficient and less biased than ad-hoc methods under MAR. Built into SEM software (lavaan, Mplus) and mixed-effects models.

        :orange[**Single Imputation (Mean/Mode/Regression)**]: Filling each missing value once with a predicted value. Convenient but artificially reduces standard errors, narrows confidence intervals, and inflates Type I error rates. Regression imputation adds bias toward the imputing model. Not recommended except for exploratory work.

        :orange[**Sensitivity Analysis for Missing Data**]: A systematic exploration of how conclusions change under different assumptions about the missingness mechanism (e.g., tipping-point analysis, delta-adjustment). Essential when MNAR cannot be ruled out — shows how far results must deviate from MAR before the substantive conclusion changes.
        """)
