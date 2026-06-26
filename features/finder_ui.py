import streamlit as st
from core.matching import find_matching_tests
from core.data import rules, FIELDS
from features.widgets import render_latex, render_test_widget
from features.glossary import render_glossary


# Cross-link: test → diagnostics
_TEST_DIAG_MAP = {
    # One-sample continuous parametric
    "One-sample t-test": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Verifies the sample follows a normal distribution"),
        ("Grubbs' Test", "Outlier Detection Tests",
         "Detects outliers that could bias the mean"),
    ],
    "One-sample z-test": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Verifies normality (or n ≥ 30 for CLT)"),
        ("Grubbs' Test", "Outlier Detection Tests",
         "Detects outliers that could bias the mean"),
    ],
    # One-sample non-parametric / binary
    "One-sample Wilcoxon Signed-Rank Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Flags extreme values that may affect ranks"),
    ],
    "Sign Test (One-sample)": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Flags extreme values that may affect the sign test"),
    ],
    # Two-sample independent parametric
    "Student's t-test (Independent)": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality within each group"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Verifies equal variances between groups"),
        ("Grubbs' Test", "Outlier Detection Tests",
         "Detects outliers within each group"),
    ],
    "Welch's t-test (Independent, Unequal Variances)": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality within each group"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Detects outliers within each group"),
    ],
    "F-Test for Two Variances": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "F-test is sensitive to non-normality"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Outliers inflate variance estimates"),
    ],
    "Equivalence Test (TOST) - Two Independent Samples": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "TOST assumes normality for accurate CIs"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Equal variances simplify equivalence bounds"),
    ],
    "Yuen's Trimmed t-test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Trimming addresses outliers, but check extent"),
    ],
    # Two-sample dependent parametric
    "Paired t-test": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality of paired differences"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Detects extreme paired differences"),
    ],
    "Wilcoxon Signed-Rank Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Flags extreme rank differences"),
    ],
    "Sign Test (Paired)": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Flags extreme differences"),
    ],
    # Multi-group parametric
    "One-way ANOVA": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality of residuals"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Verifies equal variances across groups"),
        ("Rosner's Test", "Outlier Detection Tests",
         "Detects multiple outliers across groups"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Quick outlier check per group"),
    ],
    "Two-way ANOVA": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality of residuals"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Verifies equal variances across groups"),
        ("Rosner's Test", "Outlier Detection Tests",
         "Detects multiple outliers"),
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among factors/covariates"),
    ],
    "ANCOVA": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality of residuals"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Verifies equal variances across groups"),
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among covariates"),
    ],
    "Repeated Measures ANOVA": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality of residuals"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Detects within-subject outliers"),
    ],
    "MANOVA": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks multivariate normality per group"),
        ("Mahalanobis Distance", "Outlier Detection Tests",
         "Detects multivariate outliers"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Verifies equal covariance matrices"),
    ],
    # Regression
    "Simple Linear Regression": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality of residuals"),
        ("Durbin-Watson Test", "Autocorrelation Tests",
         "Detects autocorrelation in residuals"),
        ("Breusch-Pagan Test", "Heteroscedasticity Tests",
         "Detects non-constant residual variance"),
        ("Cook's Distance", "Influence Diagnostics",
         "Identifies highly influential points"),
        ("Leverage / Hat Values", "Influence Diagnostics",
         "Flags extreme predictor values"),
        ("Grubbs' Test", "Outlier Detection Tests",
         "Detects outliers in residuals"),
    ],
    "Multiple Linear Regression": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality of residuals"),
        ("Durbin-Watson Test", "Autocorrelation Tests",
         "Detects autocorrelation in residuals"),
        ("Breusch-Pagan Test", "Heteroscedasticity Tests",
         "Detects non-constant residual variance"),
        ("White's Test", "Heteroscedasticity Tests",
         "General heteroscedasticity test"),
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Quantifies multicollinearity among predictors"),
        ("Condition Index", "Multicollinearity Diagnostics",
         "Detects severity of collinearity"),
        ("Cook's Distance", "Influence Diagnostics",
         "Identifies highly influential points"),
        ("DFFITS", "Influence Diagnostics",
         "Measures change in fitted values"),
        ("Leverage / Hat Values", "Influence Diagnostics",
         "Flags extreme predictor values"),
        ("Mahalanobis Distance", "Outlier Detection Tests",
         "Detects multivariate outliers in predictors"),
    ],
    "Logistic Regression": [
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among predictors"),
        ("Cook's Distance", "Influence Diagnostics",
         "Identifies influential observations"),
        ("Leverage / Hat Values", "Influence Diagnostics",
         "Flags extreme predictor values"),
        ("Mahalanobis Distance", "Outlier Detection Tests",
         "Detects multivariate outliers in predictors"),
    ],
    "Multinomial Logistic Regression": [
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among predictors"),
        ("Cook's Distance", "Influence Diagnostics",
         "Identifies influential observations"),
    ],
    "Ordinal Logistic Regression": [
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among predictors"),
        ("Cook's Distance", "Influence Diagnostics",
         "Identifies influential observations"),
    ],
    "Poisson Regression": [
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among predictors"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Detects outliers in count data"),
    ],
    "Negative Binomial Regression": [
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among predictors"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Detects outliers in count data"),
    ],
    # Correlation
    "Pearson Correlation": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Pearson r assumes bivariate normality"),
        ("Mahalanobis Distance", "Outlier Detection Tests",
         "Detects bivariate outliers that inflate/deflate r"),
    ],
    "Point-Biserial Correlation": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Assumes normality of the continuous variable per group"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Equal variances across binary groups"),
    ],
    # Categorical
    "Chi-Square Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns in categorical data"),
    ],
    "Chi-Square Test of Independence": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns in contingency tables"),
    ],
    "Fisher's Exact Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns in contingency tables"),
    ],
    "McNemar's Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns in paired binary data"),
    ],
    "Cochran's Q Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns across repeated binary measures"),
    ],
    "Two-sample Proportion Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns in proportion data"),
    ],
    # Non-parametric multi-group
    "Kruskal-Wallis Test": [
        ("Kolmogorov-Smirnov Test", "Normality Tests",
         "Compares distribution shapes across groups — K-S tests if groups differ in distribution"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Extreme ranks can distort the test"),
    ],
    "Mood's Median Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Extreme values affect median comparison"),
    ],
    "Friedman Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Extreme within-subject ranks"),
    ],
    # Agreement / reliability
    "Cohen's Kappa (Agreement Analysis)": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual rating patterns"),
    ],
    "Weighted Kappa": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual rating patterns with ordinal weights"),
    ],
    "Fleiss' Kappa": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns across multiple raters"),
    ],
    "Bland-Altman Analysis": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Assumes normality of differences for limits of agreement"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Extreme differences bias the limits"),
    ],
    # Survival
    "Cox Proportional Hazards Regression": [
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among predictors"),
        ("Cook's Distance", "Influence Diagnostics",
         "Identifies influential observations in Cox model"),
    ],
    # --- New tests from expansion ---
    "One-way Welch ANOVA": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks normality of residuals"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Verifies equal variances (Welch's ANOVA is robust to violations)"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Detects outliers within groups"),
    ],
    "Hotelling's T-Squared": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Checks multivariate normality per group"),
        ("Mahalanobis Distance", "Outlier Detection Tests",
         "Detects multivariate outliers"),
        ("Levene's Test", "Homogeneity of Variance Tests",
         "Verifies equal covariance matrices"),
    ],
    "Two-Sample Kolmogorov-Smirnov Test": [
        ("Kolmogorov-Smirnov Test", "Normality Tests",
         "Identifies distribution differences between groups"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Detects extreme values affecting distribution shape"),
    ],
    "Jonckheere-Terpstra Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Extreme ranks can distort ordered trend detection"),
    ],
    "Page Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Extreme within-subject ranks affect ordered trend"),
    ],
    "Mann-Kendall Trend Test": [
        ("Durbin-Watson Test", "Autocorrelation Tests",
         "Checks for serial correlation in residuals after trend removal"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Extreme values can create spurious trends"),
    ],
    "Goodness-of-Fit G-Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual cell patterns affect the G-statistic"),
    ],
    "Barnard's Exact Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns in 2x2 tables"),
    ],
    "Boschloo's Exact Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns in 2x2 tables"),
    ],
    "Stuart-Maxwell Test": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual patterns in paired categorical data"),
    ],
    "Gwet's AC1": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual rating patterns affect agreement estimates"),
    ],
    "Krippendorff's Alpha": [
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Unusual rating patterns affect reliability estimates"),
    ],
    "Intraclass Correlation Coefficient (ICC)": [
        ("Shapiro-Wilk Test", "Normality Tests",
         "Assumes normality of random effects"),
        ("IQR-Based Outlier Detection", "Outlier Detection Tests",
         "Extreme measurements inflate between-subject variance"),
    ],
    "Hosmer-Lemeshow Test": [
        ("Variance Inflation Factor (VIF)", "Multicollinearity Diagnostics",
         "Checks multicollinearity among logistic regression predictors"),
        ("Cook's Distance", "Influence Diagnostics",
         "Identifies influential observations affecting model fit"),
    ],
}


def _get_conditional(user_input):
    """Return (question_text, {option_label: [test_names]}) for disambiguation, or None."""

    obj = user_input["Objective"]
    dv = user_input["Dependent_Variable"]
    iv = user_input["Independent_Variable"]
    groups = user_input["Groups"]
    relation = user_input["Relation"]
    dist = user_input["Distribution"]

    # 1 — One-sample parametric: σ known → z-test, σ unknown → t-test
    if (
        obj == "Comparison"
        and groups == "1"
        and iv == "None"
        and dv == "Continuous"
        and dist == "Normal"
    ):
        return (
            "Do you know the population standard deviation (σ)?",
            {
                "Yes, σ is known — One-Sample Z-test": ["One-sample z-test"],
                "No, σ is unknown — One-Sample t-test": ["One-sample t-test"],
            },
        )

    # 2 — One-sample non-parametric
    if (
        obj == "Comparison"
        and groups == "1"
        and iv == "None"
        and dv in ("Ordinal", "Continuous")
        and dist == "Non-normal"
    ):
        return (
            "What is the shape of your distribution?",
            {
                "Symmetrical — One-Sample Wilcoxon Signed-Rank Test": [
                    "One-sample Wilcoxon Signed-Rank Test"
                ],
                "Heavily skewed — Sign Test": ["Sign Test (One-sample)"],
            },
        )

    # 3 — Two-sample independent parametric
    if (
        obj == "Comparison"
        and groups == "2"
        and relation == "Independent"
        and dv == "Continuous"
        and dist == "Normal"
    ):
        return (
            "What is your analysis goal?",
            {
                "Compare means, equal variances — Student's t-test": [
                    "Student's t-test (Independent)"
                ],
                "Compare means, unequal variances — Welch's t-test": [
                    "Welch's t-test (Independent, Unequal Variances)"
                ],
                "Compare means with severe outliers — Yuen's Trimmed t-test": [
                    "Yuen's Trimmed t-test"
                ],
                "Test equivalence of means within bounds — TOST": [
                    "Equivalence Test (TOST) - Two Independent Samples"
                ],
                "Compare variances — F-Test for Two Variances": [
                    "F-Test for Two Variances"
                ],
            },
        )

    # 4 — Two-sample independent non-parametric
    if (
        obj == "Comparison"
        and groups == "2"
        and relation == "Independent"
        and dv in ("Ordinal", "Continuous")
        and dist == "Non-normal"
    ):
        return (
            "Do the groups have equal spread (variance)?",
            {
                "Equal group spreads — Mann-Whitney U Test": ["Mann-Whitney U Test"],
                "Unequal group spreads — Brunner-Munzel Test": [
                    "Brunner-Munzel Test"
                ],
            },
        )

    # 5 — Two-sample binary/categorical (independent)
    if (
        obj == "Comparison"
        and groups == "2"
        and relation == "Independent"
        and dv in ("Binary/Dichotomous", "Categorical")
    ):
        return (
            "What is your analysis goal and sample size?",
            {
                "Test association (large sample) — Chi-Square Test": [
                    "Chi-Square Test"
                ],
                "Compare two specific proportions (large sample) — Two-Proportion Z-test": [
                    "Two-sample Proportion Test"
                ],
                "Small sample (expected cells < 5) — Fisher's Exact Test": [
                    "Fisher's Exact Test"
                ],
            },
        )

    # 6 — Two-sample dependent/paired non-parametric
    if (
        obj == "Comparison"
        and groups == "2"
        and relation == "Dependent"
        and dv in ("Ordinal", "Continuous")
        and dist == "Non-normal"
    ):
        return (
            "What is the shape of the paired differences?",
            {
                "Symmetrical differences — Wilcoxon Signed-Rank Test": [
                    "Wilcoxon Signed-Rank Test"
                ],
                "Heavily skewed differences — Sign Test (Paired)": ["Sign Test (Paired)"],
            },
        )

    # 7 — One-sample proportion (Binary DV)
    if (
        obj == "Comparison"
        and groups == "1"
        and iv == "None"
        and dv == "Binary/Dichotomous"
    ):
        return (
            "What is your sample size?",
            {
                "Large (np ≥ 5) — One-Sample Proportion Z-test": [
                    "One-sample Proportion Test (Binomial Test)"
                ],
                "Small (np < 5) — Exact Binomial Test": ["Binomial Test"],
            },
        )

    # 8 — One-sample categorical (multi-category DV)
    if (
        obj == "Comparison"
        and groups == "1"
        and iv == "None"
        and dv == "Categorical"
    ):
        return (
            "What hypothesis are you testing?",
            {
                "Proportions match expected (large sample) — Chi-Square GOF": [
                    "Chi-Square Goodness-of-Fit Test"
                ],
                "Proportions match expected (small sample) — Exact Multinomial Test": [
                    "Multinomial Test"
                ],
                "Data follows a Poisson distribution — Poisson GOF": [
                    "Poisson Goodness-of-Fit Test"
                ],
            },
        )

    # 9 — Multi-sample independent non-parametric
    if (
        obj == "Comparison"
        and groups == "More than 2"
        and relation == "Independent"
        and dv in ("Ordinal", "Continuous")
        and dist == "Non-normal"
        and iv == "Categorical"
    ):
        return (
            "What best describes your data?",
            {
                "Equal group shapes — Kruskal-Wallis Test": ["Kruskal-Wallis Test"],
                "Unequal shapes / outliers — Mood's Median Test": [
                    "Mood's Median Test"
                ],
            },
        )

    # 9.5 — Multi-sample independent parametric (ANOVA family)
    if (
        obj == "Comparison"
        and groups == "More than 2"
        and relation == "Independent"
        and dv == "Continuous"
        and dist == "Normal"
        and iv == "Categorical"
    ):
        return (
            "Which design applies?",
            {
                "One categorical independent variable — One-way ANOVA": [
                    "One-way ANOVA"
                ],
                "Two categorical independent variables — Two-way ANOVA": [
                    "Two-way ANOVA"
                ],
                "One categorical IV with continuous covariate(s) — ANCOVA": [
                    "ANCOVA"
                ],
            },
        )

    # 10 — Ordinal correlation (Spearman vs Kendall)
    if (
        obj == "Association/Correlation"
        and dv in ("Ordinal", "Continuous")
        and iv in ("Ordinal", "Continuous")
        and dist == "Non-normal"
    ):
        return (
            "Are there many tied ranks?",
            {
                "Few ties — Spearman Rank Correlation": ["Spearman Rank Correlation"],
                "Many ties — Kendall's Tau-b": ["Kendall's Tau-b"],
            },
        )

    # 11 — Count regression (Poisson vs Negative Binomial)
    if obj == "Prediction" and dv == "Discrete":
        return (
            "How does the variance compare to the mean?",
            {
                "Mean ≈ Variance — Poisson Regression": ["Poisson Regression"],
                "Variance > Mean (overdispersion) — Negative Binomial Regression": [
                    "Negative Binomial Regression"
                ],
            },
        )

    # 11.5 — Binary/Multinomial Logistic Regression
    if (
        obj == "Prediction"
        and dv in ("Binary/Dichotomous", "Categorical")
        and iv in ("Continuous", "Multiple Continuous", "Categorical")
    ):
        return (
            "How many outcome categories?",
            {
                "2 outcome categories — Binary Logistic Regression": [
                    "Logistic Regression"
                ],
                "3+ outcome categories — Multinomial Logistic Regression": [
                    "Multinomial Logistic Regression"
                ],
            },
        )

    # 12 — Survival two-group (Kaplan-Meier vs Log-Rank vs Cox)
    if (
        obj == "Survival Analysis"
        and dv == "Time-to-event"
        and iv == "Categorical"
        and groups == "2"
        and relation == "Independent"
    ):
        return (
            "Do you need to adjust for covariates?",
            {
                "No — estimate survival curves (Kaplan-Meier)": [
                    "Kaplan-Meier Survival Analysis"
                ],
                "No — test for group difference (Log-Rank)": ["Log-Rank Test"],
                "Yes — adjust for covariates (Cox Regression)": [
                    "Cox Proportional Hazards Regression"
                ],
            },
        )

    # 13 — Diagnostic binary metrics
    if (
        obj == "Diagnostic Accuracy"
        and dv == "Binary/Dichotomous"
        and iv == "Binary/Dichotomous"
    ):
        return (
            "What do you want to report?",
            {
                "Overall accuracy (Sensitivity / Specificity / PPV / NPV)": [
                    "Sensitivity & Specificity Analysis"
                ],
                "Likelihood ratios (LR+ / LR-)": ["Likelihood Ratio Analysis"],
            },
        )

    return None


def _categorize_tests():
    """Categorize tests by design type and parametric/non-parametric."""
    from collections import defaultdict

    def _is_parametric(rule):
        dist = rule.distribution if rule.distribution else "any"
        if isinstance(dist, str):
            return dist == "Normal"
        return "Normal" in dist and "Non-normal" not in dist

    def _is_nonparametric(rule):
        dist = rule.distribution if rule.distribution else "any"
        if isinstance(dist, str):
            return dist == "Non-normal"
        return "Non-normal" in dist and "Normal" not in dist

    def _is_any_dist(rule):
        dist = rule.distribution if rule.distribution else "any"
        if isinstance(dist, str):
            return dist == "any"
        return "any" in dist or ("Normal" in dist and "Non-normal" in dist)

    def _get_groups(groups):
        if isinstance(groups, list):
            if "1" in groups:
                return "1"
            elif "2" in groups:
                return "2"
            elif "More than 2" in groups:
                return "More than 2"
            return "any"
        return groups

    def _get_relation(relation):
        if isinstance(relation, list):
            if "Dependent" in relation:
                return "Dependent"
            elif "Independent" in relation:
                return "Independent"
            return "any"
        return relation

    categories = defaultdict(list)
    category_order = []

    for rule in rules:
        groups = _get_groups(rule.get("Groups", "any"))
        relation = _get_relation(rule.get("Relation", "any"))
        objective = rule.objective or "Unknown"

        if objective == "Association/Correlation":
            cat_key = ("Correlation & Association",)
        elif objective == "Prediction":
            cat_key = ("Regression & Prediction",)
        elif objective == "Survival Analysis":
            cat_key = ("Survival Analysis",)
        elif objective == "Diagnostic Accuracy":
            cat_key = ("Diagnostic Accuracy",)
        elif groups == "1":
            if _is_parametric(rule):
                cat_key = ("One-sample", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("One-sample", "Non-parametric")
            else:
                cat_key = ("One-sample", "Any Distribution")
        elif groups == "2" and relation == "Independent":
            if _is_parametric(rule):
                cat_key = ("Two-sample (Independent)", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("Two-sample (Independent)", "Non-parametric")
            else:
                cat_key = ("Two-sample (Independent)", "Any Distribution")
        elif groups == "2" and relation == "Dependent":
            if _is_parametric(rule):
                cat_key = ("Two-sample (Dependent/Paired)", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("Two-sample (Dependent/Paired)", "Non-parametric")
            else:
                cat_key = ("Two-sample (Dependent/Paired)", "Any Distribution")
        elif groups == "More than 2" and relation == "Independent":
            if _is_parametric(rule):
                cat_key = ("Multi-sample (Independent)", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("Multi-sample (Independent)", "Non-parametric")
            else:
                cat_key = ("Multi-sample (Independent)", "Any Distribution")
        elif groups == "More than 2" and relation == "Dependent":
            if _is_parametric(rule):
                cat_key = ("Multi-sample (Dependent/Paired)", "Parametric")
            elif _is_nonparametric(rule):
                cat_key = ("Multi-sample (Dependent/Paired)", "Non-parametric")
            else:
                cat_key = ("Multi-sample (Dependent/Paired)", "Any Distribution")
        else:
            if _is_any_dist(rule):
                cat_key = ("Other Tests", "Flexible/Any Design")
            elif _is_parametric(rule):
                cat_key = ("Other Tests", "Parametric")
            else:
                cat_key = ("Other Tests", "Non-parametric")

        if cat_key not in categories:
            category_order.append(cat_key)
        categories[cat_key].append(rule.name)

    for cat_key in categories:
        categories[cat_key] = sorted(set(categories[cat_key]))

    def _category_sort_key(key):
        priority = {
            ("One-sample", "Parametric"): 100,
            ("One-sample", "Non-parametric"): 110,
            ("One-sample", "Any Distribution"): 120,
            ("Two-sample (Independent)", "Parametric"): 200,
            ("Two-sample (Independent)", "Non-parametric"): 210,
            ("Two-sample (Independent)", "Any Distribution"): 220,
            ("Two-sample (Dependent/Paired)", "Parametric"): 300,
            ("Two-sample (Dependent/Paired)", "Non-parametric"): 310,
            ("Two-sample (Dependent/Paired)", "Any Distribution"): 320,
            ("Multi-sample (Independent)", "Parametric"): 400,
            ("Multi-sample (Independent)", "Non-parametric"): 410,
            ("Multi-sample (Independent)", "Any Distribution"): 420,
            ("Multi-sample (Dependent/Paired)", "Parametric"): 500,
            ("Multi-sample (Dependent/Paired)", "Non-parametric"): 510,
            ("Multi-sample (Dependent/Paired)", "Any Distribution"): 520,
            ("Correlation & Association",): 600,
            ("Regression & Prediction",): 700,
            ("Survival Analysis",): 800,
            ("Diagnostic Accuracy",): 900,
            ("Other Tests", "Parametric"): 1000,
            ("Other Tests", "Non-parametric"): 1010,
            ("Other Tests", "Flexible/Any Design"): 1020,
        }
        return priority.get(key, 9999)

    category_order.sort(key=_category_sort_key)

    return categories, category_order


def render_all_tests_section():
    st.divider()
    st.header("All Statistical Tests")
    st.info("Click on any test name to view it directly in the finder.")

    categories, category_order = _categorize_tests()

    prev_main_cat = None

    for cat_key in category_order:
        main_cat = cat_key[0]
        sub_cat = cat_key[1] if len(cat_key) > 1 else None

        if main_cat != prev_main_cat:
            st.divider()
            st.subheader(main_cat)
            prev_main_cat = main_cat

        if sub_cat:
            st.markdown(f"**{sub_cat}:**")
        else:
            st.markdown("")

        test_names = categories[cat_key]
        total = len(test_names)

        test_cols = st.columns(3)
        for i, test_name in enumerate(test_names):
            col_idx = i % 3
            with test_cols[col_idx]:
                btn_key = f"alltest_cat_{main_cat.replace(' ', '_')}_{test_name.replace(' ', '_')}"
                if st.button(f"📌 {test_name}", key=btn_key, use_container_width=True):
                    _open_test_directly(test_name)


def _open_test_directly(test_name):
    """Open a test directly in the finder's right panel."""
    from core.data import rules
    rule = next((r for r in rules if r.name == test_name), None)
    st.session_state.view = "detail"
    st.session_state.detail_test = test_name
    st.rerun()


# Map test names to solved example functions
_SOLVED_EXAMPLE_MAP = {}
try:
    import features.solved_examples as _solved_mod
    _SOLVED_EXAMPLE_MAP = {
        "One-sample t-test": _solved_mod._solved_ttest,
        "One-sample z-test": _solved_mod._solved_ztest,
        "One-sample Proportion Test (Binomial Test)": _solved_mod._solved_proportion,
        "Binomial Test": _solved_mod._solved_binomial_exact,
        "One-sample Wilcoxon Signed-Rank Test": _solved_mod._solved_wilcoxon,
        "Chi-Square Goodness-of-Fit Test": _solved_mod._solved_chisquare_gof,
        "Multinomial Test": _solved_mod._solved_multinomial,
        "Student's t-test (Independent)": _solved_mod._solved_students_ttest,
        "Welch's t-test (Independent, Unequal Variances)": _solved_mod._solved_welch_ttest,
        "Yuen's Trimmed t-test": _solved_mod._solved_yuen_ttest,
        "Paired t-test": _solved_mod._solved_paired_ttest,
        "Sign Test (Paired)": _solved_mod._solved_paired_sign,
        "Wilcoxon Signed-Rank Test": _solved_mod._solved_wilcoxon_paired,
    }
except Exception:
    pass


def render_test_detail_page(test_name):
    """Render a full-page detail view for a statistical test."""
    from core.data import rules

    rule = next((r for r in rules if r.name == test_name), None)
    if rule is None:
        st.error(f"Test '{test_name}' not found.")
        if st.button("← Back to Results"):
            st.session_state.view = "finder"
            st.rerun()
        return

    if st.button("← Back to Results", use_container_width=True):
        st.session_state.view = "finder"
        st.rerun()

    st.title(test_name)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        if rule.core_assumptions:
            st.subheader("Core Assumptions")
            st.info(rule.core_assumptions)

        if rule.explanation:
            st.subheader("Explanation")
            st.markdown(rule.explanation)

        if rule.formula:
            st.subheader("Formulae")
            render_latex(rule.formula)

        if rule.decision_rules:
            st.subheader("Decision Rules")
            st.info(rule.decision_rules)

        if rule.interpretation:
            st.subheader("Interpretation")
            st.success(rule.interpretation)

        if rule.post_hoc:
            st.subheader("Available Post-Hoc Tests")
            st.info("\n".join(f"- {m.strip()}" for m in rule.post_hoc.split(",")))

        # Assumption Checks — cross-link to Diagnostics app
        diag_entries = _TEST_DIAG_MAP.get(test_name)
        if diag_entries:
            st.divider()
            st.subheader("🔍 Recommended Assumption Checks")
            st.markdown(
                "Verify the test's assumptions using the **Data Screening & Diagnostics** app. "
                "Open it separately with:  \n"
                "`streamlit run apps/app_diagnostics.py`"
            )
            for diag_name, category, desc in diag_entries:
                with st.container(border=True):
                    cols = st.columns([2, 1, 4])
                    cols[0].markdown(f"**{diag_name}**")
                    cols[1].caption(f"_{category}_")
                    cols[2].markdown(desc)

    with col_right:
        if rule.example:
            st.subheader("Examples")
            st.markdown(rule.example)

        _solved_fn = _SOLVED_EXAMPLE_MAP.get(test_name)
        if _solved_fn:
            with st.expander("📝 Step-by-Step Walkthrough", expanded=False):
                _solved_fn()

        if rule.realworld_apps:
            st.subheader("Real-world Applications")
            st.info(rule.realworld_apps)

        st.subheader("Interactive Calculator")
        render_test_widget(test_name)


def render_test_finder():
    """Render the Test Finder UI."""

    if "view" not in st.session_state:
        st.session_state.view = "finder"
    if "detail_test" not in st.session_state:
        st.session_state.detail_test = None

    if st.session_state.view == "detail" and st.session_state.detail_test:
        render_test_detail_page(st.session_state.detail_test)
        return

    with st.sidebar:
        render_glossary()

    st.title("Statistical Test Finder")

    st.write(
        "Select your study characteristics to identify the appropriate statistical test."
    )

    if "results" not in st.session_state:
        st.session_state.results = None
    if "open_tests" not in st.session_state:
        st.session_state.open_tests = set()

    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("1. Research Objective")

        obj_opts = [
            "Comparison",
            "Association/Correlation",
            "Prediction",
            "Diagnostic Accuracy",
            "Survival Analysis",
        ]
        default_obj_idx = 0
        Objective = st.selectbox("What is your goal?", obj_opts, index=default_obj_idx)

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

        user_input = {
            "Objective": Objective,
            "Dependent_Variable": Dependent_Variable,
            "Independent_Variable": Independent_Variable,
            "Groups": Groups,
            "Relation": Relation,
            "Distribution": Distribution,
        }

        # Step 4: conditional refinement (shown before Find My Test)
        _cond_result = _get_conditional(user_input)
        if _cond_result:
            _question, _answer_map = _cond_result
            st.subheader("4. Refine Test Selection")
            _selected = st.radio(
                _question,
                list(_answer_map.keys()),
                index=0,
                key="finder_cond_radio",
            )

        if st.button("Find My Test", use_container_width=True):
            full_results = find_matching_tests(user_input)
            if _cond_result:
                st.session_state.results = _answer_map[_selected]
            else:
                st.session_state.results = full_results
            st.session_state.open_tests = set()
            st.session_state.power_params = None

    with col_right:
        if st.session_state.results is not None:
            if st.session_state.results:
                st.success("Recommended Statistical Test(s):")

                for test in st.session_state.results:
                    if st.button(
                        f"▶ {test}", key=f"btn_{test}", use_container_width=True
                    ):
                        st.session_state.view = "detail"
                        st.session_state.detail_test = test
                        st.rerun()

            else:
                st.error(
                    "No matching statistical test found. Try adjusting your selections."
                )
        else:
            st.info("Results will appear here once you click 'Find My Test'.")

    render_all_tests_section()
