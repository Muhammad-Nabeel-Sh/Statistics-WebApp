import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

# Mapping from test finder rule names to sample size estimation analysis types
TEST_TO_SS_TYPE = {
    "One-sample t-test": "One-sample Mean (t/z-test)",
    "One-sample z-test": "One-sample Mean (t/z-test)",
    "Student's t-test (Independent)": "Two Independent Means (t-test)",
    "Welch's t-test (unequal variances)": "Two Independent Means (t-test)",
    "Paired t-test": "Paired Means (t-test)",
    "One-sample Proportion Test (Binomial Test)": "One-sample Proportion",
    "Two Proportion Z-test": "Two Proportions",
    "One-way ANOVA": "One-way ANOVA",
    "Pearson Correlation": "Correlation (Pearson)",
    "Multiple Linear Regression": "Multiple Linear Regression",
    "Logistic Regression": "Logistic Regression",
    "Chi-Square Test of Independence": "Chi-Square Test",
    "Chi-Square Goodness of Fit Test": "Chi-Square Test",
    "Mann-Whitney U Test": "Mann-Whitney / Wilcoxon (Non-parametric)",
    "One-sample Wilcoxon Signed-Rank Test": "Wilcoxon Signed-Rank (paired)",
    "Log-Rank Test (Survival)": "Log-Rank Test (Survival)",
    "Cox Regression": "Cox Regression",
    "Kruskal-Wallis Test": "Kruskal-Wallis Test",
    "Friedman Test": "Friedman Test",
    "McNemar's Test": "McNemar's Test",
    "Fisher's Exact Test": "Fisher's Exact Test",
    "MANOVA": "MANOVA (Multivariate ANOVA)",
    "Permutation MANOVA": "MANOVA (Multivariate ANOVA)",
}

rules = [
    # Comparison Tests
    {
        "name": "One-sample t-test",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": "any",
        "Distribution": "Normal",
        "Explanation": "One-Sample t-test This test is used to determine whether the mean of a single sample is significantly different from a known or hypothesized population mean. It assumes that the data is continuous and follows a normal distribution. It is typically used when comparing a clinical measurement (like blood pressure) against a standard clinical threshold. ",
        "Example": "A researcher wants to test if the average systolic blood pressure of a group of patients is significantly different from the standard threshold of 120 mmHg. The researcher collects blood pressure readings from 30 patients and performs a one-sample t-test to compare the sample mean against the known population mean of 120 mmHg.",
        "Formula": r"""
                    $$ t = \dfrac{\bar{x} - \mu_0}{\dfrac{s}{\sqrt{n}}}$$ 
                    Where: 
                    - :orange[$\bar{x}$] is the sample mean,   
                    - :orange[$\mu_0$] is the population mean, 
                    - :orange[$s$] is the sample standard deviation, 
                    - :orange[$n$] is the sample size 
                    - :orange[$\dfrac{s}{\sqrt{n}}$] is the standard error of the mean.
                    """,
    },
    {
        "name": "One-sample z-test",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": "any",
        "Distribution": "Normal",
        "Explanation": "One-Sample z-Test This test is used to determine whether the mean of a single sample is significantly different from a known or hypothesized population mean. It assumes that the data is continuous and follows a normal distribution, and that the population standard deviation is known.",
        "Example": "A researcher wants to test if the average systolic blood pressure of a group of patients is significantly different from the standard threshold of 120 mmHg. The researcher knows the population standard deviation is 10 mmHg and collects blood pressure readings from 30 patients. The researcher performs a one-sample z-test to compare the sample mean against the known population mean of 120 mmHg.",
        "Formula": r"""
                    $$ z = \dfrac{\bar{x} - \mu_0}{\dfrac{\sigma}{\sqrt{n}}} $$
                    where:
                    - :orange[$\bar{x}$] is the sample mean,
                    - :orange[$\mu_0$] is the population mean,
                    - :orange[$\sigma$] is the population standard deviation,
                    - :orange[$n$] is the sample size.
                    - The denominator :orange[$\dfrac{\sigma}{\sqrt{n}}$] is the standard error of the mean.
                    """,
    },
    {
        "name": "One-sample Proportion Test (Binomial Test)",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": "any",
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "One-Sample Proportion Test (Binomial Test) This test is used to determine whether the proportion of successes in a single sample is significantly different from a known or hypothesized population proportion. It is typically used when analyzing categorical data, such as the proportion of patients who respond to a treatment compared to a known response rate.",
        "Example": "A clinical trial tests a new drug and finds that 18 out of 30 patients respond to the treatment. The researcher wants to determine if this response rate is significantly different from the known response rate of 50% (0.5) for existing treatments. The researcher performs a one-sample proportion test (binomial test) to compare the observed proportion of 0.6 (18/30) against the known population proportion of 0.5.",
        "Formula": r"""
                    $$ z = \dfrac{\hat{p} - p_0}{\sqrt{\dfrac{p_0(1 - p_0)}{n}}} $$
                    where: 
                    - :orange[$\hat{p}$] is the sample proportion,  
                    - :orange[$p_0$] is the population proportion, and   
                    - :orange[$n$] is the sample size.  
                    - The denominator :orange[$\sqrt{\dfrac{p_0(1 - p_0)}{n}}$] is the standard error of the proportion.
                    """,
    },
    {
        "name": "One-sample Wilcoxon Signed-Rank Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": "any",
        "Distribution": "Non-normal",
        "Explanation": "One-Sample Wilcoxon Signed-Rank Test This non-parametric test is used to determine whether the median of a single sample is significantly different from a known or hypothesized population median. It is typically used when the data is ordinal or continuous but does not follow a normal distribution.",
        "Example": "A researcher wants to test if the median pain score of patients after a treatment is significantly different from a known median pain score of 5 on a 10-point scale. The researcher collects pain scores from 30 patients and performs a one-sample Wilcoxon signed-rank test to compare the sample median against the known population median of 5.",
        "Formula": r"""
                    $$ W = \sum_{i=1}^{n} R_i \cdot sgn(X_i - M_0) $$
                    Where:  
                    - :orange[$R_i$] is the rank of the absolute difference between the observed value  
                    - :orange[$X_i$] and the hypothesized median :orange[$M_0$], and  
                    - :orange[$sgn(X_i - M_0)$] is the sign function that indicates whether :orange[$X_i$] is above, below, or equal to :orange[$M_0$].  
                    - The test statistic :orange[$W$] is then compared to a critical value from the Wilcoxon signed-rank distribution to determine significance.
                    """,
    },
    {
        "name": "Student's t-test (Independent)",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": "Normal",
        "Explanation": "Student's t-test (Independent) This test is used to compare the means of two independent groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, and that the variances of the two groups are equal.",
        "Example": "A researcher wants to compare the average blood pressure between two groups of patients: those who received a new drug and those who received a placebo. The researcher collects blood pressure readings from 30 patients in each group and performs an independent t-test to determine if there is a significant difference in mean blood pressure between the two groups.",
        "Formula": r"""
                    $$ t = \dfrac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\dfrac{1}{n_1} + \dfrac{1}{n_2}}} $$ 
                    Where:  
                    - :orange[$\bar{x}_1$] and :orange[$\bar{x}_2$] are the sample means of the two groups,  
                    - :orange[$n_1$] and :orange[$n_2$] are the sample sizes of the two groups, and  
                    - :orange[$s_p$] is the pooled standard deviation calculated as: $$ s_p = \sqrt{\dfrac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}} $$ 
                    where  
                    - :orange[$s_1^2$] and :orange[$s_2^2$] are the sample variances of the two groups.
                    """,
    },
    {
        "name": "Welch's t-test (Independent, Unequal Variances)",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": "Normal",
        "Explanation": "Welch's t-test (Independent, Unequal Variances) This test is used to compare the means of two independent groups when the variances are assumed to be unequal. It is a modification of the Student's t-test.",
        "Example": "A researcher wants to compare the average blood pressure between two groups of patients: those who received a new drug and those who received a placebo. The researcher collects blood pressure readings from 30 patients in each group and performs Welch's t-test to determine if there is a significant difference in mean blood pressure between the two groups.",
        "Formula": r"""
                    $$ t = \dfrac{\bar{x}_1 - \bar{x}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}} $$ 
                    Where: 
                    - :orange[$\bar{x}_1$] and :orange[$\bar{x}_2$] are the sample means of the two groups, 
                    - :orange[$n_1$] and :orange[$n_2$] are the sample sizes of the two groups, and 
                    - :orange[$s_1^2$] and :orange[$s_2^2$] are the sample variances of the two groups.
                    """,
    },
    {
        "name": "Paired t-test",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": "Normal",
        "Explanation": "Paired t-test This test is used to compare the means of two related groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, and that the pairs are dependent (e.g., measurements taken from the same subjects before and after a treatment).",
        "Example": "A researcher wants to test if a new drug reduces blood pressure in patients. The researcher measures the blood pressure of 30 patients before and after administering the drug. The researcher performs a paired t-test to determine if there is a significant difference in mean blood pressure before and after the treatment.",
        "Formula": r"""
                    $$ t = \dfrac{\bar{d}}{s_d / \sqrt{n}} $$ 
                    Where: 
                    - :orange[$\bar{d}$] is the mean of the differences between paired observations, 
                    - :orange[$s_d$] is the standard deviation of the differences, and 
                    - :orange[$n$] is the number of pairs. 
                    - The test statistic :orange[$t$] is then compared to a critical value from the t-distribution with :orange[$n-1$] degrees of freedom to determine significance.
                    """,
    },
    {
        "name": "One-way ANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Normal",
        "Explanation": "One-way ANOVA This test is used to compare the means of three or more independent groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, that the variances are equal across groups (homogeneity of variance), and that the groups are independent.",
        "Example": "A researcher wants to compare the average blood pressure between three groups of patients: those who received a new drug, those who received a different drug, and those who received a placebo. The researcher collects blood pressure readings from 30 patients in each group and performs one-way ANOVA to determine if there is a significant difference in mean blood pressure between the three groups.",
        "Formula": r"""
                    $$ F = \dfrac{MS_{between}}{MS_{within}} $$ 
                    Where: 
                    - :orange[$F$] is the F-statistic, 
                    - :orange[$MS_{between}$] is the mean square between groups, and 
                    - :orange[$MS_{within}$] is the mean square within groups. 
                    Mean squares are calculated as: 
                    - $$ MS_{between} = \dfrac{SS_{between}}{df_{between}} $$ and 
                    - $$ MS_{within} = \dfrac{SS_{within}}{df_{within}} $$ 
                    where:
                    - :orange[$SS$] is the sum of squares and :orange[$df$] is the degrees of freedom for between and within groups.
                    """,
    },
    {
        "name": "Wilcoxon Signed-Rank Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": "Non-normal",
        "Explanation": "Wilcoxon Signed-Rank Test This test is used to compare the medians of two related groups to determine if there is a statistically significant difference between them. It assumes that the data is ordinal or continuous but not normally distributed, and that the pairs are dependent.",
        "Example": "A researcher wants to test if a new drug reduces pain levels in patients. The researcher measures the pain levels of 30 patients before and after administering the drug. The researcher performs a Wilcoxon signed-rank test to determine if there is a significant difference in median pain levels before and after the treatment.",
        "Formula": r"""
                    $$ W = \sum_{i=1}^{n} R_i $$ 
                    Where: 
                    - :orange[$W$] is the test statistic, 
                    - :orange[$R_i$] is the rank of the :orange[$i$]-th difference, and 
                    - :orange[$n$] is the number of pairs. 
                    - The test statistic :orange[$W$] is then compared to a critical value from the Wilcoxon signed-rank distribution to determine significance. :orange[$R$] is calculated by ranking the absolute differences between paired observations and assigning ranks accordingly, with ties receiving average ranks. The sign of the difference is also considered when calculating the test statistic.
                    """,
    },
    {
        "name": "Mann-Whitney U Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": "Non-normal",
        "Explanation": "Mann-Whitney U Test This test is used to compare the medians of two independent groups to determine if there is a statistically significant difference between them. It assumes that the data is ordinal or continuous but not normally distributed, and that the groups are independent.",
        "Example": "A researcher wants to compare the pain levels between two groups of patients: those who received a new drug and those who received a placebo. The researcher measures the pain levels of 30 patients in each group and performs a Mann-Whitney U test to determine if there is a significant difference in median pain levels between the two groups.",
        "Formula": r"""
                    $$ U = \sum_{i=1}^{n_1} R_i - \dfrac{n_1(n_1+1)}{2} $$ 
                    Where: 
                    - :orange[$U$] is the test statistic, 
                    - :orange[$R_i$] is the rank of the :orange[$i$]-th observation in the combined dataset, 
                    - :orange[$n_1$] is the number of observations in group 1, and 
                    - :orange[$n_2$] is the number of observations in group 2. 
                    - The test statistic :orange[$U$] is then compared to a critical value from the Mann-Whitney U distribution to determine significance. :orange[$R$] is calculated by ranking all observations from both groups together and assigning ranks accordingly, with ties receiving average ranks. The U statistic is calculated based on the sum of ranks for one of the groups and adjusted for the number of observations in that group.
                    """,
    },
    {
        "name": "Kruskal-Wallis Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Non-normal",
        "Explanation": "Kruskal-Wallis Test This test is used to compare the medians of more than two independent groups to determine if there is a statistically significant difference between them. It assumes that the data is ordinal or continuous but not normally distributed, and that the groups are independent.",
        "Example": "A researcher wants to compare the pain levels between three groups of patients: those who received a new drug, those who received a different drug, and those who received a placebo. The researcher measures the pain levels of 30 patients in each group and performs a Kruskal-Wallis test to determine if there is a significant difference in median pain levels between the three groups.",
        "Formula": r"""
                    $$ H = \dfrac{12}{N(N+1)} \sum_{i=1}^{k} \dfrac{R_i^2}{n_i} - 3(N+1) $$ 
                    Where: 
                    - :orange[$H$] is the test statistic, 
                    - :orange[$R_i$] is the sum of ranks for group :orange[$i$], 
                    - :orange[$n_i$] is the number of observations in group :orange[$i$], 
                    - :orange[$N$] is the total number of observations, and 
                    - :orange[$k$] is the number of groups. 
                    - The test statistic :orange[$H$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
    },
    {
        "name": "Repeated Measures ANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Dependent",
        "Distribution": "Normal",
        "Explanation": "Repeated Measures ANOVA This test is used to compare the means of three or more related groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, that the variances of the differences between all pairs of repeated measures are equal (sphericity), and that the groups are dependent (e.g., measurements taken from the same subjects at multiple time points).",
        "Example": "A researcher wants to test the effect of a new drug on blood pressure over time. The researcher measures the blood pressure of 30 patients at three different time points: before treatment, after 1 month of treatment, and after 3 months of treatment. The researcher performs a repeated measures ANOVA to determine if there is a significant difference in mean blood pressure across the three time points.",
        "Formula": r"""
                    $$ F = \dfrac{MS_{between}}{MS_{error}} $$ 
                    Where: 
                    - :orange[$F$] is the F-statistic, 
                    - :orange[$MS_{between}$] is the mean square between groups (calculated based on the variability of the group means), and 
                    - :orange[$MS_{error}$] is the mean square error (calculated based on the variability of observations within groups). 
                    - The test statistic :orange[$F$] is then compared to a critical value from the F-distribution with appropriate degrees of freedom to determine significance.
                    """,
    },
    {
        "name": "MANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Multiple Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Normal",
        "Explanation": "MANOVA (Multivariate Analysis of Variance) This test is used to compare the means of multiple dependent variables across two or more independent groups. It assumes that the data is continuous, follows a multivariate normal distribution, and that the groups are independent.",
        "Example": "A researcher wants to compare the effects of three different diets on both weight loss and cholesterol levels. The researcher collects data on weight loss and cholesterol levels from 30 patients in each diet group and performs a MANOVA to determine if there are significant differences in the combined dependent variables (weight loss and cholesterol levels) across the three diet groups.",
        "Formula": r"""
                    $$ \Lambda = \dfrac{|\mathbf{E}|}{|\mathbf{E} + \mathbf{H}|} $$ 
                    Where: 
                    - :orange[$\Lambda$] is the test statistic (Wilks' Lambda), 
                    - :orange[$\mathbf{E}$] is the error sum of squares and cross-products matrix, and 
                    - :orange[$\mathbf{H}$] is the hypothesis sum of squares and cross-products matrix. 
                    - The test statistic :orange[$\Lambda$] is then transformed into an F-statistic for significance testing.
                    """,
    },
    {
        "name": "Friedman Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Dependent",
        "Distribution": "Non-normal",
        "Explanation": "Friedman Test This test is used to compare the medians of three or more related groups to determine if there is a statistically significant difference between them. It assumes that the data is ordinal or continuous but not normally distributed, and that the groups are dependent.",
        "Example": "A researcher wants to compare the effectiveness of three different teaching methods on student performance. The researcher measures the performance of 30 students using each teaching method and performs a Friedman Test to determine if there is a significant difference in median performance across the three methods.",
        "Formula": r"""
                    $$ \chi^2 = \dfrac{12}{N(N+1)} \sum_{j=1}^{k} R_j^2 - 3N(N+1) $$ 
                    Where: 
                    - :orange[$\chi^2$] is the test statistic, 
                    - :orange[$N$] is the number of subjects, 
                    - :orange[$k$] is the number of treatments, and 
                    - :orange[$R_j$] is the sum of ranks for the :orange[$j$]-th treatment. 
                    - The test statistic :orange[$\chi^2$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
    },
    {
        "name": "Permutation MANOVA or Non-Parametric MANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Multiple Continuous",
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Non-normal",
        "Explanation": "Permutation MANOVA or Non-Parametric MANOVA These tests are used to compare multivariate distributions across two or more independent groups when the assumptions of traditional MANOVA are not met. They do not assume a specific distribution for the data.",
        "Example": "A researcher wants to compare the effects of three different diets on both weight loss and cholesterol levels, but the data does not follow a normal distribution. The researcher performs a Permutation MANOVA to determine if there are significant differences in the combined dependent variables (weight loss and cholesterol levels) across the three diet groups.",
        "Formula": r"""
                    $$ F = \dfrac{MS_{between}}{MS_{error}} $$
                    Where:
                    - :orange[$F$] is the pseudo-F-statistic,
                    - :orange[$MS_{between}$] is the mean square between groups, and
                    - :orange[$MS_{error}$] is the mean square error.
                    - The test statistic :orange[$F$] is then evaluated using a permutation-based null distribution (data is randomly reshuffled many times) to compute the p-value, rather than comparing to a theoretical F-distribution.
                    """,
    },
    {
        "name": "Chi-Square Goodness-of-Fit Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Non-normal", "Normal", "any"],
        "Explanation": "Chi-Square Goodness-of-Fit Test This test is used to determine if a sample data fits a particular distribution. It compares the observed frequencies with the expected frequencies under the null hypothesis.",
        "Example": "A researcher wants to test if the distribution of blood types in a sample of 100 people matches the expected distribution in the general population. The researcher performs a Chi-Square Goodness-of-Fit Test to determine if there is a significant difference between the observed and expected distributions.",
        "Formula": r"""
                    $$ \chi^2 = \sum_{i=1}^{k} \dfrac{(O_i - E_i)^2}{E_i} $$
                    Where:
                    - :orange[$\chi^2$] is the test statistic,
                    - :orange[$O_i$] is the observed frequency for category :orange[$i$],
                    - :orange[$E_i$] is the expected frequency for category :orange[$i$], and
                    - :orange[$k$] is the number of categories.
                    - The test statistic :orange[$\chi^2$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
    },
    {
        "name": "Chi-Square Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": ["any", "2", "More than 2"],
        "Relation": "Independent",
        "Distribution": ["Non-normal", "Normal", "any"],
        "Explanation": "Chi-Square Test This test is used to determine if there is a significant association between two categorical variables. It compares the observed frequencies with the expected frequencies under the null hypothesis.",
        "Example": "A researcher wants to test if there is a significant association between gender and smoking status. The researcher collects data on gender and smoking status from 200 participants and performs a Chi-Square Test to determine if there is a significant relationship between these two variables.",
        "Formula": r"""
                    $$ \chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \dfrac{(O_{ij} - E_{ij})^2}{E_{ij}} $$
                    Where:
                    - :orange[$\chi^2$] is the test statistic,
                    - :orange[$O_{ij}$] is the observed frequency for cell :orange[$(i,j)$],
                    - :orange[$E_{ij}$] is the expected frequency for cell :orange[$(i,j)$],
                    - :orange[$r$] is the number of rows, and
                    - :orange[$c$] is the number of columns.
                    - The test statistic :orange[$\chi^2$] is then compared to a critical value from the chi-square distribution with :orange[$(r-1)(c-1)$] degrees of freedom to determine significance.
                    """,
    },
    {
        "name": "McNemar's Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": ["Non-normal", "Normal", "any"],
        "Explanation": "McNemar's Test This test is used to determine if there is a significant change in proportions for paired nominal data. It is typically used when analyzing before-and-after data or matched pairs.",
        "Example": "A researcher wants to test if there is a significant change in smoking status before and after a intervention. The researcher collects data on smoking status from 100 participants before and after the intervention and performs a McNemar's Test to determine if there is a significant change.",
        "Formula": r"""
                    $$ \chi^2 = \dfrac{(b - c)^2}{b + c} $$
                    Where:
                    - :orange[$\chi^2$] is the test statistic,
                    - :orange[$b$] is the number of pairs where the first condition is positive and the second is negative, and
                    - :orange[$c$] is the number of pairs where the first condition is negative and the second is positive.
                    - The test statistic :orange[$\chi^2$] is then compared to a critical value from the chi-square distribution with 1 degree of freedom to determine significance.
                    """,
    },
    {
        "name": "Cochran's Q Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "More than 2",
        "Relation": "Dependent",
        "Distribution": ["Non-normal", "Normal", "any"],
        "Explanation": "Cochran's Q Test This test is used to determine if there is a significant difference in proportions for three or more related groups. It is an extension of McNemar's Test.",
        "Example": "A researcher wants to test if there is a significant difference in the proportion of participants who smoke at three different time points (before, during, and after an intervention). The researcher performs a Cochran's Q Test to determine if there is a significant difference.",
        "Formula": r"""
                    $$ Q = \dfrac{(k-1) \left( k \sum_{j=1}^{k} G_j^2 - \left( \sum_{j=1}^{k} G_j \right)^2 \right)}{k \sum_{i=1}^{b} L_i - \sum_{i=1}^{b} L_i^2} $$
                    Where:
                    - :orange[$Q$] is the test statistic,
                    - :orange[$k$] is the number of conditions/treatments,
                    - :orange[$b$] is the number of subjects,
                    - :orange[$G_j$] is the column total (successes) for condition :orange[$j$], and
                    - :orange[$L_i$] is the row total (successes) for subject :orange[$i$].
                    - The test statistic :orange[$Q$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
    },
    {
        "name": "Fisher's Exact Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": ["Non-normal", "Normal", "any"],
        "Explanation": "Fisher's Exact Test This test is used to determine if there is a significant association between two categorical variables when sample sizes are small.",
        "Example": "A researcher wants to test if there is a significant association between gender and smoking status in a small sample of 20 participants. The researcher performs a Fisher's Exact Test to determine if there is a significant relationship between these two variables.",
        "Formula": r"""
                    $$ p = \dfrac{(a+b)!\,(c+d)!\,(a+c)!\,(b+d)!}{a!\,b!\,c!\,d!\,n!} $$
                    
                    Where:
                    - :orange[$a, b, c, d$] are the cell frequencies in the 2×2 contingency table,
                    - :orange[$n = a+b+c+d$] is the total sample size.
                    - The p-value is calculated by summing the probabilities of all tables with the same marginal totals that are as extreme or more extreme than the observed table.
                    """,
    },
    {
        "name": "Pearson Correlation",
        "Objective": "Association/Correlation",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": "Normal",
        "Explanation": "Pearson Correlation This test is used to measure the strength and direction of the linear relationship between two continuous variables. It assumes that the data is continuous, follows a normal distribution, and that the relationship between the variables is linear.",
        "Example": "A researcher wants to test if there is a significant correlation between hours of study and exam scores among students. The researcher collects data on hours of study and exam scores from 100 students and performs a Pearson Correlation to determine if there is a significant relationship between these two variables.",
        "Formula": r"""
                    $$ r = \dfrac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^{n} (Y_i - \bar{Y})^2}} $$ 
                    Where: 
                    - :orange[$r$] is the Pearson correlation coefficient, 
                    - :orange[$X_i$] and :orange[$Y_i$] are the individual data points for variables X and Y, 
                    - :orange[$\bar{X}$] and :orange[$\bar{Y}$] are the means of variables X and Y, and 
                    - :orange[$n$] is the number of data points. 
                    - The test statistic :orange[$r$] is then compared to a critical value from the Pearson correlation distribution to determine significance.
                    """,
    },
    {
        "name": "Spearman Rank Correlation",
        "Objective": "Association/Correlation",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Ordinal", "Continuous"],
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": "Non-normal",
        "Explanation": "Spearman Rank Correlation This test is used to measure the strength and direction of the monotonic relationship between two ordinal or continuous variables. It does not assume a linear relationship or a normal distribution.",
        "Example": "A researcher wants to test if there is a significant correlation between rankings of students in two different subjects. The researcher collects data on the rankings and performs a Spearman Rank Correlation to determine if there is a significant relationship between these two variables.",
        "Formula": r"""
                    $$ r_s = 1 - \dfrac{6 \sum d_i^2}{n(n^2 - 1)} $$ 
                    Where: 
                    - :orange[$r_s$] is the Spearman rank correlation coefficient, 
                    - :orange[$d_i$] is the difference in ranks for each pair of observations, 
                    - :orange[$n$] is the number of observations, and 
                    - the sum is over all pairs. 
                    - The test statistic :orange[$r_s$] is then compared to a critical value from the Spearman rank correlation distribution to determine significance.
                    """,
    },
    {
        "name": "Chi-Square Test of Independence",
        "Objective": "Association/Correlation",
        "Dependent_Variable": "Categorical",
        "Independent_Variable": "Categorical",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Non-normal", "Normal", "any"],
        "Explanation": "Chi-Square Test of Independence This test is used to determine if there is a significant association between two categorical variables. It compares the observed frequencies with the expected frequencies under the null hypothesis.",
        "Example": "A researcher wants to test if there is a significant association between gender and smoking status. The researcher collects data on gender and smoking status from 200 participants and performs a Chi-Square Test of Independence to determine if there is a significant relationship between these two variables.",
        "Formula": r"""
                    $$ \chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \dfrac{(O_{ij} - E_{ij})^2}{E_{ij}} $$ 
                    Where: 
                    - :orange[$\chi^2$] is the test statistic, 
                    - :orange[$O_{ij}$] is the observed frequency for cell :orange[$(i,j)$], and 
                    - :orange[$E_{ij}$] is the expected frequency for cell :orange[$(i,j)$]. 
                    - The test statistic :orange[$\chi^2$] is then compared to a critical value from the chi-square distribution with :orange[$(r-1)(c-1)$] degrees of freedom to determine significance.
                    """,
    },
    {
        "name": "Point-Biserial Correlation",
        "Objective": "Association/Correlation",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Binary/Dichotomous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Non-normal", "Normal", "any"],
        "Explanation": "Point-Biserial Correlation This test is used to measure the strength and direction of the linear relationship between a continuous variable and a binary variable. It is a special case of the Pearson correlation coefficient.",
        "Example": "A researcher wants to test if there is a significant correlation between test scores (continuous) and gender (binary). The researcher collects data on test scores and gender from 100 participants and performs a Point-Biserial Correlation to determine if there is a significant relationship between these two variables.",
        "Formula": r"""
                    $$ r_{pb} = \dfrac{M_1 - M_0}{s} \sqrt{\dfrac{n_1 n_0}{n(n-1)}} $$ 
                    Where: 
                    - :orange[$r_{pb}$] is the Point-Biserial correlation coefficient, 
                    - :orange[$M_1$] and :orange[$M_0$] are the means of the continuous variable for the two groups, 
                    - :orange[$s$] is the standard deviation of the continuous variable, 
                    - :orange[$n_1$] and :orange[$n_0$] are the sample sizes of the two groups, and 
                    - :orange[$n$] is the total sample size. 
                    - The test statistic :orange[$r_{pb}$] is then compared to a critical value from the Pearson correlation distribution to determine significance.
                    """,
    },
    # Prediction Tests
    {
        "name": "Simple Linear Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "Simple Linear Regression This test is used to model the relationship between a continuous dependent variable and a single continuous independent variable. It assumes that the dependent variable is continuous, that the residuals are normally distributed, and that the relationship between the variables is linear.",
        "Example": "A researcher wants to predict exam scores based on hours of study. The researcher collects data on hours of study and exam scores from 100 students and performs a Simple Linear Regression to determine if hours of study is a significant predictor of exam scores.",
        "Formula": r"""
                    $$ Y = \beta_0 + \beta_1 X + \epsilon $$ 
                    Where: 
                    - :orange[$Y$] is the dependent variable, 
                    - :orange[$X$] is the independent variable, 
                    - :orange[$\beta_0$] is the intercept, 
                    - :orange[$\beta_1$] is the slope coefficient, and 
                    - :orange[$\epsilon$] is the error term. 
                    - The coefficients :orange[$\beta_0$] and :orange[$\beta_1$] are estimated using the least squares method, and the significance of the predictor is determined by testing if :orange[$\beta_1$] is significantly different from zero.
                    """,
    },
    {
        "name": "Multiple Linear Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Multiple Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "Multiple Linear Regression This test is used to model the relationship between a continuous dependent variable and multiple continuous independent variables. It assumes that the dependent variable is continuous, that the residuals are normally distributed, and that the relationship between the variables is linear.",
        "Example": "A researcher wants to predict exam scores based on hours of study and attendance. The researcher collects data on hours of study, attendance, and exam scores from 100 students and performs a Multiple Linear Regression to determine if hours of study and attendance are significant predictors of exam scores.",
        "Formula": r"""
                    $$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_k X_k + \epsilon $$ 
                    Where: 
                    - :orange[$Y$] is the dependent variable, 
                    - :orange[$X_1, X_2, \ldots, X_k$] are the independent variables, 
                    - :orange[$\beta_0$] is the intercept, 
                    - :orange[$\beta_1, \beta_2, \ldots, \beta_k$] are the slope coefficients, and 
                    - :orange[$\epsilon$] is the error term. 
                    - The coefficients :orange[$\beta_0, \beta_1, \beta_2, \ldots, \beta_k$] are estimated using the least squares method, and the significance of each predictor is determined by testing if its corresponding coefficient is significantly different from zero.
                    """,
    },
    {
        "name": "Logistic Regression",
        "Objective": "Prediction",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "Logistic Regression This test is used to model the relationship between a binary dependent variable and one or more continuous independent variables. It assumes that the dependent variable is binary, and that the relationship between the independent variables and the log-odds of the dependent variable is linear.",
        "Example": "A researcher wants to predict the likelihood of a student passing an exam based on hours of study. The researcher collects data on hours of study and pass/fail status from 100 students and performs a Logistic Regression to determine if hours of study is a significant predictor of passing the exam.",
        "Formula": r"""
                    $$ \log\left(\dfrac{p}{1-p}\right) = \beta_0 + \beta_1 X $$ 
                    Where: 
                    - :orange[$p$] is the probability of the dependent variable being 1 (e.g., passing the exam), 
                    - :orange[$X$] is the independent variable, 
                    - :orange[$\beta_0$] is the intercept, and 
                    - :orange[$\beta_1$] is the slope coefficient. 
                    - The coefficients :orange[$\beta_0$] and :orange[$\beta_1$] are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if :orange[$\beta_1$] is significantly different from zero.
                    """,
    },
    {
        "name": "Multinomial Logistic Regression",
        "Objective": "Prediction",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "Multinomial Logistic Regression This test is used to model the relationship between a categorical dependent variable with more than two categories and one or more continuous independent variables. It assumes that the dependent variable is categorical, and that the relationship between the independent variables and the log-odds of each category of the dependent variable is linear.",
        "Example": "A researcher wants to predict the choice of transportation (car, bus, bike) based on hours of commute. The researcher collects data on hours of commute and transportation choice from 100 participants and performs a Multinomial Logistic Regression to determine if hours of commute is a significant predictor of transportation choice.",
        "Formula": r"""
                    $$ \log\left(\dfrac{p_j}{p_k}\right) = \beta_{0j} + \beta_{1j} X $$ 
                    Where: 
                    - :orange[$p_j$] is the probability of the dependent variable being in category :orange[$j$], 
                    - :orange[$p_k$] is the probability of the dependent variable being in the reference category :orange[$k$], 
                    - :orange[$X$] is the independent variable, 
                    - :orange[$\beta_{0j}$] is the intercept for category :orange[$j$], and 
                    - :orange[$\beta_{1j}$] is the slope coefficient for category :orange[$j$]. 
                    - The coefficients :orange[$\beta_{0j}$] and :orange[$\beta_{1j}$] are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if :orange[$\beta_{1j}$] is significantly different from zero for each category.
                    """,
    },
    {
        "name": "Ordinal Logistic Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Ordinal",
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "Ordinal Logistic Regression This test is used to model the relationship between an ordinal dependent variable and one or more continuous independent variables. It assumes that the dependent variable is ordinal, and that the relationship between the independent variables and the log-odds of each category of the dependent variable is linear.",
        "Example": "A researcher wants to predict the level of satisfaction (very unsatisfied, unsatisfied, neutral, satisfied, very satisfied) based on income. The researcher collects data on income and satisfaction levels from 100 participants and performs an Ordinal Logistic Regression to determine if income is a significant predictor of satisfaction level.",
        "Formula": r"""
                    $$ \log\left(\dfrac{P(Y \leq j)}{P(Y > j)}\right) = \beta_{0j} + \beta_{1j} X $$ 
                    Where: 
                    - :orange[$P(Y \leq j)$] is the probability of the dependent variable being in category :orange[$j$] or lower, 
                    - :orange[$P(Y > j)$] is the probability of the dependent variable being in a category higher than :orange[$j$], 
                    - :orange[$X$] is the independent variable, 
                    - :orange[$\beta_{0j}$] is the intercept for category :orange[$j$], and 
                    - :orange[$\beta_{1j}$] is the slope coefficient for category :orange[$j$]. 
                    - The coefficients :orange[$\beta_{0j}$] and :orange[$\beta_{1j}$] are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if :orange[$\beta_{1j}$] is significantly different from zero for each category.
                    """,
    },
    {
        "name": "Poisson Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Discrete",
        "Independent_Variable": "Continuous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "Poisson Regression This test is used to model the relationship between a count dependent variable and one or more continuous independent variables. It assumes that the dependent variable is a count variable, and that the relationship between the independent variables and the log of the expected count is linear.",
        "Example": "A researcher wants to predict the number of hospital visits (0, 1, 2, 3, ...) based on age and income. The researcher collects data on age, income, and hospital visits from 100 participants and performs a Poisson Regression to determine if age and income are significant predictors of hospital visits.",
        "Formula": r"""
                    $$ \log(\lambda) = \beta_{0} + \beta_{1} X_1 + \beta_{2} X_2 $$ 
                    Where: 
                    - :orange[$\lambda$] is the expected count of the dependent variable, 
                    - :orange[$X_1$] and :orange[$X_2$] are the independent variables, 
                    - :orange[$\beta_{0}$] is the intercept, and 
                    - :orange[$\beta_{1}$] and :orange[$\beta_{2}$] are the slope coefficients for each independent variable. 
                    - The coefficients :orange[$\beta_{0}$], :orange[$\beta_{1}$], and :orange[$\beta_{2}$] are estimated using maximum likelihood estimation, and the significance of the predictors is determined by testing if they are significantly different from zero.
                    """,
    },
    # Diagnostic Accuracy Tests
    {
        "name": "Sensitivity & Specificity Analysis",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Binary/Dichotomous",
        "Independent_Variable": "Binary/Dichotomous",
        "Groups": "any",
        "Relation": "any",
        "Distribution": "any",
        "Explanation": "Sensitivity and Specificity measures the performance of a binary diagnostic test against a gold standard. Sensitivity (True Positive Rate) is the ability to correctly identify those with the disease, while Specificity (True Negative Rate) is the ability to correctly identify those without the disease.",
        "Example": "A new rapid antigen test is compared against PCR (gold standard) for COVID-19. 100 people known to have the virus and 100 known to be healthy are tested to calculate the accuracy metrics.",
        "Formula": r"""
                    $$ \text{Sensitivity} = \dfrac{TP}{TP + FN} $$
                    $$ \text{Specificity} = \dfrac{TN}{TN + FP} $$
                    $$ \text{Positive Predictive Value (PPV)} = \dfrac{TP}{TP + FP} $$
                    $$ \text{Negative Predictive Value (NPV)} = \dfrac{TN}{TN + FN} $$
                    $$ \text{Accuracy} = \dfrac{TP + TN}{TP + TN + FP + FN} $$
                    $$ \text{Likelihood ratio for a positive test (LR+)} = \dfrac{\text{Sensitivity}}{1 - \text{Specificity}} $$
                    $$ \text{Likelihood ratio for a negative test (LR-)} = \dfrac{1 - \text{Sensitivity}}{\text{Specificity}} $$
                    $$ \text{F1 Score} = 2 \times \dfrac{\text{PPV} \times \text{Sensitivity}}{\text{PPV} + \text{Sensitivity}} $$
                    $$ \text{Diagnostic Odds Ratio (DOR)} = \dfrac{LR+}{LR-} $$
                    Where:
                    - :orange[$TP$]: True Positives
                    - :orange[$TN$]: True Negatives
                    - :orange[$FP$]: False Positives
                    - :orange[$FN$]: False Negatives
                    """,
    },
    {
        "name": "ROC Curve Analysis",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Binary/Dichotomous",
        "Independent_Variable": "Continuous",
        "Groups": "any",
        "Relation": "any",
        "Distribution": "any",
        "Explanation": "Receiver Operating Characteristic (ROC) analysis is used to evaluate the performance of a continuous diagnostic test. It plots Sensitivity against 1-Specificity at various thresholds. The Area Under the Curve (AUC) represents the overall accuracy.",
        "Example": "A researcher wants to determine if blood sugar levels can accurately diagnose diabetes. By plotting an ROC curve, they can find the optimal sugar level cut-off that maximizes both sensitivity and specificity.",
        "Formula": r"""
                    $$ \text{AUC} = \int_{0}^{1} \text{Sensitivity}(1-\text{Specificity}) d(1-\text{Specificity}) $$
                    - :orange[AUC = 0.5]: Random guessing
                    - :orange[AUC = 1.0]: Perfect diagnostic accuracy
                    """,
    },
    {
        "name": "Likelihood Ratio Analysis",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Binary/Dichotomous",
        "Independent_Variable": "Binary/Dichotomous",
        "Groups": "any",
        "Relation": "any",
        "Distribution": "any",
        "Explanation": "Likelihood Ratios (LR) are used to assess the value of performing a diagnostic test. LR+ indicates how much more likely a positive test is to be found in a person with the disease than in a person without. LR- indicates how much less likely a negative test is to be found in a person with the disease than in a person without.",
        "Example": "A clinician uses the LR+ of a physical exam finding to update their post-test probability of a patient having appendicitis.",
        "Formula": r"""
                    $$ LR+ = \dfrac{\text{Sensitivity}}{1 - \text{Specificity}} $$
                    $$ LR- = \dfrac{1 - \text{Sensitivity}}{\text{Specificity}} $$
                    - :orange[LR+ > 10]: Large increase in disease probability
                    - :orange[LR- < 0.1]: Large decrease in disease probability
                    """,
    },
    {
        "name": "Cohen's Kappa (Agreement Analysis)",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Categorical",
        "Independent_Variable": "Categorical",
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": "any",
        "Explanation": "Cohen's Kappa is used to measure inter-rater or intra-rater agreement for categorical variables. It accounts for the agreement occurring by chance.",
        "Example": "Two radiologists evaluate the same set of X-rays to diagnose a fracture. Cohen's Kappa measures how consistently they agree on the presence or absence of a fracture.",
        "Formula": r"""
                    $$ \kappa = \dfrac{p_o - p_e}{1 - p_e} $$
                    Where:
                    - :orange[$p_o$]: Observed proportionate agreement
                    - :orange[$p_e$]: Probability of random agreement
                    """,
    },
    # ========================================================================
    # TWO-WAY ANOVA
    # ========================================================================
    {
        "name": "Two-way ANOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Categorical",
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Normal",
        "Explanation": "Two-way ANOVA This test evaluates the effect of two categorical independent variables (factors) on a continuous dependent variable, as well as their interaction. For example, it can test the effect of treatment (drug vs. placebo) and sex (male vs. female) on blood pressure, and whether the treatment effect differs by sex. It assumes normality, homogeneity of variances, and independence of observations.",
        "Example": "A researcher wants to test the effects of a new drug and sex on blood pressure. 60 patients are divided into drug and placebo groups, each containing equal numbers of males and females. Two-way ANOVA is used to test for main effects of drug and sex, and their interaction.",
        "Formula": r"""
                    $$ SS_{total} = SS_A + SS_B + SS_{AB} + SS_{error} $$
                    $$ F_A = \frac{MS_A}{MS_{error}}, \quad F_B = \frac{MS_B}{MS_{error}}, \quad F_{AB} = \frac{MS_{AB}}{MS_{error}} $$
                    Where:
                    - :orange[$SS_A$] and :orange[$SS_B$] are sums of squares for factors A and B,
                    - :orange[$SS_{AB}$] is the sum of squares for the interaction,
                    - :orange[$MS$] values are mean squares (SS/df),
                    - :orange[$F$]-statistics test the main effects and interaction separately,
                    - Significance is determined by comparing each :orange[$F$] to the F-distribution with appropriate df.
                    """,
    },
    # ========================================================================
    # ANCOVA
    # ========================================================================
    {
        "name": "ANCOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Categorical",
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Normal",
        "Explanation": "Analysis of Covariance (ANCOVA) combines ANOVA and linear regression. It compares group means on a continuous dependent variable while statistically controlling for the effect of one or more continuous covariates. This increases statistical power by reducing within-group error variance and adjusts for baseline differences. It assumes normality, homogeneity of variances, homogeneity of regression slopes, and linearity between covariate and outcome.",
        "Example": "A researcher wants to compare post-treatment blood pressure between three drug groups while controlling for baseline blood pressure. ANCOVA adjusts the post-treatment means for baseline differences, providing a more precise estimate of treatment effects.",
        "Formula": r"""
                    $$ F = \frac{MS_{between}}{MS_{error}} $$
                    Where:
                    - The dependent variable :orange[$Y$] is modeled as: $$ Y_{ij} = \mu + \tau_j + \beta(X_{ij} - \bar{X}) + \epsilon_{ij} $$
                    - :orange[$\tau_j$] is the effect of the :orange[$j$]-th group,
                    - :orange[$\beta$] is the regression coefficient for the covariate :orange[$X$],
                    - :orange[$MS_{error}$] is reduced by the variance explained by the covariate, increasing power.
                    """,
    },
    # ========================================================================
    # COX PROPORTIONAL HAZARDS REGRESSION
    # ========================================================================
    {
        "name": "Cox Proportional Hazards Regression",
        "Objective": "Survival Analysis",
        "Dependent_Variable": "Time-to-event",
        "Independent_Variable": "Continuous",
        "Groups": "any",
        "Relation": "any",
        "Distribution": "any",
        "Explanation": "Cox Proportional Hazards Regression is the most widely used model for survival analysis. It estimates the hazard ratio (HR) for time-to-event outcomes while adjusting for multiple covariates simultaneously. It is a semi-parametric model that makes no assumption about the shape of the baseline hazard function, but assumes proportional hazards (the effect of each predictor is constant over time).",
        "Example": "A researcher wants to identify predictors of survival time after a cancer diagnosis. They collect data on age, tumor stage, and treatment type from 200 patients and perform Cox regression to estimate the hazard ratios for each predictor, indicating which factors significantly increase or decrease mortality risk.",
        "Formula": r"""
                    $$ h(t) = h_0(t) \exp(\beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_k X_k) $$
                    Where:
                    - :orange[$h(t)$] is the hazard at time :orange[$t$],
                    - :orange[$h_0(t)$] is the baseline hazard (when all predictors are zero),
                    - :orange[$\beta_1, \beta_2, \ldots$] are the log-hazard ratios for predictors :orange[$X_1, X_2, \ldots$],
                    - :orange[$\exp(\beta_i)$] is the hazard ratio for predictor :orange[$X_i$],
                    - The proportional hazards assumption means :orange[$h_0(t)$] cancels out when comparing two individuals.
                    """,
    },
    # ========================================================================
    # LOG-RANK TEST
    # ========================================================================
    {
        "name": "Log-Rank Test",
        "Objective": "Survival Analysis",
        "Dependent_Variable": "Time-to-event",
        "Independent_Variable": "Categorical",
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": "any",
        "Explanation": "The Log-Rank Test is a non-parametric test that compares the survival distributions of two or more independent groups. It tests whether the time-to-event differs significantly between groups. It makes no assumption about the shape of the survival curves but assumes that the hazard rates are proportional over time. It is commonly used alongside Kaplan-Meier survival curves.",
        "Example": "A researcher compares survival times between 50 patients receiving a new cancer drug and 50 receiving standard therapy. The Log-Rank Test determines if the survival difference between the two groups is statistically significant.",
        "Formula": r"""
                    $$ \chi^2 = \frac{(O_1 - E_1)^2}{E_1} + \frac{(O_2 - E_2)^2}{E_2} $$
                    Where:
                    - :orange[$O_1$] and :orange[$O_2$] are the observed number of events in each group,
                    - :orange[$E_1$] and :orange[$E_2$] are the expected number of events under the null hypothesis of no difference,
                    - The test statistic is compared to a chi-square distribution with 1 degree of freedom (for two groups).
                    - For more than two groups, an extension with :orange[$k-1$] degrees of freedom is used.
                    """,
    },
    # ========================================================================
    # BLAND-ALTMAN ANALYSIS
    # ========================================================================
    {
        "name": "Bland-Altman Analysis",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Continuous",
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": "any",
        "Explanation": "Bland-Altman Analysis is the standard method for assessing agreement between two quantitative measurement techniques. It plots the difference between paired measurements against their mean, calculates the mean difference (bias), and defines limits of agreement (mean difference ± 1.96 SD of differences). Unlike correlation, which measures association, Bland-Altman directly assesses interchangeability. It assumes the differences are approximately normally distributed.",
        "Example": "A researcher develops a new digital caliper for measuring tooth dimensions and wants to know if it agrees with the traditional mechanical caliper. 50 teeth are measured with both instruments. Bland-Altman Analysis shows a mean difference of 0.02 mm (negligible bias) with limits of agreement from −0.15 to +0.19 mm, confirming the new caliper can replace the old one for clinical purposes.",
        "Formula": r"""
                    $$ \bar{d} = \frac{1}{n} \sum_{i=1}^{n} (X_i - Y_i) $$
                    $$ s_d = \sqrt{\frac{\sum_{i=1}^{n} (d_i - \bar{d})^2}{n-1}} $$
                    $$ \text{Upper LoA} = \bar{d} + 1.96 \times s_d $$
                    $$ \text{Lower LoA} = \bar{d} - 1.96 \times s_d $$
                    Where:
                    - :orange[$X_i$] and :orange[$Y_i$] are paired measurements from two methods,
                    - :orange[$\bar{d}$] is the mean difference (bias),
                    - :orange[$s_d$] is the standard deviation of the differences,
                    - :orange[LoA] are the limits of agreement (95% tolerance limits).
                    """,
    },
    # ========================================================================
    # KENDALL'S TAU-B
    # ========================================================================
    {
        "name": "Kendall's Tau-b",
        "Objective": "Association/Correlation",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Ordinal", "Continuous"],
        "Groups": "any",
        "Relation": "any",
        "Distribution": "Non-normal",
        "Explanation": "Kendall's Tau-b is a non-parametric rank correlation coefficient that measures the strength and direction of monotonic association between two variables. It is more robust than Spearman's ρ when there are many tied ranks and provides a more conservative estimate. Its interpretation is more intuitive: it represents the difference between the probability of concordance and discordance.",
        "Example": "A researcher wants to assess the association between two ordinal ratings of periodontal disease severity (none, mild, moderate, severe) given by two different examiners on 100 patients. Kendall's Tau-b is preferred over Spearman's ρ due to the high number of expected ties.",
        "Formula": r"""
                    $$ \tau_b = \frac{C - D}{\sqrt{(C + D + T_X)(C + D + T_Y)}} $$
                    Where:
                    - :orange[$C$] is the number of concordant pairs,
                    - :orange[$D$] is the number of discordant pairs,
                    - :orange[$T_X$] is the number of pairs tied only on variable :orange[$X$],
                    - :orange[$T_Y$] is the number of pairs tied only on variable :orange[$Y$],
                    - :orange[$\tau_b$] ranges from −1 (perfect disagreement) to +1 (perfect agreement).
                    """,
    },
    # ========================================================================
    # NEGATIVE BINOMIAL REGRESSION
    # ========================================================================
    {
        "name": "Negative Binomial Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Discrete",
        "Independent_Variable": "Continuous",
        "Groups": "any",
        "Relation": "any",
        "Distribution": "any",
        "Explanation": "Negative Binomial Regression is used for modeling count data when the variance exceeds the mean (overdispersion), which violates the Poisson Regression assumption of equal mean and variance. It adds an extra dispersion parameter to account for unobserved heterogeneity. It is commonly used in medical research for counts with many zeros or high variability, such as hospital readmissions, number of seizures, or dental caries counts.",
        "Example": "A researcher wants to model the number of dental caries (cavities) in children based on sugar consumption, fluoride exposure, and brushing frequency. The count data shows variance much larger than the mean, so Negative Binomial Regression is chosen over Poisson Regression to account for overdispersion.",
        "Formula": r"""
                    $$ \log(\lambda_i) = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \ldots + \beta_k X_{ki} $$
                    $$ \text{Var}(Y_i) = \lambda_i + \alpha \lambda_i^2 $$
                    Where:
                    - :orange[$\lambda_i$] is the expected count for observation :orange[$i$],
                    - :orange[$\alpha$] is the dispersion parameter (:orange[$\alpha = 0$] reduces to Poisson),
                    - :orange[$\alpha > 0$] indicates overdispersion,
                    - Coefficients are estimated using maximum likelihood.
                    """,
    },
    # ========================================================================
    # WEIGHTED KAPPA
    # ========================================================================
    {
        "name": "Weighted Kappa",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Ordinal",
        "Independent_Variable": "Ordinal",
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": "any",
        "Explanation": "Weighted Kappa extends Cohen's Kappa to ordinal categorical ratings by incorporating partial credit for disagreements that are close (e.g., 'mild' vs. 'moderate' disagreement is penalized less than 'mild' vs. 'severe'). Linear weights penalize disagreements proportionally to their distance; quadratic weights penalize more severely. It is the standard agreement measure for ordinal scales in medical research.",
        "Example": "Two radiologists independently classify 100 mammograms into four categories: normal, benign, suspicious, and malignant. Weighted Kappa is used to measure their agreement, where a disagreement between 'normal' and 'benign' is penalized less than between 'normal' and 'malignant'.",
        "Formula": r"""
                    $$ \kappa_w = 1 - \frac{\sum_{i=1}^{k} \sum_{j=1}^{k} w_{ij} O_{ij}}{\sum_{i=1}^{k} \sum_{j=1}^{k} w_{ij} E_{ij}} $$
                    Where:
                    - :orange[$O_{ij}$] and :orange[$E_{ij}$] are observed and expected frequencies for cell :orange[$(i,j)$],
                    - :orange[$w_{ij}$] is the weight (0 for perfect agreement, 1 for maximum disagreement),
                    - Linear weights: :orange[$w_{ij} = \frac{|i-j|}{k-1}$],
                    - Quadratic weights: :orange[$w_{ij} = \frac{(i-j)^2}{(k-1)^2}$].
                    """,
    },
    # ========================================================================
    # FLEISS' KAPPA
    # ========================================================================
    {
        "name": "Fleiss' Kappa",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Categorical",
        "Independent_Variable": "Categorical",
        "Groups": "More than 2",
        "Relation": "Dependent",
        "Distribution": "any",
        "Explanation": "Fleiss' Kappa is an extension of Cohen's Kappa that measures inter-rater agreement for three or more raters evaluating categorical (nominal) ratings. Unlike Cohen's Kappa, which handles only two raters, Fleiss' Kappa simultaneously assesses agreement among multiple raters while correcting for chance agreement. It ranges from −1 to +1, with +1 indicating perfect agreement.",
        "Example": "Three oral pathologists independently classify 50 biopsy slides into diagnostic categories (benign, dysplastic, malignant). Fleiss' Kappa measures the overall agreement among all three pathologists simultaneously, correcting for chance.",
        "Formula": r"""
                    $$ \kappa = \frac{\bar{P} - \bar{P_e}}{1 - \bar{P_e}} $$
                    Where:
                    - :orange[$\bar{P}$] is the mean proportion of observed agreement across all raters,
                    - :orange[$\bar{P_e}$] is the mean proportion of expected agreement by chance,
                    - :orange[$\kappa = 1$]: perfect agreement,
                    - :orange[$\kappa = 0$]: agreement equivalent to chance,
                    - :orange[$\kappa < 0$]: less than chance agreement.
                    """,
    },
]

# =========================
# MATCHING ENGINE
# =========================
# MATCHING ENGINE
# =========================

CRITERIA_FIELDS = [
    "Objective",
    "Dependent_Variable",
    "Independent_Variable",
    "Groups",
    "Relation",
    "Distribution",
]


def matches_rule(user_input, rule):

    for key in CRITERIA_FIELDS:

        rule_val = rule.get(key)
        if rule_val is None:
            continue

        user_val = user_input.get(key)

        # Rule accepts anything
        if rule_val == "any":
            continue

        # Handle LISTS in rules
        if isinstance(rule_val, list):
            if user_val not in rule_val:
                return False

        # Handle normal strings
        else:
            if user_val != rule_val:
                return False

    return True


def find_matching_tests(user_input):

    matches = []

    for rule in rules:
        if matches_rule(user_input, rule):
            matches.append(rule["name"])

    return matches


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
# INTERACTIVE WIDGETS
# =========================


def render_latex(formula_text):
    """Render LaTeX formulas from text with $$ delimiters."""
    import re

    last_end = 0
    for match in re.finditer(r"\$\$(.*?)\$\$", formula_text, re.DOTALL):
        # Text before this match
        text_before = formula_text[last_end : match.start()]
        if text_before.strip():
            st.markdown(text_before)

        # The LaTeX block (without $$ delimiters)
        latex_code = match.group(1).strip()
        st.latex(latex_code)

        last_end = match.end()

    # Text after the last match
    text_after = formula_text[last_end:]
    if text_after.strip():
        st.markdown(text_after)


def render_test_widget(test_name):
    """Render interactive widget for specific statistical test."""

    # One Sample Tests

    if test_name == "One-sample t-test":

        from scipy.stats import ttest_1samp

        st.subheader("Interactive One-sample t-test")

        # =========================
        # CONTROLS
        # =========================

        population_mean = st.slider(
            "Reference Mean",
            -10.0,
            10.0,
            0.0,
            0.1,
        )

        sample_mean_shift = st.slider(
            "Sample Mean Shift",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        sd = st.slider(
            "Standard Deviation",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        sample = np.random.normal(
            population_mean + sample_mean_shift,
            sd,
            80,
        )

        t, p = ttest_1samp(sample, population_mean)

        # =========================
        # STATS
        # =========================

        st.latex(rf"t = {t:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=sample,
                nbinsx=20,
            )
        )

        fig.add_vline(
            x=population_mean,
            line_dash="dash",
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import t as t_dist
        from scipy.stats import sem

        n = len(sample)
        sample_mean = np.mean(sample)
        sample_sd = np.std(sample, ddof=1)
        se = sem(sample)
        ci = se * t_dist.ppf(0.975, n - 1)
        cohens_d = (sample_mean - population_mean) / sample_sd

        results_data = {
            "Metric": [
                "Sample Mean",
                "Reference Mean",
                "Mean Difference",
                "95% CI of Diff",
                "t-statistic",
                "df",
                "p-value",
                "Cohen's d",
            ],
            "Value": [
                f"{sample_mean:.3f}",
                f"{population_mean:.3f}",
                f"{sample_mean - population_mean:.3f}",
                f"±{ci:.3f}",
                f"{t:.3f}",
                f"{n - 1}",
                f"{p:.5f}",
                f"{cohens_d:.3f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Histogram(
                x=sample,
                nbinsx=25,
                histnorm="probability density",
                name="Sample",
                marker=dict(color="rgba(0, 123, 255, 0.5)"),
            )
        )

        x_dense = np.linspace(sample.min(), sample.max(), 200)
        from scipy.stats import norm

        y_dense = norm.pdf(x_dense, sample_mean, sample_sd)
        fig2.add_trace(
            go.Scatter(
                x=x_dense,
                y=y_dense,
                mode="lines",
                name="Normal fit",
                line=dict(color="red", width=2),
            )
        )

        fig2.add_vline(
            x=population_mean,
            line_dash="dash",
            line_color="green",
            annotation_text="Reference Mean",
        )
        fig2.add_vline(
            x=sample_mean,
            line_dash="dot",
            line_color="blue",
            annotation_text="Sample Mean",
        )
        fig2.add_vline(
            x=sample_mean - ci,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
        )
        fig2.add_vline(
            x=sample_mean + ci,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
        )

        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Value",
            yaxis_title="Density",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "One-sample z-test":

        from statsmodels.stats.weightstats import ztest

        st.subheader("Interactive One-sample z-test")

        # =========================
        # CONTROLS
        # =========================

        population_mean = st.slider(
            "Population Mean",
            -10.0,
            10.0,
            0.0,
            0.1,
        )

        shift = st.slider(
            "Sample Mean Shift",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        sample = np.random.normal(
            population_mean + shift,
            1,
            300,
        )

        z, p = ztest(sample, value=population_mean)

        # =========================
        # STATS
        # =========================

        st.latex(rf"z = {z:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Histogram(x=sample))

        fig.add_vline(
            x=population_mean,
            line_dash="dash",
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        n_z = len(sample)
        sample_mean_z = np.mean(sample)
        se_z = 1 / np.sqrt(n_z)
        ci_z = 1.96 * se_z

        results_data = {
            "Metric": [
                "Sample Mean",
                "Population Mean (μ₀)",
                "z-statistic",
                "p-value",
                "SE (σ/√n)",
                "n",
                "95% CI of Mean",
                "Known σ",
            ],
            "Value": [
                f"{sample_mean_z:.3f}",
                f"{population_mean:.3f}",
                f"{z:.3f}",
                f"{p:.5f}",
                f"{se_z:.4f}",
                f"{n_z}",
                f"{sample_mean_z - ci_z:.2f} to {sample_mean_z + ci_z:.2f}",
                "1.0",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Histogram(
                x=sample,
                nbinsx=25,
                histnorm="probability density",
                name="Sample",
                marker=dict(color="rgba(0, 123, 255, 0.5)"),
            )
        )

        x_dense = np.linspace(sample.min(), sample.max(), 200)
        from scipy.stats import norm as norm2

        y_dense = norm2.pdf(x_dense, sample_mean_z, 1 / np.sqrt(n_z) * np.sqrt(n_z))
        y_dense = norm2.pdf(x_dense, sample_mean_z, np.std(sample, ddof=1))
        fig2.add_trace(
            go.Scatter(
                x=x_dense,
                y=y_dense,
                mode="lines",
                name="Normal fit",
                line=dict(color="red", width=2),
            )
        )

        fig2.add_vline(
            x=population_mean,
            line_dash="dash",
            line_color="green",
            annotation_text="H₀ Mean",
        )
        fig2.add_vline(
            x=sample_mean_z,
            line_dash="dot",
            line_color="blue",
            annotation_text="Sample Mean",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Value",
            yaxis_title="Density",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "One-sample Proportion Test (Binomial Test)":

        from scipy.stats import binomtest

        st.subheader("Interactive One-sample Proportion Test")

        # =========================
        # CONTROLS
        # =========================

        expected_p = st.slider(
            "Expected Proportion",
            0.0,
            1.0,
            0.5,
            0.01,
        )

        observed_p = st.slider(
            "Observed Proportion",
            0.0,
            1.0,
            0.7,
            0.01,
        )

        n = st.slider(
            "Sample Size",
            10,
            500,
            100,
        )

        # =========================
        # TEST
        # =========================

        successes = int(observed_p * n)

        result = binomtest(
            successes,
            n,
            expected_p,
        )

        # =========================
        # STATS
        # =========================

        st.latex(rf"\hat{{p}} = {observed_p:.2f}")

        st.latex(rf"\text{{p-value}} = {result.pvalue:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Expected", "Observed"],
                y=[expected_p, observed_p],
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            yaxis=dict(range=[0, 1]),
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import binomtest as binomtest2
        from scipy.stats import norm as norm_prop

        p_hat = successes / n
        se_prop = np.sqrt(expected_p * (1 - expected_p) / n)
        z_prop = (p_hat - expected_p) / se_prop if se_prop > 0 else 0
        ci_prop = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n)

        results_data = {
            "Metric": [
                "Observed Proportion",
                "Expected Proportion",
                "Difference",
                "95% CI of Proportion",
                "Number of Successes",
                "Sample Size (n)",
                "z-approximation",
                "Exact p-value",
            ],
            "Value": [
                f"{p_hat:.3f}",
                f"{expected_p:.3f}",
                f"{p_hat - expected_p:.3f}",
                f"{p_hat - ci_prop:.3f} to {p_hat + ci_prop:.3f}",
                f"{successes}",
                f"{n}",
                f"{z_prop:.3f}",
                f"{result.pvalue:.5f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(
                name="Expected",
                x=["Proportion"],
                y=[expected_p],
                marker_color="rgba(255, 99, 71, 0.7)",
                width=[0.3],
                offsetgroup=0,
            )
        )
        fig2.add_trace(
            go.Bar(
                name="Observed",
                x=["Proportion"],
                y=[p_hat],
                marker_color="rgba(54, 162, 235, 0.7)",
                width=[0.3],
                offsetgroup=1,
            )
        )

        fig2.add_hline(
            y=expected_p,
            line_dash="dash",
            line_color="red",
            annotation_text="Expected",
        )
        fig2.add_hline(
            y=p_hat,
            line_dash="dot",
            line_color="blue",
            annotation_text="Observed",
        )

        # Error bar for CI
        fig2.add_trace(
            go.Scatter(
                x=["Proportion"],
                y=[p_hat],
                error_y=dict(
                    type="data",
                    symmetric=True,
                    array=[ci_prop],
                    visible=True,
                    color="blue",
                ),
                mode="markers",
                marker=dict(size=8, color="blue"),
                showlegend=False,
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=400,
            yaxis=dict(range=[0, 1]),
            barmode="group",
            xaxis_title="",
            yaxis_title="Proportion",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "One-sample Wilcoxon Signed-Rank Test":

        from scipy.stats import wilcoxon

        st.subheader("Interactive One-sample Wilcoxon Signed-Rank Test")

        # =========================
        # CONTROLS
        # =========================

        median_shift = st.slider(
            "Median Shift",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        sample = np.random.exponential(1, 80)

        sample = sample + median_shift

        stat, p = wilcoxon(sample)

        # =========================
        # STATS
        # =========================

        st.latex(rf"W = {stat:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=sample))

        fig.add_hline(y=0)

        fig.update_layout(
            template="plotly_dark",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        n_1w = len(sample)
        median_1w = np.median(sample)
        hypothesized_median = 0
        median_diff_1w = median_1w - hypothesized_median
        from scipy.stats import wilcoxon as wilcoxon_1samp

        T_1w, p_1w = wilcoxon_1samp(sample)
        r_rb_1w = 1 - 2 * T_1w / (n_1w * (n_1w + 1) / 2)

        results_data = {
            "Metric": [
                "Median",
                "Hypothesized Median",
                "W-statistic",
                "p-value",
                "Sample Size (n)",
                "Median Difference",
                "Rank-biserial r",
                "",
            ],
            "Value": [
                f"{median_1w:.3f}",
                f"{hypothesized_median}",
                f"{T_1w:.3f}",
                f"{p_1w:.5f}",
                f"{n_1w}",
                f"{median_diff_1w:.3f}",
                f"{r_rb_1w:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(go.Box(y=sample, name="Sample", boxmean="sd"))

        jitter_x = np.random.normal(0, 0.06, n_1w)
        fig2.add_trace(
            go.Scatter(
                x=jitter_x,
                y=sample,
                mode="markers",
                name="Data points",
                marker=dict(color="rgba(0, 123, 255, 0.5)", size=5),
            )
        )

        fig2.add_hline(
            y=hypothesized_median,
            line_dash="dash",
            line_color="red",
            annotation_text="H₀: median = 0",
        )

        ci_1w = (
            1.58
            * (np.percentile(sample, 75) - np.percentile(sample, 25))
            / np.sqrt(n_1w)
        )
        fig2.add_hline(
            y=median_1w + ci_1w, line_dash="dot", line_color="gray", opacity=0.5
        )
        fig2.add_hline(
            y=median_1w - ci_1w, line_dash="dot", line_color="gray", opacity=0.5
        )

        fig2.update_layout(
            template="plotly_dark",
            height=450,
            xaxis=dict(showticklabels=False),
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Chi-Square Goodness-of-Fit Test":

        from scipy.stats import chisquare

        st.subheader("Interactive Chi-Square Goodness-of-Fit Test")

        # =========================
        # CONTROLS
        # =========================

        obs1 = st.slider("Observed Category A", 1, 100, 40)
        obs2 = st.slider("Observed Category B", 1, 100, 30)
        obs3 = st.slider("Observed Category C", 1, 100, 20)

        # =========================
        # DATA
        # =========================

        observed = np.array([obs1, obs2, obs3])

        expected = np.mean(observed)

        chi2, p = chisquare(observed)

        # =========================
        # STATS
        # =========================

        st.latex(rf"\chi^2 = {chi2:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["A", "B", "C"],
                y=observed,
                name="Observed",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=["A", "B", "C"],
                y=[expected] * 3,
                mode="lines",
                name="Expected",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import chi2 as chi2_dist_gof

        n_gof = np.sum(observed)
        k_gof = len(observed)
        df_gof = k_gof - 1
        expected_val = expected
        cramer_v_gof = (
            np.sqrt(chi2 / (n_gof * (k_gof - 1))) if n_gof > 0 and k_gof > 1 else 0
        )

        results_data = {
            "Metric": [
                "Observed A",
                "Observed B",
                "Observed C",
                "Expected (mean)",
                "χ²",
                "df",
                "p-value",
                "Cramer's V",
            ],
            "Value": [
                f"{observed[0]}",
                f"{observed[1]}",
                f"{observed[2]}",
                f"{expected_val:.1f}",
                f"{chi2:.3f}",
                f"{df_gof}",
                f"{p:.5f}",
                f"{cramer_v_gof:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        categories_gof = ["A", "B", "C"]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                name="Observed",
                x=categories_gof,
                y=observed,
                marker_color="rgba(54, 162, 235, 0.7)",
            )
        )
        fig2.add_trace(
            go.Bar(
                name="Expected",
                x=categories_gof,
                y=[expected_val] * 3,
                marker_color="rgba(255, 99, 71, 0.7)",
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Category",
            yaxis_title="Count",
            barmode="group",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Categorial Tests
    elif test_name == "Chi-Square Test":

        from scipy.stats import chi2_contingency

        st.subheader("Interactive Chi-Square Test of Independence")

        # =========================
        # CONTROLS
        # =========================

        a = st.slider("Cell A", 1, 100, 40)
        b = st.slider("Cell B", 1, 100, 20)
        c = st.slider("Cell C", 1, 100, 10)
        d = st.slider("Cell D", 1, 100, 30)

        # =========================
        # TABLE
        # =========================

        table = np.array(
            [
                [a, b],
                [c, d],
            ]
        )

        chi2, p, dof, expected = chi2_contingency(table)

        # =========================
        # STATS
        # =========================

        st.latex(rf"\chi^2 = {chi2:.3f}")

        st.latex(rf"\text{{Degrees of Freedom}} = {dof}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # HEATMAP
        # =========================

        fig = go.Figure(
            data=go.Heatmap(
                z=table,
                text=table,
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import chi2 as chi2_dist_cs

        n_cs = np.sum(table)
        cramer_v_cs = (
            np.sqrt(chi2 / (n_cs * min(table.shape[0] - 1, table.shape[1] - 1)))
            if n_cs > 0
            else 0
        )

        results_data = {
            "Metric": [
                "χ²",
                "df",
                "p-value",
                "Cramer's V",
                "Cell A (R1,C1)",
                "Cell B (R1,C2)",
                "Cell C (R2,C1)",
                "Cell D (R2,C2)",
            ],
            "Value": [
                f"{chi2:.3f}",
                f"{dof}",
                f"{p:.5f}",
                f"{cramer_v_cs:.4f}",
                f"{table[0, 0]}",
                f"{table[0, 1]}",
                f"{table[1, 0]}",
                f"{table[1, 1]}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        cells_cs = ["(R1,C1)", "(R1,C2)", "(R2,C1)", "(R2,C2)"]
        observed_flat = table.flatten()
        expected_flat = expected.flatten()
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                name="Observed",
                x=cells_cs,
                y=observed_flat,
                marker_color="rgba(54, 162, 235, 0.7)",
            )
        )
        fig2.add_trace(
            go.Bar(
                name="Expected",
                x=cells_cs,
                y=expected_flat,
                marker_color="rgba(255, 99, 71, 0.7)",
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Cell",
            yaxis_title="Count",
            barmode="group",
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "McNemar's Test":

        from statsmodels.stats.contingency_tables import mcnemar

        st.subheader("Interactive McNemar's Test")

        # =========================
        # CONTROLS
        # =========================

        yes_yes = st.slider("Yes → Yes", 0, 100, 40)

        yes_no = st.slider("Yes → No", 0, 100, 10)

        no_yes = st.slider("No → Yes", 0, 100, 30)

        no_no = st.slider("No → No", 0, 100, 20)

        # =========================
        # TABLE
        # =========================

        table = np.array(
            [
                [yes_yes, yes_no],
                [no_yes, no_no],
            ]
        )

        result = mcnemar(table)

        # =========================
        # STATS
        # =========================

        st.latex(rf"\chi^2 = {result.statistic:.3f}")

        st.latex(rf"\text{{p-value}} = {result.pvalue:.5f}")

        # =========================
        # HEATMAP
        # =========================

        fig = go.Figure(
            data=go.Heatmap(
                z=table,
                text=table,
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        b_mc = table[0, 1]
        c_mc = table[1, 0]
        odds_ratio_mc = b_mc / c_mc if c_mc > 0 else float("inf")

        results_data = {
            "Metric": [
                "χ²",
                "p-value",
                "b (Yes→No)",
                "c (No→Yes)",
                "Odds Ratio (b/c)",
                "",
                "",
                "",
            ],
            "Value": [
                f"{result.statistic:.3f}",
                f"{result.pvalue:.5f}",
                f"{b_mc}",
                f"{c_mc}",
                f"{odds_ratio_mc:.3f}" if c_mc > 0 else "∞",
                "",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                name="Discordant Pairs",
                x=["b (Yes→No)", "c (No→Yes)"],
                y=[b_mc, c_mc],
                marker_color=["rgba(255, 99, 71, 0.7)", "rgba(54, 162, 235, 0.7)"],
            )
        )
        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Discordant Pair Type",
            yaxis_title="Count",
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Cochran's Q Test":

        from statsmodels.stats.contingency_tables import cochrans_q

        st.subheader("Interactive Cochran's Q Test")

        # =========================
        # CONTROLS
        # =========================

        prob1 = st.slider(
            "Condition 1 Success Probability",
            0.0,
            1.0,
            0.3,
            0.01,
        )

        prob2 = st.slider(
            "Condition 2 Success Probability",
            0.0,
            1.0,
            0.5,
            0.01,
        )

        prob3 = st.slider(
            "Condition 3 Success Probability",
            0.0,
            1.0,
            0.7,
            0.01,
        )

        subjects = st.slider(
            "Subjects",
            10,
            300,
            100,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        c1 = np.random.binomial(1, prob1, subjects)

        c2 = np.random.binomial(1, prob2, subjects)

        c3 = np.random.binomial(1, prob3, subjects)

        data = np.column_stack([c1, c2, c3])

        result = cochrans_q(data)

        # =========================
        # STATS
        # =========================

        st.latex(rf"Q = {result.statistic:.3f}")

        st.latex(rf"\text{{p-value}} = {result.pvalue:.5f}")

        # =========================
        # PLOT
        # =========================

        means = data.mean(axis=0)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Condition 1", "Condition 2", "Condition 3"],
                y=means,
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            yaxis=dict(range=[0, 1]),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        k_cq = data.shape[1]
        df_cq = k_cq - 1
        proportions = means

        results_data = {
            "Metric": [
                "Q",
                "df",
                "p-value",
                "Proportion C1",
                "Proportion C2",
                "Proportion C3",
                "",
                "",
            ],
            "Value": [
                f"{result.statistic:.3f}",
                f"{df_cq}",
                f"{result.pvalue:.5f}",
                f"{proportions[0]:.3f}",
                f"{proportions[1]:.3f}",
                f"{proportions[2]:.3f}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        cond_names = ["Condition 1", "Condition 2", "Condition 3"]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=cond_names,
                y=proportions,
                marker_color=[
                    "rgba(54, 162, 235, 0.7)",
                    "rgba(255, 99, 71, 0.7)",
                    "rgba(75, 192, 192, 0.7)",
                ],
                name="Proportion",
            )
        )
        np.random.seed(123)
        for j in range(data.shape[1]):
            jitter_x = np.random.normal(j, 0.05, size=data.shape[0])
            response_y = data[:, j]
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=response_y,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="white", size=3, opacity=0.3),
                )
            )
        fig2.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Condition",
            yaxis_title="Proportion",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Fisher's Exact Test":

        from scipy.stats import fisher_exact

        st.subheader("Interactive Fisher's Exact Test")

        # =========================
        # CONTROLS
        # =========================

        a = st.slider("Cell A", 0, 50, 8)

        b = st.slider("Cell B", 0, 50, 2)

        c = st.slider("Cell C", 0, 50, 1)

        d = st.slider("Cell D", 0, 50, 9)

        # =========================
        # TABLE
        # =========================

        table = np.array(
            [
                [a, b],
                [c, d],
            ]
        )

        odds_ratio, p = fisher_exact(table)

        # =========================
        # STATS
        # =========================

        st.latex(rf"OR = {odds_ratio:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # HEATMAP
        # =========================

        fig = go.Figure(
            data=go.Heatmap(
                z=table,
                text=table,
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import fisher_exact as fisher_exact2

        or_f, p_f = fisher_exact2(table)
        log_or = np.log(or_f) if or_f > 0 else 0
        se_log_or = np.sqrt(np.sum(1 / table[table > 0])) if np.all(table > 0) else 0
        ci_low_f = np.exp(log_or - 1.96 * se_log_or) if se_log_or > 0 else 0
        ci_high_f = np.exp(log_or + 1.96 * se_log_or) if se_log_or > 0 else float("inf")

        results_data = {
            "Metric": [
                "Odds Ratio",
                "95% CI (OR)",
                "p-value",
                "",
                "Cell A",
                "Cell B",
                "Cell C",
                "Cell D",
            ],
            "Value": [
                f"{or_f:.3f}",
                f"[{ci_low_f:.3f}, {ci_high_f:.3f}]",
                f"{p_f:.5f}",
                "",
                f"{table[0, 0]}",
                f"{table[0, 1]}",
                f"{table[1, 0]}",
                f"{table[1, 1]}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        fig2 = go.Figure(
            data=go.Heatmap(
                z=table,
                text=table,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        fig2.update_layout(template="plotly_dark", height=400)
        fig2.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text=f"OR = {or_f:.3f}, 95% CI: [{ci_low_f:.3f}, {ci_high_f:.3f}]",
            showarrow=False,
            font=dict(size=14),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Parametric Two Sample Tests

    elif test_name == "Student's t-test (Independent)":

        from scipy.stats import ttest_ind

        st.subheader("Interactive Independent t-test")

        # =========================
        # CONTROLS
        # =========================

        mean_diff = st.slider(
            "Mean Difference",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        sd = st.slider(
            "Shared Standard Deviation",
            0.5,
            5.0,
            1.5,
            0.1,
        )

        n = st.slider(
            "Sample Size per Group",
            10,
            300,
            50,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        group1 = np.random.normal(0, sd, n)

        group2 = np.random.normal(mean_diff, sd, n)

        t, p = ttest_ind(group1, group2)

        # =========================
        # STATS
        # =========================

        st.latex(rf"t = {t:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=group1, name="Group 1"))
        fig.add_trace(go.Box(y=group2, name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import t as t_dist_student

        n1_s, n2_s = len(group1), len(group2)
        m1_s, m2_s = np.mean(group1), np.mean(group2)
        sd1_s, sd2_s = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_sd = np.sqrt(
            ((n1_s - 1) * sd1_s**2 + (n2_s - 1) * sd2_s**2) / (n1_s + n2_s - 2)
        )
        mean_diff_s = m2_s - m1_s
        se_diff_s = pooled_sd * np.sqrt(1 / n1_s + 1 / n2_s)
        ci_diff_s = se_diff_s * t_dist_student.ppf(0.975, n1_s + n2_s - 2)
        cohens_d_s = mean_diff_s / pooled_sd

        results_data = {
            "Metric": [
                "Mean G1 (SD)",
                "Mean G2 (SD)",
                "Mean Difference",
                "95% CI of Diff",
                "Pooled SD",
                "t-statistic",
                "df",
                "p-value",
                "Cohen's d",
                "",
                "",
                "",
            ],
            "Value": [
                f"{m1_s:.2f} ({sd1_s:.2f})",
                f"{m2_s:.2f} ({sd2_s:.2f})",
                f"{mean_diff_s:.3f}",
                f"[{mean_diff_s - ci_diff_s:.3f}, {mean_diff_s + ci_diff_s:.3f}]",
                f"{pooled_sd:.3f}",
                f"{t:.3f}",
                f"{n1_s + n2_s - 2}",
                f"{p:.5f}",
                f"{cohens_d_s:.4f}",
                "",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(zip([group1, group2], ["Group 1", "Group 2"])):
            fig2.add_trace(
                go.Violin(
                    y=g,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Welch's t-test (Independent, Unequal Variances)":

        from scipy.stats import ttest_ind

        st.subheader("Interactive Welch's t-test")

        # =========================
        # CONTROLS
        # =========================

        mean_diff = st.slider("Mean Difference", 0.0, 10.0, 2.0, 0.1)

        sd1 = st.slider("Group 1 SD", 0.5, 8.0, 1.0, 0.1)

        sd2 = st.slider("Group 2 SD", 0.5, 8.0, 3.0, 0.1)

        n = st.slider("Sample Size", 10, 300, 50)

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        g1 = np.random.normal(0, sd1, n)

        g2 = np.random.normal(mean_diff, sd2, n)

        t, p = ttest_ind(g1, g2, equal_var=False)

        # =========================
        # STATS
        # =========================

        st.latex(rf"t = {t:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Violin(y=g1, name="Group 1"))

        fig.add_trace(go.Violin(y=g2, name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import t as t_dist_welch

        n1_w, n2_w = len(g1), len(g2)
        m1_w, m2_w = np.mean(g1), np.mean(g2)
        sd1_w, sd2_w = np.std(g1, ddof=1), np.std(g2, ddof=1)
        mean_diff_w = m2_w - m1_w
        se_w = np.sqrt(sd1_w**2 / n1_w + sd2_w**2 / n2_w)

        welch_df_num = (sd1_w**2 / n1_w + sd2_w**2 / n2_w) ** 2
        welch_df_den = (sd1_w**2 / n1_w) ** 2 / (n1_w - 1) + (
            sd2_w**2 / n2_w
        ) ** 2 / (n2_w - 1)
        welch_df = welch_df_num / welch_df_den

        ci_diff_w = se_w * t_dist_welch.ppf(0.975, welch_df)

        pooled_sd_w = np.sqrt((sd1_w**2 + sd2_w**2) / 2)
        cohens_d_w = mean_diff_w / pooled_sd_w

        results_data = {
            "Metric": [
                "Mean G1 (SD)",
                "Mean G2 (SD)",
                "Mean Difference",
                "95% CI of Diff",
                "t-statistic",
                "Welch df",
                "p-value",
                "Cohen's d",
            ],
            "Value": [
                f"{m1_w:.2f} ({sd1_w:.2f})",
                f"{m2_w:.2f} ({sd2_w:.2f})",
                f"{mean_diff_w:.3f}",
                f"[{mean_diff_w - ci_diff_w:.3f}, {mean_diff_w + ci_diff_w:.3f}]",
                f"{t:.3f}",
                f"{welch_df:.1f}",
                f"{p:.5f}",
                f"{cohens_d_w:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(zip([g1, g2], ["Group 1", "Group 2"])):
            fig2.add_trace(
                go.Violin(
                    y=g,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Paired t-test":

        from scipy.stats import ttest_rel

        st.subheader("Interactive Paired t-test")

        # =========================
        # CONTROLS
        # =========================

        effect = st.slider(
            "Treatment Effect",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        n = st.slider("Number of Subjects", 10, 200, 40)

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        before = np.random.normal(10, noise, n)

        after = before + effect + np.random.normal(0, noise, n)

        t, p = ttest_rel(before, after)

        # =========================
        # STATS
        # =========================

        st.latex(rf"t = {t:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i in range(n):

            fig.add_trace(
                go.Scatter(
                    x=["Before", "After"],
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import t as t_dist_paired

        mean_pre = np.mean(before)
        mean_post = np.mean(after)
        sd_pre = np.std(before, ddof=1)
        sd_post = np.std(after, ddof=1)
        diffs = after - before
        mean_diff_p = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        n_p = len(before)
        se_diff = sd_diff / np.sqrt(n_p)
        ci_diff_paired = se_diff * t_dist_paired.ppf(0.975, n_p - 1)
        cohens_dz = mean_diff_p / sd_diff

        results_data = {
            "Metric": [
                "Mean Pre (SD)",
                "Mean Post (SD)",
                "Mean Difference",
                "95% CI of Diff",
                "t-statistic",
                "df",
                "p-value",
                "Cohen's d_z",
            ],
            "Value": [
                f"{mean_pre:.2f} ({sd_pre:.2f})",
                f"{mean_post:.2f} ({sd_post:.2f})",
                f"{mean_diff_p:.3f}",
                f"[{mean_diff_p - ci_diff_paired:.3f}, {mean_diff_p + ci_diff_paired:.3f}]",
                f"{t:.3f}",
                f"{n_p - 1}",
                f"{p:.5f}",
                f"{cohens_dz:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i in range(n_p):
            fig2.add_trace(
                go.Scatter(
                    x=["Before", "After"],
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=["Before", "After"],
                y=[mean_pre, mean_post],
                mode="lines+markers",
                name="Mean change",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=12),
            )
        )

        fig2.add_hline(
            y=mean_pre,
            line_dash="dot",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"Pre Mean = {mean_pre:.2f}",
        )

        fig2.add_hrect(
            y0=mean_diff_p - ci_diff_paired,
            y1=mean_diff_p + ci_diff_paired,
            x0=-0.5,
            x1=1.5,
            fillcolor="rgba(255, 0, 0, 0.1)",
            line_width=0,
            name="95% CI of difference",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time Point",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Parametric Multiple Group Tests

    elif test_name == "One-way ANOVA":

        from scipy.stats import f_oneway

        st.subheader("Interactive One-way ANOVA")

        # =========================
        # CONTROLS
        # =========================

        mean_shift = st.slider(
            "Group Separation",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        noise = st.slider(
            "Within-group Variability",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        g1 = np.random.normal(0, noise, 60)
        g2 = np.random.normal(mean_shift, noise, 60)
        g3 = np.random.normal(mean_shift * 2, noise, 60)

        F, p = f_oneway(g1, g2, g3)

        # =========================
        # STATS
        # =========================

        st.latex(rf"F = {F:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=g1, name="Group 1"))
        fig.add_trace(go.Box(y=g2, name="Group 2"))
        fig.add_trace(go.Box(y=g3, name="Group 3"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from scipy.stats import f as f_dist_1w

        groups_1w = [g1, g2, g3]
        means_1w = [np.mean(g) for g in groups_1w]
        sds_1w = [np.std(g, ddof=1) for g in groups_1w]
        n_1w = [len(g) for g in groups_1w]
        n_total_1w = sum(n_1w)
        k_1w = len(groups_1w)
        grand_mean_1w = np.mean(np.concatenate(groups_1w))

        ss_between = sum(
            n_i * (m_i - grand_mean_1w) ** 2 for n_i, m_i in zip(n_1w, means_1w)
        )
        ss_within = sum((n_i - 1) * sd_i**2 for n_i, sd_i in zip(n_1w, sds_1w))
        df_between = k_1w - 1
        df_within = n_total_1w - k_1w
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        F_1w = ms_between / ms_within
        p_1w = 1 - f_dist_1w.cdf(F_1w, df_between, df_within)
        eta_sq = ss_between / (ss_between + ss_within)
        omega_sq = (ss_between - df_between * ms_within) / (
            ss_between + ss_within + ms_within
        )

        results_data = {
            "Metric": [
                "Mean G1 (SD)",
                "Mean G2 (SD)",
                "Mean G3 (SD)",
                "F",
                f"df ({df_between}, {df_within})",
                "p-value",
                "η²",
                "Partial ω²",
            ],
            "Value": [
                f"{means_1w[0]:.2f} ({sds_1w[0]:.2f})",
                f"{means_1w[1]:.2f} ({sds_1w[1]:.2f})",
                f"{means_1w[2]:.2f} ({sds_1w[2]:.2f})",
                f"{F_1w:.3f}",
                f"{df_between}, {df_within}",
                f"{p_1w:.5f}",
                f"{eta_sq:.4f}",
                f"{omega_sq:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(
            zip(groups_1w, ["Group 1", "Group 2", "Group 3"])
        ):
            fig2.add_trace(
                go.Violin(
                    y=g,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.add_hline(
            y=grand_mean_1w,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Grand Mean = {grand_mean_1w:.2f}",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Two-way ANOVA":

        from scipy.stats import f_oneway
        from scipy.stats import f as f_dist

        st.subheader("Interactive Two-way ANOVA")

        # =========================
        # CONTROLS
        # =========================

        effect_A = st.slider("Effect of Factor A (Group)", 0.0, 10.0, 2.0, 0.1)

        effect_B = st.slider("Effect of Factor B (Sex)", 0.0, 10.0, 1.0, 0.1)

        interaction = st.slider("Interaction (A × B)", -5.0, 5.0, 0.0, 0.1)

        noise = st.slider("Within-group Variability", 0.1, 5.0, 1.0, 0.1)

        # =========================
        # DATA
        # =========================

        np.random.seed(42)
        n_per_cell = 30

        A1B1 = np.random.normal(0, noise, n_per_cell)
        A1B2 = np.random.normal(effect_B, noise, n_per_cell)
        A2B1 = np.random.normal(effect_A, noise, n_per_cell)
        A2B2 = np.random.normal(effect_A + effect_B + interaction, noise, n_per_cell)

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=A1B1, name="A1 (Control), B1 (Male)"))
        fig.add_trace(go.Box(y=A1B2, name="A1 (Control), B2 (Female)"))
        fig.add_trace(go.Box(y=A2B1, name="A2 (Drug), B1 (Male)"))
        fig.add_trace(go.Box(y=A2B2, name="A2 (Drug), B2 (Female)"))

        fig.update_layout(template="plotly_dark", height=550)

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from statsmodels.formula.api import ols as ols_tw
        import statsmodels.api as sm_tw

        df_tw = pd.DataFrame(
            {
                "y": np.concatenate([A1B1, A1B2, A2B1, A2B2]),
                "A": np.repeat(["A1", "A1", "A2", "A2"], n_per_cell),
                "B": np.repeat(["B1", "B2", "B1", "B2"], n_per_cell),
            }
        )
        model_tw = ols_tw("y ~ C(A) * C(B)", data=df_tw).fit()
        anova_tw = sm_tw.stats.anova_lm(model_tw, typ=2)

        F_A = anova_tw.loc["C(A)", "F"]
        p_A = anova_tw.loc["C(A)", "PR(>F)"]
        F_B = anova_tw.loc["C(B)", "F"]
        p_B = anova_tw.loc["C(B)", "PR(>F)"]
        F_AB = anova_tw.loc["C(A):C(B)", "F"]
        p_AB = anova_tw.loc["C(A):C(B)", "PR(>F)"]

        ss_A = anova_tw.loc["C(A)", "sum_sq"]
        ss_B = anova_tw.loc["C(B)", "sum_sq"]
        ss_AB = anova_tw.loc["C(A):C(B)", "sum_sq"]
        ss_resid_tw = anova_tw.loc["Residual", "sum_sq"]

        partial_eta_A = ss_A / (ss_A + ss_resid_tw)
        partial_eta_B = ss_B / (ss_B + ss_resid_tw)
        partial_eta_AB = ss_AB / (ss_AB + ss_resid_tw)

        results_data = {
            "Metric": [
                "Mean A1B1",
                "Mean A1B2",
                "Mean A2B1",
                "Mean A2B2",
                "F_A",
                "p_A",
                "Partial η²_A",
                "",
                "F_B",
                "p_B",
                "Partial η²_B",
                "",
                "F_AB",
                "p_AB",
                "Partial η²_AB",
                "",
            ],
            "Value": [
                f"{np.mean(A1B1):.3f}",
                f"{np.mean(A1B2):.3f}",
                f"{np.mean(A2B1):.3f}",
                f"{np.mean(A2B2):.3f}",
                f"{F_A:.3f}",
                f"{p_A:.5f}",
                f"{partial_eta_A:.4f}",
                "",
                f"{F_B:.3f}",
                f"{p_B:.5f}",
                f"{partial_eta_B:.4f}",
                "",
                f"{F_AB:.3f}",
                f"{p_AB:.5f}",
                f"{partial_eta_AB:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        cell_means = {
            "A1": {"B1": np.mean(A1B1), "B2": np.mean(A1B2)},
            "A2": {"B1": np.mean(A2B1), "B2": np.mean(A2B2)},
        }
        for level_B, color, dash in [("B1", "blue", "solid"), ("B2", "red", "dash")]:
            means = [cell_means[a][level_B] for a in ["A1", "A2"]]
            fig2.add_trace(
                go.Scatter(
                    x=["A1", "A2"],
                    y=means,
                    mode="lines+markers",
                    name=f"Factor B: {level_B}",
                    line=dict(color=color, width=3, dash=dash),
                    marker=dict(size=10),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Factor A",
            yaxis_title="Cell Mean",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "ANCOVA":

        from scipy.stats import linregress

        st.subheader("Interactive ANCOVA")

        # =========================
        # CONTROLS
        # =========================

        treatment_effect = st.slider("Treatment Effect", 0.0, 10.0, 3.0, 0.1)

        covariate_strength = st.slider("Covariate Strength (β)", 0.0, 3.0, 1.0, 0.1)

        noise = st.slider("Noise", 0.1, 5.0, 1.0, 0.1)

        # =========================
        # DATA
        # =========================

        np.random.seed(42)
        n = 40

        covariate = np.random.normal(50, 10, n)

        control = covariate * covariate_strength + np.random.normal(0, noise, n)

        treatment = (
            covariate * covariate_strength
            + treatment_effect
            + np.random.normal(0, noise, n)
        )

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=covariate,
                y=control,
                mode="markers",
                name="Control",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=covariate,
                y=treatment,
                mode="markers",
                name="Treatment",
            )
        )

        # Regression lines
        slope_c, intercept_c, _, _, _ = linregress(covariate, control)
        slope_t, intercept_t, _, _, _ = linregress(covariate, treatment)

        x_line = np.linspace(covariate.min(), covariate.max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept_c + slope_c * x_line,
                mode="lines",
                name="Control (adjusted)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept_t + slope_t * x_line,
                mode="lines",
                name="Treatment (adjusted)",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Covariate (Baseline)",
            yaxis_title="Outcome",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        from statsmodels.formula.api import ols as ols_anc

        df_anc = pd.DataFrame(
            {
                "outcome": np.concatenate([control, treatment]),
                "group": np.repeat(["Control", "Treatment"], n),
                "cov": np.tile(covariate, 2),
            }
        )
        model_anc = ols_anc("outcome ~ C(group) + cov", data=df_anc).fit()
        beta_cov = model_anc.params["cov"]
        grand_mean_cov = np.mean(covariate)
        adj_control = model_anc.params["Intercept"] + beta_cov * grand_mean_cov
        adj_treatment = (
            model_anc.params["Intercept"]
            + model_anc.params["C(group)[T.Treatment]"]
            + beta_cov * grand_mean_cov
        )
        F_group = model_anc.fvalue
        p_group = model_anc.f_pvalue
        ss_resid = model_anc.ssr
        ss_expl = model_anc.ess
        partial_eta_anc = ss_expl / (ss_expl + ss_resid)

        results_data = {
            "Metric": [
                "Adj. Mean Control",
                "Adj. Mean Treatment",
                "F (group)",
                "p-value",
                "Covariate β",
                "Partial η²",
                "",
                "",
            ],
            "Value": [
                f"{adj_control:.3f}",
                f"{adj_treatment:.3f}",
                f"{F_group:.3f}",
                f"{p_group:.5f}",
                f"{beta_cov:.3f}",
                f"{partial_eta_anc:.4f}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=covariate,
                y=control,
                mode="markers",
                name="Control",
                marker=dict(color="rgba(0, 123, 255, 0.6)", size=6),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=covariate,
                y=treatment,
                mode="markers",
                name="Treatment",
                marker=dict(color="rgba(255, 65, 54, 0.6)", size=6),
            )
        )

        slope_c, intercept_c, _, _, _ = linregress(covariate, control)
        slope_t, intercept_t, _, _, _ = linregress(covariate, treatment)
        x_line = np.linspace(covariate.min(), covariate.max(), 100)
        fig2.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept_c + slope_c * x_line,
                mode="lines",
                name="Control (adjusted)",
                line=dict(color="blue", width=2, dash="dash"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept_t + slope_t * x_line,
                mode="lines",
                name="Treatment (adjusted)",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        fig2.add_vline(
            x=grand_mean_cov, line_dash="dot", line_color="gray", opacity=0.5
        )
        fig2.add_annotation(
            x=grand_mean_cov,
            y=adj_control,
            text=f"Adj. Control: {adj_control:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=-30,
            bgcolor="blue",
        )
        fig2.add_annotation(
            x=grand_mean_cov,
            y=adj_treatment,
            text=f"Adj. Treatment: {adj_treatment:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=30,
            ay=30,
            bgcolor="red",
        )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Covariate (Baseline)",
            yaxis_title="Outcome",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Repeated Measures ANOVA":

        st.subheader("Interactive Repeated Measures ANOVA")

        # =========================
        # CONTROLS
        # =========================

        trend = st.slider(
            "Time Trend",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        subjects = st.slider(
            "Subjects",
            5,
            100,
            20,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        timepoints = 4

        data = []

        for s in range(subjects):

            baseline = np.random.normal(10, noise)

            vals = [
                baseline + trend * t + np.random.normal(0, noise)
                for t in range(timepoints)
            ]

            data.append(vals)

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for vals in data:

            fig.add_trace(
                go.Scatter(
                    x=[1, 2, 3, 4],
                    y=vals,
                    mode="lines+markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time",
            yaxis_title="Measurement",
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        time_means = [
            np.mean([data[s][t] for s in range(subjects)]) for t in range(timepoints)
        ]

        df_rm = pd.DataFrame(
            {
                "subject": np.tile(range(subjects), timepoints),
                "time": np.repeat(range(timepoints), subjects),
                "y": np.array(data).T.flatten(),
            }
        )

        from statsmodels.stats.anova import AnovaRM

        rm_result = AnovaRM(df_rm, "y", "subject", within=["time"]).fit()
        rm_table = rm_result.anova_table
        F_rm = rm_table.loc["time", "F Value"]
        p_rm = rm_table.loc["time", "Pr > F"]
        df_num_rm = int(rm_table.loc["time", "Num DF"])
        df_den_rm = int(rm_table.loc["time", "Den DF"])
        partial_eta_sq_rm = (
            rm_table.loc["time", "F Value"]
            * df_num_rm
            / (rm_table.loc["time", "F Value"] * df_num_rm + df_den_rm)
        )

        results_data = {
            "Metric": [
                "Mean T1",
                "Mean T2",
                "Mean T3",
                "Mean T4",
                "F",
                f"df ({df_num_rm}, {df_den_rm})",
                "p-value",
                "Partial η²",
            ],
            "Value": [
                f"{time_means[0]:.3f}",
                f"{time_means[1]:.3f}",
                f"{time_means[2]:.3f}",
                f"{time_means[3]:.3f}",
                f"{F_rm:.3f}",
                f"{df_num_rm}, {df_den_rm}",
                f"{p_rm:.5f}",
                f"{partial_eta_sq_rm:.4f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for vals in data:
            fig2.add_trace(
                go.Scatter(
                    x=[1, 2, 3, 4],
                    y=vals,
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=[1, 2, 3, 4],
                y=time_means,
                mode="lines+markers",
                name="Mean trend",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=10),
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time",
            yaxis_title="Measurement",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "MANOVA":

        st.subheader("Interactive MANOVA")

        # =========================
        # CONTROLS
        # =========================

        separation = st.slider(
            "Group Separation",
            0.0,
            10.0,
            3.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        n = 100

        g1 = np.random.multivariate_normal(
            [0, 0, 0],
            np.eye(3),
            n,
        )

        g2 = np.random.multivariate_normal(
            [separation, separation, separation],
            np.eye(3),
            n,
        )

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=g1[:, 0],
                y=g1[:, 1],
                z=g1[:, 2],
                mode="markers",
                name="Group 1",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=g2[:, 0],
                y=g2[:, 1],
                z=g2[:, 2],
                mode="markers",
                name="Group 2",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=700,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        n_m = len(g1)
        y_all = np.vstack([g1, g2])
        grand_mean = np.mean(y_all, axis=0)
        mean1 = np.mean(g1, axis=0)
        mean2 = np.mean(g2, axis=0)

        H_m = n_m * np.outer(mean1 - grand_mean, mean1 - grand_mean) + n_m * np.outer(
            mean2 - grand_mean, mean2 - grand_mean
        )
        E_m = (n_m - 1) * np.cov(g1, rowvar=False) + (n_m - 1) * np.cov(
            g2, rowvar=False
        )
        wilks_lambda = np.linalg.det(E_m) / np.linalg.det(E_m + H_m)

        p_manova = 3
        k_manova = 2
        df1_m = p_manova
        df2_m = 2 * n_m - p_manova - 1
        F_m = ((1 - wilks_lambda) / wilks_lambda) * (df2_m / df1_m)

        from scipy.stats import f as f_dist_m

        p_m = 1 - f_dist_m.cdf(F_m, df1_m, df2_m)

        results_data = {
            "Metric": [
                "Wilks' Λ",
                "F-approx",
                f"df ({df1_m}, {df2_m:.0f})",
                "p-value",
                "Number of DVs",
                "Number of Groups",
                "",
                "",
            ],
            "Value": [
                f"{wilks_lambda:.4f}",
                f"{F_m:.3f}",
                f"{df1_m}, {df2_m:.0f}",
                f"{p_m:.5f}",
                f"{p_manova}",
                f"{k_manova}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        fig2 = go.Figure()
        for idx, (g, name, color) in enumerate(
            zip([g1, g2], ["Group 1", "Group 2"], ["blue", "red"])
        ):
            fig2.add_trace(
                go.Scatter(
                    x=g[:, 0],
                    y=g[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=6, opacity=0.5),
                )
            )
            centroid = np.mean(g[:, :2], axis=0)
            fig2.add_trace(
                go.Scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode="markers",
                    showlegend=False,
                    marker=dict(color=color, size=15, symbol="x"),
                )
            )
        fig2.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Non-parametric Two Sample Tests

    elif test_name == "Wilcoxon Signed-Rank Test":

        from scipy.stats import wilcoxon

        st.subheader("Interactive Wilcoxon Signed-Rank Test")

        # =========================
        # CONTROLS
        # =========================

        median_shift = st.slider(
            "Median Shift",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        n = st.slider(
            "Sample Size",
            10,
            200,
            40,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        before = np.random.exponential(1, n)

        after = before + median_shift + np.random.normal(0, noise, n)

        stat, p = wilcoxon(before, after)

        # =========================
        # STATS
        # =========================

        st.latex(rf"W = {stat:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i in range(n):

            fig.add_trace(
                go.Scatter(
                    x=["Before", "After"],
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        median_pre = np.median(before)
        median_post = np.median(after)
        median_diff = median_post - median_pre
        from scipy.stats import norm as norm_w

        z_w = -norm_w.ppf(p / 2)
        r_ws = z_w / np.sqrt(n) if n > 0 else 0

        results_data = {
            "Metric": [
                "Median (Pre)",
                "Median (Post)",
                "Median Difference",
                "W-statistic",
                "z",
                "p-value",
                "Rank-biserial r",
                "",
            ],
            "Value": [
                f"{median_pre:.3f}",
                f"{median_post:.3f}",
                f"{median_diff:.3f}",
                f"{stat:.3f}",
                f"{z_w:.3f}",
                f"{p:.5f}",
                f"{r_ws:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i in range(n):
            fig2.add_trace(
                go.Scatter(
                    x=["Before", "After"],
                    y=[before[i], after[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=["Before", "After"],
                y=[median_pre, median_post],
                mode="lines+markers",
                name="Median change",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=12),
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time Point",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Mann-Whitney U Test":

        from scipy.stats import mannwhitneyu

        st.subheader("Interactive Mann-Whitney U Test")

        # =========================
        # CONTROLS
        # =========================

        location_shift = st.slider(
            "Distribution Shift",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        spread = st.slider(
            "Distribution Spread",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        n = st.slider(
            "Sample Size",
            10,
            300,
            60,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        g1 = np.random.exponential(spread, n)

        g2 = np.random.exponential(spread, n) + location_shift

        u, p = mannwhitneyu(g1, g2)

        # =========================
        # STATS
        # =========================

        st.latex(rf"U = {u:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Violin(y=g1, name="Group 1"))
        fig.add_trace(go.Violin(y=g2, name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        medians_mw = [np.median(g) for g in [g1, g2]]
        iqr_mw = [
            f"{np.percentile(g, 25):.2f}–{np.percentile(g, 75):.2f}" for g in [g1, g2]
        ]
        from scipy.stats import norm as norm_mw

        z_mw = -norm_mw.ppf(p / 2)
        r_rb_mw = 1 - 2 * min(u, n * n - u) / (n * n)

        results_data = {
            "Metric": [
                "Median G1 (IQR)",
                "Median G2 (IQR)",
                "U-statistic",
                "z",
                "p-value",
                "Rank-biserial r",
                "",
                "",
            ],
            "Value": [
                f"{medians_mw[0]:.3f} ({iqr_mw[0]})",
                f"{medians_mw[1]:.3f} ({iqr_mw[1]})",
                f"{u:.3f}",
                f"{z_mw:.3f}",
                f"{p:.5f}",
                f"{r_rb_mw:.4f}",
                "",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(zip([g1, g2], ["Group 1", "Group 2"])):
            fig2.add_trace(
                go.Violin(y=g, name=name, box_visible=True, meanline_visible=True)
            )
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Non-parametric Multiple Group Tests

    elif test_name == "Kruskal-Wallis Test":

        from scipy.stats import kruskal

        st.subheader("Interactive Kruskal-Wallis Test")

        # =========================
        # CONTROLS
        # =========================

        shift = st.slider(
            "Group Separation",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        spread = st.slider(
            "Distribution Spread",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        g1 = np.random.gamma(2, spread, 60)

        g2 = np.random.gamma(2, spread, 60) + shift

        g3 = np.random.gamma(2, spread, 60) + shift * 2

        H, p = kruskal(g1, g2, g3)

        # =========================
        # STATS
        # =========================

        st.latex(rf"H = {H:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Box(y=g1, name="Group 1"))
        fig.add_trace(go.Box(y=g2, name="Group 2"))
        fig.add_trace(go.Box(y=g3, name="Group 3"))

        fig.update_layout(
            template="plotly_dark",
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        n_kw = [len(g1), len(g2), len(g3)]
        medians_kw = [np.median(g) for g in [g1, g2, g3]]
        iqr_kw = [
            f"{np.percentile(g, 25):.2f}–{np.percentile(g, 75):.2f}"
            for g in [g1, g2, g3]
        ]
        n_total_kw = sum(n_kw)
        eps_sq = H / (n_total_kw - 1) if n_total_kw > 1 else 0

        results_data = {
            "Metric": [
                "Median G1 (IQR)",
                "Median G2 (IQR)",
                "Median G3 (IQR)",
                "H",
                "df",
                "p-value",
                "ε²",
                "",
            ],
            "Value": [
                f"{medians_kw[0]:.3f} ({iqr_kw[0]})",
                f"{medians_kw[1]:.3f} ({iqr_kw[1]})",
                f"{medians_kw[2]:.3f} ({iqr_kw[2]})",
                f"{H:.3f}",
                "2",
                f"{p:.5f}",
                f"{eps_sq:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i, (g, name) in enumerate(
            zip([g1, g2, g3], ["Group 1", "Group 2", "Group 3"])
        ):
            fig2.add_trace(go.Box(y=g, name=name, boxmean="sd"))
            jitter_x = np.random.normal(i + 1, 0.06, len(g))
            fig2.add_trace(
                go.Scatter(
                    x=jitter_x,
                    y=g,
                    mode="markers",
                    showlegend=False,
                    marker=dict(color="rgba(0, 123, 255, 0.4)", size=4),
                )
            )

        fig2.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Group",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Friedman Test":

        from scipy.stats import friedmanchisquare

        st.subheader("Interactive Friedman Test")

        # =========================
        # CONTROLS
        # =========================

        trend = st.slider(
            "Repeated Trend",
            -5.0,
            5.0,
            1.0,
            0.1,
        )

        noise = st.slider(
            "Noise",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        subjects = st.slider(
            "Subjects",
            5,
            100,
            20,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        t1 = np.random.exponential(1, subjects)

        t2 = t1 + trend + np.random.normal(0, noise, subjects)

        t3 = t2 + trend + np.random.normal(0, noise, subjects)

        stat, p = friedmanchisquare(t1, t2, t3)

        # =========================
        # STATS
        # =========================

        st.latex(rf"\chi^2 = {stat:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        for i in range(subjects):

            fig.add_trace(
                go.Scatter(
                    x=["T1", "T2", "T3"],
                    y=[t1[i], t2[i], t3[i]],
                    mode="lines+markers",
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # DETAILED STATISTICS TABLE
        # =========================

        st.divider()
        st.subheader("Detailed Results")

        medians_f = [np.median(t1), np.median(t2), np.median(t3)]
        kendall_w = stat / (subjects * (3 - 1))

        results_data = {
            "Metric": [
                "Median T1",
                "Median T2",
                "Median T3",
                "χ²",
                "df",
                "p-value",
                "Kendall's W",
                "",
            ],
            "Value": [
                f"{medians_f[0]:.3f}",
                f"{medians_f[1]:.3f}",
                f"{medians_f[2]:.3f}",
                f"{stat:.3f}",
                "2",
                f"{p:.5f}",
                f"{kendall_w:.4f}",
                "",
            ],
        }
        st.table(pd.DataFrame(results_data))

        # =========================
        # ENHANCED CHART
        # =========================

        fig2 = go.Figure()

        for i in range(subjects):
            fig2.add_trace(
                go.Scatter(
                    x=["T1", "T2", "T3"],
                    y=[t1[i], t2[i], t3[i]],
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
                    marker=dict(size=3),
                )
            )

        fig2.add_trace(
            go.Scatter(
                x=["T1", "T2", "T3"],
                y=medians_f,
                mode="lines+markers",
                name="Median trend",
                line=dict(color="red", width=3),
                marker=dict(color="red", size=10),
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Time Point",
            yaxis_title="Value",
        )

        st.plotly_chart(fig2, use_container_width=True)

    elif test_name == "Permutation MANOVA or Non-Parametric MANOVA":

        st.subheader("Interactive Permutation MANOVA")

        # =========================
        # CONTROLS
        # =========================

        separation = st.slider(
            "Cluster Separation",
            0.0,
            10.0,
            2.0,
            0.1,
        )

        dispersion = st.slider(
            "Cluster Dispersion",
            0.1,
            5.0,
            1.0,
            0.1,
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        n = 150

        g1 = np.random.multivariate_normal(
            [0, 0],
            np.eye(2) * dispersion,
            n,
        )

        g2 = np.random.multivariate_normal(
            [separation, separation],
            np.eye(2) * dispersion,
            n,
        )

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=g1[:, 0],
                y=g1[:, 1],
                mode="markers",
                name="Group 1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=g2[:, 0],
                y=g2[:, 1],
                mode="markers",
                name="Group 2",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Detailed Results")

        n_pm = len(g1)
        y_all_pm = np.vstack([g1, g2])
        grand_mean_pm = np.mean(y_all_pm, axis=0)

        SSt_pm = np.sum((y_all_pm - grand_mean_pm) ** 2)
        mean1_pm = np.mean(g1, axis=0)
        mean2_pm = np.mean(g2, axis=0)
        SSb_pm = n_pm * np.sum((mean1_pm - grand_mean_pm) ** 2) + n_pm * np.sum(
            (mean2_pm - grand_mean_pm) ** 2
        )
        SSw_pm = SSt_pm - SSb_pm

        k_pm = 2
        N_pm = 2 * n_pm
        pseudo_F = (SSb_pm / (k_pm - 1)) / (SSw_pm / (N_pm - k_pm))
        R2_pm = SSb_pm / SSt_pm

        n_perms = 199
        pseudo_Fs = np.zeros(n_perms)
        combined = y_all_pm.copy()
        for perm in range(n_perms):
            np.random.shuffle(combined)
            perm_g1 = combined[:n_pm]
            perm_g2 = combined[n_pm:]
            perm_mean1 = np.mean(perm_g1, axis=0)
            perm_mean2 = np.mean(perm_g2, axis=0)
            perm_grand = np.mean(combined, axis=0)
            perm_SSb = n_pm * np.sum((perm_mean1 - perm_grand) ** 2) + n_pm * np.sum(
                (perm_mean2 - perm_grand) ** 2
            )
            perm_SSw = np.sum((combined - perm_grand) ** 2) - perm_SSb
            pseudo_Fs[perm] = (perm_SSb / (k_pm - 1)) / (perm_SSw / (N_pm - k_pm))

        p_pm = (np.sum(pseudo_Fs >= pseudo_F) + 1) / (n_perms + 1)

        results_data = {
            "Metric": ["Pseudo-F", "R²", "Permutations", "p-value"],
            "Value": [
                f"{pseudo_F:.3f}",
                f"{R2_pm:.4f}",
                f"{n_perms + 1}",
                f"{p_pm:.5f}",
            ],
        }
        st.table(pd.DataFrame(results_data))

        fig2 = go.Figure()
        for idx, (g, name, color) in enumerate(
            zip([g1, g2], ["Group 1", "Group 2"], ["blue", "red"])
        ):
            fig2.add_trace(
                go.Scatter(
                    x=g[:, 0],
                    y=g[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=6, opacity=0.5),
                )
            )
            mean_pos = np.mean(g, axis=0)
            cov_pos = np.cov(g, rowvar=False)
            theta = np.linspace(0, 2 * np.pi, 100)
            eigvals, eigvecs = np.linalg.eigh(cov_pos)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            ellipse = (
                np.column_stack([np.cos(theta), np.sin(theta)])
                @ (np.diag(np.sqrt(eigvals) * 2))
                @ eigvecs.T
                + mean_pos
            )
            fig2.add_trace(
                go.Scatter(
                    x=ellipse[:, 0],
                    y=ellipse[:, 1],
                    mode="lines",
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dash"),
                )
            )
        fig2.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation and Association Tests

    elif test_name == "Pearson Correlation":
        st.subheader("Interactive Pearson Correlation")

        # =========================
        # CONTROLS
        # =========================

        col1, col2 = st.columns(2)

        with col1:
            r = st.slider("Correlation Coefficient (r)", -1.0, 1.0, 0.5, 0.01)

        with col2:
            n = st.slider("Sample Size (n)", 3, 100, 30)

        # =========================
        # LATEX
        # =========================

        st.latex(rf"""
            r = {r:.2f}
            """)

        # =========================
        # PLOTLY FIGURE
        # =========================

        x = np.random.normal(size=n)
        y = r * x + np.sqrt(1 - r**2) * np.random.normal(size=n)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data Points",
            )
        )

        fig.update_layout(
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="X",
            yaxis_title="Y",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Spearman Rank Correlation":

        from scipy.stats import spearmanr

        st.subheader("Interactive Spearman Rank Correlation")

        # =========================
        # CONTROLS
        # =========================
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
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        x = np.linspace(0, 10, 300)

        direction_multiplier = 1 if direction == "Positive" else -1

        y = direction_multiplier * (x**curve_strength) + np.random.normal(0, noise, 300)

        rho, _ = spearmanr(x, y)

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"\rho = {rho:.3f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Ranked Data",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Ranked X",
            yaxis_title="Ranked Y",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Kendall's Tau-b":

        from scipy.stats import kendalltau

        st.subheader("Interactive Kendall's Tau-b")

        # =========================
        # CONTROLS
        # =========================

        strength = st.slider("Association Strength", 0.0, 1.0, 0.5, 0.05)

        n = st.slider("Sample Size", 10, 200, 60)

        noise = st.slider("Noise", 0.1, 5.0, 1.0, 0.1)

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        x = np.random.normal(0, 1, n)
        y = strength * x + np.random.normal(0, noise, n)

        tau, p = kendalltau(x, y)

        # =========================
        # STATS
        # =========================

        st.latex(rf"\tau_b = {tau:.3f}")
        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=y, mode="markers"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="X",
            yaxis_title="Y",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Chi-Square Test of Independence":

        from scipy.stats import chi2_contingency

        st.subheader("Interactive Chi-Square Test of Independence")

        # =========================
        # CONTROLS
        # =========================

        a = st.slider("Cell A", 1, 100, 30)
        b = st.slider("Cell B", 1, 100, 20)
        c = st.slider("Cell C", 1, 100, 10)
        d = st.slider("Cell D", 1, 100, 40)

        # =========================
        # TABLE
        # =========================

        table = np.array([[a, b], [c, d]])

        chi2, p, dof, expected = chi2_contingency(table)

        # =========================
        # STATISTICS
        # =========================

        st.latex(rf"\chi^2 = {chi2:.3f}")

        st.write(f"p-value = {p:.5f}")

        # =========================
        # HEATMAP
        # =========================

        fig = go.Figure(
            data=go.Heatmap(
                z=table,
                text=table,
                texttemplate="%{text}",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Point-Biserial Correlation":

        from scipy.stats import pointbiserialr

        st.subheader("Interactive Point-Biserial Correlation")

        # =========================
        # CONTROLS
        # =========================

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
        )

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        group = np.random.binomial(1, 0.5, 300)

        y = group * group_difference + np.random.normal(0, noise, 300)

        r, p = pointbiserialr(group, y)

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"r_{{pb}} = {r:.3f}")

        st.latex(rf"\text{{p-value}} = {p:.5f}")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Box(
                y=y[group == 0],
                name="Group 0",
            )
        )

        fig.add_trace(
            go.Box(
                y=y[group == 1],
                name="Group 1",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            yaxis_title="Continuous Variable",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Regression Tests

    elif test_name == "Logistic Regression":

        st.subheader("Interactive Logistic Regression")

        # =========================
        # CONTROLS
        # =========================

        col1, col2 = st.columns(2)

        with col1:
            beta0 = st.slider("Intercept (β₀)", -10.0, 10.0, 0.0, 0.1)

        with col2:
            beta1 = st.slider("Slope (β₁)", -5.0, 5.0, 1.0, 0.1)

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

    elif test_name == "Simple Linear Regression":
        st.subheader("Interactive Simple Linear Regression")

        # =========================
        # CONTROLS
        # =========================

        col1, col2 = st.columns(2)

        with col1:
            beta0 = st.slider("Intercept (β₀)", -20.0, 20.0, 0.0, 0.1)

        with col2:
            beta1 = st.slider("Slope (β₁)", -10.0, 10.0, 1.0, 0.1)

        # =========================
        # DATA
        # =========================

        x = np.linspace(0, 10, 500)

        y = beta0 + beta1 * x

        # =========================
        # EQUATION
        # =========================

        st.latex(rf"y = {beta0:.2f} + ({beta1:.2f})x")

        # =========================
        # PLOT
        # =========================

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Regression Line",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="x",
            yaxis_title="y",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Multiple Linear Regression":
        st.subheader("Interactive Multiple Linear Regression")

        # =========================
        # CONTROLS
        # =========================

        beta0 = st.slider("β₀", -20.0, 20.0, 0.0, 0.1)

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

    elif test_name == "Multinomial Logistic Regression":
        st.subheader("Interactive Multinomial Logistic Regression")

        # =========================
        # CONTROLS
        # =========================

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

    elif test_name == "Ordinal Logistic Regression":
        st.subheader("Interactive Ordinal Logistic Regression")

        # =========================
        # CONTROLS
        # =========================

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

    elif test_name == "Poisson Regression":
        st.subheader("Interactive Poisson Regression")

        # =========================
        # CONTROLS
        # =========================

        beta0 = st.slider("β₀", -3.0, 3.0, 0.5, 0.1)

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

    elif test_name == "Cox Proportional Hazards Regression":

        st.subheader("Interactive Cox Regression")

        # =========================
        # CONTROLS
        # =========================

        hazard_ratio = st.slider("True Hazard Ratio (exp(β))", 0.5, 4.0, 2.0, 0.1)

        n_subjects = st.slider("Number of Subjects", 20, 500, 100)

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

    elif test_name == "Log-Rank Test":

        st.subheader("Interactive Log-Rank Test")

        # =========================
        # CONTROLS
        # =========================

        sep = st.slider("Survival Separation", 0.0, 5.0, 1.5, 0.1)

        n = st.slider("Subjects per Group", 10, 200, 50)

        # =========================
        # DATA
        # =========================

        np.random.seed(42)

        t1 = np.random.exponential(10, n)
        t2 = np.random.exponential(10, n) / (1 + sep * 0.3)

        censor = np.random.exponential(15, n)
        censor2 = np.random.exponential(15, n)

        obs1 = (t1 <= censor).astype(int)
        obs2 = (t2 <= censor2).astype(int)
        t1_obs = np.minimum(t1, censor)
        t2_obs = np.minimum(t2, censor2)

        fig = go.Figure()
        t_grid = np.linspace(0, max(max(t1_obs), max(t2_obs)) * 1.1, 200)

        # Exponential fit approximations
        rate1 = 1 / np.mean(t1_obs[obs1 == 1]) if obs1.sum() > 0 else 0.1
        rate2 = 1 / np.mean(t2_obs[obs2 == 1]) if obs2.sum() > 0 else 0.1

        surv1 = np.exp(-rate1 * t_grid)
        surv2 = np.exp(-rate2 * t_grid)

        fig.add_trace(go.Scatter(x=t_grid, y=surv1, mode="lines", name="Group 1"))
        fig.add_trace(go.Scatter(x=t_grid, y=surv2, mode="lines", name="Group 2"))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Time",
            yaxis_title="Survival Probability",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif test_name == "Sensitivity & Specificity Analysis":
        st.subheader("Interactive Diagnostic Accuracy Calculator")

        # =========================
        # CONTROLS
        # =========================
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

    elif test_name == "ROC Curve Analysis":
        st.subheader("Interactive ROC Curve Analysis")

        # =========================
        # CONTROLS
        # =========================
        separation = st.slider(
            "Diagnostic Power (Group Separation)", 0.0, 5.0, 1.5, 0.1
        )

        # =========================
        # DATA
        # =========================
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

    elif test_name == "Likelihood Ratio Analysis":
        st.subheader("Interactive Likelihood Ratio Analysis")

        # =========================
        # CONTROLS
        # =========================
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

    elif test_name == "Cohen's Kappa (Agreement Analysis)":
        st.subheader("Interactive Agreement Analysis (Cohen's Kappa)")

        # =========================
        # CONTROLS
        # =========================
        st.write("Enter agreement counts between two raters:")
        c1, c2 = st.columns(2)
        with c1:
            yy = st.number_input("Both say YES", min_value=0, value=40)
            yn = st.number_input(
                "Rater 1 says YES, Rater 2 says NO", min_value=0, value=10
            )
        with c2:
            ny = st.number_input(
                "Rater 1 says NO, Rater 2 says YES", min_value=0, value=5
            )
            nn = st.number_input("Both say NO", min_value=0, value=45)

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

    elif test_name == "Bland-Altman Analysis":

        st.subheader("Interactive Bland-Altman Analysis")

        # =========================
        # CONTROLS
        # =========================

        bias = st.slider("Bias (Mean Difference)", -5.0, 5.0, 0.2, 0.1)

        agreement_sd = st.slider("SD of Differences", 0.1, 5.0, 1.0, 0.1)

        n = st.slider("Sample Size", 10, 200, 50)

        # =========================
        # DATA
        # =========================

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

    else:
        st.info("Interactive widget coming soon for this test.")


# =========================
# SAMPLE SIZE / POWER CALCULATOR
# =========================


def render_power_calculator(params):
    """Render sample size estimation results for the given analysis."""

    atype = params["type"]
    alpha = params["alpha"]
    power = params["power"]
    tails = params["tails"]
    alternative = "two-sided" if tails == "Two-tailed" else "larger"

    # Determine z critical values
    if alternative == "two-sided":
        z_alpha = norm.ppf(1 - alpha / 2)
    else:
        z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)

    # Extract global adjustment parameters (with defaults for backward compat)
    dropout_rate = params.get("dropout_rate", 0.0)
    num_tests = params.get("num_tests", 1)
    mc_method = params.get("mc_method", "None")
    cost_per = params.get("cost_per", 0.0)
    recruitment_rate = params.get("recruitment_rate", 0.0)

    # Apply multiple testing correction
    alpha_raw = alpha
    if num_tests > 1 and mc_method != "None":
        if mc_method == "Bonferroni" or mc_method == "Holm-Bonferroni":
            alpha = alpha / num_tests
        elif mc_method == "Benjamini-Hochberg (FDR)":
            alpha = alpha * (num_tests + 1) / (2 * num_tests)
        if alternative == "two-sided":
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)

    # --- Computation ---
    result = None
    n_total = None
    n_total_raw = None
    n_per_group = None
    explanation = ""
    formula_latex = ""

    if atype == "one_mean":
        d = params["effect_size"]
        from statsmodels.stats.power import TTestPower

        solver = TTestPower()
        if d > 0:
            n_total = int(
                np.ceil(
                    solver.solve_power(
                        effect_size=d, alpha=alpha, power=power, alternative=alternative
                    )
                )
            )
            n_per_group = n_total
        explanation = (
            f"Required total sample size for a one-sample {tails.lower()} t-test "
            f"to detect Cohen's d = {d:.3f} with α = {alpha} and power = {power}."
        )
        formula_latex = r"n = \left( \frac{z_{\alpha/2} + z_{\beta}}{d} \right)^2"

    elif atype == "two_means":
        d = params["effect_size"]
        ratio = params["ratio"]
        from statsmodels.stats.power import TTestIndPower

        solver = TTestIndPower()
        if d > 0:
            n1 = solver.solve_power(
                effect_size=d,
                alpha=alpha,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
            n1 = int(np.ceil(n1))
            n2 = int(np.ceil(n1 * ratio))
            n_total = n1 + n2
            n_per_group = (n1, n2)
        explanation = (
            f"Required sample size per group for an independent {tails.lower()} t-test "
            f"to detect Cohen's d = {d:.3f} with α = {alpha} and power = {power}, "
            f"allocation ratio n₂/n₁ = {ratio:.2f}."
        )
        formula_latex = r"n_1 = 2 \left( \frac{z_{\alpha/2} + z_{\beta}}{d} \right)^2 \quad n_2 = r \cdot n_1"

    elif atype == "paired":
        d = params["effect_size"]
        from statsmodels.stats.power import TTestPower

        solver = TTestPower()
        if d > 0:
            n_total = int(
                np.ceil(
                    solver.solve_power(
                        effect_size=d,
                        alpha=alpha,
                        power=power,
                        alternative=alternative,
                    )
                )
            )
            n_per_group = n_total
        explanation = (
            f"Required number of pairs for a paired {tails.lower()} t-test "
            f"to detect Cohen's d_z = {d:.3f} with α = {alpha} and power = {power}."
        )
        formula_latex = r"n = \left( \frac{z_{\alpha/2} + z_{\beta}}{d_z} \right)^2"

    elif atype == "one_prop":
        p0 = params["prop_null"]
        p1 = params["prop_alt"]
        from statsmodels.stats.proportion import proportion_effectsize
        from statsmodels.stats.power import NormalIndPower

        d_eff = proportion_effectsize(p1, p0)
        solver = NormalIndPower()
        if abs(d_eff) > 0:
            n_total = int(
                np.ceil(
                    solver.solve_power(
                        effect_size=abs(d_eff),
                        alpha=alpha,
                        power=power,
                        alternative=alternative,
                    )
                )
            )
            n_per_group = n_total
        explanation = (
            f"Required sample size for a one-sample proportion test "
            f"to detect a difference from {p0} to {p1} "
            f"with α = {alpha} and power = {power}."
        )
        formula_latex = r"n = \left( \frac{z_{\alpha/2} \sqrt{p_0(1-p_0)} + z_{\beta} \sqrt{p_1(1-p_1)}}{p_1 - p_0} \right)^2"

    elif atype == "two_prop":
        p1 = params["p1"]
        p2 = params["p2"]
        ratio = params["ratio"]
        from statsmodels.stats.proportion import proportion_effectsize
        from statsmodels.stats.power import NormalIndPower

        d_eff = proportion_effectsize(p2, p1)
        solver = NormalIndPower()
        if abs(d_eff) > 0:
            n1 = solver.solve_power(
                effect_size=abs(d_eff),
                alpha=alpha,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
            n1 = int(np.ceil(n1))
            n2 = int(np.ceil(n1 * ratio))
            n_total = n1 + n2
            n_per_group = (n1, n2)
        explanation = (
            f"Required sample size per group for a two-proportion z-test "
            f"to detect a difference between {p1} and {p2} "
            f"with α = {alpha}, power = {power}, ratio = {ratio:.2f}."
        )
        formula_latex = r"n_1 = \left( \frac{z_{\alpha/2} \sqrt{2\bar{p}(1-\bar{p})} + z_{\beta} \sqrt{p_1(1-p_1) + p_2(1-p_2)}}{p_1 - p_2} \right)^2"

    elif atype == "anova":
        f_eff = params["effect_size"]
        k = params["k"]
        from statsmodels.stats.power import FTestAnovaPower

        solver = FTestAnovaPower()
        if f_eff > 0:
            n_per_g = solver.solve_power(
                effect_size=f_eff,
                alpha=alpha,
                power=power,
                k_groups=k,
            )
            n_per_g = int(np.ceil(n_per_g))
            n_total = n_per_g * k
            n_per_group = n_per_g
        explanation = (
            f"Required sample size per group for a one-way ANOVA with {k} groups "
            f"to detect Cohen's f = {f_eff:.3f} with α = {alpha} and power = {power}."
        )
        formula_latex = r"n = \frac{\text{from non-central }F\text{ distribution}}{k} \quad f = \frac{\sigma_{\text{between}}}{\sigma_{\text{within}}}"

    elif atype == "correlation":
        r_val = params["effect_size"]
        import math

        fisher_z = math.atanh(r_val)
        n_total = int(np.ceil(3 + ((z_alpha + z_beta) / fisher_z) ** 2))
        n_per_group = n_total
        explanation = (
            f"Required sample size to detect a Pearson correlation of r = {r_val:.3f} "
            f"with α = {alpha} and power = {power} ({tails.lower()}), "
            f"based on Fisher's z-transformation."
        )
        formula_latex = r"n = 3 + \left( \frac{z_{\alpha/2} + z_{\beta}}{\text{arctanh}(r)} \right)^2"

    elif atype == "regression":
        f2 = params["effect_size"]
        k = params["k"]
        if f2 > 0:
            from scipy.stats import ncf as noncentral_f, f as f_dist

            for n_candidate in range(k + 2, 10000):
                dfd = n_candidate - k - 1
                ncp = f2 * n_candidate
                f_crit = f_dist.ppf(1 - alpha, k, dfd)
                pwr_cur = 1 - noncentral_f.cdf(f_crit, k, dfd, ncp)
                if pwr_cur >= power:
                    n_total = n_candidate
                    break
            n_per_group = n_total
        explanation = (
            f"Required total sample size for multiple linear regression "
            f"with {k} predictor(s) to detect Cohen's f² = {f2:.3f} "
            f"(R² = {f2 / (1 + f2):.3f}) with α = {alpha} and power = {power}."
        )
        formula_latex = r"n = \text{from non-central }F\text{ distribution} \quad f^2 = \frac{R^2}{1-R^2}"

    elif atype == "logistic":
        k = params["k"]
        ev_rate = params["event_rate"]
        or_val = params["or"]
        p1_log = (or_val * ev_rate) / (1 - ev_rate + or_val * ev_rate)
        d_log = abs(p1_log - ev_rate)
        p_bar = (ev_rate + p1_log) / 2
        from statsmodels.stats.power import NormalIndPower

        solver = NormalIndPower()
        se = np.sqrt(2 * p_bar * (1 - p_bar))
        d_eff_log = d_log / se if se > 0 else 0
        if d_eff_log > 0:
            n1 = solver.solve_power(
                effect_size=d_eff_log,
                alpha=alpha,
                power=power,
                alternative=alternative,
            )
            n_base = int(np.ceil(n1))
            n_total = max(n_base, 10 * k)
            n_per_group = n_total
        explanation = (
            f"Required total sample size for logistic regression "
            f"with {k} predictor(s) to detect OR = {or_val:.2f} "
            f"with baseline event rate = {ev_rate:.2f}, α = {alpha}, power = {power}. "
            f"Lower bound of 10 × {k} = {10 * k} events per predictor applied."
        )
        formula_latex = r"n = \frac{(z_{\alpha/2} + z_{\beta})^2 \bar{p}(1-\bar{p})}{(p_1 - p_0)^2} \quad \text{min } 10k"

    elif atype == "chisq":
        w = params["effect_size"]
        df = params["df"]
        from statsmodels.stats.power import GofChisquarePower

        solver = GofChisquarePower()
        if w > 0:
            n_total = int(
                np.ceil(
                    solver.solve_power(
                        effect_size=w,
                        alpha=alpha,
                        power=power,
                        n_bins=df + 1,
                    )
                )
            )
            n_per_group = n_total
        explanation = (
            f"Required total sample size for a chi-square test "
            f"with {df} degree(s) of freedom to detect Cohen's w = {w:.3f} "
            f"with α = {alpha} and power = {power}."
        )
        formula_latex = r"n = \text{from non-central }\chi^2\text{ distribution} \quad w = \sqrt{\sum \frac{(p_{0i} - p_{1i})^2}{p_{0i}}}"

    elif atype == "mannwhitney":
        p_val = params["effect_size"]
        ratio = params["ratio"]
        are = params["are"]
        from statsmodels.stats.power import NormalIndPower

        solver = NormalIndPower()
        d_mw = np.sqrt(3) * (p_val - 0.5)
        if d_mw > 0:
            n1 = solver.solve_power(
                effect_size=d_mw,
                alpha=alpha,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
            n1 = int(np.ceil(n1 / are))
            n2 = int(np.ceil(n1 * ratio))
            n_total = n1 + n2
            n_per_group = (n1, n2)
        explanation = (
            f"Required sample size for Mann-Whitney / Wilcoxon test "
            f"to detect P(X>Y) = {p_val:.3f} (d ≈ {d_mw:.3f}) with ARE = {are:.3f}, "
            f"α = {alpha}, power = {power}, ratio = {ratio:.2f}."
        )
        formula_latex = r"n_{\text{nonparam}} = \frac{n_{\text{param}}}{\text{ARE}} \quad \text{ARE} \approx 0.955"

    elif atype == "logrank":
        hr = params["hr"]
        ratio = params["ratio"]
        med_ctrl = params["median_survival"]
        study_dur = params["study_duration"]
        log_hr = np.log(hr)
        num_events = (
            ((z_alpha + z_beta) ** 2) * ((ratio + 1) ** 2) / (ratio * (log_hr**2))
        )
        lambda_ctrl = np.log(2) / med_ctrl
        p_event_ctrl = 1 - np.exp(-lambda_ctrl * study_dur)
        p_event_trt = 1 - np.exp(-lambda_ctrl / hr * study_dur)
        p_event = (p_event_ctrl + ratio * p_event_trt) / (1 + ratio)
        if p_event > 0:
            n_total = int(np.ceil(num_events / p_event))
            n1 = int(np.ceil(n_total / (1 + ratio)))
            n2 = n_total - n1
            n_per_group = (n1, n2)
        explanation = (
            f"Required sample size for log-rank test to detect HR = {hr:.2f} "
            f"with median survival {med_ctrl:.0f} months, study duration {study_dur:.0f} months, "
            f"α = {alpha}, power = {power}, ratio = {ratio:.2f}. "
            f"Events needed: {int(np.ceil(num_events))}."
        )
        formula_latex = r"E = \frac{(z_{\alpha/2}+z_{\beta})^2 (r+1)^2}{r (\log HR)^2} \quad N = \frac{E}{P(\text{event})}"

    elif atype == "cox":
        hr = params["hr"]
        k = params["k"]
        sd_x = params["sd_x"]
        r2_x = params["r2_x"]
        ev_rate = params["event_rate"]
        log_hr = np.log(hr)
        var_denom = (sd_x**2) * (log_hr**2) * (1 - r2_x)
        if var_denom > 0:
            num_events = ((z_alpha + z_beta) ** 2) / var_denom
            num_events = max(num_events, 10 * k)
            n_total = int(np.ceil(num_events / ev_rate))
            n_per_group = n_total
        explanation = (
            f"Required total sample size for Cox regression with {k} predictor(s) "
            f"to detect HR = {hr:.2f} (SD = {sd_x:.1f}, R² = {r2_x:.2f}) "
            f"with α = {alpha}, power = {power}, event rate = {ev_rate:.2f}. "
            f"Events needed: {int(np.ceil(num_events))}."
        )
        formula_latex = r"E = \frac{(z_{\alpha/2}+z_{\beta})^2}{\sigma_x^2 \beta^2 (1-R^2)} \quad N = \frac{E}{P(\text{event})}"

    elif atype == "equiv":
        margin = params["margin"]
        exp_diff = params["expected_diff"]
        sd = params["sd"]
        ratio = params["ratio"]
        equiv_type = params.get("equiv_param_type", "Mean")
        if equiv_type == "Proportion":
            p1_eq = params.get("p1_eq", 0.2)
            p2_eq = params.get("p2_eq", 0.2)
            d_prop = abs(p1_eq - p2_eq)
            d_e = margin - d_prop
            if d_e > 0:
                p_bar = (p1_eq + p2_eq) / 2
                se = np.sqrt(2 * p_bar * (1 - p_bar))
                es_e = d_e / se if se > 0 else 0
                from statsmodels.stats.power import NormalIndPower
                n1 = NormalIndPower().solve_power(
                    effect_size=es_e,
                    alpha=alpha,
                    power=power,
                    ratio=ratio,
                    alternative="larger",
                )
                n1 = int(np.ceil(n1))
                n2 = int(np.ceil(n1 * ratio))
                n_total = n1 + n2
                n_per_group = (n1, n2)
            explanation = (
                f"Required sample size for non-inferiority test of proportions "
                f"with margin = {margin:.3f}, p₁ = {p1_eq:.3f}, p₂ = {p2_eq:.3f}, "
                f"expected difference = {d_prop:.3f}, α = {alpha}, power = {power}, ratio = {ratio:.2f}."
            )
            formula_latex = (
                r"n_1 = \frac{(z_{\alpha}+z_{\beta})^2 \, 2\bar{p}(1-\bar{p})}{(\delta - |p_1-p_2|)^2}"
            )
        else:
            d_e = margin - abs(exp_diff)
            if d_e > 0:
                from statsmodels.stats.power import NormalIndPower
                es_e = d_e / sd
                n1 = NormalIndPower().solve_power(
                    effect_size=es_e,
                    alpha=alpha,
                    power=power,
                    ratio=ratio,
                    alternative="larger",
                )
                n1 = int(np.ceil(n1))
                n2 = int(np.ceil(n1 * ratio))
                n_total = n1 + n2
                n_per_group = (n1, n2)
            explanation = (
                f"Required sample size for non-inferiority test of means "
                f"with margin = {margin:.2f}, expected difference = {exp_diff:.2f}, "
                f"SD = {sd:.1f}, α = {alpha}, power = {power}, ratio = {ratio:.2f}."
            )
            formula_latex = (
                r"n_1 = \frac{(z_{\alpha}+z_{\beta})^2 \sigma^2 (1+1/r)}{(\delta - |d|)^2}"
            )

    elif atype == "rm_anova":
        f_eff = params["effect_size"]
        k = params["k"]
        m = params["m"]
        rho = params["rho"]
        epsilon = params["epsilon"]
        from statsmodels.stats.power import FTestAnovaPower

        solver = FTestAnovaPower()
        if f_eff > 0:
            n_per_g = solver.solve_power(
                effect_size=f_eff,
                alpha=alpha,
                power=power,
                k_groups=k,
            )
            design_effect = (1 + (m - 1) * rho) / m
            df_adj = (m - 1) * epsilon
            n_per_g_adj = int(
                np.ceil(n_per_g * design_effect * k / (k * df_adj / (k - 1)))
            )
            n_per_g_adj = max(n_per_g_adj, int(np.ceil(n_per_g)))
            n_total = n_per_g_adj * k
            n_per_group = n_per_g_adj
        explanation = (
            f"Required sample size per group for repeated measures ANOVA "
            f"with {k} group(s), {m} measurement(s), ρ = {rho:.2f}, ε = {epsilon:.2f}, "
            f"Cohen's f = {f_eff:.3f}, α = {alpha}, power = {power}."
        )
        formula_latex = r"\text{Adjustment: } n_{\text{adj}} = n \times \frac{1+(m-1)\rho}{m} \quad \text{GG: } \varepsilon"

    elif atype == "twoway_anova":
        f_a = params["f_a"]
        f_b = params["f_b"]
        f_ab = params["f_ab"]
        rows = params["rows"]
        cols = params["cols"]
        focus = params["focus"]
        from statsmodels.stats.power import FTestAnovaPower

        solver = FTestAnovaPower()
        if focus == "Main Effect A":
            f_use = f_a
            k_use = rows
        elif focus == "Main Effect B":
            f_use = f_b
            k_use = cols
        else:
            f_use = f_ab
            k_use = rows * cols
        if f_use > 0:
            n_per_cell = solver.solve_power(
                effect_size=f_use,
                alpha=alpha,
                power=power,
                k_groups=k_use,
            )
            n_per_cell = int(np.ceil(n_per_cell))
            n_total = n_per_cell * rows * cols
            n_per_group = n_per_cell
        explanation = (
            f"Required sample size per cell for two-way ANOVA "
            f"({rows} × {cols} design), focus on {focus}, "
            f"Cohen's f = {f_use:.3f}, α = {alpha}, power = {power}. "
            f"Total N = {n_total} across {rows * cols} cells."
        )
        formula_latex = r"n_{\text{per cell}} = \text{from non-central }F \quad N = n \times r \times c"

    elif atype == "roc_auc":
        auc = params["auc"]
        null_auc = params["null_auc"]
        ratio = params["ratio"]
        v_auc = auc * (1 - auc) + (ratio - 1) * (auc / (2 - auc) - auc**2) / (1 + ratio)
        delta_auc = auc - null_auc
        if delta_auc > 0:
            n_cases = ((z_alpha + z_beta) ** 2 * v_auc) / (delta_auc**2)
            n_cases = int(np.ceil(n_cases))
            n_controls = int(np.ceil(n_cases * ratio))
            n_total = n_cases + n_controls
            n_per_group = (n_cases, n_controls)
        explanation = (
            f"Required sample size for ROC/AUC analysis to detect AUC = {auc:.3f} "
            f"(null = {null_auc:.1f}) with case:control ratio {ratio:.2f}, "
            f"α = {alpha}, power = {power}."
        )
        formula_latex = r"n_{\text{cases}} = \frac{(z_{\alpha/2}+z_{\beta})^2 V(AUC)}{(AUC - 0.5)^2}"

    elif atype == "kappa":
        kappa = params["kappa"]
        null_kappa = params["null_kappa"]
        raters = params["raters"]
        cats = params["categories"]
        delta_k = kappa - null_kappa
        if delta_k > 0:
            n_total = int(
                np.ceil(
                    ((z_alpha + z_beta) ** 2 * null_kappa * (1 - null_kappa))
                    / (delta_k**2)
                )
            )
            n_total = max(n_total, raters * cats * 5)
            n_per_group = n_total
        explanation = (
            f"Required sample size for Cohen's Kappa with {raters} rater(s), {cats} category(ies), "
            f"expected κ = {kappa:.3f}, null κ = {null_kappa:.2f}, "
            f"α = {alpha}, power = {power}."
        )
        formula_latex = r"n = \frac{(z_{\alpha/2}+z_{\beta})^2 \kappa_0(1-\kappa_0)}{(\kappa - \kappa_0)^2}"

    elif atype == "cluster_rct":
        d = params["effect_size"]
        icc = params["icc"]
        cluster_m = params["cluster_size"]
        ratio = params["ratio"]
        deff = 1 + (cluster_m - 1) * icc
        from statsmodels.stats.power import TTestIndPower

        solver = TTestIndPower()
        if d > 0:
            n1_ind = solver.solve_power(
                effect_size=d,
                alpha=alpha,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
            n1_ind = int(np.ceil(n1_ind))
            n1_clust = int(np.ceil(n1_ind * deff))
            n1_clust = int(np.ceil(n1_clust / cluster_m)) * cluster_m
            n2_clust = int(np.ceil(n1_clust * ratio))
            n_total = n1_clust + n2_clust
            num_clusters_1 = int(np.ceil(n1_clust / cluster_m))
            num_clusters_2 = int(np.ceil(n2_clust / cluster_m))
            n_per_group = (n1_clust, n2_clust)
        explanation = (
            f"Required sample size for cluster RCT with ICC = {icc:.3f}, "
            f"cluster size m = {cluster_m}, DEFF = {deff:.2f}, "
            f"Cohen's d = {d:.3f}, α = {alpha}, power = {power}, ratio = {ratio:.2f}. "
            f"Clusters needed: {num_clusters_1} in group 1, {num_clusters_2} in group 2."
        )
        formula_latex = (
            r"N_{\text{clust}} = N_{\text{ind}} \times [1+(m-1)\rho] \quad \text{DEFF}"
        )

    elif atype == "precision":
        half_width = params["half_width"]
        conf_level = params["conf_level"]
        conf_alpha = 1 - conf_level / 100
        param_type = params["param_type"]
        if param_type == "Mean":
            sd = params["sd"]
            if sd > 0 and half_width > 0:
                from scipy.stats import t as t_dist
                n_total = int(np.ceil((norm.ppf(1 - conf_alpha / 2) * sd / half_width) ** 2))
                n_total = max(n_total, 3)
                for _ in range(20):
                    t_val = t_dist.ppf(1 - conf_alpha / 2, df=n_total - 1)
                    n_next = int(np.ceil((t_val * sd / half_width) ** 2))
                    n_next = max(n_next, 3)
                    if n_next == n_total:
                        break
                    n_total = n_next
            else:
                n_total = 3
        else:
            prop = params["prop"]
            if prop > 0 and half_width > 0:
                n_total = int(np.ceil((norm.ppf(1 - conf_alpha / 2) ** 2 * prop * (1 - prop)) / (half_width**2)))
            else:
                n_total = 3
        n_total = max(n_total, 3)
        n_per_group = n_total
        explanation = (
            f"Required sample size for precision-based estimation of a {param_type.lower()} "
            f"with {conf_level:.0f}% CI half-width = {half_width:.2f}. "
            f"σ = {sd:.2f}"
            if param_type == "Mean"
            else f"p = {prop:.3f}."
        )
        formula_latex = (
            r"n = \left(\frac{t_{\alpha/2, n-1} \sigma}{w}\right)^2"
            if param_type == "Mean"
            else r"n = \frac{z_{\alpha/2}^2 p(1-p)}{w^2}"
        )

    elif atype == "pilot":
        method = params["method"]
        if method == "Rule of thumb":
            n_total = params["n_per_group"]
            n_per_group = n_total
        elif method == "Precision-based":
            half_width = params["half_width"]
            conf_level = params["conf_level"]
            conf_alpha = 1 - conf_level / 100
            param_type = params["param_type"]
            if param_type == "Mean":
                sd = params["sd"]
                if sd > 0 and half_width > 0:
                    from scipy.stats import t as t_dist
                    n_total = int(np.ceil((norm.ppf(1 - conf_alpha / 2) * sd / half_width) ** 2))
                    n_total = max(n_total, 3)
                    for _ in range(20):
                        t_val = t_dist.ppf(1 - conf_alpha / 2, df=n_total - 1)
                        n_next = int(np.ceil((t_val * sd / half_width) ** 2))
                        n_next = max(n_next, 3)
                        if n_next == n_total:
                            break
                        n_total = n_next
            else:
                z_hw = norm.ppf(1 - conf_alpha / 2)
                prop = params["prop"]
                if prop > 0 and half_width > 0:
                    n_total = int(
                        np.ceil((z_hw**2 * prop * (1 - prop)) / (half_width**2))
                    )
            n_total = max(n_total, 5)
            n_per_group = n_total
        else:
            main_n = params["main_n"]
            fraction = params["fraction"]
            n_total = max(int(np.ceil(main_n * fraction)), 5)
            n_per_group = n_total
        explanation = (
            f"Pilot/feasibility study sample size using '{method}' method. "
            f"N = {n_total} per group. Recommended for estimating parameters "
            f"for a future definitive trial."
        )
        formula_latex = r"n_{\text{pilot}} = \text{rule of thumb: } 12\text{/group}"

    elif atype == "wilcoxon_sr":
        p_val = params["effect_size"]
        are = params["are"]
        d_wsr = np.sqrt(3) * (p_val - 0.5) * 2
        if d_wsr > 0:
            from statsmodels.stats.power import NormalIndPower

            n_total = int(
                np.ceil(
                    NormalIndPower().solve_power(
                        effect_size=d_wsr,
                        alpha=alpha,
                        power=power,
                        alternative=alternative,
                    )
                    / are
                )
            )
            n_per_group = n_total
        explanation = (
            f"Required sample size for a paired Wilcoxon signed-rank test ({tails.lower()}) "
            f"to detect Pr(positive diff) = {p_val:.3f} with α = {alpha}, power = {power}, ARE = {are:.3f}."
        )
        formula_latex = r"n = \frac{1}{\text{ARE}} \left( \frac{z_{\alpha/2} + z_{\beta}}{\sqrt{3} \cdot (P - 0.5)} \right)^2"

    elif atype == "kruskal":
        f_eff = params["effect_size"]
        k = params["k"]
        are_kw = params.get("are", 0.955)
        if f_eff > 0:
            from statsmodels.stats.power import FTestAnovaPower

            n_per_g = FTestAnovaPower().solve_power(
                effect_size=f_eff, alpha=alpha, power=power, k_groups=k
            )
            n_per_g = int(np.ceil(n_per_g / are_kw))
            n_total = n_per_g * k
            n_per_group = n_per_g
        explanation = (
            f"Required sample size per group for Kruskal-Wallis test with {k} groups "
            f"to detect Cohen's f = {f_eff:.3f} with α = {alpha}, power = {power}. "
            f"Inflated by {1/are_kw:.2f}× vs ANOVA (ARE = {are_kw:.3f}) to account for efficiency loss."
        )
        formula_latex = r"n_{\text{KW}} = \frac{n_{\text{ANOVA}}}{\text{ARE}} \quad f = \sqrt{\frac{\sum(\bar{R}_i - \bar{R})^2}{(N(N+1)/12)}}"

    elif atype == "friedman":
        k = params["k"]
        m = params["m"]
        w = params["w"]
        are = params["are"]
        if w > 0:
            from statsmodels.stats.power import FTestAnovaPower

            f_fr = np.sqrt(w / (1 - w))
            n_per_g = FTestAnovaPower().solve_power(
                effect_size=f_fr, alpha=alpha, power=power, k_groups=k
            )
            n_per_g = int(np.ceil(n_per_g * m / are))
            n_total = n_per_g * k
            n_per_group = n_per_g
        explanation = (
            f"Required sample size per group for Friedman test with {k} groups, {m} measurements, "
            f"Kendall's W = {w:.3f}, α = {alpha}, power = {power}."
        )
        formula_latex = (
            r"W = \frac{12 \sum R_i^2}{m^2 k(k^2-1)} \quad f = \sqrt{\frac{W}{1-W}}"
        )

    elif atype == "mcnemar":
        p_b = params["p_b"]
        p_c = params["p_c"]
        d_mc = abs(p_b - p_c)
        p_discordant = p_b + p_c
        if d_mc > 0 and p_discordant > 0:
            n_total = int(np.ceil((z_alpha + z_beta) ** 2 * p_discordant / (d_mc**2)))
            n_per_group = n_total
        explanation = (
            f"Required number of pairs for McNemar's test ({tails.lower()}) "
            f"to detect discordant proportions p_b = {p_b:.3f}, p_c = {p_c:.3f} "
            f"with α = {alpha}, power = {power}."
        )
        formula_latex = (
            r"n = \frac{(z_{\alpha/2} + z_{\beta})^2 (p_b + p_c)}{(p_b - p_c)^2}"
        )

    elif atype == "fisher":
        p1 = params["p1"]
        p2 = params["p2"]
        ratio = params["ratio"]
        are_f = params.get("are", 0.833)
        p_bar = (p1 + ratio * p2) / (1 + ratio)
        d_f = abs(p1 - p2)
        if d_f > 0 and p_bar > 0 and p_bar < 1:
            from statsmodels.stats.proportion import proportion_effectsize
            from statsmodels.stats.power import NormalIndPower

            d_eff = proportion_effectsize(p2, p1)
            n1 = NormalIndPower().solve_power(
                effect_size=abs(d_eff),
                alpha=alpha,
                power=power,
                ratio=ratio,
                alternative=alternative,
            )
            n1 = int(np.ceil(n1 / are_f))
            n2 = int(np.ceil(n1 * ratio))
            n_total = n1 + n2
            n_per_group = (n1, n2)
        explanation = (
            f"Required sample size per group for Fisher's exact test ({tails.lower()}) "
            f"to detect difference between {p1:.3f} and {p2:.3f} "
            f"with α = {alpha}, power = {power}, ratio = {ratio:.2f}. "
            f"Inflated by {1/are_f:.2f}× vs z-test (ARE = {are_f:.3f}) for exact-method efficiency."
        )
        formula_latex = r"n_{\text{Fisher}} \approx \frac{n_{\text{two-prop}}}{\text{ARE}} \quad \text{(exact conditional inference)}"

    elif atype == "manova":
        k = params["k"]
        dv = params["dv"]
        f2 = params["f2"]
        rho = params["rho"]
        manova_test = params.get("manova_test", "Pillai's Trace")
        if f2 > 0:
            u = dv
            v_num = k - 1
            v_den = 1e9
            n_total = None
            for n_try in range(k * dv + 2, 5000):
                v_den = n_try - k - dv
                if v_den <= 0:
                    continue
                s_val = min(u, v_num)
                df1 = u * v_num
                if manova_test == "Pillai's Trace":
                    df2 = s_val * (v_den - dv + 1) + 4
                elif manova_test == "Wilks' Lambda":
                    t_val = max(np.sqrt((u**2 * v_num**2 - 4) / max(u**2 + v_num**2 - 5, 1)), 1)
                    df2 = (v_den - (u - v_num + 1) / 2) * t_val - (u * v_num - 2) / 2
                elif manova_test == "Hotelling-Lawley Trace":
                    df2 = s_val * (v_den - dv - 1) + 4
                else:
                    df2 = s_val * (v_den - dv + 1) + 4
                ncp = f2 * n_try * (1 - rho)
                from scipy.stats import ncf as noncentral_f, f as f_dist

                f_crit = f_dist.ppf(1 - alpha, df1, df2)
                p_cur = 1 - noncentral_f.cdf(f_crit, df1, df2, ncp)
                if p_cur >= power:
                    n_total = n_try
                    break
            if n_total is not None:
                n_per_group = int(np.ceil(n_total / k))
                n_total = n_per_group * k
        explanation = (
            f"Required total sample size for MANOVA with {k} groups, {dv} dependent variables, "
            f"effect size f²(V) = {f2:.4f}, α = {alpha}, power = {power}, "
            f"DV correlation ρ = {rho:.2f}. Test: {manova_test}."
        )
        formula_latex = r"n \text{ from non-central } F \text{ with } df_1 = u \cdot v_{\text{num}}, \quad f^2(V) = \frac{V}{s - V}"

    elif atype == "binomial":
        p0 = params["p0"]
        p1 = params["p1"]
        if p0 != p1:
            from scipy.stats import binom

            n_total = None
            for n_try in range(3, 10000):
                if alternative == "two-sided":
                    alpha_lo = binom.ppf(alpha / 2, n_try, p0)
                    alpha_hi = binom.ppf(1 - alpha / 2, n_try, p0)
                    p_pow = binom.cdf(alpha_hi, n_try, p1) - binom.cdf(
                        alpha_lo - 1, n_try, p1
                    )
                elif p1 > p0:
                    crit = binom.ppf(1 - alpha, n_try, p0)
                    p_pow = 1 - binom.cdf(crit - 1, n_try, p1)
                else:
                    crit = binom.ppf(alpha, n_try, p0)
                    p_pow = binom.cdf(crit, n_try, p1)
                if p_pow >= power:
                    n_total = n_try
                    break
            n_per_group = n_total
        explanation = (
            f"Required sample size for binomial exact test ({tails.lower()}) "
            f"to detect π₁ = {p1:.3f} vs π₀ = {p0:.3f} "
            f"with α = {alpha}, power = {power}. Uses exact binomial CDF."
        )
        formula_latex = r"n = \min\{n: 1 - \Phi(\Phi^{-1}(1-\alpha/2) - \sqrt{n} d) \geq \text{power}\}"

    elif atype == "simulation":
        sim_test = params["sim_test"]
        n_sim = params["n_sim"]
        n_per = params["n_per"]
        n_total = n_per * 2
        n_per_group = (n_per, n_per)
        alpha_sim = alpha
        st.info(f"Running {n_sim} Monte Carlo simulations with N = {n_per} per group...")
        progress_bar = st.progress(0)
        rejects = 0
        np.random.seed(42)
        if sim_test == "Two-proportion z-test":
            p1_s = params["p1_s"]
            p2_s = params["p2_s"]
            for i in range(n_sim):
                x1 = np.random.binomial(1, p1_s, n_per)
                x2 = np.random.binomial(1, p2_s, n_per)
                from scipy.stats import chi2_contingency
                _, p_val, _, _ = chi2_contingency(pd.crosstab(
                    pd.Series(np.concatenate([x1, x2])),
                    pd.Series(["G1"] * n_per + ["G2"] * n_per),
                ))
                if p_val < alpha_sim:
                    rejects += 1
                if (i + 1) % max(1, n_sim // 20) == 0:
                    progress_bar.progress((i + 1) / n_sim)
        else:
            mu1 = params["mu1"]
            mu2 = params["mu2"]
            sd = params["sd"]
            dist = params.get("dist", "Normal")
            for i in range(n_sim):
                if dist == "Normal":
                    g1 = np.random.normal(mu1, sd, n_per)
                    g2 = np.random.normal(mu2, sd, n_per)
                elif dist == "Skewed (Exponential)":
                    g1 = np.random.exponential(sd, n_per) + mu1
                    g2 = np.random.exponential(sd, n_per) + mu2
                else:
                    g1 = np.random.uniform(mu1 - sd * 1.732, mu1 + sd * 1.732, n_per)
                    g2 = np.random.uniform(mu2 - sd * 1.732, mu2 + sd * 1.732, n_per)
                from scipy.stats import ttest_ind, mannwhitneyu
                if sim_test == "Independent t-test (pooled)":
                    _, p_val = ttest_ind(g1, g2, equal_var=True)
                elif sim_test == "Welch's t-test":
                    _, p_val = ttest_ind(g1, g2, equal_var=False)
                else:
                    _, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
                if p_val < alpha_sim:
                    rejects += 1
                if (i + 1) % max(1, n_sim // 20) == 0:
                    progress_bar.progress((i + 1) / n_sim)
        progress_bar.empty()
        empirical_power = rejects / n_sim
        st.success(f"Monte Carlo simulation complete — {rejects}/{n_sim} rejections.")
        params["_empirical_power"] = empirical_power
        explanation = (
            f"Monte Carlo simulation ({n_sim} iterations) using {sim_test} "
            f"with N = {n_per} per group. "
            f"Estimated power: {empirical_power:.1%} "
            f"(95% CI: {max(0, empirical_power - 1.96*np.sqrt(empirical_power*(1-empirical_power)/n_sim)):.1%} – "
            f"{min(1, empirical_power + 1.96*np.sqrt(empirical_power*(1-empirical_power)/n_sim)):.1%})."
        )
        formula_latex = r"\text{Power} = \frac{1}{N_{\text{sim}}} \sum_{i=1}^{N_{\text{sim}}} I(p_i < \alpha)"

    # --- Apply Attrition Adjustment ---
    n_total_raw = n_total
    if dropout_rate > 0 and n_total is not None:
        n_total = int(np.ceil(n_total / (1 - dropout_rate)))
        if isinstance(n_per_group, tuple):
            n1_adj = int(np.ceil(n_per_group[0] / (1 - dropout_rate)))
            n2_adj = int(np.ceil(n_per_group[1] / (1 - dropout_rate)))
            n_per_group = (n1_adj, n2_adj)
        elif n_per_group is not None:
            n_per_group = int(np.ceil(n_per_group / (1 - dropout_rate)))

    # --- Display Results ---

    if n_total is None:
        st.error(
            "Effect size is too small — consider a larger effect or different design."
        )
        return

    st.subheader("Sample Size Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sample Size (N)", n_total)
    with col2:
        if isinstance(n_per_group, tuple):
            st.metric("Group 1 (n₁)", n_per_group[0])
            st.metric("Group 2 (n₂)", n_per_group[1])
        else:
            st.metric("Per Group (n)", n_per_group)
    with col3:
        if atype == "simulation":
            emp_pwr = params.get("_empirical_power", 0)
            st.metric("Empirical Power", f"{emp_pwr:.1%}")
        else:
            st.metric("Power", f"{power:.0%}")
        st.metric("Alpha (α)", f"{alpha:.3f}")

    adjustments = []
    if dropout_rate > 0 and n_total_raw is not None and n_total_raw != n_total:
        adjustments.append(
            f"Raw N = {n_total_raw} (before {dropout_rate:.0%} dropout adjustment)"
        )
    if num_tests > 1:
        adjustments.append(
            f"{mc_method} α = {alpha:.4f} (original: {alpha_raw:.4f}) for {num_tests} comparisons"
        )
    if adjustments:
        st.caption(" | ".join(adjustments))

    st.info(explanation)

    if isinstance(n_per_group, tuple) and n_total is not None and n_total > 0:
        labels = [
            f"Group 1 (n₁ = {n_per_group[0]})",
            f"Group 2 (n₂ = {n_per_group[1]})",
        ]
        values = [n_per_group[0], n_per_group[1]]
        fig_pie = go.Figure(
            data=[
                go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent")
            ]
        )
        fig_pie.update_layout(
            template="plotly_dark",
            height=280,
            title="Group-Size Distribution",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    if atype == "simulation":
        st.info("Power curve and sensitivity plots use parametric formulas and are not available for Monte Carlo simulation. Change N above and re-run to test different sample sizes.")
        return

    # --- Power Curve ---
    st.subheader("Power Analysis Curve")
    max_n = max(n_total * 3, 30)
    n_range = np.arange(5, max_n + 1, 2)

    from statsmodels.stats.power import (
        TTestIndPower,
        TTestPower,
        NormalIndPower,
        FTestAnovaPower,
        GofChisquarePower,
    )

    alpha_curve_values = sorted(set([0.01, 0.05, 0.10, alpha]))
    alpha_curve_colors = {0.01: "#FF6B6B", 0.05: "#00BFFF", 0.10: "#51CF66"}
    if alpha not in alpha_curve_values:
        alpha_curve_colors[alpha] = "#FFD43B"

    fig = go.Figure()
    for a in alpha_curve_values:
        saved_alpha = alpha
        saved_z_alpha = z_alpha
        alpha = a
        if alternative == "two-sided":
            z_alpha = norm.ppf(1 - alpha / 2)
        else:
            z_alpha = norm.ppf(1 - alpha)

        power_vals = []
        for n in n_range:
            try:
                if atype == "one_mean":
                    solver = TTestPower()
                    pv = solver.solve_power(
                        effect_size=params["effect_size"],
                        nobs=n,
                        alpha=alpha,
                        alternative=alternative,
                    )
                elif atype == "two_means":
                    solver = TTestIndPower()
                    pv = solver.solve_power(
                        effect_size=params["effect_size"],
                        nobs1=n,
                        alpha=alpha,
                        ratio=params["ratio"],
                        alternative=alternative,
                    )
                elif atype == "paired":
                    solver = TTestPower()
                    pv = solver.solve_power(
                        effect_size=params["effect_size"],
                        nobs=n,
                        alpha=alpha,
                        alternative=alternative,
                    )
                elif atype == "one_prop":
                    from statsmodels.stats.proportion import proportion_effectsize

                    solver = NormalIndPower()
                    d_eff = proportion_effectsize(
                        params["prop_alt"], params["prop_null"]
                    )
                    pv = solver.solve_power(
                        effect_size=abs(d_eff),
                        nobs1=n,
                        alpha=alpha,
                        alternative=alternative,
                    )
                elif atype == "two_prop":
                    from statsmodels.stats.proportion import proportion_effectsize

                    solver = NormalIndPower()
                    d_eff = proportion_effectsize(params["p2"], params["p1"])
                    pv = solver.solve_power(
                        effect_size=abs(d_eff),
                        nobs1=n,
                        alpha=alpha,
                        ratio=params["ratio"],
                        alternative=alternative,
                    )
                elif atype == "anova":
                    solver = FTestAnovaPower()
                    pv = solver.solve_power(
                        effect_size=params["effect_size"],
                        nobs=n,
                        alpha=alpha,
                        k_groups=params["k"],
                    )
                elif atype == "correlation":
                    import math

                    fisher_z = math.atanh(params["effect_size"])
                    z_alpha_c = (
                        norm.ppf(1 - alpha / 2)
                        if alternative == "two-sided"
                        else norm.ppf(1 - alpha)
                    )
                    z_beta_c = (np.sqrt(n - 3) * fisher_z) - z_alpha_c
                    pv = norm.cdf(z_beta_c)
                elif atype == "regression":
                    from scipy.stats import ncf as noncentral_f, f as f_dist

                    k_r = params["k"]
                    dfd = n - k_r - 1
                    if dfd > 0:
                        ncp = params["effect_size"] * n
                        f_crit = f_dist.ppf(1 - alpha, k_r, dfd)
                        pv = 1 - noncentral_f.cdf(f_crit, k_r, dfd, ncp)
                    else:
                        pv = 0
                elif atype == "logistic":
                    k = params["k"]
                    ev_rate = params["event_rate"]
                    or_val = params["or"]
                    p1_log = (or_val * ev_rate) / (1 - ev_rate + or_val * ev_rate)
                    d_log = abs(p1_log - ev_rate)
                    p_bar = (ev_rate + p1_log) / 2
                    se = np.sqrt(2 * p_bar * (1 - p_bar))
                    d_eff_log = d_log / se if se > 0 else 0
                    solver = NormalIndPower()
                    pv = solver.solve_power(
                        effect_size=d_eff_log,
                        nobs1=n,
                        alpha=alpha,
                        alternative=alternative,
                    )
                elif atype == "chisq":
                    solver = GofChisquarePower()
                    pv = solver.solve_power(
                        effect_size=params["effect_size"],
                        nobs=n,
                        alpha=alpha,
                        n_bins=params["df"] + 1,
                    )
                elif atype == "mannwhitney":
                    p_val = params["effect_size"]
                    ratio = params["ratio"]
                    are = params["are"]
                    d_mw = np.sqrt(3) * (p_val - 0.5)
                    if d_mw > 0:
                        solver = NormalIndPower()
                        pv = solver.solve_power(
                            effect_size=d_mw,
                            nobs1=n * are,
                            alpha=alpha,
                            ratio=ratio,
                            alternative=alternative,
                        )
                    else:
                        pv = 0
                elif atype == "logrank":
                    hr = params["hr"]
                    ratio = params["ratio"]
                    med_ctrl = params["median_survival"]
                    study_dur = params["study_duration"]
                    log_hr = np.log(hr)
                    lambda_ctrl = np.log(2) / med_ctrl
                    p_event_ctrl = 1 - np.exp(-lambda_ctrl * study_dur)
                    p_event_trt = 1 - np.exp(-lambda_ctrl / hr * study_dur)
                    p_event = (p_event_ctrl + ratio * p_event_trt) / (1 + ratio)
                    num_events = n * p_event
                    if log_hr != 0 and num_events > 0:
                        z_val = (
                            abs(log_hr)
                            * np.sqrt(num_events * ratio / ((ratio + 1) ** 2))
                            - z_alpha
                        )
                        pv = norm.cdf(z_val)
                    else:
                        pv = 0
                elif atype == "cox":
                    hr = params["hr"]
                    sd_x = params["sd_x"]
                    r2_x = params["r2_x"]
                    ev_rate = params["event_rate"]
                    log_hr = np.log(hr)
                    num_events = n * ev_rate
                    if log_hr != 0 and num_events > 0:
                        z_val = (
                            abs(log_hr) * sd_x * np.sqrt(num_events * (1 - r2_x))
                            - z_alpha
                        )
                        pv = norm.cdf(z_val)
                    else:
                        pv = 0
                elif atype == "equiv":
                    margin = params["margin"]
                    exp_diff = params["expected_diff"]
                    sd = params["sd"]
                    ratio = params["ratio"]
                    equiv_type = params.get("equiv_param_type", "Mean")
                    if equiv_type == "Proportion":
                        p1_eq = params.get("p1_eq", 0.2)
                        p2_eq = params.get("p2_eq", 0.2)
                        d_prop = abs(p1_eq - p2_eq)
                        d_e = margin - d_prop
                        if d_e > 0:
                            p_bar = (p1_eq + p2_eq) / 2
                            se = np.sqrt(2 * p_bar * (1 - p_bar))
                            es_e = d_e / se if se > 0 else 0
                            solver = NormalIndPower()
                            pv = solver.solve_power(
                                effect_size=es_e, nobs1=n,
                                alpha=alpha, ratio=ratio, alternative="larger",
                            )
                        else:
                            pv = 0
                    else:
                        d_e = margin - abs(exp_diff)
                        if d_e > 0:
                            es_e = d_e / sd
                            solver = NormalIndPower()
                            pv = solver.solve_power(
                                effect_size=es_e, nobs1=n,
                                alpha=alpha, ratio=ratio, alternative="larger",
                            )
                        else:
                            pv = 0
                elif atype == "rm_anova":
                    f_eff = params["effect_size"]
                    k = params["k"]
                    m = params["m"]
                    rho = params["rho"]
                    epsilon = params["epsilon"]
                    if f_eff > 0:
                        solver = FTestAnovaPower()
                        design_effect = (1 + (m - 1) * rho) / m
                        df_adj = (m - 1) * epsilon
                        n_indep = max(n * df_adj / (design_effect * (k - 1)), k + 1)
                        pv = solver.solve_power(
                            effect_size=f_eff,
                            nobs=n_indep,
                            alpha=alpha,
                            k_groups=k,
                        )
                    else:
                        pv = 0
                elif atype == "twoway_anova":
                    f_a = params["f_a"]
                    f_b = params["f_b"]
                    f_ab = params["f_ab"]
                    rows = params["rows"]
                    cols = params["cols"]
                    focus = params["focus"]
                    if focus == "Main Effect A":
                        f_use = f_a
                        k_use = rows
                    elif focus == "Main Effect B":
                        f_use = f_b
                        k_use = cols
                    else:
                        f_use = f_ab
                        k_use = rows * cols
                    if f_use > 0:
                        solver = FTestAnovaPower()
                        pv = solver.solve_power(
                            effect_size=f_use,
                            nobs=n,
                            alpha=alpha,
                            k_groups=k_use,
                        )
                    else:
                        pv = 0
                elif atype == "roc_auc":
                    auc = params["auc"]
                    null_auc = params["null_auc"]
                    ratio = params["ratio"]
                    v_auc = auc * (1 - auc) + (ratio - 1) * (
                        auc / (2 - auc) - auc**2
                    ) / (1 + ratio)
                    delta_auc = auc - null_auc
                    if delta_auc > 0:
                        n_cases_eff = n / (1 + ratio)
                        z_val = delta_auc * np.sqrt(n_cases_eff / v_auc) - z_alpha
                        pv = norm.cdf(z_val)
                    else:
                        pv = 0
                elif atype == "kappa":
                    kappa = params["kappa"]
                    null_kappa = params["null_kappa"]
                    raters = params["raters"]
                    cats = params["categories"]
                    delta_k = kappa - null_kappa
                    if delta_k > 0:
                        z_val = (
                            delta_k * np.sqrt(n / (null_kappa * (1 - null_kappa)))
                            - z_alpha
                        )
                        pv = norm.cdf(z_val)
                    else:
                        pv = 0
                elif atype == "cluster_rct":
                    d = params["effect_size"]
                    icc = params["icc"]
                    cluster_m = params["cluster_size"]
                    ratio = params["ratio"]
                    deff = 1 + (cluster_m - 1) * icc
                    if d > 0:
                        n1_equiv = (n / (1 + ratio)) / deff
                        solver = TTestIndPower()
                        if n1_equiv > 1:
                            pv = solver.solve_power(
                                effect_size=d,
                                nobs1=n1_equiv,
                                alpha=alpha,
                                ratio=ratio,
                                alternative=alternative,
                            )
                        else:
                            pv = 0
                    else:
                        pv = 0
                elif atype == "precision":
                    half_width = params["half_width"]
                    conf_level = params["conf_level"]
                    conf_alpha = 1 - conf_level / 100
                    param_type = params["param_type"]
                    if param_type == "Mean":
                        sd = params["sd"]
                        if sd > 0 and half_width > 0:
                            from scipy.stats import t as t_dist
                            n_req = max((norm.ppf(1 - conf_alpha / 2) * sd / half_width) ** 2, 3)
                            for _ in range(20):
                                t_val = t_dist.ppf(1 - conf_alpha / 2, df=int(n_req) - 1)
                                n_next = (t_val * sd / half_width) ** 2
                                if abs(n_next - n_req) < 0.5:
                                    break
                                n_req = max(n_next, 3)
                        else:
                            n_req = 0
                    else:
                        prop = params["prop"]
                        z_hw = norm.ppf(1 - conf_alpha / 2)
                        if prop > 0 and half_width > 0:
                            n_req = (z_hw**2 * prop * (1 - prop)) / (half_width**2)
                        else:
                            n_req = 0
                    if n_req > 0:
                        pv = min(n / n_req, 1.0)
                    else:
                        pv = 0
                elif atype == "pilot":
                    method = params["method"]
                    if method == "Rule of thumb":
                        n_req = params["n_per_group"]
                        pv = min(n / n_req, 1.0) if n_req > 0 else 0
                    elif method == "Precision-based":
                        conf_level = params["conf_level"]
                        conf_alpha = 1 - conf_level / 100
                        param_type = params["param_type"]
                        if param_type == "Mean":
                            sd = params["sd"]
                            half_width = params["half_width"]
                            if sd > 0 and half_width > 0:
                                from scipy.stats import t as t_dist
                                n_req = max((norm.ppf(1 - conf_alpha / 2) * sd / half_width) ** 2, 3)
                                for _ in range(20):
                                    t_val = t_dist.ppf(1 - conf_alpha / 2, df=int(n_req) - 1)
                                    n_next = (t_val * sd / half_width) ** 2
                                    if abs(n_next - n_req) < 0.5:
                                        break
                                    n_req = max(n_next, 3)
                            else:
                                n_req = 0
                        else:
                            z_hw = norm.ppf(1 - conf_alpha / 2)
                            prop = params["prop"]
                            half_width = params["half_width"]
                            n_req = (
                                (z_hw**2 * prop * (1 - prop)) / (half_width**2)
                                if prop > 0 and half_width > 0
                                else 0
                            )
                        pv = min(n / n_req, 1.0) if n_req > 0 else 0
                    else:
                        main_n = params["main_n"]
                        fraction = params["fraction"]
                        n_req = max(int(np.ceil(main_n * fraction)), 5)
                        pv = min(n / n_req, 1.0)
                elif atype == "wilcoxon_sr":
                    p_val = params["effect_size"]
                    are = params["are"]
                    d_wsr = np.sqrt(3) * (p_val - 0.5) * 2
                    if d_wsr > 0:
                        solver = NormalIndPower()
                        pv = solver.solve_power(
                            effect_size=d_wsr,
                            nobs1=n * are,
                            alpha=alpha,
                            alternative=alternative,
                        )
                    else:
                        pv = 0
                elif atype == "kruskal":
                    f_eff = params["effect_size"]
                    k = params["k"]
                    are_kw = params.get("are", 0.955)
                    if f_eff > 0:
                        solver = FTestAnovaPower()
                        pv = solver.solve_power(
                            effect_size=f_eff, nobs=n * are_kw, alpha=alpha, k_groups=k
                        )
                    else:
                        pv = 0
                elif atype == "friedman":
                    w = params["w"]
                    k = params["k"]
                    m = params["m"]
                    are = params["are"]
                    f_fr = np.sqrt(w / (1 - w)) if w < 1 else 1
                    if f_fr > 0:
                        solver = FTestAnovaPower()
                        n_indep = n * are / m
                        pv = solver.solve_power(
                            effect_size=f_fr, nobs=n_indep, alpha=alpha, k_groups=k
                        )
                    else:
                        pv = 0
                elif atype == "mcnemar":
                    p_b = params["p_b"]
                    p_c = params["p_c"]
                    d_mc = abs(p_b - p_c)
                    p_discordant = p_b + p_c
                    if d_mc > 0 and p_discordant > 0:
                        z_val = d_mc * np.sqrt(n / p_discordant) - z_alpha
                        pv = norm.cdf(z_val)
                    else:
                        pv = 0
                elif atype == "fisher":
                    p1 = params["p1"]
                    p2 = params["p2"]
                    ratio = params["ratio"]
                    are_f = params.get("are", 0.833)
                    from statsmodels.stats.proportion import proportion_effectsize

                    d_eff = proportion_effectsize(p2, p1)
                    if abs(d_eff) > 0:
                        solver = NormalIndPower()
                        pv = solver.solve_power(
                            effect_size=abs(d_eff),
                            nobs1=n / (1 + ratio) * are_f,
                            alpha=alpha,
                            ratio=ratio,
                            alternative=alternative,
                        )
                    else:
                        pv = 0
                elif atype == "manova":
                    k = params["k"]
                    dv = params["dv"]
                    f2 = params["f2"]
                    rho = params["rho"]
                    manova_test = params.get("manova_test", "Pillai's Trace")
                    if f2 > 0:
                        u = dv
                        v_num = k - 1
                        v_den = n - k - dv
                        if v_den > 0:
                            s_val = min(u, v_num)
                            df1 = u * v_num
                            if manova_test == "Pillai's Trace":
                                df2 = s_val * (v_den - dv + 1) + 4
                            elif manova_test == "Wilks' Lambda":
                                t_val = max(np.sqrt((u**2 * v_num**2 - 4) / max(u**2 + v_num**2 - 5, 1)), 1)
                                df2 = (v_den - (u - v_num + 1) / 2) * t_val - (u * v_num - 2) / 2
                            elif manova_test == "Hotelling-Lawley Trace":
                                df2 = s_val * (v_den - dv - 1) + 4
                            else:
                                df2 = s_val * (v_den - dv + 1) + 4
                            ncp = f2 * n * (1 - rho)
                            from scipy.stats import ncf as noncentral_f, f as f_dist

                            f_crit = f_dist.ppf(1 - alpha, df1, df2)
                            pv = 1 - noncentral_f.cdf(f_crit, df1, df2, ncp)
                        else:
                            pv = 0
                    else:
                        pv = 0
                elif atype == "binomial":
                    p0 = params["p0"]
                    p1 = params["p1"]
                    if p0 != p1:
                        from scipy.stats import binom

                        if alternative == "two-sided":
                            alpha_lo = binom.ppf(alpha / 2, n, p0)
                            alpha_hi = binom.ppf(1 - alpha / 2, n, p0)
                            pv = binom.cdf(alpha_hi, n, p1) - binom.cdf(
                                alpha_lo - 1, n, p1
                            )
                        elif p1 > p0:
                            crit = binom.ppf(1 - alpha, n, p0)
                            pv = 1 - binom.cdf(crit - 1, n, p1)
                        else:
                            crit = binom.ppf(alpha, n, p0)
                            pv = binom.cdf(crit, n, p1)
                    else:
                        pv = 0
                else:
                    pv = 0
            except Exception:
                pv = 0
            power_vals.append(pv)

        alpha = saved_alpha
        z_alpha = saved_z_alpha

        is_selected = abs(a - saved_alpha) < 1e-10
        line_w = 4 if is_selected else 2
        line_d = "solid" if is_selected else "dash"
        fig.add_trace(
            go.Scatter(
                x=n_range,
                y=power_vals,
                mode="lines",
                name=f"α = {a:.3f}",
                line=dict(
                    color=alpha_curve_colors.get(a, "#FFD43B"),
                    width=line_w,
                    dash=line_d,
                ),
            )
        )

    target_power_val = params["power"]
    fig.add_hline(
        y=target_power_val,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Target Power ({target_power_val:.0%})",
    )
    fig.add_vline(
        x=n_total,
        line_dash="dot",
        line_color="green",
        annotation_text=f"N = {n_total}",
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Sample Size (N)",
        yaxis_title="Power (1 − β)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Distribution Plot ---
    st.subheader(":orange[Distribution Plot (Null & Alternative)]")
    st.caption(
        "Shows the sampling distributions under H₀ (null) and H₁ (alternative) with critical regions."
    )
    x_min = -4.5
    x_max = 4.5
    if n_total is not None and n_total > 0:
        sqrt_n = np.sqrt(n_total)
        if atype in ("one_mean", "paired", "wilcoxon_sr"):
            d = (
                params.get("effect_size", 0.5)
                if atype != "wilcoxon_sr"
                else np.sqrt(3) * (params["effect_size"] - 0.5) * 2
            )
            ncp = d * sqrt_n
        elif atype == "two_means":
            d = params.get("effect_size", 0.5)
            ncp = d * sqrt_n / np.sqrt(1 + 1 / params.get("ratio", 1))
        elif atype in ("one_prop", "binomial"):
            p0 = params.get("prop_null", params.get("p0", 0.5))
            p1 = params.get("prop_alt", params.get("p1", 0.7))
            d_eff = abs(p1 - p0) / np.sqrt(p0 * (1 - p0)) if p0 > 0 and p0 < 1 else 0
            ncp = d_eff * sqrt_n
        elif atype in ("two_prop", "fisher"):
            p1 = params["p1"]
            ratio = params.get("ratio", 1)
            d_eff = abs(p1 - params.get("p2", 0.5)) / np.sqrt(p1 * (1 - p1))
            ncp = d_eff * sqrt_n / np.sqrt(1 + ratio)
        elif atype in ("mcnemar",):
            p_b = params.get("p_b", 0.2)
            p_c = params.get("p_c", 0.4)
            d_mc = abs(p_b - p_c)
            p_disc = p_b + p_c
            ncp = d_mc * sqrt_n / np.sqrt(p_disc) if p_disc > 0 else 0
        elif atype in ("correlation",):
            r = params.get("effect_size", 0.3)
            ncp = np.arctanh(r) * sqrt_n
        elif atype in ("mannwhitney",):
            p = params.get("effect_size", 0.65)
            d_mw = np.sqrt(3) * (p - 0.5)
            ncp = d_mw * sqrt_n * np.sqrt(params.get("are", 1))
        elif atype in ("logistic",):
            or_val = params.get("or", 2)
            ev_rate = params.get("event_rate", 0.3)
            p1_log = (or_val * ev_rate) / (1 - ev_rate + or_val * ev_rate)
            d_log = abs(p1_log - ev_rate)
            p_bar = (ev_rate + p1_log) / 2
            se = np.sqrt(2 * p_bar * (1 - p_bar))
            d_eff = d_log / se if se > 0 else 0
            ncp = d_eff * sqrt_n
        elif atype in ("logrank",):
            hr = params.get("hr", 2)
            ncp = abs(np.log(hr)) * sqrt_n / 2
        elif atype in ("cox",):
            hr = params.get("hr", 2)
            sd_x = params.get("sd_x", 1)
            r2_x = params.get("r2_x", 0)
            ev_rate = params.get("event_rate", 0.5)
            ncp = abs(np.log(hr)) * sd_x * np.sqrt(ev_rate * (1 - r2_x)) * sqrt_n
        else:
            ncp = 2.5

        if ncp > 0:
            x_range = np.linspace(max(-4.5, -ncp - 4), max(4.5, ncp + 4), 400)
            null_pdf = norm.pdf(x_range, 0, 1)
            alt_pdf = norm.pdf(x_range, ncp, 1)
            z_crit = z_alpha
            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Scatter(
                    x=x_range,
                    y=null_pdf,
                    mode="lines",
                    name="H₀ (null)",
                    line=dict(color="#00BFFF", width=2),
                )
            )
            fig_dist.add_trace(
                go.Scatter(
                    x=x_range,
                    y=alt_pdf,
                    mode="lines",
                    name="H₁ (alternative)",
                    line=dict(color="#FF6B6B", width=2),
                )
            )
            if alternative == "two-sided":
                x_shade_left = x_range[x_range <= -z_crit]
                x_shade_right = x_range[x_range >= z_crit]
                if len(x_shade_left) > 0:
                    fig_dist.add_trace(
                        go.Scatter(
                            x=x_shade_left,
                            y=null_pdf[: len(x_shade_left)],
                            mode="lines",
                            fill="tozeroy",
                            name=f"α/2 ({alpha/2:.4f})",
                            line=dict(color="rgba(255,0,0,0.3)"),
                            fillcolor="rgba(255,0,0,0.2)",
                        )
                    )
                if len(x_shade_right) > 0:
                    fig_dist.add_trace(
                        go.Scatter(
                            x=x_shade_right,
                            y=null_pdf[len(null_pdf) - len(x_shade_right) :],
                            mode="lines",
                            fill="tozeroy",
                            name=f"α/2 ({alpha/2:.4f})",
                            line=dict(color="rgba(255,0,0,0.3)"),
                            fillcolor="rgba(255,0,0,0.2)",
                        )
                    )
                x_beta = x_range[x_range <= z_crit]
                if len(x_beta) > 0:
                    beta_pdf = alt_pdf[: len(x_beta)]
                    fig_dist.add_trace(
                        go.Scatter(
                            x=x_beta,
                            y=beta_pdf,
                            mode="lines",
                            fill="tozeroy",
                            name=f"β ({1-power:.2f})",
                            line=dict(color="rgba(255,165,0,0.3)"),
                            fillcolor="rgba(255,165,0,0.2)",
                        )
                    )
            else:
                ncp_pos = ncp > 0
                if ncp_pos:
                    x_shade = x_range[x_range >= z_crit]
                    x_beta = x_range[x_range <= z_crit]
                else:
                    x_shade = x_range[x_range <= -z_crit]
                    x_beta = x_range[x_range >= -z_crit]
                if len(x_shade) > 0:
                    null_shade = (
                        null_pdf[-len(x_shade) :]
                        if ncp_pos
                        else null_pdf[: len(x_shade)]
                    )
                    fig_dist.add_trace(
                        go.Scatter(
                            x=x_shade,
                            y=null_shade,
                            mode="lines",
                            fill="tozeroy",
                            name=f"α ({alpha:.4f})",
                            line=dict(color="rgba(255,0,0,0.3)"),
                            fillcolor="rgba(255,0,0,0.2)",
                        )
                    )
                if len(x_beta) > 0:
                    alt_shade = (
                        alt_pdf[-len(x_beta) :]
                        if not ncp_pos
                        else alt_pdf[: len(x_beta)]
                    )
                    fig_dist.add_trace(
                        go.Scatter(
                            x=x_beta,
                            y=alt_shade,
                            mode="lines",
                            fill="tozeroy",
                            name=f"β ({1-power:.2f})",
                            line=dict(color="rgba(255,165,0,0.3)"),
                            fillcolor="rgba(255,165,0,0.2)",
                        )
                    )

            fig_dist.add_vline(
                x=0,
                line_dash="dot",
                line_color="#00BFFF",
                annotation_text="H₀ center",
            )
            fig_dist.add_vline(
                x=ncp,
                line_dash="dot",
                line_color="#FF6B6B",
                annotation_text="H₁ center",
            )
            fig_dist.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Test Statistic (z)",
                yaxis_title="Density",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info(
                "Distribution plot not available for this test type with current parameters."
            )

    # --- Sensitivity Table ---
    st.subheader("Sensitivity Analysis")
    st.caption("How required sample size changes with different effect sizes.")

    effect_multipliers = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0]
    sens_data = []
    for mult in effect_multipliers:
        try:
            if atype in ("one_mean", "paired", "correlation"):
                adj_es = params["effect_size"] * mult
                if atype == "correlation":
                    import math

                    adj_es = max(min(adj_es, 0.99), 0.01)
                    fisher_z_s = math.atanh(adj_es)
                    n_s = int(np.ceil(3 + ((z_alpha + z_beta) / fisher_z_s) ** 2))
                elif atype in ("one_mean", "paired"):
                    solver = TTestPower()
                    n_s = int(
                        np.ceil(
                            solver.solve_power(
                                effect_size=adj_es,
                                alpha=alpha,
                                power=power,
                                alternative=alternative,
                            )
                        )
                    )
            elif atype == "two_means":
                adj_es = params["effect_size"] * mult
                solver = TTestIndPower()
                n1_s = solver.solve_power(
                    effect_size=adj_es,
                    alpha=alpha,
                    power=power,
                    ratio=params["ratio"],
                    alternative=alternative,
                )
                n1_s = int(np.ceil(n1_s))
                n_s = n1_s + int(np.ceil(n1_s * params["ratio"]))
            elif atype == "one_prop":
                adj_es = (
                    params["prop_null"]
                    + (params["prop_alt"] - params["prop_null"]) * mult
                )
                adj_es = max(min(adj_es, 0.99), 0.01)
                from statsmodels.stats.proportion import proportion_effectsize

                d_eff_s = proportion_effectsize(adj_es, params["prop_null"])
                solver = NormalIndPower()
                n_s = int(
                    np.ceil(
                        solver.solve_power(
                            effect_size=abs(d_eff_s),
                            alpha=alpha,
                            power=power,
                            alternative=alternative,
                        )
                    )
                )
            elif atype == "two_prop":
                adj_p2 = params["p1"] + (params["p2"] - params["p1"]) * mult
                adj_p2 = max(min(adj_p2, 0.99), 0.01)
                adj_es = abs(adj_p2 - params["p1"])
                from statsmodels.stats.proportion import proportion_effectsize

                d_eff_s = proportion_effectsize(adj_p2, params["p1"])
                solver = NormalIndPower()
                n1_s = solver.solve_power(
                    effect_size=abs(d_eff_s),
                    alpha=alpha,
                    power=power,
                    ratio=params["ratio"],
                    alternative=alternative,
                )
                n1_s = int(np.ceil(n1_s))
                n_s = n1_s + int(np.ceil(n1_s * params["ratio"]))
            elif atype == "logistic":
                adj_es = max(params["or"] ** mult, 1.01)
                ev_rate = params["event_rate"]
                k_log = params["k"]
                p1_log = (adj_es * ev_rate) / (1 - ev_rate + adj_es * ev_rate)
                d_log = abs(p1_log - ev_rate)
                p_bar = (ev_rate + p1_log) / 2
                se = np.sqrt(2 * p_bar * (1 - p_bar))
                d_eff_log = d_log / se if se > 0 else 0
                solver = NormalIndPower()
                n_base = int(
                    np.ceil(
                        solver.solve_power(
                            effect_size=d_eff_log,
                            alpha=alpha,
                            power=power,
                            alternative=alternative,
                        )
                    )
                )
                n_s = max(n_base, 10 * k_log)
            elif atype == "anova":
                adj_es = params["effect_size"] * mult
                solver = FTestAnovaPower()
                n_s = (
                    int(
                        np.ceil(
                            solver.solve_power(
                                effect_size=adj_es,
                                alpha=alpha,
                                power=power,
                                k_groups=params["k"],
                            )
                        )
                    )
                    * params["k"]
                )
            elif atype == "regression":
                adj_es = params["effect_size"] * mult
                k_r = params["k"]
                from scipy.stats import ncf as noncentral_f, f as f_dist

                n_s = None
                for nc in range(k_r + 2, 10000):
                    dfd = nc - k_r - 1
                    ncp = adj_es * nc
                    f_crit = f_dist.ppf(1 - alpha, k_r, dfd)
                    p_cur = 1 - noncentral_f.cdf(f_crit, k_r, dfd, ncp)
                    if p_cur >= power:
                        n_s = nc
                        break
            elif atype == "chisq":
                adj_es = params["effect_size"] * mult
                solver = GofChisquarePower()
                n_s = int(
                    np.ceil(
                        solver.solve_power(
                            effect_size=adj_es,
                            alpha=alpha,
                            power=power,
                            n_bins=params["df"] + 1,
                        )
                    )
                )
            elif atype == "mannwhitney":
                adj_p = params["effect_size"]
                adj_p = max(min(adj_p * mult, 0.99), 0.01)
                adj_es = np.sqrt(3) * (adj_p - 0.5)
                are = params["are"]
                ratio = params["ratio"]
                solver = NormalIndPower()
                n1_s = solver.solve_power(
                    effect_size=adj_es,
                    alpha=alpha,
                    power=power,
                    ratio=ratio,
                    alternative=alternative,
                )
                n1_s = int(np.ceil(n1_s / are))
                n_s = n1_s + int(np.ceil(n1_s * ratio))
            elif atype == "logrank":
                adj_es = max(params["hr"] ** mult, 1.001)
                ratio = params["ratio"]
                med_ctrl = params["median_survival"]
                study_dur = params["study_duration"]
                log_hr = np.log(adj_es)
                num_events = (
                    ((z_alpha + z_beta) ** 2)
                    * ((ratio + 1) ** 2)
                    / (ratio * (log_hr**2))
                )
                lambda_ctrl = np.log(2) / med_ctrl
                p_event_ctrl = 1 - np.exp(-lambda_ctrl * study_dur)
                p_event_trt = 1 - np.exp(-lambda_ctrl / adj_es * study_dur)
                p_event = (p_event_ctrl + ratio * p_event_trt) / (1 + ratio)
                if p_event > 0:
                    n_s = int(np.ceil(num_events / p_event))
                else:
                    n_s = None
            elif atype == "cox":
                adj_es = max(params["hr"] ** mult, 1.001)
                k = params["k"]
                sd_x = params["sd_x"]
                r2_x = params["r2_x"]
                ev_rate = params["event_rate"]
                log_hr = np.log(adj_es)
                var_denom = (sd_x**2) * (log_hr**2) * (1 - r2_x)
                if var_denom > 0:
                    num_events = ((z_alpha + z_beta) ** 2) / var_denom
                    num_events = max(num_events, 10 * k)
                    n_s = int(np.ceil(num_events / ev_rate))
                else:
                    n_s = None
            elif atype == "equiv":
                adj_margin = params["margin"] * mult
                exp_diff = params["expected_diff"]
                sd = params["sd"]
                ratio = params["ratio"]
                equiv_type = params.get("equiv_param_type", "Mean")
                if equiv_type == "Proportion":
                    p1_eq = params.get("p1_eq", 0.2)
                    p2_eq = params.get("p2_eq", 0.2)
                    d_prop = abs(p1_eq - p2_eq)
                    d_e = adj_margin - d_prop
                    if d_e > 0:
                        p_bar = (p1_eq + p2_eq) / 2
                        se = np.sqrt(2 * p_bar * (1 - p_bar))
                        es_e = d_e / se if se > 0 else 0
                        adj_es = es_e
                        solver = NormalIndPower()
                        n1_s = solver.solve_power(
                            effect_size=es_e, alpha=alpha, power=power,
                            ratio=ratio, alternative="larger",
                        )
                        n1_s = int(np.ceil(n1_s))
                        n_s = n1_s + int(np.ceil(n1_s * ratio))
                    else:
                        n_s = None
                else:
                    d_e = adj_margin - abs(exp_diff)
                    adj_es = d_e / sd if d_e > 0 else 0
                    if d_e > 0:
                        es_e = d_e / sd
                        solver = NormalIndPower()
                        n1_s = solver.solve_power(
                            effect_size=es_e, alpha=alpha, power=power,
                            ratio=ratio, alternative="larger",
                        )
                        n1_s = int(np.ceil(n1_s))
                        n_s = n1_s + int(np.ceil(n1_s * ratio))
                    else:
                        n_s = None
            elif atype == "rm_anova":
                adj_es = params["effect_size"] * mult
                k = params["k"]
                m = params["m"]
                rho = params["rho"]
                epsilon = params["epsilon"]
                solver = FTestAnovaPower()
                n_per_g = solver.solve_power(
                    effect_size=adj_es,
                    alpha=alpha,
                    power=power,
                    k_groups=k,
                )
                design_effect = (1 + (m - 1) * rho) / m
                df_adj = (m - 1) * epsilon
                n_per_g_adj = int(
                    np.ceil(n_per_g * design_effect * k / (k * df_adj / (k - 1)))
                )
                n_per_g_adj = max(n_per_g_adj, int(np.ceil(n_per_g)))
                n_s = n_per_g_adj * k
            elif atype == "twoway_anova":
                adj_f_a = params["f_a"] * mult
                adj_f_b = params["f_b"] * mult
                adj_f_ab = params["f_ab"] * mult
                rows = params["rows"]
                cols = params["cols"]
                focus = params["focus"]
                if focus == "Main Effect A":
                    adj_es = adj_f_a
                    k_use = rows
                elif focus == "Main Effect B":
                    adj_es = adj_f_b
                    k_use = cols
                else:
                    adj_es = adj_f_ab
                    k_use = rows * cols
                solver = FTestAnovaPower()
                n_per_cell = solver.solve_power(
                    effect_size=adj_es,
                    alpha=alpha,
                    power=power,
                    k_groups=k_use,
                )
                n_per_cell = int(np.ceil(n_per_cell))
                n_s = n_per_cell * rows * cols
            elif atype == "roc_auc":
                auc = params["auc"]
                adj_auc = 0.5 + (auc - 0.5) * mult
                adj_auc = max(min(adj_auc, 0.99), 0.51)
                null_auc = params["null_auc"]
                ratio = params["ratio"]
                v_auc = adj_auc * (1 - adj_auc) + (ratio - 1) * (
                    adj_auc / (2 - adj_auc) - adj_auc**2
                ) / (1 + ratio)
                delta_auc = adj_auc - null_auc
                adj_es = delta_auc
                if delta_auc > 0:
                    n_cases = ((z_alpha + z_beta) ** 2 * v_auc) / (delta_auc**2)
                    n_cases = int(np.ceil(n_cases))
                    n_controls = int(np.ceil(n_cases * ratio))
                    n_s = n_cases + n_controls
                else:
                    n_s = None
            elif atype == "kappa":
                adj_kappa = params["kappa"]
                adj_kappa = 0.5 + (adj_kappa - 0.5) * mult
                adj_kappa = max(min(adj_kappa, 0.99), 0.01)
                null_kappa = params["null_kappa"]
                adj_kappa = max(adj_kappa, null_kappa + 0.01)
                delta_k = adj_kappa - null_kappa
                raters = params["raters"]
                cats = params["categories"]
                adj_es = delta_k
                if delta_k > 0:
                    n_s = int(
                        np.ceil(
                            ((z_alpha + z_beta) ** 2 * null_kappa * (1 - null_kappa))
                            / (delta_k**2)
                        )
                    )
                    n_s = max(n_s, raters * cats * 5)
                else:
                    n_s = None
            elif atype == "cluster_rct":
                adj_es = params["effect_size"] * mult
                icc = params["icc"]
                cluster_m = params["cluster_size"]
                ratio = params["ratio"]
                deff = 1 + (cluster_m - 1) * icc
                solver = TTestIndPower()
                n1_ind = solver.solve_power(
                    effect_size=adj_es,
                    alpha=alpha,
                    power=power,
                    ratio=ratio,
                    alternative=alternative,
                )
                n1_ind = int(np.ceil(n1_ind))
                n1_clust = int(np.ceil(n1_ind * deff))
                n1_clust = int(np.ceil(n1_clust / cluster_m)) * cluster_m
                n2_clust = int(np.ceil(n1_clust * ratio))
                n_s = n1_clust + n2_clust
            elif atype == "precision":
                adj_hw = params["half_width"] * mult
                conf_level = params["conf_level"]
                conf_alpha = 1 - conf_level / 100
                param_type = params["param_type"]
                adj_es = adj_hw
                if param_type == "Mean":
                    sd = params["sd"]
                    if sd > 0 and adj_hw > 0:
                        from scipy.stats import t as t_dist
                        n_s = int(np.ceil((norm.ppf(1 - conf_alpha / 2) * sd / adj_hw) ** 2))
                        n_s = max(n_s, 3)
                        for _ in range(20):
                            t_val = t_dist.ppf(1 - conf_alpha / 2, df=n_s - 1)
                            n_next = int(np.ceil((t_val * sd / adj_hw) ** 2))
                            n_next = max(n_next, 3)
                            if n_next == n_s:
                                break
                            n_s = n_next
                    else:
                        n_s = None
                else:
                    prop = params["prop"]
                    z_hw = norm.ppf(1 - conf_alpha / 2)
                    if prop > 0 and adj_hw > 0:
                        n_s = int(np.ceil((z_hw**2 * prop * (1 - prop)) / (adj_hw**2)))
                    else:
                        n_s = None
                if n_s is not None:
                    n_s = max(n_s, 3)
            elif atype == "pilot":
                method = params["method"]
                adj_es = mult
                if method == "Rule of thumb":
                    n_s = int(np.ceil(params["n_per_group"] / max(mult, 0.1)))
                elif method == "Precision-based":
                    conf_level = params["conf_level"]
                    conf_alpha = 1 - conf_level / 100
                    param_type = params["param_type"]
                    adj_hw = params["half_width"] * mult
                    if param_type == "Mean":
                        sd = params["sd"]
                        if sd > 0 and adj_hw > 0:
                            from scipy.stats import t as t_dist
                            n_s = int(np.ceil((norm.ppf(1 - conf_alpha / 2) * sd / adj_hw) ** 2))
                            n_s = max(n_s, 3)
                            for _ in range(20):
                                t_val = t_dist.ppf(1 - conf_alpha / 2, df=n_s - 1)
                                n_next = int(np.ceil((t_val * sd / adj_hw) ** 2))
                                n_next = max(n_next, 3)
                                if n_next == n_s:
                                    break
                                n_s = n_next
                        else:
                            n_s = None
                    else:
                        z_hw = norm.ppf(1 - conf_alpha / 2)
                        prop = params["prop"]
                        if prop > 0 and adj_hw > 0:
                            n_s = int(
                                np.ceil((z_hw**2 * prop * (1 - prop)) / (adj_hw**2))
                            )
                        else:
                            n_s = None
                    if n_s is not None:
                        n_s = max(n_s, 5)
                else:
                    main_n = params["main_n"]
                    fraction = params["fraction"]
                    n_s = max(int(np.ceil(main_n * fraction / max(mult, 0.1))), 5)
            elif atype == "wilcoxon_sr":
                p_val = params["effect_size"]
                are = params["are"]
                d_wsr = np.sqrt(3) * (p_val - 0.5) * 2
                adj_es = d_wsr * max(mult, 0.1)
                if d_wsr > 0:
                    solver = NormalIndPower()
                    n_s = int(
                        np.ceil(
                            solver.solve_power(
                                effect_size=adj_es / are,
                                alpha=alpha,
                                power=power,
                                alternative=alternative,
                            )
                            / are
                        )
                    )
                else:
                    n_s = None
            elif atype == "kruskal":
                adj_es = params["effect_size"] * mult
                k = params["k"]
                are_kw = params.get("are", 0.955)
                if adj_es > 0:
                    solver = FTestAnovaPower()
                    n_per_g = solver.solve_power(
                        effect_size=adj_es, alpha=alpha, power=power, k_groups=k
                    )
                    n_per_g = int(np.ceil(n_per_g / are_kw))
                    n_s = n_per_g * k
                else:
                    n_s = None
            elif atype == "friedman":
                w = params["w"]
                k = params["k"]
                m = params["m"]
                are = params["are"]
                adj_w = max(min(w * mult, 0.99), 0.01)
                adj_es = np.sqrt(adj_w / (1 - adj_w))
                if adj_es > 0:
                    solver = FTestAnovaPower()
                    n_per_g = solver.solve_power(
                        effect_size=adj_es, alpha=alpha, power=power, k_groups=k
                    )
                    n_per_g = int(np.ceil(n_per_g * m / are))
                    n_s = n_per_g * k
                else:
                    n_s = None
            elif atype == "mcnemar":
                p_b = params["p_b"]
                p_c = params["p_c"]
                p_b_adj = max(min(p_b * mult, 0.99), 0.01)
                p_c_adj = p_c * mult
                p_c_adj = max(min(p_c_adj, 0.99), 0.01)
                d_mc = abs(p_b_adj - p_c_adj)
                p_discordant = p_b_adj + p_c_adj
                adj_es = d_mc
                if d_mc > 0 and p_discordant > 0:
                    n_s = int(
                        np.ceil((z_alpha + z_beta) ** 2 * p_discordant / (d_mc**2))
                    )
                    n_s = max(n_s, 5)
                else:
                    n_s = None
            elif atype == "fisher":
                p1 = params["p1"]
                p2 = params["p2"]
                ratio = params["ratio"]
                are_f = params.get("are", 0.833)
                adj_p2 = p1 + (p2 - p1) * mult
                adj_p2 = max(min(adj_p2, 0.99), 0.01)
                adj_es = abs(adj_p2 - p1)
                from statsmodels.stats.proportion import proportion_effectsize

                d_eff = proportion_effectsize(adj_p2, p1)
                if abs(d_eff) > 0:
                    solver = NormalIndPower()
                    n1 = solver.solve_power(
                        effect_size=abs(d_eff),
                        alpha=alpha,
                        power=power,
                        ratio=ratio,
                        alternative=alternative,
                    )
                    n1 = int(np.ceil(n1 / are_f))
                    n_s = n1 + int(np.ceil(n1 * ratio))
                else:
                    n_s = None
            elif atype == "manova":
                k = params["k"]
                dv = params["dv"]
                f2 = params["f2"]
                rho = params["rho"]
                adj_f2 = f2 * mult
                adj_es = adj_f2
                manova_test = params.get("manova_test", "Pillai's Trace")
                if adj_f2 > 0:
                    u = dv
                    v_num = k - 1
                    n_s = None
                    for n_try in range(k * dv + 2, 5000):
                        v_den = n_try - k - dv
                        if v_den <= 0:
                            continue
                        s_val = min(u, v_num)
                        df1 = u * v_num
                        if manova_test == "Pillai's Trace":
                            df2 = s_val * (v_den - dv + 1) + 4
                        elif manova_test == "Wilks' Lambda":
                            t_val = max(np.sqrt((u**2 * v_num**2 - 4) / max(u**2 + v_num**2 - 5, 1)), 1)
                            df2 = (v_den - (u - v_num + 1) / 2) * t_val - (u * v_num - 2) / 2
                        elif manova_test == "Hotelling-Lawley Trace":
                            df2 = s_val * (v_den - dv - 1) + 4
                        else:
                            df2 = s_val * (v_den - dv + 1) + 4
                        ncp = adj_f2 * n_try * (1 - rho)
                        from scipy.stats import ncf as noncentral_f, f as f_dist

                        f_crit = f_dist.ppf(1 - alpha, df1, df2)
                        p_cur = 1 - noncentral_f.cdf(f_crit, df1, df2, ncp)
                        if p_cur >= power:
                            n_s = n_try
                            break
                else:
                    n_s = None
            elif atype == "binomial":
                p0 = params["p0"]
                p1 = params["p1"]
                p1_adj = p0 + (p1 - p0) * mult
                p1_adj = max(min(p1_adj, 0.99), 0.01)
                adj_es = abs(p1_adj - p0)
                if p0 != p1_adj:
                    from scipy.stats import binom

                    n_s = None
                    for n_try in range(3, 10000):
                        if alternative == "two-sided":
                            alpha_lo = binom.ppf(alpha / 2, n_try, p0)
                            alpha_hi = binom.ppf(1 - alpha / 2, n_try, p0)
                            p_pow = binom.cdf(alpha_hi, n_try, p1_adj) - binom.cdf(
                                alpha_lo - 1, n_try, p1_adj
                            )
                        elif p1_adj > p0:
                            crit = binom.ppf(1 - alpha, n_try, p0)
                            p_pow = 1 - binom.cdf(crit - 1, n_try, p1_adj)
                        else:
                            crit = binom.ppf(alpha, n_try, p0)
                            p_pow = binom.cdf(crit, n_try, p1_adj)
                        if p_pow >= power:
                            n_s = n_try
                            break
                else:
                    n_s = None
            else:
                n_s = None
        except Exception:
            n_s = None
        if n_s is not None and n_s > 0 and n_s < 100000:
            sens_data.append(
                {
                    "Effect Size Multiplier": f"{mult:.1f}×",
                    "Adjusted Effect": (
                        f"{adj_es:.3f}" if mult != 1.0 else f"{adj_es:.3f} (baseline)"
                    ),
                    "Required N": n_s,
                }
            )

    if sens_data:
        st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)

    # --- Formula ---
    with st.expander("📐 Formula Used"):
        st.latex(formula_latex)

    # --- Interpretation Guide ---
    with st.expander("📖 How to Interpret These Results"):
        if n_per_group is not None:
            if isinstance(n_per_group, tuple):
                n1, n2 = n_per_group
                st.markdown(f"""
                - You need **at least {n1} participants in Group 1** and **{n2} in Group 2**.
                - **Total**: {n_total} participants.
                - This assumes α = {alpha}, power = {power:.0%}, and your estimated effect size.
                - Use the **Power Curve** above to see how N affects your study's power.
                - The **Sensitivity Table** shows how N changes with different effect sizes.
                """)
            else:
                st.markdown(f"""
                - You need **at least {n_per_group} participants per group** (total N = {n_total}).
                - This assumes α = {alpha}, power = {power:.0%}, and your estimated effect size.
                - Use the **Power Curve** above to see how N affects your study's power.
                - The **Sensitivity Table** shows how N changes with different effect sizes.
                """)
        st.warning(
            "**Disclaimer**: Sample size estimation is based on statistical assumptions. "
            "Always consult a biostatistician and consider practical constraints (budget, "
            "dropout rates, feasibility) when finalizing your study size."
        )

    # --- Budget & Feasibility ---
    if cost_per > 0 and n_total is not None:
        st.subheader("💰 Budget & Feasibility")
        total_cost = n_total * cost_per
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.metric("Total Study Cost", f"${total_cost:,.0f}")
        with col_b2:
            st.metric("Cost per Participant", f"${cost_per:,.0f}")
        with col_b3:
            if recruitment_rate > 0:
                months = int(np.ceil(n_total / recruitment_rate))
                st.metric("Est. Recruitment Duration", f"{months} months")
            else:
                st.metric("Recruitment Rate", "Not specified")

        if dropout_rate > 0 and n_total_raw is not None:
            st.caption(
                f"Base N = {n_total_raw}, adjusted for {dropout_rate:.0%} dropout → {n_total}. "
                f"Extra cost due to dropout: ${(n_total - n_total_raw) * cost_per:,.0f}."
            )

    # --- What-If Scenario Explorer ---
    st.subheader("🔍 What-If Scenario Explorer")
    st.caption(
        "Explore how sample size changes across different combinations of power and effect size."
    )

    whatif_powers = [0.70, 0.80, 0.90]
    whatif_mults = [0.5, 0.75, 1.0, 1.25, 1.5]

    # Compute a heatmap: rows = power, cols = effect multiplier
    heatmap_data = []
    for w_power in whatif_powers:
        w_z_beta = norm.ppf(w_power)
        row = []
        for w_mult in whatif_mults:
            try:
                if atype == "one_mean":
                    from statsmodels.stats.power import TTestPower

                    d = params["effect_size"] * w_mult
                    w_n = int(
                        np.ceil(
                            TTestPower().solve_power(
                                effect_size=d,
                                alpha=alpha,
                                power=w_power,
                                alternative=alternative,
                            )
                        )
                    )
                elif atype == "two_means":
                    from statsmodels.stats.power import TTestIndPower

                    d = params["effect_size"] * w_mult
                    n1 = TTestIndPower().solve_power(
                        effect_size=d,
                        alpha=alpha,
                        power=w_power,
                        ratio=params["ratio"],
                        alternative=alternative,
                    )
                    n1 = int(np.ceil(n1))
                    w_n = n1 + int(np.ceil(n1 * params["ratio"]))
                elif atype == "paired":
                    from statsmodels.stats.power import TTestPower

                    d = params["effect_size"] * w_mult
                    w_n = int(
                        np.ceil(
                            TTestPower().solve_power(
                                effect_size=d,
                                alpha=alpha,
                                power=w_power,
                                alternative=alternative,
                            )
                        )
                    )
                elif atype == "one_prop":
                    from statsmodels.stats.proportion import proportion_effectsize
                    from statsmodels.stats.power import NormalIndPower

                    adj_p1 = (
                        params["prop_null"]
                        + (params["prop_alt"] - params["prop_null"]) * w_mult
                    )
                    adj_p1 = max(min(adj_p1, 0.99), 0.01)
                    d_eff = proportion_effectsize(adj_p1, params["prop_null"])
                    w_n = int(
                        np.ceil(
                            NormalIndPower().solve_power(
                                effect_size=abs(d_eff),
                                alpha=alpha,
                                power=w_power,
                                alternative=alternative,
                            )
                        )
                    )
                elif atype == "two_prop":
                    from statsmodels.stats.proportion import proportion_effectsize
                    from statsmodels.stats.power import NormalIndPower

                    adj_p2 = params["p1"] + (params["p2"] - params["p1"]) * w_mult
                    adj_p2 = max(min(adj_p2, 0.99), 0.01)
                    d_eff = proportion_effectsize(adj_p2, params["p1"])
                    n1 = NormalIndPower().solve_power(
                        effect_size=abs(d_eff),
                        alpha=alpha,
                        power=w_power,
                        ratio=params["ratio"],
                        alternative=alternative,
                    )
                    n1 = int(np.ceil(n1))
                    w_n = n1 + int(np.ceil(n1 * params["ratio"]))
                elif atype == "anova":
                    from statsmodels.stats.power import FTestAnovaPower

                    f_eff = params["effect_size"] * w_mult
                    n_per_g = FTestAnovaPower().solve_power(
                        effect_size=f_eff,
                        alpha=alpha,
                        power=w_power,
                        k_groups=params["k"],
                    )
                    n_per_g = int(np.ceil(n_per_g))
                    w_n = n_per_g * params["k"]
                elif atype == "correlation":
                    import math

                    r_val = max(min(params["effect_size"] * w_mult, 0.99), 0.01)
                    fisher_z = math.atanh(r_val)
                    w_n = int(np.ceil(3 + ((z_alpha + w_z_beta) / fisher_z) ** 2))
                elif atype == "regression":
                    from scipy.stats import ncf as noncentral_f, f as f_dist

                    adj_f2 = params["effect_size"] * w_mult
                    k_r = params["k"]
                    w_n = None
                    for nc in range(k_r + 2, 10000):
                        dfd = nc - k_r - 1
                        ncp = adj_f2 * nc
                        f_crit = f_dist.ppf(1 - alpha, k_r, dfd)
                        p_cur = 1 - noncentral_f.cdf(f_crit, k_r, dfd, ncp)
                        if p_cur >= w_power:
                            w_n = nc
                            break
                elif atype == "logistic":
                    from statsmodels.stats.power import NormalIndPower

                    adj_or = max(params["or"] ** w_mult, 1.01)
                    ev_rate = params["event_rate"]
                    k_log = params["k"]
                    p1_log = (adj_or * ev_rate) / (1 - ev_rate + adj_or * ev_rate)
                    d_log = abs(p1_log - ev_rate)
                    p_bar = (ev_rate + p1_log) / 2
                    se = np.sqrt(2 * p_bar * (1 - p_bar))
                    d_eff = d_log / se if se > 0 else 0
                    w_n = int(
                        np.ceil(
                            max(
                                NormalIndPower().solve_power(
                                    effect_size=d_eff,
                                    alpha=alpha,
                                    power=w_power,
                                    alternative=alternative,
                                ),
                                10 * k_log,
                            )
                        )
                    )
                elif atype == "chisq":
                    from statsmodels.stats.power import GofChisquarePower

                    adj_w = params["effect_size"] * w_mult
                    w_n = int(
                        np.ceil(
                            GofChisquarePower().solve_power(
                                effect_size=adj_w,
                                alpha=alpha,
                                power=w_power,
                                n_bins=params["df"] + 1,
                            )
                        )
                    )
                elif atype == "mannwhitney":
                    from statsmodels.stats.power import NormalIndPower

                    adj_p = max(min(params["effect_size"] * w_mult, 0.99), 0.01)
                    d_mw = np.sqrt(3) * (adj_p - 0.5)
                    are = params["are"]
                    ratio = params["ratio"]
                    n1 = NormalIndPower().solve_power(
                        effect_size=d_mw,
                        alpha=alpha,
                        power=w_power,
                        ratio=ratio,
                        alternative=alternative,
                    )
                    n1 = int(np.ceil(n1 / are))
                    w_n = n1 + int(np.ceil(n1 * ratio))
                elif atype == "logrank":
                    adj_hr = max(params["hr"] ** w_mult, 1.001)
                    ratio = params["ratio"]
                    med_ctrl = params["median_survival"]
                    study_dur = params["study_duration"]
                    log_hr = np.log(adj_hr)
                    num_events = (
                        ((z_alpha + w_z_beta) ** 2)
                        * ((ratio + 1) ** 2)
                        / (ratio * (log_hr**2))
                    )
                    lambda_ctrl = np.log(2) / med_ctrl
                    p_event_ctrl = 1 - np.exp(-lambda_ctrl * study_dur)
                    p_event_trt = 1 - np.exp(-lambda_ctrl / adj_hr * study_dur)
                    p_event = (p_event_ctrl + ratio * p_event_trt) / (1 + ratio)
                    w_n = int(np.ceil(num_events / p_event)) if p_event > 0 else None
                elif atype == "cox":
                    adj_hr = max(params["hr"] ** w_mult, 1.001)
                    k = params["k"]
                    sd_x = params["sd_x"]
                    r2_x = params["r2_x"]
                    ev_rate = params["event_rate"]
                    log_hr = np.log(adj_hr)
                    var_denom = (sd_x**2) * (log_hr**2) * (1 - r2_x)
                    if var_denom > 0:
                        num_events = ((z_alpha + w_z_beta) ** 2) / var_denom
                        num_events = max(num_events, 10 * k)
                        w_n = int(np.ceil(num_events / ev_rate))
                    else:
                        w_n = None
                elif atype == "equiv":
                    adj_margin = params["margin"] * w_mult
                    exp_diff = params["expected_diff"]
                    sd = params["sd"]
                    ratio = params["ratio"]
                    equiv_type = params.get("equiv_param_type", "Mean")
                    if equiv_type == "Proportion":
                        p1_eq = params.get("p1_eq", 0.2)
                        p2_eq = params.get("p2_eq", 0.2)
                        d_prop = abs(p1_eq - p2_eq)
                        d_e = adj_margin - d_prop
                        if d_e > 0:
                            p_bar = (p1_eq + p2_eq) / 2
                            se = np.sqrt(2 * p_bar * (1 - p_bar))
                            es_e = d_e / se if se > 0 else 0
                            from statsmodels.stats.power import NormalIndPower
                            n1 = NormalIndPower().solve_power(
                                effect_size=es_e, alpha=alpha, power=w_power,
                                ratio=ratio, alternative="larger",
                            )
                            n1 = int(np.ceil(n1))
                            w_n = n1 + int(np.ceil(n1 * ratio))
                        else:
                            w_n = None
                    else:
                        d_e = adj_margin - abs(exp_diff)
                        if d_e > 0:
                            es_e = d_e / sd
                            from statsmodels.stats.power import NormalIndPower
                            n1 = NormalIndPower().solve_power(
                                effect_size=es_e, alpha=alpha, power=w_power,
                                ratio=ratio, alternative="larger",
                            )
                            n1 = int(np.ceil(n1))
                            w_n = n1 + int(np.ceil(n1 * ratio))
                        else:
                            w_n = None
                elif atype == "rm_anova":
                    from statsmodels.stats.power import FTestAnovaPower

                    adj_es = params["effect_size"] * w_mult
                    k = params["k"]
                    m = params["m"]
                    rho = params["rho"]
                    epsilon = params["epsilon"]
                    n_per_g = FTestAnovaPower().solve_power(
                        effect_size=adj_es, alpha=alpha, power=w_power, k_groups=k
                    )
                    design_effect = (1 + (m - 1) * rho) / m
                    df_adj = (m - 1) * epsilon
                    n_per_g_adj = max(
                        int(
                            np.ceil(
                                n_per_g * design_effect * k / (k * df_adj / (k - 1))
                            )
                        ),
                        int(np.ceil(n_per_g)),
                    )
                    w_n = n_per_g_adj * k
                elif atype == "twoway_anova":
                    from statsmodels.stats.power import FTestAnovaPower

                    rows = params["rows"]
                    cols = params["cols"]
                    focus = params["focus"]
                    if focus == "Main Effect A":
                        adj_es = params["f_a"] * w_mult
                        k_use = rows
                    elif focus == "Main Effect B":
                        adj_es = params["f_b"] * w_mult
                        k_use = cols
                    else:
                        adj_es = params["f_ab"] * w_mult
                        k_use = rows * cols
                    n_per_cell = int(
                        np.ceil(
                            FTestAnovaPower().solve_power(
                                effect_size=adj_es,
                                alpha=alpha,
                                power=w_power,
                                k_groups=k_use,
                            )
                        )
                    )
                    w_n = n_per_cell * rows * cols
                elif atype == "roc_auc":
                    auc = params["auc"]
                    adj_auc = 0.5 + (auc - 0.5) * w_mult
                    adj_auc = max(min(adj_auc, 0.99), 0.51)
                    null_auc = params["null_auc"]
                    ratio = params["ratio"]
                    v_auc = adj_auc * (1 - adj_auc) + (ratio - 1) * (
                        adj_auc / (2 - adj_auc) - adj_auc**2
                    ) / (1 + ratio)
                    delta_auc = adj_auc - null_auc
                    if delta_auc > 0:
                        n_cases = int(
                            np.ceil(
                                ((z_alpha + w_z_beta) ** 2 * v_auc) / (delta_auc**2)
                            )
                        )
                        n_controls = int(np.ceil(n_cases * ratio))
                        w_n = n_cases + n_controls
                    else:
                        w_n = None
                elif atype == "kappa":
                    adj_kappa = params["kappa"]
                    adj_kappa = 0.5 + (adj_kappa - 0.5) * w_mult
                    adj_kappa = max(min(adj_kappa, 0.99), 0.01)
                    null_kappa = params["null_kappa"]
                    adj_kappa = max(adj_kappa, null_kappa + 0.01)
                    delta_k = adj_kappa - null_kappa
                    if delta_k > 0:
                        w_n = int(
                            np.ceil(
                                (
                                    (z_alpha + w_z_beta) ** 2
                                    * null_kappa
                                    * (1 - null_kappa)
                                )
                                / (delta_k**2)
                            )
                        )
                        w_n = max(w_n, params["raters"] * params["categories"] * 5)
                    else:
                        w_n = None
                elif atype == "cluster_rct":
                    from statsmodels.stats.power import TTestIndPower

                    adj_d = params["effect_size"] * w_mult
                    icc = params["icc"]
                    cluster_m = params["cluster_size"]
                    ratio = params["ratio"]
                    deff = 1 + (cluster_m - 1) * icc
                    n1_ind = TTestIndPower().solve_power(
                        effect_size=adj_d,
                        alpha=alpha,
                        power=w_power,
                        ratio=ratio,
                        alternative=alternative,
                    )
                    n1_ind = int(np.ceil(n1_ind))
                    n1_clust = int(np.ceil(n1_ind * deff))
                    n1_clust = int(np.ceil(n1_clust / cluster_m)) * cluster_m
                    n2_clust = int(np.ceil(n1_clust * ratio))
                    w_n = n1_clust + n2_clust
                elif atype == "precision":
                    adj_hw = params["half_width"] * w_mult
                    conf_level = params["conf_level"]
                    conf_alpha = 1 - conf_level / 100
                    param_type = params["param_type"]
                    if param_type == "Mean":
                        sd = params["sd"]
                        if sd > 0 and adj_hw > 0:
                            from scipy.stats import t as t_dist
                            w_n = int(np.ceil((norm.ppf(1 - conf_alpha / 2) * sd / adj_hw) ** 2))
                            w_n = max(w_n, 3)
                            for _ in range(20):
                                t_val = t_dist.ppf(1 - conf_alpha / 2, df=w_n - 1)
                                n_next = int(np.ceil((t_val * sd / adj_hw) ** 2))
                                n_next = max(n_next, 3)
                                if n_next == w_n:
                                    break
                                w_n = n_next
                        else:
                            w_n = None
                    else:
                        prop = params["prop"]
                        z_hw = norm.ppf(1 - conf_alpha / 2)
                        w_n = (
                            int(np.ceil((z_hw**2 * prop * (1 - prop)) / (adj_hw**2)))
                            if prop > 0 and adj_hw > 0
                            else None
                        )
                    if w_n is not None:
                        w_n = max(w_n, 3)
                elif atype == "pilot":
                    method = params["method"]
                    if method == "Rule of thumb":
                        w_n = int(np.ceil(params["n_per_group"] / max(w_mult, 0.1)))
                    elif method == "Precision-based":
                        conf_level = params["conf_level"]
                        conf_alpha = 1 - conf_level / 100
                        param_type = params["param_type"]
                        adj_hw = params["half_width"] * w_mult
                        if param_type == "Mean":
                            sd = params["sd"]
                            if sd > 0 and adj_hw > 0:
                                from scipy.stats import t as t_dist
                                w_n = int(np.ceil((norm.ppf(1 - conf_alpha / 2) * sd / adj_hw) ** 2))
                                w_n = max(w_n, 3)
                                for _ in range(20):
                                    t_val = t_dist.ppf(1 - conf_alpha / 2, df=w_n - 1)
                                    n_next = int(np.ceil((t_val * sd / adj_hw) ** 2))
                                    n_next = max(n_next, 3)
                                    if n_next == w_n:
                                        break
                                    w_n = n_next
                            else:
                                w_n = None
                        else:
                            z_hw = norm.ppf(1 - conf_alpha / 2)
                            prop = params["prop"]
                            w_n = (
                                int(
                                    np.ceil((z_hw**2 * prop * (1 - prop)) / (adj_hw**2))
                                )
                                if prop > 0 and adj_hw > 0
                                else None
                            )
                        if w_n is not None:
                            w_n = max(w_n, 5)
                    else:
                        main_n = params["main_n"]
                        fraction = params["fraction"]
                        w_n = max(int(np.ceil(main_n * fraction / max(w_mult, 0.1))), 5)
                elif atype == "wilcoxon_sr":
                    p_val = params["effect_size"]
                    are = params["are"]
                    d_wsr = np.sqrt(3) * (p_val - 0.5) * 2
                    if d_wsr > 0:
                        from statsmodels.stats.power import NormalIndPower

                        w_n = int(
                            np.ceil(
                                NormalIndPower().solve_power(
                                    effect_size=d_wsr,
                                    alpha=alpha,
                                    power=w_power,
                                    alternative=alternative,
                                )
                                / are
                            )
                        )
                    else:
                        w_n = None
                elif atype == "kruskal":
                    f_eff = params["effect_size"] * w_mult
                    k = params["k"]
                    are_kw = params.get("are", 0.955)
                    if f_eff > 0:
                        from statsmodels.stats.power import FTestAnovaPower

                        n_per_g = FTestAnovaPower().solve_power(
                            effect_size=f_eff, alpha=alpha, power=w_power, k_groups=k
                        )
                        n_per_g = int(np.ceil(n_per_g / are_kw))
                        w_n = n_per_g * k
                    else:
                        w_n = None
                elif atype == "friedman":
                    w = params["w"]
                    k = params["k"]
                    m = params["m"]
                    are = params["are"]
                    adj_w = max(min(w * w_mult, 0.99), 0.01)
                    f_fr = np.sqrt(adj_w / (1 - adj_w))
                    if f_fr > 0:
                        from statsmodels.stats.power import FTestAnovaPower

                        n_per_g = FTestAnovaPower().solve_power(
                            effect_size=f_fr, alpha=alpha, power=w_power, k_groups=k
                        )
                        n_per_g = int(np.ceil(n_per_g * m / are))
                        w_n = n_per_g * k
                    else:
                        w_n = None
                elif atype == "mcnemar":
                    p_b = params["p_b"]
                    p_c = params["p_c"]
                    p_b_adj = max(min(p_b * w_mult, 0.99), 0.01)
                    p_c_adj = max(min(p_c * w_mult, 0.99), 0.01)
                    d_mc = abs(p_b_adj - p_c_adj)
                    p_discordant = p_b_adj + p_c_adj
                    if d_mc > 0 and p_discordant > 0:
                        w_n = max(
                            int(
                                np.ceil(
                                    ((z_alpha + w_z_beta) ** 2 * p_discordant)
                                    / (d_mc**2)
                                )
                            ),
                            5,
                        )
                    else:
                        w_n = None
                elif atype == "fisher":
                    p1 = params["p1"]
                    p2 = params["p2"]
                    ratio = params["ratio"]
                    are_f = params.get("are", 0.833)
                    adj_p2 = p1 + (p2 - p1) * w_mult
                    adj_p2 = max(min(adj_p2, 0.99), 0.01)
                    from statsmodels.stats.proportion import proportion_effectsize
                    from statsmodels.stats.power import NormalIndPower

                    d_eff = proportion_effectsize(adj_p2, p1)
                    if abs(d_eff) > 0:
                        n1 = NormalIndPower().solve_power(
                            effect_size=abs(d_eff),
                            alpha=alpha,
                            power=w_power,
                            ratio=ratio,
                            alternative=alternative,
                        )
                        n1 = int(np.ceil(n1 / are_f))
                        w_n = n1 + int(np.ceil(n1 * ratio))
                    else:
                        w_n = None
                elif atype == "manova":
                    k = params["k"]
                    dv = params["dv"]
                    f2 = params["f2"]
                    rho = params["rho"]
                    adj_f2 = f2 * w_mult
                    manova_test = params.get("manova_test", "Pillai's Trace")
                    if adj_f2 > 0:
                        u = dv
                        v_num = k - 1
                        w_n = None
                        for n_try in range(k * dv + 2, 5000):
                            v_den = n_try - k - dv
                            if v_den <= 0:
                                continue
                            s_val = min(u, v_num)
                            df1 = u * v_num
                            if manova_test == "Pillai's Trace":
                                df2 = s_val * (v_den - dv + 1) + 4
                            elif manova_test == "Wilks' Lambda":
                                t_val = max(np.sqrt((u**2 * v_num**2 - 4) / max(u**2 + v_num**2 - 5, 1)), 1)
                                df2 = (v_den - (u - v_num + 1) / 2) * t_val - (u * v_num - 2) / 2
                            elif manova_test == "Hotelling-Lawley Trace":
                                df2 = s_val * (v_den - dv - 1) + 4
                            else:
                                df2 = s_val * (v_den - dv + 1) + 4
                            ncp = adj_f2 * n_try * (1 - rho)
                            from scipy.stats import ncf as noncentral_f, f as f_dist
                            f_crit = f_dist.ppf(1 - alpha, df1, df2)
                            p_cur = 1 - noncentral_f.cdf(f_crit, df1, df2, ncp)
                            if p_cur >= w_power:
                                w_n = n_try
                                break
                    else:
                        w_n = None
                elif atype == "binomial":
                    p0 = params["p0"]
                    p1 = params["p1"]
                    p1_adj = p0 + (p1 - p0) * w_mult
                    p1_adj = max(min(p1_adj, 0.99), 0.01)
                    if p0 != p1_adj:
                        from scipy.stats import binom

                        w_n = None
                        for n_try in range(3, 10000):
                            if alternative == "two-sided":
                                alpha_lo = binom.ppf(alpha / 2, n_try, p0)
                                alpha_hi = binom.ppf(1 - alpha / 2, n_try, p0)
                                p_pow = binom.cdf(alpha_hi, n_try, p1_adj) - binom.cdf(
                                    alpha_lo - 1, n_try, p1_adj
                                )
                            elif p1_adj > p0:
                                crit = binom.ppf(1 - alpha, n_try, p0)
                                p_pow = 1 - binom.cdf(crit - 1, n_try, p1_adj)
                            else:
                                crit = binom.ppf(alpha, n_try, p0)
                                p_pow = binom.cdf(crit, n_try, p1_adj)
                            if p_pow >= w_power:
                                w_n = n_try
                                break
                    else:
                        w_n = None
                else:
                    w_n = None
            except Exception:
                w_n = None
            row.append(w_n if w_n and w_n < 1000000 else None)
        heatmap_data.append(row)

    if any(any(r is not None for r in row) for row in heatmap_data):
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=[f"{m:.2f}×" for m in whatif_mults],
                y=[f"Power = {p:.0%}" for p in whatif_powers],
                text=[[str(v) if v else "—" for v in row] for row in heatmap_data],
                texttemplate="%{text}",
                colorscale="Blues",
                hovertemplate="Power: %{y}<br>Effect Size: %{x}<br>N: %{text}<extra></extra>",
            )
        )
        fig_heat.update_layout(
            template="plotly_dark",
            height=250,
            xaxis_title="Effect Size Multiplier",
            yaxis_title="",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Cells show required N at each power × effect size combination. Adjust your design assumptions accordingly."
        )

    # --- Inverse Power Analysis (Minimum Detectable Effect) ---
    st.subheader("🎯 Sensitivity Analysis: Minimum Detectable Effect")
    st.caption(
        "Given a fixed sample size, what is the smallest effect size your study can detect?"
    )

    with st.expander("Enter a candidate sample size to compute the minimum detectable effect"):
        c1, c2 = st.columns([1, 3])
        with c1:
            candidate_n = st.number_input(
                "Candidate N (total)",
                min_value=5,
                max_value=100000,
                value=n_total if n_total else 100,
                step=10,
            )
        with c2:
            st.caption(" ")
        if st.button("Compute Minimum Detectable Effect", type="secondary"):
            try:
                from scipy.stats import t as t_dist_inv

                mde = None
                mde_label = ""
                mde_note = ""

                if atype == "one_mean":
                    from statsmodels.stats.power import TTestPower
                    mde = TTestPower().solve_power(
                        effect_size=None, nobs=candidate_n, alpha=alpha, power=power, alternative=alternative
                    )
                    mde_label = "Cohen's d"
                elif atype == "two_means":
                    from statsmodels.stats.power import TTestIndPower
                    ratio_val = params.get("ratio", 1)
                    n1 = candidate_n / (1 + ratio_val)
                    mde = TTestIndPower().solve_power(
                        effect_size=None, nobs1=n1, alpha=alpha, power=power, ratio=ratio_val, alternative=alternative
                    )
                    mde_label = "Cohen's d"
                elif atype == "paired":
                    from statsmodels.stats.power import TTestPower
                    mde = TTestPower().solve_power(
                        effect_size=None, nobs=candidate_n, alpha=alpha, power=power, alternative=alternative
                    )
                    mde_label = "Cohen's d_z"
                elif atype == "one_prop":
                    from statsmodels.stats.power import NormalIndPower
                    d_eff = NormalIndPower().solve_power(
                        effect_size=None, nobs1=candidate_n, alpha=alpha, power=power, alternative=alternative
                    )
                    p0 = params.get("prop_null", 0.5)
                    mde = p0 + d_eff * np.sqrt(p0 * (1 - p0))
                    mde = max(0.01, min(0.99, mde))
                    mde_label = "Detectable proportion p₁"
                    mde_note = f"(null p₀ = {p0})"
                elif atype == "two_prop":
                    from statsmodels.stats.proportion import proportion_effectsize
                    from statsmodels.stats.power import NormalIndPower
                    ratio_val = params.get("ratio", 1)
                    n1 = candidate_n / (1 + ratio_val)
                    d_eff = NormalIndPower().solve_power(
                        effect_size=None, nobs1=n1, alpha=alpha, power=power, ratio=ratio_val, alternative=alternative
                    )
                    p1_base = params.get("p1", 0.3)
                    mde = p1_base + d_eff * np.sqrt(2 * p1_base * (1 - p1_base))
                    mde = max(0.01, min(0.99, mde))
                    mde_label = "Detectable proportion p₂"
                    mde_note = f"(group 1 p₁ = {p1_base})"
                elif atype == "anova":
                    from statsmodels.stats.power import FTestAnovaPower
                    k_val = params.get("k", 3)
                    mde = FTestAnovaPower().solve_power(
                        effect_size=None, nobs=candidate_n, alpha=alpha, power=power, k_groups=k_val
                    )
                    mde_label = "Cohen's f"
                elif atype == "correlation":
                    import math
                    fisher_z = (norm.ppf(1 - alpha / 2 if alternative == "two-sided" else 1 - alpha) + z_beta) / np.sqrt(candidate_n - 3)
                    mde = min(math.tanh(fisher_z), 0.99)
                    mde_label = "Pearson r"
                elif atype == "chisq":
                    from statsmodels.stats.power import GofChisquarePower
                    df_val = params.get("df", 2)
                    mde = GofChisquarePower().solve_power(
                        effect_size=None, nobs=candidate_n, alpha=alpha, power=power, n_bins=df_val + 1
                    )
                    mde_label = "Cohen's w"
                elif atype == "mannwhitney":
                    from statsmodels.stats.power import NormalIndPower
                    are_val = params.get("are", 0.955)
                    ratio_val = params.get("ratio", 1)
                    n1_eff = candidate_n / (1 + ratio_val) * are_val
                    d_mw = NormalIndPower().solve_power(
                        effect_size=None, nobs1=n1_eff, alpha=alpha, power=power, ratio=ratio_val, alternative=alternative
                    )
                    mde = 0.5 + d_mw / np.sqrt(3)
                    mde = max(0.51, min(0.99, mde))
                    mde_label = "P(X>Y)"
                elif atype == "wilcoxon_sr":
                    from statsmodels.stats.power import NormalIndPower
                    are_val = params.get("are", 0.955)
                    d_z = NormalIndPower().solve_power(
                        effect_size=None, nobs1=candidate_n * are_val, alpha=alpha, power=power, alternative=alternative
                    )
                    mde = 0.5 + d_z / (2 * np.sqrt(3))
                    mde = max(0.51, min(0.99, mde))
                    mde_label = "Pr(positive diff)"
                elif atype == "kruskal":
                    from statsmodels.stats.power import FTestAnovaPower
                    are_val = params.get("are", 0.955)
                    k_val = params.get("k", 3)
                    mde = FTestAnovaPower().solve_power(
                        effect_size=None, nobs=candidate_n * are_val, alpha=alpha, power=power, k_groups=k_val
                    )
                    mde_label = "Cohen's f"
                elif atype == "mcnemar":
                    p_b = params.get("p_b", 0.2)
                    p_c = params.get("p_c", 0.4)
                    p_disc = p_b + p_c
                    if p_disc > 0:
                        delta = (norm.ppf(1 - alpha / 2 if alternative == "two-sided" else 1 - alpha) + z_beta) * np.sqrt(p_disc / candidate_n)
                        mde = max(delta, 0.01)
                        mde_label = "|p_b − p_c|"
                elif atype == "fisher":
                    from statsmodels.stats.proportion import proportion_effectsize
                    from statsmodels.stats.power import NormalIndPower
                    are_val = params.get("are", 0.833)
                    ratio_val = params.get("ratio", 1)
                    p1_base = params.get("p1", 0.3)
                    n1 = candidate_n / (1 + ratio_val) * are_val
                    d_eff = NormalIndPower().solve_power(
                        effect_size=None, nobs1=n1, alpha=alpha, power=power, ratio=ratio_val, alternative=alternative
                    )
                    mde = p1_base + d_eff * np.sqrt(2 * p1_base * (1 - p1_base))
                    mde = max(0.01, min(0.99, mde))
                    mde_label = "Detectable proportion p₂"
                elif atype == "logrank":
                    ratio_val = params.get("ratio", 1)
                    hr_guess = params.get("hr", 2)
                    num_events_est = candidate_n * 0.5
                    log_hr_min = (norm.ppf(1 - alpha / 2 if alternative == "two-sided" else 1 - alpha) + z_beta) * np.sqrt((ratio_val + 1) ** 2 / (ratio_val * num_events_est))
                    if log_hr_min > 0:
                        mde = np.exp(log_hr_min)
                        mde = max(1.001, min(10.0, mde))
                        mde_label = "Hazard Ratio (HR)"
                        mde_note = "(approximate, depends on event probability)"
                elif atype == "cox":
                    sd_x = params.get("sd_x", 1)
                    r2_x = params.get("r2_x", 0)
                    ev_rate = params.get("event_rate", 0.5)
                    num_events_est = candidate_n * ev_rate
                    var_denom_inv = (norm.ppf(1 - alpha / 2 if alternative == "two-sided" else 1 - alpha) + z_beta) ** 2 / num_events_est
                    if var_denom_inv > 0 and sd_x > 0:
                        log_hr_min = np.sqrt(var_denom_inv / (sd_x ** 2 * (1 - r2_x)))
                        mde = np.exp(log_hr_min)
                        mde = max(1.001, min(10.0, mde))
                        mde_label = "Hazard Ratio (HR)"
                        mde_note = "(approximate)"
                elif atype == "logistic":
                    import math
                    ev_rate = params.get("event_rate", 0.3)
                    or_val = params.get("or", 2)
                    p1_log = (or_val * ev_rate) / (1 - ev_rate + or_val * ev_rate)
                    d_log = abs(p1_log - ev_rate)
                    p_bar = (ev_rate + p1_log) / 2
                    se = np.sqrt(2 * p_bar * (1 - p_bar))
                    if se > 0:
                        d_eff = d_log / se
                        from statsmodels.stats.power import NormalIndPower
                        mde_d = NormalIndPower().solve_power(
                            effect_size=None, nobs1=candidate_n, alpha=alpha, power=power, alternative=alternative
                        )
                        p1_mde = ev_rate + mde_d * se
                        p1_mde = max(0.01, min(0.99, p1_mde))
                        mde = (p1_mde * (1 - ev_rate)) / (ev_rate * (1 - p1_mde))
                        mde = max(1.001, min(100.0, mde))
                        mde_label = "Odds Ratio (OR)"
                        mde_note = "(approximate)"
                elif atype == "regression":
                    k_r = params.get("k", 3)
                    from scipy.stats import ncf as noncentral_f, f as f_dist
                    dfd = candidate_n - k_r - 1
                    if dfd > 0:
                        mde = None
                        for f2_try in np.linspace(0.001, 2.0, 2000):
                            ncp = f2_try * candidate_n
                            f_crit = f_dist.ppf(1 - alpha, k_r, dfd)
                            p_cur = 1 - noncentral_f.cdf(f_crit, k_r, dfd, ncp)
                            if p_cur >= power:
                                mde = f2_try
                                break
                        mde_label = "Cohen's f²"
                    else:
                        mde = None
                elif atype == "binomial":
                    p0 = params.get("p0", 0.5)
                    from scipy.stats import binom
                    mde = None
                    for p1_try in np.linspace(p0 + 0.001, 0.99, 990):
                        if alternative == "two-sided":
                            alpha_lo = binom.ppf(alpha / 2, candidate_n, p0)
                            alpha_hi = binom.ppf(1 - alpha / 2, candidate_n, p0)
                            p_pow = binom.cdf(alpha_hi, candidate_n, p1_try) - binom.cdf(alpha_lo - 1, candidate_n, p1_try)
                        elif p1_try > p0:
                            crit = binom.ppf(1 - alpha, candidate_n, p0)
                            p_pow = 1 - binom.cdf(crit - 1, candidate_n, p1_try)
                        else:
                            crit = binom.ppf(alpha, candidate_n, p0)
                            p_pow = binom.cdf(crit, candidate_n, p1_try)
                        if p_pow >= power:
                            mde = p1_try
                            break
                    mde_label = "Detectable proportion p₁"
                elif atype == "equiv":
                    margin = params.get("margin", 1.0)
                    sd = params.get("sd", 1.0)
                    ratio_v = params.get("ratio", 1)
                    equiv_type = params.get("equiv_param_type", "Mean")
                    n1_eff = candidate_n / (1 + ratio_v)
                    if n1_eff > 1:
                        from statsmodels.stats.power import NormalIndPower
                        es_mde = NormalIndPower().solve_power(
                            effect_size=None, nobs1=n1_eff, alpha=alpha, power=power,
                            ratio=ratio_v, alternative="larger",
                        )
                        if equiv_type == "Proportion":
                            p_bar = (params.get("p1_eq", 0.2) + params.get("p2_eq", 0.2)) / 2
                            se = np.sqrt(2 * p_bar * (1 - p_bar))
                            mde = es_mde * se
                            mde_label = "Detectable margin remaining (δ - |p₁−p₂|)"
                        else:
                            mde = es_mde * sd
                            mde_label = "Detectable margin remaining (δ - |d|)"
                elif atype == "rm_anova":
                    from statsmodels.stats.power import FTestAnovaPower
                    f_eff = params.get("effect_size", 0.25)
                    k_v = params.get("k", 2)
                    m_v = params.get("m", 3)
                    rho_v = params.get("rho", 0.5)
                    eps_v = params.get("epsilon", 0.75)
                    design_effect = (1 + (m_v - 1) * rho_v) / m_v
                    df_adj = (m_v - 1) * eps_v
                    n_eff = candidate_n * design_effect * (k_v - 1) / (k_v * df_adj / (k_v - 1))
                    if n_eff > k_v:
                        mde = FTestAnovaPower().solve_power(
                            effect_size=None, nobs=n_eff, alpha=alpha, power=power, k_groups=k_v
                        )
                        mde_label = "Cohen's f"
                elif atype == "twoway_anova":
                    from statsmodels.stats.power import FTestAnovaPower
                    rows_v = params.get("rows", 2)
                    cols_v = params.get("cols", 2)
                    focus_v = params.get("focus", "Main Effect A")
                    k_use = rows_v if focus_v == "Main Effect A" else (cols_v if focus_v == "Main Effect B" else rows_v * cols_v)
                    n_per_cell_eff = candidate_n / (rows_v * cols_v)
                    if n_per_cell_eff > k_use:
                        mde = FTestAnovaPower().solve_power(
                            effect_size=None, nobs=n_per_cell_eff, alpha=alpha, power=power, k_groups=k_use
                        )
                        mde_label = "Cohen's f"

                if mde is not None and mde > 0:
                    st.success(f"**Minimum detectable {mde_label}** with N = {candidate_n}: **{mde:.4f}** {mde_note}")
                else:
                    st.warning("Could not compute minimum detectable effect for this analysis type.")
            except Exception as e:
                st.error(f"Could not compute minimum detectable effect: {e}")

    # --- Sample Size Justification ---
    st.subheader("📝 Sample Size Justification")
    st.caption(
        "Copy the full protocol below for your grant application, IRB submission, or research protocol."
    )

    from datetime import datetime

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    atype_label = {
        "one_mean": "One-sample Mean t/z-test",
        "two_means": "Two Independent Means t-test",
        "paired": "Paired Means t-test",
        "one_prop": "One-sample Proportion",
        "two_prop": "Two Proportions",
        "anova": "One-way ANOVA",
        "correlation": "Pearson Correlation",
        "regression": "Multiple Linear Regression",
        "logistic": "Logistic Regression",
        "chisq": "Chi-Square Test",
        "mannwhitney": "Mann-Whitney / Wilcoxon (Non-parametric)",
        "logrank": "Log-Rank Test (Survival)",
        "cox": "Cox Regression",
        "equiv": "Equivalence / Non-Inferiority",
        "rm_anova": "Repeated Measures ANOVA",
        "twoway_anova": "Two-way / Factorial ANOVA",
        "roc_auc": "ROC / AUC Analysis",
        "kappa": "Cohen's Kappa / ICC Agreement",
        "cluster_rct": "Cluster-RCT / Multilevel",
        "precision": "Precision-based (CI Width)",
        "pilot": "Pilot / Feasibility Study",
        "wilcoxon_sr": "Wilcoxon Signed-Rank (paired)",
        "kruskal": "Kruskal-Wallis Test",
        "friedman": "Friedman Test",
        "mcnemar": "McNemar's Test",
        "fisher": "Fisher's Exact Test",
        "manova": "MANOVA (Multivariate ANOVA)",
        "binomial": "Binomial Exact Test",
        "simulation": "Simulation-based Power (Monte Carlo)",
    }

    if n_per_group is not None:
        n_desc = (
            f"{n_per_group} per group"
            if not isinstance(n_per_group, tuple)
            else f"{n_per_group[0]} (Group 1) and {n_per_group[1]} (Group 2)"
        )
        protocol = f"""SAMPLE SIZE ESTIMATION PROTOCOL
Generated: {now_str}
Application: Statistical Test Finder (opencode.ai)

Analysis: {atype_label.get(atype, atype)}
Direction: {tails.lower()}
Significance Level (α): {alpha * num_tests:.4f}"""
        if num_tests > 1:
            protocol += f" ({mc_method}-adjusted from {alpha_raw:.4f}, {num_tests} comparisons)"
        protocol += f"""
Statistical Power (1−β): {power:.0%}
Effect Size: {params.get('effect_size', params.get('or', params.get('w', 'N/A'))):.4f}
"""
        if isinstance(n_per_group, tuple):
            protocol += f"""Allocation Ratio (n₂/n₁): {n_per_group[1] / n_per_group[0]:.2f}
"""
        protocol += f"""
Required Sample Size: {n_desc}
Total N: {n_total}"""
        if dropout_rate > 0 and n_total_raw is not None and n_total_raw != n_total:
            protocol += f"""
Dropout Rate: {dropout_rate:.0%}
Raw N (pre-dropout): {n_total_raw}
Adjusted N (post-dropout): {n_total}
"""
        else:
            protocol += "\n"
        if cost_per > 0:
            protocol += f"\nEstimated Study Cost: ${n_total * cost_per:,.0f}"
            if recruitment_rate > 0:
                protocol += f"\nEst. Recruitment Duration: {int(np.ceil(n_total / recruitment_rate))} months"
            protocol += "\n"
    else:
        protocol = f"""SAMPLE SIZE ESTIMATION PROTOCOL
Generated: {now_str}
Application: Statistical Test Finder (opencode.ai)

Analysis: {atype_label.get(atype, atype)}
Direction: {tails.lower()}
Significance Level (α): {alpha * num_tests:.4f}
Statistical Power (1−β): {power:.0%}
Effect Size: {params.get('effect_size', 'N/A'):.4f}
Required Total N: {n_total}
"""

    fields_str = []
    for k, v in params.items():
        if k in (
            "type",
            "alpha",
            "power",
            "tails",
            "dropout_rate",
            "num_tests",
            "cost_per",
            "recruitment_rate",
        ):
            continue
        fields_str.append(f"  {k}: {v}")
    if fields_str:
        protocol += "\nFull Parameters:\n" + "\n".join(fields_str) + "\n"

    protocol += "\n--- Generated by Statistical Test Finder ---"

    if n_per_group is not None:
        if isinstance(n_per_group, tuple):
            n1, n2 = n_per_group
            justification = (
                f"A sample size of {n1} in Group 1 and {n2} in Group 2 "
                f"(total N = {n_total}) was determined to provide {power:.0%} power "
                f"at a significance level of α = {alpha * num_tests:.3f} "
            )
            if num_tests > 1:
                justification += f"({mc_method}-adjusted for {num_tests} comparisons, per-test α = {alpha:.4f}) "
            justification += (
                f"to detect the anticipated effect size. "
                f"Sample size estimation was performed using a {tails.lower()} {explanation.split('to detect')[0].strip().lower()}. "
            )
            if dropout_rate > 0:
                justification += (
                    f"To account for an anticipated dropout rate of {dropout_rate:.0%}, "
                    f"the required sample was inflated from {n_total_raw} to {n_total} participants. "
                )
            justification += "All calculations were performed using the Statistical Test Finder application."
        else:
            justification = (
                f"A sample size of {n_per_group} per group (total N = {n_total}) "
                f"was determined to provide {power:.0%} power at a significance level of α = {alpha * num_tests:.3f} "
            )
            if num_tests > 1:
                justification += f"({mc_method}-adjusted for {num_tests} comparisons, per-test α = {alpha:.4f}) "
            justification += (
                f"to detect the anticipated effect size. "
                f"Sample size estimation was performed using a {tails.lower()} {explanation.split('to detect')[0].strip().lower()}. "
            )
            if dropout_rate > 0:
                justification += (
                    f"To account for an anticipated dropout rate of {dropout_rate:.0%}, "
                    f"the required sample was inflated from {n_total_raw} to {n_total} participants. "
                )
            justification += "All calculations were performed using the Statistical Test Finder application."
    else:
        justification = f"A sample size of {n_total} participants was determined to provide {power:.0%} power at α = {alpha * num_tests:.3f}. All calculations were performed using the Statistical Test Finder application."

    st.text(justification)

    st.divider()
    with st.expander("📄 Full Protocol Text"):
        st.text_area(
            ":orange[Full Protocol (select all, Ctrl+C / Cmd+C to copy)]",
            protocol,
            height=350,
        )

    # --- Key References ---
    with st.expander("📚 Key References"):
        st.markdown("""
        **General Sample Size & Power:**
        - Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
        - Faul, F., Erdfelder, E., Lang, A.-G., & Buchner, A. (2007). G*Power 3: A flexible statistical power analysis program. *Behavior Research Methods*, 39(2), 175–191.

        **Means & t-tests:**
        - Julious, S. A. (2004). Sample sizes for clinical trials with normal data. *Statistics in Medicine*, 23(12), 1921–1986.

        **Proportions:**
        - Fleiss, J. L., Levin, B., & Paik, M. C. (2003). *Statistical Methods for Rates and Proportions* (3rd ed.). Wiley.

        **ANOVA & F-tests:**
        - Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

        **Regression:**
        - Green, S. B. (1991). How many subjects does it take to do a regression analysis? *Multivariate Behavioral Research*, 26(3), 499–510.
        - Hsieh, F. Y., Bloch, D. A., & Larsen, M. D. (1998). A simple method of sample size calculation for linear and logistic regression. *Statistics in Medicine*, 17(14), 1623–1634.

        **Correlation:**
        - Fisher, R. A. (1915). Frequency distribution of the values of the correlation coefficient in samples from an indefinitely large population. *Biometrika*, 10(4), 507–521.

        **Survival Analysis:**
        - Schoenfeld, D. (1983). Sample-size formula for the proportional-hazards regression model. *Biometrics*, 39(2), 499–503.
        - Freedman, L. S. (1982). Tables of the number of patients required in clinical trials using the logrank test. *Statistics in Medicine*, 1(2), 121–129.

        **Non-parametric Tests:**
        - Lehmann, E. L. (2006). *Nonparametrics: Statistical Methods Based on Ranks*. Springer.

        **Equivalence / Non-Inferiority:**
        - Blackwelder, W. C. (1982). "Proving the null hypothesis" in clinical trials. *Controlled Clinical Trials*, 3(4), 345–353.

        **Cluster-RCT:**
        - Donner, A., & Klar, N. (2000). *Design and Analysis of Cluster Randomization Trials in Health Research*. Arnold.

        **Repeated Measures:**
        - Greenhouse, S. W., & Geisser, S. (1959). On methods in the analysis of profile data. *Psychometrika*, 24(2), 95–112.

        **ROC / AUC:**
        - Obuchowski, N. A. (1994). Sample size calculations in studies of test accuracy. *Statistical Methods in Medical Research*, 7(4), 371–392.

        **Kappa / Agreement:**
        - Cantor, A. B. (1996). Sample-size calculations for Cohen's kappa. *Psychological Methods*, 1(2), 150–153.

        **Pilot Studies:**
        - Julious, S. A. (2005). Sample size of 12 per group rule of thumb for a pilot study. *Pharmaceutical Statistics*, 4(4), 287–291.
        - Whitehead, A. L., et al. (2016). Estimating the sample size for a pilot randomised trial to minimise the overall trial sample size. *Journal of Clinical Epidemiology*, 71, 23–29.

        **Multiple Testing:**
        - Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle probabilità. *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*, 8, 3–62.
        """)


# =========================
# FLOWCHART VIEW
# =========================

FIELDS = [
    "Objective",
    "Dependent_Variable",
    "Independent_Variable",
    "Groups",
    "Relation",
    "Distribution",
]


def build_tree(rule_subset, fields, user_input=None, level=0):

    # No more fields → show tests
    if not fields:

        for rule in rule_subset:
            st.success(f"✅ {rule['name']}")

        return

    current_field = fields[0]
    # Skip meaningless all-any branches
    all_any = all(rule[current_field] == "any" for rule in rule_subset)

    if all_any:
        build_tree(rule_subset, fields[1:], user_input, level + 1)
        return

    # Group rules by current field
    grouped = {}

    for rule in rule_subset:

        value = rule[current_field]

        if isinstance(value, list):

            for item in value:
                grouped.setdefault(item, []).append(rule)

        else:
            grouped.setdefault(value, []).append(rule)

    # Render Groups
    for value, subrules in grouped.items():

        # Pretty label
        if isinstance(value, tuple):
            label = " OR ".join(value)
        else:
            label = str(value)

        # Highlighting logic
        is_selected = False
        if user_input and current_field in user_input:
            user_val = user_input[current_field]
            if user_val == value or (user_val == "any" and value == "any"):
                is_selected = True

        display_label = (
            f"🎯 **{current_field}: {label}** (Current Selection)"
            if is_selected
            else f"{current_field}: {label}"
        )

        with st.expander(display_label, expanded=is_selected):

            build_tree(subrules, fields[1:], user_input, level + 1)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
