import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
                    - :orange[$\frac{s}{\sqrt{n}}$] is the standard error of the mean.
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
                    - The denominator :orange[$\sqrt{\frac{p_0(1 - p_0)}{n}}$] is the standard error of the proportion.
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
        "Explanation": "One-way ANOVA This test is used to compare the means of three or more independent groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, and that the groups are independent.",
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
                    $$ U = \sum_{i=1}^{n_1} R_i - \frac{n_1(n_1+1)}{2} $$ 
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
                    $$ H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1) $$ 
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
        "Explanation": "Repeated Measures ANOVA This test is used to compare the means of three or more related groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, and that the groups are dependent (e.g., measurements taken from the same subjects at multiple time points).",
        "Example": "A researcher wants to test the effect of a new drug on blood pressure over time. The researcher measures the blood pressure of 30 patients at three different time points: before treatment, after 1 month of treatment, and after 3 months of treatment. The researcher performs a repeated measures ANOVA to determine if there is a significant difference in mean blood pressure across the three time points.",
        "Formula": r"""
                    $$ F = \frac{MS_{between}}{MS_{error}} $$ 
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
        "Explanation": "Permutation MANOVA or Non-Parametric MANOVA These tests are used to compare the means of multiple dependent variables across two or more independent groups when the assumptions of traditional MANOVA are not met. They do not assume a specific distribution for the data.",
        "Example": "A researcher wants to compare the effects of three different diets on both weight loss and cholesterol levels, but the data does not follow a normal distribution. The researcher performs a Permutation MANOVA to determine if there are significant differences in the combined dependent variables (weight loss and cholesterol levels) across the three diet groups.",
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
                    $$ Q = \sum_{j=1}^{k} \left( \frac{(X_j - \bar{X})^2}{\sigma^2} \right) $$ 
                    Where: 
                    - :orange[$Q$] is the test statistic, 
                    - :orange[$X_j$] is the observed frequency for group :orange[$j$], 
                    - :orange[$\bar{X}$] is the mean frequency, and 
                    - :orange[$\sigma^2$] is the variance. 
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
                    $$ P = \sum_{\text{all tables with } \geq O_{ij}} \frac{(N!)}{\prod_{i,j} (n_{ij}!)} \prod_{i,j} (p_{ij})^{n_{ij}} $$ 
                    Where: 
                    - :orange[$P$] is the p-value, 
                    - :orange[$O_{ij}$] is the observed frequency for cell :orange[$(i,j)$], 
                    - :orange[$n_{ij}$] is the expected frequency for cell :orange[$(i,j)$], and 
                    - :orange[$p_{ij}$] is the probability for cell :orange[$(i,j)$]. 
                    - The p-value :orange[$P$] is then compared to a significance level to determine significance.
                    """,
    },
    # Association/Correlation Tests
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
                    $$ r = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^{n} (Y_i - \bar{Y})^2}} $$ 
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
                    $$ r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} $$ 
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
                    $$ \chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$ 
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
        "Explanation": "Simple Linear Regression This test is used to model the relationship between a continuous dependent variable and a single continuous independent variable. It assumes that the data is continuous, follows a normal distribution, and that the relationship between the variables is linear.",
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
        "Explanation": "Multiple Linear Regression This test is used to model the relationship between a continuous dependent variable and multiple continuous independent variables. It assumes that the data is continuous, follows a normal distribution, and that the relationship between the variables is linear.",
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
                    $$ \log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X $$ 
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
                    $$ \log\left(\frac{p_j}{p_k}\right) = \beta_{0j} + \beta_{1j} X $$ 
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
                    $$ \log\left(\frac{P(Y \leq j)}{P(Y > j)}\right) = \beta_{0j} + \beta_{1j} X $$ 
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
                    $$ \text{Sensitivity} = \frac{TP}{TP + FN} $$
                    $$ \text{Specificity} = \frac{TN}{TN + FP} $$
                    $$ \text{Positive Predictive Value (PPV)} = \frac{TP}{TP + FP} $$
                    $$ \text{Negative Predictive Value (NPV)} = \frac{TN}{TN + FN} $$
                    $$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
                    $$ \text{Likelihood ratio for a positive test (LR+)} = \frac{\text{Sensitivity}}{1 - \text{Specificity}} $$
                    $$ \text{Likelihood ratio for a negative test (LR-)} = \frac{1 - \text{Sensitivity}}{\text{Specificity}} $$
                    $$ \text{F1 Score} = 2 \times \frac{\text{PPV} \times \text{Sensitivity}}{\text{PPV} + \text{Sensitivity}} $$
                    $$ \text{Diagnostic Odds Ratio (DOR)} = \frac{LR+}{LR-} $$
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
        "Explanation": "Likelihood Ratios (LR) are used to assess the value of performing a diagnostic test. LR+ indicates how much more likely a positive test is to be found in a person with the disease than in a person without. LR- indicates how much more likely a negative test is to be found in a person with the disease than in a person without.",
        "Example": "A clinician uses the LR+ of a physical exam finding to update their post-test probability of a patient having appendicitis.",
        "Formula": r"""
                    $$ LR+ = \frac{\text{Sensitivity}}{1 - \text{Specificity}} $$
                    $$ LR- = \frac{1 - \text{Sensitivity}}{\text{Specificity}} $$
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
                    $$ \kappa = \frac{p_o - p_e}{1 - p_e} $$
                    Where:
                    - :orange[$p_o$]: Observed proportionate agreement
                    - :orange[$p_e$]: Probability of random agreement
                    """,
    },
]


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
        st.header("📖 Statistical Glossary")
        st.write("Comprehensive reference for statistical concepts.")

        with st.expander("Variable Types"):
            st.markdown("""
            :orange[**Binary / Dichotomous**]: Two exclusive categories (e.g., Male/Female, Pass/Fail).
            
            :orange[**Categorical / Nominal**]: Categories without inherent order (e.g., Eye color, Department).
            
            :orange[**Ordinal**]: Categories with a natural rank/order, but unequal distances between them (e.g., Likert scales, Education level).
            
            :orange[**Discrete (Count)**]: Numerical values that must be whole numbers (e.g., Number of teeth, Number of children).
            
            :orange[**Continuous (Scale)**]: Numerical values that can be measured with infinite precision (e.g., Height, Lab values, Temperature).
            
            :orange[**Dependent (Outcome)**]: The variable being measured/tested (the "effect").
            
            :orange[**Independent (Predictor)**]: The variable being manipulated or used to group data (the "cause").
            """)

        with st.expander("Descriptive Statistics"):
            st.markdown("""
            :orange[**Mean**]: The mathematical average (sum divided by count). Sensitive to outliers.
            
            :orange[**Median**]: The middle value when data is ordered. Better for skewed data.
            
            :orange[**Mode**]: The most frequently occurring value.
            
            :orange[**Standard Deviation (SD)**]: Measures the spread of data around the mean.
            
            :orange[**Variance**]: The square of the Standard Deviation; measures variability.
            
            :orange[**Interquartile Range (IQR)**]: The range of the middle 50% of values (Q3 - Q1).
            """)

        with st.expander("Sampling & Research Design"):
            st.markdown("""
            :orange[**Random Sampling**]: Every member of the population has an equal chance of being selected.
            
            :orange[**Stratified Sampling**]: Dividing the population into subgroups (strata) and sampling from each.
            
            :orange[**Blinding**]: Preventing participants (Single-blind) or both (Double-blind) from knowing the treatment to reduce bias.
            
            :orange[**Randomization**]: Assigning participants to groups by chance to ensure groups are comparable.
            
            :orange[**Randomized Controlled Trial (RCT)**]: The gold standard for clinical trials where participants are randomly assigned to a treatment or control group.
            
            :orange[**Cohort Study**]: A longitudinal study that follows a group of people over time to see how exposures affect outcomes.
            
            :orange[**Case-Control Study**]: A study that compares people with a condition (cases) to those without (controls) to find retrospective causes.
            
            :orange[**Cross-sectional Study**]: A "snapshot" study that analyzes data from a population at a single point in time.
            
            :orange[**Placebo Effect**]: Improvement due to the belief in a treatment rather than the treatment itself.
            """)

        with st.expander("Hypothesis Testing"):
            st.markdown("""
            :orange[**Null Hypothesis (H₀)**]: Suggests no effect or no difference exists.
            
            :orange[**Alternative Hypothesis (H₁)**]: Suggests a significant effect or difference exists.
            
            :orange[**P-value**]: Probability that the observed result happened by chance. Low p-values (<0.05) suggest rejecting the Null Hypothesis.
            
            :orange[**Alpha (α)**]: The threshold for significance (usually 0.05).
            
            :orange[**Type I Error (α)**]: Falsely rejecting a true null hypothesis (a "false positive").
            
            :orange[**Type II Error (β)**]: Failing to reject a false null hypothesis (a "false negative").
            
            :orange[**Confidence Interval (CI)**]: A range of values likely to contain the true population parameter (usually 95%).
            """)

        with st.expander("Distribution & Assumptions"):
            st.markdown("""
            :orange[**Parametric Tests**]: Assume data follows a specific distribution (usually Normal). More powerful but strict.
            
            :orange[**Non-parametric Tests**]: "Distribution-free" tests. Used for ordinal or non-normal data.
            
            :orange[**Normality**]: When data follows a bell-shaped curve (symmetrical around the mean).
            
            :orange[**Skewness**]: Measures the lack of symmetry in a distribution (Left/Right skew).
            
            :orange[**Kurtosis**]: Measures the "tailedness" or peakedness of a distribution.
            
            :orange[**Homogeneity of Variance**]: The assumption that different groups have approximately the same spread/variance.
            
            :orange[**Outlier**]: An extreme value that deviates significantly from the rest of the dataset.
            """)

        with st.expander("Effect Size & Power"):
            st.markdown("""
            :orange[**Effect Size**]: Quantitative measure of the magnitude of a phenomenon.
            
            :orange[**Statistical Power (1-β)**]: The probability that a test will correctly reject a false null hypothesis.
            
            :orange[**Cohen's d**]: Measures the distance between two means in SD units.
            
            :orange[**Eta-squared (η²)**]: Proportion of variance in the outcome explained by a predictor in ANOVA.
            
            :orange[**Cramer's V**]: Measure of association between two nominal variables.
            """)

        with st.expander("Correlation & Regression"):
            st.markdown("""
            :orange[**Correlation**]: Measures the strength and direction of a relationship.
            
            :orange[**Pearson's r**]: Linear relationship between continuous variables (-1 to +1).
            
            :orange[**Spearman's ρ**]: Monotonic relationship (Rank-based).
            
            :orange[**Regression**]: Predicting a dependent variable based on one or more predictors.
            
            :orange[**R-squared (R²)**]: Percentage of variance explained by the model.
            
            :orange[**Adjusted R²**]: R² adjusted for the number of predictors in the model.
            
            :orange[**Residuals**]: The difference between observed and predicted values.
            """)

        with st.expander("Reliability & Validity"):
            st.markdown("""
            :orange[**Reliability**]: Consistency or stability of a measurement.
            
            :orange[**Validity**]: Accuracy; whether a tool measures what it's supposed to.
            
            :orange[**Face Validity**]: Whether a test "looks like" it measures what it's supposed to.
            
            :orange[**Content Validity**]: Whether a test covers all aspects of the concept being measured.
            
            :orange[**Construct Validity**]: Whether a test truly measures the theoretical construct it claims to.
            
            :orange[**Criterion Validity**]: How well one measure predicts an outcome based on another measure.
            
            :orange[**Cronbach's Alpha**]: Measures internal consistency (0 to 1).
            
            :orange[**Intraclass Correlation (ICC)**]: Used to measure reliability of ratings for grouped data.
            
            :orange[**Inter-rater Reliability**]: Agreement between different observers (e.g., Cohen's Kappa).
            """)

        with st.expander("Clinical/Biostatistics"):
            st.markdown("""
            :orange[**Sensitivity**]: Ability to correctly identify those *with* a condition (True Positive Rate).
            
            :orange[**Specificity**]: Ability to correctly identify those *without* a condition (True Negative Rate).
            
            :orange[**Positive Predictive Value (PPV)**]: Probability that a person with a positive test actually has the disease.
            
            :orange[**Negative Predictive Value (NPV)**]: Probability that a person with a negative test is actually healthy.
            
            :orange[**Likelihood Ratio (LR+/LR-)**]: How much a test result changes the odds of having a condition.
            
            :orange[**Odds Ratio (OR)**]: Odds of an event occurring in one group vs. another.
            
            :orange[**Relative Risk (RR)**]: Risk of an event in an exposed group vs. unexposed group.
            
            :orange[**Number Needed to Treat (NNT)**]: Number of patients who need to be treated to prevent one additional bad outcome.
            
            :orange[**Number Needed to Harm (NNH)**]: Number of patients who need to be exposed to a risk factor to cause one additional bad outcome.
            
            :orange[**Forest Plot**]: Visual representation of the results of multiple studies in a meta-analysis.
            """)

        with st.expander("Survival Analysis"):
            st.markdown("""
            :orange[**Censoring**]: Occurs when we have incomplete information about the survival time of an individual.
            
            :orange[**Kaplan-Meier Curve**]: A non-parametric statistic used to estimate the survival function from lifetime data.
            
            :orange[**Hazard Ratio (HR)**]: The ratio of the hazard rates corresponding to the conditions described by two levels of an explanatory variable.
            """)

        with st.expander("Advanced Modeling"):
            st.markdown("""
            :orange[**Multicollinearity**]: When independent variables in a regression model are highly correlated.
            
            :orange[**Interaction Effect**]: When the effect of one independent variable on the outcome depends on the level of another independent variable.
            
            :orange[**Confounding Variable**]: An outside influence that changes the effect of a dependent and independent variable.
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

        Objective = st.selectbox(
            "What is your goal?",
            [
                "Comparison",
                "Association/Correlation",
                "Prediction",
                "Diagnostic Accuracy",
            ],
        )

        # =========================
        # VARIABLES
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
        # DESIGN
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
        # USER INPUT OBJECT
        # =========================
        user_input = {
            "Objective": Objective,
            "Dependent_Variable": Dependent_Variable,
            "Independent_Variable": Independent_Variable,
            "Groups": Groups,
            "Relation": Relation,
            "Distribution": Distribution,
        }

        # =========================
        # FIND TEST
        # =========================
        if st.button("Find My Test", use_container_width=True):
            st.session_state.results = find_matching_tests(user_input)
            # Clear open states when new search is performed
            st.session_state.open_tests = set()

    with col_right:
        # Display results from session state
        if st.session_state.results is not None:
            if st.session_state.results:
                st.success("Recommended Statistical Test(s):")

                for test in st.session_state.results:
                    rule = next((r for r in rules if r["name"] == test), None)
                    if rule:
                        # Button to toggle open/close
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

                        # Show content if test is in open_tests
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
                            st.markdown("---")

            else:
                st.error(
                    "No matching statistical test found. Try adjusting your selections."
                )
        else:
            st.info("Results will appear here once you click 'Find My Test'.")

    # =========================
    # FLOWCHART MODE
    # =========================

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
            p = \frac{{1}}{{1 + e^{{-({beta0:.2f} + {beta1:.2f}x)}}}}
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

    else:
        st.info("Interactive widget coming soon for this test.")


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
