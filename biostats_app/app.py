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
                    $$ t = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}}$$ 
                    Where: 
                    - $\bar{x}$ is the sample mean,   
                    - $\mu_0$ is the population mean, 
                    - $s$ is the sample standard deviation, 
                    - $n$ is the sample size 
                    - $\frac{s}{\sqrt{n}}$ is the standard error of the mean.
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
                    $$ z = \frac{\bar{x} - \mu_0}{\frac{\sigma}{\sqrt{n}}} $$
                    where:
                    - $\bar{x}$ is the sample mean,
                    - $\mu_0$ is the population mean,
                    - $\sigma$ is the population standard deviation,
                    - $n$ is the sample size.
                    - The denominator $\frac{\sigma}{\sqrt{n}}$ is the standard error of the mean.
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
                    $$ z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1 - p_0)}{n}}} $$
                    where: 
                    - $\hat{p}$ is the sample proportion,  
                    - $p_0$ is the population proportion, and   
                    - $n$ is the sample size.  
                    - The denominator $\sqrt{\frac{p_0(1 - p_0)}{n}}$ is the standard error of the proportion.
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
                    - $R_i$ is the rank of the absolute difference between the observed value  
                    - $X_i$ and the hypothesized median $M_0$, and  
                    - $sgn(X_i - M_0)$ is the sign function that indicates whether $X_i$ is above, below, or equal to $M_0$.  
                    - The test statistic $W$ is then compared to a critical value from the Wilcoxon signed-rank distribution to determine significance.
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
                    $$ t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} $$ 
                    Where:  
                    - $\bar{x}_1$ and $\bar{x}_2$ are the sample means of the two groups,  
                    - $n_1$ and $n_2$ are the sample sizes of the two groups, and  
                    - $s_p$ is the pooled standard deviation calculated as: $$ s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}} $$ 
                    where  
                    - $s_1^2$ and $s_2^2$ are the sample variances of the two groups.
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
                    $$ t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} $$ 
                    Where: 
                    - $\bar{x}_1$ and $\bar{x}_2$ are the sample means of the two groups, 
                    - $n_1$ and $n_2$ are the sample sizes of the two groups, and 
                    - $s_1^2$ and $s_2^2$ are the sample variances of the two groups.
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
                    $$ t = \frac{\bar{d}}{s_d / \sqrt{n}} $$ 
                    Where: 
                    - $\bar{d}$ is the mean of the differences between paired observations, 
                    - $s_d$ is the standard deviation of the differences, and 
                    - $n$ is the number of pairs. 
                    - The test statistic $t$ is then compared to a critical value from the t-distribution with $n-1$ degrees of freedom to determine significance.
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
                    $$ F = \frac{MS_{between}}{MS_{within}} $$ 
                    Where: 
                    - $F$ is the F-statistic, 
                    - $MS_{between}$ is the mean square between groups, and 
                    - $MS_{within}$ is the mean square within groups. 
                    Mean squares are calculated as: 
                    - $$ MS_{between} = \frac{SS_{between}}{df_{between}} $$ and 
                    - $$ MS_{within} = \frac{SS_{within}}{df_{within}} $$ 
                    where:
                    - $SS$ is the sum of squares and $df$ is the degrees of freedom for between and within groups.
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
                    - $W$ is the test statistic, 
                    - $R_i$ is the rank of the $i$-th difference, and 
                    - $n$ is the number of pairs. 
                    - The test statistic $W$ is then compared to a critical value from the Wilcoxon signed-rank distribution to determine significance. R is calculated by ranking the absolute differences between paired observations and assigning ranks accordingly, with ties receiving average ranks. The sign of the difference is also considered when calculating the test statistic.
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
                    - $U$ is the test statistic, 
                    - $R_i$ is the rank of the $i$-th observation in the combined dataset, 
                    - $n_1$ is the number of observations in group 1, and 
                    - $n_2$ is the number of observations in group 2. 
                    - The test statistic $U$ is then compared to a critical value from the Mann-Whitney U distribution to determine significance. R is calculated by ranking all observations from both groups together and assigning ranks accordingly, with ties receiving average ranks. The U statistic is calculated based on the sum of ranks for one of the groups and adjusted for the number of observations in that group.
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
                    - $H$ is the test statistic, 
                    - $R_i$ is the sum of ranks for group $i$, 
                    - $n_i$ is the number of observations in group $i$, 
                    - $N$ is the total number of observations, and 
                    - $k$ is the number of groups. 
                    - The test statistic $H$ is then compared to a critical value from the chi-square distribution with $k-1$ degrees of freedom to determine significance.
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
                    - $F$ is the F-statistic, 
                    - $MS_{between}$ is the mean square between groups (calculated based on the variability of the group means), and 
                    - $MS_{error}$ is the mean square error (calculated based on the variability of observations within groups). 
                    - The test statistic $F$ is then compared to a critical value from the F-distribution with appropriate degrees of freedom to determine significance.
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
                    $$ \Lambda = \frac{|\mathbf{E}|}{|\mathbf{E} + \mathbf{H}|} $$ 
                    Where: 
                    - $\Lambda$ is the test statistic (Wilks' Lambda), 
                    - $\mathbf{E}$ is the error sum of squares and cross-products matrix, and 
                    - $\mathbf{H}$ is the hypothesis sum of squares and cross-products matrix. 
                    - The test statistic $\Lambda$ is then transformed into an F-statistic for significance testing.
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
                    $$ \chi^2 = \frac{12}{N(N+1)} \sum_{j=1}^{k} R_j^2 - 3N(N+1) $$ 
                    Where: 
                    - $\chi^2$ is the test statistic, 
                    - $N$ is the number of subjects, 
                    - $k$ is the number of treatments, and 
                    - $R_j$ is the sum of ranks for the $j$-th treatment. 
                    - The test statistic $\chi^2$ is then compared to a critical value from the chi-square distribution with $k-1$ degrees of freedom to determine significance.
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
                    $$ F = \frac{MS_{between}}{MS_{error}} $$ 
                    Where: 
                    - $F$ is the F-statistic, 
                    - $MS_{between}$ is the mean square between groups (calculated based on the variability of the group means), and 
                    - $MS_{error}$ is the mean square error (calculated based on the variability of observations within groups). 
                    - The test statistic $F$ is then compared to a critical value from the F-distribution with appropriate degrees of freedom to determine significance.
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
                    $$ \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} $$ 
                    Where: 
                    - $\chi^2$ is the test statistic, 
                    - $O_i$ is the observed frequency for category $i$, and 
                    - $E_i$ is the expected frequency for category $i$. 
                    - The test statistic $\chi^2$ is then compared to a critical value from the chi-square distribution with $k-1$ degrees of freedom to determine significance.
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
                    $$ \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} $$ 
                    Where: 
                    - $\chi^2$ is the test statistic, 
                    - $O_i$ is the observed frequency for cell $i$, and 
                    - $E_i$ is the expected frequency for cell $i$. 
                    - The test statistic $\chi^2$ is then compared to a critical value from the chi-square distribution with $(r-1)(c-1)$ degrees of freedom to determine significance.
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
                    $$ \chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} $$ 
                    Where: 
                    - $\chi^2$ is the test statistic, 
                    - $O_i$ is the observed frequency for cell $i$, and 
                    - $E_i$ is the expected frequency for cell $i$. 
                    - The test statistic $\chi^2$ is then compared to a critical value from the chi-square distribution with $(r-1)(c-1)$ degrees of freedom to determine significance.
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
                    - $Q$ is the test statistic, 
                    - $X_j$ is the observed frequency for group $j$, 
                    - $\bar{X}$ is the mean frequency, and 
                    - $\sigma^2$ is the variance. 
                    - The test statistic $Q$ is then compared to a critical value from the chi-square distribution with $k-1$ degrees of freedom to determine significance.
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
                    - $P$ is the p-value, 
                    - $O_{ij}$ is the observed frequency for cell $(i,j)$, 
                    - $n_{ij}$ is the expected frequency for cell $(i,j)$, and 
                    - $p_{ij}$ is the probability for cell $(i,j)$. 
                    - The p-value $P$ is then compared to a significance level to determine significance.
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
                    - $r$ is the Pearson correlation coefficient, 
                    - $X_i$ and $Y_i$ are the individual data points for variables X and Y, 
                    - $\bar{X}$ and $\bar{Y}$ are the means of variables X and Y, and 
                    - $n$ is the number of data points. 
                    - The test statistic $r$ is then compared to a critical value from the Pearson correlation distribution to determine significance.
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
                    - $r_s$ is the Spearman rank correlation coefficient, 
                    - $d_i$ is the difference in ranks for each pair of observations, 
                    - $n$ is the number of observations, and 
                    - the sum is over all pairs. 
                    - The test statistic $r_s$ is then compared to a critical value from the Spearman rank correlation distribution to determine significance.
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
                    - $\chi^2$ is the test statistic, 
                    - $O_{ij}$ is the observed frequency for cell $(i,j)$, and 
                    - $E_{ij}$ is the expected frequency for cell $(i,j)$. 
                    - The test statistic $\chi^2$ is then compared to a critical value from the chi-square distribution with $(r-1)(c-1)$ degrees of freedom to determine significance.
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
                    $$ r_{pb} = \frac{M_1 - M_0}{s} \sqrt{\frac{n_1 n_0}{n(n-1)}} $$ 
                    Where: 
                    - $r_{pb}$ is the Point-Biserial correlation coefficient, 
                    - $M_1$ and $M_0$ are the means of the continuous variable for the two groups, 
                    - $s$ is the standard deviation of the continuous variable, 
                    - $n_1$ and $n_0$ are the sample sizes of the two groups, and 
                    - $n$ is the total sample size. 
                    - The test statistic $r_{pb}$ is then compared to a critical value from the Pearson correlation distribution to determine significance.
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
                    - $Y$ is the dependent variable, 
                    - $X$ is the independent variable, 
                    - $\beta_0$ is the intercept, 
                    - $\beta_1$ is the slope coefficient, and 
                    - $\epsilon$ is the error term. 
                    - The coefficients $\beta_0$ and $\beta_1$ are estimated using the least squares method, and the significance of the predictor is determined by testing if $\beta_1$ is significantly different from zero.
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
                    - $Y$ is the dependent variable, 
                    - $X_1, X_2, \ldots, X_k$ are the independent variables, 
                    - $\beta_0$ is the intercept, 
                    - $\beta_1, \beta_2, \ldots, \beta_k$ are the slope coefficients, and 
                    - $\epsilon$ is the error term. 
                    - The coefficients $\beta_0, \beta_1, \beta_2, \ldots, \beta_k$ are estimated using the least squares method, and the significance of each predictor is determined by testing if its corresponding coefficient is significantly different from zero.
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
                    - $p$ is the probability of the dependent variable being 1 (e.g., passing the exam), 
                    - $X$ is the independent variable, 
                    - $\beta_0$ is the intercept, and 
                    - $\beta_1$ is the slope coefficient. 
                    - The coefficients $\beta_0$ and $\beta_1$ are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if $\beta_1$ is significantly different from zero.
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
                    - $p_j$ is the probability of the dependent variable being in category $j$, 
                    - $p_k$ is the probability of the dependent variable being in the reference category $k$, 
                    - $X$ is the independent variable, 
                    - $\beta_{0j}$ is the intercept for category $j$, and 
                    - $\beta_{1j}$ is the slope coefficient for category $j$. 
                    - The coefficients $\beta_{0j}$ and $\beta_{1j}$ are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if $\beta_{1j}$ is significantly different from zero for each category.
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
                    - $P(Y \leq j)$ is the probability of the dependent variable being in category $j$ or lower, 
                    - $P(Y > j)$ is the probability of the dependent variable being in a category higher than $j$, 
                    - $X$ is the independent variable, 
                    - $\beta_{0j}$ is the intercept for category $j$, and 
                    - $\beta_{1j}$ is the slope coefficient for category $j$. 
                    - The coefficients $\beta_{0j}$ and $\beta_{1j}$ are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if $\beta_{1j}$ is significantly different from zero for each category.
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
                    - $\lambda$ is the expected count of the dependent variable, 
                    - $X_1$ and $X_2$ are the independent variables, 
                    - $\beta_{0}$ is the intercept, and 
                    - $\beta_{1}$ and $\beta_{2}$ are the slope coefficients for each independent variable. 
                    - The coefficients $\beta_{0}$, $\beta_{1}$, and $\beta_{2}$ are estimated using maximum likelihood estimation, and the significance of the predictors is determined by testing if they are significantly different from zero.
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

    st.set_page_config(page_title="Statistical Test Finder")

    st.title("🔬 Statistical Test Finder")

    st.write(
        "Select your study characteristics to identify the appropriate statistical test."
    )

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
        ],
    )

    # =========================
    # VARIABLES
    # =========================
    st.subheader("2. Variables")

    Dependent_Variable = st.selectbox(
        "Dependent Variable Type",
        [
            "Binary/Dichotomous",
            "Categorical",
            "Ordinal",
            "Discrete",
            "Continuous",
            "Multiple Continuous",
        ],
    )

    Independent_Variable = st.selectbox(
        "Independent Variable Type",
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
    if st.button("Find My Test"):
        st.session_state.results = find_matching_tests(user_input)
        # Clear open states when new search is performed
        st.session_state.open_tests = set()

    # Initialize open_tests in session state
    if "open_tests" not in st.session_state:
        st.session_state.open_tests = set()

    # Display results from session state
    if hasattr(st.session_state, "results") and st.session_state.results:
        st.success("Recommended Statistical Test(s):")

        for test in st.session_state.results:
            rule = next((r for r in rules if r["name"] == test), None)
            if rule:
                # Button to toggle open/close
                is_open = test in st.session_state.open_tests
                btn_label = f"▶ {test}" if not is_open else f"▼ {test}"

                if st.button(btn_label, key=f"btn_{test}"):
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

    elif hasattr(st.session_state, "results") and not st.session_state.results:
        st.error("No matching statistical test found. Try adjusting your selections.")
    # =========================
    # FLOWCHART MODE
    # =========================

    st.divider()

    st.header("🌳 Interactive Statistical Flowchart")

    st.write(
        "Expand the branches below to navigate statistical test selection visually."
    )

    build_tree(rules, FIELDS)


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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {result.pvalue:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"Degrees of Freedom = {dof}")

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

        st.write(f"p-value = {result.pvalue:.5f}")

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

        st.write(f"p-value = {result.pvalue:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        st.write(f"p-value = {p:.5f}")

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

        x = np.linspace(-10, 10, 500)

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


def build_tree(rule_subset, fields, level=0):

    # No more fields → show tests
    if not fields:

        for rule in rule_subset:
            st.success(f"✅ {rule['name']}")

        return

    current_field = fields[0]
    # Skip meaningless all-any branches
    all_any = all(rule[current_field] == "any" for rule in rule_subset)

    if all_any:
        build_tree(rule_subset, fields[1:], level + 1)
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

        with st.expander(f"{current_field}: {label}"):

            build_tree(subrules, fields[1:], level + 1)


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
