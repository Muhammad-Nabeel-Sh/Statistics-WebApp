# Mapping from test finder rule names to sample size estimation analysis types
TEST_TO_SS_TYPE = {
    "One-sample t-test": "One-sample Mean (t/z-test)",
    "One-sample z-test": "One-sample Mean (t/z-test)",
    "Student's t-test (Independent)": "Two Independent Means (t-test)",
    "Welch's t-test (Independent, Unequal Variances)": "Two Independent Means (t-test)",
    "Paired t-test": "Paired Means (t-test)",
    "One-sample Proportion Test (Binomial Test)": "One-sample Proportion",
    "One-way ANOVA": "One-way ANOVA",
    "Pearson Correlation": "Correlation (Pearson)",
    "Multiple Linear Regression": "Multiple Linear Regression",
    "Logistic Regression": "Logistic Regression",
    "Chi-Square Test of Independence": "Chi-Square Test",
    "Chi-Square Goodness-of-Fit Test": "Chi-Square Test",
    "Mann-Whitney U Test": "Mann-Whitney / Wilcoxon (Non-parametric)",
    "One-sample Wilcoxon Signed-Rank Test": "Wilcoxon Signed-Rank (paired)",
    "Log-Rank Test": "Log-Rank Test (Survival)",
    "Cox Proportional Hazards Regression": "Cox Regression",
    "Kruskal-Wallis Test": "Kruskal-Wallis Test",
    "Friedman Test": "Friedman Test",
    "McNemar's Test": "McNemar's Test",
    "Fisher's Exact Test": "Fisher's Exact Test",
    "MANOVA": "MANOVA (Multivariate ANOVA)",
    "Permutation MANOVA or Non-Parametric MANOVA": "MANOVA (Multivariate ANOVA)",
    "Repeated Measures ANOVA": "Repeated Measures ANOVA",
    "Two-way ANOVA": "Two-way / Factorial ANOVA",
    "Spearman Rank Correlation": "Correlation (Spearman)",
    "Wilcoxon Signed-Rank Test": "Wilcoxon Signed-Rank (paired)",
    "Simple Linear Regression": "Multiple Linear Regression",
    "Cochran's Q Test": "Chi-Square Test",
    "Point-Biserial Correlation": "Correlation (Point-Biserial)",
    "Chi-Square Test": "Chi-Square Test",
    "Negative Binomial Regression": "Negative Binomial Regression",
    "Sensitivity and Specificity": "Diagnostic Accuracy",
    "Kendall's Tau-b": "Correlation (Kendall's Tau)",
    "kaplan-Meier Survival Analysis": "Kaplan-Meier Survival Analysis",
    "Logistic Regression (Binary Outcome)": "Logistic Regression",
    "Logistic Regression (Multinomial Outcome)": "Logistic Regression",
    "Logistic Regression (Ordinal Outcome)": "Logistic Regression",
    "Cox Proportional Hazards Regression": "Cox Regression",
     "Binomial Test": "One-sample Proportion",
     "Multinomial Test": "Chi-Square Test",
     "Sign Test (One-sample)": "Wilcoxon Signed-Rank (paired)",
     "Sign Test (Paired)": "Wilcoxon Signed-Rank (paired)",
     "Mood's Median Test": "Kruskal-Wallis Test",
     "Runs Test for Randomness": "Precision-based (CI Width)",
     "Equivalence Test (TOST) - Two Independent Samples": "Equivalence / Non-Inferiority",
     "F-Test for Two Variances": "Two Independent Means (t-test)",
     "Poisson Goodness-of-Fit Test": "Chi-Square Test",
 }


rules = [
    # Comparison Tests
    # One-sample tests
    {
        "name": "One-sample t-test",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": ["any", "Dependent", "Independent"],
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
        "Decision Rules": r"""
                    -  Reject the null hyposthesis $H_0$ if the absolute value of the test statistic :orange[$|t|$] is greater than the critical t-value from the t-distribution with :orange[$n-1$] degrees of freedom at the chosen significance level (e.g., α = 0.05) :orange[$|t| > t_{critical}$].
                    -  Fail to reject the null hypothesis if :orange[$|t|$] is less than or equal to the critical t-value :orange[$|t| \leq t_{critical}$].
                    -  Reject the null hyosthesis $H_0$ if the p-value is less than the chosen significance level (e.g., α = 0.05) :orange[$p < \alpha$].
                    -  Effect size (Cohen's d) can be calculated as :orange[$ d = \dfrac{\bar{x} - \mu_0}{s} $], where :orange[$d$] is the standardized mean difference, providing a measure of the magnitude of the effect independent of sample size.
        """,
    },
    {
        "name": "One-sample z-test",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": ["any", "Dependent", "Independent"],
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
        "Decision Rules": r"""
                    - Reject the null hypothesis if the absolute value of the test statistic :orange[$|z|$] is greater than the critical z-value from the standard normal distribution at the chosen significance level (e.g., α = 0.05) :orange[$|z| > z_{critical}$].
                    - Fail to reject the null hypothesis if :orange[$|z|$] is less than or equal to the critical z-value :orange[$|z| \leq z_{critical}$].
                    - Reject the null hypothesis if the p-value is less than the chosen significance level (e.g., α = 0.05) :orange[$p < \alpha$].
                    - Effect size (Cohen's d) can be calculated as :orange[ $ d = \dfrac{\bar{x} - \mu_0}{\sigma} $], where :orange[$d$] is the standardized mean difference, providing a measure of the magnitude of the effect independent of sample size.
        """,
    },
    {
        "name": "One-sample Proportion Test (Binomial Test)",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "One-Sample Proportion Test (Binomial Test) This test is used to determine whether the proportion of successes in a single sample is significantly different from a known or hypothesized population proportion. It is typically used when analyzing categorical data, such as the proportion of patients who respond to a treatment compared to a known response rate.",
        "Example": "A clinical trial tests a new drug and finds that 18 out of 30 patients respond to the treatment. The researcher wants to determine if this response rate is significantly different from the known response rate of 50% (0.5) for existing treatments. The researcher performs a one-sample proportion test (binomial test) to compare the observed proportion of 0.6 (18/30) against the known population proportion of 0.5.",
        "Formula": r"""
                    $$ z = \dfrac{\hat{p} - p_0}{\sqrt{\dfrac{p_0(1 - p_0)}{n}}} $$
                    $$ P(X = k) = \binom{n}{k} p_0^k (1 - p_0)^{n-k} $$
                    where: 
                    - :orange[$\hat{p}$] is the sample proportion,  
                    - :orange[$p_0$] is the population proportion, and   
                    - :orange[$n$] is the sample size.  
                    - :orange[$P(X = k)$] is the probability of observing exactly :orange[$k$] successes in :orange[$n$] trials under the null hypothesis, calculated using the binomial distribution.
                    - :orange[$z$] is the test statistic.
                    - :orange[$k$] is the number of successes observed in the sample.
                    - The denominator :orange[$\sqrt{\dfrac{p_0(1 - p_0)}{n}}$] is the standard error of the proportion.
                    """,
    },
    {
        "name": "Binomial Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Normal", "Non-normal", "any"],
        "Explanation": "Binomial Test (Exact) This test is used to determine whether the observed proportion of successes in a single sample differs from a hypothesized population proportion. Unlike the one-sample proportion z-test, the exact binomial test computes the p-value directly from the binomial distribution without relying on a normal approximation, making it more accurate for small sample sizes. It is appropriate when the data are binary (success/failure) and each trial is independent.",
        "Example": "A researcher wants to test if a coin is fair. They flip it 20 times and observe 15 heads. The exact binomial test calculates the probability of observing 15 or more heads (or 5 or fewer) under the null hypothesis that the true probability of heads is 0.5, providing an exact p-value without relying on the normal approximation.",
        "Formula": r"""
                    $$ P(X = k) = \binom{n}{k} p_0^k (1 - p_0)^{n-k} $$
                    $$ \text{p-value} = \sum_{j \leq k} \binom{n}{j} p_0^j (1 - p_0)^{n-j} \quad \text{(one-sided lower)} $$
                    $$ \text{p-value} = \sum_{j \geq k} \binom{n}{j} p_0^j (1 - p_0)^{n-j} \quad \text{(one-sided upper)} $$
                    $$ \text{p-value} = \sum_{j: P(X=j) \leq P(X=k)} \binom{n}{j} p_0^j (1 - p_0)^{n-j} \quad \text{(two-sided)} $$
                    where:
                    - :orange[$k$] is the observed number of successes,
                    - :orange[$n$] is the total number of trials,
                    - :orange[$p_0$] is the hypothesized probability of success under :orange[$H_0$],
                    - :orange[$\binom{n}{k}$] is the binomial coefficient.
                    """,
        "Decision Rules": r"""
                    - Reject the null hypothesis :orange[$H_0$] if the exact p-value is less than the chosen significance level (e.g., :orange[$\alpha = 0.05$]).
                    - The two-sided p-value sums the probabilities of all outcomes as extreme or more extreme than the observed outcome.
                    - This test does not rely on a normal approximation, making it valid for any sample size.
                    - Effect size can be reported as the observed proportion :orange[$\hat{p} = k / n$] with a confidence interval (Clopper-Pearson exact interval).
        """,
    },
    {
        "name": "One-sample Wilcoxon Signed-Rank Test",
        "Objective": "Comparison",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": "Non-normal",
        "Explanation": "One-Sample Wilcoxon Signed-Rank Test This non-parametric test is used to determine whether the median of a single sample is significantly different from a known or hypothesized population median. It is typically used when the data is ordinal or continuous but does not follow a normal distribution.",
        "Example": "A researcher wants to test if the median pain score of patients after a treatment is significantly different from a known median pain score of 5 on a 10-point scale. The researcher collects pain scores from 30 patients and performs a one-sample Wilcoxon signed-rank test to compare the sample median against the known population median of 5.",
        "Formula": r"""
                    $$ W = \min(W^+, W^-) $$
                    $$ W^+ = \sum_{D_i>0} \text{Rank}(|D_i|) $$
                    $$ W^- = \sum_{D_i<0} \text{Rank}(|D_i|) $$
                    $$ D_i = X_i - \theta_0 $$
                    $$ sgn(X_i - \theta_0) = \begin{cases} +1 & \text{if } X_i > \theta_0 \\ -1 & \text{if } X_i < \theta_0 \\ 0 & \text{if } X_i = \theta_0 \end{cases} $$
                    Where:  
                    - :orange[$D_i$] is the absolute difference between the observed value :orange[$X_i$] and the hypothesized median :orange[$\theta_0$] for the :orange[$i$]-th observation,
                    - :orange[$X_i$] is the observed value for the :orange[$i$]-th observation, 
                    - :orange[$\theta_0$] is the hypothesized median
                    - :orange[$sgn(X_i - \theta_0)$] is the sign function that indicates whether :orange[$X_i$] is above, below, or equal to :orange[$\theta_0$].  
                    - The test statistic :orange[$W$] is then compared to a critical value from the Wilcoxon signed-rank distribution to determine significance.

                    Large Sample Approximation
                    $$ z = \frac{W - \mu_W}{\sigma_W} $$
                    $$ \mu_W = \frac{n(n+1)}{4} $$
                    $$ \sigma_W = \sqrt{\frac{n(n+1)(2n+1)}{24}}  \quad \text{if no ties} $$
                    $$ \sigma_W = \sqrt{\frac{n(n+1)(2n+1)}{24} - \sum_{t} \frac{t(t^2 - 1)}{48}} \quad \text{if ties exist} $$
                    $$ \sigma_W = \sqrt{\frac{n(n+1)(2n+1)-\frac{1}{2}\sum^g_{j=1}{(t^3_j-t_j)}}{24}} \quad \text{if ties exist} $$
                    Where:
                    - :orange[$n$] is the number of non-zero differences,
                    - :orange[$t$] is the number of tied ranks for a particular value,
                    - :orange[$g$] is the number of groups of tied ranks.
                    - :orange[$\mu_W$] is the mean of the test statistic under the null hypothesis,
                    - :orange[$\sigma_W$] is the standard deviation of the test statistic under the null hypothesis, adjusted for ties if necessary.
                    - The test statistic :orange[$z$] is then compared to a standard normal distribution to determine significance when the sample size is large (typically n > 20).
                    - The large sample approximation is used when the sample size is sufficiently large, allowing the distribution of the test statistic to be approximated by a normal distribution, which simplifies the calculation of p-values and critical values for hypothesis testing.
                    """,
        "Decision Rules": r"""
                    - Reject the null hypothesis if the test statistic :orange[$W$] is less than or equal to the critical value from the Wilcoxon signed-rank distribution for a given significance level (e.g., α = 0.05) :orange[$W \leq W_{critical}$]. 
                    - For large samples, reject the null hypothesis if the z-score is greater than or equal to the critical z-value corresponding to the chosen significance level :orange[$ |z| \geq z_{critical} $].
                    - Fail to reject the null hypothesis if the test statistic :orange[$W$] is greater than the critical value, or if the z-score is less than the critical z-value (for larger samples).
                    - Effect size (r) can be calculated as: :orange[$$ r = \dfrac{z}{\sqrt{n}} $$]""",
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
        "name": "Multinomial Test",
        "Objective": "Comparison",
        "Dependent_Variable": "Categorical",
        "Independent_Variable": "None",
        "Groups": "1",
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["Non-normal", "Normal", "any"],
        "Explanation": "Multinomial Test This test is used to determine whether the observed proportions across multiple categories differ from a hypothesized distribution. It is an extension of the binomial test to situations with more than two categories. Unlike the Chi-Square Goodness-of-Fit test (which uses a large-sample approximation), the exact multinomial test computes the p-value by enumerating all possible outcomes consistent with the observed total sample size, summing the probabilities of all outcomes that are as extreme or more extreme than the observed outcome. It is the gold standard when sample sizes are small.",
        "Example": "A researcher expects that genotypes AA, Aa, and aa occur in a 1:2:1 ratio according to Mendelian inheritance. Among 20 offspring, they observe 8 AA, 10 Aa, and 2 aa. The multinomial test determines whether these observed counts significantly deviate from the expected 1:2:1 ratio, providing an exact p-value.",
        "Formula": r"""
                    $$ P(\mathbf{X} = \mathbf{x}) = \dfrac{n!}{x_1! \, x_2! \, \ldots \, x_k!} \, p_1^{x_1} p_2^{x_2} \ldots p_k^{x_k} $$
                    $$ \text{p-value} = \sum_{\mathbf{y}: \, P(\mathbf{Y}) \leq P(\mathbf{x})} \dfrac{n!}{y_1! \, y_2! \, \ldots \, y_k!} \, p_1^{y_1} p_2^{y_2} \ldots p_k^{y_k} $$
                    where:
                    - :orange[$n$] is the total number of trials,
                    - :orange[$k$] is the number of categories,
                    - :orange[$x_i$] is the observed count in category :orange[$i$],
                    - :orange[$p_i$] is the hypothesized probability for category :orange[$i$] under :orange[$H_0$],
                    - :orange[$\sum_{i=1}^{k} p_i = 1$] and :orange[$\sum_{i=1}^{k} x_i = n$].
                    """,
        "Decision Rules": r"""
                    - Reject the null hypothesis :orange[$H_0$] if the exact p-value is less than the chosen significance level (e.g., :orange[$\alpha = 0.05$]).
                    - The p-value sums the multinomial probabilities of all possible outcomes that are no more probable than the observed outcome.
                    - For large sample sizes, the Chi-Square Goodness-of-Fit test can be used as an approximation.
                    - This test makes no distributional assumptions beyond random sampling and independent trials.
        """,
    },
    # Two-sample tests (Parametric)
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
                    $$ MS_{between} = \dfrac{SS_{between}}{df_{between}} $$
                    $$ MS_{within} = \dfrac{SS_{within}}{df_{within}} $$
                    $$ SS_{between} = \sum_{i=1}^{k} n_i (\bar{X}_i - \bar{X})^2 $$
                    $$ SS_{within} = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2 $$
                    $$ df_{between} = k - 1 $$
                    $$ df_{within} = N - k $$
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
        "Post-Hoc": "Tukey HSD, Bonferroni, Holm-Bonferroni, Šidák, Scheffé, Dunnett, Games-Howell, Fisher LSD, Newman-Keuls",
    },
    # More two-sample tests (Parametric)
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
                    $$ MS_{between} = \dfrac{SS_{between}}{df_{between}} $$
                    $$ MS_{error} = \dfrac{SS_{error}}{df_{error}} $$
                    $$ SS_{between} = \sum_{i=1}^{k} n_i (\bar{X}_i - \bar{X})^2 $$
                    $$ SS_{error} = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2 $$
                    $$ df_{between} = k - 1 $$
                    $$ df_{error} = N - k $$
                    $$ N = \sum_{i=1}^{k} n_i $$
                    Where: 
                    - :orange[$F$] is the F-statistic, 
                    - :orange[$MS_{between}$] is the mean square between groups (calculated based on the variability of the group means), and 
                    - :orange[$MS_{error}$] is the mean square error (calculated based on the variability of observations within groups). 
                    - :orange[$SS_{between}$] is the sum of squares between groups, calculated by summing the squared differences between each group mean and the overall mean, weighted by the number of observations in each group.
                    - :orange[$SS_{error}$] is the sum of squares error, calculated by summing the squared differences between each observation and its respective group mean.
                    - :orange[$df_{between}$] is the degrees of freedom for the between-groups variability, calculated as the number of groups minus one.
                    - :orange[$df_{error}$] is the degrees of freedom for the error term, calculated as the total number of observations minus the number of groups.
                    - :orange[$N$] is the total number of observations across all groups.
                    - The test statistic :orange[$F$] is then compared to a critical value from the F-distribution with appropriate degrees of freedom to determine significance.
                    """,
        "Post-Hoc": "Pairwise Paired t, Paired t + Bonferroni, Paired t + Holm",
    },
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
        "Post-Hoc": "Tukey HSD, Bonferroni, Holm-Bonferroni, Šidák, Scheffé, Dunnett, Games-Howell, Fisher LSD, Newman-Keuls",
    },
    {
        "name": "ANCOVA",
        "Objective": "Comparison",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": ["Categorical", "Continuous"],
        "Groups": "More than 2",
        "Relation": "Independent",
        "Distribution": "Normal",
        "Explanation": "Analysis of Covariance (ANCOVA) combines ANOVA and linear regression. It compares group means on a continuous dependent variable while statistically controlling for the effect of one or more continuous covariates. This increases statistical power by reducing within-group error variance and adjusts for baseline differences. It assumes normality, homogeneity of variances, homogeneity of regression slopes, and linearity between covariate and outcome.",
        "Example": "A researcher wants to compare post-treatment blood pressure between three drug groups while controlling for baseline blood pressure. ANCOVA adjusts the post-treatment means for baseline differences, providing a more precise estimate of treatment effects.",
        "Formula": r"""
                    $$ F = \frac{MS_{between}}{MS_{error}} $$
                    $$ MS_{between} = \frac{SS_{between}}{df_{between}}, \quad MS_{error} = \frac{SS_{error}}{df_{error}} $$
                    $$ SS_{between} = \sum_{j=1}^{k} n_j (\bar{Y}_j - \bar{Y}_{adj})^2 $$
                    $$ \bar{Y}_{adj} = \bar{Y} - \beta(\bar{X} - \bar{X}_{overall}) $$
                    $$ SS_{error} = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $$
                    $$ \hat{Y}_i = \mu + \tau_j + \beta(X_i - \bar{X}) $$
                    Where:
                    - The dependent variable :orange[$Y$] is modeled as: $$ Y_{ij} = \mu + \tau_j + \beta(X_{ij} - \bar{X}) + \epsilon_{ij} $$
                    - :orange[$\tau_j$] is the effect of the :orange[$j$]-th group,
                    - :orange[$\beta$] is the regression coefficient for the covariate :orange[$X$],
                    - :orange[$MS_{error}$] is reduced by the variance explained by the covariate, increasing power.
                    """,
        "Post-Hoc": "Tukey HSD, Bonferroni, Holm-Bonferroni, Šidák (for adjusted group comparisons after covariate adjustment)",
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
        "Post-Hoc": "Discriminant Comparisons (pairwise univariate F), Canonical Contrasts (Bonferroni-corrected)",
    },
    # Two-sample tests (Non-parametric)
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
                    $$ W = \min(W^+, W^-) $$
                    $$ W^+ = \sum_{i=1}^{n} R_i^+ $$
                    $$ W^- = \sum_{i=1}^{n} R_i^- $$
                    $$ R_i = \text{rank}(|X_i - M_0|) $$
                    $$ sgn(X_i - M_0) = \begin{cases} +1 & \text{if } X_i > M_0 \\ -1 & \text{if } X_i < M_0 \\ 0 & \text{if } X_i = M_0 \end{cases} $$
                    Where:
                    - :orange[$W^+$] is the sum of the positive ranks,
                    - :orange[$W^-$] is the absolute sum of the negative ranks.
                    - :orange[$R$] is calculated by ranking the absolute differences between paired observations, excluding ties.
                    - :orange[$X_i$] is the observed value for the :orange[$i$]-th pair, and :orange[$M_0$] is the hypothesized median.
                    - :orange[$sgn(X_i - M_0)$] indicates the direction of the difference (positive, negative, or zero).
                    - :orange[$W$] is the test statistic, which is the smaller of the two sums of ranks (positive and negative).
                    - The test statistic :orange[$W$] is then compared to a critical value from the Wilcoxon signed-rank distribution to determine significance.
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
                    $$ U' = \sum_{i=1}^{n_2} R_i - \dfrac{n_2(n_2+1)}{2} $$
                    $$ U = \min(U, U') $$
                    $$ z = \dfrac{U - \mu_U}{\sigma_U} $$
                    Where: 
                    - :orange[$U$] is the test statistic, 
                    - :orange[$R_i$] is the rank of the :orange[$i$]-th observation in the combined dataset, 
                    - :orange[$n_1$] is the number of observations in group 1, and 
                    - :orange[$n_2$] is the number of observations in group 2. 
                    - The test statistic :orange[$U$] is then compared to a critical value from the Mann-Whitney U distribution to determine significance. :orange[$R$] is calculated by ranking all observations from both groups together and assigning ranks accordingly, with ties receiving average ranks. The U statistic is calculated based on the sum of ranks for one of the groups and adjusted for the number of observations in that group.
                    """,
    },
    # More two-sample tests (Non-parametric)
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
        "Post-Hoc": "Dunn, Conover, DSCF (Dwass-Steel-Critchlow-Fligner)",
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
                    $$ \chi^2_F = \dfrac{12}{N k (k+1)} \sum_{j=1}^{k} R_j^2 - 3N(k+1) $$ 
                    Where: 
                    - :orange[$\chi^2_F$] is the test statistic, 
                    - :orange[$N$] is the number of subjects (blocks), 
                    - :orange[$k$] is the number of treatments (groups), and 
                    - :orange[$R_j$] is the sum of ranks for the :orange[$j$]-th treatment. 
                    - The test statistic :orange[$\chi^2_F$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
        "Post-Hoc": "Nemenyi, Conover-Friedman, Wilcoxon + Bonferroni",
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
                    $$ MS_{between} = \dfrac{SS_{between}}{df_{between}} $$
                    $$ MS_{error} = \dfrac{SS_{error}}{df_{error}} $$
                    $$ SS_{between} = \sum_{i=1}^{k} n_i (\bar{X}_i - \bar{X})^2 $$
                    $$ SS_{error} = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2 $$
                    $$ df_{between} = k - 1 $$
                    $$ df_{error} = N - k $$
                    Where:
                    - :orange[$F$] is the pseudo-F-statistic,
                    - :orange[$MS_{between}$] is the mean square between groups, and
                    - :orange[$MS_{error}$] is the mean square error.
                    - The test statistic :orange[$F$] is then evaluated using a permutation-based null distribution (data is randomly reshuffled many times) to compute the p-value, rather than comparing to a theoretical F-distribution.
                    """,
        "Post-Hoc": "Dunn, Conover, DSCF (Dwass-Steel-Critchlow-Fligner)",
    },
    # Two or More Categorical sample tests
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
    # Association/Correlation tests
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
        "Dependent_Variable": ["Binary/Dichotomous", "Categorical"],
        "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
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
    {
        "name": "Kendall's Tau-b",
        "Objective": "Association/Correlation",
        "Dependent_Variable": ["Ordinal", "Continuous"],
        "Independent_Variable": ["Ordinal", "Continuous"],
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
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
        "Independent_Variable": ["Continuous", "Multiple Continuous", "Categorical"],
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
        "Independent_Variable": ["Continuous", "Multiple Continuous", "Categorical"],
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
        "Independent_Variable": ["Continuous", "Multiple Continuous", "Categorical"],
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
        "Independent_Variable": ["Continuous", "Multiple Continuous", "Categorical"],
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
    {
        "name": "Negative Binomial Regression",
        "Objective": "Prediction",
        "Dependent_Variable": "Discrete",
        "Independent_Variable": ["Continuous", "Multiple Continuous", "Categorical"],
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["any", "Dependent", "Independent"],
        "Distribution": ["any", "Non-normal", "Normal"],
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
    # Diagnostic Accuracy Tests
    {
        "name": "Sensitivity & Specificity Analysis",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Binary/Dichotomous",
        "Independent_Variable": "Binary/Dichotomous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["any", "Dependent", "Independent"],
        "Distribution": ["any", "Non-normal", "Normal"],
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
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["any", "Dependent", "Independent"],
        "Distribution": ["any", "Non-normal", "Normal"],
        "Explanation": "Receiver Operating Characteristic (ROC) analysis is used to evaluate the performance of a continuous diagnostic test. It plots Sensitivity against 1-Specificity at various thresholds. The Area Under the Curve (AUC) represents the overall accuracy.",
        "Example": "A researcher wants to determine if blood sugar levels can accurately diagnose diabetes. By plotting an ROC curve, they can find the optimal sugar level cut-off that maximizes both sensitivity and specificity.",
        "Formula": r"""
                    $$ \text{AUC} = \int_{0}^{1} \text{Sensitivity}(t) \, d(1 - \text{Specificity}(t)) $$
                    - :orange[AUC = 0.5]: Random guessing
                    - :orange[AUC = 1.0]: Perfect diagnostic accuracy
                    """,
    },
    {
        "name": "Likelihood Ratio Analysis",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Binary/Dichotomous",
        "Independent_Variable": "Binary/Dichotomous",
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["any", "Dependent", "Independent"],
        "Distribution": ["any", "Non-normal", "Normal"],
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
        "Distribution": ["any", "Non-normal", "Normal"],
        "Explanation": "Cohen's Kappa is used to measure inter-rater or intra-rater agreement for categorical variables. It accounts for the agreement occurring by chance.",
        "Example": "Two radiologists evaluate the same set of X-rays to diagnose a fracture. Cohen's Kappa measures how consistently they agree on the presence or absence of a fracture.",
        "Formula": r"""
                    $$ \kappa = \dfrac{p_o - p_e}{1 - p_e} $$
                    Where:
                    - :orange[$p_o$]: Observed proportionate agreement
                    - :orange[$p_e$]: Probability of random agreement
                    """,
    },
    {
        "name": "Bland-Altman Analysis",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Continuous",
        "Independent_Variable": "Continuous",
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": ["any", "Non-normal", "Normal"],
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
    {
        "name": "Weighted Kappa",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Ordinal",
        "Independent_Variable": "Ordinal",
        "Groups": "2",
        "Relation": "Dependent",
        "Distribution": ["any", "Non-normal", "Normal"],
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
    {
        "name": "Fleiss' Kappa",
        "Objective": "Diagnostic Accuracy",
        "Dependent_Variable": "Categorical",
        "Independent_Variable": "Categorical",
        "Groups": "More than 2",
        "Relation": "Dependent",
        "Distribution": ["any", "Non-normal", "Normal"],
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
    # Survival Analysis Tests
    {
        "name": "Kaplan-Meier Survival Analysis",
        "Objective": "Survival Analysis",
        "Dependent_Variable": "Time-to-event",
        "Independent_Variable": "Categorical",
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": ["any", "Non-normal", "Normal"],
        "Explanation": "Kaplan-Meier Survival Analysis is a non-parametric method used to estimate the survival function from time-to-event data. It accounts for censored data (individuals lost to follow-up or not experiencing the event by the end of the study). The Kaplan-Meier curve plots the probability of survival over time, and the Log-Rank Test can be used to compare survival curves between groups.",
        "Example": "A researcher wants to compare the survival times of patients with two different types of cancer. They collect time-to-event data (time until death or last follow-up) for 100 patients in each group and use Kaplan-Meier analysis to estimate and compare the survival curves.",
        "Formula": r"""
                    $$ \hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right) $$
                    Where:
                    - :orange[$\hat{S}(t)$] is the estimated survival probability at time :orange[$t$],
                    - :orange[$t_i$] are the distinct event times,
                    - :orange[$d_i$] is the number of events at time :orange[$t_i$],
                    - :orange[$n_i$] is the number of individuals at risk just before time :orange[$t_i$].
                    """,
    },
    {
        "name": "Cox Proportional Hazards Regression",
        "Objective": "Survival Analysis",
        "Dependent_Variable": "Time-to-event",
        "Independent_Variable": ["Continuous", "Multiple Continuous", "Categorical"],
        "Groups": ["any", "2", "More than 2"],
        "Relation": ["Independent", "Dependent", "any"],
        "Distribution": ["any", "Non-normal", "Normal"],
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
    {
        "name": "Log-Rank Test",
        "Objective": "Survival Analysis",
        "Dependent_Variable": "Time-to-event",
        "Independent_Variable": "Categorical",
        "Groups": "2",
        "Relation": "Independent",
        "Distribution": ["any", "Non-normal", "Normal"],
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
     {
         "name": "Sign Test (One-sample)",
         "Objective": "Comparison",
         "Dependent_Variable": ["Ordinal", "Continuous"],
         "Independent_Variable": "None",
         "Groups": "1",
         "Relation": ["Independent", "Dependent", "any"],
         "Distribution": ["Non-normal", "Normal", "any"],
         "Explanation": "The Sign Test is the simplest nonparametric test for location. It only examines the signs (positive/negative) of differences from a hypothesized median, completely ignoring their magnitudes. While less powerful than the Wilcoxon Signed-Rank Test, it is extremely robust to outliers and conceptually simpler — making it ideal for teaching nonparametric statistics before introducing more complex rank-based methods.",
         "Example": "A teacher tests whether a new study method improves exam scores. For 15 students, 11 score higher after the method, 3 score lower, and 1 shows no change. The Sign Test asks: If the method had no effect, is observing 11 positives out of 14 non-ties unusual?",
         "Formula": r"""
                     $$ S = \min(S^+, S^-) $$
                     $$ S^+ = \text{number of positive differences from } \theta_0 $$
                     $$ S^- = \text{number of negative differences from } \theta_0 $$
                     
                     Under H₀, S follows a **Binomial distribution** with parameters n' = S⁺ + S⁻ and p = 0.5:
                     $$ P(X = k) = \binom{n'}{k} 0.5^{n'} $$
                     
                     Two-sided p-value = 2 × min(P(X ≤ S), P(X ≥ S))
                     """,
         "Decision Rules": r"""
                     - Reject H₀ if the exact binomial p-value < α (typically 0.05)
                     - No normal approximation is needed — the binomial calculation is exact
                     - Differences equal to θ₀ are typically dropped from analysis (reducing effective sample size)
                     - Effect size can be reported as (S⁺ - S⁻)/n' or the median difference with confidence interval
         """,
     },
     {
         "name": "Sign Test (Paired)",
         "Objective": "Comparison",
         "Dependent_Variable": ["Ordinal", "Continuous"],
         "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
         "Groups": "2",
         "Relation": "Dependent",
         "Distribution": ["Non-normal", "Normal", "any"],
         "Explanation": "The Paired Sign Test compares two related (matched/paired) measurements by examining only the signs of their differences. It asks a simple question: Is one measurement typically greater than the other? Like the one-sample Sign Test, it ignores the magnitude of differences — trading power for simplicity and robustness.",
         "Example": "A sports scientist tests whether a new running shoe reduces 5K times. For 20 athletes, 14 run faster with the new shoe, 5 run slower, and 1 shows no change. The Paired Sign Test determines if this pattern (more positives) is unlikely under the null hypothesis of no difference.",
         "Formula": r"""
                     $$ D_i = X_{i,\text{after}} - X_{i,\text{before}} \quad \text{or} \quad D_i = X_{i,\text{Method A}} - X_{i,\text{Method B}} $$
                     $$ S = \min(\text{number of positive } D_i, \text{number of negative } D_i) $$
                     
                     Under H₀, S ~ Binomial(n', 0.5) where n' = number of non-zero differences.
                     
                     **Exact p-value** computed directly from the binomial distribution.
                     """,
     },
     {
         "name": "Runs Test for Randomness",
         "Objective": "Comparison",
         "Dependent_Variable": ["Binary/Dichotomous", "Categorical", "Continuous"],
         "Independent_Variable": "None",
         "Groups": "1",
         "Relation": ["Independent", "Dependent", "any"],
         "Distribution": ["any", "Non-normal", "Normal"],
         "Explanation": "The Runs Test for Randomness asks a fundamental question: Is this sequence random? A 'run' is a consecutive sequence of identical values or values above/below a threshold. Too few runs = clustered pattern; too many runs = alternating pattern. This test teaches the concept of randomness and is essential for time-series and quality control applications.",
         "Example": "A quality control engineer examines 50 consecutive parts from an assembly line: G G G G B B G G G G G G B B B G G G G B... The engineer counts runs of Good and Bad parts. If runs are too few, the process may have clustering issues (e.g., a machine that drifts out of alignment). If runs are too many, there may be an alternating systematic issue.",
         "Formula": r"""
                     Let:
                     - :orange[$n_1$] = number of observations of Type 1 (e.g., above median, or 'Successes')
                     - :orange[$n_2$] = number of observations of Type 2 (e.g., below median, or 'Failures')
                     - :orange[$R$] = number of runs observed
                     
                     Under H₀ (randomness):
                     $$ \mu_R = \frac{2 n_1 n_2}{n_1 + n_2} + 1 $$
                     $$ \sigma_R = \sqrt{\frac{2 n_1 n_2 (2 n_1 n_2 - n_1 - n_2)}{(n_1 + n_2)^2 (n_1 + n_2 - 1)}} $$
                     
                     For large samples:
                     $$ z = \frac{R - \mu_R}{\sigma_R} \sim N(0, 1) $$
                     
                     **Left-tailed**: Too few runs (clustering)
                     **Right-tailed**: Too many runs (alternating pattern)
                     **Two-tailed**: Either extreme (non-randomness of any form)
                     """,
         "Decision Rules": r"""
                     - If :orange[$R \ll \mu_R$]: Clustered pattern → reject randomness
                     - If :orange[$R \gg \mu_R$]: Alternating pattern → reject randomness
                     - Two-sided test: Reject if :orange[$|z| > z_{\alpha/2}$] or p-value < α
                     - For continuous data: typically coded as values **above** vs **below** the median (or mean)
                     - The test can also be applied to categorical sequences with more than two categories (extensions exist)
         """,
     },
     {
         "name": "Mood's Median Test",
         "Objective": "Comparison",
         "Dependent_Variable": ["Ordinal", "Continuous"],
         "Independent_Variable": "Categorical",
         "Groups": "More than 2",
         "Relation": "Independent",
         "Distribution": ["Non-normal", "Normal", "any"],
         "Explanation": "Mood's Median Test is the simplest nonparametric alternative to one-way ANOVA. It tests whether multiple independent samples come from populations with the same median. Conceptually, it works by: (1) finding the grand median of ALL observations combined, (2) counting values above and below this grand median in each group, and (3) performing a Chi-square test of independence on this contingency table. Less powerful than Kruskal-Wallis, but simpler to understand and very robust.",
         "Example": "A psychologist tests whether three different teaching methods produce different median exam scores. Scores from Method A: 72, 85, 78, 90, 65. Method B: 88, 92, 85, 95, 80. Method C: 60, 75, 70, 68, 72. The grand median of all 15 scores is computed, then counts of scores above/below are tabulated for each group. A Chi-square test determines if the pattern differs across groups.",
         "Formula": r"""
                     Step 1: Compute the **grand median** (GM) of all observations combined:
                     $$ \tilde{X}_{grand} = \text{median}(X_{11}, X_{12}, \ldots, X_{kn_k}) $$
                     
                     Step 2: Construct contingency table of counts:
                     
                     | Group | Above GM | Below or Equal |
                     |-------|-----------|----------------|
                     | 1     | :orange[$A_1$] | :orange[$B_1$] |
                     | 2     | :orange[$A_2$] | :orange[$B_2$] |
                     | ...   | ...       | ...            |
                     | k     | :orange[$A_k$] | :orange[$B_k$] |
                     
                     Step 3: Compute **Pearson's Chi-square** on this 2×k table:
                     $$ \chi^2 = \sum_{i=1}^{k} \sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$
                     
                     Where :orange[$E_{ij}$] are expected frequencies under independence.
                     
                     $$ df = (2-1)(k-1) = k-1 $$
                     """,
         "Decision Rules": r"""
                     - Reject H₀ if :orange[$\chi^2 > \chi^2_{\alpha, k-1}$] or p-value < α
                     - The test is essentially a **Chi-square test of independence** with the median dichotomy
                     - It assumes only independence of observations
                     - No normality assumption required
                     - If the Chi-square approximation is questionable (small expected frequencies), Fisher's exact test can be used for 2×2 tables (two groups)
                     - Ties at the grand median can be handled by: (a) dropping them, (b) counting half in each category, or (c) using the 'below or equal' approach shown
         """,
     },
     {
         "name": "Equivalence Test (TOST) - Two Independent Samples",
         "Objective": "Comparison",
         "Dependent_Variable": "Continuous",
         "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
         "Groups": "2",
         "Relation": "Independent",
         "Distribution": "Normal",
         "Explanation": "TOST = Two One-Sided Tests. This procedure solves a critical statistical misconception: 'p > 0.05' does NOT mean 'treatments are equivalent'. TOST explicitly reverses the burden of proof: it requires data to demonstrate that the true difference lies within a pre-specified 'equivalence range' (-Δ, +Δ). Only if BOTH one-sided tests reject their respective null hypotheses can you conclude equivalence. This is essential for teaching the difference between 'no evidence of difference' and 'evidence of no difference'.",
         "Example": "A pharmaceutical company wants to show that a new generic drug is 'equivalent' to the brand-name drug. They define equivalence as having a mean difference in blood pressure of less than 5 mmHg (Δ = 5). They test: (1) Is the true difference > -5? (2) Is the true difference < +5? If both tests reject, equivalence is established. A simple t-test showing 'no significant difference' (p > 0.05) would not be sufficient — equivalence must be demonstrated.",
         "Formula": r"""
                     **Equivalence Margin**: :orange[$\Delta$] (smallest difference considered 'meaningfully different')
                     
                     **Traditional superiority null**: :orange[$H_0: \mu_1 - \mu_2 = 0$]
                     
                     **TOST null hypotheses** (both must be rejected for equivalence):
                     $$ H_{0L}: \mu_1 - \mu_2 \leq -\Delta \quad \text{(difference is too low)} $$
                     $$ H_{0U}: \mu_1 - \mu_2 \geq +\Delta \quad \text{(difference is too high)} $$
                     
                     **Alternative hypotheses**:
                     $$ H_{1L}: \mu_1 - \mu_2 > -\Delta $$
                     $$ H_{1U}: \mu_1 - \mu_2 < +\Delta $$
                     
                     **Two one-sided t-tests**:
                     $$ t_L = \frac{(\bar{x}_1 - \bar{x}_2) - (-\Delta)}{SE} = \frac{\bar{D} + \Delta}{SE} $$
                     $$ t_U = \frac{(\bar{x}_1 - \bar{x}_2) - (+\Delta)}{SE} = \frac{\bar{D} - \Delta}{SE} $$
                     
                     Where :orange[$SE = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$] (or pooled if assuming equal variances)
                     
                     **Decision**: Equivalence concluded if :orange[$p_L < \alpha$] **AND** :orange[$p_U < \alpha$]
                     
                     **Confidence Interval approach**: Equivalence concluded if the :orange[$100(1-2\alpha)\%$] (or 90% for α=0.05) CI for :orange[$\mu_1 - \mu_2$] lies entirely within :orange[$(-\Delta, +\Delta)$].
                     """,
         "Decision Rules": r"""
                     **Key Principle**: The burden of proof is on demonstrating equivalence.
                     
                     1. **Define equivalence margin (Δ)** BEFORE data collection
                        - Typically based on clinical or practical significance
                        - Common choices: 0.2×SD of reference, or regulatory standards
                     
                     2. **Run both one-sided tests**:
                        - Reject :orange[$H_{0L}$] if :orange[$t_L > t_{\alpha, df}$] (one-tailed)
                        - Reject :orange[$H_{0U}$] if :orange[$t_U < -t_{\alpha, df}$] (one-tailed)
                     
                     3. **Conclusion**:
                        - If **both** rejected → **Equivalence demonstrated**
                        - If either not rejected → **Cannot claim equivalence**
                     
                     **Common Misconception**:
                     - ❌ 'p > 0.05 in t-test' ≠ 'Equivalent'
                     - ✅ Only TOST or CI-within-range can demonstrate equivalence
                     
                     **Visual Interpretation**: The 90% confidence interval must lie COMPLETELY inside (-Δ, +Δ) for equivalence.
         """,
     },
     {
         "name": "F-Test for Two Variances",
         "Objective": "Comparison",
         "Dependent_Variable": "Continuous",
         "Independent_Variable": ["Binary/Dichotomous", "Categorical"],
         "Groups": "2",
         "Relation": "Independent",
         "Distribution": "Normal",
         "Explanation": "The F-test for equality of variances compares the spread (variability) of two independent samples. It tests the homogeneity of variance assumption that underlies Student's t-test and ANOVA. The test statistic is simply the ratio of the two sample variances. Important: This test is EXTREMELY sensitive to non-normality — much more so than the t-test itself. For this reason, robust alternatives like Levene's test are generally preferred, but the F-test remains valuable for teaching the concept of variance ratios and the F-distribution.",
         "Example": "A researcher plans to use Student's t-test to compare two methods. First, they check the equal variance assumption: Method A (n=20) has variance = 12.5, Method B (n=20) has variance = 45.2. The F-ratio = 45.2/12.5 = 3.62. Is this large enough to reject equal variances?",
         "Formula": r"""
                     $$ F = \frac{s_1^2}{s_2^2} \quad \text{or typically} \quad F = \frac{\max(s_1^2, s_2^2)}{\min(s_1^2, s_2^2)} $$
                     
                     Where:
                     - :orange[$s_1^2$] = variance of sample 1
                     - :orange[$s_2^2$] = variance of sample 2
                     
                     Under :orange[$H_0: \sigma_1^2 = \sigma_2^2$], the F-statistic follows an **F-distribution** with:
                     - Numerator df = :orange[$n_1 - 1$]
                     - Denominator df = :orange[$n_2 - 1$]
                     
                     **Two-sided test**: Reject if :orange[$F > F_{\alpha/2, df_1, df_2}$] or :orange[$F < F_{1-\alpha/2, df_1, df_2}$]
                     
                     By convention, most software places the larger variance in the numerator, giving F ≥ 1, and then doubles the one-tailed p-value.
                     """,
         "Decision Rules": r"""
                     - Reject :orange[$H_0$] if p-value < α (typically 0.05)
                     - If significant → variances differ → use **Welch's t-test** instead of Student's t-test
                     - If not significant → equal variance assumption may be reasonable
                     
                     **CRITICAL WARNING**:
                     The F-test is **highly sensitive to non-normality**. If your data comes from a non-normal distribution, this test may:
                     - Falsely detect 'unequal variances' when the real issue is non-normality
                     - Or fail to detect real differences
                     
                     **Robust alternatives**:
                     - **Levene's test**: Uses absolute deviations from the median (more robust)
                     - **Fligner-Killeen test**: Even more robust to non-normality
                     - **Brown-Forsythe**: Levene's variant using median instead of mean
                     
                     These are generally preferred over the F-test in practice, but the F-test teaches the fundamental concept of variance ratios.
         """,
     },
     {
         "name": "Poisson Goodness-of-Fit Test",
         "Objective": "Comparison",
         "Dependent_Variable": "Categorical",
         "Independent_Variable": "None",
         "Groups": "1",
         "Relation": ["Independent", "Dependent", "any"],
         "Distribution": ["Non-normal", "Normal", "any"],
         "Explanation": "The Poisson Goodness-of-Fit test determines whether count data follows a Poisson distribution. The Poisson distribution is fundamental for modeling: (1) number of events in a fixed interval, (2) independent events, (3) constant average rate. A key property: for a true Poisson, the variance equals the mean. If variance > mean, you have **over-dispersion** (common in real data), suggesting Negative Binomial regression may be more appropriate. This test teaches the Poisson assumptions and how to check them.",
         "Example": "A hospital administrator counts the number of emergency admissions per hour over 100 hours: 0 admissions in 25 hours, 1 admission in 35 hours, 2 admissions in 25 hours, 3+ admissions in 15 hours. The administrator fits a Poisson distribution to these counts and tests whether the observed frequencies match the expected Poisson frequencies.",
         "Formula": r"""
                     **Step 1: Estimate the Poisson parameter (λ)** from the sample:
                     $$ \hat{\lambda} = \bar{x} = \text{sample mean} $$
                     
                     **Step 2: Compute expected frequencies** for each count k:
                     $$ P(X = k) = \frac{e^{-\hat{\lambda}} \hat{\lambda}^k}{k!} $$
                     $$ E_k = n \times P(X = k) $$
                     
                     **Step 3: Pool categories** so that all :orange[$E_k \geq 5$] (requirement for Chi-square approximation)
                     
                     **Step 4: Compute Chi-square statistic**:
                     $$ \chi^2 = \sum_{k} \frac{(O_k - E_k)^2}{E_k} $$
                     
                     **Step 5: Degrees of freedom**:
                     $$ df = \text{(\# categories after pooling)} - 1 - \text{(\# parameters estimated)} $$
                     
                     For Poisson GOF: **df = k - 2** (subtract 1 for total sum, subtract 1 more for estimating λ)
                     """,
         "Decision Rules": r"""
                     - Reject :orange[$H_0$] if :orange[$\chi^2 > \chi^2_{\alpha, df}$] or p-value < α
                     - If rejected → data does NOT follow Poisson distribution
                     
                     **Checking for Over-dispersion**:
                     
                     A key diagnostic for count data:
                     $$ \text{Variance/Mean Ratio} = \frac{s^2}{\bar{x}} $$
                     
                     - Ratio ≈ 1 → Consistent with Poisson
                     - Ratio > 1 → **Over-dispersion** (more variance than Poisson predicts)
                     - Ratio < 1 → **Under-dispersion** (less variance than Poisson predicts)
                     
                     **Over-dispersion is common** and suggests:
                     - Use **Negative Binomial regression** instead of Poisson regression
                     - Or consider **zero-inflated Poisson (ZIP)** if there are excess zeros
                     
                     **Exact alternatives**: For very small samples, the Kolmogorov-Smirnov test (with estimated parameters requires special tables) or simulation-based approaches.
         """,
     },
 ]


CRITERIA_FIELDS = [
    "Objective",
    "Dependent_Variable",
    "Independent_Variable",
    "Groups",
    "Relation",
    "Distribution",
]

FIELDS = [
    "Objective",
    "Dependent_Variable",
    "Independent_Variable",
    "Groups",
    "Relation",
    "Distribution",
]
