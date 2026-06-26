# Auto-generated from assets/test_definitions.json
# Edit tests directly here in Python — the dataclass gives you full IDE support,
# type checking, and autocomplete on every field.

from core.types import TestDefinition

one_sample_t_test = TestDefinition(
    name="One-sample t-test",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var="None",
    groups="1",
    relation=["any", "Dependent", "Independent"],
    distribution="Normal",
    explanation="""One-Sample t-test This test is used to determine whether the mean of a single sample is significantly different from a known or hypothesized population mean. It assumes that the data is continuous and follows a normal distribution. It is typically used when comparing a clinical measurement (like blood pressure) against a standard clinical threshold. """,
    example="""A researcher wants to test if the average systolic blood pressure of a group of patients is significantly different from the standard threshold of 120 mmHg. The researcher collects blood pressure readings from 30 patients and performs a one-sample t-test to compare the sample mean against the known population mean of 120 mmHg.""",
    formula="""
                    $$ t = \\dfrac{\\bar{x} - \\mu_0}{\\dfrac{s}{\\sqrt{n}}}$$ 
                    Where: 
                    - :orange[$\\bar{x}$] is the sample mean,   
                    - :orange[$\\mu_0$] is the population mean, 
                    - :orange[$s$] is the sample standard deviation, 
                    - :orange[$n$] is the sample size 
                    - :orange[$\\dfrac{s}{\\sqrt{n}}$] is the standard error of the mean.
                    """,
    decision_rules="""
                    -  Reject the null hyposthesis $H_0$ if the absolute value of the test statistic :orange[$|t|$] is greater than the critical t-value from the t-distribution with :orange[$n-1$] degrees of freedom at the chosen significance level (e.g., α = 0.05) :orange[$|t| > t_{critical}$].
                    -  Fail to reject the null hypothesis if :orange[$|t|$] is less than or equal to the critical t-value :orange[$|t| \\leq t_{critical}$].
                    -  Reject the null hyosthesis $H_0$ if the p-value is less than the chosen significance level (e.g., α = 0.05) :orange[$p < \\alpha$].
                    -  Effect size (Cohen's d) can be calculated as :orange[$ d = \\dfrac{\\bar{x} - \\mu_0}{s} $], where :orange[$d$] is the standardized mean difference, providing a measure of the magnitude of the effect independent of sample size.
        """,
    core_assumptions="""- **Continuous data** — the dependent variable is measured on a continuous scale
- **Independence** — each observation is independent of others (random sampling)
- **Normal distribution** — the data follows a normal distribution (or n ≥ 30 for CLT robustness)
- **No significant outliers** — extreme values can bias the mean and inflate the standard error""",
    interpretation="""- If **p < α** (typically 0.05), reject H₀: the sample mean significantly differs from the hypothesized population mean
- Report: **t(df)** = t-value, **p** = p-value, **d** = Cohen's d
- **Cohen's d** effect size: 0.2 = small, 0.5 = medium, 0.8 = large
- **Confidence interval** for the mean difference provides a range of plausible values
- Always check the assumptions were met before drawing conclusions""",
    realworld_apps="""- **Clinical research**: comparing patient blood pressure to a healthy threshold
- **Education**: testing if class exam scores differ from the national average
- **Quality control**: checking if manufactured part dimensions meet specifications
- **Environmental science**: comparing pollutant levels to regulatory limits
- **Psychology**: comparing reaction times to a known baseline""",
)

one_sample_z_test = TestDefinition(
    name="One-sample z-test",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var="None",
    groups="1",
    relation=["any", "Dependent", "Independent"],
    distribution="Normal",
    explanation="""One-Sample z-Test This test is used to determine whether the mean of a single sample is significantly different from a known or hypothesized population mean. It assumes that the data is continuous and follows a normal distribution, and that the population standard deviation is known.""",
    example="""A researcher wants to test if the average systolic blood pressure of a group of patients is significantly different from the standard threshold of 120 mmHg. The researcher knows the population standard deviation is 10 mmHg and collects blood pressure readings from 30 patients. The researcher performs a one-sample z-test to compare the sample mean against the known population mean of 120 mmHg.""",
    formula="""
                    $$ z = \\dfrac{\\bar{x} - \\mu_0}{\\dfrac{\\sigma}{\\sqrt{n}}} $$
                    where:
                    - :orange[$\\bar{x}$] is the sample mean,
                    - :orange[$\\mu_0$] is the population mean,
                    - :orange[$\\sigma$] is the population standard deviation,
                    - :orange[$n$] is the sample size.
                    - The denominator :orange[$\\dfrac{\\sigma}{\\sqrt{n}}$] is the standard error of the mean.
                    """,
    decision_rules="""
                    - Reject the null hypothesis if the absolute value of the test statistic :orange[$|z|$] is greater than the critical z-value from the standard normal distribution at the chosen significance level (e.g., α = 0.05) :orange[$|z| > z_{critical}$].
                    - Fail to reject the null hypothesis if :orange[$|z|$] is less than or equal to the critical z-value :orange[$|z| \\leq z_{critical}$].
                    - Reject the null hypothesis if the p-value is less than the chosen significance level (e.g., α = 0.05) :orange[$p < \\alpha$].
                    - Effect size (Cohen's d) can be calculated as :orange[ $ d = \\dfrac{\\bar{x} - \\mu_0}{\\sigma} $], where :orange[$d$] is the standardized mean difference, providing a measure of the magnitude of the effect independent of sample size.
        """,
    core_assumptions="""- **Continuous data** — the dependent variable is measured on a continuous scale
- **Independence** — each observation is independent of others
- **Normal distribution** — the data follows a normal distribution (or n is large for CLT)
- **Known σ** — the population standard deviation must be known (rare in practice)
- **Random sampling** — the sample is representative of the population""",
    interpretation="""- If **p < α**, reject H₀: the sample mean significantly differs from the hypothesized population mean
- Report: **z** = z-statistic, **p** = p-value, **d** = Cohen's d
- **Confidence interval**: x̄ ± z_(α/2) × σ/√n
- Note: the z-test is only appropriate when σ is **known**; otherwise use the one-sample t-test
- In practice, σ is almost never known, making the z-test less common than the t-test""",
    realworld_apps="""- **Standardized testing**: comparing IQ scores where population variance is known
- **Industrial QC**: checking measurements against known process variation
- **Clinical trials**: comparing against well-established population parameters
- **Educational assessment**: comparing class performance to published national norms
- **Manufacturing**: verifying product specifications when process variation is well-characterized""",
)

one_sample_proportion_test_binomial_test = TestDefinition(
    name="One-sample Proportion Test (Binomial Test)",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var="None",
    groups="1",
    relation=["Independent", "Dependent", "any"],
    distribution=["Normal", "Non-normal", "any"],
    explanation="""One-Sample Proportion Test (Binomial Test) This test is used to determine whether the proportion of successes in a single sample is significantly different from a known or hypothesized population proportion. It is typically used when analyzing categorical data, such as the proportion of patients who respond to a treatment compared to a known response rate.""",
    example="""A clinical trial tests a new drug and finds that 18 out of 30 patients respond to the treatment. The researcher wants to determine if this response rate is significantly different from the known response rate of 50% (0.5) for existing treatments. The researcher performs a one-sample proportion test (binomial test) to compare the observed proportion of 0.6 (18/30) against the known population proportion of 0.5.""",
    formula="""
                    $$ z = \\dfrac{\\hat{p} - p_0}{\\sqrt{\\dfrac{p_0(1 - p_0)}{n}}} $$
                    $$ P(X = k) = \\binom{n}{k} p_0^k (1 - p_0)^{n-k} $$
                    where: 
                    - :orange[$\\hat{p}$] is the sample proportion,  
                    - :orange[$p_0$] is the population proportion, and   
                    - :orange[$n$] is the sample size.  
                    - :orange[$P(X = k)$] is the probability of observing exactly :orange[$k$] successes in :orange[$n$] trials under the null hypothesis, calculated using the binomial distribution.
                    - :orange[$z$] is the test statistic.
                    - :orange[$k$] is the number of successes observed in the sample.
                    - The denominator :orange[$\\sqrt{\\dfrac{p_0(1 - p_0)}{n}}$] is the standard error of the proportion.
                    """,
    core_assumptions="""- **Binary data** — the outcome has exactly two categories (success/failure)
- **Independence** — observations are independent of each other
- **Random sampling** — the sample is representative
- **Large sample** — n × p₀ ≥ 5 **and** n × (1−p₀) ≥ 5 for the normal approximation to be valid
- **Fixed sample size** — n is fixed in advance (not the number of successes)""",
    interpretation="""- If **p < α**, reject H₀: the sample proportion differs significantly from the hypothesized proportion
- Report: **p̂** = sample proportion, **z** = z-statistic, **p** = p-value
- **Confidence interval** (Wald): p̂ ± z_(α/2) × √(p̂(1−p̂)/n)
- For small samples (np < 5), use the **Exact Binomial Test** instead
- The Wilson score interval provides better coverage than the Wald interval for proportions""",
    realworld_apps="""- **Clinical trials**: comparing treatment response rate to a historical control rate
- **Market research**: testing if customer satisfaction exceeds a target percentage
- **Quality control**: checking if defect rate exceeds the acceptable limit
- **Political polling**: comparing candidate support to previous election results
- **Epidemiology**: testing if disease prevalence differs from a known population rate""",
)

binomial_test = TestDefinition(
    name="Binomial Test",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var="None",
    groups="1",
    relation=["Independent", "Dependent", "any"],
    distribution=["Normal", "Non-normal", "any"],
    explanation="""Binomial Test (Exact) This test is used to determine whether the observed proportion of successes in a single sample differs from a hypothesized population proportion. Unlike the one-sample proportion z-test, the exact binomial test computes the p-value directly from the binomial distribution without relying on a normal approximation, making it more accurate for small sample sizes. It is appropriate when the data are binary (success/failure) and each trial is independent.""",
    example="""A researcher wants to test if a coin is fair. They flip it 20 times and observe 15 heads. The exact binomial test calculates the probability of observing 15 or more heads (or 5 or fewer) under the null hypothesis that the true probability of heads is 0.5, providing an exact p-value without relying on the normal approximation.""",
    formula="""
                    $$ P(X = k) = \\binom{n}{k} p_0^k (1 - p_0)^{n-k} $$
                    $$ \\text{p-value} = \\sum_{j \\leq k} \\binom{n}{j} p_0^j (1 - p_0)^{n-j} \\quad \\text{(one-sided lower)} $$
                    $$ \\text{p-value} = \\sum_{j \\geq k} \\binom{n}{j} p_0^j (1 - p_0)^{n-j} \\quad \\text{(one-sided upper)} $$
                    $$ \\text{p-value} = \\sum_{j: P(X=j) \\leq P(X=k)} \\binom{n}{j} p_0^j (1 - p_0)^{n-j} \\quad \\text{(two-sided)} $$
                    where:
                    - :orange[$k$] is the observed number of successes,
                    - :orange[$n$] is the total number of trials,
                    - :orange[$p_0$] is the hypothesized probability of success under :orange[$H_0$],
                    - :orange[$\\binom{n}{k}$] is the binomial coefficient.
                    """,
    decision_rules="""
                    - Reject the null hypothesis :orange[$H_0$] if the exact p-value is less than the chosen significance level (e.g., :orange[$\\alpha = 0.05$]).
                    - The two-sided p-value sums the probabilities of all outcomes as extreme or more extreme than the observed outcome.
                    - This test does not rely on a normal approximation, making it valid for any sample size.
                    - Effect size can be reported as the observed proportion :orange[$\\hat{p} = k / n$] with a confidence interval (Clopper-Pearson exact interval).
        """,
    core_assumptions="""- **Binary data** — the outcome has exactly two categories (success/failure)
- **Independence** — each trial is independent
- **Random sampling** — the sample is representative
- **Fixed sample size** — n is fixed in advance
- **No distributional approximation** — valid for any sample size, even very small
- **Constant probability** — the probability of success is constant under H₀""",
    interpretation="""- If the **exact p-value** < α, reject H₀: the observed proportion differs significantly from the hypothesized proportion
- Report: **k/n** = observed proportion, **exact p-value**, **Clopper-Pearson CI**
- Unlike the z-test, this test is valid even when np < 5 or n(1−p) < 5
- The **Clopper-Pearson interval** provides an exact confidence interval for the proportion
- Effect size: the difference p̂ − p₀ with its confidence interval""",
    realworld_apps="""- **Coin fairness tests** — the classic binomial experiment
- **Small clinical trials** — testing treatment efficacy with limited patients
- **A/B testing** — comparing conversion rates when traffic is low
- **Genetic studies** — testing Mendelian inheritance ratios
- **Rare event analysis** — studying outcomes that occur infrequently
- **Drug safety** — monitoring adverse event rates in small patient populations""",
)

sign_test_one_sample = TestDefinition(
    name="Sign Test (One-sample)",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var="None",
    groups="1",
    relation=["Independent", "Dependent", "any"],
    distribution="Non-normal",
    explanation="""The Sign Test is the simplest nonparametric test for location. It only examines the signs (positive/negative) of differences from a hypothesized median, completely ignoring their magnitudes. While less powerful than the Wilcoxon Signed-Rank Test, it is extremely robust to outliers and conceptually simpler — making it ideal for teaching nonparametric statistics before introducing more complex rank-based methods.""",
    example="""A teacher tests whether a new study method improves exam scores. For 15 students, 11 score higher after the method, 3 score lower, and 1 shows no change. The Sign Test asks: If the method had no effect, is observing 11 positives out of 14 non-ties unusual?""",
    formula="""
                     $$ S = \\min(S^+, S^-) $$
                     $$ S^+ = \\text{number of positive differences from } \\theta_0 $$
                     $$ S^- = \\text{number of negative differences from } \\theta_0 $$
                     
                     Under H₀, S follows a **Binomial distribution** with parameters n' = S⁺ + S⁻ and p = 0.5:
                     $$ P(X = k) = \\binom{n'}{k} 0.5^{n'} $$
                     
                     Two-sided p-value = 2 × min(P(X ≤ S), P(X ≥ S))
                     """,
    decision_rules="""
                     - Reject H₀ if the exact binomial p-value < α (typically 0.05)
                     - No normal approximation is needed — the binomial calculation is exact
                     - Differences equal to θ₀ are typically dropped from analysis (reducing effective sample size)
                     - Effect size can be reported as (S⁺ - S⁻)/n' or the median difference with confidence interval
         """,
    core_assumptions="""- **Ordinal or continuous data** — the variable is at least ordinal
- **Independence** — observations are independent
- **Continuous distribution** — needed to avoid excessive ties (differences exactly equal to θ₀)
- **No shape assumptions** — does NOT require symmetry or normality
- **Meaningful median** — the hypothesized median θ₀ is clinically or scientifically meaningful
- **Ties are dropped** — differences exactly equal to θ₀ are excluded, reducing effective sample size""",
    interpretation="""- If the **exact p-value** < α, reject H₀: the population median differs from the hypothesized value
- Report: **S⁺** = positive signs, **S⁻** = negative signs, **n'** = effective sample size, **p-value**
- **Effect size**: proportion of positive differences = S⁺/n', or the **median difference**
- The Sign Test is **less powerful** than the Wilcoxon Signed-Rank Test when assumptions hold
- However, it is **extremely robust** and minimally affected by outliers or heavy skew
- Ties reduce power — avoid them with precise measurement when possible""",
    realworld_apps="""- **Paired preferences** — taste tests where judges pick which product they prefer
- **Before-after studies** — only the direction of change matters, not the magnitude
- **Pilot studies** — very small samples where parametric assumptions cannot be verified
- **Educational research** — ordinal outcomes like pass/fail after an intervention
- **Quality control** — parts above or below specification, ignoring how far""",
)

one_sample_wilcoxon_signed_rank_test = TestDefinition(
    name="One-sample Wilcoxon Signed-Rank Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var="None",
    groups="1",
    relation=["Independent", "Dependent", "any"],
    distribution="Non-normal",
    explanation="""One-Sample Wilcoxon Signed-Rank Test This non-parametric test is used to determine whether the median of a single sample is significantly different from a known or hypothesized population median. It is typically used when the data is ordinal or continuous but does not follow a normal distribution.""",
    example="""A researcher wants to test if the median pain score of patients after a treatment is significantly different from a known median pain score of 5 on a 10-point scale. The researcher collects pain scores from 30 patients and performs a one-sample Wilcoxon signed-rank test to compare the sample median against the known population median of 5.""",
    formula="""
                    $$ W = \\min(W^+, W^-) $$
                    $$ W^+ = \\sum_{D_i>0} \\text{Rank}(|D_i|) $$
                    $$ W^- = \\sum_{D_i<0} \\text{Rank}(|D_i|) $$
                    $$ D_i = X_i - \\theta_0 $$
                    $$ sgn(X_i - \\theta_0) = \\begin{cases} +1 & \\text{if } X_i > \\theta_0 \\\\ -1 & \\text{if } X_i < \\theta_0 \\\\ 0 & \\text{if } X_i = \\theta_0 \\end{cases} $$
                    Where:  
                    - :orange[$D_i$] is the absolute difference between the observed value :orange[$X_i$] and the hypothesized median :orange[$\\theta_0$] for the :orange[$i$]-th observation,
                    - :orange[$X_i$] is the observed value for the :orange[$i$]-th observation, 
                    - :orange[$\\theta_0$] is the hypothesized median
                    - :orange[$sgn(X_i - \\theta_0)$] is the sign function that indicates whether :orange[$X_i$] is above, below, or equal to :orange[$\\theta_0$].  
                    - The test statistic :orange[$W$] is then compared to a critical value from the Wilcoxon signed-rank distribution to determine significance.

                    Large Sample Approximation
                    $$ z = \\frac{W - \\mu_W}{\\sigma_W} $$
                    $$ \\mu_W = \\frac{n(n+1)}{4} $$
                    $$ \\sigma_W = \\sqrt{\\frac{n(n+1)(2n+1)}{24}}  \\quad \\text{if no ties} $$
                    $$ \\sigma_W = \\sqrt{\\frac{n(n+1)(2n+1)}{24} - \\sum_{t} \\frac{t(t^2 - 1)}{48}} \\quad \\text{if ties exist} $$
                    $$ \\sigma_W = \\sqrt{\\frac{n(n+1)(2n+1)-\\frac{1}{2}\\sum^g_{j=1}{(t^3_j-t_j)}}{24}} \\quad \\text{if ties exist} $$
                    Where:
                    - :orange[$n$] is the number of non-zero differences,
                    - :orange[$t$] is the number of tied ranks for a particular value,
                    - :orange[$g$] is the number of groups of tied ranks.
                    - :orange[$\\mu_W$] is the mean of the test statistic under the null hypothesis,
                    - :orange[$\\sigma_W$] is the standard deviation of the test statistic under the null hypothesis, adjusted for ties if necessary.
                    - The test statistic :orange[$z$] is then compared to a standard normal distribution to determine significance when the sample size is large (typically n > 20).
                    - The large sample approximation is used when the sample size is sufficiently large, allowing the distribution of the test statistic to be approximated by a normal distribution, which simplifies the calculation of p-values and critical values for hypothesis testing.
                    """,
    decision_rules="""
                    - Reject the null hypothesis if the test statistic :orange[$W$] is less than or equal to the critical value from the Wilcoxon signed-rank distribution for a given significance level (e.g., α = 0.05) :orange[$W \\leq W_{critical}$]. 
                    - For large samples, reject the null hypothesis if the z-score is greater than or equal to the critical z-value corresponding to the chosen significance level :orange[$ |z| \\geq z_{critical} $].
                    - Fail to reject the null hypothesis if the test statistic :orange[$W$] is greater than the critical value, or if the z-score is less than the critical z-value (for larger samples).
                    - Effect size (r) can be calculated as: :orange[$$ r = \\dfrac{z}{\\sqrt{n}} $$]""",
    core_assumptions="""- **Ordinal or continuous data** — the variable is at least ordinal
- **Independence** — observations are independent of each other
- **Symmetric distribution** — the population distribution is symmetric around the median
- **Ordinal scale** — data can be meaningfully ranked
- **Meaningful median** — the hypothesized median is meaningful
- **No normality required** — this is a non-parametric test""",
    interpretation="""- If **p < α**, reject H₀: the population median differs from the hypothesized median
- Report: **W** = test statistic, **z** = z-score (large sample), **p** = p-value, **r** = effect size
- **Effect size r** = |z|/√n where: 0.1 = small, 0.3 = medium, 0.5 = large
- The test is **more powerful** than the Sign Test when the symmetry assumption holds
- If the distribution is heavily skewed, the Sign Test may be more appropriate
- The **median difference** or **Hodges-Lehmann estimator** provides a measure of effect location""",
    realworld_apps="""- **Pain research** — comparing patient pain scores to a clinical threshold
- **Satisfaction surveys** — testing if median satisfaction ratings differ from neutral
- **Income analysis** — comparing income to a poverty threshold (income is typically skewed)
- **Psychology** — evaluating psychological scale scores against a reference value
- **Toxicology** — analyzing biomarker levels with non-normal distributions""",
)

chi_square_goodness_of_fit_test = TestDefinition(
    name="Chi-Square Goodness-of-Fit Test",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var="None",
    groups="1",
    relation=["Independent", "Dependent", "any"],
    distribution=["Non-normal", "Normal", "any"],
    explanation="""Chi-Square Goodness-of-Fit Test This test is used to determine if a sample data fits a particular distribution. It compares the observed frequencies with the expected frequencies under the null hypothesis.""",
    example="""A researcher wants to test if the distribution of blood types in a sample of 100 people matches the expected distribution in the general population. The researcher performs a Chi-Square Goodness-of-Fit Test to determine if there is a significant difference between the observed and expected distributions.""",
    formula="""
                    $$ \\chi^2 = \\sum_{i=1}^{k} \\dfrac{(O_i - E_i)^2}{E_i} $$
                    Where:
                    - :orange[$\\chi^2$] is the test statistic,
                    - :orange[$O_i$] is the observed frequency for category :orange[$i$],
                    - :orange[$E_i$] is the expected frequency for category :orange[$i$], and
                    - :orange[$k$] is the number of categories.
                    - The test statistic :orange[$\\chi^2$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
    core_assumptions="""- **Categorical data** — the variable is nominal (unordered categories)
- **Independence** — observations are independent of each other
- **Random sampling** — the sample is representative
- **Expected frequencies ≥ 5** — each expected cell count should be at least 5 for the χ² approximation
- **Mutually exclusive categories** — each observation belongs to exactly one category
- **Exhaustive categories** — all possible categories are included""",
    interpretation="""- If **p < α**, reject H₀: the observed frequencies significantly differ from the expected distribution
- Report: **χ²** = test statistic, **df** = degrees of freedom (k−1), **p** = p-value
- The test is **omnibus** — it detects differences but does not identify which categories differ
- **Standardized residuals** (O−E)/√E indicate which categories contribute most to significance
- **Effect size**: Cramér's V = √(χ²/(n·k)) or the contingency coefficient
- If expected frequencies < 5, consider pooling adjacent categories or using the Exact Multinomial Test""",
    realworld_apps="""- **Genetics** — testing if observed genotype frequencies follow Mendelian ratios
- **Market research** — testing if brand preferences match expected market shares
- **Quality control** — checking if defect types follow historical patterns
- **Social science** — testing if survey responses match population demographics
- **Ecology** — comparing species distribution to expected proportions""",
)

multinomial_test = TestDefinition(
    name="Multinomial Test",
    objective="Comparison",
    dependent_var="Categorical",
    independent_var="None",
    groups="1",
    relation=["Independent", "Dependent", "any"],
    distribution=["Non-normal", "Normal", "any"],
    explanation="""Multinomial Test This test is used to determine whether the observed proportions across multiple categories differ from a hypothesized distribution. It is an extension of the binomial test to situations with more than two categories. Unlike the Chi-Square Goodness-of-Fit test (which uses a large-sample approximation), the exact multinomial test computes the p-value by enumerating all possible outcomes consistent with the observed total sample size, summing the probabilities of all outcomes that are as extreme or more extreme than the observed outcome. It is the gold standard when sample sizes are small.""",
    example="""A researcher expects that genotypes AA, Aa, and aa occur in a 1:2:1 ratio according to Mendelian inheritance. Among 20 offspring, they observe 8 AA, 10 Aa, and 2 aa. The multinomial test determines whether these observed counts significantly deviate from the expected 1:2:1 ratio, providing an exact p-value.""",
    formula="""
                    $$ P(\\mathbf{X} = \\mathbf{x}) = \\dfrac{n!}{x_1! \\, x_2! \\, \\ldots \\, x_k!} \\, p_1^{x_1} p_2^{x_2} \\ldots p_k^{x_k} $$
                    $$ \\text{p-value} = \\sum_{\\mathbf{y}: \\, P(\\mathbf{Y}) \\leq P(\\mathbf{x})} \\dfrac{n!}{y_1! \\, y_2! \\, \\ldots \\, y_k!} \\, p_1^{y_1} p_2^{y_2} \\ldots p_k^{y_k} $$
                    where:
                    - :orange[$n$] is the total number of trials,
                    - :orange[$k$] is the number of categories,
                    - :orange[$x_i$] is the observed count in category :orange[$i$],
                    - :orange[$p_i$] is the hypothesized probability for category :orange[$i$] under :orange[$H_0$],
                    - :orange[$\\sum_{i=1}^{k} p_i = 1$] and :orange[$\\sum_{i=1}^{k} x_i = n$].
                    """,
    decision_rules="""
                    - Reject the null hypothesis :orange[$H_0$] if the exact p-value is less than the chosen significance level (e.g., :orange[$\\alpha = 0.05$]).
                    - The p-value sums the multinomial probabilities of all possible outcomes that are no more probable than the observed outcome.
                    - For large sample sizes, the Chi-Square Goodness-of-Fit test can be used as an approximation.
                    - This test makes no distributional assumptions beyond random sampling and independent trials.
        """,
    core_assumptions="""- **Categorical data** — the variable is nominal with 3+ categories (for 2 categories, use the Binomial Test)
- **Independence** — observations are independent
- **Random sampling** — the sample is representative
- **Mutually exclusive and exhaustive categories**
- **No minimum expected frequency** — valid for any sample size
- **Computationally intensive** — exact enumeration can be demanding for large n and many categories""",
    interpretation="""- If the **exact p-value** < α, reject H₀: observed proportions differ from the hypothesized distribution
- Report: **exact p-value**, observed vs. expected proportions for each category
- This test is the **gold standard** for small samples where the χ² approximation is invalid
- For large samples, the **Chi-Square GOF** test provides a good approximation at lower computational cost
- The test enumerates all possible outcomes and sums probabilities of those as/more extreme than observed
- Effect size: report observed proportions with exact confidence intervals per category""",
    realworld_apps="""- **Genetics** — testing Mendelian inheritance ratios in small breeding experiments
- **Ecological studies** — comparing species abundance distributions with limited samples
- **Small-scale market research** — testing multi-brand preferences with few respondents
- **Educational research** — analyzing categorical achievement levels in small classes
- **Linguistics** — comparing word category frequencies to theoretical language models""",
)

runs_test_for_randomness = TestDefinition(
    name="Runs Test for Randomness",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical", "Continuous"],
    independent_var="None",
    groups="1",
    relation=["Independent", "Dependent", "any"],
    distribution=["any", "Non-normal", "Normal"],
    explanation="""The Runs Test for Randomness asks a fundamental question: Is this sequence random? A 'run' is a consecutive sequence of identical values or values above/below a threshold. Too few runs = clustered pattern; too many runs = alternating pattern. This test teaches the concept of randomness and is essential for time-series and quality control applications.""",
    example="""A quality control engineer examines 50 consecutive parts from an assembly line: G G G G B B G G G G G G B B B G G G G B... The engineer counts runs of Good and Bad parts. If runs are too few, the process may have clustering issues (e.g., a machine that drifts out of alignment). If runs are too many, there may be an alternating systematic issue.""",
    formula="""
                     Let:
                     - :orange[$n_1$] = number of observations of Type 1 (e.g., above median, or 'Successes')
                     - :orange[$n_2$] = number of observations of Type 2 (e.g., below median, or 'Failures')
                     - :orange[$R$] = number of runs observed
                     
                     Under H₀ (randomness):
                     $$ \\mu_R = \\frac{2 n_1 n_2}{n_1 + n_2} + 1 $$
                     $$ \\sigma_R = \\sqrt{\\frac{2 n_1 n_2 (2 n_1 n_2 - n_1 - n_2)}{(n_1 + n_2)^2 (n_1 + n_2 - 1)}} $$
                     
                     For large samples:
                     $$ z = \\frac{R - \\mu_R}{\\sigma_R} \\sim N(0, 1) $$
                     
                     **Left-tailed**: Too few runs (clustering)
                     **Right-tailed**: Too many runs (alternating pattern)
                     **Two-tailed**: Either extreme (non-randomness of any form)
                     """,
    decision_rules="""
                     - If :orange[$R \\ll \\mu_R$]: Clustered pattern → reject randomness
                     - If :orange[$R \\gg \\mu_R$]: Alternating pattern → reject randomness
                     - Two-sided test: Reject if :orange[$|z| > z_{\\alpha/2}$] or p-value < α
                     - For continuous data: typically coded as values **above** vs **below** the median (or mean)
                     - The test can also be applied to categorical sequences with more than two categories (extensions exist)
         """,
    core_assumptions="""- **Dichotomous data** — the sequence can be classified into two categories (or above/below a threshold)
- **Independent observations** — this is what the test checks
- **Pre-specified threshold** — the split value (e.g., median) should be determined before seeing the data
- **Meaningful order** — the sequence order is important and reflects the process
- **For continuous data** — convert to binary using above/below median (or another meaningful cutoff)""",
    interpretation="""- If **p < α**, reject H₀: the sequence is **not random**
- **Too few runs** (R ≪ μ_R) → **positive autocorrelation** (clustering): similar values tend to occur together
- **Too many runs** (R ≫ μ_R) → **negative autocorrelation** (alternating): values systematically alternate
- Report: **R** = observed runs, **μ_R** = expected runs, **p-value**
- Use a **two-sided test** for general non-randomness
- Use a **one-sided test** if specifically looking for clustering (left-tailed) or alternation (right-tailed)
- This is a fundamental diagnostic **before** applying time-series models""",
    realworld_apps="""- **Quality control** — checking if production defects occur randomly or in clusters
- **Time-series analysis** — testing for independence before fitting ARIMA models
- **Financial analysis** — testing if stock price movements are truly random (weak-form EMH)
- **Clinical trials** — verifying that treatment assignments in a sequence are random
- **Environmental monitoring** — testing if pollution events occur randomly over time
- **Genomics** — testing if DNA sequence patterns occur randomly along a chromosome""",
)

poisson_goodness_of_fit_test = TestDefinition(
    name="Poisson Goodness-of-Fit Test",
    objective="Comparison",
    dependent_var="Categorical",
    independent_var="None",
    groups="1",
    relation=["Independent", "Dependent", "any"],
    distribution=["Non-normal", "Normal", "any"],
    explanation="""The Poisson Goodness-of-Fit test determines whether count data follows a Poisson distribution. The Poisson distribution is fundamental for modeling: (1) number of events in a fixed interval, (2) independent events, (3) constant average rate. A key property: for a true Poisson, the variance equals the mean. If variance > mean, you have **over-dispersion** (common in real data), suggesting Negative Binomial regression may be more appropriate. This test teaches the Poisson assumptions and how to check them.""",
    example="""A hospital administrator counts the number of emergency admissions per hour over 100 hours: 0 admissions in 25 hours, 1 admission in 35 hours, 2 admissions in 25 hours, 3+ admissions in 15 hours. The administrator fits a Poisson distribution to these counts and tests whether the observed frequencies match the expected Poisson frequencies.""",
    formula="""
                     **Step 1: Estimate the Poisson parameter (λ)** from the sample:
                     $$ \\hat{\\lambda} = \\bar{x} = \\text{sample mean} $$
                     
                     **Step 2: Compute expected frequencies** for each count k:
                     $$ P(X = k) = \\frac{e^{-\\hat{\\lambda}} \\hat{\\lambda}^k}{k!} $$
                     $$ E_k = n \\times P(X = k) $$
                     
                     **Step 3: Pool categories** so that all :orange[$E_k \\geq 5$] (requirement for Chi-square approximation)
                     
                     **Step 4: Compute Chi-square statistic**:
                     $$ \\chi^2 = \\sum_{k} \\frac{(O_k - E_k)^2}{E_k} $$
                     
                     **Step 5: Degrees of freedom**:
                     $$ df = \\text{(\\# categories after pooling)} - 1 - \\text{(\\# parameters estimated)} $$
                     
                     For Poisson GOF: **df = k - 2** (subtract 1 for total sum, subtract 1 more for estimating λ)
                     """,
    decision_rules="""
                     - Reject :orange[$H_0$] if :orange[$\\chi^2 > \\chi^2_{\\alpha, df}$] or p-value < α
                     - If rejected → data does NOT follow Poisson distribution
                     
                     **Checking for Over-dispersion**:
                     
                     A key diagnostic for count data:
                     $$ \\text{Variance/Mean Ratio} = \\frac{s^2}{\\bar{x}} $$
                     
                     - Ratio ≈ 1 → Consistent with Poisson
                     - Ratio > 1 → **Over-dispersion** (more variance than Poisson predicts)
                     - Ratio < 1 → **Under-dispersion** (less variance than Poisson predicts)
                     
                     **Over-dispersion is common** and suggests:
                     - Use **Negative Binomial regression** instead of Poisson regression
                     - Or consider **zero-inflated Poisson (ZIP)** if there are excess zeros
                     
                     **Exact alternatives**: For very small samples, the Kolmogorov-Smirnov test (with estimated parameters requires special tables) or simulation-based approaches.
         """,
    core_assumptions="""- **Count data** — observations are non-negative integers (0, 1, 2, …)
- **Independence** — events occur independently of each other
- **Constant rate** — the average rate λ is constant over the observation period
- **No interaction** — events do not influence the occurrence of subsequent events
- **Pooled expected ≥ 5** — after pooling categories, all expected frequencies should be ≥ 5
- **Random sampling** — the data is a representative sample of the process""",
    interpretation="""- If **p < α**, reject H₀: the data does **not** follow a Poisson distribution
- **Variance/Mean ratio**: ≈ 1 → Poisson, > 1 → over-dispersion, < 1 → under-dispersion
- **Over-dispersion** (ratio > 1) suggests **Negative Binomial regression** may be more appropriate
- **Under-dispersion** (ratio < 1) is less common but can occur with highly regular processes
- Report: **χ²** = test statistic, **df** = (categories pooled − 2), **p-value**, **variance/mean ratio**
- The variance/mean ratio is a simple and powerful diagnostic tool for count data modeling""",
    realworld_apps="""- **Epidemiology** — modeling the number of disease cases per day/week
- **Insurance** — modeling the number of claims per policy period
- **Manufacturing** — modeling the number of defects per unit area or time interval
- **Biology** — modeling cell counts per microscopic field
- **Call centers** — modeling incoming call volumes per minute/hour
- **Traffic engineering** — modeling vehicle arrivals at an intersection per minute""",
)

student_s_t_test_independent = TestDefinition(
    name="Student's t-test (Independent)",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="Normal",
    explanation="""Student's t-test (Independent) This test is used to compare the means of two independent groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, and that the variances of the two groups are equal.""",
    example="""A researcher wants to compare the average blood pressure between two groups of patients: those who received a new drug and those who received a placebo. The researcher collects blood pressure readings from 30 patients in each group and performs an independent t-test to determine if there is a significant difference in mean blood pressure between the two groups.""",
    formula="""
                    $$ t = \\dfrac{\\bar{x}_1 - \\bar{x}_2}{s_p \\sqrt{\\dfrac{1}{n_1} + \\dfrac{1}{n_2}}} $$ 
                    Where:  
                    - :orange[$\\bar{x}_1$] and :orange[$\\bar{x}_2$] are the sample means of the two groups,  
                    - :orange[$n_1$] and :orange[$n_2$] are the sample sizes of the two groups, and  
                    - :orange[$s_p$] is the pooled standard deviation calculated as: $$ s_p = \\sqrt{\\dfrac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}} $$ 
                    where  
                    - :orange[$s_1^2$] and :orange[$s_2^2$] are the sample variances of the two groups.
                    """,
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Independence** — observations between groups are independent
- **Normal distribution** — the DV is approximately normally distributed in each group
- **Homogeneity of variances** — the variance is roughly equal across groups (check with Levene's test)
- **Random sampling** — samples are representative of their populations""",
    interpretation="""- If **p < α** (typically 0.05), reject H₀: the group means are significantly different
- Report: **t(df)** = t-value, **p** = p-value, **d** = Cohen's d
- **Cohen's d** effect size: 0.2 = small, 0.5 = medium, 0.8 = large
- **Confidence interval** for the mean difference indicates the precision and direction of the effect
- If Levene's test is significant (variances unequal), use **Welch's t-test** instead""",
    realworld_apps="""- **Clinical research**: comparing blood pressure between treatment and placebo groups
- **Education**: comparing test scores between two teaching methods
- **Manufacturing**: comparing product measurements from two production lines
- **Psychology**: comparing reaction times between two experimental conditions
- **Agriculture**: comparing crop yields between two fertilizer treatments""",
)

welch_s_t_test_independent_unequal_variances = TestDefinition(
    name="Welch's t-test (Independent, Unequal Variances)",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="Normal",
    explanation="""Welch's t-test (Independent, Unequal Variances) This test is used to compare the means of two independent groups when the variances are assumed to be unequal. It is a modification of the Student's t-test.""",
    example="""A researcher wants to compare the average blood pressure between two groups of patients: those who received a new drug and those who received a placebo. The researcher collects blood pressure readings from 30 patients in each group and performs Welch's t-test to determine if there is a significant difference in mean blood pressure between the two groups.""",
    formula="""
                    $$ t = \\dfrac{\\bar{x}_1 - \\bar{x}_2}{\\sqrt{\\dfrac{s_1^2}{n_1} + \\dfrac{s_2^2}{n_2}}} $$ 
                    Where: 
                    - :orange[$\\bar{x}_1$] and :orange[$\\bar{x}_2$] are the sample means of the two groups, 
                    - :orange[$n_1$] and :orange[$n_2$] are the sample sizes of the two groups, and 
                    - :orange[$s_1^2$] and :orange[$s_2^2$] are the sample variances of the two groups.
                    """,
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Independence** — observations between groups are independent
- **Normal distribution** — the DV is approximately normally distributed in each group (robust with large samples)
- **Does NOT assume equal variances** — the key advantage over Student's t-test
- **Random sampling** — samples are representative""",
    interpretation="""- If **p < α**, reject H₀: the group means are significantly different
- Report: **t(df)** = t-value with Welch-Satterthwaite adjusted df, **p** = p-value
- Degrees of freedom are typically **non-integer** and smaller than Student's t-test
- **Effect size**: report Cohen's d or the mean difference with confidence interval
- Many statisticians recommend **always using Welch's test** by default, even when variances appear equal""",
    realworld_apps="""- **Clinical research**: comparing groups with inherently different variability (e.g., healthy vs diseased)
- **Education**: comparing test scores across classes with different score spreads
- **Environmental science**: comparing pollution levels across sites with different variability
- **Economics**: comparing income between regions with different income inequality
- **Genomics**: comparing gene expression across conditions with unequal variances""",
)

f_test_for_two_variances = TestDefinition(
    name="F-Test for Two Variances",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="Normal",
    explanation="""The F-test for equality of variances compares the spread (variability) of two independent samples. It tests the homogeneity of variance assumption that underlies Student's t-test and ANOVA. The test statistic is simply the ratio of the two sample variances. Important: This test is EXTREMELY sensitive to non-normality — much more so than the t-test itself. For this reason, robust alternatives like Levene's test are generally preferred, but the F-test remains valuable for teaching the concept of variance ratios and the F-distribution.""",
    example="""A researcher plans to use Student's t-test to compare two methods. First, they check the equal variance assumption: Method A (n=20) has variance = 12.5, Method B (n=20) has variance = 45.2. The F-ratio = 45.2/12.5 = 3.62. Is this large enough to reject equal variances?""",
    formula="""
                     $$ F = \\frac{s_1^2}{s_2^2} \\quad \\text{or typically} \\quad F = \\frac{\\max(s_1^2, s_2^2)}{\\min(s_1^2, s_2^2)} $$
                     
                     Where:
                     - :orange[$s_1^2$] = variance of sample 1
                     - :orange[$s_2^2$] = variance of sample 2
                     
                     Under :orange[$H_0: \\sigma_1^2 = \\sigma_2^2$], the F-statistic follows an **F-distribution** with:
                     - Numerator df = :orange[$n_1 - 1$]
                     - Denominator df = :orange[$n_2 - 1$]
                     
                     **Two-sided test**: Reject if :orange[$F > F_{\\alpha/2, df_1, df_2}$] or :orange[$F < F_{1-\\alpha/2, df_1, df_2}$]
                     
                     By convention, most software places the larger variance in the numerator, giving F ≥ 1, and then doubles the one-tailed p-value.
                     """,
    decision_rules="""
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
    core_assumptions="""- **Continuous DV** — the variable is measured on a continuous scale
- **Independence** — samples are independent of each other
- **Normal distribution** — both populations are normally distributed (F-test is very sensitive to non-normality)
- **Random sampling** — samples are representative""",
    interpretation="""- If **p < α**, reject H₀: the variances are significantly different
- Report: **F** = F-statistic (ratio of larger variance to smaller), **df₁** = n₁−1, **df₂** = n₂−1, **p** = p-value
- A significant result means the equal variance assumption (for t-test/ANOVA) is violated
- **Caution**: the F-test is highly sensitive to non-normality — use Levene's test for a robust alternative
- This test is most useful as an **assumption check** before applying parametric tests""",
    realworld_apps="""- **Quality control**: comparing the consistency (variability) of two production processes
- **Method comparison**: checking if two measurement instruments have equal precision
- **Pre-study checks**: verifying equal variance assumption before Student's t-test or ANOVA
- **Finance**: comparing the volatility (risk) of two investment portfolios
- **Manufacturing**: checking if a new process reduces variability compared to the old process""",
)

equivalence_test_tost_two_independent_samples = TestDefinition(
    name="Equivalence Test (TOST) - Two Independent Samples",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="Normal",
    explanation="""TOST = Two One-Sided Tests. This procedure solves a critical statistical misconception: 'p > 0.05' does NOT mean 'treatments are equivalent'. TOST explicitly reverses the burden of proof: it requires data to demonstrate that the true difference lies within a pre-specified 'equivalence range' (-Δ, +Δ). Only if BOTH one-sided tests reject their respective null hypotheses can you conclude equivalence. This is essential for teaching the difference between 'no evidence of difference' and 'evidence of no difference'.""",
    example="""A pharmaceutical company wants to show that a new generic drug is 'equivalent' to the brand-name drug. They define equivalence as having a mean difference in blood pressure of less than 5 mmHg (Δ = 5). They test: (1) Is the true difference > -5? (2) Is the true difference < +5? If both tests reject, equivalence is established. A simple t-test showing 'no significant difference' (p > 0.05) would not be sufficient — equivalence must be demonstrated.""",
    formula="""
                     **Equivalence Margin**: :orange[$\\Delta$] (smallest difference considered 'meaningfully different')
                     
                     **Traditional superiority null**: :orange[$H_0: \\mu_1 - \\mu_2 = 0$]
                     
                     **TOST null hypotheses** (both must be rejected for equivalence):
                     $$ H_{0L}: \\mu_1 - \\mu_2 \\leq -\\Delta \\quad \\text{(difference is too low)} $$
                     $$ H_{0U}: \\mu_1 - \\mu_2 \\geq +\\Delta \\quad \\text{(difference is too high)} $$
                     
                     **Alternative hypotheses**:
                     $$ H_{1L}: \\mu_1 - \\mu_2 > -\\Delta $$
                     $$ H_{1U}: \\mu_1 - \\mu_2 < +\\Delta $$
                     
                     **Two one-sided t-tests**:
                     $$ t_L = \\frac{(\\bar{x}_1 - \\bar{x}_2) - (-\\Delta)}{SE} = \\frac{\\bar{D} + \\Delta}{SE} $$
                     $$ t_U = \\frac{(\\bar{x}_1 - \\bar{x}_2) - (+\\Delta)}{SE} = \\frac{\\bar{D} - \\Delta}{SE} $$
                     
                     Where :orange[$SE = \\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}}$] (or pooled if assuming equal variances)
                     
                     **Decision**: Equivalence concluded if :orange[$p_L < \\alpha$] **AND** :orange[$p_U < \\alpha$]
                     
                     **Confidence Interval approach**: Equivalence concluded if the :orange[$100(1-2\\alpha)\\%$] (or 90% for α=0.05) CI for :orange[$\\mu_1 - \\mu_2$] lies entirely within :orange[$(-\\Delta, +\\Delta)$].
                     """,
    decision_rules="""
                     **Key Principle**: The burden of proof is on demonstrating equivalence.
                     
                     1. **Define equivalence margin (Δ)** BEFORE data collection
                        - Typically based on clinical or practical significance
                        - Common choices: 0.2×SD of reference, or regulatory standards
                     
                     2. **Run both one-sided tests**:
                        - Reject :orange[$H_{0L}$] if :orange[$t_L > t_{\\alpha, df}$] (one-tailed)
                        - Reject :orange[$H_{0U}$] if :orange[$t_U < -t_{\\alpha, df}$] (one-tailed)
                     
                     3. **Conclusion**:
                        - If **both** rejected → **Equivalence demonstrated**
                        - If either not rejected → **Cannot claim equivalence**
                     
                     **Common Misconception**:
                     - ❌ 'p > 0.05 in t-test' ≠ 'Equivalent'
                     - ✅ Only TOST or CI-within-range can demonstrate equivalence
                     
                     **Visual Interpretation**: The 90% confidence interval must lie COMPLETELY inside (-Δ, +Δ) for equivalence.
         """,
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Independence** — observations between groups are independent
- **Normal distribution** — the DV is approximately normally distributed in each group
- **Pre-specified equivalence bounds** — define the equivalence margin (Δ) in advance, representing the largest negligible difference
- **Homogeneity of variances** (for the Student's version)""",
    interpretation="""- TOST **reverses** the traditional hypothesis: you seek to **confirm** equivalence within bounds
- If both one-sided tests are significant (p < α), conclude **statistical equivalence** within ±Δ
- Report: **t₁** and **t₂** (the two test statistics), **p** = larger of the two p-values, **90% CI** for mean difference
- The equivalence bounds Δ must be chosen **before** seeing the data, based on practical significance
- A non-significant TOST does NOT prove groups differ — it means equivalence could not be demonstrated""",
    realworld_apps="""- **Bioequivalence studies**: testing if a generic drug has the same effect as the brand-name drug
- **Generic drug approval**: FDA requires demonstrating equivalence within ±20% of the reference mean
- **Method validation**: showing a new measurement method agrees with the gold standard
- **Manufacturing**: proving a new (cheaper) process produces equivalent quality
- **Environmental monitoring**: showing pollution levels at a site are within acceptable limits""",
)

paired_t_test = TestDefinition(
    name="Paired t-test",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Dependent",
    distribution="Normal",
    explanation="""Paired t-test This test is used to compare the means of two related groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, and that the pairs are dependent (e.g., measurements taken from the same subjects before and after a treatment).""",
    example="""A researcher wants to test if a new drug reduces blood pressure in patients. The researcher measures the blood pressure of 30 patients before and after administering the drug. The researcher performs a paired t-test to determine if there is a significant difference in mean blood pressure before and after the treatment.""",
    formula="""
                    $$ t = \\dfrac{\\bar{d}}{s_d / \\sqrt{n}} $$ 
                    Where: 
                    - :orange[$\\bar{d}$] is the mean of the differences between paired observations, 
                    - :orange[$s_d$] is the standard deviation of the differences, and 
                    - :orange[$n$] is the number of pairs. 
                    - The test statistic :orange[$t$] is then compared to a critical value from the t-distribution with :orange[$n-1$] degrees of freedom to determine significance.
                    """,
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Paired observations** — each subject is measured twice (before/after) or subjects are matched pairs
- **Independence between pairs** — each pair is independent of other pairs
- **Normal differences** — the **difference scores** are approximately normally distributed
- **No extreme outliers** in the difference scores""",
    interpretation="""- If **p < α**, reject H₀: the mean difference between paired observations is significantly different from zero
- Report: **t(df)** = t-value (df = n_pairs − 1), **p** = p-value, **d** = Cohen's d_z
- **Cohen's d_z** = mean_difference / SD_of_differences
- Each subject serves as their own control, removing between-subject variability for **greater power**
- **Confidence interval** for the mean difference provides the range of plausible effect sizes""",
    realworld_apps="""- **Before-after studies**: measuring blood pressure before and after treatment in the same patients
- **Matched case-control**: comparing patients matched by age, sex, and confounders
- **Crossover trials**: subjects receive both treatment and placebo in random order
- **Product testing**: comparing ratings of the same product under different conditions
- **Longitudinal research**: measuring cognitive scores at baseline and follow-up""",
)

one_way_anova = TestDefinition(
    name="One-way ANOVA",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="More than 2",
    relation="Independent",
    distribution="Normal",
    explanation="""One-way ANOVA This test is used to compare the means of three or more independent groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, that the variances are equal across groups (homogeneity of variance), and that the groups are independent.""",
    example="""A researcher wants to compare the average blood pressure between three groups of patients: those who received a new drug, those who received a different drug, and those who received a placebo. The researcher collects blood pressure readings from 30 patients in each group and performs one-way ANOVA to determine if there is a significant difference in mean blood pressure between the three groups.""",
    formula="""
                    $$ F = \\dfrac{MS_{between}}{MS_{within}} $$ 
                    $$ MS_{between} = \\dfrac{SS_{between}}{df_{between}} $$
                    $$ MS_{within} = \\dfrac{SS_{within}}{df_{within}} $$
                    $$ SS_{between} = \\sum_{i=1}^{k} n_i (\\bar{X}_i - \\bar{X})^2 $$
                    $$ SS_{within} = \\sum_{i=1}^{k} \\sum_{j=1}^{n_i} (X_{ij} - \\bar{X}_i)^2 $$
                    $$ df_{between} = k - 1 $$
                    $$ df_{within} = N - k $$
                    Where: 
                    - :orange[$F$] is the F-statistic, 
                    - :orange[$MS_{between}$] is the mean square between groups, and 
                    - :orange[$MS_{within}$] is the mean square within groups. 
                    Mean squares are calculated as: 
                    - $$ MS_{between} = \\dfrac{SS_{between}}{df_{between}} $$ and 
                    - $$ MS_{within} = \\dfrac{SS_{within}}{df_{within}} $$ 
                    where:
                    - :orange[$SS$] is the sum of squares and :orange[$df$] is the degrees of freedom for between and within groups.
                    """,
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Independence** — observations between groups are independent
- **Normal distribution** — the DV is approximately normally distributed in **each group**
- **Homogeneity of variances** — variances are roughly equal across all groups (Levene's test)
- **One categorical IV** — with 3+ levels (groups)""",
    interpretation="""- If **p < α**, reject H₀: at least one group mean is significantly different from the others
- Report: **F(df_between, df_within)** = F-value, **p** = p-value, **η²** = eta-squared
- **Eta-squared (η²)** = SS_between / SS_total — proportion of variance explained by group membership
- The F-test is **omnibus** — it only tells you differences exist, NOT which groups differ
- **Post-hoc tests** (e.g., Tukey HSD, Bonferroni) identify specific group differences
- **η² interpretation**: 0.01 = small, 0.06 = medium, 0.14 = large effect""",
    post_hoc="""Tukey HSD, Bonferroni, Holm-Bonferroni, Šidák, Scheffé, Dunnett, Games-Howell, Fisher LSD, Newman-Keuls""",
    realworld_apps="""- **Clinical trials**: comparing outcomes across three or more dosage levels
- **Agriculture**: comparing crop yields across multiple fertilizer types
- **Education**: comparing exam scores across teaching methods (lecture, online, hybrid)
- **Marketing**: comparing customer satisfaction across multiple product versions
- **Ecology**: comparing species diversity across different habitat types""",
)

repeated_measures_anova = TestDefinition(
    name="Repeated Measures ANOVA",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="More than 2",
    relation="Dependent",
    distribution="Normal",
    explanation="""Repeated Measures ANOVA This test is used to compare the means of three or more related groups to determine if there is a statistically significant difference between them. It assumes that the data is continuous, follows a normal distribution, that the variances of the differences between all pairs of repeated measures are equal (sphericity), and that the groups are dependent (e.g., measurements taken from the same subjects at multiple time points).""",
    example="""A researcher wants to test the effect of a new drug on blood pressure over time. The researcher measures the blood pressure of 30 patients at three different time points: before treatment, after 1 month of treatment, and after 3 months of treatment. The researcher performs a repeated measures ANOVA to determine if there is a significant difference in mean blood pressure across the three time points.""",
    formula="""
                    $$ F = \\dfrac{MS_{between}}{MS_{error}} $$ 
                    $$ MS_{between} = \\dfrac{SS_{between}}{df_{between}} $$
                    $$ MS_{error} = \\dfrac{SS_{error}}{df_{error}} $$
                    $$ SS_{between} = \\sum_{i=1}^{k} n_i (\\bar{X}_i - \\bar{X})^2 $$
                    $$ SS_{error} = \\sum_{i=1}^{k} \\sum_{j=1}^{n_i} (X_{ij} - \\bar{X}_i)^2 $$
                    $$ df_{between} = k - 1 $$
                    $$ df_{error} = N - k $$
                    $$ N = \\sum_{i=1}^{k} n_i $$
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
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Within-subjects factor** — the same subjects are measured at 3+ time points or conditions
- **Independence between subjects** — subjects are independent of each other
- **Sphericity** — the variances of differences between all pairs of repeated measures are equal
- **Normal distribution** — the DV is normally distributed at each time point""",
    interpretation="""- If **p < α**, reject H₀: means differ significantly across time points or conditions
- Report: **F(df_num, df_den)** = F-value, **p** = p-value, **η²_p** = partial eta-squared
- If **Mauchly's test** is significant (sphericity violated), use **Greenhouse-Geisser** or **Huynh-Feldt** correction
- **Post-hoc**: paired t-tests with Bonferroni correction or Tukey's test for repeated measures
- This design has **greater statistical power** because each subject serves as their own control""",
    post_hoc="Pairwise Paired t, Paired t + Bonferroni, Paired t + Holm",
    realworld_apps="""- **Clinical trials**: measuring blood pressure at baseline, 1 month, 3 months, and 6 months
- **Psychology**: tracking mood ratings daily over a week-long intervention
- **Exercise science**: measuring strength before training, after 4 weeks, and after 8 weeks
- **Cognitive research**: testing reaction times across multiple task difficulty levels
- **Nutrition studies**: measuring biomarkers at multiple time points after a dietary intervention""",
)

two_way_anova = TestDefinition(
    name="Two-way ANOVA",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var="Categorical",
    groups="More than 2",
    relation="Independent",
    distribution="Normal",
    explanation="""Two-way ANOVA This test evaluates the effect of two categorical independent variables (factors) on a continuous dependent variable, as well as their interaction. For example, it can test the effect of treatment (drug vs. placebo) and sex (male vs. female) on blood pressure, and whether the treatment effect differs by sex. It assumes normality, homogeneity of variances, and independence of observations.""",
    example="""A researcher wants to test the effects of a new drug and sex on blood pressure. 60 patients are divided into drug and placebo groups, each containing equal numbers of males and females. Two-way ANOVA is used to test for main effects of drug and sex, and their interaction.""",
    formula="""
                    $$ SS_{total} = SS_A + SS_B + SS_{AB} + SS_{error} $$
                    $$ F_A = \\frac{MS_A}{MS_{error}}, \\quad F_B = \\frac{MS_B}{MS_{error}}, \\quad F_{AB} = \\frac{MS_{AB}}{MS_{error}} $$
                    Where:
                    - :orange[$SS_A$] and :orange[$SS_B$] are sums of squares for factors A and B,
                    - :orange[$SS_{AB}$] is the sum of squares for the interaction,
                    - :orange[$MS$] values are mean squares (SS/df),
                    - :orange[$F$]-statistics test the main effects and interaction separately,
                    - Significance is determined by comparing each :orange[$F$] to the F-distribution with appropriate df.
                    """,
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Independence** — observations are independent
- **Normal distribution** — the DV is normally distributed in each combination of factor levels
- **Homogeneity of variances** — variances are equal across all groups
- **Two categorical IVs** — two independent variables (factors), each with 2+ levels
- **Balanced design preferred** — equal sample sizes per cell make the test more robust""",
    interpretation="""- Report **three F-tests**: main effect of Factor A, Factor B, and the A×B interaction
- If the **interaction** is significant, interpret it first — main effects may be misleading
- Report: **F(df₁, df₂)** = F-value, **p** = p-value, **η²_p** = partial eta-squared
- **Simple main effects** should be examined following a significant interaction
- **Partial η²**: 0.01 = small, 0.06 = medium, 0.14 = large""",
    post_hoc="""Tukey HSD, Bonferroni, Holm-Bonferroni, Šidák, Scheffé, Dunnett, Games-Howell, Fisher LSD, Newman-Keuls""",
    realworld_apps="""- **Clinical research**: testing drug effect across both sexes (male vs female)
- **Education**: testing teaching method effectiveness across student ability levels
- **Marketing**: analyzing how advertising type and price jointly affect sales
- **Ergonomics**: studying how workstation design and shift length affect worker fatigue
- **Sports science**: investigating how training program and diet interact to affect performance""",
)

ancova = TestDefinition(
    name="ANCOVA",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Categorical", "Continuous"],
    groups="More than 2",
    relation="Independent",
    distribution="Normal",
    explanation="""Analysis of Covariance (ANCOVA) combines ANOVA and linear regression. It compares group means on a continuous dependent variable while statistically controlling for the effect of one or more continuous covariates. This increases statistical power by reducing within-group error variance and adjusts for baseline differences. It assumes normality, homogeneity of variances, homogeneity of regression slopes, and linearity between covariate and outcome.""",
    example="""A researcher wants to compare post-treatment blood pressure between three drug groups while controlling for baseline blood pressure. ANCOVA adjusts the post-treatment means for baseline differences, providing a more precise estimate of treatment effects.""",
    formula="""
                    $$ F = \\frac{MS_{between}}{MS_{error}} $$
                    $$ MS_{between} = \\frac{SS_{between}}{df_{between}}, \\quad MS_{error} = \\frac{SS_{error}}{df_{error}} $$
                    $$ SS_{between} = \\sum_{j=1}^{k} n_j (\\bar{Y}_j - \\bar{Y}_{adj})^2 $$
                    $$ \\bar{Y}_{adj} = \\bar{Y} - \\beta(\\bar{X} - \\bar{X}_{overall}) $$
                    $$ SS_{error} = \\sum_{i=1}^{n} (Y_i - \\hat{Y}_i)^2 $$
                    $$ \\hat{Y}_i = \\mu + \\tau_j + \\beta(X_i - \\bar{X}) $$
                    Where:
                    - The dependent variable :orange[$Y$] is modeled as: $$ Y_{ij} = \\mu + \\tau_j + \\beta(X_{ij} - \\bar{X}) + \\epsilon_{ij} $$
                    - :orange[$\\tau_j$] is the effect of the :orange[$j$]-th group,
                    - :orange[$\\beta$] is the regression coefficient for the covariate :orange[$X$],
                    - :orange[$MS_{error}$] is reduced by the variance explained by the covariate, increasing power.
                    """,
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Continuous covariate(s)** — one or more continuous variables to adjust the DV
- **Independence** — observations are independent
- **Normal distribution** — residuals are normally distributed
- **Homogeneity of variances** — variances are equal across groups
- **Homogeneity of regression slopes** — covariate-DV relationship is similar across groups
- **Linear relationship** — the covariate has a linear relationship with the DV""",
    interpretation="""- Tests whether group means differ **after adjusting for the covariate(s)**
- If **p < α** for the group effect, reject H₀: adjusted group means are significantly different
- Report: **F(df₁, df₂)** = F-value, **p** = p-value, **η²_p** = partial eta-squared
- **Adjusted means** (least-squares means) are the estimated group means at the mean covariate value
- ANCOVA **increases statistical power** by reducing the error term (controlling for the covariate)""",
    post_hoc="""Tukey HSD, Bonferroni, Holm-Bonferroni, Šidák (for adjusted group comparisons after covariate adjustment)""",
    realworld_apps="""- **Clinical trials**: comparing post-treatment outcomes while controlling for baseline values
- **Education**: comparing teaching methods while controlling for pre-test scores
- **Psychology**: comparing therapy outcomes while controlling for initial symptom severity
- **Nutrition**: comparing diet effects on weight loss while controlling for baseline BMI
- **Epidemiology**: comparing health outcomes across groups controlling for age and SES""",
)

manova = TestDefinition(
    name="MANOVA",
    objective="Comparison",
    dependent_var="Multiple Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="More than 2",
    relation="Independent",
    distribution="Normal",
    explanation="""MANOVA (Multivariate Analysis of Variance) This test is used to compare the means of multiple dependent variables across two or more independent groups. It assumes that the data is continuous, follows a multivariate normal distribution, and that the groups are independent.""",
    example="""A researcher wants to compare the effects of three different diets on both weight loss and cholesterol levels. The researcher collects data on weight loss and cholesterol levels from 30 patients in each diet group and performs a MANOVA to determine if there are significant differences in the combined dependent variables (weight loss and cholesterol levels) across the three diet groups.""",
    formula="""
                    $$ \\Lambda = \\dfrac{|\\mathbf{E}|}{|\\mathbf{E} + \\mathbf{H}|} $$ 
                    Where: 
                    - :orange[$\\Lambda$] is the test statistic (Wilks' Lambda), 
                    - :orange[$\\mathbf{E}$] is the error sum of squares and cross-products matrix, and 
                    - :orange[$\\mathbf{H}$] is the hypothesis sum of squares and cross-products matrix. 
                    - The test statistic :orange[$\\Lambda$] is then transformed into an F-statistic for significance testing.
                    """,
    core_assumptions="""- **Multiple continuous DVs** — two or more correlated outcome variables
- **Independence** — observations are independent
- **Multivariate normality** — the DVs jointly follow a multivariate normal distribution in each group
- **Homogeneity of covariance matrices** — variance-covariance matrices are equal across groups
- **No extreme multicollinearity** — DVs should be correlated but not excessively (r < 0.90)""",
    interpretation="""- If **p < α**, reject H₀: groups differ on the **combination** of DVs
- Report: **Wilks' Λ**, **Pillai's Trace**, or **Hotelling's Trace** with corresponding F-approximation
- **Pillai's Trace** is the most robust to assumption violations and recommended for general use
- Following significant MANOVA, examine **univariate ANOVAs** for each DV to identify specific differences
- **Discriminant Function Analysis** can be used post-hoc to understand how DVs combine to distinguish groups""",
    post_hoc="""Discriminant Comparisons (pairwise univariate F), Canonical Contrasts (Bonferroni-corrected)""",
    realworld_apps="""- **Clinical trials**: testing treatment effects on multiple correlated outcomes (pain, swelling, mobility)
- **Psychology**: comparing groups on multiple personality subscales simultaneously
- **Neuroscience**: analyzing multiple brain region activation levels across conditions
- **Education**: evaluating interventions using multiple achievement measures (reading, math, science)
- **Marketing**: comparing brand perceptions across multiple dimensions (quality, value, style)""",
)

sign_test_paired = TestDefinition(
    name="Sign Test (Paired)",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Dependent",
    distribution="Non-normal",
    explanation="""The Paired Sign Test compares two related (matched/paired) measurements by examining only the signs of their differences. It asks a simple question: Is one measurement typically greater than the other? Like the one-sample Sign Test, it ignores the magnitude of differences — trading power for simplicity and robustness.""",
    example="""A sports scientist tests whether a new running shoe reduces 5K times. For 20 athletes, 14 run faster with the new shoe, 5 run slower, and 1 shows no change. The Paired Sign Test determines if this pattern (more positives) is unlikely under the null hypothesis of no difference.""",
    formula="""
                     $$ D_i = X_{i,\\text{after}} - X_{i,\\text{before}} \\quad \\text{or} \\quad D_i = X_{i,\\text{Method A}} - X_{i,\\text{Method B}} $$
                     $$ S = \\min(\\text{number of positive } D_i, \\text{number of negative } D_i) $$
                     
                     Under H₀, S ~ Binomial(n', 0.5) where n' = number of non-zero differences.
                     
                     **Exact p-value** computed directly from the binomial distribution.
                     """,
    core_assumptions="""- **Paired observations** — each subject contributes two measurements (or subjects are matched)
- **Ordinal or continuous data** — the variable is at least ordinal
- **Independence between pairs** — each pair is independent of other pairs
- **No distributional assumptions** — does NOT require normality or symmetry
- **Continuous distribution preferred** — avoids ties (differences = 0 are dropped)""",
    interpretation="""- If the **exact p-value** < α, reject H₀: the median difference is not zero
- Report: **S⁺** (positive differences), **S⁻** (negative), **n'** (effective n), **p-value**
- **Effect size**: proportion of positive differences = S⁺/n' or the median difference
- This is the **paired version** of the one-sample Sign Test
- Ties (differences = 0) are excluded, reducing power
- Less powerful than the paired Wilcoxon test but makes NO symmetry assumptions""",
    realworld_apps="""- **Preference testing**: asking consumers which of two products they prefer
- **Before-after studies**: recording whether symptoms improved, worsened, or stayed the same
- **Medical screening**: comparing diagnostic outcomes before and after a new protocol
- **Behavioral research**: counting whether behavior increased or decreased after intervention
- **Quality improvement**: recording whether defects increased or decreased after a process change""",
)

wilcoxon_signed_rank_test = TestDefinition(
    name="Wilcoxon Signed-Rank Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Dependent",
    distribution="Non-normal",
    explanation="""Wilcoxon Signed-Rank Test This test is used to compare the medians of two related groups to determine if there is a statistically significant difference between them. It assumes that the data is ordinal or continuous but not normally distributed, and that the pairs are dependent.""",
    example="""A researcher wants to test if a new drug reduces pain levels in patients. The researcher measures the pain levels of 30 patients before and after administering the drug. The researcher performs a Wilcoxon signed-rank test to determine if there is a significant difference in median pain levels before and after the treatment.""",
    formula="""
                    $$ W = \\min(W^+, W^-) $$
                    $$ W^+ = \\sum_{i=1}^{n} R_i^+ $$
                    $$ W^- = \\sum_{i=1}^{n} R_i^- $$
                    $$ R_i = \\text{rank}(|X_i - M_0|) $$
                    $$ sgn(X_i - M_0) = \\begin{cases} +1 & \\text{if } X_i > M_0 \\\\ -1 & \\text{if } X_i < M_0 \\\\ 0 & \\text{if } X_i = M_0 \\end{cases} $$
                    Where:
                    - :orange[$W^+$] is the sum of the positive ranks,
                    - :orange[$W^-$] is the absolute sum of the negative ranks.
                    - :orange[$R$] is calculated by ranking the absolute differences between paired observations, excluding ties.
                    - :orange[$X_i$] is the observed value for the :orange[$i$]-th pair, and :orange[$M_0$] is the hypothesized median.
                    - :orange[$sgn(X_i - M_0)$] indicates the direction of the difference (positive, negative, or zero).
                    - :orange[$W$] is the test statistic, which is the smaller of the two sums of ranks (positive and negative).
                    - The test statistic :orange[$W$] is then compared to a critical value from the Wilcoxon signed-rank distribution to determine significance.
                    """,
    core_assumptions="""- **Paired observations** — each subject is measured twice, or subjects are matched pairs
- **Ordinal or continuous data** — the variable is at least ordinal
- **Independence between pairs** — each pair is independent
- **Symmetric differences** — the distribution of difference scores is approximately symmetric
- **No normality required** — non-parametric alternative to the paired t-test""",
    interpretation="""- If **p < α**, reject H₀: the median difference is significantly different from zero
- Report: **W** = test statistic, **z** = z-score (large sample), **p** = p-value, **r** = effect size
- **Effect size r** = |z|/√n: 0.1 = small, 0.3 = medium, 0.5 = large
- Considers both the **direction** (sign) and **magnitude** (rank) of differences
- More **powerful** than the Sign Test when the symmetry assumption holds
- This is the non-parametric equivalent of the paired t-test""",
    realworld_apps="""- **Clinical research**: comparing pain scores before and after treatment in the same patients
- **Psychology**: comparing depression scores pre- and post-therapy (skewed data)
- **Education**: comparing pre-test and post-test scores on an ordinal grading scale
- **Market research**: comparing preference ratings for two products by the same panelists
- **Environmental science**: comparing pollution levels at same sites before and after a policy change""",
)

mann_whitney_u_test = TestDefinition(
    name="Mann-Whitney U Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="Non-normal",
    explanation="""Mann-Whitney U Test This test is used to compare the medians of two independent groups to determine if there is a statistically significant difference between them. It assumes that the data is ordinal or continuous but not normally distributed, and that the groups are independent.""",
    example="""A researcher wants to compare the pain levels between two groups of patients: those who received a new drug and those who received a placebo. The researcher measures the pain levels of 30 patients in each group and performs a Mann-Whitney U test to determine if there is a significant difference in median pain levels between the two groups.""",
    formula="""
                    $$ U = \\sum_{i=1}^{n_1} R_i - \\dfrac{n_1(n_1+1)}{2} $$ 
                    $$ U' = \\sum_{i=1}^{n_2} R_i - \\dfrac{n_2(n_2+1)}{2} $$
                    $$ U = \\min(U, U') $$
                    $$ z = \\dfrac{U - \\mu_U}{\\sigma_U} $$
                    Where: 
                    - :orange[$U$] is the test statistic, 
                    - :orange[$R_i$] is the rank of the :orange[$i$]-th observation in the combined dataset, 
                    - :orange[$n_1$] is the number of observations in group 1, and 
                    - :orange[$n_2$] is the number of observations in group 2. 
                    - The test statistic :orange[$U$] is then compared to a critical value from the Mann-Whitney U distribution to determine significance. :orange[$R$] is calculated by ranking all observations from both groups together and assigning ranks accordingly, with ties receiving average ranks. The U statistic is calculated based on the sum of ranks for one of the groups and adjusted for the number of observations in that group.
                    """,
    core_assumptions="""- **Ordinal or continuous DV** — the outcome variable is at least ordinal
- **Independence** — observations between groups are independent
- **Independent groups** — two unrelated groups being compared
- **Similar distribution shapes** — groups have approximately the same distribution shape
- **No normality required** — non-parametric alternative to the independent t-test""",
    interpretation="""- If **p < α**, reject H₀: one group tends to have larger values than the other (stochastic dominance)
- Report: **U** = Mann-Whitney U statistic, **z** = z-score, **p** = p-value, **r** = effect size
- **Effect size r** = |z|/√n: 0.1 = small, 0.3 = medium, 0.5 = large
- With similar shapes, the test compares **medians**; with different shapes, it compares stochastic dominance
- **Hodges-Lehmann estimator**: median of all pairwise between-group differences""",
    realworld_apps="""- **Clinical research**: comparing pain scores (ordinal) between treatment and placebo
- **Psychology**: comparing well-being ratings between two intervention groups
- **Education**: comparing rankings (ordinal) between two teaching methods
- **Environmental science**: comparing contamination levels (non-normal) across two sites
- **Market research**: comparing Likert-scale satisfaction between two customer segments""",
)

mood_s_median_test = TestDefinition(
    name="Mood's Median Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var="Categorical",
    groups="More than 2",
    relation="Independent",
    distribution="Non-normal",
    explanation="""Mood's Median Test is the simplest nonparametric alternative to one-way ANOVA. It tests whether multiple independent samples come from populations with the same median. Conceptually, it works by: (1) finding the grand median of ALL observations combined, (2) counting values above and below this grand median in each group, and (3) performing a Chi-square test of independence on this contingency table. Less powerful than Kruskal-Wallis, but simpler to understand and very robust.""",
    example="""A psychologist tests whether three different teaching methods produce different median exam scores. Scores from Method A: 72, 85, 78, 90, 65. Method B: 88, 92, 85, 95, 80. Method C: 60, 75, 70, 68, 72. The grand median of all 15 scores is computed, then counts of scores above/below are tabulated for each group. A Chi-square test determines if the pattern differs across groups.""",
    formula="""
                     Step 1: Compute the **grand median** (GM) of all observations combined:
                     $$ \\tilde{X}_{grand} = \\text{median}(X_{11}, X_{12}, \\ldots, X_{kn_k}) $$
                     
                     Step 2: Construct contingency table of counts:
                     
                     | Group | Above GM | Below or Equal |
                     |-------|-----------|----------------|
                     | 1     | :orange[$A_1$] | :orange[$B_1$] |
                     | 2     | :orange[$A_2$] | :orange[$B_2$] |
                     | ...   | ...       | ...            |
                     | k     | :orange[$A_k$] | :orange[$B_k$] |
                     
                     Step 3: Compute **Pearson's Chi-square** on this 2×k table:
                     $$ \\chi^2 = \\sum_{i=1}^{k} \\sum_{j=1}^{2} \\frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$
                     
                     Where :orange[$E_{ij}$] are expected frequencies under independence.
                     
                     $$ df = (2-1)(k-1) = k-1 $$
                     """,
    decision_rules="""
                     - Reject H₀ if :orange[$\\chi^2 > \\chi^2_{\\alpha, k-1}$] or p-value < α
                     - The test is essentially a **Chi-square test of independence** with the median dichotomy
                     - It assumes only independence of observations
                     - No normality assumption required
                     - If the Chi-square approximation is questionable (small expected frequencies), Fisher's exact test can be used for 2×2 tables (two groups)
                     - Ties at the grand median can be handled by: (a) dropping them, (b) counting half in each category, or (c) using the 'below or equal' approach shown
         """,
    core_assumptions="""- **Ordinal or continuous DV** — the outcome variable is at least ordinal
- **Independence** — observations between groups are independent
- **Independent groups** — two or more unrelated groups
- **Does NOT assume equal distribution shapes** — more robust than Kruskal-Wallis
- **No normality required**""",
    interpretation="""- If **p < α**, reject H₀: at least one group has a different median from the others
- Report: **χ²** = chi-square statistic, **df** = k−1, **p** = p-value
- Counts observations above and below the **overall median** in a 2×k contingency table
- **Effect size**: proportion of observations above the overall median in each group
- Less **powerful** than Kruskal-Wallis when shapes are similar, but **more robust** when they differ""",
    realworld_apps="""- **Clinical research**: comparing median biomarker levels across groups with different variances
- **Environmental science**: comparing contamination where sites show very different variability
- **Psychology**: comparing well-being scores when groups differ in both central tendency and spread
- **Genomics**: comparing gene expression medians across multiple experimental conditions
- **Engineering**: comparing product durability (different failure patterns) across batches""",
)

kruskal_wallis_test = TestDefinition(
    name="Kruskal-Wallis Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="More than 2",
    relation="Independent",
    distribution="Non-normal",
    explanation="""Kruskal-Wallis Test This test is used to compare the medians of more than two independent groups to determine if there is a statistically significant difference between them. It assumes that the data is ordinal or continuous but not normally distributed, and that the groups are independent.""",
    example="""A researcher wants to compare the pain levels between three groups of patients: those who received a new drug, those who received a different drug, and those who received a placebo. The researcher measures the pain levels of 30 patients in each group and performs a Kruskal-Wallis test to determine if there is a significant difference in median pain levels between the three groups.""",
    formula="""
                    $$ H = \\dfrac{12}{N(N+1)} \\sum_{i=1}^{k} \\dfrac{R_i^2}{n_i} - 3(N+1) $$ 
                    Where: 
                    - :orange[$H$] is the test statistic, 
                    - :orange[$R_i$] is the sum of ranks for group :orange[$i$], 
                    - :orange[$n_i$] is the number of observations in group :orange[$i$], 
                    - :orange[$N$] is the total number of observations, and 
                    - :orange[$k$] is the number of groups. 
                    - The test statistic :orange[$H$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
    core_assumptions="""- **Ordinal or continuous DV** — the outcome variable is at least ordinal
- **Independence** — observations between groups are independent
- **Independent groups** — two or more unrelated groups
- **Similar distribution shapes** — groups have approximately the same shape
- **No normality required** — non-parametric alternative to one-way ANOVA""",
    interpretation="""- If **p < α**, reject H₀: at least one group tends to have larger values than others
- Report: **H** = Kruskal-Wallis H statistic, **df** = k−1, **p** = p-value
- **Dunn's post-hoc test** (with Bonferroni correction) identifies which specific groups differ
- **Effect size**: ε² = H/(n²−1)/(n+1) or the difference in mean ranks
- With similar distribution shapes, the test compares **medians**
- This is the non-parametric extension of Mann-Whitney for 3+ groups""",
    post_hoc="Dunn, Conover, DSCF (Dwass-Steel-Critchlow-Fligner)",
    realworld_apps="""- **Clinical research**: comparing pain scores across three or more treatment groups
- **Education**: comparing exam rankings across multiple teaching methods
- **Ecology**: comparing species diversity across different habitat types (non-normal data)
- **Social science**: comparing Likert-scale satisfaction across multiple demographic groups
- **Market research**: comparing product preference rankings across consumer segments""",
)

friedman_test = TestDefinition(
    name="Friedman Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="More than 2",
    relation="Dependent",
    distribution="Non-normal",
    explanation="""Friedman Test This test is used to compare the medians of three or more related groups to determine if there is a statistically significant difference between them. It assumes that the data is ordinal or continuous but not normally distributed, and that the groups are dependent.""",
    example="""A researcher wants to compare the effectiveness of three different teaching methods on student performance. The researcher measures the performance of 30 students using each teaching method and performs a Friedman Test to determine if there is a significant difference in median performance across the three methods.""",
    formula="""
                    $$ \\chi^2_F = \\dfrac{12}{N k (k+1)} \\sum_{j=1}^{k} R_j^2 - 3N(k+1) $$ 
                    Where: 
                    - :orange[$\\chi^2_F$] is the test statistic, 
                    - :orange[$N$] is the number of subjects (blocks), 
                    - :orange[$k$] is the number of treatments (groups), and 
                    - :orange[$R_j$] is the sum of ranks for the :orange[$j$]-th treatment. 
                    - The test statistic :orange[$\\chi^2_F$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
    core_assumptions="""- **Ordinal or continuous DV** — the outcome variable is at least ordinal
- **Dependent groups** — same subjects measured under 3+ conditions, or matched subjects
- **Independence between subjects** — subjects (or blocks) are independent
- **No normality required** — non-parametric alternative to repeated measures ANOVA
- **Rankings must be meaningful** within each block""",
    interpretation="""- If **p < α**, reject H₀: at least one condition differs significantly from the others
- Report: **χ²_r** = Friedman chi-square statistic, **df** = k−1, **p** = p-value
- **Post-hoc**: Wilcoxon Signed-Rank Test with Bonferroni correction or Conover's test
- **Effect size**: Kendall's W = χ²_r / (n(k−1)) — 0 = no agreement, 1 = perfect agreement
- **Kendall's W**: 0.1 = small, 0.3 = medium, 0.5 = large effect""",
    post_hoc="Nemenyi, Conover-Friedman, Wilcoxon + Bonferroni",
    realworld_apps="""- **Clinical trials**: measuring patient outcomes at baseline, 1 month, and 3 months (ordinal scale)
- **Psychology**: comparing mood ratings across multiple time points in the same subjects
- **Consumer testing**: ranking multiple products (e.g., coffee blends) by the same panelists
- **Sensor technology**: comparing readings from multiple sensors at the same locations
- **Rehabilitation**: measuring functional ability at pre-, mid-, and post-treatment""",
)

permutation_manova_or_non_parametric_manova = TestDefinition(
    name="Permutation MANOVA or Non-Parametric MANOVA",
    objective="Comparison",
    dependent_var="Multiple Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="More than 2",
    relation="Independent",
    distribution="Non-normal",
    explanation="""Permutation MANOVA or Non-Parametric MANOVA These tests are used to compare multivariate distributions across two or more independent groups when the assumptions of traditional MANOVA are not met. They do not assume a specific distribution for the data.""",
    example="""A researcher wants to compare the effects of three different diets on both weight loss and cholesterol levels, but the data does not follow a normal distribution. The researcher performs a Permutation MANOVA to determine if there are significant differences in the combined dependent variables (weight loss and cholesterol levels) across the three diet groups.""",
    formula="""
                    $$ F = \\dfrac{MS_{between}}{MS_{error}} $$
                    $$ MS_{between} = \\dfrac{SS_{between}}{df_{between}} $$
                    $$ MS_{error} = \\dfrac{SS_{error}}{df_{error}} $$
                    $$ SS_{between} = \\sum_{i=1}^{k} n_i (\\bar{X}_i - \\bar{X})^2 $$
                    $$ SS_{error} = \\sum_{i=1}^{k} \\sum_{j=1}^{n_i} (X_{ij} - \\bar{X}_i)^2 $$
                    $$ df_{between} = k - 1 $$
                    $$ df_{error} = N - k $$
                    Where:
                    - :orange[$F$] is the pseudo-F-statistic,
                    - :orange[$MS_{between}$] is the mean square between groups, and
                    - :orange[$MS_{error}$] is the mean square error.
                    - The test statistic :orange[$F$] is then evaluated using a permutation-based null distribution (data is randomly reshuffled many times) to compute the p-value, rather than comparing to a theoretical F-distribution.
                    """,
    core_assumptions="""- **Multiple continuous or ordinal DVs** — two or more correlated outcome variables
- **Independence** — observations are independent
- **No normality required** — permutation tests are distribution-free
- **Exchangeability under H₀** — observations can be randomly permuted between groups
- **Similar dispersion** — groups should have similar multivariate dispersion""",
    interpretation="""- If **p < α** (from permutation distribution), reject H₀: groups differ in multivariate location or dispersion
- Report: **Pseudo-F** (or similar statistic), **p** = permutation p-value, **R²** = variance explained
- The p-value is the proportion of permuted test statistics ≥ the observed statistic
- **Advantage**: no distributional assumptions — valid for any underlying distribution
- **Post-hoc**: pairwise PERMANOVAs with Bonferroni correction""",
    post_hoc="Dunn, Conover, DSCF (Dwass-Steel-Critchlow-Fligner)",
    realworld_apps="""- **Ecology**: comparing community composition across habitat types (multiple species abundances)
- **Genomics**: comparing gene expression profiles across experimental conditions
- **Microbiome research**: comparing microbial community structure across treatment groups
- **Environmental science**: comparing multivariate pollution profiles across regions
- **Psychology**: comparing multivariate personality profiles across demographic groups""",
)

chi_square_test = TestDefinition(
    name="Chi-Square Test",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups=["any", "2", "More than 2"],
    relation="Independent",
    distribution=["Non-normal", "Normal", "any"],
    explanation="""Chi-Square Test This test is used to determine if there is a significant association between two categorical variables. It compares the observed frequencies with the expected frequencies under the null hypothesis.""",
    example="""A researcher wants to test if there is a significant association between gender and smoking status. The researcher collects data on gender and smoking status from 200 participants and performs a Chi-Square Test to determine if there is a significant relationship between these two variables.""",
    formula="""
                    $$ \\chi^2 = \\sum_{i=1}^{r} \\sum_{j=1}^{c} \\dfrac{(O_{ij} - E_{ij})^2}{E_{ij}} $$
                    Where:
                    - :orange[$\\chi^2$] is the test statistic,
                    - :orange[$O_{ij}$] is the observed frequency for cell :orange[$(i,j)$],
                    - :orange[$E_{ij}$] is the expected frequency for cell :orange[$(i,j)$],
                    - :orange[$r$] is the number of rows, and
                    - :orange[$c$] is the number of columns.
                    - The test statistic :orange[$\\chi^2$] is then compared to a critical value from the chi-square distribution with :orange[$(r-1)(c-1)$] degrees of freedom to determine significance.
                    """,
    core_assumptions="""- **Categorical data** — both variables are nominal (categorical)
- **Independence** — observations are independently classified
- **Mutually exclusive categories** — each observation falls into exactly one cell
- **Expected frequencies ≥ 5** — no more than 20% of expected cell counts should be < 5
- **Random sampling** — the sample is representative""",
    interpretation="""- If **p < α**, reject H₀: the two categorical variables are **associated** (not independent)
- Report: **χ²** = chi-square statistic, **df** = (r−1)(c−1), **p** = p-value
- **Effect size (Cramér's V)** = √(χ²/(n·min(r−1,c−1))): 0.1 = small, 0.3 = medium, 0.5 = large
- **Standardized residuals** identify which cells contribute most to the association
- The test indicates **whether** association exists, not its strength or direction""",
    realworld_apps="""- **Medical research**: testing if smoking is associated with lung cancer
- **Market research**: testing if brand preference is associated with age group
- **Political science**: testing if voting preference is associated with education level
- **Genetics**: testing if genotype frequencies are associated with disease status
- **Social science**: testing if marital status is associated with employment category""",
)

mcnemar_s_test = TestDefinition(
    name="McNemar's Test",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Dependent",
    distribution=["Non-normal", "Normal", "any"],
    explanation="""McNemar's Test This test is used to determine if there is a significant change in proportions for paired nominal data. It is typically used when analyzing before-and-after data or matched pairs.""",
    example="""A researcher wants to test if there is a significant change in smoking status before and after a intervention. The researcher collects data on smoking status from 100 participants before and after the intervention and performs a McNemar's Test to determine if there is a significant change.""",
    formula="""
                    $$ \\chi^2 = \\dfrac{(b - c)^2}{b + c} $$
                    Where:
                    - :orange[$\\chi^2$] is the test statistic,
                    - :orange[$b$] is the number of pairs where the first condition is positive and the second is negative, and
                    - :orange[$c$] is the number of pairs where the first condition is negative and the second is positive.
                    - The test statistic :orange[$\\chi^2$] is then compared to a critical value from the chi-square distribution with 1 degree of freedom to determine significance.
                    """,
    core_assumptions="""- **Paired binary data** — each subject has two binary measurements (before/after) or matched pairs
- **Dependent observations** — the two measurements are paired within subjects
- **Discordant pairs** — the test only uses pairs where the two measurements differ
- **Independence between pairs** — subjects (pairs) are independent
- **No minimum expected frequency** — exact binomial version is valid for any sample size""",
    interpretation="""- If **p < α**, reject H₀: the proportion of 'yes' responses is the same before and after
- Report: **χ²** = McNemar's chi-square (or exact binomial p-value), **df** = 1, **p** = p-value
- Based on **discordant pairs** only — pairs where before and after responses differ
- **Effect size**: odds ratio = b/c where b = yes→no pairs and c = no→yes pairs
- For small samples (< 25 discordant pairs), use the **exact binomial test**""",
    realworld_apps="""- **Clinical trials**: measuring if diagnosis (positive/negative) changes before and after treatment
- **Epidemiology**: testing if exposure status changes from one survey wave to the next
- **Education**: testing pass/fail rates before and after a training program
- **Marketing**: testing consumer brand preference change after seeing an advertisement
- **Quality control**: testing if defect rates change after implementing a process improvement""",
)

cochran_s_q_test = TestDefinition(
    name="Cochran's Q Test",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="More than 2",
    relation="Dependent",
    distribution=["Non-normal", "Normal", "any"],
    explanation="""Cochran's Q Test This test is used to determine if there is a significant difference in proportions for three or more related groups. It is an extension of McNemar's Test.""",
    example="""A researcher wants to test if there is a significant difference in the proportion of participants who smoke at three different time points (before, during, and after an intervention). The researcher performs a Cochran's Q Test to determine if there is a significant difference.""",
    formula="""
                    $$ Q = \\dfrac{(k-1) \\left( k \\sum_{j=1}^{k} G_j^2 - \\left( \\sum_{j=1}^{k} G_j \\right)^2 \\right)}{k \\sum_{i=1}^{b} L_i - \\sum_{i=1}^{b} L_i^2} $$
                    Where:
                    - :orange[$Q$] is the test statistic,
                    - :orange[$k$] is the number of conditions/treatments,
                    - :orange[$b$] is the number of subjects,
                    - :orange[$G_j$] is the column total (successes) for condition :orange[$j$], and
                    - :orange[$L_i$] is the row total (successes) for subject :orange[$i$].
                    - The test statistic :orange[$Q$] is then compared to a critical value from the chi-square distribution with :orange[$k-1$] degrees of freedom to determine significance.
                    """,
    core_assumptions="""- **Binary DV** — the outcome has exactly two categories (yes/no, success/failure)
- **Dependent observations** — same subjects (or matched) are measured under 3+ conditions
- **Independence between subjects** — subjects are independent
- **No normality required** — extension of McNemar's test to 3+ conditions
- **Complete blocks** — all subjects are measured under ALL conditions""",
    interpretation="""- If **p < α**, reject H₀: the proportion of 'successes' differs across at least one pair of conditions
- Report: **Q** = Cochran's Q statistic, **df** = k−1, **p** = p-value
- **Post-hoc**: pairwise McNemar's tests with Bonferroni correction
- **Effect size**: report the proportion of successes under each condition
- This is the **binary-data equivalent** of the Friedman Test""",
    realworld_apps="""- **Clinical trials**: measuring if patients test positive for an allergen across multiple time points
- **Psychology**: recording whether subjects correctly identify a stimulus under 3+ conditions
- **Market research**: testing if consumers select a product across multiple product variants
- **Medical education**: tracking if trainees correctly diagnose across different case types
- **Usability testing**: recording successful task completion across multiple interface designs""",
)

fisher_s_exact_test = TestDefinition(
    name="Fisher's Exact Test",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution=["Non-normal", "Normal", "any"],
    explanation="""Fisher's Exact Test This test is used to determine if there is a significant association between two categorical variables when sample sizes are small.""",
    example="""A researcher wants to test if there is a significant association between gender and smoking status in a small sample of 20 participants. The researcher performs a Fisher's Exact Test to determine if there is a significant relationship between these two variables.""",
    formula="""
                    $$ p = \\dfrac{(a+b)!\\,(c+d)!\\,(a+c)!\\,(b+d)!}{a!\\,b!\\,c!\\,d!\\,n!} $$
                    
                    Where:
                    - :orange[$a, b, c, d$] are the cell frequencies in the 2×2 contingency table,
                    - :orange[$n = a+b+c+d$] is the total sample size.
                    - The p-value is calculated by summing the probabilities of all tables with the same marginal totals that are as extreme or more extreme than the observed table.
                    """,
    core_assumptions="""- **Categorical data** — two binary or nominal variables
- **Independence** — observations are independent
- **Fixed row and column totals** — the margins are fixed (or conditioned upon)
- **No minimum expected frequency** — valid for any sample size, especially small samples
- **Mutually exclusive categories** — each observation falls into exactly one cell""",
    interpretation="""- If **p < α** (exact p-value), reject H₀: the two categorical variables are associated
- Report: **p** = exact p-value, **odds ratio** (for 2×2 tables), **95% CI** for odds ratio
- The p-value enumerates **all possible tables** with the same margins, summing extreme probabilities
- **Mid-p adjustment**: a less conservative variant averaging the observed table probability
- For larger tables (r×c), exact computation can be intensive — Monte Carlo approximation may be used""",
    realworld_apps="""- **Clinical trials**: analyzing treatment response in small trials (e.g., 5 patients per group)
- **Genetics**: testing if a rare genetic variant is associated with disease status
- **Epidemiology**: analyzing outbreak data with small numbers of cases
- **Pharmaceutical safety**: testing if adverse events are associated with treatment
- **Educational research**: analyzing categorical outcomes in small classroom studies""",
)

pearson_correlation = TestDefinition(
    name="Pearson Correlation",
    objective="Association/Correlation",
    dependent_var="Continuous",
    independent_var="Continuous",
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution="Normal",
    explanation="""Pearson Correlation This test is used to measure the strength and direction of the linear relationship between two continuous variables. It assumes that the data is continuous, follows a normal distribution, and that the relationship between the variables is linear.""",
    example="""A researcher wants to test if there is a significant correlation between hours of study and exam scores among students. The researcher collects data on hours of study and exam scores from 100 students and performs a Pearson Correlation to determine if there is a significant relationship between these two variables.""",
    formula="""
                    $$ r = \\dfrac{\\sum_{i=1}^{n} (X_i - \\bar{X})(Y_i - \\bar{Y})}{\\sqrt{\\sum_{i=1}^{n} (X_i - \\bar{X})^2} \\sqrt{\\sum_{i=1}^{n} (Y_i - \\bar{Y})^2}} $$ 
                    Where: 
                    - :orange[$r$] is the Pearson correlation coefficient, 
                    - :orange[$X_i$] and :orange[$Y_i$] are the individual data points for variables X and Y, 
                    - :orange[$\\bar{X}$] and :orange[$\\bar{Y}$] are the means of variables X and Y, and 
                    - :orange[$n$] is the number of data points. 
                    - The test statistic :orange[$r$] is then compared to a critical value from the Pearson correlation distribution to determine significance.
                    """,
    core_assumptions="""- **Continuous variables** — both X and Y are measured on continuous scales
- **Linear relationship** — the relationship between X and Y is approximately linear
- **Independence** — observations are independent of each other
- **Normal distribution** — both variables are approximately normally distributed
- **Homoscedasticity** — the variability of Y is roughly constant across levels of X
- **No significant outliers** — extreme values can greatly influence r""",
    interpretation="""- Report: **r** = Pearson correlation coefficient, **p** = p-value (test of r ≠ 0)
- **r ranges** from −1 to +1: −1 = perfect negative, 0 = none, +1 = perfect positive linear
- **r²** (coefficient of determination): proportion of variance in Y explained by X
- **Interpretation of |r|**: 0.1 = small, 0.3 = medium, 0.5 = large effect
- **Always visualize** with a scatterplot — r is misleading for non-linear relationships""",
    realworld_apps="""- **Medical research**: correlating BMI with blood pressure across a population
- **Psychology**: correlating years of education with annual income
- **Environmental science**: correlating temperature with ice melt rate
- **Finance**: correlating stock returns between two companies
- **Education**: correlating hours studied with exam scores""",
)

spearman_rank_correlation = TestDefinition(
    name="Spearman Rank Correlation",
    objective="Association/Correlation",
    dependent_var=["Ordinal", "Continuous"],
    independent_var=["Ordinal", "Continuous"],
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution="Non-normal",
    explanation="""Spearman Rank Correlation This test is used to measure the strength and direction of the monotonic relationship between two ordinal or continuous variables. It does not assume a linear relationship or a normal distribution.""",
    example="""A researcher wants to test if there is a significant correlation between rankings of students in two different subjects. The researcher collects data on the rankings and performs a Spearman Rank Correlation to determine if there is a significant relationship between these two variables.""",
    formula="""
                    $$ r_s = 1 - \\dfrac{6 \\sum d_i^2}{n(n^2 - 1)} $$ 
                    Where: 
                    - :orange[$r_s$] is the Spearman rank correlation coefficient, 
                    - :orange[$d_i$] is the difference in ranks for each pair of observations, 
                    - :orange[$n$] is the number of observations, and 
                    - the sum is over all pairs. 
                    - The test statistic :orange[$r_s$] is then compared to a critical value from the Spearman rank correlation distribution to determine significance.
                    """,
    core_assumptions="""- **Ordinal or continuous variables** — both X and Y are at least ordinal
- **Monotonic relationship** — as X increases, Y consistently increases or decreases (not necessarily linear)
- **Independence** — observations are independent
- **No normality required** — non-parametric alternative to Pearson
- **Data can be meaningfully ranked**""",
    interpretation="""- Report: **ρ** (rho) or **r_s** = Spearman rank correlation, **p** = p-value
- **r_s ranges** from −1 to +1, interpreted similarly to Pearson's r but for **monotonic** relationships
- **Interpretation of |r_s|**: 0.1 = small, 0.3 = medium, 0.5 = large
- Uses the **ranks** of data rather than raw values — robust to outliers
- Appropriate when linearity or normality assumptions are violated for Pearson""",
    realworld_apps="""- **Psychology**: correlating well-being rank with income rank (ordinal data)
- **Education**: correlating class rankings before and after an intervention
- **Environmental science**: correlating pollution with distance (non-linear relationship)
- **Market research**: correlating satisfaction rankings with likelihood-to-recommend
- **Sports analytics**: correlating draft position with career performance rankings""",
)

chi_square_test_of_independence = TestDefinition(
    name="Chi-Square Test of Independence",
    objective="Association/Correlation",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["Non-normal", "Normal", "any"],
    explanation="""Chi-Square Test of Independence This test is used to determine if there is a significant association between two categorical variables. It compares the observed frequencies with the expected frequencies under the null hypothesis.""",
    example="""A researcher wants to test if there is a significant association between gender and smoking status. The researcher collects data on gender and smoking status from 200 participants and performs a Chi-Square Test of Independence to determine if there is a significant relationship between these two variables.""",
    formula="""
                    $$ \\chi^2 = \\sum_{i=1}^{r} \\sum_{j=1}^{c} \\dfrac{(O_{ij} - E_{ij})^2}{E_{ij}} $$ 
                    Where: 
                    - :orange[$\\chi^2$] is the test statistic, 
                    - :orange[$O_{ij}$] is the observed frequency for cell :orange[$(i,j)$], and 
                    - :orange[$E_{ij}$] is the expected frequency for cell :orange[$(i,j)$]. 
                    - The test statistic :orange[$\\chi^2$] is then compared to a critical value from the chi-square distribution with :orange[$(r-1)(c-1)$] degrees of freedom to determine significance.
                    """,
    core_assumptions="""- **Categorical variables** — both variables are nominal (categorical)
- **Independence** — observations are independently classified
- **Mutually exclusive categories** — each observation falls into exactly one cell
- **Expected frequencies ≥ 5** — no more than 20% of expected counts should be < 5
- **Random sampling** — the sample is representative""",
    interpretation="""- If **p < α**, reject H₀: the two categorical variables are **associated** (not independent)
- Report: **χ²** = chi-square statistic, **df** = (r−1)(c−1), **p** = p-value
- **Effect size**: Cramér's V (for r×c) or φ (phi, for 2×2 tables)
- **Standardized residuals** identify which observed frequencies deviate most from expected
- Mathematically identical to the Chi-Square Test but framed as a test of **independence**""",
    realworld_apps="""- **Medical research**: testing if blood type is independent of disease status
- **Market research**: testing if product preference is independent of age group
- **Political science**: testing if voting preference is independent of geographic region
- **Sociology**: testing if religious affiliation is independent of marital status
- **Genetics**: testing if genotype distribution is independent of environmental exposure""",
)

point_biserial_correlation = TestDefinition(
    name="Point-Biserial Correlation",
    objective="Association/Correlation",
    dependent_var="Continuous",
    independent_var="Binary/Dichotomous",
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["Non-normal", "Normal", "any"],
    explanation="""Point-Biserial Correlation This test is used to measure the strength and direction of the linear relationship between a continuous variable and a binary variable. It is a special case of the Pearson correlation coefficient.""",
    example="""A researcher wants to test if there is a significant correlation between test scores (continuous) and gender (binary). The researcher collects data on test scores and gender from 100 participants and performs a Point-Biserial Correlation to determine if there is a significant relationship between these two variables.""",
    formula="""
                    $$ r_{pb} = \\dfrac{M_1 - M_0}{s} \\sqrt{\\dfrac{n_1 n_0}{n(n-1)}} $$ 
                    Where: 
                    - :orange[$r_{pb}$] is the Point-Biserial correlation coefficient, 
                    - :orange[$M_1$] and :orange[$M_0$] are the means of the continuous variable for the two groups, 
                    - :orange[$s$] is the standard deviation of the continuous variable, 
                    - :orange[$n_1$] and :orange[$n_0$] are the sample sizes of the two groups, and 
                    - :orange[$n$] is the total sample size. 
                    - The test statistic :orange[$r_{pb}$] is then compared to a critical value from the Pearson correlation distribution to determine significance.
                    """,
    core_assumptions="""- **Binary variable** — one variable is truly dichotomous (e.g., male/female, treatment/control)
- **Continuous variable** — the other variable is measured on a continuous scale
- **Linear relationship** — the continuous variable has roughly equal variance in both groups
- **Normal distribution** — the continuous variable is approximately normal in each group
- **Independence** — observations are independent""",
    interpretation="""- Report: **r_pb** = point-biserial correlation, **p** = p-value
- **r_pb ranges** from −1 to +1, interpreted similarly to Pearson's r
- **r_pb²** = proportion of variance in the continuous variable explained by the binary grouping
- **Related to t-test**: significant t-test → significant point-biserial correlation
- **Effect size**: r_pb = 0.1 = small, 0.3 = medium, 0.5 = large""",
    realworld_apps="""- **Test development**: correlating item responses (correct/incorrect) with total test scores
- **Clinical research**: correlating treatment group with continuous outcomes
- **Psychology**: correlating sex (male/female) with continuous personality trait scores
- **Education**: correlating pass/fail status with hours studied
- **HR analytics**: correlating hiring decision with interview scores""",
)

kendall_s_tau_b = TestDefinition(
    name="Kendall's Tau-b",
    objective="Association/Correlation",
    dependent_var=["Ordinal", "Continuous"],
    independent_var=["Ordinal", "Continuous"],
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution="Non-normal",
    explanation="""Kendall's Tau-b is a non-parametric rank correlation coefficient that measures the strength and direction of monotonic association between two variables. It is more robust than Spearman's ρ when there are many tied ranks and provides a more conservative estimate. Its interpretation is more intuitive: it represents the difference between the probability of concordance and discordance.""",
    example="""A researcher wants to assess the association between two ordinal ratings of periodontal disease severity (none, mild, moderate, severe) given by two different examiners on 100 patients. Kendall's Tau-b is preferred over Spearman's ρ due to the high number of expected ties.""",
    formula="""
                    $$ \\tau_b = \\frac{C - D}{\\sqrt{(C + D + T_X)(C + D + T_Y)}} $$
                    Where:
                    - :orange[$C$] is the number of concordant pairs,
                    - :orange[$D$] is the number of discordant pairs,
                    - :orange[$T_X$] is the number of pairs tied only on variable :orange[$X$],
                    - :orange[$T_Y$] is the number of pairs tied only on variable :orange[$Y$],
                    - :orange[$\\tau_b$] ranges from −1 (perfect disagreement) to +1 (perfect agreement).
                    """,
    core_assumptions="""- **Ordinal or continuous variables** — both X and Y are at least ordinal
- **Monotonic relationship** — the association is monotonic (not necessarily linear)
- **Independence** — observations are independent
- **No normality required**
- **Handles ties well** — key advantage over Spearman with many tied ranks""",
    interpretation="""- Report: **τ_b** = Kendall's tau-b coefficient, **p** = p-value
- **τ_b ranges** from −1 to +1, measuring **concordance** between rankings
- **Interpretation of |τ_b|**: 0.1 = small, 0.3 = medium, 0.5 = large
- Includes a **correction for ties**, more reliable than Spearman when many observations share values
- The difference from Spearman is usually small; τ_b gives slightly lower absolute values""",
    realworld_apps="""- **Survey research**: correlating Likert-scale responses with many tied ratings
- **Education**: correlating letter grades (A, B, C, D, F) across two subjects
- **Medical research**: correlating ordinal disease severity ratings from two clinicians
- **Social science**: correlating social class with educational attainment categories
- **Sensory science**: correlating taste test rankings from multiple judges with ties""",
)

simple_linear_regression = TestDefinition(
    name="Simple Linear Regression",
    objective="Prediction",
    dependent_var="Continuous",
    independent_var="Continuous",
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["Normal", "Non-normal", "any"],
    explanation="""Simple Linear Regression This test is used to model the relationship between a continuous dependent variable and a single continuous independent variable. It assumes that the dependent variable is continuous, that the residuals are normally distributed, and that the relationship between the variables is linear.""",
    example="""A researcher wants to predict exam scores based on hours of study. The researcher collects data on hours of study and exam scores from 100 students and performs a Simple Linear Regression to determine if hours of study is a significant predictor of exam scores.""",
    formula="""
                    $$ Y = \\beta_0 + \\beta_1 X + \\epsilon $$ 
                    Where: 
                    - :orange[$Y$] is the dependent variable, 
                    - :orange[$X$] is the independent variable, 
                    - :orange[$\\beta_0$] is the intercept, 
                    - :orange[$\\beta_1$] is the slope coefficient, and 
                    - :orange[$\\epsilon$] is the error term. 
                    - The coefficients :orange[$\\beta_0$] and :orange[$\\beta_1$] are estimated using the least squares method, and the significance of the predictor is determined by testing if :orange[$\\beta_1$] is significantly different from zero.
                    """,
    core_assumptions="""- **Continuous DV** — the outcome variable (Y) is continuous
- **Continuous or binary IV** — the predictor (X) can be continuous or binary
- **Linear relationship** — the relationship between X and Y is approximately linear
- **Independence** — observations are independent
- **Homoscedasticity** — constant variance of residuals across X levels
- **Normal residuals** — residuals are approximately normally distributed
- **No significant outliers** — extreme values can disproportionately influence the line""",
    interpretation="""- The **regression coefficient (β₁)** = change in Y for a one-unit change in X
- **R²** = proportion of variance in Y explained by X
- Report: **β₁** = slope, **SE**, **t** = t-statistic, **p** = p-value, **R²**, **F** = overall F-statistic
- Always check **residuals vs. fitted plot** for homoscedasticity and **Q-Q plot** for normality
- **Standardized coefficient (beta)** allows comparison when predictors have different scales""",
    realworld_apps="""- **Medical research**: predicting blood pressure from age
- **Economics**: predicting consumer spending from disposable income
- **Environmental science**: predicting plant growth from rainfall
- **Education**: predicting exam scores from hours studied
- **Marketing**: predicting sales from advertising spend""",
)

multiple_linear_regression = TestDefinition(
    name="Multiple Linear Regression",
    objective="Prediction",
    dependent_var="Continuous",
    independent_var="Multiple Continuous",
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["Normal", "Non-normal", "any"],
    explanation="""Multiple Linear Regression This test is used to model the relationship between a continuous dependent variable and multiple continuous independent variables. It assumes that the dependent variable is continuous, that the residuals are normally distributed, and that the relationship between the variables is linear.""",
    example="""A researcher wants to predict exam scores based on hours of study and attendance. The researcher collects data on hours of study, attendance, and exam scores from 100 students and performs a Multiple Linear Regression to determine if hours of study and attendance are significant predictors of exam scores.""",
    formula="""
                    $$ Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\ldots + \\beta_k X_k + \\epsilon $$ 
                    Where: 
                    - :orange[$Y$] is the dependent variable, 
                    - :orange[$X_1, X_2, \\ldots, X_k$] are the independent variables, 
                    - :orange[$\\beta_0$] is the intercept, 
                    - :orange[$\\beta_1, \\beta_2, \\ldots, \\beta_k$] are the slope coefficients, and 
                    - :orange[$\\epsilon$] is the error term. 
                    - The coefficients :orange[$\\beta_0, \\beta_1, \\beta_2, \\ldots, \\beta_k$] are estimated using the least squares method, and the significance of each predictor is determined by testing if its corresponding coefficient is significantly different from zero.
                    """,
    core_assumptions="""- **Continuous DV** — the outcome variable (Y) is continuous
- **Two+ IVs** — predictors are continuous, categorical (dummy-coded), or a mix
- **Linear relationship** — each predictor has a linear relationship with Y (accounting for others)
- **Independence** — observations are independent
- **Homoscedasticity** — constant variance of residuals
- **Normal residuals** — residuals are approximately normally distributed
- **No perfect multicollinearity** — predictors should not be too highly correlated (VIF < 5−10)
- **No significant outliers or influential points**""",
    interpretation="""- Report: **R²** = variance explained, **F** = overall F-test, **p** = p-value
- Each predictor has: **β** (coefficient), **SE**, **t**, **p-value**, **95% CI**
- **Standardized coefficients (beta)** allow comparison of predictor importance
- **VIF** (Variance Inflation Factor) diagnoses multicollinearity — VIF > 5−10 is serious
- **Adjusted R²** penalizes for adding irrelevant predictors""",
    realworld_apps="""- **Clinical research**: predicting patient outcomes from age, baseline health, and treatment
- **Economics**: predicting housing prices from sq footage, bedrooms, location, age
- **Psychology**: predicting well-being from income, social support, and physical activity
- **Marketing**: predicting sales from price, advertising spend, and competitor activity
- **Environmental science**: predicting air quality from traffic, temperature, and wind""",
)

logistic_regression = TestDefinition(
    name="Logistic Regression",
    objective="Prediction",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Continuous", "Multiple Continuous", "Categorical"],
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["Normal", "Non-normal", "any"],
    explanation="""Logistic Regression This test is used to model the relationship between a binary dependent variable and one or more continuous independent variables. It assumes that the dependent variable is binary, and that the relationship between the independent variables and the log-odds of the dependent variable is linear.""",
    example="""A researcher wants to predict the likelihood of a student passing an exam based on hours of study. The researcher collects data on hours of study and pass/fail status from 100 students and performs a Logistic Regression to determine if hours of study is a significant predictor of passing the exam.""",
    formula="""
                    $$ \\log\\left(\\dfrac{p}{1-p}\\right) = \\beta_0 + \\beta_1 X $$ 
                    Where: 
                    - :orange[$p$] is the probability of the dependent variable being 1 (e.g., passing the exam), 
                    - :orange[$X$] is the independent variable, 
                    - :orange[$\\beta_0$] is the intercept, and 
                    - :orange[$\\beta_1$] is the slope coefficient. 
                    - The coefficients :orange[$\\beta_0$] and :orange[$\\beta_1$] are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if :orange[$\\beta_1$] is significantly different from zero.
                    """,
    core_assumptions="""- **Binary DV** — the outcome has two categories (0/1, yes/no, success/failure)
- **Independence** — observations are independent
- **Logit linearity** — the log-odds of the outcome have a linear relationship with continuous predictors
- **No multicollinearity** — predictors should not be too highly correlated
- **Large sample** — rule of thumb: at least 10 events per predictor variable (EPV)
- **No extreme outliers** — influential points can distort the model""",
    interpretation="""- Report: **odds ratios (OR)** = exp(β) with **95% CI**, **p-values**, **model fit statistics**
- **OR > 1**: higher X → higher odds of the outcome; **OR < 1**: higher X → lower odds
- **Pseudo-R²** (McFadden's, Nagelkerke's) measures model fit (lower than OLS R² is expected)
- **Hosmer-Lemeshow test** assesses calibration (non-significant = good fit)
- **Classification accuracy** (confusion matrix, AUC-ROC) assesses predictive performance""",
    realworld_apps="""- **Clinical research**: predicting disease development (yes/no) from risk factors
- **Credit scoring**: predicting whether a customer defaults on a loan
- **Epidemiology**: predicting disease presence from exposure and demographics
- **Marketing**: predicting whether a customer makes a purchase
- **HR analytics**: predicting employee turnover from job satisfaction and salary""",
)

multinomial_logistic_regression = TestDefinition(
    name="Multinomial Logistic Regression",
    objective="Prediction",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Continuous", "Multiple Continuous", "Categorical"],
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["Normal", "Non-normal", "any"],
    explanation="""Multinomial Logistic Regression This test is used to model the relationship between a categorical dependent variable with more than two categories and one or more continuous independent variables. It assumes that the dependent variable is categorical, and that the relationship between the independent variables and the log-odds of each category of the dependent variable is linear.""",
    example="""A researcher wants to predict the choice of transportation (car, bus, bike) based on hours of commute. The researcher collects data on hours of commute and transportation choice from 100 participants and performs a Multinomial Logistic Regression to determine if hours of commute is a significant predictor of transportation choice.""",
    formula="""
                    $$ \\log\\left(\\dfrac{p_j}{p_k}\\right) = \\beta_{0j} + \\beta_{1j} X $$ 
                    Where: 
                    - :orange[$p_j$] is the probability of the dependent variable being in category :orange[$j$], 
                    - :orange[$p_k$] is the probability of the dependent variable being in the reference category :orange[$k$], 
                    - :orange[$X$] is the independent variable, 
                    - :orange[$\\beta_{0j}$] is the intercept for category :orange[$j$], and 
                    - :orange[$\\beta_{1j}$] is the slope coefficient for category :orange[$j$]. 
                    - The coefficients :orange[$\\beta_{0j}$] and :orange[$\\beta_{1j}$] are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if :orange[$\\beta_{1j}$] is significantly different from zero for each category.
                    """,
    core_assumptions="""- **Nominal DV** — the outcome has 3+ unordered categories
- **Independence** — observations are independent
- **IIA assumption** — odds of choosing one category over another are independent of other categories
- **Logit linearity** — log-odds have a linear relationship with continuous predictors
- **Large sample** — sufficient observations per outcome category per predictor""",
    interpretation="""- Estimates coefficients for each outcome category **relative to a reference category**
- Report: **relative risk ratios (RRR)** = exp(β) with **95% CI**, **p-values**
- **RRR**: for a one-unit increase in X, odds of being in category j vs. reference multiply by RRR
- **Pseudo-R²** and **log-likelihood** assess overall model fit
- **IIA assumption** can be tested with Hausman or Small-Hsiao tests""",
    realworld_apps="""- **Transportation**: predicting travel mode (car, bus, train, bike) from cost and time
- **Political science**: predicting voting choice (Dem, Rep, Independent) from demographics
- **Marketing**: predicting brand choice among 3+ competing brands
- **Medical research**: predicting disease subtype from risk factors
- **Education**: predicting college major choice from aptitude scores""",
)

ordinal_logistic_regression = TestDefinition(
    name="Ordinal Logistic Regression",
    objective="Prediction",
    dependent_var="Ordinal",
    independent_var=["Continuous", "Multiple Continuous", "Categorical"],
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["Normal", "Non-normal", "any"],
    explanation="""Ordinal Logistic Regression This test is used to model the relationship between an ordinal dependent variable and one or more continuous independent variables. It assumes that the dependent variable is ordinal, and that the relationship between the independent variables and the log-odds of each category of the dependent variable is linear.""",
    example="""A researcher wants to predict the level of satisfaction (very unsatisfied, unsatisfied, neutral, satisfied, very satisfied) based on income. The researcher collects data on income and satisfaction levels from 100 participants and performs an Ordinal Logistic Regression to determine if income is a significant predictor of satisfaction level.""",
    formula="""
                    $$ \\log\\left(\\dfrac{P(Y \\leq j)}{P(Y > j)}\\right) = \\beta_{0j} + \\beta_{1j} X $$ 
                    Where: 
                    - :orange[$P(Y \\leq j)$] is the probability of the dependent variable being in category :orange[$j$] or lower, 
                    - :orange[$P(Y > j)$] is the probability of the dependent variable being in a category higher than :orange[$j$], 
                    - :orange[$X$] is the independent variable, 
                    - :orange[$\\beta_{0j}$] is the intercept for category :orange[$j$], and 
                    - :orange[$\\beta_{1j}$] is the slope coefficient for category :orange[$j$]. 
                    - The coefficients :orange[$\\beta_{0j}$] and :orange[$\\beta_{1j}$] are estimated using maximum likelihood estimation, and the significance of the predictor is determined by testing if :orange[$\\beta_{1j}$] is significantly different from zero for each category.
                    """,
    core_assumptions="""- **Ordinal DV** — the outcome has 3+ ordered categories (e.g., Low, Medium, High)
- **Independence** — observations are independent
- **Proportional odds** — predictor effects are constant across all cumulative logits (parallel regression)
- **Logit linearity** — log-odds have a linear relationship with continuous predictors
- **Large sample** — sufficient observations per outcome level""",
    interpretation="""- Models **cumulative logits**: log[P(Y ≤ j) / P(Y > j)] for each threshold j
- Report: **odds ratios (OR)** = exp(β) with **95% CI**, threshold parameters (cut-points)
- **OR**: for a one-unit increase in X, odds of being in a **higher** category multiply by OR
- **Brant test** checks proportional odds — if significant, consider multinomial logistic instead
- **Predicted probabilities** can be calculated for each outcome level""",
    realworld_apps="""- **Medical research**: predicting pain severity (None, Mild, Moderate, Severe)
- **Education**: predicting achievement (Below Basic, Basic, Proficient, Advanced)
- **Survey research**: predicting satisfaction (Very Dissatisfied to Very Satisfied)
- **Clinical trials**: predicting response category (Complete, Partial, Stable, Progression)
- **Social science**: predicting SES (Low, Middle, High) from demographics""",
)

poisson_regression = TestDefinition(
    name="Poisson Regression",
    objective="Prediction",
    dependent_var="Discrete",
    independent_var=["Continuous", "Multiple Continuous", "Categorical"],
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["Normal", "Non-normal", "any"],
    explanation="""Poisson Regression This test is used to model the relationship between a count dependent variable and one or more continuous independent variables. It assumes that the dependent variable is a count variable, and that the relationship between the independent variables and the log of the expected count is linear.""",
    example="""A researcher wants to predict the number of hospital visits (0, 1, 2, 3, ...) based on age and income. The researcher collects data on age, income, and hospital visits from 100 participants and performs a Poisson Regression to determine if age and income are significant predictors of hospital visits.""",
    formula="""
                    $$ \\log(\\lambda) = \\beta_{0} + \\beta_{1} X_1 + \\beta_{2} X_2 $$ 
                    Where: 
                    - :orange[$\\lambda$] is the expected count of the dependent variable, 
                    - :orange[$X_1$] and :orange[$X_2$] are the independent variables, 
                    - :orange[$\\beta_{0}$] is the intercept, and 
                    - :orange[$\\beta_{1}$] and :orange[$\\beta_{2}$] are the slope coefficients for each independent variable. 
                    - The coefficients :orange[$\\beta_{0}$], :orange[$\\beta_{1}$], and :orange[$\\beta_{2}$] are estimated using maximum likelihood estimation, and the significance of the predictors is determined by testing if they are significantly different from zero.
                    """,
    core_assumptions="""- **Count DV** — the outcome is a non-negative integer (0, 1, 2, ...)
- **Independence** — observations are independent
- **Equidispersion** — the conditional mean equals the conditional variance
- **Log-linear relationship** — log(expected count) has a linear relationship with predictors
- **Large sample** — sufficient observations per predictor""",
    interpretation="""- Report: **incidence rate ratios (IRR)** = exp(β) with **95% CI**, **p-values**
- **IRR**: for a one-unit increase in X, the expected count multiplies by IRR
- **Deviance/df** ratio > 1.5 indicates over-dispersion → consider Negative Binomial
- **Offset term**: log(population/exposure) when modeling rates
- **AIC/BIC** for model comparison""",
    realworld_apps="""- **Epidemiology**: modeling disease case counts per region
- **Insurance**: modeling the number of claims per policyholder
- **Criminology**: modeling crimes per neighborhood from socioeconomic predictors
- **Biology**: modeling cell counts per microscopic field
- **Sports analytics**: modeling goals scored per game from team statistics""",
)

negative_binomial_regression = TestDefinition(
    name="Negative Binomial Regression",
    objective="Prediction",
    dependent_var="Discrete",
    independent_var=["Continuous", "Multiple Continuous", "Categorical"],
    groups=["any", "2", "More than 2"],
    relation=["any", "Dependent", "Independent"],
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Negative Binomial Regression is used for modeling count data when the variance exceeds the mean (overdispersion), which violates the Poisson Regression assumption of equal mean and variance. It adds an extra dispersion parameter to account for unobserved heterogeneity. It is commonly used in medical research for counts with many zeros or high variability, such as hospital readmissions, number of seizures, or dental caries counts.""",
    example="""A researcher wants to model the number of dental caries (cavities) in children based on sugar consumption, fluoride exposure, and brushing frequency. The count data shows variance much larger than the mean, so Negative Binomial Regression is chosen over Poisson Regression to account for overdispersion.""",
    formula="""
                    $$ \\log(\\lambda_i) = \\beta_0 + \\beta_1 X_{1i} + \\beta_2 X_{2i} + \\ldots + \\beta_k X_{ki} $$
                    $$ \\text{Var}(Y_i) = \\lambda_i + \\alpha \\lambda_i^2 $$
                    Where:
                    - :orange[$\\lambda_i$] is the expected count for observation :orange[$i$],
                    - :orange[$\\alpha$] is the dispersion parameter (:orange[$\\alpha = 0$] reduces to Poisson),
                    - :orange[$\\alpha > 0$] indicates overdispersion,
                    - Coefficients are estimated using maximum likelihood.
                    """,
    core_assumptions="""- **Count DV** — the outcome is a non-negative integer
- **Independence** — observations are independent
- **Over-dispersion** — variance exceeds the mean (handled by the dispersion parameter θ)
- **Log-linear relationship** — log(expected count) has a linear relationship with predictors
- **Dispersion parameter (θ)** captures extra variability beyond Poisson""",
    interpretation="""- Report: **IRR** = exp(β) with **95% CI**, **dispersion parameter (θ)**, **p-values**
- Smaller θ = more over-dispersion; as θ → ∞, NB → Poisson
- **IRR**: for a one-unit increase in X, expected count multiplies by IRR
- **LR test** comparing NB to Poisson: significant LR confirms over-dispersion
- Provides **better-calibrated standard errors** than Poisson when over-dispersion is present""",
    realworld_apps="""- **Public health**: modeling hospital readmission counts (patients vary greatly in risk)
- **Ecology**: modeling species abundance (some sites have vastly more individuals)
- **Insurance**: modeling claim counts (some policyholders file many claims)
- **Genomics**: modeling gene expression counts (RNA-Seq over-dispersion)
- **Manufacturing**: modeling defect counts across batches of varying quality""",
)

sensitivity_specificity_analysis = TestDefinition(
    name="Sensitivity & Specificity Analysis",
    objective="Diagnostic Accuracy",
    dependent_var="Binary/Dichotomous",
    independent_var="Binary/Dichotomous",
    groups=["any", "2", "More than 2"],
    relation=["any", "Dependent", "Independent"],
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Sensitivity and Specificity measures the performance of a binary diagnostic test against a gold standard. Sensitivity (True Positive Rate) is the ability to correctly identify those with the disease, while Specificity (True Negative Rate) is the ability to correctly identify those without the disease.""",
    example="""A new rapid antigen test is compared against PCR (gold standard) for COVID-19. 100 people known to have the virus and 100 known to be healthy are tested to calculate the accuracy metrics.""",
    formula="""
                    $$ \\text{Sensitivity} = \\dfrac{TP}{TP + FN} $$
                    $$ \\text{Specificity} = \\dfrac{TN}{TN + FP} $$
                    $$ \\text{Positive Predictive Value (PPV)} = \\dfrac{TP}{TP + FP} $$
                    $$ \\text{Negative Predictive Value (NPV)} = \\dfrac{TN}{TN + FN} $$
                    $$ \\text{Accuracy} = \\dfrac{TP + TN}{TP + TN + FP + FN} $$
                    $$ \\text{Likelihood ratio for a positive test (LR+)} = \\dfrac{\\text{Sensitivity}}{1 - \\text{Specificity}} $$
                    $$ \\text{Likelihood ratio for a negative test (LR-)} = \\dfrac{1 - \\text{Sensitivity}}{\\text{Specificity}} $$
                    $$ \\text{F1 Score} = 2 \\times \\dfrac{\\text{PPV} \\times \\text{Sensitivity}}{\\text{PPV} + \\text{Sensitivity}} $$
                    $$ \\text{Diagnostic Odds Ratio (DOR)} = \\dfrac{LR+}{LR-} $$
                    Where:
                    - :orange[$TP$]: True Positives
                    - :orange[$TN$]: True Negatives
                    - :orange[$FP$]: False Positives
                    - :orange[$FN$]: False Negatives
                    """,
    core_assumptions="""- **Binary reference standard** — a gold standard determines true disease status
- **Binary test result** — the diagnostic test yields a binary result (positive/negative)
- **Independent comparison** — test result vs. reference standard in a blinded manner
- **Representative sample** — study population reflects the target population
- **Verification bias avoided** — all subjects receive both test and reference standard""",
    interpretation="""- Report: **Sensitivity**, **Specificity**, **PPV**, **NPV**, **Accuracy**
- **Sensitivity** = TP/(TP+FN) — proportion of diseased correctly identified
- **Specificity** = TN/(TN+FP) — proportion of non-diseased correctly ruled out
- **PPV** = proportion of positive tests that are true positives (depends on prevalence)
- **NPV** = proportion of negative tests that are true negatives
- Higher sensitivity often comes at the cost of lower specificity (and vice versa)
- **95% CI** should be reported for all metrics""",
    realworld_apps="""- **Medical screening**: evaluating a new COVID-19 rapid test against PCR gold standard
- **Radiology**: assessing mammography accuracy for breast cancer detection
- **Laboratory medicine**: validating new biomarker tests against established assays
- **Diagnostic imaging**: comparing MRI vs. CT scan accuracy for detecting tumors
- **Public health**: evaluating field diagnostic tests for infectious disease surveillance""",
)

roc_curve_analysis = TestDefinition(
    name="ROC Curve Analysis",
    objective="Diagnostic Accuracy",
    dependent_var="Binary/Dichotomous",
    independent_var="Continuous",
    groups=["any", "2", "More than 2"],
    relation=["any", "Dependent", "Independent"],
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Receiver Operating Characteristic (ROC) analysis is used to evaluate the performance of a continuous diagnostic test. It plots Sensitivity against 1-Specificity at various thresholds. The Area Under the Curve (AUC) represents the overall accuracy.""",
    example="""A researcher wants to determine if blood sugar levels can accurately diagnose diabetes. By plotting an ROC curve, they can find the optimal sugar level cut-off that maximizes both sensitivity and specificity.""",
    formula="""
                    $$ \\text{AUC} = \\int_{0}^{1} \\text{Sensitivity}(t) \\, d(1 - \\text{Specificity}(t)) $$
                    - :orange[AUC = 0.5]: Random guessing
                    - :orange[AUC = 1.0]: Perfect diagnostic accuracy
                    """,
    core_assumptions="""- **Binary reference standard** — a gold standard determines true disease status
- **Continuous or ordinal test result** — the diagnostic test yields a score (not just binary)
- **Independent observations** — subjects are independent
- **Representative sample** — covers the full spectrum of disease severity""",
    interpretation="""- **AUC** (Area Under the ROC Curve) measures the test's overall discriminatory ability
- **AUC**: 0.5 = no discrimination, 0.7−0.8 = acceptable, 0.8−0.9 = excellent, > 0.9 = outstanding
- ROC curve plots **Sensitivity vs. 1−Specificity** across all possible cut-off values
- **Optimal cut-off** maximizes Youden's Index (Sensitivity + Specificity − 1)
- **95% CI for AUC** quantifies precision; **DeLong's test** compares AUCs of two tests""",
    realworld_apps="""- **Clinical diagnostics**: finding optimal cut-off for a new biomarker (e.g., HbA1c for diabetes)
- **Radiology**: comparing AUC of different imaging modalities for cancer detection
- **Machine learning**: evaluating classifier performance (AUC is threshold-independent)
- **Laboratory medicine**: setting reference ranges for new diagnostic tests
- **Risk prediction**: evaluating discriminative ability of prognostic models (e.g., Framingham Score)""",
)

likelihood_ratio_analysis = TestDefinition(
    name="Likelihood Ratio Analysis",
    objective="Diagnostic Accuracy",
    dependent_var="Binary/Dichotomous",
    independent_var="Binary/Dichotomous",
    groups=["any", "2", "More than 2"],
    relation=["any", "Dependent", "Independent"],
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Likelihood Ratios (LR) are used to assess the value of performing a diagnostic test. LR+ indicates how much more likely a positive test is to be found in a person with the disease than in a person without. LR- indicates how much less likely a negative test is to be found in a person with the disease than in a person without.""",
    example="""A clinician uses the LR+ of a physical exam finding to update their post-test probability of a patient having appendicitis.""",
    formula="""
                    $$ LR+ = \\dfrac{\\text{Sensitivity}}{1 - \\text{Specificity}} $$
                    $$ LR- = \\dfrac{1 - \\text{Sensitivity}}{\\text{Specificity}} $$
                    - :orange[LR+ > 10]: Large increase in disease probability
                    - :orange[LR- < 0.1]: Large decrease in disease probability
                    """,
    core_assumptions="""- **Binary reference standard** — gold standard for true disease status
- **Binary test result** — the test produces positive or negative results
- **Independent observations** — subjects are independent
- **Representative sample** — appropriate disease spectrum is represented""",
    interpretation="""- **LR+** = Sensitivity / (1 − Specificity) — how much a positive result increases disease odds
- **LR−** = (1 − Sensitivity) / Specificity — how much a negative result decreases disease odds
- **LR+ > 10**: large increase; **5−10**: moderate; **2−5**: small; **1−2**: minimal
- **LR− < 0.1**: large decrease; **0.1−0.2**: moderate; **0.2−0.5**: small; **0.5−1**: minimal
- **Post-test odds** = Pre-test odds × LR (apply Bayes' theorem to update probability)
- Likelihood ratios are **prevalence-independent** (unlike PPV/NPV)""",
    realworld_apps="""- **Clinical decision-making**: updating disease probability after a diagnostic test result
- **Evidence-based medicine**: comparing clinical utility of multiple diagnostic tests
- **Medical education**: teaching Bayesian reasoning in diagnosis
- **Systematic reviews**: meta-analyzing diagnostic accuracy across studies
- **Guideline development**: determining which tests provide meaningful information""",
)

cohen_s_kappa_agreement_analysis = TestDefinition(
    name="Cohen's Kappa (Agreement Analysis)",
    objective="Diagnostic Accuracy",
    dependent_var="Categorical",
    independent_var="Categorical",
    groups="2",
    relation="Dependent",
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Cohen's Kappa is used to measure inter-rater or intra-rater agreement for categorical variables. It accounts for the agreement occurring by chance.""",
    example="""Two radiologists evaluate the same set of X-rays to diagnose a fracture. Cohen's Kappa measures how consistently they agree on the presence or absence of a fracture.""",
    formula="""
                    $$ \\kappa = \\dfrac{p_o - p_e}{1 - p_e} $$
                    Where:
                    - :orange[$p_o$]: Observed proportionate agreement
                    - :orange[$p_e$]: Probability of random agreement
                    """,
    core_assumptions="""- **Categorical ratings** — two raters classify items into the same nominal categories
- **Independent ratings** — each rating is made independently
- **Same categories** — both raters use the same set of categories
- **Fixed number of categories** — predetermined and exhaustive
- **Rater independence** — ratings are not influenced by the other rater""",
    interpretation="""- κ (kappa) measures **agreement beyond chance** between two raters
- κ ranges from −1 to +1: negative = worse than chance, 0 = chance, 1 = perfect
- **Landis & Koch**: ≤ 0 = poor, 0−0.20 = slight, 0.21−0.40 = fair, 0.41−0.60 = moderate, 0.61−0.80 = substantial, 0.81−1.00 = almost perfect
- Report: κ, **95% CI**, **observed agreement (%)**, **p-value**
- **Paradox**: κ can be low despite high agreement when prevalence is extreme
- Use **Weighted Kappa** for ordinal categories (gives partial credit for near-agreements)""",
    realworld_apps="""- **Radiology**: measuring agreement between two radiologists interpreting X-rays
- **Psychiatry**: assessing diagnostic agreement between two clinicians using DSM criteria
- **Medical coding**: checking consistency of ICD-10 code assignment across coders
- **Content analysis**: measuring inter-coder reliability in qualitative research
- **Pathology**: evaluating agreement between pathologists grading tissue samples""",
)

bland_altman_analysis = TestDefinition(
    name="Bland-Altman Analysis",
    objective="Diagnostic Accuracy",
    dependent_var="Continuous",
    independent_var="Continuous",
    groups="2",
    relation="Dependent",
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Bland-Altman Analysis is the standard method for assessing agreement between two quantitative measurement techniques. It plots the difference between paired measurements against their mean, calculates the mean difference (bias), and defines limits of agreement (mean difference ± 1.96 SD of differences). Unlike correlation, which measures association, Bland-Altman directly assesses interchangeability. It assumes the differences are approximately normally distributed.""",
    example="""A researcher develops a new digital caliper for measuring tooth dimensions and wants to know if it agrees with the traditional mechanical caliper. 50 teeth are measured with both instruments. Bland-Altman Analysis shows a mean difference of 0.02 mm (negligible bias) with limits of agreement from −0.15 to +0.19 mm, confirming the new caliper can replace the old one for clinical purposes.""",
    formula="""
                    $$ \\bar{d} = \\frac{1}{n} \\sum_{i=1}^{n} (X_i - Y_i) $$
                    $$ s_d = \\sqrt{\\frac{\\sum_{i=1}^{n} (d_i - \\bar{d})^2}{n-1}} $$
                    $$ \\text{Upper LoA} = \\bar{d} + 1.96 \\times s_d $$
                    $$ \\text{Lower LoA} = \\bar{d} - 1.96 \\times s_d $$
                    Where:
                    - :orange[$X_i$] and :orange[$Y_i$] are paired measurements from two methods,
                    - :orange[$\\bar{d}$] is the mean difference (bias),
                    - :orange[$s_d$] is the standard deviation of the differences,
                    - :orange[LoA] are the limits of agreement (95% tolerance limits).
                    """,
    core_assumptions="""- **Continuous measurements** — both methods produce measurements on a continuous scale
- **Paired measurements** — each subject is measured by both methods
- **Independent subjects** — subjects are independent
- **Approximately normal differences** — for calculating limits of agreement
- **Homoscedasticity** — variability of differences is constant across the measurement range""",
    interpretation="""- **Bland-Altman plot**: difference between two methods vs. their mean
- **Mean difference** = estimate of **fixed bias** (one method reads consistently higher/lower)
- **Limits of Agreement (LoA)** = mean diff ± 1.96 × SD of differences (95% of differences fall here)
- **Clinical judgment**: LoA must be judged against pre-defined acceptable limits
- **Proportional bias**: if differences fan out as mean increases, assumptions are violated
- This is a **descriptive** method — no p-value is produced""",
    realworld_apps="""- **Medical devices**: validating a new blood pressure monitor against mercury sphygmomanometer
- **Lab methods**: comparing a new rapid assay against the established method
- **Wearable technology**: validating step counts from a fitness tracker against observation
- **Imaging**: comparing tumor size measurements from CT and MRI
- **Nutrition**: validating a food frequency questionnaire against 24-hour recall""",
)

weighted_kappa = TestDefinition(
    name="Weighted Kappa",
    objective="Diagnostic Accuracy",
    dependent_var="Ordinal",
    independent_var="Ordinal",
    groups="2",
    relation="Dependent",
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Weighted Kappa extends Cohen's Kappa to ordinal categorical ratings by incorporating partial credit for disagreements that are close (e.g., 'mild' vs. 'moderate' disagreement is penalized less than 'mild' vs. 'severe'). Linear weights penalize disagreements proportionally to their distance; quadratic weights penalize more severely. It is the standard agreement measure for ordinal scales in medical research.""",
    example="""Two radiologists independently classify 100 mammograms into four categories: normal, benign, suspicious, and malignant. Weighted Kappa is used to measure their agreement, where a disagreement between 'normal' and 'benign' is penalized less than between 'normal' and 'malignant'.""",
    formula="""
                    $$ \\kappa_w = 1 - \\frac{\\sum_{i=1}^{k} \\sum_{j=1}^{k} w_{ij} O_{ij}}{\\sum_{i=1}^{k} \\sum_{j=1}^{k} w_{ij} E_{ij}} $$
                    Where:
                    - :orange[$O_{ij}$] and :orange[$E_{ij}$] are observed and expected frequencies for cell :orange[$(i,j)$],
                    - :orange[$w_{ij}$] is the weight (0 for perfect agreement, 1 for maximum disagreement),
                    - Linear weights: :orange[$w_{ij} = \\frac{|i-j|}{k-1}$],
                    - Quadratic weights: :orange[$w_{ij} = \\frac{(i-j)^2}{(k-1)^2}$].
                    """,
    core_assumptions="""- **Ordinal ratings** — two raters classify items into ordered categories
- **Independent ratings** — each rating is made independently
- **Same ordinal scale** — both raters use the same ordered categories
- **Pre-specified weights** — quadratic or linear weights determine disagreement penalty
- **Fixed number of categories**""",
    interpretation="""- **Weighted κ** extends Cohen's Kappa to ordinal data, giving **partial credit** for near-agreements
- **Linear weights**: penalty proportional to distance between categories
- **Quadratic weights** (most common): penalty proportional to squared distance — equivalent to ICC
- Uses same Landis & Koch benchmarks as Cohen's Kappa
- Report: **weighted κ**, **95% CI**, **p-value**, and whether linear or quadratic weights used
- Quadratic weighting is generally preferred for ordinal disagreements""",
    realworld_apps="""- **Radiology staging**: measuring agreement between radiologists grading cancer stages (I, II, III, IV)
- **Pain research**: assessing agreement on ordinal pain severity ratings (0−10 scale)
- **Pathology**: evaluating agreement on tissue grading (Normal, Dysplasia, CIS, Invasive)
- **Psychiatry**: measuring agreement on ordinal symptom severity scales
- **Rehabilitation**: assessing agreement on functional independence ratings""",
)

fleiss_kappa = TestDefinition(
    name="Fleiss' Kappa",
    objective="Diagnostic Accuracy",
    dependent_var="Categorical",
    independent_var="Categorical",
    groups="More than 2",
    relation="Dependent",
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Fleiss' Kappa is an extension of Cohen's Kappa that measures inter-rater agreement for three or more raters evaluating categorical (nominal) ratings. Unlike Cohen's Kappa, which handles only two raters, Fleiss' Kappa simultaneously assesses agreement among multiple raters while correcting for chance agreement. It ranges from −1 to +1, with +1 indicating perfect agreement.""",
    example="""Three oral pathologists independently classify 50 biopsy slides into diagnostic categories (benign, dysplastic, malignant). Fleiss' Kappa measures the overall agreement among all three pathologists simultaneously, correcting for chance.""",
    formula="""
                    $$ \\kappa = \\frac{\\bar{P} - \\bar{P_e}}{1 - \\bar{P_e}} $$
                    Where:
                    - :orange[$\\bar{P}$] is the mean proportion of observed agreement across all raters,
                    - :orange[$\\bar{P_e}$] is the mean proportion of expected agreement by chance,
                    - :orange[$\\kappa = 1$]: perfect agreement,
                    - :orange[$\\kappa = 0$]: agreement equivalent to chance,
                    - :orange[$\\kappa < 0$]: less than chance agreement.
                    """,
    core_assumptions="""- **Categorical ratings** — three or more raters classify items into nominal categories
- **Independent raters** — each rater makes their judgment independently
- **Same categories** — all raters use the same set of categories
- **Complete design** — all items are rated by all raters
- **Rater independence** — no rater is influenced by others""",
    interpretation="""- **Fleiss' κ** measures agreement among 3+ raters beyond chance for nominal categories
- κ ranges from −1 to +1, interpreted with the same Landis & Koch benchmarks
- Report: κ, **95% CI** (bootstrap or asymptotic), **p-value**
- Tends to be **lower** than the average pairwise Cohen's κ
- Assumes the **same set of raters** for all subjects
- **Pairwise Cohen's Kappa** can complement Fleiss' to identify specific rater pairs with poor agreement""",
    realworld_apps="""- **Multi-center clinical trials**: measuring diagnostic agreement across multiple pathologists
- **Medical imaging**: assessing inter-rater reliability among multiple radiologists
- **Qualitative research**: measuring agreement among multiple coders in content analysis
- **Psychiatric diagnosis**: evaluating consistency across multiple clinicians
- **Drug safety**: assessing agreement among multiple reviewers classifying adverse events""",
)

kaplan_meier_survival_analysis = TestDefinition(
    name="Kaplan-Meier Survival Analysis",
    objective="Survival Analysis",
    dependent_var="Time-to-event",
    independent_var="Categorical",
    groups="2",
    relation="Independent",
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Kaplan-Meier Survival Analysis is a non-parametric method used to estimate the survival function from time-to-event data. It accounts for censored data (individuals lost to follow-up or not experiencing the event by the end of the study). The Kaplan-Meier curve plots the probability of survival over time, and the Log-Rank Test can be used to compare survival curves between groups.""",
    example="""A researcher wants to compare the survival times of patients with two different types of cancer. They collect time-to-event data (time until death or last follow-up) for 100 patients in each group and use Kaplan-Meier analysis to estimate and compare the survival curves.""",
    formula="""
                    $$ \\hat{S}(t) = \\prod_{t_i \\leq t} \\left(1 - \\frac{d_i}{n_i}\\right) $$
                    Where:
                    - :orange[$\\hat{S}(t)$] is the estimated survival probability at time :orange[$t$],
                    - :orange[$t_i$] are the distinct event times,
                    - :orange[$d_i$] is the number of events at time :orange[$t_i$],
                    - :orange[$n_i$] is the number of individuals at risk just before time :orange[$t_i$].
                    """,
    core_assumptions="""- **Time-to-event data** — the outcome is the time until an event occurs
- **Right censoring** — subjects who do not experience the event by end of follow-up are censored
- **Non-informative censoring** — censored subjects have the same future survival probability as those remaining
- **Independent censoring** — censoring is unrelated to the event
- **No shape assumptions** — non-parametric method estimating survival probabilities at each event time""",
    interpretation="""- **Kaplan-Meier curve** plots estimated survival probability vs. time with step-downs at event times
- **Median survival time** = time when survival probability drops to 0.5 (if observed)
- **Survival probability** at specific time points with 95% CI (e.g., 1-year survival)
- **Number at risk** shown below the curve indicates remaining subjects at each time point
- **Censored observations** are marked as tick marks on the curve
- KM curve does NOT test group differences — use the **Log-Rank Test** for comparisons""",
    realworld_apps="""- **Clinical trials**: estimating overall survival in cancer patients receiving treatments
- **Epidemiology**: estimating time to disease onset in longitudinal cohort studies
- **Engineering**: estimating time to equipment failure (reliability analysis)
- **Social science**: estimating time to re-employment after job loss
- **Demography**: estimating survival from life-table data in population studies""",
)

cox_proportional_hazards_regression = TestDefinition(
    name="Cox Proportional Hazards Regression",
    objective="Survival Analysis",
    dependent_var="Time-to-event",
    independent_var=["Continuous", "Multiple Continuous", "Categorical"],
    groups=["any", "2", "More than 2"],
    relation=["Independent", "Dependent", "any"],
    distribution=["any", "Non-normal", "Normal"],
    explanation="""Cox Proportional Hazards Regression is the most widely used model for survival analysis. It estimates the hazard ratio (HR) for time-to-event outcomes while adjusting for multiple covariates simultaneously. It is a semi-parametric model that makes no assumption about the shape of the baseline hazard function, but assumes proportional hazards (the effect of each predictor is constant over time).""",
    example="""A researcher wants to identify predictors of survival time after a cancer diagnosis. They collect data on age, tumor stage, and treatment type from 200 patients and perform Cox regression to estimate the hazard ratios for each predictor, indicating which factors significantly increase or decrease mortality risk.""",
    formula="""
                    $$ h(t) = h_0(t) \\exp(\\beta_1 X_1 + \\beta_2 X_2 + \\ldots + \\beta_k X_k) $$
                    Where:
                    - :orange[$h(t)$] is the hazard at time :orange[$t$],
                    - :orange[$h_0(t)$] is the baseline hazard (when all predictors are zero),
                    - :orange[$\\beta_1, \\beta_2, \\ldots$] are the log-hazard ratios for predictors :orange[$X_1, X_2, \\ldots$],
                    - :orange[$\\exp(\\beta_i)$] is the hazard ratio for predictor :orange[$X_i$],
                    - The proportional hazards assumption means :orange[$h_0(t)$] cancels out when comparing two individuals.
                    """,
    core_assumptions="""- **Time-to-event data** — the outcome is the time until an event
- **Right censoring** — censored observations are allowed
- **Non-informative censoring** — censoring is unrelated to the event
- **Proportional Hazards (PH)** — hazard ratios are constant over time (test with Schoenfeld residuals)
- **Log-linear relationship** — continuous predictors have a linear relationship with the log-hazard
- **Independence** — observations are independent""",
    interpretation="""- Report: **Hazard Ratios (HR)** = exp(β) with **95% CI**, **p-values**, **Wald statistics**
- **HR > 1**: higher predictor → increased hazard (shorter survival)
- **HR < 1**: higher predictor → decreased hazard (longer survival)
- **Schoenfeld residuals test** checks PH assumption — if significant (p < 0.05), PH is violated
- **C-index** (concordance) = proportion of pairs where model correctly predicts event order (like AUC)
- Cox is a **semi-parametric** model (no baseline hazard estimation required)""",
    realworld_apps="""- **Clinical trials**: estimating treatment effect on survival adjusting for age, sex, comorbidities
- **Epidemiology**: identifying risk factors for disease mortality in cohort studies
- **Health services**: modeling time to hospital readmission adjusting for patient characteristics
- **Pharmacovigilance**: estimating drug effect on time to adverse event controlling for confounders
- **Engineering**: analyzing time to equipment failure with covariates (temp, usage, maintenance)""",
)

log_rank_test = TestDefinition(
    name="Log-Rank Test",
    objective="Survival Analysis",
    dependent_var="Time-to-event",
    independent_var="Categorical",
    groups="2",
    relation="Independent",
    distribution=["any", "Non-normal", "Normal"],
    explanation="""The Log-Rank Test is a non-parametric test that compares the survival distributions of two or more independent groups. It tests whether the time-to-event differs significantly between groups. It makes no assumption about the shape of the survival curves but assumes that the hazard rates are proportional over time. It is commonly used alongside Kaplan-Meier survival curves.""",
    example="""A researcher compares survival times between 50 patients receiving a new cancer drug and 50 receiving standard therapy. The Log-Rank Test determines if the survival difference between the two groups is statistically significant.""",
    formula="""
                    $$ \\chi^2 = \\frac{(O_1 - E_1)^2}{E_1} + \\frac{(O_2 - E_2)^2}{E_2} $$
                    Where:
                    - :orange[$O_1$] and :orange[$O_2$] are the observed number of events in each group,
                    - :orange[$E_1$] and :orange[$E_2$] are the expected number of events under the null hypothesis of no difference,
                    - The test statistic is compared to a chi-square distribution with 1 degree of freedom (for two groups).
                     - For more than two groups, an extension with :orange[$k-1$] degrees of freedom is used.
        """,
    core_assumptions="""- **Time-to-event data** — the outcome is the time until an event
- **Two independent groups** — comparison of survival between two groups
- **Non-informative censoring** — censoring is unrelated to the event in both groups
- **Proportional hazards** — the hazard ratio between groups is approximately constant over time
- **Independent observations** — subjects are independent within and between groups""",
    interpretation="""- If **p < α**, reject H₀: the survival distributions of the two groups are significantly different
- Report: **χ²** = chi-square statistic, **df** = 1, **p** = p-value
- Compares **observed vs. expected** events in each group at each event time
- **Hazard Ratio (HR)** can be estimated alongside—HR > 1 means treatment group has events sooner
- **Kaplan-Meier curves** should always accompany the Log-Rank Test for visual interpretation
- Most powerful when **proportional hazards** holds; if curves cross, consider alternatives""",
    realworld_apps="""- **Clinical trials**: comparing survival between a new cancer drug and standard of care
- **Medical device**: comparing time to device failure between two implant designs
- **Epidemiology**: comparing time to disease onset between exposed and unexposed groups
- **Public health**: comparing time to hospital readmission between intervention and control
- **Reliability engineering**: comparing time to failure between two product designs""",
)

cochran_mantel_haenszel_test = TestDefinition(
    name="Cochran-Mantel-Haenszel Test",
    objective="Association/Correlation",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups=["2", "More than 2"],
    relation="Independent",
    distribution=["any", "Non-normal", "Normal"],
    explanation="""The Cochran-Mantel-Haenszel (CMH) test is a statistical test used to assess the association between two binary variables after adjusting for one or more confounding/stratifying variables. It combines information across multiple 2×2 contingency tables (strata) to produce a single summary odds ratio and test of conditional independence. The test assumes that the odds ratio is consistent across strata (homogeneity assumption).""",
    example="""A researcher wants to test whether a new treatment is associated with improved recovery, but suspects that patient age group (young vs. old) may confound the results. They collect data stratified by age group, creating separate 2×2 tables for each age stratum, and apply the CMH test to obtain an adjusted odds ratio and assess the treatment effect while controlling for age.""",
    formula="""
                    $$ \\chi^2_{MH} = \\frac{\\left(\\sum_i a_i - \\sum_i \\frac{n_{1i}m_{1i}}{N_i}\\right)^2}{\\sum_i \\frac{n_{1i}n_{2i}m_{1i}m_{2i}}{N_i^2(N_i-1)}} $$
                    Where:
                    - :orange[$a_i$] is the observed count in cell (1,1) of stratum :orange[$i$],
                    - :orange[$n_{1i}$] and :orange[$n_{2i}$] are the row totals for stratum :orange[$i$],
                    - :orange[$m_{1i}$] and :orange[$m_{2i}$] are the column totals for stratum :orange[$i$],
                    - :orange[$N_i$] is the total sample size in stratum :orange[$i$],
                    - The test statistic follows a chi-square distribution with 1 degree of freedom.
                    """,
    core_assumptions="""- **Binary outcome** — the DV is binary (success/failure)
- **Binary exposure** — the IV is binary (exposed/unexposed)
- **Stratification variable(s)** — one or more categorical variables defining strata
- **No three-way interaction** — the odds ratio is consistent across strata (homogeneity)
- **Independence** — observations are independent within and between strata""",
    interpretation="""- If **p < α**, reject H₀: exposure is associated with outcome after controlling for stratification
- Report: **CMH χ²** = chi-square statistic, **df** = 1, **p** = p-value
- **Mantel-Haenszel odds ratio** = pooled (common) odds ratio across strata
- **Breslow-Day test** checks homogeneity — if significant, ORs differ across strata
- The CMH test is a **confounder control** method, adjusting for the stratification variable""",
    realworld_apps="""- **Epidemiology**: testing if smoking is associated with lung cancer controlling for age groups
- **Clinical trials**: testing treatment efficacy across multiple study centers
- **Public health**: testing risk factor association across geographic regions
- **Health services**: testing if insurance type is associated with readmission across hospitals
- **Pharmacovigilance**: testing drug-adverse event association across patient age groups""",
)

two_sample_proportion_test = TestDefinition(
    name="Two-sample Proportion Test",
    objective="Comparison",
    dependent_var=["Binary/Dichotomous", "Categorical"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution=["any", "Normal", "Non-normal"],
    explanation="""The Two-sample Proportion Test (also known as the two-proportion z-test) is used to determine whether there is a statistically significant difference between the proportions of two independent groups. It compares the observed proportions in two samples to assess whether they likely come from populations with the same proportion. The test uses the normal approximation to the binomial distribution, which works well when the sample sizes are sufficiently large.""",
    example="""A researcher wants to compare the proportion of patients who recover after receiving either a new drug or a placebo. Out of 120 patients receiving the drug, 80 recovered, while out of 110 patients receiving the placebo, 60 recovered. The two-sample proportion test determines whether the recovery rate differs significantly between the two groups.""",
    formula="""
                    $$ z = \\frac{p_1 - p_2}{\\sqrt{\\bar{p}(1-\\bar{p})\\left(\\frac{1}{n_1} + \\frac{1}{n_2}\\right)}} $$
                    Where:
                    - :orange[$p_1$] and :orange[$p_2$] are the sample proportions in each group,
                    - :orange[$n_1$] and :orange[$n_2$] are the sample sizes,
                    - :orange[$\\bar{p} = \\frac{x_1 + x_2}{n_1 + n_2}$] is the pooled proportion,
                    - The test statistic :orange[$z$] follows a standard normal distribution under the null hypothesis.
                    """,
    core_assumptions="""- **Binary DV** — the outcome has exactly two categories (success/failure) in each group
- **Independence** — observations are independent within and between groups
- **Random sampling** — each group is a random sample from its population
- **Large sample** — n₁p̂₁ ≥ 5, n₁(1−p̂₁) ≥ 5, n₂p̂₂ ≥ 5, n₂(1−p̂₂) ≥ 5
- **Fixed sample sizes** — n₁ and n₂ are fixed (not the number of successes)""",
    interpretation="""- If **p < α**, reject H₀: the two population proportions are significantly different
- Report: **z** = z-statistic, **p** = p-value, **difference in proportions** = p̂₁ − p̂₂
- **Confidence interval**: (p̂₁−p̂₂) ± z_(α/2) × SE
- **Effect size**: risk difference, risk ratio (p̂₁/p̂₂), or odds ratio
- For small samples, **Fisher's Exact Test** is preferred""",
    realworld_apps="""- **Clinical trials**: comparing response rates between treatment and control groups
- **A/B testing**: comparing conversion rates between two website designs
- **Public health**: comparing vaccination rates between two regions
- **Political polling**: comparing candidate support between male and female voters
- **Quality control**: comparing defect rates between two production shifts""",
)

yuen_s_trimmed_t_test = TestDefinition(
    name="Yuen's Trimmed t-test",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="Normal",
    explanation="""Yuen's Trimmed t-test is a robust alternative to the independent t-test that is relatively insensitive to outliers. It works by trimming a proportion of observations from each tail of both groups before computing the test statistic. It tests whether the trimmed means of two independent groups differ, and is particularly recommended when data contain severe outliers that cannot be resolved by transformation.""",
    example="""A researcher compares cortisol levels between a stress intervention group and a control group. The data contain several extreme values (cortisol spikes). Rather than excluding outliers arbitrarily, Yuen's trimmed t-test (with 20% trimming) provides a robust comparison of the groups central tendencies while downweighting the impact of extreme observations.""",
    formula="""
                    Let :orange[$h_j = n_j - 2 \\lfloor n_j g \\rfloor$] be the effective sample size after trimming proportion :orange[$g$] from each tail.
                    
                    **Step 1:** Sort each sample and trim :orange[$g$] proportion from each tail (typically :orange[$g = 0.1$] or :orange[$g = 0.2$])
                    
                    **Step 2:** Compute trimmed means :orange[$\\bar{X}_{t1}$], :orange[$\\bar{X}_{t2}$]
                    
                    **Step 3:** Compute winsorized variances :orange[$s_{w1}^2$], :orange[$s_{w2}^2$]
                    
                    :orange[**Yuen's test statistic:**]
                    Trimmed means test statistic is calculated as:
                    $$ t_y = \\dfrac{\\bar{X}_{t1} - \\bar{X}_{t2}}{\\sqrt{\\dfrac{(n_1 - 1)s_{w1}^2}{h_1(h_1 - 1)} + \\dfrac{(n_2 - 1)s_{w2}^2}{h_2(h_2 - 1)}}} $$
                    utilizing the winsorized variances to account for the trimmed data, and the sample sizes after trimming $h = n - 2 \\lfloor ng \\rfloor$.
                    
                    :orange[**Winsorized Means Test**]
                    Winsorized means test statistic is calculated as:
                    $$ t_w = \\dfrac{\\bar{X}_{w1} - \\bar{X}_{w2}}{\\sqrt{\\dfrac{s_{w1}^2}{n_1} + \\dfrac{s_{w2}^2}{n_2}}} $$
                    utilizing the winsorized means and variances, which replace extreme values with the nearest non-extreme values. Sample sizes remain the same as the original data $n$.
                    
                    :orange[**Degrees of freedom**] (Welch-type approximation):
                    $$ \\nu = \\dfrac{\\left(\\dfrac{(n_1-1)s_{w1}^2}{h_1(h_1-1)} + \\dfrac{(n_2-1)s_{w2}^2}{h_2(h_2-1)}\\right)^2}{\\dfrac{\\left(\\dfrac{(n_1-1)s_{w1}^2}{h_1(h_1-1)}\\right)^2}{h_1 - 1} + \\dfrac{\\left(\\frac{(n_2-1)s_{w2}^2}{h_2(h_2-1)}\\right)^2}{h_2 - 1}} $$
                    """,
    decision_rules="""
                    - Reject :orange[$H_0: \\mu_{t1} = \\mu_{t2}$] if :orange[$|t_y| > t_{\\alpha/2, \\nu}$] or p-value < :orange[$\\alpha$].
                    - Yuen's test is recommended when outliers are present and the assumptions of Student's t-test are violated.
                    - A trimming proportion of 20% (:orange[$g = 0.2$]) is commonly used as a default.
                    - The test assumes symmetry of the distributions after trimming.
                    - Effect size can be reported as the difference in trimmed means with a bootstrap confidence interval.
        """,
    core_assumptions="""- **Continuous DV** — the outcome variable is measured on a continuous scale
- **Independence** — observations between groups are independent
- **Robust to non-normality** — the trimmed mean removes the influence of outliers
- **Symmetric trimming** — typically 20% trimming (removing top and bottom 10%) is used
- **Large enough sample** — after trimming, sufficient observations remain for valid inference""",
    interpretation="""- If **p < α**, reject H₀: the trimmed group means are significantly different
- Report: **t_y** = Yuen's test statistic, **trimmed mean difference**, **p-value**
- The test compares **trimmed means** rather than arithmetic means — reducing outlier influence
- **Effect size**: trimmed Cohen's d or the difference in trimmed means with confidence interval
- Preferred when data has **heavy tails** or **outliers** that cannot be justifiably removed""",
    realworld_apps="""- **Biomedical research**: analyzing biomarker data with extreme values (e.g., cytokine levels)
- **Economics**: comparing income where outliers (billionaires) distort the mean
- **Psychology**: analyzing reaction time data with very slow responses from distraction
- **Environmental science**: comparing pollutant measurements with occasional extreme readings
- **Sports analytics**: comparing athlete performance when a few exceptional performances skew data""",
)

brunner_munzel_test = TestDefinition(
    name="Brunner-Munzel Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="Non-normal",
    explanation="""The Brunner-Munzel Test is a non-parametric test for comparing two independent groups. Unlike the Mann-Whitney U test, it does NOT assume equal group spreads (i.e., it does not require the two distributions to have the same shape). It tests the hypothesis :orange[$P(X < Y) = P(X > Y)$], i.e., whether one group tends to produce larger values than the other. It is often recommended as a general replacement for Mann-Whitney U because it maintains correct Type I error rates even when variances differ.""",
    example="""A researcher compares depression scores between two treatment groups. The distributions have different spreads — one group shows more variability than the other. The Brunner-Munzel Test is preferred over Mann-Whitney U because it remains valid when the shapes differ.""",
    formula="""
                    Let :orange[$R_{ij}$] be the rank of observation :orange[$j$] in group :orange[$i$] among the combined sample.
                    
                    **Relative effect:**
                    $$ \\hat{p} = \\frac{1}{n_1 n_2} \\sum_{i=1}^{n_1} \\sum_{j=1}^{n_2} I(X_i < Y_j) + \\frac{1}{2} I(X_i = Y_j) $$
                    
                    **Test statistic:**
                    $$ W = \\frac{\\hat{p} - \\frac{1}{2}}{\\hat{\\sigma}} $$
                    
                    Where :orange[$\\hat{\\sigma}$] is estimated from the ranks and accounts for potentially different group variances.
                    
                    The test statistic follows a t-distribution with approximated degrees of freedom (Welch-type).
                    """,
    decision_rules="""
                    - Reject :orange[$H_0: P(X < Y) = P(X > Y)$] if p-value < :orange[$\\alpha$].
                    - The Brunner-Munzel Test is valid even when group distributions have different shapes or variances.
                    - When group spreads are equal, both Mann-Whitney U and Brunner-Munzel are valid, but Mann-Whitney may have slightly more power.
                    - The relative effect :orange[$\\hat{p}$] can be interpreted as the probability that a random observation from group 1 is larger than a random observation from group 2.
                    - Effect size: :orange[$\\hat{p} - 0.5$] with confidence interval.
        """,
    core_assumptions="""- **Ordinal or continuous DV** — the outcome variable is at least ordinal
- **Independence** — observations between groups are independent
- **Independent groups** — two unrelated groups
- **Does NOT assume equal distribution shapes** — key advantage over Mann-Whitney
- **No normality required**""",
    interpretation="""- If **p < α**, reject H₀: there is a stochastic tendency for one group to have larger values
- Report: **W** = test statistic, **df** = Welch-type adjusted df, **p** = p-value
- **Relative effect** = P(X > Y) + 0.5×P(X = Y) — probability a random observation from group 1 exceeds one from group 2
- Relative effect = 0.5 means groups are stochastically equal; > 0.5 means group 1 tends higher
- Preferred over Mann-Whitney when groups have **different distribution shapes**""",
    realworld_apps="""- **Clinical research**: comparing outcomes between groups with different variability
- **Ecology**: comparing species abundance between sites with different distribution shapes
- **Psychology**: comparing reaction times when one group shows more variability (older vs younger)
- **Economics**: comparing income distributions with different degrees of inequality
- **Environmental monitoring**: comparing pollutants when reference and contaminated sites have different shapes""",
)

one_way_welch_anova = TestDefinition(
    name="One-way Welch ANOVA",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var=["Categorical", "Binary/Dichotomous"],
    groups="More than 2",
    relation="Independent",
    distribution="Normal",
    explanation="""Welch's ANOVA is an adaptation of the one-way ANOVA that does NOT assume equal variances across groups. It uses a Welch-type variance-weighted adjustment to the F-statistic and degrees of freedom. It is the preferred ANOVA approach when the homogeneity of variance assumption is violated, and it maintains good Type I error rates even when group sizes are unequal. The test generates a pseudo-F statistic (Welch's F) that is approximately F-distributed with adjusted denominator degrees of freedom.""",
    example="""A researcher compares exam scores across three teaching methods. Levene's test indicates unequal variances across the groups. Instead of standard one-way ANOVA, the researcher uses Welch's ANOVA followed by Games-Howell post-hoc tests.""",
    formula="""
                    **Welch's F-statistic:**
                    $$ F_w = \\frac{\\frac{1}{k-1} \\sum_{i=1}^k w_i (\\bar{x}_i - \\tilde{x})^2}{1 + \\frac{2(k-2)}{k^2-1} \\sum_{i=1}^k \\frac{(1 - w_i / W)^2}{n_i - 1}} $$
                    Where :orange[$w_i = n_i / s_i^2$] are the group weights, :orange[$W = \\sum w_i$], :orange[$\\tilde{x} = \\frac{1}{W} \\sum w_i \\bar{x}_i$] is the weighted grand mean, :orange[$n_i$] and :orange[$s_i^2$] are group sample sizes and variances, and :orange[$k$] is the number of groups.
                    The adjusted denominator degrees of freedom are:
                    $$ df_2 = \\frac{1}{\\frac{3}{k^2-1} \\sum_{i=1}^k \\frac{(1 - w_i/W)^2}{n_i - 1}} $$
                    """,
    decision_rules="""
                    - Reject :orange[$H_0: \\mu_1 = \\mu_2 = \\dots = \\mu_k$] if p-value < :orange[$\\alpha$].
                    - Welch's F is approximately F-distributed with :orange[$k-1$] (numerator) and :orange[$df_2$] (adjusted denominator) degrees of freedom.
                    - Pairwise comparisons use Games-Howell post-hoc tests (does not assume equal variances or equal sample sizes).
                    - Effect size can be reported as :orange[$\\eta^2$] or :orange[$\\omega^2$].
                    - Welch's ANOVA maintains correct Type I error even with unequal variances and unequal group sizes.
        """,
    core_assumptions="""- **Continuous DV** — the outcome variable is continuous
- **Independence** — observations are independent across and within groups
- **Normal residuals** — the residuals within each group should be approximately normal (ANOVA is robust to moderate violations with n ≥ 20 per group)
- **Does NOT assume equal variances** — the key advantage over standard one-way ANOVA
- **No extreme outliers** — outliers can distort group means and variances""",
    interpretation="""- If **p < α**, reject H₀: at least one group mean is significantly different from the others
- Report: **F_w**(df₁, df₂) = value, **p** = p-value, effect size (**η²**, **ω²**)
- Unlike standard ANOVA, Welch's uses adjusted denominator degrees of freedom — they may not be integers
- Follow significant results with **Games-Howell** post-hoc tests for pairwise comparisons
- Welch's ANOVA is recommended as the default over standard one-way ANOVA, even when variances appear equal""",
    realworld_apps="""- **Clinical trials**: comparing treatment outcomes when variance differs across dose groups
- **Education**: comparing test scores across schools with different variability
- **Manufacturing**: comparing product quality across production lines with different consistency levels
- **Psychology**: comparing cognitive performance across age groups with different variability (e.g., older vs younger adults)""",
)

hotelling_s_t_squared = TestDefinition(
    name="Hotelling's T-Squared",
    objective="Comparison",
    dependent_var="Multiple Continuous",
    independent_var="Binary/Dichotomous",
    groups="2",
    relation="Independent",
    distribution="Normal",
    explanation="""Hotelling's T² is a multivariate generalization of the independent two-sample t-test. It tests whether the mean vectors of two groups differ significantly across multiple dependent variables simultaneously. It accounts for correlations among the dependent variables and provides a single omnibus test. The test statistic follows an F-distribution (after transformation) with p and n₁+n₂-p-1 degrees of freedom, where p is the number of dependent variables.""",
    example="""A researcher measures three biomarkers (e.g., blood pressure, cholesterol, glucose) in a treatment and placebo group. Rather than conducting three separate t-tests (which inflates Type I error), Hotelling's T² tests whether the entire vector of biomarker means differs between groups simultaneously.""",
    formula="""
                    **Hotelling's T² statistic:**
                    $$ T^2 = \\frac{n_1 n_2}{n_1 + n_2} (\\bar{\\mathbf{x}}_1 - \\bar{\\mathbf{x}}_2)^T \\mathbf{S}_p^{-1} (\\bar{\\mathbf{x}}_1 - \\bar{\\mathbf{x}}_2) $$
                    Where :orange[$\\bar{\\mathbf{x}}_1, \\bar{\\mathbf{x}}_2$] are the sample mean vectors, :orange[$\\mathbf{S}_p$] is the pooled covariance matrix, and :orange[$n_1, n_2$] are group sizes.
                    
                    **Converted to F-statistic:**
                    $$ F = \\frac{n_1 + n_2 - p - 1}{p (n_1 + n_2 - 2)} T^2 $$
                    Follows :orange[$F(p, n_1 + n_2 - p - 1)$] under H₀.
                    """,
    decision_rules="""
                    - Reject :orange[$H_0: \\boldsymbol{\\mu}_1 = \\boldsymbol{\\mu}_2$] (both groups have the same mean vector) if p-value < :orange[$\\alpha$].
                    - The test requires :orange[$n_1 + n_2 > p$] (more observations than variables) and each group must have more observations than variables.
                    - After a significant result, examine univariate t-tests or discriminant function coefficients to identify which variables drive the difference.
                    - Report effect size: Mahalanobis distance :orange[$D^2$] or partial :orange[$\\eta^2$].
                    - Hotelling's T² is sensitive to multivariate outliers and non-normality.
        """,
    core_assumptions="""- **Multiple continuous DVs** — at least 2 dependent variables measured on continuous scales
- **Independence** — observations between groups are independent
- **Multivariate normality** — the data within each group should follow a multivariate normal distribution
- **Equal covariance matrices** — the two groups share a common covariance matrix (Box's M test can verify this; Welch's adaptation exists if violated)
- **No multivariate outliers** — Mahalanobis distance can detect these
- **Adequate sample size** — n₁ + n₂ > p (observations > variables)""",
    interpretation="""- If **p < α**, reject H₀: the mean vectors differ significantly between groups
- Report: **T²** = value, **F**(p, n₁+n₂-p-1) = value, **p** = p-value
- Follow up with separate univariate t-tests (with Bonferroni correction) to identify which variables differ
- Alternatively, examine standardized discriminant function coefficients to understand variable contributions
- Effect size: Mahalanobis D² or partial η² — interpret as multivariate distance between groups
- Always check the Box's M test result for covariance homogeneity""",
    realworld_apps="""- **Medical research**: comparing multiple biomarkers between disease and healthy groups
- **Marketing**: comparing brand perceptions across multiple dimensions between two segments
- **Psychology**: comparing cognitive test batteries between clinical and control groups
- **Genomics**: comparing gene expression profiles across multiple genes between two conditions
- **Finance**: comparing financial ratios between profitable and non-profitable firms""",
)

two_sample_kolmogorov_smirnov_test = TestDefinition(
    name="Two-Sample Kolmogorov-Smirnov Test",
    objective="Comparison",
    dependent_var="Continuous",
    independent_var="Binary/Dichotomous",
    groups="2",
    relation="Independent",
    distribution="any",
    explanation="""The Two-Sample Kolmogorov-Smirnov (K-S) Test is a non-parametric test that compares the cumulative distribution functions (CDFs) of two independent samples. It is sensitive to any difference between the two distributions — location (median), spread (variance), shape (skewness), or modality. The null hypothesis is that both samples come from the same population distribution. The test statistic D is the maximum absolute difference between the two empirical CDFs.""",
    example="""A researcher wants to compare the distribution of income between two cities. They use the two-sample K-S test to determine whether the distributions differ in any way (not just central tendency).""",
    formula="""
                    **Test statistic:**
                    $$ D = \\max_x |F_1(x) - F_2(x)| $$
                    Where :orange[$F_1(x)$] and :orange[$F_2(x)$] are the empirical cumulative distribution functions of the two samples.
                    
                    For large samples, the critical value at significance level :orange[$\\alpha$] is:
                    $$ D_{\\alpha} = c(\\alpha) \\sqrt{\\frac{n_1 + n_2}{n_1 n_2}} $$
                    Where :orange[$c(0.05) = 1.36$] and :orange[$c(0.01) = 1.63$].
                    """,
    decision_rules="""
                    - Reject :orange[$H_0: F_1(x) = F_2(x)$] for all :orange[$x$] if p-value < :orange[$\\alpha$] or if :orange[$D > D_{\\alpha}$].
                    - The K-S test is **distribution-free** — it does not assume any specific underlying distribution.
                    - It is sensitive to all types of distributional differences: central tendency, dispersion, skewness, and even multimodality.
                    - The test is most powerful near the center of the distributions and less sensitive to differences in the tails.
                    - For detecting only location shifts, the Mann-Whitney U test may be more powerful.
                    - Effect size: :orange[$D$] itself is the maximum absolute difference between CDFs, ranging from 0 to 1.
        """,
    core_assumptions="""- **Continuous DV** — the variable is measured on a continuous scale (ties reduce power slightly)
- **Independence** — observations are independent both within and between samples
- **Two independent groups** — samples are not paired or otherwise dependent
- **No distributional assumptions** — completely non-parametric, works for any distribution
- **Random sampling** — samples should be representative of the populations""",
    interpretation="""- If **p < α**, reject H₀: the two samples come from different population distributions
- Report: **D** = test statistic (max CDF difference), **p** = p-value
- The test detects ANY distributional difference — median, variance, shape, etc.
- Plot the two empirical CDFs to visualize where the difference occurs (the D statistic)
- If you only care about location (median), Mann-Whitney is more powerful
- Available as an assumption check for Mann-Whitney U (to verify equal distribution shapes)""",
    realworld_apps="""- **Environmental monitoring**: comparing pollutant distributions between two locations
- **Quality control**: comparing product specifications from two manufacturing processes
- **Finance**: comparing return distributions between two investment strategies
- **Medicine**: comparing distribution of a biomarker between healthy and diseased populations
- **Social science**: comparing income distributions between two demographic groups""",
)

jonckheere_terpstra_test = TestDefinition(
    name="Jonckheere-Terpstra Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var="Categorical",
    groups="More than 2",
    relation="Independent",
    distribution="Non-normal",
    explanation="""The Jonckheere-Terpstra Test is a non-parametric test for ordered differences across k independent groups. It is more powerful than the Kruskal-Wallis test when there is a natural ordering among the groups (e.g., low dose, medium dose, high dose). The null hypothesis is that all groups come from the same population. The alternative hypothesis is that there is a systematic increasing (or decreasing) trend across the groups in the order specified. The test statistic J counts the number of concordant pairs across adjacent groups.""",
    example="""A researcher tests the effect of three increasing doses of a drug (10mg, 20mg, 40mg) on pain scores. Since there is a natural ordering by dose level, the Jonckheere-Terpstra Test is chosen over Kruskal-Wallis for its superior power in detecting dose-response trends.""",
    formula="""
                    **Test statistic:**
                    $$ J = \\sum_{i < j} U_{ij} $$
                    Where :orange[$U_{ij}$] is the Mann-Whitney statistic comparing group :orange[$i$] and group :orange[$j$] (with :orange[$i < j$] in the hypothesized order).
                    
                    **Standardized test statistic:**
                    $$ Z = \\frac{J - E[J]}{\\sqrt{Var(J)}} $$
                    
                    Where :orange[$E[J] = \\frac{N^2 - \\sum n_k^2}{4}$] and :orange[$Var(J)$] is computed under H₀.
                    """,
    decision_rules="""
                    - Reject :orange[$H_0$] (no trend) if p-value < :orange[$\\alpha$], concluding the groups follow the specified monotonic trend.
                    - The alternative can be **directed** (:orange[$M_1 \\leq M_2 \\leq \\dots \\leq M_k$] with at least one strict inequality) or two-sided.
                    - The test requires that groups are specified in the hypothesized order before examining the data.
                    - Effect size: :orange[$\\hat{\\tau} = \\frac{2J}{N^2 - \\sum n_k^2} - 1$], ranging from -1 to +1.
                    - Post-hoc: pairwise Mann-Whitney tests with Bonferroni correction, or examine the pattern of medians.
        """,
    core_assumptions="""- **Ordinal or continuous DV** — the outcome is at least ordinal
- **Independence** — observations are independent across and within groups
- **Ordered groups** — the independent variable must have a natural, meaningful order (specified before analysis)
- **Monotonic trend** — the alternative is that the response tends to increase (or decrease) across the ordered groups
- **Similar distribution shapes** — for the test to detect primarily location shifts rather than shape differences""",
    interpretation="""- If **p < α**, reject H₀: there is a statistically significant monotonic trend across the ordered groups
- Report: **J** = test statistic (or **Z** = standardized statistic), **p** = p-value
- Specify whether the trend is increasing or decreasing (one-sided) or any trend (two-sided)
- The standardized Z statistic is approximately normal for moderate-to-large sample sizes
- The Jonckheere-Terpstra Test is **more powerful** than Kruskal-Wallis when groups have a natural order
- Effect size τ ranges from -1 (perfect decreasing trend) to +1 (perfect increasing trend)""",
    realworld_apps="""- **Dose-response studies**: testing for increasing effect across increasing dose levels
- **Education**: comparing performance across grade levels (3rd, 4th, 5th grade)
- **Clinical research**: testing improvement across treatment stages (baseline, 1-month, 3-month)
- **Economics**: comparing income across education levels (high school, bachelor's, master's, PhD)
- **Psychology**: comparing stress levels across increasing workload conditions""",
)

page_test = TestDefinition(
    name="Page Test",
    objective="Comparison",
    dependent_var=["Ordinal", "Continuous"],
    independent_var="Categorical",
    groups="More than 2",
    relation="Dependent",
    distribution="Non-normal",
    explanation="""The Page Test is a non-parametric test for ordered differences across k related (matched or repeated) groups. It is an extension of the Friedman Test specifically designed for ordered alternatives. When the treatments or conditions have a natural order (e.g., time points, increasing doses), the Page Test has greater statistical power than the Friedman Test because it incorporates the ordering information. The test uses the agreement between the hypothesized order and the observed ranks within each block.""",
    example="""A researcher measures patient pain scores at three time points: baseline, 1 week, and 4 weeks after treatment. Since the researcher expects pain to decrease monotonically over time, the Page Test (which incorporates this ordering) is preferred over the Friedman Test.""",
    formula="""
                    **Test statistic:**
                    $$ L = \\sum_{j=1}^k j \\cdot R_j $$
                    Where :orange[$R_j$] is the sum of ranks assigned to condition :orange[$j$] across all :orange[$n$] subjects, and conditions are ordered :orange[$1, 2, \\dots, k$] in the hypothesized direction.
                    
                    **Standardized test statistic:**
                    $$ Z = \\frac{12L - 3n k(k+1)^2}{\\sqrt{n k(k+1)(k^2-1)}} $$
                    
                    Under :orange[$H_0$], the standardized :orange[$L$] follows a standard normal distribution for moderate sample sizes.
                    """,
    decision_rules="""
                    - Reject :orange[$H_0$] (no ordered difference) if p-value < :orange[$\\alpha$], concluding the conditions follow the specified monotonic order.
                    - The alternative is **directed**: :orange[$\\theta_1 \\leq \\theta_2 \\leq \\dots \\leq \\theta_k$] with at least one strict inequality.
                    - The ordering must be specified **a priori** (before looking at the data).
                    - The Page Test has **greater power** than Friedman when the ordered alternative holds.
                    - Effect size: :orange[$\\rho = \\frac{12L - 3nk(k+1)^2}{n(k^3 - k)}$], ranging from -1 to +1.
        """,
    core_assumptions="""- **Ordinal or continuous DV** — the outcome is at least ordinal
- **Related groups** — the same subjects are measured under all :orange[$k$] conditions (matched or repeated measures)
- **a priori ordering** — the conditions must have a natural order specified before analysis
- **Monotonic trend** — the alternative is that measurements increase (or decrease) across the ordered conditions
- **No extreme outliers** — outliers can distort rank assignments
- **Complete blocks** — each subject provides data for all conditions""",
    interpretation="""- If **p < α**, reject H₀: there is a statistically significant monotonic trend across the ordered repeated measures
- Report: **L** = test statistic (or **Z** = standardized statistic), **p** = p-value
- Specify the hypothesized direction (increasing or decreasing)
- The Page Test is **more powerful** than Friedman when there is a natural ordering and a directional hypothesis
- The standardized Z can be used for one-sided testing
- Effect size ρ (Kendall coefficient of agreement) indicates the strength of the trend""",
    realworld_apps="""- **Clinical trials**: tracking improvement across treatment time points
- **Psychology**: measuring learning curves across repeated test sessions
- **Education**: assessing skill development across increasing grade levels (within same students)
- **Ergonomics**: comparing task performance across different work-rest schedules in ordered sequence
- **Sensory science**: evaluating preference ratings across increasing concentrations of a flavor""",
)

mann_kendall_trend_test = TestDefinition(
    name="Mann-Kendall Trend Test",
    objective="Association/Correlation",
    dependent_var="Continuous",
    independent_var="None",
    groups="1",
    relation="Dependent",
    distribution="any",
    explanation="""The Mann-Kendall Trend Test is a non-parametric test for detecting monotonic trends in time series data. It tests whether the values tend to consistently increase or decrease over time (or another ordered index). The test statistic S is based on the sum of the signs of differences between all pairs of observations. It does not assume any specific distribution and is robust to outliers. The null hypothesis is that there is no monotonic trend.""",
    example="""An environmental scientist analyzes 20 years of annual temperature readings to determine if there is a significant increasing trend. The Mann-Kendall test is chosen because it makes no distributional assumptions and handles missing values well.""",
    formula="""
                    **Test statistic:**
                    $$ S = \\sum_{i=1}^{n-1} \\sum_{j=i+1}^n \\text{sgn}(x_j - x_i) $$
                    Where :orange[$\\text{sgn}(x_j - x_i) = 1$] if :orange[$x_j > x_i$], :orange[$0$] if :orange[$x_j = x_i$], and :orange[$-1$] if :orange[$x_j < x_i$].
                    
                    **Variance of S (no ties):**
                    $$ Var(S) = \\frac{n(n-1)(2n+5)}{18} $$
                    
                    **Standardized test statistic:**
                    $$ Z = \\frac{S - \\text{sgn}(S)}{\\sqrt{Var(S)}} $$
                    
                    **Sen's slope (trend magnitude):**
                    $$ \\beta = \\text{median}\\left(\\frac{x_j - x_i}{j - i}\\right) \\ \\text{for all } i < j $$
                    """,
    decision_rules="""
                    - Reject :orange[$H_0$] (no monotonic trend) if p-value < :orange[$\\alpha$].
                    - Positive :orange[$S$] (or :orange[$Z$]) indicates an increasing trend; negative indicates decreasing trend.
                    - The test detects **monotonic** (not necessarily linear) trends.
                    - **Sen's slope** provides the estimated magnitude of the trend (median slope).
                    - The test assumes the observations are ordered (typically by time).
                    - The test is valid even with missing values (pairwise deletion).
                    - Seasonal Mann-Kendall handles seasonal patterns by stratifying by season.
        """,
    core_assumptions="""- **Continuous DV** — the variable is measured on a continuous scale
- **Time-ordered** — observations are ordered by time (or another index)
- **Independence between pairs** — the test is based on pairwise comparisons; no additional independence structure
- **No seasonal cycles** — for data with seasonal patterns, use Seasonal Mann-Kendall instead
- **Monotonic trend** — the test detects monotonic trends (not necessarily linear)
- **No distributional assumptions** — completely non-parametric, robust to outliers""",
    interpretation="""- If **p < α**, reject H₀: there is a statistically significant monotonic trend over time
- Report: **S** = test statistic (or **Z** = standardized), **p** = p-value
- Positive S = increasing trend; Negative S = decreasing trend
- **Sen's slope** β gives the magnitude of change per time unit (e.g., °C per year)
- Plot the data with the Sen's slope trend line overlaid
- The Mann-Kendall test does NOT estimate the trend shape — only confirms its presence
- Use Seasonal Mann-Kendall for cyclical data (e.g., monthly environmental data)""",
    realworld_apps="""- **Climate science**: detecting warming trends in temperature records
- **Hydrology**: analyzing trends in rainfall, river flow, or groundwater levels
- **Ecology**: monitoring population trends over time
- **Air quality**: tracking pollutant concentration trends
- **Economics**: analyzing long-term economic indicator trends
- **Medicine**: monitoring disease incidence rates over calendar years""",
)

goodness_of_fit_g_test = TestDefinition(
    name="Goodness-of-Fit G-Test",
    objective="Comparison",
    dependent_var="Categorical",
    independent_var="None",
    groups="1",
    relation="any",
    distribution="any",
    explanation="""The Likelihood Ratio Test (G-test) is an alternative to the chi-square goodness-of-fit test for categorical data. It tests whether observed frequencies match expected frequencies under a specified null hypothesis. The G-test is based on the log-likelihood ratio statistic G, which follows a chi-square distribution under the null hypothesis. The G-test is often preferred over the chi-square test in small samples and has better theoretical properties for nested model comparisons.""",
    example="""A geneticist tests whether pea plant phenotypes follow a 9:3:3:1 Mendelian inheritance ratio. The G-test compares observed counts to the expected Mendelian proportions.""",
    formula="""
                    **G-statistic:**
                    $$ G = 2 \\sum_{i=1}^k O_i \\ln\\left(\\frac{O_i}{E_i}\\right) $$
                    Where :orange[$O_i$] are observed frequencies, :orange[$E_i$] are expected frequencies under :orange[$H_0$], and :orange[$k$] is the number of categories.
                    
                    Under :orange[$H_0$], :orange[$G \\sim \\chi^2_{k-1-m}$], where :orange[$m$] is the number of parameters estimated from the data.
                    
                    **Relationship to chi-square:**
                    $$ G \\approx \\sum \\frac{(O_i - E_i)^2}{E_i} = \\chi^2 $$
                    The two statistics converge for large samples, but G tends to better approximate the chi-square distribution in small samples.
                    """,
    decision_rules="""
                    - Reject :orange[$H_0$] (observed frequencies match expected frequencies) if p-value < :orange[$\\alpha$].
                    - The G-test assesses how well observed counts fit the expected distribution.
                    - It is valid when expected frequencies are ≥ 5 for most categories (more robust than chi-square for smaller samples).
                    - A significant result indicates lack of fit: the data do not follow the hypothesized distribution.
                    - The G-test can also be used for **test of independence** (contingency tables), similar to chi-square.
                    - Effect size: Cramér's V (for contingency tables) or Freeman-Tukey statistic.
        """,
    core_assumptions="""- **Categorical DV** — the data consist of counts in mutually exclusive categories
- **Independence** — each observation falls into exactly one category
- **Fixed total N** — the total sample size is fixed
- **Expected frequencies** — most expected frequencies should be ≥ 5 (the G-test is more robust than chi-square for smaller expected counts)
- **Random sampling** — categories are mutually exclusive and exhaustive""",
    interpretation="""- If **p < α**, reject H₀: the observed frequencies do not fit the hypothesized distribution
- Report: **G** = test statistic, **df** = degrees of freedom, **p** = p-value
- The G-statistic is always non-negative; larger values indicate greater departure from H₀
- The G-test converges to the chi-square distribution; results are similar to Pearson's chi-square for large samples
- For small samples, the G-test is generally preferred over Pearson's chi-square
- The G-test can also be used for contingency tables (test of independence) and model comparison
- The Williams' correction (q = 1 + (k²-1)/(6n(k-1))) improves the chi-square approximation for small samples""",
    realworld_apps="""- **Genetics**: testing Mendelian inheritance ratios
- **Ecology**: comparing species abundance distributions to expected models
- **Linguistics**: testing word frequency distributions
- **Survey research**: comparing observed response distributions to expected distributions
- **Model selection**: comparing nested models in logistic regression and categorical data analysis""",
)

barnard_s_exact_test = TestDefinition(
    name="Barnard's Exact Test",
    objective="Comparison",
    dependent_var="Binary/Dichotomous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="any",
    explanation="""Barnard's Exact Test is an unconditional exact test for comparing two independent binomial proportions in a 2×2 contingency table. Unlike Fisher's Exact Test, which conditions on both margins (treats row and column totals as fixed), Barnard's test treats only one margin as fixed. This makes it an unconditional test that is generally more powerful than Fisher's Exact Test, especially for small samples. Barnard's test uses a nuisance parameter approach to maximize the p-value over all possible values of the common success probability.""",
    example="""A clinical trial compares a new drug to placebo in 20 patients (10 each). The outcome is binary (recovered/not recovered). With small expected cell counts, Barnard's Exact Test provides greater statistical power than Fisher's Exact Test.""",
    formula="""
                    **Test statistic:**
                    $$ W = \\frac{\\hat{p}_1 - \\hat{p}_2}{\\sqrt{\\bar{p}(1-\\bar{p})(1/n_1 + 1/n_2)}} $$
                    Where :orange[$\\hat{p}_1$], :orange[$\\hat{p}_2$] are observed proportions, and :orange[$\\bar{p}$] is the pooled proportion.
                    
                    **P-value calculation:**
                    $$ p = \\max_{0 \\leq \\pi \\leq 1} \\sum_{T \\geq T_0} \\binom{n_1}{x_1} \\binom{n_2}{x_2} \\pi^{x_1+x_2} (1-\\pi)^{n_1+n_2-x_1-x_2} $$
                    Where :orange[$\\pi$] is the nuisance parameter (common success probability under H₀), and :orange[$T_0$] is the observed test statistic.
                    """,
    decision_rules="""
                    - Reject :orange[$H_0: p_1 = p_2$] if p-value < :orange[$\\alpha$].
                    - Barnard's test is **unconditional** — it does not condition on both margins being fixed.
                    - It is generally **more powerful** than Fisher's Exact Test, especially for small and unbalanced samples.
                    - The p-value is computed by maximizing over the nuisance parameter :orange[$\\pi$], which makes it computationally intensive.
                    - For 2×2 tables with expected cell counts < 5, Barnard's test is recommended over both Fisher's and the chi-square test.
                    - Effect size: risk difference :orange[$\\hat{p}_1 - \\hat{p}_2$] or odds ratio with confidence interval.
        """,
    core_assumptions="""- **Binary DV** — the outcome is binary (success/failure, yes/no)
- **Two independent groups** — each subject belongs to exactly one of two groups
- **Independence** — observations are independent within and between groups
- **Random sampling** — subjects are randomly assigned or sampled
- **Unconditional test** — only one margin (typically column totals) is considered fixed
- **Small samples** — Barnard's test is particularly suitable when expected counts are small""",
    interpretation="""- If **p < α**, reject H₀: the two groups have significantly different success probabilities
- Report: **W** = test statistic, **p** = p-value (nuisance-parameter maximized)
- The p-value is computed by maximizing over all possible values of the common success probability under H₀
- Barnard's test is computationally more expensive than Fisher's but generally provides greater statistical power
- The test is available in the `scipy.stats.barnard_exact` function (SciPy ≥ 1.11)
- Report effect size as the risk difference or odds ratio with a 95% confidence interval
- Barnard's test is preferred over Fisher's for small 2×2 tables with at least one cell < 5""",
    realworld_apps="""- **Clinical trials**: comparing response rates between small treatment and control groups
- **Medical research**: analyzing rare events in small samples
- **Epidemiology**: analyzing case-control studies with small sample sizes
- **Product testing**: comparing defect rates between two manufacturing processes with limited data
- **A/B testing**: comparing conversion rates when sample sizes are small""",
)

boschloo_s_exact_test = TestDefinition(
    name="Boschloo's Exact Test",
    objective="Comparison",
    dependent_var="Binary/Dichotomous",
    independent_var=["Binary/Dichotomous", "Categorical"],
    groups="2",
    relation="Independent",
    distribution="any",
    explanation="""Boschloo's Exact Test is an unconditional exact test for 2×2 contingency tables that is uniformly more powerful than Fisher's Exact Test. Like Barnard's test, it treats only one margin as fixed (unconditional), but uses the p-value from Fisher's Exact Test as the test statistic. Boschloo's test then maximizes the p-value over the nuisance parameter (common success proportion). This approach is known as an 'exact unconditional test with the Berger-Boos correction'. Boschloo's test is generally recommended as the default exact test for 2×2 tables when power is a concern.""",
    example="""A researcher tests whether a new training program improves pass rates. With only 12 participants in each group, the expected cell counts are small. Boschloo's Exact Test is chosen because it offers greater power than Fisher's Exact Test while maintaining exact Type I error control.""",
    formula="""
                    **Test statistic:**
                    $$ T = \\text{p-value from Fisher's Exact Test (one-sided)} $$
                    
                    **P-value calculation:**
                    $$ p_{\\text{Boschloo}} = \\max_{0 \\leq \\pi \\leq 1} \\sum_{T \\leq T_0} P(\\mathbf{X} = \\mathbf{x} \\mid \\pi) $$
                    Where :orange[$P(\\mathbf{X} = \\mathbf{x} \\mid \\pi)$] is the probability of the observed table under the unconditional product-binomial model with common success probability :orange[$\\pi$].
                    
                    **Berger-Boos correction:**
                    Instead of maximizing over the entire interval :orange[$[0, 1]$], the optimization is restricted to :orange[$[\\hat{\\pi}_L, \\hat{\\pi}_U]$], which improves computational efficiency.
                    """,
    decision_rules="""
                    - Reject :orange[$H_0: p_1 = p_2$] if p-value < :orange[$\\alpha$].
                    - Boschloo's test is **uniformly more powerful** than Fisher's Exact Test.
                    - It is **unconditional** — does not condition on both margins being fixed.
                    - The test statistic :orange[$T$] is Fisher's p-value itself.
                    - Boschloo's test is computationally intensive but feasible for small-to-moderate sample sizes.
                    - It maintains exact Type I error control (guarantees :orange[$\\alpha$] level).
                    - Effect size: odds ratio, risk difference, or relative risk with confidence interval.
        """,
    core_assumptions="""- **Binary DV** — the outcome is binary (success/failure)
- **Two independent groups** — each subject belongs to one of two independent groups
- **Independence** — observations are independent within and between groups
- **Unconditional test** — only column totals are fixed (rows are not conditioned on)
- **Small samples** — particularly suitable when expected cell counts are small
- **Exact test** — Boschloo's test guarantees exact Type I error control (results are not asymptotic)""",
    interpretation="""- If **p < α**, reject H₀: the two groups have significantly different success probabilities
- Report: test statistic = Fisher's p-value used as statistic, **p_{\\text{Boschloo}}** = Boschloo p-value
- Boschloo's test is **uniformly more powerful** than Fisher's Exact Test
- The null hypothesis is the same as Fisher's: independence of rows and columns in the 2×2 table
- Boschloo's test is computationally intensive; for large samples, the chi-square or two-proportion z-test is sufficient
- Available in `scipy.stats.boschloo_exact` (SciPy ≥ 1.9.0)
- Report effect size as odds ratio or risk difference with 95% CI""",
    realworld_apps="""- **Clinical trials**: comparing adverse event rates in small treatment groups
- **Epidemiology**: analyzing small case-control studies
- **Pre-clinical research**: comparing response rates in animal studies with few subjects
- **Educational research**: comparing pass/fail rates in small class pilot studies
- **Usability testing**: comparing success rates between two interface designs with few participants""",
)

stuart_maxwell_test = TestDefinition(
    name="Stuart-Maxwell Test",
    objective="Comparison",
    dependent_var="Categorical",
    independent_var="Categorical",
    groups="2",
    relation="Dependent",
    distribution="any",
    explanation="""The Stuart-Maxwell Test is an extension of McNemar's Test for paired categorical data with more than two categories. While McNemar's test handles binary (2×2) paired data, the Stuart-Maxwell test handles k×k tables where k ≥ 3. It tests whether the marginal distributions of two matched categorical variables are equal (marginal homogeneity). The test is often used for pretest-posttest designs with multi-category outcomes or for comparing two raters' classifications on a nominal scale with multiple categories.""",
    example="""A researcher classifies patients' disease severity as Mild, Moderate, or Severe at baseline and after 6 months of treatment. The Stuart-Maxwell Test tests whether the distribution of severity ratings has changed from baseline to follow-up.""",
    formula="""
                    **Test statistic:**
                    $$ \\chi^2 = \\mathbf{d}^T \\mathbf{V}^{-1} \\mathbf{d} $$
                    Where :orange[$\\mathbf{d}$] is the vector of differences between marginal frequencies (:orange[$d_i = n_{i\\bullet} - n_{\\bullet i}$]), and :orange[$\\mathbf{V}$] is the estimated variance-covariance matrix of :orange[$\\mathbf{d}$].
                    
                    For a :orange[$k \\times k$] table, the test statistic follows :orange[$\\chi^2_{k-1}$] under :orange[$H_0$].
                    
                    **Simplified formula (for k=3):**
                    $$ \\chi^2 = \\frac{n_{23}d_1^2 + n_{13}d_2^2 + n_{12}d_3^2}{2(n_{12}n_{13} + n_{12}n_{23} + n_{13}n_{23})} $$
                    Where :orange[$n_{ij}$] are cell counts and :orange[$d_i = n_{i\\bullet} - n_{\\bullet i}$] are the marginal differences.
                    """,
    decision_rules="""
                    - Reject :orange[$H_0$] (marginal homogeneity) if p-value < :orange[$\\alpha$].
                    - A significant result indicates that the row and column distributions differ systematically.
                    - The test assesses **global** marginal homogeneity across all categories simultaneously.
                    - After a significant result, examine individual categories using McNemar's tests with Bonferroni correction.
                    - The test is appropriate for :orange[$k \\times k$] tables (same categories for rows and columns).
                    - The test requires paired/dependent observations.
        """,
    core_assumptions="""- **Categorical DV** — the outcome is categorical with ≥ 3 categories (nominal or ordinal)
- **Paired observations** — each subject provides a response under two conditions (e.g., pretest/posttest, two raters)
- **Same categories** — both variables use the same set of categories
- **Square table** — the contingency table has equal rows and columns (:orange[$k \\times k$])
- **Independence between subjects** — subjects are independent of each other
- **Adequate sample size** — the variance-covariance matrix must be invertible (no zero off-diagonal pairs)""",
    interpretation="""- If **p < α**, reject H₀: the marginal distributions are not equal across the two conditions
- Report: **χ²**(df) = value, **p** = p-value (df = k-1)
- A significant result means the distribution of responses differs between the two matched conditions
- For **2×2** tables, use McNemar's Test (Stuart-Maxwell reduces to McNemar's for k=2)
- For **ordinal** categories with more than 2 levels and a directional hypothesis, consider the Marginal Homogeneity Test (a related method)
- Follow-up: use McNemar's Test on individual categories with Bonferroni correction to identify which categories changed
- The test is also known as the Stuart-Maxwell χ² Test for Marginal Homogeneity""",
    realworld_apps="""- **Longitudinal studies**: comparing disease severity distributions from baseline to follow-up
- **Diagnostic agreement**: comparing classification distributions between two raters
- **Survey research**: comparing response distributions before and after an intervention
- **Market research**: comparing brand preference distributions before and after an advertising campaign
- **Clinical psychology**: comparing symptom category distributions before and after therapy""",
)

gwet_s_ac1 = TestDefinition(
    name="Gwet's AC1",
    objective="Association/Correlation",
    dependent_var="Categorical",
    independent_var=["Categorical", "Binary/Dichotomous"],
    groups="2",
    relation="Independent",
    distribution="any",
    explanation="""Gwet's AC1 is an agreement coefficient for inter-rater reliability with categorical data. It addresses the 'kappa paradox' where Cohen's Kappa can be unexpectedly low even when agreement is high, due to sensitivity to marginal distributions. Gwet's AC1 uses a different chance-agreement correction that is more robust to trait prevalence and marginal asymmetry. It is recommended as a general-purpose alternative to Cohen's Kappa for two raters with nominal categories.""",
    example="""Two radiologists independently classify 100 mammogram images as either 'Normal', 'Benign', or 'Malignant'. The prevalence of 'Malignant' is low (5%). Cohen's Kappa gives a surprisingly low value due to high marginal asymmetry, so Gwet's AC1 is used instead to obtain a more interpretable agreement measure.""",
    formula="""
                    **Observed agreement:**
                    $$ p_o = \\frac{1}{n} \\sum_{i=1}^k n_{ii} $$
                    
                    **Chance agreement (Gwet's approach):**
                    $$ p_e = \\frac{1}{k-1} \\sum_{j=1}^k \\pi_j (1 - \\pi_j) $$
                    Where :orange[$\\pi_j = \\frac{n_{j\\bullet} + n_{\\bullet j}}{2n}$] is the average proportion of ratings in category :orange[$j$].
                    
                    **Gwet's AC1:**
                    $$ \\gamma_1 = \\frac{p_o - p_e}{1 - p_e} $$
                    """,
    decision_rules="""
                    - :orange[$\\gamma_1 = 1$] indicates perfect agreement; :orange[$\\gamma_1 = 0$] indicates agreement no better than chance; negative values indicate agreement worse than chance.
                    - Standard interpretation: > 0.80 = almost perfect, 0.61-0.80 = substantial, 0.41-0.60 = moderate, 0.21-0.40 = fair, ≤ 0.20 = slight.
                    - Gwet's AC1 is **not affected** by the kappa paradox (low kappa despite high agreement).
                    - It is more stable than Cohen's Kappa across different trait prevalences.
                    - Standard errors and confidence intervals are available via the delta method.
                    - The AC1 coefficient ranges from -1 (complete disagreement) to +1 (perfect agreement).
        """,
    core_assumptions="""- **Categorical DV** — ratings are in nominal categories (≥ 2 categories)
- **Two raters** — exactly two raters or measurement methods
- **Same subjects** — each rater classifies the same set of :orange[$n$] subjects
- **Independent ratings** — raters classify independently of each other
- **Fixed number of categories** — both raters use the same predefined categories
- **No missing categories** — all categories should be used by at least one rater""",
    interpretation="""- **γ₁ > 0.80**: almost perfect agreement beyond chance
- **γ₁ = 0.61-0.80**: substantial agreement
- **γ₁ = 0.41-0.60**: moderate agreement
- **γ₁ = 0.21-0.40**: fair agreement
- **γ₁ ≤ 0.20**: slight agreement
- Report: **γ₁** = value, **SE** = standard error, **95% CI** = confidence interval
- Gwet's AC1 is preferred over Cohen's Kappa when marginal distributions are asymmetric or when trait prevalence is very low or very high
- The AC1 coefficient solves the 'kappa paradox' and provides more interpretable agreement estimates
- Gwet's AC2 extends this approach to ordinal and continuous ratings""",
    realworld_apps="""- **Medical imaging**: assessing inter-radiologist agreement with rare conditions
- **Psychiatric diagnosis**: evaluating diagnostic agreement with low-prevalence disorders
- **Content analysis**: coding rare categories in text analysis
- **Quality control**: classifying defect types with a dominant non-defective category
- **Forensic science**: evaluating fingerprint or DNA analyst agreement with imbalanced categories""",
)

krippendorff_s_alpha = TestDefinition(
    name="Krippendorff's Alpha",
    objective="Association/Correlation",
    dependent_var=["Ordinal", "Continuous", "Categorical"],
    independent_var="Categorical",
    groups="More than 2",
    relation="Independent",
    distribution="any",
    explanation="""Krippendorff's Alpha is a versatile reliability coefficient that measures agreement among any number of raters, works with any number of raters per subject, handles missing data, and supports multiple measurement levels (nominal, ordinal, interval, ratio). It compares the observed disagreement to the disagreement expected under statistical independence. Alpha values range from 0 (no agreement beyond chance) to 1 (perfect agreement). A common threshold for acceptable reliability is α ≥ 0.667 for tentative conclusions and α ≥ 0.80 for firm conclusions.""",
    example="""Five content analysts code 50 news articles for political bias on a 7-point ordinal scale. Some articles are coded by only 3 of the 5 analysts due to scheduling constraints. Krippendorff's Alpha is the only reliability measure that can handle this incomplete design with mixed numbers of raters per subject.""",
    formula="""
                    **Krippendorff's Alpha:**
                    $$ \\alpha = 1 - \\frac{D_o}{D_e} $$
                    Where :orange[$D_o$] is the observed disagreement and :orange[$D_e$] is the disagreement expected under independence.
                    
                    **Observed disagreement:**
                    $$ D_o = \\frac{1}{n} \\sum_{c} \\sum_{k} o_{ck} \\cdot \\text{dist}(c,k)^2 $$
                    Where :orange[$o_{ck}$] is the frequency of pairs with values :orange[$c$] and :orange[$k$], and :orange[$\\text{dist}(c,k)^2$] is a metric difference function.
                    
                    **Metric functions:**
                    - Nominal: :orange[$\\text{dist}(c,k)^2 = 0$] if :orange[$c = k$], :orange[$1$] otherwise
                    - Ordinal: :orange[$\\text{dist}(c,k)^2 = (c-k)^2$]
                    - Interval: :orange[$\\text{dist}(c,k)^2 = (c-k)^2$]
                    - Ratio: :orange[$\\text{dist}(c,k)^2 = ((c-k)/(c+k))^2$] (for positive values)
                    """,
    decision_rules="""
                    - Reject :orange[$H_0$] (no agreement beyond chance) if the confidence interval excludes zero.
                    - :orange[$\\alpha \\geq 0.800$]: firm conclusions can be drawn from the data.
                    - :orange[$\\alpha \\geq 0.667$]: tentative conclusions are acceptable (common in content analysis).
                    - :orange[$\\alpha < 0.667$]: reliability is insufficient; improve coding procedures.
                    - Krippendorff's Alpha can handle any number of raters and any level of measurement.
                    - It naturally handles missing data (raters who did not code all subjects).
                    - Bootstrapped confidence intervals are recommended for inference.
        """,
    core_assumptions="""- **Any measurement level** — works with nominal, ordinal, interval, or ratio data
- **Multiple raters (≥ 2)** — any number of raters can be used
- **Possibly incomplete data** — not every rater needs to evaluate every subject
- **Independent ratings** — raters code independently
- **Subjects as units** — subjects (units) are independent
- **Sufficient variability** — there should be at least some variability in the data (α is undefined for constant data)""",
    interpretation="""- **α ≥ 0.800**: excellent reliability, firm conclusions supported
- **α ≥ 0.667**: acceptable for tentative conclusions
- **α < 0.667**: insufficient reliability, revise coding scheme or train raters
- Report: **α** = coefficient, **95% CI** = bootstrapped confidence interval
- Krippendorff's Alpha is the **most general** reliability coefficient available
- Choose the appropriate metric function based on the measurement level of your data
- Always report the metric function used (nominal, ordinal, interval, or ratio)
- Bootstrap confidence intervals provide more robust inference than asymptotic methods""",
    realworld_apps="""- **Content analysis**: coding themes in open-ended survey responses with multiple coders
- **Behavioral coding**: coding video-recorded behaviors with multiple observers
- **Medical records review**: extracting data from records with multiple abstractors
- **Sentiment analysis**: rating sentiment on ordinal scales across multiple annotators
- **Systematic reviews**: coding study characteristics with multiple reviewers
- **Linguistics**: transcribing and coding speech patterns with multiple annotators""",
)

intraclass_correlation_coefficient_icc = TestDefinition(
    name="Intraclass Correlation Coefficient (ICC)",
    objective="Association/Correlation",
    dependent_var="Continuous",
    independent_var="Categorical",
    groups="More than 2",
    relation="Independent",
    distribution="Normal",
    explanation="""The Intraclass Correlation Coefficient (ICC) measures the reliability or agreement of continuous measurements made by multiple raters or methods on the same subjects. It describes how strongly units within the same group resemble each other. There are multiple ICC variants for different study designs: ICC(1) for one-way random effects (each subject rated by different raters), ICC(2) for two-way random effects (same raters for all subjects, generalizable), and ICC(3) for two-way mixed effects (same raters, not generalizable). Each variant also has a 'consistency' vs 'absolute agreement' form.""",
    example="""A researcher measures the resting heart rate of 20 subjects using three different devices simultaneously. ICC is used to assess the test-retest reliability and absolute agreement among the three measurement devices.""",
    formula="""
                    **One-way random effects model (ICC(1)):**
                    $$ x_{ij} = \\mu + s_i + e_{ij} $$
                    $$ ICC(1) = \\frac{\\sigma_s^2}{\\sigma_s^2 + \\sigma_e^2} $$
                    
                    **Two-way random effects (ICC(2), absolute agreement):**
                    $$ x_{ij} = \\mu + s_i + r_j + e_{ij} $$
                    $$ ICC(2,1) = \\frac{\\sigma_s^2}{\\sigma_s^2 + \\sigma_r^2 + \\sigma_e^2} $$
                    
                    **Two-way mixed effects (ICC(3), consistency):**
                    $$ ICC(3,1) = \\frac{\\sigma_s^2}{\\sigma_s^2 + \\sigma_e^2} $$
                    
                    Where :orange[$\\sigma_s^2$] is between-subject variance, :orange[$\\sigma_r^2$] is between-rater variance, and :orange[$\\sigma_e^2$] is residual (error) variance.
                    """,
    decision_rules="""
                    - ICC ranges from 0 to 1 (or rarely negative when true reliability is near zero).
                    - Common interpretation: < 0.40 = poor, 0.40-0.59 = fair, 0.60-0.74 = good, ≥ 0.75 = excellent.
                    - Choose ICC variant based on study design: ICC(1) = different raters per subject; ICC(2) = same raters, generalizable; ICC(3) = same raters, not generalizable.
                    - Choose **consistency** if systematic rater differences are irrelevant, or **absolute agreement** if scale calibration matters.
                    - Report confidence intervals for the ICC estimate (typically bootstrapped or F-based).
                    - ICC is related to Pearson r but accounts for systematic bias and multiple raters.
        """,
    core_assumptions="""- **Continuous DV** — measurements are on a continuous scale
- **Multiple raters/methods (≥ 2)** — at least two raters, devices, or measurement occasions
- **Subjects are random** — subjects are a random sample from the target population
- **Independence between subjects** — different subjects are independent
- **Normality** — the random effects (subject and residual) are approximately normally distributed
- **Homoscedasticity** — the residual variance is approximately constant across subjects and raters
- **No extreme outliers** — outliers can inflate between-subject variance and bias ICC upward""",
    interpretation="""- **ICC < 0.40**: poor reliability
- **ICC = 0.40-0.59**: fair reliability
- **ICC = 0.60-0.74**: good reliability
- **ICC ≥ 0.75**: excellent reliability
- Report: **ICC** estimate, **95% CI** = confidence interval, **ICC type** = (1,1), (2,1), (3,1), etc.
- Specify whether the reported ICC is for **single measures** (individual rating reliability) or **average measures** (mean of k ratings)
- ICC(2,1) includes systematic rater differences in the error term → lower but more conservative estimate
- ICC(3,1) excludes systematic rater differences → higher estimate, appropriate when raters are fixed
- Always report which ICC variant was used and whether it reflects consistency or absolute agreement""",
    realworld_apps="""- **Medical devices**: comparing blood pressure readings from multiple devices
- **Imaging**: assessing inter-radiologist reliability for continuous measurements (e.g., tumor size)
- **Psychometrics**: evaluating test-retest reliability of psychological scales
- **Clinical research**: assessing inter-observer reliability for physical measurements
- **Wearable technology**: validating fitness trackers against gold-standard measurements
- **Laboratory methods**: comparing different assay methods for the same biomarker""",
)

hosmer_lemeshow_test = TestDefinition(
    name="Hosmer-Lemeshow Test",
    objective="Diagnostic Accuracy",
    dependent_var="Binary/Dichotomous",
    independent_var="Continuous",
    groups="2",
    relation="Independent",
    distribution="any",
    explanation="""The Hosmer-Lemeshow Test is a goodness-of-fit test for logistic regression models. It assesses whether the observed event rates match the expected event rates in subgroups of the model population. Observations are sorted by predicted probability and grouped into deciles (or g groups). Within each group, the observed and expected counts are compared using a Pearson chi-square statistic. A non-significant result (p > 0.05) indicates the model fits the data well. The test is sensitive to the number of groups chosen and has relatively low power for detecting certain types of model misspecification.""",
    example="""A researcher fits a logistic regression model to predict hospital readmission based on age, comorbidities, and lab values. The Hosmer-Lemeshow test is run to check whether the predicted probabilities match observed readmission rates across risk deciles.""",
    formula="""
                    **Hosmer-Lemeshow C-statistic:**
                    $$ \\hat{C} = \\sum_{g=1}^G \\frac{(O_g - E_g)^2}{E_g (1 - E_g / n_g)} $$
                    Where :orange[$G$] is the number of groups (typically 10, corresponding to deciles of predicted risk), :orange[$O_g$] is the observed number of events in group :orange[$g$], :orange[$E_g$] is the sum of predicted probabilities (expected events) in group :orange[$g$], and :orange[$n_g$] is the number of observations in group :orange[$g$].
                    
                    Under :orange[$H_0$] (the model fits well), :orange[$\\hat{C} \\sim \\chi^2_{G-2}$].
                    """,
    decision_rules="""
                    - **Fail to reject** :orange[$H_0$] (p > 0.05) indicates the model fits adequately — the test does NOT guarantee good fit, but suggests no gross misspecification.
                    - **Reject** :orange[$H_0$] (p < 0.05) indicates significant lack of fit — consider model revisions (interaction terms, non-linear terms, alternative link function).
                    - The test has **low power** — a non-significant result does not guarantee good calibration.
                    - The test is sensitive to the choice of groups :orange[$G$]; 10 groups (deciles) is standard.
                    - The test is known to be **conservative** and may fail to detect certain types of poor calibration.
                    - Do not rely solely on the Hosmer-Lemeshow test; also examine calibration plots and other fit diagnostics.
        """,
    core_assumptions="""- **Binary DV** — the outcome is binary (event/non-event)
- **Logistic regression model** — the test evaluates a fitted logistic regression model
- **Independent observations** — subjects are independent
- **Adequate sample size** — the test has low power for small samples; recommended n ≥ 100
- **Groups based on predicted probabilities** — observations grouped by deciles of predicted risk
- **The test is a model diagnostic** — it does not test a substantive hypothesis about the data itself""",
    interpretation="""- **p > 0.05**: insufficient evidence of poor fit; the model's predicted probabilities appear adequately calibrated across risk strata
- **p ≤ 0.05**: evidence of poor calibration; the model does not adequately fit the data
- Report: **Ĉ**(df) = value, **p** = p-value, **G** = number of groups
- A significant Hosmer-Lemeshow test suggests the model lacks calibration — consider:
  - Adding interaction terms or nonlinear terms (polynomials, splines)
  - Changing the link function
  - Examining influential observations
- The Hosmer-Lemeshow test should be used alongside other diagnostics:
  - **Calibration plot** (observed vs predicted proportions)
  - **ROC curve** (discrimination, not calibration)
  - **Brier score** (overall prediction error)
- The test is most informative when the sample is large enough to have stable groups (≥ 50 per decile)""",
    realworld_apps="""- **Medical risk prediction**: evaluating the calibration of hospital readmission risk models
- **Credit scoring**: testing whether predicted default probabilities match observed default rates
- **Epidemiology**: assessing disease risk prediction models
- **Marketing**: validating customer churn prediction models
- **Public health**: evaluating predictive models for disease outbreak risk
- **Clinical decision support**: validating risk calculators before clinical deployment""",
)

ALL_TESTS: list[TestDefinition] = [
    one_sample_t_test,
    one_sample_z_test,
    one_sample_proportion_test_binomial_test,
    binomial_test,
    sign_test_one_sample,
    one_sample_wilcoxon_signed_rank_test,
    chi_square_goodness_of_fit_test,
    multinomial_test,
    runs_test_for_randomness,
    poisson_goodness_of_fit_test,
    student_s_t_test_independent,
    welch_s_t_test_independent_unequal_variances,
    f_test_for_two_variances,
    equivalence_test_tost_two_independent_samples,
    paired_t_test,
    one_way_anova,
    repeated_measures_anova,
    two_way_anova,
    ancova,
    manova,
    sign_test_paired,
    wilcoxon_signed_rank_test,
    mann_whitney_u_test,
    mood_s_median_test,
    kruskal_wallis_test,
    friedman_test,
    permutation_manova_or_non_parametric_manova,
    chi_square_test,
    mcnemar_s_test,
    cochran_s_q_test,
    fisher_s_exact_test,
    pearson_correlation,
    spearman_rank_correlation,
    chi_square_test_of_independence,
    point_biserial_correlation,
    kendall_s_tau_b,
    simple_linear_regression,
    multiple_linear_regression,
    logistic_regression,
    multinomial_logistic_regression,
    ordinal_logistic_regression,
    poisson_regression,
    negative_binomial_regression,
    sensitivity_specificity_analysis,
    roc_curve_analysis,
    likelihood_ratio_analysis,
    cohen_s_kappa_agreement_analysis,
    bland_altman_analysis,
    weighted_kappa,
    fleiss_kappa,
    kaplan_meier_survival_analysis,
    cox_proportional_hazards_regression,
    log_rank_test,
    cochran_mantel_haenszel_test,
    two_sample_proportion_test,
    yuen_s_trimmed_t_test,
    brunner_munzel_test,
    one_way_welch_anova,
    hotelling_s_t_squared,
    two_sample_kolmogorov_smirnov_test,
    jonckheere_terpstra_test,
    page_test,
    mann_kendall_trend_test,
    goodness_of_fit_g_test,
    barnard_s_exact_test,
    boschloo_s_exact_test,
    stuart_maxwell_test,
    gwet_s_ac1,
    krippendorff_s_alpha,
    intraclass_correlation_coefficient_icc,
    hosmer_lemeshow_test,
]


def get_test(name: str) -> TestDefinition | None:
    for t in ALL_TESTS:
        if t.name == name:
            return t
    return None
