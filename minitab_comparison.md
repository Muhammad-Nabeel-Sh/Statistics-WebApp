# Feature Comparison: Statistics WebApp vs Minitab Statistical Software

> **Our App** = Statistics WebApp (free, web-based, educational)
> **Minitab** = Commercial statistical software (paid license, desktop)

---

## 1. Hypothesis Testing

| Feature | Our App | Minitab |
|---------|---------|---------|
| One-sample t-test | ✅ Interactive widget, sliders, plot, effect sizes, CI, Cohen's d, interpretations | ✅ Basic Statistics menu |
| One-sample z-test | ✅ Interactive widget, known SD | ✅ Basic Statistics |
| One-sample proportion (Binomial) | ✅ Normal approx + Exact versions | ✅ 1 Proportion |
| One-sample Poisson rate | ⚠ Listed as compatible test, no dedicated widget | ✅ 1-Sample Poisson Rate |
| One-sample Wilcoxon Signed-Rank | ✅ Interactive widget | ✅ Nonparametrics menu |
| Sign Test (One-sample) | ✅ Interactive widget | ✅ Nonparametrics menu |
| **Two-sample independent t-test** | ✅ Student's + Welch's side-by-side, group names, n, Mean, SD, SE, df, CI | ✅ Basic Statistics |
| Two-sample proportion test | ⚠ Not available | ✅ 2 Proportions |
| F-Test for Two Variances | ✅ Interactive widget | ✅ Basic Statistics (2 Variances) |
| Mann-Whitney U Test | ✅ Interactive widget | ✅ Nonparametrics menu |
| Equivalence Test (TOST) | ✅ Two independent samples | ✅ Equivalence Tests menu (one/two sample, paired, 2x2 crossover) |
| **Paired t-test** | ✅ Interactive widget | ✅ Basic Statistics |
| Wilcoxon Signed-Rank (Paired) | ✅ Interactive widget | ✅ Nonparametrics menu |
| Sign Test (Paired) | ✅ Interactive widget | ✅ Nonparametrics menu |
| McNemar's Test | ✅ Interactive widget | ✅ Tables menu |
| **One-way ANOVA** | ✅ Interactive widget, post-hoc, eta-squared | ✅ ANOVA menu |
| Two-way ANOVA | ✅ Interactive widget | ✅ ANOVA menu |
| ANCOVA | ✅ Interactive widget | ✅ ANOVA (GLM) |
| Kruskal-Wallis Test | ✅ Interactive widget, post-hoc | ✅ Nonparametrics menu |
| Mood's Median Test | ✅ Interactive widget | ✅ Nonparametrics menu |
| Repeated Measures ANOVA | ✅ Interactive widget | ✅ ANOVA (GLM) |
| Friedman Test | ✅ Interactive widget, post-hoc | ✅ Nonparametrics menu |
| Cochran's Q Test | ✅ Interactive widget | ⚠ Not listed |
| Permutation MANOVA | ✅ Interactive widget | ⚠ Not listed |
| **Chi-Square GOF** | ✅ Interactive widget | ✅ Tables menu |
| Multinomial Test | ✅ Interactive widget | ⚠ Not listed |
| Runs Test for Randomness | ✅ Interactive widget | ✅ Nonparametrics menu |
| Poisson GOF Test | ✅ Interactive widget | ✅ Basic Statistics |
| Chi-Square Independence | ✅ Interactive widget | ✅ Tables menu |
| Fisher's Exact Test | ✅ Interactive widget | ✅ Tables (via Chi-Square) |
| Cohen's Kappa | ✅ Interactive widget | ✅ Multivariate (Item Analysis) |
| Weighted Kappa | ✅ Interactive widget | ⚠ Not listed |
| Fleiss' Kappa | ✅ Interactive widget | ⚠ Not listed |
| Bland-Altman Analysis | ✅ Interactive widget | ⚠ Not listed |
| **Pearson Correlation** | ✅ Interactive widget | ✅ Basic Statistics |
| Spearman Rank Correlation | ✅ Interactive widget | ✅ Basic Statistics |
| Kendall's Tau-b | ✅ Interactive widget | ⚠ Not listed |
| Point-Biserial Correlation | ✅ Interactive widget | ⚠ Not listed |
| **Simple Linear Regression** | ✅ Interactive widget | ✅ Regression menu |
| Multiple Linear Regression | ✅ Interactive widget | ✅ Regression menu |
| Logistic Regression (Binary) | ✅ Interactive widget | ✅ Regression menu |
| Multinomial Logistic Regression | ✅ Interactive widget | ✅ Regression menu |
| Ordinal Logistic Regression | ✅ Interactive widget | ✅ Regression menu |
| Poisson Regression | ✅ Interactive widget | ✅ Regression menu |
| Negative Binomial Regression | ✅ Interactive widget | ⚠ Not listed |
| MANOVA | ✅ Interactive widget | ✅ ANOVA menu |
| **Kaplan-Meier Survival** | ✅ Interactive widget | ✅ Reliability/Survival |
| Cox Proportional Hazards | ✅ Interactive widget | ✅ Regression menu |
| Log-Rank Test | ✅ Interactive widget | ✅ Reliability/Survival |
| **Sensitivity & Specificity** | ✅ Interactive widget | ⚠ Not listed (part of MSA) |
| ROC Curve Analysis | ✅ Interactive widget | ⚠ Not listed (part of Predictive Analytics) |
| Likelihood Ratio Analysis | ✅ Interactive widget | ⚠ Not listed |
| **Outlier Test (Grubbs)** | ✅ In diagnostics | ✅ Basic Statistics |
| Normality Test (Shapiro-Wilk, etc.) | ✅ In diagnostics | ✅ Basic Statistics |

**Summary:** Our app has more interactive hypothesis test widgets (41+ vs ~40 listed in Minitab Basic + Nonparametrics + Tables). Minitab's advantage is in Poisson rate tests and the equivalence test 2x2 crossover design. Our advantage includes Cochran's Q, Permutation MANOVA, Multinomial, Weighted Kappa, Fleiss' Kappa, Bland-Altman, Kendall's Tau, Point-Biserial, Negative Binomial, and diagnostic accuracy tests — not available in base Minitab.

---

## 2. Data Visualization & Graphics

| Feature | Our App | Minitab |
|---------|---------|---------|
| Total graph types | **83** interactive widgets across 10 categories | Extensive (no exact count listed, roughly 40-50 types) |
| Interactive (sliders, real-time update) | ✅ All 83 graphs are interactive with real-time controls | ❌ Static graphs (update when data change, but no interactive parameter sliders) |
| Distribution Plots | 13 types (histogram, KDE, boxplot, violin, Q-Q, P-P, stem-leaf, frequency polygon, Pareto, dot, polar density, PDF, raincloud, population pyramid, ridgeline) | ✅ Histogram, boxplot, dotplot, probability plots, etc. |
| Comparison Plots | 12 types (grouped bar, error bar, paired line, boxplot comp, violin comp, raincloud, beeswarm, time series, pie, area, stacked bar, Cleveland dot, Sankey, Venn) | ✅ Bar, pie, time series, etc. |
| Correlation Plots | 6 types (scatter, heatmap, bubble, monotonic vs linear, SPLOM, hexbin) | ✅ Scatterplot, matrix, bubble, heatmap |
| Regression Plots | 10 types (linear, multiple surface, logistic sigmoid, multinomial boundaries, ordinal probability, Poisson count, residuals vs fitted, polynomial, regularization path, growth curve) | ✅ Residual, factorial, contour, surface, etc. |
| Diagnostic Accuracy Plots | 6 types (confusion matrix, ROC, PR curve, sensitivity-specificity threshold, calibration plot) | ⚠ Not listed |
| Agreement Plots | 3 types (Bland-Altman, Kappa matrix, ICC) | ⚠ Not listed |
| Multivariate Plots | 6 types (PCA scatter, MANOVA groups, cluster, 3D scatter, parallel coordinates, contour) | ✅ 3D plots, contour, etc. |
| Survival Analysis Plots | 7 types (KM curve, Nelson-Aalen, hazard, cumulative hazard, Cox effect forest, survival heatmap) | ✅ Distribution, probability, hazard, survival plots |
| Meta-Analysis Plots | 5 types (forest, funnel, Galbraith, Baujat, leave-one-out) | ❌ Not available |
| Post-Hoc Plots | 6 types (CI comparison, mean diff forest, CLD, significance heatmap, pairwise network, estimation/Gardner-Altman) | ❌ Not available (basic multiple comparisons tables only) |
| Graph Builder | ❌ Not available | ✅ Drag-and-drop Graph Builder |
| Pareto Chart | ✅ Interactive widget | ✅ Quality Tools menu |
| Control Charts | ❌ Not available | ✅ 20+ control chart types (XBar, R, S, P, NP, C, U, EWMA, CUSUM, etc.) |
| Automated graph updates | ✅ Real-time slider interaction | ✅ Updates when data change |
| Export formats | ✅ PNG (300 DPI) + SVG vector | ✅ TIF, JPEG, PNG, BMP, GIF, EMF |

**Summary:** Our app offers more graph types (83 vs ~40-50) and every single graph is **interactive** with real-time parameter sliders — a major educational advantage. Minitab wins on Graph Builder (drag-and-drop), control charts (20+ types), and export format variety. Our unique categories: meta-analysis, post-hoc plots, diagnostic accuracy, and agreement plots — not available in base Minitab.

---

## 3. Probability Distributions

| Feature | Our App | Minitab |
|---------|---------|---------|
| Discrete distributions | 7 (Bernoulli, Binomial, Poisson, Geometric, NegBinomial, Hypergeometric, Discrete Uniform) | ✅ Available |
| Continuous distributions | 13 (Normal, StdNormal, t, Chi-Square, F, Exponential, Gamma, Beta, Uniform, LogNormal, Weibull, Cauchy, Logistic) | ✅ Available (roughly similar count) |
| Interactive PDF plots | ✅ All 20 distributions with sliders for each parameter | ❌ Static probability distribution plots |
| Random Data Generator | ✅ 16 distributions, multi-column, min/max, rounding, CSV download | ✅ Random number generation |
| CLT Simulator | ✅ Interactive Central Limit Theorem demo | ❌ Not available |
| Distribution Overlay Comparison | ✅ Side-by-side comparison of 2 distributions | ❌ Not available |
| Sampling Simulator | ✅ Empirical vs theoretical distribution comparison | ❌ Not available |
| Interactive CDF / inverse CDF | ❌ Not available | ✅ CDF and inverse CDF functions |

**Summary:** Our app has a clear educational edge with all 20 distributions being **interactive** (sliders update PDF/CDF plots in real-time) plus 4 educational tools (CLT Simulator, Overlay Comparison, Sampling Simulator, Random Data Generator). Minitab provides computational CDF/inverse CDF functions but lacks the interactive educational approach.

---

## 4. Power & Sample Size

| Feature | Our App | Minitab |
|---------|---------|---------|
| Total analysis types | **30** analysis types | ~15-20 types |
| A Priori (compute N) | ✅ All 30 types | ✅ |
| Post Hoc (compute power) | ✅ All 30 types | ✅ |
| Sensitivity (detectable effect) | ✅ All 30 types | ✅ |
| Criterion (required alpha) | ✅ All 30 types | ❌ Not listed |
| Compromise (cost ratio) | ✅ All 30 types | ❌ Not listed |
| One-sample mean (t/z) | ✅ | ✅ |
| Two independent means | ✅ | ✅ |
| Paired means | ✅ | ✅ |
| One-sample proportion | ✅ | ✅ |
| Two proportions | ✅ | ✅ (new in recent versions) |
| One-way ANOVA | ✅ | ✅ |
| Correlation (Pearson) | ✅ | ⚠ Not listed |
| Multiple Linear Regression | ✅ | ✅ (for factors) |
| Logistic Regression | ✅ | ⚠ Not listed |
| Chi-Square Test | ✅ | ⚠ Not listed |
| Mann-Whitney / Wilcoxon | ✅ | ⚠ Not listed |
| Log-Rank (Survival) | ✅ | ⚠ Not listed |
| Cox Regression | ✅ | ⚠ Not listed |
| Equivalence / Non-Inferiority | ✅ | ✅ |
| Repeated Measures ANOVA | ✅ | ⚠ Not listed |
| Two-way / Factorial ANOVA | ✅ | ✅ (factorial designs) |
| ROC / AUC Analysis | ✅ | ⚠ Not listed |
| Cohen's Kappa / ICC | ✅ | ⚠ Not listed |
| Cluster-RCT / Multilevel | ✅ | ⚠ Not listed |
| Precision-based (CI width) | ✅ | ✅ (Sample Size for Estimation) |
| Pilot / Feasibility | ✅ | ❌ Not available |
| Wilcoxon Signed-Rank (paired) | ✅ | ⚠ Not listed |
| Kruskal-Wallis | ✅ | ⚠ Not listed |
| Friedman Test | ✅ | ⚠ Not listed |
| McNemar's Test | ✅ | ⚠ Not listed |
| Fisher's Exact Test | ✅ | ⚠ Not listed |
| MANOVA | ✅ | ⚠ Not listed |
| Binomial Exact Test | ✅ | ⚠ Not listed |
| Monte Carlo Simulation | ✅ | ⚠ Not listed |
| Tolerance Intervals | ❌ Not available | ✅ |
| DOE power | ⚠ Not available | ✅ |
| **Adjustment parameters** | ✅ Dropout rate, multiple testing correction, cost, recruitment rate | ❌ Not listed |
| **Effect size converter** | ✅ d ↔ r ↔ f ↔ f² ↔ OR ↔ w | ❌ Not listed |
| **Power curves** | ✅ Interactive power curves | ✅ Power curves |

**Summary:** Our power analysis module is significantly more comprehensive (30 types vs ~15-20) with **5 analysis modes per type** (A Priori, Post Hoc, Sensitivity, Criterion, Compromise) vs Minitab's 3-4 modes. We also include adjustment parameters (dropout, FDR correction, cost) and an effect size converter. Minitab wins on power for design of experiments and tolerance intervals, which we lack.

---

## 5. Regression

| Feature | Our App | Minitab |
|---------|---------|---------|
| Linear Regression | ✅ Interactive widget | ✅ Regression menu |
| Nonlinear Regression | ❌ Not available | ✅ |
| Binary Logistic Regression | ✅ Interactive widget | ✅ |
| Ordinal Logistic Regression | ✅ Interactive widget | ✅ |
| Nominal Logistic Regression | ✅ Multinomial | ✅ |
| Poisson Regression | ✅ Interactive widget | ✅ |
| Negative Binomial Regression | ✅ Interactive widget | ❌ Not listed |
| Cox Regression | ✅ Interactive widget | ✅ Regression menu |
| Partial Least Squares | ❌ Not available | ✅ |
| Orthogonal Regression | ❌ Not available | ✅ |
| Multivariate Adaptive Regression Splines | ❌ Not available | ✅ |
| Stepwise selection | ✅ (via interactive selection) | ✅ (p-value, AICc, BIC) |
| Best subsets | ❌ Not available | ✅ |
| Stability studies | ❌ Not available | ✅ |
| Model validation | ⚠ Limited (R², AIC, BIC displayed) | ✅ |
| Regression plots | ✅ 10 types (residual, surface, etc.) | ✅ Residual, factorial, contour, surface |
| Response prediction/optimization | ❌ Not available | ✅ |

**Summary:** Minitab is stronger in regression with nonlinear, PLS, orthogonal regression, MARS, best subsets, stability studies, and response optimization. Our app covers the most commonly used regression types (linear, logistic, Poisson, Negative Binomial, Cox) with interactive widgets, which covers most educational needs.

---

## 6. ANOVA

| Feature | Our App | Minitab |
|---------|---------|---------|
| One-way ANOVA | ✅ Interactive widget, post-hoc, eta-squared, effect sizes | ✅ |
| Two-way ANOVA | ✅ Interactive widget | ✅ |
| ANCOVA | ✅ Interactive widget | ✅ (GLM) |
| MANOVA | ✅ Interactive widget | ✅ |
| Repeated Measures ANOVA | ✅ Interactive widget | ✅ (GLM) |
| General Linear Models | ❌ Not available | ✅ |
| Mixed Models | ❌ Not available | ✅ |
| Multiple Comparisons | ✅ **23 post-hoc methods** | ✅ Tukey, Dunnett, etc. (fewer methods) |
| Test for Equal Variances | ✅ Levene's, Bartlett's (in diagnostics) | ✅ |
| Response prediction/optimization | ❌ Not available | ✅ |
| Analysis of Means | ❌ Not available | ✅ |
| ANOVA plots | ✅ Residual, interaction, main effects | ✅ Residual, factorial, contour, surface |

**Summary:** Comparable for standard ANOVA types. Minitab wins on GLM, mixed models, and analysis of means. Our app wins on post-hoc test variety (23 methods vs Minitab's basic set).

---

## 7. Nonparametric Statistics

| Feature | Our App | Minitab |
|---------|---------|---------|
| Sign Test | ✅ One-sample + Paired | ✅ |
| Wilcoxon Signed-Rank | ✅ One-sample + Paired | ✅ |
| Mann-Whitney U | ✅ | ✅ |
| Kruskal-Wallis | ✅ + Dunn/Conover/DSCF post-hoc | ✅ |
| Mood's Median | ✅ | ✅ |
| Friedman Test | ✅ + Nemenyi/Conover/Wilcoxon post-hoc | ✅ |
| Runs Test | ✅ | ✅ |
| **Nonparametric post-hoc methods** | **6 methods** (Dunn, Conover, DSCF, Nemenyi, Conover-Friedman, Wilcoxon+Bonferroni) | ❌ Limited or none |
| Permutation MANOVA | ✅ | ❌ Not available |
| Cochran's Q | ✅ | ❌ Not listed |

**Summary:** Our app covers all 7 Minitab nonparametric tests plus adds nonparametric post-hoc methods (6 methods) and Permutation MANOVA. The post-hoc capabilities are a significant educational advantage.

---

## 8. Statistical Process Control (SPC)

| Feature | Our App | Minitab |
|---------|---------|---------|
| XBar, R, S charts | ❌ Not available | ✅ |
| I-MR, I-MR-R/S | ❌ Not available | ✅ |
| P, NP, C, U charts | ❌ Not available | ✅ |
| Laney P' and U' | ❌ Not available | ✅ |
| EWMA, CUSUM | ❌ Not available | ✅ |
| Multivariate T² | ❌ Not available | ✅ |
| Rare events (G and T) | ❌ Not available | ✅ |
| Process Capability (normal, non-normal, attribute) | ❌ Not available | ✅ |
| Process Capability Sixpack™ | ❌ Not available | ✅ |
| Tolerance Intervals | ❌ Not available | ✅ |
| Acceptance Sampling | ❌ Not available | ✅ (with OC curves) |
| Run Chart | ❌ Not available | ✅ |
| Multi-Vari Chart | ❌ Not available | ✅ |
| Variability Chart | ❌ Not available | ✅ |
| Box-Cox / Johnson transformations | ⚠ In diagnostics only | ✅ |
| Individual Distribution Identification | ❌ Not available | ✅ |
| Nonparametric capability | ❌ Not available | ✅ (new) |

**Summary:** This is Minitab's strongest area and a major gap in our app. We have **no SPC/quality control features** — control charts, process capability, acceptance sampling, etc. This is the category where adding coverage would require the most effort.

---

## 9. Design of Experiments (DOE)

| Feature | Our App | Minitab |
|---------|---------|---------|
| Two-level factorial | ❌ Not available | ✅ |
| General factorial | ❌ Not available | ✅ |
| Response surface | ❌ Not available | ✅ |
| Mixture designs | ❌ Not available | ✅ |
| D-optimal designs | ❌ Not available | ✅ |
| Plackett-Burman | ❌ Not available | ✅ |
| Definitive screening | ❌ Not available | ✅ |
| Split-plot | ❌ Not available | ✅ |
| Taguchi designs | ❌ Not available | ✅ |
| User-specified designs | ❌ Not available | ✅ |
| Binary response analysis | ❌ Not available | ✅ |

**Summary:** We have no DOE capabilities. Minitab's DOE module is extensive with 10+ design types. Adding DOE would be a large new feature area.

---

## 10. Reliability / Survival Analysis

| Feature | Our App | Minitab |
|---------|---------|---------|
| Kaplan-Meier Survival | ✅ Interactive widget | ✅ |
| Cox Proportional Hazards | ✅ Interactive widget | ✅ |
| Log-Rank Test | ✅ Interactive widget | ✅ |
| Nelson-Aalen | ✅ In graph explorer | ⚠ Not listed |
| Parametric distribution analysis | ❌ Not available | ✅ |
| Accelerated life testing | ❌ Not available | ✅ |
| Regression with life data | ❌ Not available | ✅ |
| Test plans | ❌ Not available | ✅ |
| Threshold parameter distributions | ❌ Not available | ✅ |
| Repairable systems | ❌ Not available | ✅ |
| Multiple failure modes | ❌ Not available | ✅ |
| Probit analysis | ❌ Not available | ✅ |
| Weibayes analysis | ❌ Not available | ✅ |
| Warranty analysis | ❌ Not available | ✅ |
| Survival probability heatmap | ✅ In graph explorer | ❌ Not available |
| Parametric survival plots | ❌ Not available | ✅ |

**Summary:** Our app covers basic survival analysis (KM, Cox, Log-Rank) with interactive visualization. Minitab's reliability module is extensive with 15+ features (parametric models, ALT, repairable systems, warranty analysis, test plans) — a major gap.

---

## 11. Time Series & Forecasting

| Feature | Our App | Minitab |
|---------|---------|---------|
| Time series plots | ✅ Basic time series in graph explorer | ✅ |
| Trend analysis | ❌ Not available | ✅ |
| Decomposition | ❌ Not available | ✅ |
| Moving average | ❌ Not available | ✅ |
| Exponential smoothing | ❌ Not available | ✅ |
| Winters' method | ❌ Not available | ✅ |
| ACF/PACF | ❌ Not available | ✅ |
| ARIMA | ❌ Not available | ✅ |
| Box-Cox transformation | ⚠ In diagnostics | ✅ (new) |
| Augmented Dickey-Fuller | ❌ Not available | ✅ (new) |
| Best ARIMA model selection | ❌ Not available | ✅ (new) |

**Summary:** No time series features except a basic time series plot. Minitab has a complete time series module with ARIMA, decomposition, ACF/PACF, and forecasting. Another major gap.

---

## 12. Multivariate Analysis

| Feature | Our App | Minitab |
|---------|---------|---------|
| PCA | ✅ PCA scatter plot in graph explorer | ✅ |
| Factor Analysis | ❌ Not available | ✅ |
| Discriminant Analysis | ❌ Not available | ✅ |
| Cluster Analysis | ✅ K-means visualization in graph explorer | ✅ |
| Correspondence Analysis | ❌ Not available | ✅ |
| Cronbach's Alpha / Item Analysis | ❌ Not available | ✅ |
| Multivariate Adaptive Regression Splines | ❌ Not available | ✅ |

**Summary:** Our app has basic PCA and cluster visualization in the graph explorer. Minitab has a full multivariate module with factor analysis, discriminant, correspondence analysis, and Cronbach's alpha. Significant gap.

---

## 13. Measurement Systems Analysis (MSA)

| Feature | Our App | Minitab |
|---------|---------|---------|
| Gage R&R Crossed | ❌ Not available | ✅ |
| Gage R&R Nested | ❌ Not available | ✅ |
| Gage R&R Expanded | ❌ Not available | ✅ |
| Gage linearity and bias | ❌ Not available | ✅ |
| Type 1 Gage Study | ❌ Not available | ✅ |
| Attribute Gage Study | ❌ Not available | ✅ |
| Attribute Agreement Analysis | ❌ Not available | ✅ |
| EMP Crossed | ❌ Not available | ✅ (new) |

**Summary:** No MSA capabilities. This is an industrial/manufacturing area that our educational app doesn't target.

---

## 14. Diagnostic Accuracy (our unique area)

| Feature | Our App | Minitab |
|---------|---------|---------|
| Sensitivity & Specificity | ✅ Interactive widget + tables | ❌ Not available |
| ROC Curve Analysis | ✅ Interactive with threshold explorer | ❌ Not available |
| Likelihood Ratios (LR+/LR-) | ✅ Interactive widget | ❌ Not available |
| PPV / NPV | ✅ Interactive widget | ❌ Not available |
| Calibration Plot | ✅ Interactive | ❌ Not available |
| Confusion Matrix | ✅ Interactive heatmap | ❌ Not available |
| Precision-Recall Curve | ✅ Interactive | ❌ Not available |
| Diagnostic 2x2 table | ✅ In tabulation module | ❌ Not available |
| Youden's Index | ✅ Automated | ❌ Not available |

**Summary:** This is a unique strength of our app — diagnostic accuracy analysis with 8+ features not available in Minitab. Valuable for clinical/biostatistics education.

---

## 15. Post-Hoc Tests (our unique area)

| Feature | Our App | Minitab |
|---------|---------|---------|
| Total methods | **23 methods** across parametric/nonparametric contexts | ~6-8 basic methods |
| Fisher LSD | ✅ | ✅ |
| Tukey HSD | ✅ | ✅ |
| Bonferroni | ✅ | ✅ |
| Holm-Bonferroni | ✅ | ⚠ Not listed |
| Sidak | ✅ | ⚠ Not listed |
| Scheffe | ✅ | ✅ |
| Dunnett | ✅ | ✅ |
| Games-Howell | ✅ | ⚠ Not listed |
| Newman-Keuls | ✅ | ⚠ Not listed |
| Dunn's (nonparametric) | ✅ | ⚠ Not listed |
| Conover (nonparametric) | ✅ | ⚠ Not listed |
| DSCF (nonparametric) | ✅ | ⚠ Not listed |
| Nemenyi (Friedman) | ✅ | ⚠ Not listed |
| Conover-Friedman | ✅ | ⚠ Not listed |
| Discriminant/Canonical (MANOVA) | 2 methods | ⚠ Not listed |
| Post-hoc visualizations | ✅ 6 plot types (forest, heatmap, CLD, network, etc.) | ❌ Not available |

**Summary:** Our post-hoc system is far more comprehensive than Minitab's — 23 methods (vs ~6-8) with nonparametric, repeated measures, and MANOVA-specific methods, plus 6 dedicated visualization types. Minitab has basic multiple comparisons in ANOVA but nothing close to this breadth.

---

## 16. Meta-Analysis (our unique area)

| Feature | Our App | Minitab |
|---------|---------|---------|
| Forest Plot | ✅ In graph explorer | ❌ Not available |
| Funnel Plot | ✅ In graph explorer | ❌ Not available |
| Galbraith (Radial) Plot | ✅ In graph explorer | ❌ Not available |
| Baujat Plot | ✅ In graph explorer | ❌ Not available |
| Leave-One-Out Plot | ✅ In graph explorer | ❌ Not available |

**Summary:** Meta-analysis visualization is unique to our app — Minitab has no meta-analysis features at all.

---

## 17. Tables, Tabulation & Contingency Tables

| Feature | Our App | Minitab |
|---------|---------|---------|
| Frequency Tables | ✅ Interactive with class intervals | ✅ Tally |
| Cross-Tabulation | ✅ 2x2 and R x C contingency tables | ✅ Cross Tabulation |
| Proportion/Percentage Tables | ✅ Row %, Column %, Expected frequency | ⚠ Limited |
| Descriptive Statistics Tables | ✅ Mean, median, SD, skew, kurtosis, etc. | ✅ Descriptive Statistics |
| Grouped Summary Tables | ✅ By-group summaries | ✅ |
| Pivot Tables | ✅ | ✅ |
| Diagnostic Accuracy Tables | ✅ 2x2 with sensitivity/specificity/PPV/NPV/LR | ❌ Not available |
| Agreement Tables (Kappa, ICC) | ✅ Interactive | ✅ |
| Regression Summary Tables | ✅ Coefficients, odds ratios, model fit, ANOVA, residuals | ✅ |
| Effect Size Tables | ✅ Cohen's d, eta-squared, Cramer's V, OR, RR | ⚠ Limited |
| Kaplan-Meier Life Table | ✅ Interactive | ✅ |
| Multiple Testing Correction | ✅ Bonferroni, Holm, Sidak, BH-FDR | ⚠ Not listed |
| Power Curve Table | ✅ | ⚠ Not listed |
| Post-Hoc Tables | ✅ Pairwise comparisons, adjusted p-values, CIs | ✅ |
| APA/Journal Export | ✅ APA-formatted tables + CSV/LaTeX/HTML download | ❌ Not available |
| **Educational Explorers** | 6 interactive explorers (Frequency, Cross-Tab, Expected, Odds & Risk, Conditional Probability, Bayesian Updater) | ❌ **Not available** |
| Tabulated statistics | ❌ Not available | ✅ (Graph Builder feature) |
| Chi-square / Fisher's exact (from tables) | ✅ Interactive | ✅ |

**Summary:** Our tabulation module is more expansive with 13 sections vs Minitab's basic Tally/Cross-Tabulation/Descriptive Statistics. We add educational explorers, diagnostic accuracy tables, effect size tables, multiple testing corrections, APA export, and post-hoc tables. Minitab's tabulated statistics in Graph Builder is a nice feature we lack.

---

## 18. Data Import & Management

| Feature | Our App | Minitab |
|---------|---------|---------|
| CSV upload | ✅ | ✅ |
| Excel upload | ✅ | ✅ |
| Built-in datasets | ✅ 20 curated datasets | ✅ Many sample datasets |
| Drag-and-drop | ❌ Not available | ✅ |
| Database connectors | ❌ Not available | ✅ |
| Data transformation | ⚠ Limited (structure detection) | ✅ Extensive |
| Data pivoting | ⚠ Manual organization | ✅ |
| Stacking/unstacking columns | ❌ Not available | ✅ |
| Missing data handling | ⚠ Not addressed | ✅ |
| Simulated data | ✅ Via sliders (educational) | ✅ Via random generation |
| Dual-mode (simulated + real) | ✅ Core design philosophy | ❌ Not applicable |

**Summary:** Minitab has more robust data management (database connectors, transformations, pivoting, missing data handling). Our app's dual-mode design (simulated for education + real data import) is a unique educational approach. Our 20 built-in datasets are curated for learning.

---

## 19. Diagnostics / Assumption Checking

| Feature | Our App | Minitab |
|---------|---------|---------|
| Shapiro-Wilk Normality | ✅ Interactive | ✅ |
| Kolmogorov-Smirnov | ✅ Interactive | ✅ |
| Anderson-Darling | ✅ Interactive | ✅ |
| Jarque-Bera | ✅ Interactive | ⚠ Not listed |
| D'Agostino-Pearson | ✅ Interactive | ⚠ Not listed |
| Levene's Test | ✅ Interactive | ✅ |
| Bartlett's Test | ✅ Interactive | ✅ |
| Fligner-Killeen | ✅ Interactive | ⚠ Not listed |
| Cochran's C | ✅ Interactive | ⚠ Not listed |
| Durbin-Watson | ✅ Interactive | ✅ |
| Breusch-Pagan | ✅ Interactive | ⚠ Not listed |
| White's Test | ✅ Interactive | ⚠ Not listed |
| Grubbs' Outlier | ✅ Interactive | ✅ |
| Rosner's (ESD) Outlier | ✅ Interactive | ⚠ Not listed |
| Mahalanobis Distance | ✅ Interactive | ⚠ Not listed |
| IQR Outlier Detection | ✅ Interactive | ⚠ Not listed |
| Variance Inflation Factor (VIF) | ✅ Interactive | ✅ |
| Condition Index | ✅ Interactive | ⚠ Not listed |
| Cook's Distance | ✅ Interactive | ✅ |
| DFFITS | ✅ Interactive | ⚠ Not listed |
| Leverage / Hat Values | ✅ Interactive | ✅ |
| Visualizations per test | ✅ Q-Q plots, histograms, boxplots, residual plots with every test | ✅ Diagnostic graphs |
| Box-Cox / Johnson transformations | ⚠ In diagnostics | ✅ |

**Summary:** Our app has 21 diagnostic tests across 7 categories — more than Minitab's listed diagnostics. We include several tests (Fligner-Killeen, Breusch-Pagan, White, Rosner's, Mahalanobis, Condition Index, DFFITS) not found in Minitab's Basic Statistics menu. Minitab has the edge on data transformation tools.

---

## 20. Educational Features (our unique advantage)

| Feature | Our App | Minitab |
|---------|---------|---------|
| **Test Finder** | ✅ Interactive wizard: objective → IV type → DV type → groups → relationship → distribution → recommendation | ❌ Not available |
| **Interactive test widgets** | **41 widgets** — all with real-time sliders that update plots and statistics instantly | ❌ Static output |
| **Interactive graph explorer** | **83 graph types** — all with real-time parameter sliders | ❌ Static graphs |
| **Step-by-step solved examples** | ✅ 7 examples with full LaTeX formulas, step-by-step solutions, decision rules | ❌ Not available |
| **Glossary** | ✅ ~150 terms across 14 categories, all interactive | ❌ Not available |
| **Educational tooltips** | ✅ `st.info()` / `st.expander()` integrated into every feature | ❌ Not available |
| **"When to use this test"** | ✅ Built into every widget | ❌ Not available |
| **Effect size interpretations** | ✅ Automatic text: Trivial/Small/Medium/Large | ⚠ Numerical only |
| **Interpretation guidance** | ✅ Full sentences explaining results in plain language | ❌ Numerical output only |
| **CLT Simulator** | ✅ Interactive Central Limit Theorem visualization | ❌ Not available |
| **Distribution Overlay Comparison** | ✅ Side-by-side distribution comparison | ❌ Not available |
| **Sampling Simulator** | ✅ Interactive sampling demonstration | ❌ Not available |
| **Educational Table Explorers** | ✅ 6 educational modules (Frequency, Cross-Tab, Expected, Odds & Risk, Conditional Probability, Bayesian Updater) | ❌ Not available |
| **Data Workspace** | ✅ Two-column: data management + educational test picker | ❌ Not applicable |
| **Dual-mode data** | ✅ Default = simulated (educational), optional = real data | ❌ Not applicable |
| **APA-formatted output** | ✅ Professional tables + download in CSV/LaTeX/HTML | ⚠ Basic formatting |
| **Web-based / zero install** | ✅ Runs in browser | ❌ Desktop install |
| **Free** | ✅ Completely free | ❌ ~$1,500/year license |

**Summary:** This is our app's **strongest differentiator**. We're designed as an **educational interactive statistics platform**, while Minitab is designed as a **professional analysis tool**. Every feature in our app is built with teaching in mind — real-time sliders, interpretations, step-by-step examples, glossary, test finder wizard, and educational explorers.

---

## 21. Cross-Cutting Summary

| Category | Our App | Minitab | Advantage |
|----------|---------|---------|-----------|
| **Hypothesis Tests** | 41+ widgets, more variety | ~40 tests, standard set | **Our App** (more types, interactive) |
| **Graphs** | 83 types, all interactive | 40-50 types, static | **Our App** (more types, interactive) |
| **Distributions** | 20 distributions, all interactive | ~20 distributions, static | **Our App** (interactivity, CLT, etc.) |
| **Power Analysis** | 30 types, 5 modes, adjustments | ~15-20 types, 3-4 modes | **Our App** (more comprehensive) |
| **Regression** | 8 types | 10+ types | **Minitab** (more methods) |
| **ANOVA** | Good coverage | Complete (GLM, mixed) | **Minitab** (GLM, mixed models) |
| **Nonparametrics** | 9 tests + 6 post-hoc methods | 7 tests | **Our App** (more methods) |
| **SPC / Quality** | ❌ None | ✅ Extensive | **Minitab** |
| **DOE** | ❌ None | ✅ Extensive | **Minitab** |
| **Reliability** | Basic (KM, Cox) | ✅ Extensive (15+ features) | **Minitab** |
| **Time Series** | ❌ None | ✅ Complete module | **Minitab** |
| **Multivariate** | Basic (PCA, cluster viz) | ✅ Complete module | **Minitab** |
| **MSA** | ❌ None | ✅ Complete module | **Minitab** |
| **Diagnostic Accuracy** | ✅ 8+ features | ❌ None | **Our App** |
| **Post-Hoc Tests** | ✅ 23 methods + 6 visualizations | ~6-8 basic methods | **Our App** |
| **Meta-Analysis** | ✅ 5 plot types | ❌ None | **Our App** |
| **Tabulation** | ✅ 13 sections, 6 explorers | Basic | **Our App** |
| **Diagnostics** | 21 tests (7 categories) | ~10 tests | **Our App** (more variety) |
| **Educational Features** | ✅ Extensive | ❌ None | **Our App** |
| **Data Management** | Basic (CSV/Excel) | ✅ Professional | **Minitab** |
| **Collaboration** | ❌ None | ✅ Project sharing | **Minitab** |
| **Scripting/API** | ❌ None | ✅ Python, R, macros | **Minitab** |
| **Cost** | **Free** | ~$1,500/year | **Our App** |

---

## 22. Strategic Summary

### Our App's Unique Value Proposition (vs Minitab)

| What we do better | What Minitab does better |
|-------------------|--------------------------|
| **Interactive widgets** (real-time sliders on every test/graph) | **Statistical Process Control** (all control charts, capability analysis) |
| **Educational design** (interpretations, when-to-use, step-by-step) | **Design of Experiments** (10+ design types) |
| **Post-hoc methods** (23 methods with 6 visualizations) | **Reliability/Survival** (ALT, warranty, repairable systems) |
| **Diagnostic accuracy** (8+ clinical/biostatistics features) | **Time Series & Forecasting** (ARIMA, decomposition, ACF/PACF) |
| **Meta-analysis** (5 plot types) | **Multivariate Analysis** (factor, discriminant, correspondence analysis) |
| **Power analysis** (30 types, 5 modes, adjustments) | **Measurement Systems Analysis** (Gage R&R, EMP) |
| **Graph variety** (83 types, all interactive) | **Data Management** (transformations, database connectors) |
| **Test finder** (interactive wizard for test selection) | **Regression** (nonlinear, PLS, MARS, best subsets) |
| **Glossary + Solved Examples** (integrated learning) | **Scripting** (Python, R, macro support) |
| **Completely free** (no license, zero install) | **Collaboration** (project sharing, versioning) |
| **Built-in datasets** (20 curated for learning) | **ANOVA** (GLM, mixed models, analysis of means) |

### Unique Features (no Minitab equivalent)

| Feature | What it does |
|---------|-------------|
| **Test Finder Wizard** | Interactive 6-step questionnaire → test recommendation |
| **41 Interactive Test Widgets** | Real-time sliders update plots/stats instantly |
| **83 Interactive Graph Types** | All with adjustable parameters |
| **23 Post-Hoc Methods** | Nonparametric, MANOVA, and repeated measures methods |
| **6 Educational Table Explorers** | Interactive learning modules for cross-tab concepts |
| **CLT Simulator** | Interactive Central Limit Theorem demonstration |
| **Distribution Overlay** | Compare two distributions side-by-side |
| **Sampling Simulator** | Interactive sampling from distributions |
| **Diagnostic Accuracy Module** | Complete clinical test evaluation toolbox |
| **Meta-Analysis Plots** | 5 publication-ready meta-analysis visualizations |
| **Step-by-Step Solved Examples** | Full LaTeX walkthroughs with interpretations |
| **Glossary with ~150 Terms** | 14 categories, always accessible |
| **APA Journal Export** | Professional formatted tables |
| **Effect Size Converter** | d ↔ r ↔ f ↔ f² ↔ OR ↔ w |

### Suggested Future Development (to close gaps with Minitab)

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | **Control charts** (XBar, R, P, EWMA, CUSUM) | High | Opens SPC/quality audience |
| 2 | **Time series plots** (ACF/PACF, trend, decomposition) | Medium | Useful for many fields |
| 3 | **Factor analysis / Cronbach's alpha** | Medium | Psychometrics education |
| 4 | **Nonlinear regression** | Medium | Common in biology/chemistry |
| 5 | **Database/API data connectors** | Medium | Professional usability |
| 6 | **Reliability analysis** (parametric distributions, Weibull) | High | Engineering audience |
| 7 | **DOE basics** (factorial designs) | Very High | Advanced statistics education |
| 8 | **Data transformation** (Box-Cox, Johnson, log, sqrt) | Low | Complements diagnostics |
| 9 | **Mixed models / GLM** | Very High | Advanced statistics |
| 10 | **Session scripting / reproducibility** | Medium | Research use |

---

*Generated May 2026. Our app version: ~28,000 lines across 7 apps, 12 feature modules, 4 core modules. Minitab version: Minitab Statistical Software (latest).*
