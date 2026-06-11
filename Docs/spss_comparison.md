# Feature Comparison: Statistics WebApp vs IBM SPSS Statistics

> **Our App** = Statistics WebApp (free, web-based, educational)
> **SPSS** = IBM SPSS Statistics (commercial, desktop + server, paid license)

**Key difference in philosophy:**
- **Our App**: Educational-first, every feature is interactive (real-time sliders), dual-mode data (simulated + real), interpretive guidance built-in
- **SPSS**: Professional analysis tool, batch-oriented (point-and-click or syntax), comprehensive data management, enterprise integrations, AI-assisted output

SPSS is modular — each numbered section is a separate paid add-on.

---

## 0. Architecture & Pricing

| Dimension | Our App | SPSS |
|-----------|---------|------|
| Cost | **Free** (zero cost) | **$105–$300+/month** or $1,500+/year per user |
| Add-on modules | None (all-in-one) | **10+ paid add-ons** (each $500–$1,500/year extra) |
| Full-featured version | Single app | **Premium edition** = Base + Advanced + Regression + all add-ons (~$8,000+/year) |
| Platform | **Web browser** (zero install) | Desktop (Windows/Mac/Linux) or server |
| Deployment | Streamlit cloud or self-hosted | On-prem, cloud (SaaS), or hybrid |
| Scripting/automation | ❌ None | ✅ **Syntax editor**, Python/R integration, macros |
| AI assistance | ❌ None | ✅ **AI Output Assistant** (watsonx.ai, v32) |
| Collaboration | ❌ None | ✅ Project files, shared output |
| Data size limit | Moderate (in-memory) | **Enterprise-scale** (server edition, database connectors) |

---

## 1. Descriptive Statistics & Tables

| Feature | Our App | SPSS |
|---------|---------|------|
| Frequencies (one-way tables) | ✅ Interactive frequency tables with class intervals | ✅ Frequencies procedure |
| Descriptive statistics (mean, SD, etc.) | ✅ Interactive, all standard stats | ✅ Descriptives, Explore |
| Cross-tabulation / contingency tables | ✅ 2x2 + R x C + proportion/row%/col%/expected | ✅ Crosstabs (chi-square, phi, Cramer's V, risk, etc.) |
| Custom/presentation tables | ✅ APA-formatted export (CSV/LaTeX/HTML) | ✅ **Custom Tables** (drag-and-drop, 160+ stats, add-on) |
| Grouped summary tables | ✅ By-group summaries | ✅ Means, Report procedures |
| Pivot tables | ✅ Interactive | ✅ Pivot table editor in Output Viewer |
| **Educational table explorers** | ✅ **6 interactive modules** (Frequency, Cross-Tab, Expected, Odds & Risk, Conditional Probability, Bayesian Updater) | ❌ Not available |
| Codebook | ❌ Not available | ✅ Codebook procedure |
| Explore (detailed EDA) | ❌ Not available | ✅ Explore (boxplots, stem-leaf, normality tests, M-estimators) |
| OLAP cubes | ❌ Not available | ✅ OLAP Cubes |

**Summary:** Our app has interactive tables and unique educational explorers. SPSS has far more robust data description tools (Explore, Codebook, OLAP) and the Custom Tables add-on (drag-and-drop, 160+ stats).

---

## 2. Hypothesis Testing

| Test | Our App | SPSS Base | Notes |
|------|---------|-----------|-------|
| **One-sample t-test** | ✅ Interactive widget | ✅ Base | |
| One-sample z-test | ✅ Interactive widget | ✅ Base (via syntax) | |
| **One-sample proportion** (Binomial) | ✅ Normal approx + Exact | ✅ Base (Binomial) | |
| One-sample Poisson rate | ⚠ Listed, no dedicated widget | ✅ Base (Rare Events) | |
| Wilcoxon Signed-Rank (one-sample) | ✅ Interactive widget | ✅ Base (Nonparametric) | |
| Sign Test (one-sample) | ✅ Interactive widget | ✅ Base (Nonparametric) | |
| **Paired t-test** | ✅ Interactive widget | ✅ Base | |
| Wilcoxon Signed-Rank (paired) | ✅ Interactive widget | ✅ Base (2 Related Samples) | |
| Sign Test (paired) | ✅ Interactive widget | ✅ Base (2 Related Samples) | |
| McNemar's Test | ✅ Interactive widget | ✅ Base (2 Related Samples) | |
| **Independent t-test** (Student's + Welch's) | ✅ Side-by-side, group names, n, Mean, SD, SE, df, CI | ✅ Base (Independent T-Test) | |
| Mann-Whitney U | ✅ Interactive widget | ✅ Base (2 Independent Samples) | |
| F-Test for Two Variances | ✅ Interactive widget | ✅ Base (Levene's in t-test, but no dedicated widget) | |
| Equivalence Test (TOST) | ✅ Interactive widget | ❌ Not available in Base | Requires custom syntax |
| **One-way ANOVA** | ✅ Interactive, post-hoc, eta-squared, effect sizes | ✅ Base (One-Way ANOVA) | |
| Two-way ANOVA | ✅ Interactive widget | ✅ Advanced Statistics (GLM) | |
| ANCOVA | ✅ Interactive widget | ✅ Advanced Statistics (GLM) | |
| Repeated Measures ANOVA | ✅ Interactive widget | ✅ Advanced Statistics (GLM) | |
| MANOVA | ✅ Interactive widget | ✅ Advanced Statistics (GLM) | |
| Kruskal-Wallis | ✅ Interactive + post-hoc | ✅ Base (K Independent Samples) | |
| Mood's Median | ✅ Interactive widget | ❌ Not in Base (requires syntax) | |
| Friedman Test | ✅ Interactive + post-hoc | ✅ Base (K Related Samples) | |
| Cochran's Q | ✅ Interactive widget | ✅ Base (K Related Samples) | |
| Permutation MANOVA | ✅ Interactive widget | ❌ Not available | |
| **Pearson Correlation** | ✅ Interactive widget | ✅ Base (Bivariate) | |
| Spearman Rank Correlation | ✅ Interactive widget | ✅ Base (Bivariate) | |
| Kendall's Tau-b | ✅ Interactive widget | ✅ Base (Bivariate) | |
| Point-Biserial Correlation | ✅ Interactive widget | ⚠ Via syntax or correlation menu | |
| Partial Correlation | ❌ Not available | ✅ Base (Partial) | |
| Distance Correlation | ❌ Not available | ✅ **v32 new feature** | |
| **Chi-Square GOF** | ✅ Interactive widget | ✅ Base (Nonparametric) | |
| Chi-Square Independence | ✅ Interactive widget | ✅ Base (Crosstabs) | |
| Fisher's Exact Test | ✅ Interactive widget | ✅ Base (Crosstabs) | |
| Cohen's Kappa | ✅ Interactive widget | ✅ Base (Crosstabs via kappa) | |
| Weighted Kappa | ✅ Interactive widget | ⚠ Not standard in Base | |
| Fleiss' Kappa | ✅ Interactive widget | ❌ Not available | |
| Bland-Altman Analysis | ✅ Interactive widget | ❌ Not available | |
| ROC Curve Analysis | ✅ Interactive widget | ✅ Base (ROC Curve) | |
| Sensitivity & Specificity | ✅ Interactive widget | ❌ Not available as dedicated test | |
| Likelihood Ratio Analysis | ✅ Interactive widget | ❌ Not available as dedicated test | |
| **Simple Linear Regression** | ✅ Interactive widget | ✅ Base (Linear) | |
| Multiple Linear Regression | ✅ Interactive widget | ✅ Base (Linear) | |
| Binary Logistic Regression | ✅ Interactive widget | ✅ Regression add-on | |
| Multinomial Logistic Regression | ✅ Interactive widget | ✅ Regression add-on | |
| Ordinal Logistic Regression | ✅ Interactive widget | ✅ Regression add-on | |
| Poisson Regression | ✅ Interactive widget | ✅ Advanced Statistics (GENLIN) | |
| Negative Binomial Regression | ✅ Interactive widget | ✅ Advanced Statistics (GENLIN) | |
| Nonlinear Regression | ❌ Not available | ✅ Regression add-on | |
| Probit Analysis | ❌ Not available | ✅ Regression add-on | |
| 2-Stage Least Squares | ❌ Not available | ✅ Regression add-on | |
| Ridge / Lasso / Elastic Net | ❌ Not dedicated widget (regularization in graph explorer) | ✅ Regression add-on (v31+) | |
| **Kaplan-Meier Survival** | ✅ Interactive widget | ✅ Advanced Statistics | |
| Cox Regression (PH) | ✅ Interactive widget | ✅ Advanced Statistics | |
| Log-Rank Test | ✅ Interactive widget | ✅ Advanced Statistics | |
| Life Tables | ❌ Not available | ✅ Advanced Statistics | |
| **Outlier Tests** (Grubbs, Rosner, Mahalanobis, IQR) | ✅ **4 methods** in diagnostics | ⚠ Limited (Explore, Regression diagnostics) | |
| Normality Tests (Shapiro-Wilk, K-S, A-D, etc.) | ✅ **5 methods** in diagnostics | ✅ Base (Explore, NPAR) | |

**Summary:** Our app covers **more individual test types** (71 vs Base ~25-30) with interactive widgets. SPSS has key tests we lack (partial correlation, nonlinear regression, probit, 2SLS, life tables). SPSS's Exact Tests add-on (30+ exact tests) overlaps with many of our interactive tests. **Our unique tests**: Fleiss' Kappa, Weighted Kappa, Bland-Altman, Permutation MANOVA, TOST, point-biserial, all diagnostic accuracy tests.

---

## 3. Regression Modeling

| Feature | Our App | SPSS |
|---------|---------|------|
| Linear regression | ✅ Interactive | ✅ Base |
| Multiple regression | ✅ Interactive | ✅ Base |
| Binary logistic | ✅ Interactive | ✅ Regression add-on |
| Multinomial logistic | ✅ Interactive | ✅ Regression add-on |
| Ordinal logistic | ✅ Interactive | ✅ Regression add-on |
| Poisson / Negative binomial | ✅ Interactive | ✅ Advanced Statistics (GENLIN) |
| Nonlinear regression | ❌ Not available | ✅ Regression add-on |
| Probit | ❌ Not available | ✅ Regression add-on |
| 2-Stage Least Squares | ❌ Not available | ✅ Regression add-on |
| Weighted Least Squares | ❌ Not available | ✅ Regression add-on |
| Ridge / Lasso / Elastic Net | ✅ Regularization path in graph explorer (no data fitting) | ✅ Regression add-on (v31+) |
| Partial Least Squares | ❌ Not available | ❌ Not standard (in SPSS Modeler) |
| Cox regression | ✅ Interactive | ✅ Advanced Statistics |
| Stepwise selection | ✅ Manual via checkbox | ✅ Automatic (p, AICc, BIC) |
| Best subsets | ❌ Not available | ✅ Base |
| Response prediction | ❌ Not available | ✅ Regression add-on |
| Model validation | ⚠ R², AIC, BIC only | ✅ Regression add-on |
| Regression diagnostics plots | ✅ 8 plot types (residual, surface, etc.) | ✅ Base (residual plots, partial plots) |
| Linear mixed models / HLM | ❌ Not available | ✅ Advanced Statistics |
| Generalized linear models (GENLIN) | ❌ Not available | ✅ Advanced Statistics |
| Generalized estimating equations (GEE) | ❌ Not available | ✅ Advanced Statistics |
| Generalized linear mixed models (GLMM) | ❌ Not available | ✅ Advanced Statistics |

**Summary:** SPSS regression is far more extensive — nonlinear, probit, 2SLS, WLS, mixed models, GENLIN, GEE, GLMM are entirely absent from our app. Our app covers the 8 most commonly taught regression types interactively, which covers educational needs well.

---

## 4. ANOVA & General Linear Models

| Feature | Our App | SPSS |
|---------|---------|------|
| One-way ANOVA | ✅ Interactive | ✅ Base |
| Two-way ANOVA | ✅ Interactive | ✅ Advanced Statistics (GLM) |
| ANCOVA | ✅ Interactive | ✅ Advanced Statistics (GLM) |
| MANOVA | ✅ Interactive | ✅ Advanced Statistics (GLM) |
| Repeated Measures ANOVA | ✅ Interactive | ✅ Advanced Statistics (GLM) |
| General Linear Models (GLM) | ❌ Not available | ✅ Advanced Statistics |
| Linear Mixed Models (HLM) | ❌ Not available | ✅ Advanced Statistics |
| Generalized Linear Models (GENLIN) | ❌ Not available | ✅ Advanced Statistics |
| Generalized Estimating Equations (GEE) | ❌ Not available | ✅ Advanced Statistics |
| Generalized Linear Mixed Models (GLMM) | ❌ Not available | ✅ Advanced Statistics |
| Post-hoc comparisons | ✅ **23 methods** + 6 visualizations | ✅ **Tukey, Bonferroni, Sidak, Scheffe, Dunnett, LSD, etc.** (fewer methods) |
| Analysis of Means | ❌ Not available | ❌ Not in SPSS (Minitab feature) |
| Profile plots | ❌ Not available | ✅ Advanced Statistics |
| Contrasts | ❌ Not customizable | ✅ Advanced Statistics (custom contrasts) |
| Unianova (ANOVA with multiple designs) | ❌ Not available | ✅ Base (Unianova) |

**Summary:** SPSS has much more comprehensive ANOVA/GLM capabilities (GLM, mixed, GENLIN, GEE, GLMM, custom contrasts, profile plots). Our app's **major advantage is 23 post-hoc methods with 6 visualizations** — SPSS has fewer and no dedicated post-hoc visualizations.

---

## 5. Nonparametric Statistics

| Feature | Our App | SPSS |
|---------|---------|------|
| Sign Test | ✅ 1-sample + paired | ✅ Base (Legacy) |
| Wilcoxon Signed-Rank | ✅ 1-sample + paired | ✅ Base |
| Mann-Whitney U | ✅ | ✅ Base |
| Kruskal-Wallis | ✅ + post-hoc (Dunn, Conover, DSCF) | ✅ Base (no post-hoc) |
| Mood's Median | ✅ | ❌ Not in Base |
| Friedman | ✅ + post-hoc (Nemenyi, Conover, Wilcoxon+Bonferroni) | ✅ Base (no post-hoc) |
| Runs Test | ✅ | ✅ Base |
| Cochran's Q | ✅ | ✅ Base |
| **Exact tests** (for small samples) | ✅ Each widget provides exact p-values | ✅ **Exact Tests add-on** (30+ exact methods) |
| Monte Carlo p-values | ⚠ Not available | ✅ Exact Tests add-on |
| Permutation MANOVA | ✅ | ❌ Not available |
| Post-hoc methods (nonparametric) | **6 methods** (Dunn, Conover, DSCF, Nemenyi, Conover-Friedman, Wilcoxon+Bonferroni) | ❌ **None** — SPSS does not provide nonparametric post-hoc tests |
| Kolmogorov-Smirnov (2-sample) | ❌ Not available | ✅ Base (2 Independent Samples) |
| Wald-Wolfowitz Runs | ❌ Not available | ✅ Base (2 Independent Samples) |
| Moses Extreme Reactions | ❌ Not available | ✅ Base (2 Independent Samples) |
| Jonckheere-Terpstra | ❌ Not available | ✅ Base (K Independent Samples) |

**Summary:** Comparable in test coverage. Our app adds nonparametric post-hoc methods (a major gap in SPSS), Mood's Median, and Permutation MANOVA. SPSS adds Jonckheere-Terpstra, K-S 2-sample, Wald-Wolfowitz, and Moses Extreme Reactions, plus the Exact Tests add-on (30+ exact methods). SPSS has the edge for researchers needing exact inference; our app has the edge for teaching (interactive widgets + post-hoc).

---

## 6. Diagnostics / Assumption Checking

| Feature | Our App | SPSS |
|---------|---------|------|
| Shapiro-Wilk | ✅ Interactive | ✅ Base (Explore) |
| Kolmogorov-Smirnov (Lilliefors) | ✅ Interactive | ✅ Base (Explore) |
| Anderson-Darling | ✅ Interactive | ✅ Base (Explore) |
| Jarque-Bera | ✅ Interactive | ❌ Not in Base |
| D'Agostino-Pearson | ✅ Interactive | ❌ Not in Base |
| Levene's Test | ✅ Interactive | ✅ Base |
| Bartlett's Test | ✅ Interactive | ✅ Base (via syntax) |
| Fligner-Killeen | ✅ Interactive | ❌ Not in Base |
| Cochran's C | ✅ Interactive | ❌ Not in Base |
| Durbin-Watson | ✅ Interactive | ✅ Base (in Regression) |
| Breusch-Pagan | ✅ Interactive | ❌ Not in Base |
| White's Test | ✅ Interactive | ❌ Not in Base |
| Grubbs' Outlier | ✅ Interactive | ❌ Not in Base |
| Rosner's (ESD) Outlier | ✅ Interactive | ❌ Not in Base |
| Mahalanobis Distance | ✅ Interactive | ✅ Base (in Regression) |
| IQR Outlier Detection | ✅ Interactive | ✅ Base (via Explore/Boxplot) |
| Variance Inflation Factor (VIF) | ✅ Interactive | ✅ Base (in Regression) |
| Condition Index | ✅ Interactive | ❌ Not in Base |
| Cook's Distance | ✅ Interactive | ✅ Base (in Regression) |
| DFFITS | ✅ Interactive | ❌ Not in Base |
| Leverage / Hat Values | ✅ Interactive | ✅ Base (in Regression) |
| **Visualizations per test** | ✅ Interactive Q-Q, histogram, boxplot, residual plots | ⚠ Static diagnostic plots |
| Box-Cox / Johnson transformations | ⚠ Not available (concept only) | ✅ Base (via syntax) |
| Missing data diagnostics | ❌ Not available | ✅ **Missing Values add-on** (patterns, EM, MICE imputation) |

**Summary:** Our app has **more diagnostic tests (21 vs ~12)** with interactive visualizations. SPSS has missing data diagnostics (a major gap in our app) and the Explore procedure for comprehensive EDA.

---

## 7. Power & Sample Size

| Feature | Our App | SPSS |
|---------|---------|------|
| **Total analysis types** | **30** | ~15 (Base includes t-tests, proportions, correlations, ANOVA, regression, precision) |
| A Priori (compute N) | ✅ All 30 types | ✅ Limited set |
| Post Hoc (compute power) | ✅ All 30 types | ✅ Limited set |
| Sensitivity (detectable effect) | ✅ All 30 types | ✅ Limited set |
| Criterion (required alpha) | ✅ All 30 types | ❌ Not available |
| Compromise (cost ratio) | ✅ All 30 types | ❌ Not available |
| **Adjustment parameters** | ✅ Dropout, FDR correction, cost, recruitment rate | ❌ Not available |
| **Effect size converter** | ✅ d ↔ r ↔ f ↔ f² ↔ OR ↔ w | ❌ Not available |
| **Power curves** | ✅ Interactive for all 30 types | ⚠ Limited (2D/3D charts for some tests) |
| **Monte Carlo simulation** | ✅ For t-tests, Welch, Mann-Whitney, proportions | ✅ Base (Power Analysis) |
| **Precision-based (CI width)** | ✅ | ✅ Power Analysis: Precision |
| **One-sample t-test power** | ✅ | ✅ Base |
| **Two-sample t-test power** | ✅ | ✅ Base |
| **Paired t-test power** | ✅ | ✅ Base |
| **One-way ANOVA power** | ✅ | ✅ Base |
| **Correlation power** | ✅ | ✅ Base |
| **Regression power** | ✅ | ✅ Base |
| **Proportion power** | ✅ | ✅ Base |
| **Equivalence / Non-Inferiority** | ✅ | ❌ Not in Base |
| **Logistic Regression** | ✅ | ❌ Not in Base |
| **Survival (Log-Rank, Cox)** | ✅ | ❌ Not in Base |
| **Repeated measures ANOVA** | ✅ | ❌ Not in Base |
| **MANOVA** | ✅ | ❌ Not in Base |
| **ROC / AUC** | ✅ | ❌ Not in Base |
| **Kappa / ICC** | ✅ | ❌ Not in Base |
| **Cluster-RCT / Multilevel** | ✅ | ❌ Not in Base |
| **Nonparametric (Mann-Whitney, Kruskal-Wallis, Friedman)** | ✅ | ❌ Not in Base |
| **McNemar, Fisher's Exact** | ✅ | ❌ Not in Base |
| **Pilot / Feasibility** | ✅ | ❌ Not in Base |
| **Binary Exact** | ✅ | ❌ Not in Base |
| **Grid power values** | ❌ Not available | ✅ Power Analysis: Grid Values |

**Summary:** Our power analysis is vastly more comprehensive (30 types, 5 modes, adjustments, converter). SPSS has basic power in Base (t-tests, proportions, correlations, ANOVA, regression, partial correlation) plus grid values. Our app has **every analysis type SPSS lacks** — equivalence, survival, repeated measures, MANOVA, ROC, Kappa, multilevel, nonparametric, etc.

---

## 8. Time Series & Forecasting

| Feature | Our App | SPSS |
|---------|---------|------|
| Time series plots | ✅ Basic in graph explorer | ✅ Forecasting add-on |
| Trend analysis | ❌ Not available | ✅ Forecasting |
| Seasonal decomposition | ❌ Not available | ✅ Forecasting |
| Exponential smoothing | ❌ Not available | ✅ Forecasting |
| ARIMA | ❌ Not available | ✅ Forecasting |
| Expert Modeler (auto ARIMA/ES) | ❌ Not available | ✅ Forecasting |
| ACF / PACF | ❌ Not available | ✅ Forecasting |
| Time Series Filtering | ❌ Not available | ✅ **v32 new feature** |
| Box-Cox transformation | ⚠ Not available | ✅ Forecasting |
| Augmented Dickey-Fuller | ❌ Not available | ❌ Not in SPSS |
| Moving average | ❌ Not available | ✅ Base (via syntax) |
| Spectral analysis | ❌ Not available | ✅ Forecasting |

**Summary:** Our app has **virtually no time series features** — only a basic time series plot in graph explorer. SPSS has a complete Forecasting add-on with ARIMA, exponential smoothing, decomposition, expert modeler, and seasonal adjustment. This is a major gap.

---

## 9. Multivariate Analysis

| Feature | Our App | SPSS |
|---------|---------|------|
| **Principal Component Analysis (PCA)** | ✅ Scatter plot in graph explorer | ✅ Base (Factor) |
| Factor Analysis | ❌ Not available | ✅ Base (Factor) |
| Discriminant Analysis | ❌ Not available | ✅ Base (Discriminant) |
| Cluster Analysis (K-means) | ✅ Visualization in graph explorer | ✅ Base |
| Cluster Analysis (Hierarchical) | ❌ Not available | ✅ Base |
| **Correspondence Analysis** | ❌ Not available | ✅ Categories add-on |
| Multiple Correspondence Analysis | ❌ Not available | ✅ Categories add-on |
| **Categorical Regression (CATREG)** | ❌ Not available | ✅ Categories add-on |
| Optimal Scaling | ❌ Not available | ✅ Categories add-on |
| Perceptual Mapping | ❌ Not available | ✅ Categories add-on |
| Preference Scaling (PREFSCAL) | ❌ Not available | ✅ Categories add-on |
| Nonlinear Canonical Correlation | ❌ Not available | ✅ Categories add-on |
| Proximity Scaling (PROXSCAL) | ❌ Not available | ✅ Categories add-on |
| **Cronbach's Alpha / Item Analysis** | ❌ Not available | ✅ Base (Reliability) |
| **Conjoint Analysis** | ❌ Not available | ✅ Conjoint add-on |
| **Multidimensional Scaling (PROXSCAL/PREFSCAL)** | ❌ Not available | ✅ Categories add-on |
| **Distance Correlation** | ❌ Not available | ✅ **v32 new feature** |
| **Proximity Mapping** | ❌ Not available | ✅ **v32 new feature** |
| MANOVA | ✅ Interactive widget | ✅ Advanced Statistics |
| Permutation MANOVA | ✅ Interactive widget | ❌ Not available |

**Summary:** SPSS dominates multivariate analysis — factor analysis, discriminant, hierarchical clustering, correspondence, MDS, conjoint, reliability, optimal scaling, CATREG, and the new v32 features (distance correlation, proximity mapping). Our app has basic PCA and cluster visualization only. This is a **major gap** in our app.

---

## 10. Survival Analysis

| Feature | Our App | SPSS |
|---------|---------|------|
| Kaplan-Meier | ✅ Interactive widget | ✅ Advanced Statistics |
| Cox Proportional Hazards | ✅ Interactive widget | ✅ Advanced Statistics |
| Log-Rank Test | ✅ Interactive widget | ✅ Advanced Statistics |
| Life Tables | ❌ Not available | ✅ Advanced Statistics |
| **Nelson-Aalen** | ✅ In graph explorer | ❌ Not standard in SPSS |
| Accelerated Failure Time | ❌ Not available | ❌ Not in SPSS |
| Competing Risks | ❌ Not available | ❌ Not in SPSS |
| Parametric survival models | ❌ Not available | ❌ Not in SPSS |
| **Survival plots** | 7 types (KM, hazard, cumulative hazard, Cox forest, survival heatmap) | ✅ KM curves, hazard plots (fewer types) |

**Summary:** Both cover basic survival (KM, Cox, Log-Rank). SPSS adds life tables. Our app adds Nelson-Aalen and 7 interactive survival plot types including unique visualizations (survival heatmap, Cox effect forest plot).

---

## 11. Data Visualization & Charts

| Feature | Our App | SPSS |
|---------|---------|------|
| **Total chart types** | **83 interactive widgets** | **~50+ chart types** (Chart Builder) |
| **Interactivity** | ✅ **All** charts have real-time sliders/parameters | ❌ **Static** — generate and view |
| Chart Builder (drag-and-drop) | ❌ Not available | ✅ Chart Builder |
| Distribution plots | ✅ 15 types | ✅ Histogram, boxplot, etc. |
| Comparison plots | ✅ 14 types | ✅ Bar, pie, error bar, etc. |
| Correlation plots | ✅ 6 types | ✅ Scatter, matrix, bubble, heatmap |
| Regression plots | ✅ 10 types | ✅ Scatter with fit line, residual, etc. |
| Diagnostic accuracy plots | ✅ 6 types | ❌ Not available |
| Agreement plots | ✅ 3 types | ❌ Not available (Kappa table only) |
| Multivariate plots | ✅ 6 types | ✅ 3D scatter, etc. |
| **Meta-analysis plots** | **5 types** (forest, funnel, Galbraith, Baujat, leave-one-out) | ❌ Not available |
| **Post-hoc visualizations** | **6 types** (forest, heatmap, CLD, network, etc.) | ❌ Not available |
| Survival plots | ✅ 7 types | ✅ Basic KM, hazard |
| Control charts | ❌ Not available | ✅ **Quality Control** (XBar, R, P, NP, etc.) |
| ROC curves | ✅ Interactive, multiple thresholds | ✅ Static ROC |
| **3D interactive** | ✅ 3D scatter explorer | ⚠ Limited (static, rotatable) |
| **Export formats** | ✅ PNG (300 DPI) + SVG | ✅ Many formats (PNG, JPG, TIF, EMF, etc.) |
| **Graph templates** | ❌ Not available | ✅ Chart templates (.sgt) |
| **Automated graph updates** | ✅ Real-time slider interaction | ❌ Static (must re-run) |

**Summary:** Our app has **more chart types (83 vs ~50)**, all interactive — a major educational advantage. SPSS has a professional Chart Builder (drag-and-drop). Our unique categories: meta-analysis (5 types), post-hoc visualizations (6 types), diagnostic accuracy (6 types), and agreement plots (3 types) — entirely absent from SPSS. SPSS wins on quality control charts, chart templates, and export variety.

---

## 12. Data Import & Management

| Feature | Our App | SPSS |
|---------|---------|------|
| CSV import | ✅ | ✅ |
| Excel import | ✅ | ✅ |
| SAS / Stata / R import | ❌ Not available | ✅ |
| SPSS .sav files | ❌ Not available | ✅ Native format |
| **Database connectors** | ❌ Not available | ✅ SQL, ODBC, etc. |
| **Data transformation** | ⚠ Basic (structure detection only) | ✅ **Extensive** (compute, recode, rank, lag, etc.) |
| Variable labeling | ❌ Not available | ✅ Variable + value labels |
| Missing value coding | ❌ Not available | ✅ Systematic missing value handling |
| **Data aggregation** | ❌ Not available | ✅ Aggregate, split file |
| **Data merging** (merge files, add cases) | ❌ Not available | ✅ |
| **Weighting** | ❌ Not available | ✅ Weight cases |
| **Built-in datasets** | ✅ 20 curated datasets | ✅ Many sample datasets |
| **Simulated data** | ✅ Dual-mode (sliders for education) | ✅ Random number generation |
| **Stacking/unstacking** | ❌ Not available | ✅ Restructure Data Wizard |
| **Date/time handling** | ❌ Not available | ✅ Date/time wizard |
| **Automated data preparation** | ❌ Not available | ✅ Data Preparation add-on |

**Summary:** SPSS has **vastly superior data management** — database connectors, transformations, labeling, merging, weighting, and the Data Preparation add-on. Our app's data management is minimal (CSV/Excel upload, structure detection). The dual-mode design (simulated defaults with real data as option) is a unique educational feature.

---

## 13. Decision Trees & Machine Learning

| Feature | Our App | SPSS |
|---------|---------|------|
| Decision Trees (CHAID, CART, QUEST) | ❌ Not available | ✅ Decision Trees add-on |
| Neural Networks | ❌ Not available | ✅ Neural Networks add-on |
| Random Forests | ❌ Not available | ❌ Not in SPSS (in SPSS Modeler) |
| SVM | ❌ Not available | ❌ Not in SPSS |
| K-means clustering | ⚠ Basic visualization in graph explorer | ✅ Base |
| Hierarchical clustering | ❌ Not available | ✅ Base |
| Two-step clustering | ❌ Not available | ✅ Base |
| Discriminant analysis | ❌ Not available | ✅ Base |
| **Automated machine learning** | ❌ Not available | ❌ Not in SPSS (in SPSS Modeler) |

**Summary:** SPSS has more ML capabilities (decision trees, neural networks, hierarchical clustering, discriminant analysis). Our app has only basic K-means visualization. SPSS Modeler (separate product) handles advanced ML.

---

## 14. Complex Survey / Research Design

| Feature | Our App | SPSS |
|---------|---------|------|
| Complex sampling design | ❌ Not available | ✅ Complex Samples add-on |
| Sampling weights | ❌ Not available | ✅ Base (Weight) + Complex Samples |
| Stratified/cluster sampling | ❌ Not available | ✅ Complex Samples |
| Conjoint analysis | ❌ Not available | ✅ Conjoint add-on |
| Direct marketing (RFM, etc.) | ❌ Not available | ✅ Direct Marketing add-on |
| Missing data (multiple imputation) | ❌ Not available | ✅ Missing Values add-on |
| Bootstrapping | ❌ Not available | ✅ Bootstrapping add-on |
| **Exact tests (small samples)** | ✅ Built into each widget | ✅ Exact Tests add-on |

**Summary:** SPSS dominates in research design features (complex sampling, conjoint, direct marketing, missing data, bootstrapping). Our app has no equivalent for most of these — a major gap for survey/market researchers.

---

## 15. Educational Features (our unique advantage)

| Feature | Our App | SPSS |
|---------|---------|------|
| **Test Finder Wizard** | ✅ 6-step interactive wizard | ❌ Not available |
| **Interactive test widgets** | ✅ **71 widgets** with real-time sliders | ❌ Static output |
| **Interactive graph explorer** | ✅ **83 graph types** with real-time sliders | ❌ Static graphs |
| **Step-by-step solved examples** | ✅ 7 examples, LaTeX formulas, full walkthrough | ❌ Not available |
| **Glossary** | ✅ ~150 terms, 14 categories | ❌ Not available |
| **Educational tooltips** | ✅ Built into every feature | ❌ Not available |
| **"When to use this test"** | ✅ Every widget | ❌ Not available |
| **Effect size interpretations** | ✅ Automatic text (Trivial/Small/Medium/Large) | ⚠ Numerical only |
| **CLT Simulator** | ✅ Interactive CLT demonstration | ❌ Not available |
| **Distribution Overlay Comparison** | ✅ Side-by-side distribution explorer | ❌ Not available |
| **Sampling Simulator** | ✅ Interactive sampling | ❌ Not available |
| **Educational Table Explorers** | ✅ 6 modules | ❌ Not available |
| **APA-formatted export** | ✅ Tables + CSV/LaTeX/HTML | ⚠ Basic formatting via Custom Tables |
| **Free + zero install** | ✅ **Web browser** | ❌ Pay + install |
| **AI Output Assistant** | ❌ Not available | ✅ **v32** — translates output to plain language |

**Summary:** This is our app's strongest differentiator — it's designed as a learning tool with interactive widgets, interpretations, glossary, step-by-step examples, and educational simulators. SPSS has none of these (v32 adds AI Output Assistant for plain-language interpretations, partially bridging the gap). Our app's **interactive approach is fundamentally different** from SPSS's static output.

---

## 16. Cross-Cutting Summary

| Category | Our App | SPSS | Advantage |
|----------|---------|------|-----------|
| **Cost** | **Free** | $105–$300+/month | **Our App** |
| **Installation** | **Zero** (web browser) | Desktop install | **Our App** |
| **Hypothesis Tests** | **71 tests, all interactive** | 25-30 Base + add-ons (static) | **Our App** (breadth + interactivity) |
| **Nonparametric Post-Hoc** | ✅ **6 methods** | ❌ **None** | **Our App** |
| **Power Analysis** | **30 types, 5 modes, adjustments** | ~15 types, 3 modes | **Our App** |
| **Graphs** | **83 types, all interactive** | ~50 types, static | **Our App** (count + interactivity) |
| **Educational Features** | ✅ **Extensive** | ❌ None (AI Assistant in v32) | **Our App** |
| **Regression** | 8 types | 15+ types | **SPSS** (nonlinear, probit, 2SLS, ridge/lasso, mixed models) |
| **ANOVA / GLM** | Basic coverage | **Complete** (GLM, mixed, GENLIN, GEE, GLMM) | **SPSS** |
| **Multivariate** | Basic (PCA, cluster viz) | **Complete** (factor, discriminant, correspondence, MDS, CATREG) | **SPSS** |
| **Time Series** | ❌ None | **Complete** (ARIMA, ES, decomposition, expert modeler) | **SPSS** |
| **SPC / Quality** | ❌ None | ✅ Control charts in Base | **SPSS** |
| **Data Management** | Basic (CSV/Excel) | **Professional** (transformations, labeling, merging, DB connectors) | **SPSS** |
| **Scripting / Automation** | ❌ None | ✅ Syntax, Python, R | **SPSS** |
| **Missing Data** | ❌ None | ✅ **Multiple imputation, EM, pattern analysis** (add-on) | **SPSS** |
| **Complex Surveys** | ❌ None | ✅ **Complex Samples add-on** | **SPSS** |
| **Decision Trees / ML** | ❌ None | ✅ Decision Trees, Neural Networks (add-ons) | **SPSS** |
| **Research Design** | ❌ None | ✅ Conjoint, Direct Marketing (add-ons) | **SPSS** |
| **Collaboration** | ❌ None | ✅ Project files, shared output | **SPSS** |
| **AI Assistance** | ❌ None | ✅ AI Output Assistant (v32) | **SPSS** |

---

## 17. Strategic Summary

### Our App's Unique Value Proposition (vs SPSS)

| What we do better | What SPSS does better |
|-------------------|----------------------|
| **Interactive widgets** (71 tests, 83 graphs — all real-time sliders) | **Data management** (transformations, labeling, merging, DB connectors) |
| **Educational design** (interpretations, test finder, glossary, examples) | **Regression** (nonlinear, probit, 2SLS, mixed models, GENLIN, GEE, GLMM) |
| **Power analysis** (30 types vs ~15, 5 modes vs 3, adjustments, converter) | **Multivariate analysis** (factor, discriminant, correspondence, MDS, CATREG) |
| **Post-hoc methods** (23 methods + 6 visualizations vs basic set) | **Time series** (ARIMA, exponential smoothing, expert modeler, decomposition) |
| **Nonparametric post-hoc** (6 methods vs **none** in SPSS) | **Missing data** (multiple imputation, EM, pattern analysis) |
| **Diagnostic accuracy** (8+ clinical features) | **Complex surveys** (sampling weights, design effects) |
| **Meta-analysis** (5 plot types — SPSS has none) | **Scripting/automation** (syntax, Python, R integration) |
| **APA export** (CSV/LaTeX/HTML tables) | **Enterprise data** (large datasets, server deployment) |
| **Completely free + web-based** | **Decision Trees / ML** (CHAID, CART, QUEST, neural nets) |
| **Built-in datasets** (20 curated for learning) | **Collaboration** (project files, shared output) |
| **CLT Simulator, Sampling Simulator, Distribution Overlay** | **AI Output Assistant** (v32 — natural language explanations) |

### Features in Our App with No SPSS Equivalent

| Feature | What it does |
|---------|-------------|
| **Test Finder Wizard** | Interactive 6-step questionnaire → test recommendation |
| **71 Interactive Test Widgets** | All with real-time parameter sliders |
| **83 Interactive Graph Types** | All with adjustable parameters |
| **23 Post-Hoc Methods** | Including nonparametric, MANOVA, repeated measures contexts |
| **6 Educational Table Explorers** | Interactive learning modules for cross-tab concepts |
| **CLT Simulator** | Interactive Central Limit Theorem demonstration |
| **Distribution Overlay Comparison** | Compare two distributions side-by-side |
| **Sampling Simulator** | Interactive sampling from distributions |
| **Diagnostic Accuracy Module** | 8+ clinical test evaluation tools |
| **Meta-Analysis Plots** | 5 publication-ready visualizations |
| **Step-by-Step Solved Examples** | Full LaTeX walkthroughs |
| **Glossary with ~150 Terms** | 14 categories, always accessible |
| **Effect Size Converter** | d ↔ r ↔ f ↔ f² ↔ OR ↔ w |
| **6 Nonparametric Post-Hoc Methods** | SPSS provides **zero** nonparametric post-hoc tests |

### Features in SPSS with No Equivalent in Our App

| Feature | What it does |
|---------|-------------|
| **Generalized Linear Mixed Models (GLMM)** | Handles non-normal, correlated, hierarchical data |
| **Linear Mixed Models (HLM)** | Multilevel/hierarchical modeling |
| **Generalized Estimating Equations (GEE)** | Correlated longitudinal data |
| **Nonlinear Regression** | Models with nonlinear parameters |
| **Probit Analysis** | Binary response with stimulus intensity |
| **2-Stage Least Squares** | Instrumental variables regression |
| **Ridge / Lasso / Elastic Net** | Regularized regression |
| **Factor Analysis** | Latent variable discovery |
| **Discriminant Analysis** | Classification with linear boundaries |
| **Correspondence Analysis** | Categorical data dimension reduction |
| **Categorical Regression (CATREG)** | Optimal scaling regression |
| **Conjoint Analysis** | Preference measurement |
| **ARIMA / Exponential Smoothing** | Time series forecasting |
| **Expert Modeler** | Automatic ARIMA/ES model selection |
| **Multiple Imputation** | Missing data handling |
| **Complex Samples** | Survey data with design effects |
| **Decision Trees (CHAID, CART, QUEST)** | Tree-based classification |
| **Neural Networks** | Deep learning models |
| **Control Charts** | Statistical process control |
| **Data transformations** | Compute, recode, rank, lag, aggregate |
| **Syntax / Scripting** | Reproducible analysis pipelines |
| **AI Output Assistant** | Natural language interpretation |

---

## 18. Feature Coverage by Edition

This shows what you can access at each price point vs our app:

| Feature Area | Our App | SPSS Base (~$1,500/yr) | SPSS Professional (+~$1,000/yr) | SPSS Premium (+~$2,000+/yr) |
|--------------|---------|------------------------|--------------------------------|-----------------------------|
| **Cost** | **$0** | ~$1,500/yr | ~$2,500/yr | ~$4,500+/yr |
| **Basic stats + t-tests** | ✅ | ✅ | ✅ | ✅ |
| **ANOVA (one-way)** | ✅ | ✅ | ✅ | ✅ |
| **Nonparametric tests** | ✅ | ✅ (basic) | ✅ (basic) | ✅ (basic) |
| **Correlation** | ✅ | ✅ | ✅ | ✅ |
| **Linear regression** | ✅ | ✅ | ✅ | ✅ |
| **Graphs** | ✅ **83 interactive** | ✅ ~50 static | ✅ ~50 static | ✅ ~50 static |
| **Power analysis** | ✅ **30 types** | ✅ ~15 types | ✅ ~15 types | ✅ ~15 types |
| **Meta-analysis** | ✅ **5 plot types** | ✅ Base | ✅ Base | ✅ Base |
| **Custom Tables** | ⚠ Basic + APA export | ❌ Add-on | ❌ Add-on | ❌ Add-on |
| **Advanced Statistics (GLM, mixed, survival)** | ⚠ Basic version (ANOVA, Cox) | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Logistic regression** | ✅ Binary + Multinomial + Ordinal | ❌ Add-on | ✅ Included | ✅ Included |
| **Forecasting** | ❌ None | ❌ Add-on | ✅ Included | ✅ Included |
| **Decision Trees** | ❌ None | ❌ Add-on | ✅ Included | ✅ Included |
| **Complex Samples** | ❌ None | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Missing Values** | ❌ None | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Exact Tests** | ✅ Built-in | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Categories** | ❌ None | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Neural Networks** | ❌ None | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Conjoint** | ❌ None | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Bootstrapping** | ❌ None | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Data Preparation** | ❌ None | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Direct Marketing** | ❌ None | ❌ Add-on | ❌ Add-on | ✅ Included |
| **Nonparametric Post-Hoc** | ✅ **6 methods** | ❌ **None at any tier** | ❌ | ❌ |
| **Educational features** | ✅ **Extensive** | ❌ None | ❌ None | ❌ None |
| **AI Output Assistant** | ❌ None | ❌ (v32) | ❌ (v32) | ✅ (v32) |

---

## 19. Suggested Development Priorities (to close gaps with SPSS)

| Priority | Feature to Add | Effort | Impact |
|----------|---------------|--------|--------|
| 1 | **Missing data handling** (imputation, MCAR/MAR/MNAR diagnostics) | Medium | Major for research users |
| 2 | **Time series plots** (ACF/PACF, trend, decomposition) | Medium | Useful across many fields |
| 3 | **Factor analysis + Cronbach's alpha** | Medium | Opens psychometrics/survey audience |
| 4 | **Multiple imputation** | High | Complements missing data |
| 5 | **Hierarchical/multilevel modeling (HLM) basics** | Very High | Advanced statistics education |
| 6 | **Data transformation** (Box-Cox, log, sqrt, standardize) | Low | Complements diagnostics |
| 7 | **Basic control charts** (XBar, R, P) | High | SPC education |
| 8 | **Nonlinear regression** | Medium | Biology/chemistry education |
| 9 | **Syntax/scripting for reproducibility** | High | Research use |
| 10 | **Ridge/Lasso/Elastic Net regression** | Medium | ML education |
| 11 | **Correspondence analysis** | Medium | Market research education |
| 12 | **Conjoint analysis basics** | High | Market research |

---

*Generated June 2026. Our app: ~40,000 lines across 8 apps (1 legacy), 14 feature modules, 4 core modules. SPSS: IBM SPSS Statistics v32 (released 2026), 10+ add-on modules.*

*Bottom line: Our app covers roughly 40% of SPSS's feature set (higher in hypothesis testing / power / post-hoc, much lower in data management / multivariate / time series / mixed models / survey design). Our interactive educational features, nonparametric post-hoc methods, meta-analysis, and comprehensive power analysis have **no SPSS equivalent at any price tier**.*
