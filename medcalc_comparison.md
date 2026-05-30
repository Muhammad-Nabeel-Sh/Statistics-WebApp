# Feature Comparison: Statistics WebApp vs MedCalc

> **Our App** = Statistics WebApp (free, web-based, educational)
> **MedCalc** = Commercial biomedical statistics software (paid perpetual license, Windows desktop)

**Key difference in philosophy:**
- **Our App**: Educational-first, interactive real-time widgets, dual-mode data (simulated + uploaded), interpretive guidance built-in, runs in any browser
- **MedCalc**: Professional biomedical research tool, Windows desktop with integrated spreadsheet, reference-standard ROC and method-comparison tools, scripting for automation

---

## 0. Architecture & Pricing

| Dimension | Our App | MedCalc |
|-----------|---------|---------|
| Cost | **Free** (zero cost) | **€1,195** (single-user perpetual license) |
| Licensing model | Open-source, self-host or Streamlit Cloud | Perpetual license, free updates, 15-day trial |
| Platform | **Web browser** (cross-platform) | **Windows only** (no Mac/Linux) |
| Deployment | Streamlit Cloud or self-hosted | On-premises desktop |
| Scripting/automation | ❌ None | ✅ **Built-in scripting language** (automation, bootstrap, file I/O) |
| Data row limit | Moderate (in-memory, ~100K rows) | **1,048,576 rows × 16,384 columns** (spreadsheet) |
| Built-in spreadsheet | ❌ Uses pandas DataFrames | ✅ **Full integrated spreadsheet** with formulas, filters, transformations |
| Collaboration | ❌ None | ✅ Free file viewer (.mc1) for sharing results |
| Mobile support | ✅ Any browser | ❌ Desktop only |
| Auto-update | ✅ Via git/deploy | ✅ Built-in auto-update |

---

## 1. Statistical Tests — Core Coverage

| Test | Our App | MedCalc | Notes |
|------|---------|---------|-------|
| **One-sample t-test** | ✅ Interactive widget | ✅ | Both have effect sizes, CI |
| One-sample z-test | ✅ Interactive widget | ❌ Not listed | Rare in medical stats |
| One-sample proportion (Binomial) | ✅ Normal approx + Exact | ✅ | Both have exact binomial |
| **Two-sample independent t-test** | ✅ Student's + Welch's | ✅ Welch correction optional | |
| Paired t-test | ✅ Interactive widget | ✅ | |
| Two-sample proportion test | ❌ Not available | ✅ Tests on summarized data | MedCalc can compute from 2×2 |
| F-Test for Two Variances | ✅ Interactive widget | ✅ Variance ratio test | |
| Equivalence Test (TOST) | ✅ Two independent samples | ❌ Not listed | MedCalc lacks TOST |
| **One-way ANOVA** | ✅ Interactive widget, post-hoc | ✅ | |
| Two-way ANOVA | ✅ Interactive widget | ✅ | |
| Repeated Measures ANOVA | ✅ Interactive widget | ✅ | |
| ANCOVA | ✅ Interactive widget | ✅ | |
| MANOVA | ✅ Interactive widget | ⚠ PCA only | MedCalc lacks full MANOVA |
| **Kruskal-Wallis** | ✅ Interactive widget | ✅ | |
| Friedman Test | ✅ Interactive widget | ✅ | |
| Cochran's Q | ✅ Interactive widget | ✅ | |
| Mann-Whitney U | ✅ Interactive widget | ✅ | |
| Wilcoxon Signed-Rank (paired) | ✅ Interactive widget | ✅ | |
| Sign Test (one/paired) | ✅ Interactive widget | ❌ Not listed | |
| Mood's Median Test | ✅ Interactive widget | ❌ Not listed | |
| Permutation MANOVA | ✅ Interactive widget | ❌ Not listed | |

**Summary:** Core parametric/nonparametric coverage is comparable. Our app has a slight edge in breadth (z-test, sign test, TOST, permutation MANOVA). MedCalc has a dedicated "Tests on Summarized Data" menu for entering means/SDs/proportions directly (useful for published data).

---

## 2. Categorical Data Analysis

| Test | Our App | MedCalc |
|------|---------|---------|
| **Chi-square test of independence** | ✅ Interactive widget | ✅ |
| Chi-square goodness-of-fit | ✅ Interactive widget | ✅ |
| Fisher's exact test | ✅ Interactive widget | ✅ |
| McNemar's test | ✅ Interactive widget | ✅ |
| Relative risk / Odds ratio | ✅ In tabulation module | ✅ Dedicated menu |
| Cochran-Mantel-Haenszel | ❌ Not available | ✅ Stratified analysis |
| Poisson goodness-of-fit | ✅ Interactive widget | ❌ Not listed |
| Multinomial test (exact) | ✅ Interactive widget | ❌ Not listed |
| Cohen's Kappa | ✅ Interactive widget | ✅ Inter-rater agreement |
| Fleiss' Kappa | ✅ Interactive widget | ❌ Not listed |
| Weighted Kappa | ✅ Interactive widget | ❌ Not listed |
| Cronbach's alpha | ✅ In factor analysis | ✅ Reliability analysis |
| Intraclass correlation (ICC) | ❌ Not available | ✅ ICC with CI |

**Summary:** Our app has wider categorical test breadth (multinomial exact, Fleiss/weighted kappa, Poisson GOF). MedCalc has CMH test and ICC which we lack.

---

## 3. Regression & Correlation

| Test | Our App | MedCalc |
|------|---------|---------|
| **Pearson correlation** | ✅ Interactive widget | ✅ |
| Spearman rank correlation | ✅ Interactive widget | ✅ |
| Kendall's Tau-b | ✅ Interactive widget | ✅ Partial correlation |
| Point-biserial correlation | ✅ Interactive widget | ❌ Not listed |
| **Simple linear regression** | ✅ Interactive widget | ✅ |
| **Multiple linear regression** | ✅ Interactive widget | ✅ |
| Logistic regression (binary) | ✅ Interactive widget | ✅ |
| Multinomial logistic regression | ✅ Interactive widget | ❌ Not listed |
| Ordinal logistic regression | ✅ Interactive widget | ❌ Not listed |
| Poisson / Negative binomial regression | ✅ Interactive widget | ❌ Not listed |
| Cox proportional hazards | ✅ Interactive widget | ✅ Survival analysis |
| Non-linear regression | ❌ Not available | ✅ Dedicated module |
| Polynomial regression | ❌ Not available | ✅ Built into linear regression |
| Passing-Bablok regression | ❌ Not available | ✅ **Best-in-class method comparison** |
| Deming regression | ❌ Not available | ✅ **Clinical chemistry standard** |

**Summary:** Our app has more regression families (multinomial, ordinal, Poisson, neg. binomial). MedCalc has Method Comparison regression (Passing-Bablok, Deming) which are essential in clinical chemistry and completely absent from our app.

---

## 4. ROC Curve Analysis ⭐ (MedCalc's Signature Feature)

| Feature | Our App | MedCalc |
|---------|---------|---------|
| **ROC curve with AUC** | ✅ Basic ROC widget | ✅ **Reference-standard implementation** |
| Sensitivity / Specificity at all thresholds | ✅ | ✅ |
| Youden index (J) | ❌ Not available | ✅ With BCa bootstrap CI |
| DeLong SE method | ❌ Always Hanley | ✅ Choose DeLong or Hanley & McNeil |
| Binomial exact CI for AUC | ❌ Not available | ✅ **Recommended for clinical use** |
| 95% confidence bounds on ROC curve | ❌ Not available | ✅ Plotted as shaded region |
| Comparison of up to 6 ROC curves | ❌ Not available | ✅ **Correlated + independent** comparison |
| Partial AUC analysis | ❌ Not available | ✅ **Partial area under curve** |
| Precision-Recall curves | ❌ Not available | ✅ With comparison |
| Interactive dot diagram (threshold selection) | ❌ Not available | ✅ **Interactive threshold explorer** |
| Sensitivity/specificity vs criterion plot | ❌ Not available | ✅ |
| Predictive values vs prevalence plot | ❌ Not available | ✅ |
| Interval likelihood ratios | ❌ Not available | ✅ |
| BCa bootstrap CIs for fixed specificity | ❌ Not available | ✅ **Advanced bootstrap methodology** |
| Optimal criterion with cost/prevalence | ❌ Not available | ✅ Weighted by misclassification costs |
| Sample size for ROC analysis | ❌ Not available | ✅ AUC + comparison of curves |
| **Sample size for comparison of ROC curves** | ❌ Not available | ✅ |

**Summary:** ROC analysis is the **single biggest gap** between our app and MedCalc. MedCalc is widely considered the reference software for ROC analysis in biomedical research. Our app has only a basic ROC widget. Adding advanced ROC features (DeLong comparison, partial AUC, bootstrap CIs, interactive dot diagram, Youden index, precision-recall) would be the highest-impact enhancement for clinical users.

---

## 5. Method Comparison / Bland-Altman ⭐ (MedCalc's Signature Feature)

| Feature | Our App | MedCalc |
|---------|---------|---------|
| **Bland-Altman plot** | ✅ Basic widget | ✅ **3 methodologies** (parametric, non-parametric, regression-based) |
| Multiple measurements per subject | ❌ Not available | ✅ Bland & Altman 2007 extension |
| Comparison of multiple methods | ❌ Not available | ✅ Krouwer method, aligned bias plots |
| Passing-Bablok regression | ❌ Not available | ✅ **Non-parametric method comparison** (CLSI EP09c) |
| Deming regression | ❌ Not available | ✅ Error-in-variables regression |
| Mountain plot | ❌ Not available | ✅ Folded empirical CDF |
| Coefficient of variation from duplicates | ❌ Not available | ✅ |
| Concordance correlation coefficient | ❌ Not available | ✅ |
| Absolute percentage error (APE) | ❌ Not available | ✅ MdAPE, 95th percentile APE |
| Area between LoA curves (LoAA) | ❌ Not available | ✅ Summary disagreement measure |

**Summary:** Method comparison is MedCalc's **second signature strength**. Our app has a single Bland-Altman widget. Adding Passing-Bablok regression, Deming regression, multiple-measurement BA, and mountain plots would be essential for clinical chemistry/laboratory medicine use cases.

---

## 6. Survival Analysis

| Feature | Our App | MedCalc |
|---------|---------|---------|
| **Kaplan-Meier curves** | ✅ Widget + graph explorer | ✅ Up to 6 subgroups |
| Log-rank test | ✅ Widget | ✅ |
| Log-rank test for trend | ❌ Not available | ✅ Across ordered groups |
| **Cox proportional hazards** | ✅ Widget | ✅ |
| Hazard ratios with 95% CI | ✅ | ✅ |
| Mean/median survival time | ⚠ Via KM output | ✅ With 95% CI (Brookmeyer & Crowley) |
| Restricted Mean Survival Time (RMST) | ❌ Not available | ✅ Alternative to Cox when PH violated |
| Number at risk table below curve | ❌ Not available | ✅ Standard Kaplan-Meier output |
| Nelson-Aalen plot | ✅ In graph explorer | ❌ Not listed |
| Cumulative hazard plot | ✅ In graph explorer | ❌ Not listed |
| Survival probability heatmap | ✅ In graph explorer | ❌ Not listed |

**Summary:** Comparable core survival features (KM, log-rank, Cox). Our app has more survival visualization variety (Nelson-Aalen, heatmap). MedCalc has RMST (important when PH assumption fails) and number-at-risk tables.

---

## 7. Meta-Analysis

| Feature | Our App | MedCalc |
|---------|---------|---------|
| **Generic inverse variance** | ❌ Not available | ✅ Fixed + random effects |
| **Meta-analysis of AUC** | ❌ Not available | ✅ Summary ROC (SROC) |
| Forest plot | ✅ In graph explorer | ✅ |
| Funnel plot | ✅ In graph explorer | ✅ |
| Galbraith (radial) plot | ✅ In graph explorer | ❌ Not listed |
| Baujat plot | ✅ In graph explorer | ❌ Not listed |
| Leave-one-out plot | ✅ In graph explorer | ❌ Not listed |
| Heterogeneity (Cochran's Q, I²) | ❌ Not available | ✅ |
| Pooled effect sizes | ❌ Not available | ✅ With fixed/random models |

**Summary:** Our app has meta-analysis **visualization plots** (forest, funnel, Galbraith, Baujat, LOO) but no **computational meta-analysis engine**. MedCalc computes pooled effects, heterogeneity, and SROC meta-analysis. Adding the computation layer would turn our plots into a full meta-analysis tool.

---

## 8. Sample Size / Power Analysis

| Feature | Our App | MedCalc |
|---------|---------|---------|
| **One-sample mean** | ✅ A priori, Post-hoc, Sensitivity, Compromise, Criterion | ✅ |
| **Two independent means** | ✅ All 5 analysis modes | ✅ |
| **Paired means** | ✅ All 5 modes | ❌ Not listed |
| **One-sample proportion** | ✅ | ✅ |
| **Two proportions** | ✅ | ✅ |
| **One-way ANOVA** | ✅ | ❌ Not listed |
| **Correlation (Pearson)** | ✅ | ❌ Not listed |
| **Multiple linear regression** | ✅ | ❌ Not listed |
| **Logistic regression** | ✅ | ❌ Not listed |
| **Chi-square test** | ✅ | ❌ Not listed |
| **Equivalence / Non-inferiority** | ✅ | ❌ Not listed |
| **Repeated measures ANOVA** | ✅ | ❌ Not listed |
| **Two-way / Factorial ANOVA** | ✅ | ❌ Not listed |
| **ROC / AUC analysis** | ❌ Not available | ✅ **ROC sample size** (single + comparison) |
| **Survival (log-rank)** | ✅ | ✅ |
| **Cox regression** | ✅ | ❌ Not listed |
| **Cluster-RCT / Multilevel** | ✅ | ❌ Not listed |
| **Precision-based (CI width)** | ✅ | ✅ |
| **Kappa / ICC agreement** | ✅ | ❌ Not listed |
| **Non-parametric (MW, Wilcoxon)** | ✅ | ❌ Not listed |
| **Kruskal-Wallis / Friedman** | ✅ | ❌ Not listed |
| **McNemar / Fisher exact** | ✅ | ❌ Not listed |
| **MANOVA** | ✅ | ❌ Not listed |
| **Binomial exact** | ✅ | ❌ Not listed |
| **Simulation-based power (Monte Carlo)** | ✅ | ❌ Not listed |
| **Reference limits sample size** | ❌ Not available | ✅ Bellera & Hanley method |
| **Power curve visualization** | ✅ Interactive | ❌ Not listed |
| **Sensitivity analysis heatmap** | ✅ | ❌ Not listed |
| **Budget & feasibility** | ✅ | ❌ Not listed |
| **What-if scenario explorer** | ✅ | ❌ Not listed |
| **Effect size converter** | ✅ | ❌ Not listed |
| **Sample size justification text** | ✅ | ❌ Not listed |

**Summary:** Our **power analysis module is dramatically more comprehensive** than MedCalc's. We cover 29 test types in 5 analysis modes with interactive curves, heatmaps, Monte Carlo, budget planning, and justification text. MedCalc covers only ~8 test types. This is our strongest comparative advantage.

---

## 9. Reference Intervals (MedCalc Signature Feature)

| Feature | Our App | MedCalc |
|---------|---------|---------|
| Standard reference intervals (90/95/99%) | ❌ Not available | ✅ Normal, non-parametric, robust (CLSI C28-A3) |
| Age-related reference intervals | ❌ Not available | ✅ **Polynomial regression with centile charts** |
| Box-Cox transformation | ✅ In data diagnostics | ✅ Built into reference interval engine |
| Outlier testing (Reed/Tukey) | ❌ Not available | ✅ Before reference limit calculation |
| z-score analysis for model evaluation | ❌ Not available | ✅ Normality test, skewness, scatter plots |
| Bootstrap CI for reference limits | ❌ Not available | ✅ Wright & Royston bootstrap method |

**Summary:** Reference intervals are a signature MedCalc feature for clinical laboratory medicine, entirely absent from our app. Adding them would serve the clinical chemistry audience.

---

## 10. Data Management

| Feature | Our App | MedCalc |
|---------|---------|---------|
| **Data upload** | ✅ CSV, Excel | ✅ CSV, Excel, SPSS, DBase, Lotus, SYLK, DIF |
| **Built-in datasets** | ✅ **30 curated datasets** | ❌ No built-in datasets |
| **Data cleaning** | ✅ NA handling, drop cols, filter rows, type conversion | ✅ Outlier exclusion, filter system |
| **Column transformation** | ✅ Eval-based expressions, 50+ expression gallery | ✅ LOG, SQRT, Box-Cox, categorisation |
| **Computed columns** | ✅ Eval-based expressions | ✅ Formula-based columns in spreadsheet |
| **Categorical → numeric mapping** | ✅ Manual, ordinal encode, one-hot encode | ⚠ Via recoding |
| **Pivot / Reshape (melt)** | ✅ Long-to-Wide + Wide-to-Long | ✅ Stack columns tool |
| **Data editing** | ✅ AG Grid (sort, filter, edit inline) | ✅ Spreadsheet (formulas, cell addressing) |
| **Filter rows** | ✅ Numeric range + categorical multiselect | ✅ Logical AND/OR filter system |
| **Group-by aggregation** | ✅ Mean, Sum, Count, Median, Min, Max, Std, Var | ✅ Via spreadsheet statistics |
| **Deduplication** | ✅ Detection + removal | ❌ Not listed |
| **Duplicate measurements CV** | ❌ Not available | ✅ Coefficient of variation from duplicates |
| **Define status / recode** | ❌ Not available | ✅ Dichotomous recoding tool |
| **Create groups from continuous** | ❌ Bin slider in some widgets | ✅ Dedicated tool |

**Summary:** Our app has better built-in educational datasets and a cleaner web-based data editing experience (AG Grid). MedCalc has a more powerful spreadsheet engine (formulas, cell addressing) and more import formats. Both are roughly comparable for basic data management.

---

## 11. Graphics / Visualization

| Feature | Our App | MedCalc |
|---------|---------|---------|
| **Histogram** | ✅ + density overlay | ✅ + normal curve |
| **Boxplot / Violin plot** | ✅ + raincloud, beeswarm | ✅ + notched boxplot |
| **Scatterplot** | ✅ + bubble, hexbin, SPLOM | ✅ + regression line, CI |
| **Bar chart** | ✅ Grouped, stacked, error bars | ✅ Error bars, multiple variables |
| **Pie chart** | ✅ | ✅ |
| **Line chart / Time series** | ✅ | ✅ |
| **Correlation heatmap** | ✅ | ❌ Not listed |
| **Q-Q / P-P plot** | ✅ | ❌ Not listed |
| **3D scatter / contour / parallel coordinates** | ✅ | ❌ Not listed |
| **ROC curve with confidence bounds** | ❌ Not available | ✅ **Shaded 95% CI region** |
| **Bland-Altman (3 variants)** | ❌ Basic only | ✅ Parametric + non-parametric + regression |
| **Passing-Bablok scatter** | ❌ Not available | ✅ |
| **Kaplan-Meier with at-risk table** | ❌ Not available | ✅ |
| **Forest / Funnel plot** | ✅ | ✅ |
| **Cleveland dot plot** | ✅ | ❌ Not listed |
| **Sankey diagram** | ✅ | ❌ Not listed |
| **Ridgeline plot** | ✅ | ❌ Not listed |
| **Polar density / population pyramid** | ✅ | ❌ Not listed |
| **Interactive graph builder** | ✅ **Custom graph builder** | ❌ Fixed graph types only |
| **Export formats** | ✅ PNG, SVG, HTML | ✅ SVG, PNG, JPG, GIF, BMP, PCX, TIF, **PowerPoint (PPTX)** |
| **Graph annotations** | ⚠ Limited | ✅ Text boxes, lines, arrows, connectors |
| **Dark theme by default** | ✅ plotly_dark template | ✅ Dark template aesthetic |

**Summary:** Our app has **more visualization variety** (50+ graph types vs MedCalc's ~20) including interactive 3D, Sankey, ridgeline, polar plots, and a custom graph builder. MedCalc has better graph annotations and PPTX export. **The biggest gap is specialized clinical graphs** (ROC with CI bounds, Bland-Altman variants, Passing-Bablok, KM with at-risk table) where MedCalc excels.

---

## 12. Quality Control / SPC

| Feature | Our App | MedCalc |
|---------|---------|---------|
| Xbar-R Chart | ✅ | ❌ Not listed |
| Xbar-S Chart | ✅ | ❌ Not listed |
| I-MR Chart | ✅ | ❌ Not listed |
| p-Chart / np-Chart | ✅ | ❌ Not listed |
| c-Chart / u-Chart | ✅ | ❌ Not listed |
| CUSUM Chart | ✅ | ❌ Not listed |
| EWMA Chart | ✅ | ❌ Not listed |
| Shewhart rule violations (Rules 1-8) | ✅ | ❌ Not listed |
| Process capability (Cp, Cpk, Pp, Ppk, Cpm) | ✅ | ❌ Not listed |
| Phase 1 / Phase 2 analysis | ✅ | ❌ Not listed |
| **Westgard multi-rules** | ❌ Not available | ✅ **1:2S, 1:3S, 2:2S, 4:1S, 10:X** |
| Custom control chart (blank form) | ❌ Not available | ✅ Printable manual recording form |

**Summary:** Our SPC module is **far more comprehensive** than MedCalc's (9 chart types + capability + Shewhart rules). MedCalc's strength is Westgard rules for clinical lab QC, which we lack.

---

## 13. Educational & Exploratory Features (Our App's Strengths)

| Feature | Our App | MedCalc |
|---------|---------|---------|
| **Interactive test finder** | ✅ Filter by objective, variables, design → get test recommendation | ❌ Not available |
| **100+ statistical definitions** (glossary) | ✅ 13 sections, searchable | ❌ No glossary |
| **Step-by-step solved examples** | ✅ 7 one-sample tests with full worked solutions | ❌ Not available |
| **Probability distribution explorer** | ✅ 20+ distributions with PMF, CDF, sampling simulation, CLT demo | ❌ Not available |
| **Graph builder** | ✅ Custom drag-and-drop graph construction | ❌ Not available |
| **Data transformation explorer** | ✅ 11 transformations with before/after comparison | ❌ Not available |
| **Educational wide-vs-long explorer** | ✅ Interactive pivoting/melting/transposing demo | ❌ Not available |
| **Bayesian updating table** | ✅ Interactive Bayes theorem with population simulation | ❌ Not available |
| **Frequency / Cross-tab / Expected frequency explorers** | ✅ 6 interactive modules | ❌ Not available |
| **Built-in dataset library** | ✅ 30 datasets with semantic descriptions | ❌ Not available |
| **APA-format table export** | ✅ | ❌ Not listed |
| **Post-hoc explorer** | ✅ 8 parametric + 6 non-parametric methods with heatmaps | ❌ Limited Tukey/Dunnett |

**Summary:** Our app's **educational and exploratory features are unmatched** by MedCalc. The interactive test finder, distribution explorer, solved examples, glossary, transformation explorer, and Bayesian updater make it a powerful learning tool. MedCalc has none of these — it's a professional tool, not a teaching platform.

---

## 14. Summary: What MedCalc Has That We Don't (High-Impact Gaps)

| Priority | Feature | Clinical Use Case |
|----------|---------|-------------------|
| 🔴 **Critical** | **Advanced ROC analysis** (DeLong comparison, partial AUC, Youden index, BCa bootstrap, precision-recall, interactive dot diagram) | Diagnostic test evaluation, biomarker discovery |
| 🔴 **Critical** | **Method comparison suite** (Passing-Bablok, Deming, multiple-measurement BA, mountain plot) | Clinical chemistry, lab instrument validation |
| 🟠 **High** | **Reference intervals** (age-related polynomial regression, centile charts, CLSI C28-A3) | Laboratory medicine reference ranges |
| 🟠 **High** | **Westgard QC rules** (1:2S, 1:3S, 2:2S, 4:1S, 10:X) | Clinical laboratory QC |
| 🟡 **Medium** | **Meta-analysis computation** (pooled effects, heterogeneity, SROC) | Evidence synthesis, diagnostic test accuracy meta-analysis |
| 🟡 **Medium** | **RMST (Restricted Mean Survival Time)** | Survival analysis when PH assumption fails |
| 🟡 **Medium** | **Intraclass correlation (ICC)** | Reliability studies |
| 🟡 **Medium** | **Cochran-Mantel-Haenszel test** | Stratified categorical analysis |
| 🟡 **Medium** | **Passing-Bablok & Deming regression** | Method comparison (overlaps with #2) |
| 🟢 **Low** | **PowerPoint export, graph annotations** | Publication workflow |

## 15. Summary: What We Have That MedCalc Doesn't (Our Differentiators)

| Category | Our Advantage |
|----------|---------------|
| **Power analysis** | 29 test types × 5 analysis modes × interactive visuals — vastly more comprehensive |
| **SPC / Control charts** | 9 chart types + capability + Shewhart rules vs MedCalc's single chart |
| **Educational features** | Test finder, glossary, solved examples, distribution explorer, Bayesian updater, graph builder — none exist in MedCalc |
| **Graph variety** | 50+ graph types including 3D, Sankey, ridgeline, polar, Cleveland dot, interactive graph builder |
| **Regression breadth** | Multinomial logistic, ordinal logistic, Poisson, negative binomial — MedCalc has none of these |
| **Web platform** | Zero install, cross-platform, mobile-friendly, cloud-deployable |
| **Cost** | Free vs €1,195 |
| **Open source** | Full transparency, forkable, extendable |

## 16. Enhancement Roadmap

Based on this comparison, the following enhancements would most narrow the gap with MedCalc:

### Phase 1 — Highest Impact (Clinical/Diagnostic)
1. **Advanced ROC Analysis** — DeLong comparison of up to 6 curves, partial AUC, Youden index with bootstrap CI, interactive dot diagram, precision-recall curves, BCa bootstrap for sensitivity at fixed specificity
2. **Method Comparison Suite** — Passing-Bablok regression, Deming regression, multiple-measurement Bland-Altman, mountain plot, concordance correlation coefficient

### Phase 2 — Lab Medicine
3. **Reference Intervals** — Normal/robust/non-parametric methods, age-related polynomial regression centile charts, Box-Cox transformation integration
4. **Westgard QC Rules** — Add to existing SPC module for clinical lab compliance

### Phase 3 — Advanced Methods
5. **Meta-Analysis Computation** — Add pooled effect sizes, heterogeneity statistics, SROC to existing forest/funnel plots
6. **RMST for Survival** — Alternative to Cox when PH assumption is violated
7. **Cochran-Mantel-Haenszel** — Stratified categorical analysis
8. **ICC** — Intraclass correlation coefficient for reliability studies

### Phase 4 — Publication Workflow
9. **PowerPoint export** — Direct export of graphs to PPTX slides
10. **Graph annotations** — Text boxes, arrows, reference lines on plots
