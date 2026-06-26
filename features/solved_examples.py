import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb, gammaln


def _step_header(step_num, title):
    st.markdown(f"**Step {step_num}: {title}**")


def _step_formula(formula):
    st.latex(formula)


def _step_result(text):
    st.markdown(f"> {text}")


def _step_subresult(text):
    st.markdown(text)


def _section(title):
    st.subheader(title)
    st.divider()


def render_solved_examples():
    st.title("Step-by-Step Solved Examples")
    st.markdown(
        "Enter your data below to see a complete step-by-step solution for each test."
    )

    test_choice = st.selectbox(
        "Select a test",
        [
            "--- One-sample tests ---",
            "One-sample t-test",
            "One-sample z-test",
            "One-sample Proportion Test (Binomial Test)",
            "Binomial Test (Exact)",
            "One-sample Wilcoxon Signed-Rank Test",
            "Chi-Square Goodness-of-Fit Test",
            "Multinomial Test",
            "--- Two-sample (independent) ---",
            "Student's t-test (Independent)",
            "Welch's t-test (Independent, Unequal Variances)",
            "Yuen's Trimmed t-test",
            "--- Two-sample (paired/dependent) ---",
            "Paired t-test",
            "Sign Test (Paired)",
            "Wilcoxon Signed-Rank Test",
        ],
        key="solved_test_selector",
    )

    if test_choice == "One-sample t-test":
        _solved_ttest()
    elif test_choice == "One-sample z-test":
        _solved_ztest()
    elif test_choice == "One-sample Proportion Test (Binomial Test)":
        _solved_proportion()
    elif test_choice == "Binomial Test (Exact)":
        _solved_binomial_exact()
    elif test_choice == "One-sample Wilcoxon Signed-Rank Test":
        _solved_wilcoxon()
    elif test_choice == "Chi-Square Goodness-of-Fit Test":
        _solved_chisquare_gof()
    elif test_choice == "Multinomial Test":
        _solved_multinomial()
    elif test_choice == "Student's t-test (Independent)":
        _solved_students_ttest()
    elif test_choice == "Welch's t-test (Independent, Unequal Variances)":
        _solved_welch_ttest()
    elif test_choice == "Yuen's Trimmed t-test":
        _solved_yuen_ttest()
    elif test_choice == "Paired t-test":
        _solved_paired_ttest()
    elif test_choice == "Sign Test (Paired)":
        _solved_paired_sign()
    elif test_choice == "Wilcoxon Signed-Rank Test":
        _solved_wilcoxon_paired()


# =========================================================================
# 1. ONE-SAMPLE T-TEST
# =========================================================================
def _solved_ttest():
    _section("One-Sample t-Test — Solved Example")

    st.markdown("Enter your summary statistics or raw data:")

    input_mode = st.radio(
        "Input mode",
        ["Summary statistics", "Raw data (comma-separated)"],
        key="ttest_input_mode",
    )

    if input_mode == "Summary statistics":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sample_mean = st.number_input(
                "Sample mean ($\\bar{x}$)",
                value=52.0,
                step=0.1,
                format="%.2f",
                key="ttest_xbar",
            )
        with col2:
            mu0 = st.number_input(
                "Hypothesized mean ($\\mu_0$)",
                value=50.0,
                step=0.1,
                format="%.2f",
                key="ttest_mu0",
            )
        with col3:
            s = st.number_input(
                "Sample SD ($s$)",
                value=8.0,
                min_value=0.01,
                step=0.1,
                format="%.2f",
                key="ttest_s",
            )
        with col4:
            n = st.number_input(
                "Sample size ($n$)",
                value=25,
                min_value=2,
                step=1,
                format="%d",
                key="ttest_n",
            )
    else:
        raw = st.text_area(
            "Enter data values (comma-separated, e.g. 45, 52, 48, 55, 50)",
            "45, 52, 48, 55, 50, 53, 49, 51, 54, 47, 56, 46, 50, 52, 48, 53, 51, 49, 55, 50",
            key="ttest_raw",
        )
        try:
            vals = np.array([float(x.strip()) for x in raw.split(",") if x.strip()])
        except ValueError:
            st.error("Invalid input. Use comma-separated numbers.")
            return
        n = len(vals)
        sample_mean = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
        mu0 = st.number_input(
            "Hypothesized mean ($\\mu_0$)",
            value=50.0,
            step=0.1,
            format="%.2f",
            key="ttest_mu0_raw",
        )

    if not st.button("Calculate Step-by-Step", key="btn_ttest"):
        return

    df = n - 1
    se = s / np.sqrt(n)
    t_stat = (sample_mean - mu0) / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    cohens_d = (sample_mean - mu0) / s

    _section("Solution")

    _step_header(1, "State the hypotheses")
    if mu0 is not None:
        _step_formula(rf"H_0: \mu = {mu0} \quad \text{{vs}} \quad H_1: \mu \neq {mu0}")

    _step_header(2, "Compute the standard error of the mean")
    _step_formula(
        rf"SE = \frac{{s}}{{\sqrt{{n}}}} = \frac{{{s:.3f}}}{{\sqrt{{{n}}}}} = {se:.4f}"
    )

    _step_header(3, "Calculate the test statistic")
    _step_formula(
        rf"t = \frac{{\bar{{x}} - \mu_0}}{{SE}} = \frac{{{sample_mean:.3f} - ({mu0:.3f})}}{{{se:.4f}}} = {t_stat:.4f}"
    )

    _step_header(4, "Determine the degrees of freedom")
    _step_formula(rf"df = n - 1 = {n} - 1 = {df}")

    _step_header(5, "Find the p-value")
    _step_formula(rf"p = 2 \times P(T_{{{df}}} > |{t_stat:.4f}|) = {p_value:.6f}")

    _step_header(6, "Decision")
    alpha = 0.05
    if p_value < alpha:
        _step_formula(
            rf"\text{{Since p = }}{p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. There is sufficient evidence that the population mean differs from}} {mu0}."
        )
    else:
        _step_formula(
            rf"\text{{Since p = }}{p_value:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀. There is insufficient evidence that the population mean differs from}} {mu0}."
        )

    _step_header(7, "Effect size (Cohen's d)")
    _step_formula(
        rf"d = \frac{{\bar{{x}} - \mu_0}}{{s}} = \frac{{{sample_mean:.3f} - {mu0:.3f}}}{{{s:.3f}}} = {cohens_d:.4f}"
    )
    d_abs = abs(cohens_d)
    if d_abs < 0.2:
        interpretation = "negligible"
    elif d_abs < 0.5:
        interpretation = "small"
    elif d_abs < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    _step_formula(
        rf"\text{{Cohen's }} d = {cohens_d:.4f} \text{{— this is a }} {interpretation} \text{{ effect size. }}"
    )

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": [
                "Sample mean",
                "Reference mean",
                "Sample SD",
                "n",
                "SE",
                "t",
                "df",
                "p-value",
                "Cohen's d",
            ],
            "Value": [
                f"{sample_mean:.3f}",
                f"{mu0:.3f}",
                f"{s:.3f}",
                f"{n}",
                f"{se:.4f}",
                f"{t_stat:.4f}",
                f"{df}",
                f"{p_value:.6f}",
                f"{cohens_d:.4f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name="ttest_results.csv",
        mime="text/csv",
        key="dl_ttest",
    )


# =========================================================================
# 2. ONE-SAMPLE Z-TEST
# =========================================================================
def _solved_ztest():
    _section("One-Sample z-Test — Solved Example")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sample_mean = st.number_input(
            "Sample mean ($\\bar{x}$)",
            value=52.0,
            step=0.1,
            format="%.2f",
            key="ztest_xbar",
        )
    with col2:
        mu0 = st.number_input(
            "Population mean ($\\mu_0$)",
            value=50.0,
            step=0.1,
            format="%.2f",
            key="ztest_mu0",
        )
    with col3:
        sigma = st.number_input(
            "Population SD ($\\sigma$)",
            value=10.0,
            min_value=0.01,
            step=0.1,
            format="%.2f",
            key="ztest_sigma",
        )
    with col4:
        n = st.number_input(
            "Sample size ($n$)",
            value=30,
            min_value=1,
            step=1,
            format="%d",
            key="ztest_n",
        )

    if not st.button("Calculate Step-by-Step", key="btn_ztest"):
        return

    se = sigma / np.sqrt(n)
    z_stat = (sample_mean - mu0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(rf"H_0: \mu = {mu0} \quad \text{{vs}} \quad H_1: \mu \neq {mu0}")

    _step_header(2, "Compute the standard error")
    _step_formula(
        rf"SE = \frac{{\sigma}}{{\sqrt{{n}}}} = \frac{{{sigma:.3f}}}{{\sqrt{{{n}}}}} = {se:.4f}"
    )

    _step_header(3, "Calculate the z-statistic")
    _step_formula(
        rf"z = \frac{{\bar{{x}} - \mu_0}}{{SE}} = \frac{{{sample_mean:.3f} - {mu0:.3f}}}{{{se:.4f}}} = {z_stat:.4f}"
    )

    _step_header(4, "Find the p-value")
    _step_formula(rf"p = 2 \times P(Z > |{z_stat:.4f}|) = {p_value:.6f}")

    _step_header(5, "Decision")
    alpha = 0.05
    z_crit = stats.norm.ppf(0.975)
    _step_formula(
        rf"\text{{Critical value (two-tailed, }}\alpha = {alpha}):  z_{{critical}} = ±{z_crit:.4f}"
    )
    if abs(z_stat) > z_crit:
        _step_formula(
            rf"|z| = {abs(z_stat):.4f} > {z_crit:.4f} \rightarrow \text{{reject H₀.}}"
        )
    else:
        _step_formula(
            rf"|z| = {abs(z_stat):.4f} ≤ {z_crit:.4f} \rightarrow \text{{fail to reject H₀.}}"
        )
    if p_value < alpha:
        _step_formula(
            rf"\text{{p = }}{p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀.}} \newline \text{{There is sufficient evidence that the population mean differs from }} {mu0}."
        )
    else:
        _step_formula(
            rf"\text{{p = }}{p_value:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀.}} \newline \text{{There is insufficient evidence that the population mean differs from }} {mu0}."
        )

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": [
                "Sample mean",
                "Population mean",
                "Population SD",
                "n",
                "SE",
                "z",
                "p-value",
            ],
            "Value": [
                f"{sample_mean:.3f}",
                f"{mu0:.3f}",
                f"{sigma:.3f}",
                f"{n}",
                f"{se:.4f}",
                f"{z_stat:.4f}",
                f"{p_value:.6f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name="ztest_results.csv",
        mime="text/csv",
        key="dl_ztest",
    )


# =========================================================================
# 3. ONE-SAMPLE PROPORTION TEST (BINOMIAL TEST — NORMAL APPROXIMATION)
# =========================================================================
def _solved_proportion():
    _section(
        "One-Sample Proportion Test (Binomial Test — Normal Approximation) — Solved Example"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input(
            "Total trials ($n$)",
            value=30,
            min_value=1,
            step=1,
            format="%d",
            key="prop_n",
        )
    with col2:
        successes = st.number_input(
            "Number of successes ($k$)",
            value=18,
            min_value=0,
            max_value=int(n),
            step=1,
            format="%d",
            key="prop_k",
        )
    with col3:
        p0 = st.number_input(
            "Hypothesized proportion ($p_0$)",
            value=0.50,
            min_value=0.001,
            max_value=0.999,
            step=0.01,
            format="%.3f",
            key="prop_p0",
        )

    if not st.button("Calculate Step-by-Step", key="btn_prop"):
        return

    p_hat = successes / n
    se = np.sqrt(p0 * (1 - p0) / n)
    z_stat = (p_hat - p0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(rf"H_0: p = {p0} \quad \text{{vs}} \quad H_1: p \neq {p0}")

    _step_header(2, "Compute the sample proportion")
    _step_formula(
        rf"\hat{{p}} = \frac{{k}}{{n}} = \frac{{{successes}}}{{{n}}} = {p_hat:.4f}"
    )

    _step_header(3, "Check the normality assumption")
    np0 = n * p0
    nq0 = n * (1 - p0)
    _step_formula(rf"n × p₀ = {n} × {p0:.3f} = {np0:.2f}")
    _step_formula(rf"n × (1 − p₀) = {n} × {1-p0:.3f} = {nq0:.2f}")
    if np0 >= 5 and nq0 >= 5:
        _step_formula(
            r"\text{Both} \geq 5 \rightarrow \text{normal approximation is valid.}"
        )
    else:
        _step_formula(
            r"\text{One or both} < 5 \rightarrow \text{normal approximation may be inaccurate; consider the Exact Binomial Test.}"
        )

    _step_header(4, "Compute the standard error under H₀")
    _step_formula(
        rf"SE = \sqrt{{\frac{{p_0(1-p_0)}}{{n}}}} = \sqrt{{\frac{{{p0:.3f} \times {1-p0:.3f}}}{{{n}}}}} = {se:.4f}"
    )

    _step_header(5, "Calculate the z-statistic")
    _step_formula(
        rf"z = \frac{{\hat{{p}} - p_0}}{{SE}} = \frac{{{p_hat:.4f} - {p0:.4f}}}{{{se:.4f}}} = {z_stat:.4f}"
    )

    _step_header(6, "Find the p-value")
    _step_formula(rf"p = 2 \times P(Z > |{z_stat:.4f}|) = {p_value:.6f}")

    _step_header(7, "Decision")
    alpha = 0.05
    if p_value < alpha:
        _step_formula(
            rf"\text{{p}} = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. The proportion differs significantly from}} {p0}."
        )
    else:
        _step_formula(
            rf"\text{{p}} = {p_value:.6f} ≥ \alpha = {alpha} \rightarrow \text{{fail to reject H₀. The proportion does not differ significantly from}} {p0}."
        )

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": [
                "n",
                "Successes",
                "Sample proportion",
                "Hypothesized p₀",
                "SE",
                "z",
                "p-value",
            ],
            "Value": [
                f"{n}",
                f"{successes}",
                f"{p_hat:.4f}",
                f"{p0:.3f}",
                f"{se:.4f}",
                f"{z_stat:.4f}",
                f"{p_value:.6f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name="proportion_test_results.csv",
        mime="text/csv",
        key="dl_prop",
    )


# =========================================================================
# 4. BINOMIAL TEST (EXACT)
# =========================================================================
def _solved_binomial_exact():
    _section("Binomial Test (Exact) — Solved Example")

    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input(
            "Total trials ($n$)",
            value=20,
            min_value=1,
            step=1,
            format="%d",
            key="binom_n",
        )
    with col2:
        successes = st.number_input(
            "Number of successes ($k$)",
            value=15,
            min_value=0,
            max_value=int(n),
            step=1,
            format="%d",
            key="binom_k",
        )
    with col3:
        p0 = st.number_input(
            "Hypothesized probability ($p_0$)",
            value=0.50,
            min_value=0.001,
            max_value=0.999,
            step=0.01,
            format="%.3f",
            key="binom_p0",
        )

    if not st.button("Calculate Step-by-Step", key="btn_binom"):
        return

    p_hat = successes / n

    # Two-sided exact p-value: sum probabilities of all outcomes <= P(observed)
    k_vals = np.arange(0, n + 1)
    probs = comb(n, k_vals) * (p0**k_vals) * ((1 - p0) ** (n - k_vals))
    observed_prob = comb(n, successes) * (p0**successes) * ((1 - p0) ** (n - successes))
    p_value = np.sum(probs[probs <= observed_prob + 1e-12])

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(rf"H_0: p = {p0} \quad \text{{vs}} \quad H_1: p \neq {p0}")

    _step_header(2, "Observed data")
    _step_subresult(f"k = {successes} successes out of n = {n} trials")
    _step_formula(rf"\hat{{p}} = \frac{{{successes}}}{{{n}}} = {p_hat:.4f}")

    _step_header(3, "Probability of the observed outcome under H₀")
    _step_formula(
        rf"P(X = {successes}) = \binom{{{n}}}{{{successes}}} \times {p0}^{{{successes}}} \times ({1-p0:.4f})^{{{n - successes}}}"
    )
    _step_formula(
        rf"= {comb(n, successes):.0f} × {p0 ** successes:.6f} × {(1-p0) ** (n - successes):.6f}"
    )
    _step_formula(rf"P(X = {successes}) = {observed_prob:.6f}")

    _step_header(4, "Compute the exact two-sided p-value")
    _step_subresult(
        "Sum the probabilities of all outcomes with probability ≤ P(observed):"
    )
    tbl_data = []
    for k, prob in zip(k_vals, probs):
        included = "✓" if prob <= observed_prob + 1e-12 else ""
        tbl_data.append([k, f"{prob:.6f}", included])
    df_binom = pd.DataFrame(tbl_data, columns=["k", "P(X = k)", "Included?"])
    st.dataframe(df_binom, use_container_width=True, height=400)
    csv_binom = df_binom.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_binom,
        file_name="binomial_enumeration.csv",
        mime="text/csv",
        key="dl_binom_enum",
    )
    _step_formula(rf"\text{{Exact two-sided p-value}} = {p_value:.6f}")

    _step_header(5, "Decision")
    alpha = 0.05
    if p_value < alpha:
        _step_formula(
            rf"p = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀.}} \newline \text{{The true probability differs significantly from }} {p0}."
        )
    else:
        _step_formula(
            rf"p = {p_value:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀.}} \newline \text{{The true probability does not differ significantly from }} {p0}."
        )

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": [
                "n",
                "Successes (k)",
                "Sample proportion",
                "Hypothesized p₀",
                "P(X = k) under H₀",
                "Exact p-value",
            ],
            "Value": [
                f"{n}",
                f"{successes}",
                f"{p_hat:.4f}",
                f"{p0:.3f}",
                f"{observed_prob:.6f}",
                f"{p_value:.6f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name="binomial_exact_results.csv",
        mime="text/csv",
        key="dl_binom",
    )


# =========================================================================
# 5. ONE-SAMPLE WILCOXON SIGNED-RANK TEST
# =========================================================================
def _solved_wilcoxon():
    _section("One-Sample Wilcoxon Signed-Rank Test — Solved Example")

    raw = st.text_area(
        "Enter data values (comma-separated, e.g. 23, 19, 25, 21, 28, 18, 22, 20, 27, 24)",
        "23, 19, 25, 21, 28, 18, 22, 20, 27, 24",
        key="wilcox_raw",
    )
    try:
        vals = np.array([float(x.strip()) for x in raw.split(",") if x.strip()])
    except ValueError:
        st.error("Invalid input. Use comma-separated numbers.")
        return

    theta0 = st.number_input(
        "Hypothesized median ($\\theta_0$)",
        value=20.0,
        step=0.5,
        format="%.1f",
        key="wilcox_theta",
    )

    if not st.button("Calculate Step-by-Step", key="btn_wilcox"):
        return

    n_total = len(vals)
    diffs = vals - theta0
    non_zero = diffs[diffs != 0]
    n = len(non_zero)
    abs_diffs = np.abs(non_zero)
    ranks = stats.rankdata(abs_diffs, method="average")
    signed_ranks = ranks * np.sign(non_zero)
    W_plus = np.sum(signed_ranks[signed_ranks > 0])
    W_minus = abs(np.sum(signed_ranks[signed_ranks < 0]))
    W_stat = min(W_plus, W_minus)

    # Normal approximation
    mu_W = n * (n + 1) / 4
    sigma_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    # Tie correction
    unique_abs = np.unique(abs_diffs)
    tie_correction = 0
    for u in unique_abs:
        t = np.sum(abs_diffs == u)
        if t > 1:
            tie_correction += t * (t**2 - 1)
    if tie_correction > 0:
        sigma_W = np.sqrt((n * (n + 1) * (2 * n + 1) - tie_correction / 2) / 24)
    z_stat = (W_stat - mu_W) / sigma_W
    p_value = 2 * stats.norm.cdf(z_stat)

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(
        rf"H_0: \theta = {theta0} \quad \text{{vs}} \quad H_1: \theta \neq {theta0}"
    )

    _step_header(2, "Compute deviations and absolute deviations")
    tbl_dev = []
    for i, v in enumerate(vals, start=1):
        d = v - theta0
        tbl_dev.append([f"{i}", f"{v:.1f}", f"{d:+.2f}", f"{abs(d):.2f}"])
    df_dev = pd.DataFrame(
        tbl_dev, columns=["Observation", "Value", "dᵢ = Xᵢ − θ₀", "|dᵢ|"]
    )
    st.dataframe(df_dev, use_container_width=True)
    csv_dev = df_dev.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_dev,
        file_name="wilcoxon_deviations.csv",
        mime="text/csv",
        key="dl_wilcox_dev",
    )

    _step_header(3, "Rank the absolute deviations (ignoring zeros)")
    tbl_rank = []
    for i, v in enumerate(non_zero):
        ad = abs_diffs[i]
        r = ranks[i]
        sr = signed_ranks[i]
        sign = "+ve" if sr > 0 else "-ve"
        tbl_rank.append(
            [
                f"{ad:.2f}",
                f"{r:.1f}",
                sign,
                f"{sr:+.1f}",
            ]
        )
    df_rank = pd.DataFrame(tbl_rank, columns=["|dᵢ|", "Rank", "Sign", "Signed Rank"])
    st.dataframe(df_rank, use_container_width=True)
    csv_rank = df_rank.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_rank,
        file_name="wilcoxon_ranks.csv",
        mime="text/csv",
        key="dl_wilcox_rank",
    )

    _step_header(4, "Sum the signed ranks")
    _step_formula(rf"W⁺ = \text{{sum of positive ranks}} = {W_plus:.1f}")
    _step_formula(rf"W⁻ = \text{{sum of negative ranks}} = {W_minus:.1f}")
    _step_formula(
        rf"W = \min(W^+, W^-) = \min({W_plus:.1f}, {W_minus:.1f}) = {W_stat:.1f}"
    )

    _step_header(5, "Normal approximation (large sample)")
    _step_formula(rf"n = {n} \text{{non-zero differences}}")
    _step_formula(
        rf"\mu_W = \frac{{n(n+1)}}{{4}} = \frac{{{n} \times {n+1}}}{{4}} = {mu_W:.2f}"
    )
    _step_formula(
        rf"\sigma_W = \sqrt{{\frac{{n(n+1)(2n+1)}}{{24}}}} = \sqrt{{\frac{{{n} \times {n+1} \times {2*n+1}}}{{24}}}} = {sigma_W:.4f}"
    )
    _step_formula(
        rf"z = \frac{{W - \mu_W}}{{\sigma_W}} = \frac{{{W_stat:.1f} - {mu_W:.2f}}}{{{sigma_W:.4f}}} = {z_stat:.4f}"
    )
    _step_formula(rf"p = 2 \times P(Z < {z_stat:.4f}) = {p_value:.6f}")

    _step_header(6, "Decision")
    alpha = 0.05
    if p_value < alpha:
        _step_formula(
            rf"p = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. }} \newline \text{{The median differs significantly from}} {theta0}."
        )
    else:
        _step_formula(
            rf"p = {p_value:.6f} ≥ \alpha = {alpha} \rightarrow \text{{fail to reject H₀ }}. \newline \text{{The median does not differ significantly from}} {theta0}."
        )

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": ["n (non-zero)", "W⁺", "W⁻", "W", "μ_W", "σ_W", "z", "p-value"],
            "Value": [
                f"{n}",
                f"{W_plus:.1f}",
                f"{W_minus:.1f}",
                f"{W_stat:.1f}",
                f"{mu_W:.2f}",
                f"{sigma_W:.4f}",
                f"{z_stat:.4f}",
                f"{p_value:.6f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name="wilcoxon_results.csv",
        mime="text/csv",
        key="dl_wilcox",
    )


# =========================================================================
# 6. CHI-SQUARE GOODNESS-OF-FIT TEST
# =========================================================================
def _solved_chisquare_gof():
    _section("Chi-Square Goodness-of-Fit Test — Solved Example")

    st.markdown("Enter the category labels, observed counts, and expected proportions:")

    n_cats = st.number_input(
        "Number of categories",
        value=4,
        min_value=2,
        max_value=10,
        step=1,
        format="%d",
        key="gof_ncats",
    )

    labels_input = st.text_input(
        "Category labels (comma-separated)",
        "Type A, Type B, Type C, Type D",
        key="gof_labels",
    )
    observed_input = st.text_input(
        "Observed counts (comma-separated)", "25, 30, 20, 25", key="gof_obs"
    )
    expected_input = st.text_input(
        "Expected proportions (comma-separated, sum to 1)",
        "0.25, 0.25, 0.25, 0.25",
        key="gof_exp",
    )

    try:
        labels = [x.strip() for x in labels_input.split(",") if x.strip()]
        observed = np.array(
            [float(x.strip()) for x in observed_input.split(",") if x.strip()],
            dtype=float,
        )
        expected_props = np.array(
            [float(x.strip()) for x in expected_input.split(",") if x.strip()],
            dtype=float,
        )
    except ValueError:
        st.error("Invalid input. Ensure all entries are numbers.")
        return

    if (
        len(labels) != n_cats
        or len(observed) != n_cats
        or len(expected_props) != n_cats
    ):
        st.error(f"All three inputs must have exactly {n_cats} entries.")
        return

    if abs(sum(expected_props) - 1.0) > 0.001:
        st.warning(
            f"Expected proportions sum to {sum(expected_props):.4f}, not 1. They will be normalized."
        )
        expected_props = expected_props / sum(expected_props)

    total_n = np.sum(observed)
    expected = expected_props * total_n

    if not st.button("Calculate Step-by-Step", key="btn_gof"):
        return

    chi_sq = np.sum((observed - expected) ** 2 / expected)
    df = n_cats - 1
    p_value = 1 - stats.chi2.cdf(chi_sq, df)

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(
        rf"H_0: \text{{The observed frequencies follow the specified distribution}}"
    )
    _step_formula(
        rf"H_1: \text{{At least one proportion differs from the specified distribution}}"
    )

    _step_header(2, "Compute expected frequencies")
    _step_subresult(f"Total N = {int(total_n)}")
    tbl_exp = []
    for i in range(n_cats):
        tbl_exp.append(
            [
                labels[i],
                int(observed[i]),
                f"{expected_props[i]:.4f}",
                f"{expected[i]:.2f}",
            ]
        )
    df_exp = pd.DataFrame(
        tbl_exp,
        columns=["Category", "Observed (O)", "Expected Proportion", "Expected (E)"],
    )
    st.dataframe(df_exp, use_container_width=True)
    csv_exp = df_exp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_exp,
        file_name="gof_expected.csv",
        mime="text/csv",
        key="dl_gof_exp",
    )

    _step_header(3, "Compute the chi-square contributions")
    contributions = (observed - expected) ** 2 / expected
    tbl_contrib = []
    for i in range(n_cats):
        diff = observed[i] - expected[i]
        tbl_contrib.append(
            [
                labels[i],
                int(observed[i]),
                f"{expected[i]:.2f}",
                f"{diff:+.2f}",
                f"{diff**2:.2f}",
                f"{contributions[i]:.4f}",
            ]
        )
    df_contrib = pd.DataFrame(
        tbl_contrib, columns=["Category", "O", "E", "O−E", "(O−E)²", "(O−E)²/E"]
    )
    st.dataframe(df_contrib, use_container_width=True)
    csv_contrib = df_contrib.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_contrib,
        file_name="gof_contributions.csv",
        mime="text/csv",
        key="dl_gof_contrib",
    )

    _step_header(4, "Calculate the test statistic")
    _step_formula(rf"\chi^2 = \sum \frac{{(O - E)^2}}{{E}} = {chi_sq:.4f}")

    _step_header(5, "Degrees of freedom")
    _step_formula(rf"df = k - 1 = {n_cats} - 1 = {df}")

    _step_header(6, "Find the p-value")
    _step_formula(rf"p = P(\chi^2_{{{df}}} > {chi_sq:.4f}) = {p_value:.6f}")

    _step_header(7, "Decision")
    alpha = 0.05
    critical = stats.chi2.ppf(0.95, df)
    _step_formula(
        rf"\text{{Critical value}} (\alpha = {alpha}, df = {df}): χ^2_{{critical}} = {critical:.4f}"
    )
    if chi_sq > critical:
        _step_formula(
            rf"χ² = {chi_sq:.4f} > {critical:.4f} \rightarrow \text{{reject}} H₀."
        )
    else:
        _step_formula(
            rf"χ² = {chi_sq:.4f} ≤ {critical:.4f} \rightarrow \text{{fail to reject}} H₀."
        )
    if p_value < alpha:
        _step_formula(
            rf"p = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀.}} \newline \text{{The observed frequencies do not follow the specified distribution.}}"
        )
    else:
        _step_formula(
            rf"p = {p_value:.6f} ≥ \alpha = {alpha} \rightarrow \text{{fail to reject H₀.}} \newline \text{{The observed frequencies follow the specified distribution.}}"
        )

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": ["χ²", "df", "p-value", "Critical value"],
            "Value": [f"{chi_sq:.4f}", f"{df}", f"{p_value:.6f}", f"{critical:.4f}"],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name="gof_results.csv",
        mime="text/csv",
        key="dl_gof",
    )


# =========================================================================
# 7. MULTINOMIAL TEST
# =========================================================================
def _solved_multinomial():
    _section("Multinomial Test (Exact) — Solved Example")
    st.markdown(
        "The exact multinomial test evaluates whether observed counts across multiple categories "
        "match a hypothesized probability distribution. The p-value is computed by summing the "
        "probabilities of all possible outcomes that are as extreme or more extreme than the observed."
    )

    n_cats = st.number_input(
        "Number of categories",
        value=3,
        min_value=2,
        max_value=6,
        step=1,
        format="%d",
        key="multi_ncats",
    )

    labels_input = st.text_input(
        "Category labels (comma-separated)", "AA, Aa, aa", key="multi_labels"
    )
    observed_input = st.text_input(
        "Observed counts (comma-separated)", "8, 10, 2", key="multi_obs"
    )
    expected_input = st.text_input(
        "Expected probabilities (comma-separated, sum to 1)",
        "0.25, 0.50, 0.25",
        key="multi_exp",
    )

    try:
        labels = [x.strip() for x in labels_input.split(",") if x.strip()]
        observed = np.array(
            [int(x.strip()) for x in observed_input.split(",") if x.strip()], dtype=int
        )
        expected_probs = np.array(
            [float(x.strip()) for x in expected_input.split(",") if x.strip()],
            dtype=float,
        )
    except ValueError:
        st.error(
            "Invalid input. Observed counts must be integers; probabilities must be numbers."
        )
        return

    if (
        len(labels) != n_cats
        or len(observed) != n_cats
        or len(expected_probs) != n_cats
    ):
        st.error(f"All three inputs must have exactly {n_cats} entries.")
        return

    if abs(sum(expected_probs) - 1.0) > 0.001:
        st.warning(
            f"Expected probabilities sum to {sum(expected_probs):.4f}, not 1. They will be normalized."
        )
        expected_probs = expected_probs / sum(expected_probs)

    total_n = int(np.sum(observed))
    expected_counts = expected_probs * total_n

    if not st.button("Calculate Step-by-Step", key="btn_multi"):
        return

    # Compute the probability of the observed outcome under H0
    def multinomial_prob(counts, probs):
        n_total = int(np.sum(counts))
        log_prob = (
            gammaln(n_total + 1)
            - np.sum(gammaln(counts + 1))
            + np.sum(counts * np.log(probs))
        )
        return np.exp(log_prob)

    obs_prob = multinomial_prob(observed, expected_probs)

    # Enumerate all possible count vectors that sum to total_n
    def _enumerate_counts(n, k):
        if k == 1:
            yield [n]
        else:
            for i in range(n + 1):
                for rest in _enumerate_counts(n - i, k - 1):
                    yield [i] + rest

    all_counts = list(_enumerate_counts(total_n, n_cats))
    all_probs = np.array(
        [multinomial_prob(np.array(c), expected_probs) for c in all_counts]
    )
    p_value = np.sum(all_probs[all_probs <= obs_prob + 1e-15])

    # Chi-square approximation for reference
    chi_sq = np.sum((observed - expected_counts) ** 2 / expected_counts)
    df = n_cats - 1
    chi_p = 1 - stats.chi2.cdf(chi_sq, df)

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(
        rf"H_0: \text{{The counts follow probabilities }} ({', '.join(f'{p:.3f}' for p in expected_probs)})"
    )
    _step_formula(
        r"H_1: \text{At least one probability differs from the specified distribution}"
    )

    _step_header(2, "Observed data")
    tbl_obs = []
    for i in range(n_cats):
        tbl_obs.append([labels[i], observed[i], f"{expected_counts[i]:.2f}"])
    df_obs = pd.DataFrame(
        tbl_obs, columns=["Category", "Observed", "Expected (under H₀)"]
    )
    st.dataframe(df_obs, use_container_width=True)
    csv_obs = df_obs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_obs,
        file_name="multinomial_observed.csv",
        mime="text/csv",
        key="dl_multi_obs",
    )

    _step_header(3, "Probability of the observed outcome under H₀")
    _step_formula(
        rf"P_{{(\text{{observed}})}} = \frac{{{total_n}!}}{{{'!'.join(str(o) for o in observed)}!}} \times "
        rf"{' × '.join(f'{expected_probs[i]}^{observed[i]}' for i in range(n_cats))}"
    )
    _step_formula(rf"P_{{(\text{{observed}})}} = {obs_prob:.6e}")

    _step_header(4, "Enumerate all possible outcomes and compute p-value")
    st.markdown(
        f"Total possible outcomes (compositions of {total_n} into {n_cats} parts): {len(all_counts)}"
    )
    tbl_enumerate = []
    for idx in range(len(all_counts)):
        c = all_counts[idx]
        p = all_probs[idx]
        included = "✓" if p <= obs_prob + 1e-15 else ""
        tbl_enumerate.append([str(c), f"{p:.6e}", included])
    df_enumerate = pd.DataFrame(
        tbl_enumerate, columns=["Outcome", "Probability", "≤ Observed?"]
    )
    st.dataframe(df_enumerate, use_container_width=True, height=400)
    csv = df_enumerate.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv,
        file_name="multinomial_enumeration.csv",
        mime="text/csv",
    )
    _step_formula(rf"\text{{Exact multinomial p-value}} = {p_value:.6f}")

    _step_header(5, "Chi-square approximation (reference)")
    _step_formula(rf"\chi^2 = \sum \frac{{(O - E)^2}}{{E}} = {chi_sq:.4f}")
    _step_formula(rf"df = k − 1 = {n_cats} − 1 = {df}")
    _step_formula(rf"\text{{Approximate p-value (chi-square)}} = {chi_p:.6f}")

    _step_header(6, "Decision")
    alpha = 0.05
    if p_value < alpha:
        _step_formula(
            rf"\text{{Exact p}} = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{**reject H₀**.}} \newline \text{{The observed distribution differs significantly from the hypothesized distribution.}}"
        )
    else:
        _step_formula(
            rf"\text{{Exact p}} = {p_value:.6f} ≥ \alpha = {alpha} \rightarrow \text{{**fail to reject H₀**.}} \newline \text{{The observed distribution does not differ significantly from the hypothesized distribution.}}  "
        )

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": [
                "P(observed)",
                "Exact p-value",
                "χ² (approx)",
                "df",
                "χ² p-value",
            ],
            "Value": [
                f"{obs_prob:.6e}",
                f"{p_value:.6f}",
                f"{chi_sq:.4f}",
                f"{df}",
                f"{chi_p:.6f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv_data,
        file_name="multinomial_results.csv",
        mime="text/csv",
        key="dl_multi",
    )


# =========================================================================
# 8. STUDENT'S T-TEST (INDEPENDENT)
# =========================================================================
def _solved_students_ttest():
    _section("Student's t-Test (Independent) — Solved Example")

    st.markdown("Enter summary statistics or raw data for **two independent groups**:")

    input_mode = st.radio(
        "Input mode",
        ["Summary statistics", "Raw data (comma-separated, two groups)"],
        key="stt_input_mode",
    )

    if input_mode == "Summary statistics":
        col1, col2, col3 = st.columns(3)
        with col1:
            n1 = st.number_input("n₁", value=12, min_value=3, step=1, format="%d", key="stt_n1")
            mean1 = st.number_input("Mean₁ (x̄₁)", value=85.0, step=0.1, format="%.2f", key="stt_m1")
            s1 = st.number_input("SD₁ (s₁)", value=8.0, min_value=0.01, step=0.1, format="%.2f", key="stt_s1")
        with col2:
            n2 = st.number_input("n₂", value=12, min_value=3, step=1, format="%d", key="stt_n2")
            mean2 = st.number_input("Mean₂ (x̄₂)", value=75.0, step=0.1, format="%.2f", key="stt_m2")
            s2 = st.number_input("SD₂ (s₂)", value=9.0, min_value=0.01, step=0.1, format="%.2f", key="stt_s2")
    else:
        raw1 = st.text_area(
            "Group 1 data (comma-separated)",
            "78, 92, 83, 76, 88, 84, 90, 79, 85, 81, 87, 82",
            key="stt_raw1",
        )
        raw2 = st.text_area(
            "Group 2 data (comma-separated)",
            "65, 72, 68, 78, 80, 71, 74, 69, 77, 73, 70, 66",
            key="stt_raw2",
        )
        try:
            v1 = np.array([float(x.strip()) for x in raw1.split(",") if x.strip()])
            v2 = np.array([float(x.strip()) for x in raw2.split(",") if x.strip()])
        except ValueError:
            st.error("Invalid input.")
            return
        n1, n2 = len(v1), len(v2)
        mean1, mean2 = float(np.mean(v1)), float(np.mean(v2))
        s1, s2 = float(np.std(v1, ddof=1)), float(np.std(v2, ddof=1))

    if not st.button("Calculate Step-by-Step", key="btn_stt"):
        return

    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
    sp = np.sqrt(sp2)
    se = sp * np.sqrt(1 / n1 + 1 / n2)
    t_stat = (mean1 - mean2) / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    cohens_d = (mean1 - mean2) / sp

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(r"H_0: \mu_1 = \mu_2 \quad \text{vs} \quad H_1: \mu_1 \neq \mu_2")

    _step_header(2, "Compute the pooled standard deviation")
    _step_formula(
        rf"s_p^2 = \frac{{(n_1-1)s_1^2 + (n_2-1)s_2^2}}{{n_1 + n_2 - 2}}"
    )
    _step_formula(
        rf"s_p^2 = \frac{{({n1-1})({s1:.3f})^2 + ({n2-1})({s2:.3f})^2}}{{{n1}+{n2}-2}} = {sp2:.4f}"
    )
    _step_formula(rf"s_p = \sqrt{{{sp2:.4f}}} = {sp:.4f}")

    _step_header(3, "Compute the standard error of the difference")
    _step_formula(
        rf"SE = s_p \sqrt{{\frac{{1}}{{n_1}} + \frac{{1}}{{n_2}}}}"
    )
    _step_formula(
        rf"SE = {sp:.4f} \times \sqrt{{\frac{{1}}{{{n1}}} + \frac{{1}}{{{n2}}}}} = {se:.4f}"
    )

    _step_header(4, "Calculate the t-statistic")
    _step_formula(
        rf"t = \frac{{\bar{{x}}_1 - \bar{{x}}_2}}{{SE}} = \frac{{{mean1:.3f} - {mean2:.3f}}}{{{se:.4f}}} = {t_stat:.4f}"
    )

    _step_header(5, "Degrees of freedom")
    _step_formula(rf"df = n_1 + n_2 - 2 = {n1} + {n2} - 2 = {df}")

    _step_header(6, "Find the p-value")
    _step_formula(rf"p = 2 \times P(T_{{{df}}} > |{t_stat:.4f}|) = {p_value:.6f}")

    _step_header(7, "Decision")
    alpha = 0.05
    t_crit = stats.t.ppf(0.975, df)
    _step_formula(rf"\text{{Critical value (two-tailed, }}\alpha={alpha}\text{{, df=}}{df}\text{{): }} t_{{critical}} = \pm{t_crit:.4f}")
    if abs(t_stat) > t_crit:
        _step_formula(rf"|t| = {abs(t_stat):.4f} > {t_crit:.4f} \rightarrow \text{{reject H₀.}}")
    else:
        _step_formula(rf"|t| = {abs(t_stat):.4f} \leq {t_crit:.4f} \rightarrow \text{{fail to reject H₀.}}")
    if p_value < alpha:
        _step_formula(rf"p = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. The group means differ significantly.}}")
    else:
        _step_formula(rf"p = {p_value:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀. The group means are not significantly different.}}")

    _step_header(8, "Effect size (Cohen's d)")
    _step_formula(
        rf"d = \frac{{\bar{{x}}_1 - \bar{{x}}_2}}{{s_p}} = \frac{{{mean1:.3f} - {mean2:.3f}}}{{{sp:.3f}}} = {cohens_d:.4f}"
    )
    d_abs = abs(cohens_d)
    if d_abs < 0.2:
        interp = "negligible"
    elif d_abs < 0.5:
        interp = "small"
    elif d_abs < 0.8:
        interp = "medium"
    else:
        interp = "large"
    _step_formula(rf"\text{{Cohen's }} d = {cohens_d:.4f} \text{{ — }} {interp} \text{{ effect.}}")

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": ["x̄₁", "x̄₂", "s₁", "s₂", "n₁", "n₂", "s_p", "SE", "t", "df", "p-value", "Cohen's d"],
            "Value": [
                f"{mean1:.3f}", f"{mean2:.3f}", f"{s1:.3f}", f"{s2:.3f}",
                f"{n1}", f"{n2}", f"{sp:.4f}", f"{se:.4f}",
                f"{t_stat:.4f}", f"{df}", f"{p_value:.6f}", f"{cohens_d:.4f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_data, file_name="students_ttest_results.csv", mime="text/csv", key="dl_stt")


# =========================================================================
# 9. WELCH'S T-TEST (INDEPENDENT, UNEQUAL VARIANCES)
# =========================================================================
def _solved_welch_ttest():
    _section("Welch's t-Test (Unequal Variances) — Solved Example")

    st.markdown("Enter summary statistics or raw data for **two independent groups**:")

    input_mode = st.radio(
        "Input mode",
        ["Summary statistics", "Raw data (comma-separated, two groups)"],
        key="welch_input_mode",
    )

    if input_mode == "Summary statistics":
        col1, col2, col3 = st.columns(3)
        with col1:
            n1 = st.number_input("n₁", value=10, min_value=3, step=1, format="%d", key="welch_n1")
            mean1 = st.number_input("Mean₁ (x̄₁)", value=88.0, step=0.1, format="%.2f", key="welch_m1")
            s1 = st.number_input("SD₁ (s₁)", value=5.0, min_value=0.01, step=0.1, format="%.2f", key="welch_s1")
        with col2:
            n2 = st.number_input("n₂", value=15, min_value=3, step=1, format="%d", key="welch_n2")
            mean2 = st.number_input("Mean₂ (x̄₂)", value=78.0, step=0.1, format="%.2f", key="welch_m2")
            s2 = st.number_input("SD₂ (s₂)", value=12.0, min_value=0.01, step=0.1, format="%.2f", key="welch_s2")
    else:
        raw1 = st.text_area(
            "Group 1 data (comma-separated)",
            "85, 92, 88, 90, 84, 91, 86, 89, 83, 87",
            key="welch_raw1",
        )
        raw2 = st.text_area(
            "Group 2 data (comma-separated)",
            "65, 82, 72, 78, 95, 70, 74, 68, 80, 85, 79, 76, 71, 77, 69",
            key="welch_raw2",
        )
        try:
            v1 = np.array([float(x.strip()) for x in raw1.split(",") if x.strip()])
            v2 = np.array([float(x.strip()) for x in raw2.split(",") if x.strip()])
        except ValueError:
            st.error("Invalid input.")
            return
        n1, n2 = len(v1), len(v2)
        mean1, mean2 = float(np.mean(v1)), float(np.mean(v2))
        s1, s2 = float(np.std(v1, ddof=1)), float(np.std(v2, ddof=1))

    if not st.button("Calculate Step-by-Step", key="btn_welch"):
        return

    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite df
    num = (s1**2 / n1 + s2**2 / n2) ** 2
    denom = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    df_w = num / denom

    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_w))

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(r"H_0: \mu_1 = \mu_2 \quad \text{vs} \quad H_1: \mu_1 \neq \mu_2")

    _step_header(2, "Compute the standard error (unequal variances)")
    _step_formula(
        rf"SE = \sqrt{{\frac{{s_1^2}}{{n_1}} + \frac{{s_2^2}}{{n_2}}}}"
    )
    _step_formula(
        rf"SE = \sqrt{{\frac{{{s1:.3f}^2}}{{{n1}}} + \frac{{{s2:.3f}^2}}{{{n2}}}}} = {se:.4f}"
    )

    _step_header(3, "Calculate the t-statistic")
    _step_formula(
        rf"t = \frac{{\bar{{x}}_1 - \bar{{x}}_2}}{{SE}} = \frac{{{mean1:.3f} - {mean2:.3f}}}{{{se:.4f}}} = {t_stat:.4f}"
    )

    _step_header(4, "Welch-Satterthwaite degrees of freedom")
    _step_formula(
        rf"\nu = \frac{{ \left( \frac{{s_1^2}}{{n_1}} + \frac{{s_2^2}}{{n_2}} \right)^2 }}{{ \frac{{ (s_1^2/n_1)^2 }}{{n_1-1}} + \frac{{ (s_2^2/n_2)^2 }}{{n_2-1}} }}"
    )
    _step_formula(
        rf"\nu = \frac{{ ({s1**2/n1:.4f} + {s2**2/n2:.4f})^2 }}{{ {((s1**2/n1)**2/(n1-1)):.4f} + {((s2**2/n2)**2/(n2-1)):.4f} }} = {df_w:.2f}"
    )

    _step_header(5, "Find the p-value")
    _step_formula(rf"p = 2 \times P(T_{{{df_w:.2f}}} > |{t_stat:.4f}|) = {p_value:.6f}")

    _step_header(6, "Decision")
    alpha = 0.05
    t_crit = stats.t.ppf(0.975, df_w)
    _step_formula(rf"\text{{Critical value (two-tailed, }}\alpha={alpha}\text{{, df=}}{df_w:.2f}\text{{): }} t_{{critical}} = \pm{t_crit:.4f}")
    if abs(t_stat) > t_crit:
        _step_formula(rf"|t| = {abs(t_stat):.4f} > {t_crit:.4f} \rightarrow \text{{reject H₀.}}")
    else:
        _step_formula(rf"|t| = {abs(t_stat):.4f} \leq {t_crit:.4f} \rightarrow \text{{fail to reject H₀.}}")
    if p_value < alpha:
        _step_formula(rf"p = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. The group means differ significantly.}}")
    else:
        _step_formula(rf"p = {p_value:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀. The group means are not significantly different.}}")

    _step_header(7, "Effect size (Cohen's d using pooled SD for reference)")
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    sp = np.sqrt(sp2)
    cohens_d = (mean1 - mean2) / sp
    _step_formula(rf"s_p = \sqrt{{ \frac{{(n_1-1)s_1^2 + (n_2-1)s_2^2}}{{n_1+n_2-2}} }} = {sp:.4f}")
    _step_formula(rf"d = \frac{{\bar{{x}}_1 - \bar{{x}}_2}}{{s_p}} = {cohens_d:.4f}")
    d_abs = abs(cohens_d)
    if d_abs < 0.2:
        interp = "negligible"
    elif d_abs < 0.5:
        interp = "small"
    elif d_abs < 0.8:
        interp = "medium"
    else:
        interp = "large"
    _step_formula(rf"\text{{Cohen's }} d = {cohens_d:.4f} \text{{ — }} {interp} \text{{ effect.}}")

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": ["x̄₁", "x̄₂", "s₁", "s₂", "n₁", "n₂", "SE", "t", "df (Welch)", "p-value", "Cohen's d"],
            "Value": [
                f"{mean1:.3f}", f"{mean2:.3f}", f"{s1:.3f}", f"{s2:.3f}",
                f"{n1}", f"{n2}", f"{se:.4f}",
                f"{t_stat:.4f}", f"{df_w:.2f}", f"{p_value:.6f}", f"{cohens_d:.4f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_data, file_name="welch_ttest_results.csv", mime="text/csv", key="dl_welch")


# =========================================================================
# 10. YUEN'S TRIMMED T-TEST
# =========================================================================
def _solved_yuen_ttest():
    _section("Yuen's Trimmed t-Test — Solved Example")

    st.markdown(
        "Yuen's test compares trimmed means of two independent groups. "
        "It trims a proportion *g* of observations from each tail, reducing the influence of outliers."
    )

    gamma = st.slider(
        "Trimming proportion (γ) — fraction trimmed from each tail",
        min_value=0.05, max_value=0.25, value=0.10, step=0.05,
        format="%.2f",
        key="yuen_gamma",
    )

    raw1 = st.text_area(
        "Group 1 data (comma-separated)",
        "23, 45, 38, 42, 36, 41, 155, 39, 44, 37, 40, 43",
        key="yuen_raw1",
    )
    raw2 = st.text_area(
        "Group 2 data (comma-separated)",
        "30, 28, 25, 35, 22, 32, 27, 29, 31, 26, 33, 160",
        key="yuen_raw2",
    )

    try:
        v1 = np.array([float(x.strip()) for x in raw1.split(",") if x.strip()])
        v2 = np.array([float(x.strip()) for x in raw2.split(",") if x.strip()])
    except ValueError:
        st.error("Invalid input.")
        return

    if not st.button("Calculate Step-by-Step", key="btn_yuen"):
        return

    n1, n2 = len(v1), len(v2)
    v1_sorted = np.sort(v1)
    v2_sorted = np.sort(v2)
    g1 = int(np.floor(n1 * gamma))
    g2 = int(np.floor(n2 * gamma))
    h1 = n1 - 2 * g1
    h2 = n2 - 2 * g2

    # Trimmed means
    trim1 = v1_sorted[g1:n1 - g1]
    trim2 = v2_sorted[g2:n2 - g2]
    mt1 = float(np.mean(trim1))
    mt2 = float(np.mean(trim2))

    # Winsorized variances
    w1 = np.concatenate([np.full(g1, v1_sorted[g1]), trim1, np.full(g1, v1_sorted[n1 - g1 - 1])])
    w2 = np.concatenate([np.full(g2, v2_sorted[g2]), trim2, np.full(g2, v2_sorted[n2 - g2 - 1])])
    sw1_2 = float(np.var(w1, ddof=1))
    sw2_2 = float(np.var(w2, ddof=1))

    # Yuen's test statistic
    denom = np.sqrt((n1 - 1) * sw1_2 / (h1 * (h1 - 1)) + (n2 - 1) * sw2_2 / (h2 * (h2 - 1)))
    t_y = (mt1 - mt2) / denom

    # Welch-type df
    num_df = ((n1 - 1) * sw1_2 / (h1 * (h1 - 1)) + (n2 - 1) * sw2_2 / (h2 * (h2 - 1))) ** 2
    denom_df = ((n1 - 1) * sw1_2 / (h1 * (h1 - 1))) ** 2 / (h1 - 1) + ((n2 - 1) * sw2_2 / (h2 * (h2 - 1))) ** 2 / (h2 - 1)
    df_y = num_df / denom_df

    p_value = 2 * (1 - stats.t.cdf(abs(t_y), df_y))

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(r"H_0: \mu_{t1} = \mu_{t2} \quad \text{vs} \quad H_1: \mu_{t1} \neq \mu_{t2}")
    _step_subresult(f"Trimming proportion γ = {gamma:.2f}")

    _step_header(2, "Sort each group and determine trimming limits")
    _step_formula(rf"n_1 = {n1},\quad n_2 = {n2}")
    _step_formula(rf"g_1 = \lfloor {n1} \times {gamma:.2f} \rfloor = {g1}")
    _step_formula(rf"g_2 = \lfloor {n2} \times {gamma:.2f} \rfloor = {g2}")
    _step_formula(rf"h_1 = n_1 - 2g_1 = {n1} - 2({g1}) = {h1}")
    _step_formula(rf"h_2 = n_2 - 2g_2 = {n2} - 2({g2}) = {h2}")
    _step_subresult(f"Observations to trim per tail: {g1} from Group 1, {g2} from Group 2")

    _step_header(3, "Sorted data with trimming markers")
    def _make_trim_table(vals, g, n):
        rows = []
        for i, x in enumerate(vals):
            if i < g or i >= n - g:
                label = "trimmed"
            else:
                label = "kept"
            rows.append([i + 1, f"{x:.2f}", label])
        return rows
    df_t1 = pd.DataFrame(_make_trim_table(v1_sorted, g1, n1), columns=["Rank", "Value", "Status"])
    df_t2 = pd.DataFrame(_make_trim_table(v2_sorted, g2, n2), columns=["Rank", "Value", "Status"])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Group 1 (sorted):**")
        st.dataframe(df_t1, use_container_width=True, height=300)
    with col2:
        st.markdown("**Group 2 (sorted):**")
        st.dataframe(df_t2, use_container_width=True, height=300)

    _step_header(4, "Compute trimmed means")
    _step_formula(rf"\bar{{X}}_{{t1}} = \text{{mean of kept values in Group 1}} = {mt1:.4f}")
    _step_formula(rf"\bar{{X}}_{{t2}} = \text{{mean of kept values in Group 2}} = {mt2:.4f}")

    _step_header(5, "Compute winsorized variances")
    _step_subresult("Winsorization replaces trimmed values with the nearest retained value.")
    _step_formula(rf"s_{{w1}}^2 = {sw1_2:.4f}")
    _step_formula(rf"s_{{w2}}^2 = {sw2_2:.4f}")

    _step_header(6, "Calculate Yuen's test statistic")
    _step_formula(
        rf"t_y = \frac{{\bar{{X}}_{{t1}} - \bar{{X}}_{{t2}}}}{{\sqrt{{ \frac{{(n_1-1)s_{{w1}}^2}}{{h_1(h_1-1)}} + \frac{{(n_2-1)s_{{w2}}^2}}{{h_2(h_2-1)}} }}}}"
    )
    _step_formula(
        rf"t_y = \frac{{{mt1:.4f} - {mt2:.4f}}}{{{denom:.4f}}} = {t_y:.4f}"
    )

    _step_header(7, "Degrees of freedom (Welch-type approximation)")
    _step_formula(rf"\nu = {df_y:.2f}")

    _step_header(8, "Find the p-value")
    _step_formula(rf"p = 2 \times P(T_{{{df_y:.2f}}} > |{t_y:.4f}|) = {p_value:.6f}")

    _step_header(9, "Decision")
    alpha = 0.05
    if p_value < alpha:
        _step_formula(rf"p = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. The trimmed group means differ significantly.}}")
    else:
        _step_formula(rf"p = {p_value:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀. The trimmed group means are not significantly different.}}")

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": [
                "n₁, n₂", "g₁, g₂", "h₁, h₂",
                "Trimmed mean₁", "Trimmed mean₂",
                "Winsorized var₁", "Winsorized var₂",
                "t_y", "df", "p-value",
            ],
            "Value": [
                f"{n1}, {n2}", f"{g1}, {g2}", f"{h1}, {h2}",
                f"{mt1:.4f}", f"{mt2:.4f}",
                f"{sw1_2:.4f}", f"{sw2_2:.4f}",
                f"{t_y:.4f}", f"{df_y:.2f}", f"{p_value:.6f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_data, file_name="yuen_ttest_results.csv", mime="text/csv", key="dl_yuen")


# =========================================================================
# 11. PAIRED T-TEST
# =========================================================================
def _solved_paired_ttest():
    _section("Paired t-Test — Solved Example")

    st.markdown("Enter paired (before/after) data:")

    input_mode = st.radio(
        "Input mode",
        ["Raw paired data (two columns)", "Summary of differences"],
        key="ptt_input_mode",
    )

    if input_mode == "Summary of differences":
        col1, col2, col3 = st.columns(3)
        with col1:
            d_bar = st.number_input("Mean difference (d̄)", value=4.5, step=0.1, format="%.2f", key="ptt_dbar")
        with col2:
            s_d = st.number_input("SD of differences (s_d)", value=6.0, min_value=0.01, step=0.1, format="%.2f", key="ptt_sd")
        with col3:
            n = st.number_input("Number of pairs (n)", value=15, min_value=2, step=1, format="%d", key="ptt_n")
    else:
        raw_before = st.text_area(
            "Before / Control (comma-separated)",
            "78, 82, 75, 80, 77, 85, 79, 74, 81, 76, 83, 73, 84, 72, 78",
            key="ptt_before",
        )
        raw_after = st.text_area(
            "After / Treatment (comma-separated)",
            "85, 88, 80, 86, 82, 90, 84, 78, 87, 81, 89, 79, 91, 77, 83",
            key="ptt_after",
        )
        try:
            before = np.array([float(x.strip()) for x in raw_before.split(",") if x.strip()])
            after = np.array([float(x.strip()) for x in raw_after.split(",") if x.strip()])
        except ValueError:
            st.error("Invalid input. Use comma-separated numbers.")
            return
        if len(before) != len(after):
            st.error("Both groups must have the same number of observations.")
            return
        diffs = after - before
        n = len(diffs)
        d_bar = float(np.mean(diffs))
        s_d = float(np.std(diffs, ddof=1))

    if not st.button("Calculate Step-by-Step", key="btn_ptt"):
        return

    se = s_d / np.sqrt(n)
    t_stat = d_bar / se
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    d_z = d_bar / s_d

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(r"H_0: \mu_d = 0 \quad \text{vs} \quad H_1: \mu_d \neq 0")
    _step_subresult("Where μ_d is the mean of the paired differences (After − Before).")

    if input_mode == "Raw paired data (two columns)":
        _step_header(2, "Compute differences")
        tbl = []
        for i in range(n):
            tbl.append([i + 1, f"{before[i]:.2f}", f"{after[i]:.2f}", f"{diffs[i]:+.2f}"])
        df_diffs = pd.DataFrame(tbl, columns=["Pair", "Before", "After", "d = After − Before"])
        st.dataframe(df_diffs, use_container_width=True)
        csv_d = df_diffs.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", data=csv_d, file_name="paired_diffs.csv", mime="text/csv", key="dl_ptt_diffs")

    _step_header(3, "Calculate the mean and SD of differences")
    _step_formula(rf"\bar{{d}} = \frac{{\sum d_i}}{{n}} = {d_bar:.4f}")
    _step_formula(rf"s_d = {s_d:.4f}")

    _step_header(4, "Compute the standard error")
    _step_formula(rf"SE = \frac{{s_d}}{{\sqrt{{n}}}} = \frac{{{s_d:.4f}}}{{\sqrt{{{n}}}}} = {se:.4f}")

    _step_header(5, "Calculate the t-statistic")
    _step_formula(rf"t = \frac{{\bar{{d}}}}{{SE}} = \frac{{{d_bar:.4f}}}{{{se:.4f}}} = {t_stat:.4f}")

    _step_header(6, "Degrees of freedom")
    _step_formula(rf"df = n - 1 = {n} - 1 = {df}")

    _step_header(7, "Find the p-value")
    _step_formula(rf"p = 2 \times P(T_{{{df}}} > |{t_stat:.4f}|) = {p_value:.6f}")

    _step_header(8, "Decision")
    alpha = 0.05
    if p_value < alpha:
        _step_formula(rf"p = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. The mean difference is significantly different from zero.}}")
    else:
        _step_formula(rf"p = {p_value:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀. The mean difference is not significantly different from zero.}}")

    _step_header(9, "Effect size (Cohen's d_z)")
    _step_formula(rf"d_z = \frac{{\bar{{d}}}}{{s_d}} = \frac{{{d_bar:.4f}}}{{{s_d:.4f}}} = {d_z:.4f}")
    dz_abs = abs(d_z)
    if dz_abs < 0.2:
        interp = "negligible"
    elif dz_abs < 0.5:
        interp = "small"
    elif dz_abs < 0.8:
        interp = "medium"
    else:
        interp = "large"
    _step_formula(rf"\text{{Cohen's }} d_z = {d_z:.4f} \text{{ — }} {interp} \text{{ effect.}}")

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": ["d̄", "s_d", "n", "SE", "t", "df", "p-value", "Cohen's d_z"],
            "Value": [
                f"{d_bar:.4f}", f"{s_d:.4f}", f"{n}",
                f"{se:.4f}", f"{t_stat:.4f}", f"{df}",
                f"{p_value:.6f}", f"{d_z:.4f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_data, file_name="paired_ttest_results.csv", mime="text/csv", key="dl_ptt")


# =========================================================================
# 12. SIGN TEST (PAIRED)
# =========================================================================
def _solved_paired_sign():
    _section("Sign Test (Paired) — Solved Example")

    st.markdown(
        "The Sign Test compares paired observations using only the **direction** (sign) of differences, "
        "ignoring magnitudes. It tests whether the median difference is zero."
    )

    raw_before = st.text_area(
        "Before / Control (comma-separated)",
        "23, 19, 25, 21, 28, 18, 22, 20, 27, 24, 29, 26",
        key="psign_before",
    )
    raw_after = st.text_area(
        "After / Treatment (comma-separated)",
        "30, 22, 28, 25, 31, 20, 24, 19, 29, 27, 33, 30",
        key="psign_after",
    )

    try:
        before = np.array([float(x.strip()) for x in raw_before.split(",") if x.strip()])
        after = np.array([float(x.strip()) for x in raw_after.split(",") if x.strip()])
    except ValueError:
        st.error("Invalid input.")
        return
    if len(before) != len(after):
        st.error("Both groups must have the same number of observations.")
        return

    if not st.button("Calculate Step-by-Step", key="btn_psign"):
        return

    diffs = after - before
    n_total = len(diffs)
    signs = np.sign(diffs)
    n_pos = int(np.sum(signs == 1))
    n_neg = int(np.sum(signs == -1))
    n_zero = int(np.sum(signs == 0))
    n_eff = n_pos + n_neg
    k = min(n_pos, n_neg)

    # Exact binomial p-value (two-sided)
    p_value_exact = 2 * min(
        stats.binom.cdf(k, n_eff, 0.5),
        1 - stats.binom.cdf(k - 1, n_eff, 0.5),
    )
    p_value_exact = min(p_value_exact, 1.0)

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(r"H_0: \text{median difference} = 0 \quad \text{vs} \quad H_1: \text{median difference} \neq 0")

    _step_header(2, "Compute signs of differences")
    tbl = []
    for i in range(n_total):
        d = diffs[i]
        if d > 0:
            s = "+"
        elif d < 0:
            s = "−"
        else:
            s = "0"
        tbl.append([i + 1, f"{before[i]:.2f}", f"{after[i]:.2f}", f"{d:+.2f}", s])
    df_signs = pd.DataFrame(tbl, columns=["Pair", "Before", "After", "d", "Sign"])
    st.dataframe(df_signs, use_container_width=True)
    csv_s = df_signs.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_s, file_name="sign_test_diffs.csv", mime="text/csv", key="dl_psign_diffs")

    _step_header(3, "Count signs")
    _step_formula(rf"\text{{Positive differences}}: {n_pos}")
    _step_formula(rf"\text{{Negative differences}}: {n_neg}")
    _step_formula(rf"\text{{Zeros (dropped)}}: {n_zero}")
    _step_formula(rf"\text{{Effective sample size }} n' = {n_pos} + {n_neg} = {n_eff}")

    _step_header(4, "Calculate the test statistic")
    _step_formula(rf"S = \min(S^+, S^-) = \min({n_pos}, {n_neg}) = {k}")
    _step_subresult(f"Under H₀, S ~ Binomial(n' = {n_eff}, p = 0.5)")

    _step_header(5, "Compute the exact p-value")
    _step_formula(rf"p = 2 \times P(\text{{Binomial}}({n_eff}, 0.5) \leq {k})")
    _step_formula(rf"p = {p_value_exact:.6f}")

    _step_header(6, "Decision")
    alpha = 0.05
    if p_value_exact < alpha:
        _step_formula(rf"p = {p_value_exact:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. The median difference is significantly different from zero.}}")
    else:
        _step_formula(rf"p = {p_value_exact:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀. The median difference is not significantly different from zero.}}")

    _step_header(7, "Effect size")
    prop_pos = n_pos / n_eff
    _step_formula(rf"\text{{Proportion of positive differences}} = \frac{{{n_pos}}}{{{n_eff}}} = {prop_pos:.4f}")
    _step_formula(rf"\text{{Median difference}} = {np.median(diffs):.4f}")

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": ["n_total", "n' (non-zero)", "S⁺", "S⁻", "S", "Exact p-value", "Prop. positive"],
            "Value": [
                f"{n_total}", f"{n_eff}", f"{n_pos}", f"{n_neg}", f"{k}",
                f"{p_value_exact:.6f}", f"{prop_pos:.4f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_data, file_name="sign_test_results.csv", mime="text/csv", key="dl_psign")


# =========================================================================
# 13. WILCOXON SIGNED-RANK TEST (PAIRED)
# =========================================================================
def _solved_wilcoxon_paired():
    _section("Wilcoxon Signed-Rank Test (Paired) — Solved Example")

    st.markdown(
        "The Wilcoxon Signed-Rank Test compares paired observations using both the **direction** and "
        "**magnitude** of differences (via ranks). It is the non-parametric alternative to the paired t-test."
    )

    raw_before = st.text_area(
        "Before / Control (comma-separated)",
        "23, 19, 25, 21, 28, 18, 22, 20, 27, 24",
        key="wilsign_before",
    )
    raw_after = st.text_area(
        "After / Treatment (comma-separated)",
        "30, 22, 28, 25, 31, 20, 24, 19, 29, 27",
        key="wilsign_after",
    )

    try:
        before = np.array([float(x.strip()) for x in raw_before.split(",") if x.strip()])
        after = np.array([float(x.strip()) for x in raw_after.split(",") if x.strip()])
    except ValueError:
        st.error("Invalid input.")
        return
    if len(before) != len(after):
        st.error("Both groups must have the same number of observations.")
        return

    if not st.button("Calculate Step-by-Step", key="btn_wilsign"):
        return

    diffs = after - before
    n_total = len(diffs)
    non_zero = diffs[diffs != 0]
    n = len(non_zero)
    abs_diffs = np.abs(non_zero)
    ranks = stats.rankdata(abs_diffs, method="average")
    signed_ranks = ranks * np.sign(non_zero)
    W_plus = float(np.sum(signed_ranks[signed_ranks > 0]))
    W_minus = float(abs(np.sum(signed_ranks[signed_ranks < 0])))
    W_stat = min(W_plus, W_minus)

    # Normal approximation with tie correction
    mu_W = n * (n + 1) / 4
    sigma_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    unique_abs = np.unique(abs_diffs)
    tie_correction = sum(t * (t**2 - 1) for u in unique_abs if (t := int(np.sum(abs_diffs == u))) > 1)
    if tie_correction:
        sigma_W = np.sqrt((n * (n + 1) * (2 * n + 1) - tie_correction / 2) / 24)
    z_stat = (W_stat - mu_W) / sigma_W
    p_value = 2 * stats.norm.cdf(z_stat)

    _section("Solution")

    _step_header(1, "State the hypotheses")
    _step_formula(r"H_0: \text{median difference} = 0 \quad \text{vs} \quad H_1: \text{median difference} \neq 0")

    _step_header(2, "Compute differences and absolute differences")
    tbl = []
    for i in range(n_total):
        d = diffs[i]
        tbl.append([i + 1, f"{before[i]:.2f}", f"{after[i]:.2f}", f"{d:+.2f}", f"{abs(d):.2f}"])
    df_d = pd.DataFrame(tbl, columns=["Pair", "Before", "After", "d", "|d|"])
    st.dataframe(df_d, use_container_width=True)
    csv_d = df_d.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_d, file_name="wilcoxon_paired_diffs.csv", mime="text/csv", key="dl_wp_diffs")

    _step_header(3, "Rank absolute differences (exclude zeros)")
    _step_formula(rf"\text{{Non-zero differences}}: n = {n}")
    tbl_r = []
    for i in range(n):
        ad = abs_diffs[i]
        r = ranks[i]
        sr = signed_ranks[i]
        sign_label = "+" if sr > 0 else "−"
        tbl_r.append([f"{ad:.2f}", f"{r:.1f}", sign_label, f"{sr:+.1f}"])
    df_r = pd.DataFrame(tbl_r, columns=["|d|", "Rank", "Sign", "Signed Rank"])
    st.dataframe(df_r, use_container_width=True)
    csv_r = df_r.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_r, file_name="wilcoxon_paired_ranks.csv", mime="text/csv", key="dl_wp_ranks")

    _step_header(4, "Sum the signed ranks")
    _step_formula(rf"W^+ = {W_plus:.1f}, \quad W^- = {W_minus:.1f}")
    _step_formula(rf"W = \min(W^+, W^-) = \min({W_plus:.1f}, {W_minus:.1f}) = {W_stat:.1f}")

    _step_header(5, "Normal approximation (large sample)")
    _step_formula(rf"\mu_W = \frac{{n(n+1)}}{{4}} = \frac{{{n} \times {n+1}}}{{4}} = {mu_W:.2f}")
    _step_formula(rf"\sigma_W = \sqrt{{\frac{{n(n+1)(2n+1)}}{{24}}}} = \sqrt{{\frac{{{n} \times {n+1} \times {2*n+1}}}{{24}}}} = {sigma_W:.4f}")
    _step_formula(rf"z = \frac{{W - \mu_W}}{{\sigma_W}} = \frac{{{W_stat:.1f} - {mu_W:.2f}}}{{{sigma_W:.4f}}} = {z_stat:.4f}")
    _step_formula(rf"p = 2 \times P(Z < {z_stat:.4f}) = {p_value:.6f}")

    _step_header(6, "Decision")
    alpha = 0.05
    if p_value < alpha:
        _step_formula(rf"p = {p_value:.6f} < \alpha = {alpha} \rightarrow \text{{reject H₀. The median difference is significantly different from zero.}}")
    else:
        _step_formula(rf"p = {p_value:.6f} \geq \alpha = {alpha} \rightarrow \text{{fail to reject H₀. The median difference is not significantly different from zero.}}")

    _step_header(7, "Effect size")
    r_effect = abs(z_stat) / np.sqrt(n)
    _step_formula(rf"r = \frac{{|z|}}{{\sqrt{{n}}}} = \frac{{{abs(z_stat):.4f}}}{{\sqrt{{{n}}}}} = {r_effect:.4f}")
    if r_effect < 0.1:
        interp = "negligible"
    elif r_effect < 0.3:
        interp = "small"
    elif r_effect < 0.5:
        interp = "medium"
    else:
        interp = "large"
    _step_formula(rf"r = {r_effect:.4f} \text{{ — }} {interp} \text{{ effect.}}")

    st.divider()
    summary = pd.DataFrame(
        {
            "Metric": ["n (non-zero)", "W⁺", "W⁻", "W", "μ_W", "σ_W", "z", "p-value", "Effect size r"],
            "Value": [
                f"{n}", f"{W_plus:.1f}", f"{W_minus:.1f}", f"{W_stat:.1f}",
                f"{mu_W:.2f}", f"{sigma_W:.4f}", f"{z_stat:.4f}",
                f"{p_value:.6f}", f"{r_effect:.4f}",
            ],
        }
    )
    st.subheader("Summary of Results")
    st.dataframe(summary, use_container_width=True)
    csv_data = summary.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv_data, file_name="wilcoxon_paired_results.csv", mime="text/csv", key="dl_wp")
