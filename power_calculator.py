import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

def _solve_with_mode(solver, analysis_mode, effect_size=None, nobs=None, alpha=None, power=None,
                     nobs_name="nobs", **extra_kw):
    """Call solver.solve_power with the parameter to solve for set to None."""
    effective = {}
    for k, v in [("effect_size", effect_size), ("alpha", alpha), ("power", power)]:
        if v is not None:
            effective[k] = v
    effective[nobs_name] = nobs
    effective.update(extra_kw)

    if analysis_mode == "Post Hoc":
        effective["power"] = None
    elif analysis_mode == "Sensitivity":
        effective["effect_size"] = None
    elif analysis_mode == "Criterion":
        effective["alpha"] = None
    else:  # A Priori (or Compromise, handled separately)
        effective[nobs_name] = None

    return solver.solve_power(**effective)


def _solve_n_binary(func_n, target_n, lo=2, hi=1000000):
    """Binary search an integer N to make func_n(N) >= target_n (max 100 iterations)."""
    for _ in range(100):
        mid = (lo + hi) // 2
        if func_n(mid):
            hi = mid
        else:
            lo = mid + 1
        if lo >= hi:
            break
    return hi


def _solve_compromise(solver, effect_size, nobs, cost_ratio, alternative="two-sided",
                      nobs_name="nobs", **extra_kw):
    """Brent's method to find α such that β/α = cost_ratio for given N and effect size.
    Returns dict {'alpha': adjusted_alpha, 'power': achieved_power}.
    """
    from scipy.optimize import brentq

    def f(alpha_candidate):
        kw = {nobs_name: nobs, "effect_size": effect_size, "alpha": alpha_candidate, "power": None}
        kw.update(extra_kw)
        try:
            achieved = solver.solve_power(**kw)
        except Exception:
            return 1e6
        beta = max(1e-10, 1 - achieved)
        return beta / alpha_candidate - cost_ratio

    lo, hi = 1e-8, 0.5
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return None
    alpha_solved = brentq(f, lo, hi)
    kw = {nobs_name: nobs, "effect_size": effect_size, "alpha": alpha_solved, "power": None}
    kw.update(extra_kw)
    power_solved = solver.solve_power(**kw)
    return {"alpha": alpha_solved, "power": power_solved}


def render_power_calculator(params, analysis_mode="A Priori"):
    """Render power analysis results (a_priori / post_hoc / sensitivity / criterion / compromise)."""

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
    _z_sub = r"z_{1-\alpha/2}" if alternative == "two-sided" else r"z_{1-\alpha}"

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
    computed_value = None
    computed_label = ""
    _skip_computation = False

    # Compromise mode: generic solver using z-approximation for all test types
    if analysis_mode == "Compromise":
        from scipy.optimize import brentq
        n_total = params.get("n_total", 0)
        if n_total > 0:
            es_val = params.get("effect_size")
            n_eff = n_total
            if atype == "one_prop":
                from statsmodels.stats.proportion import proportion_effectsize
                es_val = abs(proportion_effectsize(params["prop_alt"], params["prop_null"]))
            elif atype == "two_prop":
                from statsmodels.stats.proportion import proportion_effectsize
                es_val = abs(proportion_effectsize(params["p2"], params["p1"]))
                ratio = params.get("ratio", 1.0)
                n_eff = int(n_total / (1 + ratio)) if ratio > 0 else int(n_total / 2)
            elif atype == "mannwhitney":
                es_val = np.sqrt(3) * (params["effect_size"] - 0.5)
                ratio = params.get("ratio", 1.0)
                n_eff = int(n_total / (1 + ratio)) if ratio > 0 else int(n_total / 2)
            elif atype == "regression":
                es_val = params.get("effect_size", 0.15)
            elif atype == "logistic":
                ev_cr = params.get("event_rate", 0.3)
                or_cr = params.get("or", 2.0)
                p1_cr = (or_cr * ev_cr) / (1 - ev_cr + or_cr * ev_cr)
                d_cr = abs(p1_cr - ev_cr)
                p_bar_cr = (ev_cr + p1_cr) / 2
                se_cr = np.sqrt(2 * p_bar_cr * (1 - p_bar_cr))
                es_val = d_cr / se_cr if se_cr > 0 else 0

            if es_val is not None and es_val > 0:
                def f(alpha_candidate):
                    z_a = norm.ppf(1 - alpha_candidate / 2) if alternative == "two-sided" else norm.ppf(1 - alpha_candidate)
                    z_b = es_val * np.sqrt(n_eff) - z_a
                    achieved = norm.cdf(z_b)
                    beta = max(1e-10, 1 - achieved)
                    return beta / alpha_candidate - params.get("cost_ratio", 1.0)

                lo, hi = 1e-8, 0.5
                try:
                    if f(lo) * f(hi) < 0:
                        alpha_solved = brentq(f, lo, hi)
                        z_a_solved = norm.ppf(1 - alpha_solved / 2) if alternative == "two-sided" else norm.ppf(1 - alpha_solved)
                        z_b_solved = es_val * np.sqrt(n_eff) - z_a_solved
                        power_solved = norm.cdf(z_b_solved)
                        computed_value = {"alpha": alpha_solved, "power": power_solved}
                        n_per_group = n_total
                        _skip_computation = True
                except Exception:
                    pass

            if computed_value is None:
                explanation = "Compromise analysis could not converge with current parameters. Try adjusting N or cost ratio."
            else:
                explanation = (
                    f"Compromise power analysis with N = {n_total}, "
                    f"effect size = {es_val:.4f}, cost ratio β/α = {params.get('cost_ratio', 1.0):.2f}. "
                    f"Adjusted α = {computed_value['alpha']:.4f}, achieved power = {computed_value['power']:.1%}."
                )
            formula_latex = r"\text{Find } \alpha, \beta \text{ s.t. } \beta/\alpha = q \text{ via } \Phi^{-1}"

    if _skip_computation:
        pass
    elif atype == "one_mean":
        d = params["effect_size"]
        from statsmodels.stats.power import TTestPower

        solver = TTestPower()
        if d > 0:
            if analysis_mode == "Compromise":
                n_total = params.get("n_total", 0)
                n_per_group = n_total
                comp = _solve_compromise(solver, d, n_total, params.get("cost_ratio", 1.0), alternative=alternative)
                computed_value = comp
            else:
                raw = _solve_with_mode(
                    solver, analysis_mode, effect_size=d,
                    nobs=params.get("n_total"), alpha=alpha, power=power,
                    alternative=alternative,
                )
                if analysis_mode == "A Priori":
                    n_total = int(np.ceil(raw))
                    n_per_group = n_total
                else:
                    n_total = params.get("n_total", 0)
                    n_per_group = n_total
                computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required total sample size for a one-sample {tails.lower()} t-test "
                f"to detect Cohen's d = {d:.3f} with α = {alpha} and power = {power}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for a one-sample {tails.lower()} t-test "
                f"with N = {n_total}, Cohen's d = {d:.3f}, α = {alpha}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable effect size for a one-sample {tails.lower()} t-test "
                f"with N = {n_total}, α = {alpha}, power = {power}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for a one-sample {tails.lower()} t-test "
                f"with N = {n_total}, Cohen's d = {d:.3f}, power = {power}."
            )
        elif analysis_mode == "Compromise" and isinstance(computed_value, dict):
            explanation = (
                f"Compromise power analysis for a one-sample {tails.lower()} t-test "
                f"with N = {n_total}, Cohen's d = {d:.3f}, "
                f"cost ratio β/α = {params.get('cost_ratio', 1.0):.2f}. "
                f"Adjusted α = {computed_value['alpha']:.4f}, achieved power = {computed_value['power']:.1%}."
            )
        formula_latex = rf"n = \left( \frac{{{_z_sub} + z_{{1-\beta}}}}{{d}} \right)^2"

    elif atype == "two_means":
        d = params["effect_size"]
        ratio = params["ratio"]
        from statsmodels.stats.power import TTestIndPower

        solver = TTestIndPower()
        if d > 0:
            raw = _solve_with_mode(
                solver, analysis_mode, effect_size=d,
                nobs=params.get("n_total"), alpha=alpha, power=power,
                nobs_name="nobs1", ratio=ratio, alternative=alternative,
            )
            if analysis_mode == "A Priori":
                n1 = int(np.ceil(raw))
                n2 = int(np.ceil(n1 * ratio))
                n_total = n1 + n2
                n_per_group = (n1, n2)
            else:
                n_total = params.get("n_total", 0)
                n1 = int(np.ceil(n_total / (1 + ratio))) if ratio > 0 else 0
                n2 = n_total - n1
                n_per_group = (n1, n2)
            computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required sample size per group for an independent {tails.lower()} t-test "
                f"to detect Cohen's d = {d:.3f} with α = {alpha} and power = {power}, "
                f"allocation ratio n₂/n₁ = {ratio:.2f}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for an independent {tails.lower()} t-test "
                f"with N = {n_total}, Cohen's d = {d:.3f}, α = {alpha}, ratio = {ratio:.2f}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable effect size for an independent {tails.lower()} t-test "
                f"with N = {n_total}, α = {alpha}, power = {power}, ratio = {ratio:.2f}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for an independent {tails.lower()} t-test "
                f"with N = {n_total}, Cohen's d = {d:.3f}, power = {power}, ratio = {ratio:.2f}."
            )
        formula_latex = rf"n_1 = 2 \left( \frac{{{_z_sub} + z_{{1-\beta}}}}{{d}} \right)^2 \quad n_2 = r \cdot n_1"

    elif atype == "paired":
        d = params["effect_size"]
        from statsmodels.stats.power import TTestPower

        solver = TTestPower()
        if d > 0:
            raw = _solve_with_mode(
                solver, analysis_mode, effect_size=d,
                nobs=params.get("n_total"), alpha=alpha, power=power,
                alternative=alternative,
            )
            if analysis_mode == "A Priori":
                n_total = int(np.ceil(raw))
                n_per_group = n_total
            else:
                n_total = params.get("n_total", 0)
                n_per_group = n_total
            computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required number of pairs for a paired {tails.lower()} t-test "
                f"to detect Cohen's d_z = {d:.3f} with α = {alpha} and power = {power}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for a paired {tails.lower()} t-test "
                f"with {n_total} pairs, Cohen's d_z = {d:.3f}, α = {alpha}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable effect size for a paired {tails.lower()} t-test "
                f"with {n_total} pairs, α = {alpha}, power = {power}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for a paired {tails.lower()} t-test "
                f"with {n_total} pairs, Cohen's d_z = {d:.3f}, power = {power}."
            )
        formula_latex = rf"n = \left( \frac{{{_z_sub} + z_{{1-\beta}}}}{{d_z}} \right)^2"

    elif atype == "one_prop":
        p0 = params["prop_null"]
        p1 = params["prop_alt"]
        from statsmodels.stats.proportion import proportion_effectsize
        from statsmodels.stats.power import NormalIndPower

        d_eff = proportion_effectsize(p1, p0)
        solver = NormalIndPower()
        if abs(d_eff) > 0:
            raw = _solve_with_mode(
                solver, analysis_mode, effect_size=abs(d_eff),
                nobs=params.get("n_total"), alpha=alpha, power=power,
                nobs_name="nobs1", alternative=alternative,
            )
            if analysis_mode == "A Priori":
                n_total = int(np.ceil(raw))
                n_per_group = n_total
            else:
                n_total = params.get("n_total", 0)
                n_per_group = n_total
            computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required sample size for a one-sample proportion test "
                f"to detect a difference from {p0} to {p1} "
                f"with α = {alpha} and power = {power}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for a one-sample proportion test "
                f"with N = {n_total}, difference {p0} → {p1}, α = {alpha}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable proportion difference "
                f"for a one-sample proportion test with N = {n_total}, α = {alpha}, power = {power}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for a one-sample proportion test "
                f"with N = {n_total}, difference {p0} → {p1}, power = {power}."
            )
        formula_latex = rf"n = \left( \frac{{{_z_sub} \sqrt{{p_0(1-p_0)}} + z_{{1-\beta}} \sqrt{{p_1(1-p_1)}}}}{{{{p_1 - p_0}}}} \right)^2"

    elif atype == "two_prop":
        p1 = params["p1"]
        p2 = params["p2"]
        ratio = params["ratio"]
        from statsmodels.stats.proportion import proportion_effectsize
        from statsmodels.stats.power import NormalIndPower

        d_eff = proportion_effectsize(p2, p1)
        solver = NormalIndPower()
        if abs(d_eff) > 0:
            raw = _solve_with_mode(
                solver, analysis_mode, effect_size=abs(d_eff),
                nobs=params.get("n_total"), alpha=alpha, power=power,
                nobs_name="nobs1", ratio=ratio, alternative=alternative,
            )
            if analysis_mode == "A Priori":
                n1 = int(np.ceil(raw))
                n2 = int(np.ceil(n1 * ratio))
                n_total = n1 + n2
                n_per_group = (n1, n2)
            else:
                n_total = params.get("n_total", 0)
                n1 = int(np.ceil(n_total / (1 + ratio))) if ratio > 0 else 0
                n2 = n_total - n1
                n_per_group = (n1, n2)
            computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required sample size per group for a two-proportion z-test "
                f"to detect a difference between {p1} and {p2} "
                f"with α = {alpha}, power = {power}, ratio = {ratio:.2f}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for a two-proportion z-test "
                f"with N = {n_total}, proportions {p1} vs {p2}, α = {alpha}, ratio = {ratio:.2f}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable proportion difference "
                f"for a two-proportion z-test with N = {n_total}, α = {alpha}, power = {power}, ratio = {ratio:.2f}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for a two-proportion z-test "
                f"with N = {n_total}, proportions {p1} vs {p2}, power = {power}, ratio = {ratio:.2f}."
            )
        formula_latex = rf"n_1 = \left( \frac{{{_z_sub} \sqrt{{2\bar{{p}}(1-\bar{{p}})}} + z_{{1-\beta}} \sqrt{{p_1(1-p_1) + p_2(1-p_2)}}}}{{{{p_1 - p_2}}}} \right)^2"

    elif atype == "anova":
        f_eff = params["effect_size"]
        k = params["k"]
        from statsmodels.stats.power import FTestAnovaPower

        solver = FTestAnovaPower()
        if f_eff > 0:
            raw = _solve_with_mode(
                solver, analysis_mode, effect_size=f_eff,
                nobs=params.get("n_total"), alpha=alpha, power=power,
                k_groups=k,
            )
            if analysis_mode == "A Priori":
                n_per_g = int(np.ceil(raw))
                n_total = n_per_g * k
                n_per_group = n_per_g
            else:
                n_total = params.get("n_total", 0)
                n_per_g = max(1, n_total // k)
                n_per_group = n_per_g
            computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required sample size per group for a one-way ANOVA with {k} groups "
                f"to detect Cohen's f = {f_eff:.3f} with α = {alpha} and power = {power}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for a one-way ANOVA with {k} groups "
                f"with N = {n_total}, Cohen's f = {f_eff:.3f}, α = {alpha}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable effect size for a one-way ANOVA with {k} groups "
                f"with N = {n_total}, α = {alpha}, power = {power}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for a one-way ANOVA with {k} groups "
                f"with N = {n_total}, Cohen's f = {f_eff:.3f}, power = {power}."
            )
        formula_latex = r"n = \frac{\text{from non-central }F\text{ distribution}}{k} \quad f = \frac{\sigma_{\text{between}}}{\sigma_{\text{within}}}"

    elif atype == "correlation":
        r_val = params["effect_size"]
        import math as cmath

        fisher_z = cmath.atanh(r_val)
        if analysis_mode == "A Priori":
            n_total = int(np.ceil(3 + ((z_alpha + z_beta) / fisher_z) ** 2))
        elif analysis_mode == "Post Hoc":
            n_total = params.get("n_total", 0)
            z_beta_c = (cmath.sqrt(max(0, n_total - 3)) * fisher_z) - z_alpha
            computed_value = norm.cdf(z_beta_c)
        elif analysis_mode == "Sensitivity":
            n_total = params.get("n_total", 0)
            z_beta_c = norm.ppf(power)
            fisher_z_c = (z_alpha + z_beta_c) / cmath.sqrt(max(1, n_total - 3))
            computed_value = cmath.tanh(fisher_z_c)
        elif analysis_mode == "Criterion":
            n_total = params.get("n_total", 0)
            z_beta_c = norm.ppf(power)
            z_alpha_c = cmath.sqrt(max(0, n_total - 3)) * fisher_z - z_beta_c
            computed_value = 2 * (1 - norm.cdf(z_alpha_c))
        n_per_group = n_total
        if analysis_mode == "A Priori":
            explanation = (
                f"Required sample size to detect a Pearson correlation of r = {r_val:.3f} "
                f"with α = {alpha} and power = {power} ({tails.lower()}), "
                f"based on Fisher's z-transformation."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power to detect Pearson correlation r = {r_val:.3f} "
                f"with N = {n_total}, α = {alpha}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable Pearson correlation "
                f"with N = {n_total}, α = {alpha}, power = {power}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level to detect Pearson correlation r = {r_val:.3f} "
                f"with N = {n_total}, power = {power}."
            )
        formula_latex = rf"n = 3 + \left( \frac{{{_z_sub} + z_{{1-\beta}}}}{{\text{{arctanh}}(r)}} \right)^2"

    elif atype == "regression":
        f2 = params["effect_size"]
        k = params["k"]
        if f2 > 0:
            from scipy.stats import ncf as noncentral_f, f as f_dist

            if analysis_mode == "A Priori":
                for n_candidate in range(k + 2, 10000):
                    dfd = n_candidate - k - 1
                    ncp = f2 * n_candidate
                    f_crit = f_dist.ppf(1 - alpha, k, dfd)
                    pwr_cur = 1 - noncentral_f.cdf(f_crit, k, dfd, ncp)
                    if pwr_cur >= power:
                        n_total = n_candidate
                        break
            elif analysis_mode == "Post Hoc":
                n_total = params.get("n_total", 0)
                dfd = n_total - k - 1
                ncp = f2 * n_total
                f_crit = f_dist.ppf(1 - alpha, k, dfd)
                computed_value = 1 - noncentral_f.cdf(f_crit, k, dfd, ncp)
            elif analysis_mode == "Sensitivity":
                n_total = params.get("n_total", 0)
                from scipy.optimize import brentq
                def power_for_f2(f2_try):
                    dfd = n_total - k - 1
                    ncp = f2_try * n_total
                    f_crit = f_dist.ppf(1 - alpha, k, dfd)
                    return 1 - noncentral_f.cdf(f_crit, k, dfd, ncp) - power
                computed_value = brentq(power_for_f2, 1e-6, 5.0)
            elif analysis_mode == "Criterion":
                n_total = params.get("n_total", 0)
                from scipy.optimize import brentq
                def power_for_alpha(a_try):
                    dfd = n_total - k - 1
                    ncp = f2 * n_total
                    f_crit = f_dist.ppf(1 - a_try, k, dfd)
                    return 1 - noncentral_f.cdf(f_crit, k, dfd, ncp) - power
                computed_value = brentq(power_for_alpha, 1e-8, 0.5)
            n_per_group = n_total
        if analysis_mode == "A Priori":
            explanation = (
                f"Required total sample size for multiple linear regression "
                f"with {k} predictor(s) to detect Cohen's f² = {f2:.3f} "
                f"(R² = {f2 / (1 + f2):.3f}) with α = {alpha} and power = {power}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for multiple linear regression "
                f"with {k} predictor(s), N = {n_total}, Cohen's f² = {f2:.3f}, α = {alpha}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable Cohen's f² for multiple linear regression "
                f"with {k} predictor(s), N = {n_total}, α = {alpha}, power = {power}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for multiple linear regression "
                f"with {k} predictor(s), N = {n_total}, Cohen's f² = {f2:.3f}, power = {power}."
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
            raw = _solve_with_mode(
                solver, analysis_mode, effect_size=d_eff_log,
                nobs=params.get("n_total"), alpha=alpha, power=power,
                nobs_name="nobs1", alternative=alternative,
            )
            if analysis_mode == "A Priori":
                n_base = int(np.ceil(raw))
                n_total = max(n_base, 10 * k)
            else:
                n_total = params.get("n_total", 0)
            n_per_group = n_total
            computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required total sample size for logistic regression "
                f"with {k} predictor(s) to detect OR = {or_val:.2f} "
                f"with baseline event rate = {ev_rate:.2f}, α = {alpha}, power = {power}. "
                f"Lower bound of 10 × {k} = {10 * k} events per predictor applied."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for logistic regression "
                f"with {k} predictor(s), N = {n_total}, OR = {or_val:.2f}, "
                f"event rate = {ev_rate:.2f}, α = {alpha}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable odds ratio for logistic regression "
                f"with {k} predictor(s), N = {n_total}, event rate = {ev_rate:.2f}, "
                f"α = {alpha}, power = {power}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for logistic regression "
                f"with {k} predictor(s), N = {n_total}, OR = {or_val:.2f}, "
                f"event rate = {ev_rate:.2f}, power = {power}."
            )
        formula_latex = rf"n = \frac{{({_z_sub} + z_{{1-\beta}})^2 \bar{{p}}(1-\bar{{p}})}}{{{{(p_1 - p_0)^2}}}} \quad \text{{min }} 10k"

    elif atype == "chisq":
        w = params["effect_size"]
        df = params["df"]
        from statsmodels.stats.power import GofChisquarePower

        solver = GofChisquarePower()
        if w > 0:
            raw = _solve_with_mode(
                solver, analysis_mode, effect_size=w,
                nobs=params.get("n_total"), alpha=alpha, power=power,
                nobs_name="nobs", n_bins=df + 1,
            )
            if analysis_mode == "A Priori":
                n_total = int(np.ceil(raw))
            else:
                n_total = params.get("n_total", 0)
            n_per_group = n_total
            computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required total sample size for a chi-square test "
                f"with {df} degree(s) of freedom to detect Cohen's w = {w:.3f} "
                f"with α = {alpha} and power = {power}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for a chi-square test "
                f"with N = {n_total}, {df} DF, Cohen's w = {w:.3f}, α = {alpha}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable Cohen's w for a chi-square test "
                f"with N = {n_total}, {df} DF, α = {alpha}, power = {power}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for a chi-square test "
                f"with N = {n_total}, {df} DF, Cohen's w = {w:.3f}, power = {power}."
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
            raw = _solve_with_mode(
                solver, analysis_mode, effect_size=d_mw,
                nobs=params.get("n_total"), alpha=alpha, power=power,
                nobs_name="nobs1", ratio=ratio, alternative=alternative,
            )
            if analysis_mode == "A Priori":
                n1 = int(np.ceil(raw / are))
                n2 = int(np.ceil(n1 * ratio))
                n_total = n1 + n2
                n_per_group = (n1, n2)
            else:
                n_total = params.get("n_total", 0)
                n1 = int(np.ceil(n_total / (1 + ratio))) if ratio > 0 else 0
                n2 = n_total - n1
                n_per_group = (n1, n2)
            computed_value = raw
        if analysis_mode == "A Priori":
            explanation = (
                f"Required sample size for Mann-Whitney / Wilcoxon test "
                f"to detect P(X>Y) = {p_val:.3f} (d ≈ {d_mw:.3f}) with ARE = {are:.3f}, "
                f"α = {alpha}, power = {power}, ratio = {ratio:.2f}."
            )
        elif analysis_mode == "Post Hoc":
            explanation = (
                f"Achieved power for Mann-Whitney / Wilcoxon test "
                f"with N = {n_total}, P(X>Y) = {p_val:.3f}, α = {alpha}, ratio = {ratio:.2f}."
            )
        elif analysis_mode == "Sensitivity":
            explanation = (
                f"Minimum detectable P(X>Y) for Mann-Whitney / Wilcoxon test "
                f"with N = {n_total}, α = {alpha}, power = {power}, ratio = {ratio:.2f}."
            )
        elif analysis_mode == "Criterion":
            explanation = (
                f"Required significance level for Mann-Whitney / Wilcoxon test "
                f"with N = {n_total}, P(X>Y) = {p_val:.3f}, power = {power}, ratio = {ratio:.2f}."
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
        formula_latex = rf"E = \frac{{({_z_sub}+z_{{1-\beta}})^2 (r+1)^2}}{{r (\log HR)^2}} \quad N = \frac{{E}}{{P(\text{{event}})}}"

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
        formula_latex = rf"E = \frac{{({_z_sub}+z_{{1-\beta}})^2}}{{\sigma_x^2 \beta^2 (1-R^2)}} \quad N = \frac{{E}}{{P(\text{{event}})}}"

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
                r"n_1 = \frac{(z_{1-\alpha}+z_{1-\beta})^2 \, 2\bar{p}(1-\bar{p})}{(\delta - |p_1-p_2|)^2}"
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
                r"n_1 = \frac{(z_{1-\alpha}+z_{1-\beta})^2 \sigma^2 (1+1/r)}{(\delta - |d|)^2}"
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
        formula_latex = rf"n_{{\text{{cases}}}} = \frac{{({_z_sub}+z_{{1-\beta}})^2 V(AUC)}}{{{{(AUC - 0.5)^2}}}}"

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
        formula_latex = rf"n = \frac{{({_z_sub}+z_{{1-\beta}})^2 \kappa_0(1-\kappa_0)}}{{(\kappa - \kappa_0)^2}}"

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
            r"n = \left(\frac{t_{1-\alpha/2, n-1} \sigma}{w}\right)^2"
            if param_type == "Mean"
            else r"n = \frac{z_{1-\alpha/2}^2 p(1-p)}{w^2}"
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
        formula_latex = rf"n = \frac{{1}}{{\text{{ARE}}}} \left( \frac{{{_z_sub} + z_{{1-\beta}}}}{{\sqrt{{3}} \cdot (P - 0.5)}} \right)^2"

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
            rf"n = \frac{{({_z_sub} + z_{{1-\beta}})^2 (p_b + p_c)}}{{{{(p_b - p_c)^2}}}}"
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
        _z_sub_phi = r"1-\alpha/2" if alternative == "two-sided" else r"1-\alpha"
        formula_latex = rf"n = \min\{{n: 1 - \Phi(\Phi^{{-1}}({_z_sub_phi}) - \sqrt{{n}} d) \geq \text{{power}}\}}"

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

    # --- Compromise fallback for test types not explicitly handled ---
    if analysis_mode == "Compromise" and computed_value is None and n_total is not None and n_total > 0:
        from scipy.optimize import brentq
        es_val = params.get("effect_size", None)
        if es_val is not None and es_val > 0:
            from statsmodels.stats.power import TTestPower, NormalIndPower
            if atype in ("one_mean", "paired", "correlation"):
                solver = TTestPower()
            elif atype in ("one_prop", "mannwhitney", "wilcoxon_sr"):
                solver = NormalIndPower()
            else:
                solver = None
            if solver is not None:
                cost_ratio = params.get("cost_ratio", 1.0)
                try:
                    comp = _solve_compromise(solver, es_val, n_total, cost_ratio, alternative=alternative)
                    if comp is not None:
                        computed_value = comp
                        explanation = (
                            f"Compromise power analysis for {atype} with N = {n_total}, "
                            f"effect size = {es_val:.4f}, cost ratio β/α = {cost_ratio:.2f}. "
                            f"Adjusted α = {computed_value['alpha']:.4f}, achieved power = {computed_value['power']:.1%}."
                        )
                except Exception:
                    pass

    # --- Apply Attrition Adjustment ---
    n_total_raw = n_total
    if analysis_mode == "A Priori" and dropout_rate > 0 and n_total is not None:
        n_total = int(np.ceil(n_total / (1 - dropout_rate)))
        if isinstance(n_per_group, tuple):
            n1_adj = int(np.ceil(n_per_group[0] / (1 - dropout_rate)))
            n2_adj = int(np.ceil(n_per_group[1] / (1 - dropout_rate)))
            n_per_group = (n1_adj, n2_adj)
        elif n_per_group is not None:
            n_per_group = int(np.ceil(n_per_group / (1 - dropout_rate)))

    # --- Display Results ---

    if n_total is None and computed_value is None:
        st.error(
            "Effect size is too small — consider a larger effect or different design."
        )
        return

    mode_titles = {
        "A Priori": "Sample Size Results",
        "Post Hoc": "Post-Hoc Power Analysis Results",
        "Sensitivity": "Sensitivity Analysis Results",
        "Criterion": "Criterion Analysis Results",
        "Compromise": "Compromise Power Analysis Results",
    }
    st.subheader(mode_titles.get(analysis_mode, "Power Analysis Results"))

    if analysis_mode == "A Priori":
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

    elif analysis_mode == "Post Hoc":
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
            pwr_display = f"{computed_value:.1%}" if computed_value is not None else "—"
            st.metric("Achieved Power", pwr_display)
            st.metric("Alpha (α)", f"{alpha:.3f}")

    elif analysis_mode == "Sensitivity":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sample Size (N)", n_total)
        with col2:
            st.metric("Power", f"{power:.0%}")
            st.metric("Alpha (α)", f"{alpha:.3f}")
        with col3:
            es_display = f"{computed_value:.4f}" if computed_value is not None else "—"
            st.metric("Min Detectable Effect", es_display)

    elif analysis_mode == "Criterion":
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
            st.metric("Power", f"{power:.0%}")
            a_display = f"{computed_value:.4f}" if computed_value is not None else "—"
            st.metric("Required α", a_display)

    elif analysis_mode == "Compromise":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sample Size (N)", n_total)
        with col2:
            st.metric("Adjusted α", f"{computed_value['alpha']:.4f}" if isinstance(computed_value, dict) else "—")
        with col3:
            st.metric("Achieved Power", f"{computed_value['power']:.1%}" if isinstance(computed_value, dict) else "—")

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

    es_val = params.get('effect_size', params.get('or', params.get('w', params.get('f2', params.get('f', 'N/A')))))
    es_str = f"{es_val:.4f}" if isinstance(es_val, (int, float, np.integer, np.floating)) else str(es_val)
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
Effect Size: {es_str}
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
Effect Size: {es_str}
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


