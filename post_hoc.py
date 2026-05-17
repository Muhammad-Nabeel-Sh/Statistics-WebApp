import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as sp_stats
from itertools import combinations


# =========================
# HELPERS
# =========================

def _apply_holm(p_values):
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(m)
    for i in range(m):
        adjusted[i] = np.clip(sorted_p[i] * (m - i), 0, 1)
    # Enforce monotonicity
    for i in range(m - 2, -1, -1):
        adjusted[i] = max(adjusted[i], adjusted[i + 1])
    result = np.zeros(m)
    result[sorted_idx] = adjusted
    return result.tolist()


def _apply_sidak(p_values):
    m = len(p_values)
    return [np.clip(1 - (1 - p) ** m, 0, 1) for p in p_values]


def _pairwise_result(label_i, label_j, i, j, diff, p, cl, cu):
    return {"Pair": f"{label_i} vs {label_j}", "i": i, "j": j,
            "Diff": diff, "p": p, "CI_low": cl, "CI_high": cu}


# =========================
# PARAMETRIC — ANOVA
# =========================

def _pooled_stats(groups):
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    vars_ = np.array([g.var(ddof=1) for g in groups])
    n_total = np.sum(n)
    n_groups = len(groups)
    pooled_var = np.sum((n - 1) * vars_) / (n_total - n_groups)
    df = n_total - n_groups
    return n, means, vars_, pooled_var, df


def _raw_pairwise_t(groups):
    n, means, vars_, pooled_var, df = _pooled_stats(groups)
    labels = [f"Group {i+1}" for i in range(len(groups))]
    results = []
    for (i, j) in combinations(range(len(groups)), 2):
        diff = means[i] - means[j]
        se = np.sqrt(pooled_var * (1/n[i] + 1/n[j]))
        t_stat = diff / se if se > 0 else 0
        p = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df))
        cl = diff - sp_stats.t.ppf(0.975, df) * se
        cu = diff + sp_stats.t.ppf(0.975, df) * se
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff, p, cl, cu))
    return results


def _fisher_lsd(groups):
    return _raw_pairwise_t(groups)


def _bonferroni(groups):
    results = _raw_pairwise_t(groups)
    n = len(results)
    for r in results:
        r["p"] = np.clip(r["p"] * n, 0, 1)
    return results


def _holm_bonferroni(groups):
    results = _raw_pairwise_t(groups)
    p_vals = [r["p"] for r in results]
    adjusted = _apply_holm(p_vals)
    for r, p in zip(results, adjusted):
        r["p"] = p
    return results


def _sidak(groups):
    results = _raw_pairwise_t(groups)
    m = len(results)
    for r in results:
        r["p"] = np.clip(1 - (1 - r["p"]) ** m, 0, 1)
    return results


def _tukey_hsd(groups):
    from scipy.stats import studentized_range
    n, means, vars_, pooled_var, df = _pooled_stats(groups)
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n_eff = sp_stats.hmean(n) if np.all(n > 0) else n.min()
    mse = pooled_var
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = means[i] - means[j]
        se = np.sqrt(mse / n_eff)
        q = abs(diff) / se if se > 0 else 0
        p = 1 - studentized_range.cdf(q, n_groups, df)
        cl = diff - studentized_range.ppf(0.95, n_groups, df) * se
        cu = diff + studentized_range.ppf(0.95, n_groups, df) * se
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff, p, cl, cu))
    return results


def _scheffe(groups):
    n, means, vars_, pooled_var, df_w = _pooled_stats(groups)
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    df_b = n_groups - 1
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = means[i] - means[j]
        se = np.sqrt(pooled_var * (1/n[i] + 1/n[j]))
        f_stat = (diff**2) / ((df_b) * se**2) if se > 0 else 0
        p = 1 - sp_stats.f.cdf(f_stat, df_b, df_w)
        crit = np.sqrt(df_b * sp_stats.f.ppf(0.95, df_b, df_w))
        cl = diff - crit * se
        cu = diff + crit * se
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff, p, cl, cu))
    return results


def _dunnett(groups, control=0):
    n, means, vars_, pooled_var, df = _pooled_stats(groups)
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    results = []
    for j in range(n_groups):
        if j == control:
            continue
        diff = means[j] - means[control]
        se = np.sqrt(pooled_var * (1/n[j] + 1/n[control]))
        t_stat = diff / se if se > 0 else 0
        p = 1 - sp_stats.norm.cdf(abs(t_stat))
        p = np.clip(p * (n_groups - 1), 0, 1)
        cl = diff - sp_stats.t.ppf(0.975, df) * se
        cu = diff + sp_stats.t.ppf(0.975, df) * se
        results.append(_pairwise_result(labels[control], labels[j], control, j, diff, p, cl, cu))
    return results


def _games_howell(groups):
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    vars_ = np.array([g.var(ddof=1) for g in groups])
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = means[i] - means[j]
        se = np.sqrt(vars_[i]/n[i] + vars_[j]/n[j])
        df_num = (vars_[i]/n[i] + vars_[j]/n[j])**2
        df_den = (vars_[i]/n[i])**2/(n[i]-1) + (vars_[j]/n[j])**2/(n[j]-1)
        df = df_num / df_den if df_den > 0 else 1
        t_stat = abs(diff) / se if se > 0 else 0
        p = 2 * (1 - sp_stats.t.cdf(t_stat, df))
        from scipy.stats import studentized_range
        q_crit = studentized_range.ppf(0.95, n_groups, df) / np.sqrt(2)
        cl = diff - q_crit * se
        cu = diff + q_crit * se
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff, p, cl, cu))
    return results


def _newman_keuls(groups):
    """Stepwise studentized range (NK). Uses Tukey critical values (conservative)."""
    results = _tukey_hsd(groups)
    n_groups = len(groups)
    n, means, _, pooled_var, df = _pooled_stats(groups)
    n_eff = sp_stats.hmean(n) if np.all(n > 0) else n.min()
    mse = pooled_var
    se = np.sqrt(mse / n_eff)
    labels = [f"Group {i+1}" for i in range(len(groups))]
    from scipy.stats import studentized_range
    sorted_idx = np.argsort(means)
    rank = {sorted_idx[i]: i for i in range(n_groups)}
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = means[i] - means[j]
        step = abs(rank[i] - rank[j]) + 1
        q_crit = studentized_range.ppf(0.95, step, df)
        p = 1 - studentized_range.cdf(abs(diff) / se if se > 0 else 0, step, df)
        cl = diff - q_crit * se
        cu = diff + q_crit * se
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff, p, cl, cu))
    return results


# =========================
# PARAMETRIC — REPEATED MEASURES
# =========================

def _paired_ttest(x, y):
    d = np.array(x) - np.array(y)
    n = len(d)
    mean_d = d.mean()
    se_d = d.std(ddof=1) / np.sqrt(n)
    t_stat = mean_d / se_d if se_d > 0 else 0
    p = 2 * (1 - sp_stats.t.cdf(abs(t_stat), n - 1))
    cl = mean_d - sp_stats.t.ppf(0.975, n - 1) * se_d
    cu = mean_d + sp_stats.t.ppf(0.975, n - 1) * se_d
    return mean_d, p, cl, cu


def _pairwise_paired(groups):
    n_groups = len(groups)
    labels = [f"Time {i+1}" for i in range(n_groups)]
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff, p, cl, cu = _paired_ttest(groups[i], groups[j])
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff, p, cl, cu))
    return results


def _bonferroni_paired(groups):
    results = _pairwise_paired(groups)
    n = len(results)
    for r in results:
        r["p"] = np.clip(r["p"] * n, 0, 1)
    return results


def _holm_paired(groups):
    results = _pairwise_paired(groups)
    p_vals = [r["p"] for r in results]
    adjusted = _apply_holm(p_vals)
    for r, p in zip(results, adjusted):
        r["p"] = p
    return results


# =========================
# PARAMETRIC — MANOVA (simplified: univariate F per DV)
# =========================

def _discriminant_comparisons(groups):
    """Compare each group via pairwise Hotelling-like approach."""
    n_groups = len(groups)
    labels = [f"DV {i+1}" for i in range(n_groups)]
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        x, y = np.array(groups[i]), np.array(groups[j])
        from scipy.stats import ttest_ind
        t_stat, p = ttest_ind(x, y, equal_var=True)
        diff = x.mean() - y.mean()
        cl = diff - sp_stats.t.ppf(0.975, len(x) + len(y) - 2) * np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))
        cu = diff + sp_stats.t.ppf(0.975, len(x) + len(y) - 2) * np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff, p, cl, cu))
    return results


def _canonical_contrasts(groups):
    """Contrast-based comparison with Roy-Bargmann stepdown correction."""
    results = _discriminant_comparisons(groups)
    n = len(results)
    for r in results:
        r["p"] = np.clip(r["p"] * n, 0, 1)
    return results


# =========================
# NON-PARAMETRIC — KRUSKAL-WALLIS
# =========================

def _dunn_test(groups):
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    all_data = np.concatenate(groups)
    all_ranks = sp_stats.rankdata(all_data)
    idx = np.concatenate([np.full(n[i], i) for i in range(n_groups)])
    mean_ranks = np.array([all_ranks[idx == i].mean() for i in range(n_groups)])
    ties = np.unique(all_data, return_counts=True)[1]
    tie_corr = 1 - np.sum(ties**3 - ties) / (len(all_data)**3 - len(all_data))
    n_total = len(all_data)
    var_overall = (n_total * (n_total + 1) / 12) * tie_corr
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        z = (mean_ranks[i] - mean_ranks[j]) / np.sqrt(var_overall * (1/n[i] + 1/n[j])) if var_overall > 0 else 0
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff=mean_ranks[i] - mean_ranks[j], p=p, cl=0, cu=0))
    return results


def _conover_test(groups):
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    all_data = np.concatenate(groups)
    all_ranks = sp_stats.rankdata(all_data)
    idx = np.concatenate([np.full(n[i], i) for i in range(n_groups)])
    mean_ranks = np.array([all_ranks[idx == i].mean() for i in range(n_groups)])
    n_total = len(all_data)
    grand_mean_rank = all_ranks.mean()
    var_rank = np.sum((all_ranks - grand_mean_rank)**2) / (n_total - 1)
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        t_num = abs(mean_ranks[i] - mean_ranks[j])
        t_den = np.sqrt(var_rank * (n_total - 1 - np.sum(mean_ranks**2 / n_total)) /
                        (n_total - n_groups) * (1/n[i] + 1/n[j]))
        t_stat = t_num / t_den if t_den > 0 else 0
        df = n_total - n_groups
        p = 2 * (1 - sp_stats.t.cdf(t_stat, df))
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff=mean_ranks[i] - mean_ranks[j], p=p, cl=0, cu=0))
    return results


def _dscf(groups):
    """Dwass-Steel-Critchlow-Fligner: pairwise MW with studentized range."""
    from scipy.stats import studentized_range
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        x, y = groups[i], groups[j]
        u_stat, p_raw = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        n_total = len(x) + len(y)
        z_val = sp_stats.norm.ppf(1 - p_raw / 2)
        q_val = z_val * np.sqrt(2)
        p = 1 - studentized_range.cdf(q_val, n_groups, np.inf)
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff=np.median(x) - np.median(y), p=p, cl=0, cu=0))
    return results


# =========================
# NON-PARAMETRIC — FRIEDMAN
# =========================

def _nemenyi_test(groups):
    from scipy.stats import studentized_range
    n_groups = len(groups)
    labels = [f"Time {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    all_data = np.concatenate(groups)
    all_ranks = sp_stats.rankdata(all_data)
    idx = np.concatenate([np.full(n[i], i) for i in range(n_groups)])
    mean_ranks = np.array([all_ranks[idx == i].mean() for i in range(n_groups)])
    n_eff = sp_stats.hmean(n) if np.all(n > 0) else n.min()
    se = np.sqrt(n_groups * (n_groups + 1) / (12 * n_eff))
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = mean_ranks[i] - mean_ranks[j]
        q = abs(diff) / se if se > 0 else 0
        p = 1 - studentized_range.cdf(q, n_groups, np.inf)
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff=diff, p=p, cl=0, cu=0))
    return results


def _conover_friedman(groups):
    """Conover's test for Friedman: pairwise rank comparison with t-distribution."""
    n_groups = len(groups)
    labels = [f"Time {i+1}" for i in range(n_groups)]
    n_subjects = len(groups[0])
    # Rank within each subject
    ranks = np.array([sp_stats.rankdata([g[i] for g in groups]) for i in range(n_subjects)])
    mean_ranks = ranks.mean(axis=0)
    # Overall mean rank
    grand_mean = (n_groups + 1) / 2
    ss_total = np.sum((ranks - grand_mean)**2)
    ss_subjects = n_groups * np.sum((ranks.mean(axis=1) - grand_mean)**2)
    var_rank = (ss_total - ss_subjects) / ((n_subjects - 1) * (n_groups - 1))
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = mean_ranks[i] - mean_ranks[j]
        se = np.sqrt(2 * var_rank / n_subjects)
        t_stat = abs(diff) / se if se > 0 else 0
        df = (n_subjects - 1) * (n_groups - 1)
        p = 2 * (1 - sp_stats.t.cdf(t_stat, df))
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff=diff, p=p, cl=0, cu=0))
    return results


def _wilcoxon_pairwise(groups):
    """Pairwise Wilcoxon signed-rank with Bonferroni correction."""
    n_groups = len(groups)
    labels = [f"Time {i+1}" for i in range(n_groups)]
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        from scipy.stats import wilcoxon
        diff_arr = np.array(groups[i]) - np.array(groups[j])
        mask = diff_arr != 0
        if mask.sum() > 1:
            w_stat, p = wilcoxon(np.array(groups[i])[mask], np.array(groups[j])[mask])
        elif mask.sum() == 1:
            p = 1.0
        else:
            p = 1.0
        diff = np.median(np.array(groups[i]) - np.array(groups[j]))
        results.append(_pairwise_result(labels[i], labels[j], i, j, diff=diff, p=p, cl=0, cu=0))
    n = len(results)
    for r in results:
        r["p"] = np.clip(r["p"] * n, 0, 1)
    return results


# =========================
# REGISTRY
# =========================

POST_HOC_METHODS = {
    # Parametric — ANOVA
    "Fisher LSD": {"type": "parametric", "fn": _fisher_lsd, "ci": True, "context": "ANOVA"},
    "Tukey HSD": {"type": "parametric", "fn": _tukey_hsd, "ci": True, "context": "ANOVA"},
    "Bonferroni": {"type": "parametric", "fn": _bonferroni, "ci": True, "context": "ANOVA"},
    "Holm-Bonferroni": {"type": "parametric", "fn": _holm_bonferroni, "ci": True, "context": "ANOVA"},
    "Šidák": {"type": "parametric", "fn": _sidak, "ci": True, "context": "ANOVA"},
    "Scheffé": {"type": "parametric", "fn": _scheffe, "ci": True, "context": "ANOVA"},
    "Dunnett": {"type": "parametric", "fn": _dunnett, "ci": True, "context": "ANOVA"},
    "Games-Howell": {"type": "parametric", "fn": _games_howell, "ci": True, "context": "ANOVA"},
    "Newman-Keuls": {"type": "parametric", "fn": _newman_keuls, "ci": True, "context": "ANOVA"},
    # Parametric — Repeated Measures
    "Pairwise Paired t": {"type": "parametric", "fn": _pairwise_paired, "ci": True, "context": "Repeated"},
    "Paired t + Bonferroni": {"type": "parametric", "fn": _bonferroni_paired, "ci": True, "context": "Repeated"},
    "Paired t + Holm": {"type": "parametric", "fn": _holm_paired, "ci": True, "context": "Repeated"},
    # Parametric — MANOVA
    "Discriminant Comparisons": {"type": "parametric", "fn": _discriminant_comparisons, "ci": True, "context": "MANOVA"},
    "Canonical Contrasts": {"type": "parametric", "fn": _canonical_contrasts, "ci": True, "context": "MANOVA"},
    # Nonparametric — Kruskal-Wallis
    "Dunn": {"type": "nonparametric", "fn": _dunn_test, "ci": False, "context": "Kruskal-Wallis"},
    "Conover": {"type": "nonparametric", "fn": _conover_test, "ci": False, "context": "Kruskal-Wallis"},
    "DSCF": {"type": "nonparametric", "fn": _dscf, "ci": False, "context": "Kruskal-Wallis"},
    # Nonparametric — Friedman
    "Nemenyi": {"type": "nonparametric", "fn": _nemenyi_test, "ci": False, "context": "Friedman"},
    "Conover-Friedman": {"type": "nonparametric", "fn": _conover_friedman, "ci": False, "context": "Friedman"},
    "Wilcoxon + Bonferroni": {"type": "nonparametric", "fn": _wilcoxon_pairwise, "ci": False, "context": "Friedman"},
}


# =========================
# RENDER ENGINE
# =========================

def _pairwise_heatmap(pairwise_results, n_groups, title):
    labels = [f"Group {i+1}" for i in range(n_groups)]
    p_mat = np.ones((n_groups, n_groups))
    for r in pairwise_results:
        i, j = r["i"], r["j"]
        p_mat[i, j] = r["p"]
        p_mat[j, i] = r["p"]
    fig = px.imshow(
        p_mat, x=labels, y=labels, color_continuous_scale="RdBu_r",
        zmin=0, zmax=1, text_auto=".3f", aspect="equal",
        title=title,
    )
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(tickformat=".0s")
    fig.update_yaxes(tickformat=".0s")
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                fig.add_annotation(x=i, y=j, text="—", showarrow=False,
                                   font=dict(color="white", size=11))
    return fig


def _ci_plot(pairwise_results, method):
    pairs = [r["Pair"] for r in pairwise_results]
    diffs = [r["Diff"] for r in pairwise_results]
    lo = [r["CI_low"] for r in pairwise_results]
    hi = [r["CI_high"] for r in pairwise_results]
    has_ci = any(l != 0 or h != 0 for l, h in zip(lo, hi))

    fig = go.Figure()
    if has_ci:
        for idx, (p, d, l, h) in enumerate(zip(pairs, diffs, lo, hi)):
            color = "#4C78A8" if l <= 0 <= h else "#E45756"
            fig.add_trace(go.Scatter(
                x=[l, h], y=[idx, idx], mode="lines",
                line=dict(color=color, width=3),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[d], y=[idx], mode="markers",
                marker=dict(color=color, size=8),
                showlegend=False,
                hovertemplate=f"{p}<br>Diff={d:.3f}<br>CI=[{l:.3f}, {h:.3f}]<extra></extra>",
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    else:
        for idx, (p, d) in enumerate(zip(pairs, diffs)):
            color = "#4C78A8"
            fig.add_trace(go.Scatter(
                x=[d], y=[idx], mode="markers",
                marker=dict(color=color, size=8),
                showlegend=False,
                hovertemplate=f"{p}<br>Diff={d:.3f}<extra></extra>",
            ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Difference" if has_ci else "Rank Difference",
        yaxis=dict(tickmode="array", tickvals=list(range(len(pairs))), ticktext=pairs),
        showlegend=False,
    )
    return fig


def render_post_hoc(groups, param_type="parametric", key="ph"):
    n_groups = len(groups)
    if n_groups < 3:
        st.info("Post-hoc tests require at least 3 groups.")
        return

    methods = {k: v for k, v in POST_HOC_METHODS.items() if v["type"] == param_type}

    # Group methods by context
    contexts = sorted(set(v["context"] for v in methods.values()))
    if len(contexts) > 1:
        ctx = st.selectbox("Category", contexts, key=f"{key}_ctx")
        ctx_methods = {k: v for k, v in methods.items() if v["context"] == ctx}
    else:
        ctx_methods = methods

    selected = st.selectbox("Post-hoc Method", list(ctx_methods.keys()), key=f"{key}_method")
    method_info = ctx_methods[selected]
    results = method_info["fn"](groups)

    n_comparisons = len(results)
    sig_results = [r for r in results if r["p"] < 0.05]
    n_sig = len(sig_results)

    col1, col2, col3 = st.columns(3)
    col1.metric("Comparisons", str(n_comparisons))
    col2.metric("Significant (p<0.05)", str(n_sig))
    col3.metric("Method", selected)

    tab_p, tab_h, tab_ci = st.tabs(["P-Values", "Pairwise Heatmap", "Confidence Intervals"])

    with tab_p:
        df = pd.DataFrame(results)
        df_display = df[["Pair", "Diff", "p"]].copy()
        df_display["p"] = df_display["p"].apply(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")
        df_display["Diff"] = df_display["Diff"].apply(lambda x: f"{x:.3f}")
        df_display.columns = ["Pair", "Difference", "Adjusted p"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with tab_h:
        fig = _pairwise_heatmap(results, n_groups, f"{selected} — Adjusted P-Values")
        st.plotly_chart(fig, use_container_width=True)

    with tab_ci:
        fig = _ci_plot(results, selected)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Interpretation"):
        st.markdown(f"""
        - **{selected}** post-hoc test applied to {n_comparisons} pairwise comparisons
        - {n_sig} comparison{'s' if n_sig != 1 else ''} significant at α = 0.05
        - Adjusted p-values control for multiple testing
        """)
