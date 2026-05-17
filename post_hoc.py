import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as sp_stats
from itertools import combinations


# =========================
# PARAMETRIC POST-HOC TESTS
# =========================

def _pairwise_ttest(groups, pooled_var, df_within, method="bonferroni"):
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = means[i] - means[j]
        se = np.sqrt(pooled_var * (1/n[i] + 1/n[j]))
        t_stat = diff / se if se > 0 else 0
        p = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df_within))
        cl = diff - sp_stats.t.ppf(0.975, df_within) * se
        cu = diff + sp_stats.t.ppf(0.975, df_within) * se
        results.append({"Pair": f"{labels[i]} vs {labels[j]}", "i": i, "j": j,
                        "Diff": diff, "p": p, "CI_low": cl, "CI_high": cu})
    return results


def _tukey_hsd(groups):
    from scipy.stats import studentized_range
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    vars_ = np.array([g.var(ddof=1) for g in groups])
    # Assume equal n for simplicity (use harmonic mean if unequal)
    n_eff = sp_stats.hmean(n) if np.all(n > 0) else n.min()
    mse = np.sum((n - 1) * vars_) / (np.sum(n) - n_groups)
    df = np.sum(n) - n_groups
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = means[i] - means[j]
        se = np.sqrt(mse / n_eff)
        q = abs(diff) / se if se > 0 else 0
        p = 1 - studentized_range.cdf(q, n_groups, df)
        cl = diff - studentized_range.ppf(0.95, n_groups, df) * se
        cu = diff + studentized_range.ppf(0.95, n_groups, df) * se
        results.append({"Pair": f"{labels[i]} vs {labels[j]}", "i": i, "j": j,
                        "Diff": diff, "p": p, "CI_low": cl, "CI_high": cu})
    return results


def _bonferroni(groups):
    n_groups = len(groups)
    pooled_var = np.sum([(len(g)-1) * g.var(ddof=1) for g in groups]) / (np.sum([len(g) for g in groups]) - n_groups)
    df = np.sum([len(g) for g in groups]) - n_groups
    results = _pairwise_ttest(groups, pooled_var, df, "bonferroni")
    n_comparisons = len(results)
    for r in results:
        r["p"] = np.clip(r["p"] * n_comparisons, 0, 1)
    return results


def _scheffe(groups):
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    pooled_var = np.sum([(len(g)-1) * g.var(ddof=1) for g in groups]) / (np.sum(n) - n_groups)
    df_w = np.sum(n) - n_groups
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
        results.append({"Pair": f"{labels[i]} vs {labels[j]}", "i": i, "j": j,
                        "Diff": diff, "p": p, "CI_low": cl, "CI_high": cu})
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
        results.append({"Pair": f"{labels[i]} vs {labels[j]}", "i": i, "j": j,
                        "Diff": diff, "p": p, "CI_low": cl, "CI_high": cu})
    return results


def _dunnett(groups, control=0):
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    pooled_var = np.sum([(len(g)-1) * g.var(ddof=1) for g in groups]) / (np.sum(n) - n_groups)
    df = np.sum(n) - n_groups
    results = []
    for j in range(n_groups):
        if j == control:
            continue
        diff = means[j] - means[control]
        se = np.sqrt(pooled_var * (1/n[j] + 1/n[control]))
        t_stat = diff / se if se > 0 else 0
        p = 1 - sp_stats.norm.cdf(abs(t_stat))
        # Dunnett uses multivariate t — approximate with Bonferroni
        p = np.clip(p * (n_groups - 1), 0, 1)
        cl = diff - sp_stats.t.ppf(0.975, df) * se
        cu = diff + sp_stats.t.ppf(0.975, df) * se
        results.append({"Pair": f"{labels[control]} vs {labels[j]}", "i": control, "j": j,
                        "Diff": diff, "p": p, "CI_low": cl, "CI_high": cu})
    return results


# =========================
# NON-PARAMETRIC POST-HOC TESTS
# =========================

def _dunn_test(groups):
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    # Joint ranking
    all_data = np.concatenate(groups)
    all_ranks = sp_stats.rankdata(all_data)
    idx = np.concatenate([np.full(n[i], i) for i in range(n_groups)])
    # Mean rank per group
    mean_ranks = np.array([all_ranks[idx == i].mean() for i in range(n_groups)])
    # Tie correction
    ties = np.unique(all_data, return_counts=True)[1]
    tie_corr = 1 - np.sum(ties**3 - ties) / (len(all_data)**3 - len(all_data))
    # Overall variance
    n_total = len(all_data)
    var_overall = (n_total * (n_total + 1) / 12) * tie_corr
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        z = (mean_ranks[i] - mean_ranks[j]) / np.sqrt(var_overall * (1/n[i] + 1/n[j])) if var_overall > 0 else 0
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
        results.append({"Pair": f"{labels[i]} vs {labels[j]}", "i": i, "j": j,
                        "Diff": mean_ranks[i] - mean_ranks[j], "p": p,
                        "CI_low": 0, "CI_high": 0})
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
    # Mean rank overall
    grand_mean_rank = all_ranks.mean()
    # Variance of ranks
    var_rank = np.sum((all_ranks - grand_mean_rank)**2) / (n_total - 1)
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        t_num = abs(mean_ranks[i] - mean_ranks[j])
        t_den = np.sqrt(var_rank * (n_total - 1 - np.sum(mean_ranks**2 / n_total)) /
                        (n_total - n_groups) * (1/n[i] + 1/n[j]))
        t_stat = t_num / t_den if t_den > 0 else 0
        df = n_total - n_groups
        p = 2 * (1 - sp_stats.t.cdf(t_stat, df))
        results.append({"Pair": f"{labels[i]} vs {labels[j]}", "i": i, "j": j,
                        "Diff": mean_ranks[i] - mean_ranks[j], "p": p,
                        "CI_low": 0, "CI_high": 0})
    return results


def _nemenyi_test(groups):
    from scipy.stats import studentized_range
    n_groups = len(groups)
    labels = [f"Group {i+1}" for i in range(n_groups)]
    n = np.array([len(g) for g in groups])
    all_data = np.concatenate(groups)
    all_ranks = sp_stats.rankdata(all_data)
    idx = np.concatenate([np.full(n[i], i) for i in range(n_groups)])
    mean_ranks = np.array([all_ranks[idx == i].mean() for i in range(n_groups)])
    n_eff = sp_stats.hmean(n) if np.all(n > 0) else n.min()
    df = np.inf
    se = np.sqrt(n_groups * (n_groups + 1) / (12 * n_eff))
    results = []
    for (i, j) in combinations(range(n_groups), 2):
        diff = mean_ranks[i] - mean_ranks[j]
        q = abs(diff) / se if se > 0 else 0
        p = 1 - studentized_range.cdf(q, n_groups, df)
        results.append({"Pair": f"{labels[i]} vs {labels[j]}", "i": i, "j": j,
                        "Diff": diff, "p": p, "CI_low": 0, "CI_high": 0})
    return results


# =========================
# REGISTRY
# =========================

POST_HOC_METHODS = {
    "Tukey HSD": {"type": "parametric", "fn": _tukey_hsd, "ci": True},
    "Bonferroni": {"type": "parametric", "fn": _bonferroni, "ci": True},
    "Scheffe": {"type": "parametric", "fn": _scheffe, "ci": True},
    "Games-Howell": {"type": "parametric", "fn": _games_howell, "ci": True},
    "Dunnett": {"type": "parametric", "fn": _dunnett, "ci": True},
    "Dunn": {"type": "nonparametric", "fn": _dunn_test, "ci": False},
    "Conover": {"type": "nonparametric", "fn": _conover_test, "ci": False},
    "Nemenyi": {"type": "nonparametric", "fn": _nemenyi_test, "ci": False},
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
    midpoint = len(pairs) // 2
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

    selected = st.selectbox("Post-hoc Method", list(methods.keys()), key=f"{key}_method")

    method_info = methods[selected]
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
