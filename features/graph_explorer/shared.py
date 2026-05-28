import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import scipy.optimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =========================
# DATA GENERATION HELPERS
# =========================

_rng = np.random.default_rng(42)
def _gen_corr(n, r, noise=0.2, heteroscedastic=False, outlier=False):
    x = np.random.normal(0, 1, n)
    y = r * x + np.sqrt(1 - r**2) * np.random.normal(0, 1, n)
    if heteroscedastic:
        y = y * (1 + 0.5 * np.abs(x))
    if outlier:
        x[-1] = 4
        y[-1] = -4 if r > 0 else 4
    return x, y



def _gen_reg(n, beta=1.0, noise=1.0):
    x = np.random.uniform(0, 10, n)
    y = beta * x + np.random.normal(0, noise, n)
    return x, y


def _swarm_positions(values, size=0.3):
    n = len(values)
    idx = np.argsort(values)
    sv = values[idx]
    pos = np.zeros((n, 2))
    placed = []
    for i, v in enumerate(sv):
        for r in range(50):
            for s in [1, -1] if r > 0 else [1]:
                x = s * r * size * 0.3
                ok = True
                for px, py in placed:
                    if abs(v - py) < size * 0.5 and abs(x - px) < size * 0.4:
                        ok = False
                        break
                if ok:
                    pos[i] = [x, v]
                    placed.append((x, v))
                    break
    return pos[np.argsort(idx)]


def _gen_surv_data(n, hr, cens_frac):
    np.random.seed(42)
    t_ctrl = np.random.exponential(12, n)
    t_trt = np.random.exponential(12 / hr, n)
    times = np.concatenate([t_ctrl, t_trt])
    groups = np.array([0] * n + [1] * n)
    cens = np.random.uniform(0, 25, 2 * n)
    obs = np.minimum(times, cens)
    event = (times <= cens).astype(int)
    return obs, event, groups


def _km(t, e):
    df = pd.DataFrame({"t": t, "e": e}).sort_values("t")
    ts = sorted(df["t"].unique())
    s = 1.0
    ot, os = [0], [1.0]
    for ti in ts:
        nr = (df["t"] >= ti).sum()
        ne = df.loc[df["t"] == ti, "e"].sum()
        if nr > 0:
            s *= 1 - ne / nr
        ot.extend([ti, ti])
        os.extend([os[-1], s])
    return np.array(ot), np.array(os)


def _na(t, e):
    df = pd.DataFrame({"t": t, "e": e}).sort_values("t")
    h = 0.0
    ot, oh = [0], [0.0]
    for ti in sorted(df["t"].unique()):
        nr = (df["t"] >= ti).sum()
        ne = df.loc[df["t"] == ti, "e"].sum()
        if nr > 0:
            h += ne / nr
        ot.extend([ti, ti])
        oh.extend([oh[-1], h])
    return np.array(ot), np.array(oh)


def _gen_meta_data(k, eff, het):
    np.random.seed(42)
    se = np.random.uniform(0.05, 0.5, k)
    y = np.random.normal(eff, np.sqrt(se**2 + het**2))
    return y, se


def _gen_meta_bias(k, eff, het, bias):
    np.random.seed(42)
    se = np.random.uniform(0.05, 0.5, k)
    y = np.random.normal(eff, np.sqrt(se**2 + het**2))
    if bias > 0:
        for i in range(k):
            if se[i] > 0.15:
                p = 2 * (1 - stats.norm.cdf(abs(y[i] / se[i])))
                if p > 0.05 and np.random.random() < bias:
                    y[i] = np.nan
                    se[i] = np.nan
    m = ~np.isnan(y)
    return y[m], se[m]


def _gen_ph_data(n_groups, n_per_group, effect_size_label):
    _rng_ph = np.random.default_rng(137)
    effect_map = {"None": 0.0, "Small": 0.2, "Medium": 0.5, "Large": 0.8}
    em = effect_map[effect_size_label]
    bm = np.array([0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0, 2.5][:n_groups]) * em
    bm -= bm.mean()
    gd = []
    for i in range(n_groups):
        gd.append(bm[i] + _rng_ph.standard_normal(n_per_group))
    gm = [g.mean() for g in gd]
    pairs = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            x, y = gd[i], gd[j]
            m1, m2 = gm[i], gm[j]
            s1, s2 = x.std(ddof=1), y.std(ddof=1)
            sp = math.sqrt(((n_per_group - 1) * s1**2 + (n_per_group - 1) * s2**2) / (2 * n_per_group - 2))
            se = sp * math.sqrt(2 / n_per_group)
            md = m1 - m2
            d = md / sp
            t_stat = md / se
            p_val = 2 * stats.t.sf(abs(t_stat), 2 * n_per_group - 2)
            ci_lo = md - 1.96 * se
            ci_hi = md + 1.96 * se
            pairs.append({
                "i": i, "j": j,
                "pair": f"G{i+1} vs G{j+1}",
                "pair_long": f"Group {i+1} vs Group {j+1}",
                "mean_i": m1, "mean_j": m2,
                "md": md, "se": se, "d": d,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "t": t_stat, "p": p_val,
            })
    m = len(pairs)
    for idx in range(m):
        pairs[idx]["p_bonf"] = min(pairs[idx]["p"] * m, 1.0)
    sidx = np.argsort([p["p"] for p in pairs])
    for rk, idx in enumerate(sidx):
        pairs[idx]["p_holm"] = min(pairs[idx]["p"] * (m - rk), 1.0)
    pmat = np.eye(n_groups)
    dmat = np.eye(n_groups)
    for p in pairs:
        pmat[p["i"], p["j"]] = pmat[p["j"], p["i"]] = p["p"]
        dmat[p["i"], p["j"]] = dmat[p["j"], p["i"]] = p["d"]
    return {
        "gd": gd, "gm": gm, "pairs": pairs, "pmat": pmat, "dmat": dmat,
        "ng": n_groups, "np": n_per_group,
        "labels": [f"Group {i+1}" for i in range(n_groups)],
    }


def _apa_table_ge(df, title):
    st.markdown(f"**{title}**")
    st.dataframe(df, use_container_width=True)


def _assign_cld(means, pmat, alpha):
    n = len(means)
    order = np.argsort(-np.array(means))
    non_sig = pmat > alpha
    letters = ["" for _ in range(n)]
    next_l = 97
    for idx in order:
        friends = [j for j in range(n) if non_sig[idx, j] or idx == j]
        fl = set()
        for j in friends:
            for ch in letters[j]:
                fl.add(ch)
        assigned = False
        for ch in sorted(fl):
            ok = True
            for j in range(n):
                if ch in letters[j] and not non_sig[idx, j]:
                    ok = False
                    break
            if ok:
                letters[idx] += ch
                assigned = True
        if not assigned:
            letters[idx] = chr(next_l)
            next_l += 1
    return letters


def _circle_overlap_area(d, r1, r2):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    return (
        r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        + r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        - 0.5
        * math.sqrt(
            max(0, (r1 + r2 + d) * (r1 + r2 - d) * (r1 - r2 + d) * (-r1 + r2 + d))
        )
    )


def _ideal_sep(r1, r2, overlap, s1, s2):
    if overlap <= 0:
        return r1 + r2 + 0.5
    max_possible = math.pi * min(r1, r2) ** 2
    ratio = overlap / min(s1, s2) if min(s1, s2) > 0 else 0
    target = min(ratio, 0.99) * max_possible
    lo, hi = abs(r1 - r2) * 1.001, (r1 + r2) * 0.999
    try:
        return scipy.optimize.brentq(
            lambda dd: _circle_overlap_area(dd, r1, r2) - target, lo, hi
        )
    except (ValueError, RuntimeError):
        return (r1 + r2) * max(0.1, 1 - ratio * 0.6)


def _region_pos(cx, cy, r, dx, dy):
    mag = math.hypot(dx, dy)
    if mag == 0:
        return cx, cy
    return cx + dx / mag * r * 0.55, cy + dy / mag * r * 0.55


