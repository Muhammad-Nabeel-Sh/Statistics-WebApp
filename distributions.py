import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

# =========================
# HELPERS
# =========================

def _metric_row(mu, var, skew, kurt):
    c = st.columns(4)
    c[0].metric("Mean", f"{mu:.4f}")
    c[1].metric("Variance", f"{var:.4f}")
    c[2].metric("Skewness", f"{skew:.4f}")
    c[3].metric("Kurtosis", f"{kurt:.4f}")


def _pmf_plot(dist, x_max, param_str, x_min=0):
    x = np.arange(x_min, x_max + 1)
    pmf = dist.pmf(x)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=pmf, marker_color="#4C78A8", width=0.7))
    fig.update_xaxes(tickformat=".0f", dtick=1)
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="x",
        yaxis_title="P(X = x)",
        title=f"PMF {param_str}",
    )
    return fig


def _pdf_plot(x, pdf, param_str, x_label="x", shade=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
        name="PDF", hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>",
    ))
    if shade:
        lo, hi = shade
        mask = (x >= lo) & (x <= hi)
        fig.add_trace(go.Scatter(
            x=x[mask], y=pdf[mask], mode="lines",
            fill="tozeroy", fillcolor="rgba(228,87,86,0.25)",
            line=dict(width=0), name=f"P({lo} ≤ X ≤ {hi})",
            hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>",
        ))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=x_label,
        yaxis_title="f(x)",
        title=f"PDF {param_str}",
    )
    return fig


def _cdf_plot(dist, x, param_str, continuous=True):
    cdf = dist.cdf(x) if continuous else dist.cdf(np.floor(x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=cdf, mode="lines", line=dict(color="#E45756", width=2.5),
        name="CDF", hovertemplate="x=%{x:.2f}<br>F(x)=%{y:.4f}<extra></extra>",
    ))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="x",
        yaxis_title="F(x)",
        title=f"CDF {param_str}",
    )
    return fig


def _interpret_card(title, body):
    with st.expander(f"{title}", expanded=False):
        st.markdown(body)


def _display_distribution(dist_name, formula, controls_fn, plot_fn,
                          interpretation, applications, tests,
                          moments_fn=None):
    st.subheader(dist_name)
    cols = st.columns([1.2, 2])
    with cols[0]:
        st.markdown("### Formula")
        st.latex(formula)
        st.markdown("---")
        st.markdown("### Parameters")
        controls_fn()
    with cols[1]:
        fig = plot_fn()
        st.plotly_chart(fig, use_container_width=True)
        if moments_fn:
            moments_fn()
    tab_i, tab_a, tab_t = st.tabs(["Interpretation", "Applications", "Associated Tests"])
    with tab_i:
        st.markdown(interpretation)
    with tab_a:
        for app in applications:
            st.markdown(f"- {app}")
    with tab_t:
        for test in tests:
            st.markdown(f"- {test}")


# =========================
# DISCRETE DISTRIBUTIONS
# =========================

def bernoulli_widget():
    st.subheader("Bernoulli Distribution")
    p = st.slider("Success Probability (p)", 0.01, 0.99, 0.5, 0.01, key="bern_p")
    st.latex(r"P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}")
    x = np.array([0, 1])
    pmf = sp_stats.bernoulli.pmf(x, p)
    fig = go.Figure(go.Bar(x=x, y=pmf, marker_color="#4C78A8", width=0.4))
    fig.update_xaxes(tickformat=".0f", dtick=1)
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="P(X = x)")
    st.plotly_chart(fig, use_container_width=True)
    _metric_row(p, p * (1 - p), (1 - 2 * p) / np.sqrt(p * (1 - p)),
                (1 - 6 * p * (1 - p)) / (p * (1 - p)))
    _interpret_card("Interpretation", f"A single trial with success probability **p = {p:.2f}**. Models binary outcomes like coin flips, treatment success/failure.")
    _interpret_card("Applications", "- Coin flips\n- Treatment success/failure\n- A/B test outcomes\n- Binary classification")
    _interpret_card("Associated Tests", "- Binomial test\n- One-sample proportion test\n- Z-test for proportions\n- Logistic regression")


def binomial_widget():
    st.subheader("Binomial Distribution")
    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("Number of Trials (n)", 1, 100, 20, key="binom_n")
    with c2:
        p = st.slider("Success Probability (p)", 0.01, 0.99, 0.3, 0.01, key="binom_p")
    st.latex(r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n")
    dist = sp_stats.binom(n, p)
    x = np.arange(0, n + 1)
    pmf = dist.pmf(x)
    fig = go.Figure(go.Bar(x=x, y=pmf, marker_color="#4C78A8", width=0.7))
    fig.update_xaxes(tickformat=".0f", dtick=max(1, n // 10))
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="k (Number of Successes)", yaxis_title="P(X = k)")
    st.plotly_chart(fig, use_container_width=True)
    mu = n * p
    var = n * p * (1 - p)
    skew = (1 - 2 * p) / np.sqrt(var) if var > 0 else 0
    kurt = (1 - 6 * p * (1 - p)) / var if var > 0 else 0
    _metric_row(mu, var, skew, kurt)
    _interpret_card("Interpretation", f"Sum of **n = {n}** independent Bernoulli trials with success probability **p = {p:.2f}**. Mean = {mu:.2f}, Variance = {var:.2f}. {'Approaches normality as n increases.' if n > 20 else 'For small n, the distribution is skewed.' if abs(p - 0.5) > 0.2 else 'Symmetric when p ≈ 0.5.'}")
    _interpret_card("Applications", "- Number of heads in coin flips\n- Number of defective items in batch\n- Survey yes/no counts\n- Clinical trial success counts")
    _interpret_card("Associated Tests", "- Binomial test\n- One-sample proportion z-test\n- Chi-square goodness-of-fit\n- Logistic regression")


def poisson_widget():
    st.subheader("Poisson Distribution")
    lam = st.slider("Rate (λ)", 0.1, 20.0, 3.0, 0.1, key="pois_lam")
    st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots")
    dist = sp_stats.poisson(lam)
    x_max = int(lam + 5 * np.sqrt(lam)) + 5
    x = np.arange(0, x_max + 1)
    pmf = dist.pmf(x)
    fig = go.Figure(go.Bar(x=x, y=pmf, marker_color="#4C78A8", width=0.7))
    fig.update_xaxes(tickformat=".0f", dtick=max(1, x_max // 10))
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="k (Count)", yaxis_title="P(X = k)")
    st.plotly_chart(fig, use_container_width=True)
    _metric_row(lam, lam, 1 / np.sqrt(lam), 1 / lam if lam > 0 else 0)
    normal_note = ""
    if lam >= 10:
        normal_note = " The distribution approximates a Normal(λ, λ) due to the Central Limit Theorem."
    _interpret_card("Interpretation", f"Models the count of rare events over a fixed interval with rate **λ = {lam:.1f}**. Mean = Variance = {lam:.1f}.{normal_note}")
    _interpret_card("Applications", "- Number of ER arrivals per hour\n- Defects per unit area\n- Website visits per minute\n- Insurance claims per period")
    _interpret_card("Associated Tests", "- Chi-square goodness-of-fit\n- Poisson regression\n- Rare event hypothesis tests\n- CUSUM charts")


def geometric_widget():
    st.subheader("Geometric Distribution")
    p = st.slider("Success Probability (p)", 0.01, 0.99, 0.2, 0.01, key="geom_p")
    st.latex(r"P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots")
    x_max = int(np.ceil(5 / p)) + 1
    x = np.arange(1, x_max + 1)
    pmf = sp_stats.geom.pmf(x, p)
    fig = go.Figure(go.Bar(x=x, y=pmf, marker_color="#4C78A8", width=0.7))
    fig.update_xaxes(tickformat=".0f", dtick=max(1, x_max // 10))
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="k (Trial of First Success)", yaxis_title="P(X = k)")
    st.plotly_chart(fig, use_container_width=True)
    mu = 1 / p
    var = (1 - p) / (p ** 2)
    skew = (2 - p) / np.sqrt(1 - p)
    kurt = 6 + (p ** 2) / (1 - p)
    _metric_row(mu, var, skew, kurt)
    _interpret_card("Interpretation", f"Models the number of trials needed for the **first success** with success probability **p = {p:.2f}**. Mean = {mu:.1f} trials. The distribution is right-skewed — most successes occur early, but occasional long waits inflate the mean.")
    _interpret_card("Applications", "- Number of coin flips to get heads\n- Number of calls until a sale\n- Number of attempts until system failure\n- Waiting times (discrete)")
    _interpret_card("Associated Tests", "- Geometric regression\n- Survival analysis (discrete)\n- Negative binomial regression")


def negbinom_widget():
    st.subheader("Negative Binomial Distribution")
    c1, c2 = st.columns(2)
    with c1:
        r = st.slider("Number of Successes (r)", 1, 30, 5, key="nb_r")
    with c2:
        p = st.slider("Success Probability (p)", 0.01, 0.99, 0.3, 0.01, key="nb_p")
    st.latex(r"P(X = k) = \binom{k+r-1}{r-1} p^r (1-p)^k, \quad k = 0, 1, 2, \ldots")
    x_max = int(sp_stats.nbinom.ppf(0.995, r, p)) + 5
    x = np.arange(0, x_max + 1)
    pmf = sp_stats.nbinom.pmf(x, r, p)
    fig = go.Figure(go.Bar(x=x, y=pmf, marker_color="#4C78A8", width=0.7))
    fig.update_xaxes(tickformat=".0f", dtick=max(1, x_max // 10))
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="k (Failures)", yaxis_title="P(X = k)")
    st.plotly_chart(fig, use_container_width=True)
    mu = r * (1 - p) / p
    var = r * (1 - p) / (p ** 2)
    skew = (2 - p) / np.sqrt(r * (1 - p))
    kurt = 6 / r + (p ** 2) / (r * (1 - p))
    _metric_row(mu, var, skew, kurt)
    _interpret_card("Interpretation", f"Models the number of **failures** before **r = {r}** successes with success probability **p = {p:.2f}**. Mean = {mu:.1f} failures, Variance = {var:.1f}. The variance exceeds the mean, making this suitable for **overdispersed** count data.")
    _interpret_card("Applications", "- Overdispersed count data\n- Epidemiology (disease clusters)\n- Insurance claim counts\n- Ecology (species abundance)")
    _interpret_card("Associated Tests", "- Negative binomial regression\n- Quasi-Poisson models\n- Zero-inflated models")


def hypergeom_widget():
    st.subheader("Hypergeometric Distribution")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.slider("Population Size (N)", 10, 500, 50, key="hg_N")
    with c2:
        K = st.slider("Successes in Population (K)", 1, 500, 20, key="hg_K")
    with c3:
        n = st.slider("Sample Size (n)", 1, 500, 10, key="hg_n")
    K = min(K, N)
    n = min(n, N)
    st.latex(r"P(X = k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}")
    dist = sp_stats.hypergeom(N, K, n)
    x = np.arange(max(0, n - (N - K)), min(n, K) + 1)
    pmf = dist.pmf(x)
    fig = go.Figure(go.Bar(x=x, y=pmf, marker_color="#4C78A8", width=0.7))
    fig.update_xaxes(tickformat=".0f", dtick=1)
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="k (Successes in Sample)", yaxis_title="P(X = k)")
    st.plotly_chart(fig, use_container_width=True)
    mu = n * K / N
    var = n * K / N * (1 - K / N) * (N - n) / (N - 1)
    skew = ((N - 2 * K) * (N - 2 * n) / (N - 2)) * np.sqrt((N - 1) / (n * K * (N - K) * (N - n))) if var > 0 else 0
    kurt = 3 + (N - 2 * n) * (N - 2 * K) * (2 * N - n - K - 2) / (n * K * (N - K) * (N - n)) - 6 / (N - 1) if var > 0 else 0
    _metric_row(mu, var, skew, kurt)
    _interpret_card("Interpretation", f"Sampling **n = {n}** items from a population of **N = {N}** containing **K = {K}** successes, **without replacement**. Unlike the binomial distribution, trials are **dependent** — each draw changes the composition of the remaining population. The finite population correction factor {((N - n) / (N - 1)):.3f} shrinks the variance.")
    _interpret_card("Applications", "- Sampling without replacement\n- Quality control (defect counting)\n- Lottery odds\n- Survey sampling from finite populations")
    _interpret_card("Associated Tests", "- Fisher's exact test\n- Hypergeometric test (enrichment analysis)\n- Gene set enrichment analysis")


def discrete_uniform_widget():
    st.subheader("Discrete Uniform Distribution")
    a = st.slider("Minimum (a)", 0, 20, 1, key="du_a")
    b = st.slider("Maximum (b)", a + 1, 30, 6, key="du_b")
    st.latex(r"P(X = k) = \frac{1}{b - a + 1}, \quad k = a, a+1, \ldots, b")
    x = np.arange(a, b + 1)
    pmf = np.full_like(x, 1 / len(x), dtype=float)
    fig = go.Figure(go.Bar(x=x, y=pmf, marker_color="#4C78A8", width=0.7))
    fig.update_xaxes(tickformat=".0f", dtick=1)
    fig.update_yaxes(tickformat=".2f", range=[0, max(pmf) * 1.3])
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="k", yaxis_title="P(X = k)")
    st.plotly_chart(fig, use_container_width=True)
    n_vals = len(x)
    mu = (a + b) / 2
    var = ((b - a + 1) ** 2 - 1) / 12
    _metric_row(mu, var, 0, -6 * (n_vals ** 2 + 1) / (5 * (n_vals ** 2 - 1)))
    _interpret_card("Interpretation", f"All **{n_vals}** outcomes between {a} and {b} are equally likely. Each has probability **{1/n_vals:.3f}**. This is the simplest discrete distribution, representing pure randomness with no bias.")
    _interpret_card("Applications", "- Fair dice rolls\n- Random number generation\n- Lottery number selection\n- Baseline comparison distribution")
    _interpret_card("Associated Tests", "- Chi-square goodness-of-fit\n- Runs test for randomness")


# =========================
# CONTINUOUS DISTRIBUTIONS
# =========================

def normal_widget():
    st.subheader("Normal (Gaussian) Distribution")
    c1, c2 = st.columns(2)
    with c1:
        mu = st.slider("Mean (μ)", -10.0, 10.0, 0.0, 0.1, key="norm_mu")
    with c2:
        sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.05, key="norm_sigma")
    st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)")
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    pdf = sp_stats.norm.pdf(x, mu, sigma)
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    _metric_row(mu, sigma ** 2, 0, 0)
    _interpret_card("Interpretation", f"The bell curve with mean **μ = {mu:.2f}** and standard deviation **σ = {sigma:.2f}**. Symmetric about μ. ~68% of data within μ±σ, ~95% within μ±2σ, ~99.7% within μ±3σ (Empirical Rule).")
    _interpret_card("Applications", "- Biological measurements\n- Measurement error\n- Sampling distributions (CLT)\n- IQ scores, heights, blood pressure")
    _interpret_card("Associated Tests", "- Z-test\n- T-test\n- ANOVA\n- Linear regression\n- Paired tests")


def std_normal_widget():
    st.subheader("Standard Normal Distribution")
    st.latex(r"f(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}")
    st.markdown("Special case: **μ = 0**, **σ = 1**. Every Normal distribution can be **standardized**:")
    st.latex(r"Z = \frac{X - \mu}{\sigma}")
    c1, c2 = st.columns(2)
    x = np.linspace(-4, 4, 500)
    pdf = sp_stats.norm.pdf(x, 0, 1)
    with c1:
        z_val = st.slider("z-score", -4.0, 4.0, 1.96, 0.01, key="stdn_z")
    with c2:
        tail = st.selectbox("Tail", ["Two-tailed", "Left-tailed", "Right-tailed"], key="stdn_tail")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                             name="Standard Normal"))
    if tail == "Two-tailed":
        mask = (x >= -abs(z_val)) & (x <= abs(z_val))
        label = f"P(|Z| ≤ {abs(z_val):.2f})"
    elif tail == "Left-tailed":
        mask = x <= z_val
        label = f"P(Z ≤ {z_val:.2f})"
    else:
        mask = x >= z_val
        label = f"P(Z ≥ {z_val:.2f})"
    prob = sp_stats.norm.cdf(z_val) if tail == "Left-tailed" else (1 - sp_stats.norm.cdf(z_val)) if tail == "Right-tailed" else 2 * (1 - sp_stats.norm.cdf(abs(z_val)))
    fig.add_trace(go.Scatter(x=x[mask], y=pdf[mask], mode="lines",
                             fill="tozeroy", fillcolor="rgba(228,87,86,0.3)",
                             line=dict(width=0), name=label))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="z", yaxis_title="f(z)")
    st.plotly_chart(fig, use_container_width=True)
    c1.metric("Cumulative Probability", f"{sp_stats.norm.cdf(z_val):.4f}")
    c2.metric("P-value (two-tailed)", f"{prob:.4f}")
    _interpret_card("Interpretation", f"At **z = {z_val:.2f}**, P(Z ≤ z) = {sp_stats.norm.cdf(z_val):.4f}. The shaded area represents the {tail.lower()} probability = **{prob:.4f}**. z-scores measure distance from the mean in standard deviation units.")
    _interpret_card("Applications", "- Hypothesis testing (z-tests)\n- Confidence intervals\n- Standardized scores\n- Effect size interpretation")
    _interpret_card("Associated Tests", "- One-sample z-test\n- Two-sample z-test\n- Proportion z-test")


def t_widget():
    st.subheader("Student's t-Distribution")
    df = st.slider("Degrees of Freedom (ν)", 1, 100, 5, 1, key="t_df")
    st.latex(r"f(t) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\,\Gamma(\frac{\nu}{2})} \left(1+\frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}")
    x = np.linspace(-5, 5, 500)
    pdf_t = sp_stats.t.pdf(x, df)
    pdf_norm = sp_stats.norm.pdf(x, 0, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pdf_t, mode="lines", line=dict(color="#4C78A8", width=2.5),
                             name=f"t (df={df})"))
    fig.add_trace(go.Scatter(x=x, y=pdf_norm, mode="lines", line=dict(color="#E45756", width=1.5, dash="dash"),
                             name="N(0,1)"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="t", yaxis_title="f(t)")
    st.plotly_chart(fig, use_container_width=True)
    mu_t = 0 if df > 1 else float("nan")
    var_t = df / (df - 2) if df > 2 else float("inf")
    kurt_t = 6 / (df - 4) + 3 if df > 4 else float("inf")
    _metric_row(mu_t if not np.isnan(mu_t) else 0,
                var_t if var_t != float("inf") else 0,
                0,
                kurt_t if kurt_t != float("inf") else 0)
    converge_note = ""
    if df >= 30:
        converge_note = " The t-distribution closely approximates the standard normal (shown in red dashed line)."
    elif df >= 10:
        converge_note = " The t-distribution approaches the standard normal as df increases."
    else:
        converge_note = " Heavy tails at low df inflate the variance."
    _interpret_card("Interpretation", f"Heavier tails than normal, especially at low **df = {df}**. Variance = {df / (df - 2) if df > 2 else 'undefined (df ≤ 2)'}.{converge_note}")
    _interpret_card("Applications", "- Small sample inference\n- Unknown population variance\n- Linear regression coefficients\n- Confidence intervals")
    _interpret_card("Associated Tests", "- One-sample t-test\n- Independent t-test\n- Paired t-test\n- Regression t-tests")


def chi2_widget():
    st.subheader("Chi-Square (χ²) Distribution")
    df = st.slider("Degrees of Freedom (k)", 1, 50, 5, 1, key="chi2_df")
    st.latex(r"f(x; k) = \frac{x^{k/2-1} e^{-x/2}}{2^{k/2} \Gamma(k/2)}, \quad x > 0")
    x = np.linspace(0.001, sp_stats.chi2.ppf(0.999, df), 500)
    pdf = sp_stats.chi2.pdf(x, df)
    crit = sp_stats.chi2.ppf(0.95, df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                             name=f"χ² (df={df})"))
    mask = x >= crit
    fig.add_trace(go.Scatter(x=x[mask], y=pdf[mask], mode="lines",
                             fill="tozeroy", fillcolor="rgba(228,87,86,0.3)",
                             line=dict(width=0), name=f"α=0.05 (χ²={crit:.2f})"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    _metric_row(df, 2 * df, np.sqrt(8 / df), 12 / df + 3)
    _interpret_card("Interpretation", f"Sum of **k = {df}** squared independent standard normals. Mean = **{df}**, Variance = **{2 * df}**. Right-skewed (skewness = {np.sqrt(8 / df):.2f}). The critical value at α = 0.05 is **{crit:.2f}**.")
    _interpret_card("Applications", "- Variance estimation\n- Goodness-of-fit tests\n- Contingency tables\n- Model comparisons (likelihood ratio)")
    _interpret_card("Associated Tests", "- Chi-square goodness-of-fit\n- Chi-square test of independence\n- Likelihood ratio test\n- Wald test")


def f_widget():
    st.subheader("F-Distribution")
    c1, c2 = st.columns(2)
    with c1:
        d1 = st.slider("Numerator df (d₁)", 1, 50, 5, 1, key="f_d1")
    with c2:
        d2 = st.slider("Denominator df (d₂)", 1, 100, 20, 1, key="f_d2")
    st.latex(r"f(x; d_1, d_2) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x\,B\!\left(\frac{d_1}{2}, \frac{d_2}{2}\right)}")
    x = np.linspace(0.001, sp_stats.f.ppf(0.995, d1, d2), 500)
    pdf = sp_stats.f.pdf(x, d1, d2)
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="F", yaxis_title="f(F)")
    st.plotly_chart(fig, use_container_width=True)
    mu_f = d2 / (d2 - 2) if d2 > 2 else float("nan")
    var_f = 2 * d2**2 * (d1 + d2 - 2) / (d1 * (d2 - 2)**2 * (d2 - 4)) if d2 > 4 else float("nan")
    _metric_row(mu_f if not np.isnan(mu_f) else 0,
                var_f if not np.isnan(var_f) else 0,
                0, 0)
    _interpret_card("Interpretation", f"Ratio of two independent chi-square variables divided by their dfs. Mean ≈ **{mu_f:.3f}** (approaches 1 as d₂ → ∞). Right-skewed. Used to compare variances across groups.")
    _interpret_card("Applications", "- ANOVA (between/within group variance)\n- Comparing two variances (F-test)\n- Regression model comparison\n- Nested model tests")
    _interpret_card("Associated Tests", "- One-way ANOVA\n- Two-way ANOVA\n- F-test for variance equality\n- Regression F-test")


def exponential_widget():
    st.subheader("Exponential Distribution")
    lam = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.05, key="exp_lam")
    st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
    x = np.linspace(0, sp_stats.expon.ppf(0.995, scale=1/lam), 500)
    pdf = sp_stats.expon.pdf(x, scale=1/lam)
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    mu = 1 / lam
    var = 1 / (lam ** 2)
    _metric_row(mu, var, 2, 6)
    _interpret_card("Interpretation", f"Models waiting times between Poisson events with rate **λ = {lam:.2f}**. Mean = **{mu:.2f}**, Variance = **{var:.2f}**. The exponential distribution has the **memoryless property**: P(X > s+t | X > s) = P(X > t). It is the continuous analogue of the geometric distribution.")
    _interpret_card("Applications", "- Time between radioactive decays\n- Equipment failure times\n- Service/waiting times\n- Survival analysis (constant hazard)")
    _interpret_card("Associated Tests", "- Exponential regression\n- Survival analysis\n- Hazard rate models\n- Poisson process tests")


def gamma_widget():
    st.subheader("Gamma Distribution")
    c1, c2 = st.columns(2)
    with c1:
        shape = st.slider("Shape (k)", 0.5, 10.0, 2.0, 0.1, key="gam_k")
    with c2:
        scale = st.slider("Scale (θ)", 0.5, 5.0, 2.0, 0.1, key="gam_theta")
    st.latex(r"f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}, \quad x > 0")
    x = np.linspace(0.001, sp_stats.gamma.ppf(0.995, a=shape, scale=scale), 500)
    pdf = sp_stats.gamma.pdf(x, a=shape, scale=scale)
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    mu = shape * scale
    var = shape * (scale ** 2)
    sk = 2 / np.sqrt(shape)
    kurt = 6 / shape + 3
    _metric_row(mu, var, sk, kurt)
    special = ""
    if abs(shape - 1) < 0.01:
        special = " When k = 1, this reduces to the **Exponential(λ)** distribution."
    if abs(shape - sp_stats.gamma.ppf(0.5, a=shape)) < 0.1 and shape > 5:
        special = " For large k, the Gamma distribution approaches a Normal distribution."
    _interpret_card("Interpretation", f"A flexible two-parameter family with shape **k = {shape:.1f}** and scale **θ = {scale:.1f}**. Mean = **{mu:.1f}**, Variance = **{var:.1f}**. Skewness = {sk:.2f}. The Gamma generalizes the Exponential (k=1) and Chi-square (k=ν/2, θ=2) distributions.{special}")
    _interpret_card("Applications", "- Waiting times for k events\n- Insurance claim sizes\n- Rainfall amounts\n- Bayesian conjugate prior for λ")
    _interpret_card("Associated Tests", "- Gamma regression\n- Generalized linear models\n- Survival analysis\n- Bayesian analysis")


def beta_widget():
    st.subheader("Beta Distribution")
    c1, c2 = st.columns(2)
    with c1:
        alpha = st.slider("α (alpha)", 0.1, 10.0, 2.0, 0.1, key="beta_a")
    with c2:
        beta = st.slider("β (beta)", 0.1, 10.0, 2.0, 0.1, key="beta_b")
    st.latex(r"f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1")
    x = np.linspace(0.001, 0.999, 500)
    pdf = sp_stats.beta.pdf(x, alpha, beta)
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.3f}<br>f(x)=%{y:.3f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    mu = alpha / (alpha + beta)
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    sk = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1) / ((alpha + beta + 2) * np.sqrt(alpha * beta))
    kurt = 6 * ((alpha - beta) ** 2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)) / (alpha * beta * (alpha + beta + 2) * (alpha + beta + 3))
    _metric_row(mu, var, sk, kurt)
    shape_desc = ""
    if alpha < beta:
        shape_desc = "left-skewed (α < β)"
    elif alpha > beta:
        shape_desc = "right-skewed (α > β)"
    else:
        shape_desc = "symmetric (α = β)"
    uniform_note = " At α = β = 1, it is the **Uniform(0,1)** distribution." if abs(alpha - 1) < 0.1 and abs(beta - 1) < 0.1 else ""
    _interpret_card("Interpretation", f"Distribution on **[0, 1]** with shape parameters α = **{alpha:.1f}**, β = **{beta:.1f}**. The distribution is {shape_desc}. Mean = **{mu:.3f}**.{uniform_note} The Beta is the **conjugate prior** for the Binomial likelihood in Bayesian statistics.")
    _interpret_card("Applications", "- Bayesian priors (proportions)\n- Prevalence estimation\n- Random probabilities\n- A/B testing (posterior distribution)")
    _interpret_card("Associated Tests", "- Bayesian proportion tests\n- Beta-binomial model\n- Bayesian A/B testing\n- Prevalence estimation")


def uniform_cont_widget():
    st.subheader("Continuous Uniform Distribution")
    c1, c2 = st.columns(2)
    with c1:
        a = st.slider("Minimum (a)", -10.0, 10.0, 0.0, 0.5, key="unif_a")
    with c2:
        b = st.slider("Maximum (b)", a + 0.5, 10.0, 5.0, 0.5, key="unif_b")
    st.latex(r"f(x) = \frac{1}{b - a}, \quad a \leq x \leq b")
    x = np.linspace(a - (b - a) * 0.2, b + (b - a) * 0.2, 500)
    pdf = sp_stats.uniform.pdf(x, a, b - a)
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f", range=[-0.1, 1.5 / (b - a)])
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    mu = (a + b) / 2
    var = (b - a) ** 2 / 12
    _metric_row(mu, var, 0, -6 / 5)
    _interpret_card("Interpretation", f"All values between **{a:.1f}** and **{b:.1f}** are equally likely. The PDF is constant at **{1/(b-a):.4f}**. This distribution represents **maximum ignorance** — no value is more likely than any other within the range.")
    _interpret_card("Applications", "- Random number generation\n- Non-informative priors\n- Rounding error models\n- Baseline distribution")
    _interpret_card("Associated Tests", "- Kolmogorov-Smirnov test\n- Runs test\n- Goodness-of-fit tests")


def lognormal_widget():
    st.subheader("Log-Normal Distribution")
    c1, c2 = st.columns(2)
    with c1:
        mu_ln = st.slider("μ (log-scale)", -2.0, 4.0, 0.0, 0.1, key="ln_mu")
    with c2:
        sigma_ln = st.slider("σ (log-scale)", 0.1, 2.0, 0.5, 0.05, key="ln_sigma")
    st.latex(r"f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0")
    x_max = sp_stats.lognorm.ppf(0.995, s=sigma_ln, scale=np.exp(mu_ln))
    x = np.linspace(0.001, x_max, 500)
    pdf = sp_stats.lognorm.pdf(x, s=sigma_ln, scale=np.exp(mu_ln))
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    mu_x = np.exp(mu_ln + sigma_ln ** 2 / 2)
    var_x = (np.exp(sigma_ln ** 2) - 1) * np.exp(2 * mu_ln + sigma_ln ** 2)
    sk_ln = (np.exp(sigma_ln ** 2) + 2) * np.sqrt(np.exp(sigma_ln ** 2) - 1)
    kurt_ln = np.exp(4 * sigma_ln ** 2) + 2 * np.exp(3 * sigma_ln ** 2) + 3 * np.exp(2 * sigma_ln ** 2) - 3
    _metric_row(mu_x, var_x, sk_ln, kurt_ln)
    _interpret_card("Interpretation", f"If **ln(X) ~ N(μ, σ²)**, then **X** is log-normal. Positive-only, right-skewed. As **σ = {sigma_ln:.2f}** increases, the distribution becomes more skewed. Important in biology and finance where multiplicative effects dominate.")
    _interpret_card("Applications", "- Stock prices (geometric Brownian motion)\n- Biological measurements (concentrations)\n- Income distributions\n- Survival times")
    _interpret_card("Associated Tests", "- Log-transformed t-tests\n- Log-normal regression\n- Survival analysis\n- Financial modeling")


def weibull_widget():
    st.subheader("Weibull Distribution")
    c1, c2 = st.columns(2)
    with c1:
        k_w = st.slider("Shape (k)", 0.3, 5.0, 1.5, 0.05, key="wei_k")
    with c2:
        lam_w = st.slider("Scale (λ)", 0.5, 5.0, 1.0, 0.1, key="wei_lam")
    st.latex(r"f(x; k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}, \quad x \geq 0")
    x = np.linspace(0.001, sp_stats.weibull_min.ppf(0.995, c=k_w, scale=lam_w), 500)
    pdf = sp_stats.weibull_min.pdf(x, c=k_w, scale=lam_w)
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    from scipy.special import gammaln
    mu_w = lam_w * np.exp(gammaln(1 + 1 / k_w))
    var_w = lam_w ** 2 * (np.exp(gammaln(1 + 2 / k_w)) - np.exp(gammaln(1 + 1 / k_w)) ** 2)
    _metric_row(mu_w, var_w, 0, 0)
    shape_desc = ""
    if k_w < 1:
        shape_desc = "**decreasing** hazard (infant mortality / early failures)"
    elif abs(k_w - 1) < 0.05:
        shape_desc = "**constant** hazard (equivalent to Exponential distribution)"
    elif k_w > 1:
        shape_desc = "**increasing** hazard (wear-out / aging effects)"
    _interpret_card("Interpretation", f"Shape parameter **k = {k_w:.2f}** produces a {shape_desc}. The Weibull is the only distribution that can model all three hazard regimes. Originally developed for material strength testing, now widely used in survival analysis and reliability engineering.")
    _interpret_card("Applications", "- Survival analysis / time-to-event\n- Reliability engineering\n- Wind speed modeling\n- Failure time analysis")
    _interpret_card("Associated Tests", "- Cox proportional hazards\n- Parametric survival models\n- Weibull regression\n- Accelerated failure time models")


def cauchy_widget():
    st.subheader("Cauchy Distribution")
    c1, c2 = st.columns(2)
    with c1:
        x0 = st.slider("Location (x₀)", -5.0, 5.0, 0.0, 0.1, key="cau_x0")
    with c2:
        gamma = st.slider("Scale (γ)", 0.1, 5.0, 1.0, 0.05, key="cau_g")
    st.latex(r"f(x; x_0, \gamma) = \frac{1}{\pi\gamma\left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]}")
    x = np.linspace(x0 - 10 * gamma, x0 + 10 * gamma, 1000)
    pdf = sp_stats.cauchy.pdf(x, loc=x0, scale=gamma)
    pdf_norm = sp_stats.norm.pdf(x, loc=x0, scale=gamma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                             name="Cauchy"))
    fig.add_trace(go.Scatter(x=x, y=pdf_norm, mode="lines", line=dict(color="#E45756", width=1.5, dash="dash"),
                             name="Normal (same scale)"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    _metric_row(0, 0, 0, 0)
    st.caption("Mean, Variance, Skewness, and Kurtosis are **undefined** for the Cauchy distribution.")
    _interpret_card("Interpretation", "The Cauchy distribution has **no defined mean or variance** — its tails are so heavy that the integrals diverge. It is the ratio of two independent standard normals (Z₁/Z₂). The Normal comparison (red dashed) shows how much heavier the Cauchy tails are. A classic pathological example in probability theory.")
    _interpret_card("Applications", "- Physics (resonance behavior)\n- Ratio distributions\n- Robustness demonstrations\n- Heavy-tail modeling")
    _interpret_card("Associated Tests", "- Cauchy regression\n- Ratio of normals tests")


def logistic_widget():
    st.subheader("Logistic Distribution")
    c1, c2 = st.columns(2)
    with c1:
        mu_l = st.slider("Location (μ)", -10.0, 10.0, 0.0, 0.1, key="log_mu")
    with c2:
        s_l = st.slider("Scale (s)", 0.2, 5.0, 1.0, 0.05, key="log_s")
    st.latex(r"f(x; \mu, s) = \frac{e^{-(x-\mu)/s}}{s\left(1 + e^{-(x-\mu)/s}\right)^2}")
    x = np.linspace(mu_l - 6 * s_l, mu_l + 6 * s_l, 500)
    pdf = sp_stats.logistic.pdf(x, loc=mu_l, scale=s_l)
    fig = go.Figure(go.Scatter(x=x, y=pdf, mode="lines", line=dict(color="#4C78A8", width=2.5),
                               hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    var_l = (s_l ** 2) * (np.pi ** 2) / 3
    _metric_row(mu_l, var_l, 0, 6 / 5)
    _interpret_card("Interpretation", f"Similar to the Normal distribution but with **heavier tails** (kurtosis = 1.2 vs 0 for Normal). Mean = **{mu_l:.2f}**, Variance ≈ **{var_l:.2f}**. The logistic CDF is the **logit function**, which forms the basis of logistic regression.")
    _interpret_card("Applications", "- Logistic regression foundation\n- Population growth models\n- Item response theory\n- Chess rating (Elo) systems")
    _interpret_card("Associated Tests", "- Logistic regression\n- Logit models\n- Binary classification\n- Proportional odds models")


# =========================
# EDUCATIONAL FEATURES
# =========================

def clt_simulator():
    st.subheader("Central Limit Theorem Simulator")
    st.markdown("""
    The CLT states that the **sampling distribution of the mean** approaches a Normal distribution
    as the sample size increases, **regardless of the source distribution's shape**.
    """)
    c1, c2, c3 = st.columns(3)
    with c1:
        source = st.selectbox("Source Distribution", [
            "Uniform", "Exponential", "Bernoulli", "Chi-Square", "Lognormal"
        ], key="clt_source")
    with c2:
        n_samples = st.slider("Samples per Mean (n)", 2, 100, 5, 1, key="clt_n")
    with c3:
        n_reps = st.slider("Number of Repetitions", 100, 5000, 1000, 100, key="clt_reps")

    np.random.seed(42)
    dist_map = {
        "Uniform": lambda s: np.random.uniform(0, 1, s),
        "Exponential": lambda s: np.random.exponential(1, s),
        "Bernoulli": lambda s: np.random.binomial(1, 0.3, s),
        "Chi-Square": lambda s: np.random.chisquare(2, s),
        "Lognormal": lambda s: np.random.lognormal(0, 1, s),
    }
    means = np.array([dist_map[source](n_samples).mean() for _ in range(n_reps)])
    x_fit = np.linspace(means.min(), means.max(), 200)
    mu_sim = means.mean()
    sigma_sim = means.std(ddof=0)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Histogram of {n_reps} Sample Means", "Source Distribution (PDF)"))
    fig.add_trace(go.Histogram(x=means, nbinsx=40, marker_color="#4C78A8", opacity=0.7,
                               name="Sample Means", histnorm="probability density"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_fit, y=sp_stats.norm.pdf(x_fit, mu_sim, sigma_sim),
                             mode="lines", line=dict(color="#E45756", width=2.5),
                             name="Normal Fit"), row=1, col=1)

    x_src = np.linspace(0.001, 5, 500)
    if source == "Uniform":
        pdf_src = sp_stats.uniform.pdf(x_src, 0, 1)
    elif source == "Exponential":
        pdf_src = sp_stats.expon.pdf(x_src)
    elif source == "Bernoulli":
        x_src_d = np.array([0, 1])
        pdf_src_d = sp_stats.bernoulli.pmf(0.3, 0.3)
        x_src = x_src_d
        pdf_src = pdf_src_d
    elif source == "Chi-Square":
        pdf_src = sp_stats.chi2.pdf(x_src, 2)
    else:
        pdf_src = sp_stats.lognorm.pdf(x_src, s=1)
    fig.add_trace(go.Scatter(x=x_src, y=pdf_src, mode="lines",
                             line=dict(color="#54A24B", width=2),
                             name=f"{source} PDF"), row=1, col=2)
    if source == "Bernoulli":
        fig.update_xaxes(row=1, col=2, tickformat=".0f", dtick=1)

    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(title_text="Sample Mean", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean of Means", f"{mu_sim:.4f}")
    c2.metric("SD of Means (SE)", f"{sigma_sim:.4f}")
    c3.metric("Theoretical SE (σ/√n)", f"{np.sqrt(sp_stats.describe(dist_map[source](10000))[3]) / np.sqrt(n_samples):.4f}" if source != "Bernoulli" else "N/A")

    normal_note = ""
    if n_samples >= 30:
        normal_note = " The sampling distribution is nearly Normal even for n=30 (the traditional threshold)."
    elif n_samples >= 10:
        normal_note = " The CLT is already working — the sampling distribution is becoming bell-shaped."
    else:
        normal_note = " With small n, the sampling distribution still resembles the source distribution."
    _interpret_card("Interpretation", f"Sampling from a **{source}** distribution with **n = {n_samples}** per sample, {n_reps} repetitions. The histogram of sample means {normal_note}")


def overlay_comparison():
    st.subheader("Distribution Overlay Comparison")
    st.markdown("Compare two distributions side by side to understand how they differ.")
    c1, c2 = st.columns(2)
    dists = ["Normal", "t", "Chi-Square", "F", "Exponential", "Gamma", "Beta", "Uniform", "Logistic", "Cauchy"]
    with c1:
        d1_name = st.selectbox("Distribution 1", dists, index=0, key="ov_d1")
    with c2:
        d2_name = st.selectbox("Distribution 2", dists, index=1, key="ov_d2")

    param_fns = {
        "Normal": lambda: (st.slider("Mean (μ)", -5.0, 5.0, 0.0, 1.0, key="ov_n_mu"), st.slider("SD (σ)", 0.1, 5.0, 1.0, 0.1, key="ov_n_sd")),
        "t": lambda: (st.slider("DF", 1, 50, 5, 1, key="ov_t_df"),),
        "Chi-Square": lambda: (st.slider("DF", 1, 30, 5, 1, key="ov_c2_df"),),
        "F": lambda: (st.slider("d₁", 1, 30, 5, 1, key="ov_f_d1"), st.slider("d₂", 1, 50, 20, 1, key="ov_f_d2")),
        "Exponential": lambda: (st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1, key="ov_e_lam"),),
        "Gamma": lambda: (st.slider("Shape (k)", 0.5, 10.0, 2.0, 0.1, key="ov_g_k"), st.slider("Scale (θ)", 0.5, 5.0, 2.0, 0.1, key="ov_g_t")),
        "Beta": lambda: (st.slider("α", 0.1, 10.0, 2.0, 0.1, key="ov_b_a"), st.slider("β", 0.1, 10.0, 2.0, 0.1, key="ov_b_b")),
        "Uniform": lambda: (st.slider("Min", -5.0, 5.0, 0.0, 1.0, key="ov_u_a"), st.slider("Max", -4.0, 10.0, 5.0, 1.0, key="ov_u_b")),
        "Logistic": lambda: (st.slider("μ", -5.0, 5.0, 0.0, 1.0, key="ov_l_mu"), st.slider("s", 0.2, 5.0, 1.0, 0.1, key="ov_l_s")),
        "Cauchy": lambda: (st.slider("x₀", -5.0, 5.0, 0.0, 1.0, key="ov_c_x0"), st.slider("γ", 0.1, 5.0, 1.0, 0.1, key="ov_c_g")),
    }
    cols1, cols2 = st.columns(2)
    with cols1:
        st.markdown(f"**{d1_name}** parameters:")
        p1 = param_fns[d1_name]()
    with cols2:
        st.markdown(f"**{d2_name}** parameters:")
        p2 = param_fns[d2_name]()

    def make_pdf(name, params, x):
        mapping = {
            "Normal": lambda: sp_stats.norm.pdf(x, params[0], params[1]),
            "t": lambda: sp_stats.t.pdf(x, params[0]),
            "Chi-Square": lambda: sp_stats.chi2.pdf(x, params[0]),
            "F": lambda: sp_stats.f.pdf(x, params[0], params[1]),
            "Exponential": lambda: sp_stats.expon.pdf(x, scale=1/params[0]),
            "Gamma": lambda: sp_stats.gamma.pdf(x, a=params[0], scale=params[1]),
            "Beta": lambda: sp_stats.beta.pdf(x, params[0], params[1]),
            "Uniform": lambda: sp_stats.uniform.pdf(x, params[0], params[1] - params[0]),
            "Logistic": lambda: sp_stats.logistic.pdf(x, loc=params[0], scale=params[1]),
            "Cauchy": lambda: sp_stats.cauchy.pdf(x, loc=params[0], scale=params[1]),
        }
        return mapping[name]()

    x = np.linspace(-8, 8, 1000)
    pdf1 = make_pdf(d1_name, p1 if isinstance(p1, tuple) else (p1,), x)
    pdf2 = make_pdf(d2_name, p2 if isinstance(p2, tuple) else (p2,), x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pdf1, mode="lines", line=dict(color="#4C78A8", width=2.5), name=d1_name))
    fig.add_trace(go.Scatter(x=x, y=pdf2, mode="lines", line=dict(color="#E45756", width=2.5), name=d2_name))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="f(x)")
    st.plotly_chart(fig, use_container_width=True)
    _interpret_card("Interpretation", f"Comparing **{d1_name}** vs **{d2_name}**. Different shapes, tail behaviors, and parameterizations become visible through direct overlay.")


def sampling_simulator():
    st.subheader("Sampling Simulation")
    st.markdown("Draw random samples from a distribution and observe how the empirical histogram approximates the theoretical PDF.")
    c1, c2, c3 = st.columns(3)
    with c1:
        dist_name = st.selectbox("Distribution", [
            "Normal", "Exponential", "Gamma", "Beta", "Uniform", "Lognormal"
        ], key="ss_dist")
    with c2:
        n_draws = st.slider("Number of Samples", 50, 5000, 500, 50, key="ss_n")
    with c3:
        bins = st.slider("Histogram Bins", 10, 100, 40, 5, key="ss_bins")

    np.random.seed(42)
    sample_params = {
        "Normal": lambda: np.random.normal(0, 1, n_draws),
        "Exponential": lambda: np.random.exponential(1, n_draws),
        "Gamma": lambda: np.random.gamma(2, 2, n_draws),
        "Beta": lambda: np.random.beta(2, 5, n_draws),
        "Uniform": lambda: np.random.uniform(0, 1, n_draws),
        "Lognormal": lambda: np.random.lognormal(0, 0.5, n_draws),
    }
    sample = sample_params[dist_name]()
    x = np.linspace(max(0.001, sample.min()), sample.max(), 500)
    pdf_map = {
        "Normal": sp_stats.norm.pdf(x, 0, 1),
        "Exponential": sp_stats.expon.pdf(x),
        "Gamma": sp_stats.gamma.pdf(x, a=2, scale=2),
        "Beta": sp_stats.beta.pdf(x, 2, 5),
        "Uniform": sp_stats.uniform.pdf(x, 0, 1),
        "Lognormal": sp_stats.lognorm.pdf(x, s=0.5),
    }
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sample, nbinsx=bins, histnorm="probability density",
                               marker_color="#4C78A8", opacity=0.6, name="Sample"))
    fig.add_trace(go.Scatter(x=x, y=pdf_map[dist_name], mode="lines",
                             line=dict(color="#E45756", width=2.5), name="Theoretical PDF"))
    fig.update_xaxes(tickformat=".2f")
    fig.update_yaxes(tickformat=".2f")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="x", yaxis_title="Density")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample Mean", f"{sample.mean():.4f}")
    c2.metric("Sample Variance", f"{sample.var():.4f}")
    c3.metric("Sample Size", str(n_draws))
    _interpret_card("Interpretation", f"As the number of samples increases, the histogram converges to the theoretical PDF. With **n = {n_draws}**, the empirical distribution {'closely matches' if n_draws > 1000 else 'approximates'} the true density.")


# =========================
# REGISTRY
# =========================

DISTRIBUTION_REGISTRY = {
    "Bernoulli Distribution": {"type": "Discrete", "widget": bernoulli_widget},
    "Binomial Distribution": {"type": "Discrete", "widget": binomial_widget},
    "Poisson Distribution": {"type": "Discrete", "widget": poisson_widget},
    "Geometric Distribution": {"type": "Discrete", "widget": geometric_widget},
    "Negative Binomial Distribution": {"type": "Discrete", "widget": negbinom_widget},
    "Hypergeometric Distribution": {"type": "Discrete", "widget": hypergeom_widget},
    "Discrete Uniform Distribution": {"type": "Discrete", "widget": discrete_uniform_widget},
    "Normal Distribution": {"type": "Continuous", "widget": normal_widget},
    "Standard Normal Distribution": {"type": "Continuous", "widget": std_normal_widget},
    "Student's t-Distribution": {"type": "Continuous", "widget": t_widget},
    "Chi-Square Distribution": {"type": "Continuous", "widget": chi2_widget},
    "F-Distribution": {"type": "Continuous", "widget": f_widget},
    "Exponential Distribution": {"type": "Continuous", "widget": exponential_widget},
    "Gamma Distribution": {"type": "Continuous", "widget": gamma_widget},
    "Beta Distribution": {"type": "Continuous", "widget": beta_widget},
    "Continuous Uniform Distribution": {"type": "Continuous", "widget": uniform_cont_widget},
    "Log-Normal Distribution": {"type": "Continuous", "widget": lognormal_widget},
    "Weibull Distribution": {"type": "Continuous", "widget": weibull_widget},
    "Cauchy Distribution": {"type": "Continuous", "widget": cauchy_widget},
    "Logistic Distribution": {"type": "Continuous", "widget": logistic_widget},
}

EDUCATIONAL_FEATURES = {
    "CLT Simulator": {"widget": clt_simulator},
    "Distribution Overlay": {"widget": overlay_comparison},
    "Sampling Simulator": {"widget": sampling_simulator},
}


# =========================
# RENDER ENGINE
# =========================

def render_distributions():
    st.title(" Probability Distributions Explorer")
    st.markdown("""
    Probability distributions are the mathematical foundation of all statistical inference.
    Explore how parameters change the shape, spread, and behavior of each distribution.
    """)

    tab_discrete, tab_continuous, tab_features = st.tabs([
        "Discrete Distributions",
        "Continuous Distributions",
        "Educational Features",
    ])

    with tab_discrete:
        st.markdown("## Discrete Distributions (PMFs)")
        st.markdown("These distributions model **counts** and **event frequencies** — the probability of each specific outcome is non-zero.")
        discrete_dists = {k: v for k, v in DISTRIBUTION_REGISTRY.items() if v["type"] == "Discrete"}
        for i, (name, info) in enumerate(discrete_dists.items()):
            with st.expander(name, expanded=False):
                info["widget"]()

    with tab_continuous:
        st.markdown("## Continuous Distributions (PDFs)")
        st.markdown("These distributions model **measurements** and **continuous quantities** — probability is defined over intervals.")
        continuous_dists = {k: v for k, v in DISTRIBUTION_REGISTRY.items() if v["type"] == "Continuous"}
        for i, (name, info) in enumerate(continuous_dists.items()):
            with st.expander(name, expanded=False):
                info["widget"]()

    with tab_features:
        st.markdown("## Educational Features")
        st.markdown("Interactive tools that build intuition about sampling, the Central Limit Theorem, and distribution relationships.")
        for name, info in EDUCATIONAL_FEATURES.items():
            with st.expander(name, expanded=False):
                info["widget"]()
