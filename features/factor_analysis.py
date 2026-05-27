import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as sp_stats
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from core.utils import _apa_table

_rng = np.random.default_rng(42)


# ── HELPERS ──


def _varimax(loadings, gamma=1.0, max_iter=500, tol=1e-6):
    p, k = loadings.shape
    rotation = np.eye(k)
    d = 0
    for i in range(max_iter):
        loadings_rot = loadings @ rotation
        u, s, vh = np.linalg.svd(
            loadings.T
            @ (
                loadings_rot**3
                - (gamma / p)
                * (loadings_rot * np.sum(loadings_rot**2, axis=0, keepdims=True))
            )
        )
        rotation_new = u @ vh
        d_new = np.sum(s)
        if np.abs(d_new - d) < tol:
            break
        rotation = rotation_new
        d = d_new
    return loadings @ rotation, rotation


def _promax(loadings_rotated, power=4):
    k = loadings_rotated.shape[1]
    target = (
        loadings_rotated
        / (np.sum(loadings_rotated**2, axis=1, keepdims=True) + 1e-10) ** 0.5
    )
    target = np.abs(target) ** (power - 1) * target
    t_mat = np.linalg.inv(loadings_rotated.T @ loadings_rotated) @ (
        loadings_rotated.T @ target
    )
    promax_loadings = loadings_rotated @ t_mat
    return promax_loadings / np.sqrt(np.sum(promax_loadings**2, axis=0, keepdims=True))


def _kmo(corr_matrix):
    corr_inv = np.linalg.inv(corr_matrix)
    partial_cov = -corr_inv / np.sqrt(np.outer(np.diag(corr_inv), np.diag(corr_inv)))
    np.fill_diagonal(partial_cov, 0)
    partial_corr = -partial_cov
    np.fill_diagonal(partial_corr, 1)
    corr_sq = corr_matrix**2
    np.fill_diagonal(corr_sq, 0)
    partial_sq = partial_corr**2
    np.fill_diagonal(partial_sq, 0)
    kmo_num = np.sum(corr_sq, axis=1)
    kmo_den = kmo_num + np.sum(partial_sq, axis=1)
    kmo_per_var = kmo_num / kmo_den
    kmo_total = np.sum(corr_sq) / (np.sum(corr_sq) + np.sum(partial_sq))
    return kmo_total, kmo_per_var


def _bartlett_sphericity(corr_matrix, n):
    p = corr_matrix.shape[0]
    det_corr = np.linalg.det(corr_matrix)
    chi2 = -((n - 1) - (2 * p + 5) / 6) * np.log(det_corr)
    df = p * (p - 1) // 2
    p_val = 1 - sp_stats.chi2.cdf(chi2, df)
    return chi2, df, p_val


def _parallel_analysis(data, n_sim=100, seed=42):
    n, p = data.shape
    rng = np.random.default_rng(seed)
    obs_eigenvalues = np.linalg.eigh(np.corrcoef(data, rowvar=False))[0][::-1]
    sim_eigenvalues = np.zeros((n_sim, p))
    for i in range(n_sim):
        sim_data = rng.normal(0, 1, size=(n, p))
        sim_corr = np.corrcoef(sim_data, rowvar=False)
        sim_eigenvalues[i] = np.linalg.eigh(sim_corr)[0][::-1]
    sim_mean = sim_eigenvalues.mean(axis=0)
    sim_95 = np.percentile(sim_eigenvalues, 95, axis=0)
    return obs_eigenvalues, sim_mean, sim_95


def _get_sample_data(choice, random_seed=42):
    rng = np.random.default_rng(random_seed)
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if choice == "Iris (4 numerical variables)":
        df = pd.read_csv(os.path.join(_BASE, "datasets", "iris_bezdek.csv"))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Keep only 4 original numerical columns, rename for clarity
        df = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]
        df["_species"] = df["species"]
        return df
    elif choice == "Palmer Penguins (4 numerical variables)":
        df = pd.read_csv(os.path.join(_BASE, "datasets", "penguins.csv"))
        df = df[
            [
                "species",
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ]
        ].dropna()
        df["species"] = df["species"].astype("category")
        return df
    elif choice == "Big Five Personality (simulated)":
        n = 200
        factors = rng.normal(0, 1, size=(n, 5))
        loadings = np.array(
            [
                [0.7, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.7, 0.0, 0.0, 0.0],
                [0.0, 0.8, 0.0, 0.0, 0.0],
                [0.0, 0.6, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.7, 0.0],
                [0.0, 0.0, 0.0, 0.8, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.7],
                [0.0, 0.0, 0.0, 0.0, 0.8],
                [0.0, 0.0, 0.0, 0.0, 0.6],
            ]
        )
        noise = rng.normal(0, 0.3, size=(n, 14))
        X = factors @ loadings.T + noise
        cols = [
            "Extraversion_1",
            "Extraversion_2",
            "Extraversion_3",
            "Agreeableness_1",
            "Agreeableness_2",
            "Agreeableness_3",
            "Conscientiousness_1",
            "Conscientiousness_2",
            "Conscientiousness_3",
            "Neuroticism_1",
            "Neuroticism_2",
            "Openness_1",
            "Openness_2",
            "Openness_3",
        ]
        df = pd.DataFrame(X, columns=cols)
        return df
    elif choice == "Exam Scores (simulated)":
        n = 150
        ability = rng.normal(0, 1, n)
        math = 0.8 * ability + rng.normal(0, 0.6, n) + 50
        physics = 0.75 * ability + rng.normal(0, 0.66, n) + 50
        chemistry = 0.7 * ability + rng.normal(0, 0.71, n) + 50
        reading = 0.3 * ability + rng.normal(0, 0.95, n) + 50
        writing = 0.35 * ability + rng.normal(0, 0.94, n) + 50
        vocab = 0.25 * ability + rng.normal(0, 0.97, n) + 50
        df = pd.DataFrame(
            {
                "Math": math,
                "Physics": physics,
                "Chemistry": chemistry,
                "Reading": reading,
                "Writing": writing,
                "Vocabulary": vocab,
            }
        )
        return df

    # Fallback: correlated multivariate normal
    n = 150
    n_vars = 6
    n_factors = 2
    true_loadings = np.zeros((n_vars, n_factors))
    true_loadings[:3, 0] = [0.8, 0.7, 0.75]
    true_loadings[3:, 1] = [0.8, 0.7, 0.75]
    factors = rng.normal(0, 1, size=(n, n_factors))
    noise = rng.normal(0, 0.4, size=(n, n_vars))
    X = factors @ true_loadings.T + noise
    cols = [f"Var_{i+1}" for i in range(n_vars)]
    df = pd.DataFrame(X, columns=cols)
    return df


def _st_plot_with_download(fig, key, use_container_width=True, height=None):
    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": key,
            "height": height or fig.layout.height or 500,
            "width": fig.layout.width or 800,
        },
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        "displaylogo": False,
        "scrollZoom": True,
    }
    st.plotly_chart(
        fig, use_container_width=use_container_width, config=config, key=key
    )


# ── MAIN ──


def render_factor_analysis():
    st.title("Factor Analysis")
    st.markdown("""
    Explore latent structure in multivariate data using Principal Component Analysis (PCA)
    and Common Factor Analysis with interactive rotation. Identify which variables cluster together
    and uncover the unobserved factors that drive the correlations in your data.
    """)

    with st.sidebar:
        st.markdown("##### :orange[Factor Analysis Controls]")
        data_source = st.radio(
            "Data Source",
            ["Sample Dataset", "Upload CSV"],
            key="fa_data_source",
        )

    if data_source == "Sample Dataset":
        dataset = st.selectbox(
            "Choose Dataset",
            [
                "Iris (4 numerical variables)",
                "Palmer Penguins (4 numerical variables)",
                "Big Five Personality (simulated)",
                "Exam Scores (simulated)",
                "Synthetic (2-factor, 6 variables)",
            ],
            key="fa_dataset",
        )
        df = _get_sample_data(dataset)
    else:
        uploaded = st.file_uploader(
            "Upload CSV (numerical columns only)",
            type=["csv"],
            key="fa_upload",
        )
        if uploaded is None:
            st.info("Upload a CSV file with numerical columns to begin.")
            return
        df = pd.read_csv(uploaded)

    # Separate numerical vs categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(numerical_cols) < 3:
        st.error("Factor analysis requires at least 3 numerical variables.")
        return

    n, p = len(df), len(numerical_cols)
    col_names = numerical_cols

    # Drop rows with NaN
    clean = df[numerical_cols].dropna()
    if len(clean) < n:
        st.warning(f"Dropped {n - len(clean)} row(s) with missing values. "
                   f"Using {len(clean)} complete observations.")
        n = len(clean)
    if n < 3:
        st.error("Not enough complete observations after removing missing data.")
        return

    # Check for zero-variance columns
    variances = clean.var()
    zero_var = variances[variances < 1e-10].index.tolist()
    if zero_var:
        st.error(f"Columns with zero or near-zero variance detected: {', '.join(zero_var)}. "
                 f"These cannot be used in factor analysis. Please remove them and try again.")
        return

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)
    corr_matrix = np.corrcoef(X_scaled, rowvar=False)

    # Guard: ensure correlation matrix is finite
    if not np.all(np.isfinite(corr_matrix)):
        st.error("The correlation matrix contains NaN or infinite values. "
                 "Check your data for constant or missing columns.")
        return

    # ── SECTION 1: CORRELATION MATRIX ──
    st.subheader("Step 1: Examine the Correlation Matrix", divider="orange")
    st.markdown("""
    Factor analysis starts with the correlation matrix. Variables that correlate highly
    within clusters and weakly across clusters suggest latent factors.
    """)

    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr_matrix,
            x=col_names,
            y=col_names,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig_corr.update_layout(
        template="plotly_dark",
        height=500,
        title="Correlation Matrix",
        xaxis=dict(tickangle=45),
        margin=dict(l=10, r=10, t=40, b=80),
    )
    _st_plot_with_download(fig_corr, "fa_corr_matrix")

    # ── SECTION 2: SAMPLING ADEQUACY ──
    st.subheader("Step 2: Assess Sampling Adequacy", divider="orange")

    kmo_total, kmo_per_var = _kmo(corr_matrix)
    chi2, bartlett_df, bartlett_p = _bartlett_sphericity(corr_matrix, n)

    c1, c2, c3 = st.columns(3)
    c1.metric("KMO (Overall)", f"{kmo_total:.4f}")
    c2.metric("Bartlett χ²", f"{chi2:.2f}")
    c3.metric("Bartlett p-value", f"{bartlett_p:.6f}")

    kmo_interpret = (
        "Marvelous"
        if kmo_total >= 0.9
        else (
            "Meritorious"
            if kmo_total >= 0.8
            else (
                "Middling"
                if kmo_total >= 0.7
                else "Mediocre" if kmo_total >= 0.6 else "Miserable"
            )
        )
    )
    st.success(
        f"**KMO = {kmo_total:.4f}** — {kmo_interpret}. "
        + (
            "Variable correlations are adequate for factor analysis."
            if kmo_total >= 0.6
            else "Data may not be suitable for factor analysis."
        )
    )
    if bartlett_p < 0.05:
        st.success(
            f"Bartlett's test is significant (p = {bartlett_p:.6f}) — the correlation matrix is not an identity matrix. Data are suitable for factor analysis."
        )
    else:
        st.warning(
            f"Bartlett's test is NOT significant (p = {bartlett_p:.6f}). The correlation matrix may not differ enough from identity for meaningful factor extraction."
        )

    kmo_df = pd.DataFrame(
        {
            "Variable": col_names,
            "KMO": [f"{v:.4f}" for v in kmo_per_var],
        }
    )
    _apa_table(kmo_df, title="KMO per Variable")

    # ── SECTION 3: EXTRACTION ──
    st.subheader("Step 3: Extract Factors", divider="orange")

    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox(
            "Extraction Method",
            ["Principal Component Analysis (PCA)", "Principal Axis Factoring (PAF)"],
            key="fa_method",
        )
    with c2:
        criterion = st.selectbox(
            "Factor Retention Criterion",
            [
                "Eigenvalue > 1 (Kaiser)",
                "Scree Plot + Parallel Analysis",
                "Fixed Number",
            ],
            key="fa_criterion",
        )
    with c3:
        rotation = st.selectbox(
            "Rotation Method",
            ["Varimax (Orthogonal)", "Promax (Oblique)", "None"],
            key="fa_rotation",
        )

    # Eigenanalysis
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    except np.linalg.LinAlgError:
        st.error("Eigenvalue decomposition did not converge. "
                 "This usually happens when the data matrix is near-singular. "
                 "Try removing highly correlated variables or increasing sample size.")
        return
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    total_var = eigenvalues.sum()

    # Parallel analysis
    try:
        obs_ev, sim_mean_ev, sim_95_ev = _parallel_analysis(X_scaled, n_sim=100, seed=42)
    except np.linalg.LinAlgError:
        st.warning("Parallel analysis could not converge. Using Kaiser criterion only.")
        obs_ev = eigenvalues.copy()
        sim_mean_ev = eigenvalues * 0.5
        sim_95_ev = eigenvalues * 0.7

    if criterion == "Eigenvalue > 1 (Kaiser)":
        n_factors = int(np.sum(eigenvalues > 1))
        n_factors = max(1, n_factors)
        st.info(
            f"Kaiser criterion retains **{n_factors}** factor(s) with eigenvalue > 1."
        )
    elif criterion == "Fixed Number":
        max_f = min(p, 8)
        n_factors = st.slider(
            "Number of Factors", 1, max_f, min(3, max_f), key="fa_n_factors"
        )
    else:
        # Parallel analysis: retain where obs > sim_95
        n_k1 = int(np.sum(eigenvalues > 1))
        n_pa = int(np.sum(obs_ev > sim_95_ev))
        n_factors = max(1, n_pa)
        st.info(
            f"Parallel analysis (95th percentile) retains **{n_pa}** factor(s). "
            f"Kaiser criterion retains **{n_k1}**. Adjust below if needed."
        )
        max_f = min(p, 8)
        n_factors = st.slider(
            "Number of Factors", 1, max_f, n_factors, key="fa_n_factors_pa"
        )

    if n_factors >= p:
        n_factors = p - 1
        st.warning(
            f"Reduced to {n_factors} factors (cannot extract as many as variables)."
        )

    # Scree plot with parallel analysis overlay
    fig_scree = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Scree Plot", "Variance Explained"],
        column_widths=[0.6, 0.4],
    )
    fig_scree.add_trace(
        go.Scatter(
            x=list(range(1, p + 1)),
            y=eigenvalues,
            mode="lines+markers",
            marker=dict(color="#4C78A8", size=8),
            line=dict(color="#4C78A8"),
            name="Observed",
        ),
        row=1,
        col=1,
    )
    fig_scree.add_trace(
        go.Scatter(
            x=list(range(1, p + 1)),
            y=sim_mean_ev,
            mode="lines+markers",
            marker=dict(color="#54A24B", size=6, symbol="diamond"),
            line=dict(color="#54A24B", dash="dot"),
            name="Parallel (mean)",
        ),
        row=1,
        col=1,
    )
    fig_scree.add_trace(
        go.Scatter(
            x=list(range(1, p + 1)),
            y=sim_95_ev,
            mode="lines+markers",
            marker=dict(color="#E45756", size=6, symbol="diamond-open"),
            line=dict(color="#E45756", dash="dash"),
            name="Parallel (95th %ile)",
        ),
        row=1,
        col=1,
    )
    fig_scree.add_hline(
        y=1,
        line_dash="dash",
        line_color="white",
        opacity=0.3,
        annotation_text="Kaiser (λ=1)",
        row=1,
        col=1,
    )
    fig_scree.add_trace(
        go.Bar(
            x=list(range(1, p + 1)),
            y=eigenvalues / total_var * 100,
            marker_color="#54A24B",
            name="% Variance",
        ),
        row=1,
        col=2,
    )
    cum_var = np.cumsum(eigenvalues) / total_var * 100
    fig_scree.add_trace(
        go.Scatter(
            x=list(range(1, p + 1)),
            y=cum_var,
            mode="lines+markers",
            marker=dict(color="#B279A2", size=6),
            line=dict(color="#B279A2"),
            name="Cumulative %",
        ),
        row=1,
        col=2,
    )
    fig_scree.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig_scree.update_xaxes(title_text="Factor Number", row=1, col=1)
    fig_scree.update_xaxes(title_text="Factor Number", row=1, col=2)
    fig_scree.update_yaxes(title_text="Eigenvalue", row=1, col=1)
    fig_scree.update_yaxes(title_text="Variance (%)", row=1, col=2)
    _st_plot_with_download(fig_scree, "fa_scree")

    # Variance table
    var_df = pd.DataFrame(
        {
            "Factor": [f"F{i+1}" for i in range(p)],
            "Eigenvalue": [f"{eigenvalues[i]:.4f}" for i in range(p)],
            "Parallel (95th)": [f"{sim_95_ev[i]:.4f}" for i in range(p)],
            "% Variance": [f"{eigenvalues[i] / total_var * 100:.2f}" for i in range(p)],
            "Cumulative %": [f"{cum_var[i]:.2f}" for i in range(p)],
        }
    )
    _apa_table(var_df, title="Eigenvalues and Variance Explained")

    # ── SECTION 4: LOADINGS ──
    st.subheader("Step 4: Interpret Factor Loadings", divider="orange")
    st.markdown(
        f"**Extracting {n_factors} factors using {method.split('(')[0].strip()}**"
    )

    c_suppress = st.checkbox(
        "Suppress loadings |r| < 0.3 (cleaner view)", True, key="fa_suppress"
    )

    def _pca_loadings():
        return eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    if "PCA" in method:
        loadings = _pca_loadings()
        communalities = np.sum(loadings**2, axis=1)
    else:
        try:
            h2_init = 1 - 1 / np.diag(np.linalg.inv(corr_matrix))
        except np.linalg.LinAlgError:
            st.warning("PAF could not invert the correlation matrix. Falling back to PCA-based loadings.")
            loadings = _pca_loadings()
            communalities = np.sum(loadings**2, axis=1)
            paf_ok = False
        else:
            paf_ok = True

        if paf_ok:
            h2 = h2_init.copy()
            reduced = corr_matrix.copy()
            for iteration in range(50):
                np.fill_diagonal(reduced, h2)
                try:
                    evals, evecs = np.linalg.eigh(reduced)
                except np.linalg.LinAlgError:
                    st.warning("PAF iteration failed. Falling back to PCA-based loadings.")
                    loadings = _pca_loadings()
                    communalities = np.sum(loadings**2, axis=1)
                    break
                evals = evals[::-1]
                evecs = evecs[:, ::-1]
                loadings = evecs[:, :n_factors] * np.sqrt(np.maximum(evals[:n_factors], 0))
                h2_new = np.sum(loadings**2, axis=1)
                if np.max(np.abs(h2_new - h2)) < 1e-4:
                    break
                h2 = h2_new
                np.fill_diagonal(reduced, 1)
            else:
                communalities = h2

    # Rotation
    if rotation == "Varimax (Orthogonal)":
        loadings_rot, _ = _varimax(loadings)
        rot_label = "Varimax-rotated"
    elif rotation == "Promax (Oblique)":
        loadings_var, _ = _varimax(loadings)
        loadings_rot = _promax(loadings_var, power=4)
        rot_label = "Promax-rotated"
    else:
        loadings_rot = loadings.copy()
        rot_label = "Unrotated"

    # Loading display with optional suppression
    loadings_display = loadings_rot.copy()
    if c_suppress:
        loadings_display[np.abs(loadings_display) < 0.3] = np.nan

    loading_df = pd.DataFrame(
        loadings_rot,
        index=col_names,
        columns=[f"Factor {i+1}" for i in range(n_factors)],
    )

    fig_load = go.Figure(
        data=go.Heatmap(
            z=loadings_display,
            x=[f"F{i+1}" for i in range(n_factors)],
            y=col_names,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(loadings_rot, 3),
            texttemplate="%{text}",
            hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig_load.update_layout(
        template="plotly_dark",
        height=min(600, 40 * p + 100),
        title=f"Factor Loadings ({rot_label})"
        + (" [|r| < 0.3 hidden]" if c_suppress else ""),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    _st_plot_with_download(fig_load, "fa_loadings_heatmap")

    _apa_table(loading_df, title=f"Factor Loading Matrix ({rot_label})")

    # Primary factor clustering
    primary = np.argmax(np.abs(loadings_rot), axis=1)
    cluster_df = pd.DataFrame(
        {
            "Variable": col_names,
            "Primary Factor": [f"F{p+1}" for p in primary],
            "Loading": [f"{loadings_rot[i, primary[i]]:.4f}" for i in range(p)],
        }
    )
    _apa_table(cluster_df, title="Variable Clustering by Primary Factor")

    # Communalities
    st.subheader("Step 5: Communalities", divider="orange")
    comm_df = pd.DataFrame(
        {
            "Variable": col_names,
            "Communality (h²)": [f"{communalities[i]:.4f}" for i in range(p)],
            "Uniqueness": [f"{1 - communalities[i]:.4f}" for i in range(p)],
        }
    )
    _apa_table(comm_df, title="Communalities (variance explained by all factors)")

    st.markdown("""
    - **Communality (h²)**: Proportion of each variable's variance explained by the extracted factors.
    - **Uniqueness**: Proportion of variance unique to that variable (not shared with others).
    - Variables with low communality (e.g., < 0.3) may not be well-represented by the factor solution and
    could be candidates for removal.
    """)

    # ── SECTION 5.5: RELIABILITY ANALYSIS ──
    st.subheader("Step 6: Reliability Analysis (Cronbach's Alpha)", divider="orange")
    st.markdown("""
    **Cronbach's alpha** measures internal consistency — how closely related the items are as a group.
    High alpha (≥ 0.70) suggests the items measure the same latent construct, supporting scale reliability.
    """)

    alpha_data = df[numerical_cols].dropna()
    if len(alpha_data) < 3:
        st.warning("Need at least 3 observations for reliability analysis.")
    else:
        k = alpha_data.shape[1]
        item_vars = alpha_data.var(ddof=1, axis=0)
        total_var = alpha_data.sum(axis=1).var(ddof=1)
        alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Cronbach's α", f"{alpha:.4f}")
        if alpha >= 0.9:
            alpha_interp = "Excellent"
        elif alpha >= 0.8:
            alpha_interp = "Good"
        elif alpha >= 0.7:
            alpha_interp = "Acceptable"
        elif alpha >= 0.6:
            alpha_interp = "Questionable"
        elif alpha >= 0.5:
            alpha_interp = "Poor"
        else:
            alpha_interp = "Unacceptable"
        col_b.metric("Interpretation", alpha_interp)
        col_c.metric("Number of Items", k)

        st.markdown("#### Item-Total Statistics")
        item_means = alpha_data.mean(axis=0)
        item_sds = alpha_data.std(ddof=1, axis=0)
        item_rest_corr = []
        alpha_if_deleted = []
        for i in range(k):
            rest = alpha_data.drop(columns=[alpha_data.columns[i]]).sum(axis=1)
            corr = np.corrcoef(alpha_data.iloc[:, i], rest)[0, 1]
            item_rest_corr.append(corr)

            rest_vars = item_vars.drop(index=alpha_data.columns[i])
            rest_total_var = rest.var(ddof=1)
            alpha_del = (
                ((k - 1) / (k - 2)) * (1 - rest_vars.sum() / rest_total_var)
                if (k - 1) > 1
                else np.nan
            )
            alpha_if_deleted.append(alpha_del)

        item_stats = pd.DataFrame(
            {
                "Item": alpha_data.columns,
                "Mean": [f"{m:.2f}" for m in item_means],
                "SD": [f"{s:.2f}" for s in item_sds],
                "Item-Rest r": [f"{r:.4f}" for r in item_rest_corr],
                "α if Deleted": [
                    f"{a:.4f}" if not np.isnan(a) else "—" for a in alpha_if_deleted
                ],
            }
        )
        _apa_table(item_stats, title="Item-Total Statistics")

        notable = []
        for i in range(k):
            if alpha_if_deleted[i] > alpha and not np.isnan(alpha_if_deleted[i]):
                notable.append(
                    f"- Removing **{alpha_data.columns[i]}** raises α to **{alpha_if_deleted[i]:.4f}**"
                )
        if notable:
            st.markdown("#### Items That May Improve Reliability")
            for note in notable:
                st.markdown(note)

        if alpha < 0.7:
            st.warning(
                "⚠️ Alpha below 0.70 suggests the items may not reliably measure a single construct. "
                "Consider reviewing which items are reducing consistency (see 'α if Deleted' column above)."
            )
        else:
            st.success(
                "✅ Reliability is acceptable or better (α ≥ 0.70). "
                "The items demonstrate good internal consistency."
            )

    # ── SECTION 6: SCORE PLOT ──
    if n_factors >= 2:
        st.subheader("Step 7: Factor Score Plot", divider="orange")
        st.markdown("""
        Plot observations in the space of the first two factors. Points that cluster together
        share similar factor profiles. Arrows show variable loading directions (biplot).
        """)

        if "PCA" in method:
            scores = X_scaled @ eigenvectors[:, :n_factors]
        else:
            scores = X_scaled @ loadings_rot

        # Color-by dropdown
        group_opts = ["None"] + [c for c in cat_cols if df[c].nunique() <= 20]
        # Also check for low-cardinality integer columns not in the analysis set
        for c in df.columns:
            if c not in numerical_cols and c not in cat_cols:
                if df[c].nunique() <= 10:
                    group_opts.append(c)

        group_by = st.selectbox("Color by", group_opts, key="fa_group")

        fig_scores = go.Figure()
        colors = px.colors.qualitative.Plotly

        if group_by != "None" and group_by in df.columns:
            # Align color-by with the rows used in the analysis (NaN rows dropped)
            df_aligned = df.loc[clean.index]
            unique_vals = df_aligned[group_by].unique()
            for idx, g in enumerate(unique_vals):
                mask = df_aligned[group_by] == g
                fig_scores.add_trace(
                    go.Scatter(
                        x=(
                            scores[mask.values, 0]
                            if isinstance(mask, pd.Series)
                            else scores[mask, 0]
                        ),
                        y=(
                            scores[mask.values, 1]
                            if isinstance(mask, pd.Series)
                            else scores[mask, 1]
                        ),
                        mode="markers",
                        marker=dict(size=6, color=colors[idx % len(colors)]),
                        name=str(g),
                        legendgroup=str(g),
                    )
                )
        else:
            fig_scores.add_trace(
                go.Scatter(
                    x=scores[:, 0],
                    y=scores[:, 1],
                    mode="markers",
                    marker=dict(color="#4C78A8", size=6, opacity=0.6),
                    name="Observations",
                )
            )

        # Biplot: overlay loading vectors
        show_biplot = st.checkbox(
            "Show biplot (loading vectors)", True, key="fa_biplot"
        )
        if show_biplot:
            # Scale loadings for visibility
            score_range = max(np.abs(scores[:, 0]).max(), np.abs(scores[:, 1]).max())
            scale_factor = score_range * 0.85
            biplot_loadings = loadings_rot[:, :2] * scale_factor

            for i in range(p):
                fig_scores.add_annotation(
                    x=biplot_loadings[i, 0],
                    y=biplot_loadings[i, 1],
                    text=col_names[i][:12],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="#E45756",
                    ax=0,
                    ay=0,
                    font=dict(color="#E45756", size=10),
                    opacity=0.8,
                )

            # Unit circle reference
            theta = np.linspace(0, 2 * np.pi, 100)
            fig_scores.add_trace(
                go.Scatter(
                    x=scale_factor * np.cos(theta),
                    y=scale_factor * np.sin(theta),
                    mode="lines",
                    line=dict(color="white", dash="dot", width=1),
                    opacity=0.15,
                    name="Unit circle",
                    showlegend=False,
                )
            )

        var_explained_f1 = eigenvalues[0] / total_var * 100
        var_explained_f2 = eigenvalues[1] / total_var * 100
        fig_scores.update_layout(
            template="plotly_dark",
            height=550,
            title=(
                f"Factor Score Plot with Biplot ({rot_label})"
                if show_biplot
                else f"Factor Score Plot ({rot_label})"
            ),
            xaxis_title=f"Factor 1 ({var_explained_f1:.1f}% variance)",
            yaxis_title=f"Factor 2 ({var_explained_f2:.1f}% variance)",
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode="closest",
        )
        fig_scores.update_yaxes(scaleanchor="x", scaleratio=1)
        _st_plot_with_download(fig_scores, "fa_score_plot")

    # ── SECTION 6: INTERPRETATION ──
    st.subheader("Interpretation & Guidelines", divider="orange")
    with st.expander("How to Interpret Factor Analysis Results", expanded=True):
        st.markdown(f"""
        **Factor loadings** are correlations between variables and factors (rotated loadings range from -1 to 1).

        **Rules of thumb:**
        - |Loading| ≥ 0.70 → Excellent (variable is strongly associated with the factor)
        - |Loading| ≥ 0.50 → Practically significant
        - |Loading| ≥ 0.30 → Minimally acceptable
        - Cross-loadings (|load| ≥ 0.30 on two+ factors) suggest a variable is ambiguous

        **Current solution:**
        - {n_factors} factor(s) extracted using {method.split('(')[0].strip()} with {rot_label} rotation
        - KMO = {kmo_total:.4f} ({kmo_interpret})
        - Total variance explained: {cum_var[n_factors - 1]:.1f}% by {n_factors} factor(s)
        """)

    with st.expander("Methodological Notes", expanded=False):
        st.markdown("""
        - **Kaiser criterion (eigenvalue > 1)** retains factors that explain more variance than a single standardized variable.
        This is the default in most software but can over- or under-extract.
        - **Scree plot** shows the "elbow" where eigenvalues level off — factors before the elbow are retained.
        - **Parallel analysis** compares observed eigenvalues to those from random data of the same size. Factors with
        eigenvalues exceeding the 95th percentile of random eigenvalues are retained. This is considered the most
        accurate retention method.
        - **Varimax** (orthogonal) assumes factors are uncorrelated. **Promax** (oblique) allows correlated factors and is
        often more realistic for psychological/social science data.
        - Factor analysis assumes **multivariate normality** for maximum likelihood estimation, but PCA-based extraction
        is descriptive and does not require distributional assumptions.
        - **Sample size guidelines**: n ≥ 100 or n ≥ 5× the number of variables is generally recommended.
        """)

    with st.expander("Common Pitfalls", expanded=False):
        st.markdown("""
        - **Over-factoring**: Extracting too many factors leads to uninterpretable "factor splitting."
        - **Under-factoring**: Too few factors merge distinct constructs, losing information.
        - **Ignoring cross-loadings**: Variables that load on multiple factors may need to be removed or theoretically reconsidered.
        - **Reifying factors**: A factor is only as meaningful as the variables that define it. Naming a factor does not
        validate its existence — theoretical grounding is essential.
        - **Running FA on too-small samples**: Estimates are unstable with n < 50 or variable-to-factor ratios below 5:1.
        - **Treating ordinal variables as continuous**: Polychoric correlations are preferred for ordinal data.
        """)
