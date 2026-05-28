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

from .shared import _rng, _gen_corr, _gen_reg

def pca_widget():
    st.markdown("## PCA Scatter Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="pca_n")
        n_groups = st.selectbox(
            "Number of Groups", [2, 3, 4], index=1, key="pca_groups"
        )
        separation = st.slider("Group Separation", 0.0, 5.0, 2.0, 0.1, key="pca_sep")
        n_features = st.selectbox(
            "Number of Features", [5, 10, 20], index=0, key="pca_feat"
        )
        show_ellipse = st.toggle("Show Confidence Ellipses", True, key="pca_ellipse")
    k, n_f = int(n_groups), int(n_features)
    np.random.seed(42)
    X_list, y_list = [], []
    for i in range(k):
        center = np.random.uniform(-separation, separation, n_f) * (i / max(k - 1, 1))
        X_list.append(np.random.normal(center, 1, (n // k, n_f)))
        y_list.append(np.full(n // k, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    var_expl = pca.explained_variance_ratio_
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    for i in range(k):
        mask = y == i
        fig.add_trace(
            go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode="markers",
                name=f"Group {i}",
                marker=dict(color=colors[i], size=5, opacity=0.7),
                hovertemplate=f"Group {i}<br>PC1 = %{{x:.2f}}<br>PC2 = %{{y:.2f}}<extra></extra>",
            )
        )
        if show_ellipse and mask.sum() > 2:
            x_c, y_c = np.mean(X_pca[mask, 0]), np.mean(X_pca[mask, 1])
            angle = np.linspace(0, 2 * np.pi, 50)
            cov_ = np.cov(X_pca[mask].T)
            try:
                eigvals2, eigvecs = np.linalg.eigh(cov_)
                order = eigvals2.argsort()[::-1]
                eigvals2, eigvecs = eigvals2[order], eigvecs[:, order]
                theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
                a, b = 2 * np.sqrt(eigvals2[0]), 2 * np.sqrt(eigvals2[1])
                x_e = (
                    x_c
                    + a * np.cos(theta) * np.cos(angle)
                    - b * np.sin(theta) * np.sin(angle)
                )
                y_e = (
                    y_c
                    + a * np.sin(theta) * np.cos(angle)
                    + b * np.cos(theta) * np.sin(angle)
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_e,
                        y=y_e,
                        mode="lines",
                        line=dict(color=colors[i], width=1.5, dash="dash"),
                        showlegend=False,
                    )
                )
            except Exception:
                pass
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title=f"PC1 ({var_expl[0]:.1%} variance)",
        yaxis_title=f"PC2 ({var_expl[1]:.1%} variance)",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- PCs = directions of max variance\n"
                "- Close points = similar profiles\n"
                "- Separated groups = distinct clusters\n"
                "- Ellipses = 95% confidence region\n"
                "- Axis labels show % variance explained"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Reduce dimensionality\n"
                "- Visualize high-dim data\n"
                "- Check for natural clusters\n"
                "- Exploratory data analysis"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- MANOVA (on PCs or raw)\n"
                "- Factor analysis\n"
                "- K-means clustering\n"
                "- PERMANOVA"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "PCA assumes linear relationships. Non-linear "
                "structure (e.g., U-shape, spiral) will NOT "
                "be captured. Use t-SNE or UMAP for non-linear "
                "dimensionality reduction."
            )



def manova_widget():
    st.markdown("## MANOVA Group Clouds")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Samples per Group", 10, 100, 30, key="man_n")
        n_groups = st.selectbox("Number of Groups", [2, 3], index=1, key="man_groups")
        n_dims = st.selectbox("Number of DVs", [2, 3], index=0, key="man_dims")
        effect = st.slider("Group Separation Effect", 0.0, 5.0, 2.0, 0.1, key="man_eff")
        show_centroids = st.toggle("Show Group Centroids", True, key="man_cent")
    k = int(n_groups)
    d = int(n_dims)
    np.random.seed(42)
    groups, names, colors = [], [], []
    for i in range(k):
        center = np.full(d, i * effect)
        cov = np.eye(d) * 0.5
        groups.append(np.random.multivariate_normal(center, cov, n))
        names.append(f"Group {chr(65 + i)}")
        colors.append(px.colors.qualitative.Plotly[i])
    fig = go.Figure()
    if d == 2:
        for i, (g, name, color) in enumerate(zip(groups, names, colors)):
            fig.add_trace(
                go.Scatter(
                    x=g[:, 0],
                    y=g[:, 1],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=5, opacity=0.7),
                    hovertemplate=f"{name}<br>DV1 = %{{x:.2f}}<br>DV2 = %{{y:.2f}}<extra></extra>",
                )
            )
            if show_centroids:
                fig.add_trace(
                    go.Scatter(
                        x=[np.mean(g[:, 0])],
                        y=[np.mean(g[:, 1])],
                        mode="markers",
                        marker=dict(color=color, size=15, symbol="x"),
                        showlegend=False,
                    )
                )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            hovermode="closest",
            xaxis_title="DV1",
            yaxis_title="DV2",
        )
    else:
        for i, (g, name, color) in enumerate(zip(groups, names, colors)):
            fig.add_trace(
                go.Scatter3d(
                    x=g[:, 0],
                    y=g[:, 1],
                    z=g[:, 2],
                    mode="markers",
                    name=name,
                    marker=dict(color=color, size=4, opacity=0.7),
                    hovertemplate=f"{name}<br>DV1=%{{x:.2f}}<br>DV2=%{{y:.2f}}<br>DV3=%{{z:.2f}}<extra></extra>",
                )
            )
            if show_centroids:
                fig.add_trace(
                    go.Scatter3d(
                        x=[np.mean(g[:, 0])],
                        y=[np.mean(g[:, 1])],
                        z=[np.mean(g[:, 2])],
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="diamond"),
                        showlegend=False,
                    )
                )
        fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
            scene=dict(
                xaxis_title="DV1",
                yaxis_title="DV2",
                zaxis_title="DV3",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            ),
        )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each color = one group\n"
                "- Distance between = group difference\n"
                "- Overlap = no significant difference\n"
                "- X marks = group centroid (mean)\n"
                "- Ellipsoid shape = covariance"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Compare groups on multiple DVs\n"
                "- Control for correlated outcomes\n"
                "- Multivariate experimental design\n"
                "- Protect against inflated Type I error"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Pillai's Trace\n"
                "- Wilks' Lambda\n"
                "- Hotelling-Lawley Trace\n"
                "- Roy's Largest Root"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "MANOVA requires multivariate normality and "
                "homogeneity of covariance matrices. Violations "
                "inflate Type I error. When assumptions fail, "
                "consider PERMANOVA or non-parametric alternatives."
            )



def cluster_widget():
    st.markdown("## Cluster Visualization")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 30, 500, 150, key="clust_n")
        k_true = st.selectbox(
            "True Number of Clusters", [2, 3, 4, 5], index=1, key="clust_k"
        )
        separation = st.slider(
            "Cluster Separation", 0.5, 5.0, 2.0, 0.1, key="clust_sep"
        )
        n_features = st.selectbox(
            "Project from Features", [2, 5, 10], index=0, key="clust_feat"
        )
        show_centers = st.toggle("Show Cluster Centers", True, key="clust_center")
    k, n_f = int(k_true), int(n_features)
    np.random.seed(42)
    if n_f == 2:
        centers = [
            [
                np.cos(2 * np.pi * i / k) * separation,
                np.sin(2 * np.pi * i / k) * separation,
            ]
            for i in range(k)
        ]
        X = np.vstack(
            [np.random.normal(centers[i], 0.5, (n // k, 2)) for i in range(k)]
        )
    else:
        X_list = []
        for i in range(k):
            center = np.random.uniform(-separation, separation, n_f) * (
                i / max(k - 1, 1)
            )
            X_list.append(np.random.normal(center, 1, (n // k, n_f)))
        X = np.vstack(X_list)
        X = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    centers2 = kmeans.cluster_centers_
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    for i in range(k):
        mask = labels == i
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=f"Cluster {i}",
                marker=dict(color=colors[i], size=5, opacity=0.6),
                hovertemplate=f"Cluster {i}<br>x = %{{x:.2f}}<br>y = %{{y:.2f}}<extra></extra>",
            )
        )
    if show_centers:
        fig.add_trace(
            go.Scatter(
                x=centers2[:, 0],
                y=centers2[:, 1],
                mode="markers",
                marker=dict(
                    color="black",
                    size=12,
                    symbol="x",
                    line=dict(color="white", width=1),
                ),
                name="Centers",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        xaxis_title="PC1 / Feature 1",
        yaxis_title="PC2 / Feature 2",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each color = discovered cluster\n"
                "- X marks = cluster center\n"
                "- Tight clusters = well-separated\n"
                "- Overlap = ambiguous assignment\n"
                "- K-means assumes spherical clusters"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Discover natural groupings\n"
                "- Segment patients/populations\n"
                "- Pattern recognition\n"
                "- Exploratory data mining"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Silhouette score\n"
                "- Elbow method (WCSS)\n"
                "- Gap statistic\n"
                "- Davies-Bouldin index"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "K-means requires specifying k beforehand and "
                "assumes spherical, equally-sized clusters. "
                "Elongated or irregular clusters will be "
                "incorrectly split. Use DBSCAN or GMM for "
                "complex shapes."
            )



def scatter3d_widget():
    st.markdown("## 3D Scatter Explorer")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 20, 500, 100, key="s3d_n")
        n_groups = st.selectbox(
            "Number of Groups", [1, 2, 3], index=1, key="sd3_groups"
        )
        correlation = st.slider(
            "Variable Correlation",
            -1.0,
            1.0,
            0.5,
            0.05,
            key="s3d_corr",
            help="Correlation among the 3 dimensions",
        )
        spread = st.slider("Data Spread", 0.2, 3.0, 1.0, 0.1, key="s3d_spread")
        rotate = st.toggle("Auto-Rotate", True, key="s3d_rotate")
    k = int(n_groups)
    np.random.seed(42)
    cov_mat = np.array(
        [
            [1, correlation, correlation],
            [correlation, 1, correlation],
            [correlation, correlation, 1],
        ]
    )
    eigvals = np.linalg.eigvalsh(cov_mat)
    if min(eigvals) <= 0:
        cov_mat += np.eye(3) * (abs(min(eigvals)) + 0.01)
    colors = px.colors.qualitative.Plotly[:k]
    fig = go.Figure()
    for i in range(k):
        offset = np.full(3, i * spread * 1.5)
        data = np.random.multivariate_normal(offset, cov_mat * spread, n // k)
        fig.add_trace(
            go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode="markers",
                name=f"Group {chr(65 + i)}" if k > 1 else "Data",
                marker=dict(
                    color=colors[i] if k > 1 else "#4C78A8", size=4, opacity=0.7
                ),
                hovertemplate=f"{'Group ' + chr(65 + i) if k > 1 else 'Point'}"
                f"<br>X=%{{x:.2f}}<br>Y=%{{y:.2f}}<br>Z=%{{z:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each axis = one variable\n"
                "- Position in 3D space = multi-dim profile\n"
                "- Clusters = groups with similar profiles\n"
                "- Rotation reveals different patterns\n"
                "- Elliptical shape = correlated variables"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Explore 3-variable relationships\n"
                "- Identify 3D clusters\n"
                "- Present multivariate patterns\n"
                "- Interactive data exploration"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- MANOVA\n"
                "- Multivariate regression\n"
                "- Canonical correlation\n"
                "- 3D PCA visualization"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "3D plots can obscure patterns depending on "
                "viewing angle. Always rotate and view from "
                "multiple perspectives. Pre-projected 2D views "
                "(PCA) often reveal structure more clearly."
            )



def parallel_coords_widget():
    st.markdown("## Parallel Coordinates Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 50, 500, 200, key="parcoords_n")
        n_dims = st.selectbox(
            "Number of Dimensions", [4, 5, 6, 7, 8], index=1, key="parcoords_k"
        )
        n_clusters = st.selectbox(
            "Number of Clusters", [2, 3, 4], index=0, key="parcoords_clust"
        )
    np.random.seed(42)
    k = int(n_dims)
    c = int(n_clusters)
    cluster_centers = np.random.uniform(-3, 3, (c, k))
    data_list = []
    labels = []
    for i in range(c):
        n_per = n // c
        data_list.append(np.random.normal(cluster_centers[i], 0.6, (n_per, k)))
        labels.extend([i] * n_per)
    X = np.vstack(data_list)[:n]
    labels = np.array(labels[:n])
    col_names = [f"Dim {i+1}" for i in range(k)]
    df = pd.DataFrame(X, columns=col_names)
    df["Cluster"] = labels
    fig = px.parallel_coordinates(
        df, color="Cluster", dimensions=col_names, color_continuous_scale="Viridis"
    )
    fig.update_layout(
        template="plotly_dark", height=500, margin=dict(l=10, r=10, t=30, b=10)
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Each vertical axis = one variable\n"
                "- Each line = one observation\n"
                "- Crossing lines = negative correlation\n"
                "- Parallel lines = positive correlation"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Visualize high-dimensional data\n"
                "- Identify variable relationships\n"
                "- Find multivariate patterns\n"
                "- Complement to PCA"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- MANOVA\n"
                "- Canonical correlation\n"
                "- Discriminant analysis\n"
                "- Cluster validation"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Axis order affects interpretation. "
                "Reorder axes to highlight patterns. "
                "Too many observations creates clutter "
                "— consider sampling."
            )



def contour_widget():
    st.markdown("## Contour Plot")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        n = st.slider("Sample Size", 100, 1000, 300, key="cont_n")
        bw = st.slider("Bandwidth", 0.1, 1.0, 0.3, 0.05, key="cont_bw")
        dist = st.selectbox(
            "Distribution",
            ["Bivariate Normal", "Two Clusters", "Donut"],
            key="cont_dist",
        )
    np.random.seed(42)
    if dist == "Bivariate Normal":
        x = np.random.normal(0, 1, n)
        y = np.random.normal(0, 1, n)
    elif dist == "Two Clusters":
        x = np.concatenate(
            [np.random.normal(-2, 0.8, n // 2), np.random.normal(2, 0.8, n // 2)]
        )
        y = np.concatenate(
            [np.random.normal(0, 0.8, n // 2), np.random.normal(0, 0.8, n // 2)]
        )
    else:
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.normal(2, 0.4, n)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
    kde = stats.gaussian_kde(np.vstack([x, y]), bw_method=bw)
    xi, yi = np.meshgrid(
        np.linspace(min(x) - 1, max(x) + 1, 50), np.linspace(min(y) - 1, max(y) + 1, 50)
    )
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xi[0],
            y=yi[:, 0],
            z=zi,
            colorscale="Viridis",
            contours=dict(coloring="heatmap"),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>density=%{z:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="white", size=2, opacity=0.3),
            name="Data",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="X",
        yaxis_title="Y",
    )
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("📖 Interpretation & Guidance", expanded=True):
        col_i, col_w, col_t, col_m = st.columns(4)
        with col_i:
            st.info(
                "**Interpretation**\n\n"
                "- Lines = constant density\n"
                "- Closer lines = steeper gradient\n"
                "- Peaks = dense regions\n"
                "- Color = density intensity"
            )
        with col_w:
            st.success(
                "**When To Use**\n\n"
                "- Visualize 2D distribution\n"
                "- Identify density peaks\n"
                "- Replace scatter for large n\n"
                "- Topographic data display"
            )
        with col_t:
            st.warning(
                "**Associated Tests**\n\n"
                "- Bivariate normality test\n"
                "- Hotelling's T-squared\n"
                "- Multivariate outlier test\n"
                "- Kernel density estimation"
            )
        with col_m:
            st.error(
                "**Common Mistake**\n\n"
                "Bandwidth changes contours "
                "dramatically. Too low = spikes, "
                "too high = oversmoothed. "
                "Use cross-validation."
            )




GRAPHS = {
    "PCA Scatter Plot": pca_widget,
    "MANOVA Group Clouds": manova_widget,
    "Cluster Visualization": cluster_widget,
    "3D Scatter Explorer": scatter3d_widget,
    "Parallel Coordinates Plot": parallel_coords_widget,
    "Contour Plot": contour_widget
}