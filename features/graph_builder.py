"""Interactive Graph Builder — flexible point-and-click charting from user data.

Lets users import data (CSV/Excel or built-in), choose a chart type,
map variables to aesthetics, and configure every aspect of the chart.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import StringIO
from scipy import stats as scipy_stats
from features.builtin_datasets import load_builtin_dataset, get_all_dataset_names, get_dataset_info


# ── Chart taxonomy: grouped by analytical objective ──────────────────────

CHART_CATEGORIES = [
    "Distribution",
    "Comparison",
    "Relationship",
    "Composition",
    "Multivariate",
    "Time Series",
    "Quality / SPC",
]

CHARTS_BY_CATEGORY = {
    "Distribution": [
        "Histogram", "Box Plot", "Violin Plot", "Density Plot",
        "ECDF Plot", "Q-Q Plot", "Ridgeline Plot", "Individual Value Plot",
    ],
    "Comparison": [
        "Bar Chart", "Stacked Bar Chart", "Dot Plot",
        "Funnel Chart", "Radar / Spider Chart",
        "Pareto Chart", "Interval Plot",
    ],
    "Relationship": [
        "Scatter Plot", "Line Plot", "Bubble Plot",
        "Scatterplot Matrix (SPLOM)", "Contour Plot",
        "3D Scatter Plot", "Hexbin Plot",
        "Marginal Plot", "Dual Y-axis Combo",
    ],
    "Composition": [
        "Pie Chart", "Area Chart",
        "Sunburst", "Treemap", "Waterfall Chart",
    ],
    "Multivariate": [
        "Heatmap", "Correlation Heatmap",
        "Parallel Coordinates", "Parallel Categories",
        "Correlogram",
    ],
    "Time Series": [
        "Time Series Plot", "Run Chart",
    ],
    "Quality / SPC": [
        "Pareto Chart",
    ],
}

GRAPH_TYPES = sum(CHARTS_BY_CATEGORY.values(), [])

SCALE_OPTIONS = ["linear", "log", "ln", "square root", "reversed"]
TICK_FORMATS = ["none", "comma", "percent", "scientific"]
LEGEND_POSITIONS = ["top", "bottom", "left", "right", "none"]
COLOR_PALETTES = {
    "Plotly": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3,
    "G10": px.colors.qualitative.G10,
    "T10": px.colors.qualitative.T10,
    "Alphabet": px.colors.qualitative.Alphabet,
    "Dark2": px.colors.qualitative.Dark2,
    "Pastel1": px.colors.qualitative.Pastel1,
    "Set1": px.colors.qualitative.Set1,
    "Set2": px.colors.qualitative.Set2,
    "Bold": px.colors.qualitative.Bold,
}
COLORSCALES = ["Viridis", "Plasma", "Inferno", "Magma", "RdBu", "Blues", "Greens", "YlOrRd", "Portland", "Electric"]

AGGREGATIONS = [None, "Mean", "Sum", "Count", "Median", "Min", "Max"]


def _format_ticks(fig, axis, fmt):
    tickformat = ""
    ticksuffix = ""
    if fmt == "comma":
        tickformat = ","
    elif fmt == "percent":
        ticksuffix = "%"
    elif fmt == "scientific":
        tickformat = ".2e"
    (fig.update_xaxes if axis == "x" else fig.update_yaxes)(tickformat=tickformat, ticksuffix=ticksuffix)


def _aggregate(df, x_col, y_col, agg, color_col=None):
    if agg is None or agg == "None":
        return df
    if agg == "Count":
        return df.groupby(x_col)[y_col].size().reset_index(name=y_col)
    if color_col:
        return df.groupby([x_col, color_col])[y_col].agg(agg.lower()).reset_index()
    return df.groupby(x_col)[y_col].agg(agg.lower()).reset_index()


def _make_chart(df, graph_type, mapping, opts):
    x_col = mapping.get("x") or None
    y_col = mapping.get("y") or None
    y2_col = mapping.get("y2") or None
    z_col = mapping.get("z") or None
    color_col = mapping.get("color") or None
    facet_col = mapping.get("facet") or None
    size_col = mapping.get("size") or None

    _REQUIRES = {
        "Scatter Plot": ("x", "y"), "Line Plot": ("x", "y"),
        "Bar Chart": ("x",), "Stacked Bar Chart": ("x", "y"),
        "Histogram": ("x",), "Box Plot": (), "Violin Plot": (),
        "Density Plot": ("x", "y"), "Area Chart": ("x", "y"),
        "Pie Chart": ("x",), "Heatmap": ("x", "y"),
        "ECDF Plot": ("x",), "Bubble Plot": ("x", "y"),
        "Scatterplot Matrix (SPLOM)": (), "Hexbin Plot": ("x", "y"),
        "Correlation Heatmap": (), "Q-Q Plot": ("x",),
        "Parallel Coordinates": (), "Contour Plot": ("x", "y"),
        "Dot Plot": ("x", "y"), "Ridgeline Plot": ("x", "y"),
        "3D Scatter Plot": ("x", "y", "z"),
        "Sunburst": ("x",), "Treemap": ("x",),
        "Radar / Spider Chart": (), "Funnel Chart": ("x", "y"),
        "Parallel Categories": (), "Waterfall Chart": ("x", "y"),
        # New types
        "Pareto Chart": ("x",),
        "Individual Value Plot": ("x", "y"),
        "Interval Plot": ("x", "y"),
        "Dual Y-axis Combo": ("x", "y", "y2"),
        "Marginal Plot": ("x", "y"),
        "Time Series Plot": ("x", "y"),
        "Run Chart": ("x", "y"),
        "Correlogram": (),
    }

    required = _REQUIRES.get(graph_type, ())
    missing = [{"x": "X variable", "y": "Y variable", "z": "Z variable", "y2": "Y2 variable"}.get(r, r) for r in required if not mapping.get(r)]
    if missing:
        fig = go.Figure()
        fig.add_annotation(text=f"Select {', '.join(missing)} to build chart", showarrow=False,
                           font=dict(size=16), x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    # Pre-aggregate data for chart types that support it
    agg = opts.get("aggregation")
    CAN_AGGREGATE = {"Bar Chart", "Line Plot", "Dot Plot", "Interval Plot"}
    if agg and agg != "None" and graph_type in CAN_AGGREGATE and x_col and y_col:
        df = _aggregate(df, x_col, y_col, agg, color_col)

    fig = None
    palette = COLOR_PALETTES.get(opts.get("palette", "Plotly"))

    try:
        if graph_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                             size=size_col, trendline="ols" if opts.get("regression_line") else None,
                             opacity=0.8, color_discrete_sequence=palette)
        elif graph_type == "Line Plot":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                          markers=opts.get("show_markers", True), color_discrete_sequence=palette)
        elif graph_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                         barmode="group", text_auto=opts.get("show_values", False),
                         color_discrete_sequence=palette)
        elif graph_type == "Stacked Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                         barmode="stack", text_auto=opts.get("show_values", False),
                         color_discrete_sequence=palette)
        elif graph_type == "Histogram":
            nb = opts.get("nbins", 30)
            marginal = opts.get("marginal", "none")
            bm = opts.get("barmode", "overlay")
            is_subplot = marginal != "none"
            if is_subplot:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.02, row_heights=[0.75, 0.25])
            else:
                fig = go.Figure()
            vals = df[x_col].dropna()
            if color_col:
                groups = df.groupby(color_col)[x_col]
                for i, (name, grp) in enumerate(groups):
                    c = palette[i % len(palette)] if palette else None
                    kw = dict(row=1, col=1) if is_subplot else {}
                    fig.add_trace(go.Histogram(x=grp.dropna(), nbinsx=nb, name=str(name),
                                               marker_color=c, opacity=0.75), **kw)
                    if is_subplot:
                        if marginal == "rug":
                            fig.add_trace(go.Scatter(x=grp.dropna(), y=[i]*len(grp.dropna()),
                                                     mode="markers", marker_color=c,
                                                     showlegend=False, name=str(name)),
                                          row=2, col=1)
                        elif marginal == "box":
                            fig.add_trace(go.Box(x=grp.dropna(), name=str(name),
                                                 marker_color=c, showlegend=False),
                                          row=2, col=1)
                        elif marginal == "violin":
                            fig.add_trace(go.Violin(x=grp.dropna(), name=str(name),
                                                    marker_color=c, showlegend=False),
                                          row=2, col=1)
                fig.update_layout(barmode=bm)
            else:
                kw = dict(row=1, col=1) if is_subplot else {}
                fig.add_trace(go.Histogram(x=vals, nbinsx=nb, marker_color=palette[0] if palette else None,
                                           opacity=0.75), **kw)
                if is_subplot:
                    if marginal == "rug":
                        fig.add_trace(go.Scatter(x=vals, y=[0]*len(vals), mode="markers",
                                                 marker_color=palette[0] if palette else None,
                                                 showlegend=False),
                                      row=2, col=1)
                    elif marginal == "box":
                        fig.add_trace(go.Box(x=vals, marker_color=palette[0] if palette else None,
                                             showlegend=False),
                                      row=2, col=1)
                    elif marginal == "violin":
                        fig.add_trace(go.Violin(x=vals, marker_color=palette[0] if palette else None,
                                                showlegend=False),
                                      row=2, col=1)
            if is_subplot:
                fig.update_xaxes(matches="x", row=2, col=1)

            # Distribution overlay (Normal / KDE)
            dist_overlay = opts.get("dist_overlay", "none")
            if dist_overlay != "none":
                xv = vals.values
                x_lin = np.linspace(xv.min(), xv.max(), 200)
                bin_width = np.ptp(xv) / nb
                if dist_overlay in ("normal", "both"):
                    n = scipy_stats.norm.pdf(x_lin, xv.mean(), xv.std())
                    fig.add_trace(go.Scatter(x=x_lin, y=n * len(xv) * bin_width, mode="lines",
                                             name="Normal", line=dict(color="red", width=2, dash="dash")))
                if dist_overlay in ("kde", "both"):
                    kde = scipy_stats.gaussian_kde(xv)
                    fig.add_trace(go.Scatter(x=x_lin, y=kde(x_lin) * len(xv) * bin_width, mode="lines",
                                             name="KDE", line=dict(color="orange", width=2)))
        elif graph_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                         points=opts.get("box_points", "outliers"), notched=opts.get("notched", False),
                         color_discrete_sequence=palette)
        elif graph_type == "Violin Plot":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                            box=opts.get("show_box", True), points=opts.get("violin_points", "outliers"),
                            color_discrete_sequence=palette)
        elif graph_type == "Density Plot":
            fig = px.density_contour(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                     color_discrete_sequence=palette)
        elif graph_type == "Area Chart":
            fig = px.area(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                          line_shape=opts.get("line_shape", "linear"), color_discrete_sequence=palette)
        elif graph_type == "Pie Chart":
            if y_col:
                vals = df.groupby(x_col)[y_col].sum().reset_index() if y_col != x_col else df
                fig = px.pie(vals, names=x_col, values=y_col, color_discrete_sequence=palette,
                             hole=opts.get("donut_hole", 0.0))
            else:
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, "count"]
                fig = px.pie(counts, names=x_col, values="count", color_discrete_sequence=palette,
                             hole=opts.get("donut_hole", 0.0))
        elif graph_type == "Heatmap":
            ct = pd.crosstab(df[x_col], df[y_col])
            fig = px.imshow(ct.values, x=ct.columns, y=ct.index,
                            color_continuous_scale=opts.get("colorscale", "Viridis"),
                            text_auto=opts.get("show_values", False), aspect="auto")
            fig.update_xaxes(title_text=y_col)
            fig.update_yaxes(title_text=x_col)
        elif graph_type == "ECDF Plot":
            fig = px.ecdf(df, x=x_col, color=color_col, facet_col=facet_col,
                          markers=opts.get("show_markers", False), color_discrete_sequence=palette)
        elif graph_type == "Bubble Plot":
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col or df[y_col],
                             color=color_col, facet_col=facet_col,
                             size_max=opts.get("bubble_max_size", 40),
                             opacity=0.7, color_discrete_sequence=palette)
        elif graph_type == "Scatterplot Matrix (SPLOM)":
            dims = opts.get("splom_dims", [x_col, y_col] if x_col and y_col else [])
            if len(dims) >= 2:
                fig = px.scatter_matrix(df, dimensions=dims, color=color_col,
                                        color_discrete_sequence=palette)
        elif graph_type == "Hexbin Plot":
            fig = px.density_heatmap(df, x=x_col, y=y_col, nbinsx=opts.get("hex_bins", 20),
                                     nbinsy=opts.get("hex_bins", 20),
                                     color_continuous_scale=opts.get("colorscale", "Viridis"))
        elif graph_type == "Correlation Heatmap":
            num_df = df.select_dtypes(include=["float64", "int64"])
            if num_df.shape[1] > 1:
                corr = num_df.corr()
                fig = px.imshow(corr.values, x=corr.columns, y=corr.index,
                                color_continuous_scale=opts.get("colorscale", "RdBu"),
                                text_auto=".2f", aspect="auto", zmin=-1, zmax=1)
        elif graph_type == "Q-Q Plot":
            vals = df[x_col].dropna().values
            theoretical = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, len(vals)), vals.mean(), vals.std())
            sorted_vals = np.sort(vals)
            r, _ = scipy_stats.pearsonr(theoretical, sorted_vals)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=theoretical, y=sorted_vals, mode="markers",
                                     marker=dict(color=palette[0] if palette else "#4C78A8", size=5),
                                     name="Observed"))
            lm = min(theoretical.min(), sorted_vals.min())
            lx = max(theoretical.max(), sorted_vals.max())
            fig.add_trace(go.Scatter(x=[lm, lx], y=[lm, lx], mode="lines", name="Expected",
                                     line=dict(color="red", dash="dash")))
            fig.update_layout(title=f"Q-Q Plot (r={r:.4f})")
        elif graph_type == "Parallel Coordinates":
            num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            dims_list = opts.get("pc_dims", num_cols[:6])
            if len(dims_list) >= 2:
                fig = px.parallel_coordinates(df, dimensions=dims_list, color=color_col or dims_list[0],
                                              color_continuous_scale=opts.get("colorscale", "Viridis"))
        elif graph_type == "Contour Plot":
            fig = px.density_contour(df, x=x_col, y=y_col, color=color_col, facet_col=facet_col,
                                     color_discrete_sequence=palette)
        elif graph_type == "Dot Plot":
            if y_col:
                agg = df.groupby(x_col)[y_col].mean().reset_index()
                err = df.groupby(x_col)[y_col].sem().fillna(0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=agg[y_col], y=agg[x_col], mode="markers",
                                         marker=dict(size=10, color=palette[0] if palette else "#4C78A8"),
                                         error_x=dict(type="data", array=err.values, visible=True)))
            else:
                agg = df[x_col].value_counts().reset_index()
                fig = px.scatter(agg, x=agg.columns[1], y=agg.columns[0], color_discrete_sequence=palette)
        elif graph_type == "Ridgeline Plot":
            if color_col:
                groups = df.groupby(color_col)
                fig = go.Figure()
                for i, (name, grp) in enumerate(groups):
                    vals = grp[y_col].dropna()
                    if len(vals) > 1:
                        kde = scipy_stats.gaussian_kde(vals)
                        xs = np.linspace(vals.min(), vals.max(), 200)
                        ys = kde(xs)
                        fig.add_trace(go.Scatter(x=xs, y=ys + i * ys.max() * 0.3, fill="tonexty",
                                                 name=str(name), mode="lines",
                                                 line=dict(color=palette[i % len(palette)] if palette else None)))
                fig.update_yaxes(showticklabels=False)
        elif graph_type == "3D Scatter Plot":
            opacity = max(0.0, min(1.0, opts.get("opacity_3d", 0.8)))
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                                size=size_col, opacity=opacity,
                                color_discrete_sequence=palette)
            fig.update_layout(scene=dict(
                xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col,
            ))
        elif graph_type == "Sunburst":
            path = [x_col]
            if color_col and color_col != x_col:
                path.append(color_col)
            fig = px.sunburst(df, path=path, values=size_col or None,
                              color=color_col if color_col and color_col not in path else None,
                              color_discrete_sequence=palette)
        elif graph_type == "Treemap":
            path = [x_col]
            if color_col and color_col != x_col:
                path.append(color_col)
            fig = px.treemap(df, path=path, values=size_col or None,
                             color=color_col if color_col and color_col not in path else None,
                             color_discrete_sequence=palette)
        elif graph_type == "Radar / Spider Chart":
            rad_vars = opts.get("radar_vars", [])
            if not rad_vars:
                rad_vars = [c for c in [x_col, y_col] if c]
            if len(rad_vars) >= 2:
                fig = go.Figure()
                if color_col:
                    for name, grp in df.groupby(color_col):
                        vals = [grp[v].mean() for v in rad_vars]
                        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]],
                                                      theta=rad_vars + [rad_vars[0]],
                                                      fill="toself", name=str(name)))
                else:
                    vals = [df[v].mean() for v in rad_vars]
                    fig.add_trace(go.Scatterpolar(r=vals + [vals[0]],
                                                  theta=rad_vars + [rad_vars[0]],
                                                  fill="toself"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
        elif graph_type == "Funnel Chart":
            fig = px.funnel(df, x=x_col, y=y_col, color=color_col,
                            color_discrete_sequence=palette)
        elif graph_type == "Parallel Categories":
            dims_list = opts.get("pc_dims", [])
            if not dims_list:
                dims_list = [c for c in [x_col, y_col, color_col] if c]
            if len(dims_list) >= 2:
                color_cont = color_col if color_col and color_col in df.select_dtypes(include=["int64", "float64"]).columns else None
                fig = px.parallel_categories(df, dimensions=dims_list, color=color_cont,
                                             color_continuous_scale=opts.get("colorscale", "Viridis"))
        elif graph_type == "Waterfall Chart":
            agg = df.groupby(x_col)[y_col].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Waterfall(
                name="", orientation="v",
                measure=["relative"] * (len(agg) - 1) + ["total"],
                x=agg[x_col].tolist(),
                y=agg[y_col].tolist(),
                connector=dict(line=dict(color="rgb(63, 63, 63)")),
            ))
            fig.update_layout(title=f"Waterfall: {y_col} by {x_col}")

        # ═══════════════ NEW CHART TYPES ═══════════════

        elif graph_type == "Pareto Chart":
            if y_col:
                agg = df.groupby(x_col)[y_col].sum().reset_index()
                agg.columns = [x_col, "value"]
            else:
                agg = df[x_col].value_counts().reset_index()
                agg.columns = [x_col, "value"]
            agg = agg.sort_values("value", ascending=False).reset_index(drop=True)
            agg["cum_pct"] = agg["value"].cumsum() / agg["value"].sum() * 100
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=agg[x_col], y=agg["value"], name="Count",
                                 marker_color=palette[0] if palette else "#4C78A8"), secondary_y=False)
            fig.add_trace(go.Scatter(x=agg[x_col], y=agg["cum_pct"], name="Cumulative %",
                                     mode="lines+markers", line=dict(color="red", width=2),
                                     marker=dict(size=6)), secondary_y=True)
            fig.update_yaxes(title_text="Count", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])

        elif graph_type == "Individual Value Plot":
            fig = go.Figure()
            rng = np.random.default_rng(42)
            if color_col:
                for name, grp in df.groupby(color_col):
                    yv = grp[y_col].dropna()
                    x_pos = np.full(len(yv), float(len(fig.data)), dtype=float) + rng.uniform(-0.3, 0.3, len(yv))
                    fig.add_trace(go.Scatter(x=x_pos, y=yv, mode="markers",
                                             name=str(name),
                                             marker=dict(color=palette[len(fig.data) % len(palette)] if palette else None,
                                                         size=5, opacity=0.7)))
            else:
                x_pos = np.ones(len(df)) + np.random.default_rng(42).uniform(-0.3, 0.3, len(df))
                fig.add_trace(go.Scatter(x=x_pos, y=df[y_col].dropna(),
                                         mode="markers",
                                         marker=dict(color=palette[0] if palette else "#4C78A8",
                                                     size=5, opacity=0.7)))
            fig.update_xaxes(showticklabels=False)

        elif graph_type == "Interval Plot":
            fig = go.Figure()
            if color_col:
                groups = df.groupby([x_col, color_col])[y_col]
                for (x_val, c_val), grp in groups:
                    yv = grp.dropna()
                    if len(yv) > 1:
                        m, se = yv.mean(), yv.std() / np.sqrt(len(yv))
                        ci = se * scipy_stats.t.ppf(0.975, len(yv) - 1)
                        ci = 0 if np.isnan(ci) else ci
                        fig.add_trace(go.Scatter(
                            x=[str(x_val)], y=[m],
                            error_y=dict(type="data", array=[ci], visible=True),
                            mode="markers", name=str(c_val),
                            marker=dict(size=8, color=palette[len(fig.data) % len(palette)] if palette else None),
                            legendgroup=str(c_val),
                        ))
            else:
                for x_val in sorted(df[x_col].unique()):
                    yv = df.loc[df[x_col] == x_val, y_col].dropna()
                    if len(yv) > 1:
                        m, se = yv.mean(), yv.std() / np.sqrt(len(yv))
                        ci = se * scipy_stats.t.ppf(0.975, len(yv) - 1)
                        ci = 0 if np.isnan(ci) else ci
                        fig.add_trace(go.Scatter(
                            x=[str(x_val)], y=[m],
                            error_y=dict(type="data", array=[ci], visible=True),
                            mode="markers", name=str(x_val),
                            marker=dict(size=8, color=palette[len(fig.data) % len(palette)] if palette else None),
                        ))
            fig.update_layout(showlegend=color_col is not None)

        elif graph_type == "Dual Y-axis Combo":
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if color_col:
                for name, grp in df.groupby(color_col):
                    fig.add_trace(go.Bar(x=grp[x_col], y=grp[y_col], name=f"{y_col} ({name})",
                                         marker_color=palette[0] if palette else None),
                                  secondary_y=False)
                    fig.add_trace(go.Scatter(x=grp[x_col], y=grp[y2_col], mode="lines+markers",
                                             name=f"{y2_col} ({name})",
                                             marker_color=palette[1] if palette else None,
                                             line=dict(color=palette[1] if palette else None)),
                                  secondary_y=True)
            else:
                fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], name=y_col,
                                     marker_color=palette[0] if palette else "#4C78A8"),
                              secondary_y=False)
                fig.add_trace(go.Scatter(x=df[x_col], y=df[y2_col], mode="lines+markers",
                                         name=y2_col,
                                         marker_color=palette[1] if palette else "#E45756",
                                         line=dict(color=palette[1] if palette else "#E45756")),
                              secondary_y=True)
            fig.update_yaxes(title_text=y_col, secondary_y=False)
            fig.update_yaxes(title_text=y2_col, secondary_y=True)

        elif graph_type == "Marginal Plot":
            marginal = opts.get("marginal", "histogram")
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             marginal_x=marginal, marginal_y=marginal,
                             opacity=0.7, color_discrete_sequence=palette)

        elif graph_type == "Time Series Plot":
            fig = go.Figure()
            try:
                time_vals = pd.to_datetime(df[x_col])
            except Exception:
                time_vals = df[x_col]
            if color_col:
                for name, grp in df.groupby(color_col):
                    idx = grp.index
                    fig.add_trace(go.Scatter(x=time_vals.loc[idx], y=grp[y_col],
                                             mode="lines+markers" if opts.get("show_markers", False) else "lines",
                                             name=str(name),
                                             line=dict(color=palette[len(fig.data) % len(palette)] if palette else None)))
            else:
                fig.add_trace(go.Scatter(x=time_vals, y=df[y_col],
                                         mode="lines+markers" if opts.get("show_markers", False) else "lines",
                                         marker_color=palette[0] if palette else None,
                                         line=dict(color=palette[0] if palette else "#4C78A8")))
            fig.update_xaxes(title_text=x_col)

        elif graph_type == "Run Chart":
            fig = go.Figure()
            med = df[y_col].median()
            order = df[x_col] if x_col and x_col in df.columns else df.index
            if color_col:
                for name, grp in df.groupby(color_col):
                    idx = grp.index
                    fig.add_trace(go.Scatter(x=order.loc[idx], y=grp[y_col], mode="lines+markers",
                                             name=str(name),
                                             line=dict(color=palette[len(fig.data) % len(palette)] if palette else None)))
            else:
                fig.add_trace(go.Scatter(x=order, y=df[y_col], mode="lines+markers",
                                         marker_color=palette[0] if palette else None,
                                         line=dict(color=palette[0] if palette else "#4C78A8")))
            fig.add_hline(y=med, line_dash="dash", line_color="red",
                          annotation_text=f"Median={med:.2f}")
            fig.update_xaxes(title_text=x_col or "Order")

        elif graph_type == "Correlogram":
            num_df = df.select_dtypes(include=["float64", "int64"])
            if num_df.shape[1] > 1:
                corr = num_df.corr()
                names = corr.columns
                fig = go.Figure()
                for i, r in enumerate(names):
                    for j, c in enumerate(names):
                        if i == j:
                            continue
                        r_val = corr.iloc[i, j]
                        size = max(4, abs(r_val) * 30)
                        color = "red" if r_val < 0 else "#4C78A8"
                        fig.add_trace(go.Scatter(
                            x=[j], y=[i], mode="markers",
                            marker=dict(size=size, color=color, opacity=0.7,
                                        line=dict(width=1, color="white")),
                            text=f"{r} vs {c}: {r_val:.2f}",
                            hoverinfo="text", showlegend=False,
                        ))
                fig.update_xaxes(tickvals=list(range(len(names))), ticktext=names)
                fig.update_yaxes(tickvals=list(range(len(names))), ticktext=names,
                                 autorange="reversed")
                fig.update_layout(height=500,
                                  title="Correlogram")
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Chart error: {e}", showarrow=False,
                           font=dict(size=14), x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    if fig is None:
        fig = go.Figure()
        fig.add_annotation(text="Select variables to build chart", showarrow=False,
                           font=dict(size=16), x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    # ---- Apply style options ----
    subtitle = opts.get("subtitle", "")
    full_title = f"{opts.get('title', '')}<br><sup>{subtitle}</sup>" if subtitle else opts.get("title", "")

    fig.update_layout(
        title=full_title or None,
        xaxis_title=opts.get("x_title", x_col or ""),
        yaxis_title=opts.get("y_title", y_col or ""),
        template="plotly_dark", height=550,
        margin=dict(l=60, r=30, t=60, b=60),
        hovermode="x unified" if graph_type in ("Line Plot", "Area Chart", "ECDF Plot", "Run Chart") else "closest",
        showlegend=opts.get("legend_position", "top") != "none",
        legend=dict(
            title=opts.get("legend_title", ""),
            orientation="h" if opts.get("legend_position") in ("top", "bottom") else "v",
            y=1.12 if opts.get("legend_position") == "top" else -0.15 if opts.get("legend_position") == "bottom" else 1,
            x=0.5 if opts.get("legend_position") in ("top", "bottom") else 1.02,
            xanchor="center" if opts.get("legend_position") in ("top", "bottom") else "left",
            yanchor="bottom" if opts.get("legend_position") == "top" else "top",
        ),
    )

    # Axis range
    xr, yr = opts.get("x_range"), opts.get("y_range")
    if xr and len(xr) == 2: fig.update_xaxes(range=list(xr))
    if yr and len(yr) == 2: fig.update_yaxes(range=list(yr))

    # Scale
    xs, ys = opts.get("x_scale"), opts.get("y_scale")
    if xs == "log": fig.update_xaxes(type="log")
    elif xs == "reversed": fig.update_xaxes(autorange="reversed")
    if ys == "log": fig.update_yaxes(type="log")
    elif ys == "reversed": fig.update_yaxes(autorange="reversed")

    _format_ticks(fig, "x", opts.get("x_tick_format", "none"))
    _format_ticks(fig, "y", opts.get("y_tick_format", "none"))

    if not opts.get("show_grid", True):
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    if opts.get("zero_baseline", False):
        fig.update_xaxes(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)")
        fig.update_yaxes(zeroline=True, zerolinecolor="rgba(255,255,255,0.3)")

    # ---- Overlays ----
    if opts.get("error_bars") and y_col and graph_type in ("Scatter Plot", "Line Plot", "Bar Chart"):
        y_vals = df[y_col].dropna()
        if len(y_vals) > 1:
            se = y_vals.std() / np.sqrt(len(y_vals))
            fig.update_traces(error_y=dict(type="data", array=[se] * len(y_vals), visible=True))

    if opts.get("regression_line") and graph_type == "Scatter Plot" and x_col and y_col:
        clean = df[[x_col, y_col]].dropna()
        if len(clean) > 1:
            xv, yv = clean[x_col].values, clean[y_col].values
            slope, intercept, r_val, _, _ = scipy_stats.linregress(xv, yv)
            x_line = np.linspace(xv.min(), xv.max(), 100)
            fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                                     name=f"Regression (R\u00b2={r_val**2:.3f})",
                                     line=dict(color="red", width=2, dash="dash")))

    if opts.get("trend_line") and x_col and y_col:
        from numpy.polynomial.polynomial import polyfit
        clean = df[[x_col, y_col]].dropna()
        if len(clean) > 3:
            coeffs = polyfit(clean[x_col], clean[y_col], opts.get("trend_degree", 2))
            x_smooth = np.linspace(clean[x_col].min(), clean[x_col].max(), 200)
            fig.add_trace(go.Scatter(x=x_smooth, y=sum(c * x_smooth ** i for i, c in enumerate(coeffs)),
                                     mode="lines", name=f"Trend (deg {opts.get('trend_degree', 2)})",
                                     line=dict(color="orange", width=2)))

    if opts.get("ref_line_x") is not None and isinstance(opts["ref_line_x"], (int, float)):
        try:
            fig.add_vline(x=opts["ref_line_x"], line_dash="dash", line_color="cyan",
                          annotation_text=f"x={opts['ref_line_x']}")
        except ValueError:
            pass
    if opts.get("ref_line_y") is not None and isinstance(opts["ref_line_y"], (int, float)):
        try:
            fig.add_hline(y=opts["ref_line_y"], line_dash="dash", line_color="cyan",
                          annotation_text=f"y={opts['ref_line_y']}")
        except ValueError:
            pass

    if opts.get("mean_line"):
        use_col = y_col if y_col is not None else x_col
        if use_col and pd.api.types.is_numeric_dtype(df[use_col]):
            mv = df[use_col].dropna().mean()
            dir_ = opts.get("mean_line_dir", "auto")
            if dir_ == "auto":
                use_h = y_col is not None
            else:
                use_h = dir_ == "horizontal"
            try:
                if use_h:
                    fig.add_hline(y=mv, line_dash="dot", line_color="lime", annotation_text=f"Mean={mv:.2f}")
                else:
                    fig.add_vline(x=mv, line_dash="dot", line_color="lime", annotation_text=f"Mean={mv:.2f}")
            except ValueError:
                pass
    if opts.get("median_line"):
        use_col = y_col if y_col is not None else x_col
        if use_col and pd.api.types.is_numeric_dtype(df[use_col]):
            mv = df[use_col].dropna().median()
            dir_ = opts.get("median_line_dir", "auto")
            if dir_ == "auto":
                use_h = y_col is not None
            else:
                use_h = dir_ == "horizontal"
            try:
                if use_h:
                    fig.add_hline(y=mv, line_dash="dot", line_color="magenta", annotation_text=f"Median={mv:.2f}")
                else:
                    fig.add_vline(x=mv, line_dash="dot", line_color="magenta", annotation_text=f"Median={mv:.2f}")
            except ValueError:
                pass

    if opts.get("highlight_groups") and color_col and opts.get("highlight_vals"):
        for trace in fig.data:
            if trace.name in opts["highlight_vals"]:
                trace.update(marker=dict(size=10, line=dict(width=2, color="yellow")))

    return fig


def render_graph_builder():
    st.title("Graph Builder")
    st.caption("Import data, pick a chart type, map variables, and customize every aspect.")

    # ==================== STATE INIT ====================
    for k in ("gb_df", "gb_df_name", "gb_category"):
        if k not in st.session_state:
            st.session_state[k] = None if k == "gb_df" else ("" if k == "gb_df_name" else "Distribution")

    # ==================== DATA SOURCE ====================
    with st.expander("1. Data Source", expanded=st.session_state.gb_df is None):
        src_mode = st.radio("Source", ["Upload CSV/Excel", "Use built-in dataset"],
                            key="gb_src_mode", horizontal=True)
        df = None
        if src_mode == "Upload CSV/Excel":
            uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], key="gb_file")
            if uploaded:
                try:
                    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                except Exception as e:
                    st.error(f"Error loading: {e}")
        else:
            selected = st.selectbox("Choose dataset", [""] + get_all_dataset_names(), key="gb_builtin")
            if selected:
                df = load_builtin_dataset(selected)
                info = get_dataset_info(selected) or {}
                if "description" in info:
                    st.caption(info["description"])

        if df is not None and not df.empty:
            st.session_state.gb_df = df
            st.session_state.gb_df_name = selected if src_mode == "Use built-in dataset" else uploaded.name
            st.success(f"Loaded {len(df)} rows \u00d7 {len(df.columns)} columns")
        elif df is not None and df.empty:
            st.warning("File is empty.")

    df = st.session_state.gb_df
    if df is None:
        st.info("Load a dataset above to begin.")
        return

    # ==================== DATA PREVIEW (collapsed) ====================
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)
        st.caption(f"{len(df)} rows, {len(df.columns)} columns \u2014 {st.session_state.gb_df_name}")
        col_types = pd.DataFrame({"Column": df.columns, "Type": [str(df[c].dtype) for c in df.columns],
                                   "Non-null": [df[c].notna().sum() for c in df.columns]})
        st.dataframe(col_types, use_container_width=True, hide_index=True)

    # ==================== COLUMN CATEGORIES ====================
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    # ==================== CHART TYPE SELECTOR (grouped by objective) ====================
    st.markdown("### Chart Type")
    cat_col1, cat_col2 = st.columns([1, 2])
    with cat_col1:
        cat = st.selectbox("Category", CHART_CATEGORIES, key="gb_category")
    with cat_col2:
        group_types = CHARTS_BY_CATEGORY[cat]
        graph_type = st.selectbox("Chart Type", group_types, key="gb_type")

    # ==================== 3-COLUMN CONTROLS ====================
    opts = {}
    with st.expander("2. Map Variables", expanded=True):
        c1, c2, c3 = st.columns(3)

    # ---- COLUMN 1: Variable Mapping ----
    with c1:
        st.markdown("### Variables")

        def col_opts(gt):
            if gt in ("Box Plot", "Violin Plot", "Ridgeline Plot", "Dot Plot", "Individual Value Plot", "Interval Plot"):
                return categorical_cols or all_cols
            if gt in ("Bar Chart", "Stacked Bar Chart", "Pie Chart", "Heatmap",
                      "Sunburst", "Treemap", "Funnel Chart", "Waterfall Chart",
                      "Pareto Chart"):
                return categorical_cols or all_cols
            if gt in ("Histogram", "ECDF Plot", "Q-Q Plot"):
                return numeric_cols
            if gt in ("Time Series Plot", "Run Chart"):
                return all_cols
            return numeric_cols

        mapping = {}

        x_opts = col_opts(graph_type)
        mapping["x"] = st.selectbox("X variable", [""] + x_opts, key="gb_x")

        needs_y = graph_type not in ("Histogram", "ECDF Plot", "Q-Q Plot", "Sunburst", "Treemap",
                                     "Correlogram", "Pareto Chart")
        if needs_y:
            y_opts = numeric_cols if graph_type not in ("Heatmap", "Dual Y-axis Combo") else all_cols
            mapping["y"] = st.selectbox("Y variable", [""] + y_opts, key="gb_y")

        # Dual Y-axis second Y
        if graph_type == "Dual Y-axis Combo":
            y2_opts = [c for c in numeric_cols if c != mapping.get("y")]
            mapping["y2"] = st.selectbox("Y2 variable (line)", [""] + y2_opts, key="gb_y2")

        # 3D scatter Z
        if graph_type == "3D Scatter Plot":
            z_opts = [c for c in numeric_cols if c not in (mapping.get("x"), mapping.get("y"))]
            mapping["z"] = st.selectbox("Z variable", [""] + z_opts, key="gb_z")

        # Color (hidden for some types)
        if graph_type not in ("Correlation Heatmap", "Correlogram"):
            mapping["color"] = st.selectbox("Color / Group", [""] + (categorical_cols or all_cols), key="gb_color")

        # Facet (hidden for some types)
        if graph_type not in ("Heatmap", "Correlation Heatmap", "Parallel Coordinates",
                              "Parallel Categories", "Radar / Spider Chart", "Sunburst", "Treemap",
                              "Correlogram", "Dual Y-axis Combo", "Marginal Plot",
                              "Time Series Plot", "Run Chart", "Pareto Chart"):
            facet_opts = categorical_cols
            mapping["facet"] = st.selectbox("Facet / Subplot", [""] + facet_opts, key="gb_facet")

        # Size (for applicable types)
        if graph_type in ("Scatter Plot", "Bubble Plot", "3D Scatter Plot", "Sunburst", "Treemap"):
            mapping["size"] = st.selectbox("Size variable", [""] + numeric_cols, key="gb_size")

        # SPLOM dimensions
        if graph_type == "Scatterplot Matrix (SPLOM)":
            splom_defaults = numeric_cols[:min(4, len(numeric_cols))]
            opts["splom_dims"] = st.multiselect("Dimensions", numeric_cols, default=splom_defaults, key="gb_splom_dims")

        # Parallel Coordinates dimensions
        if graph_type == "Parallel Coordinates":
            pc_defaults = numeric_cols[:min(6, len(numeric_cols))]
            opts["pc_dims"] = st.multiselect("Dimensions", all_cols, default=pc_defaults, key="gb_pc_dims")

        # Parallel Categories dimensions
        if graph_type == "Parallel Categories":
            cat_dims = categorical_cols or all_cols
            pc_defaults = cat_dims[:min(4, len(cat_dims))]
            opts["pc_dims"] = st.multiselect("Dimensions (categorical)", cat_dims, default=pc_defaults, key="gb_parcat_dims")

        # Radar variables
        if graph_type == "Radar / Spider Chart":
            radar_defaults = numeric_cols[:min(4, len(numeric_cols))]
            opts["radar_vars"] = st.multiselect("Radar variables (numeric)", numeric_cols,
                                                default=radar_defaults, key="gb_radar_vars")

        # Correlogram
        if graph_type == "Correlogram":
            corr_defaults = numeric_cols[:min(6, len(numeric_cols))]
            opts["corr_dims"] = st.multiselect("Numeric variables", numeric_cols,
                                               default=corr_defaults, key="gb_corr_dims")

        # Aggregation (for applicable chart types)
        CAN_AGGREGATE = {"Bar Chart", "Line Plot", "Dot Plot", "Interval Plot"}
        if graph_type in CAN_AGGREGATE and mapping.get("x") and mapping.get("y"):
            x_is_cat = mapping["x"] in categorical_cols
            if x_is_cat:
                st.markdown("### Aggregation")
                opts["aggregation"] = st.selectbox("Y-axis statistic",
                                                   [a if a else "None" for a in AGGREGATIONS],
                                                   key="gb_agg")
                if opts["aggregation"] == "None":
                    opts["aggregation"] = None

        st.markdown("### Axis")
        opts["x_scale"] = st.selectbox("X scale", SCALE_OPTIONS, key="gb_x_scale")
        opts["y_scale"] = st.selectbox("Y scale", SCALE_OPTIONS, key="gb_y_scale")

        sx1, sx2 = st.columns(2)
        with sx1:
            x_min = st.text_input("X min", key="gb_x_min", placeholder="auto")
            x_max = st.text_input("X max", key="gb_x_max", placeholder="auto")
        with sx2:
            y_min = st.text_input("Y min", key="gb_y_min", placeholder="auto")
            y_max = st.text_input("Y max", key="gb_y_max", placeholder="auto")

        try:
            opts["x_range"] = [float(x_min), float(x_max)] if x_min and x_max else None
            opts["y_range"] = [float(y_min), float(y_max)] if y_min and y_max else None
        except ValueError:
            opts["x_range"] = opts["y_range"] = None

        opts["x_tick_format"] = st.selectbox("X tick format", TICK_FORMATS, key="gb_x_tick")
        opts["y_tick_format"] = st.selectbox("Y tick format", TICK_FORMATS, key="gb_y_tick")
        opts["show_grid"] = st.toggle("Gridlines", True, key="gb_grid")
        opts["zero_baseline"] = st.toggle("Zero baseline", False, key="gb_zero")

    # ---- COLUMN 2: Titles, Axis, Colors ----
    with c2:
        st.markdown("### Labels")
        opts["title"] = st.text_input("Chart title", key="gb_title")
        opts["subtitle"] = st.text_input("Subtitle", key="gb_subtitle")
        opts["caption"] = st.text_input("Caption", key="gb_caption")
        opts["formula"] = st.text_input("Formula / Note", key="gb_formula")

        x_label = mapping.get("x", "") or "X"
        y_label = mapping.get("y", "") or "Y"
        opts["x_title"] = st.text_input("X-axis title", value=x_label, key="gb_x_title")
        opts["y_title"] = st.text_input("Y-axis title", value=y_label, key="gb_y_title")

        st.markdown("### Legend")
        opts["legend_position"] = st.selectbox("Position", LEGEND_POSITIONS, key="gb_legend_pos")
        opts["legend_title"] = st.text_input("Legend title", key="gb_legend_title")

        st.markdown("### Colors")
        opts["palette"] = st.selectbox("Color palette", list(COLOR_PALETTES.keys()), key="gb_palette")
        if graph_type in ("Hexbin Plot", "Heatmap", "Correlation Heatmap", "Parallel Coordinates", "Parallel Categories"):
            opts["colorscale"] = st.selectbox("Color scale", COLORSCALES, key="gb_cscale")

    # ---- COLUMN 3: Overlays & Type-specific ----
    with c3:
        st.markdown("### Overlays")
        opts["regression_line"] = st.checkbox("Regression line", graph_type == "Scatter Plot", key="gb_reg")
        opts["trend_line"] = st.checkbox("Polynomial trend", False, key="gb_trend")
        if opts["trend_line"]:
            opts["trend_degree"] = st.slider("Degree", 1, 6, 2, key="gb_trend_deg")
        opts["error_bars"] = st.checkbox("Error bars (SE)", False, key="gb_err")
        opts["ref_line_x"] = st.number_input("Reference line X", value=None, key="gb_ref_x")
        opts["ref_line_y"] = st.number_input("Reference line Y", value=None, key="gb_ref_y")
        opts["mean_line"] = st.checkbox("Mean line", False, key="gb_mean")
        if opts["mean_line"]:
            opts["mean_line_dir"] = st.selectbox("Mean line direction", ["auto", "horizontal", "vertical"],
                                                  key="gb_mean_dir")
        opts["median_line"] = st.checkbox("Median line", False, key="gb_median")
        if opts["median_line"]:
            opts["median_line_dir"] = st.selectbox("Median line direction", ["auto", "horizontal", "vertical"],
                                                    key="gb_median_dir")

        if mapping.get("color"):
            highlight_opts = df[mapping["color"]].unique().tolist() if mapping["color"] in df.columns else []
            hv = st.multiselect("Highlight groups", highlight_opts, key="gb_highlight")
            opts["highlight_groups"] = len(hv) > 0
            opts["highlight_vals"] = hv

        st.markdown("### Type Options")
        # Bar chart types
        if graph_type in ("Bar Chart", "Stacked Bar Chart"):
            opts["show_values"] = st.checkbox("Show values on bars", False, key="gb_bar_vals")
        # Histogram
        elif graph_type == "Histogram":
            opts["nbins"] = st.slider("Number of bins", 5, 100, 30, key="gb_nbins")
            opts["marginal"] = st.selectbox("Marginal plot", ["none", "rug", "box", "violin"], key="gb_marginal")
            opts["barmode"] = st.selectbox("Bar mode", ["overlay", "stack", "group"], key="gb_hist_mode")
            opts["dist_overlay"] = st.selectbox("Distribution overlay",
                                                ["none", "normal", "kde", "both"], key="gb_dist_overlay")
        # Box Plot
        elif graph_type == "Box Plot":
            opts["box_points"] = st.selectbox("Show points", ["all", "outliers", "suspectedoutliers", False], key="gb_box_pts")
            opts["notched"] = st.checkbox("Notched boxes", False, key="gb_notched")
        # Violin Plot
        elif graph_type == "Violin Plot":
            opts["show_box"] = st.checkbox("Show inner box", True, key="gb_violin_box")
            opts["violin_points"] = st.selectbox("Show points", ["all", "outliers", False], key="gb_violin_pts")
        # Pie Chart
        elif graph_type == "Pie Chart":
            opts["donut_hole"] = st.slider("Donut hole", 0.0, 0.8, 0.0, 0.05, key="gb_donut")
        # Heatmap
        elif graph_type in ("Heatmap",):
            opts["show_values"] = st.checkbox("Show values in cells", False, key="gb_heat_vals")
        # Line Plot
        elif graph_type == "Line Plot":
            opts["show_markers"] = st.checkbox("Show markers", True, key="gb_line_markers")
        # Area Chart
        elif graph_type == "Area Chart":
            opts["line_shape"] = st.selectbox("Line shape", ["linear", "spline", "hv", "vh", "hvh", "vhv"], key="gb_area_shape")
        # ECDF Plot
        elif graph_type == "ECDF Plot":
            opts["show_markers"] = st.checkbox("Show markers", False, key="gb_ecdf_markers")
        # Bubble Plot
        elif graph_type == "Bubble Plot":
            opts["bubble_max_size"] = st.slider("Max bubble size", 10, 80, 40, key="gb_bubble_size")
        # Hexbin Plot
        elif graph_type == "Hexbin Plot":
            opts["hex_bins"] = st.slider("Number of bins", 5, 50, 20, key="gb_hex_bins")
        # 3D Scatter
        elif graph_type == "3D Scatter Plot":
            opts["opacity_3d"] = st.slider("Point opacity", 0.0, 1.0, 0.8, 0.05, key="gb_opacity_3d")
        # Marginal Plot
        elif graph_type == "Marginal Plot":
            opts["marginal"] = st.selectbox("Marginal type", ["histogram", "box", "violin", "rug"], key="gb_marginal_mp")
        # Time Series / Run Chart
        elif graph_type in ("Time Series Plot", "Run Chart"):
            opts["show_markers"] = st.checkbox("Show markers", False, key="gb_ts_markers")

    # ==================== BUILD & DISPLAY CHART ====================
    fig = _make_chart(df, graph_type, mapping, opts)
    st.plotly_chart(fig, use_container_width=True)

    caption = opts.get("caption", "")
    if caption:
        st.markdown(caption)

    formula = opts.get("formula", "")
    if formula:
        st.latex(formula)

    # ==================== DOWNLOAD & CODE ====================
    dl, code_col = st.columns(2)
    with dl:
        with st.expander("Download Chart", expanded=False):
            fmt_img = st.selectbox("Image format", ["png", "svg", "jpeg", "html"], key="gb_dl_fmt")
            scale = st.slider("Scale", 1, 4, 2, key="gb_dl_scale")
            if fmt_img == "html":
                st.download_button("Download HTML", fig.to_html(include_plotlyjs="cdn"),
                                   "chart.html", "text/html", key="gb_dl_html")
            else:
                st.download_button(f"Download {fmt_img.upper()}", fig.to_image(format=fmt_img, scale=scale),
                                   f"chart.{fmt_img}", f"image/{fmt_img}", key="gb_dl_img")

    with code_col:
        with st.expander("Show Python Code", expanded=False):
            st.code(
                f"""import plotly.express as px
import pandas as pd

df = pd.read_csv("{st.session_state.gb_df_name}")
fig = px.{graph_type.lower().replace(' ', '_').replace('/', '_').replace('-', '_')}(
    df, x={mapping.get('x')!r}, y={mapping.get('y')!r},
    color={mapping.get('color')!r},
    title={opts.get('title')!r},
)
fig.update_layout(template="plotly_dark")
fig.show()""",
                language="python",
            )
