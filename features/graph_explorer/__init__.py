"""Graph Explorer package.

Each category module exports a GRAPHS dict mapping name -> widget_function.
This module merges them and provides the public API.
"""

import streamlit as st

from features.graph_builder import render_graph_builder

from .distribution import GRAPHS as _distribution_graphs
from .comparison import GRAPHS as _comparison_graphs
from .correlation import GRAPHS as _correlation_graphs
from .regression import GRAPHS as _regression_graphs
from .diagnostic import GRAPHS as _diagnostic_graphs
from .agreement import GRAPHS as _agreement_graphs
from .multivariate import GRAPHS as _multivariate_graphs
from .survival import GRAPHS as _survival_graphs
from .meta_analysis import GRAPHS as _meta_analysis_graphs
from .post_hoc import GRAPHS as _post_hoc_graphs

_CATEGORY_MODULES = [
    ("Distribution Plots", _distribution_graphs),
    ("Comparison Plots", _comparison_graphs),
    ("Correlation Plots", _correlation_graphs),
    ("Regression Plots", _regression_graphs),
    ("Diagnostic Accuracy Plots", _diagnostic_graphs),
    ("Agreement Plots", _agreement_graphs),
    ("Multivariate Plots", _multivariate_graphs),
    ("Survival Analysis Plots", _survival_graphs),
    ("Meta-Analysis Visualizations", _meta_analysis_graphs),
    ("Post-Hoc Plots", _post_hoc_graphs),
]

CATEGORIES = [c for c, _ in _CATEGORY_MODULES]

graphs = {}
for category, module_graphs in _CATEGORY_MODULES:
    for name, func in module_graphs.items():
        graphs[name] = {"widget_function": func, "category": category}

BUILDER_MODE = "Graph Builder"


def render_graph_explorer():
    if "graph_mode" not in st.session_state:
        st.session_state.graph_mode = BUILDER_MODE

    st.sidebar.markdown("### Mode")

    if st.sidebar.button("Graph Builder", key="mode_builder", use_container_width=True,
                         type="primary" if st.session_state.graph_mode == BUILDER_MODE else "secondary"):
        st.session_state.graph_mode = BUILDER_MODE
        st.rerun()

    st.sidebar.markdown("**Pre-built Graphs**")
    for cat in CATEGORIES:
        if st.sidebar.button(cat, key=f"graph_cat_{cat}", use_container_width=True):
            st.session_state.graph_mode = cat
            st.rerun()

    mode = st.session_state.graph_mode

    if mode == BUILDER_MODE:
        return render_graph_builder()

    category_graphs = {
        k: v for k, v in graphs.items() if v.get("category") == mode
    }

    st.header(f"{mode}", divider="orange")

    for name, info in category_graphs.items():
        with st.expander(f"**{name}**", expanded=True):
            info["widget_function"]()
