"""Robust extraction of graph_explorer.py into per-category modules."""

import re
import os
import ast

SHARED_IMPORTS = """import math
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
"""

CATEGORY_MAP = {
    "Distribution Plots": "distribution",
    "Comparison Plots": "comparison",
    "Correlation Plots": "correlation",
    "Regression Plots": "regression",
    "Diagnostic Accuracy Plots": "diagnostic",
    "Agreement Plots": "agreement",
    "Multivariate Plots": "multivariate",
    "Survival Analysis Plots": "survival",
    "Meta-Analysis Visualizations": "meta_analysis",
    "Post-Hoc Plots": "post_hoc",
}


def main():
    source_path = os.path.join(os.path.dirname(__file__), "features", "graph_explorer.py")
    output_dir = os.path.join(os.path.dirname(__file__), "features", "graph_explorer")
    os.makedirs(output_dir, exist_ok=True)

    with open(source_path, encoding="utf-8") as f:
        source_lines = f.readlines()

    source_lines = [l.rstrip("\n").rstrip("\r") for l in source_lines]

    # Step 1: Extract graph metadata from the graphs dict
    graphs_start = None
    graphs_end = None
    for i, line in enumerate(source_lines):
        if line.strip() == "graphs = {":
            graphs_start = i
        if graphs_start is not None and line.strip() == "}":
            graphs_end = i
            break

    # Parse graphs dict to get name -> category mapping
    name_pattern = re.compile(r'^\s*"(.+?)":\s*\{')
    cat_pattern = re.compile(r'"category":\s*"(.+?)"')
    func_pattern = re.compile(r'"widget_function":\s*(\w+)')

    graph_info = {}  # name -> {"category": ..., "func_name": ...}
    current_name = None
    in_graph = False
    brace_depth = 0
    current_cat = None
    current_func = None

    for i in range(graphs_start or 0, graphs_end or len(source_lines)):
        line = source_lines[i]
        if not in_graph:
            m = name_pattern.match(line)
            if m:
                current_name = m.group(1)
                current_cat = None
                current_func = None
                in_graph = True
                brace_depth = line.count("{") - line.count("}")
                # Check if category is on same line
                cm = re.search(r'"category":\s*"(.+?)"', line)
                if cm:
                    current_cat = cm.group(1)
                fm = re.search(r'"widget_function":\s*(\w+)', line)
                if fm:
                    current_func = fm.group(1)
        else:
            if not current_cat:
                cm = re.search(r'"category":\s*"(.+?)"', line)
                if cm:
                    current_cat = cm.group(1)
            if not current_func:
                fm = re.search(r'"widget_function":\s*(\w+)', line)
                if fm:
                    current_func = fm.group(1)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                if current_name and current_cat and current_func:
                    graph_info[current_name] = {"category": current_cat, "func_name": current_func}
                current_name = None
                in_graph = False

    print(f"Found {len(graph_info)} graph entries")

    # Step 2: Find all function definitions and their line ranges
    func_ranges = {}  # func_name -> (start_line, end_line)
    func_def_pattern = re.compile(r"^def (\w+)\(.*\):.*")
    
    i = 0
    while i < len(source_lines):
        line = source_lines[i]
        m = func_def_pattern.match(line)
        if m:
            func_name = m.group(1)
            # Calculate indentation of this function
            func_indent = len(line) - len(line.lstrip())
            # Find end: next def at same indentation or end of file
            end_idx = len(source_lines) - 1
            for j in range(i + 1, len(source_lines)):
                next_line = source_lines[j]
                stripped = next_line.strip()
                if stripped == "":
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                # Stop at next def at same or lesser indentation
                if stripped.startswith("def ") and next_indent <= func_indent:
                    end_idx = j - 1
                    break
                # Stop at graphs dict or any top-level assignment
                if j >= graphs_start:
                    end_idx = j - 1
                    break
            func_ranges[func_name] = (i, end_idx)
            i = end_idx + 1
        else:
            i += 1

    print(f"Found {len(func_ranges)} function definitions")

    # Step 3: Group graph entries by category
    category_widgets = {}  # category -> [(graph_name, func_name)]
    for gname, ginfo in graph_info.items():
        cat = ginfo["category"]
        if cat not in category_widgets:
            category_widgets[cat] = []
        category_widgets[cat].append((gname, ginfo["func_name"]))

    # Step 4: Extract data generation helpers into shared.py
    shared_funcs = ["_gen_corr", "_gen_reg", "_rng"]
    shared_lines = [SHARED_IMPORTS]
    shared_lines.append("")
    shared_lines.append("# =========================")
    shared_lines.append("# DATA GENERATION HELPERS")
    shared_lines.append("# =========================\n")

    # Add _rng declaration
    for line_num, line in enumerate(source_lines):
        if line.strip().startswith("_rng = "):
            shared_lines.append(line)
            break

    for func_name in ["_gen_corr", "_gen_reg"]:
        if func_name in func_ranges:
            start, end = func_ranges[func_name]
            for j in range(start, end + 1):
                shared_lines.append(source_lines[j])
            shared_lines.append("")

    shared_path = os.path.join(output_dir, "shared.py")
    with open(shared_path, "w", encoding="utf-8") as f:
        f.write("\n".join(shared_lines))
    print(f"  shared.py: OK ({len(shared_lines)} lines)")

    # Step 5: For each category, create a module
    # Also track which functions are used so we know which to include
    used_functions = set()
    for cat, widgets in category_widgets.items():
        for gname, func_name in widgets:
            used_functions.add(func_name)

    # Step 5: Generate per-category modules
    total_widgets = 0
    for cat, widgets in category_widgets.items():
        module_name = CATEGORY_MAP.get(cat)
        if not module_name:
            print(f"  SKIPPING category: {cat}")
            continue

        mod_lines = [SHARED_IMPORTS]
        mod_lines.append("from .shared import _rng, _gen_corr, _gen_reg\n")

        for gname, func_name in widgets:
            if func_name in func_ranges:
                start, end = func_ranges[func_name]
                for j in range(start, end + 1):
                    if j >= graphs_start and j <= graphs_end:
                        continue
                    mod_lines.append(source_lines[j])
                mod_lines.append("")
                total_widgets += 1

        # Add a GRAPHS dict for this module
        mod_lines.append(f"\nGRAPHS = {{")
        for i, (gname, func_name) in enumerate(widgets):
            ending = "," if i < len(widgets) - 1 else ""
            mod_lines.append(f'    "{gname}": {func_name}{ending}')
        mod_lines.append("}")

        output_path = os.path.join(output_dir, f"{module_name}.py")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mod_lines))

        # Validate
        try:
            with open(output_path, encoding="utf-8") as f:
                ast.parse(f.read())
            print(f"  {module_name}.py: OK ({len(mod_lines)} lines)")
        except SyntaxError as e:
            print(f"  {module_name}.py: SYNTAX ERROR at line {e.lineno}: {e.msg}")
            if e.lineno:
                ctx_start = max(0, e.lineno - 2)
                ctx_end = min(len(mod_lines), e.lineno + 2)
                for ci in range(ctx_start, ctx_end):
                    marker = ">>>" if ci == e.lineno - 1 else "   "
                    print(f"{marker}{ci+1}: {mod_lines[ci]}")

    # Check for unused functions
    for func_name in func_ranges:
        if func_name not in used_functions and not func_name.startswith("_") and func_name.endswith("_widget"):
            print(f"  WARNING: Unused widget function: {func_name}")

    print(f"\nTotal widgets distributed: {total_widgets}")


if __name__ == "__main__":
    main()
