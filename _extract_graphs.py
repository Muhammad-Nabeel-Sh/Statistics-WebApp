"""Extract graph_explorer.py into per-category modules."""

import re
import os
import ast
import json

CATEGORY_MODULE_MAP = {
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

CATEGORY_TO_DIR = {v: k for k, v in CATEGORY_MODULE_MAP.items()}

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


def slugify(name):
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def extract_functions_and_registry(source_path):
    """Extract widget functions and the graphs dict."""
    with open(source_path, encoding="utf-8") as f:
        source = f.read()

    # Parse the file to find function definitions
    lines = source.split("\n")

    # Find the _rng declaration line (after imports, before first widget)
    # Find the line where the graphs dict starts
    graphs_start = None
    graphs_end = None
    for i, line in enumerate(lines):
        if line.strip() == "graphs = {":
            graphs_start = i
        if graphs_start is not None and line.strip() == "}":
            graphs_end = i

    if graphs_start is None or graphs_end is None:
        print("ERROR: Could not find graphs dict boundaries")
        return {}, {}

    print(f"Graphs dict: lines {graphs_start + 1} to {graphs_end + 1}")

    # Parse the graphs dict to extract info per graph
    # Use a simple approach - find widget_function references
    func_pattern = re.compile(r'"widget_function":\s*(\w+)')
    name_pattern = re.compile(r'^\s*"(.+?)":\s*\{')

    graph_info = {}  # graph_name -> {"func_name": xxx, "category": yyy, "metadata": {...}}
    current_name = None
    current_category = None
    current_meta_lines = []
    in_graph = False
    brace_depth = 0
    meta_start = None

    for i in range(graphs_start, graphs_end + 1):
        line = lines[i]

        if not in_graph:
            m = name_pattern.match(line)
            if m:
                current_name = m.group(1)
                current_category = None
                current_meta_lines = []
                in_graph = True
                brace_depth = line.count("{") - line.count("}")
                meta_start = i
                # Extract what we can from this first line
                cat_m = re.search(r'"category":\s*"(.+?)"', line)
                if cat_m:
                    current_category = cat_m.group(1)
        else:
            if not current_category:
                cat_m = re.search(r'"category":\s*"(.+?)"', line)
                if cat_m:
                    current_category = cat_m.group(1)

            current_meta_lines.append(line)

            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0 or (line.strip() == "}," and brace_depth <= 1):
                # Graph entry complete
                # Find widget function
                func_m = func_pattern.search("\n".join(current_meta_lines))
                func_name = func_m.group(1) if func_m else None
                if current_name and current_category and func_name:
                    graph_info[current_name] = {
                        "func_name": func_name,
                        "category": current_category,
                    }
                current_name = None
                current_category = None
                current_meta_lines = []
                in_graph = False

    print(f"Found {len(graph_info)} graph entries")

    # Now extract each function's source code
    # Find all function definitions and their line ranges
    function_ranges = {}
    func_def_pattern = re.compile(r"^def (\w+)\(.*\):")
    for i, line in enumerate(lines):
        m = func_def_pattern.match(line)
        if m:
            func_name = m.group(1)
            # Find the end of the function (next def at same level or EOF)
            func_indent = len(line) - len(line.lstrip())
            end_idx = len(lines)
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip() == "":
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_line.startswith("def ") and next_indent == func_indent:
                    end_idx = j
                    break
                if next_line.strip() and next_indent <= func_indent and not next_line.strip().startswith("#"):
                    # This might be a non-def line at same/smaller indent - only for helper functions
                    pass
            function_ranges[func_name] = (i, end_idx - 1)

    # Organize: category -> list of (graph_name, func_name, func_lines)
    category_graphs = {}
    function_bodies = {}

    for func_name, (start, end) in function_ranges.items():
        func_lines = lines[start:end + 1]
        function_bodies[func_name] = func_lines

    # Group by category
    for graph_name, info in graph_info.items():
        cat = info["category"]
        func_name = info["func_name"]
        if cat not in category_graphs:
            category_graphs[cat] = []
        category_graphs[cat].append({
            "graph_name": graph_name,
            "func_name": func_name,
        })

    print("\nCategory breakdown:")
    for cat, graphs in category_graphs.items():
        print(f"  {cat}: {len(graphs)} graphs")

    return category_graphs, function_bodies, graph_info


def generate_category_module(cat_name, graphs, function_bodies, graph_info, module_name):
    """Generate a category module file."""
    module_dir = os.path.join(os.path.dirname(__file__), "features", "graph_explorer")
    os.makedirs(module_dir, exist_ok=True)
    
    lines = [SHARED_IMPORTS]
    lines.append("")
    lines.append(f"# {cat_name}")
    lines.append("")

    # Track which functions we've written
    written_funcs = set()
    extra_funcs = {}  # helper functions not directly in graph_info

    for g in graphs:
        func_name = g["func_name"]
        if func_name in written_funcs:
            continue
        
        if func_name in function_bodies:
            func_lines = function_bodies[func_name]
            for fl in func_lines:
                lines.append(fl)
            lines.append("")
            written_funcs.add(func_name)

    lines.append("")
    lines.append(f"GRAPHS = {{")
    for i, g in enumerate(graphs):
        gn = g["graph_name"]
        fn = g["func_name"]
        comma = "," if i < len(graphs) - 1 else ""
        lines.append(f'    "{gn}": {fn},{comma}')
    lines.append("}")
    lines.append("")

    output_path = os.path.join(module_dir, f"{module_name}.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Validate
    try:
        ast.parse("\n".join(lines))
        print(f"  {module_name}.py: OK ({len(lines)} lines)")
    except SyntaxError as e:
        print(f"  {module_name}.py: SYNTAX ERROR at line {e.lineno}: {e.msg}")

    return output_path


def main():
    source_path = os.path.join(os.path.dirname(__file__), "features", "graph_explorer.py")
    category_graphs, function_bodies, graph_info = extract_functions_and_registry(source_path)

    # Extract graph metadata to JSON (without function references)
    metadata = {}
    for graph_name, info in graph_info.items():
        metadata[graph_name] = {
            "category": info["category"],
        }
    
    # Generate modules
    for cat_name, graphs in category_graphs.items():
        module_name = CATEGORY_MODULE_MAP.get(cat_name)
        if not module_name:
            print(f"  SKIPPING unknown category: {cat_name}")
            continue
        generate_category_module(cat_name, graphs, function_bodies, graph_info, module_name)


if __name__ == "__main__":
    main()
