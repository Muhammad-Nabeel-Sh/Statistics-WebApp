"""Robust extraction of graph_explorer.py into per-category modules + __init__.py."""

import re
import os
import ast
from collections import defaultdict

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

CATEGORY_MAP_ORDERED = [
    ("Distribution Plots", "distribution"),
    ("Comparison Plots", "comparison"),
    ("Correlation Plots", "correlation"),
    ("Regression Plots", "regression"),
    ("Diagnostic Accuracy Plots", "diagnostic"),
    ("Agreement Plots", "agreement"),
    ("Multivariate Plots", "multivariate"),
    ("Survival Analysis Plots", "survival"),
    ("Meta-Analysis Visualizations", "meta_analysis"),
    ("Post-Hoc Plots", "post_hoc"),
]


def extract_graphs_dict(lines):
    """Parse the graphs dict and return graph_info."""
    graph_info = {}  # graph_name -> {"category": ..., "func_name": ...}
    name_pattern = re.compile(r'^\s*"(.+?)":\s*{')
    
    current_name = None
    current_cat = None
    current_func = None
    brace_depth = 0
    in_graph = False

    for line in lines:
        if not in_graph:
            m = name_pattern.match(line)
            if m:
                current_name = m.group(1)
                current_cat = None
                current_func = None
                in_graph = True
                brace_depth = line.count("{") - line.count("}")
                cm = re.search(r'"category":\s*"(.+?)"', line)
                if cm: current_cat = cm.group(1)
                fm = re.search(r'"widget_function":\s*(\w+)', line)
                if fm: current_func = fm.group(1)
        else:
            if not current_cat:
                cm = re.search(r'"category":\s*"(.+?)"', line)
                if cm: current_cat = cm.group(1)
            if not current_func:
                fm = re.search(r'"widget_function":\s*(\w+)', line)
                if fm: current_func = fm.group(1)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                if current_name and current_cat and current_func:
                    graph_info[current_name] = {"category": current_cat, "func_name": current_func}
                current_name = None
                in_graph = False
    return graph_info


def find_function_ranges(lines, graphs_start, graphs_end):
    """Find line ranges for all function definitions."""
    func_ranges = {}
    func_def_pattern = re.compile(r"^def (\w+)\(.*\):.*")
    
    i = 0
    while i < len(lines):
        line = lines[i]
        m = func_def_pattern.match(line)
        if m:
            func_name = m.group(1)
            func_indent = len(line) - len(line.lstrip())
            end_idx = len(lines) - 1
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if graphs_start <= j <= graphs_end:
                    continue
                stripped = next_line.strip()
                if stripped == "":
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if stripped.startswith("def ") and next_indent <= func_indent:
                    end_idx = j - 1
                    break
            func_ranges[func_name] = (i, end_idx)
            i = end_idx + 1
        else:
            i += 1
    return func_ranges


def main():
    source_path = os.path.join(os.path.dirname(__file__), "features", "graph_explorer.py")
    output_dir = os.path.join(os.path.dirname(__file__), "features", "graph_explorer")
    os.makedirs(output_dir, exist_ok=True)

    with open(source_path, encoding="utf-8") as f:
        source_lines = f.readlines()
    source_lines = [l.rstrip("\n").rstrip("\r") for l in source_lines]

    # Find graphs dict boundaries
    graphs_start = graphs_end = None
    for i, line in enumerate(source_lines):
        if line.strip() == "graphs = {":
            graphs_start = i
        if graphs_start is not None and line.strip() == "}":
            graphs_end = i
            break

    graph_info = extract_graphs_dict(source_lines[graphs_start:graphs_end + 1])
    func_ranges = find_function_ranges(source_lines, graphs_start, graphs_end)
    print(f"Found {len(graph_info)} graph entries, {len(func_ranges)} functions")

    # Determine which functions each category needs
    category_funcs = defaultdict(set)  # category -> set of func_names
    for gname, ginfo in graph_info.items():
        category_funcs[ginfo["category"]].add(ginfo["func_name"])

    # Determine "home" module for each function (first category that needs it)
    func_home = {}
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        for func_name in category_funcs.get(cat_name, set()):
            if func_name not in func_home:
                func_home[func_name] = module_name

    # Create shared.py
    shared_lines = [SHARED_IMPORTS]
    shared_lines.append("\n# =========================\n# DATA GENERATION HELPERS\n# =========================\n")
    for line in source_lines:
        if line.strip().startswith("_rng = "):
            shared_lines.append(line)
            break
    for fn in ["_gen_corr", "_gen_reg"]:
        if fn in func_ranges:
            s, e = func_ranges[fn]
            for j in range(s, e + 1):
                shared_lines.append(source_lines[j])
            shared_lines.append("")

    shared_path = os.path.join(output_dir, "shared.py")
    with open(shared_path, "w", encoding="utf-8") as f:
        f.write("\n".join(shared_lines))
    print(f"  shared.py: OK ({len(shared_lines)} lines)")

    # Create category modules
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        mod_lines = [SHARED_IMPORTS]
        mod_lines.append("from .shared import _rng, _gen_corr, _gen_reg\n")

        # figure out which functions to include and which to import
        own_funcs = set()
        import_funcs = {}
        for func_name in category_funcs.get(cat_name, set()):
            if func_name not in func_ranges:
                print(f"  WARNING: {func_name} (needed by {module_name}) not found in function definitions")
                continue
            home_mod = func_home.get(func_name, module_name)
            if home_mod == module_name:
                own_funcs.add(func_name)
            else:
                # Import from home module
                import_key = home_mod
                if import_key not in import_funcs:
                    import_funcs[import_key] = []
                import_funcs[import_key].append(func_name)

        # Add imports from other modules
        for src_mod, fnames in sorted(import_funcs.items()):
            mod_lines.append(f"from .{src_mod} import {', '.join(sorted(fnames))}")

        # Add function definitions for owned functions
        for func_name in sorted(own_funcs, key=lambda f: func_ranges[f][0]):
            s, e = func_ranges[func_name]
            for j in range(s, e + 1):
                mod_lines.append(source_lines[j])
            mod_lines.append("")

        # Build GRAPHS dict
        category_graph_entries = [(gname, ginfo["func_name"]) for gname, ginfo in sorted(graph_info.items()) if ginfo["category"] == cat_name]
        mod_lines.append("\nGRAPHS = {")
        for i, (gname, func_name) in enumerate(category_graph_entries):
            ending = "," if i < len(category_graph_entries) - 1 else ""
            mod_lines.append(f'    "{gname}": {func_name}{ending}')
        mod_lines.append("}")

        output_path = os.path.join(output_dir, f"{module_name}.py")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(mod_lines))

        try:
            ast.parse("\n".join(mod_lines))
            print(f"  {module_name}.py: OK ({len(mod_lines)} lines, {len(own_funcs)} own funcs)")
        except SyntaxError as e:
            print(f"  {module_name}.py: SYNTAX ERROR at line {e.lineno}: {e.msg}")

    # Create __init__.py
    init_lines = ['"""Graph Explorer package - split by plot category."""']
    init_lines.append("")
    init_lines.append("import streamlit as st")
    init_lines.append("")
    init_lines.append("# Import all category modules to trigger GRAPHS registration")
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        init_lines.append(f"from .{module_name} import GRAPHS as _{module_name}_graphs")

    init_lines.append("")
    init_lines.append("# Merge all GRAPHS dicts")
    init_lines.append("graphs = {}")
    init_lines.append("_CATEGORY_ORDER = []")
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        init_lines.append(f"graphs.update(_{module_name}_graphs)")
        init_lines.append(f'_CATEGORY_ORDER.append("{cat_name}")')

    init_lines.append("")
    init_lines.append('CATEGORIES = _CATEGORY_ORDER')
    init_lines.append("")

    # Add render_graph_explorer function
    init_lines.append("""
def render_graph_explorer():
    st.title("Interactive Graph Explorer")
    st.write(
        "Explore statistical graphs interactively. Adjust controls to build visual intuition."
    )

    graph_category = st.sidebar.radio(
        "Graph Category",
        CATEGORIES,
        key="graph_category_radio",
    )

    category_graphs = {
        k: v for k, v in graphs.items() if v.get("category") == graph_category
    }

    st.header(f"{graph_category}", divider="orange")

    for name, info in category_graphs.items():
        widget_func = info.get("widget_function") or info.get("func") or globals().get(info.get("widget_func_name", ""))
        with st.expander(f"**{name}**", expanded=True):
            info["widget_function"]()
""")

    # But wait, the GRAPHS dicts from modules only have function references, not full metadata.
    # We need to add the metadata (description, category, etc.) back.
    # Extract metadata from original graphs dict entries.
    # Let me re-read the original file and extract the full metadata for each entry.

    # Actually, the modules' GRAPHS dicts map name->function reference.
    # The __init__.py needs to build the full dict with metadata.
    # Let me extract the metadata from the original file.
    
    # Read the original graphs dict area
    raw_meta = {}
    with open(source_path, encoding="utf-8") as f:
        original = f.read()

    # Simple extraction: find each graph entry block
    orig_lines = original.split("\n")
    graph_meta = {}
    name_pattern = re.compile(r'^\s*"(.+?)":\s*\{')
    in_graph = False
    brace_depth = 0
    current_entry = None
    current_name = None
    
    for line in orig_lines:
        if not in_graph:
            m = name_pattern.match(line)
            if m:
                current_name = m.group(1)
                current_entry = [line]
                in_graph = True
                brace_depth = line.count("{") - line.count("}")
        else:
            current_entry.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                # End of entry - store metadata without function reference
                entry_str = "\n".join(current_entry)
                # Remove the widget_function line
                entry_str_no_func = re.sub(r',?\s*"widget_function":\s*\w+', '', entry_str)
                try:
                    meta = ast.literal_eval(entry_str_no_func)
                    graph_meta[current_name] = meta
                except:
                    # Fall back to eval without widget_function
                    try:
                        meta = ast.literal_eval(entry_str_no_func)
                        graph_meta[current_name] = meta
                    except:
                        pass
                current_name = None
                in_graph = False
    
    # Now build final __init__.py
    init_lines2 = ['"""Graph Explorer package - split by plot category."""']
    init_lines2.append("")
    init_lines2.append("import streamlit as st")
    init_lines2.append("")
    init_lines2.append("# Import all category modules")
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        init_lines2.append(f"from .{module_name} import GRAPHS as _{module_name}_graphs")

    init_lines2.append("")
    init_lines2.append("# Build full graphs dict with metadata")
    init_lines2.append("graphs = {}")

    # For each category, add entries with metadata
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        init_lines2.append(f"")
        init_lines2.append(f"# {cat_name}")
        init_lines2.append(f"for _name, _func in _{module_name}_graphs.items():")
        init_lines2.append(f"    _meta = {repr(graph_meta.get(list(cat for cat in [cat_name])[0] if False else None, {}))}")
        init_lines2.append(f"    _meta['widget_function'] = _func")
        init_lines2.append(f"    _meta['category'] = '{cat_name}'")
        init_lines2.append(f"    graphs[_name] = _meta")

    # Wait, that's not right. Let me statically generate the metadata for each entry.
    # I should iterate over the entries from the module GRAPHS dicts.
    
    # Let me redo the init
    init_lines3 = ['"""Graph Explorer package - split by plot category."""']
    init_lines3.append("")
    init_lines3.append("import streamlit as st")
    init_lines3.append("")

    # Build import dicts
    module_names = {}
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        module_names[cat_name] = module_name

    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        init_lines3.append(f"from .{module_name} import GRAPHS")

    init_lines3.append("")
    init_lines3.append("# Build full graphs dict with metadata")
    init_lines3.append("graphs = {}")

    # Generate static code for each graph entry
    entry_num = 0
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        category_entries = sorted([(gname, ginfo["func_name"]) for gname, ginfo in graph_info.items() if ginfo["category"] == cat_name])
        for gname, func_name in category_entries:
            meta = graph_meta.get(gname, {})
            # Remove function reference from meta dict
            meta_str = repr(meta)
            init_lines3.append(f"")
            init_lines3.append(f"graphs['{gname}'] = {meta_str}")

    init_lines3.append("")
    init_lines3.append('CATEGORIES = [')
    for cat_name, module_name in CATEGORY_MAP_ORDERED:
        init_lines3.append(f'    "{cat_name}",')
    init_lines3.append(']')
    init_lines3.append("")

    # Add render_graph_explorer
    init_lines3.append("""
def render_graph_explorer():
    st.title("Interactive Graph Explorer")
    st.write(
        "Explore statistical graphs interactively. Adjust controls to build visual intuition."
    )

    graph_category = st.sidebar.radio(
        "Graph Category",
        CATEGORIES,
        key="graph_category_radio",
    )

    category_graphs = {
        k: v for k, v in graphs.items() if v.get("category") == graph_category
    }

    st.header(f"{graph_category}", divider="orange")

    for name, info in category_graphs.items():
        with st.expander(f"**{name}**", expanded=True):
            info["widget_function"]()
""")

    init_path = os.path.join(output_dir, "__init__.py")
    with open(init_path, "w", encoding="utf-8") as f:
        f.write("\n".join(init_lines3))
    print(f"  __init__.py: OK ({len(init_lines3)} lines)")

    # Rewrite features/graph_explorer.py as wrapper
    wrapper = '''"""
Backward-compatible wrapper for graph_explorer package.

All graph widgets have been refactored into features/graph_explorer/ directory.
This file re-exports the public API for backward compatibility.
"""

from features.graph_explorer import render_graph_explorer, graphs, CATEGORIES
'''

    wrapper_path = os.path.join(os.path.dirname(__file__), "features", "graph_explorer.py")
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper)
    print(f"  features/graph_explorer.py: rewritten as wrapper")

    print("\nDone!")


if __name__ == "__main__":
    main()
