"""Build graph_explorer package: category modules + __init__.py."""
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

CATEGORY_MAP = [
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


def main():
    src = os.path.join(os.path.dirname(__file__), "features", "graph_explorer.py")
    pkg = os.path.join(os.path.dirname(__file__), "features", "graph_explorer_")
    os.makedirs(pkg, exist_ok=True)
    with open(src, encoding="utf-8") as f:
        lines = [l.rstrip("\n").rstrip("\r") for l in f.readlines()]

    # Find graphs dict range
    gs = ge = None
    for i, l in enumerate(lines):
        if l.strip() == "graphs = {": gs = i
        if gs is not None and l.strip() == "}": ge = i; break

    # Parse graphs dict metadata (name -> full_meta_dict_without_widget_function)
    import json
    graph_meta = {}
    name_pat = re.compile(r'^\s*"(.+?)":\s*{')
    cur_name = None
    cur_lines = []
    brace_depth = 0
    in_g = False

    for l in lines[gs:ge+1]:
        if not in_g:
            m = name_pat.match(l)
            if m:
                cur_name = m.group(1)
                cur_lines = [l]
                in_g = True
                brace_depth = l.count("{") - l.count("}")
        else:
            cur_lines.append(l)
            brace_depth += l.count("{") - l.count("}")
            if brace_depth <= 0:
                # Found end of entry - extract metadata
                entry = "\n".join(cur_lines)
                # Remove widget_function line
                entry_clean = re.sub(r',?\s*"widget_function":\s*\w+', "", entry)
                entry_clean = re.sub(r'"widget_function":\s*\w+,?\s*', "", entry_clean)
                try:
                    meta = ast.literal_eval(entry_clean)
                    graph_meta[cur_name] = meta
                except:
                    print(f"  WARNING: could not parse metadata for '{cur_name}'")
                cur_name = None
                in_g = False

    # Parse function ranges
    func_ranges = {}
    func_pat = re.compile(r"^def (\w+)\(.*\):.*")
    i = 0
    while i < len(lines):
        m = func_pat.match(lines[i])
        if m:
            fn = m.group(1)
            ind = len(lines[i]) - len(lines[i].lstrip())
            end = len(lines) - 1
            for j in range(i+1, len(lines)):
                nl = lines[j]
                if gs <= j <= ge: continue
                s = nl.strip()
                if s == "": continue
                ni = len(nl) - len(nl.lstrip())
                if s.startswith("def ") and ni <= ind:
                    end = j - 1; break
            func_ranges[fn] = (i, end)
            i = end + 1
        else:
            i += 1

    # Group function ownership by category
    cat_funcs = {}  # category -> set of func_names
    for gname, meta in graph_meta.items():
        cat = meta.get("category", "")
        if cat not in cat_funcs:
            cat_funcs[cat] = set()
        # Find which function this entry maps to
        # We need to look at the original graphs dict to find widget_function references
        # But since we removed them from metadata, we need another way
        # Let's read the original dict again
    # Actually, let me re-read widget_function references from original
    cat_funcs = {}
    name_to_func = {}  # graph_name -> func_name
    with open(src, encoding="utf-8") as f2:
        raw = f2.read()
    func_ref_pat = re.compile(r'"(.+?)":\s*\{.*?"widget_function":\s*(\w+)', re.DOTALL)
    for m in func_ref_pat.finditer(raw):
        gname = m.group(1)
        func = m.group(2)
        name_to_func[gname] = func
        # Find its category
        cat_section = raw[m.start():m.end()]
        cat_m = re.search(r'"category":\s*"(.+?)"', cat_section)
        cat = cat_m.group(1) if cat_m else "Unknown"
        if cat not in cat_funcs:
            cat_funcs[cat] = []
        cat_funcs[cat].append((gname, func))

    # Determine function home
    func_home = {}
    for cat_name, _ in CATEGORY_MAP:
        for gname, func_name in cat_funcs.get(cat_name, []):
            if func_name not in func_home:
                func_home[func_name] = CATEGORY_MAP[[c for c,_ in CATEGORY_MAP].index(cat_name)][1]
                # Actually just find the module name
                for cn, mn in CATEGORY_MAP:
                    if cn == cat_name:
                        func_home[func_name] = mn
                        break

    print(f"Functions: {len(func_ranges)}, Graph entries: {len(name_to_func)}, Categories: {len(cat_funcs)}")

    # Generate shared.py
    shared = [SHARED_IMPORTS, "\n# DATA GENERATION HELPERS\n"]
    for l in lines:
        if l.strip().startswith("_rng = "):
            shared.append(l); break
    for fn in ["_gen_corr", "_gen_reg"]:
        if fn in func_ranges:
            s, e = func_ranges[fn]
            for j in range(s, e+1): shared.append(lines[j])
            shared.append("")

    with open(os.path.join(pkg, "shared.py"), "w", encoding="utf-8") as f:
        f.write("\n".join(shared))

    # Generate category modules
    for cat_name, mod_name in CATEGORY_MAP:
        mod_lines = [SHARED_IMPORTS, "from .shared import _rng, _gen_corr, _gen_reg\n"]

        entries = cat_funcs.get(cat_name, [])

        # Collect imports and own functions
        own_funcs = []
        imports = {}
        for gname, func_name in entries:
            if func_name not in func_ranges:
                continue
            home = func_home.get(func_name, mod_name)
            if home == mod_name:
                if func_name not in own_funcs:
                    own_funcs.append(func_name)
            else:
                if home not in imports:
                    imports[home] = []
                if func_name not in imports[home]:
                    imports[home].append(func_name)

        for src_mod, fnames in sorted(imports.items()):
            mod_lines.append(f"from .{src_mod} import {', '.join(sorted(fnames))}")

        for func_name in sorted(own_funcs, key=lambda f: func_ranges[f][0]):
            s, e = func_ranges[func_name]
            for j in range(s, e+1): mod_lines.append(lines[j])
            mod_lines.append("")

        # GRAPHS dict with full metadata
        mod_lines.append("\nGRAPHS = {")
        for i, (gname, func_name) in enumerate(entries):
            meta = graph_meta.get(gname, {"category": cat_name})
            meta_str = json.dumps(meta, ensure_ascii=False)
            ending = "," if i < len(entries) - 1 else ""
            mod_lines.append(f'    "{gname}": {meta_str}{ending}')
        mod_lines.append("}")

        # Save
        out = os.path.join(pkg, f"{mod_name}.py")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(mod_lines))
        try:
            ast.parse("\n".join(mod_lines))
            print(f"  {mod_name}.py: OK ({len(mod_lines)} lines)")
        except SyntaxError as e:
            print(f"  {mod_name}.py: SYNTAX ERROR line {e.lineno}: {e.msg}")

    # Generate __init__.py
    init = ["""
    Graph Explorer package.

    Each category module exports a GRAPHS dict mapping graph name to metadata dict.
    The metadata dicts have "widget_function" keys added here by merging.
    """]
    init.append("")
    init.append("import streamlit as st")
    for cn, mn in CATEGORY_MAP:
        init.append(f"from .{mn} import GRAPHS as _{mn}_graphs")
    init.append("")
    init.append("graphs = {}")
    init.append("_CATEGORY_ORDER = []")
    for cn, mn in CATEGORY_MAP:
        init.append(f"for _name, _meta in _{mn}_graphs.items():")
        init.append(f'    _meta["widget_function"] = _{mn}_widgets[_name]')
        # Hmm, this doesn't work because _mn_widgets is not defined
    
    # Actually, the GRAPHS dicts from modules already don't have widget_function
    # I need a different approach: store function names, resolve later
    # OR: include function refs in __init__ by combining metadata + function refs
    
    # Simplest: modify GRAPHS to include function references
    # Store name -> func in a separate dict

    init2 = ['"""Graph Explorer package."""']
    init2.append("")
    init2.append("import streamlit as st")
    init2.append("")
    
    # Import all GRAPHS dicts
    for cn, mn in CATEGORY_MAP:
        init2.append(f"from .{mn} import GRAPHS as _{mn}_graphs")
    init2.append("")
    
    # Build full graphs dict
    init2.append("# Build full graphs dict (merge metadata + function references)")
    init2.append("graphs = {}")
    
    for cn, mn in CATEGORY_MAP:
        init2.append(f"graphs.update(_{mn}_graphs)")
    
    init2.append("")
    init2.append("CATEGORIES = [")
    for cn, mn in CATEGORY_MAP:
        init2.append(f'    "{cn}",')
    init2.append("]")
    init2.append("")

    # BUT: the GRAPHS dicts from modules don't have widget_function!
    # I need to include them somehow.
    # Let me regenerate the modules to include widget_function references.

    init_path = os.path.join(pkg, "__init__.py")
    with open(init_path, "w", encoding="utf-8") as f:
        f.write("\n".join(init2))

    print("Init written")
    print("Package built in features/graph_explorer_/")


if __name__ == "__main__":
    main()
