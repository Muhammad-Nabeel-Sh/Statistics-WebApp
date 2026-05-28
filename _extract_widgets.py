"""Extract widget elif blocks from widgets.py into family modules."""

import re
import os
import ast

FAMILIES = {
    "parametric": [
        "One-sample t-test",
        "One-sample z-test",
        "Student's t-test (Independent)",
        "Welch's t-test (Independent, Unequal Variances)",
        "Paired t-test",
        "One-way ANOVA",
        "One-way Welch ANOVA",
        "Two-way ANOVA",
        "ANCOVA",
        "Repeated Measures ANOVA",
        "MANOVA",
        "F-Test for Two Variances",
        "Equivalence Test (TOST) - Two Independent Samples",
    ],
    "nonparametric": [
        "One-sample Wilcoxon Signed-Rank Test",
        "Sign Test (One-sample)",
        "Wilcoxon Signed-Rank Test",
        "Sign Test (Paired)",
        "Mann-Whitney U Test",
        "Kruskal-Wallis Test",
        "Mood's Median Test",
        "Friedman Test",
        "Permutation MANOVA or Non-Parametric MANOVA",
    ],
    "categorical": [
        "One-sample Proportion Test (Binomial Test)",
        "Binomial Test",
        "Multinomial Test",
        "Chi-Square Goodness-of-Fit Test",
        "Poisson Goodness-of-Fit Test",
        "Chi-Square Test",
        "Chi-Square Test of Independence",
        "McNemar's Test",
        "Cochran's Q Test",
        "Fisher's Exact Test",
    ],
    "regression": [
        "Pearson Correlation",
        "Spearman Rank Correlation",
        "Kendall's Tau-b",
        "Point-Biserial Correlation",
        "Logistic Regression",
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Multinomial Logistic Regression",
        "Ordinal Logistic Regression",
        "Poisson Regression",
        "Negative Binomial Regression",
        "Cox Proportional Hazards Regression",
    ],
    "survival": [
        "Kaplan-Meier Survival Analysis",
        "Log-Rank Test",
    ],
    "diagnostic": [
        "Sensitivity & Specificity Analysis",
        "ROC Curve Analysis",
        "Likelihood Ratio Analysis",
    ],
    "agreement": [
        "Cohen's Kappa (Agreement Analysis)",
        "Fleiss' Kappa",
        "Weighted Kappa",
        "Bland-Altman Analysis",
    ],
    "other": [
        "Runs Test for Randomness",
    ],
}

MODULE_IMPORTS = """import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from core.post_hoc import render_post_hoc
from core.utils import (
    format_p_value,
    cohens_d_one_sample_ci,
    cohens_d_independent_ci,
    hedges_g,
    omega_squared_partial,
    format_effect_size_with_ci,
    st_plot_with_download,
    interpret_cohens_d,
    interpret_eta_squared,
    data_source_toggle,
)
from features.widgets import register_test
"""


def slugify(name):
    """Create a valid function name from a test name."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    s = re.sub(r"_+", "_", s)
    if s and s[0].isdigit():
        s = "t_" + s
    return s


def extract_blocks(source_lines):
    """Extract all test blocks from render_test_widget function."""
    func_start = None
    for i, line in enumerate(source_lines):
        if line.startswith("def render_test_widget("):
            func_start = i
            break

    if func_start is None:
        return []

    # The if/elif/else chain is at 4-space indentation (one level inside func)
    # Find the else: at exactly 4 spaces (not deeper nested else)
    else_line = None
    for i in range(func_start + 1, len(source_lines)):
        if source_lines[i] == "    else:\n" or source_lines[i] == "    else:":
            else_line = i
            break

    if else_line is None:
        return []

    # Now find all if/elif blocks between func_start and else_line
    blocks = []
    test_name_pattern = re.compile(r'^    (?:if|elif)\s+test_name\s*==\s*"(.+?)"\s*:\s*$')

    current_name = None
    current_start = None

    for i in range(func_start + 1, else_line):
        line = source_lines[i]
        m = test_name_pattern.match(line)
        if m:
            if current_name is not None and current_start is not None:
                blocks.append((current_name, current_start, i - 1))
            current_name = m.group(1)
            current_start = i

    # Add last block
    if current_name is not None and current_start is not None:
        blocks.append((current_name, current_start, else_line - 1))

    return blocks


def main():
    source_path = os.path.join(os.path.dirname(__file__), "features", "widgets.py")
    output_dir = os.path.join(os.path.dirname(__file__), "features", "widgets")

    with open(source_path, encoding="utf-8") as f:
        source_lines = f.readlines()

    blocks = extract_blocks(source_lines)
    print(f"Found {len(blocks)} test blocks")

    for name, start, end in blocks[:3]:
        print(f"  '{name}': lines {start+1}-{end+1}")

    # Group by family
    name_to_block = {name: (start, end) for name, start, end in blocks}

    for family, test_names in FAMILIES.items():
        family_lines = [MODULE_IMPORTS]

        for test_name in test_names:
            if test_name not in name_to_block:
                print(f"  WARNING: '{test_name}' not in blocks")
                continue

            start, end = name_to_block[test_name]
            func_name = f"render_{slugify(test_name)}"

            header = source_lines[start]
            header_match = re.search(r'test_name\s*==\s*"(.+?)"', header)
            exact_name = header_match.group(1) if header_match else test_name

            family_lines.append(f'\n@register_test("{exact_name}")')
            family_lines.append(f"def {func_name}(external_data=None):")

            for j in range(start + 1, end + 1):
                line = source_lines[j].rstrip("\n").rstrip("\r")
                # Remove 4 leading spaces
                if line.startswith("    "):
                    line = line[4:]
                family_lines.append(line)

            family_lines.append("")

        output_path = os.path.join(output_dir, f"{family}.py")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(family_lines))

        try:
            with open(output_path, encoding="utf-8") as f:
                ast.parse(f.read())
            line_count = len(family_lines)
            print(f"  {family}.py: OK ({line_count} lines)")
        except SyntaxError as e:
            print(f"  {family}.py: SYNTAX ERROR at line {e.lineno}: {e.msg}")
            if e.lineno:
                ctx_s = max(0, e.lineno - 3)
                ctx_e = min(len(family_lines), e.lineno + 2)
                for ci in range(ctx_s, ctx_e):
                    marker = " >>> " if ci == e.lineno - 1 else "     "
                    print(f"{marker}{ci+1}: {family_lines[ci]}")

    all_assigned = set()
    for tests in FAMILIES.values():
        all_assigned.update(tests)
    for name in sorted(name_to_block.keys()):
        if name not in all_assigned:
            print(f"  UNASSIGNED: '{name}'")


if __name__ == "__main__":
    main()
