"""Statistical test widget registry.

Each widget module registers its tests via @register_test("Test Name").
"""

import streamlit as st
import re

_registry = {}


def register_test(name):
    """Decorator to register a test widget function."""
    def decorator(func):
        _registry[name] = func
        return func
    return decorator


def render_latex(formula_text):
    """Render LaTeX formulas from text with $$ delimiters."""
    last_end = 0
    for match in re.finditer(r"\$\$(.*?)\$\$", formula_text, re.DOTALL):
        text_before = formula_text[last_end : match.start()]
        if text_before.strip():
            st.markdown(text_before)
        latex_code = match.group(1).strip()
        st.latex(latex_code)
        last_end = match.end()
    text_after = formula_text[last_end:]
    if text_after.strip():
        st.markdown(text_after)


def render_test_widget(test_name, external_data=None):
    """Render interactive widget for specific statistical test.

    Parameters
    ----------
    test_name : str
    external_data : dict or None
    """
    if test_name in _registry:
        _registry[test_name](external_data)
    else:
        st.info("Interactive widget coming soon for this test.")


def list_registered_tests():
    """Return list of all registered test names."""
    return list(_registry.keys())


import features.widgets.parametric
import features.widgets.nonparametric
import features.widgets.categorical
import features.widgets.regression
import features.widgets.survival
import features.widgets.diagnostic
import features.widgets.agreement
import features.widgets.other
