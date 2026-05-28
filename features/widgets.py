"""Backward-compatible wrapper for the widget registry.

All test widgets have been refactored into features/widgets/ directory.
This file re-exports the public API for backward compatibility.
"""

from features.widgets import render_test_widget, render_latex
