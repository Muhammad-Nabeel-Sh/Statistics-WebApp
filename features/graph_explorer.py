"""Backward-compatible wrapper for the graph_explorer package.

All graph widgets have been refactored into features/graph_explorer/ directory.
This file re-exports the public API for backward compatibility.
"""

from features.graph_explorer import render_graph_explorer, graphs, CATEGORIES
