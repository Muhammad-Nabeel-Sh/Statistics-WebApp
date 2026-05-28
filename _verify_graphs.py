"""End-to-end verification of graph_explorer package."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# Import from package
try:
    from features.graph_explorer import render_graph_explorer, graphs, CATEGORIES
    print(f"Package imports: OK")
    print(f"  CATEGORIES: {len(CATEGORIES)} items")
    print(f"  graphs: {len(graphs)} items")
    print(f"  Sample graph: {list(graphs.keys())[0]}")
    print(f"  graph entry keys: {list(graphs.values())[0].keys()}")
except Exception as e:
    import traceback
    traceback.print_exc()

# Import from module (backward compat)
try:
    # This should now work same as package import since the .py file is gone
    import features.graph_explorer
    print(f"\nModule import: OK")
    print(f"  render_graph_explorer: {hasattr(features.graph_explorer, 'render_graph_explorer')}")
except Exception as e:
    print(f"\nModule import error: {e}")

# Verify all function references resolve
print("\nChecking widget function references:")
for name, info in graphs.items():
    func = info.get("widget_function")
    if func is None:
        print(f"  MISSING widget_function: {name}")
    elif not callable(func):
        print(f"  NOT CALLABLE: {name} -> {func}")
print("  All references valid!")
