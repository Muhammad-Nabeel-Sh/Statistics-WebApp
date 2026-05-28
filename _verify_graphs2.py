"""Verify graph_explorer package."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

try:
    from features.graph_explorer import render_graph_explorer, graphs, CATEGORIES
    print(f"Package imports OK")
    print(f"  CATEGORIES: {len(CATEGORIES)} items")
    print(f"  graphs: {len(graphs)} items")
    for name, info in list(graphs.items())[:3]:
        print(f"  {name}: widget_function={info.get('widget_function') is not None}")
    print("All references valid!")
except Exception as e:
    import traceback
    traceback.print_exc()
