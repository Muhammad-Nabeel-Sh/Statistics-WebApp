import sys
sys.path.insert(0, ".")

import features.graph_explorer as ge
print(f"Module file: {ge.__file__}")
print(f"Has render_graph_explorer: {hasattr(ge, 'render_graph_explorer')}")
print(f"Has graphs: {hasattr(ge, 'graphs')}")
print(f"Has CATEGORIES: {hasattr(ge, 'CATEGORIES')}")
print(f"Has __path__: {hasattr(ge, '__path__')}")

# If it's a package, list the submodules
if hasattr(ge, '__path__'):
    import pkgutil
    for mod in pkgutil.iter_modules(ge.__path__):
        print(f"  Submodule: {mod.name}")
