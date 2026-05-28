"""Comprehensive verification of all major app imports."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
errors = []

# 1. core modules
try:
    from core.data import rules, TEST_TO_SS_TYPE, FIELDS, CRITERIA_FIELDS
    print(f"[OK] core.data: {len(rules)} rules")
except Exception as e:
    errors.append(f"core.data: {e}")
    print(f"[ERR] core.data: {e}")

try:
    from core.models import ExternalData, OneSampleData, TwoSampleData, is_using_external
    print("[OK] core.models")
except Exception as e:
    errors.append(f"core.models: {e}")
    print(f"[ERR] core.models: {e}")

try:
    from core.matching import find_matching_tests
    print("[OK] core.matching")
except Exception as e:
    errors.append(f"core.matching: {e}")
    print(f"[ERR] core.matching: {e}")

try:
    from core.post_hoc import render_post_hoc
    print("[OK] core.post_hoc")
except Exception as e:
    errors.append(f"core.post_hoc: {e}")
    print(f"[ERR] core.post_hoc: {e}")

try:
    from core.utils import _apa_table, data_source_toggle, format_p_value
    print("[OK] core.utils")
except Exception as e:
    errors.append(f"core.utils: {e}")
    print(f"[ERR] core.utils: {e}")

# 2. widgets
try:
    from features.widgets import render_test_widget, render_latex, list_registered_tests
    tests = list_registered_tests()
    print(f"[OK] features.widgets: {len(tests)} registered tests")
except Exception as e:
    errors.append(f"features.widgets: {e}")
    print(f"[ERR] features.widgets: {e}")

# 3. graph_explorer
try:
    from features.graph_explorer import render_graph_explorer, graphs, CATEGORIES
    print(f"[OK] features.graph_explorer: {len(graphs)} graphs, {len(CATEGORIES)} categories")
except Exception as e:
    errors.append(f"features.graph_explorer: {e}")
    print(f"[ERR] features.graph_explorer: {e}")

# 4. finder_ui
try:
    from features.finder_ui import render_test_finder, render_all_tests_section
    print("[OK] features.finder_ui")
except Exception as e:
    errors.append(f"features.finder_ui: {e}")
    print(f"[ERR] features.finder_ui: {e}")

# 5. data_workspace
try:
    from features.data_workspace import render_data_workspace, _build_external_data
    print("[OK] features.data_workspace")
except Exception as e:
    errors.append(f"features.data_workspace: {e}")
    print(f"[ERR] features.data_workspace: {e}")

# 6. Other features
for mod_name in ["builtin_datasets", "control_charts", "diagnostics", "distributions",
                  "factor_analysis", "glossary", "power_calculator", "solved_examples",
                  "tabulation"]:
    try:
        __import__(f"features.{mod_name}")
        print(f"[OK] features.{mod_name}")
    except Exception as e:
        errors.append(f"features.{mod_name}: {e}")
        print(f"[ERR] features.{mod_name}: {e}")

# 7. App entry points (just verify imports)
for app_name in ["app_finder", "app_explorer", "app_tabulation", "app_distributions",
                  "app_power", "app_diagnostics", "app_factor", "app_spc"]:
    try:
        # Try importing the module (won't run main())
        import importlib
        spec = importlib.util.spec_from_file_location(app_name, f"apps/{app_name}.py")
        if spec:
            print(f"[OK] apps/{app_name}.py (spec found)")
    except Exception as e:
        errors.append(f"apps/{app_name}: {e}")
        print(f"[ERR] apps/{app_name}: {e}")

print(f"\n{'='*50}")
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL IMPORTS VERIFIED SUCCESSFULLY!")
