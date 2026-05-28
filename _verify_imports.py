"""End-to-end verification that imports work correctly."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# 1. core data
from core.data import rules, TEST_TO_SS_TYPE, FIELDS, CRITERIA_FIELDS
print(f"core.data: {len(rules)} rules, {len(TEST_TO_SS_TYPE)} mappings")

# 2. matching
from core.matching import find_matching_tests
print(f"matching.find_matching_tests: ok")

# 3. widgets (backward compat)
from features.widgets import render_test_widget, render_latex
print(f"widgets.render_test_widget: ok")

# 4. widget registry
from features.widgets import list_registered_tests
tests = list_registered_tests()
print(f"widgets registry: {len(tests)} tests")

# 5. finder_ui
from features.finder_ui import render_test_finder
print(f"finder_ui: ok")

# 6. data_workspace
from features.data_workspace import render_data_workspace
print(f"data_workspace: ok")

# 7. post_hoc
from core.post_hoc import render_post_hoc
print(f"post_hoc: ok")

# 8. utils
from core.utils import _apa_table, data_source_toggle, format_p_value
print(f"utils: ok")

print("\nAll imports verified successfully!")
