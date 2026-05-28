"""Verify extracted widget modules load correctly."""
import sys
import os
sys.path.insert(0, ".")

# Test loading all widget modules
try:
    from features.widgets import render_test_widget, render_latex, list_registered_tests
    tests = list_registered_tests()
    print(f"Registered tests: {len(tests)}")
    for t in sorted(tests):
        print(f"  - {t}")
    print("\nWidget registry loaded successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
