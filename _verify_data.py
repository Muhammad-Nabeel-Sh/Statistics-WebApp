from core.data import rules, TEST_TO_SS_TYPE, CRITERIA_FIELDS, FIELDS
print(f"rules: {len(rules)} items")
print(f"TEST_TO_SS_TYPE: {len(TEST_TO_SS_TYPE)} items")
print(f"CRITERIA_FIELDS: {len(CRITERIA_FIELDS)} items")
print(f"FIELDS: {len(FIELDS)} items")
print(f"First rule name: {rules[0]['name']}")
print(f"First SS mapping: {list(TEST_TO_SS_TYPE.items())[0]}")
