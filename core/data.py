import json
import os
from core.types import TestDefinition
from core.test_definitions import ALL_TESTS

_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

def _load_json(filename):
    path = os.path.join(_data_dir, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

rules: list[TestDefinition] = list(ALL_TESTS)
TEST_TO_SS_TYPE = _load_json("test_to_ss_type.json")
CRITERIA_FIELDS = _load_json("criteria_fields.json")
FIELDS = _load_json("fields.json")
