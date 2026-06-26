from dataclasses import dataclass
from typing import Any

_JSON_KEY_TO_ATTR = {
    "Objective": "objective",
    "Dependent_Variable": "dependent_var",
    "Independent_Variable": "independent_var",
    "Groups": "groups",
    "Relation": "relation",
    "Distribution": "distribution",
    "Explanation": "explanation",
    "Example": "example",
    "Formula": "formula",
    "Decision Rules": "decision_rules",
    "Core_Assumptions": "core_assumptions",
    "Interpretation": "interpretation",
    "Post-Hoc": "post_hoc",
    "Realworld_Applications": "realworld_apps",
}


@dataclass
class TestDefinition:
    name: str
    objective: str | list[str] | None = None
    dependent_var: str | list[str] | None = None
    independent_var: str | list[str] | None = None
    groups: str | list[str] | None = None
    relation: str | list[str] | None = None
    distribution: str | list[str] | None = None
    explanation: str | None = None
    example: str | None = None
    formula: str | None = None
    decision_rules: str | None = None
    core_assumptions: str | None = None
    interpretation: str | None = None
    post_hoc: str | None = None
    realworld_apps: str | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "name":
            return self.name
        attr = _JSON_KEY_TO_ATTR.get(key, key.lower())
        return getattr(self, attr)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default

    def __contains__(self, key: str) -> bool:
        val = self.get(key)
        return val is not None
