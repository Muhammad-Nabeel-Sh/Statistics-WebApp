"""Typed data models for external_data and related structures.

Uses Python dataclasses (no external dependencies) to provide type safety
for the external_data dictionary passed between Data Workspace and Widgets.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Format types (string literals for backward compatibility)
# ---------------------------------------------------------------------------

#: Supported data format modes
DATA_FORMATS = {
    "one_sample",       # Single sample of values
    "two_sample",       # Two independent groups
    "multi_sample",     # Three or more independent groups
    "paired",           # Two paired measurements
    "repeated",         # Three or more repeated measurements
    "correlation",      # Two continuous variables (X, Y)
    "categorical_one",  # Single categorical variable (frequency table)
    "categorical_two",  # Two categorical variables (contingency table)
}


# ---------------------------------------------------------------------------
# Data payloads
# ---------------------------------------------------------------------------

@dataclass
class OneSampleData:
    """Data format for one-sample tests (t-test, z-test, Wilcoxon, etc.)."""
    values: np.ndarray


@dataclass
class TwoSampleData:
    """Data format for two-group independent tests."""
    group1: np.ndarray
    group2: np.ndarray
    group_names: list[str] = field(default_factory=list)


@dataclass
class MultiSampleData:
    """Data format for multi-group independent tests (ANOVA, Kruskal-Wallis)."""
    groups: list[np.ndarray]
    group_names: list[str] = field(default_factory=list)


@dataclass
class PairedData:
    """Data format for paired tests (paired t-test, Wilcoxon signed-rank)."""
    values1: np.ndarray
    values2: np.ndarray


@dataclass
class RepeatedData:
    """Data format for repeated measures (3+ measurements per subject)."""
    measurements: list[np.ndarray]
    col_names: list[str] = field(default_factory=list)


@dataclass
class CorrelationData:
    """Data format for correlation/regression tests."""
    x: np.ndarray
    y: np.ndarray
    col_names: list[str] = field(default_factory=list)


@dataclass
class CategoricalOneData:
    """Data format for single categorical variable (Chi-Square GOF)."""
    categories: list[Any]
    counts: np.ndarray


@dataclass
class CategoricalTwoData:
    """Data format for two categorical variables (contingency table)."""
    contingency_table: Any  # pd.DataFrame
    col_a: str = ""
    col_b: str = ""
    col_a_vals: list[Any] = field(default_factory=list)
    col_b_vals: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# External Data container (replaces the loose dict)
# ---------------------------------------------------------------------------

@dataclass
class ExternalData:
    """Typed container for external data passed from Data Workspace to widgets.

    This replaces the loose ``{"mode": ..., "data": ..., "using_uploaded": ..., "_format": ...}``
    dict with a validated, self-documenting dataclass.

    Usage::

        ext = ExternalData.from_format("one_sample", {"values": np.array([1,2,3])})
        if ext.using_uploaded:
            payload = ext.get_payload(OneSampleData)
            # payload.values is typed as np.ndarray
    """
    mode: str = "simulated"
    data: Optional[Any] = None
    using_uploaded: bool = False
    _format: str = ""

    @classmethod
    def simulated(cls) -> "ExternalData":
        """Create a simulated (empty) external data."""
        return cls(mode="simulated", data=None, using_uploaded=False, _format="")

    @classmethod
    def from_format(cls, fmt: str, data_payload: Any, **extra) -> "ExternalData":
        """Create uploaded external data from a format type and payload."""
        fmt = fmt.lstrip("_")  # allow "_format" values with or without underscore
        return cls(
            mode="uploaded",
            data=data_payload,
            using_uploaded=True,
            _format=fmt,
            **extra,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ExternalData":
        """Construct from a legacy dict (backward compatible)."""
        if not d:
            return cls.simulated()
        return cls(
            mode=d.get("mode", "simulated"),
            data=d.get("data"),
            using_uploaded=d.get("using_uploaded", False),
            _format=d.get("_format", ""),
        )

    def get_payload(self, payload_type: type) -> Any:
        """Extract and validate the data payload as a specific dataclass.

        Example::

            data = ext.get_payload(OneSampleData)
            # data.values  # typed np.ndarray
        """
        if self.data is None:
            raise ValueError("No data payload available (simulated mode).")
        # If data is already a dict, convert to the target dataclass
        if isinstance(self.data, dict):
            return payload_type(**self.data)
        return self.data

    def to_dict(self) -> dict:
        """Convert back to legacy dict format for backward compatibility."""
        result = {
            "mode": self.mode,
            "data": self.data,
            "using_uploaded": self.using_uploaded,
        }
        if self._format:
            result["_format"] = self._format
        return result

    # ---- dict-compatible interface (backward compat) ----

    def __getitem__(self, key: str) -> Any:
        if key == "_format":
            return self._format
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "_format":
            self._format = value
        else:
            setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key == "_format"

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (AttributeError, KeyError, TypeError):
            return default

    def keys(self) -> Iterator[str]:
        for k in ("mode", "data", "using_uploaded", "_format"):
            yield k


# ---------------------------------------------------------------------------
# Helper functions for working with external_data
# ---------------------------------------------------------------------------

def is_using_external(src) -> bool:
    """Check if a data source dict or ExternalData is using uploaded data."""
    if isinstance(src, ExternalData):
        return src.using_uploaded
    if isinstance(src, dict):
        return src.get("using_uploaded", False)
    return False
