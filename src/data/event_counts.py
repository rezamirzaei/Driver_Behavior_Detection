"""Utilities for counting threshold-driven driving events."""

from __future__ import annotations

from typing import Literal

import pandas as pd


def count_event_starts(mask: pd.Series) -> int:
    """Count contiguous True segments as distinct events."""
    active = pd.Series(mask, copy=False).fillna(False).astype(bool)
    starts = active & ~active.shift(1, fill_value=False)
    return int(starts.sum())


def count_threshold_events(values: pd.Series, threshold: float, direction: Literal["below", "above"]) -> int:
    """Count events where a signal crosses into a threshold region."""
    signal = pd.to_numeric(values, errors="coerce")
    if direction == "below":
        return count_event_starts(signal < threshold)
    return count_event_starts(signal > threshold)
