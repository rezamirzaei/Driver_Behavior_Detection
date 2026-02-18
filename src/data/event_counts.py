"""Utilities for counting threshold-driven driving events."""

from __future__ import annotations

from typing import Literal

import pandas as pd

# Threshold constants for event detection (in g-force units)
# Based on analysis of UAH-DriveSet accelerometer data:
# - acc_x_kf values typically range from -0.05 to +0.05 g
# - acc_x_std is around 0.02-0.04 g
BRAKE_THRESHOLD = -0.1  # Moderate braking
ACCEL_THRESHOLD = 0.1  # Acceleration event
TURN_THRESHOLD = 0.1  # Moderate turning (lateral)
SHARP_TURN_THRESHOLD = 0.2  # Sharp turning (was 0.3, adjusted for dataset)


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
