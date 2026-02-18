"""Tests for threshold-event counting utilities and feature extraction integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.classification import data as classification_data
from src.data import raw_loader
from src.data.event_counts import (
    count_event_starts,
    count_threshold_events,
)


def test_count_event_starts_counts_contiguous_segments() -> None:
    mask = pd.Series([False, True, True, False, True, False, True, True])
    assert count_event_starts(mask) == 3


def test_count_threshold_events_handles_above_and_below() -> None:
    signal = pd.Series([np.nan, -0.2, -0.25, -0.05, -0.31, -0.32, 0.0, 0.2, 0.25, 0.0, 0.15])
    assert count_threshold_events(signal, threshold=-0.1, direction="below") == 2
    assert count_threshold_events(signal, threshold=-0.3, direction="below") == 1
    assert count_threshold_events(signal, threshold=0.1, direction="above") == 2


def test_classification_extract_raw_features_counts_event_starts(monkeypatch) -> None:
    gps_df = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3, 4, 5],
            "speed": [30, 31, 32, 33, 34, 35],
            "course": [0, 1, 2, 3, 2, 1],
        }
    )
    acc_df = pd.DataFrame(
        {
            "acc_x_kf": [-0.2, -0.25, -0.05, -0.35, -0.36, 0.0, 0.2, 0.25, 0.0, 0.15],
            "acc_y_kf": [0.0, 0.2, 0.25, 0.0, 0.35, 0.36, 0.0, 0.0, 0.2, 0.0],
            "acc_z_kf": [0.0] * 10,
        }
    )

    monkeypatch.setattr(classification_data, "load_raw_gps", lambda _: gps_df)
    monkeypatch.setattr(classification_data, "load_raw_accelerometer", lambda _: acc_df)

    features = classification_data.extract_raw_features(Path("trip"))
    assert features["brake_count"] == 2
    assert "hard_brake_count" not in features
    assert features["accel_count"] == 2
    assert features["turn_count"] == 3
    # With SHARP_TURN_THRESHOLD=0.2, values 0.25 (event 1) and 0.35,0.36 (event 2) = 2 events
    assert features["sharp_turn_count"] == 2


def test_raw_loader_extract_raw_features_counts_event_starts(monkeypatch) -> None:
    gps_df = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3, 4, 5],
            "speed": [30, 31, 32, 33, 34, 35],
            "diff_course": [0, 1, 2, 3, 2, 1],
        }
    )
    acc_df = pd.DataFrame(
        {
            "acc_x_kf": [-0.2, -0.25, -0.05, -0.35, -0.36, 0.0, 0.2, 0.25, 0.0, 0.15],
            "acc_y_kf": [0.0, 0.2, 0.25, 0.0, 0.35, 0.36, 0.0, 0.0, 0.2, 0.0],
            "acc_z_kf": [0.0] * 10,
        }
    )

    monkeypatch.setattr(raw_loader, "load_raw_gps", lambda _: gps_df)
    monkeypatch.setattr(raw_loader, "load_raw_accelerometer", lambda _: acc_df)
    monkeypatch.setattr(raw_loader, "load_inertial_events", lambda _: None)

    features = raw_loader.extract_raw_features(Path("trip"))
    assert features["brake_count"] == 2
    assert "hard_brake_count" not in features
    assert features["accel_count"] == 2
    assert features["turn_count"] == 3
    # With SHARP_TURN_THRESHOLD=0.2, values 0.25 (event 1) and 0.35,0.36 (event 2) = 2 events
    assert features["sharp_turn_count"] == 2
