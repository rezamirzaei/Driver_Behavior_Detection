"""Tests for reusable Pydantic-based sample validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import ValidationError
import pytest

from src.data.sample_models import ClassificationTripSample, EPAVehicleSample, TripInfoSample, UAHRawGPSSample
from src.data.validation import validate_dataframe_records, validate_record


def test_trip_info_sample_normalizes_casing() -> None:
    sample = TripInfoSample.model_validate(
        {
            "path": Path("trip"),
            "driver": "d1",
            "behavior": "normal",
            "road_type": "motorway",
        }
    )
    assert sample.driver == "D1"
    assert sample.behavior == "NORMAL"
    assert sample.road_type == "MOTORWAY"


def test_validate_record_raises_when_strict() -> None:
    with pytest.raises(ValueError):
        validate_record({"timestamp": "bad", "speed": 10}, UAHRawGPSSample, strict=True, context="gps")


def test_validate_dataframe_drops_invalid_rows_when_not_strict() -> None:
    df = pd.DataFrame(
        [
            {"timestamp": 1.0, "speed": 50.0, "lat": 10.0},
            {"timestamp": "invalid", "speed": 55.0, "lat": 11.0},
        ]
    )
    with pytest.warns(UserWarning, match="dropped 1 invalid row"):
        validated = validate_dataframe_records(df, UAHRawGPSSample, strict=False, context="gps_df")
    assert len(validated) == 1
    assert validated.iloc[0]["speed"] == 50.0


def test_classification_trip_sample_validates_counts_non_negative() -> None:
    payload = {
        "driver": "D1",
        "behavior": "NORMAL",
        "road_type": "SECONDARY",
        "speed_mean": 40.0,
        "brake_count": 2,
    }
    parsed = ClassificationTripSample.model_validate(payload)
    assert parsed.brake_count == 2

    with pytest.raises(ValidationError):
        ClassificationTripSample.model_validate({**payload, "brake_count": -1})


def test_epa_vehicle_sample_requires_valid_target_range() -> None:
    ok = EPAVehicleSample.model_validate({"comb08": 35.5, "year": 2020, "make": "Ford"})
    assert ok.comb08 == 35.5

    with pytest.raises(ValidationError):
        EPAVehicleSample.model_validate({"comb08": 0})
