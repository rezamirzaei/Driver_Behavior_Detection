"""Tests for classification dataset cache write fallback behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.classification import data as classification_data


def test_load_or_build_dataset_does_not_raise_on_cache_write_failure(monkeypatch, tmp_path) -> None:
    """Cache write errors should not crash dataset build in read-only environments."""
    sample_df = pd.DataFrame(
        [
            {"driver": "D1", "behavior": "NORMAL", "road_type": "MOTORWAY", "speed_mean": 30.0},
            {"driver": "D2", "behavior": "DROWSY", "road_type": "SECONDARY", "speed_mean": 22.0},
        ]
    )

    monkeypatch.setattr(classification_data, "get_all_trips", lambda _: [object(), object()])
    monkeypatch.setattr(classification_data, "build_raw_dataset", lambda trips: sample_df.copy())
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, index=False: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )

    result = classification_data.load_or_build_dataset(
        data_dir=tmp_path,
        cache_path=Path("/app/data/processed/uah_raw_features.csv"),
        force_rebuild=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "speed_mean" in result.columns
