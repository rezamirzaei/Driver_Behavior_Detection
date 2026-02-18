"""Tests for sqlite-backed training run repository."""

from __future__ import annotations

from pathlib import Path

from src.api.run_repository import TrainingRunRepository


def test_cache_roundtrip(tmp_path: Path) -> None:
    repo = TrainingRunRepository(f"sqlite:///{tmp_path / 'runs.sqlite'}")
    payload = {"task": "classification", "validation_metrics": {"accuracy": 0.82}}

    repo.set_cached_result(
        signature="sig-1",
        task="classification",
        operation="custom_learning",
        model_name="Random Forest",
        feature_names=["speed_mean", "speed_std"],
        params={"cv_folds": 3},
        data_version="data-v1",
        result_payload=payload,
    )

    cached = repo.get_cached_result("sig-1")
    assert cached is not None
    assert cached["validation_metrics"]["accuracy"] == 0.82


def test_run_lifecycle_roundtrip(tmp_path: Path) -> None:
    repo = TrainingRunRepository(f"sqlite:///{tmp_path / 'runs.sqlite'}")

    run_id = repo.start_run(
        signature="sig-2",
        task="classification",
        operation="custom_learning",
        model_name="Random Forest",
        feature_names=["speed_mean"],
        params={"cv_folds": 1},
        data_version="data-v1",
    )

    result_payload = {
        "task": "classification",
        "model_name": "Random Forest",
        "validation_metrics": {"accuracy": 0.84},
    }
    repo.complete_run(
        run_id=run_id,
        cache_hit=False,
        duration_ms=123.0,
        metrics={"validation": {"accuracy": 0.84}},
        cv_summary={},
        artifact_id="",
        artifact_path="",
        result_payload=result_payload,
    )

    rows = repo.list_runs(task="classification", limit=10)
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["status"] == "completed"
    assert rows[0]["feature_count"] == 1

    detail = repo.get_run(run_id)
    assert detail is not None
    assert detail["result"]["validation_metrics"]["accuracy"] == 0.84


def test_artifact_roundtrip(tmp_path: Path) -> None:
    repo = TrainingRunRepository(f"sqlite:///{tmp_path / 'runs.sqlite'}")
    repo.upsert_model_artifact(
        task="classification",
        artifact_id="artifact-1",
        model_name="Random Forest",
        signature="sig-1",
        data_version="data-v1",
        feature_names=["speed_mean", "speed_std"],
        reference_stats={"numeric": {"speed_mean": {"mean": 30.0, "std": 5.0}}},
        artifact_file_path="results/model_artifacts/classification-artifact-1.joblib",
        metadata_file_path="results/model_artifacts/classification-artifact-1.json",
        result_payload={"validation_metrics": {"accuracy": 0.8}},
    )

    rows = repo.list_model_artifacts(task="classification", limit=10)
    assert len(rows) == 1
    assert rows[0]["artifact_id"] == "artifact-1"

    detail = repo.get_model_artifact(task="classification", artifact_id="artifact-1")
    assert detail is not None
    assert detail["feature_names"] == ["speed_mean", "speed_std"]
    assert detail["result_payload"]["validation_metrics"]["accuracy"] == 0.8
