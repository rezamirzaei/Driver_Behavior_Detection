"""Auth and quota contract tests for API middleware."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from fastapi.testclient import TestClient

import src.api.app as app_module


class _FakeService:
    runtimes = {"classification": object(), "regression": object()}

    def custom_learning(
        self,
        task: str,
        model_name: str,
        feature_names: list[str],
        cv_folds: int = 1,
        persist_artifact: bool = False,
        artifact_id: str | None = None,
    ):
        del task, model_name, feature_names, cv_folds, persist_artifact, artifact_id
        return {}

    def metadata(self, task: str):
        return {
            "task": task,
            "dataset_name": "Fake Dataset",
            "target_name": "target",
            "n_rows": 10,
            "n_features": 2,
            "n_numeric_features": 2,
            "features": [
                {
                    "name": "feature_a",
                    "description": "Feature A",
                    "is_numeric": True,
                    "source_type": "processed",
                    "source_summary": "Processed",
                    "lineage": "mock lineage",
                },
                {
                    "name": "feature_b",
                    "description": "Feature B",
                    "is_numeric": True,
                    "source_type": "processed",
                    "source_summary": "Processed",
                    "lineage": "mock lineage",
                },
            ],
            "models": ["Random Forest"],
            "model_details": [
                {
                    "name": "Random Forest",
                    "task": task,
                    "family": "Ensemble",
                    "description": "Mock model",
                    "supports_staged_predictions": False,
                }
            ],
        }


@contextmanager
def _auth_client(monkeypatch) -> Iterator[TestClient]:
    fake_service = _FakeService()
    monkeypatch.setattr(app_module, "build_service", lambda _: fake_service)
    app_module._service = None
    app_module._job_manager = None
    app_module._observability = None
    app_module._security_manager = None

    settings = app_module.settings
    original_api_auth_enabled = settings.api_auth_enabled
    original_api_keys = list(settings.api_keys)
    original_api_key_header = settings.api_key_header
    original_api_quota_per_minute = settings.api_quota_per_minute
    original_api_auth_exempt_paths = list(settings.api_auth_exempt_paths)
    settings.api_auth_enabled = True
    settings.api_keys = ["test-key"]
    settings.api_key_header = "X-API-Key"
    settings.api_quota_per_minute = 2
    settings.api_auth_exempt_paths = ["/api/health"]

    try:
        with TestClient(app_module.app) as client:
            yield client
    finally:
        settings.api_auth_enabled = original_api_auth_enabled
        settings.api_keys = original_api_keys
        settings.api_key_header = original_api_key_header
        settings.api_quota_per_minute = original_api_quota_per_minute
        settings.api_auth_exempt_paths = original_api_auth_exempt_paths
        app_module._service = None
        app_module._job_manager = None
        app_module._observability = None
        app_module._security_manager = None


def test_auth_rejects_missing_or_invalid_key(monkeypatch) -> None:
    with _auth_client(monkeypatch) as client:
        health = client.get("/api/health")
        assert health.status_code == 200

        missing = client.get("/api/metadata", params={"task": "classification"})
        assert missing.status_code == 401

        invalid = client.get("/api/metadata", params={"task": "classification"}, headers={"X-API-Key": "wrong"})
        assert invalid.status_code == 401


def test_auth_quota_limit(monkeypatch) -> None:
    headers = {"X-API-Key": "test-key"}
    with _auth_client(monkeypatch) as client:
        ok1 = client.get("/api/metadata", params={"task": "classification"}, headers=headers)
        ok2 = client.get("/api/metadata", params={"task": "classification"}, headers=headers)
        limited = client.get("/api/metadata", params={"task": "classification"}, headers=headers)

    assert ok1.status_code == 200
    assert ok2.status_code == 200
    assert limited.status_code == 429
    assert limited.headers.get("Retry-After") is not None
