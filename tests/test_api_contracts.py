"""API contract tests for custom-learning sync and async flows."""

from __future__ import annotations

import time

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
        loaded_artifact = bool(artifact_id)
        final_artifact_id = artifact_id or "artifact-001"
        selected_features = [
            {
                "name": name,
                "description": f"Feature {name}",
                "is_numeric": True,
                "source_type": "processed",
                "source_summary": "Processed feature",
            }
            for name in feature_names
        ]
        return {
            "task": task,
            "model_name": model_name,
            "selected_features": selected_features,
            "train_metrics": {"accuracy": 0.9} if task == "classification" else {"r2": 0.7},
            "validation_metrics": {"accuracy": 0.82} if task == "classification" else {"r2": 0.63},
            "explanation": f"trained {model_name}",
            "train_confusion_matrix_url": "data:image/png;base64,abc" if task == "classification" else "",
            "validation_confusion_matrix_url": "data:image/png;base64,abc" if task == "classification" else "",
            "validation_diagnostics_plot_url": "data:image/png;base64,abc" if task == "regression" else "",
            "error_plot_url": "data:image/png;base64,abc",
            "training_history": {
                "score_metric": "accuracy" if task == "classification" else "r2",
                "error_metric": "classification_error" if task == "classification" else "one_minus_r2",
                "points": [
                    {
                        "iteration": 1,
                        "train_score": 0.8,
                        "validation_score": 0.75,
                        "train_error": 0.2,
                        "validation_error": 0.25,
                    }
                ],
            },
            "cv_metric_name": "f1_weighted" if task == "classification" else "r2",
            "cv_scores": [0.8, 0.82, 0.79] if cv_folds > 1 else [],
            "cv_mean": 0.8033 if cv_folds > 1 else None,
            "cv_std": 0.0125 if cv_folds > 1 else None,
            "artifact_id": final_artifact_id if (persist_artifact or loaded_artifact) else "",
            "artifact_saved": bool(persist_artifact or loaded_artifact),
            "artifact_path": f"results/model_artifacts/{task}-{final_artifact_id}.json"
            if (persist_artifact or loaded_artifact)
            else "",
        }

    def list_training_runs(self, task: str | None = None, limit: int = 30):
        del limit
        return [
            {
                "run_id": "run-001",
                "signature": "sig-001",
                "task": task or "classification",
                "operation": "custom_learning",
                "model_name": "Random Forest",
                "status": "completed",
                "cache_hit": True,
                "started_at": "2026-02-18T00:00:00+00:00",
                "finished_at": "2026-02-18T00:00:01+00:00",
                "duration_ms": 100.0,
                "artifact_id": "artifact-001",
                "artifact_path": "results/model_artifacts/classification-artifact-001.json",
                "feature_count": 2,
                "metrics": {"validation": {"accuracy": 0.82}},
                "cv_summary": {"metric_name": "f1_weighted"},
                "error": None,
            }
        ]

    def get_training_run(self, run_id: str):
        if run_id != "run-001":
            raise ValueError("Run not found")
        return {
            "run_id": "run-001",
            "signature": "sig-001",
            "task": "classification",
            "operation": "custom_learning",
            "model_name": "Random Forest",
            "status": "completed",
            "cache_hit": True,
            "started_at": "2026-02-18T00:00:00+00:00",
            "finished_at": "2026-02-18T00:00:01+00:00",
            "duration_ms": 100.0,
            "artifact_id": "artifact-001",
            "artifact_path": "results/model_artifacts/classification-artifact-001.json",
            "feature_count": 2,
            "metrics": {"validation": {"accuracy": 0.82}},
            "cv_summary": {"metric_name": "f1_weighted"},
            "error": None,
            "feature_names": ["speed_mean", "speed_std"],
            "params": {"cv_folds": 3},
            "data_version": "abc123",
            "result": self.custom_learning(
                task="classification",
                model_name="Random Forest",
                feature_names=["speed_mean", "speed_std"],
                cv_folds=3,
            ),
        }


def _client(monkeypatch):
    fake_service = _FakeService()
    monkeypatch.setattr(app_module, "build_service", lambda _: fake_service)
    app_module._service = None
    app_module._job_manager = None
    app_module._observability = None
    return TestClient(app_module.app)


def test_custom_learning_sync_contract(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        response = client.post(
            "/api/model/custom-learning",
            json={
                "task": "classification",
                "model_name": "Random Forest",
                "feature_names": ["speed_mean", "speed_std"],
                "cv_folds": 3,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "classification"
    assert payload["model_name"] == "Random Forest"
    assert len(payload["selected_features"]) == 2
    assert payload["cv_metric_name"] == "f1_weighted"
    assert isinstance(payload["cv_scores"], list)


def test_custom_learning_job_flow_contract(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        job_response = client.post(
            "/api/model/custom-learning/job",
            json={
                "task": "regression",
                "model_name": "Ridge (L2)",
                "feature_names": ["year", "displ"],
                "cv_folds": 3,
            },
        )
        assert job_response.status_code == 200
        job_id = job_response.json()["job_id"]

        status_payload = None
        for _ in range(15):
            poll = client.get(f"/api/jobs/{job_id}")
            assert poll.status_code == 200
            status_payload = poll.json()
            if status_payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["result"]["task"] == "regression"
    assert status_payload["result"]["validation_diagnostics_plot_url"].startswith("data:image/png;base64,")


def test_training_runs_contract(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        list_response = client.get("/api/training-runs", params={"task": "classification", "limit": 10})
        assert list_response.status_code == 200
        runs = list_response.json()["runs"]
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-001"

        detail_response = client.get("/api/training-runs/run-001")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["run_id"] == "run-001"
        assert detail["result"]["model_name"] == "Random Forest"


def test_observability_metrics_contract(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        _ = client.get("/api/health")
        response = client.get("/api/observability/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert "requests" in payload
    assert isinstance(payload["requests"], dict)
