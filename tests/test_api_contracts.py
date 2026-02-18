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

    def list_artifacts(self, task: str | None = None, limit: int = 30):
        del limit
        return [
            {
                "task": task or "classification",
                "artifact_id": "artifact-001",
                "model_name": "Random Forest",
                "signature": "sig-001",
                "data_version": "abc123",
                "updated_at": "2026-02-18T00:00:01+00:00",
                "artifact_file_path": "results/model_artifacts/classification-artifact-001.joblib",
                "metadata_file_path": "results/model_artifacts/classification-artifact-001.json",
            }
        ]

    def get_artifact(self, task: str, artifact_id: str):
        if task != "classification" or artifact_id != "artifact-001":
            raise ValueError("Artifact not found")
        return {
            "task": "classification",
            "artifact_id": "artifact-001",
            "model_name": "Random Forest",
            "signature": "sig-001",
            "data_version": "abc123",
            "updated_at": "2026-02-18T00:00:01+00:00",
            "artifact_file_path": "results/model_artifacts/classification-artifact-001.joblib",
            "metadata_file_path": "results/model_artifacts/classification-artifact-001.json",
            "feature_names": ["speed_mean", "speed_std"],
            "reference_stats": {"numeric": {"speed_mean": {"mean": 30.0, "std": 5.0}}},
            "result_payload": self.custom_learning(
                task="classification",
                model_name="Random Forest",
                feature_names=["speed_mean", "speed_std"],
            ),
        }

    def predict_with_artifact(self, task: str, artifact_id: str, records: list[dict[str, object]]):
        del records
        if task != "classification" or artifact_id != "artifact-001":
            raise ValueError("Artifact not found")
        return {
            "task": "classification",
            "artifact_id": "artifact-001",
            "n_records": 2,
            "predictions": ["NORMAL", "AGGRESSIVE"],
            "probabilities": [
                {"NORMAL": 0.8, "AGGRESSIVE": 0.1, "DROWSY": 0.1},
                {"NORMAL": 0.1, "AGGRESSIVE": 0.8, "DROWSY": 0.1},
            ],
        }

    def detect_artifact_drift(self, task: str, artifact_id: str, records: list[dict[str, object]]):
        del records
        if task != "classification" or artifact_id != "artifact-001":
            raise ValueError("Artifact not found")
        return {
            "task": "classification",
            "artifact_id": "artifact-001",
            "n_records": 2,
            "overall_drift_score": 0.4,
            "flagged_feature_count": 0,
            "is_drifted": False,
            "alert_id": None,
            "feature_reports": [
                {
                    "feature": "speed_mean",
                    "type": "numeric",
                    "score": 0.4,
                    "flagged": False,
                    "reference_mean": 30.0,
                    "current_mean": 32.0,
                }
            ],
        }

    def list_drift_alerts(
        self,
        task: str | None = None,
        artifact_id: str | None = None,
        limit: int = 50,
        status: str | None = None,
    ):
        del task, artifact_id, limit, status
        return [
            {
                "alert_id": "alert-001",
                "task": "classification",
                "artifact_id": "artifact-001",
                "overall_drift_score": 1.8,
                "flagged_feature_count": 2,
                "status": "open",
                "detected_at": "2026-02-18T00:05:00+00:00",
                "acknowledged_at": None,
                "acknowledged_by": None,
            }
        ]

    def acknowledge_drift_alert(self, *, alert_id: str, acknowledged_by: str) -> bool:
        del acknowledged_by
        return alert_id == "alert-001"

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
    app_module._security_manager = None
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


def test_cancel_job_contract(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        job_response = client.post(
            "/api/model/custom-learning/job",
            json={
                "task": "classification",
                "model_name": "Random Forest",
                "feature_names": ["speed_mean", "speed_std"],
            },
        )
        assert job_response.status_code == 200
        job_id = job_response.json()["job_id"]

        cancel_response = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] in {"cancel_requested", "canceled", "completed", "failed"}


def test_artifact_endpoints_contract(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        list_response = client.get("/api/artifacts", params={"task": "classification"})
        assert list_response.status_code == 200
        artifacts = list_response.json()["artifacts"]
        assert artifacts[0]["artifact_id"] == "artifact-001"

        detail_response = client.get("/api/artifacts/classification/artifact-001")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["model_name"] == "Random Forest"
        assert detail["feature_names"] == ["speed_mean", "speed_std"]

        predict_response = client.post(
            "/api/artifacts/classification/artifact-001/predict",
            json={"records": [{"speed_mean": 20.0}, {"speed_mean": 40.0}]},
        )
        assert predict_response.status_code == 200
        assert predict_response.json()["n_records"] == 2

        drift_response = client.post(
            "/api/artifacts/classification/artifact-001/drift",
            json={"records": [{"speed_mean": 20.0}, {"speed_mean": 40.0}]},
        )
        assert drift_response.status_code == 200
        assert drift_response.json()["is_drifted"] is False


def test_drift_alert_endpoints_contract(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        list_response = client.get(
            "/api/drift-alerts", params={"task": "classification", "artifact_id": "artifact-001"}
        )
        assert list_response.status_code == 200
        alerts = list_response.json()["alerts"]
        assert alerts[0]["alert_id"] == "alert-001"

        ack_response = client.post("/api/drift-alerts/alert-001/ack", json={"acknowledged_by": "qa"})
        assert ack_response.status_code == 200
        assert ack_response.json()["status"] == "acknowledged"
