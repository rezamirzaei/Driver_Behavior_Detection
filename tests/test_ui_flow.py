"""UI flow contract: feature subset -> train job -> diagnostics payload."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

import src.api.app as app_module


class _UiFlowFakeService:
    runtimes = {"classification": object(), "regression": object()}

    def metadata(self, task: str):
        return {
            "task": task,
            "dataset_name": "UAH-DriveSet",
            "target_name": "behavior",
            "n_rows": 40,
            "n_features": 12,
            "n_numeric_features": 12,
            "features": self.list_feature_payload(task),
            "models": self.list_models(task),
            "model_details": self.list_model_payload(task),
        }

    def list_feature_payload(self, task: str):
        del task
        return [
            {
                "name": "speed_mean",
                "description": "Average speed",
                "is_numeric": True,
                "source_type": "processed",
                "source_summary": "Processed feature",
                "lineage": "RAW_GPS.speed mean over trip samples.",
            },
            {
                "name": "speed_std",
                "description": "Speed standard deviation",
                "is_numeric": True,
                "source_type": "processed",
                "source_summary": "Processed feature",
                "lineage": "RAW_GPS.speed std over trip samples.",
            },
        ]

    def list_models(self, task: str):
        del task
        return ["Random Forest"]

    def list_model_payload(self, task: str):
        del task
        return [
            {
                "name": "Random Forest",
                "task": "classification",
                "family": "Ensemble Tree",
                "description": "Bagged trees",
                "supports_staged_predictions": False,
            }
        ]

    def custom_learning(
        self,
        task: str,
        model_name: str,
        feature_names: list[str],
        cv_folds: int = 1,
        persist_artifact: bool = False,
        artifact_id: str | None = None,
    ):
        del artifact_id, cv_folds, persist_artifact
        return {
            "task": task,
            "model_name": model_name,
            "selected_features": self.list_feature_payload(task),
            "train_metrics": {"accuracy": 0.91},
            "validation_metrics": {"accuracy": 0.84},
            "explanation": f"trained {model_name} on {len(feature_names)} feature(s)",
            "train_confusion_matrix_url": "data:image/png;base64,abc",
            "validation_confusion_matrix_url": "data:image/png;base64,abc",
            "validation_diagnostics_plot_url": "",
            "error_plot_url": "data:image/png;base64,abc",
            "training_history": {
                "score_metric": "accuracy",
                "error_metric": "classification_error",
                "points": [
                    {
                        "iteration": 1,
                        "train_score": 0.85,
                        "validation_score": 0.8,
                        "train_error": 0.15,
                        "validation_error": 0.2,
                    }
                ],
            },
            "cv_metric_name": "",
            "cv_scores": [],
            "cv_mean": None,
            "cv_std": None,
            "artifact_id": "",
            "artifact_saved": False,
            "artifact_path": "",
        }

    def list_training_runs(self, task: str | None = None, limit: int = 30):
        del limit
        return [
            {
                "run_id": "run-ui-1",
                "signature": "sig-ui-1",
                "task": task or "classification",
                "operation": "custom_learning",
                "model_name": "Random Forest",
                "status": "completed",
                "cache_hit": False,
                "started_at": "2026-02-18T00:00:00+00:00",
                "finished_at": "2026-02-18T00:00:01+00:00",
                "duration_ms": 50.0,
                "artifact_id": "",
                "artifact_path": "",
                "feature_count": 2,
                "metrics": {"validation": {"accuracy": 0.84}},
                "cv_summary": {},
                "error": None,
            }
        ]

    def get_training_run(self, run_id: str):
        if run_id != "run-ui-1":
            raise ValueError("Run not found")
        return {
            **self.list_training_runs(task="classification")[0],
            "feature_names": ["speed_mean", "speed_std"],
            "params": {"cv_folds": 1},
            "data_version": "ui-test",
            "result": self.custom_learning(
                task="classification",
                model_name="Random Forest",
                feature_names=["speed_mean", "speed_std"],
            ),
        }


def _client(monkeypatch):
    fake_service = _UiFlowFakeService()
    monkeypatch.setattr(app_module, "build_service", lambda _: fake_service)
    app_module._service = None
    app_module._job_manager = None
    app_module._observability = None
    return TestClient(app_module.app)


def test_ui_flow_contract(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        index_response = client.get("/")
        assert index_response.status_code == 200
        html = index_response.text
        assert "Learning Studio" in html
        assert "studioFeatureQuery" in html
        assert "studioSourceFilter" in html
        assert "Recent Runs" in html

        metadata_response = client.get("/api/metadata", params={"task": "classification"})
        assert metadata_response.status_code == 200
        metadata_payload = metadata_response.json()
        assert metadata_payload["features"][0]["lineage"]

        job_response = client.post(
            "/api/model/custom-learning/job",
            json={
                "task": "classification",
                "model_name": "Random Forest",
                "feature_names": ["speed_mean", "speed_std"],
                "cv_folds": 1,
            },
        )
        assert job_response.status_code == 200
        job_id = job_response.json()["job_id"]

        final_payload = None
        for _ in range(20):
            poll_response = client.get(f"/api/jobs/{job_id}")
            assert poll_response.status_code == 200
            final_payload = poll_response.json()
            if final_payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)

        assert final_payload is not None
        assert final_payload["status"] == "completed"
        result = final_payload["result"]
        assert result["train_confusion_matrix_url"].startswith("data:image/png;base64,")
        assert result["validation_confusion_matrix_url"].startswith("data:image/png;base64,")
        assert result["error_plot_url"].startswith("data:image/png;base64,")

        history_response = client.get("/api/training-runs", params={"task": "classification"})
        assert history_response.status_code == 200
        assert history_response.json()["runs"][0]["run_id"] == "run-ui-1"
