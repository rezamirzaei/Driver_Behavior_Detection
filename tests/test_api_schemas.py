"""Tests for API schema validation."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from src.api.schemas import (
    ALLOWED_FEATURES_BY_TASK,
    ALLOWED_MODELS_BY_TASK,
    ConfusionMatrixRequest,
    ConfusionMatrixResponse,
    CorrelationMatrixResponse,
    CustomLearningRequest,
    CustomLearningResponse,
    FeatureInfo,
    FeatureRequest,
    FeatureResponse,
    HealthResponse,
    ModelComparisonResponse,
    ModelInfo,
    ModelRequest,
    ObservabilityMetricsResponse,
    RegressionDiagnosticsRequest,
    RegressionDiagnosticsResponse,
    TrainingHistoryPayload,
    TrainingHistoryPoint,
    TrainingRunDetailResponse,
    TrainingRunListResponse,
    TrainingRunSummary,
    TwoFeatureRequest,
    TwoFeatureResponse,
)


def _first_feature(task: str) -> str:
    return ALLOWED_FEATURES_BY_TASK[task][0]


def _first_model(task: str) -> str:
    return ALLOWED_MODELS_BY_TASK[task][0]


class TestFeatureRequest:
    def test_valid_feature_request(self) -> None:
        req = FeatureRequest(task="classification", feature_name=_first_feature("classification"))
        assert req.task == "classification"

    def test_invalid_feature_request(self) -> None:
        with pytest.raises(ValidationError):
            FeatureRequest(task="classification", feature_name="not_a_real_feature")


class TestTwoFeatureRequest:
    def test_valid_numeric_pair(self) -> None:
        numeric_features = ALLOWED_FEATURES_BY_TASK["classification"][:2]
        req = TwoFeatureRequest(task="classification", feature1=numeric_features[0], feature2=numeric_features[1])
        assert req.feature1 != req.feature2

    def test_same_feature_rejected(self) -> None:
        same_feature = _first_feature("classification")
        with pytest.raises(ValidationError):
            TwoFeatureRequest(task="classification", feature1=same_feature, feature2=same_feature)


class TestModelRequest:
    def test_classification_features_exclude_removed_hard_brake_count(self) -> None:
        features = ALLOWED_FEATURES_BY_TASK["classification"]
        assert "hard_brake_count" not in features
        assert "hard_break_count" not in features

    def test_classification_models_include_notebook_set(self) -> None:
        models = ALLOWED_MODELS_BY_TASK["classification"]
        assert len(models) >= 18
        assert "Naive Bayes" in models
        assert "Logistic (SCAD)" in models
        assert "Gradient Boosting" in models

    def test_valid_model_request(self) -> None:
        req = ModelRequest(task="regression", model_name=_first_model("regression"))
        assert req.model_name in ALLOWED_MODELS_BY_TASK["regression"]

    def test_invalid_model_request(self) -> None:
        with pytest.raises(ValidationError):
            ModelRequest(task="regression", model_name="invalid_model")


class TestSpecializedModelRequests:
    def test_confusion_matrix_request_requires_classification(self) -> None:
        with pytest.raises(ValidationError):
            ConfusionMatrixRequest(task="regression", model_name=_first_model("regression"))

    def test_regression_request_requires_regression(self) -> None:
        with pytest.raises(ValidationError):
            RegressionDiagnosticsRequest(task="classification", model_name=_first_model("classification"))

    def test_custom_learning_request_validates_feature_and_model(self) -> None:
        req = CustomLearningRequest(
            task="classification",
            model_name=_first_model("classification"),
            feature_names=ALLOWED_FEATURES_BY_TASK["classification"][:3],
        )
        assert len(req.feature_names) == 3

    def test_custom_learning_request_rejects_invalid_feature(self) -> None:
        with pytest.raises(ValidationError):
            CustomLearningRequest(
                task="classification",
                model_name=_first_model("classification"),
                feature_names=["not_a_real_feature"],
            )

    def test_custom_learning_request_accepts_regression(self) -> None:
        req = CustomLearningRequest(
            task="regression",
            model_name=_first_model("regression"),
            feature_names=ALLOWED_FEATURES_BY_TASK["regression"][:3],
            cv_folds=3,
        )
        assert req.task == "regression"

    def test_custom_learning_request_accepts_artifact_without_features(self) -> None:
        req = CustomLearningRequest(
            task="classification",
            model_name=_first_model("classification"),
            feature_names=[],
            artifact_id="abc123",
        )
        assert req.artifact_id == "abc123"

    def test_custom_learning_request_rejects_invalid_cv_folds(self) -> None:
        with pytest.raises(ValidationError):
            CustomLearningRequest(
                task="classification",
                model_name=_first_model("classification"),
                feature_names=ALLOWED_FEATURES_BY_TASK["classification"][:2],
                cv_folds=0,
            )


class TestResponseModels:
    def test_feature_response(self) -> None:
        payload = FeatureResponse(
            task="classification",
            feature_name="speed_mean",
            statistics={"mean": 1.0},
            explanation="ok",
            plot_url="data:image/png;base64,abc",
        )
        assert payload.task == "classification"

    def test_two_feature_response(self) -> None:
        payload = TwoFeatureResponse(
            task="classification",
            feature1="speed_mean",
            feature2="speed_std",
            correlation=0.5,
            explanation="ok",
            plot_url="data:image/png;base64,abc",
        )
        assert payload.correlation == 0.5

    def test_correlation_matrix_response(self) -> None:
        payload = CorrelationMatrixResponse(
            task="regression",
            matrix=[[1.0]],
            feature_names=["f1"],
            high_correlation_pairs=[],
            explanation="ok",
            plot_url="data:image/png;base64,abc",
        )
        assert payload.task == "regression"

    def test_confusion_matrix_response(self) -> None:
        payload = ConfusionMatrixResponse(
            task="classification",
            model_name="Random Forest",
            accuracy=0.95,
            metrics={"f1_score": 0.9},
            explanation="ok",
            plot_url="data:image/png;base64,abc",
        )
        assert payload.accuracy > 0.0

    def test_regression_diagnostics_response(self) -> None:
        payload = RegressionDiagnosticsResponse(
            task="regression",
            model_name="Ridge",
            metrics={"r2": 0.8, "rmse": 3.2, "mae": 2.1},
            explanation="ok",
            plot_url="data:image/png;base64,abc",
        )
        assert "r2" in payload.metrics

    def test_model_comparison_response(self) -> None:
        payload = ModelComparisonResponse(
            task="classification",
            results=[{"Model": "Random Forest", "F1 Score": 0.9}],
            best_model="Random Forest",
            ranking_metric="F1 Score",
            explanation="ok",
            plot_url="data:image/png;base64,abc",
        )
        assert payload.best_model == "Random Forest"

    def test_feature_info(self) -> None:
        feature = FeatureInfo(name="speed_mean", description="Average speed", is_numeric=True)
        assert feature.is_numeric is True
        assert feature.source_type in {"raw", "processed"}
        assert feature.lineage == ""

    def test_model_info(self) -> None:
        model = ModelInfo(
            name="Random Forest",
            task="classification",
            family="Ensemble Tree",
            description="Tree ensemble.",
            supports_staged_predictions=False,
        )
        assert model.task == "classification"

    def test_training_history_payload(self) -> None:
        payload = TrainingHistoryPayload(
            score_metric="accuracy",
            error_metric="classification_error",
            points=[
                TrainingHistoryPoint(
                    iteration=1,
                    train_score=0.9,
                    validation_score=0.8,
                    train_error=0.1,
                    validation_error=0.2,
                )
            ],
        )
        assert payload.points[0].iteration == 1

    def test_custom_learning_response(self) -> None:
        payload = CustomLearningResponse(
            task="classification",
            model_name="Random Forest",
            selected_features=[FeatureInfo(name="speed_mean", description="Average speed", is_numeric=True)],
            train_metrics={"accuracy": 0.99},
            validation_metrics={"accuracy": 0.95},
            explanation="ok",
            train_confusion_matrix_url="data:image/png;base64,abc",
            validation_confusion_matrix_url="data:image/png;base64,abc",
            error_plot_url="data:image/png;base64,abc",
        )
        assert payload.task == "classification"
        assert payload.selected_features[0].name == "speed_mean"

    def test_custom_learning_response_for_regression(self) -> None:
        payload = CustomLearningResponse(
            task="regression",
            model_name="Ridge (L2)",
            selected_features=[FeatureInfo(name="year", description="Model year", is_numeric=True, source_type="raw")],
            train_metrics={"r2": 0.8},
            validation_metrics={"r2": 0.75},
            explanation="ok",
            validation_diagnostics_plot_url="data:image/png;base64,abc",
            error_plot_url="data:image/png;base64,abc",
        )
        assert payload.task == "regression"
        assert payload.validation_diagnostics_plot_url

    def test_health_response(self) -> None:
        health = HealthResponse(status="ok", version="2.0.0", tasks_loaded={"classification": True, "regression": True})
        assert health.tasks_loaded["classification"]

    def test_training_run_models(self) -> None:
        summary = TrainingRunSummary(
            run_id="run-1",
            signature="sig-1",
            task="classification",
            operation="custom_learning",
            model_name="Random Forest",
            status="completed",
            cache_hit=True,
            started_at="2026-02-18T00:00:00+00:00",
            finished_at="2026-02-18T00:00:01+00:00",
            duration_ms=100.0,
            feature_count=2,
            metrics={"validation": {"accuracy": 0.8}},
            cv_summary={},
        )
        wrapper = TrainingRunListResponse(runs=[summary])
        assert wrapper.runs[0].run_id == "run-1"

        detail = TrainingRunDetailResponse(
            **summary.model_dump(),
            feature_names=["speed_mean", "speed_std"],
            params={"cv_folds": 3},
            data_version="hash-1",
            result={"model_name": "Random Forest"},
        )
        assert detail.feature_names[0] == "speed_mean"

    def test_observability_metrics_response(self) -> None:
        payload = ObservabilityMetricsResponse(
            started_at="2026-02-18T00:00:00+00:00",
            generated_at="2026-02-18T00:00:01+00:00",
            requests={"GET /api/health": {"count": 1, "avg_duration_ms": 10.0}},
            training={},
        )
        assert "GET /api/health" in payload.requests
