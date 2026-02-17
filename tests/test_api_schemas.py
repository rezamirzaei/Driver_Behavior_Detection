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
    FeatureInfo,
    FeatureRequest,
    FeatureResponse,
    HealthResponse,
    ModelComparisonResponse,
    ModelInfo,
    ModelRequest,
    RegressionDiagnosticsRequest,
    RegressionDiagnosticsResponse,
    TrainingHistoryPayload,
    TrainingHistoryPoint,
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

    def test_health_response(self) -> None:
        health = HealthResponse(status="ok", version="2.0.0", tasks_loaded={"classification": True, "regression": True})
        assert health.tasks_loaded["classification"]
