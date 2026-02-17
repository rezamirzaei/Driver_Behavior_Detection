"""Tests for API schema validation and endpoint structure."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from src.api.schemas import (
    ALLOWED_FEATURES,
    ALLOWED_MODELS,
    ConfusionMatrixResponse,
    CorrelationMatrixResponse,
    FeatureRequest,
    FeatureResponse,
    HealthResponse,
    ModelComparisonResponse,
    ModelRequest,
    TwoFeatureRequest,
    TwoFeatureResponse,
)


class TestFeatureRequest:
    def test_valid_feature(self):
        req = FeatureRequest(feature_name="score_total")
        assert req.feature_name == "score_total"

    def test_invalid_feature(self):
        with pytest.raises(ValidationError):
            FeatureRequest(feature_name="invalid_feature")

    def test_all_allowed_features(self):
        for f in ALLOWED_FEATURES:
            req = FeatureRequest(feature_name=f)
            assert req.feature_name == f


class TestTwoFeatureRequest:
    def test_valid_two_features(self):
        req = TwoFeatureRequest(feature1="score_total", feature2="score_brakings")
        assert req.feature1 == "score_total"
        assert req.feature2 == "score_brakings"

    def test_invalid_feature1(self):
        with pytest.raises(ValidationError):
            TwoFeatureRequest(feature1="invalid", feature2="score_total")

    def test_invalid_feature2(self):
        with pytest.raises(ValidationError):
            TwoFeatureRequest(feature1="score_total", feature2="invalid")


class TestModelRequest:
    def test_valid_model(self):
        req = ModelRequest(model_name="Random Forest")
        assert req.model_name == "Random Forest"

    def test_invalid_model(self):
        with pytest.raises(ValidationError):
            ModelRequest(model_name="NonexistentModel")

    def test_all_allowed_models(self):
        for m in ALLOWED_MODELS:
            req = ModelRequest(model_name=m)
            assert req.model_name == m


class TestResponseModels:
    def test_feature_response(self):
        resp = FeatureResponse(
            feature_name="score_total",
            statistics={"mean": 50.0, "std": 10.0},
            explanation="Test",
            plot_url="data:image/png;base64,abc",
        )
        assert resp.feature_name == "score_total"

    def test_two_feature_response(self):
        resp = TwoFeatureResponse(
            feature1="score_total",
            feature2="score_brakings",
            correlation=0.75,
            plot_url="data:image/png;base64,abc",
            explanation="Test",
        )
        assert resp.correlation == 0.75

    def test_correlation_matrix_response(self):
        resp = CorrelationMatrixResponse(
            matrix=[[1.0, 0.5], [0.5, 1.0]],
            feature_names=["a", "b"],
            high_correlation_pairs=[],
            explanation="Test",
            plot_url="data:image/png;base64,abc",
        )
        assert len(resp.matrix) == 2

    def test_confusion_matrix_response(self):
        resp = ConfusionMatrixResponse(
            model_name="Random Forest",
            accuracy=0.95,
            metrics={"f1_score": 0.94},
            plot_url="data:image/png;base64,abc",
        )
        assert resp.accuracy == 0.95

    def test_model_comparison_response(self):
        resp = ModelComparisonResponse(
            results=[{"Model": "RF", "Accuracy": 0.95}],
            plot_url="data:image/png;base64,abc",
            best_model="RF",
        )
        assert resp.best_model == "RF"

    def test_health_response(self):
        resp = HealthResponse(status="ok", version="1.0.0", dataset_loaded=True)
        assert resp.dataset_loaded is True
