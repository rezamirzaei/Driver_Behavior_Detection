"""Pydantic request/response models for the API."""

from typing import Dict, List

from pydantic import BaseModel, field_validator

ALLOWED_FEATURES = [
    "score_total",
    "score_accelerations",
    "score_brakings",
    "score_turnings",
    "score_weaving",
    "score_drifting",
    "score_overspeeding",
    "score_following",
    "ratio_normal",
    "ratio_drowsy",
    "ratio_aggressive",
]

ALLOWED_MODELS = [
    "Random Forest",
    "Gradient Boosting",
    "Logistic (L2)",
    "SVM (RBF)",
    "KNN (k=5)",
]


class FeatureRequest(BaseModel):
    """Request model for single feature analysis."""

    feature_name: str

    @field_validator("feature_name")
    @classmethod
    def validate_feature(cls, v: str) -> str:
        if v not in ALLOWED_FEATURES:
            raise ValueError(f"Invalid feature '{v}'. Allowed: {ALLOWED_FEATURES}")
        return v


class TwoFeatureRequest(BaseModel):
    """Request model for two-feature comparison."""

    feature1: str
    feature2: str

    @field_validator("feature1", "feature2")
    @classmethod
    def validate_features(cls, v: str) -> str:
        if v not in ALLOWED_FEATURES:
            raise ValueError(f"Invalid feature '{v}'. Allowed: {ALLOWED_FEATURES}")
        return v


class ModelRequest(BaseModel):
    """Request model for model evaluation."""

    model_name: str

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Invalid model '{v}'. Allowed: {ALLOWED_MODELS}")
        return v


class FeatureResponse(BaseModel):
    """Response model for single feature analysis."""

    feature_name: str
    statistics: Dict[str, float]
    explanation: str
    plot_url: str


class TwoFeatureResponse(BaseModel):
    """Response model for two-feature comparison."""

    feature1: str
    feature2: str
    correlation: float
    plot_url: str
    explanation: str


class CorrelationMatrixResponse(BaseModel):
    """Response model for correlation matrix."""

    matrix: List[List[float]]
    feature_names: List[str]
    high_correlation_pairs: List[List[str]]
    explanation: str
    plot_url: str


class ConfusionMatrixResponse(BaseModel):
    """Response model for confusion matrix."""

    model_name: str
    accuracy: float
    metrics: Dict[str, float]
    plot_url: str


class ModelComparisonResponse(BaseModel):
    """Response model for model comparison."""

    results: List[Dict[str, object]]
    plot_url: str
    best_model: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    dataset_loaded: bool
