"""Pydantic request/response contracts for the API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.api.catalog import build_task_catalogs
from src.api.config import get_settings

TaskType = Literal["classification", "regression"]
FeatureSourceType = Literal["raw", "processed"]


_settings = get_settings()
_catalogs = build_task_catalogs(
    _settings.resolve_path(_settings.classification_cache_csv),
    _settings.resolve_path(_settings.regression_cache_csv),
    random_state=_settings.random_state,
)

ALLOWED_FEATURES_BY_TASK: Dict[str, List[str]] = {task: catalog.features for task, catalog in _catalogs.items()}
ALLOWED_NUMERIC_FEATURES_BY_TASK: Dict[str, List[str]] = {
    task: catalog.numeric_features for task, catalog in _catalogs.items()
}
ALLOWED_MODELS_BY_TASK: Dict[str, List[str]] = {task: catalog.models for task, catalog in _catalogs.items()}


class TaskQuery(BaseModel):
    """Query model for task-scoped GET endpoints."""

    task: TaskType


class FeatureRequest(BaseModel):
    """Request for single-feature analysis."""

    task: TaskType
    feature_name: str = Field(min_length=1)

    @field_validator("feature_name")
    @classmethod
    def normalize_feature_name(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_feature(self) -> "FeatureRequest":
        allowed = ALLOWED_FEATURES_BY_TASK[self.task]
        if self.feature_name not in allowed:
            raise ValueError(f"Feature '{self.feature_name}' is not valid for task '{self.task}'.")
        return self


class TwoFeatureRequest(BaseModel):
    """Request for two-feature analysis."""

    task: TaskType
    feature1: str = Field(min_length=1)
    feature2: str = Field(min_length=1)

    @field_validator("feature1", "feature2")
    @classmethod
    def normalize_feature(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_features(self) -> "TwoFeatureRequest":
        allowed = ALLOWED_FEATURES_BY_TASK[self.task]
        numeric_allowed = ALLOWED_NUMERIC_FEATURES_BY_TASK[self.task]

        if self.feature1 == self.feature2:
            raise ValueError("feature1 and feature2 must be different.")
        if self.feature1 not in allowed or self.feature2 not in allowed:
            raise ValueError("Both features must be valid for the selected task.")
        if self.feature1 not in numeric_allowed or self.feature2 not in numeric_allowed:
            raise ValueError("Two-feature analysis requires numeric features.")

        return self


class ModelRequest(BaseModel):
    """Request for model-specific analysis."""

    task: TaskType
    model_name: str = Field(min_length=1)

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_model(self) -> "ModelRequest":
        allowed = ALLOWED_MODELS_BY_TASK[self.task]
        if self.model_name not in allowed:
            raise ValueError(f"Model '{self.model_name}' is not valid for task '{self.task}'.")
        return self


class ConfusionMatrixRequest(ModelRequest):
    """Request for classification confusion matrix."""

    @model_validator(mode="after")
    def validate_task(self) -> "ConfusionMatrixRequest":
        if self.task != "classification":
            raise ValueError("Confusion matrix is only available for classification task.")
        return self


class RegressionDiagnosticsRequest(ModelRequest):
    """Request for regression diagnostics plot."""

    @model_validator(mode="after")
    def validate_task(self) -> "RegressionDiagnosticsRequest":
        if self.task != "regression":
            raise ValueError("Regression diagnostics is only available for regression task.")
        return self


class CustomLearningRequest(BaseModel):
    """Request for task-specific custom learning with selected feature subset."""

    task: TaskType = "classification"
    model_name: str = Field(min_length=1)
    feature_names: List[str] = Field(default_factory=list)
    cv_folds: int = Field(default=1, ge=1, le=10)
    persist_artifact: bool = False
    artifact_id: Optional[str] = None

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("feature_names")
    @classmethod
    def normalize_feature_names(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("feature_names must not contain duplicates.")
        return cleaned

    @model_validator(mode="after")
    def validate_selection(self) -> "CustomLearningRequest":
        allowed_models = ALLOWED_MODELS_BY_TASK[self.task]
        if self.model_name not in allowed_models:
            raise ValueError(f"Model '{self.model_name}' is not valid for task '{self.task}'.")

        if not self.feature_names and not self.artifact_id:
            raise ValueError("At least one feature must be selected when artifact_id is not provided.")

        allowed_features = ALLOWED_FEATURES_BY_TASK[self.task]
        invalid_features = [name for name in self.feature_names if name not in allowed_features]
        if invalid_features:
            raise ValueError(f"Invalid feature(s) for task '{self.task}': " + ", ".join(invalid_features))

        return self


class FeatureInfo(BaseModel):
    """Feature metadata for UI controls."""

    name: str
    description: str
    is_numeric: bool
    source_type: FeatureSourceType = "processed"
    source_summary: str = ""
    lineage: str = ""


class ModelInfo(BaseModel):
    """Model metadata for UI controls."""

    name: str
    task: TaskType
    family: str
    description: str
    supports_staged_predictions: bool = False


class FeatureListResponse(BaseModel):
    """Response for list-features endpoint."""

    task: TaskType
    features: List[FeatureInfo]


class ModelsResponse(BaseModel):
    """Response for list-models endpoint."""

    task: TaskType
    models: List[str]
    model_details: List[ModelInfo] = Field(default_factory=list)


class TaskMetadataResponse(BaseModel):
    """Response for task metadata endpoint."""

    task: TaskType
    dataset_name: str
    target_name: str
    n_rows: int
    n_features: int
    n_numeric_features: int
    features: List[FeatureInfo]
    models: List[str]
    model_details: List[ModelInfo] = Field(default_factory=list)


class FeatureResponse(BaseModel):
    """Response for single-feature analysis."""

    task: TaskType
    feature_name: str
    statistics: Dict[str, float]
    explanation: str
    plot_url: str
    feature_info: Optional[FeatureInfo] = None


class TwoFeatureResponse(BaseModel):
    """Response for two-feature analysis."""

    task: TaskType
    feature1: str
    feature2: str
    correlation: float
    explanation: str
    plot_url: str


class CorrelationPair(BaseModel):
    """One high-correlation pair."""

    feature1: str
    feature2: str
    correlation: float


class CorrelationMatrixResponse(BaseModel):
    """Response for correlation matrix endpoint."""

    task: TaskType
    matrix: List[List[float]]
    feature_names: List[str]
    high_correlation_pairs: List[CorrelationPair]
    explanation: str
    plot_url: str


class TrainingHistoryPoint(BaseModel):
    """One train/validation point for model diagnostics over iterations."""

    iteration: int
    train_score: float
    validation_score: Optional[float] = None
    train_error: float
    validation_error: Optional[float] = None


class TrainingHistoryPayload(BaseModel):
    """Training/validation trajectory used by diagnostics UI."""

    score_metric: str = "score"
    error_metric: str = "error"
    points: List[TrainingHistoryPoint] = Field(default_factory=list)


class ConfusionMatrixResponse(BaseModel):
    """Response for classification confusion matrix endpoint."""

    task: Literal["classification"]
    model_name: str
    accuracy: float
    metrics: Dict[str, float]
    explanation: str
    plot_url: str
    error_plot_url: str = ""
    training_history: TrainingHistoryPayload = Field(default_factory=TrainingHistoryPayload)


class RegressionDiagnosticsResponse(BaseModel):
    """Response for regression diagnostics endpoint."""

    task: Literal["regression"]
    model_name: str
    metrics: Dict[str, float]
    explanation: str
    plot_url: str
    error_plot_url: str = ""
    training_history: TrainingHistoryPayload = Field(default_factory=TrainingHistoryPayload)


class CustomLearningResponse(BaseModel):
    """Response for custom feature-subset training."""

    task: TaskType
    model_name: str
    selected_features: List[FeatureInfo] = Field(default_factory=list)
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    explanation: str
    train_confusion_matrix_url: str = ""
    validation_confusion_matrix_url: str = ""
    validation_diagnostics_plot_url: str = ""
    error_plot_url: str = ""
    training_history: TrainingHistoryPayload = Field(default_factory=TrainingHistoryPayload)
    cv_metric_name: str = ""
    cv_scores: List[float] = Field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    artifact_id: str = ""
    artifact_saved: bool = False
    artifact_path: str = ""


class JobStartResponse(BaseModel):
    """Response for creating an asynchronous job."""

    job_id: str
    status: Literal["pending", "running"]


class JobStatusResponse(BaseModel):
    """Response for job polling endpoint."""

    job_id: str
    status: Literal["pending", "running", "cancel_requested", "canceled", "completed", "failed"]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobCancelResponse(BaseModel):
    """Response for job cancellation endpoint."""

    job_id: str
    status: Literal["cancel_requested", "canceled", "completed", "failed", "not_found"]


class DataVersionResponse(BaseModel):
    """Response for data-version manifest endpoint."""

    versions: Dict[str, str]


class ObservabilityMetricsResponse(BaseModel):
    """Response payload for in-process observability metrics."""

    started_at: str
    generated_at: str
    requests: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    training: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class TrainingRunSummary(BaseModel):
    """Compact training-run summary for history views."""

    run_id: str
    signature: str
    task: TaskType
    operation: str
    model_name: str
    status: Literal["running", "completed", "failed", "canceled"]
    cache_hit: bool = False
    started_at: str
    finished_at: Optional[str] = None
    duration_ms: Optional[float] = None
    artifact_id: str = ""
    artifact_path: str = ""
    feature_count: int = 0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    cv_summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class TrainingRunListResponse(BaseModel):
    """Response for recent training-run list endpoint."""

    runs: List[TrainingRunSummary] = Field(default_factory=list)


class TrainingRunDetailResponse(TrainingRunSummary):
    """Full persisted training-run payload."""

    feature_names: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    data_version: str = ""
    result: Dict[str, Any] = Field(default_factory=dict)


class ArtifactInfo(BaseModel):
    """Artifact metadata for list/detail responses."""

    task: TaskType
    artifact_id: str
    model_name: str
    signature: str
    data_version: str
    updated_at: str
    artifact_file_path: str
    metadata_file_path: str


class ArtifactListResponse(BaseModel):
    """Response for artifact listing endpoint."""

    artifacts: List[ArtifactInfo] = Field(default_factory=list)


class ArtifactDetailResponse(ArtifactInfo):
    """Detailed artifact metadata including feature references."""

    feature_names: List[str] = Field(default_factory=list)
    reference_stats: Dict[str, Any] = Field(default_factory=dict)
    result_payload: Dict[str, Any] = Field(default_factory=dict)


class ArtifactPredictRequest(BaseModel):
    """Inference request against a persisted artifact."""

    records: List[Dict[str, Any]] = Field(min_length=1)


class ArtifactPredictResponse(BaseModel):
    """Inference response for a persisted artifact."""

    task: TaskType
    artifact_id: str
    n_records: int
    predictions: List[str] = Field(default_factory=list)
    probabilities: List[Dict[str, float]] = Field(default_factory=list)


class ArtifactDriftRequest(BaseModel):
    """Drift check request against artifact reference data."""

    records: List[Dict[str, Any]] = Field(min_length=1)


class ArtifactDriftFeatureReport(BaseModel):
    """One feature drift report entry."""

    feature: str
    type: Literal["numeric", "categorical"]
    score: float
    flagged: bool
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    metric: str = ""


class ArtifactDriftResponse(BaseModel):
    """Drift check response for an artifact."""

    task: TaskType
    artifact_id: str
    n_records: int
    overall_drift_score: float
    flagged_feature_count: int
    is_drifted: bool
    alert_id: Optional[str] = None
    feature_reports: List[ArtifactDriftFeatureReport] = Field(default_factory=list)


class DriftAlertInfo(BaseModel):
    """Summary payload for one drift alert."""

    alert_id: str
    task: TaskType
    artifact_id: str
    overall_drift_score: float
    flagged_feature_count: int
    status: Literal["open", "acknowledged"]
    detected_at: str
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None


class DriftAlertListResponse(BaseModel):
    """Response payload for drift alert listing."""

    alerts: List[DriftAlertInfo] = Field(default_factory=list)


class DriftAlertAcknowledgeRequest(BaseModel):
    """Acknowledge drift alert request body."""

    acknowledged_by: str = Field(default="api-user", min_length=1, max_length=128)


class DriftAlertAcknowledgeResponse(BaseModel):
    """Drift alert acknowledge response."""

    alert_id: str
    status: Literal["acknowledged", "not_found"]


class ModelComparisonResponse(BaseModel):
    """Response for all-model comparison endpoint."""

    task: TaskType
    results: List[Dict[str, Any]]
    best_model: str
    ranking_metric: str
    explanation: str
    plot_url: str


class HealthResponse(BaseModel):
    """Health response."""

    status: Literal["ok"]
    version: str
    tasks_loaded: Dict[str, bool]
