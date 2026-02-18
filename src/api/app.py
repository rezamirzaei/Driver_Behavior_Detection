"""FastAPI application for classification/regression analytics."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional
import uuid

import matplotlib

matplotlib.use("Agg")
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.config import get_settings
from src.api.job_manager import JobManager
from src.api.observability import ObservabilityRegistry
from src.api.schemas import (
    ArtifactDetailResponse,
    ArtifactDriftFeatureReport,
    ArtifactDriftRequest,
    ArtifactDriftResponse,
    ArtifactInfo,
    ArtifactListResponse,
    ArtifactPredictRequest,
    ArtifactPredictResponse,
    ConfusionMatrixRequest,
    ConfusionMatrixResponse,
    CorrelationMatrixResponse,
    CorrelationPair,
    CustomLearningRequest,
    CustomLearningResponse,
    DataVersionResponse,
    DriftAlertAcknowledgeRequest,
    DriftAlertAcknowledgeResponse,
    DriftAlertInfo,
    DriftAlertListResponse,
    FeatureInfo,
    FeatureListResponse,
    FeatureRequest,
    FeatureResponse,
    HealthResponse,
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
    ModelComparisonResponse,
    ModelInfo,
    ModelsResponse,
    ObservabilityMetricsResponse,
    RegressionDiagnosticsRequest,
    RegressionDiagnosticsResponse,
    TaskMetadataResponse,
    TaskQuery,
    TaskType,
    TrainingHistoryPayload,
    TrainingRunDetailResponse,
    TrainingRunListResponse,
    TrainingRunSummary,
    TwoFeatureRequest,
    TwoFeatureResponse,
)
from src.api.security import ApiSecurityManager
from src.api.services import AnalyticsService, build_service
from src.data.versioning import build_data_manifest

logger = logging.getLogger(__name__)

_service: Optional[AnalyticsService] = None
_job_manager: Optional[JobManager] = None
_observability: Optional[ObservabilityRegistry] = None
_security_manager: Optional[ApiSecurityManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize runtime service during startup."""
    del app
    global _job_manager, _observability, _security_manager, _service

    settings = get_settings()
    try:
        _service = build_service(settings)
        _observability = ObservabilityRegistry()
        _security_manager = ApiSecurityManager(settings)
        if hasattr(_service, "set_observability"):
            _service.set_observability(_observability)
        _job_manager = JobManager(
            custom_learning_handler=_service.custom_learning,
            backend=settings.async_job_backend,
            max_workers=settings.async_job_workers,
            celery_broker_url=settings.celery_broker_url,
            celery_result_backend=settings.celery_result_backend,
            celery_task_max_retries=settings.celery_task_max_retries,
            celery_retry_backoff=settings.celery_retry_backoff,
            celery_retry_backoff_max=settings.celery_retry_backoff_max,
            celery_soft_time_limit_seconds=settings.celery_soft_time_limit_seconds,
            celery_time_limit_seconds=settings.celery_time_limit_seconds,
        )
        logger.info("API service initialized")
    except Exception:  # pragma: no cover - startup should fail loudly in runtime
        logger.exception("Failed to initialize analytics service")
        _service = None
        _job_manager = None
        _observability = None
        _security_manager = None

    yield


app = FastAPI(
    title="ABAX Model Analytics API",
    version="2.0.0",
    description="Task-aware analytics API for UAH classification and EPA regression workflows.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Emit request timing metrics for observability."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    decision = _security_manager.evaluate(request) if _security_manager is not None else None
    if decision is not None and not decision.allowed:
        headers = {"X-Request-ID": request_id}
        if decision.retry_after_seconds > 0:
            headers["Retry-After"] = str(decision.retry_after_seconds)
        if decision.limit > 0:
            headers["X-RateLimit-Limit"] = str(decision.limit)
            headers["X-RateLimit-Remaining"] = str(decision.remaining)
        response = JSONResponse(status_code=decision.status_code, content={"detail": decision.detail}, headers=headers)
        if _observability is not None:
            _observability.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=0.0,
            )
        return response

    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if _observability is not None:
        _observability.record_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=elapsed_ms,
        )

    logger.info(
        "request_completed request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    response.headers["X-Request-ID"] = request_id
    if decision is not None and decision.limit > 0:
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    return response


settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def require_service() -> AnalyticsService:
    """Return initialized service or raise HTTP 503."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Service is not initialized")
    return _service


def require_job_manager() -> JobManager:
    """Return initialized job manager or raise HTTP 503."""
    if _job_manager is None:
        raise HTTPException(status_code=503, detail="Job manager is not initialized")
    return _job_manager


@app.get("/api/health", response_model=HealthResponse)
def health_check(service: AnalyticsService = Depends(require_service)) -> HealthResponse:
    """Health endpoint."""
    tasks_loaded: Dict[str, bool] = {task: task in service.runtimes for task in ["classification", "regression"]}
    return HealthResponse(status="ok", version=settings.app_version, tasks_loaded=tasks_loaded)


@app.get("/api/data/version", response_model=DataVersionResponse)
def data_versions() -> DataVersionResponse:
    """Return lightweight dataset version hashes for reproducibility checks."""
    roots = {
        "classification_raw": settings.resolve_path(settings.classification_raw_dir),
        "processed": settings.resolve_path("data/processed"),
    }
    versions = build_data_manifest(roots)
    return DataVersionResponse(versions=versions)


@app.get("/api/observability/metrics", response_model=ObservabilityMetricsResponse)
def get_observability_metrics() -> ObservabilityMetricsResponse:
    """Return in-process request/training metrics snapshot."""
    if _observability is None:
        raise HTTPException(status_code=503, detail="Observability registry is not initialized")
    return ObservabilityMetricsResponse(**_observability.snapshot())


@app.get("/api/metadata", response_model=TaskMetadataResponse)
def get_metadata(
    query: TaskQuery = Depends(),
    service: AnalyticsService = Depends(require_service),
) -> TaskMetadataResponse:
    """Return metadata for selected task."""
    payload: Dict[str, Any] = service.metadata(query.task)
    return TaskMetadataResponse(**payload)


@app.get("/api/features", response_model=FeatureListResponse)
def list_features(
    query: TaskQuery = Depends(),
    service: AnalyticsService = Depends(require_service),
) -> FeatureListResponse:
    """List features available for selected task."""
    features = service.list_feature_payload(query.task)
    feature_models = [FeatureInfo(**feature) for feature in features]
    return FeatureListResponse(task=query.task, features=feature_models)


@app.get("/api/models", response_model=ModelsResponse)
def list_models(
    query: TaskQuery = Depends(),
    service: AnalyticsService = Depends(require_service),
) -> ModelsResponse:
    """List models available for selected task."""
    model_details = [ModelInfo(**item) for item in service.list_model_payload(query.task)]
    return ModelsResponse(
        task=query.task,
        models=service.list_models(query.task),
        model_details=model_details,
    )


@app.post("/api/feature", response_model=FeatureResponse)
def analyze_feature(
    request: FeatureRequest,
    service: AnalyticsService = Depends(require_service),
) -> FeatureResponse:
    """Compute single-feature plots/statistics."""
    stats, explanation, plot_url, feature_info = service.analyze_feature(request.task, request.feature_name)
    return FeatureResponse(
        task=request.task,
        feature_name=request.feature_name,
        statistics=stats,
        explanation=explanation,
        plot_url=plot_url,
        feature_info=FeatureInfo(**feature_info),
    )


@app.post("/api/two-features", response_model=TwoFeatureResponse)
def analyze_two_features(
    request: TwoFeatureRequest,
    service: AnalyticsService = Depends(require_service),
) -> TwoFeatureResponse:
    """Compute two-feature scatter and correlation."""
    corr, explanation, plot_url = service.analyze_two_features(request.task, request.feature1, request.feature2)
    return TwoFeatureResponse(
        task=request.task,
        feature1=request.feature1,
        feature2=request.feature2,
        correlation=round(corr, 4),
        explanation=explanation,
        plot_url=plot_url,
    )


@app.get("/api/correlation-matrix", response_model=CorrelationMatrixResponse)
def get_correlation_matrix(
    query: TaskQuery = Depends(),
    service: AnalyticsService = Depends(require_service),
) -> CorrelationMatrixResponse:
    """Return correlation matrix and detected high-correlation pairs."""
    matrix, high_pairs, explanation, plot_url, feature_names = service.correlation_matrix(query.task)
    pair_models = [CorrelationPair(**pair) for pair in high_pairs]
    return CorrelationMatrixResponse(
        task=query.task,
        matrix=matrix,
        feature_names=feature_names,
        high_correlation_pairs=pair_models,
        explanation=explanation,
        plot_url=plot_url,
    )


@app.post("/api/model/confusion-matrix", response_model=ConfusionMatrixResponse)
def get_confusion_matrix(
    request: ConfusionMatrixRequest,
    service: AnalyticsService = Depends(require_service),
) -> ConfusionMatrixResponse:
    """Train selected classifier and return confusion matrix."""
    metrics, plot_url, history_payload, error_plot_url = service.classification_confusion_matrix(request.model_name)

    summary_metrics = {
        "balanced_accuracy": round(metrics.balanced_accuracy, 4),
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1_score": round(metrics.f1_score, 4),
    }
    explanation = (
        f"{request.model_name} achieved accuracy={metrics.accuracy:.4f} on held-out test data. "
        "The confusion matrix shows class-level error distribution."
    )

    return ConfusionMatrixResponse(
        task="classification",
        model_name=request.model_name,
        accuracy=round(metrics.accuracy, 4),
        metrics=summary_metrics,
        explanation=explanation,
        plot_url=plot_url,
        error_plot_url=error_plot_url,
        training_history=TrainingHistoryPayload(**history_payload),
    )


@app.post("/api/model/regression-diagnostics", response_model=RegressionDiagnosticsResponse)
def get_regression_diagnostics(
    request: RegressionDiagnosticsRequest,
    service: AnalyticsService = Depends(require_service),
) -> RegressionDiagnosticsResponse:
    """Train selected regressor and return residual diagnostics."""
    metrics, explanation, plot_url, history_payload, error_plot_url = service.regression_diagnostics(request.model_name)
    payload = {
        "r2": round(metrics.r2, 4),
        "rmse": round(metrics.rmse, 4),
        "mae": round(metrics.mae, 4),
    }
    if metrics.mape is not None:
        payload["mape"] = round(metrics.mape, 4)

    return RegressionDiagnosticsResponse(
        task="regression",
        model_name=request.model_name,
        metrics=payload,
        explanation=explanation,
        plot_url=plot_url,
        error_plot_url=error_plot_url,
        training_history=TrainingHistoryPayload(**history_payload),
    )


@app.post("/api/model/custom-learning", response_model=CustomLearningResponse)
def run_custom_learning(
    request: CustomLearningRequest,
    service: AnalyticsService = Depends(require_service),
) -> CustomLearningResponse:
    """Train selected model on selected feature subset and return diagnostics."""
    payload = service.custom_learning(
        task=request.task,
        model_name=request.model_name,
        feature_names=request.feature_names,
        cv_folds=request.cv_folds,
        persist_artifact=request.persist_artifact,
        artifact_id=request.artifact_id,
    )
    return CustomLearningResponse(**payload)


@app.post("/api/model/custom-learning/job", response_model=JobStartResponse)
def run_custom_learning_job(
    request: CustomLearningRequest,
    job_manager: JobManager = Depends(require_job_manager),
) -> JobStartResponse:
    """Queue custom learning run as background job and return job id."""
    record = job_manager.submit_custom_learning(
        {
            "task": request.task,
            "model_name": request.model_name,
            "feature_names": request.feature_names,
            "cv_folds": request.cv_folds,
            "persist_artifact": request.persist_artifact,
            "artifact_id": request.artifact_id,
        }
    )
    return JobStartResponse(job_id=record.job_id, status="running" if record.status == "running" else "pending")


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, job_manager: JobManager = Depends(require_job_manager)) -> JobStatusResponse:
    """Poll one asynchronous job state."""
    record = job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobStatusResponse(**record.to_payload())


@app.post("/api/jobs/{job_id}/cancel", response_model=JobCancelResponse)
def cancel_job(job_id: str, job_manager: JobManager = Depends(require_job_manager)) -> JobCancelResponse:
    """Cancel one asynchronous job if still pending/running."""
    cancelled, status = job_manager.cancel(job_id)
    if not cancelled and status == "not_found":
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobCancelResponse(job_id=job_id, status=status)


@app.get("/api/training-runs", response_model=TrainingRunListResponse)
def get_training_runs(
    task: Optional[TaskType] = None,
    limit: int = 30,
    service: AnalyticsService = Depends(require_service),
) -> TrainingRunListResponse:
    """List recent training runs for dashboard history."""
    rows = service.list_training_runs(task=task, limit=limit)
    return TrainingRunListResponse(runs=[TrainingRunSummary(**row) for row in rows])


@app.get("/api/training-runs/{run_id}", response_model=TrainingRunDetailResponse)
def get_training_run(run_id: str, service: AnalyticsService = Depends(require_service)) -> TrainingRunDetailResponse:
    """Get one persisted training run by id."""
    try:
        payload = service.get_training_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TrainingRunDetailResponse(**payload)


@app.get("/api/artifacts", response_model=ArtifactListResponse)
def list_artifacts(
    task: Optional[TaskType] = None,
    limit: int = 30,
    service: AnalyticsService = Depends(require_service),
) -> ArtifactListResponse:
    """List persisted model artifacts."""
    payload = service.list_artifacts(task=task, limit=limit)
    return ArtifactListResponse(artifacts=[ArtifactInfo(**item) for item in payload])


@app.get("/api/artifacts/{task}/{artifact_id}", response_model=ArtifactDetailResponse)
def get_artifact(
    task: TaskType,
    artifact_id: str,
    service: AnalyticsService = Depends(require_service),
) -> ArtifactDetailResponse:
    """Return one persisted artifact metadata payload."""
    try:
        payload = service.get_artifact(task=task, artifact_id=artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ArtifactDetailResponse(**payload)


@app.post("/api/artifacts/{task}/{artifact_id}/predict", response_model=ArtifactPredictResponse)
def predict_from_artifact(
    task: TaskType,
    artifact_id: str,
    request: ArtifactPredictRequest,
    service: AnalyticsService = Depends(require_service),
) -> ArtifactPredictResponse:
    """Run batch inference from a persisted artifact."""
    try:
        payload = service.predict_with_artifact(task=task, artifact_id=artifact_id, records=request.records)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ArtifactPredictResponse(**payload)


@app.post("/api/artifacts/{task}/{artifact_id}/drift", response_model=ArtifactDriftResponse)
def drift_from_artifact(
    task: TaskType,
    artifact_id: str,
    request: ArtifactDriftRequest,
    service: AnalyticsService = Depends(require_service),
) -> ArtifactDriftResponse:
    """Run drift checks against artifact training references."""
    try:
        payload = service.detect_artifact_drift(task=task, artifact_id=artifact_id, records=request.records)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload["feature_reports"] = [ArtifactDriftFeatureReport(**item) for item in payload.get("feature_reports", [])]
    return ArtifactDriftResponse(**payload)


@app.get("/api/drift-alerts", response_model=DriftAlertListResponse)
def list_drift_alerts(
    task: Optional[TaskType] = None,
    artifact_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    service: AnalyticsService = Depends(require_service),
) -> DriftAlertListResponse:
    """List persisted drift alerts for operational monitoring."""
    rows = service.list_drift_alerts(task=task, artifact_id=artifact_id, limit=limit, status=status)
    return DriftAlertListResponse(alerts=[DriftAlertInfo(**row) for row in rows])


@app.post("/api/drift-alerts/{alert_id}/ack", response_model=DriftAlertAcknowledgeResponse)
def acknowledge_drift_alert(
    alert_id: str,
    request: DriftAlertAcknowledgeRequest,
    service: AnalyticsService = Depends(require_service),
) -> DriftAlertAcknowledgeResponse:
    """Acknowledge one open drift alert."""
    ok = service.acknowledge_drift_alert(alert_id=alert_id, acknowledged_by=request.acknowledged_by)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Drift alert '{alert_id}' not found")
    return DriftAlertAcknowledgeResponse(alert_id=alert_id, status="acknowledged")


@app.get("/api/model/compare", response_model=ModelComparisonResponse)
def compare_models(
    query: TaskQuery = Depends(),
    service: AnalyticsService = Depends(require_service),
) -> ModelComparisonResponse:
    """Compare all task-relevant models."""
    results, best_model, ranking_metric, plot_url = service.compare_models(query.task)

    explanation = f"Compared {len(results)} models for {query.task}. Best model by {ranking_metric} is {best_model}."

    return ModelComparisonResponse(
        task=query.task,
        results=results,
        best_model=best_model,
        ranking_metric=ranking_metric,
        explanation=explanation,
        plot_url=plot_url,
    )


index_html = static_dir / "index.html"


@app.get("/")
def serve_index() -> Any:
    """Serve AngularJS dashboard."""
    if index_html.is_file():
        return FileResponse(str(index_html))
    return {"message": "UI not found. Use /api/health for backend status."}
