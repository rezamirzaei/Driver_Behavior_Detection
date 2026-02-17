"""FastAPI application for classification/regression analytics."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.config import get_settings
from src.api.schemas import (
    ConfusionMatrixRequest,
    ConfusionMatrixResponse,
    CorrelationMatrixResponse,
    CorrelationPair,
    FeatureInfo,
    FeatureListResponse,
    FeatureRequest,
    FeatureResponse,
    HealthResponse,
    ModelComparisonResponse,
    ModelsResponse,
    RegressionDiagnosticsRequest,
    RegressionDiagnosticsResponse,
    TaskMetadataResponse,
    TaskQuery,
    TwoFeatureRequest,
    TwoFeatureResponse,
)
from src.api.services import AnalyticsService, build_service

logger = logging.getLogger(__name__)

_service: Optional[AnalyticsService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize runtime service during startup."""
    del app
    global _service

    settings = get_settings()
    try:
        _service = build_service(settings)
        logger.info("API service initialized")
    except Exception:  # pragma: no cover - startup should fail loudly in runtime
        logger.exception("Failed to initialize analytics service")
        _service = None

    yield


app = FastAPI(
    title="ABAX Model Analytics API",
    version="2.0.0",
    description="Task-aware analytics API for UAH classification and EPA regression workflows.",
    lifespan=lifespan,
)

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


@app.get("/api/health", response_model=HealthResponse)
def health_check(service: AnalyticsService = Depends(require_service)) -> HealthResponse:
    """Health endpoint."""
    tasks_loaded: Dict[str, bool] = {task: task in service.runtimes for task in ["classification", "regression"]}
    return HealthResponse(status="ok", version=settings.app_version, tasks_loaded=tasks_loaded)


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
    return ModelsResponse(
        task=query.task,
        models=service.list_models(query.task),
        model_details=service.list_model_payload(query.task),
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
        training_history=history_payload,
    )


@app.post("/api/model/regression-diagnostics", response_model=RegressionDiagnosticsResponse)
def get_regression_diagnostics(
    request: RegressionDiagnosticsRequest,
    service: AnalyticsService = Depends(require_service),
) -> RegressionDiagnosticsResponse:
    """Train selected regressor and return residual diagnostics."""
    metrics, explanation, plot_url, history_payload, error_plot_url = service.regression_diagnostics(
        request.model_name
    )
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
        training_history=history_payload,
    )


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
