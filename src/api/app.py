"""FastAPI backend API for serving ML project data."""

import base64
from contextlib import asynccontextmanager
import io
import logging
import pathlib
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt  # noqa: E402 – must follow matplotlib.use()
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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
from src.core.schemas import ClassificationMetrics
from src.data.splitter import split_by_driver
from src.data.uah_loader import load_uah_driveset
from src.features.analysis import analyze_correlations, compute_feature_statistics
from src.models.comparison import compare_classifiers
from src.models.evaluation import evaluate_classifier
from src.visualization.plots import (
    plot_classifier_comparison,
    plot_confusion_matrix,
    plot_correlation_matrix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_dataset: Optional[Any] = None
_df: Optional[pd.DataFrame] = None
_feature_cols: Optional[List[str]] = None
_X_train: Optional[Any] = None
_X_test: Optional[Any] = None
_y_train: Optional[Any] = None
_y_test: Optional[Any] = None
_class_names: Optional[List[str]] = None
_comparison: Optional[Any] = None

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "score_total": "Overall driving score (0-100) aggregating all behavior sub-scores",
    "score_accelerations": "Score for acceleration events – lower means more harsh accelerations",
    "score_brakings": "Score for braking events – lower means more harsh brakings",
    "score_turnings": "Score for turning events – lower means more aggressive turns",
    "score_weaving": "Lane discipline score based on lateral movement patterns",
    "score_drifting": "Lane keeping score derived from GPS trajectory analysis",
    "score_overspeeding": "Speed compliance score relative to road speed limits",
    "score_following": "Following distance score estimating headway to the vehicle ahead",
    "ratio_normal": "Proportion of the trip classified as normal driving (0-1)",
    "ratio_drowsy": "Proportion of the trip showing drowsy driving indicators (0-1)",
    "ratio_aggressive": "Proportion of the trip showing aggressive driving indicators (0-1)",
}

API_MODELS: Dict[str, Any] = {
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    ),
    "Logistic (L2)": lambda: LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    ),
    "SVM (RBF)": lambda: SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=42,
        probability=True,
    ),
    "KNN (k=5)": lambda: KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def plot_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64-encoded data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def _load_data() -> None:
    """Load the UAH dataset and prepare train/test splits (called once at startup)."""
    global _dataset, _df, _feature_cols
    global _X_train, _X_test, _y_train, _y_test, _class_names

    logger.info("Loading UAH-DriveSet …")
    _dataset = load_uah_driveset(
        data_dir="data/UAH-DRIVESET-v1",
        task="classification",
        return_driver_info=True,
    )

    _feature_cols = [c for c in _dataset.feature_names if c != "driver"]

    # Build a combined DataFrame for analysis / plotting
    _df = _dataset.X.copy()
    _df["behavior"] = _dataset.y.values

    # Driver-aware split
    _X_train, _X_test, _y_train, _y_test = split_by_driver(
        _dataset.X,
        _dataset.y,
        test_drivers=["D6"],
    )
    _class_names = sorted(_dataset.y.unique().tolist())
    logger.info(
        "Dataset loaded: %d samples, %d features, classes=%s",
        len(_dataset.y),
        len(_feature_cols),
        _class_names,
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 – required by FastAPI lifespan protocol
    """FastAPI lifespan handler: load dataset on startup."""
    try:
        _load_data()
    except Exception:
        logger.exception("Failed to load dataset on startup")
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Driver Behavior Detection API",
    version="1.0.0",
    description="REST API for UAH-DriveSet driver behavior analysis",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for the AngularJS frontend
_static_dir = pathlib.Path(__file__).resolve().parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        dataset_loaded=_dataset is not None,
    )


@app.get("/api/features")
def list_features():
    """List all available features with descriptions."""
    return {"features": [{"name": f, "description": FEATURE_DESCRIPTIONS.get(f, "")} for f in ALLOWED_FEATURES]}


@app.post("/api/feature", response_model=FeatureResponse)
def get_feature_stats(req: FeatureRequest):
    """Get statistics and distribution plot for a single feature."""
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        stats = compute_feature_statistics(_df[req.feature_name], req.feature_name)

        fig, ax = plt.subplots(figsize=(8, 5))
        for label in sorted(_df["behavior"].unique()):
            subset = _df.loc[_df["behavior"] == label, req.feature_name].dropna()
            ax.hist(subset, bins=20, alpha=0.5, label=str(label), edgecolor="white")
        ax.set_title(f"Distribution of {req.feature_name}", fontweight="bold")
        ax.set_xlabel(req.feature_name)
        ax.set_ylabel("Frequency")
        ax.legend(title="Behavior")
        fig.tight_layout()
        plot_url = plot_to_base64(fig)

        desc = FEATURE_DESCRIPTIONS.get(req.feature_name, "")
        explanation = (
            f"{desc}. "
            f"Mean={stats.mean:.4f}, Median={stats.median:.4f}, "
            f"Std={stats.std:.4f}, Min={stats.min:.4f}, Max={stats.max:.4f}, "
            f"Skewness={stats.skewness:.4f}."
        )

        return FeatureResponse(
            feature_name=req.feature_name,
            statistics={
                "mean": stats.mean,
                "median": stats.median,
                "std": stats.std,
                "min": stats.min,
                "max": stats.max,
                "skewness": stats.skewness,
                "n_missing": float(stats.n_missing),
                "n_unique": float(stats.n_unique),
            },
            explanation=explanation,
            plot_url=plot_url,
        )
    except Exception as exc:
        logger.exception("Error computing feature stats")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/two-features", response_model=TwoFeatureResponse)
def get_two_feature_comparison(req: TwoFeatureRequest):
    """Scatter plot comparing two features colour-coded by behaviour class."""
    if _df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        corr_val = float(_df[req.feature1].corr(_df[req.feature2]))

        color_map = {"NORMAL": "#2ecc71", "AGGRESSIVE": "#e74c3c", "DROWSY": "#f39c12"}
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in sorted(_df["behavior"].unique()):
            subset = _df[_df["behavior"] == label]
            ax.scatter(
                subset[req.feature1],
                subset[req.feature2],
                label=str(label),
                alpha=0.7,
                edgecolors="white",
                linewidth=0.5,
                color=color_map.get(str(label), None),
            )
        ax.set_xlabel(req.feature1)
        ax.set_ylabel(req.feature2)
        ax.set_title(
            f"{req.feature1} vs {req.feature2} (r={corr_val:.3f})",
            fontweight="bold",
        )
        ax.legend(title="Behavior")
        fig.tight_layout()
        plot_url = plot_to_base64(fig)

        explanation = f"Pearson correlation between {req.feature1} and {req.feature2} is {corr_val:.4f}. "
        if abs(corr_val) > 0.8:
            explanation += "These features are highly correlated."
        elif abs(corr_val) > 0.5:
            explanation += "These features show moderate correlation."
        else:
            explanation += "These features show weak correlation."

        return TwoFeatureResponse(
            feature1=req.feature1,
            feature2=req.feature2,
            correlation=round(corr_val, 4),
            plot_url=plot_url,
            explanation=explanation,
        )
    except Exception as exc:
        logger.exception("Error in two-feature comparison")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/correlation-matrix", response_model=CorrelationMatrixResponse)
def get_correlation_matrix():
    """Return the full correlation matrix with a heatmap plot."""
    if _df is None or _feature_cols is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        analysis = analyze_correlations(_df, columns=_feature_cols)

        fig = plot_correlation_matrix(_df, columns=_feature_cols)
        plot_url = plot_to_base64(fig)

        corr_matrix = _df[_feature_cols].corr()
        matrix_list = corr_matrix.values.tolist()

        high_pairs = [[f1, f2, str(round(c, 4))] for f1, f2, c in analysis.high_correlation_pairs]

        explanation = (
            f"Correlation matrix for {len(_feature_cols)} features. {len(high_pairs)} pair(s) with |r| > 0.8. "
        )
        if analysis.multicollinearity_warning:
            explanation += "⚠️ Severe multicollinearity detected (|r| > 0.9)."

        return CorrelationMatrixResponse(
            matrix=matrix_list,
            feature_names=_feature_cols,
            high_correlation_pairs=high_pairs,
            explanation=explanation,
            plot_url=plot_url,
        )
    except Exception as exc:
        logger.exception("Error computing correlation matrix")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/model/confusion-matrix", response_model=ConfusionMatrixResponse)
def get_confusion_matrix(req: ModelRequest):
    """Train a model and return its confusion matrix."""
    if _X_train is None or _X_test is None or _y_train is None or _y_test is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        model = API_MODELS[req.model_name]()
        model.fit(_X_train, _y_train)
        metrics: ClassificationMetrics = evaluate_classifier(
            model,
            _X_test,
            _y_test,
            _class_names,
        )

        fig = plot_confusion_matrix(metrics)
        plot_url = plot_to_base64(fig)

        return ConfusionMatrixResponse(
            model_name=req.model_name,
            accuracy=round(metrics.accuracy, 4),
            metrics={
                "balanced_accuracy": round(metrics.balanced_accuracy, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1_score": round(metrics.f1_score, 4),
            },
            plot_url=plot_url,
        )
    except Exception as exc:
        logger.exception("Error computing confusion matrix")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/model/compare", response_model=ModelComparisonResponse)
def compare_models():
    """Compare all available models and return results with a chart."""
    global _comparison

    if _X_train is None or _X_test is None or _y_train is None or _y_test is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded")

    try:
        models = {name: fn() for name, fn in API_MODELS.items()}

        comparison = compare_classifiers(
            _X_train,
            _y_train,
            _X_test,
            _y_test,
            models=models,
            class_names=_class_names,
            feature_names=_feature_cols,
        )
        _comparison = comparison

        results_df: pd.DataFrame = comparison.results
        fig = plot_classifier_comparison(results_df)
        plot_url = plot_to_base64(fig)

        results_list = results_df.to_dict(orient="records")

        return ModelComparisonResponse(
            results=results_list,
            plot_url=plot_url,
            best_model=comparison.best_model_name,
        )
    except Exception as exc:
        logger.exception("Error comparing models")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models")
def list_models():
    """List available model names."""
    return {"models": ALLOWED_MODELS}


# ---------------------------------------------------------------------------
# Frontend catch-all
# ---------------------------------------------------------------------------
_index_html = _static_dir / "index.html"


@app.get("/")
def serve_index():
    """Serve the AngularJS frontend."""
    if _index_html.is_file():
        return FileResponse(str(_index_html))
    return {"message": "Frontend not built. Use /api/health to verify the API."}
