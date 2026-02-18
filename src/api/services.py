"""Task-aware analytics services for API endpoints."""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from threading import Lock
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Avoid oversubscription/deadlocks from mixed OpenMP runtimes in containerized environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from src.api.catalog import TaskCatalog, TaskType, build_task_catalogs
from src.api.config import AppSettings
from src.api.model_registry import ModelRegistry
from src.api.observability import ObservabilityRegistry
from src.api.run_repository import TrainingRunRepository
from src.core.schemas import ClassificationMetrics, RegressionMetrics, SplitData, TrainingHistory
from src.data.cache_io import read_dataframe_cache, write_dataframe_cache
from src.data.epa_loader import load_epa_fuel_economy
from src.data.sample_models import ClassificationTripSample, EPAVehicleSample
from src.data.splitter import split_by_driver
from src.data.validation import validate_dataframe_records
from src.features.analysis import analyze_correlations, compute_feature_statistics
from src.features.preprocessing import engineer_regression_features, preprocess_features
from src.models.evaluation import evaluate_classifier, evaluate_regressor
from src.models.trainer import train_model
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_model_comparison_detailed,
    plot_residual_analysis,
    plot_training_error_history,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*Intel OpenMP.*", category=RuntimeWarning)

REMOVED_CLASSIFICATION_COLUMNS = {"hard_brake_count", "hard_break_count"}
REGRESSION_CACHE_SCHEMA_VERSION = "1.0.0"


@dataclass
class TaskRuntime:
    """Runtime artifacts for one task."""

    catalog: TaskCatalog
    dataframe: pd.DataFrame
    feature_columns: List[str]
    numeric_feature_columns: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: List[str]


class AnalyticsService:
    """Service exposing analytics operations for both tasks."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

        classification_csv = settings.resolve_path(settings.classification_cache_csv)
        regression_csv = settings.resolve_path(settings.regression_cache_csv)

        self.catalogs = build_task_catalogs(
            classification_csv,
            regression_csv,
            random_state=settings.random_state,
        )
        self.model_registry = ModelRegistry(settings.random_state)

        self.runtimes = {
            "classification": self._load_classification_runtime(),
            "regression": self._load_regression_runtime(),
        }
        self._runtime_data_versions = {
            task: self._build_runtime_data_version(runtime)
            for task, runtime in self.runtimes.items()
        }

        self._training_cache_enabled = bool(self.settings.training_cache_enabled)
        self._training_cache_max_entries = int(self.settings.training_cache_max_entries)
        self._training_cache: Dict[str, Any] = {}
        self._training_cache_order: List[str] = []
        self._training_cache_lock = Lock()
        self.run_repository = TrainingRunRepository(self.settings.resolve_path(self.settings.run_store_path))
        self.observability: Optional[ObservabilityRegistry] = None

    def set_observability(self, registry: ObservabilityRegistry) -> None:
        """Attach shared observability registry from API app."""
        self.observability = registry

    @staticmethod
    def _looks_like_legacy_event_counts(df: pd.DataFrame) -> bool:
        """Detect likely legacy caches where event counts need recalculation.

        This detects legacy caches where event counts stored sample totals
        (e.g., rates > 0.5 events/second), which indicates a pre-event-start-count
        extraction bug.
        """
        required_columns = {"trip_duration", "brake_count"}
        if not required_columns.issubset(df.columns):
            return False

        duration = pd.to_numeric(df["trip_duration"], errors="coerce").replace(0, np.nan).abs()
        brake_count = pd.to_numeric(df["brake_count"], errors="coerce").fillna(0.0)

        # Check for suspiciously high rates (legacy sample counts)
        rate = (brake_count / duration).replace([np.inf, -np.inf], np.nan).dropna()
        if not rate.empty and bool(rate.median() > 0.5):
            return True

        return False

    @staticmethod
    def _sanitize_classification_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Drop deprecated/corrupted classification columns from runtime datasets."""
        legacy_cols = [name for name in REMOVED_CLASSIFICATION_COLUMNS if name in df.columns]
        if not legacy_cols:
            return df
        return df.drop(columns=legacy_cols)

    def _load_classification_runtime(self) -> TaskRuntime:
        catalog = self.catalogs["classification"]
        cache_path = self.settings.resolve_path(self.settings.classification_cache_csv)
        from src.classification.data import load_or_build_dataset

        if cache_path.exists():
            df = read_dataframe_cache(cache_path)
            df = validate_dataframe_records(df, ClassificationTripSample, strict=False, context="classification_cache")
            df = self._sanitize_classification_dataframe(df)
            if self._looks_like_legacy_event_counts(df):
                logger.warning(
                    "Classification cache appears to use legacy sample-based event counts; rebuilding from raw data."
                )
                raw_dir = self.settings.resolve_path(self.settings.classification_raw_dir)
                df = load_or_build_dataset(data_dir=raw_dir, cache_path=cache_path, force_rebuild=True)
                df = validate_dataframe_records(
                    df,
                    ClassificationTripSample,
                    strict=False,
                    context="classification_cache_rebuild",
                )
                df = self._sanitize_classification_dataframe(df)
        else:
            raw_dir = self.settings.resolve_path(self.settings.classification_raw_dir)
            df = load_or_build_dataset(data_dir=raw_dir, cache_path=cache_path)
            df = validate_dataframe_records(df, ClassificationTripSample, strict=False, context="classification_raw")
            df = self._sanitize_classification_dataframe(df)

        if "behavior" not in df.columns or "driver" not in df.columns:
            raise ValueError("Classification dataset must include 'behavior' and 'driver' columns")

        feature_columns = [col for col in catalog.features if col in df.columns]
        if not feature_columns:
            feature_columns = [col for col in df.columns if col not in {"driver", "behavior", "road_type"}]

        numeric_feature_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]

        data = df.copy()
        data["behavior"] = data["behavior"].astype(str).str.upper()
        for col in numeric_feature_columns:
            median_value = float(data[col].median())
            data[col] = data[col].fillna(median_value)

        X_for_split = data[feature_columns + ["driver"]].copy()
        y = data["behavior"].copy()

        X_train, X_test, y_train, y_test = split_by_driver(
            X_for_split,
            y,
            test_drivers=["D6"],
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
        )

        split_data = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_columns,
            target_name="behavior",
        )
        train_features, test_features = preprocess_features(split_data, scaler_type="robust")

        class_names = sorted(data["behavior"].unique().tolist())

        return TaskRuntime(
            catalog=catalog,
            dataframe=data,
            feature_columns=feature_columns,
            numeric_feature_columns=numeric_feature_columns,
            X_train=np.asarray(train_features.X),
            X_test=np.asarray(test_features.X),
            y_train=np.asarray(y_train),
            y_test=np.asarray(y_test),
            class_names=class_names,
        )

    def _load_regression_runtime(self) -> TaskRuntime:
        catalog = self.catalogs["regression"]
        cache_path = self.settings.resolve_path(self.settings.regression_cache_csv)
        cache_needs_refresh = False

        def _load_epa_dataframe(context: str) -> pd.DataFrame:
            dataset = load_epa_fuel_economy(sample_size=5000, random_state=self.settings.random_state)
            frame = pd.DataFrame(dataset.X).copy()
            frame["comb08"] = dataset.y
            return validate_dataframe_records(frame, EPAVehicleSample, strict=False, context=context)

        if cache_path.exists():
            df = read_dataframe_cache(cache_path)
            df = validate_dataframe_records(df, EPAVehicleSample, strict=False, context="regression_cache")
            if df.empty:
                logger.warning(
                    "Regression cache '%s' yielded zero valid rows; rebuilding from raw EPA dataset.",
                    cache_path,
                )
                df = _load_epa_dataframe(context="regression_cache_fallback")
                cache_needs_refresh = True
        else:
            df = _load_epa_dataframe(context="regression_raw")
            cache_needs_refresh = True

        if cache_needs_refresh:
            try:
                write_dataframe_cache(
                    df,
                    cache_path,
                    dataset_name="epa_fuel_economy",
                    schema_version=REGRESSION_CACHE_SCHEMA_VERSION,
                )
            except OSError as exc:
                logger.warning("Could not persist regression cache to %s: %s", cache_path, exc)

        if "comb08" not in df.columns:
            raise ValueError("Regression dataset must include 'comb08' target column")

        data = engineer_regression_features(df)
        data = data.dropna(subset=["comb08"]).reset_index(drop=True)

        feature_columns = [col for col in catalog.features if col in data.columns and col != "comb08"]
        if not feature_columns:
            feature_columns = [col for col in data.columns if col != "comb08"]

        numeric_feature_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(data[col])]

        X = data[feature_columns].copy()
        y = data["comb08"].astype(float).copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
        )

        split_data = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_columns,
            target_name="comb08",
        )
        train_features, test_features = preprocess_features(split_data, scaler_type="robust")

        return TaskRuntime(
            catalog=catalog,
            dataframe=data,
            feature_columns=feature_columns,
            numeric_feature_columns=numeric_feature_columns,
            X_train=np.asarray(train_features.X),
            X_test=np.asarray(test_features.X),
            y_train=np.asarray(y_train),
            y_test=np.asarray(y_test),
            class_names=[],
        )

    @staticmethod
    def figure_to_data_url(fig: plt.Figure) -> str:
        """Encode a matplotlib figure as data URL."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{encoded}"

    def _runtime(self, task: TaskType) -> TaskRuntime:
        runtime = self.runtimes.get(task)
        if runtime is None:
            raise ValueError(f"Unsupported task: {task}")
        return runtime

    @staticmethod
    def _build_runtime_data_version(runtime: TaskRuntime) -> str:
        """Build a stable hash representing runtime dataset state."""
        digest = hashlib.sha256()
        digest.update(str(runtime.dataframe.shape).encode("utf-8"))
        digest.update("|".join(runtime.dataframe.columns.astype(str).tolist()).encode("utf-8"))

        preview = runtime.dataframe.head(200)
        if not preview.empty:
            row_hashes = pd.util.hash_pandas_object(preview, index=True)
            digest.update(row_hashes.to_numpy(dtype=np.uint64).tobytes())

        return digest.hexdigest()

    def _training_signature(
        self,
        operation: str,
        task: TaskType,
        model_name: str,
        feature_names: List[str],
        params: Dict[str, Any],
    ) -> str:
        payload = {
            "operation": operation,
            "task": task,
            "model": model_name,
            "features": sorted(set(feature_names)),
            "params": params,
            "data_version": self._runtime_data_versions.get(task, "unknown"),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Any]:
        if not self._training_cache_enabled:
            return None
        with self._training_cache_lock:
            if key not in self._training_cache:
                return None

            # Maintain recency order for lightweight LRU behavior.
            if key in self._training_cache_order:
                self._training_cache_order.remove(key)
            self._training_cache_order.append(key)
            return copy.deepcopy(self._training_cache[key])

    def _cache_set(self, key: str, value: Any) -> None:
        if not self._training_cache_enabled:
            return

        with self._training_cache_lock:
            self._training_cache[key] = copy.deepcopy(value)
            if key in self._training_cache_order:
                self._training_cache_order.remove(key)
            self._training_cache_order.append(key)

            while len(self._training_cache_order) > self._training_cache_max_entries:
                evicted = self._training_cache_order.pop(0)
                self._training_cache.pop(evicted, None)

    @staticmethod
    def _run_metrics_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "train": payload.get("train_metrics", {}),
            "validation": payload.get("validation_metrics", {}),
        }

    @staticmethod
    def _run_cv_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
        cv_metric_name = str(payload.get("cv_metric_name", ""))
        if not cv_metric_name:
            return {}
        return {
            "metric_name": cv_metric_name,
            "scores": payload.get("cv_scores", []),
            "mean": payload.get("cv_mean"),
            "std": payload.get("cv_std"),
        }

    def list_training_runs(self, task: Optional[TaskType], limit: int = 30) -> List[Dict[str, Any]]:
        """List recent persisted runs for dashboard history."""
        safe_limit = max(1, min(int(limit), 200))
        return self.run_repository.list_runs(task=task, limit=safe_limit)

    def get_training_run(self, run_id: str) -> Dict[str, Any]:
        """Get one persisted run by id."""
        payload = self.run_repository.get_run(run_id)
        if payload is None:
            raise ValueError(f"Run '{run_id}' not found.")
        return payload

    def _record_training_metric(
        self,
        *,
        operation: str,
        task: TaskType,
        model_name: str,
        duration_ms: float,
        cache_hit: bool,
    ) -> None:
        if self.observability is None:
            return
        self.observability.record_training(
            operation=operation,
            task=task,
            model_name=model_name,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
        )

    def _artifact_dir(self) -> Path:
        return self.settings.resolve_path("results/model_artifacts")

    def _artifact_file(self, task: TaskType, artifact_id: str) -> Path:
        return self._artifact_dir() / f"{task}-{artifact_id}.json"

    def _persist_artifact_payload(self, task: TaskType, artifact_id: str, payload: Dict[str, Any]) -> str:
        artifact_path = self._artifact_file(task, artifact_id)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(artifact_path)

    def _load_artifact_payload(self, task: TaskType, artifact_id: str) -> Dict[str, Any]:
        artifact_path = self._artifact_file(task, artifact_id)
        if not artifact_path.exists():
            raise ValueError(f"Artifact '{artifact_id}' not found for task '{task}'.")
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Artifact '{artifact_id}' is invalid.")
        return payload

    @staticmethod
    def _feature_source_summary(task: TaskType, source_type: str) -> str:
        if task == "classification":
            return "Processed feature aggregated from raw GPS/accelerometer driving signals."
        if source_type == "raw":
            return "Raw EPA attribute taken directly from the source dataset."
        return "Processed feature engineered from one or more raw EPA vehicle attributes."

    def list_feature_payload(self, task: TaskType) -> List[Dict[str, Any]]:
        runtime = self._runtime(task)
        return [
            {
                "name": feature,
                "description": runtime.catalog.feature_descriptions.get(feature, ""),
                "is_numeric": feature in runtime.numeric_feature_columns,
                "source_type": runtime.catalog.feature_sources.get(feature, "processed"),
                "source_summary": self._feature_source_summary(
                    task,
                    runtime.catalog.feature_sources.get(feature, "processed"),
                ),
                "lineage": runtime.catalog.feature_lineage.get(feature, ""),
            }
            for feature in runtime.feature_columns
        ]

    def list_model_payload(self, task: TaskType) -> List[Dict[str, Any]]:
        return self.model_registry.list_model_payload(task)

    def list_models(self, task: TaskType) -> List[str]:
        return self.model_registry.list_models(task)

    def metadata(self, task: TaskType) -> Dict[str, Any]:
        runtime = self._runtime(task)
        return {
            "task": task,
            "dataset_name": runtime.catalog.dataset_name,
            "target_name": runtime.catalog.target_name,
            "n_rows": int(len(runtime.dataframe)),
            "n_features": int(len(runtime.feature_columns)),
            "n_numeric_features": int(len(runtime.numeric_feature_columns)),
            "features": self.list_feature_payload(task),
            "models": runtime.catalog.models,
            "model_details": self.list_model_payload(task),
        }

    def analyze_feature(self, task: TaskType, feature_name: str) -> Tuple[Dict[str, float], str, str, Dict[str, Any]]:
        runtime = self._runtime(task)
        series = runtime.dataframe[feature_name]
        feature_desc = runtime.catalog.feature_descriptions.get(feature_name, "")
        source_type = runtime.catalog.feature_sources.get(feature_name, "processed")

        fig: plt.Figure
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            stats = compute_feature_statistics(numeric_series, feature_name)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(numeric_series.dropna(), bins=30, edgecolor="white", color="#2563eb", alpha=0.8)
            axes[0].set_title(f"Distribution: {feature_name}", fontweight="bold")
            axes[0].set_xlabel(feature_name)
            axes[0].set_ylabel("Frequency")

            if task == "classification":
                for label in sorted(runtime.dataframe["behavior"].unique()):
                    subset = runtime.dataframe.loc[runtime.dataframe["behavior"] == label, feature_name]
                    axes[1].hist(
                        pd.to_numeric(subset, errors="coerce").dropna(),
                        bins=20,
                        alpha=0.5,
                        label=str(label),
                        edgecolor="white",
                    )
                axes[1].legend(title="Behavior")
                axes[1].set_title("Class-wise distributions", fontweight="bold")
                axes[1].set_xlabel(feature_name)
                axes[1].set_ylabel("Frequency")
            else:
                axes[1].boxplot(numeric_series.dropna())
                axes[1].set_title("Boxplot", fontweight="bold")
                axes[1].set_ylabel(feature_name)

            fig.tight_layout()

            statistics = {
                "mean": round(float(stats.mean), 4),
                "median": round(float(stats.median), 4),
                "std": round(float(stats.std), 4),
                "min": round(float(stats.min), 4),
                "max": round(float(stats.max), 4),
                "skewness": round(float(stats.skewness), 4),
                "missing_count": float(stats.n_missing),
                "unique_count": float(stats.n_unique),
            }
            explanation = (
                f"{feature_desc} Source type: {source_type}. Mean={stats.mean:.3f}, "
                f"median={stats.median:.3f}, std={stats.std:.3f}. "
                f"Skewness={stats.skewness:.3f}, missing={stats.n_missing}, unique={stats.n_unique}."
            ).strip()
        else:
            value_counts = series.astype(str).fillna("N/A").value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(value_counts.index.astype(str), value_counts.values, color="#0ea5e9", edgecolor="black")
            ax.set_title(f"Top categories: {feature_name}", fontweight="bold")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()

            statistics = {
                "missing_count": float(series.isna().sum()),
                "unique_count": float(series.nunique(dropna=True)),
                "top_category_count": float(value_counts.iloc[0] if len(value_counts) > 0 else 0),
            }
            top_label = str(value_counts.index[0]) if len(value_counts) > 0 else "N/A"
            explanation = (
                f"{feature_desc} Source type: {source_type}. Categorical distribution with "
                f"{int(statistics['unique_count'])} unique values. Most frequent category is '{top_label}'."
            ).strip()

        feature_info = next(
            (item for item in self.list_feature_payload(task) if item["name"] == feature_name),
            {
                "name": feature_name,
                "description": feature_desc,
                "is_numeric": feature_name in runtime.numeric_feature_columns,
                "source_type": source_type,
                "source_summary": self._feature_source_summary(task, source_type),
                "lineage": runtime.catalog.feature_lineage.get(feature_name, ""),
            },
        )

        return statistics, explanation, self.figure_to_data_url(fig), feature_info

    def analyze_two_features(self, task: TaskType, feature1: str, feature2: str) -> Tuple[float, str, str]:
        runtime = self._runtime(task)
        df = runtime.dataframe

        x = pd.to_numeric(df[feature1], errors="coerce")
        y = pd.to_numeric(df[feature2], errors="coerce")
        corr = float(x.corr(y)) if len(df) > 1 else 0.0
        if np.isnan(corr):
            corr = 0.0

        fig, ax = plt.subplots(figsize=(8, 6))
        if task == "classification":
            for label in sorted(df["behavior"].unique()):
                subset = df[df["behavior"] == label]
                ax.scatter(
                    pd.to_numeric(subset[feature1], errors="coerce"),
                    pd.to_numeric(subset[feature2], errors="coerce"),
                    alpha=0.75,
                    label=str(label),
                    edgecolors="white",
                    linewidth=0.5,
                )
            ax.legend(title="Behavior")
        else:
            ax.scatter(x, y, alpha=0.6, edgecolors="white", linewidth=0.5, color="#16a34a")
            valid = ~(x.isna() | y.isna())
            if valid.sum() > 2:
                coeff = np.polyfit(x[valid], y[valid], deg=1)
                fit_x = np.linspace(float(x[valid].min()), float(x[valid].max()), 100)
                fit_y = coeff[0] * fit_x + coeff[1]
                ax.plot(fit_x, fit_y, color="#dc2626", linewidth=2)

        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(f"{feature1} vs {feature2} (r={corr:.3f})", fontweight="bold")
        fig.tight_layout()

        strength = "weak"
        if abs(corr) >= 0.8:
            strength = "strong"
        elif abs(corr) >= 0.5:
            strength = "moderate"
        explanation = (
            f"Pearson correlation between {feature1} and {feature2} is {corr:.4f}, indicating {strength} association."
        )

        return corr, explanation, self.figure_to_data_url(fig)

    def correlation_matrix(self, task: TaskType) -> Tuple[List[List[float]], List[Dict[str, Any]], str, str, List[str]]:
        runtime = self._runtime(task)
        numeric_cols = runtime.numeric_feature_columns
        corr_df = runtime.dataframe[numeric_cols].corr().fillna(0.0)

        analysis = analyze_correlations(runtime.dataframe, columns=numeric_cols, threshold=0.8)
        high_pairs = [
            {
                "feature1": f1,
                "feature2": f2,
                "correlation": float(corr_value),
            }
            for f1, f2, corr_value in analysis.high_correlation_pairs
        ]

        figure = plot_correlation_matrix(runtime.dataframe, columns=numeric_cols)
        explanation = (
            f"Computed correlation matrix for {len(numeric_cols)} numeric features. "
            f"Detected {len(high_pairs)} feature pairs with |r| > 0.8."
        )
        if analysis.multicollinearity_warning:
            explanation += " Severe multicollinearity exists for at least one pair (|r| > 0.9)."

        return (
            corr_df.values.tolist(),
            high_pairs,
            explanation,
            self.figure_to_data_url(figure),
            numeric_cols,
        )

    def _build_model(self, task: TaskType, model_name: str) -> Any:
        return self.model_registry.create_model(task, model_name)

    @staticmethod
    def _score_to_error(metric_name: str, score: float | None) -> float | None:
        if score is None:
            return None

        if metric_name in {"accuracy", "balanced_accuracy", "f1_score"}:
            return max(0.0, 1.0 - score)
        if metric_name in {"r2", "r2_score"}:
            return 1.0 - score
        return score

    @staticmethod
    def _error_metric_name(metric_name: str) -> str:
        if metric_name in {"accuracy", "balanced_accuracy", "f1_score"}:
            return "classification_error"
        if metric_name in {"r2", "r2_score"}:
            return "one_minus_r2"
        return metric_name

    def _history_payload(self, history: TrainingHistory) -> Dict[str, Any]:
        points: List[Dict[str, Any]] = []

        for idx, iteration in enumerate(history.iterations):
            train_score = history.train_scores[idx] if idx < len(history.train_scores) else 0.0
            val_score = history.val_scores[idx] if idx < len(history.val_scores) else None
            points.append(
                {
                    "iteration": int(iteration),
                    "train_score": float(train_score),
                    "validation_score": float(val_score) if val_score is not None else None,
                    "train_error": float(self._score_to_error(history.metric_name, train_score) or 0.0),
                    "validation_error": (
                        float(self._score_to_error(history.metric_name, val_score)) if val_score is not None else None
                    ),
                }
            )

        return {
            "score_metric": history.metric_name,
            "error_metric": self._error_metric_name(history.metric_name),
            "points": points,
        }

    def _train_with_history(self, task: TaskType, model_name: str) -> Tuple[Any, TrainingHistory]:
        runtime = self._runtime(task)
        model = self._build_model(task, model_name)

        trained = train_model(
            runtime.X_train,
            runtime.y_train,
            model,
            model_name,
            X_val=runtime.X_test,
            y_val=runtime.y_test,
            n_iterations=self.settings.training_history_iterations,
        )
        return trained.model, trained.history

    def _classification_split_for_features(
        self,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        runtime = self._runtime("classification")
        missing = [feature for feature in feature_names if feature not in runtime.feature_columns]
        if missing:
            raise ValueError(f"Selected feature(s) not available: {', '.join(missing)}")

        data = runtime.dataframe.copy()
        X_for_split = data[feature_names + ["driver"]].copy()
        y = data["behavior"].copy()

        X_train, X_test, y_train, y_test = split_by_driver(
            X_for_split,
            y,
            test_drivers=["D6"],
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
        )

        split_data = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            target_name="behavior",
        )
        train_features, test_features = preprocess_features(split_data, scaler_type="robust")
        class_names = sorted(data["behavior"].astype(str).str.upper().unique().tolist())
        return (
            np.asarray(train_features.X),
            np.asarray(test_features.X),
            np.asarray(y_train),
            np.asarray(y_test),
            class_names,
        )

    @staticmethod
    def _classification_metrics_payload(metrics: ClassificationMetrics) -> Dict[str, float]:
        return {
            "accuracy": round(metrics.accuracy, 4),
            "balanced_accuracy": round(metrics.balanced_accuracy, 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1_score": round(metrics.f1_score, 4),
        }

    @staticmethod
    def _regression_metrics_payload(metrics: RegressionMetrics) -> Dict[str, float]:
        payload = {
            "r2": round(metrics.r2, 4),
            "rmse": round(metrics.rmse, 4),
            "mae": round(metrics.mae, 4),
        }
        if metrics.mape is not None:
            payload["mape"] = round(metrics.mape, 4)
        return payload

    def classification_confusion_matrix(
        self,
        model_name: str,
    ) -> Tuple[ClassificationMetrics, str, Dict[str, Any], str]:
        signature = self._training_signature(
            operation="classification_confusion_matrix",
            task="classification",
            model_name=model_name,
            feature_names=[],
            params={
                "test_size": self.settings.test_size,
                "iterations": self.settings.training_history_iterations,
                "random_state": self.settings.random_state,
            },
        )
        cached = self._cache_get(signature)
        if cached is not None:
            self._record_training_metric(
                operation="classification_confusion_matrix",
                task="classification",
                model_name=model_name,
                duration_ms=0.0,
                cache_hit=True,
            )
            logger.info("classification_confusion_matrix_completed model=%s cache_hit=true", model_name)
            return cached

        start = time.perf_counter()
        runtime = self._runtime("classification")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model, history = self._train_with_history("classification", model_name)

        metrics = evaluate_classifier(
            model,
            runtime.X_test,
            runtime.y_test,
            runtime.class_names,
        )

        confusion_fig = plot_confusion_matrix(metrics)
        history_fig = plot_training_error_history(
            history,
            title=f"{model_name} - Train/Validation Error by Iteration",
        )
        payload = (
            metrics,
            self.figure_to_data_url(confusion_fig),
            self._history_payload(history),
            self.figure_to_data_url(history_fig),
        )
        self._cache_set(signature, payload)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "classification_confusion_matrix_completed model=%s duration_ms=%.2f cache_hit=false",
            model_name,
            duration_ms,
        )
        self._record_training_metric(
            operation="classification_confusion_matrix",
            task="classification",
            model_name=model_name,
            duration_ms=duration_ms,
            cache_hit=False,
        )
        return payload

    def custom_classification_learning(
        self,
        model_name: str,
        feature_names: List[str],
        cv_folds: int = 1,
    ) -> Dict[str, Any]:
        unique_features = list(dict.fromkeys(feature_names))
        if not unique_features:
            raise ValueError("At least one feature must be selected.")

        X_train, X_test, y_train, y_test, class_names = self._classification_split_for_features(unique_features)
        model = self._build_model("classification", model_name)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            trained = train_model(
                X_train,
                y_train,
                model,
                model_name,
                X_val=X_test,
                y_val=y_test,
                n_iterations=self.settings.training_history_iterations,
                feature_names=unique_features,
            )

        train_metrics = evaluate_classifier(trained.model, X_train, y_train, class_names)
        validation_metrics = evaluate_classifier(trained.model, X_test, y_test, class_names)

        cv_metric_name = ""
        cv_scores: List[float] = []
        cv_mean: Optional[float] = None
        cv_std: Optional[float] = None
        if cv_folds > 1 and len(y_train) >= 4:
            train_labels = pd.Series(y_train)
            if not train_labels.empty:
                min_per_class = int(train_labels.value_counts().min())
                effective_folds = min(cv_folds, min_per_class)
                if effective_folds >= 2:
                    cv_model = self._build_model("classification", model_name)
                    cv_splitter = StratifiedKFold(
                        n_splits=effective_folds,
                        shuffle=True,
                        random_state=self.settings.random_state,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        scores = cross_val_score(
                            cv_model,
                            X_train,
                            y_train,
                            scoring="f1_weighted",
                            cv=cv_splitter,
                            n_jobs=None,
                        )
                    cv_metric_name = "f1_weighted"
                    cv_scores = [round(float(score), 4) for score in scores.tolist()]
                    cv_mean = round(float(np.mean(scores)), 4)
                    cv_std = round(float(np.std(scores)), 4)

        train_fig = plot_confusion_matrix(train_metrics)
        validation_fig = plot_confusion_matrix(validation_metrics)
        history_fig = plot_training_error_history(
            trained.history,
            title=f"{model_name} - Train/Validation Error by Iteration",
        )

        feature_lookup = {item["name"]: item for item in self.list_feature_payload("classification")}
        selected_feature_payload = [feature_lookup[name] for name in unique_features if name in feature_lookup]

        explanation = (
            f"Trained {model_name} using {len(unique_features)} selected feature(s). "
            f"Validation accuracy={validation_metrics.accuracy:.4f}, F1={validation_metrics.f1_score:.4f}."
        )

        return {
            "task": "classification",
            "model_name": model_name,
            "selected_features": selected_feature_payload,
            "train_metrics": self._classification_metrics_payload(train_metrics),
            "validation_metrics": self._classification_metrics_payload(validation_metrics),
            "explanation": explanation,
            "train_confusion_matrix_url": self.figure_to_data_url(train_fig),
            "validation_confusion_matrix_url": self.figure_to_data_url(validation_fig),
            "error_plot_url": self.figure_to_data_url(history_fig),
            "training_history": self._history_payload(trained.history),
            "cv_metric_name": cv_metric_name,
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

    def _regression_split_for_features(
        self,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        runtime = self._runtime("regression")
        missing = [feature for feature in feature_names if feature not in runtime.feature_columns]
        if missing:
            raise ValueError(f"Selected feature(s) not available: {', '.join(missing)}")

        data = runtime.dataframe.copy()
        X = data[feature_names].copy()
        y = data["comb08"].astype(float).copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
        )

        split_data = SplitData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            target_name="comb08",
        )
        train_features, test_features = preprocess_features(split_data, scaler_type="robust")
        return (
            np.asarray(train_features.X),
            np.asarray(test_features.X),
            np.asarray(y_train),
            np.asarray(y_test),
        )

    def custom_regression_learning(
        self,
        model_name: str,
        feature_names: List[str],
        cv_folds: int = 1,
    ) -> Dict[str, Any]:
        unique_features = list(dict.fromkeys(feature_names))
        if not unique_features:
            raise ValueError("At least one feature must be selected.")

        X_train, X_test, y_train, y_test = self._regression_split_for_features(unique_features)
        model = self._build_model("regression", model_name)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            trained = train_model(
                X_train,
                y_train,
                model,
                model_name,
                X_val=X_test,
                y_val=y_test,
                n_iterations=self.settings.training_history_iterations,
                feature_names=unique_features,
            )

        train_metrics = evaluate_regressor(trained.model, X_train, y_train)
        validation_metrics = evaluate_regressor(trained.model, X_test, y_test)
        validation_pred = trained.model.predict(X_test)

        cv_metric_name = ""
        cv_scores: List[float] = []
        cv_mean: Optional[float] = None
        cv_std: Optional[float] = None
        if cv_folds > 1 and len(y_train) >= 4:
            effective_folds = min(cv_folds, len(y_train))
            if effective_folds >= 2:
                cv_model = self._build_model("regression", model_name)
                cv_splitter = KFold(
                    n_splits=effective_folds,
                    shuffle=True,
                    random_state=self.settings.random_state,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    scores = cross_val_score(
                        cv_model,
                        X_train,
                        y_train,
                        scoring="r2",
                        cv=cv_splitter,
                        n_jobs=None,
                    )
                cv_metric_name = "r2"
                cv_scores = [round(float(score), 4) for score in scores.tolist()]
                cv_mean = round(float(np.mean(scores)), 4)
                cv_std = round(float(np.std(scores)), 4)

        diagnostics_fig = plot_residual_analysis(
            y_test,
            np.asarray(validation_pred),
            model_name=model_name,
            r2=validation_metrics.r2,
        )
        history_fig = plot_training_error_history(
            trained.history,
            title=f"{model_name} - Train/Validation Error by Iteration",
        )

        feature_lookup = {item["name"]: item for item in self.list_feature_payload("regression")}
        selected_feature_payload = [feature_lookup[name] for name in unique_features if name in feature_lookup]

        explanation = (
            f"Trained {model_name} using {len(unique_features)} selected feature(s). "
            f"Validation R2={validation_metrics.r2:.4f}, RMSE={validation_metrics.rmse:.4f}, "
            f"MAE={validation_metrics.mae:.4f}."
        )

        return {
            "task": "regression",
            "model_name": model_name,
            "selected_features": selected_feature_payload,
            "train_metrics": self._regression_metrics_payload(train_metrics),
            "validation_metrics": self._regression_metrics_payload(validation_metrics),
            "explanation": explanation,
            "validation_diagnostics_plot_url": self.figure_to_data_url(diagnostics_fig),
            "error_plot_url": self.figure_to_data_url(history_fig),
            "training_history": self._history_payload(trained.history),
            "cv_metric_name": cv_metric_name,
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

    def custom_learning(
        self,
        task: TaskType,
        model_name: str,
        feature_names: List[str],
        cv_folds: int = 1,
        persist_artifact: bool = False,
        artifact_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "cv_folds": cv_folds,
            "test_size": self.settings.test_size,
            "iterations": self.settings.training_history_iterations,
            "random_state": self.settings.random_state,
        }
        data_version = self._runtime_data_versions.get(task, "unknown")
        signature = self._training_signature(
            operation="custom_learning",
            task=task,
            model_name=model_name,
            feature_names=feature_names,
            params=params,
        )

        if artifact_id:
            signature = hashlib.sha256(f"artifact:{task}:{artifact_id}".encode("utf-8")).hexdigest()

        run_id = self.run_repository.start_run(
            signature=signature,
            task=task,
            operation="custom_learning",
            model_name=model_name,
            feature_names=feature_names,
            params=params,
            data_version=data_version,
        )
        start = time.perf_counter()

        try:
            if artifact_id:
                payload = self._load_artifact_payload(task, artifact_id)
                payload["artifact_id"] = artifact_id
                payload["artifact_saved"] = True
                payload["artifact_path"] = str(self._artifact_file(task, artifact_id))
                duration_ms = (time.perf_counter() - start) * 1000.0
                self.run_repository.complete_run(
                    run_id=run_id,
                    cache_hit=True,
                    duration_ms=duration_ms,
                    metrics=self._run_metrics_summary(payload),
                    cv_summary=self._run_cv_summary(payload),
                    artifact_id=payload.get("artifact_id", ""),
                    artifact_path=payload.get("artifact_path", ""),
                    result_payload=payload,
                )
                logger.info(
                    "custom_learning_loaded_artifact task=%s model=%s artifact_id=%s",
                    task,
                    model_name,
                    artifact_id,
                )
                self._record_training_metric(
                    operation="custom_learning",
                    task=task,
                    model_name=model_name,
                    duration_ms=duration_ms,
                    cache_hit=True,
                )
                return payload

            cached = self._cache_get(signature)
            if cached is not None:
                duration_ms = (time.perf_counter() - start) * 1000.0
                self.run_repository.complete_run(
                    run_id=run_id,
                    cache_hit=True,
                    duration_ms=duration_ms,
                    metrics=self._run_metrics_summary(cached),
                    cv_summary=self._run_cv_summary(cached),
                    artifact_id=cached.get("artifact_id", ""),
                    artifact_path=cached.get("artifact_path", ""),
                    result_payload=cached,
                )
                logger.info(
                    "custom_learning_completed task=%s model=%s features=%d cv_folds=%d cache_hit=true",
                    task,
                    model_name,
                    len(feature_names),
                    cv_folds,
                )
                self._record_training_metric(
                    operation="custom_learning",
                    task=task,
                    model_name=model_name,
                    duration_ms=duration_ms,
                    cache_hit=True,
                )
                return cached

            persisted_cached = self.run_repository.get_cached_result(signature)
            if persisted_cached is not None:
                payload = copy.deepcopy(persisted_cached)
                payload.setdefault("artifact_id", "")
                payload.setdefault("artifact_saved", False)
                payload.setdefault("artifact_path", "")

                if persist_artifact and not payload.get("artifact_saved"):
                    persisted_artifact_id = signature[:16]
                    artifact_path = self._persist_artifact_payload(task, persisted_artifact_id, payload)
                    payload["artifact_id"] = persisted_artifact_id
                    payload["artifact_saved"] = True
                    payload["artifact_path"] = artifact_path

                self._cache_set(signature, payload)
                duration_ms = (time.perf_counter() - start) * 1000.0
                self.run_repository.complete_run(
                    run_id=run_id,
                    cache_hit=True,
                    duration_ms=duration_ms,
                    metrics=self._run_metrics_summary(payload),
                    cv_summary=self._run_cv_summary(payload),
                    artifact_id=payload.get("artifact_id", ""),
                    artifact_path=payload.get("artifact_path", ""),
                    result_payload=payload,
                )
                logger.info(
                    "custom_learning_completed task=%s model=%s features=%d cv_folds=%d cache_hit=true cache_source=persistent",
                    task,
                    model_name,
                    len(feature_names),
                    cv_folds,
                )
                self._record_training_metric(
                    operation="custom_learning",
                    task=task,
                    model_name=model_name,
                    duration_ms=duration_ms,
                    cache_hit=True,
                )
                return payload

            if task == "classification":
                payload = self.custom_classification_learning(
                    model_name=model_name,
                    feature_names=feature_names,
                    cv_folds=cv_folds,
                )
            else:
                payload = self.custom_regression_learning(
                    model_name=model_name,
                    feature_names=feature_names,
                    cv_folds=cv_folds,
                )

            payload.setdefault("artifact_id", "")
            payload.setdefault("artifact_saved", False)
            payload.setdefault("artifact_path", "")

            if persist_artifact:
                persisted_artifact_id = signature[:16]
                artifact_path = self._persist_artifact_payload(task, persisted_artifact_id, payload)
                payload["artifact_id"] = persisted_artifact_id
                payload["artifact_saved"] = True
                payload["artifact_path"] = artifact_path

            self._cache_set(signature, payload)
            self.run_repository.set_cached_result(
                signature=signature,
                task=task,
                operation="custom_learning",
                model_name=model_name,
                feature_names=feature_names,
                params=params,
                data_version=data_version,
                result_payload=payload,
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.run_repository.complete_run(
                run_id=run_id,
                cache_hit=False,
                duration_ms=duration_ms,
                metrics=self._run_metrics_summary(payload),
                cv_summary=self._run_cv_summary(payload),
                artifact_id=payload.get("artifact_id", ""),
                artifact_path=payload.get("artifact_path", ""),
                result_payload=payload,
            )
            logger.info(
                "custom_learning_completed task=%s model=%s features=%d cv_folds=%d duration_ms=%.2f cache_hit=false artifact_saved=%s",
                task,
                model_name,
                len(feature_names),
                cv_folds,
                duration_ms,
                bool(payload.get("artifact_saved")),
            )
            self._record_training_metric(
                operation="custom_learning",
                task=task,
                model_name=model_name,
                duration_ms=duration_ms,
                cache_hit=False,
            )
            return payload
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.run_repository.fail_run(
                run_id=run_id,
                duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def regression_diagnostics(
        self,
        model_name: str,
    ) -> Tuple[RegressionMetrics, str, str, Dict[str, Any], str]:
        signature = self._training_signature(
            operation="regression_diagnostics",
            task="regression",
            model_name=model_name,
            feature_names=[],
            params={
                "test_size": self.settings.test_size,
                "iterations": self.settings.training_history_iterations,
                "random_state": self.settings.random_state,
            },
        )
        cached = self._cache_get(signature)
        if cached is not None:
            self._record_training_metric(
                operation="regression_diagnostics",
                task="regression",
                model_name=model_name,
                duration_ms=0.0,
                cache_hit=True,
            )
            logger.info("regression_diagnostics_completed model=%s cache_hit=true", model_name)
            return cached

        start = time.perf_counter()
        runtime = self._runtime("regression")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model, history = self._train_with_history("regression", model_name)

        y_pred = model.predict(runtime.X_test)
        metrics = evaluate_regressor(model, runtime.X_test, runtime.y_test)

        diagnostics_fig = plot_residual_analysis(
            runtime.y_test,
            np.asarray(y_pred),
            model_name=model_name,
            r2=metrics.r2,
        )
        history_fig = plot_training_error_history(
            history,
            title=f"{model_name} - Train/Validation Error by Iteration",
        )
        explanation = (
            f"{model_name} regression diagnostics: R2={metrics.r2:.4f}, RMSE={metrics.rmse:.4f}, MAE={metrics.mae:.4f}."
        )
        payload = (
            metrics,
            explanation,
            self.figure_to_data_url(diagnostics_fig),
            self._history_payload(history),
            self.figure_to_data_url(history_fig),
        )
        self._cache_set(signature, payload)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "regression_diagnostics_completed model=%s duration_ms=%.2f cache_hit=false",
            model_name,
            duration_ms,
        )
        self._record_training_metric(
            operation="regression_diagnostics",
            task="regression",
            model_name=model_name,
            duration_ms=duration_ms,
            cache_hit=False,
        )
        return payload

    def compare_models(self, task: TaskType) -> Tuple[List[Dict[str, Any]], str, str, str]:
        runtime = self._runtime(task)
        models = self.list_models(task)
        X_train_fit = runtime.X_train
        y_train_fit = runtime.y_train

        # Keep all classification samples; downsample larger regression training sets
        # so the interactive dashboard remains responsive.
        if task == "regression" and len(runtime.X_train) > 1500:
            rng = np.random.default_rng(self.settings.random_state)
            sample_idx = rng.choice(len(runtime.X_train), size=1500, replace=False)
            X_train_fit = runtime.X_train[sample_idx]
            y_train_fit = runtime.y_train[sample_idx]

        rows: List[Dict[str, Any]] = []
        for model_name in models:
            model = self._build_model(task, model_name)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    model.fit(X_train_fit, y_train_fit)

                if task == "classification":
                    cls_metrics = evaluate_classifier(model, runtime.X_test, runtime.y_test, runtime.class_names)
                    rows.append(
                        {
                            "Model": model_name,
                            "Accuracy": float(cls_metrics.accuracy),
                            "Balanced Accuracy": float(cls_metrics.balanced_accuracy),
                            "Precision": float(cls_metrics.precision),
                            "Recall": float(cls_metrics.recall),
                            "F1 Score": float(cls_metrics.f1_score),
                        }
                    )
                else:
                    reg_metrics = evaluate_regressor(model, runtime.X_test, runtime.y_test)
                    rows.append(
                        {
                            "Model": model_name,
                            "R2": float(reg_metrics.r2),
                            "RMSE": float(reg_metrics.rmse),
                            "MAE": float(reg_metrics.mae),
                        }
                    )
            except Exception as exc:  # pragma: no cover - estimator-specific runtime failures
                logger.warning("Skipping model '%s' for task '%s': %s", model_name, task, exc)
                continue

        if not rows:
            raise ValueError(f"No models could be trained for task '{task}'.")

        results_df = pd.DataFrame(rows)

        if task == "classification":
            results_df = results_df.sort_values("F1 Score", ascending=False)
            metric_for_best = "F1 Score"
            best_model = str(results_df.iloc[0]["Model"])
            fig = plot_model_comparison_detailed(
                results_df,
                metrics=["Accuracy", "F1 Score", "Balanced Accuracy"],
                higher_better=[True, True, True],
                title="Classification Model Comparison",
            )
        else:
            results_df = results_df.sort_values("R2", ascending=False)
            metric_for_best = "R2"
            best_model = str(results_df.iloc[0]["Model"])
            fig = plot_model_comparison_detailed(
                results_df,
                metrics=["R2", "RMSE", "MAE"],
                higher_better=[True, False, False],
                title="Regression Model Comparison",
            )

        rounded_rows: List[Dict[str, Any]] = []
        for _, row in results_df.iterrows():
            item: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, float):
                    item[key] = round(value, 4)
                else:
                    item[key] = value
            rounded_rows.append(item)

        return rounded_rows, best_model, metric_for_best, self.figure_to_data_url(fig)


def build_service(settings: AppSettings) -> AnalyticsService:
    """Build and initialize the analytics service."""
    logger.info("Initializing analytics service")
    return AnalyticsService(settings)
