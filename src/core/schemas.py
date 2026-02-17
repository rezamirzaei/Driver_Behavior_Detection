"""
Core data structures using Pydantic for type-safe data handling.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class DrivingBehavior(str, Enum):
    """Driving behavior categories."""

    NORMAL = "NORMAL"
    AGGRESSIVE = "AGGRESSIVE"
    DROWSY = "DROWSY"


class RoadType(str, Enum):
    """Road type categories."""

    MOTORWAY = "MOTORWAY"
    SECONDARY = "SECONDARY"


class DatasetInfo(BaseModel):
    """Metadata about a loaded dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Dataset name")
    n_samples: int = Field(..., ge=0, description="Number of samples")
    n_features: int = Field(..., ge=0, description="Number of features")
    feature_names: List[str] = Field(default_factory=list)
    target_name: Optional[str] = Field(default=None)
    task_type: Literal["classification", "regression"] = Field(...)
    class_distribution: Optional[Dict[str, float]] = Field(default=None)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"Dataset: {self.name}",
            f"Samples: {self.n_samples:,}",
            f"Features: {self.n_features}",
            f"Task: {self.task_type}",
            f"Target: {self.target_name}",
        ]
        if self.class_distribution:
            lines.append("Class Distribution:")
            for cls, ratio in self.class_distribution.items():
                lines.append(f"  - {cls}: {ratio:.1%}")
        return "\n".join(lines)


class Dataset(BaseModel):
    """Container for a dataset with features and target."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: Any = Field(..., description="Feature matrix")
    y: Any = Field(..., description="Target vector")
    feature_names: List[str] = Field(default_factory=list)
    target_name: str = Field(default="target")
    info: Optional[DatasetInfo] = Field(default=None)

    @property
    def n_samples(self) -> int:
        return int(len(self.y))

    @property
    def n_features(self) -> int:
        if hasattr(self.X, "shape") and len(self.X.shape) > 1:
            return int(self.X.shape[1])
        return len(self.feature_names)


class SplitData(BaseModel):
    """Container for train/test split data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X_train: Any = Field(...)
    X_test: Any = Field(...)
    y_train: Any = Field(...)
    y_test: Any = Field(...)
    feature_names: List[str] = Field(default_factory=list)
    target_name: str = Field(default="target")

    @property
    def n_train(self) -> int:
        return int(len(self.y_train))

    @property
    def n_test(self) -> int:
        return int(len(self.y_test))


class FeatureSet(BaseModel):
    """Container for extracted/preprocessed features."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X: Any = Field(...)
    feature_names: List[str] = Field(default_factory=list)
    scaler: Optional[Any] = Field(default=None)
    encoder: Optional[Any] = Field(default=None)

    @property
    def n_features(self) -> int:
        if hasattr(self.X, "shape") and len(self.X.shape) > 1:
            return int(self.X.shape[1])
        return len(self.feature_names)


class OutlierAnalysisResult(BaseModel):
    """Result of outlier analysis using IQR method."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    column: str = Field(..., description="Column analyzed")
    lower_bound: float = Field(..., description="Lower IQR bound")
    upper_bound: float = Field(..., description="Upper IQR bound")
    n_outliers: int = Field(..., ge=0, description="Number of outliers")
    outlier_percentage: float = Field(..., ge=0, le=100, description="Percentage of outliers")
    outlier_indices: List[int] = Field(default_factory=list, description="Indices of outliers")

    @property
    def has_significant_outliers(self) -> bool:
        """True if outlier percentage > 5%."""
        return self.outlier_percentage > 5.0


class CorrelationAnalysisResult(BaseModel):
    """Result of correlation analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    correlation_matrix: Any = Field(..., description="Correlation matrix as DataFrame")
    high_correlation_pairs: List[Tuple[str, str, float]] = Field(
        default_factory=list, description="List of (feature1, feature2, correlation) with |r| > threshold"
    )
    target_correlations: Optional[Dict[str, float]] = Field(
        default=None, description="Correlations with target variable"
    )
    multicollinearity_warning: bool = Field(default=False, description="True if severe multicollinearity detected")


class ClassDistributionResult(BaseModel):
    """Result of class distribution analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    counts: Dict[str, int] = Field(..., description="Counts per class")
    percentages: Dict[str, float] = Field(..., description="Percentages per class")
    imbalance_ratio: float = Field(..., ge=0, description="Max/min class ratio")
    is_imbalanced: bool = Field(default=False, description="True if imbalance ratio > 3")
    recommendation: str = Field(default="", description="Recommendation based on analysis")


class FeatureStatisticsResult(BaseModel):
    """Statistics for a single feature."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Feature name")
    mean: float = Field(...)
    median: float = Field(...)
    std: float = Field(...)
    min: float = Field(...)
    max: float = Field(...)
    skewness: float = Field(...)
    n_missing: int = Field(default=0)
    n_unique: int = Field(default=0)


class DataQualityReport(BaseModel):
    """Comprehensive data quality report."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_samples: int = Field(..., ge=0)
    n_features: int = Field(..., ge=0)
    n_numerical: int = Field(default=0)
    n_categorical: int = Field(default=0)
    missing_values: Dict[str, int] = Field(default_factory=dict)
    feature_statistics: List[FeatureStatisticsResult] = Field(default_factory=list)
    outlier_analysis: List[OutlierAnalysisResult] = Field(default_factory=list)
    class_distribution: Optional[ClassDistributionResult] = Field(default=None)
    correlation_analysis: Optional[CorrelationAnalysisResult] = Field(default=None)

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "📊 Data Quality Report",
            f"   Samples: {self.n_samples:,}",
            f"   Features: {self.n_features} ({self.n_numerical} numerical, {self.n_categorical} categorical)",
        ]
        if self.missing_values:
            total_missing = sum(self.missing_values.values())
            lines.append(f"   Missing values: {total_missing}")
        if self.class_distribution:
            lines.append(f"   Imbalanced: {'Yes' if self.class_distribution.is_imbalanced else 'No'}")
        if self.correlation_analysis and self.correlation_analysis.multicollinearity_warning:
            lines.append("   ⚠️ Multicollinearity detected")
        return "\n".join(lines)


class ScoreMappingInfo(BaseModel):
    """Documentation for UAH-DriveSet score mapping from accelerometer data.

    The UAH-DriveSet uses the DriveSafe app which computes behavioral scores
    from raw accelerometer and GPS data. This class documents that mapping.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    description: str = Field(
        default="""
UAH-DriveSet Score Mapping (from DriveSafe App)
================================================

The scores are computed by the DriveSafe iOS app from accelerometer data:

1. **Raw Data Collection**:
   - Accelerometer: 3-axis acceleration at ~50Hz
   - GPS: Position, speed at ~1Hz

2. **Event Detection** (from SEMANTIC_FINAL.txt):
   - Low/Medium/High Accelerations (Lacc, Macc, Hacc)
   - Low/Medium/High Brakings (Lbra, Mbra, Hbra)
   - Low/Medium/High Turnings (Ltur, Mtur, Htur)

   Events are classified by acceleration magnitude thresholds:
   - Low: 0.2-0.5g
   - Medium: 0.5-0.8g
   - High: >0.8g

3. **Score Computation** (0-100 scale):
   - score_accelerations: Based on Lacc, Macc, Hacc counts
   - score_brakings: Based on Lbra, Mbra, Hbra counts
   - score_turnings: Based on Ltur, Mtur, Htur counts
   - score_weaving: Lane discipline from lateral movement
   - score_drifting: Lane keeping from GPS trajectory
   - score_overspeeding: Speed compliance vs road limits
   - score_following: Following distance estimation
   - score_total: Mean of all 7 scores

4. **Behavior Ratios** (0-1 scale):
   - ratio_normal: Proportion of trip with normal driving
   - ratio_drowsy: Proportion with drowsy indicators
   - ratio_aggressive: Proportion with aggressive indicators

   Computed using sliding 60-second windows.

5. **Aggregation for Classification**:
   We use the FINAL scores (last row of SEMANTIC_ONLINE.txt)
   which represent the complete trip statistics.
   This avoids fixed-window assumptions for variable trip lengths.
""",
        description="Detailed explanation of score mapping",
    )

    score_features: List[str] = Field(
        default=[
            "score_total",
            "score_accelerations",
            "score_brakings",
            "score_turnings",
            "score_weaving",
            "score_drifting",
            "score_overspeeding",
            "score_following",
        ],
        description="Score features (0-100 scale)",
    )

    ratio_features: List[str] = Field(
        default=["ratio_normal", "ratio_drowsy", "ratio_aggressive"], description="Ratio features (0-1 scale)"
    )

    acceleration_thresholds: Dict[str, Tuple[float, float]] = Field(
        default={"low": (0.2, 0.5), "medium": (0.5, 0.8), "high": (0.8, float("inf"))},
        description="Acceleration thresholds in g units",
    )


class TrainingHistory(BaseModel):
    """Training history with metrics per iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    iterations: List[int] = Field(default_factory=list)
    train_scores: List[float] = Field(default_factory=list)
    val_scores: List[float] = Field(default_factory=list)
    metric_name: str = Field(default="score")

    def add(self, iteration: int, train_score: float, val_score: Optional[float] = None) -> None:
        """Add a training iteration record."""
        self.iterations.append(iteration)
        self.train_scores.append(train_score)
        if val_score is not None:
            self.val_scores.append(val_score)

    @property
    def best_iteration(self) -> int:
        """Return the iteration with best validation score."""
        if self.val_scores:
            return self.iterations[int(np.argmax(self.val_scores))]
        return self.iterations[-1] if self.iterations else 0


class ClassificationMetrics(BaseModel):
    """Classification evaluation metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    accuracy: float = Field(..., ge=0, le=1)
    balanced_accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    confusion_matrix: Any = Field(...)
    class_names: List[str] = Field(default_factory=list)
    report: str = Field(default="")

    def summary(self) -> str:
        return (
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Balanced Accuracy: {self.balanced_accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1 Score: {self.f1_score:.4f}"
        )


class RegressionMetrics(BaseModel):
    """Regression evaluation metrics."""

    mse: float = Field(..., ge=0)
    rmse: float = Field(..., ge=0)
    mae: float = Field(..., ge=0)
    r2: float = Field(...)
    mape: Optional[float] = Field(default=None)

    def summary(self) -> str:
        lines = [
            f"RMSE: {self.rmse:.4f}",
            f"MAE: {self.mae:.4f}",
            f"R²: {self.r2:.4f}",
        ]
        if self.mape is not None:
            lines.append(f"MAPE: {self.mape:.2f}%")
        return "\n".join(lines)


class TrainedModel(BaseModel):
    """Container for a trained model with history and metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(...)
    model_name: str = Field(...)
    history: TrainingHistory = Field(default_factory=TrainingHistory)
    train_metrics: Optional[Any] = Field(default=None)
    test_metrics: Optional[Any] = Field(default=None)
    feature_names: List[str] = Field(default_factory=list)

    def predict(self, X: Any) -> Any:
        """Make predictions."""
        return self.model.predict(X)


class ModelComparison(BaseModel):
    """Container for comparing multiple models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: Any = Field(...)
    trained_models: Dict[str, TrainedModel] = Field(default_factory=dict)
    best_model_name: str = Field(default="")
    metric_used: str = Field(default="")

    @property
    def best_model(self) -> Optional[TrainedModel]:
        if self.best_model_name in self.trained_models:
            return self.trained_models[self.best_model_name]
        return None
