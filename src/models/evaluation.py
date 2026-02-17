"""
Model evaluation module.
"""

from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.core.schemas import ClassificationMetrics, RegressionMetrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute R², RMSE, MAE for regression.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Tuple of (r2, rmse, mae).
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def train_and_evaluate_regressor(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Train model and compute metrics.

    Args:
        model: Sklearn regressor.
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        Tuple of (predictions, r2, rmse, mae).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2, rmse, mae = compute_regression_metrics(y_test, y_pred)
    return y_pred, r2, rmse, mae


def train_and_evaluate_classifier(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Train classifier and compute metrics.

    Args:
        model: Sklearn classifier.
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        Tuple of (predictions, accuracy, f1_score).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return y_pred, acc, f1


class QuantileRegressor:
    """
    Quantile regression for prediction intervals.

    Uses GradientBoostingRegressor with quantile loss.
    """

    def __init__(
        self,
        quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9),
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            quantiles: (lower, median, upper) quantiles for prediction interval.
            n_estimators: Number of boosting stages.
            max_depth: Max tree depth.
            random_state: Random seed.
        """
        self.quantiles = quantiles
        self.models: Dict[float, Any] = {}
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileRegressor":
        """Fit quantile models."""
        from sklearn.ensemble import GradientBoostingRegressor

        for q in self.quantiles:
            self.models[q] = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            self.models[q].fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict with intervals.

        Returns:
            Dict with 'lower', 'median', 'upper' predictions.
        """
        q_lower, q_median, q_upper = self.quantiles
        return {
            "lower": self.models[q_lower].predict(X),
            "median": self.models[q_median].predict(X),
            "upper": self.models[q_upper].predict(X),
        }

    def predict_interval(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with interval bounds.

        Returns:
            Tuple of (lower_bound, point_estimate, upper_bound).
        """
        preds = self.predict(X)
        return preds["lower"], preds["median"], preds["upper"]


def compute_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Compute what fraction of true values fall within prediction interval.

    Args:
        y_true: True values.
        lower: Lower bounds.
        upper: Upper bounds.

    Returns:
        Coverage fraction (0-1).
    """
    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


def compute_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute average prediction interval width."""
    return float(np.mean(upper - lower))


class ClassificationEvaluator:
    """Evaluator for classification models."""

    def evaluate(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> ClassificationMetrics:
        """
        Evaluate a classification model.

        Args:
            model: Trained classifier.
            X_test: Test features.
            y_test: True labels.
            class_names: Names of classes.

        Returns:
            ClassificationMetrics.
        """
        y_pred = model.predict(X_test)

        if class_names is None:
            if hasattr(model, "classes_"):
                class_names = [str(c) for c in model.classes_]
            else:
                class_names = [str(c) for c in np.unique(y_test)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return ClassificationMetrics(
                accuracy=accuracy_score(y_test, y_pred),
                balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
                recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
                f1_score=f1_score(y_test, y_pred, average="weighted", zero_division=0),
                confusion_matrix=confusion_matrix(y_test, y_pred),
                class_names=class_names,
                report=classification_report(y_test, y_pred, target_names=class_names, zero_division=0),
            )


class RegressionEvaluator:
    """Evaluator for regression models."""

    def evaluate(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> RegressionMetrics:
        """
        Evaluate a regression model.

        Args:
            model: Trained regressor.
            X_test: Test features.
            y_test: True values.

        Returns:
            RegressionMetrics.
        """
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # MAPE
        mape = None
        nonzero_mask = y_test != 0
        if np.any(nonzero_mask):
            mape = np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])) * 100

        return RegressionMetrics(mse=mse, rmse=np.sqrt(mse), mae=mae, r2=r2, mape=mape)


def evaluate_classifier(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """Evaluate a classification model."""
    evaluator = ClassificationEvaluator()
    return evaluator.evaluate(model, X_test, y_test, class_names)


def evaluate_regressor(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> RegressionMetrics:
    """Evaluate a regression model."""
    evaluator = RegressionEvaluator()
    return evaluator.evaluate(model, X_test, y_test)
