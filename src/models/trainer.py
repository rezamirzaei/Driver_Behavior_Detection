"""
Model training module.
"""

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score

from src.core.schemas import TrainedModel, TrainingHistory


class ModelTrainer:
    """Trainer for sklearn models with iteration tracking."""

    def __init__(self, model: BaseEstimator, model_name: str):
        self.model = model
        self.model_name = model_name
        self.history = TrainingHistory()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_iterations: int = 100,
        feature_names: Optional[List[str]] = None,
    ) -> TrainedModel:
        """
        Train the model with optional iteration tracking.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            n_iterations: Number of iterations for iterative models.
            feature_names: List of feature names.

        Returns:
            TrainedModel with trained model and history.
        """
        feature_names = feature_names or []
        is_classifier = hasattr(self.model, "classes_") or "Classifier" in type(self.model).__name__

        # Check if model supports staged prediction
        is_gradient_boosting = hasattr(self.model, "staged_predict")

        if is_gradient_boosting:
            self._train_iterative(X_train, y_train, X_val, y_val, n_iterations, is_classifier)
        else:
            self._train_single(X_train, y_train, X_val, y_val, is_classifier)

        return TrainedModel(
            model=self.model, model_name=self.model_name, history=self.history, feature_names=feature_names
        )

    def _train_iterative(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        n_iterations: int,
        is_classifier: bool,
    ) -> None:
        """Train with iteration tracking for Gradient Boosting models."""
        if hasattr(self.model, "n_estimators"):
            self.model.set_params(n_estimators=n_iterations)

        self.model.fit(X_train, y_train)

        self.history.metric_name = "accuracy" if is_classifier else "r2"

        for i, y_pred_train in enumerate(self.model.staged_predict(X_train)):
            if is_classifier:
                train_score = accuracy_score(y_train, y_pred_train)
            else:
                train_score = r2_score(y_train, y_pred_train)

            val_score = None
            if X_val is not None and y_val is not None:
                y_pred_val = list(self.model.staged_predict(X_val))[i]
                if is_classifier:
                    val_score = accuracy_score(y_val, y_pred_val)
                else:
                    val_score = r2_score(y_val, y_pred_val)

            self.history.add(i + 1, train_score, val_score)

    def _train_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        is_classifier: bool,
    ) -> None:
        """Train without iteration tracking."""
        self.model.fit(X_train, y_train)

        y_pred_train = self.model.predict(X_train)

        self.history.metric_name = "accuracy" if is_classifier else "r2"

        if is_classifier:
            train_score = accuracy_score(y_train, y_pred_train)
        else:
            train_score = r2_score(y_train, y_pred_train)

        val_score = None
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            if is_classifier:
                val_score = accuracy_score(y_val, y_pred_val)
            else:
                val_score = r2_score(y_val, y_pred_val)

        self.history.add(1, train_score, val_score)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: BaseEstimator,
    model_name: str,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_iterations: int = 100,
    feature_names: Optional[List[str]] = None,
) -> TrainedModel:
    """
    Train a model with optional iteration tracking.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model: Sklearn model.
        model_name: Name for the model.
        X_val: Validation features.
        y_val: Validation labels.
        n_iterations: Number of iterations.
        feature_names: Feature names.

    Returns:
        TrainedModel with model and history.
    """
    trainer = ModelTrainer(model, model_name)
    return trainer.train(
        X_train, y_train, X_val=X_val, y_val=y_val, n_iterations=n_iterations, feature_names=feature_names
    )
