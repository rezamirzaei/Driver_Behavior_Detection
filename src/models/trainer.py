"""
Model training module.
"""

from copy import deepcopy
from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import accuracy_score, mean_squared_error

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
        X_train_arr = np.asarray(X_train)
        y_train_arr = np.asarray(y_train)
        X_val_arr = np.asarray(X_val) if X_val is not None else None
        y_val_arr = np.asarray(y_val) if y_val is not None else None

        model_is_classifier = is_classifier(self.model)

        # Check if model supports staged prediction
        supports_staged_predict = hasattr(self.model, "staged_predict")

        if supports_staged_predict:
            self._train_iterative(X_train_arr, y_train_arr, X_val_arr, y_val_arr, n_iterations, model_is_classifier)
        else:
            self._train_single(
                X_train_arr,
                y_train_arr,
                X_val_arr,
                y_val_arr,
                model_is_classifier,
                n_iterations=n_iterations,
            )

        return TrainedModel(
            model=self.model, model_name=self.model_name, history=self.history, feature_names=feature_names
        )

    @staticmethod
    def _clone_model(model: BaseEstimator) -> BaseEstimator:
        """Clone estimator and fall back to deepcopy for custom estimators."""
        try:
            return clone(model)
        except Exception:
            return deepcopy(model)

    @staticmethod
    def _metric_name(is_classifier_task: bool) -> str:
        return "accuracy" if is_classifier_task else "rmse"

    @staticmethod
    def _score(y_true: np.ndarray, y_pred: np.ndarray, is_classifier_task: bool) -> float:
        if is_classifier_task:
            return float(accuracy_score(y_true, y_pred))
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

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
            self.model.set_params(n_estimators=max(2, n_iterations))

        self.model.fit(X_train, y_train)

        self.history.metric_name = self._metric_name(is_classifier)

        val_iter = None
        if X_val is not None and y_val is not None:
            val_iter = self.model.staged_predict(X_val)

        for i, y_pred_train in enumerate(self.model.staged_predict(X_train), start=1):
            train_score = self._score(y_train, y_pred_train, is_classifier_task=is_classifier)

            val_score = None
            if val_iter is not None and y_val is not None:
                y_pred_val = next(val_iter, None)
                if y_pred_val is not None:
                    val_score = self._score(y_val, y_pred_val, is_classifier_task=is_classifier)

            self.history.add(i, train_score, val_score)

    def _train_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        is_classifier: bool,
        n_iterations: int,
    ) -> None:
        """Train models without staged_predict using progressive subsets for history."""
        self.history.metric_name = self._metric_name(is_classifier)
        n_steps = max(1, min(20, int(n_iterations)))

        if X_val is not None and y_val is not None and n_steps > 1 and len(X_train) > 8:
            self._build_progressive_history(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                is_classifier=is_classifier,
                n_steps=n_steps,
            )

        # Final fit always uses full training data.
        self.model.fit(X_train, y_train)

        if not self.history.iterations:
            y_pred_train = self.model.predict(X_train)
            train_score = self._score(y_train, y_pred_train, is_classifier_task=is_classifier)

            val_score = None
            if X_val is not None and y_val is not None:
                y_pred_val = self.model.predict(X_val)
                val_score = self._score(y_val, y_pred_val, is_classifier_task=is_classifier)

            self.history.add(1, train_score, val_score)

    def _build_progressive_history(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        is_classifier: bool,
        n_steps: int,
    ) -> None:
        """Fit cloned models on progressively larger subsets to emulate iteration curves."""
        total_samples = int(len(X_train))
        capped_samples = min(total_samples, 1500)
        if capped_samples < 4:
            return

        rng = np.random.default_rng(42)
        sampled_indices = rng.permutation(total_samples)[:capped_samples]

        min_size = max(10, int(capped_samples / max(n_steps, 2)))
        if is_classifier:
            min_size = max(min_size, int(len(np.unique(y_train))))

        step_sizes = np.linspace(min_size, capped_samples, num=n_steps, dtype=int)
        step_sizes = np.unique(step_sizes)

        history_rows: List[tuple[int, float, Optional[float]]] = []
        for step_index, size in enumerate(step_sizes, start=1):
            subset_indices = sampled_indices[: int(size)]
            X_subset = X_train[subset_indices]
            y_subset = y_train[subset_indices]

            model_for_step = self._clone_model(self.model)
            try:
                model_for_step.fit(X_subset, y_subset)
                train_pred = model_for_step.predict(X_subset)
                train_score = self._score(y_subset, train_pred, is_classifier_task=is_classifier)

                val_pred = model_for_step.predict(X_val)
                val_score = self._score(y_val, val_pred, is_classifier_task=is_classifier)
                history_rows.append((step_index, train_score, val_score))
            except Exception:
                # Some estimators may fail on very small subsets.
                continue

        for iteration, train_score, val_score in history_rows:
            self.history.add(iteration, train_score, val_score)


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
