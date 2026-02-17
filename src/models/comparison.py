"""
Model comparison utilities.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

from src.core.schemas import ModelComparison
from src.models.evaluation import evaluate_classifier, evaluate_regressor
from src.models.trainer import train_model


def get_classifiers(
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Dict[str, BaseEstimator]:
    """
    Get dictionary of classifiers for comparison.

    Includes:
    - Linear models (L1/L2 regularization)
    - SVM (linear L1, RBF, polynomial kernels)
    - KNN (K-Nearest Neighbors with different k values)
    - Ensemble methods (RF, GradientBoosting)
    """
    return {
        # Linear models
        "Logistic (L2)": LogisticRegression(
            penalty='l2', class_weight=class_weight, max_iter=1000, random_state=random_state
        ),
        "Logistic (L1 Sparse)": LogisticRegression(
            penalty='l1', solver='saga', class_weight=class_weight,
            max_iter=1000, random_state=random_state
        ),
        "Logistic (ElasticNet)": LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            class_weight=class_weight, max_iter=1000, random_state=random_state
        ),
        # SVM variants
        "SVM (Linear L1)": LinearSVC(
            penalty='l1', dual=False, class_weight=class_weight,
            max_iter=2000, random_state=random_state
        ),
        "SVM (Linear L2)": LinearSVC(
            penalty='l2', class_weight=class_weight,
            max_iter=2000, random_state=random_state
        ),
        "SVM (RBF Kernel)": SVC(
            kernel='rbf', class_weight=class_weight,
            random_state=random_state, probability=True
        ),
        "SVM (Poly Kernel)": SVC(
            kernel='poly', degree=3, class_weight=class_weight,
            random_state=random_state, probability=True
        ),
        # KNN variants (different k values and distance metrics)
        "KNN (k=3)": KNeighborsClassifier(
            n_neighbors=3, weights='uniform', metric='euclidean', n_jobs=-1
        ),
        "KNN (k=5, weighted)": KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='euclidean', n_jobs=-1
        ),
        "KNN (k=7, manhattan)": KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='manhattan', n_jobs=-1
        ),
        # Ensemble
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight=class_weight,
            random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=random_state
        ),
    }


def get_regressors(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """
    Get dictionary of regressors for comparison.

    Includes:
    - OLS baseline
    - Regularized (Ridge L2, Lasso L1, ElasticNet)
    - Robust (Huber, RANSAC)
    - SVM (linear, RBF kernel)
    - KNN (K-Nearest Neighbors with different k values)
    - Ensemble (RF, GradientBoosting)
    """
    return {
        # Baseline
        "OLS (Baseline)": LinearRegression(),
        # Regularized
        "Ridge (L2)": Ridge(alpha=1.0, random_state=random_state),
        "Lasso (L1 Sparse)": Lasso(alpha=0.1, max_iter=2000, random_state=random_state),
        "ElasticNet (L1+L2)": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=random_state),
        # Robust
        "Huber (Robust)": HuberRegressor(epsilon=1.35, max_iter=1000),
        # SVM
        "SVR (Linear)": LinearSVR(epsilon=0.1, max_iter=2000, random_state=random_state),
        "SVR (RBF Kernel)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        # KNN variants (different k values and distance metrics)
        "KNN (k=3)": KNeighborsRegressor(
            n_neighbors=3, weights='uniform', metric='euclidean', n_jobs=-1
        ),
        "KNN (k=5, weighted)": KNeighborsRegressor(
            n_neighbors=5, weights='distance', metric='euclidean', n_jobs=-1
        ),
        "KNN (k=7, manhattan)": KNeighborsRegressor(
            n_neighbors=7, weights='distance', metric='manhattan', n_jobs=-1
        ),
        # Ensemble
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=random_state
        ),
    }


def get_knn_classifiers() -> Dict[str, BaseEstimator]:
    """
    Get dictionary of KNN classifiers with different configurations.

    Returns:
        Dict with KNN classifiers varying in:
        - k values (3, 5, 7)
        - Weight schemes (uniform, distance)
        - Distance metrics (euclidean, manhattan)
    """
    return {
        "KNN (k=3)": KNeighborsClassifier(
            n_neighbors=3, weights='uniform', metric='euclidean', n_jobs=-1
        ),
        "KNN (k=5, weighted)": KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='euclidean', n_jobs=-1
        ),
        "KNN (k=7, manhattan)": KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='manhattan', n_jobs=-1
        ),
    }


def get_knn_regressors() -> Dict[str, BaseEstimator]:
    """
    Get dictionary of KNN regressors with different configurations.

    Returns:
        Dict with KNN regressors varying in:
        - k values (3, 5, 7)
        - Weight schemes (uniform, distance)
        - Distance metrics (euclidean, manhattan)
    """
    return {
        "KNN (k=3)": KNeighborsRegressor(
            n_neighbors=3, weights='uniform', metric='euclidean', n_jobs=-1
        ),
        "KNN (k=5, weighted)": KNeighborsRegressor(
            n_neighbors=5, weights='distance', metric='euclidean', n_jobs=-1
        ),
        "KNN (k=7, manhattan)": KNeighborsRegressor(
            n_neighbors=7, weights='distance', metric='manhattan', n_jobs=-1
        ),
    }


def compare_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Optional[Dict[str, BaseEstimator]] = None,
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> ModelComparison:
    """
    Compare multiple classifiers.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        models: Dict of models. None = use defaults.
        class_names: Class names.
        feature_names: Feature names.

    Returns:
        ModelComparison with results.
    """
    if models is None:
        models = get_classifiers()

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        trained = train_model(
            X_train, y_train,
            clone(model), name,
            X_val=X_test, y_val=y_test,
            feature_names=feature_names
        )

        metrics = evaluate_classifier(trained.model, X_test, y_test, class_names)
        trained.test_metrics = metrics

        results.append({
            "Model": name,
            "Accuracy": metrics.accuracy,
            "Balanced Accuracy": metrics.balanced_accuracy,
            "Precision": metrics.precision,
            "Recall": metrics.recall,
            "F1 Score": metrics.f1_score,
        })

        trained_models[name] = trained

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]

    return ModelComparison(
        results=results_df,
        trained_models=trained_models,
        best_model_name=best_model_name,
        metric_used="F1 Score"
    )


def compare_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Optional[Dict[str, BaseEstimator]] = None,
    feature_names: Optional[List[str]] = None,
) -> ModelComparison:
    """
    Compare multiple regressors.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        models: Dict of models. None = use defaults.
        feature_names: Feature names.

    Returns:
        ModelComparison with results.
    """
    if models is None:
        models = get_regressors()

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")

        try:
            trained = train_model(
                X_train, y_train,
                clone(model), name,
                X_val=X_test, y_val=y_test,
                feature_names=feature_names
            )

            metrics = evaluate_regressor(trained.model, X_test, y_test)
            trained.test_metrics = metrics

            results.append({
                "Model": name,
                "RMSE": metrics.rmse,
                "MAE": metrics.mae,
                "R²": metrics.r2,
                "MAPE (%)": metrics.mape,
            })

            trained_models[name] = trained

        except Exception as e:
            print(f"  Warning: {name} failed - {e}")
            continue

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]

    return ModelComparison(
        results=results_df,
        trained_models=trained_models,
        best_model_name=best_model_name,
        metric_used="R²"
    )
