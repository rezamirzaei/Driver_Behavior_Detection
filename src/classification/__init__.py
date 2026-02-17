"""
Classification Pipeline Module.

This module provides a complete, clean API for driver behavior classification.
All logic is encapsulated here - notebooks only call these functions.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.classification.data import (
    TripInfo as TripInfo,
    build_raw_dataset as build_raw_dataset,
    compute_acceleration_magnitude as compute_acceleration_magnitude,
    extract_raw_features as extract_raw_features,
    get_all_trips as get_all_trips,
    load_inertial_events as load_inertial_events,
    load_or_build_dataset as load_or_build_dataset,
    load_raw_accelerometer as load_raw_accelerometer,
    load_raw_gps as load_raw_gps,
)

# Import submodules
from src.classification.sparse_models import MCPLogisticRegression, SCADLogisticRegression

# Import types first (no dependencies)
from src.classification.types import ClassificationResult, DataSplit
from src.classification.visualization import (
    plot_class_distribution as plot_class_distribution,
    plot_confusion_matrix as plot_confusion_matrix,
    plot_correlation_matrix as plot_correlation_matrix,
    plot_driver_distribution as plot_driver_distribution,
    plot_feature_distributions as plot_feature_distributions,
    plot_feature_importance as plot_feature_importance,
    plot_model_comparison as plot_model_comparison,
)


def get_all_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Get all classification models to compare.

    Returns:
        Dictionary of model_name -> model instance
    """
    return {
        # Linear Models
        "Logistic (L2)": LogisticRegression(
            penalty="l2", max_iter=1000, random_state=random_state, class_weight="balanced"
        ),
        "Logistic (L1)": LogisticRegression(
            penalty="l1", solver="saga", max_iter=2000, random_state=random_state, class_weight="balanced"
        ),
        "Logistic (ElasticNet)": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Logistic (MCP)": MCPLogisticRegression(alpha=0.1, gamma=3.0, max_iter=1000, random_state=random_state),
        "Logistic (SCAD)": SCADLogisticRegression(alpha=0.1, a=3.7, max_iter=1000, random_state=random_state),
        # SVM
        "SVM (Linear)": SVC(kernel="linear", random_state=random_state, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", random_state=random_state, class_weight="balanced"),
        "SVM (Poly)": SVC(kernel="poly", degree=3, random_state=random_state, class_weight="balanced"),
        # KNN
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan"),
        # Trees
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=random_state, class_weight="balanced"),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=10, random_state=random_state, class_weight="balanced"
        ),
        # Ensemble
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state),
        # Neural Network
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state),
        # Probabilistic
        "Naive Bayes": GaussianNB(),
    }


def prepare_classification_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "behavior",
    driver_col: str = "driver",
    test_drivers: List[str] = ["D6"],
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """
    Prepare data for classification with driver-level splitting.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names (auto-detect if None)
        target_col: Name of target column
        driver_col: Name of driver column
        test_drivers: List of drivers to hold out for testing
        test_size: Target test size ratio
        random_state: Random seed

    Returns:
        DataSplit object with all prepared data
    """
    from src.data import split_by_driver

    # Auto-detect feature columns
    if feature_cols is None:
        exclude_cols = [target_col, driver_col, "road_type"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Prepare X with driver column for splitting
    X = df[feature_cols + [driver_col]].copy()
    X[feature_cols] = X[feature_cols].fillna(0)
    y = df[target_col].values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split by driver
    X_train, X_test, y_train, y_test = split_by_driver(
        X, y_enc, test_drivers=test_drivers, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return DataSplit(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        class_names=list(le.classes_),
        scaler=scaler,
        label_encoder=le,
    )


def train_single_classifier(model: Any, model_name: str, data: DataSplit) -> ClassificationResult:
    """
    Train a single classifier and return results.

    Args:
        model: Sklearn-compatible classifier
        model_name: Name for the model
        data: DataSplit object with train/test data

    Returns:
        ClassificationResult with metrics and predictions
    """
    model.fit(data.X_train, data.y_train)

    y_pred = model.predict(data.X_test)
    train_pred = model.predict(data.X_train)

    return ClassificationResult(
        model_name=model_name,
        train_accuracy=accuracy_score(data.y_train, train_pred),
        test_accuracy=accuracy_score(data.y_test, y_pred),
        f1_score=f1_score(data.y_test, y_pred, average="weighted"),
        predictions=y_pred,
        model=model,
    )


def train_all_classifiers(
    data: DataSplit, classifiers: Optional[Dict[str, Any]] = None, verbose: bool = True
) -> List[ClassificationResult]:
    """
    Train all classifiers and return sorted results.

    Args:
        data: DataSplit object with prepared data
        classifiers: Dict of classifiers (uses default if None)
        verbose: Print progress

    Returns:
        List of ClassificationResult sorted by test accuracy
    """
    if classifiers is None:
        classifiers = get_all_classifiers()

    results = []
    for name, model in classifiers.items():
        try:
            result = train_single_classifier(model, name, data)
            results.append(result)
            if verbose:
                print(
                    f"✅ {name}: Train={result.train_accuracy:.3f}, "
                    f"Test={result.test_accuracy:.3f}, F1={result.f1_score:.3f}"
                )
        except Exception as e:
            if verbose:
                print(f"❌ {name}: {e}")

    # Sort by test accuracy
    results.sort(key=lambda x: x.test_accuracy, reverse=True)
    return results


def get_feature_importance(model: Any, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model (LogisticRegression or tree-based)
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with Feature and Importance columns
    """
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models - use absolute coefficient values
        if model.coef_.ndim == 1:
            importance = np.abs(model.coef_)
        else:
            importance = np.abs(model.coef_).mean(axis=0)
    else:
        raise ValueError(f"Model {type(model)} doesn't support feature importance")

    df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values("Importance", ascending=False)

    return df.head(top_n)


def run_logo_cv(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, models: Optional[Dict[str, Any]] = None, verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run Leave-One-Group-Out cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels (driver IDs)
        models: Dict of models to evaluate
        verbose: Print results

    Returns:
        Dict of model_name -> {mean, std, scores}
    """
    if models is None:
        models = {
            "Logistic (L2)": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5),
        }

    logo = LeaveOneGroupOut()
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=logo, groups=groups, scoring="accuracy")
        results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
        if verbose:
            print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


def results_to_dataframe(results: List[ClassificationResult]) -> pd.DataFrame:
    """Convert list of ClassificationResult to DataFrame."""
    return pd.DataFrame(
        [
            {
                "Model": r.model_name,
                "Train Acc": r.train_accuracy,
                "Test Acc": r.test_accuracy,
                "F1-Score": r.f1_score,
                "Overfit": r.train_accuracy - r.test_accuracy,
            }
            for r in results
        ]
    )


def get_best_model(results: List[ClassificationResult]) -> ClassificationResult:
    """Get the best model by test accuracy."""
    return max(results, key=lambda x: x.test_accuracy)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> str:
    """Get formatted classification report."""
    return classification_report(y_true, y_pred, target_names=class_names)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Get confusion matrix."""
    return confusion_matrix(y_true, y_pred)
