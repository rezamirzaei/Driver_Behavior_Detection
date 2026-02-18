"""
Leave-One-Driver-Out Cross-Validation Utility.

This module provides robust cross-validation for driver behavior classification
by holding out each driver in turn as a test set.
"""

from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score


def leave_one_driver_out_cv(
    X: pd.DataFrame,
    y: pd.Series,
    driver_ids: pd.Series,
    model: BaseEstimator,
) -> Dict[str, any]:
    """
    Perform leave-one-driver-out cross-validation.

    For each unique driver, holds that driver out as the test set and trains
    on all other drivers. This provides a robust estimate of model performance
    on unseen drivers.

    Args:
        X: Feature DataFrame with shape (n_samples, n_features).
        y: Target Series with shape (n_samples,).
        driver_ids: Series of driver identifiers with shape (n_samples,).
        model: sklearn BaseEstimator to evaluate.

    Returns:
        Dictionary with:
            - 'per_fold_results': List of dicts, each containing:
                - 'driver': Driver identifier
                - 'accuracy': Test accuracy for this fold
                - 'f1': Weighted F1-score for this fold
                - 'n_test': Number of test samples
            - 'mean_accuracy': Mean accuracy across all folds
            - 'std_accuracy': Standard deviation of accuracy across folds
            - 'mean_f1': Mean F1-score across all folds
            - 'std_f1': Standard deviation of F1-score across folds

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(random_state=42)
        >>> results = leave_one_driver_out_cv(X, y, driver_ids, model)
        >>> print(f"Mean accuracy: {results['mean_accuracy']:.3f}")
    """
    unique_drivers = driver_ids.unique()
    per_fold_results: List[Dict[str, any]] = []

    for driver in unique_drivers:
        # Split data: hold out current driver as test set
        test_mask = driver_ids == driver
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Clone model to get a fresh copy for this fold
        model_fold = clone(model)

        # Train and predict
        model_fold.fit(X_train, y_train)
        y_pred = model_fold.predict(X_test)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        per_fold_results.append({
            "driver": driver,
            "accuracy": accuracy,
            "f1": f1,
            "n_test": len(y_test),
        })

    # Compute aggregate statistics
    accuracies = [fold["accuracy"] for fold in per_fold_results]
    f1_scores = [fold["f1"] for fold in per_fold_results]

    return {
        "per_fold_results": per_fold_results,
        "mean_accuracy": pd.Series(accuracies).mean(),
        "std_accuracy": pd.Series(accuracies).std(),
        "mean_f1": pd.Series(f1_scores).mean(),
        "std_f1": pd.Series(f1_scores).std(),
    }
