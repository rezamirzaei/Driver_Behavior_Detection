#!/usr/bin/env python
"""Complete pipeline tests.

These are intentionally lightweight smoke tests to ensure the end-to-end
pipelines run without errors in a clean environment.

Notes:
- Avoid hard-coded absolute paths so tests run on any machine.
- Pytest tests should use asserts (not return values).
"""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")


def test_classification_pipeline():
    """Test the full classification pipeline."""
    # Ensure relative paths (like data/UAH-DRIVESET-v1) resolve from repo root
    # without relying on a developer-specific absolute path.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    from src.data import load_uah_driveset, split_data

    dataset = load_uah_driveset("data/UAH-DRIVESET-v1")
    assert dataset.n_samples > 0

    split = split_data(dataset, test_size=0.2, stratify=True)

    from src.features import encode_target, preprocess_features

    train_feat, test_feat = preprocess_features(split, scaler_type="robust")
    y_train, y_test, encoder = encode_target(split.y_train, split.y_test)

    from sklearn.ensemble import RandomForestClassifier

    from src.models import evaluate_classifier, train_model

    rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")
    trained = train_model(
        train_feat.X,
        y_train,
        rf,
        "Random Forest",
        X_val=test_feat.X,
        y_val=y_test,
    )

    metrics = evaluate_classifier(trained.model, test_feat.X, y_test, list(encoder.classes_))

    # Smoke assertions: not about hitting a specific metric, but ensuring
    # the pipeline produces sensible outputs.
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert metrics.confusion_matrix is not None


def test_regression_pipeline():
    """Test the regression pipeline with sample data."""
    import numpy as np
    import pandas as pd

    from src.core import Dataset, DatasetInfo
    from src.data import split_data
    from src.features import preprocess_features
    from src.models import evaluate_regressor, train_model

    # Synthetic regression dataset (keeps tests fast and offline)
    rng = np.random.default_rng(42)
    n_samples = 500

    X = pd.DataFrame(
        {
            "feature1": rng.standard_normal(n_samples),
            "feature2": rng.standard_normal(n_samples),
            "feature3": rng.standard_normal(n_samples),
            "category": rng.choice(["A", "B", "C"], n_samples),
        }
    )
    y = pd.Series(3 * X["feature1"] - 2 * X["feature2"] + rng.standard_normal(n_samples) * 0.5)

    dataset = Dataset(
        X=X,
        y=y,
        feature_names=list(X.columns),
        target_name="target",
        info=DatasetInfo(
            name="Synthetic",
            n_samples=n_samples,
            n_features=4,
            feature_names=list(X.columns),
            target_name="target",
            task_type="regression",
        ),
    )

    split = split_data(dataset, test_size=0.2, stratify=False)
    train_feat, test_feat = preprocess_features(split, scaler_type="robust")

    y_train = split.y_train.values if hasattr(split.y_train, "values") else split.y_train
    y_test = split.y_test.values if hasattr(split.y_test, "values") else split.y_test

    from sklearn.linear_model import HuberRegressor, LinearRegression

    trained_ols = train_model(train_feat.X, y_train, LinearRegression(), "OLS")
    metrics_ols = evaluate_regressor(trained_ols.model, test_feat.X, y_test)

    trained_huber = train_model(
        train_feat.X,
        y_train,
        HuberRegressor(epsilon=1.35, max_iter=200),
        "Huber",
    )
    metrics_huber = evaluate_regressor(trained_huber.model, test_feat.X, y_test)

    # Smoke assertions
    assert metrics_ols.rmse >= 0.0
    assert -1.0 <= metrics_ols.r2 <= 1.0
    assert metrics_huber.rmse >= 0.0
    assert -1.0 <= metrics_huber.r2 <= 1.0


if __name__ == "__main__":
    # Allow running as a standalone script too.
    test_classification_pipeline()
    test_regression_pipeline()
    print("ALL TESTS PASSED ✅")
