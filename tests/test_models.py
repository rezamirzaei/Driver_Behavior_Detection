"""Tests for models module."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LinearRegression, LogisticRegression

from src.core.schemas import TrainedModel
from src.models.comparison import get_classifiers, get_regressors
from src.models.evaluation import evaluate_classifier, evaluate_regressor
from src.models.trainer import train_model


class TestModelTrainer:
    """Tests for model trainer."""

    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test

    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 5)
        y_test = np.random.randn(20)
        return X_train, y_train, X_test, y_test

    def test_train_classifier(self, classification_data):
        """Test training a classifier."""
        X_train, y_train, X_test, y_test = classification_data

        model = LogisticRegression()
        trained = train_model(X_train, y_train, model, "LogReg")

        assert isinstance(trained, TrainedModel)
        assert trained.model_name == "LogReg"
        assert len(trained.history.train_scores) > 0

    def test_train_regressor(self, regression_data):
        """Test training a regressor."""
        X_train, y_train, X_test, y_test = regression_data

        model = LinearRegression()
        trained = train_model(X_train, y_train, model, "OLS")

        assert isinstance(trained, TrainedModel)
        assert trained.model_name == "OLS"

    def test_train_with_validation(self, classification_data):
        """Test training with validation data."""
        X_train, y_train, X_test, y_test = classification_data

        model = LogisticRegression()
        trained = train_model(
            X_train, y_train, model, "LogReg",
            X_val=X_test, y_val=y_test
        )

        assert len(trained.history.val_scores) > 0


class TestEvaluation:
    """Tests for model evaluation."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        return model, X, y

    @pytest.fixture
    def trained_regressor(self):
        """Create a trained regressor."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = LinearRegression()
        model.fit(X, y)
        return model, X, y

    def test_evaluate_classifier(self, trained_classifier):
        """Test classifier evaluation."""
        model, X, y = trained_classifier
        metrics = evaluate_classifier(model, X, y)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1_score <= 1
        assert metrics.confusion_matrix is not None
        assert len(metrics.class_names) == 3

    def test_evaluate_regressor(self, trained_regressor):
        """Test regressor evaluation."""
        model, X, y = trained_regressor
        metrics = evaluate_regressor(model, X, y)

        assert metrics.rmse >= 0
        assert metrics.mae >= 0
        assert -1 <= metrics.r2 <= 1


class TestModelComparison:
    """Tests for model comparison."""

    def test_get_classifiers(self):
        """Test getting default classifiers."""
        classifiers = get_classifiers()
        assert len(classifiers) > 0
        assert "Random Forest" in classifiers

    def test_get_regressors(self):
        """Test getting default regressors."""
        regressors = get_regressors()
        assert len(regressors) > 0
        assert "Huber (Robust)" in regressors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
