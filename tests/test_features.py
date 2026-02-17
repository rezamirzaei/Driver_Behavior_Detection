"""Tests for features module."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.schemas import FeatureSet, SplitData
from src.features.preprocessing import (
    FeaturePreprocessor,
    encode_target,
    engineer_regression_features,
    preprocess_features,
)


class TestFeaturePreprocessor:
    """Tests for feature preprocessor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with numeric and categorical features."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "num1": np.random.randn(100),
                "num2": np.random.randn(100),
                "cat1": np.random.choice(["A", "B", "C"], 100),
            }
        )
        return df

    def test_fit_transform(self, sample_data):
        """Test fit_transform."""
        preprocessor = FeaturePreprocessor(scaler_type="robust")
        X_transformed = preprocessor.fit_transform(sample_data)

        assert X_transformed.shape[0] == 100
        assert preprocessor.numeric_cols == ["num1", "num2"]
        assert preprocessor.categorical_cols == ["cat1"]

    def test_transform(self, sample_data):
        """Test transform on new data."""
        preprocessor = FeaturePreprocessor(scaler_type="robust")
        preprocessor.fit_transform(sample_data)

        new_data = sample_data.copy()
        X_transformed = preprocessor.transform(new_data)

        assert X_transformed.shape[0] == 100


class TestPreprocessFeatures:
    """Tests for preprocess_features function."""

    @pytest.fixture
    def split_data(self):
        """Create split data for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "num1": np.random.randn(80),
                "num2": np.random.randn(80),
            }
        )
        X_test = pd.DataFrame(
            {
                "num1": np.random.randn(20),
                "num2": np.random.randn(20),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, 80))
        y_test = pd.Series(np.random.randint(0, 2, 20))

        return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, feature_names=["num1", "num2"])

    def test_preprocess_features(self, split_data):
        """Test preprocessing pipeline."""
        train_feat, test_feat = preprocess_features(split_data)

        assert isinstance(train_feat, FeatureSet)
        assert isinstance(test_feat, FeatureSet)
        assert train_feat.X.shape[0] == 80
        assert test_feat.X.shape[0] == 20


class TestTargetEncoder:
    """Tests for target encoder."""

    def test_encode_target(self):
        """Test target encoding."""
        y_train = pd.Series(["A", "B", "C", "A", "B"])
        y_test = pd.Series(["A", "C"])

        y_train_enc, y_test_enc, encoder = encode_target(y_train, y_test)

        assert len(y_train_enc) == 5
        assert len(y_test_enc) == 2
        assert len(encoder.classes_) == 3

    def test_inverse_transform(self):
        """Test inverse transform."""
        y_train = pd.Series(["A", "B", "C"])
        y_test = pd.Series(["A"])

        _, _, encoder = encode_target(y_train, y_test)

        original = encoder.inverse_transform([0, 1, 2])
        assert list(original) == ["A", "B", "C"]


class TestEngineerRegressionFeatures:
    """Tests for engineer_regression_features."""

    @pytest.fixture
    def epa_like_data(self):
        """Create a small DataFrame resembling the EPA dataset."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "displ": [2.0, 3.5, 0.0, 1.8, 4.0],
                "cylinders": [4, 6, 0, 4, 8],
                "year": [2018, 2019, 2020, 2021, 2022],
                "rangeCity": [250, 0, 300, 200, 100],
                "rangeHwy": [350, 0, 400, 300, 200],
                "sCharger": ["S", "", None, "", "S"],
                "tCharger": ["", "T", None, "", ""],
                "evMotor": [None, None, "AC motor", None, None],
                "startStop": ["Y", "N", "Y", "N", "Y"],
                "guzzler": [None, None, None, "G", None],
                "comb08": [30, 25, 110, 35, 20],
            }
        )

    def test_does_not_mutate_input(self, epa_like_data):
        """engineer_regression_features must not modify the input df."""
        original_cols = list(epa_like_data.columns)
        engineer_regression_features(epa_like_data)
        assert list(epa_like_data.columns) == original_cols

    def test_adds_expected_columns(self, epa_like_data):
        """Check that all expected engineered columns are present."""
        result = engineer_regression_features(epa_like_data)
        expected_new = [
            "displ_per_cyl",
            "displ_x_cyl",
            "displ_squared",
            "range_total",
            "range_city_ratio",
            "has_supercharger",
            "has_turbocharger",
            "is_electric",
            "has_start_stop",
            "is_guzzler",
            "is_forced_induction",
            "vehicle_age",
        ]
        for col in expected_new:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_displ_per_cyl_avoids_division_by_zero(self, epa_like_data):
        """When cylinders == 0, displ_per_cyl should be NaN, not raise."""
        result = engineer_regression_features(epa_like_data)
        # Row index 2 has cylinders=0
        assert np.isnan(result.loc[2, "displ_per_cyl"])

    def test_range_city_ratio_avoids_division_by_zero(self, epa_like_data):
        """When range_total == 0, range_city_ratio should be NaN."""
        result = engineer_regression_features(epa_like_data)
        # Row index 1 has rangeCity=0, rangeHwy=0
        assert np.isnan(result.loc[1, "range_city_ratio"])

    def test_binary_flags_are_int(self, epa_like_data):
        """Binary indicator columns should contain only 0 and 1."""
        result = engineer_regression_features(epa_like_data)
        for col in [
            "has_supercharger",
            "has_turbocharger",
            "is_electric",
            "has_start_stop",
            "is_guzzler",
            "is_forced_induction",
        ]:
            assert set(result[col].unique()).issubset({0, 1}), f"{col} not binary"

    def test_vehicle_age_correctness(self, epa_like_data):
        """vehicle_age should equal max(year) - year."""
        result = engineer_regression_features(epa_like_data)
        max_year = epa_like_data["year"].max()
        expected = max_year - epa_like_data["year"]
        pd.testing.assert_series_equal(result["vehicle_age"], expected, check_names=False)

    def test_graceful_with_missing_columns(self):
        """Should still work when optional columns are absent."""
        df = pd.DataFrame({"displ": [2.0, 3.0], "cylinders": [4, 6]})
        result = engineer_regression_features(df)
        assert "displ_per_cyl" in result.columns
        # Columns based on missing source should simply not appear
        assert "range_total" not in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
