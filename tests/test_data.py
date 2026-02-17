"""Tests for data loading modules."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.schemas import Dataset, DatasetInfo
from src.data.splitter import split_data
from src.data.uah_loader import UAHDataLoader, load_uah_driveset


class TestUAHDataLoader:
    """Tests for UAH data loader."""

    @pytest.fixture
    def data_dir(self):
        return "data/UAH-DRIVESET-v1"

    def test_loader_init(self, data_dir):
        """Test loader initialization."""
        if Path(data_dir).exists():
            loader = UAHDataLoader(data_dir)
            assert loader.data_dir == Path(data_dir)
        else:
            pytest.skip("UAH data not available")

    def test_load_dataset(self, data_dir):
        """Test loading full dataset."""
        if not Path(data_dir).exists():
            pytest.skip("UAH data not available")

        dataset = load_uah_driveset(data_dir)

        assert isinstance(dataset, Dataset)
        assert dataset.n_samples > 0
        assert dataset.n_features > 0
        assert dataset.info is not None
        assert dataset.info.task_type == "classification"

    def test_load_filtered(self, data_dir):
        """Test loading with filters."""
        if not Path(data_dir).exists():
            pytest.skip("UAH data not available")

        dataset = load_uah_driveset(data_dir, drivers=["D1"], behaviors=["NORMAL"])

        assert dataset.n_samples > 0
        # All behaviors should be NORMAL
        unique_behaviors = dataset.y.unique()
        assert len(unique_behaviors) == 1
        assert unique_behaviors[0] == "NORMAL"

    def test_load_regression(self, data_dir):
        """Test loading for regression task."""
        if not Path(data_dir).exists():
            pytest.skip("UAH data not available")

        dataset = load_uah_driveset(data_dir, task="regression", target_variable="score_total")

        assert isinstance(dataset, Dataset)
        assert dataset.info.task_type == "regression"
        assert dataset.target_name == "score_total"
        assert "score_total" not in dataset.feature_names


class TestSplitter:
    """Tests for data splitting."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        import numpy as np
        import pandas as pd

        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(["A"] * 50 + ["B"] * 50)

        return Dataset(
            X=X,
            y=y,
            feature_names=list(X.columns),
            target_name="target",
            info=DatasetInfo(name="test", n_samples=100, n_features=5, task_type="classification"),
        )

    def test_split_data(self, sample_dataset):
        """Test data splitting."""
        split = split_data(sample_dataset, test_size=0.2)

        assert split.n_train == 80
        assert split.n_test == 20
        assert len(split.feature_names) == 5

    def test_stratified_split(self, sample_dataset):
        """Test stratified splitting."""
        split = split_data(sample_dataset, test_size=0.2, stratify=True)

        # Check class proportions are preserved
        train_prop = (split.y_train == "A").sum() / len(split.y_train)
        test_prop = (split.y_test == "A").sum() / len(split.y_test)

        assert abs(train_prop - test_prop) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
