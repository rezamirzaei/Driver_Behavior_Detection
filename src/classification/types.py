"""
Classification Data Structures.

Dataclasses used across the classification module.
"""

from dataclasses import dataclass
from typing import Any, List

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class ClassificationResult:
    """Container for classification results."""
    model_name: str
    train_accuracy: float
    test_accuracy: float
    f1_score: float
    predictions: np.ndarray
    model: Any


@dataclass
class DataSplit:
    """Container for train/test data split."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    scaler: StandardScaler
    label_encoder: LabelEncoder

