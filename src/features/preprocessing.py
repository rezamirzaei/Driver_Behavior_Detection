"""
Feature preprocessing module.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler

from src.core.schemas import FeatureSet, SplitData


def engineer_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for regression on the EPA fuel economy dataset.

    Adds interaction terms, ratios, and categorical flags that capture
    domain-specific relationships between vehicle characteristics and fuel
    economy.  Works on a copy of the input DataFrame so the original is
    not mutated.

    Args:
        df: DataFrame with raw EPA features (must contain at least
            ``displ`` and ``cylinders`` columns; other columns are
            optional).

    Returns:
        DataFrame with original columns plus new engineered columns.
    """
    df = df.copy()

    # --- Interaction / ratio features ---
    if "displ" in df.columns and "cylinders" in df.columns:
        cyl = df["cylinders"].replace(0, np.nan)
        df["displ_per_cyl"] = df["displ"] / cyl
        df["displ_x_cyl"] = df["displ"] * df["cylinders"]

    if "displ" in df.columns:
        df["displ_squared"] = df["displ"] ** 2

    # --- Range features ---
    if "rangeCity" in df.columns and "rangeHwy" in df.columns:
        total = df["rangeCity"] + df["rangeHwy"]
        total_safe = total.replace(0, np.nan)
        df["range_total"] = total
        df["range_city_ratio"] = df["rangeCity"] / total_safe

    # --- Binary indicator flags ---
    for col, flag in [
        ("sCharger", "has_supercharger"),
        ("tCharger", "has_turbocharger"),
        ("evMotor", "is_electric"),
        ("startStop", "has_start_stop"),
        ("guzzler", "is_guzzler"),
    ]:
        if col in df.columns:
            df[flag] = (df[col].notna() & (df[col] != "") & (df[col] != "N")).astype(int)

    if "has_supercharger" in df.columns and "has_turbocharger" in df.columns:
        df["is_forced_induction"] = ((df["has_supercharger"] == 1) | (df["has_turbocharger"] == 1)).astype(int)

    # --- Year-based feature ---
    if "year" in df.columns:
        df["vehicle_age"] = df["year"].max() - df["year"]

    return df


def encode_and_scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply target encoding to categoricals and standardize all features.

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training target (for target encoding).
        categorical_cols: Categorical columns. None = auto-detect.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled).
    """
    from category_encoders import TargetEncoder
    from sklearn.impute import SimpleImputer

    if categorical_cols is None:
        categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Target encoding for categoricals
    if categorical_cols:
        encoder = TargetEncoder(cols=categorical_cols, smoothing=1.0)
        X_train_encoded = encoder.fit_transform(X_train, y_train)
        X_test_encoded = encoder.transform(X_test)
    else:
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

    # Convert to numpy
    X_train_arr = X_train_encoded.values if hasattr(X_train_encoded, "values") else X_train_encoded
    X_test_arr = X_test_encoded.values if hasattr(X_test_encoded, "values") else X_test_encoded

    # Impute NaN values with median (robust to outliers)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_arr)
    X_test_imputed = imputer.transform(X_test_arr)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled


class FeaturePreprocessor:
    """Preprocessor for features with scaling and encoding."""

    def __init__(self, scaler_type: str = "robust"):
        """
        Initialize preprocessor.

        Args:
            scaler_type: Type of scaler ('standard', 'robust', 'none').
        """
        self.scaler_type = scaler_type
        self.preprocessor: Optional[ColumnTransformer] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Fit on training data and transform.

        Args:
            X_train: Training features.
            numeric_cols: Numeric column names. None = auto-detect.
            categorical_cols: Categorical column names. None = auto-detect.

        Returns:
            Transformed feature array.
        """
        # Auto-detect column types
        if numeric_cols is None:
            self.numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_cols = numeric_cols

        if categorical_cols is None:
            self.categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.categorical_cols = categorical_cols

        # Build transformers
        transformers = []

        if self.numeric_cols:
            if self.scaler_type == "standard":
                scaler = StandardScaler()
            elif self.scaler_type == "robust":
                scaler = RobustScaler()
            else:
                scaler = "passthrough"
            transformers.append(("num", scaler, self.numeric_cols))

        if self.categorical_cols:
            encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            transformers.append(("cat", encoder, self.categorical_cols))

        if not transformers:
            return X_train.values

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return self.preprocessor.fit_transform(X_train)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return self.preprocessor.transform(X)

    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        if self.preprocessor is None:
            return []

        names = []
        for name, trans, cols in self.preprocessor.transformers_:
            if name == "cat" and hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols).tolist())
            elif name != "remainder":
                names.extend(cols)
        return names


def preprocess_features(
    split_data: SplitData,
    scaler_type: str = "robust",
) -> Tuple[FeatureSet, FeatureSet]:
    """
    Preprocess features for train and test sets.

    Args:
        split_data: SplitData with train/test splits.
        scaler_type: Type of scaler to use.

    Returns:
        Tuple of (train_features, test_features).
    """
    X_train = split_data.X_train
    X_test = split_data.X_test

    # Convert to DataFrame if needed
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=split_data.feature_names)
        X_test = pd.DataFrame(X_test, columns=split_data.feature_names)
    else:
        X_train = X_train.copy()
        X_test = X_test.copy()

    # Handle missing values in numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    preprocessor = FeaturePreprocessor(scaler_type=scaler_type)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Final NaN check and fill
    if np.any(np.isnan(X_train_processed)):
        X_train_processed = np.nan_to_num(X_train_processed, nan=0.0)
    if np.any(np.isnan(X_test_processed)):
        X_test_processed = np.nan_to_num(X_test_processed, nan=0.0)

    feature_names = preprocessor.get_feature_names()

    train_features = FeatureSet(X=X_train_processed, feature_names=feature_names, scaler=preprocessor.preprocessor)

    test_features = FeatureSet(X=X_test_processed, feature_names=feature_names)

    return train_features, test_features


class TargetEncoder:
    """Encoder for categorical target variables."""

    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes_ = None

    def fit_transform(self, y) -> np.ndarray:
        """Fit and transform target."""
        y_encoded = self.encoder.fit_transform(y)
        self.classes_ = self.encoder.classes_
        return y_encoded

    def transform(self, y) -> np.ndarray:
        """Transform target."""
        return self.encoder.transform(y)

    def inverse_transform(self, y) -> np.ndarray:
        """Inverse transform encoded target."""
        return self.encoder.inverse_transform(y)


def encode_target(y_train, y_test) -> Tuple[np.ndarray, np.ndarray, TargetEncoder]:
    """
    Encode categorical target variable.

    Args:
        y_train: Training target.
        y_test: Test target.

    Returns:
        Tuple of (y_train_encoded, y_test_encoded, encoder).
    """
    encoder = TargetEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded, encoder
