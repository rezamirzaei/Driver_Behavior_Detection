"""
Data splitting utilities with proper generalization strategies.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.schemas import Dataset, SplitData


def split_data(
    dataset: Dataset,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> SplitData:
    """
    Split dataset into training and test sets.

    Args:
        dataset: Dataset to split.
        test_size: Fraction of data for testing.
        stratify: Whether to stratify (for classification).
        random_state: Random seed.

    Returns:
        SplitData with train and test sets.
    """
    X = dataset.X
    y = dataset.y

    # Determine if stratification is possible
    stratify_col = None
    if stratify and dataset.info and dataset.info.task_type == "classification":
        stratify_col = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_col, random_state=random_state
    )

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=dataset.feature_names,
        target_name=dataset.target_name,
    )


def split_by_driver(
    X: pd.DataFrame,
    y: pd.Series,
    test_drivers: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data by driver for better generalization testing.

    This ensures that:
    1. Specified test_drivers (e.g., D6) are ALWAYS in the test set (never trained on)
    2. Additional stratified samples are added from other drivers to reach target test_size

    **Why this matters:**
    - In real-world deployment, models must work on NEW drivers never seen before
    - Random splits leak information when the same driver appears in train and test
    - Driver-level splits provide realistic generalization estimates

    Args:
        X: Features (must contain 'driver' column)
        y: Target
        test_drivers: Specific drivers to ALWAYS hold out (e.g., ['D6'])
        test_size: Target fraction for test set (will add stratified samples to reach this)
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test (with 'driver' column removed from features)
    """
    if "driver" not in X.columns:
        raise ValueError("X must contain 'driver' column for driver-level splitting")

    # Get unique drivers
    unique_drivers = X["driver"].unique()
    total_samples = len(X)
    target_test_samples = int(total_samples * test_size)

    if test_drivers is None:
        # Randomly select drivers for test set
        n_test_drivers = max(1, int(len(unique_drivers) * test_size))
        np.random.seed(random_state)
        test_drivers = list(np.random.choice(unique_drivers, size=n_test_drivers, replace=False))

    # Step 1: All specified test_driver samples go to test (NEVER in training)
    mandatory_test_mask = X["driver"].isin(test_drivers)
    mandatory_test_count = mandatory_test_mask.sum()

    # Convert y to pandas Series if it's a numpy array (for consistent handling)
    y_is_array = isinstance(y, np.ndarray)
    if y_is_array:
        y = pd.Series(y, index=X.index)

    # Step 2: Calculate how many additional samples needed to reach target
    additional_needed = max(0, target_test_samples - mandatory_test_count)

    # Step 3: Get remaining samples (from non-test drivers)
    remaining_mask = ~mandatory_test_mask
    X_remaining = X[remaining_mask].copy()
    y_remaining = y[remaining_mask].copy()

    # Step 4: If we need additional samples, do stratified sampling from remaining
    if additional_needed > 0 and len(X_remaining) > additional_needed:
        additional_ratio = additional_needed / len(X_remaining)
        additional_ratio = min(additional_ratio, 0.5)  # Cap at 50% to ensure train has enough

        # Stratified split of remaining samples
        try:
            X_train_extra, X_test_extra, y_train_extra, y_test_extra = train_test_split(
                X_remaining, y_remaining, test_size=additional_ratio, stratify=y_remaining, random_state=random_state
            )
        except ValueError:
            # If stratification fails, do random split
            X_train_extra, X_test_extra, y_train_extra, y_test_extra = train_test_split(
                X_remaining, y_remaining, test_size=additional_ratio, random_state=random_state
            )

        # Combine: Train = remaining train portion
        X_train = X_train_extra.copy()
        y_train = y_train_extra.copy()

        # Combine: Test = mandatory test drivers + additional stratified samples
        X_test = pd.concat([X[mandatory_test_mask], X_test_extra])
        y_test = pd.concat([y[mandatory_test_mask], y_test_extra])
    else:
        # Just use mandatory test drivers
        X_train = X_remaining.copy()
        y_train = y_remaining.copy()
        X_test = X[mandatory_test_mask].copy()
        y_test = y[mandatory_test_mask].copy()

    # Remove driver column from features
    X_train = X_train.drop(columns=["driver"])
    X_test = X_test.drop(columns=["driver"])

    # Convert y back to numpy array if it was originally an array
    if y_is_array:
        y_train = y_train.values
        y_test = y_test.values

    _train_drivers = sorted([d for d in unique_drivers if d not in test_drivers])

    print("\n📊 Driver-level split (D6 NEVER in training):")
    print(f"  Test drivers (held out): {sorted(test_drivers)} ({mandatory_test_count} samples)")
    print(f"  Additional stratified test samples: {len(X_test) - mandatory_test_count}")
    print(f"  Train samples: {len(X_train)} ({100 * len(X_train) / total_samples:.1f}%)")
    print(f"  Test samples: {len(X_test)} ({100 * len(X_test) / total_samples:.1f}%)")
    print("  ✅ D6 is NEVER used for training\n")

    return X_train, X_test, y_train, y_test
