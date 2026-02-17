"""
Feature analysis module for exploratory data analysis.

Provides reusable functions for EDA that return structured Pydantic outputs.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.core.schemas import (
    ClassDistributionResult,
    CorrelationAnalysisResult,
    DataQualityReport,
    FeatureStatisticsResult,
    OutlierAnalysisResult,
    ScoreMappingInfo,
)


def get_feature_columns(
    df: pd.DataFrame,
    target_col: Optional[str] = None
) -> tuple:
    """
    Get numerical and categorical column names from DataFrame.

    Args:
        df: Input DataFrame.
        target_col: Target column to exclude from features.

    Returns:
        Tuple of (numerical_cols, categorical_cols).
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_col and target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    return numerical_cols, categorical_cols


def get_correlations_with_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None
) -> pd.Series:
    """
    Get correlations of features with target, sorted by absolute value.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        feature_cols: Feature columns (None = all numerical).

    Returns:
        Series of correlations sorted by absolute value descending.
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)

    correlations = df[feature_cols].corrwith(df[target_col])
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def analyze_outliers(
    data: pd.Series,
    column_name: str,
    method: str = "mad",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    mad_threshold: float = 3.5,
) -> OutlierAnalysisResult:
    """
    Analyze outliers in a single column using robust statistical methods.

    Args:
        data: Series of values to analyze.
        column_name: Name of the column.
        method: Detection method - 'iqr', 'zscore', or 'mad' (recommended).
        iqr_multiplier: Multiplier for IQR bounds (default 1.5).
        z_threshold: Threshold for z-score method (default 3.0).
        mad_threshold: Threshold for MAD method (default 3.5).

    Returns:
        OutlierAnalysisResult with bounds, counts, and indices.

    Note:
        - 'iqr': Traditional but sensitive to skewed distributions
        - 'zscore': Assumes normality, affected by outliers themselves
        - 'mad': Modified Z-score using Median Absolute Deviation (most robust)
    """
    data_clean = data.dropna()

    if len(data_clean) == 0:
        return OutlierAnalysisResult(
            column=column_name,
            lower_bound=0.0,
            upper_bound=0.0,
            n_outliers=0,
            outlier_percentage=0.0,
            outlier_indices=[],
        )

    if method == "iqr":
        Q1 = data_clean.quantile(0.25)
        Q3 = data_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        outlier_mask = (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        mean = data_clean.mean()
        std = data_clean.std()
        if std == 0:
            lower_bound = mean
            upper_bound = mean
            outlier_mask = pd.Series([False] * len(data), index=data.index)
        else:
            z_scores = (data - mean) / std
            outlier_mask = abs(z_scores) > z_threshold
            lower_bound = mean - z_threshold * std
            upper_bound = mean + z_threshold * std

    elif method == "mad":
        # Modified Z-score using Median Absolute Deviation (most robust)
        median = data_clean.median()
        mad = np.median(np.abs(data_clean - median))

        # MAD = 0 means all values are identical
        if mad == 0:
            lower_bound = median
            upper_bound = median
            outlier_mask = pd.Series([False] * len(data), index=data.index)
        else:
            # Modified z-score: 0.6745 is the scaling constant for normal distribution
            modified_z = 0.6745 * (data - median) / mad
            outlier_mask = abs(modified_z) > mad_threshold
            # Convert threshold back to original scale for bounds
            lower_bound = median - mad_threshold * mad / 0.6745
            upper_bound = median + mad_threshold * mad / 0.6745
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr', 'zscore', or 'mad'.")

    outlier_indices = data.index[outlier_mask].tolist()
    n_outliers = len(outlier_indices)
    outlier_percentage = (n_outliers / len(data)) * 100

    return OutlierAnalysisResult(
        column=column_name,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        n_outliers=n_outliers,
        outlier_percentage=round(outlier_percentage, 2),
        outlier_indices=outlier_indices[:100],  # Limit to first 100
    )


def analyze_outliers_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "mad",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    mad_threshold: float = 3.5,
) -> List[OutlierAnalysisResult]:
    """
    Analyze outliers for multiple columns in a DataFrame.

    Args:
        df: DataFrame to analyze.
        columns: Columns to analyze. None = all numeric columns.
        method: Detection method - 'iqr', 'zscore', or 'mad' (recommended).
        iqr_multiplier: Multiplier for IQR bounds.
        z_threshold: Threshold for z-score method.
        mad_threshold: Threshold for MAD method.

    Returns:
        List of OutlierAnalysisResult for each column.

    Example:
        >>> results = analyze_outliers_dataframe(df, method='mad')
        >>> for r in results:
        ...     if r.has_significant_outliers:
        ...         print(f"{r.column}: {r.outlier_percentage:.1f}% outliers")
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in columns:
        if col in df.columns:
            result = analyze_outliers(
                df[col], col,
                method=method,
                iqr_multiplier=iqr_multiplier,
                z_threshold=z_threshold,
                mad_threshold=mad_threshold,
            )
            results.append(result)

    return results


def print_outlier_summary(results: List[OutlierAnalysisResult], method: str = "MAD") -> None:
    """
    Print a formatted summary of outlier analysis results.

    Args:
        results: List of OutlierAnalysisResult from analyze_outliers_dataframe.
        method: Name of method used for display.
    """
    print(f"\n🔍 Outlier Analysis ({method} method):")

    _total_outliers = sum(r.n_outliers for r in results)
    cols_with_outliers = sum(1 for r in results if r.n_outliers > 0)

    for result in results:
        if result.has_significant_outliers:
            status = "⚠️"
        elif result.n_outliers > 0:
            status = "📌"
        else:
            status = "✅"
        print(f"   {status} {result.column}: {result.n_outliers} outliers ({result.outlier_percentage:.1f}%)")

    print(f"\n   Summary: {cols_with_outliers}/{len(results)} columns have outliers")

    significant = [r for r in results if r.has_significant_outliers]
    if significant:
        print(f"   ⚠️ {len(significant)} columns have significant outliers (>5%)")
        print("   → Consider robust methods (Huber, RANSAC) or winsorization")


def analyze_correlations(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    threshold: float = 0.8,
) -> CorrelationAnalysisResult:
    """
    Analyze feature correlations and detect multicollinearity.

    Args:
        df: DataFrame with features.
        columns: Columns to analyze. None = all numeric.
        target_column: Optional target column for target correlations.
        threshold: Threshold for high correlation warning.

    Returns:
        CorrelationAnalysisResult with matrix and high-correlation pairs.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in columns:
            columns.remove(target_column)

    corr_matrix = df[columns].corr()

    # Find high correlation pairs
    high_corr_pairs = find_high_correlation_pairs(corr_matrix, columns, threshold)

    # Target correlations (only for numerical targets)
    target_correlations = None
    if target_column and target_column in df.columns:
        # Only compute correlation if target is numerical
        if pd.api.types.is_numeric_dtype(df[target_column]):
            target_corrs = df[columns].corrwith(df[target_column])
            target_correlations = {col: round(val, 4) for col, val in target_corrs.items()}

    multicollinearity_warning = any(abs(c) > 0.9 for _, _, c in high_corr_pairs)

    return CorrelationAnalysisResult(
        correlation_matrix=corr_matrix,
        high_correlation_pairs=high_corr_pairs,
        target_correlations=target_correlations,
        multicollinearity_warning=multicollinearity_warning,
    )


def find_high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.8,
) -> List[tuple]:
    """
    Find pairs of features with correlation above threshold.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        columns: Column names (uses matrix columns if None).
        threshold: Absolute correlation threshold.

    Returns:
        List of tuples: [(feature1, feature2, correlation), ...]
    """
    if columns is None:
        columns = corr_matrix.columns.tolist()

    high_corr_pairs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append((columns[i], columns[j], round(corr_val, 4)))

    return high_corr_pairs


def analyze_class_distribution(
    y: pd.Series,
    imbalance_threshold: float = 3.0,
) -> ClassDistributionResult:
    """
    Analyze class distribution for classification problems.

    Args:
        y: Target series with class labels.
        imbalance_threshold: Ratio threshold for imbalance warning.

    Returns:
        ClassDistributionResult with counts, percentages, and recommendations.
    """
    counts = y.value_counts().to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}

    percentages = y.value_counts(normalize=True).to_dict()
    percentages = {str(k): round(v * 100, 2) for k, v in percentages.items()}

    max_count = max(counts.values())
    min_count = min(counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    is_imbalanced = imbalance_ratio > imbalance_threshold

    if is_imbalanced:
        recommendation = (
            f"⚠️ Class imbalance detected (ratio: {imbalance_ratio:.1f}). "
            "Recommendations:\n"
            "  - Use class_weight='balanced' in models\n"
            "  - Evaluate with F1-score instead of accuracy\n"
            "  - Consider SMOTE or undersampling"
        )
    else:
        recommendation = f"✅ Classes reasonably balanced (ratio: {imbalance_ratio:.1f})"

    return ClassDistributionResult(
        counts=counts,
        percentages=percentages,
        imbalance_ratio=round(imbalance_ratio, 2),
        is_imbalanced=is_imbalanced,
        recommendation=recommendation,
    )


def compute_feature_statistics(
    data: pd.Series,
    feature_name: str,
) -> FeatureStatisticsResult:
    """
    Compute comprehensive statistics for a single feature.

    Args:
        data: Series of feature values.
        feature_name: Name of the feature.

    Returns:
        FeatureStatisticsResult with all statistics.
    """
    data_clean = data.dropna()

    return FeatureStatisticsResult(
        name=feature_name,
        mean=float(data_clean.mean()) if len(data_clean) > 0 else 0.0,
        median=float(data_clean.median()) if len(data_clean) > 0 else 0.0,
        std=float(data_clean.std()) if len(data_clean) > 0 else 0.0,
        min=float(data_clean.min()) if len(data_clean) > 0 else 0.0,
        max=float(data_clean.max()) if len(data_clean) > 0 else 0.0,
        skewness=float(stats.skew(data_clean.values)) if len(data_clean) > 2 else 0.0,
        n_missing=int(data.isna().sum()),
        n_unique=int(data.nunique()),
    )


def generate_data_quality_report(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: str = "classification",
) -> DataQualityReport:
    """
    Generate comprehensive data quality report.

    Args:
        df: DataFrame to analyze.
        target_column: Optional target column name.
        task_type: 'classification' or 'regression'.

    Returns:
        DataQualityReport with all analyses.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_column:
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

    # Missing values
    missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}

    # Feature statistics
    feature_stats = [
        compute_feature_statistics(df[col], col)
        for col in numerical_cols
    ]

    # Outlier analysis
    outlier_analysis = analyze_outliers_dataframe(df, columns=numerical_cols)

    # Class distribution (classification only)
    class_distribution = None
    if task_type == "classification" and target_column and target_column in df.columns:
        class_distribution = analyze_class_distribution(df[target_column])

    # Correlation analysis
    correlation_analysis = None
    if len(numerical_cols) >= 2:
        correlation_analysis = analyze_correlations(
            df, columns=numerical_cols, target_column=target_column
        )

    return DataQualityReport(
        n_samples=len(df),
        n_features=len(numerical_cols) + len(categorical_cols),
        n_numerical=len(numerical_cols),
        n_categorical=len(categorical_cols),
        missing_values=missing,
        feature_statistics=feature_stats,
        outlier_analysis=outlier_analysis,
        class_distribution=class_distribution,
        correlation_analysis=correlation_analysis,
    )


def get_score_mapping_info() -> ScoreMappingInfo:
    """
    Get documentation about how UAH-DriveSet maps accelerometer data to scores.

    Returns:
        ScoreMappingInfo with full documentation.
    """
    return ScoreMappingInfo()


def print_score_mapping_explanation() -> str:
    """
    Print human-readable explanation of UAH score mapping.

    Returns:
        Formatted string with explanation.
    """
    info = get_score_mapping_info()
    return info.description


def identify_discriminative_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    top_n: int = 10,
) -> Dict[str, float]:
    """
    Identify most discriminative features using ANOVA F-statistic.

    Args:
        df: DataFrame with features and target.
        feature_columns: List of feature column names.
        target_column: Target column name.
        top_n: Number of top features to return.

    Returns:
        Dict of feature -> F-statistic (sorted descending).
    """
    from sklearn.feature_selection import f_classif

    X = df[feature_columns].values
    y = df[target_column].values

    # Handle NaN
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    if len(np.unique(y)) < 2:
        return {}

    f_scores, _ = f_classif(X, y)

    feature_scores = dict(zip(feature_columns, f_scores))
    sorted_features = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n])

    return {k: round(v, 4) for k, v in sorted_features.items()}


def analyze_driver_distribution(
    df: pd.DataFrame,
    driver_column: str = "driver",
    behavior_column: str = "behavior",
) -> pd.DataFrame:
    """
    Analyze trip distribution per driver and behavior.

    Args:
        df: DataFrame with driver and behavior columns.
        driver_column: Name of driver column.
        behavior_column: Name of behavior column.

    Returns:
        Cross-tabulation DataFrame.
    """
    if driver_column not in df.columns or behavior_column not in df.columns:
        return pd.DataFrame()

    return pd.crosstab(df[driver_column], df[behavior_column])

