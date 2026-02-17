"""
Reporting utilities for clean, consistent output formatting.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

if TYPE_CHECKING:
    from src.core.schemas import ClassDistributionResult, DatasetInfo


def print_dataset_info(info: "DatasetInfo") -> None:
    """Print dataset information summary."""
    print(f"\n📊 Dataset: {info.name}")
    print(f"   Samples: {info.n_samples}")
    print(f"   Features: {info.n_features}")
    print(f"   Task: {info.task_type}")


def print_class_distribution_result(result: "ClassDistributionResult") -> None:
    """Print class distribution analysis results."""
    print("\n📊 Class Distribution Analysis:")
    print(f"   Counts: {result.counts}")
    print(f"   Percentages: {dict((k, f'{v:.1f}%') for k, v in result.percentages.items())}")
    if result.recommendation:
        print(f"\n{result.recommendation}")


def print_header(title: str, emoji: str = "📊", width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"{emoji} {title.upper()}")
    print("=" * width)


def print_subheader(title: str, char: str = "-", width: int = 40) -> None:
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_metrics(accuracy: float, f1_score: float, additional: Optional[Dict[str, float]] = None) -> None:
    """Print model metrics in a clean format."""
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")
    if additional:
        for name, value in additional.items():
            print(f"  {name}: {value:.4f}")


def print_classification_results(
    model_name: str,
    accuracy: float,
    f1: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    emoji: str = "📊",
) -> None:
    """Print complete classification results for a model."""
    print_header(model_name, emoji)
    print_metrics(accuracy, f1)
    print("\n" + classification_report(y_true, y_pred, target_names=class_names))


def print_data_summary(shape: Tuple[int, int], target_col: str, class_counts: pd.Series) -> None:
    """Print a summary of loaded data."""
    print(f"📊 Loaded: {shape}")
    print(f"\n🎯 Classes:\n{class_counts}")


def print_split_summary(
    X_train_shape: Tuple[int, int], X_test_shape: Tuple[int, int], train_info: str = "", test_info: str = ""
) -> None:
    """Print train/test split summary."""
    print("\n✅ Split complete")
    print(f"   Train: {X_train_shape} {train_info}")
    print(f"   Test: {X_test_shape} {test_info}")


def print_training_analysis(final_train_loss: float, final_val_loss: float, overfitting_threshold: float = 0.5) -> None:
    """Print training convergence analysis."""
    overfitting_gap = final_val_loss - final_train_loss

    print("\n📈 Training Analysis:")
    print(f"   Final train loss: {final_train_loss:.4f}")
    print(f"   Final val loss: {final_val_loss:.4f}")
    print(f"   Overfitting gap: {overfitting_gap:.4f}")

    if overfitting_gap > overfitting_threshold:
        print("   ⚠️ Significant overfitting")
        print("   → Consider: Dropout, L2 regularization")
    else:
        print("   ✅ Good generalization")


def print_model_comparison(comparison_df: pd.DataFrame, metric_col: str = "Accuracy", model_col: str = "Model") -> str:
    """Print model comparison table and return best model name."""
    print_header("MODEL COMPARISON", "🏆")
    print(comparison_df.to_string(index=False))

    best_model = comparison_df.loc[comparison_df[metric_col].idxmax(), model_col]
    print(f"\n✨ Best: {best_model}")

    return best_model


def print_confused_classes(confusion_matrix: np.ndarray, class_names: List[str], threshold: float = 0.2) -> None:
    """Print significantly confused class pairs from confusion matrix."""
    cm_norm = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]

    print("\n🔍 Most confused class pairs:")
    found_any = False

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and confusion_matrix[i, j] > 0:
                confusion_rate = cm_norm[i, j]
                if confusion_rate > threshold:
                    print(f"   {class_names[i]} → {class_names[j]}: {confusion_rate * 100:.1f}%")
                    found_any = True

    if not found_any:
        print("   No significant confusions (all < 20%)")


def print_feature_importance(feature_names: List[str], importances: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """Print top feature importances and return DataFrame."""
    df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    print(f"\n🎯 Top {top_n} Features:")
    print(df.head(top_n).to_string(index=False))

    return df


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️ {message}")


def print_regression_results(
    model_name: str, r2: float, rmse: float, mae: float, emoji: str = "📊", baseline_r2: Optional[float] = None
) -> None:
    """Print complete regression results for a model."""
    print_header(model_name, emoji)
    print(f"  R² Score: {r2:.4f} ← Higher = Better (Max = 1.0)")
    print(f"  RMSE: {rmse:.4f} ← Lower = Better")
    print(f"  MAE: {mae:.4f} ← Lower = Better")

    if baseline_r2 is not None:
        improvement = (r2 - baseline_r2) * 100
        print(f"\n  📈 Improvement over baseline: {improvement:.2f}% (R²)")

    if r2 < 0.5:
        print("\n  ⚠️ Low R² suggests poor fit or outlier effects")
    elif r2 >= 0.8:
        print("\n  ✅ Excellent fit!")
    else:
        print("\n  ✅ Good fit")


def print_target_statistics(y: np.ndarray, target_name: str = "Target") -> None:
    """Print target variable statistics."""
    print(f"\n📊 {target_name} Statistics:")
    print(f"   Mean: {np.mean(y):.2f}")
    print(f"   Median: {np.median(y):.2f}")
    print(f"   Std: {np.std(y):.2f}")
    print(f"   Min: {np.min(y):.2f}")
    print(f"   Max: {np.max(y):.2f}")


def print_skewness_check(skewness: float, threshold: float = 1.0) -> None:
    """Print skewness check result."""
    print(f"   Skewness: {skewness:.2f}")
    if abs(skewness) > threshold:
        print("\n⚠️ Target is skewed → Consider robust regression")
    else:
        print("\n✅ Target approximately normal")


def print_outlier_analysis(lower_bound: float, upper_bound: float, n_outliers: int, total: int) -> None:
    """Print outlier analysis results."""
    pct = n_outliers / total * 100

    print("\n🔍 Outlier Analysis (IQR method):")
    print(f"   Lower bound: {lower_bound:.2f}")
    print(f"   Upper bound: {upper_bound:.2f}")
    print(f"   Outliers: {n_outliers} ({pct:.1f}%)")

    if pct > 5:
        print(f"\n⚠️ Significant outliers ({pct:.1f}%)")
        print("   → PERFECT case for ROBUST REGRESSION (Huber)")
    else:
        print("\n✅ Few outliers")


def print_residual_statistics(residuals: np.ndarray) -> None:
    """Print residual analysis statistics."""
    print("\n📊 Residual Statistics:")
    print(f"   Mean: {np.mean(residuals):.4f} (should be ~0)")
    print(f"   Std: {np.std(residuals):.4f}")
    print(f"   Min: {np.min(residuals):.4f}")
    print(f"   Max: {np.max(residuals):.4f}")


def print_regressor_comparison(
    comparison_df: pd.DataFrame, metric_col: str = "R² Score", model_col: str = "Model"
) -> str:
    """Print regression model comparison table and return best model name."""
    print_header("MODEL COMPARISON", "🏆")
    print(comparison_df.to_string(index=False))

    best_model = comparison_df.loc[comparison_df[metric_col].idxmax(), model_col]
    print(f"\n✨ Best: {best_model}")

    return best_model


def print_feature_types(numerical_cols: List[str], categorical_cols: List[str]) -> None:
    """Print summary of feature types."""
    print("\n📊 Feature Types:")
    print(f"   Numerical ({len(numerical_cols)}): {numerical_cols}")
    print(f"   Categorical ({len(categorical_cols)}): {categorical_cols}")


def print_categorical_cardinality(
    df: pd.DataFrame, categorical_cols: List[str], high_cardinality_threshold: int = 50
) -> List[str]:
    """
    Print cardinality of categorical features.

    Returns list of low-cardinality columns suitable for visualization.
    """
    print("\n🏷️ Categorical Cardinality:")
    low_cardinality = []

    for col in categorical_cols:
        n_unique = df[col].nunique()
        print(f"   {col}: {n_unique} unique")
        if n_unique > high_cardinality_threshold:
            print("      → HIGH cardinality! Use target encoding")
        elif n_unique < 20:
            low_cardinality.append(col)

    return low_cardinality


def print_correlation_with_target(correlations: pd.Series, target_name: str = "target") -> None:
    """Print correlations with target variable."""
    print(f"\n📊 Correlations with {target_name}:")
    for feat, corr in correlations.items():
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        print(f"   {feat}: {corr:.3f} ({strength})")


def print_high_correlation_pairs(pairs: List[Tuple[str, str, float]], threshold: float = 0.8) -> None:
    """Print highly correlated feature pairs (multicollinearity warning)."""
    if pairs:
        print(f"\n⚠️ High multicollinearity (|r| > {threshold}):")
        for f1, f2, corr in pairs:
            print(f"   {f1} ↔ {f2}: {corr:.3f}")
        print("   → Consider Ridge/Lasso regression")
    else:
        print("\n✅ No severe multicollinearity")


def print_save_confirmation(
    path: str, shape: Tuple[int, int], target_name: str = "target", n_numerical: int = 0, n_categorical: int = 0
) -> None:
    """Print data save confirmation."""
    print(f"\n💾 Saved to: {path}")
    print(f"   Shape: {shape}")
    print(f"   Target: {target_name}")
    if n_numerical or n_categorical:
        print(f"   Features: {n_numerical} numerical, {n_categorical} categorical")


def print_prediction_interval_results(
    coverage: float,
    mean_width: float,
    expected_coverage: float = 0.8,
) -> None:
    """Print prediction interval analysis results."""
    print_header("PREDICTION INTERVALS (Quantile Regression)", "📊")
    print(f"  Expected coverage: {expected_coverage:.0%}")
    print(f"  Actual coverage: {coverage:.1%}")
    print(f"  Mean interval width: {mean_width:.2f}")

    if coverage >= expected_coverage - 0.05:
        print("\n  ✅ Good calibration - intervals are reliable")
    else:
        print("\n  ⚠️ Under-coverage - intervals may be too narrow")


def print_sparse_model_results(
    model_name: str,
    n_features: int,
    n_nonzero: int,
    metric_value: float,
    metric_name: str = "R²",
) -> None:
    """Print sparse model feature selection results."""
    sparsity = (n_features - n_nonzero) / n_features * 100
    print(f"\n🎯 {model_name} - Sparse Feature Selection:")
    print(f"   Total features: {n_features}")
    print(f"   Non-zero coefficients: {n_nonzero}")
    print(f"   Sparsity: {sparsity:.1f}%")
    print(f"   {metric_name}: {metric_value:.4f}")


def print_feature_overview(
    df: pd.DataFrame,
    target_col: str = "comb08",
) -> pd.DataFrame:
    """
    Print an overview of features with descriptive statistics and
    modelling decisions.

    Args:
        df: DataFrame (features + target).
        target_col: Name of the target column.

    Returns:
        Summary DataFrame with one row per feature.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    rows: list = []

    for col in feature_cols:
        _dtype = str(df[col].dtype)
        n_unique = int(df[col].nunique())
        missing = int(df[col].isna().sum())
        missing_pct = missing / len(df) * 100

        if pd.api.types.is_numeric_dtype(df[col]):
            kind = "numerical"
            corr = df[col].corr(df[target_col]) if target_col in df.columns else np.nan
        else:
            kind = "categorical"
            corr = np.nan

        # Decision logic
        if missing_pct > 50:
            decision = "Drop (>50% missing)"
        elif kind == "categorical" and n_unique > 50:
            decision = "Target encode (high cardinality)"
        elif kind == "categorical":
            decision = "One-hot / Target encode"
        elif not np.isnan(corr) and abs(corr) > 0.9:
            decision = "Keep (strong predictor)"
        elif not np.isnan(corr) and abs(corr) > 0.5:
            decision = "Keep (moderate predictor)"
        elif not np.isnan(corr) and abs(corr) < 0.05:
            decision = "Consider dropping (weak)"
        else:
            decision = "Keep"

        rows.append(
            {
                "Feature": col,
                "Type": kind,
                "Unique": n_unique,
                "Missing%": round(missing_pct, 1),
                "Corr(target)": round(corr, 3) if not np.isnan(corr) else "—",
                "Decision": decision,
            }
        )

    summary = pd.DataFrame(rows)

    print_header("FEATURE OVERVIEW", "🔎")
    print(f"  Total features: {len(feature_cols)}")
    n_num = sum(1 for r in rows if r["Type"] == "numerical")
    n_cat = sum(1 for r in rows if r["Type"] == "categorical")
    print(f"  Numerical: {n_num}  |  Categorical: {n_cat}")

    n_drop = sum(1 for r in rows if r["Decision"].startswith("Drop"))
    n_strong = sum(1 for r in rows if "strong" in r["Decision"])
    print(f"  Strong predictors: {n_strong}  |  Candidates to drop: {n_drop}")
    print()
    print(summary.to_string(index=False))

    return summary
