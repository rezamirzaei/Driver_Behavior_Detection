"""
Visualization module for plots.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.schemas import (
    ClassDistributionResult,
    ClassificationMetrics,
    OutlierAnalysisResult,
    TrainingHistory,
)


def setup_style() -> None:
    """Set up consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_class_distribution(
    y: pd.Series,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plot class distribution."""
    fig, ax = plt.subplots(figsize=figsize)

    counts = y.value_counts()
    colors = sns.color_palette("viridis", len(counts))

    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor='black')

    for bar, count in zip(bars, counts.values):
        pct = count / len(y) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.01,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_class_distribution_from_result(
    result: ClassDistributionResult,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot class distribution from ClassDistributionResult.

    Args:
        result: ClassDistributionResult from analyze_class_distribution().
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    classes = list(result.counts.keys())
    counts = list(result.counts.values())
    percentages = list(result.percentages.values())

    colors = ['#2ecc71', '#f39c12', '#e74c3c'][:len(classes)]

    # Bar plot
    bars = axes[0].bar(classes, counts, color=colors, edgecolor='black')
    axes[0].set_title('Class Distribution (Counts)', fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')

    for bar, count, pct in zip(bars, counts, percentages):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    # Pie chart
    axes[1].pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Class Proportions', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_driver_behavior_distribution(
    crosstab: pd.DataFrame,
    title: str = "Trips per Driver by Behavior",
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot driver × behavior distribution.

    Args:
        crosstab: Cross-tabulation DataFrame from analyze_driver_distribution().
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#2ecc71', '#f39c12', '#e74c3c'][:len(crosstab.columns)]
    crosstab.plot(kind='bar', ax=ax, color=colors, edgecolor='black')

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Driver')
    ax.set_ylabel('Number of Trips')
    ax.legend(title='Behavior')
    plt.xticks(rotation=0)

    plt.tight_layout()
    return fig


def plot_feature_by_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 12),
) -> plt.Figure:
    """
    Plot feature distributions by target class using boxplots.

    Args:
        df: DataFrame with features and target.
        feature_columns: List of feature column names.
        target_column: Target column name.
        n_cols: Number of columns in subplot grid.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    n_features = len(feature_columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_features == 1 else list(axes)

    for idx, feature in enumerate(feature_columns):
        ax = axes[idx]
        df.boxplot(column=feature, by=target_column, ax=ax)
        ax.set_title(f'{feature} by {target_column}', fontsize=10)
        ax.set_xlabel(target_column)
        ax.set_ylabel(feature)
        plt.sca(ax)
        plt.xticks(rotation=45)

    for idx in range(len(feature_columns), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Distributions by Target', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_outlier_summary(
    outlier_results: List[OutlierAnalysisResult],
    title: str = "Outlier Summary",
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot summary of outlier analysis across features.

    Args:
        outlier_results: List of OutlierAnalysisResult from analyze_outliers_dataframe().
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    columns = [r.column for r in outlier_results]
    percentages = [r.outlier_percentage for r in outlier_results]

    colors = ['#e74c3c' if p > 5 else '#3498db' for p in percentages]

    bars = ax.barh(columns, percentages, color=colors, edgecolor='black')
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='5% threshold')

    ax.set_xlabel('Outlier Percentage (%)')
    ax.set_title(title, fontweight='bold')
    ax.legend()

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_correlation_with_target(
    correlations: Dict[str, float],
    title: str = "Feature Correlations with Target",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot feature correlations with target variable.

    Args:
        correlations: Dict of feature -> correlation from CorrelationAnalysisResult.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by absolute correlation
    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]

    ax.barh(features, values, color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=0.5)

    ax.set_xlabel('Correlation Coefficient')
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_target_distribution_regression(
    y: np.ndarray,
    target_name: str = "Target",
    figsize: Tuple[int, int] = (15, 4),
) -> plt.Figure:
    """
    Plot target distribution for regression problems.

    Args:
        y: Target array.
        target_name: Name of target variable.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Histogram
    axes[0].hist(y, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(np.mean(y), color='red', linestyle='--', label=f'Mean: {np.mean(y):.1f}')
    axes[0].axvline(np.median(y), color='green', linestyle='--', label=f'Median: {np.median(y):.1f}')
    axes[0].set_xlabel(target_name)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Target Distribution', fontweight='bold')
    axes[0].legend()

    # Boxplot
    axes[1].boxplot(y)
    axes[1].set_ylabel(target_name)
    axes[1].set_title('Box Plot (Outliers Visible)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Q-Q Plot
    sp_stats.probplot(y, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality)', fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    hue: Optional[str] = None,
    n_cols: int = 4,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Plot distributions of multiple features."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:16]

    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else ([axes] if n_features == 1 else list(axes))

    for i, col in enumerate(columns):
        ax = axes[i]

        if hue and hue in df.columns:
            for label in df[hue].unique():
                data = df[df[hue] == label][col].dropna()
                sns.kdeplot(data, ax=ax, label=str(label), fill=True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            data = df[col].dropna()
            sns.histplot(data, kde=True, ax=ax, color='steelblue', edgecolor='white')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label='Mean')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=1.5, label='Median')
            ax.legend(fontsize=7)

        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_xlabel('')

    # Hide empty subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot correlation matrix heatmap."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=len(columns) <= 12, fmt='.2f',
        cmap='RdBu_r', center=0, square=True, linewidths=0.5,
        vmin=-1, vmax=1, ax=ax
    )

    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_boxplots(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize: Tuple[int, int] = (16, 10),
) -> plt.Figure:
    """Plot boxplots for outlier detection."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()[:12]

    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.boxplot(data=df, x=col, ax=ax, color='steelblue')
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_xlabel('')

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Boxplots (Outlier Detection)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=figsize)

    cm = metrics.confusion_matrix

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Confusion Matrix (Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=metrics.class_names, yticklabels=metrics.class_names, ax=ax
    )

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_confusion_matrix_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix with both count and normalized views side by side.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    cm = sk_confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Count matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=axes[0]
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')

    # Normalized matrix
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Oranges',
        xticklabels=class_names, yticklabels=class_names, ax=axes[1]
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_classifier_comparison(
    comparison_df: pd.DataFrame,
    model_col: str = 'Model',
    title: str = "Classifier Comparison",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot side-by-side comparison of accuracy and F1-score.

    Args:
        comparison_df: DataFrame with Model, Accuracy, F1-Score columns.
        model_col: Column name for model names.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Accuracy
    comparison_df.plot(
        x=model_col, y='Accuracy', kind='bar',
        ax=axes[0], legend=False, color='skyblue', edgecolor='black'
    )
    axes[0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)

    # F1-Score
    comparison_df.plot(
        x=model_col, y='F1-Score', kind='bar',
        ax=axes[1], legend=False, color='lightcoral', edgecolor='black'
    )
    axes[1].set_title('F1-Score Comparison', fontweight='bold')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig



def plot_training_history(
    history: TrainingHistory,
    title: str = "Training History",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot training and validation scores over iterations."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(history.iterations, history.train_scores, 'b-', linewidth=2,
            label='Training', marker='o', markersize=4)

    if history.val_scores:
        ax.plot(history.iterations, history.val_scores, 'r--', linewidth=2,
                label='Validation', marker='s', markersize=4)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(history.metric_name.replace('_', ' ').title())
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot feature importance as horizontal bar chart."""
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(indices))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(indices)))[::-1]

    ax.barh(y_pos, importances[indices], color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Analysis",
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot residual diagnostics for regression."""
    from scipy import stats

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.5, edgecolor='none')
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted', fontweight='bold')

    # Residual Distribution
    ax2 = axes[1]
    sns.histplot(residuals, kde=True, ax=ax2, color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals')
    ax2.set_title('Residual Distribution', fontweight='bold')

    # Q-Q Plot
    ax3 = axes[2]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot', fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolor='none')

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric_col: str,
    model_col: str = "Model",
    title: str = "Model Comparison",
    higher_is_better: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot comparison of multiple models."""
    fig, ax = plt.subplots(figsize=figsize)

    sorted_df = results_df.sort_values(metric_col, ascending=not higher_is_better)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_df)))
    if not higher_is_better:
        colors = colors[::-1]

    bars = ax.barh(sorted_df[model_col], sorted_df[metric_col], color=colors, edgecolor='black')

    for bar, val in zip(bars, sorted_df[metric_col]):
        ax.text(val + (sorted_df[metric_col].max() * 0.01), bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)

    ax.set_xlabel(metric_col)
    ax.set_title(title, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_learning_curves(history, title='Model Learning Curves'):
    """
    Plot loss and accuracy from Keras history or PyTorch history dict.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Handle Keras History object vs PyTorch dict
    if hasattr(history, 'history'):
        metrics = history.history
    else:
        metrics = history

    # Loss
    if 'loss' in metrics:
        ax1.plot(metrics['loss'], label='Train Loss')
    if 'val_loss' in metrics:
        ax1.plot(metrics['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy
    if 'accuracy' in metrics:
        ax2.plot(metrics['accuracy'], label='Train Acc')
        if 'val_accuracy' in metrics:
            ax2.plot(metrics['val_accuracy'], label='Val Acc')
        ax2.set_title(f'{title} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

    plt.tight_layout()
    return fig


def plot_regressor_comparison(
    comparison_df: pd.DataFrame,
    model_col: str = 'Model',
    title: str = "Regressor Comparison",
    figsize: Tuple[int, int] = (15, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot side-by-side comparison of R², RMSE, and MAE.

    Args:
        comparison_df: DataFrame with Model, R² Score, RMSE, MAE columns.
        model_col: Column name for model names.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # R² Score
    comparison_df.plot(
        x=model_col, y='R² Score', kind='bar',
        ax=axes[0], legend=False, color='skyblue', edgecolor='black'
    )
    axes[0].set_title('R² Score (Higher = Better)', fontweight='bold')
    axes[0].set_ylabel('R²')
    axes[0].set_ylim([0, 1])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)

    # RMSE
    comparison_df.plot(
        x=model_col, y='RMSE', kind='bar',
        ax=axes[1], legend=False, color='lightcoral', edgecolor='black'
    )
    axes[1].set_title('RMSE (Lower = Better)', fontweight='bold')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)

    # MAE
    comparison_df.plot(
        x=model_col, y='MAE', kind='bar',
        ax=axes[2], legend=False, color='lightgreen', edgecolor='black'
    )
    axes[2].set_title('MAE (Lower = Better)', fontweight='bold')
    axes[2].set_ylabel('MAE')
    axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    r2: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comprehensive residual analysis.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        model_name: Name of the model.
        r2: Optional R² score to display.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    from scipy import stats as sp_stats

    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title(f'Actual vs Predicted ({model_name})', fontweight='bold')
    if r2 is not None:
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}',
                        transform=axes[0, 0].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        verticalalignment='top')

    # Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot', fontweight='bold')

    # Residual Distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution', fontweight='bold')

    # Q-Q Plot
    sp_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_target_vs_numerical_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot scatter plots of target vs numerical features with correlation.

    Args:
        df: DataFrame with features and target.
        target_col: Target column name.
        feature_cols: List of feature columns. None = first 6 numerical.
        n_cols: Number of columns in subplot grid.
        figsize: Figure size. None = auto-calculate.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure object.
    """
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        feature_cols = numeric_cols[:6]

    n_features = len(feature_cols)
    n_rows = (n_features + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else ([axes] if n_features == 1 else list(axes))

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        ax.scatter(df[col], df[target_col], alpha=0.5, s=10)
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        ax.set_title(f'{target_col} vs {col}', fontweight='bold')

        # Add correlation annotation
        corr = df[col].corr(df[target_col])
        ax.text(0.05, 0.95, f'r = {corr:.2f}',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_target_vs_categorical_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    max_features: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot boxplots of target distribution by categorical features.

    Args:
        df: DataFrame with features and target.
        target_col: Target column name.
        feature_cols: List of categorical columns. None = auto-detect.
        max_features: Maximum number of features to plot.
        figsize: Figure size. None = auto-calculate.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure object.
    """
    if feature_cols is None:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Prefer low-cardinality categoricals
        feature_cols = [c for c in cat_cols if df[c].nunique() < 20][:max_features]
    else:
        feature_cols = feature_cols[:max_features]

    if not feature_cols:
        # Return empty figure if no categorical features
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No categorical features to plot',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    n_features = len(feature_cols)

    if figsize is None:
        figsize = (5 * n_features, 5)

    fig, axes = plt.subplots(1, n_features, figsize=figsize)
    axes = [axes] if n_features == 1 else list(axes)

    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        df.boxplot(column=target_col, by=col, ax=ax)
        ax.set_title(f'{target_col} by {col}', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

    plt.suptitle('')  # Remove automatic title
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_categorical_distributions(
    df: pd.DataFrame,
    columns: List[str],
    top_n: int = 10,
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bar charts showing top N values for each categorical column.

    Args:
        df: DataFrame with categorical columns.
        columns: List of categorical column names.
        top_n: Number of top values to show per column.
        n_cols: Number of columns in subplot grid.
        figsize: Figure size. None = auto-calculate.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure object.
    """
    if not columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No categorical columns to plot',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle different axes array shapes
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for idx, col in enumerate(columns):
        ax = axes[idx]
        top_values = df[col].value_counts().head(top_n)

        # Handle empty series
        if len(top_values) == 0:
            ax.text(0.5, 0.5, f'{col}\n(No data)', ha='center', va='center')
            ax.axis('off')
            continue

        top_values.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
        ax.set_title(f'{col} (Top {top_n})', fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    coverage: Optional[float] = None,
    title: str = "Prediction Intervals (Quantile Regression)",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot prediction intervals from quantile regression.

    Args:
        y_true: True values.
        y_pred: Point predictions (median).
        lower: Lower bound predictions.
        upper: Upper bound predictions.
        coverage: Optional coverage fraction to display.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sort by true value for cleaner visualization
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    lower_sorted = lower[sort_idx]
    upper_sorted = upper[sort_idx]

    # Left: Prediction intervals
    ax = axes[0]
    x = np.arange(len(y_true_sorted))
    ax.fill_between(x, lower_sorted, upper_sorted, alpha=0.3, color='blue', label='Prediction Interval')
    ax.plot(x, y_pred_sorted, 'b-', linewidth=1, label='Predicted (Median)')
    ax.plot(x, y_true_sorted, 'r.', markersize=2, alpha=0.5, label='Actual')
    ax.set_xlabel('Sample (sorted by actual)')
    ax.set_ylabel('Value')
    ax.set_title('Prediction Intervals', fontweight='bold')
    ax.legend(loc='upper left')

    if coverage is not None:
        ax.text(0.95, 0.05, f'Coverage: {coverage:.1%}',
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: Interval width distribution
    ax = axes[1]
    widths = upper - lower
    ax.hist(widths, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(widths), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(widths):.2f}')
    ax.set_xlabel('Interval Width')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Interval Width Distribution', fontweight='bold')
    ax.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_model_comparison_detailed(
    comparison_df: pd.DataFrame,
    metrics: List[str],
    model_col: str = 'Model',
    title: str = "Model Comparison",
    higher_better: Optional[List[bool]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot detailed model comparison for multiple metrics.

    Args:
        comparison_df: DataFrame with model results.
        metrics: List of metric column names.
        model_col: Column name for model names.
        title: Plot title.
        higher_better: List of bools indicating if higher is better for each metric.
        figsize: Figure size.
        save_path: Optional path to save.

    Returns:
        matplotlib Figure object.
    """
    n_metrics = len(metrics)

    if figsize is None:
        figsize = (5 * n_metrics, 5)

    if higher_better is None:
        higher_better = [True] * n_metrics

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    axes = [axes] if n_metrics == 1 else list(axes)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(comparison_df)))

    for idx, (metric, higher) in enumerate(zip(metrics, higher_better)):
        ax = axes[idx]
        sorted_df = comparison_df.sort_values(metric, ascending=not higher)

        bars = ax.barh(sorted_df[model_col], sorted_df[metric], color=colors, edgecolor='black')
        ax.set_xlabel(metric)

        direction = "→ Higher=Better" if higher else "→ Lower=Better"
        ax.set_title(f'{metric}\n{direction}', fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(val, bar.get_y() + bar.get_height()/2,
                    f' {val:.3f}', va='center', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig



