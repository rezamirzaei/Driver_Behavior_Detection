"""
Classification Visualization Module.

Clean, reusable plotting functions for classification results.
All plots use consistent styling matching the LaTeX technical report.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.classification.types import ClassificationResult

# Consistent color palette matching LaTeX report
COLORS = {
    'NORMAL': '#27ae60',      # Green
    'DROWSY': '#3498db',      # Blue
    'AGGRESSIVE': '#e74c3c',  # Red
    'primary': '#2c3e50',     # Dark blue
    'secondary': '#7f8c8d',   # Gray
    'accent': '#f39c12',      # Orange
}

def setup_plot_style():
    """Set up consistent plot style matching LaTeX report figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

# Apply style on import
setup_plot_style()


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = 'behavior',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """Plot class distribution as bar and pie charts."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    counts = df[target_col].value_counts()
    colors = [COLORS.get(b.upper(), '#95a5a6') for b in counts.index]

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=1)
    axes[0].set_title('Class Distribution (Trip Count)', fontweight='bold')
    axes[0].set_xlabel('Behavior')
    axes[0].set_ylabel('Number of Trips')
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(count), ha='center', fontsize=11, fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=[0.02]*len(counts),
                wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    axes[1].set_title('Class Proportions', fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'behavior',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Plot feature distributions by class."""
    setup_plot_style()
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, feat in zip(axes[:n_features], features):
        for behavior in ['NORMAL', 'DROWSY', 'AGGRESSIVE']:
            data = df[df[target_col].str.upper() == behavior][feat].dropna()
            if len(data) > 0:
                ax.hist(data, alpha=0.6, label=behavior, bins=12,
                       color=COLORS[behavior], edgecolor='white')
        ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel(feat)
        ax.legend(fontsize=8)

    # Hide unused axes
    for ax in axes[n_features:]:
        ax.set_visible(False)

    plt.suptitle('Feature Distributions by Behavior Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig


def plot_driver_distribution(
    df: pd.DataFrame,
    driver_col: str = 'driver',
    target_col: str = 'behavior',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot trips per driver by behavior."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    crosstab = pd.crosstab(df[driver_col], df[target_col])
    colors = [COLORS.get(col.upper(), '#95a5a6') for col in crosstab.columns]

    crosstab.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1)
    ax.set_title('Trips per Driver by Behavior', fontweight='bold', fontsize=14)
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_ylabel('Number of Trips', fontsize=12)
    ax.legend(title='Behavior', fontsize=10)
    plt.xticks(rotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """Plot feature correlation matrix."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig


def plot_model_comparison(
    results: List[ClassificationResult],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 7)
) -> plt.Figure:
    """Plot model comparison with train vs test accuracy."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sort by test accuracy
    results = sorted(results, key=lambda x: x.test_accuracy)
    names = [r.model_name for r in results]
    test_accs = [r.test_accuracy for r in results]
    train_accs = [r.train_accuracy for r in results]

    # Test accuracy
    colors = plt.cm.RdYlGn(np.array(test_accs))
    axes[0].barh(range(len(results)), test_accs, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(range(len(results)))
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Model Comparison - Test Accuracy (D6 Held Out)', fontweight='bold', fontsize=13)
    axes[0].set_xlim(0, 1.05)
    axes[0].axvline(x=0.875, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Best: 87.5%')
    for i, acc in enumerate(test_accs):
        axes[0].text(acc + 0.02, i, f'{acc:.3f}', va='center', fontsize=9)
    axes[0].legend(loc='lower right')

    # Train vs Test
    x = np.arange(len(results))
    width = 0.35
    axes[1].barh(x - width/2, train_accs, width, label='Train', color='#3498db', edgecolor='black', linewidth=0.5)
    axes[1].barh(x + width/2, test_accs, width, label='Test', color='#e74c3c', edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel('Accuracy', fontsize=12)
    axes[1].set_title('Train vs Test Accuracy (Overfitting Analysis)', fontweight='bold', fontsize=13)
    axes[1].legend(loc='lower right', fontsize=11)
    axes[1].set_xlim(0, 1.1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str = 'Model',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'},
                linewidths=1, linecolor='white')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=13)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = 'Feature Importance',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    """Plot feature importance as horizontal bar chart."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    df_sorted = importance_df.sort_values('Importance', ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))

    bars = ax.barh(range(len(df_sorted)), df_sorted['Importance'].values,
                   color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([n.replace('_', ' ').title() for n in df_sorted['Feature'].values])
    ax.set_xlabel('Importance (Absolute Coefficient)', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13)

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, df_sorted['Importance'].values)):
        ax.text(imp + 0.01, i, f'{imp:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig


def plot_behavior_comparison(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'behavior',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Plot feature comparison across behavior classes using boxplots."""
    setup_plot_style()
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    # Extended palette to handle NORMAL1, NORMAL2 variations
    palette = {
        'NORMAL': COLORS['NORMAL'],
        'NORMAL1': COLORS['NORMAL'],
        'NORMAL2': COLORS['NORMAL'],
        'DROWSY': COLORS['DROWSY'],
        'AGGRESSIVE': COLORS['AGGRESSIVE']
    }

    for i, feature in enumerate(features):
        ax = axes[i]
        df_plot = df.copy()
        df_plot[target_col] = df_plot[target_col].str.upper()

        # Get unique behaviors in data and filter palette
        behaviors_in_data = df_plot[target_col].unique().tolist()
        filtered_palette = {k: v for k, v in palette.items() if k in behaviors_in_data}

        sns.boxplot(data=df_plot, x=target_col, y=feature, ax=ax,
                   hue=target_col, palette=filtered_palette, legend=False)
        ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Feature Comparison Across Behavior Classes', fontweight='bold', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig


def plot_raw_accelerometer(
    acc_df: pd.DataFrame,
    gps_df: Optional[pd.DataFrame] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Plot raw accelerometer data with optional GPS speed overlay."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot accelerometer axes
    time = acc_df['timestamp'] - acc_df['timestamp'].min()
    axes[0].plot(time, acc_df['acc_x'], label='X (lateral)', alpha=0.8, linewidth=1.2, color='#3498db')
    axes[0].plot(time, acc_df['acc_y'], label='Y (longitudinal)', alpha=0.8, linewidth=1.2, color='#e67e22')
    axes[0].plot(time, acc_df['acc_z'], label='Z (vertical)', alpha=0.8, linewidth=1.2, color='#9b59b6')
    axes[0].set_ylabel('Acceleration (m/s²)', fontsize=11)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_title('Raw Accelerometer Data', fontweight='bold', fontsize=13)
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Plot magnitude
    magnitude = np.sqrt(acc_df['acc_x']**2 + acc_df['acc_y']**2 + acc_df['acc_z']**2)
    axes[1].plot(time, magnitude, color=COLORS['AGGRESSIVE'], alpha=0.8, linewidth=1.2, label='Magnitude')
    axes[1].axhline(y=9.81, color='gray', linestyle='--', linewidth=1.5, label='Gravity (9.81)')
    axes[1].fill_between(time, 9.81, magnitude, alpha=0.3, color=COLORS['AGGRESSIVE'])
    axes[1].set_xlabel('Time (s)', fontsize=11)
    axes[1].set_ylabel('Magnitude (m/s²)', fontsize=11)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].set_title('Acceleration Magnitude', fontweight='bold', fontsize=13)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    return fig

