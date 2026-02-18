#!/usr/bin/env python3
"""
Generate all figures for LaTeX report by running the same code as notebooks.
This ensures figures match exactly what notebooks produce.
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path('/Users/rezami/PycharmProjects/ABAX')
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from src (same as notebooks)
from src.classification import (
    get_all_trips, load_raw_accelerometer, load_raw_gps,
    load_or_build_dataset, prepare_classification_data,
    get_all_classifiers, train_all_classifiers, get_best_model,
    get_feature_importance, results_to_dataframe, ClassificationResult
)
from src.classification.visualization import (
    plot_class_distribution, plot_feature_distributions,
    plot_driver_distribution, plot_correlation_matrix,
    plot_model_comparison, plot_confusion_matrix,
    plot_feature_importance, plot_behavior_comparison,
    setup_plot_style, COLORS
)
from src.models import SimpleNNClassifier, plot_nn_training_history
from sklearn.metrics import accuracy_score, f1_score

# Paths
DATA_DIR = PROJECT_ROOT / 'data' / 'UAH-DRIVESET-v1'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
CACHE_PATH = PROJECT_ROOT / 'data' / 'processed' / 'uah_raw_features.csv'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_figure1_raw_accelerometer():
    """Generate Figure 1: Raw accelerometer comparison across behaviors."""
    print("\n📊 Figure 1: Raw Accelerometer Data Comparison")
    setup_plot_style()

    trips = get_all_trips(DATA_DIR)

    # Get one trip of each behavior
    sample_trips = {}
    for behavior in ['AGGRESSIVE', 'NORMAL', 'DROWSY']:
        for trip in trips:
            trip_behavior = trip.behavior.upper()
            # Match behavior (NORMAL, NORMAL1, NORMAL2 all count as NORMAL)
            if trip_behavior.startswith(behavior) and behavior not in sample_trips:
                acc = load_raw_accelerometer(trip.path)
                if acc is not None and len(acc) > 100:
                    sample_trips[behavior] = {'trip': trip, 'acc': acc}
                    print(f"    Found {behavior}: {trip.driver}, {len(acc)} samples")
                    break

    if len(sample_trips) < 3:
        print(f"  ⚠️ Could not find all 3 behaviors, found: {list(sample_trips.keys())}")
        # Try to still generate with what we have
        if len(sample_trips) == 0:
            return

    n_behaviors = len(sample_trips)
    fig, axes = plt.subplots(n_behaviors, 2, figsize=(14, 3.5*n_behaviors))
    if n_behaviors == 1:
        axes = axes.reshape(1, 2)

    behaviors_found = list(sample_trips.keys())
    for idx, behavior in enumerate(behaviors_found):
        acc = sample_trips[behavior]['acc']
        color = COLORS[behavior]

        # Take 500 samples from middle of trip
        start_idx = len(acc) // 4
        acc = acc.iloc[start_idx:start_idx+500].copy().reset_index(drop=True)

        # Time axis
        time = np.arange(len(acc)) * 0.1

        # Left: Accelerometer X, Y, Z
        ax1 = axes[idx, 0]
        # Use available columns - check what we have
        if 'acc_x_kf' in acc.columns:
            acc_x = acc['acc_x_kf'] * 9.81
            acc_y = acc['acc_y_kf'] * 9.81
            acc_z = acc['acc_z_kf'] * 9.81
        elif 'acc_x' in acc.columns:
            acc_x = acc['acc_x']
            acc_y = acc['acc_y']
            acc_z = acc['acc_z']
            # Check if values are small (in g) and convert
            if acc_x.abs().max() < 2:
                acc_x = acc_x * 9.81
                acc_y = acc_y * 9.81
                acc_z = acc_z * 9.81
        else:
            print(f"    Warning: No accelerometer columns found. Columns: {acc.columns.tolist()}")
            continue

        ax1.plot(time, acc_x.values, label='X (longitudinal)', alpha=0.9, linewidth=1.2, color='#3498db')
        ax1.plot(time, acc_y.values, label='Y (lateral)', alpha=0.9, linewidth=1.2, color='#e67e22')
        ax1.plot(time, acc_z.values, label='Z (vertical)', alpha=0.9, linewidth=1.2, color='#9b59b6')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title(f'{behavior} - Accelerometer', fontweight='bold', color=color)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        y_max = max(abs(acc_x.max()), abs(acc_y.max()), abs(acc_z.max()), 0.5) * 1.2
        ax1.set_ylim(-y_max, y_max)

        # Right: Jerk magnitude
        ax2 = axes[idx, 1]
        dt = 0.1
        jerk_x = np.abs(np.diff(acc_x.values)) / dt
        jerk_y = np.abs(np.diff(acc_y.values)) / dt
        jerk = np.sqrt(jerk_x**2 + jerk_y**2)
        time_jerk = time[:-1]

        # Smooth
        window = 5
        if len(jerk) > window:
            jerk_smooth = np.convolve(jerk, np.ones(window)/window, mode='valid')
            time_smooth = time_jerk[:len(jerk_smooth)]
        else:
            jerk_smooth = jerk
            time_smooth = time_jerk

        ax2.fill_between(time_smooth, 0, jerk_smooth, alpha=0.4, color=color)
        ax2.plot(time_smooth, jerk_smooth, color=color, linewidth=1.5)
        threshold = np.percentile(jerk_smooth, 90)
        ax2.axhline(y=threshold, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)

        n_events = np.sum(jerk_smooth > threshold)
        ax2.set_ylabel('Jerk (m/s³)')
        ax2.set_title(f'{behavior} - Jerk (Smoothness)', fontweight='bold', color=color)
        ax2.text(0.02, 0.95, f'Mean: {np.mean(jerk_smooth):.1f}\nHarsh: {n_events}',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax2.set_ylim(0, min(np.max(jerk_smooth)*1.3, np.percentile(jerk_smooth, 99)*2))

        if idx == n_behaviors - 1:
            ax1.set_xlabel('Time (seconds)')
            ax2.set_xlabel('Time (seconds)')

    plt.suptitle('Raw Sensor Data: AGGRESSIVE vs NORMAL vs DROWSY\n'
                 '(Left: Acceleration | Right: Jerk - higher = harsher driving)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'raw_accelerometer_data.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  ✓ Saved: raw_accelerometer_data.png")

def generate_classification_figures():
    """Generate all classification figures (same as notebook cells)."""
    print("\n📊 Classification Figures")

    # Load dataset
    df = load_or_build_dataset(data_dir=DATA_DIR, cache_path=CACHE_PATH)
    feature_cols = [c for c in df.columns if c not in ['driver', 'behavior', 'road_type']]

    # Figure 2: Class Distribution
    print("  Figure 2: Class Distribution")
    plot_class_distribution(df, save_path=FIGURES_DIR / 'class_distribution.png')

    # Figure 3: Feature Distributions (excluding event counts)
    print("  Figure 3: Feature Distributions")
    key_features = ['speed_mean', 'speed_std', 'acc_magnitude_std',
                    'jerk_x_std', 'jerk_y_std', 'course_change_std']
    # Filter to features that exist
    key_features = [f for f in key_features if f in df.columns]
    plot_feature_distributions(df, key_features, save_path=FIGURES_DIR / 'feature_distributions_classification.png')

    # Figure 4: Driver Distribution
    print("  Figure 4: Driver Distribution")
    plot_driver_distribution(df, save_path=FIGURES_DIR / 'driver_behavior_distribution.png')

    # Figure 5: Correlation Matrix (only valid features)
    print("  Figure 5: Correlation Matrix")
    # Filter out event counts and features with no variance
    exclude = ['sharp_turn_count', 'brake_count', 'turn_count']
    valid_features = [c for c in feature_cols if c not in exclude and df[c].std() > 0 and df[c].notna().sum() > 0]
    plot_correlation_matrix(df, valid_features, save_path=FIGURES_DIR / 'correlation_matrix_classification.png')

    # Figure 6: Behavior Comparison
    print("  Figure 6: Behavior Comparison")
    comparison_features = ['speed_mean', 'speed_std', 'acc_magnitude_std',
                          'jerk_x_std', 'jerk_y_std', 'course_change_std']
    comparison_features = [f for f in comparison_features if f in df.columns]
    plot_behavior_comparison(df, comparison_features, save_path=FIGURES_DIR / 'behavior_comparison_raw.png')

    # Prepare data and train models
    print("  Training classifiers...")
    data = prepare_classification_data(df, test_drivers=['D6'])
    classifiers = get_all_classifiers()
    results = train_all_classifiers(data, classifiers, verbose=False)

    # Add Neural Network
    print("  Training Neural Network...")
    nn_clf = SimpleNNClassifier(
        hidden_sizes=[64, 32], dropout=0.3, epochs=100, batch_size=8,
        learning_rate=0.001, weight_decay=1e-4, early_stopping_patience=20,
        verbose=0, random_state=42
    )
    nn_clf.fit(data.X_train, data.y_train)
    y_pred_nn = nn_clf.predict(data.X_test)
    y_pred_nn_enc = nn_clf.le_.transform(y_pred_nn)
    acc_nn = accuracy_score(data.y_test, y_pred_nn_enc)
    f1_nn = f1_score(data.y_test, y_pred_nn_enc, average='weighted')
    train_acc_nn = nn_clf.score(data.X_train, data.y_train)
    results.append(ClassificationResult(
        model_name='Neural Network', train_accuracy=train_acc_nn,
        test_accuracy=acc_nn, f1_score=f1_nn,
        predictions=y_pred_nn_enc, model=nn_clf
    ))

    # Figure 7: Model Comparison
    print("  Figure 7: Model Comparison")
    plot_model_comparison(results, save_path=FIGURES_DIR / 'classifier_comparison.png')

    # Figure 8: Confusion Matrix
    print("  Figure 8: Confusion Matrix")
    best = get_best_model(results)
    plot_confusion_matrix(data.y_test, best.predictions, data.class_names,
                         model_name=best.model_name,
                         save_path=FIGURES_DIR / 'confusion_matrix_classification.png')

    # Figure 9: Feature Importance
    # Use a model that supports feature importance (Logistic L1 or Random Forest)
    print("  Figure 9: Feature Importance")
    feature_model = None
    feature_model_name = None
    for r in results:
        if 'Logistic' in r.model_name and 'L1' in r.model_name:
            feature_model = r.model
            feature_model_name = r.model_name
            break
    if feature_model is None:
        for r in results:
            if 'Random Forest' in r.model_name:
                feature_model = r.model
                feature_model_name = r.model_name
                break
    if feature_model is None:
        # Fall back to first model with coef_ or feature_importances_
        for r in results:
            if hasattr(r.model, 'coef_') or hasattr(r.model, 'feature_importances_'):
                feature_model = r.model
                feature_model_name = r.model_name
                break

    if feature_model:
        importance_df = get_feature_importance(feature_model, data.feature_names)
        plot_feature_importance(importance_df, title=f'Top 15 Features ({feature_model_name})',
                               save_path=FIGURES_DIR / 'feature_importance_classification.png')
    else:
        print("    ⚠️ No model supports feature importance")

    # Figure 10: Neural Network Learning Curves
    print("  Figure 10: NN Learning Curves")
    history = nn_clf.get_training_history()
    plot_nn_training_history(history, save_path=str(FIGURES_DIR / 'nn_learning_curves_classification.png'))

def generate_regression_figures():
    """Generate regression figures."""
    print("\n📊 Regression Figures")

    from src.features import get_feature_columns, encode_and_scale
    from src.models import get_regressors, train_and_evaluate_regressor
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    setup_plot_style()

    # Load data
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'epa_fuel_economy.csv')
    y = df['comb08'].values
    X = df.drop(columns=['comb08'])

    # Save original feature names before encoding
    original_feature_names = list(X.columns)

    numerical_cols, categorical_cols = get_feature_columns(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = encode_and_scale(X_train, X_test, y_train, categorical_cols)

    # Train models
    print("  Training regressors...")
    regressors = get_regressors(random_state=42)
    results = []
    for name, model in regressors.items():
        # Skip RANSAC as it has very high error
        if 'RANSAC' in name.upper():
            continue
        try:
            y_pred, r2, rmse, mae = train_and_evaluate_regressor(
                model, X_train_scaled, y_train, X_test_scaled, y_test
            )
            results.append({'Model': name, 'R²': r2, 'RMSE': rmse, 'MAE': mae, 'y_pred': y_pred})
        except Exception as e:
            print(f"    ⚠️ {name}: {e}")

    comparison = pd.DataFrame([{k: v for k, v in r.items() if k != 'y_pred'} for r in results])
    comparison = comparison.sort_values('R²', ascending=False)

    # Figure 11: Regressor Comparison
    print("  Figure 11: Regressor Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: R² comparison
    colors = plt.cm.RdYlGn(comparison['R²'].values / comparison['R²'].max())
    axes[0].barh(range(len(comparison)), comparison['R²'].values, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(range(len(comparison)))
    axes[0].set_yticklabels(comparison['Model'].values)
    axes[0].set_xlabel('R² Score', fontsize=12)
    axes[0].set_title('Model Comparison by R²', fontweight='bold', fontsize=13)
    axes[0].set_xlim(0, 1.05)
    for i, r2 in enumerate(comparison['R²'].values):
        axes[0].text(r2 + 0.02, i, f'{r2:.3f}', va='center', fontsize=9)

    # Right: RMSE comparison
    comparison_rmse = comparison.sort_values('RMSE')
    colors_rmse = plt.cm.RdYlGn_r(comparison_rmse['RMSE'].values / comparison_rmse['RMSE'].max())
    axes[1].barh(range(len(comparison_rmse)), comparison_rmse['RMSE'].values, color=colors_rmse, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(range(len(comparison_rmse)))
    axes[1].set_yticklabels(comparison_rmse['Model'].values)
    axes[1].set_xlabel('RMSE (MPG)', fontsize=12)
    axes[1].set_title('Model Comparison by RMSE', fontweight='bold', fontsize=13)
    for i, rmse in enumerate(comparison_rmse['RMSE'].values):
        axes[1].text(rmse + 0.2, i, f'{rmse:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'regressor_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Get best model predictions
    best_result = max(results, key=lambda x: x['R²'])
    y_pred_best = best_result['y_pred']

    # Figure 12: Actual vs Predicted
    print("  Figure 12: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred_best, alpha=0.5, edgecolors='black', linewidth=0.5, s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual MPG', fontsize=12)
    ax.set_ylabel('Predicted MPG', fontsize=12)
    ax.set_title(f'Actual vs Predicted ({best_result["Model"]})\nR² = {best_result["R²"]:.3f}', fontweight='bold', fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Figure 13: Feature Importance
    print("  Figure 13: Feature Importance")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Get feature names - try from scaled data first, then use original names
    if hasattr(X_train_scaled, 'columns'):
        feature_names = list(X_train_scaled.columns)
    elif len(original_feature_names) == X_train_scaled.shape[1]:
        feature_names = original_feature_names
    else:
        # Mapping may have changed dimensions, use generic names with original where possible
        feature_names = []
        for i in range(X_train_scaled.shape[1]):
            if i < len(original_feature_names):
                feature_names.append(original_feature_names[i])
            else:
                feature_names.append(f'encoded_{i}')

    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf.feature_importances_})
    imp_df = imp_df.nlargest(15, 'Importance').sort_values('Importance', ascending=True)

    # Clean up feature names for display
    def clean_feature_name(name):
        """Make feature names more readable."""
        name = str(name)
        # Common abbreviations
        replacements = {
            'VClass': 'Vehicle Class',
            'displ': 'Displacement',
            'cylinders': 'Cylinders',
            'fuelType': 'Fuel Type',
            'drive': 'Drive Type',
            'trany': 'Transmission',
            'sCharger': 'Supercharger',
            'tCharger': 'Turbocharger',
            'atvType': 'Alt Vehicle Type',
            'startStop': 'Start/Stop',
            'phevBlended': 'PHEV Blended',
            'evMotor': 'EV Motor',
            'hlv': 'Hatchback LV',
            'hpv': 'Hatchback PV',
            'lv2': 'Luggage Vol 2',
            'lv4': 'Luggage Vol 4',
            'pv2': 'Passenger Vol 2',
            'pv4': 'Passenger Vol 4',
        }
        for old, new in replacements.items():
            if name == old:
                return new
        return name.replace('_', ' ').title()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
    ax.barh(range(len(imp_df)), imp_df['Importance'].values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels([clean_feature_name(n) for n in imp_df['Feature'].values])
    ax.set_xlabel('Importance (Mean Decrease in Impurity)', fontsize=12)
    ax.set_title('Feature Importance for Fuel Economy (Random Forest)', fontweight='bold', fontsize=13)

    # Add value labels
    for i, imp in enumerate(imp_df['Importance'].values):
        ax.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'feature_importance_regression.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Figure 14: Residuals
    print("  Figure 14: Residuals")
    residuals = y_test - y_pred_best

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred_best, residuals, alpha=0.5, edgecolors='black', linewidth=0.5, s=30)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted MPG', fontsize=12)
    axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted', fontweight='bold', fontsize=13)

    # Residual Distribution
    axes[1].hist(residuals, bins=30, color='#3498db', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual (MPG)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontweight='bold', fontsize=13)

    plt.suptitle(f'Residual Analysis ({best_result["Model"]})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'residuals.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("  ✓ All regression figures saved")

def main():
    print("=" * 60)
    print("GENERATING ALL FIGURES FOR LATEX REPORT")
    print("(Using same code as notebooks)")
    print("=" * 60)

    generate_figure1_raw_accelerometer()
    generate_classification_figures()
    generate_regression_figures()

    print("\n" + "=" * 60)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    main()
