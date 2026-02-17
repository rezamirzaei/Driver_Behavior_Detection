"""Generate Kalman filter visualization figures for the technical report."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import Kalman filter functions
from src.features.kalman import (
    apply_kalman_filter_1d,
    apply_kalman_filter_2d,
    compute_kalman_features
)

def generate_kalman_figures():
    """Generate Kalman filter demonstration figures."""

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'

    # Find sample raw data
    data_dir = Path('data/UAH-DRIVESET-v1')
    sample_trip_path = None

    for driver in ['D1', 'D2', 'D3']:
        driver_path = data_dir / driver
        if driver_path.exists():
            for trip_folder in driver_path.iterdir():
                if trip_folder.is_dir():
                    raw_accel = trip_folder / 'RAW_ACCELEROMETERS.txt'
                    if raw_accel.exists():
                        sample_trip_path = trip_folder
                        break
        if sample_trip_path:
            break

    if not sample_trip_path:
        print("No sample data found. Using synthetic data.")
        # Generate synthetic noisy signal
        t = np.linspace(0, 10, 500)
        clean_signal = np.sin(t) + 0.5 * np.sin(3*t)
        raw_signal = clean_signal + np.random.normal(0, 0.3, len(t))
    else:
        print(f"Using sample trip: {sample_trip_path}")
        raw_accel_file = sample_trip_path / 'RAW_ACCELEROMETERS.txt'
        raw_df = pd.read_csv(raw_accel_file, sep=' ', header=None,
                             names=['timestamp', 'accX', 'accY', 'accZ'])
        raw_signal = raw_df['accX'].values[:500]

    # Apply Kalman filters
    filtered_low = apply_kalman_filter_1d(raw_signal, process_noise=0.1, measurement_noise=0.1)
    filtered_med = apply_kalman_filter_1d(raw_signal, process_noise=0.01, measurement_noise=0.5)
    filtered_high = apply_kalman_filter_1d(raw_signal, process_noise=0.001, measurement_noise=1.0)

    # Apply 2D Kalman for velocity estimation
    filtered_2d, velocity = apply_kalman_filter_2d(raw_signal, dt=0.1)

    # Create output directory
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Raw vs Filtered comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time = np.arange(len(raw_signal))

    # Raw vs filtered comparison
    axes[0, 0].plot(time, raw_signal, 'b-', alpha=0.5, linewidth=0.5, label='Raw')
    axes[0, 0].plot(time, filtered_med, 'r-', linewidth=1.5, label='Kalman Filtered')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Acceleration X (m/s²)')
    axes[0, 0].set_title('Raw vs Kalman Filtered Signal')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Different smoothing levels
    axes[0, 1].plot(time[:100], raw_signal[:100], 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    axes[0, 1].plot(time[:100], filtered_low[:100], 'g-', linewidth=1, label='Low smoothing (Q=0.1, R=0.1)')
    axes[0, 1].plot(time[:100], filtered_med[:100], 'orange', linewidth=1, label='Medium (Q=0.01, R=0.5)')
    axes[0, 1].plot(time[:100], filtered_high[:100], 'r-', linewidth=1.5, label='High (Q=0.001, R=1.0)')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Acceleration X (m/s²)')
    axes[0, 1].set_title('Effect of Kalman Filter Parameters')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Noise distribution
    noise = raw_signal - filtered_med
    axes[1, 0].hist(noise, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].set_xlabel('Residual (Raw - Filtered)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Extracted Noise Distribution (std={np.std(noise):.4f})')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].grid(True, alpha=0.3)

    # 2D Kalman velocity estimation
    axes[1, 1].plot(time[:200], raw_signal[:200], 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    axes[1, 1].plot(time[:200], filtered_2d[:200], 'r-', linewidth=1.5, label='Filtered')
    ax2 = axes[1, 1].twinx()
    ax2.plot(time[:200], velocity[:200], 'g-', linewidth=1, alpha=0.7, label='Velocity (jerk)')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Acceleration (m/s²)', color='r')
    ax2.set_ylabel('Velocity/Jerk', color='g')
    axes[1, 1].set_title('2D Kalman: Position + Velocity Estimation')
    axes[1, 1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'kalman_filter_demo.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'kalman_filter_demo.png'}")

    # Compute and print features
    features = compute_kalman_features(raw_signal)
    print("\nKalman Features Extracted:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")

    print("\nKalman filter figures generated successfully!")
    return True


if __name__ == '__main__':
    generate_kalman_figures()

