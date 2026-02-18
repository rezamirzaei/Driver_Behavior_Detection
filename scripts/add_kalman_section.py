"""Add Kalman filter section to the classification notebook."""

import json
import os

def create_kalman_cells():
    """Create notebook cells for Kalman filter section."""
    cells = []

    # Markdown header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Kalman Filter Signal Processing\n",
            "\n",
            "This section demonstrates the application of **Kalman filtering** for signal smoothing and noise reduction in sensor data.\n",
            "\n",
            "### Background\n",
            "\n",
            "The Kalman filter is an optimal recursive algorithm for state estimation in linear dynamic systems with Gaussian noise. For driving behavior analysis, it provides:\n",
            "\n",
            "- **Noise Reduction**: Smooths noisy accelerometer and gyroscope measurements\n",
            "- **Signal Estimation**: Estimates the true underlying signal from noisy observations\n",
            "- **Velocity Estimation**: 2D Kalman filter can estimate signal rate of change\n",
            "\n",
            "The filter operates in two steps:\n",
            "1. **Predict**: Project state estimate forward based on system dynamics\n",
            "2. **Update**: Incorporate new measurement to refine state estimate"
        ]
    })

    # Import cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import Kalman filter functions\n",
            "from src.features.kalman import (\n",
            "    KalmanFilter1D,\n",
            "    KalmanFilter2D,\n",
            "    apply_kalman_filter_1d,\n",
            "    apply_kalman_filter_2d,\n",
            "    smooth_sensor_data,\n",
            "    compute_kalman_features,\n",
            "    extract_kalman_features_from_trip\n",
            ")"
        ]
    })

    # Load sample data
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9.1 Load Sample Raw Sensor Data"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load sample raw trip data for Kalman filter demonstration\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "# Find a sample trip with raw accelerometer data\n",
            "data_dir = Path('data/UAH-DRIVESET-v1')\n",
            "sample_trip_path = None\n",
            "\n",
            "for driver in ['D1', 'D2', 'D3']:\n",
            "    driver_path = data_dir / driver\n",
            "    if driver_path.exists():\n",
            "        for trip_folder in driver_path.iterdir():\n",
            "            if trip_folder.is_dir():\n",
            "                raw_accel = trip_folder / 'RAW_ACCELEROMETERS.txt'\n",
            "                if raw_accel.exists():\n",
            "                    sample_trip_path = trip_folder\n",
            "                    break\n",
            "    if sample_trip_path:\n",
            "        break\n",
            "\n",
            "if sample_trip_path:\n",
            "    print(f\"Sample trip: {sample_trip_path}\")\n",
            "    raw_accel_file = sample_trip_path / 'RAW_ACCELEROMETERS.txt'\n",
            "    raw_df = pd.read_csv(raw_accel_file, sep=' ', header=None, \n",
            "                         names=['timestamp', 'accX', 'accY', 'accZ'])\n",
            "    print(f\"Raw data shape: {raw_df.shape}\")\n",
            "    print(raw_df.head())\n",
            "else:\n",
            "    print(\"No sample trip found\")"
        ]
    })

    # Apply Kalman filter
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9.2 Apply 1D Kalman Filter\n",
            "\n",
            "We apply the 1D Kalman filter to smooth accelerometer readings. The filter parameters are:\n",
            "- **Process noise (Q)**: Model uncertainty - higher values allow faster tracking\n",
            "- **Measurement noise (R)**: Sensor uncertainty - higher values mean more smoothing"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Apply Kalman filter to accelerometer X\n",
            "if sample_trip_path and len(raw_df) > 0:\n",
            "    # Take a subset for visualization\n",
            "    subset = raw_df.head(500).copy()\n",
            "    \n",
            "    # Apply 1D Kalman filter with different parameters\n",
            "    raw_signal = subset['accX'].values\n",
            "    \n",
            "    # Low smoothing (responsive)\n",
            "    filtered_low = apply_kalman_filter_1d(raw_signal, process_noise=0.1, measurement_noise=0.1)\n",
            "    \n",
            "    # Medium smoothing\n",
            "    filtered_med = apply_kalman_filter_1d(raw_signal, process_noise=0.01, measurement_noise=0.5)\n",
            "    \n",
            "    # High smoothing (very smooth)\n",
            "    filtered_high = apply_kalman_filter_1d(raw_signal, process_noise=0.001, measurement_noise=1.0)\n",
            "    \n",
            "    print(\"Kalman filter applied with different smoothing levels\")\n",
            "    print(f\"Raw signal std: {np.std(raw_signal):.4f}\")\n",
            "    print(f\"Filtered (low smoothing) std: {np.std(filtered_low):.4f}\")\n",
            "    print(f\"Filtered (medium smoothing) std: {np.std(filtered_med):.4f}\")\n",
            "    print(f\"Filtered (high smoothing) std: {np.std(filtered_high):.4f}\")"
        ]
    })

    # Visualization
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9.3 Visualize Kalman Filter Effect"
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "\n",
            "if sample_trip_path and len(raw_df) > 0:\n",
            "    fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "    \n",
            "    time = np.arange(len(raw_signal))\n",
            "    \n",
            "    # Raw vs filtered comparison\n",
            "    axes[0, 0].plot(time, raw_signal, 'b-', alpha=0.5, linewidth=0.5, label='Raw')\n",
            "    axes[0, 0].plot(time, filtered_med, 'r-', linewidth=1.5, label='Kalman Filtered')\n",
            "    axes[0, 0].set_xlabel('Sample')\n",
            "    axes[0, 0].set_ylabel('Acceleration X (m/s²)')\n",
            "    axes[0, 0].set_title('Raw vs Kalman Filtered Signal')\n",
            "    axes[0, 0].legend()\n",
            "    axes[0, 0].grid(True, alpha=0.3)\n",
            "    \n",
            "    # Different smoothing levels\n",
            "    axes[0, 1].plot(time[:100], raw_signal[:100], 'b-', alpha=0.3, linewidth=0.5, label='Raw')\n",
            "    axes[0, 1].plot(time[:100], filtered_low[:100], 'g-', linewidth=1, label='Low smoothing')\n",
            "    axes[0, 1].plot(time[:100], filtered_med[:100], 'orange', linewidth=1, label='Medium smoothing')\n",
            "    axes[0, 1].plot(time[:100], filtered_high[:100], 'r-', linewidth=1.5, label='High smoothing')\n",
            "    axes[0, 1].set_xlabel('Sample')\n",
            "    axes[0, 1].set_ylabel('Acceleration X (m/s²)')\n",
            "    axes[0, 1].set_title('Effect of Kalman Filter Parameters')\n",
            "    axes[0, 1].legend()\n",
            "    axes[0, 1].grid(True, alpha=0.3)\n",
            "    \n",
            "    # Noise distribution\n",
            "    noise = raw_signal - filtered_med\n",
            "    axes[1, 0].hist(noise, bins=50, edgecolor='black', alpha=0.7)\n",
            "    axes[1, 0].set_xlabel('Residual (Raw - Filtered)')\n",
            "    axes[1, 0].set_ylabel('Frequency')\n",
            "    axes[1, 0].set_title(f'Noise Distribution (std={np.std(noise):.4f})')\n",
            "    axes[1, 0].axvline(x=0, color='r', linestyle='--')\n",
            "    axes[1, 0].grid(True, alpha=0.3)\n",
            "    \n",
            "    # Apply to all axes\n",
            "    smoothed_df = smooth_sensor_data(\n",
            "        subset, \n",
            "        columns=['accX', 'accY', 'accZ'],\n",
            "        process_noise=0.01,\n",
            "        measurement_noise=0.5\n",
            "    )\n",
            "    \n",
            "    axes[1, 1].plot(time[:200], smoothed_df['accX'].values[:200], label='X', alpha=0.8)\n",
            "    axes[1, 1].plot(time[:200], smoothed_df['accY'].values[:200], label='Y', alpha=0.8)\n",
            "    axes[1, 1].plot(time[:200], smoothed_df['accZ'].values[:200], label='Z', alpha=0.8)\n",
            "    axes[1, 1].set_xlabel('Sample')\n",
            "    axes[1, 1].set_ylabel('Acceleration (m/s²)')\n",
            "    axes[1, 1].set_title('Kalman Filtered 3-Axis Accelerometer')\n",
            "    axes[1, 1].legend()\n",
            "    axes[1, 1].grid(True, alpha=0.3)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig('results/figures/kalman_filter_demo.png', dpi=150, bbox_inches='tight')\n",
            "    plt.show()\n",
            "    print(\"Figure saved to results/figures/kalman_filter_demo.png\")"
        ]
    })

    # 2D Kalman with velocity
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9.4 2D Kalman Filter with Velocity Estimation\n",
            "\n",
            "The 2D Kalman filter estimates both the signal value and its rate of change (velocity), useful for detecting acceleration patterns."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if sample_trip_path and len(raw_df) > 0:\n",
            "    # Apply 2D Kalman filter\n",
            "    filtered_2d, velocity = apply_kalman_filter_2d(\n",
            "        raw_signal,\n",
            "        dt=0.1,  # Assuming 10Hz sampling\n",
            "        process_noise_pos=0.01,\n",
            "        process_noise_vel=0.1,\n",
            "        measurement_noise=0.5\n",
            "    )\n",
            "    \n",
            "    fig, axes = plt.subplots(2, 1, figsize=(12, 8))\n",
            "    \n",
            "    # Filtered signal\n",
            "    axes[0].plot(time, raw_signal, 'b-', alpha=0.3, linewidth=0.5, label='Raw')\n",
            "    axes[0].plot(time, filtered_2d, 'r-', linewidth=1.5, label='2D Kalman Filtered')\n",
            "    axes[0].set_xlabel('Sample')\n",
            "    axes[0].set_ylabel('Acceleration X (m/s²)')\n",
            "    axes[0].set_title('2D Kalman Filter - Position Estimate')\n",
            "    axes[0].legend()\n",
            "    axes[0].grid(True, alpha=0.3)\n",
            "    \n",
            "    # Estimated velocity (rate of change)\n",
            "    axes[1].plot(time, velocity, 'g-', linewidth=1)\n",
            "    axes[1].set_xlabel('Sample')\n",
            "    axes[1].set_ylabel('Rate of Change')\n",
            "    axes[1].set_title('2D Kalman Filter - Velocity Estimate (Signal Derivative)')\n",
            "    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)\n",
            "    axes[1].grid(True, alpha=0.3)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig('results/figures/kalman_2d_velocity.png', dpi=150, bbox_inches='tight')\n",
            "    plt.show()\n",
            "    print(\"Figure saved to results/figures/kalman_2d_velocity.png\")"
        ]
    })

    # Feature extraction
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9.5 Kalman-Based Feature Extraction\n",
            "\n",
            "We can extract features from Kalman-filtered signals that characterize the signal quality and noise properties."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if sample_trip_path and len(raw_df) > 0:\n",
            "    # Compute Kalman features for each axis\n",
            "    features_x = compute_kalman_features(raw_df['accX'].values)\n",
            "    features_y = compute_kalman_features(raw_df['accY'].values)\n",
            "    features_z = compute_kalman_features(raw_df['accZ'].values)\n",
            "    \n",
            "    print(\"Kalman Features for Accelerometer X:\")\n",
            "    for k, v in features_x.items():\n",
            "        print(f\"  {k}: {v:.4f}\")\n",
            "    \n",
            "    print(\"\\nKalman Features for Accelerometer Y:\")\n",
            "    for k, v in features_y.items():\n",
            "        print(f\"  {k}: {v:.4f}\")\n",
            "    \n",
            "    print(\"\\nKalman Features for Accelerometer Z:\")\n",
            "    for k, v in features_z.items():\n",
            "        print(f\"  {k}: {v:.4f}\")"
        ]
    })

    # Summary
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9.6 Summary and Takeaways\n",
            "\n",
            "**Key Findings:**\n",
            "\n",
            "1. **Noise Reduction**: Kalman filter effectively reduces high-frequency noise while preserving signal trends\n",
            "\n",
            "2. **Parameter Tuning**: \n",
            "   - Higher measurement noise (R) → more smoothing, slower response\n",
            "   - Higher process noise (Q) → less smoothing, faster response\n",
            "\n",
            "3. **2D Filter Benefits**: Provides velocity estimates useful for detecting sudden changes (braking, acceleration)\n",
            "\n",
            "4. **Feature Extraction**: Noise reduction ratio and smoothness improvement can characterize sensor quality\n",
            "\n",
            "**Applications in Driving Behavior Analysis:**\n",
            "- Preprocessing raw sensor data before feature extraction\n",
            "- Detecting true acceleration events vs sensor noise\n",
            "- Improving classification accuracy by reducing noise-induced variance"
        ]
    })

    return cells


def add_kalman_section_to_notebook():
    """Add Kalman filter section to the classification notebook."""
    notebook_path = 'notebooks/02_classification.ipynb'

    # Read the notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Create Kalman filter cells
    kalman_cells = create_kalman_cells()

    # Add cells before the last cell (which is usually conclusion)
    # Find a good insertion point (before conclusion if exists)
    insert_idx = len(nb['cells'])

    for i, cell in enumerate(nb['cells']):
        source = cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)
        if 'conclusion' in source.lower() or 'summary' in source.lower():
            insert_idx = i
            break

    # Insert cells
    for i, cell in enumerate(kalman_cells):
        nb['cells'].insert(insert_idx + i, cell)

    # Save the notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Added {len(kalman_cells)} cells for Kalman filter section")
    print(f"Notebook saved to {notebook_path}")


if __name__ == '__main__':
    add_kalman_section_to_notebook()
