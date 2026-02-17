"""Kalman Filter for signal smoothing and noise reduction in sensor data.

This module provides Kalman filter implementations for preprocessing
raw accelerometer and gyroscope data from driving behavior datasets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class KalmanFilter1D:
    """1D Kalman Filter for smoothing noisy sensor measurements.

    The Kalman filter provides optimal state estimation for linear
    dynamic systems with Gaussian noise. It's particularly useful
    for smoothing accelerometer and gyroscope signals.

    Attributes:
        process_noise: Q - Process noise covariance
        measurement_noise: R - Measurement noise covariance
        state_estimate: x - Current state estimate
        error_covariance: P - Estimation error covariance
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_state: float = 0.0,
        initial_covariance: float = 1.0,
    ):
        """Initialize the Kalman filter.

        Args:
            process_noise: Q - Variance of process noise (model uncertainty)
            measurement_noise: R - Variance of measurement noise (sensor uncertainty)
            initial_state: Initial state estimate
            initial_covariance: Initial estimation error covariance
        """
        self.Q = process_noise
        self.R = measurement_noise
        self.x = initial_state
        self.P = initial_covariance

    def predict(self) -> float:
        """Prediction step: Project state and covariance ahead.

        For a simple constant model (no control input):
        x_k|k-1 = x_k-1|k-1  (state remains same)
        P_k|k-1 = P_k-1|k-1 + Q  (covariance increases)

        Returns:
            Predicted state estimate
        """
        # State prediction (constant model - no change)
        self.x_pred = self.x
        # Covariance prediction
        self.P_pred = self.P + self.Q
        return self.x_pred

    def update(self, measurement: float) -> float:
        """Update step: Incorporate new measurement.

        Args:
            measurement: New sensor measurement

        Returns:
            Updated state estimate
        """
        # Kalman gain
        K = self.P_pred / (self.P_pred + self.R)

        # State update
        self.x = self.x_pred + K * (measurement - self.x_pred)

        # Covariance update
        self.P = (1 - K) * self.P_pred

        return self.x

    def filter(self, measurement: float) -> float:
        """Combined predict and update step.

        Args:
            measurement: New sensor measurement

        Returns:
            Filtered state estimate
        """
        self.predict()
        return self.update(measurement)

    def reset(self, initial_state: float = 0.0, initial_covariance: float = 1.0):
        """Reset the filter state.

        Args:
            initial_state: Initial state estimate
            initial_covariance: Initial error covariance
        """
        self.x = initial_state
        self.P = initial_covariance


class KalmanFilter2D:
    """2D Kalman Filter with velocity estimation.

    This filter estimates both position and velocity, making it
    suitable for tracking accelerometer data where we want to
    estimate both the signal value and its rate of change.

    State vector: [position, velocity]
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        measurement_noise: float = 0.5,
    ):
        """Initialize the 2D Kalman filter.

        Args:
            dt: Time step between measurements
            process_noise_pos: Process noise for position
            process_noise_vel: Process noise for velocity
            measurement_noise: Measurement noise variance
        """
        self.dt = dt

        # State transition matrix
        self.F = np.array([[1, dt], [0, 1]])

        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0]])

        # Process noise covariance
        self.Q = np.array([[process_noise_pos, 0], [0, process_noise_vel]])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

        # Initial state [position, velocity]
        self.x = np.array([[0], [0]])

        # Initial covariance
        self.P = np.eye(2)

    def predict(self) -> np.ndarray:
        """Prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, measurement: float) -> np.ndarray:
        """Update step with new measurement."""
        z = np.array([[measurement]])

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        identity = np.eye(2)
        self.P = (identity - K @ self.H) @ self.P

        return self.x

    def filter(self, measurement: float) -> Tuple[float, float]:
        """Combined predict and update.

        Returns:
            Tuple of (filtered_position, estimated_velocity)
        """
        self.predict()
        self.update(measurement)
        return self.x[0, 0], self.x[1, 0]


def apply_kalman_filter_1d(
    signal: np.ndarray, process_noise: float = 0.01, measurement_noise: float = 0.1
) -> np.ndarray:
    """Apply 1D Kalman filter to a signal.

    Args:
        signal: Input signal array
        process_noise: Q parameter
        measurement_noise: R parameter

    Returns:
        Filtered signal
    """
    kf = KalmanFilter1D(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_state=signal[0] if len(signal) > 0 else 0.0,
    )

    filtered = np.zeros_like(signal)
    for i, measurement in enumerate(signal):
        filtered[i] = kf.filter(measurement)

    return filtered


def apply_kalman_filter_2d(
    signal: np.ndarray,
    dt: float = 0.1,
    process_noise_pos: float = 0.01,
    process_noise_vel: float = 0.1,
    measurement_noise: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply 2D Kalman filter to a signal.

    Args:
        signal: Input signal array
        dt: Time step between measurements
        process_noise_pos: Process noise for position
        process_noise_vel: Process noise for velocity
        measurement_noise: Measurement noise variance

    Returns:
        Tuple of (filtered_signal, estimated_velocity)
    """
    kf = KalmanFilter2D(
        dt=dt,
        process_noise_pos=process_noise_pos,
        process_noise_vel=process_noise_vel,
        measurement_noise=measurement_noise,
    )

    # Initialize with first measurement
    if len(signal) > 0:
        kf.x = np.array([[signal[0]], [0]])

    filtered = np.zeros_like(signal)
    velocity = np.zeros_like(signal)

    for i, measurement in enumerate(signal):
        pos, vel = kf.filter(measurement)
        filtered[i] = pos
        velocity[i] = vel

    return filtered, velocity


def smooth_sensor_data(
    df: pd.DataFrame,
    columns: List[str],
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
    suffix: str = "_kalman",
) -> pd.DataFrame:
    """Apply Kalman filter smoothing to sensor columns in a DataFrame.

    Args:
        df: Input DataFrame with sensor data
        columns: List of column names to smooth
        process_noise: Q parameter for Kalman filter
        measurement_noise: R parameter for Kalman filter
        suffix: Suffix to add to filtered column names

    Returns:
        DataFrame with additional filtered columns
    """
    df_result = df.copy()

    for col in columns:
        if col in df.columns:
            signal = df[col].values
            # Handle NaN values
            mask = ~np.isnan(signal)
            if mask.sum() > 0:
                filtered = apply_kalman_filter_1d(
                    signal[mask], process_noise=process_noise, measurement_noise=measurement_noise
                )
                # Put filtered values back
                result = np.full_like(signal, np.nan)
                result[mask] = filtered
                df_result[f"{col}{suffix}"] = result

    return df_result


def compute_kalman_features(
    signal: np.ndarray, process_noise: float = 0.01, measurement_noise: float = 0.1
) -> Dict[str, float]:
    """Compute features from Kalman-filtered signal.

    Args:
        signal: Raw signal array
        process_noise: Q parameter
        measurement_noise: R parameter

    Returns:
        Dictionary of features extracted from filtered signal
    """
    if len(signal) == 0 or np.all(np.isnan(signal)):
        return {
            "filtered_mean": np.nan,
            "filtered_std": np.nan,
            "noise_reduction_ratio": np.nan,
            "smoothness_improvement": np.nan,
        }

    # Remove NaN values
    clean_signal = signal[~np.isnan(signal)]
    if len(clean_signal) == 0:
        return {
            "filtered_mean": np.nan,
            "filtered_std": np.nan,
            "noise_reduction_ratio": np.nan,
            "smoothness_improvement": np.nan,
        }

    # Apply Kalman filter
    filtered = apply_kalman_filter_1d(clean_signal, process_noise=process_noise, measurement_noise=measurement_noise)

    # Compute features
    raw_std = np.std(clean_signal)
    filtered_std = np.std(filtered)

    # Noise reduction ratio
    noise_reduction = 1 - (filtered_std / raw_std) if raw_std > 0 else 0

    # Smoothness (based on second derivative)
    raw_diff2 = np.diff(clean_signal, n=2) if len(clean_signal) > 2 else np.array([0])
    filtered_diff2 = np.diff(filtered, n=2) if len(filtered) > 2 else np.array([0])

    raw_roughness = np.mean(np.abs(raw_diff2))
    filtered_roughness = np.mean(np.abs(filtered_diff2))

    smoothness_improvement = 1 - (filtered_roughness / raw_roughness) if raw_roughness > 0 else 0

    return {
        "filtered_mean": np.mean(filtered),
        "filtered_std": filtered_std,
        "noise_reduction_ratio": noise_reduction,
        "smoothness_improvement": smoothness_improvement,
    }


def extract_kalman_features_from_trip(
    trip_data: pd.DataFrame,
    sensor_columns: Optional[List[str]] = None,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> Dict[str, float]:
    """Extract Kalman-based features from a trip's sensor data.

    Args:
        trip_data: DataFrame with sensor measurements
        sensor_columns: Columns to process (default: accelerometer columns)
        process_noise: Q parameter
        measurement_noise: R parameter

    Returns:
        Dictionary of Kalman-based features
    """
    if sensor_columns is None:
        # Default accelerometer columns
        sensor_columns = [
            "accX",
            "accY",
            "accZ",
            "ACCELEROMETER X (m/s²)",
            "ACCELEROMETER Y (m/s²)",
            "ACCELEROMETER Z (m/s²)",
            "acc_x",
            "acc_y",
            "acc_z",
        ]

    features = {}

    for col in sensor_columns:
        if col in trip_data.columns:
            signal = trip_data[col].values
            col_features = compute_kalman_features(
                signal, process_noise=process_noise, measurement_noise=measurement_noise
            )

            # Prefix with column name
            col_prefix = col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            for key, value in col_features.items():
                features[f"{col_prefix}_{key}"] = value

    return features


def tune_kalman_parameters(
    signal: np.ndarray,
    target_smoothness: float = 0.5,
    q_range: Tuple[float, float] = (0.001, 0.1),
    r_range: Tuple[float, float] = (0.01, 1.0),
    n_trials: int = 20,
) -> Tuple[float, float, float]:
    """Tune Kalman filter parameters for desired smoothness.

    Args:
        signal: Input signal to tune on
        target_smoothness: Target smoothness improvement (0-1)
        q_range: Range for process noise parameter
        r_range: Range for measurement noise parameter
        n_trials: Number of random trials

    Returns:
        Tuple of (best_q, best_r, achieved_smoothness)
    """
    best_q, best_r = 0.01, 0.1
    best_score = float("inf")
    best_smoothness = 0.0

    clean_signal = signal[~np.isnan(signal)] if len(signal) > 0 else signal
    if len(clean_signal) < 3:
        return best_q, best_r, 0.0

    for _ in range(n_trials):
        q = np.random.uniform(q_range[0], q_range[1])
        r = np.random.uniform(r_range[0], r_range[1])

        features = compute_kalman_features(clean_signal, q, r)
        smoothness = float(features.get("smoothness_improvement", 0.0))

        score = abs(smoothness - target_smoothness)
        if score < best_score:
            best_score = score
            best_q, best_r = q, r
            best_smoothness = smoothness

    return best_q, best_r, best_smoothness
