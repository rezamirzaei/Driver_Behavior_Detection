"""
Raw Data Loading and Feature Extraction Module.

Clean, reusable functions for loading UAH-DriveSet data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TripInfo:
    """Container for trip metadata."""

    path: Path
    driver: str
    behavior: str
    road_type: str


def get_all_trips(data_dir: Path) -> List[TripInfo]:
    """
    Discover all trips in the UAH-DriveSet directory.

    Args:
        data_dir: Path to UAH-DRIVESET-v1 directory

    Returns:
        List of TripInfo objects
    """
    trips = []

    for driver_dir in sorted(data_dir.glob("D*")):
        if not driver_dir.is_dir():
            continue
        driver = driver_dir.name

        for trip_dir in sorted(driver_dir.iterdir()):
            if not trip_dir.is_dir():
                continue

            # Format: 20151111132348-25km-D1-DROWSY-MOTORWAY
            # parts[0]=timestamp, parts[1]=distance, parts[2]=driver, parts[3]=behavior, parts[4]=road_type
            parts = trip_dir.name.split("-")
            if len(parts) >= 5:
                behavior = parts[3].upper()
                road_type = parts[4].upper()

                # Normalize behavior names: NORMAL1, NORMAL2 -> NORMAL
                # UAH-DriveSet has multiple "normal" driving sessions per driver
                if behavior.startswith("NORMAL"):
                    behavior = "NORMAL"

                trips.append(TripInfo(path=trip_dir, driver=driver, behavior=behavior, road_type=road_type))

    return trips


def load_raw_gps(trip_path: Path) -> Optional[pd.DataFrame]:
    """Load raw GPS data from a trip directory."""
    gps_file = trip_path / "RAW_GPS.txt"
    if not gps_file.exists():
        return None

    try:
        df = pd.read_csv(
            gps_file, sep=" ", header=None, names=["timestamp", "lat", "lon", "speed", "course", "altitude"]
        )
        return df
    except Exception:
        return None


def load_raw_accelerometer(trip_path: Path) -> Optional[pd.DataFrame]:
    """Load raw accelerometer data from a trip directory."""
    acc_file = trip_path / "RAW_ACCELEROMETERS.txt"
    if not acc_file.exists():
        return None

    try:
        # File format: timestamp flag acc_x acc_y acc_z acc_x_kf acc_y_kf acc_z_kf [gravity_x gravity_y gravity_z]
        df = pd.read_csv(acc_file, sep=r"\s+", header=None)
        # Handle variable number of columns
        if len(df.columns) >= 8:
            df.columns = ["timestamp", "flag", "acc_x", "acc_y", "acc_z", "acc_x_kf", "acc_y_kf", "acc_z_kf"] + [
                f"col_{i}" for i in range(8, len(df.columns))
            ]
        elif len(df.columns) == 7:
            df.columns = ["timestamp", "acc_x", "acc_y", "acc_z", "acc_x_kf", "acc_y_kf", "acc_z_kf"]
        return df
    except Exception:
        return None


def load_inertial_events(trip_path: Path) -> Optional[pd.DataFrame]:
    """Load inertial events from a trip directory."""
    events_file = trip_path / "EVENTS_INERTIAL.txt"
    if not events_file.exists():
        return None

    try:
        df = pd.read_csv(
            events_file,
            sep=" ",
            header=None,
            names=["timestamp", "event_code", "event_name", "level_code", "level_name"],
        )
        return df
    except Exception:
        return None


def compute_acceleration_magnitude(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> np.ndarray:
    """Compute acceleration magnitude from 3-axis data."""
    return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)


def extract_raw_features(trip_path: Path) -> Dict[str, Any]:
    """
    Extract statistical features from raw sensor data for a single trip.

    Args:
        trip_path: Path to trip directory

    Returns:
        Dictionary of feature_name -> value
    """
    features = {}

    # GPS features
    gps = load_raw_gps(trip_path)
    if gps is not None and len(gps) > 1:
        features["speed_mean"] = gps["speed"].mean()
        features["speed_std"] = gps["speed"].std()
        features["speed_max"] = gps["speed"].max()
        features["speed_min"] = gps["speed"].min()
        features["speed_range"] = gps["speed"].max() - gps["speed"].min()
        features["speed_q25"] = gps["speed"].quantile(0.25)
        features["speed_q75"] = gps["speed"].quantile(0.75)
        speed_mean = gps["speed"].mean()
        features["speed_cv"] = (gps["speed"].std() / speed_mean) if speed_mean > 0 else 0.0
        features["speed_change_mean"] = gps["speed"].diff().abs().mean()
        features["speed_change_std"] = gps["speed"].diff().abs().std()
        features["course_change_mean"] = gps["course"].diff().abs().mean()
        features["course_change_std"] = gps["course"].diff().abs().std()
        features["course_change_max"] = gps["course"].diff().abs().max()
        features["trip_duration"] = gps["timestamp"].max() - gps["timestamp"].min()

    # Accelerometer features
    acc = load_raw_accelerometer(trip_path)
    if acc is not None and len(acc) > 1:
        features["acc_x_mean"] = acc["acc_x_kf"].mean()
        features["acc_x_std"] = acc["acc_x_kf"].std()
        features["acc_x_range"] = acc["acc_x_kf"].max() - acc["acc_x_kf"].min()
        features["acc_y_mean"] = acc["acc_y_kf"].mean()
        features["acc_y_std"] = acc["acc_y_kf"].std()
        features["acc_y_range"] = acc["acc_y_kf"].max() - acc["acc_y_kf"].min()
        features["acc_z_mean"] = acc["acc_z_kf"].mean()
        features["acc_z_std"] = acc["acc_z_kf"].std()

        # Magnitude
        mag = compute_acceleration_magnitude(acc["acc_x_kf"].values, acc["acc_y_kf"].values, acc["acc_z_kf"].values)
        features["acc_magnitude_mean"] = mag.mean()
        features["acc_magnitude_std"] = mag.std()
        features["acc_magnitude_max"] = mag.max()
        features["acc_magnitude_q95"] = float(np.percentile(mag, 95))

        # RMS acceleration (energy measure)
        features["acc_rms"] = float(np.sqrt(np.mean(mag**2)))

        # Jerk (rate of acceleration change)
        jerk_x = acc["acc_x_kf"].diff()
        jerk_y = acc["acc_y_kf"].diff()
        features["jerk_x_mean"] = jerk_x.abs().mean()
        features["jerk_x_std"] = jerk_x.std()
        features["jerk_y_mean"] = jerk_y.abs().mean()
        features["jerk_y_std"] = jerk_y.std()
        jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2)
        features["jerk_magnitude_std"] = jerk_mag.std()

        # Event-like features from thresholds
        features["brake_count"] = (acc["acc_x_kf"] < -0.1).sum()
        features["hard_brake_count"] = (acc["acc_x_kf"] < -0.3).sum()
        features["accel_count"] = (acc["acc_x_kf"] > 0.1).sum()
        features["turn_count"] = (acc["acc_y_kf"].abs() > 0.1).sum()
        features["sharp_turn_count"] = (acc["acc_y_kf"].abs() > 0.3).sum()

        # Normalized event rates (events per second)
        duration = features.get("trip_duration", 0)
        if duration > 0:
            features["brake_rate"] = features["brake_count"] / duration
            features["turn_rate"] = features["turn_count"] / duration
        else:
            features["brake_rate"] = 0.0
            features["turn_rate"] = 0.0

    # NOTE: We intentionally DO NOT use event features from EVENTS_INERTIAL.txt
    # (event_braking_low, event_braking_medium, event_braking_high, etc.)
    # because these are derived from the DriveSafe scoring algorithm which uses
    # similar heuristics to the behavior labels. Using them would create circular logic.
    # Instead, we use raw sensor statistics computed directly above.

    return features


def build_raw_dataset(trips: List[TripInfo], verbose: bool = True) -> pd.DataFrame:
    """
    Build complete dataset from all trips.

    Args:
        trips: List of TripInfo objects
        verbose: Print progress

    Returns:
        DataFrame with one row per trip
    """
    rows = []

    for i, trip in enumerate(trips):
        features = extract_raw_features(trip.path)
        features["driver"] = trip.driver
        features["behavior"] = trip.behavior
        features["road_type"] = trip.road_type
        rows.append(features)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(trips)} trips")

    df = pd.DataFrame(rows)

    if verbose:
        print(f"✅ Built dataset: {df.shape}")

    return df


def load_or_build_dataset(
    data_dir: Path, cache_path: Optional[Path] = None, force_rebuild: bool = False
) -> pd.DataFrame:
    """
    Load cached dataset or build from raw data.

    Args:
        data_dir: Path to UAH-DRIVESET-v1 directory
        cache_path: Path to cache CSV file
        force_rebuild: Force rebuild even if cache exists

    Returns:
        DataFrame with extracted features
    """
    if cache_path and cache_path.exists() and not force_rebuild:
        print(f"📂 Loading cached dataset: {cache_path}")
        df = pd.read_csv(cache_path)
        # Normalize behavior names (NORMAL1, NORMAL2 -> NORMAL)
        if "behavior" in df.columns:
            df["behavior"] = df["behavior"].apply(
                lambda x: "NORMAL" if str(x).upper().startswith("NORMAL") else str(x).upper()
            )
        return df

    print("🔧 Building dataset from raw data...")
    trips = get_all_trips(data_dir)
    df = build_raw_dataset(trips)

    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"💾 Saved to: {cache_path}")

    return df
