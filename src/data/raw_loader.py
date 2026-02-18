"""
Raw data analysis utilities for UAH-DriveSet.

This module provides functions for:
1. Loading raw sensor data (GPS, Accelerometer, Events)
2. Extracting features from raw data
3. Building datasets for classification
"""

from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd

from src.data.event_counts import (
    ACCEL_THRESHOLD,
    BRAKE_THRESHOLD,
    SHARP_TURN_THRESHOLD,
    TURN_THRESHOLD,
    count_event_starts,
    count_threshold_events,
)
from src.data.sample_models import (
    ClassificationFeatureValuesSample,
    ClassificationTripSample,
    TripInfoSample,
    UAHInertialEventSample,
    UAHRawAccelerometerSample,
    UAHRawGPSSample,
)
from src.data.validation import validate_dataframe_records, validate_record


def get_all_trips(data_dir: Path) -> List[Dict]:
    """
    Get all trip folders with their metadata.

    Args:
        data_dir: Path to UAH-DRIVESET-v1 directory

    Returns:
        List of dicts with driver, behavior, road_type, and path
    """
    trips = []
    data_dir = Path(data_dir)

    for driver_folder in sorted(data_dir.iterdir()):
        if not driver_folder.is_dir() or not driver_folder.name.startswith("D"):
            continue
        driver = driver_folder.name

        for trip_folder in sorted(driver_folder.iterdir()):
            if not trip_folder.is_dir():
                continue

            # Parse folder name
            parts = trip_folder.name.split("-")
            behavior = None
            road_type = None

            for part in parts:
                if part in ["NORMAL", "NORMAL1", "NORMAL2", "AGGRESSIVE", "DROWSY"]:
                    behavior = part.replace("1", "").replace("2", "")
                elif part in ["MOTORWAY", "SECONDARY"]:
                    road_type = part

            if behavior:
                trip = validate_record(
                    {
                        "driver": driver,
                        "behavior": behavior,
                        "road_type": road_type,
                        "path": trip_folder,
                    },
                    TripInfoSample,
                    strict=True,
                    context=f"trip_info:{trip_folder.name}",
                )
                trips.append(trip)

    return trips


def load_raw_gps(trip_path: Path) -> Optional[pd.DataFrame]:
    """
    Load RAW_GPS.txt with proper column names.

    GPS data at 1Hz containing:
    - timestamp: Time in seconds
    - speed: Speed in km/h
    - lat/long: GPS coordinates
    - altitude: Altitude in meters
    - course: Heading direction in degrees
    - diff_course: Course variation in degrees

    Args:
        trip_path: Path to trip folder

    Returns:
        DataFrame with GPS data or None if file doesn't exist
    """
    gps_file = Path(trip_path) / "RAW_GPS.txt"
    if not gps_file.exists():
        return None

    cols = ["timestamp", "speed", "lat", "long", "altitude", "v_accuracy", "h_accuracy", "course", "diff_course"]

    try:
        df = pd.read_csv(gps_file, sep=r"\s+", header=None, usecols=range(9))
        df.columns = cols
        return validate_dataframe_records(
            df,
            UAHRawGPSSample,
            strict=False,
            context=f"RAW_GPS:{trip_path.name}",
        )
    except Exception as e:
        warnings.warn(f"Error loading GPS from {gps_file}: {e}")
        return None


def load_raw_accelerometer(trip_path: Path) -> Optional[pd.DataFrame]:
    """
    Load RAW_ACCELEROMETERS.txt with proper column names.

    Accelerometer data containing:
    - timestamp: Time in seconds
    - active: Boolean if system activated (speed > 50 km/h)
    - acc_x/y/z: Raw acceleration in g-force
    - acc_x/y/z_kf: Kalman-filtered acceleration
    - roll/pitch/yaw: Orientation in degrees

    Args:
        trip_path: Path to trip folder

    Returns:
        DataFrame with accelerometer data or None if file doesn't exist
    """
    acc_file = Path(trip_path) / "RAW_ACCELEROMETERS.txt"
    if not acc_file.exists():
        return None

    cols = [
        "timestamp",
        "active",
        "acc_x",
        "acc_y",
        "acc_z",
        "acc_x_kf",
        "acc_y_kf",
        "acc_z_kf",
        "roll",
        "pitch",
        "yaw",
    ]

    try:
        df = pd.read_csv(acc_file, sep=r"\s+", header=None, usecols=range(11))
        df.columns = cols
        return validate_dataframe_records(
            df,
            UAHRawAccelerometerSample,
            strict=False,
            context=f"RAW_ACCELEROMETERS:{trip_path.name}",
        )
    except Exception as e:
        warnings.warn(f"Error loading accelerometer from {acc_file}: {e}")
        return None


def load_inertial_events(trip_path: Path) -> Optional[pd.DataFrame]:
    """
    Load EVENTS_INERTIAL.txt - the detected driving events.

    Events contain:
    - timestamp: Time of event
    - event_type: 1=braking, 2=turning, 3=acceleration
    - level: 1=low, 2=medium, 3=high severity
    - lat/long: GPS coordinates

    Args:
        trip_path: Path to trip folder

    Returns:
        DataFrame with events or None if file doesn't exist
    """
    events_file = Path(trip_path) / "EVENTS_INERTIAL.txt"
    if not events_file.exists():
        return None

    cols = ["timestamp", "event_type", "level", "lat", "long", "datetime"]

    try:
        df = pd.read_csv(events_file, sep=r"\s+", header=None, usecols=range(6))
        df.columns = cols

        # Map event types and levels to names
        df["event_name"] = df["event_type"].map({1: "braking", 2: "turning", 3: "acceleration"})
        df["level_name"] = df["level"].map({1: "low", 2: "medium", 3: "high"})
        return validate_dataframe_records(
            df,
            UAHInertialEventSample,
            strict=False,
            context=f"EVENTS_INERTIAL:{trip_path.name}",
        )
    except Exception as e:
        warnings.warn(f"Error loading events from {events_file}: {e}")
        return None


def load_semantic_online(trip_path: Path) -> Optional[pd.DataFrame]:
    """
    Load SEMANTIC_ONLINE.txt with cumulative behavior scores.

    Contains running scores and ratios computed during the trip.
    Scores range 0-100 (100 = perfect).

    Args:
        trip_path: Path to trip folder

    Returns:
        DataFrame with semantic data or None if file doesn't exist
    """
    semantic_file = Path(trip_path) / "SEMANTIC_ONLINE.txt"
    if not semantic_file.exists():
        return None

    cols = [
        "timestamp",
        "lat",
        "long",
        "score_total_w",
        "score_acc_w",
        "score_brake_w",
        "score_turn_w",
        "score_weave_w",
        "score_drift_w",
        "score_speed_w",
        "score_follow_w",
        "ratio_normal_w",
        "ratio_drowsy_w",
        "ratio_agg_w",
        "ratio_distract_w",
        "score_total",
        "score_acc",
        "score_brake",
        "score_turn",
        "score_weave",
        "score_drift",
        "score_speed",
        "score_follow",
        "ratio_normal",
        "ratio_drowsy",
        "ratio_agg",
        "ratio_distract",
    ]

    try:
        df = pd.read_csv(semantic_file, sep=r"\s+", header=None)
        df.columns = cols[: len(df.columns)]
        return df
    except Exception as e:
        warnings.warn(f"Error loading semantic from {semantic_file}: {e}")
        return None


def compute_acceleration_magnitude(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> np.ndarray:
    """
    Compute acceleration magnitude from 3-axis components.

    Args:
        acc_x, acc_y, acc_z: Acceleration components in g-force

    Returns:
        Magnitude array
    """
    return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)


def extract_raw_features(trip_path: Path) -> Optional[Dict]:
    """
    Extract comprehensive features from raw sensor data.

    Features extracted:
    - GPS: speed stats, course change stats, trip duration
    - Accelerometer: acceleration stats, jerk, event counts
    - Events: counts by type and severity

    Args:
        trip_path: Path to trip folder

    Returns:
        Dictionary of features or None if data cannot be loaded
    """
    gps = load_raw_gps(trip_path)
    acc = load_raw_accelerometer(trip_path)
    events = load_inertial_events(trip_path)

    if gps is None or acc is None:
        return None

    features = {}

    # GPS-based features
    if gps is not None and len(gps) > 0:
        features["speed_mean"] = gps["speed"].mean()
        features["speed_std"] = gps["speed"].std()
        features["speed_max"] = gps["speed"].max()
        features["speed_min"] = gps["speed"].min()

        # Speed variability
        speed_diff = gps["speed"].diff().dropna()
        features["speed_change_mean"] = speed_diff.abs().mean()
        features["speed_change_std"] = speed_diff.std()

        # Course changes (turning indicator)
        if "diff_course" in gps.columns:
            features["course_change_mean"] = gps["diff_course"].abs().mean()
            features["course_change_std"] = gps["diff_course"].std()
            features["course_change_max"] = gps["diff_course"].abs().max()

        # Trip duration
        features["trip_duration"] = gps["timestamp"].max() - gps["timestamp"].min()

    # Accelerometer-based features
    if acc is not None and len(acc) > 0:
        # Use Kalman-filtered values for cleaner signal
        features["acc_x_mean"] = acc["acc_x_kf"].mean()
        features["acc_x_std"] = acc["acc_x_kf"].std()
        features["acc_y_mean"] = acc["acc_y_kf"].mean()
        features["acc_y_std"] = acc["acc_y_kf"].std()

        # Magnitude
        acc_mag = compute_acceleration_magnitude(acc["acc_x_kf"].values, acc["acc_y_kf"].values, acc["acc_z_kf"].values)
        features["acc_magnitude_mean"] = acc_mag.mean()
        features["acc_magnitude_std"] = acc_mag.std()
        features["acc_magnitude_max"] = acc_mag.max()

        # Jerk (rate of change of acceleration) - indicates smoothness
        jerk_x = acc["acc_x_kf"].diff().dropna()
        jerk_y = acc["acc_y_kf"].diff().dropna()
        features["jerk_x_std"] = jerk_x.std()
        features["jerk_y_std"] = jerk_y.std()

        # Threshold-driven events: count event starts instead of sample totals.
        features["brake_count"] = count_threshold_events(acc["acc_x_kf"], threshold=BRAKE_THRESHOLD, direction="below")
        features["accel_count"] = count_threshold_events(acc["acc_x_kf"], threshold=ACCEL_THRESHOLD, direction="above")
        features["turn_count"] = count_event_starts(acc["acc_y_kf"].abs() > TURN_THRESHOLD)
        features["sharp_turn_count"] = count_event_starts(acc["acc_y_kf"].abs() > SHARP_TURN_THRESHOLD)

    # Event-based features (from EVENTS_INERTIAL.txt)
    if events is not None and len(events) > 0:
        for event_type in ["braking", "turning", "acceleration"]:
            type_events = events[events["event_name"] == event_type]
            features[f"event_{event_type}_count"] = len(type_events)

            # Count by severity
            for level in ["low", "medium", "high"]:
                level_events = type_events[type_events["level_name"] == level]
                features[f"event_{event_type}_{level}"] = len(level_events)
    else:
        # Fill with zeros if no events file
        for event_type in ["braking", "turning", "acceleration"]:
            features[f"event_{event_type}_count"] = 0
            for level in ["low", "medium", "high"]:
                features[f"event_{event_type}_{level}"] = 0

    return validate_record(
        features,
        ClassificationFeatureValuesSample,
        strict=True,
        context=f"raw_features:{trip_path.name}",
    )


def build_raw_dataset(trips: List[Dict]) -> pd.DataFrame:
    """
    Build dataset from all trips using raw data features.

    Args:
        trips: List of trip metadata dicts from get_all_trips()

    Returns:
        DataFrame with features and metadata columns
    """
    all_data = []

    for trip in trips:
        features = extract_raw_features(trip["path"])
        if features is None:
            continue

        row = validate_record(
            {
                **features,
                "driver": trip["driver"],
                "behavior": trip["behavior"],
                "road_type": trip["road_type"],
            },
            ClassificationTripSample,
            strict=True,
            context=f"raw_trip:{Path(trip['path']).name}",
        )

        all_data.append(row)

    return pd.DataFrame(all_data)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding metadata).

    Args:
        df: DataFrame from build_raw_dataset()

    Returns:
        List of feature column names
    """
    metadata_cols = ["driver", "behavior", "road_type"]
    return [c for c in df.columns if c not in metadata_cols]


def summarize_events(events: pd.DataFrame) -> Dict:
    """
    Summarize event counts and distributions.

    Args:
        events: DataFrame from load_inertial_events()

    Returns:
        Dictionary with event counts and distributions
    """
    if events is None or len(events) == 0:
        return {"total_events": 0}

    summary = {
        "total_events": len(events),
        "event_counts": events["event_name"].value_counts().to_dict(),
        "severity_dist": events.groupby(["event_name", "level_name"]).size().unstack(fill_value=0).to_dict(),
    }

    return summary
