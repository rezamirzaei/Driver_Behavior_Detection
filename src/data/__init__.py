"""Data module exports."""

from src.data.epa_loader import EPADataLoader, load_epa_fuel_economy
from src.data.raw_loader import (
    build_raw_dataset,
    compute_acceleration_magnitude,
    extract_raw_features,
    get_all_trips,
    get_feature_columns,
    load_inertial_events,
    load_raw_accelerometer,
    load_raw_gps,
    load_semantic_online,
    summarize_events,
)
from src.data.splitter import split_by_driver, split_data
from src.data.uah_loader import UAHDataLoader, load_uah_driveset

__all__ = [
    "UAHDataLoader",
    "load_uah_driveset",
    "EPADataLoader",
    "load_epa_fuel_economy",
    "split_data",
    "split_by_driver",
    # Raw data utilities
    "get_all_trips",
    "load_raw_gps",
    "load_raw_accelerometer",
    "load_inertial_events",
    "load_semantic_online",
    "extract_raw_features",
    "build_raw_dataset",
    "get_feature_columns",
    "compute_acceleration_magnitude",
    "summarize_events",
]
