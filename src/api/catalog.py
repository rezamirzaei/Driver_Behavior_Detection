"""Feature/model catalogs used by API schemas and runtime services."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.api.model_registry import ModelRegistry
from src.features.preprocessing import engineer_regression_features

TaskType = Literal["classification", "regression"]
FeatureSourceType = Literal["raw", "processed"]

DEFAULT_CLASSIFICATION_FEATURES: List[str] = [
    "speed_mean",
    "speed_std",
    "speed_max",
    "speed_min",
    "speed_change_mean",
    "speed_change_std",
    "course_change_mean",
    "course_change_std",
    "course_change_max",
    "trip_duration",
    "acc_x_mean",
    "acc_x_std",
    "acc_y_mean",
    "acc_y_std",
    "acc_magnitude_mean",
    "acc_magnitude_std",
    "acc_magnitude_max",
    "jerk_x_std",
    "jerk_y_std",
    "brake_count",
    "accel_count",
    "turn_count",
    "sharp_turn_count",
]

REMOVED_CLASSIFICATION_FEATURES = {"hard_brake_count", "hard_break_count"}

DEFAULT_REGRESSION_BASE_FEATURES: List[str] = [
    "year",
    "make",
    "model",
    "VClass",
    "sCharger",
    "tCharger",
    "atvType",
    "drive",
    "trany",
    "trans_dscr",
    "cylinders",
    "displ",
    "eng_dscr",
    "engId",
    "fuelType",
    "fuelType1",
    "fuelType2",
    "evMotor",
    "mfrCode",
    "c240Dscr",
    "charge240b",
    "c240bDscr",
    "range",
    "rangeCity",
    "rangeHwy",
    "rangeA",
    "hlv",
    "hpv",
    "lv2",
    "lv4",
    "pv2",
    "pv4",
    "startStop",
    "phevBlended",
    "guzzler",
    "phevCity",
    "phevHwy",
    "phevComb",
]

REGRESSION_ENGINEERED_FEATURES: List[str] = [
    "displ_per_cyl",
    "displ_x_cyl",
    "displ_squared",
    "range_total",
    "range_city_ratio",
    "has_supercharger",
    "has_turbocharger",
    "is_electric",
    "has_start_stop",
    "is_guzzler",
    "is_forced_induction",
    "vehicle_age",
]

FEATURE_DESCRIPTIONS_CLASSIFICATION: Dict[str, str] = {
    "speed_mean": "Average trip speed, reflecting overall driving tempo.",
    "speed_std": "Speed variability across the trip, capturing consistency.",
    "speed_max": "Maximum speed reached during the trip.",
    "speed_min": "Minimum non-zero trip speed observed.",
    "speed_change_mean": "Average absolute speed change between samples.",
    "speed_change_std": "Variability of speed changes, related to smoothness.",
    "course_change_mean": "Average heading change, associated with turning behavior.",
    "course_change_std": "Variability of heading changes across the trip.",
    "course_change_max": "Maximum heading shift, highlighting sharp maneuvers.",
    "trip_duration": "Total duration of the trip from sensor timestamps.",
    "acc_x_mean": "Mean longitudinal acceleration (forward/backward axis).",
    "acc_x_std": "Variability in longitudinal acceleration.",
    "acc_y_mean": "Mean lateral acceleration (side-to-side axis).",
    "acc_y_std": "Variability in lateral acceleration.",
    "acc_magnitude_mean": "Average acceleration magnitude across 3 axes.",
    "acc_magnitude_std": "Spread of acceleration magnitude, linked to aggressiveness.",
    "acc_magnitude_max": "Peak acceleration magnitude observed.",
    "jerk_x_std": "Variation in longitudinal jerk (acceleration change rate).",
    "jerk_y_std": "Variation in lateral jerk.",
    "brake_count": "Count of braking events (contiguous threshold crossings) from longitudinal acceleration.",
    "accel_count": "Count of acceleration events (contiguous positive threshold crossings).",
    "turn_count": "Count of turning events from contiguous lateral acceleration threshold crossings.",
    "sharp_turn_count": "Count of sharp-turn events from stronger lateral acceleration threshold crossings.",
}

FEATURE_DESCRIPTIONS_REGRESSION: Dict[str, str] = {
    "comb08": "EPA combined fuel economy target (miles per gallon equivalent).",
    "year": "Vehicle model year.",
    "make": "Vehicle manufacturer brand.",
    "model": "Vehicle model name as reported by EPA.",
    "VClass": "EPA vehicle class category.",
    "drive": "Drivetrain configuration (e.g., FWD, RWD, AWD).",
    "trany": "Transmission type and gear specification.",
    "cylinders": "Engine cylinder count.",
    "displ": "Engine displacement (liters).",
    "fuelType1": "Primary fuel type.",
    "rangeCity": "Estimated city range.",
    "rangeHwy": "Estimated highway range.",
    "displ_per_cyl": "Engine displacement normalized by cylinders.",
    "displ_x_cyl": "Interaction between displacement and cylinder count.",
    "displ_squared": "Non-linear displacement effect.",
    "range_total": "Combined city and highway electric range.",
    "range_city_ratio": "Share of total range attributed to city driving.",
    "has_supercharger": "Binary processed flag indicating supercharger presence.",
    "has_turbocharger": "Binary processed flag indicating turbocharger presence.",
    "is_electric": "Binary processed flag indicating electric propulsion indicators.",
    "has_start_stop": "Binary processed flag for start-stop technology.",
    "is_guzzler": "Binary processed flag for gas-guzzler designation.",
    "is_forced_induction": "Binary processed flag for turbo/supercharged powertrain.",
    "vehicle_age": "Vehicle age relative to the newest model year in data.",
}


class TaskCatalog(BaseModel):
    """Catalog for one ML task."""

    model_config = ConfigDict(frozen=True)

    task: TaskType
    dataset_name: str
    target_name: str
    features: List[str] = Field(default_factory=list)
    numeric_features: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    feature_descriptions: Dict[str, str] = Field(default_factory=dict)
    feature_sources: Dict[str, FeatureSourceType] = Field(default_factory=dict)
    model_descriptions: Dict[str, str] = Field(default_factory=dict)
    model_families: Dict[str, str] = Field(default_factory=dict)


def _load_classification_features(classification_csv: Path) -> Tuple[List[str], List[str]]:
    if classification_csv.exists():
        df = pd.read_csv(classification_csv, nrows=10)
        feature_cols = [
            col
            for col in df.columns
            if col not in {"driver", "behavior", "road_type"} and col not in REMOVED_CLASSIFICATION_FEATURES
        ]
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        return feature_cols, numeric_cols

    return DEFAULT_CLASSIFICATION_FEATURES, DEFAULT_CLASSIFICATION_FEATURES


def _load_regression_features(regression_csv: Path) -> Tuple[List[str], List[str]]:
    if regression_csv.exists():
        df = pd.read_csv(regression_csv, nrows=200)
        df = engineer_regression_features(df)
        feature_cols = [col for col in df.columns if col != "comb08"]
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        return feature_cols, numeric_cols

    fallback_features = DEFAULT_REGRESSION_BASE_FEATURES + REGRESSION_ENGINEERED_FEATURES
    fallback_numeric = [
        "year",
        "cylinders",
        "displ",
        "charge240b",
        "range",
        "rangeCity",
        "rangeHwy",
        "rangeA",
        "hlv",
        "hpv",
        "lv2",
        "lv4",
        "pv2",
        "pv4",
        "phevCity",
        "phevHwy",
        "phevComb",
        *REGRESSION_ENGINEERED_FEATURES,
    ]
    return fallback_features, fallback_numeric


def _build_feature_sources(task: TaskType, features: List[str]) -> Dict[str, FeatureSourceType]:
    if task == "classification":
        return {feature: "processed" for feature in features}

    engineered = set(REGRESSION_ENGINEERED_FEATURES)
    return {feature: ("processed" if feature in engineered else "raw") for feature in features}


def _fallback_feature_description(task: TaskType, feature: str, source: FeatureSourceType) -> str:
    if task == "classification":
        return f"Processed driving-signal feature '{feature}' derived from raw UAH GPS/accelerometer telemetry windows."

    if source == "raw":
        return f"Raw EPA vehicle attribute '{feature}' used directly from the source dataset."
    return f"Processed/engineered feature '{feature}' derived from raw EPA attributes."


def _build_feature_descriptions(
    task: TaskType,
    features: List[str],
    feature_sources: Dict[str, FeatureSourceType],
) -> Dict[str, str]:
    base = FEATURE_DESCRIPTIONS_CLASSIFICATION if task == "classification" else FEATURE_DESCRIPTIONS_REGRESSION
    descriptions: Dict[str, str] = {}

    for feature in features:
        if feature in base:
            descriptions[feature] = base[feature]
        else:
            descriptions[feature] = _fallback_feature_description(task, feature, feature_sources[feature])

    return descriptions


def _build_model_metadata(registry: ModelRegistry, task: TaskType) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    model_names = registry.list_models(task)
    payload = registry.list_model_payload(task)
    model_descriptions = {item["name"]: item["description"] for item in payload}
    model_families = {item["name"]: item["family"] for item in payload}
    return model_names, model_descriptions, model_families


def build_task_catalogs(
    classification_csv: Path,
    regression_csv: Path,
    random_state: int = 42,
) -> Dict[str, TaskCatalog]:
    """Build static task catalogs from available local data files."""
    cls_features, cls_numeric = _load_classification_features(classification_csv)
    reg_features, reg_numeric = _load_regression_features(regression_csv)

    registry = ModelRegistry(random_state=random_state)

    cls_sources = _build_feature_sources("classification", cls_features)
    reg_sources = _build_feature_sources("regression", reg_features)

    cls_models, cls_model_descriptions, cls_model_families = _build_model_metadata(registry, "classification")
    reg_models, reg_model_descriptions, reg_model_families = _build_model_metadata(registry, "regression")

    return {
        "classification": TaskCatalog(
            task="classification",
            dataset_name="UAH-DriveSet",
            target_name="behavior",
            features=cls_features,
            numeric_features=cls_numeric,
            models=cls_models,
            feature_descriptions=_build_feature_descriptions("classification", cls_features, cls_sources),
            feature_sources=cls_sources,
            model_descriptions=cls_model_descriptions,
            model_families=cls_model_families,
        ),
        "regression": TaskCatalog(
            task="regression",
            dataset_name="EPA Fuel Economy",
            target_name="comb08",
            features=reg_features,
            numeric_features=reg_numeric,
            models=reg_models,
            feature_descriptions=_build_feature_descriptions("regression", reg_features, reg_sources),
            feature_sources=reg_sources,
            model_descriptions=reg_model_descriptions,
            model_families=reg_model_families,
        ),
    }
