"""Pydantic row/sample models for data ingestion and feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

BehaviorType = Literal["NORMAL", "AGGRESSIVE", "DROWSY"]
RoadType = Literal["MOTORWAY", "SECONDARY"]


class DataSampleModel(BaseModel):
    """Base model for validated data samples."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")


class TripInfoSample(DataSampleModel):
    """Trip metadata sample from UAH folder structure."""

    path: Path
    driver: str = Field(min_length=2)
    behavior: BehaviorType
    road_type: RoadType

    @field_validator("driver")
    @classmethod
    def normalize_driver(cls, value: str) -> str:
        return value.upper()

    @field_validator("behavior", mode="before")
    @classmethod
    def normalize_behavior(cls, value: str) -> str:
        return str(value).upper()

    @field_validator("road_type", mode="before")
    @classmethod
    def normalize_road_type(cls, value: str) -> str:
        return str(value).upper()


class UAHRawGPSSample(DataSampleModel):
    """One GPS sample row from UAH raw data files."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="allow")

    timestamp: float
    speed: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    long: Optional[float] = None
    course: Optional[float] = None
    altitude: Optional[float] = None
    v_accuracy: Optional[float] = None
    h_accuracy: Optional[float] = None
    diff_course: Optional[float] = None


class UAHRawAccelerometerSample(DataSampleModel):
    """One accelerometer sample row from UAH raw data files."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="allow")

    timestamp: float
    acc_x_kf: float
    acc_y_kf: float
    acc_z_kf: float
    active: Optional[int] = None
    flag: Optional[int] = None
    acc_x: Optional[float] = None
    acc_y: Optional[float] = None
    acc_z: Optional[float] = None
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None


class UAHInertialEventSample(DataSampleModel):
    """One inertial event row from UAH EVENTS_INERTIAL file."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="allow")

    timestamp: float
    event_type: Optional[int] = None
    level: Optional[int] = None
    event_code: Optional[int] = None
    event_name: Optional[str] = None
    level_code: Optional[int] = None
    level_name: Optional[str] = None
    lat: Optional[float] = None
    long: Optional[float] = None
    datetime: Optional[str] = None


class UAHSemanticSummarySample(DataSampleModel):
    """Trip-level semantic summary sample from SEMANTIC_ONLINE."""

    score_total: float = Field(ge=0)
    score_accelerations: float = Field(ge=0)
    score_brakings: float = Field(ge=0)
    score_turnings: float = Field(ge=0)
    score_weaving: float = Field(ge=0)
    score_drifting: float = Field(ge=0)
    score_overspeeding: float = Field(ge=0)
    score_following: float = Field(ge=0)
    ratio_normal: float = Field(ge=0, le=1)
    ratio_drowsy: float = Field(ge=0, le=1)
    ratio_aggressive: float = Field(ge=0, le=1)


class UAHTripSummarySample(UAHSemanticSummarySample):
    """UAH summary sample with metadata for one trip."""

    driver: str = Field(min_length=2)
    behavior: BehaviorType
    road_type: Optional[RoadType] = None

    @field_validator("driver")
    @classmethod
    def normalize_driver(cls, value: str) -> str:
        return value.upper()

    @field_validator("behavior", mode="before")
    @classmethod
    def normalize_behavior(cls, value: str) -> str:
        return str(value).upper()

    @field_validator("road_type", mode="before")
    @classmethod
    def normalize_road_type(cls, value: Optional[str]) -> Optional[str]:
        return str(value).upper() if value is not None else value


class ClassificationFeatureValuesSample(DataSampleModel):
    """Validated extracted feature set for one trip."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="allow")

    speed_mean: Optional[float] = None
    speed_std: Optional[float] = None
    speed_max: Optional[float] = None
    speed_min: Optional[float] = None
    speed_range: Optional[float] = None
    speed_q25: Optional[float] = None
    speed_q75: Optional[float] = None
    speed_cv: Optional[float] = None
    speed_change_mean: Optional[float] = None
    speed_change_std: Optional[float] = None
    course_change_mean: Optional[float] = None
    course_change_std: Optional[float] = None
    course_change_max: Optional[float] = None
    trip_duration: Optional[float] = None

    acc_x_mean: Optional[float] = None
    acc_x_std: Optional[float] = None
    acc_x_range: Optional[float] = None
    acc_y_mean: Optional[float] = None
    acc_y_std: Optional[float] = None
    acc_y_range: Optional[float] = None
    acc_z_mean: Optional[float] = None
    acc_z_std: Optional[float] = None
    acc_magnitude_mean: Optional[float] = None
    acc_magnitude_std: Optional[float] = None
    acc_magnitude_max: Optional[float] = None
    acc_magnitude_q95: Optional[float] = None
    acc_rms: Optional[float] = None
    jerk_x_mean: Optional[float] = None
    jerk_x_std: Optional[float] = None
    jerk_y_mean: Optional[float] = None
    jerk_y_std: Optional[float] = None
    jerk_magnitude_std: Optional[float] = None

    brake_count: Optional[int] = Field(default=None, ge=0)
    hard_brake_count: Optional[int] = Field(default=None, ge=0)
    accel_count: Optional[int] = Field(default=None, ge=0)
    turn_count: Optional[int] = Field(default=None, ge=0)
    sharp_turn_count: Optional[int] = Field(default=None, ge=0)
    brake_rate: Optional[float] = Field(default=None, ge=0)
    turn_rate: Optional[float] = Field(default=None, ge=0)

    event_braking_count: Optional[int] = Field(default=None, ge=0)
    event_braking_low: Optional[int] = Field(default=None, ge=0)
    event_braking_medium: Optional[int] = Field(default=None, ge=0)
    event_braking_high: Optional[int] = Field(default=None, ge=0)
    event_turning_count: Optional[int] = Field(default=None, ge=0)
    event_turning_low: Optional[int] = Field(default=None, ge=0)
    event_turning_medium: Optional[int] = Field(default=None, ge=0)
    event_turning_high: Optional[int] = Field(default=None, ge=0)
    event_acceleration_count: Optional[int] = Field(default=None, ge=0)
    event_acceleration_low: Optional[int] = Field(default=None, ge=0)
    event_acceleration_medium: Optional[int] = Field(default=None, ge=0)
    event_acceleration_high: Optional[int] = Field(default=None, ge=0)


class ClassificationTripSample(ClassificationFeatureValuesSample):
    """Validated classification dataset row with metadata."""

    driver: str = Field(min_length=2)
    behavior: BehaviorType
    road_type: Optional[RoadType] = None

    @field_validator("driver")
    @classmethod
    def normalize_driver(cls, value: str) -> str:
        return value.upper()

    @field_validator("behavior", mode="before")
    @classmethod
    def normalize_behavior(cls, value: str) -> str:
        return str(value).upper()

    @field_validator("road_type", mode="before")
    @classmethod
    def normalize_road_type(cls, value: Optional[str]) -> Optional[str]:
        return str(value).upper() if value is not None else value


class EPAVehicleSample(DataSampleModel):
    """Validated EPA vehicle sample."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="allow")

    comb08: float = Field(gt=0, lt=200)
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    VClass: Optional[str] = None
    sCharger: Optional[str] = None
    tCharger: Optional[str] = None
    atvType: Optional[str] = None
    drive: Optional[str] = None
    trany: Optional[str] = None
    trans_dscr: Optional[str] = None
    cylinders: Optional[float] = None
    displ: Optional[float] = None
    eng_dscr: Optional[str] = None
    engId: Optional[float] = None
    fuelType: Optional[str] = None
    fuelType1: Optional[str] = None
    fuelType2: Optional[str] = None
    evMotor: Optional[str] = None
    mfrCode: Optional[str] = None
    c240Dscr: Optional[str] = None
    charge240b: Optional[float] = None
    c240bDscr: Optional[str] = None
    range: Optional[float] = None
    rangeCity: Optional[float] = None
    rangeHwy: Optional[float] = None
    rangeA: Optional[float] = None
    hlv: Optional[float] = None
    hpv: Optional[float] = None
    lv2: Optional[float] = None
    lv4: Optional[float] = None
    pv2: Optional[float] = None
    pv4: Optional[float] = None
    startStop: Optional[str] = None
    phevBlended: Optional[str] = None
    guzzler: Optional[str] = None
    phevCity: Optional[float] = None
    phevHwy: Optional[float] = None
    phevComb: Optional[float] = None
