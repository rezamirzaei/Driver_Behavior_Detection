"""Runtime configuration for the API."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppSettings(BaseSettings):
    """Environment-driven settings for API runtime."""

    model_config = SettingsConfigDict(env_prefix="ABAX_", env_file=".env", env_file_encoding="utf-8")

    app_name: str = "ABAX Model Analytics API"
    app_version: str = "2.0.0"

    random_state: int = 42
    test_size: float = Field(default=0.2, gt=0.05, lt=0.5)
    training_history_iterations: int = Field(default=12, ge=2, le=40)
    training_cache_enabled: bool = True
    training_cache_max_entries: int = Field(default=256, ge=16, le=4096)
    async_job_workers: int = Field(default=2, ge=1, le=8)
    async_job_backend: Literal["local", "celery"] = "local"
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    classification_cache_csv: str = "data/processed/uah_raw_features.parquet"
    classification_raw_dir: str = "data/UAH-DRIVESET-v1"
    regression_cache_csv: str = "data/processed/epa_fuel_economy.parquet"
    data_manifest_path: str = "data/processed/data_manifest.json"
    database_url: str = "sqlite:///results/analytics_runs.sqlite"
    run_store_path: str = "results/analytics_runs.sqlite"
    artifact_dir: str = "results/model_artifacts"
    celery_task_max_retries: int = Field(default=3, ge=0, le=10)
    celery_retry_backoff: bool = True
    celery_retry_backoff_max: int = Field(default=60, ge=1, le=600)
    celery_soft_time_limit_seconds: int = Field(default=240, ge=30, le=3600)
    celery_time_limit_seconds: int = Field(default=300, ge=60, le=7200)
    api_auth_enabled: bool = False
    api_keys: List[str] = Field(default_factory=list)
    api_key_header: str = "X-API-Key"
    api_quota_per_minute: int = Field(default=120, ge=1, le=10000)
    api_auth_exempt_paths: List[str] = Field(default_factory=lambda: ["/api/health"])
    drift_alerts_enabled: bool = True
    drift_alert_score_threshold: float = Field(default=1.0, ge=0.0, le=20.0)
    drift_alert_flagged_feature_threshold: int = Field(default=1, ge=1, le=500)
    drift_alert_cooldown_seconds: int = Field(default=300, ge=0, le=86400)
    drift_alert_webhook_url: str = ""
    drift_alert_webhook_timeout_seconds: float = Field(default=4.0, gt=0.1, le=30.0)

    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator("api_keys", "api_auth_exempt_paths", mode="before")
    @classmethod
    def _parse_list_field(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    def resolve_path(self, raw_path: str) -> Path:
        """Resolve project-relative paths while preserving absolute paths."""
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return (PROJECT_ROOT / candidate).resolve()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached settings instance."""
    return AppSettings()
