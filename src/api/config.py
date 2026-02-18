"""Runtime configuration for the API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field
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
    run_store_path: str = "results/analytics_runs.sqlite"

    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

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
