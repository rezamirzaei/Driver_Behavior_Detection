"""Helpers for dataset cache read/write with metadata sidecars."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

CacheFormat = Literal["csv", "parquet"]


class CacheMetadata(BaseModel):
    """Metadata persisted alongside cached datasets."""

    dataset_name: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    cache_format: CacheFormat
    row_count: int = Field(ge=0)
    columns: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)
    updated_at: str


def detect_cache_format(cache_path: Path) -> CacheFormat:
    """Infer cache format from file extension."""
    suffix = cache_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return "csv"
    return "parquet"


def cache_metadata_path(cache_path: Path) -> Path:
    """Return metadata sidecar path for a cache file."""
    return Path(f"{cache_path}.meta.json")


def read_dataframe_cache(cache_path: Path) -> pd.DataFrame:
    """Read cached dataframe from CSV or Parquet path."""
    cache_format = detect_cache_format(cache_path)
    if cache_format == "csv":
        return pd.read_csv(cache_path)
    return pd.read_parquet(cache_path)


def write_dataframe_cache(
    frame: pd.DataFrame,
    cache_path: Path,
    *,
    dataset_name: str,
    schema_version: str,
) -> None:
    """Persist cached dataframe and metadata sidecar."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_format = detect_cache_format(cache_path)
    if cache_format == "csv":
        frame.to_csv(cache_path, index=False)
    else:
        frame.to_parquet(cache_path, index=False)

    metadata = CacheMetadata(
        dataset_name=dataset_name,
        schema_version=schema_version,
        cache_format=cache_format,
        row_count=int(len(frame)),
        columns=[str(column) for column in frame.columns.tolist()],
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        updated_at=datetime.now(tz=timezone.utc).isoformat(),
    )

    meta_path = cache_metadata_path(cache_path)
    meta_path.write_text(json.dumps(metadata.model_dump(), indent=2), encoding="utf-8")


def read_cache_metadata(cache_path: Path) -> Optional[CacheMetadata]:
    """Read cache metadata sidecar if present."""
    meta_path = cache_metadata_path(cache_path)
    if not meta_path.exists():
        return None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return CacheMetadata.model_validate(payload)
