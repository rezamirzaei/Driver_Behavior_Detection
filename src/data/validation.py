"""Reusable validation helpers for record- and dataframe-level sample checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any, List, TypeVar
import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


def _normalize_value(value: Any) -> Any:
    """Normalize pandas/numpy scalars and NaN values for Pydantic parsing."""
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value):
            return None
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA:
        return None
    return value


def normalize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return a plain python dict with NaNs mapped to None."""
    return {key: _normalize_value(value) for key, value in mapping.items()}


def validate_record(
    payload: Mapping[str, Any],
    model_cls: type[ModelT],
    *,
    strict: bool = True,
    context: str = "record",
) -> dict[str, Any]:
    """Validate one record and return model-dumped dictionary."""
    normalized = normalize_mapping(payload)
    try:
        model = model_cls.model_validate(normalized)
        return model.model_dump(mode="python")
    except ValidationError as exc:
        if strict:
            raise ValueError(f"{context} failed validation: {exc}") from exc
        warnings.warn(f"{context} dropped due to validation error: {exc}", stacklevel=2)
        return {}


def validate_records(
    records: Iterable[Mapping[str, Any]],
    model_cls: type[ModelT],
    *,
    strict: bool = False,
    context: str = "records",
) -> List[dict[str, Any]]:
    """Validate a collection of records, optionally dropping invalid rows."""
    validated: List[dict[str, Any]] = []
    errors = 0

    for index, record in enumerate(records):
        item_context = f"{context}[{index}]"
        try:
            model = model_cls.model_validate(normalize_mapping(record))
            validated.append(model.model_dump(mode="python"))
        except ValidationError as exc:
            errors += 1
            if strict:
                raise ValueError(f"{item_context} failed validation: {exc}") from exc

    if errors and not strict:
        warnings.warn(
            f"{context}: dropped {errors} invalid row(s) during Pydantic validation.",
            stacklevel=2,
        )

    return validated


def validate_dataframe_records(
    frame: pd.DataFrame,
    model_cls: type[ModelT],
    *,
    strict: bool = False,
    context: str = "dataframe",
) -> pd.DataFrame:
    """Validate dataframe rows with a Pydantic sample model."""
    if frame.empty:
        return frame.copy()

    records = frame.to_dict(orient="records")
    validated_rows = validate_records(records, model_cls, strict=strict, context=context)

    if not validated_rows:
        if strict:
            raise ValueError(f"{context}: validation removed every row.")
        return frame.iloc[0:0].copy()

    validated = pd.DataFrame(validated_rows)
    original_cols = list(frame.columns)
    remaining_cols = [col for col in validated.columns if col not in original_cols]
    ordered = [col for col in original_cols if col in validated.columns] + remaining_cols
    return validated[ordered]
