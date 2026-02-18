"""In-process observability metrics for API and training operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class _RequestMetric:
    count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    last_duration_ms: float = 0.0

    def to_payload(self) -> Dict[str, Any]:
        avg = self.total_duration_ms / self.count if self.count else 0.0
        return {
            "count": self.count,
            "error_count": self.error_count,
            "avg_duration_ms": round(avg, 3),
            "max_duration_ms": round(self.max_duration_ms, 3),
            "last_duration_ms": round(self.last_duration_ms, 3),
        }


@dataclass
class _TrainingMetric:
    count: int = 0
    cache_hit_count: int = 0
    total_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    last_duration_ms: float = 0.0

    def to_payload(self) -> Dict[str, Any]:
        avg = self.total_duration_ms / self.count if self.count else 0.0
        hit_rate = self.cache_hit_count / self.count if self.count else 0.0
        return {
            "count": self.count,
            "cache_hit_count": self.cache_hit_count,
            "cache_hit_rate": round(hit_rate, 4),
            "avg_duration_ms": round(avg, 3),
            "max_duration_ms": round(self.max_duration_ms, 3),
            "last_duration_ms": round(self.last_duration_ms, 3),
        }


class ObservabilityRegistry:
    """Thread-safe in-memory metrics registry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_metrics: Dict[str, _RequestMetric] = {}
        self._training_metrics: Dict[str, _TrainingMetric] = {}
        self._started_at = _utc_now_iso()

    def record_request(self, *, method: str, path: str, status_code: int, duration_ms: float) -> None:
        key = f"{method.upper()} {path}"
        with self._lock:
            metric = self._request_metrics.get(key)
            if metric is None:
                metric = _RequestMetric()
                self._request_metrics[key] = metric

            metric.count += 1
            if status_code >= 400:
                metric.error_count += 1
            metric.total_duration_ms += duration_ms
            metric.max_duration_ms = max(metric.max_duration_ms, duration_ms)
            metric.last_duration_ms = duration_ms

    def record_training(
        self,
        *,
        operation: str,
        task: str,
        model_name: str,
        duration_ms: float,
        cache_hit: bool,
    ) -> None:
        key = f"{operation}|{task}|{model_name}"
        with self._lock:
            metric = self._training_metrics.get(key)
            if metric is None:
                metric = _TrainingMetric()
                self._training_metrics[key] = metric

            metric.count += 1
            if cache_hit:
                metric.cache_hit_count += 1
            metric.total_duration_ms += duration_ms
            metric.max_duration_ms = max(metric.max_duration_ms, duration_ms)
            metric.last_duration_ms = duration_ms

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            request_payload = {key: metric.to_payload() for key, metric in self._request_metrics.items()}
            training_payload = {key: metric.to_payload() for key, metric in self._training_metrics.items()}

        return {
            "started_at": self._started_at,
            "generated_at": _utc_now_iso(),
            "requests": request_payload,
            "training": training_payload,
        }
