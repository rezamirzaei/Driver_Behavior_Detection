"""Celery task entrypoints for ABAX async training jobs."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, cast

try:
    from celery import Celery  # type: ignore
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    raise RuntimeError("Celery is required to import src.api.celery_tasks") from exc

from src.api.config import get_settings
from src.api.services import build_service

settings = get_settings()
celery_app = Celery(
    "abax-worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=1,
)


@celery_app.task(
    bind=True,
    name="src.api.celery_tasks.run_custom_learning_task",
    acks_late=True,
    track_started=True,
    soft_time_limit=settings.celery_soft_time_limit_seconds,
    time_limit=settings.celery_time_limit_seconds,
)
def run_custom_learning_task(
    self,
    payload: Dict[str, Any],
    runtime_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one custom-learning request with retry/backoff policy."""
    service = build_service(settings)
    task_name = str(payload["task"])
    if task_name not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task '{task_name}' for custom learning.")
    task = cast(Literal["classification", "regression"], task_name)

    try:
        return service.custom_learning(
            task=task,
            model_name=str(payload["model_name"]),
            feature_names=list(payload.get("feature_names", [])),
            cv_folds=int(payload.get("cv_folds", 1)),
            persist_artifact=bool(payload.get("persist_artifact", False)),
            artifact_id=payload.get("artifact_id"),
        )
    except ValueError:
        # Validation errors should fail fast and not retry.
        raise
    except Exception as exc:
        policy = runtime_policy or {}
        max_retries = int(policy.get("max_retries", settings.celery_task_max_retries))
        retry_backoff = bool(policy.get("retry_backoff", settings.celery_retry_backoff))
        retry_backoff_max = int(policy.get("retry_backoff_max", settings.celery_retry_backoff_max))
        retry_count = int(getattr(self.request, "retries", 0))

        if retry_count >= max_retries:
            raise

        countdown = 1
        if retry_backoff:
            countdown = min(2**retry_count, retry_backoff_max)
        raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)
