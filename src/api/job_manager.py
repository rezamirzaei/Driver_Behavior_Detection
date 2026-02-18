"""Background job manager with local and optional Celery backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
import traceback
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple
import uuid

JobStatus = Literal["pending", "running", "cancel_requested", "canceled", "completed", "failed"]
CustomLearningHandler = Callable[..., Dict[str, Any]]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class JobRecord:
    """Runtime state for one asynchronous job."""

    job_id: str
    status: JobStatus = "pending"
    created_at: datetime = field(default_factory=_utc_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        """Serialize job state for API responses."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "result": self.result,
            "error": self.error,
        }


class _JobBackend(Protocol):
    def submit_custom_learning(self, payload: Dict[str, Any]) -> JobRecord: ...

    def get(self, job_id: str) -> Optional[JobRecord]: ...

    def cancel(
        self,
        job_id: str,
    ) -> Tuple[bool, Literal["cancel_requested", "canceled", "completed", "failed", "not_found"]]: ...


class _LocalJobBackend:
    """Thread-backed local job execution backend."""

    def __init__(
        self, custom_learning_handler: CustomLearningHandler, max_workers: int, max_retained_jobs: int
    ) -> None:
        self._custom_learning_handler = custom_learning_handler
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="abax-job")
        self._max_retained_jobs = max_retained_jobs
        self._lock = Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._order: List[str] = []
        self._cancel_requested: Dict[str, bool] = {}

    def submit_custom_learning(self, payload: Dict[str, Any]) -> JobRecord:
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id)

        with self._lock:
            self._jobs[job_id] = record
            self._order.append(job_id)
            self._trim_locked()

        self._executor.submit(self._run_job, job_id, payload)
        return record

    def _run_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            if record.status == "canceled":
                record.finished_at = _utc_now()
                return
            record.status = "running"
            record.started_at = _utc_now()

        try:
            result = self._custom_learning_handler(**payload)
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                if self._cancel_requested.get(job_id, False):
                    record.result = None
                    record.status = "canceled"
                    record.finished_at = _utc_now()
                    record.error = "Canceled by user while running."
                    return
                record.result = result
                record.status = "completed"
                record.finished_at = _utc_now()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "failed"
                record.finished_at = _utc_now()
                record.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(
        self,
        job_id: str,
    ) -> Tuple[bool, Literal["cancel_requested", "canceled", "completed", "failed", "not_found"]]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return False, "not_found"

            if record.status == "pending":
                record.status = "canceled"
                record.finished_at = _utc_now()
                record.error = "Canceled by user before execution."
                return True, "canceled"

            if record.status == "running":
                record.status = "cancel_requested"
                self._cancel_requested[job_id] = True
                return True, "cancel_requested"

            if record.status == "cancel_requested":
                return True, "cancel_requested"

            if record.status == "canceled":
                return True, "canceled"

            if record.status == "completed":
                return True, "completed"
            if record.status == "failed":
                return True, "failed"

            return False, "not_found"

    def _trim_locked(self) -> None:
        if len(self._order) <= self._max_retained_jobs:
            return

        removable_count = len(self._order) - self._max_retained_jobs
        removed = 0
        kept_order: List[str] = []

        for job_id in self._order:
            if removed >= removable_count:
                kept_order.append(job_id)
                continue

            record = self._jobs.get(job_id)
            if record is None:
                removed += 1
                continue

            if record.status in {"completed", "failed", "canceled"}:
                self._jobs.pop(job_id, None)
                self._cancel_requested.pop(job_id, None)
                removed += 1
            else:
                kept_order.append(job_id)

        self._order = kept_order


class _CeleryJobBackend:
    """Celery-backed backend using Redis/Rabbit broker and result backend."""

    def __init__(
        self,
        broker_url: str,
        result_backend: str,
        *,
        max_retries: int,
        retry_backoff: bool,
        retry_backoff_max: int,
        soft_time_limit_seconds: int,
        time_limit_seconds: int,
    ) -> None:
        try:
            from celery import Celery  # type: ignore
            from celery.result import AsyncResult  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on runtime extras
            raise RuntimeError(
                "Celery backend requested but 'celery' is not installed. "
                "Install celery and redis extras or use ABAX_ASYNC_JOB_BACKEND=local."
            ) from exc

        self._async_result_cls = AsyncResult
        self._app = Celery("abax-api-client", broker=broker_url, backend=result_backend)
        self._task_name = "src.api.celery_tasks.run_custom_learning_task"
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._retry_backoff_max = retry_backoff_max
        self._soft_time_limit_seconds = soft_time_limit_seconds
        self._time_limit_seconds = time_limit_seconds
        self._lock = Lock()
        self._submitted: Dict[str, JobRecord] = {}

    def submit_custom_learning(self, payload: Dict[str, Any]) -> JobRecord:
        runtime_policy = {
            "max_retries": self._max_retries,
            "retry_backoff": self._retry_backoff,
            "retry_backoff_max": self._retry_backoff_max,
        }
        async_result = self._app.send_task(
            self._task_name,
            kwargs={"payload": payload, "runtime_policy": runtime_policy},
            soft_time_limit=self._soft_time_limit_seconds,
            time_limit=self._time_limit_seconds,
        )
        record = JobRecord(job_id=str(async_result.id), status="pending")
        with self._lock:
            self._submitted[record.job_id] = record
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            seed = self._submitted.get(job_id, JobRecord(job_id=job_id))

        async_result = self._async_result_cls(job_id, app=self._app)
        state = str(async_result.state).upper()
        payload = JobRecord(job_id=job_id, created_at=seed.created_at)

        if state in {"PENDING", "RECEIVED"}:
            payload.status = "pending"
            return payload
        if state in {"STARTED", "RETRY"}:
            payload.status = "running"
            payload.started_at = seed.started_at or _utc_now()
            with self._lock:
                cached = self._submitted.get(job_id)
                if cached is not None and cached.started_at is None:
                    cached.started_at = payload.started_at
            return payload
        if state == "SUCCESS":
            payload.status = "completed"
            payload.started_at = seed.started_at
            payload.finished_at = _utc_now()
            payload.result = async_result.result
            return payload
        if state == "REVOKED":
            payload.status = "canceled"
            payload.started_at = seed.started_at
            payload.finished_at = _utc_now()
            payload.error = "Canceled by user."
            return payload
        if state == "FAILURE":
            payload.status = "failed"
            payload.started_at = seed.started_at
            payload.finished_at = _utc_now()
            payload.error = str(async_result.result)
            return payload

        payload.status = "pending"
        return payload

    def cancel(
        self,
        job_id: str,
    ) -> Tuple[bool, Literal["cancel_requested", "canceled", "completed", "failed", "not_found"]]:
        with self._lock:
            seed = self._submitted.get(job_id)
            if seed is None:
                return False, "not_found"
            if seed.status == "completed":
                return True, "completed"
            if seed.status == "failed":
                return True, "failed"
            seed.status = "cancel_requested"

        self._app.control.revoke(job_id, terminate=True, signal="SIGTERM")
        return True, "cancel_requested"


class JobManager:
    """Facade over local threaded jobs and optional Celery-backed jobs."""

    def __init__(
        self,
        *,
        custom_learning_handler: CustomLearningHandler,
        backend: str = "local",
        max_workers: int = 2,
        max_retained_jobs: int = 300,
        celery_broker_url: str = "redis://redis:6379/0",
        celery_result_backend: str = "redis://redis:6379/1",
        celery_task_max_retries: int = 3,
        celery_retry_backoff: bool = True,
        celery_retry_backoff_max: int = 60,
        celery_soft_time_limit_seconds: int = 240,
        celery_time_limit_seconds: int = 300,
    ) -> None:
        if backend == "celery":
            self._backend: _JobBackend = _CeleryJobBackend(
                broker_url=celery_broker_url,
                result_backend=celery_result_backend,
                max_retries=celery_task_max_retries,
                retry_backoff=celery_retry_backoff,
                retry_backoff_max=celery_retry_backoff_max,
                soft_time_limit_seconds=celery_soft_time_limit_seconds,
                time_limit_seconds=celery_time_limit_seconds,
            )
        else:
            self._backend = _LocalJobBackend(
                custom_learning_handler=custom_learning_handler,
                max_workers=max_workers,
                max_retained_jobs=max_retained_jobs,
            )

    def submit_custom_learning(self, payload: Dict[str, Any]) -> JobRecord:
        """Queue one custom-learning payload asynchronously."""
        return self._backend.submit_custom_learning(payload)

    def get(self, job_id: str) -> Optional[JobRecord]:
        """Get job status/result by id."""
        return self._backend.get(job_id)

    def cancel(
        self,
        job_id: str,
    ) -> Tuple[bool, Literal["cancel_requested", "canceled", "completed", "failed", "not_found"]]:
        """Cancel one job (best effort depending on backend)."""
        return self._backend.cancel(job_id)
