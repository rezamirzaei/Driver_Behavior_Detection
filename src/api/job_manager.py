"""In-memory background job manager for long-running API tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
import traceback
from typing import Any, Callable, Dict, List, Literal, Optional
import uuid

JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class JobRecord:
    """Runtime state for one asynchronous job."""

    job_id: str
    status: JobStatus = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
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


class JobManager:
    """Thread-backed background job execution with polling support."""

    def __init__(self, max_workers: int = 2, max_retained_jobs: int = 300) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="abax-job")
        self._max_retained_jobs = max_retained_jobs
        self._lock = Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._order: List[str] = []

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> JobRecord:
        """Queue a job for asynchronous execution."""
        job_id = str(uuid.uuid4())
        record = JobRecord(job_id=job_id)

        with self._lock:
            self._jobs[job_id] = record
            self._order.append(job_id)
            self._trim_locked()

        self._executor.submit(self._run_job, job_id, fn, *args, **kwargs)
        return record

    def _run_job(self, job_id: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "running"
            record.started_at = datetime.now(tz=timezone.utc)

        try:
            result = fn(*args, **kwargs)
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.result = result
                record.status = "completed"
                record.finished_at = datetime.now(tz=timezone.utc)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "failed"
                record.finished_at = datetime.now(tz=timezone.utc)
                record.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    def get(self, job_id: str) -> Optional[JobRecord]:
        """Fetch one job record by id."""
        with self._lock:
            return self._jobs.get(job_id)

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

            if record.status in {"completed", "failed"}:
                self._jobs.pop(job_id, None)
                removed += 1
            else:
                kept_order.append(job_id)

        self._order = kept_order
