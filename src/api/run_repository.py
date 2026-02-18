"""SQLite-backed persistence for training cache and run history."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any, Dict, List, Optional
import uuid


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _json_dump(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _json_load(payload: Optional[str]) -> Dict[str, Any]:
    if not payload:
        return {}
    loaded = json.loads(payload)
    if isinstance(loaded, dict):
        return loaded
    return {}


class TrainingRunRepository:
    """Persistent run metadata and signature-based result cache."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_cache (
                    signature TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    feature_names_json TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    data_version TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    task TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    feature_names_json TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    data_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms REAL,
                    artifact_id TEXT,
                    artifact_path TEXT,
                    metrics_json TEXT,
                    cv_json TEXT,
                    error TEXT,
                    result_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_runs_task_started ON training_runs(task, started_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_runs_signature ON training_runs(signature)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)"
            )
            conn.commit()

    def start_run(
        self,
        *,
        signature: str,
        task: str,
        operation: str,
        model_name: str,
        feature_names: List[str],
        params: Dict[str, Any],
        data_version: str,
    ) -> str:
        """Insert a new running record and return run id."""
        run_id = str(uuid.uuid4())
        started_at = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_runs (
                    run_id, signature, task, operation, model_name, feature_names_json,
                    params_json, data_version, status, cache_hit, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    signature,
                    task,
                    operation,
                    model_name,
                    json.dumps(feature_names),
                    _json_dump(params),
                    data_version,
                    "running",
                    0,
                    started_at,
                ),
            )
            conn.commit()
        return run_id

    def complete_run(
        self,
        *,
        run_id: str,
        cache_hit: bool,
        duration_ms: float,
        metrics: Dict[str, Any],
        cv_summary: Dict[str, Any],
        artifact_id: str,
        artifact_path: str,
        result_payload: Dict[str, Any],
    ) -> None:
        """Mark a run as completed and persist result summary/payload."""
        finished_at = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET status = ?, cache_hit = ?, finished_at = ?, duration_ms = ?,
                    artifact_id = ?, artifact_path = ?, metrics_json = ?, cv_json = ?, result_json = ?
                WHERE run_id = ?
                """,
                (
                    "completed",
                    1 if cache_hit else 0,
                    finished_at,
                    float(duration_ms),
                    artifact_id,
                    artifact_path,
                    _json_dump(metrics),
                    _json_dump(cv_summary),
                    _json_dump(result_payload),
                    run_id,
                ),
            )
            conn.commit()

    def fail_run(self, *, run_id: str, duration_ms: float, error: str) -> None:
        """Mark a run as failed."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE training_runs
                SET status = ?, finished_at = ?, duration_ms = ?, error = ?
                WHERE run_id = ?
                """,
                ("failed", _utc_now_iso(), float(duration_ms), error, run_id),
            )
            conn.commit()

    def get_cached_result(self, signature: str) -> Optional[Dict[str, Any]]:
        """Return a cached payload by signature if available."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM training_cache WHERE signature = ?",
                (signature,),
            ).fetchone()
            if row is None:
                return None

            conn.execute(
                "UPDATE training_cache SET hit_count = hit_count + 1, updated_at = ? WHERE signature = ?",
                (_utc_now_iso(), signature),
            )
            conn.commit()

        return _json_load(row["result_json"])

    def set_cached_result(
        self,
        *,
        signature: str,
        task: str,
        operation: str,
        model_name: str,
        feature_names: List[str],
        params: Dict[str, Any],
        data_version: str,
        result_payload: Dict[str, Any],
    ) -> None:
        """Persist or update a cached payload by signature."""
        updated_at = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_cache (
                    signature, task, operation, model_name, feature_names_json,
                    params_json, data_version, result_json, updated_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(signature) DO UPDATE SET
                    task = excluded.task,
                    operation = excluded.operation,
                    model_name = excluded.model_name,
                    feature_names_json = excluded.feature_names_json,
                    params_json = excluded.params_json,
                    data_version = excluded.data_version,
                    result_json = excluded.result_json,
                    updated_at = excluded.updated_at
                """,
                (
                    signature,
                    task,
                    operation,
                    model_name,
                    json.dumps(feature_names),
                    _json_dump(params),
                    data_version,
                    _json_dump(result_payload),
                    updated_at,
                ),
            )
            conn.commit()

    def list_runs(self, *, task: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """List most recent runs with compact metadata."""
        query = """
            SELECT run_id, signature, task, operation, model_name, status, cache_hit,
                   started_at, finished_at, duration_ms, artifact_id, artifact_path,
                   feature_names_json, metrics_json, cv_json, error
            FROM training_runs
        """
        params: List[Any] = []
        if task:
            query += " WHERE task = ?"
            params.append(task)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "run_id": str(row["run_id"]),
                "signature": str(row["signature"]),
                "task": str(row["task"]),
                "operation": str(row["operation"]),
                "model_name": str(row["model_name"]),
                "status": str(row["status"]),
                "cache_hit": bool(row["cache_hit"]),
                "started_at": str(row["started_at"]),
                "finished_at": str(row["finished_at"]) if row["finished_at"] else None,
                "duration_ms": float(row["duration_ms"]) if row["duration_ms"] is not None else None,
                "artifact_id": str(row["artifact_id"]) if row["artifact_id"] else "",
                "artifact_path": str(row["artifact_path"]) if row["artifact_path"] else "",
                "feature_count": len(json.loads(str(row["feature_names_json"]) or "[]")),
                "metrics": _json_load(row["metrics_json"]),
                "cv_summary": _json_load(row["cv_json"]),
                "error": str(row["error"]) if row["error"] else None,
            }
            for row in rows
        ]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one run with full payload."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, signature, task, operation, model_name, status, cache_hit,
                       started_at, finished_at, duration_ms, artifact_id, artifact_path,
                       feature_names_json, params_json, data_version, metrics_json,
                       cv_json, error, result_json
                FROM training_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "run_id": str(row["run_id"]),
            "signature": str(row["signature"]),
            "task": str(row["task"]),
            "operation": str(row["operation"]),
            "model_name": str(row["model_name"]),
            "status": str(row["status"]),
            "cache_hit": bool(row["cache_hit"]),
            "started_at": str(row["started_at"]),
            "finished_at": str(row["finished_at"]) if row["finished_at"] else None,
            "duration_ms": float(row["duration_ms"]) if row["duration_ms"] is not None else None,
            "artifact_id": str(row["artifact_id"]) if row["artifact_id"] else "",
            "artifact_path": str(row["artifact_path"]) if row["artifact_path"] else "",
            "feature_names": json.loads(str(row["feature_names_json"]) or "[]"),
            "params": _json_load(row["params_json"]),
            "data_version": str(row["data_version"]),
            "metrics": _json_load(row["metrics_json"]),
            "cv_summary": _json_load(row["cv_json"]),
            "error": str(row["error"]) if row["error"] else None,
            "result": _json_load(row["result_json"]),
        }
