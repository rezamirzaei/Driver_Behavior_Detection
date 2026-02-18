"""Database-backed persistence for training cache, runs, and model artifacts.

Supports SQLite and PostgreSQL via SQLAlchemy, with built-in schema migrations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, RowMapping

MIGRATIONS: Sequence[Tuple[str, str]] = (
    (
        "001_initial",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

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
        );

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
        );

        CREATE INDEX IF NOT EXISTS idx_training_runs_task_started ON training_runs(task, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_training_runs_signature ON training_runs(signature);
        CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
        """,
    ),
    (
        "002_artifacts",
        """
        CREATE TABLE IF NOT EXISTS model_artifacts (
            artifact_key TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            signature TEXT NOT NULL,
            data_version TEXT NOT NULL,
            feature_names_json TEXT NOT NULL,
            reference_stats_json TEXT NOT NULL,
            artifact_file_path TEXT NOT NULL,
            metadata_file_path TEXT NOT NULL,
            result_payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_model_artifacts_task_artifact_id
            ON model_artifacts(task, artifact_id);
        CREATE INDEX IF NOT EXISTS idx_model_artifacts_updated_at ON model_artifacts(updated_at DESC);
        """,
    ),
)


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


def _resolve_database_url(database_url: str, project_root: Optional[Path]) -> str:
    if not database_url.startswith("sqlite:///"):
        return database_url

    sqlite_path = database_url.removeprefix("sqlite:///")
    candidate = Path(sqlite_path)
    if candidate.is_absolute() or project_root is None:
        return database_url
    resolved = (project_root / candidate).resolve()
    return f"sqlite:///{resolved}"


class TrainingRunRepository:
    """Persistent run metadata and signature-based result cache."""

    def __init__(self, database_url: str, *, project_root: Optional[Path] = None) -> None:
        self._lock = Lock()
        self._database_url = _resolve_database_url(database_url, project_root=project_root)
        self._engine = self._build_engine(self._database_url)
        self._migrate()

    @staticmethod
    def _build_engine(database_url: str) -> Engine:
        if database_url.startswith("sqlite:///"):
            sqlite_path = Path(database_url.removeprefix("sqlite:///"))
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            return create_engine(
                database_url,
                future=True,
                pool_pre_ping=True,
                connect_args={"check_same_thread": False},
            )
        return create_engine(database_url, future=True, pool_pre_ping=True)

    @staticmethod
    def _split_sql_script(script: str) -> List[str]:
        statements = [stmt.strip() for stmt in script.split(";")]
        return [stmt for stmt in statements if stmt]

    def _migrate(self) -> None:
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    )
                    """
                )
            )

            rows = conn.execute(text("SELECT version FROM schema_migrations")).mappings().all()
            applied = {str(row["version"]) for row in rows}

            for version, script in MIGRATIONS:
                if version in applied:
                    continue

                for statement in self._split_sql_script(script):
                    conn.exec_driver_sql(statement)

                conn.execute(
                    text("INSERT INTO schema_migrations (version, applied_at) VALUES (:version, :applied_at)"),
                    {"version": version, "applied_at": _utc_now_iso()},
                )

    @staticmethod
    def _row_to_dict(row: RowMapping) -> Dict[str, Any]:
        return {str(key): value for key, value in row.items()}

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
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO training_runs (
                        run_id, signature, task, operation, model_name, feature_names_json,
                        params_json, data_version, status, cache_hit, started_at
                    ) VALUES (
                        :run_id, :signature, :task, :operation, :model_name, :feature_names_json,
                        :params_json, :data_version, :status, :cache_hit, :started_at
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "signature": signature,
                    "task": task,
                    "operation": operation,
                    "model_name": model_name,
                    "feature_names_json": json.dumps(feature_names),
                    "params_json": _json_dump(params),
                    "data_version": data_version,
                    "status": "running",
                    "cache_hit": 0,
                    "started_at": started_at,
                },
            )
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
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE training_runs
                    SET status = :status,
                        cache_hit = :cache_hit,
                        finished_at = :finished_at,
                        duration_ms = :duration_ms,
                        artifact_id = :artifact_id,
                        artifact_path = :artifact_path,
                        metrics_json = :metrics_json,
                        cv_json = :cv_json,
                        result_json = :result_json
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "status": "completed",
                    "cache_hit": 1 if cache_hit else 0,
                    "finished_at": finished_at,
                    "duration_ms": float(duration_ms),
                    "artifact_id": artifact_id,
                    "artifact_path": artifact_path,
                    "metrics_json": _json_dump(metrics),
                    "cv_json": _json_dump(cv_summary),
                    "result_json": _json_dump(result_payload),
                    "run_id": run_id,
                },
            )

    def mark_run_canceled(self, *, run_id: str, duration_ms: float = 0.0, error: str = "") -> None:
        """Mark a run as canceled."""
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE training_runs
                    SET status = :status,
                        finished_at = :finished_at,
                        duration_ms = :duration_ms,
                        error = :error
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "status": "canceled",
                    "finished_at": _utc_now_iso(),
                    "duration_ms": float(duration_ms),
                    "error": error,
                    "run_id": run_id,
                },
            )

    def fail_run(self, *, run_id: str, duration_ms: float, error: str) -> None:
        """Mark a run as failed."""
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE training_runs
                    SET status = :status,
                        finished_at = :finished_at,
                        duration_ms = :duration_ms,
                        error = :error
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "status": "failed",
                    "finished_at": _utc_now_iso(),
                    "duration_ms": float(duration_ms),
                    "error": error,
                    "run_id": run_id,
                },
            )

    def get_cached_result(self, signature: str) -> Optional[Dict[str, Any]]:
        """Return a cached payload by signature if available."""
        with self._lock, self._engine.begin() as conn:
            row = (
                conn.execute(
                    text("SELECT result_json FROM training_cache WHERE signature = :signature"),
                    {"signature": signature},
                )
                .mappings()
                .first()
            )
            if row is None:
                return None

            conn.execute(
                text(
                    """
                    UPDATE training_cache
                    SET hit_count = hit_count + 1, updated_at = :updated_at
                    WHERE signature = :signature
                    """
                ),
                {"updated_at": _utc_now_iso(), "signature": signature},
            )

        return _json_load(row.get("result_json"))

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
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO training_cache (
                        signature, task, operation, model_name, feature_names_json,
                        params_json, data_version, result_json, updated_at, hit_count
                    ) VALUES (
                        :signature, :task, :operation, :model_name, :feature_names_json,
                        :params_json, :data_version, :result_json, :updated_at, 0
                    )
                    ON CONFLICT(signature) DO UPDATE SET
                        task = excluded.task,
                        operation = excluded.operation,
                        model_name = excluded.model_name,
                        feature_names_json = excluded.feature_names_json,
                        params_json = excluded.params_json,
                        data_version = excluded.data_version,
                        result_json = excluded.result_json,
                        updated_at = excluded.updated_at
                    """
                ),
                {
                    "signature": signature,
                    "task": task,
                    "operation": operation,
                    "model_name": model_name,
                    "feature_names_json": json.dumps(feature_names),
                    "params_json": _json_dump(params),
                    "data_version": data_version,
                    "result_json": _json_dump(result_payload),
                    "updated_at": updated_at,
                },
            )

    def upsert_model_artifact(
        self,
        *,
        task: str,
        artifact_id: str,
        model_name: str,
        signature: str,
        data_version: str,
        feature_names: List[str],
        reference_stats: Dict[str, Any],
        artifact_file_path: str,
        metadata_file_path: str,
        result_payload: Dict[str, Any],
    ) -> None:
        """Persist artifact metadata row."""
        now = _utc_now_iso()
        artifact_key = f"{task}:{artifact_id}"
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO model_artifacts (
                        artifact_key, task, artifact_id, model_name, signature, data_version,
                        feature_names_json, reference_stats_json, artifact_file_path, metadata_file_path,
                        result_payload_json, created_at, updated_at, is_active
                    ) VALUES (
                        :artifact_key, :task, :artifact_id, :model_name, :signature, :data_version,
                        :feature_names_json, :reference_stats_json, :artifact_file_path, :metadata_file_path,
                        :result_payload_json, :created_at, :updated_at, 1
                    )
                    ON CONFLICT(artifact_key) DO UPDATE SET
                        model_name = excluded.model_name,
                        signature = excluded.signature,
                        data_version = excluded.data_version,
                        feature_names_json = excluded.feature_names_json,
                        reference_stats_json = excluded.reference_stats_json,
                        artifact_file_path = excluded.artifact_file_path,
                        metadata_file_path = excluded.metadata_file_path,
                        result_payload_json = excluded.result_payload_json,
                        updated_at = excluded.updated_at,
                        is_active = 1
                    """
                ),
                {
                    "artifact_key": artifact_key,
                    "task": task,
                    "artifact_id": artifact_id,
                    "model_name": model_name,
                    "signature": signature,
                    "data_version": data_version,
                    "feature_names_json": json.dumps(feature_names),
                    "reference_stats_json": _json_dump(reference_stats),
                    "artifact_file_path": artifact_file_path,
                    "metadata_file_path": metadata_file_path,
                    "result_payload_json": _json_dump(result_payload),
                    "created_at": now,
                    "updated_at": now,
                },
            )

    def get_model_artifact(self, *, task: str, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one active artifact metadata row."""
        artifact_key = f"{task}:{artifact_id}"
        with self._lock, self._engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        """
                    SELECT artifact_key, task, artifact_id, model_name, signature, data_version,
                           feature_names_json, reference_stats_json, artifact_file_path,
                           metadata_file_path, result_payload_json, created_at, updated_at
                    FROM model_artifacts
                    WHERE artifact_key = :artifact_key AND is_active = 1
                    """
                    ),
                    {"artifact_key": artifact_key},
                )
                .mappings()
                .first()
            )

        if row is None:
            return None

        payload = self._row_to_dict(row)
        payload["feature_names"] = json.loads(str(payload.pop("feature_names_json", "[]")))
        payload["reference_stats"] = _json_load(payload.pop("reference_stats_json", ""))
        payload["result_payload"] = _json_load(payload.pop("result_payload_json", ""))
        return payload

    def list_model_artifacts(self, *, task: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """List active artifacts sorted by newest update."""
        query = """
            SELECT task, artifact_id, model_name, signature, data_version, updated_at,
                   artifact_file_path, metadata_file_path
            FROM model_artifacts
            WHERE is_active = 1
        """
        params: Dict[str, Any] = {}
        if task:
            query += " AND task = :task"
            params["task"] = task
        query += " ORDER BY updated_at DESC LIMIT :limit"
        params["limit"] = int(limit)

        with self._lock, self._engine.begin() as conn:
            rows = conn.execute(text(query), params).mappings().all()

        return [self._row_to_dict(row) for row in rows]

    def list_runs(self, *, task: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """List most recent runs with compact metadata."""
        query = """
            SELECT run_id, signature, task, operation, model_name, status, cache_hit,
                   started_at, finished_at, duration_ms, artifact_id, artifact_path,
                   feature_names_json, metrics_json, cv_json, error
            FROM training_runs
        """
        params: Dict[str, Any] = {}
        if task:
            query += " WHERE task = :task"
            params["task"] = task
        query += " ORDER BY started_at DESC LIMIT :limit"
        params["limit"] = int(limit)

        with self._lock, self._engine.begin() as conn:
            rows = conn.execute(text(query), params).mappings().all()

        payloads: List[Dict[str, Any]] = []
        for row in rows:
            item = self._row_to_dict(row)
            feature_names = json.loads(str(item.get("feature_names_json") or "[]"))
            payloads.append(
                {
                    "run_id": str(item["run_id"]),
                    "signature": str(item["signature"]),
                    "task": str(item["task"]),
                    "operation": str(item["operation"]),
                    "model_name": str(item["model_name"]),
                    "status": str(item["status"]),
                    "cache_hit": bool(item["cache_hit"]),
                    "started_at": str(item["started_at"]),
                    "finished_at": str(item["finished_at"]) if item.get("finished_at") else None,
                    "duration_ms": float(item["duration_ms"]) if item.get("duration_ms") is not None else None,
                    "artifact_id": str(item["artifact_id"]) if item.get("artifact_id") else "",
                    "artifact_path": str(item["artifact_path"]) if item.get("artifact_path") else "",
                    "feature_count": len(feature_names),
                    "metrics": _json_load(item.get("metrics_json")),
                    "cv_summary": _json_load(item.get("cv_json")),
                    "error": str(item["error"]) if item.get("error") else None,
                }
            )
        return payloads

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one run with full payload."""
        with self._lock, self._engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        """
                    SELECT run_id, signature, task, operation, model_name, status, cache_hit,
                           started_at, finished_at, duration_ms, artifact_id, artifact_path,
                           feature_names_json, params_json, data_version, metrics_json,
                           cv_json, error, result_json
                    FROM training_runs
                    WHERE run_id = :run_id
                    """
                    ),
                    {"run_id": run_id},
                )
                .mappings()
                .first()
            )

        if row is None:
            return None

        item = self._row_to_dict(row)
        return {
            "run_id": str(item["run_id"]),
            "signature": str(item["signature"]),
            "task": str(item["task"]),
            "operation": str(item["operation"]),
            "model_name": str(item["model_name"]),
            "status": str(item["status"]),
            "cache_hit": bool(item["cache_hit"]),
            "started_at": str(item["started_at"]),
            "finished_at": str(item["finished_at"]) if item.get("finished_at") else None,
            "duration_ms": float(item["duration_ms"]) if item.get("duration_ms") is not None else None,
            "artifact_id": str(item["artifact_id"]) if item.get("artifact_id") else "",
            "artifact_path": str(item["artifact_path"]) if item.get("artifact_path") else "",
            "feature_names": json.loads(str(item.get("feature_names_json") or "[]")),
            "params": _json_load(item.get("params_json")),
            "data_version": str(item["data_version"]),
            "metrics": _json_load(item.get("metrics_json")),
            "cv_summary": _json_load(item.get("cv_json")),
            "error": str(item["error"]) if item.get("error") else None,
            "result": _json_load(item.get("result_json")),
        }
