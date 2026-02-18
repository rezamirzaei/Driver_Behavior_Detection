"""Database-backed persistence for training cache, runs, artifacts, and drift alerts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, RowMapping

from src.api.db_migrations import apply_migrations


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
        self._project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        self._database_url = _resolve_database_url(database_url, project_root=self._project_root)
        apply_migrations(self._database_url, project_root=self._project_root)
        self._engine = self._build_engine(self._database_url)

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

    def create_drift_alert(
        self,
        *,
        task: str,
        artifact_id: str,
        overall_drift_score: float,
        flagged_feature_count: int,
        payload: Dict[str, Any],
    ) -> str:
        """Persist one drift alert and return its identifier."""
        alert_id = str(uuid.uuid4())
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO drift_alerts (
                        alert_id, task, artifact_id, overall_drift_score, flagged_feature_count,
                        payload_json, status, detected_at
                    ) VALUES (
                        :alert_id, :task, :artifact_id, :overall_drift_score, :flagged_feature_count,
                        :payload_json, :status, :detected_at
                    )
                    """
                ),
                {
                    "alert_id": alert_id,
                    "task": task,
                    "artifact_id": artifact_id,
                    "overall_drift_score": float(overall_drift_score),
                    "flagged_feature_count": int(flagged_feature_count),
                    "payload_json": _json_dump(payload),
                    "status": "open",
                    "detected_at": _utc_now_iso(),
                },
            )
        return alert_id

    def get_latest_drift_alert(self, *, task: str, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Return latest drift alert for one task/artifact."""
        with self._lock, self._engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        """
                        SELECT alert_id, task, artifact_id, overall_drift_score, flagged_feature_count,
                               payload_json, status, detected_at, acknowledged_at, acknowledged_by
                        FROM drift_alerts
                        WHERE task = :task AND artifact_id = :artifact_id
                        ORDER BY detected_at DESC
                        LIMIT 1
                        """
                    ),
                    {"task": task, "artifact_id": artifact_id},
                )
                .mappings()
                .first()
            )

        if row is None:
            return None
        payload = self._row_to_dict(row)
        payload["payload"] = _json_load(payload.pop("payload_json", ""))
        return payload

    def list_drift_alerts(
        self,
        *,
        task: Optional[str],
        artifact_id: Optional[str],
        limit: int,
        status: Optional[str],
    ) -> List[Dict[str, Any]]:
        """List drift alerts, newest first."""
        query = """
            SELECT alert_id, task, artifact_id, overall_drift_score, flagged_feature_count,
                   payload_json, status, detected_at, acknowledged_at, acknowledged_by
            FROM drift_alerts
            WHERE 1 = 1
        """
        params: Dict[str, Any] = {"limit": int(limit)}
        if task:
            query += " AND task = :task"
            params["task"] = task
        if artifact_id:
            query += " AND artifact_id = :artifact_id"
            params["artifact_id"] = artifact_id
        if status:
            query += " AND status = :status"
            params["status"] = status
        query += " ORDER BY detected_at DESC LIMIT :limit"

        with self._lock, self._engine.begin() as conn:
            rows = conn.execute(text(query), params).mappings().all()

        payloads: List[Dict[str, Any]] = []
        for row in rows:
            item = self._row_to_dict(row)
            item["payload"] = _json_load(item.pop("payload_json", ""))
            payloads.append(item)
        return payloads

    def acknowledge_drift_alert(self, *, alert_id: str, acknowledged_by: str) -> bool:
        """Mark one drift alert as acknowledged. Returns False when not found."""
        with self._lock, self._engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE drift_alerts
                    SET status = :status,
                        acknowledged_at = :acknowledged_at,
                        acknowledged_by = :acknowledged_by
                    WHERE alert_id = :alert_id
                    """
                ),
                {
                    "status": "acknowledged",
                    "acknowledged_at": _utc_now_iso(),
                    "acknowledged_by": acknowledged_by,
                    "alert_id": alert_id,
                },
            )
        return bool(result.rowcount)
