"""Initial runtime schema for cache, runs, artifacts, and drift alerts.

Revision ID: 20260218_0001
Revises:
Create Date: 2026-02-18
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260218_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
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
    op.execute(
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
    op.execute(
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
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_alerts (
            alert_id TEXT PRIMARY KEY,
            task TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            overall_drift_score REAL NOT NULL,
            flagged_feature_count INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            acknowledged_at TEXT,
            acknowledged_by TEXT
        )
        """
    )

    op.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_task_started ON training_runs(task, started_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_signature ON training_runs(signature)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status)")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_model_artifacts_task_artifact_id "
        "ON model_artifacts(task, artifact_id)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_artifacts_updated_at ON model_artifacts(updated_at DESC)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_drift_alerts_task_artifact_detected "
        "ON drift_alerts(task, artifact_id, detected_at DESC)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_drift_alerts_status_detected ON drift_alerts(status, detected_at DESC)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_drift_alerts_status_detected")
    op.execute("DROP INDEX IF EXISTS idx_drift_alerts_task_artifact_detected")
    op.execute("DROP INDEX IF EXISTS idx_model_artifacts_updated_at")
    op.execute("DROP INDEX IF EXISTS idx_model_artifacts_task_artifact_id")
    op.execute("DROP INDEX IF EXISTS idx_training_runs_status")
    op.execute("DROP INDEX IF EXISTS idx_training_runs_signature")
    op.execute("DROP INDEX IF EXISTS idx_training_runs_task_started")

    op.execute("DROP TABLE IF EXISTS drift_alerts")
    op.execute("DROP TABLE IF EXISTS model_artifacts")
    op.execute("DROP TABLE IF EXISTS training_runs")
    op.execute("DROP TABLE IF EXISTS training_cache")
