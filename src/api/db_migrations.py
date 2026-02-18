"""Alembic migration helpers for runtime database schema."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from alembic.config import Config

from alembic import command

_lock = Lock()
_applied: set[str] = set()


def apply_migrations(database_url: str, *, project_root: Path) -> None:
    """Apply Alembic migrations once per database URL in this process."""
    cache_key = f"{project_root.resolve()}::{database_url}"

    with _lock:
        if cache_key in _applied:
            return

        config_path = project_root / "alembic.ini"
        script_path = project_root / "alembic"
        if not config_path.exists() or not script_path.exists():
            raise RuntimeError(f"Alembic configuration not found. Expected '{config_path}' and '{script_path}'.")

        alembic_cfg = Config(str(config_path))
        alembic_cfg.set_main_option("script_location", str(script_path))
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        alembic_cfg.set_main_option("project_root", str(project_root))

        command.upgrade(alembic_cfg, "head")
        _applied.add(cache_key)
