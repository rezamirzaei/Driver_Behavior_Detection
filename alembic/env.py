"""Alembic environment for ABAX persistence schema."""

from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Raw SQL migrations are used; metadata autogenerate is intentionally disabled.
target_metadata = None


def _normalize_sqlite_url(url: str) -> str:
    if not url.startswith("sqlite:///"):
        return url
    raw = url.removeprefix("sqlite:///")
    candidate = Path(raw)
    if candidate.is_absolute():
        return url
    root = Path(config.get_main_option("project_root") or Path(__file__).resolve().parents[1])
    return f"sqlite:///{(root / candidate).resolve()}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = _normalize_sqlite_url(config.get_main_option("sqlalchemy.url"))
    context.configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"})

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _normalize_sqlite_url(section["sqlalchemy.url"])

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
