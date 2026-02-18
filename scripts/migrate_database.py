"""Apply DB migrations for run/cache/artifact storage."""

from __future__ import annotations

from pathlib import Path
import sys

# Allow running this script directly from repository root without package install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.config import get_settings
from src.api.db_migrations import apply_migrations
from src.api.run_repository import _resolve_database_url


def main() -> None:
    settings = get_settings()
    database_url = _resolve_database_url(settings.database_url, project_root=PROJECT_ROOT)
    apply_migrations(database_url=database_url, project_root=PROJECT_ROOT)
    print(f"Applied Alembic migrations successfully for database URL: {database_url}")


if __name__ == "__main__":
    main()
