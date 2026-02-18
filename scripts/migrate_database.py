"""Apply DB migrations for run/cache/artifact storage."""

from __future__ import annotations

from pathlib import Path
import sys

# Allow running this script directly from repository root without package install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.config import get_settings
from src.api.run_repository import TrainingRunRepository


def main() -> None:
    settings = get_settings()
    _ = TrainingRunRepository(settings.database_url, project_root=settings.resolve_path("."))
    print(f"Applied migrations successfully for database URL: {settings.database_url}")


if __name__ == "__main__":
    main()
