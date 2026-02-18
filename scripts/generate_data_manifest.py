"""Generate lightweight dataset version manifest for reproducibility."""

from __future__ import annotations

from pathlib import Path

from src.data.versioning import write_data_manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "data" / "processed" / "data_manifest.json"
    roots = {
        "uah_raw": project_root / "data" / "UAH-DRIVESET-v1",
        "processed": project_root / "data" / "processed",
    }
    versions = write_data_manifest(roots, manifest_path)
    print(f"Wrote data manifest to {manifest_path}")
    for name, version in versions.items():
        print(f"  {name}: {version}")


if __name__ == "__main__":
    main()
