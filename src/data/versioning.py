"""Lightweight dataset version tracking utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def compute_tree_version(root: Path) -> str:
    """Compute a stable version hash for a directory tree."""
    digest = hashlib.sha256()
    digest.update(str(root.resolve()).encode("utf-8"))

    for file_path in _iter_files(root):
        rel = file_path.relative_to(root)
        stat = file_path.stat()
        digest.update(str(rel).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime)).encode("utf-8"))

    return digest.hexdigest()


def build_data_manifest(roots: Mapping[str, Path]) -> Dict[str, str]:
    """Build dataset-name -> version-hash mapping."""
    manifest: Dict[str, str] = {}
    for name, root in roots.items():
        manifest[name] = compute_tree_version(root) if root.exists() else "missing"
    return manifest


def write_data_manifest(roots: Mapping[str, Path], output_path: Path) -> Dict[str, str]:
    """Persist data manifest to disk and return the computed versions."""
    versions = build_data_manifest(roots)
    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "versions": versions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return versions


def read_data_manifest(path: Path) -> Optional[Dict[str, str]]:
    """Read persisted data manifest if available."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    versions = payload.get("versions")
    if not isinstance(versions, dict):
        return None
    return {str(key): str(value) for key, value in versions.items()}
