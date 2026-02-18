"""Containerized end-to-end smoke test wrapper.

This test is opt-in to keep local/unit test loops fast.
Enable with: ABAX_RUN_DOCKER_E2E=1 pytest -q tests/test_containerized_e2e.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.integration
def test_docker_compose_e2e_smoke() -> None:
    if os.getenv("ABAX_RUN_DOCKER_E2E") != "1":
        pytest.skip("Set ABAX_RUN_DOCKER_E2E=1 to run containerized e2e smoke test.")
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not available.")

    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "docker_e2e_smoke.sh"
    subprocess.run(["bash", str(script_path)], cwd=str(root), check=True)
