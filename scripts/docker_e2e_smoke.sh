#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

cleanup() {
  docker compose down --remove-orphans
}
trap cleanup EXIT

docker compose up -d --build

echo "Waiting for API health..."
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null; then
    echo "API is healthy."
    break
  fi
  sleep 2
done

python - <<'PY'
import json
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"


def get_json(path: str, params: dict | None = None) -> dict:
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{BASE}{path}{query}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))


_ = get_json("/api/health")
meta = get_json("/api/metadata", {"task": "classification"})
features = [item["name"] for item in meta.get("features", [])][:2]
models = meta.get("models", [])
if len(features) < 2 or not models:
    raise RuntimeError("Metadata does not include enough features/models for e2e smoke test.")

job = post_json(
    "/api/model/custom-learning/job",
    {
        "task": "classification",
        "model_name": models[0],
        "feature_names": features,
        "cv_folds": 1,
        "persist_artifact": True,
    },
)
job_id = job["job_id"]

status_payload = None
for _ in range(180):
    status_payload = get_json(f"/api/jobs/{job_id}")
    if status_payload["status"] in {"completed", "failed", "canceled"}:
        break
    time.sleep(1)

if status_payload is None:
    raise RuntimeError("No job status received.")
if status_payload["status"] != "completed":
    raise RuntimeError(f"Job did not complete successfully: {status_payload}")

result = status_payload.get("result", {})
artifact_id = result.get("artifact_id")
if not artifact_id:
    raise RuntimeError("Custom learning result did not return artifact_id.")

artifacts = get_json("/api/artifacts", {"task": "classification", "limit": 10})
if not artifacts.get("artifacts"):
    raise RuntimeError("No artifacts listed after persisted run.")

sample_record = {features[0]: 0.0, features[1]: 0.0}
pred = post_json(f"/api/artifacts/classification/{artifact_id}/predict", {"records": [sample_record]})
if pred.get("n_records") != 1:
    raise RuntimeError(f"Unexpected prediction payload: {pred}")

drift = post_json(f"/api/artifacts/classification/{artifact_id}/drift", {"records": [sample_record]})
if "overall_drift_score" not in drift:
    raise RuntimeError(f"Unexpected drift payload: {drift}")

runs = get_json("/api/training-runs", {"task": "classification"})
if not runs.get("runs"):
    raise RuntimeError("Training runs endpoint returned empty history.")

metrics = get_json("/api/observability/metrics")
if "requests" not in metrics:
    raise RuntimeError("Observability endpoint missing requests payload.")

print(json.dumps({"status": "ok", "artifact_id": artifact_id}))
PY
