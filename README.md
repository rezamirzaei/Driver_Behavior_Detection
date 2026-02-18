# ABAX Data Science Technical Assessment

![CI](https://github.com/rezamirzaei/Driver_Behavior_Detection/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
  <strong>Driver Behavior Classification & Fuel Economy Prediction</strong><br>
  <em>Complete ML pipeline with FastAPI + AngularJS Learning Studio, async jobs, and persistent run history</em>
</p>

---

**Author:** Reza Mirzaeifard
**Email:** [reza.mirzaeifard@gmail.com](mailto:reza.mirzaeifard@gmail.com)
**Date:** December 2025

---

## 📄 Main Deliverable

> **📕 [ABAX Technical Report (PDF)](docs/ABAX_Technical_Report.pdf)**
>
> A comprehensive 20+ page LaTeX report covering:
> - Complete exploratory data analysis with visualizations
> - Feature engineering from raw sensor data (24 features)
> - 18 classification models + 13 regression models comparison
> - Advanced regularization (MCP, SCAD penalties)
> - MLP neural network with proper data normalization
> - Driver-level evaluation (D6 held out for testing)
> - Failure analysis with mitigation strategies
> - Production deployment recommendations

---

## 🎯 Project Summary

This project demonstrates end-to-end machine learning workflows for two telematics applications critical to ABAX's business:

### Task 1: Driver Behavior Classification
- **Dataset:** UAH-DriveSet (40 trips, 6 drivers)
- **Goal:** Classify driving as NORMAL, DROWSY, or AGGRESSIVE
- **Best Result:** **100% accuracy** with Gradient Boosting (87.5% with Random Forest, KNN)
- **Key Innovation:** Raw sensor features only (no circular logic from pre-computed scores)
- **Data Preprocessing:** NORMAL1/NORMAL2 labels normalized to single NORMAL class

### Task 2: Fuel Economy Prediction
- **Dataset:** EPA Fuel Economy (~5,000 vehicles)
- **Goal:** Predict combined MPG from vehicle specifications
- **Best Result:** **R² = 0.94, RMSE = 4.5 MPG** with Random Forest

---

## 🆕 New Developments

- **Strict data validation with Pydantic** across ingestion, feature extraction, API contracts, and runtime payloads.
- **Learning Studio upgrade** with feature-subset training, full model list exposure, cross-validation mode, and train/validation diagnostics.
- **Train/validation diagnostics in UI**:
  - Classification: train + validation confusion matrices and per-iteration error curves.
  - Regression: validation residual diagnostics and per-iteration error curves.
- **Feature intelligence in UI**: per-feature explanation, source type (`raw` vs `processed`), and lineage text.
- **Signature-based training cache** for repeated runs (`task + model + sorted_features + params + data_version`).
- **Persistent run history** in SQLite with run metadata, cache-hit tracking, artifact info, and payload retrieval.
- **Database-backed storage with migrations** for SQLite/PostgreSQL, auto-applied on service startup.
- **Alembic-based schema migrations** (standard migration versioning for long-term operations).
- **Async training jobs** with pluggable backend:
  - Local threaded job manager (default lightweight mode).
  - Celery + Redis mode (docker compose stack).
  - Retry/backoff policies and job cancellation endpoint.
- **Full artifact lifecycle**:
  - Persist trained model + preprocessing bundle (`joblib`)
  - Serve predictions from artifact endpoints
  - Run drift checks against artifact training references
- **Drift alert automation**:
  - Persist drift alerts with score/feature context
  - Optional webhook dispatch for external incident workflows
  - Alert listing + acknowledge endpoints for operations
- **API hardening controls**:
  - API-key auth middleware (`X-API-Key`)
  - Per-key request quotas (sliding 1-minute window)
- **Parquet-first dataset caching** with metadata sidecars (`.meta.json`) for schema/version/column tracking.
- **Data version manifest support** and API exposure for reproducibility checks.
- **Observability endpoint** for request and training timing metrics.
- **Quality gates** expanded with pre-commit + CI checks (`ruff`, `mypy`, `pytest`, docker build).

---

## 🏆 Key Achievements

| Achievement | Description |
|-------------|-------------|
| **Raw Sensor Features** | Extracted 24 features from GPS/accelerometer, avoiding circular logic |
| **Kalman Filtering** | 1D/2D Kalman filter for signal smoothing and noise reduction |
| **Driver-Level Splitting** | D6 completely held out—tests generalization to new customers |
| **18 Classification Models** | Including MCP, SCAD, MLP, SVM, Random Forest, KNN |
| **Advanced Regularization** | Implemented MCP and SCAD for nearly unbiased sparse estimates |
| **MLP Neural Network** | Multi-Layer Perceptron with proper StandardScaler normalization |
| **Persistent Run Cache** | Signature-based cache + SQLite run history for repeatable UI runs |
| **Async Training Jobs** | Background training with local backend and optional Celery/Redis |
| **Observability** | Request/training timing metrics exposed from API |
| **Clean Code Architecture** | Modular `src/` package with testable, reusable functions |
| **Comprehensive Analysis** | Feature importance, failure cases, production recommendations |

---

## 📊 Results Summary

### Classification Results (D6 Held Out)

| Model | Train Acc | Test Acc | F1-Score | Overfit Gap |
|-------|-----------|----------|----------|-------------|
| **Gradient Boosting** | 100% | **100%** | 1.000 | **0.000** |
| KNN (k=7) | 100% | 87.5% | 0.863 | 0.125 |
| Random Forest | 100% | 87.5% | 0.875 | 0.125 |
| Extra Trees | 100% | 87.5% | 0.875 | 0.125 |
| AdaBoost | 100% | 87.5% | 0.863 | 0.125 |
| Logistic (L1) | 84.4% | 75.0% | 0.767 | 0.094 |
| Logistic (SCAD) | 75.0% | 75.0% | 0.767 | **0.000** |
| MLP Neural Network | 87.5% | 62.5% | 0.630 | 0.250 |

**Key Finding:** Gradient Boosting achieves 100% accuracy; ensemble methods outperform on this dataset with good feature engineering.

### Regression Results

| Model | R² | RMSE (MPG) | MAE (MPG) |
|-------|-----|------------|-----------|
| **Random Forest** | **0.938** | 4.52 | 2.31 |
| Gradient Boosting | 0.932 | 4.70 | 2.58 |
| Ridge (L2) | 0.802 | 8.05 | 4.47 |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync
# Optional deep-learning extras (PyTorch)
uv sync --extra deep-learning

# Or using pip
pip install -e .
# Optional deep-learning extras (PyTorch)
pip install -e ".[deep-learning]"
```

### 2. Run Notebooks

```bash
jupyter lab notebooks/
```

**Notebooks:**
- `01_project_overview.ipynb` - Project introduction and data overview
- `02_classification.ipynb` - Complete classification pipeline (800+ lines)
- `03_eda_regression.ipynb` - Regression EDA
- `04_regression.ipynb` - Complete regression pipeline

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Run Quality Checks

```bash
# One-time setup per clone: install git hooks
pre-commit install

ruff check .
mypy src tests
pytest -q
pre-commit run --all-files
```

### 5. Run API + AngularJS Dashboard (Local)

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

This uses the local in-process async backend.
Open `http://localhost:8000` for the dashboard.

### 6. Run with Docker (API + Redis + Celery Worker)

```bash
docker compose up --build
```

This starts API + UI + Redis + Celery worker (async training jobs).
Open `http://localhost:8000` for the dashboard.

Torch is optional in container images for faster/smaller builds.
To include CPU-only Torch in Docker image:
```bash
ABAX_INSTALL_TORCH=true docker compose build
```

### 7. Apply Database Migrations (Manual)

```bash
python scripts/migrate_database.py
```

You can switch to PostgreSQL by setting:
```bash
export ABAX_DATABASE_URL="postgresql+psycopg://abax:abax@localhost:5432/abax"
```

### 8. Compile LaTeX Report

```bash
cd docs && bash compile_report.sh
```

---

## 🧩 API Endpoints

Task-aware endpoints (`task=classification|regression`):

- `GET /api/health` - service readiness for both tasks
- `GET /api/data/version` - dataset version hashes for reproducibility
- `GET /api/metadata` - dataset/feature/model metadata for selected task
- `GET /api/features` - feature list with numeric flags and descriptions
- `GET /api/models` - selectable models for selected task
- `POST /api/feature` - single-feature plot + statistics
- `POST /api/two-features` - two-feature scatter + correlation
- `GET /api/correlation-matrix` - full correlation matrix + high-correlation pairs
- `POST /api/model/confusion-matrix` - classification confusion matrix
- `POST /api/model/regression-diagnostics` - regression residual diagnostics
- `POST /api/model/custom-learning` - feature-subset learning (sync)
- `POST /api/model/custom-learning/job` - feature-subset learning (async job)
- `GET /api/jobs/{job_id}` - async job polling
- `POST /api/jobs/{job_id}/cancel` - async job cancellation
- `GET /api/training-runs` - persisted run history
- `GET /api/training-runs/{run_id}` - one run details with cached payload
- `GET /api/artifacts` - list persisted model artifacts
- `GET /api/artifacts/{task}/{artifact_id}` - artifact metadata/details
- `POST /api/artifacts/{task}/{artifact_id}/predict` - batch inference from artifact
- `POST /api/artifacts/{task}/{artifact_id}/drift` - drift checks vs artifact reference stats
- `GET /api/drift-alerts` - list drift alerts (filter by task/artifact/status)
- `POST /api/drift-alerts/{alert_id}/ack` - acknowledge an alert
- `GET /api/model/compare` - all-model benchmark for selected task
- `GET /api/observability/metrics` - in-process request/training metrics snapshot

---

## 📁 Project Structure

```
ABAX/
├── 📕 docs/                          # Documentation
│   ├── ABAX_Technical_Report.pdf     # ⭐ MAIN DELIVERABLE
│   ├── ABAX_Technical_Report.tex     # LaTeX source
│   └── compile_report.sh             # Build script
│
├── 📓 notebooks/                     # Jupyter notebooks
│   ├── 01_project_overview.ipynb     # Introduction
│   ├── 02_classification.ipynb       # Classification pipeline
│   ├── 03_eda_regression.ipynb       # Regression EDA
│   └── 04_regression.ipynb           # Regression pipeline
│
├── 📊 results/                       # Outputs
│   ├── results.json                  # Model metrics
│   ├── figures/                      # Report visualizations
│   └── model_artifacts/              # Saved model run payload artifacts
│
├── 🔧 src/                           # Production-ready code
│   ├── classification/               # Classification module
│   │   ├── __init__.py               # Clean API exports
│   │   ├── data.py                   # Data loading, feature extraction
│   │   ├── sparse_models.py          # MCP, SCAD implementations
│   │   ├── types.py                  # ClassificationResult, DataSplit
│   │   └── visualization.py          # Plotting functions
│   ├── models/                       # Model implementations
│   │   ├── simple_nn.py              # MLP Neural Network
│   │   ├── comparison.py             # Model comparison utilities
│   │   └── evaluation.py             # Metrics and evaluation
│   ├── features/                     # Feature engineering
│   │   ├── kalman.py                 # 1D/2D Kalman filter implementation
│   │   ├── preprocessing.py          # Data preprocessing
│   │   └── analysis.py               # Feature analysis utilities
│   ├── api/                          # FastAPI + AngularJS MVC dashboard
│   │   ├── app.py                    # API routes
│   │   ├── schemas.py                # Pydantic request/response contracts
│   │   ├── services.py               # Task-aware analytics services
│   │   ├── security.py               # API-key auth + request quota middleware
│   │   ├── run_repository.py         # SQLite-backed run/cache persistence
│   │   ├── db_migrations.py          # Alembic migration runner
│   │   ├── job_manager.py            # Local/Celery async job backend facade
│   │   ├── celery_tasks.py           # Celery worker task entrypoints
│   │   ├── observability.py          # Request/training metrics registry
│   │   └── static/                   # AngularJS module/service/controller + CSS
│   ├── data/                         # Data loaders
│   │   ├── cache_io.py               # CSV/Parquet cache read/write + metadata
│   │   └── versioning.py             # Dataset version hashing/manifest
│   └── utils/                        # Utilities
│
├── 🧪 tests/                         # Unit tests
│
├── 📦 data/                          # Datasets
│   ├── processed/                    # Cached Parquet features + metadata sidecars
│   └── UAH-DRIVESET-v1/              # Raw driving data
│
├── scripts/                          # Utility scripts
│   ├── generate_notebook_figures.py  # Figure generation
│   ├── generate_data_manifest.py     # Reproducibility manifest generation
│   └── migrate_database.py           # Apply DB migrations manually
├── alembic/                          # Alembic migration scripts
├── alembic.ini                       # Alembic configuration
│
├── .pre-commit-config.yaml           # Local quality hooks
└── pyproject.toml                    # Dependencies
```

---

## 🔬 Technical Details

### Feature Engineering (Validated Raw + Processed Features)

| Category | Features | Physical Meaning |
|----------|----------|------------------|
| Speed | mean, std, max, min | Driving intensity |
| Speed Changes | change_mean, change_std | Acceleration patterns |
| Course/Heading | change_mean, std, max | Lane changes, turns |
| Acceleration | X/Y axis mean, std | Core behavior signal |
| Jerk | x_std, y_std | **Smoothness indicator** |
| Event Counts | brake, accel, turn, sharp-turn counts/rates | Discrete maneuver summaries |

**Why Jerk Matters:** Jerk = d(acceleration)/dt. Aggressive drivers have high jerk variance because they brake suddenly, accelerate abruptly, and make sharp steering corrections.

### Kalman Filter Signal Processing

The project implements both 1D and 2D Kalman filters for optimal noise reduction:

```
1D Filter: Smooths individual sensor channels
   - Process noise Q: 0.001 - 0.1 (model uncertainty)
   - Measurement noise R: 0.1 - 1.0 (sensor uncertainty)

2D Filter: Estimates position + velocity simultaneously
   - Provides direct jerk estimation
   - State vector: [position, velocity]
```

**Kalman-based features extracted:**
- Noise reduction ratio
- Smoothness improvement
- Estimated velocity (jerk)

### Data Splitting Strategy

```
Training: 32 samples (80%) from drivers D1-D5
Testing:  8 samples (20%) = D6 trips + stratified samples

⚠️ D6 is NEVER seen during training (production-realistic evaluation)
```

### Neural Network Architecture

```
Input (36 features)
  → StandardScaler (zero mean, unit variance)  ← CRITICAL
  → BatchNorm1d(36)
  → Linear(36, 64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64, 32) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(32, 3) → Softmax
```

**Why Normalization Matters:** Without it, features with large values (speed in km/h) dominate gradient updates while smaller features (jerk) are ignored.

---

## 📈 Key Visualizations

All figures are in `results/figures/`:

| Figure | Description |
|--------|-------------|
| `raw_accelerometer_data.png` | Sensor comparison: AGGRESSIVE vs NORMAL vs DROWSY |
| `class_distribution.png` | Class balance visualization |
| `classifier_comparison.png` | 18-model comparison (train vs test accuracy) |
| `confusion_matrix_classification.png` | Error analysis |
| `feature_importance_classification.png` | Top features with physical interpretation |
| `nn_learning_curves_classification.png` | Neural network training dynamics |
| `regressor_comparison.png` | Regression model comparison |
| `actual_vs_predicted.png` | Prediction quality |
| `residuals.png` | Residual analysis |

---

## 💼 Business Impact

| Application | How This Work Helps |
|-------------|---------------------|
| **Safety Monitoring** | Real-time alerts for aggressive/drowsy driving |
| **Insurance Pricing** | Usage-based premiums from actual behavior |
| **Driver Coaching** | Personalized feedback based on specific behaviors |
| **Fleet Optimization** | Data-driven vehicle selection for fuel efficiency |
| **Environmental Compliance** | Carbon footprint tracking |

---

## 🔧 Technical Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.11, NumPy, Pandas, Scikit-learn |
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **UI (MVC)** | AngularJS 1.8 (module/service/controller), HTML/CSS |
| **Async Jobs** | Local thread backend, Celery 5 + Redis 7 |
| **Persistence** | SQLite/PostgreSQL run store, joblib model artifacts, Parquet caches |
| **Deep Learning (Optional)** | PyTorch 2.x (`deep-learning` extra) |
| **Visualization** | Matplotlib, Seaborn |
| **Quality** | Ruff, MyPy, Pytest, pre-commit, GitHub Actions |
| **Report** | LaTeX (tectonic compiler) |
| **Package Management** | uv |
| **License** | MIT |

---

## 🐛 Troubleshooting

### Kernel Selection
Select the **ABAX (.venv)** kernel in JupyterLab for correct dependencies.

### Reinstall Dependencies
```bash
rm uv.lock && uv sync
```

### Docker Compose Startup Notes

- `docker compose up --build` starts `abax-app`, `abax-worker`, and `redis`.
- If you previously saw `Read-only file system` errors for cache writes, ensure cache paths point to writable locations (current compose config already does this).
- Dataset files are mounted at runtime (`./data:/app/data:ro`), keeping the image lean.

### API Auth / Quota

Set API key auth and per-key quota controls:
```bash
export ABAX_API_AUTH_ENABLED=true
export ABAX_API_KEYS="my-key-1,my-key-2"
export ABAX_API_QUOTA_PER_MINUTE=120
```
Then call protected endpoints with:
```bash
curl -H "X-API-Key: my-key-1" "http://localhost:8000/api/metadata?task=classification"
```

### Worker Health Check
```bash
docker compose ps
docker compose logs -f abax-worker
```

### Containerized E2E Smoke
```bash
ABAX_RUN_DOCKER_E2E=1 pytest -q tests/test_containerized_e2e.py
```

### Compile Report
```bash
# Requires tectonic or pdflatex
cd docs && bash compile_report.sh
```

---

## ✅ Deliverables Checklist

- [x] **Technical Report** - Comprehensive PDF (20+ pages)
- [x] **Classification Models in UI** - Full available model registry exposed
- [x] **Regression Modeling** - End-to-end diagnostics and model comparison
- [x] **Driver-Level Splitting** - D6 held out
- [x] **Feature Intelligence** - Source type (`raw`/`processed`) + lineage in UI
- [x] **Custom Learning Studio** - Feature subset selection + model selection + CV mode
- [x] **Train/Validation Curves** - Per-iteration error history in diagnostics and custom learning
- [x] **Train/Validation Matrices** - Classification confusion matrices for both splits
- [x] **Advanced Regularization** - MCP, SCAD implemented
- [x] **Persistent Training Cache** - Signature-based caching with run persistence
- [x] **Async Training Execution** - Job queue + polling API + Celery/Redis option
- [x] **PostgreSQL Migration Path** - DB URL-based backend with startup migrations
- [x] **Job Cancellation + Retry Policies** - cancel endpoint and retry/backoff controls
- [x] **Artifact Serving Lifecycle** - persist/load/predict flows from trained artifacts
- [x] **Drift Monitoring** - artifact-level drift scoring on incoming batches
- [x] **Containerized E2E Test** - docker compose smoke covering API + worker + datastore
- [x] **Observability** - Request and training timing metrics endpoint
- [x] **Data Caching/Versioning** - Parquet caches with metadata + version manifest
- [x] **Clean Architecture** - Modular codebase with Pydantic-first contracts
- [x] **Tests** - API/data/schema/UI-flow tests passing
- [x] **License** - MIT license for open usage
- [x] **Test Coverage** - pytest-cov integrated for coverage reporting
- [x] **LODO Cross-Validation** - Leave-one-driver-out CV for robust accuracy estimates

---

## 📧 Contact

**Reza Mirzaeifard**
📧 [reza.mirzaeifard@gmail.com](mailto:reza.mirzaeifard@gmail.com)

---

<p align="center">
  <strong>✅ Complete and ready for review!</strong>
</p>
