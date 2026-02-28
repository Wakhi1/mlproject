# ERS Compliance Risk Model (CRM)
>
> A lightweight MLOps pipeline for taxpayer compliance risk scoring at the Eswatini Revenue Service.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Versioning](#monitoring--versioning)
- [Retraining with New Data](#retraining-with-new-data)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)

---

## Overview

The **ERS Compliance Risk Model** is a machine learning system that assigns a **compliance risk score (0–100)** to taxpayers registered with the Eswatini Revenue Service. A higher score indicates a higher risk of non-compliance, tax evasion, or fraud.

ERS staff can:

- Query the risk score of a specific taxpayer by TIN (Taxpayer Identification Number)
- Submit new structured data to update and retrain the model
- Monitor model performance over time via a lightweight dashboard

The system is intentionally **lean** — no heavy orchestration frameworks. It uses:

- **FastAPI** — REST API for scoring and data ingestion
- **scikit-learn / XGBoost** — model training
- **MLflow** — model versioning and experiment tracking
- **Docker + Docker Compose** — containerized deployment
- **GitHub Actions** — CI/CD pipeline
- **Prometheus + Grafana** — runtime monitoring (AUC, prediction drift, request latency)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ERS Staff / Frontend                  │
│              (Internal Web Portal or API Client)             │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
                          ▼
              ┌───────────────────────┐
              │   FastAPI Service     │  ← /predict, /ingest, /retrain
              │   (Port 8000)         │
              └──────────┬────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐  ┌──────────┐  ┌───────────────┐
   │  MLflow    │  │ Model    │  │  PostgreSQL    │
   │  Tracking  │  │ Registry │  │  (Taxpayer     │
   │  (Port     │  │ (Local   │  │   Records DB)  │
   │   5000)    │  │  FS/S3)  │  └───────────────┘
   └────────────┘  └──────────┘
          │
          ▼
   ┌─────────────────────┐
   │ Prometheus + Grafana │  ← AUC, Drift, Latency dashboards
   │ (Port 3000)          │
   └─────────────────────┘
```

---

## Project Structure

```
ers-compliance-risk-model/
│
├── api/
│   ├── main.py                  # FastAPI entry point
│   ├── routes/
│   │   ├── predict.py           # POST /predict — score a taxpayer
│   │   ├── ingest.py            # POST /ingest — add new labelled records
│   │   └── retrain.py           # POST /retrain — trigger model retraining
│   ├── schemas/
│   │   ├── taxpayer.py          # Pydantic input/output models
│   └── dependencies.py          # Shared deps (DB, model loader)
│
├── model/
│   ├── train.py                 # Training script (XGBoost pipeline)
│   ├── evaluate.py              # AUC, F1, confusion matrix output
│   ├── preprocess.py            # Feature engineering & encoding
│   └── artifacts/               # Saved model artifacts (gitignored)
│
├── data/
│   ├── ers_sample_dataset.csv   # Seed dataset (Eswatini context)
│   └── schema.md                # Column definitions
│
├── monitoring/
│   ├── prometheus.yml           # Scrape config
│   └── grafana/
│       └── dashboard.json       # Pre-built compliance risk dashboard
│
├── mlflow/
│   └── mlruns/                  # MLflow experiment store (gitignored)
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # Lint, test, build on PR
│       └── cd.yml               # Deploy to server on merge to main
│
├── docker-compose.yml           # Orchestrates all services
├── Dockerfile                   # API container
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dataset

The model trains on structured taxpayer data. Each record represents one taxpayer filing period. See `data/ers_sample_dataset.csv` for the seed data.

### Feature Columns

| Column | Type | Description |
|---|---|---|
| `tin` | string | Taxpayer Identification Number |
| `taxpayer_type` | categorical | `Individual`, `Company`, `NGO`, `Parastatal` |
| `region` | categorical | `Hhohho`, `Manzini`, `Lubombo`, `Shiselweni` |
| `industry_sector` | categorical | e.g. `Retail`, `Agriculture`, `Manufacturing` |
| `years_registered` | int | Years since registration with ERS |
| `annual_turnover_szl` | float | Declared annual turnover in Eswatini Lilangeni |
| `vat_registered` | bool | Whether taxpayer is VAT registered |
| `paye_registered` | bool | Whether taxpayer deducts PAYE |
| `num_employees_declared` | int | Number of employees declared |
| `filings_due_last_12m` | int | Expected filings in the last 12 months |
| `filings_submitted_last_12m` | int | Actual filings submitted |
| `late_filings_count` | int | Number of late filings |
| `amended_returns_count` | int | Number of amended/corrected returns |
| `outstanding_tax_szl` | float | Current outstanding tax debt (SZL) |
| `penalty_count` | int | Number of penalties issued |
| `prior_audit_flag` | bool | Whether previously audited |
| `prior_audit_finding` | bool | Whether prior audit found discrepancy |
| `days_since_last_payment` | int | Days since most recent tax payment |
| `payment_plan_active` | bool | Currently on a payment arrangement |
| `cross_border_transactions` | bool | Has declared cross-border activity |
| `is_non_compliant` | bool | **Target label** — confirmed non-compliant |

---

## Model

The model is an **XGBoost classifier** wrapped in a scikit-learn `Pipeline` with preprocessing steps (encoding, imputation, scaling).

**Training:**

```bash
python model/train.py --data data/ers_sample_dataset.csv --experiment ers_crm_v1
```

**Output:**

- Trained model logged to MLflow with AUC, F1, Precision, Recall
- Model artifact saved and registered as `ers-compliance-model` in MLflow Model Registry
- Automatically promoted to `Staging` if AUC > 0.80, `Production` if AUC > 0.88

---

## API Usage

### Score a Taxpayer

```http
POST /predict
Content-Type: application/json

{
  "tin": "E0012345678",
  "taxpayer_type": "Company",
  "region": "Manzini",
  "industry_sector": "Retail",
  "years_registered": 5,
  "annual_turnover_szl": 1500000,
  "vat_registered": true,
  "paye_registered": true,
  "num_employees_declared": 12,
  "filings_due_last_12m": 12,
  "filings_submitted_last_12m": 9,
  "late_filings_count": 4,
  "amended_returns_count": 2,
  "outstanding_tax_szl": 45000,
  "penalty_count": 1,
  "prior_audit_flag": false,
  "prior_audit_finding": false,
  "days_since_last_payment": 120,
  "payment_plan_active": false,
  "cross_border_transactions": false
}
```

**Response:**

```json
{
  "tin": "E0012345678",
  "risk_score": 74.3,
  "risk_level": "HIGH",
  "model_version": "ers-compliance-model/3",
  "scored_at": "2025-09-01T10:23:00Z"
}
```

### Ingest New Labelled Record

```http
POST /ingest
```

Accepts the same schema as `/predict` plus `is_non_compliant: true/false`. Records are stored to the database and used in the next retraining cycle.

### Trigger Retraining

```http
POST /retrain
```

Pulls all records from the database, retrains the pipeline, evaluates, and registers a new model version in MLflow. Protected by an internal API key.

---

## Docker Deployment

### Prerequisites

- Docker >= 24
- Docker Compose >= 2.20

### Run All Services

```bash
cp .env.example .env
# Edit .env with your secrets
docker compose up -d --build
```

### Services Started

| Service | Port | Purpose |
|---|---|---|
| `api` | 8000 | FastAPI scoring & ingestion |
| `mlflow` | 5000 | Experiment tracking UI |
| `postgres` | 5432 | Taxpayer record storage |
| `prometheus` | 9090 | Metrics scraping |
| `grafana` | 3000 | Monitoring dashboard |

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_version": "ers-compliance-model/3"}
```

---

## CI/CD Pipeline

Powered by **GitHub Actions** with two workflows:

### `ci.yml` — Runs on every Pull Request

1. Lint with `flake8`
2. Run unit tests (`pytest`)
3. Build Docker image (no push)
4. Validate model training script with sample data

### `cd.yml` — Runs on merge to `main`

1. Build and push Docker image to container registry (GHCR or Docker Hub)
2. SSH into deployment server
3. Pull latest image and restart services via `docker compose up -d`
4. Run smoke test against `/health` endpoint
5. Notify on failure

> Secrets required: `SERVER_HOST`, `SERVER_USER`, `SERVER_SSH_KEY`, `REGISTRY_TOKEN`

---

## Monitoring & Versioning

### MLflow (Model Versioning)

- Every training run logs: AUC-ROC, F1-Score, Precision, Recall, feature importances
- Models are versioned and tagged (`Staging` / `Production`)
- Access MLflow UI: `http://localhost:5000`

### Prometheus + Grafana (Runtime Monitoring)

The API exposes a `/metrics` endpoint. Grafana dashboards track:

| Metric | Description |
|---|---|
| `ers_crm_auc_score` | AUC of currently deployed model |
| `ers_crm_prediction_total` | Total predictions served |
| `ers_crm_high_risk_ratio` | Ratio of HIGH risk predictions (drift indicator) |
| `ers_crm_api_latency_seconds` | API response time |
| `ers_crm_ingest_total` | Total new records ingested |

> If `ers_crm_high_risk_ratio` shifts significantly from baseline, a Grafana alert fires — this is an early signal of **data drift** and triggers a review for retraining.

Access Grafana: `http://localhost:3000` (default login: `admin / admin`)

---

## Retraining with New Data

The model is designed to **improve over time** as ERS staff use it:

1. Staff query a taxpayer → model returns a risk score
2. If an audit later confirms or refutes the prediction, staff submit the labelled outcome via `POST /ingest`
3. Records accumulate in PostgreSQL
4. Periodically (manually or via a cron job), staff call `POST /retrain`
5. A new model version is trained, evaluated, and — if it outperforms the current production model on AUC — automatically promoted

This creates a **feedback loop** that continuously improves accuracy with real ERS audit outcomes.

---

## Environment Variables

```env
# API
API_KEY=your_internal_api_key
APP_ENV=production

# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=ers_crm
DB_USER=ers_user
DB_PASSWORD=strong_password

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=ers_crm

# Model
MODEL_NAME=ers-compliance-model
MODEL_STAGE=Production
MIN_AUC_FOR_PROMOTION=0.85
```

---

## Contributing

This project is maintained by the ERS ICT & Data Analytics Division.

- Branch naming: `feature/`, `fix/`, `chore/`
- All PRs require passing CI before merge
- Model changes must include updated evaluation metrics in the PR description
- Never commit real taxpayer data — use anonymized or synthetic records only

---

*Built for the Eswatini Revenue Service — Kingdom of Eswatini*