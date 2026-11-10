# SentinelGraph

SentinelGraph is an end-to-end, real-time financial fraud detection platform built around the PaySim synthetic transaction dataset. It trains a leakage-aware XGBoost model, scores transactions through a FastAPI service, persists high-risk events and analyst cases in PostgreSQL, and presents operational insights through a React dashboard.

The application is deployed as a cloud-hosted web application:

- **Frontend:** Vercel  
- **Backend API:** Render  
- **Database:** Render PostgreSQL  
- **Streaming:** Apache Kafka through Docker  

## Live application

| Component | URL |
|---|---|
| React dashboard | https://sentinelgraph-frontend.vercel.app/ |
| FastAPI service | https://sentinelgraph-api-n090.onrender.com |
| API documentation | https://sentinelgraph-api-n090.onrender.com/docs |
| Health check | https://sentinelgraph-api-n090.onrender.com/health |

## What it does

SentinelGraph supports the following workflow:

1. Financial transactions are published to a Kafka topic.
2. A Kafka consumer sends each transaction to the FastAPI scoring endpoint.
3. The realtime-safe XGBoost model calculates a fraud probability and risk score.
4. High-risk transactions are persisted as alerts and analyst cases.
5. The React dashboard displays live alerts, case status, graph analysis, model performance, and system health.
6. Analysts can update case status and add review feedback.

## Key features

- **Machine learning pipeline:** CSV validation, chronological train/validation/test splits, feature engineering, XGBoost training, evaluation, and MLflow tracking.
- **Real-time scoring:** FastAPI endpoint for transaction risk scoring.
- **Fraud case management:** Automatic case creation for flagged transactions, analyst status updates, and feedback capture.
- **Streaming architecture:** Kafka producer and consumer for transaction replay and real-time scoring.
- **Cloud deployment:** React frontend on Vercel, FastAPI backend and PostgreSQL database on Render.
- **Graph analysis:** Account-relationship analysis to identify potentially fraud-linked accounts.
- **Model monitoring:** Model performance metrics, evaluation reports, and drift-analysis tooling.
- **Analyst interfaces:** React dashboard for the primary application and Streamlit dashboard for operational review.

## Deployed architecture

```text
PaySim transaction data
        |
        v
Kafka producer
        |
        v
Kafka topic: fraud-transactions
        |
        v
Kafka consumer
        |
        v
Render FastAPI /predict
        |
        v
Realtime-safe XGBoost model
        |
        v
Render PostgreSQL
        |
        +--> Fraud alerts
        +--> Analyst cases
        |
        v
Vercel React dashboard
```

The deployed model is `realtime_safe_xgboost`. It uses transaction type, amount, pre-transaction balances, and derived pre-transaction indicators. Post-transaction balance fields are used only by the diagnostic `leakage_suspect_xgboost` experiment and are not used by the primary scoring path.

## Repository layout

```text
backend/
  api/                  FastAPI application, schemas, and predictor
  dashboard/            Streamlit analyst dashboard
  feature_store/        Feast repository and snapshot generation
  graph/                Account graph construction and inspection
  monitoring/           Evidently drift report generation
  persistence/          SQLAlchemy database layer
  src/fraud_detection/  Data loading, features, training, and evaluation
  streaming/            Kafka producer and consumer
  tests/                Backend tests

frontend/               React + TypeScript + Vite dashboard
infra/docker/           API, PostgreSQL, and Kafka Compose definitions
infra/k8s/              Kubernetes API manifests
data/                   Local dataset and generated transaction data
models/                 Local trained model artifacts
reports/                Local evaluation and training outputs
```

## Requirements

- Python 3.10
- Node.js 20 and npm
- Docker Desktop, required for local Kafka
- PaySim dataset
- Git

## Local setup

Create and activate a Python virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm ci
cd ..
```

## Dataset setup

Download the PaySim dataset and place it at:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

The application validates the required PaySim columns before training. Do not commit the CSV, trained model binaries, MLflow artifacts, generated reports, or local application database.

## Run locally

### Start Kafka

```powershell
docker compose -f infra\docker\docker-compose.kafka.yml up -d
docker ps
```

Kafka runs locally on:

```text
localhost:9092
```

### Start the backend

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1

$env:PYTHONPATH = "$PWD\backend\src;$PWD\backend"
python -m uvicorn api.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

### Start the Streamlit dashboard

In another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run backend\dashboard\app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

## Real-time streaming workflow

Start the Kafka consumer in one terminal:

```powershell
cd D:\path\to\realtime-financial-fraud-detection
.\.venv\Scripts\Activate.ps1
cd backend

$env:API_KEY = "your_api_key"

$env:PYTHONPATH = "$PWD\..;$PWD;$PWD\src"

python -m streaming.consumer
```

Start the Kafka producer in another terminal:

```powershell
cd D:\path\to\realtime-financial-fraud-detection
.\.venv\Scripts\Activate.ps1
cd backend

$env:PYTHONPATH = "$PWD\..;$PWD;$PWD\src"

python -m streaming.producer --limit 20 --delay-seconds 0.5 --fraud-only
```

The producer publishes PaySim transactions to Kafka. The consumer sends each event to the scoring API and persists scored transactions, predictions, alerts, and cases.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health status |
| GET | `/readyz` | Database, model, and report readiness |
| POST | `/predict` | Score a transaction and persist high-risk alerts |
| GET | `/cases` | Return analyst cases and fraud alerts |
| PATCH | `/cases/{transaction_id}` | Update case status and analyst feedback |
| GET | `/graph/summary` | Return account-graph metrics |
| GET | `/metrics/overview` | Return dashboard KPIs |
| GET | `/metrics/performance` | Return model performance metrics |

## Model evaluation

The current local evaluation artifact records the following for the realtime-safe model:

| Metric | Value |
|---|---:|
| Test average precision | 0.9926426882 |
| Test precision | 0.4553283100 |
| Test recall | 1.0000000000 |
| Test F2 score | 0.8069439145 |
| Test alert rate | 0.0269869858 |
| Test false-positive rate | 0.0148819152 |

These values describe the checked-in code and locally generated evaluation report. They are not a production performance guarantee.

## Train and evaluate

With the dataset in place:

```powershell
$env:PYTHONPATH = "$PWD\backend\src;$PWD\backend"
python -m fraud_detection.train
python -m fraud_detection.evaluate
```

Training writes model files to `models/` and a summary to `reports/training_summary.json`. Evaluation writes metrics and plots to `reports/`.

## Docker Compose

For the API with PostgreSQL, copy `.env.example` to `.env`, update the values, and run:

```powershell
Copy-Item .env.example .env
docker compose --env-file .env -f infra\docker\docker-compose.yml up --build
```

The API is exposed on port 8000.

## Verification

Run backend checks:

```powershell
$env:PYTHONPATH = "$PWD\backend\src;$PWD\backend"
ruff check backend
pytest -q
```

Build the frontend:

```powershell
cd frontend
npm ci
npm run build
```

## Configuration

`.env.example` documents local configuration:

- `POSTGRES_PASSWORD`: PostgreSQL password for Docker Compose.
- `CORS_ALLOWED_ORIGINS`: Comma-separated frontend origins.
- `API_KEY`: API key enforced by FastAPI middleware when set.
- `PREDICT_RETRY_COUNT`: Kafka consumer retry count for prediction requests.

Do not commit `.env` or real credentials.

## Security notes

- API requests are protected using an `X-API-Key` header.
- CORS is configured through the `CORS_ALLOWED_ORIGINS` environment variable.
- Sensitive values, database credentials, and API keys must not be committed to Git.
- The deployed frontend uses Vercel environment variables for API configuration.

## Current deployment status

| Component | Deployment target | Status |
|---|---|---|
| React frontend | Vercel | Deployed |
| FastAPI backend | Render | Deployed |
| PostgreSQL database | Render | Deployed |
| Kafka | Local Docker | Operational |
| ML model | Render container | Operational |
| Fraud alert workflow | Render + PostgreSQL | Operational |
| Analyst case management | Render + PostgreSQL | Operational |

## Future improvements

- Deploy Kafka to a managed cloud service.
- Add user authentication and role-based access control.
- Add automated database backups and retention policies.
- Add centralized logging, tracing, and alerting.
- Add load testing and horizontal scaling.
- Add a custom domain and HTTPS certificate management.
- Expand automated end-to-end tests.