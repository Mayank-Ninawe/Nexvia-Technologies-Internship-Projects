# Real-Time Financial Fraud Detection Platform — Project Plan

## Project overview

This project implements **SentinelGraph**, a real-time financial fraud detection platform. It ingests PaySim transactions, engineers leakage-safe features, scores transactions with an XGBoost model through FastAPI, streams events through Kafka, presents alerts in Streamlit, tracks models with MLflow, monitors drift with Evidently, and deploys the scoring API to Minikube Kubernetes.

The project focuses on an auditable, explainable, and deployable fraud decision workflow rather than only training a classifier.

## Core flow

```text
PaySim transactions
→ Data validation
→ Leakage-safe feature engineering
→ XGBoost model scoring
→ FastAPI prediction service
→ Kafka event streaming
→ Alert persistence
→ Streamlit analyst dashboard
→ MLflow tracking
→ Evidently drift monitoring
→ Minikube Kubernetes deployment
```

## Dataset

The project uses the PaySim synthetic financial transaction dataset.

- Total source rows: 6,362,620
- Preserved processed rows: 6,362,620
- Fraud rows: 8,213
- Non-fraud rows: 6,354,407
- Fraud rate: approximately 0.129%
- Time range: hourly simulated steps across approximately 30 days

A preprocessing issue that initially removed 16 zero-amount fraudulent transactions was identified and fixed. The final adapter preserves every source transaction and fraud label.

## Technology stack

| Layer | Technology |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Feature store | Feast |
| Machine learning | XGBoost, scikit-learn |
| Model tracking | MLflow |
| Drift monitoring | Evidently |
| API serving | FastAPI, Uvicorn |
| Streaming | Apache Kafka |
| Dashboard | Streamlit |
| Containerization | Docker, Docker Compose |
| Orchestration | Minikube, Kubernetes |
| Testing and quality | Pytest, Ruff |
| Version control | Git and GitHub |

## Implementation phases

### Phase 1 — Project setup

Created the repository structure, Python environment, Git workflow, `.gitignore`, project documentation, and foundational package configuration.

**Deliverables**

- GitHub repository
- Python virtual environment
- Project directory structure
- Initial documentation
- Reproducible development setup

**Status:** Completed.

### Phase 2 — Data ingestion

Implemented the PaySim ingestion adapter and normalized raw transaction records into a consistent internal schema.

**Deliverables**

- `scripts/adapters/paysim.py`
- `data/processed/transactions.parquet`
- Transaction ID generation
- Event-time normalization
- Fraud-label preservation
- Data-preservation checks

**Status:** Completed.

### Phase 3 — Exploratory data analysis

Validated dataset quality and analyzed fraud distribution, transaction types, amounts, timestamps, missing values, and duplicates.

**Deliverables**

- `notebooks/01_eda.ipynb`
- Data-validation checks
- Fraud-rate analysis
- Transaction-type analysis
- Time-range verification

**Status:** Completed.

### Phase 4 — Leakage-safe feature engineering

Built features using only information available before each transaction. The current transaction’s label, amount, receiver, and other post-event information were excluded from historical aggregations.

**Deliverables**

- Feature schema
- Feature dictionary
- Historical transaction features
- Time-window features
- Leakage-prevention tests
- Generated feature dataset

**Status:** Completed.

### Phase 5 — Model development

Trained and evaluated an XGBoost fraud-detection model using chronologically appropriate data splits and imbalance-aware evaluation.

**Deliverables**

- Training pipeline
- XGBoost model
- Evaluation metrics
- Model artifact
- Model metadata
- Reproducible training workflow

**Status:** Completed.

### Phase 6 — Feast feature store

Created a local Feast feature repository to keep training features and online serving features consistent.

**Deliverables**

- Feast feature definitions
- Feature views
- Entity definitions
- Feature repository configuration
- Offline/online consistency checks

**Status:** Completed.

### Phase 7 — Graph feasibility and explainability

Implemented graph-based analysis to identify suspicious account relationships and assess coordinated fraud-ring detection.

**Deliverables**

- Graph construction logic
- Account relationship analysis
- Fraud-ring feasibility checks
- Explainability documentation
- `graph_feasibility.md`

**Status:** Completed.

### Phase 8 — Real-time streaming

Implemented Kafka-based transaction streaming. A producer replays PaySim events, while a consumer consumes events, requests predictions from FastAPI, and persists scored alerts.

**Deliverables**

- Kafka producer
- Kafka consumer
- Docker Compose Kafka configuration
- Event replay workflow
- Consumer-to-API integration
- Alert persistence

**Status:** Completed.

### Phase 9 — API serving

Implemented FastAPI model serving with health checks and prediction endpoints.

**Deliverables**

- FastAPI application
- Model-loading logic
- Prediction endpoint
- Health endpoint
- Request/response schemas
- Local API verification

**Status:** Completed.

### Phase 10 — Analyst dashboard

Created a Streamlit dashboard for analysts to inspect scored transactions, fraud probabilities, and alerts.

**Deliverables**

- Streamlit application
- API integration
- Alert visualization
- Analyst review workflow

**Status:** Completed.

### Phase 11 — MLOps and monitoring

Added MLflow experiment tracking and Evidently drift monitoring.

**Deliverables**

- MLflow tracking workflow
- Model metadata and artifacts
- Evidently drift report
- Drift-review trigger
- Monitoring reports

**Status:** Completed.

### Phase 12 — Containerization and deployment

Containerized the FastAPI scoring service and deployed it to a local Minikube Kubernetes cluster.

**Deliverables**

- `infra/docker/Dockerfile.api`
- `requirements-api.txt`
- `infra/docker/docker-compose.yml`
- `.dockerignore`
- Kubernetes Deployment manifest
- Kubernetes Service manifest
- Minikube deployment verification
- `deployment_status.md`

**Status:** Completed.

### Phase 10 — Production Refactoring & Full-Stack Platform Integration

Enhanced the system from isolated components into a cohesive, high-throughput, and observable fraud detection platform:
- **Transactional Streaming Persistence:** Kafka consumer output is persisted through the shared database boundary with idempotent transaction, prediction, alert, case, and audit records. JSONL remains only for local consumer-error diagnostics.
- **RESTful API Expansion & CORS:** Added `/cases`, `/alerts`, `/cases/{id}`, `/graph/summary`, `/metrics/overview`, and `/metrics/performance` with full CORS support.
- **Modern React Command Center (SPA):** Built a React 18 + TypeScript + Tailwind CSS analyst console with live queue triage, dynamic SVG cluster graph, and sub-40ms latency diagnostics.
- **Feast Architecture Standard:** Clarified Feast's role as the offline-to-online schema and definition parity validator while maintaining sub-10ms in-memory feature engineering for live scoring.
- **Graph Page Runtime Resilience:** Resolved undefined metric access bugs and enforced null-safe rendering.
- **Test Suite Expansion:** Expanded pytest suite to 27 comprehensive tests (all passing).

**Status:** Completed.

## API deployment flow

```text
Docker image build
→ Minikube image load
→ Kubernetes Deployment
→ fraud-api pod
→ Kubernetes ClusterIP Service
→ Minikube local service proxy
→ Health and prediction requests
```

## Verification results

### Data validation

```text
Raw rows: 6,362,620
Processed rows: 6,362,620
Raw fraud rows: 8,213
Processed fraud rows: 8,213
Zero-amount fraud rows preserved: 16
```

### Kubernetes API deployment

```text
NAME                         READY   STATUS    RESTARTS   AGE
fraud-api-77cf7f7cdb-shcqx   1/1     Running   0          102s
```

### API health response

```json
{
  "status": "healthy",
  "service": "fraud-detection-api"
}
```

## Success criteria

The project meets the following criteria:

- End-to-end transaction-to-alert workflow is reproducible.
- Features avoid future-data leakage.
- Model scoring is available through FastAPI.
- Kafka streaming is demonstrated.
- Analyst dashboard is operational.
- Model experiments and drift monitoring are tracked.
- API is containerized and deployed to Kubernetes.
- Deployment status and limitations are documented honestly.

## Future improvements

- Deploy Kafka inside Kubernetes using StatefulSets and persistent volumes.
- Add production-grade secrets management.
- Add horizontal pod autoscaling.
- Add CI/CD for tests, linting, image builds, and deployments.
- Benchmark p95 prediction latency under concurrent load.
- Implement a complete analyst-feedback retraining loop.
- Extend graph analysis into a scheduled fraud-ring detection service.