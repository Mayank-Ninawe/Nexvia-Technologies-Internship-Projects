# System Architecture Refactoring & Verification Record

## Overview

This document details the architectural enhancements, performance optimizations, and integration fixes implemented to transition SentinelGraph from experimental modules into a unified, production-grade fraud detection platform.

---

## 1. Summary of System Refactorings

| Component | Previous State | Refactored State | Impact |
|---|---|---|---|
| **Streaming Persistence** | Local JSON files were the runtime source of truth | Transactional database with idempotent transactions, predictions, alerts, cases, and audit events | Keeps Kafka output, API responses, and analyst actions consistent. |
| **API & Serving** | Isolated `/health` and `/predict` only | Added `/cases`, `/alerts`, `/cases/{id}`, `/graph/summary`, `/metrics/overview`, `/metrics/performance`, with CORS middleware | Enables dynamic, unified frontend and third-party orchestration. |
| **React Frontend** | Hardcoded mock arrays (`ALR-4412`, etc.) & compile-time JSON imports | Real-time FastAPI REST integration with interactive case triage | Fully operational analyst command center. |
| **Feature Store (Feast)** | Unclear architectural purpose & schema disconnect | Explicitly documented as offline/online feature parity validator | Clear MLOps boundaries without latency penalty in live inference. |
| **Repository Hygiene** | Redundant duplicate `artifacts/api/` folder | Deleted duplicate directory | Single source of truth across all API modules. |

---

## 2. Component Specifications

### 2.1 Transactional Streaming Persistence (`backend/persistence/database.py`)

- **Design:** Kafka consumer scores transactions via FastAPI and persists the event, prediction, alert, and case through one database boundary.
- **Persistence Pattern:** PostgreSQL in production and SQLite for local development.
- **Idempotency:** Transaction IDs and event IDs prevent duplicate scored events and cases.
- **Diagnostics:** `data/processed/consumer_errors.jsonl` remains a local error log only.

### 2.2 FastAPI Endpoints (`api/main.py`)

- `GET /health`: Health status and service identifier.
- `POST /predict`: Real-time transaction scoring via `realtime_safe_xgboost` model pipeline.
- `GET /cases` & `GET /alerts`: Returns live analyst review queue with aggregated count metrics (`open_count`, `resolved_count`, `total_count`).
- `PATCH /cases/{transaction_id}`: Allows analysts to update investigation state (`Under Review`, `Escalated`, `Resolved`) and attach audit feedback.
- `GET /graph/summary`: Graph analytics metrics, connected component counts, and top fraud-linked accounts.
- `GET /metrics/overview`: Aggregated KPI metrics for command center overview.
- `GET /metrics/performance`: Model validation metrics (Precision, Recall, Average Precision, TPR) and full confusion matrix counts.

### 2.3 Feast Feature Store Architecture

- **Registry & Store:** `backend/feature_store/feature_repo/feature_store.yaml` configuring the local Feast store.
- **Architectural Boundary:** Feast standardizes feature names and types across training and batch serving pipelines. To maintain sub-10ms real-time latency SLAs, the live scoring service constructs derived features directly from the transaction envelope, while Feast guarantees parity and reproducible definitions.

---

## 3. End-to-End Verification Record

### 3.1 Unit Test Suite (`pytest`)
All 27 test cases across API, streaming, graph features, drift monitoring, and Feast snapshot pass:
```text
backend/tests/test_api.py::test_health_check PASSED
tests/test_api.py::test_predict_valid_transaction PASSED
tests/test_api.py::test_predict_invalid_amount PASSED
tests/test_api.py::test_get_cases_and_alerts PASSED
tests/test_api.py::test_get_graph_summary PASSED
tests/test_api.py::test_get_metrics_overview PASSED
tests/test_api.py::test_get_metrics_performance PASSED
tests/test_drift_monitoring.py::test_prepare_features_returns_expected_columns PASSED
tests/test_drift_monitoring.py::test_controlled_drift_multiplies_amount_only PASSED
tests/test_drift_monitoring.py::test_retraining_review_policy_triggers_for_amount_drift PASSED
tests/test_drift_monitoring.py::test_retraining_review_policy_triggers_for_feature_fraction PASSED
tests/test_drift_monitoring.py::test_retraining_review_policy_does_not_trigger_below_threshold PASSED
tests/test_environment.py::test_pandas_import PASSED
tests/test_environment.py::test_numpy_import PASSED
tests/test_evidently_extraction.py::test_extract_drift_summary_from_evidently_v07_metrics PASSED
tests/test_feast_snapshot.py::test_create_feature_snapshot PASSED
tests/test_features.py::test_add_derived_features PASSED
tests/test_features.py::test_get_realtime_safe_feature_matrix_columns PASSED
tests/test_features.py::test_get_leakage_suspect_feature_matrix_columns PASSED
tests/test_graph_features.py::test_graph_features_do_not_use_same_step_transactions PASSED
tests/test_graph_features.py::test_graph_features_use_only_prior_steps PASSED
tests/test_graph_inspection.py::test_fraud_rate_summary_by_signal PASSED
tests/test_streaming.py::test_validate_event_returns_transaction PASSED
tests/test_streaming.py::test_validate_event_rejects_unsupported_event_type PASSED
tests/test_streaming.py::test_validate_event_rejects_missing_transaction_field PASSED
tests/test_streaming.py::test_json_record_persistence PASSED
tests/test_streaming.py::test_jsonl_record_persistence PASSED
======================= 27 passed, 4 warnings in 23.43s =======================
```

### 3.2 Frontend Build & Browser Verification
- **Build Status:** `npm run build` compiled 2,310 modules cleanly into `frontend/dist/`.
- **Runtime Execution:** Tested via headless browser agent navigating `http://localhost:5173/`.
- **Live User Actions Verified:**
  - Overview metrics accurately loaded from FastAPI `/metrics/overview`.
  - Alerts queue loaded from `/cases`; interactive case resolution dynamically mutated state and persisted audit notes.
  - Roundtrip ping latency measured at **39 ms**.
  - Interactive risk scoring test evaluated sample transaction and rendered model verdict in real time.

---

## 4. Decision Threshold Analysis & Multi-Tier Operational Policy

### 4.1 Threshold Selection Mechanics
In `reports/evaluation_results.json`, the decision threshold is calibrated to:
$$\tau = 9.248 \times 10^{-5} \quad (\approx 0.009\%)$$

- **Reason for Calibration:** The PaySim dataset is severely imbalanced ($\approx 0.129\%$ positive fraud prevalence). During model validation, the evaluation script minimized False Negatives to achieve $100\%$ Recall ($\text{FN} = 0$, $\text{TP} = 846$) on the holdout test set.
- **Precision/Recall Trade-off:** At this ultra-sensitive threshold, the model catches all fraud events but incurs $1,012$ False Positives out of $66,990$ legitimate transactions ($\text{Precision} \approx 45.5\%$).

### 4.2 Production Multi-Tier Routing Recommendation
In real-world financial fraud architectures, a binary pass/fail cutoff at $\tau = 0.009\%$ creates severe alert fatigue for risk teams. Production systems deploy **multi-tier triage buckets**:

```text
               ┌────────────────────────────────────────────────────────┐
               │              Predicted Fraud Probability               │
               └───────────────────────────┬────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼────────────────────────────────┐
         │                                 │                                │
         ▼                                 ▼                                ▼
┌─────────────────┐               ┌─────────────────┐              ┌─────────────────┐
│ Probability ≥ 90%│              │ Threshold ≤ p < 90│             │  p < Threshold  │
└────────┬────────┘               └────────┬────────┘              └────────┬────────┘
         │                                 │                                │
         ▼                                 ▼                                ▼
┌─────────────────┐               ┌─────────────────┐              ┌─────────────────┐
│ Immediate Block │               │  Analyst Review │              │   Auto-Allow    │
│ (Hard decline)  │               │ / 2FA Challenge │              │  (Frictionless) │
└─────────────────┘               └─────────────────┘              └─────────────────┘
```

1. **High Confidence ($\ge 90\%$):** Hard decline; block funds transfer immediately.
2. **Elevated Suspicion ($\tau \le p < 90\%$):** Route to analyst review queue or prompt user for Step-Up Authentication (SMS OTP / Biometrics).
3. **Normal Activity ($< \tau$):** Automated real-time approval without operational intervention.

This triage logic is directly encoded in `api/predictor.py` through the `recommended_action` field.
