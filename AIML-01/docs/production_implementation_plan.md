# Production Implementation Plan

## Current baseline

The current leakage-safe model is a research/demo baseline, not a production
approval decision system:

| Metric | Current measured value |
|---|---:|
| Average precision | 0.99264 |
| Precision at selected threshold | 45.53% |
| Recall | 100% |
| True positives | 846 |
| False positives | 1,012 |
| False negatives | 0 |
| Threshold | 0.00009248 |
| Alert rate | 2.70% |
| False-positive rate | 1.49% |
| Brier score | 0.000737 |

These values come from the local holdout artifact in
`reports/evaluation_results.json`. They are not a guarantee of production
accuracy. PaySim is synthetic, and no real-world generalization claim should
be made from it.

The current chronological stability report shows precision of 25.65%, 54.10%,
and 100% across the three fixed-model test windows. The final window contains
only fraud rows (296 of 296), so its 100% precision is not evidence of stable
production performance.

## Target architecture

```text
frontend/                 React analyst console
backend/api/              FastAPI HTTP boundary
backend/src/              ML training, features, evaluation, model registry client
backend/streaming/        Kafka producer and consumer workers
backend/feature_store/    Feast definitions and materialization jobs
backend/graph/            Time-aware graph features and investigations
backend/monitoring/       Drift and model monitoring jobs
backend/dashboard/        Legacy Streamlit console during migration
backend/tests/            Backend unit and integration tests
infra/docker/             Local container and Kafka composition
infra/k8s/                Deployment, service, probes, and future worker manifests
data/                     Local development data only
models/                   Local development model artifacts only
reports/                  Reproducible evaluation outputs
```

The first structural refactor is complete. Backend tests and Ruff must remain
passing after every subsequent phase.

## Implementation phases

### Phase 1 — Repository and developer experience

- Keep the `backend/`, `frontend/`, and `infra/` boundaries.
- Add backend and frontend CI checks.
- Add one documented command for local API, frontend, Kafka, and tests.
- Remove stale documentation and duplicate generated artifacts.
- Keep secrets, datasets, model binaries, caches, and local databases out of Git.

### Phase 2 — Correct runtime data flow

- Replace local JSON case storage with a transactional database.
- Create an explicit `alerts` and `cases` schema with idempotent event IDs.
- Make the Kafka consumer persist every scored event and create/update an alert
  in the same reliable workflow.
- Make `/cases`, `/alerts`, and dashboard pages read from that database.
- Remove hardcoded overview metrics and synthetic trend values.
- Standardize on one event contract and one persistence format.

### Phase 3 — API production hardening

- Add versioned routes under `/api/v1`.
- Add authentication, authorization, request IDs, structured logging, and
  consistent error responses.
- Replace wildcard CORS with configured trusted origins.
- Add pagination, filtering, sorting, and bounded request sizes.
- Add database health, model health, Kafka health, and readiness endpoints.
- Add idempotency and retry handling for prediction and event ingestion.

### Phase 4 — Model accuracy and decision policy

- Rebuild evaluation with multiple chronological windows, not one split only.
- Measure PR-AUC, precision, recall, F-beta, false-positive rate, alert rate,
  recall at fixed alert budgets, calibration, and cost-weighted loss.
- Define business constraints before changing the threshold:
  maximum alert volume, minimum recall, investigation capacity, and cost of
  false positives versus false negatives.
- Evaluate probability calibration and threshold stability across time windows.
- Add a shadow/canary model workflow and model version metadata.
- Do not advertise a single “accuracy” number; accuracy is misleading for this
  highly imbalanced dataset.

### Phase 5 — Monitoring and governance

- Schedule feature drift, prediction drift, data-quality, latency, and error
  monitoring.
- Track model version, feature version, threshold, and decision reason per alert.
- Alert on drift and degradation; never retrain or promote automatically without
  an approval gate.
- Store monitoring outputs in durable, queryable storage.

### Phase 6 — Reliability and security

- Add durable Kafka topics, consumer retries, dead-letter handling, and replay.
- Add database migrations, backups, retention, and recovery procedures.
- Add container image scanning, dependency scanning, non-root images, and secret
  management.
- Add resource limits, autoscaling policy, PodDisruptionBudget, and network
  policy for Kubernetes.
- Add load tests and establish p50/p95/p99 latency and throughput budgets.

### Phase 7 — Delivery and production verification

- CI pipeline: lint, type-check, unit tests, integration tests, frontend build,
  image build, vulnerability scan, and migration checks.
- CD pipeline: deploy to staging, smoke test, approve, then deploy production.
- Run end-to-end tests from Kafka event through prediction, alert, case update,
  and frontend display.
- Verify rollback, replay, backup restore, and degraded dependency behavior.
- Publish a release report containing measured evidence, not documentation-only
  claims.

## Accuracy acceptance gates

The model is not production-ready until all values are measured on unseen,
chronological data and approved against business limits:

1. No feature uses the current transaction label or post-decision information.
2. Recall and precision are reported for each evaluation window.
3. Alert volume fits analyst capacity.
4. Precision, recall, calibration, and latency remain within agreed limits for
   every canary window.
5. Threshold and model version are reproducible from stored artifacts.
6. A rollback model and a monitored failure path exist.

## Immediate next implementation order

1. Finish documentation and path cleanup from the repository reorganization.
2. Implement the persistent alert/case data model and Kafka-to-alert workflow.
3. Remove hardcoded dashboard metrics and add API integration tests for live data.
4. Add cost-based multi-window model evaluation and threshold selection.
5. Harden authentication, observability, retries, migrations, and deployment.
6. Re-run the complete verification matrix before calling the platform production
   ready.
