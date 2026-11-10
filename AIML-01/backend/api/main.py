"""FastAPI service for real-time financial fraud scoring and analyst platform."""

import json
import logging
import os
import time
from uuid import uuid4
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fraud_detection.config import PROJECT_ROOT, REPORT_DIR
from persistence.database import (
    initialize_database,
    list_cases,
    persist_scored_event,
    runtime_counts,
    update_case,
)
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from api.predictor import score_transaction
from api.schemas import CaseUpdate, PredictionResponse, TransactionRequest

GRAPH_SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "graph" / "graph_summary.json"
EVALUATION_PATH = REPORT_DIR / "evaluation_results.json"
DATABASE = initialize_database()
LOGGER = logging.getLogger("fraud_detection.api")


def _configured_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOWED_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173"]


def _configured_api_key() -> str | None:
    value = os.getenv("API_KEY", "").strip()
    return value or None


def _is_public_path(path: str) -> bool:
    return path in {
        "/health",
        "/api/v1/health",
        "/readyz",
        "/api/v1/readyz",
        "/docs",
        "/openapi.json",
        "/redoc",
    }


app = FastAPI(
    title="Real-Time Financial Fraud Detection API",
    description=(
        "Scores financial transactions using an XGBoost model trained on "
        "the PaySim dataset. The deployed model uses realtime-safe features."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_configured_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    configured_key = _configured_api_key()
    if configured_key and not _is_public_path(request.url.path):
        supplied_key = request.headers.get("X-API-Key")
        if supplied_key != configured_key:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid API key."},
                headers={"X-Request-ID": request_id},
            )

    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    LOGGER.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": elapsed_ms,
        },
    )
    return response


@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    """Return service health status."""
    return {"status": "healthy", "service": "fraud-detection-api"}


@app.get("/readyz")
@app.get("/api/v1/readyz")
def readiness_check():
    """Report whether required local database and model artifacts are available."""
    database_ready = True
    try:
        runtime_counts(DATABASE)
    except SQLAlchemyError as error:
        database_ready = False
        database_error = str(error)
    model_ready = (PROJECT_ROOT / "models" / "realtime_safe_xgboost.joblib").exists()
    report_ready = EVALUATION_PATH.exists()

    checks = {
        "database": database_ready,
        "model": model_ready,
        "evaluation_report": report_ready,
    }
    if not all(checks.values()):
        detail = {"status": "not_ready", "checks": checks}
        if not database_ready:
            detail["database_error"] = database_error
        raise HTTPException(status_code=503, detail=detail)
    return {"status": "ready", "checks": checks}


@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_transaction(transaction: TransactionRequest) -> PredictionResponse:
    """Score a single financial transaction and persist high-risk alerts."""
    try:
        result = score_transaction(transaction.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))

    persist_scored_event(
        DATABASE,
        event_id=f"api_{result['transaction_id']}",
        event_timestamp=datetime.now(timezone.utc).isoformat(),
        transaction=transaction.model_dump(),
        prediction=result,
    )

    return PredictionResponse(**result)


@app.get("/cases")
@app.get("/alerts")
@app.get("/api/v1/cases")
@app.get("/api/v1/alerts")
def get_cases(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Return analyst cases and flagged fraud alerts."""
    all_cases = list_cases(DATABASE)
    cases = all_cases[offset : offset + limit]
    open_cases = [
        c for c in all_cases
        if c.get("case_status") in ("Under Review", "Escalated", "Open")
    ]
    resolved_cases = [
        c for c in all_cases
        if c.get("case_status") in ("Resolved", "Closed", "False Positive")
    ]

    return {
        "cases": cases,
        "open_count": len(open_cases),
        "resolved_count": len(resolved_cases),
        "total_count": len(all_cases),
        "limit": limit,
        "offset": offset,
    }


@app.patch("/cases/{transaction_id}")
@app.patch("/api/v1/cases/{transaction_id}")
def update_case_status(transaction_id: str, update: CaseUpdate):
    """Update analyst review status and feedback for a case."""
    target_case = update_case(
        DATABASE,
        transaction_id,
        update.case_status,
        update.analyst_feedback,
    )
    if target_case is None:
        raise HTTPException(status_code=404, detail=f"Case {transaction_id} not found.")

    return target_case


@app.get("/graph/summary")
@app.get("/api/v1/graph/summary")
def get_graph_summary():
    """Return entity graph metrics and top fraud-linked accounts."""
    if not GRAPH_SUMMARY_PATH.exists():
        raise HTTPException(status_code=404, detail="Graph summary artifact not found.")

    with open(GRAPH_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/metrics/overview")
@app.get("/api/v1/metrics/overview")
def get_overview_metrics():
    """Return summary KPIs and trend metrics for command center overview."""
    graph_data = {}
    if GRAPH_SUMMARY_PATH.exists():
        with open(GRAPH_SUMMARY_PATH, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

    eval_data = {}
    if EVALUATION_PATH.exists():
        with open(EVALUATION_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

    counts = runtime_counts(DATABASE)
    total_txns = counts["total_transactions"]
    fraud_txns = counts["alert_count"]
    fraud_rate = round((fraud_txns / total_txns) * 100, 2) if total_txns else 0.0

    realtime_eval = eval_data.get("realtime_safe", {})
    avg_precision = realtime_eval.get("test_average_precision", 0.0)
    threshold = realtime_eval.get("threshold", 0.0)

    top_accounts = graph_data.get("top_fraud_linked_accounts", [])[:5]

    return {
        "total_transactions": total_txns,
        "fraud_transactions": fraud_txns,
        "fraud_rate": fraud_rate,
        "model_average_precision": avg_precision,
        "threshold": threshold,
        "top_fraud_linked_accounts": top_accounts,
        "trend_data": [],
    }


@app.get("/metrics/performance")
@app.get("/api/v1/metrics/performance")
def get_performance_metrics():
    """Return model performance metrics and comparison parameters."""
    if not EVALUATION_PATH.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found.")

    with open(EVALUATION_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    realtime = eval_data.get("realtime_safe", {})
    leakage = eval_data.get("leakage_suspect", {})

    tp = realtime.get("true_positives", 0)
    fp = realtime.get("false_positives", 0)
    fn = realtime.get("false_negatives", 0)
    tn = realtime.get("true_negatives", 0)

    tpr = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    return {
        "model_name": "realtime_safe_xgboost",
        "precision": realtime.get("test_precision", 0.0),
        "recall": realtime.get("test_recall", 0.0),
        "average_precision": realtime.get("test_average_precision", 0.0),
        "threshold": realtime.get("threshold", 0.0),
        "f2_score": realtime.get("f2_score", 0.0),
        "alert_rate": realtime.get("alert_rate", 0.0),
        "false_positive_rate": realtime.get("false_positive_rate", 0.0),
        "brier_score": realtime.get("brier_score", 0.0),
        "tpr": tpr,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        },
        "shadow_model": {
            "model_name": "leakage_suspect_xgboost",
            "average_precision": leakage.get("test_average_precision", 0.0),
            "threshold": leakage.get("threshold", 0.0),
        },
        "temporal_windows": eval_data.get("realtime_safe_temporal_windows", []),
    }