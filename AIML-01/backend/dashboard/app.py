"""SentinelGraph analyst dashboard.

The dashboard is intentionally evidence-led: historical PaySim labels, local
stream output, evaluation artifacts, and live API responses are kept distinct.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from persistence.database import (
    initialize_database,
    list_scored_events,
    persist_scored_event,
    runtime_counts,
    update_case,
)
from persistence.database import (
    list_cases as list_persisted_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_URL = "http://127.0.0.1:8000"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "realtime_safe_xgboost.joblib"
EVALUATION_PATH = PROJECT_ROOT / "reports" / "evaluation_results.json"
TRAINING_SUMMARY_PATH = PROJECT_ROOT / "reports" / "training_summary.json"
GRAPH_SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "graph" / "graph_summary.json"
GRAPH_EDGES_PATH = PROJECT_ROOT / "artifacts" / "graph" / "edge_summary.csv"
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

st.set_page_config(
    page_title="SentinelGraph | Fraud Detection",
    page_icon="SG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --navy:#091a2d; --navy-2:#102943; --blue:#2166d1; --blue-2:#e9f1ff;
            --ink:#142338; --muted:#708096; --line:#e2e8f0; --page:#f6f8fb;
            --green:#177245; --amber:#a65d00; --red:#b42318; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:var(--page); color:var(--ink); }
    [data-testid="stSidebar"] { background:var(--navy); border-right:1px solid #173451; }
    [data-testid="stSidebar"] * { color:#eaf1fa; }
    [data-testid="stSidebar"] .stCaption { color:#92a7c2; }
    [data-testid="stSidebar"] .stButton { margin:.12rem 0; }
    [data-testid="stSidebar"] .stButton button { width:100%; text-align:left; border:1px solid transparent;
      background:transparent; color:#c9d6e6; border-radius:9px; padding:.62rem .8rem; }
    [data-testid="stSidebar"] .stButton button:hover { background:var(--navy-2); color:#fff; border-color:#285078; }
    [data-testid="stSidebar"] .stButton button[kind="primary"] { background:#1d568f; color:#fff; border-color:#3c80bc; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.025em; color:var(--ink); }
    h1 { font-size:2.15rem; margin:.1rem 0 .25rem; } h2 { font-size:1.35rem; margin-top:1.3rem; }
    h3 { font-size:1.04rem; } .eyebrow { color:var(--blue); font-size:.7rem; font-weight:700;
      letter-spacing:.14em; text-transform:uppercase; } .subtitle { color:var(--muted); font-size:1rem; margin-bottom:1.25rem; }
    .brand { padding:.45rem .1rem 1rem; } .brand-mark { display:inline-flex; width:36px; height:36px;
      align-items:center; justify-content:center; border:1px solid #5489bd; border-radius:11px;
      color:#d9ecff; font-family:'Space Grotesk'; font-weight:700; margin-right:.55rem; }
    .brand-name { font-family:'Space Grotesk'; font-weight:700; font-size:1.25rem; vertical-align:middle; }
    .brand-sub { color:#91a8c1; font-size:.73rem; line-height:1.35; margin-top:.65rem; }
    .panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1.15rem 1.25rem;
      box-shadow:0 5px 18px rgba(24,45,75,.035); } .panel-tight { padding:.85rem 1rem; }
    .status-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.75rem 1rem; }
    .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; background:#97a5b6; }
    .status-dot.ok { background:#22a06b; } .status-dot.warn { background:#e59a24; } .status-dot.bad { background:#d9534f; }
    .small-note { color:var(--muted); font-size:.8rem; } .hero-note { color:#52657d; font-size:.9rem; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.75rem 1rem; }
    [data-testid="stMetricLabel"] { color:var(--muted); } [data-testid="stMetricValue"] { color:var(--ink); }
    .risk-panel { border:1px solid #cdddf4; background:linear-gradient(135deg,#f8fbff,#edf4ff); border-radius:14px; padding:1.2rem; }
    .risk-number { color:var(--blue); font-family:'Space Grotesk'; font-size:2rem; font-weight:700; }
    .badge { display:inline-block; border-radius:99px; padding:.22rem .6rem; font-size:.73rem; font-weight:700; }
    .badge-red { color:#9b241a; background:#fdeceb; } .badge-green { color:#176c45; background:#e8f7ef; }
    .badge-amber { color:#875000; background:#fff3df; } .arch-node { border:1px solid var(--line); background:#fff;
      border-radius:11px; padding:.75rem; text-align:center; min-height:70px; } .arch-node strong { display:block; }
    .arch-arrow { color:#91a1b5; text-align:center; padding-top:1.1rem; font-size:1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


@st.cache_data(ttl=300, show_spinner=False)
def historical_summary() -> dict[str, Any]:
    if not RAW_DATA_PATH.exists():
        return {"available": False, "reason": f"Missing dataset: {RAW_DATA_PATH}"}
    row_count = fraud_count = 0
    type_counts: dict[str, int] = {}
    step_counts: dict[int, int] = {}
    try:
        for chunk in pd.read_csv(RAW_DATA_PATH, usecols=["step", "type", "isFraud"], chunksize=250_000):
            row_count += len(chunk)
            fraud_count += int(chunk["isFraud"].sum())
            for key, count in chunk["type"].value_counts().items():
                type_counts[str(key)] = type_counts.get(str(key), 0) + int(count)
            for key, count in chunk["step"].value_counts().items():
                step_counts[int(key)] = step_counts.get(int(key), 0) + int(count)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        return {"available": False, "reason": f"Could not read dataset: {error}"}
    return {
        "available": True, "row_count": row_count, "fraud_count": fraud_count,
        "genuine_count": row_count - fraud_count,
        "fraud_rate": fraud_count / row_count if row_count else 0.0,
        "type_counts": type_counts, "step_counts": step_counts,
    }


@st.cache_data(ttl=300, show_spinner=False)
def recent_historical_rows(limit: int = 25) -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        return pd.DataFrame()
    try:
        tail: pd.DataFrame | None = None
        for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=250_000):
            tail = pd.concat([tail, chunk], ignore_index=True).tail(limit)
        return tail if tail is not None else pd.DataFrame()
    except (OSError, pd.errors.ParserError, ValueError):
        return pd.DataFrame()


DATABASE = initialize_database()


def load_cases() -> list[dict]:
    return list_persisted_cases(DATABASE)


def load_streamed_records() -> list[dict]:
    return list_scored_events(DATABASE)


def request_payload(transaction_id: str, transaction_type: str, amount: float, step: int,
                    sender_id: str, receiver_id: str, sender_old: float, sender_new: float,
                    receiver_old: float, receiver_new: float) -> dict:
    return {
        "transaction_id": transaction_id.strip(), "step": int(step), "type": transaction_type,
        "amount": float(amount), "nameOrig": sender_id.strip(), "oldbalanceOrg": float(sender_old),
        "newbalanceOrig": float(sender_new), "nameDest": receiver_id.strip(),
        "oldbalanceDest": float(receiver_old), "newbalanceDest": float(receiver_new),
    }


@st.cache_data(ttl=15, show_spinner=False)
def check_api_status() -> dict[str, str]:
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        health.raise_for_status()
    except requests.exceptions.RequestException as error:
        return {"state": "API unavailable", "detail": str(error)}
    payload = request_payload("dashboard_readiness_check", "TRANSFER", 1.0, 1,
                              "C1000001", "C2000001", 1.0, 0.0, 0.0, 1.0)
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
        response.raise_for_status()
        body = response.json()
        if not {"risk_score", "threshold", "model_name"}.issubset(body):
            return {"state": "API reachable; prediction not verified",
                    "detail": "The response is missing expected scoring fields."}
        return {"state": "Model loaded and prediction ready", "detail": f"Serving {body['model_name']}."}
    except requests.exceptions.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        return {"state": "API reachable; model not ready", "detail": detail}
    except (requests.exceptions.RequestException, ValueError, TypeError) as error:
        return {"state": "API reachable; prediction not verified", "detail": str(error)}


def status_html(status: dict[str, str]) -> str:
    state = status["state"]
    dot = "ok" if state == "Model loaded and prediction ready" else ("bad" if state == "API unavailable" else "warn")
    return (f'<div class="status-card"><span class="status-dot {dot}"></span><strong>{html.escape(state)}</strong>'
            f'<br><span class="small-note">{html.escape(status["detail"])}</span></div>')


def show_prediction(prediction: dict, cases: list[dict], payload: dict) -> None:
    probability = float(prediction["fraud_probability"])
    risk = float(prediction["risk_score"])
    st.markdown('<div class="risk-panel">', unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f'<div class="small-note">MODEL RISK SCORE</div><div class="risk-number">{risk:.2f}<span style="font-size:1rem"> / 100</span></div>', unsafe_allow_html=True)
        badge = "badge-red" if prediction["is_alert"] else "badge-green"
        label = "Review recommended" if prediction["is_alert"] else "No alert at threshold"
        st.markdown(f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True)
    with right:
        st.write(f"**Prediction:** {'Alert' if prediction['is_alert'] else 'No alert'}")
        st.write(f"**Fraud probability:** `{probability:.4%}`  ·  **Decision threshold:** `{float(prediction['threshold']):.4%}`")
        st.write(f"**Recommended action:** {prediction['recommended_action']}")
        st.caption("This is a model risk estimate for analyst triage, not proof that a transaction is fraudulent.")
    st.markdown("</div>", unsafe_allow_html=True)
    if prediction["is_alert"]:
        persist_scored_event(
            DATABASE,
            event_id=f"manual_{prediction['transaction_id']}",
            event_timestamp=datetime.now(timezone.utc).isoformat(),
            transaction=payload,
            prediction=prediction,
        )
        st.info("The analyst case was persisted in the shared database.")


def show_overview(summary: dict[str, Any], streamed: list[dict], cases: list[dict]) -> None:
    st.markdown('<div class="eyebrow">Command center / evidence overview</div>', unsafe_allow_html=True)
    st.title("Overview")
    st.markdown('<div class="subtitle">A grounded view of historical fraud labels, local stream output, and model readiness.</div>', unsafe_allow_html=True)
    st.markdown(status_html(check_api_status()), unsafe_allow_html=True)
    if not summary.get("available"):
        st.warning(summary["reason"])
        return
    alerts = [r for r in streamed if r.get("prediction", {}).get("is_alert") is True]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical transactions", f"{summary['row_count']:,}")
    c2.metric("Historical fraud labels", f"{summary['fraud_count']:,}")
    c3.metric("Historical fraud rate", f"{summary['fraud_rate']:.3%}")
    c4.metric("Persisted stream alerts", f"{len(alerts):,}")
    st.caption("Historical cards use PaySim `isFraud` labels. Stream alerts are persisted consumer output; they are not current live traffic.")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Transaction activity by simulation step")
        st.area_chart(pd.Series(summary["step_counts"], name="Transactions").sort_index(), height=250, color="#2166d1")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Transaction type distribution")
        type_frame = pd.DataFrame.from_dict(summary["type_counts"], orient="index", columns=["Transactions"]).sort_values("Transactions", ascending=False)
        st.bar_chart(type_frame, height=250, color="#5c8fdb")
        st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("Recent historical transactions")
    recent = recent_historical_rows()
    if recent.empty:
        st.info("No historical rows are available.")
    else:
        cols = [c for c in ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"] if c in recent.columns]
        st.dataframe(recent[cols].sort_values("step", ascending=False), width="stretch", hide_index=True)
        st.caption("Historical records include ground-truth labels and are not live predictions.")
    if cases:
        st.subheader("Local case queue")
        frame = pd.DataFrame(cases)
        cols = [c for c in ["transaction_id", "risk_score", "case_status", "source", "created_at"] if c in frame.columns]
        st.dataframe(frame[cols].sort_values("created_at", ascending=False).head(8), width="stretch", hide_index=True)


def show_analyze(cases: list[dict]) -> None:
    st.markdown('<div class="eyebrow">Live scoring / highest confidence workflow</div>', unsafe_allow_html=True)
    st.title("Analyze transaction")
    st.markdown('<div class="subtitle">Submit only transaction fields accepted by the realtime-safe FastAPI scoring schema. Ground-truth labels are never sent.</div>', unsafe_allow_html=True)
    with st.form("transaction_form"):
        c1, c2, c3 = st.columns(3)
        transaction_id = c1.text_input("Transaction ID", "txn_demo_001")
        transaction_type = c2.selectbox("Transaction type", TRANSACTION_TYPES)
        step = c3.number_input("Simulation step", min_value=1, value=100, step=1)
        c1, c2 = st.columns(2)
        sender_id = c1.text_input("Sender account", "C1000001")
        receiver_id = c2.text_input("Receiver account", "C2000001")
        c1, c2 = st.columns(2)
        amount = c1.number_input("Amount", min_value=0.0, value=150000.0, step=100.0)
        sender_old = c2.number_input("Sender balance before", min_value=0.0, value=200000.0)
        c1, c2 = st.columns(2)
        sender_new = c1.number_input("Sender balance after", min_value=0.0, value=50000.0)
        receiver_old = c2.number_input("Receiver balance before", min_value=0.0, value=0.0)
        receiver_new = st.number_input("Receiver balance after", min_value=0.0, value=150000.0)
        submitted = st.form_submit_button("Analyze transaction", type="primary", use_container_width=True)
    if not submitted:
        st.markdown('<div class="panel"><strong>How to use this view</strong><br><span class="hero-note">Enter a transaction, call the live API, then inspect the score, threshold, and recommended action. A failed service call remains an explicit error.</span></div>', unsafe_allow_html=True)
        return
    if not transaction_id.strip() or not sender_id.strip() or not receiver_id.strip():
        st.error("Transaction ID, sender account, and receiver account are required.")
        return
    payload = request_payload(transaction_id, transaction_type, amount, int(step), sender_id, receiver_id,
                              sender_old, sender_new, receiver_old, receiver_new)
    with st.spinner("Sending transaction to the FastAPI model service..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=12)
            response.raise_for_status()
            prediction = response.json()
        except requests.exceptions.HTTPError as error:
            detail = error.response.text if error.response is not None else str(error)
            st.error(f"Prediction service rejected the request: {detail}")
            return
        except requests.exceptions.RequestException as error:
            st.error(f"Prediction service is unavailable: {error}")
            return
        except ValueError as error:
            st.error(f"Prediction service returned invalid JSON: {error}")
            return
    show_prediction(prediction, cases, payload)


def show_alerts(cases: list[dict], streamed: list[dict]) -> None:
    st.markdown('<div class="eyebrow">Investigation queue / persisted evidence</div>', unsafe_allow_html=True)
    st.title("Fraud alerts")
    alerts = [r for r in streamed if r.get("prediction", {}).get("is_alert") is True]
    if not alerts:
        st.info("No persisted stream alerts are available. This view does not fabricate live alerts.")
        return
    query = st.text_input("Search alerts", placeholder="Transaction ID, account, or type")
    rows = []
    for record in alerts:
        tx = record.get("transaction", {}); prediction = record.get("prediction", {})
        row = {"Transaction ID": prediction.get("transaction_id"), "Amount": tx.get("amount"),
               "Type": tx.get("type"), "Risk score": prediction.get("risk_score"),
               "Status": "Alert", "Processed at": record.get("processed_at"),
               "Source": record.get("source", "stream consumer")}
        rows.append(row)
    frame = pd.DataFrame(rows)
    if query.strip():
        mask = frame.astype(str).apply(lambda column: column.str.contains(query, case=False, na=False)).any(axis=1)
        frame = frame[mask]
    st.caption(f"{len(frame):,} persisted alert record(s) shown. Analyst review status is only tracked in local cases.")
    st.dataframe(frame.sort_values("Processed at", ascending=False), width="stretch", hide_index=True)
    if st.button("Create local cases for displayed alerts", type="secondary"):
        created = 0
        for record in alerts:
            prediction = record.get("prediction", {}); tx = record.get("transaction", {})
            if not any(case.get("transaction_id") == prediction.get("transaction_id") for case in cases):
                persist_scored_event(
                    DATABASE,
                    event_id=f"dashboard_{prediction['transaction_id']}",
                    event_timestamp=record.get("processed_at"),
                    transaction=tx,
                    prediction=prediction,
                )
                created += 1
        if created:
            st.success(f"Created {created} database case(s).")
        else:
            st.info("All persisted alerts already have local cases.")


def show_model_performance() -> None:
    st.markdown('<div class="eyebrow">Evidence and governance / primary artifact</div>', unsafe_allow_html=True)
    st.title("Model performance")
    evaluation = load_json(EVALUATION_PATH, {}); training = load_json(TRAINING_SUMMARY_PATH, {})
    metrics = evaluation.get("realtime_safe") if isinstance(evaluation, dict) else None
    if not isinstance(metrics, dict):
        st.warning(f"Evaluation artifact unavailable: {EVALUATION_PATH}")
        return
    precision = float(metrics["test_precision"]); recall = float(metrics["test_recall"])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test PR-AUC / AP", f"{float(metrics['test_average_precision']):.4f}")
    c2.metric("Test precision", f"{precision:.3f}"); c3.metric("Test recall", f"{recall:.3f}"); c4.metric("Derived test F1", f"{f1:.3f}")
    st.caption("Average precision is the area-under-precision-recall summary. These values are from the held-out test split, not validation.")
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True); st.subheader("Test confusion matrix")
        matrix = pd.DataFrame([[metrics["true_negatives"], metrics["false_positives"]],
                               [metrics["false_negatives"], metrics["true_positives"]]],
                              index=["Genuine", "Fraud"], columns=["Predicted genuine", "Predicted fraud"])
        st.dataframe(matrix, width="stretch"); st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True); st.subheader("Evaluation provenance")
        st.write(f"**Algorithm:** `{metrics['model']}`"); st.write(f"**Feature set:** `{metrics['feature_set']}`")
        st.write(f"**Decision threshold:** `{float(metrics['threshold']):.6%}`")
        st.write(f"**Test rows:** `{int(metrics['true_negatives']) + int(metrics['false_positives']) + int(metrics['false_negatives']) + int(metrics['true_positives']):,}`")
        st.write(f"**Model artifact:** `{'Available' if MODEL_PATH.exists() else 'Not found'}`")
        st.write(f"**Train / validation rows:** `{training.get('train_rows', 'not recorded'):,} / {training.get('validation_rows', 'not recorded'):,}`" if isinstance(training.get("train_rows"), int) else "**Train / validation rows:** not recorded")
        st.markdown("</div>", unsafe_allow_html=True)
    st.warning("The primary model is realtime-safe, but PaySim is synthetic. Treat these metrics as demonstration evidence, not a production-quality claim.")


def show_graph_analysis() -> None:
    st.markdown('<div class="eyebrow">Historical investigation / relationship evidence</div>', unsafe_allow_html=True)
    st.title("Graph analysis")
    st.markdown('<div class="subtitle">Relationships are derived from the bounded, labelled PaySim graph window. Connectivity alone does not establish a fraud ring.</div>', unsafe_allow_html=True)
    summary = load_json(GRAPH_SUMMARY_PATH, {}); edges = pd.DataFrame()
    if GRAPH_EDGES_PATH.exists():
        try: edges = pd.read_csv(GRAPH_EDGES_PATH)
        except (OSError, pd.errors.ParserError): pass
    if not summary or edges.empty:
        st.info("No saved graph result is available. No illustrative relationships are shown.")
        return
    metrics = summary.get("graph_metrics", {}); c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", f"{metrics.get('node_count', 0):,}"); c2.metric("Relationships", f"{metrics.get('edge_count', 0):,}")
    c3.metric("Fraud-linked relationships", f"{metrics.get('fraud_linked_relationship_count', 0):,}"); c4.metric("Largest component", f"{metrics.get('largest_weakly_connected_component_size', 0):,}")
    fraud_edges = edges[edges["fraud_transaction_count"] > 0].sort_values(["fraud_transaction_count", "total_amount"], ascending=False).head(20)
    dot = ['digraph {', 'graph [bgcolor="transparent", rankdir=LR];',
           'node [shape=box, style="rounded,filled", fillcolor="#e9f1ff", color="#2166d1", fontname="Arial"];',
           'edge [color="#b42318", penwidth=1.8, arrowsize=0.7];']
    for row in fraud_edges.itertuples(index=False):
        source = str(row.nameOrig).replace('"', '\\"'); target = str(row.nameDest).replace('"', '\\"')
        dot.append(f'"{source}" -> "{target}" [label="fraud links: {int(row.fraud_transaction_count)}"];')
    dot.append("}"); st.graphviz_chart("\n".join(dot), width="stretch")
    st.dataframe(fraud_edges, width="stretch", hide_index=True)
    st.caption(summary.get("interpretation", "Historical graph interpretation unavailable."))


def show_system_health() -> None:
    st.markdown('<div class="eyebrow">Operational context / verified status only</div>', unsafe_allow_html=True)
    st.title("System health / architecture")
    st.markdown('<div class="subtitle">File presence is not treated as proof that a service is running. API readiness is verified with live health and prediction checks.</div>', unsafe_allow_html=True)
    api = check_api_status()
    statuses = [("FastAPI scoring API", api["state"], api["detail"]),
                ("Realtime-safe XGBoost artifact", "Available" if MODEL_PATH.exists() else "Not found", str(MODEL_PATH)),
                ("PaySim dataset", "Available" if RAW_DATA_PATH.exists() else "Not found", str(RAW_DATA_PATH)),
                ("Scored event store", f"{runtime_counts(DATABASE)['total_transactions']:,} persisted transaction(s)", "Configured application database."),
                ("Graph analysis artifact", "Available" if GRAPH_SUMMARY_PATH.exists() and GRAPH_EDGES_PATH.exists() else "Not configured", "Saved bounded graph window"),
                ("Drift monitoring", "Not verified", "No live monitoring endpoint is configured in the dashboard.")]
    cols = st.columns(3)
    for index, (name, state, detail) in enumerate(statuses):
        with cols[index % 3]:
            tone = "ok" if state in {"Model loaded and prediction ready", "Available", "Persisted locally"} else ("bad" if state in {"API unavailable", "Not found"} else "warn")
            st.markdown(f'<div class="status-card" style="margin-bottom:.8rem"><span class="status-dot {tone}"></span><strong>{html.escape(name)}</strong><br><span class="small-note">{html.escape(state)} · {html.escape(detail)}</span></div>', unsafe_allow_html=True)
    st.subheader("Verified architecture")
    nodes = [("PaySim dataset", "historical source", "Available" if RAW_DATA_PATH.exists() else "Not found"),
             ("Feature engineering", "realtime-safe features", "Available"),
             ("XGBoost model", "scoring artifact", "Available" if MODEL_PATH.exists() else "Not found"),
             ("FastAPI", "live prediction API", api["state"]),
             ("Streamlit", "analyst workspace", "Connected")]
    for index in range(0, len(nodes), 5):
        row = st.columns(9)
        for offset, (name, detail, state) in enumerate(nodes[index:index + 5]):
            with row[offset * 2]:
                st.markdown(f'<div class="arch-node"><strong>{html.escape(name)}</strong><span class="small-note">{html.escape(detail)}<br>{html.escape(state)}</span></div>', unsafe_allow_html=True)
            if offset < len(nodes[index:index + 5]) - 1:
                with row[offset * 2 + 1]: st.markdown('<div class="arch-arrow">→</div>', unsafe_allow_html=True)
    st.caption("Kafka, Feast, MLflow, and Evidently are documented repository components, but this dashboard does not claim they are running without a verified service check.")


def show_cases(cases: list[dict]) -> None:
    st.markdown('<div class="eyebrow">Analyst workflow / local persistence</div>', unsafe_allow_html=True)
    st.title("Local case management")
    st.caption("Statuses and feedback persist in the configured application database.")
    if not cases:
        st.info("No local cases exist yet. Analyze a transaction or import a persisted stream alert.")
        return
    selected = st.selectbox("Case", [case["transaction_id"] for case in cases])
    index = next(i for i, case in enumerate(cases) if case["transaction_id"] == selected); case = cases[index]
    st.json(case.get("transaction_details", {}))
    statuses = ["New", "Under Review", "Resolved"]
    status = st.selectbox("Case status", statuses, index=statuses.index(case.get("case_status", "New")))
    feedback = st.text_area("Analyst feedback", value=case.get("analyst_feedback", ""))
    if st.button("Save case update", type="primary"):
        update_case(DATABASE, case["transaction_id"], status, feedback)
        st.success("Case update saved.")


summary = historical_summary(); cases = load_cases(); streamed = load_streamed_records()
pages = [("Overview", "01 · Overview"), ("Analyze transaction", "02 · Analyze transaction"),
         ("Fraud alerts", "03 · Fraud alerts"), ("Graph analysis", "04 · Graph analysis"),
         ("Model performance", "05 · Model performance"), ("System health / Architecture", "06 · System health / Architecture"),
         ("Case management", "07 · Case management")]
if "page" not in st.session_state:
    st.session_state.page = "Overview"
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">SG</span><span class="brand-name">SentinelGraph</span><div class="brand-sub">Real-Time Financial Fraud Detection</div></div>', unsafe_allow_html=True)
    st.caption("WORKSPACE")
    for key, label in pages:
        if st.button(label, key=f"nav_{key}", type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key; st.rerun()
    st.divider()
    st.caption("EVIDENCE SOURCES")
    st.caption("PaySim historical labels")
    st.caption("FastAPI model responses")
    st.caption("Local Kafka consumer output")
    st.caption("Saved evaluation artifacts")
    if st.button("Refresh evidence", use_container_width=True):
        st.cache_data.clear(); st.rerun()

if st.session_state.page == "Overview": show_overview(summary, streamed, cases)
elif st.session_state.page == "Analyze transaction": show_analyze(cases)
elif st.session_state.page == "Fraud alerts": show_alerts(cases, streamed)
elif st.session_state.page == "Graph analysis": show_graph_analysis()
elif st.session_state.page == "Model performance": show_model_performance()
elif st.session_state.page == "System health / Architecture": show_system_health()
else: show_cases(cases)
