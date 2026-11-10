"""Generate Evidently data-drift reports for realtime-safe PaySim features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from fraud_detection.config import PROJECT_ROOT, RAW_DATA_PATH

MONITORING_DIR = PROJECT_ROOT / "monitoring_reports"
REPORTS_DIR = MONITORING_DIR / "reports"
SUMMARIES_DIR = MONITORING_DIR / "summaries"
TRIGGERS_DIR = MONITORING_DIR / "triggers"

MONITORED_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "type",
    "is_transfer_or_cash_out",
    "sender_balance_before_zero",
    "receiver_balance_before_zero",
]

RAW_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

DRIFT_FEATURE_FRACTION_THRESHOLD = 0.30
CONTROLLED_AMOUNT_MULTIPLIER = 5.0


def load_monitoring_windows(
    reference_rows: int = 50000,
    current_rows: int = 50000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load two non-overlapping chronological PaySim windows."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"PaySim CSV not found: {RAW_DATA_PATH}")

    total_rows = reference_rows + current_rows

    dataframe = pd.read_csv(
        RAW_DATA_PATH,
        usecols=RAW_COLUMNS,
        nrows=total_rows,
    )

    if len(dataframe) < total_rows:
        raise ValueError(
            "Dataset does not contain enough rows for requested "
            "reference and current windows."
        )

    reference = dataframe.iloc[:reference_rows].copy()
    current = dataframe.iloc[
        reference_rows:reference_rows + current_rows
    ].copy()

    return prepare_features(reference), prepare_features(current)


def prepare_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create monitored realtime-safe features from raw PaySim fields."""
    result = dataframe.copy()

    result["is_transfer_or_cash_out"] = (
        result["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    )
    result["sender_balance_before_zero"] = (
        result["oldbalanceOrg"] == 0
    ).astype(int)
    result["receiver_balance_before_zero"] = (
        result["oldbalanceDest"] == 0
    ).astype(int)

    return result[MONITORED_COLUMNS]


def apply_controlled_drift(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create controlled covariate shift for monitoring demonstration."""
    drifted = dataframe.copy()
    drifted["amount"] = drifted["amount"] * CONTROLLED_AMOUNT_MULTIPLIER
    return drifted


def extract_drift_summary(report_dict: dict) -> dict:
    """Extract drift results from Evidently 0.7 metric output."""
    metric_results = report_dict.get("metrics", [])

    if not metric_results:
        raise ValueError("Evidently report contains no metric results.")

    feature_results = {}
    drifted_columns = []

    for metric in metric_results:
        config = metric.get("config", {})
        metric_type = config.get("type", "")

        if not metric_type.endswith("ValueDrift"):
            continue

        column_name = config.get("column")
        threshold = config.get("threshold")
        drift_score = metric.get("value")

        if column_name is None or threshold is None or drift_score is None:
            continue

        drift_detected = float(drift_score) > float(threshold)

        feature_results[column_name] = {
            "method": config.get("method"),
            "drift_score": float(drift_score),
            "threshold": float(threshold),
            "drift_detected": drift_detected,
        }

        if drift_detected:
            drifted_columns.append(column_name)

    monitored_feature_count = len(MONITORED_COLUMNS)
    drifted_feature_count = len(drifted_columns)
    drifted_feature_fraction = (
        drifted_feature_count / monitored_feature_count
        if monitored_feature_count
        else 0.0
    )

    dataset_drift_detected = (
        drifted_feature_fraction >= DRIFT_FEATURE_FRACTION_THRESHOLD
    )

    return {
        "drifted_columns": drifted_columns,
        "drifted_feature_count": drifted_feature_count,
        "monitored_feature_count": monitored_feature_count,
        "drifted_feature_fraction": drifted_feature_fraction,
        "dataset_drift_detected": dataset_drift_detected,
        "feature_results": feature_results,
    }


def should_request_retraining_review(drift_summary: dict) -> bool:
    """Apply the documented review-trigger policy."""
    return (
        "amount" in drift_summary["drifted_columns"]
        or drift_summary["drifted_feature_fraction"]
        >= DRIFT_FEATURE_FRACTION_THRESHOLD
    )


def save_retraining_review_request(
    report_name: str,
    drift_summary: dict,
) -> Path:
    """Write a human-review request; do not automatically retrain."""
    TRIGGERS_DIR.mkdir(parents=True, exist_ok=True)

    request = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_name": report_name,
        "status": "review_required",
        "reason": (
            "Data drift policy threshold exceeded. "
            "Validate data quality and model performance before retraining "
            "or promoting any replacement model."
        ),
        "policy": {
            "amount_drifted": "amount"
            in drift_summary["drifted_columns"],
            "drifted_feature_fraction": drift_summary[
                "drifted_feature_fraction"
            ],
            "required_fraction_threshold": DRIFT_FEATURE_FRACTION_THRESHOLD,
        },
        "drift_summary": drift_summary,
    }

    request_path = TRIGGERS_DIR / f"{report_name}_retraining_review.json"

    with open(request_path, "w", encoding="utf-8") as file:
        json.dump(request, file, indent=2)

    return request_path


def run_drift_report(
    report_name: str,
    controlled_drift: bool,
) -> dict:
    """Generate a report, write outputs, and evaluate the review trigger."""
    reference, current = load_monitoring_windows()

    if controlled_drift:
        current = apply_controlled_drift(current)

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    report_result = report.run(
        reference_data=reference,
        current_data=current,
    )

    report_dict = report_result.dict()
    drift_summary = extract_drift_summary(report_dict)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    html_path = REPORTS_DIR / f"{report_name}.html"
    json_path = SUMMARIES_DIR / f"{report_name}.json"

    report_result.save_html(str(html_path))

    output = {
        "report_name": report_name,
        "controlled_drift": controlled_drift,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "policy_triggered": should_request_retraining_review(drift_summary),
        "drift_summary": drift_summary,
    }

    if output["policy_triggered"]:
        request_path = save_retraining_review_request(
            report_name=report_name,
            drift_summary=drift_summary,
        )
        output["retraining_review_request"] = str(request_path)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(json.dumps(output, indent=2))
    print(f"HTML report saved to: {html_path}")
    return output


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate an Evidently data-drift report."
    )
    parser.add_argument(
        "--controlled-drift",
        action="store_true",
        help="Multiply current transaction amounts for a controlled drift test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    report_name = (
        "controlled_amount_drift"
        if arguments.controlled_drift
        else "natural_window_drift"
    )

    run_drift_report(
        report_name=report_name,
        controlled_drift=arguments.controlled_drift,
    )