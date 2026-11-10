"""Regression tests for Evidently 0.7 drift metric extraction."""

from monitoring.drift_report import extract_drift_summary


def test_extract_drift_summary_from_evidently_v07_metrics():
    report_dict = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "config": {
                    "type": "evidently:metric_v2:DriftedColumnsCount",
                    "drift_share": 0.5,
                },
                "value": {
                    "count": 2.0,
                    "share": 2 / 7,
                },
            },
            {
                "metric_name": (
                    "ValueDrift(column=amount,method=Wasserstein,"
                    "threshold=0.1)"
                ),
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "amount",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 2.4613,
            },
            {
                "metric_name": (
                    "ValueDrift(column=type,method=Jensen-Shannon,"
                    "threshold=0.1)"
                ),
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "type",
                    "method": "Jensen-Shannon distance",
                    "threshold": 0.1,
                },
                "value": 0.0777,
            },
        ]
    }

    summary = extract_drift_summary(report_dict)

    assert summary["drifted_columns"] == ["amount"]
    assert summary["drifted_feature_count"] == 1
    assert summary["monitored_feature_count"] == 7
    assert summary["drifted_feature_fraction"] == 1 / 7
    assert summary["feature_results"]["amount"]["drift_detected"]
    assert not summary["feature_results"]["type"]["drift_detected"]