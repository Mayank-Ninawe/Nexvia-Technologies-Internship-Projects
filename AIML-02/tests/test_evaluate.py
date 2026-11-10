"""Tests for classification evaluation metrics."""

import torch

from ml.classification.dataset import LABEL_COLUMNS
from ml.classification.evaluate import compute_metrics


def test_compute_metrics_returns_expected_labels():
    logits = torch.randn(16, len(LABEL_COLUMNS))
    targets = torch.randint(0, 2, (16, len(LABEL_COLUMNS))).float()

    metrics = compute_metrics(logits, targets, threshold=0.5)

    assert metrics["test_samples"] == 16
    assert set(metrics["per_label"].keys()) == set(LABEL_COLUMNS)


def test_compute_metrics_handles_label_without_positives():
    logits = torch.randn(8, len(LABEL_COLUMNS))
    targets = torch.zeros(8, len(LABEL_COLUMNS))

    metrics = compute_metrics(logits, targets, threshold=0.5)
    hernia_metrics = metrics["per_label"]["Hernia"]

    assert hernia_metrics["support_positive"] == 0
    assert hernia_metrics["auroc"] is None
    assert "No positive test samples" in hernia_metrics["note"]


def test_compute_metrics_handles_label_without_negatives():
    logits = torch.randn(8, len(LABEL_COLUMNS))
    targets = torch.ones(8, len(LABEL_COLUMNS))

    metrics = compute_metrics(logits, targets, threshold=0.5)
    hernia_metrics = metrics["per_label"]["Hernia"]

    assert hernia_metrics["support_positive"] == 8
    assert hernia_metrics["auroc"] is None
    assert "Only one target class" in hernia_metrics["note"]


def test_compute_metrics_perfect_classifier():
    targets = torch.zeros(32, len(LABEL_COLUMNS))
    targets[:16, 1] = 1.0

    logits = torch.zeros(32, len(LABEL_COLUMNS))
    logits[:16, 1] = 10.0
    logits[16:, 1] = -10.0

    metrics = compute_metrics(logits, targets, threshold=0.5)
    cardiomegaly = metrics["per_label"]["Cardiomegaly"]

    assert cardiomegaly["support_positive"] == 16
    assert cardiomegaly["auroc"] == 1.0
    assert cardiomegaly["average_precision"] == 1.0
    assert cardiomegaly["f1"] == 1.0