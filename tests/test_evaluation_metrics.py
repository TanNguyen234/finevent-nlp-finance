"""Evaluation metrics and benchmark integrity tests."""

from __future__ import annotations

import json
import pathlib
import pytest
from src.eval.metrics import (
    compute_classification_metrics,
    compute_argument_metrics_by_id,
)


def test_missing_boundary_penalty():
    """Verify that predictions missing a required range boundary are penalized."""
    gold = [
        {"id": "s1", "metric": "revenue", "new_low": 50.0, "new_high": 55.0},
    ]
    # Prediction only has single value (low bound), missing upper bound
    pred_partial = [
        {"id": "s1", "metric": "revenue", "new_low": 50.0, "new_high": None},
    ]
    res = compute_argument_metrics_by_id(gold, pred_partial)
    assert res["boundary_penalties"] == 1
    assert res["range_exact_match"] == 0.0


def test_id_alignment_and_no_silent_truncation():
    """Verify evaluation evaluates all IDs and does not silently truncate."""
    gold = [
        {"id": "s1", "metric": "revenue", "new_low": 50.0, "new_high": 55.0},
        {"id": "s2", "metric": "gross_margin", "new_low": 40.0, "new_high": 42.0},
    ]
    pred = [
        {"id": "s1", "metric": "revenue", "new_low": 50.0, "new_high": 55.0},
    ]
    res = compute_argument_metrics_by_id(gold, pred)
    assert res["total_evaluated"] == 2
    assert res["missing_predictions"] == 1
    assert res["metric_exact_match"] == 0.5


def test_no_synthetic_metrics_as_real():
    """Verify report artifacts explicitly track data origin."""
    syn_report = pathlib.Path("reports/synthetic_baseline_metrics.json")
    if syn_report.exists():
        data = json.loads(syn_report.read_text(encoding="utf-8"))
        assert data.get("dataset_origin") == "SYNTHETIC_TEMPLATE"
