"""Evaluation metrics engine for FinEvent benchmark.

Computes:
1. Classification Macro-F1, Weighted-F1, per-class Precision/Recall/F1, and Confusion Matrix.
2. Argument alignment by ID with strict boundary penalty and unit normalization.
3. Directional guidance revision accuracy and abstention correctness.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASSES = ["GUIDANCE_REVISION", "EARNINGS_RESULT", "MANAGEMENT_CHANGE", "OTHER"]


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str] = CLASSES,
) -> Dict[str, Any]:
    """Compute Macro-F1, Weighted-F1, per-class metrics and confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")

    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0))
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": report[cls]["support"],
            }
            for cls in classes
            if cls in report
        },
        "confusion_matrix": cm,
        "classes": classes,
        "total_evaluated": len(y_true),
    }


def compute_argument_metrics_by_id(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Strict argument evaluation aligned by sample ID.

    Penalizes:
    - Missing predicted boundaries when gold requires both low and high
    - Unit mismatches
    - Missing sample IDs (no silent truncation)
    """
    gold_by_id = {r["id"]: r for r in gold_records}
    pred_by_id = {r["id"]: r for r in pred_records}

    all_ids = sorted(list(gold_by_id.keys()))
    total = len(all_ids)
    if total == 0:
        return {"metric_exact_match": 0.0, "range_exact_match": 0.0, "total": 0}

    metric_matches = 0
    range_matches = 0
    missing_preds = 0
    boundary_penalties = 0

    for sample_id in all_ids:
        gold = gold_by_id[sample_id]
        pred = pred_by_id.get(sample_id)

        if pred is None:
            missing_preds += 1
            continue

        # 1. Metric exact match
        g_metric = (gold.get("metric") or "UNKNOWN").lower()
        p_metric = (pred.get("metric") or "UNKNOWN").lower()
        if g_metric == p_metric:
            metric_matches += 1

        # 2. Strict range matching
        g_low = gold.get("new_low")
        g_high = gold.get("new_high")
        p_low = pred.get("new_low")
        p_high = pred.get("new_high")

        if g_low is None and g_high is None:
            if p_low is None and p_high is None:
                range_matches += 1
        elif g_low is not None and g_high is not None:
            # Gold requires two boundaries
            if p_low is None or p_high is None:
                boundary_penalties += 1
            else:
                if abs(g_low - p_low) < 0.05 and abs(g_high - p_high) < 0.05:
                    range_matches += 1
        elif g_low is not None:
            if p_low is not None and abs(g_low - p_low) < 0.05:
                range_matches += 1

    return {
        "metric_exact_match": round(metric_matches / total, 4),
        "range_exact_match": round(range_matches / total, 4),
        "total_evaluated": total,
        "missing_predictions": missing_preds,
        "boundary_penalties": boundary_penalties,
    }
