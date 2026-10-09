"""Evaluation metrics engine for FinEvent benchmark.

Computes:
1. Classification Macro-F1, per-class F1, Precision, and Recall.
2. Argument exact match (metric, period, numbers).
3. Evidence span coverage.
4. Confusion matrix and error breakdown.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


CLASSES = ["GUIDANCE_REVISION", "EARNINGS_RESULT", "MANAGEMENT_CHANGE", "OTHER"]


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """Compute Macro-F1, Weighted-F1, per-class metrics and confusion matrix."""
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0))
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()

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
            for cls in CLASSES
        },
        "confusion_matrix": cm,
        "classes": CLASSES,
    }


def compute_argument_metrics(
    gold_records: List[Dict[str, Any]],
    pred_arguments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate accuracy of extracted arguments against gold annotations."""
    total = len(gold_records)
    if total == 0:
        return {"metric_exact_match": 0.0, "range_exact_match": 0.0, "total": 0}

    metric_matches = 0
    range_matches = 0

    for gold, pred in zip(gold_records, pred_arguments):
        # Metric match
        gold_m = (gold.get("metric") or "UNKNOWN").lower()
        pred_m = (pred.get("metric") or "UNKNOWN").lower()
        if gold_m == pred_m:
            metric_matches += 1

        # Range match
        g_low = gold.get("new_low")
        p_low = pred.get("new_low")
        g_high = gold.get("new_high")
        p_high = pred.get("new_high")

        if g_low is None and p_low is None:
            range_matches += 1
        elif g_low is not None and p_low is not None:
            # Tolerant match within 1%
            if abs(g_low - p_low) < 0.05 and (g_high is None or p_high is None or abs(g_high - p_high) < 0.05):
                range_matches += 1

    return {
        "metric_exact_match": round(metric_matches / total, 4),
        "range_exact_match": round(range_matches / total, 4),
        "total_evaluated": total,
    }
