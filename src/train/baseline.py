"""Baseline models training and evaluation (E0: Rule & E1: TF-IDF + LogReg).

Implements the deterministic benchmark floor and lightweight statistical NLP model.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from src.eval.metrics import compute_classification_metrics, CLASSES

logger = logging.getLogger(__name__)

MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = pathlib.Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class RuleBaselineE0:
    """E0: Deterministic keyword rule baseline."""

    def predict_one(self, text: str) -> str:
        t_low = text.lower()
        if any(k in t_low for k in ["appoint", "depart", "step down", "resigned", "officer", "board of directors", "ceo", "cfo", "elected"]):
            return "MANAGEMENT_CHANGE"
        if any(k in t_low for k in ["revis", "expects revenue to", "expect gross margin", "guidance", "outlook", "forecasts total", "raises total revenue"]):
            return "GUIDANCE_REVISION"
        if any(k in t_low for k in ["posted quarterly", "reported financial results", "quarterly revenue of", "diluted earnings per share", "march quarter record"]):
            return "EARNINGS_RESULT"
        return "OTHER"

    def predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t) for t in texts]


def train_e1_tfidf_logreg(
    train_df: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """E1: TF-IDF n-grams + Logistic Regression."""
    pipeline = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=4000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            C=1.0,
            random_state=random_state,
        ),
    )
    pipeline.fit(train_df["text"], train_df["event_type"])
    return pipeline


def run_all_baselines() -> Dict[str, Any]:
    """Train and evaluate E0 and E1 on train/val/test splits."""
    train_df = pd.read_json("data/labels/train.jsonl", lines=True)
    val_df = pd.read_json("data/labels/val.jsonl", lines=True)
    test_df = pd.read_json("data/labels/test.jsonl", lines=True)

    print(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 1. E0: Rule Baseline
    e0_model = RuleBaselineE0()
    e0_val_preds = e0_model.predict(val_df["text"].tolist())
    e0_test_preds = e0_model.predict(test_df["text"].tolist())

    e0_val_metrics = compute_classification_metrics(val_df["event_type"].tolist(), e0_val_preds)
    e0_test_metrics = compute_classification_metrics(test_df["event_type"].tolist(), e0_test_preds)

    print("\n--- E0: Rule Baseline Results ---")
    print(f"Validation Macro-F1: {e0_val_metrics['macro_f1']:.4f}")
    print(f"Test Macro-F1:       {e0_test_metrics['macro_f1']:.4f}")

    # 2. E1: TF-IDF + Logistic Regression
    e1_pipeline = train_e1_tfidf_logreg(train_df)
    e1_val_preds = e1_pipeline.predict(val_df["text"].tolist()).tolist()
    e1_test_preds = e1_pipeline.predict(test_df["text"].tolist()).tolist()

    e1_val_metrics = compute_classification_metrics(val_df["event_type"].tolist(), e1_val_preds)
    e1_test_metrics = compute_classification_metrics(test_df["event_type"].tolist(), e1_test_preds)

    print("\n--- E1: TF-IDF + Logistic Regression Results ---")
    print(f"Validation Macro-F1: {e1_val_metrics['macro_f1']:.4f}")
    print(f"Test Macro-F1:       {e1_test_metrics['macro_f1']:.4f}")

    # Save model artifact
    model_save_path = MODELS_DIR / "baseline_tfidf_logreg.joblib"
    joblib.dump(e1_pipeline, model_save_path)
    print(f"\nSaved E1 model to {model_save_path}")

    # Compile benchmark summary
    summary = {
        "E0_Rule": {
            "validation": e0_val_metrics,
            "test": e0_test_metrics,
        },
        "E1_TFIDF_LogReg": {
            "validation": e1_val_metrics,
            "test": e1_test_metrics,
        },
    }

    with open(REPORTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_baselines()
