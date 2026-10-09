"""Baseline models training and evaluation (E0: Rule & E1: TF-IDF + LogReg).

Evaluates on both:
1. Synthetic sanity benchmark (data/synthetic/)
2. Verified seeds benchmark (data/labels/)
Explicitly tracks data origin and prevents claiming synthetic scores as real SEC performance.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List

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
        if any(k in t_low for k in ["posted quarterly", "reported financial results", "quarterly revenue of", "diluted earnings per share", "march quarter record", "quarterly diluted earnings"]):
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
            min_df=1,
            max_features=3000,
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


def evaluate_split_dataset(data_dir: pathlib.Path, origin_label: str) -> Dict[str, Any]:
    """Train and evaluate baselines on a specific dataset partition."""
    train_df = pd.read_json(data_dir / "train.jsonl", lines=True)
    val_df = pd.read_json(data_dir / "val.jsonl", lines=True)
    test_df = pd.read_json(data_dir / "test.jsonl", lines=True)

    # 1. E0 Rule
    e0 = RuleBaselineE0()
    e0_test_preds = e0.predict(test_df["text"].tolist())
    e0_test_metrics = compute_classification_metrics(test_df["event_type"].tolist(), e0_test_preds)

    # 2. E1 TF-IDF + LogReg
    e1_pipe = train_e1_tfidf_logreg(train_df)
    e1_test_preds = e1_pipe.predict(test_df["text"].tolist()).tolist()
    e1_test_metrics = compute_classification_metrics(test_df["event_type"].tolist(), e1_test_preds)

    return {
        "dataset_origin": origin_label,
        "sample_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "E0_Rule": e0_test_metrics,
        "E1_TFIDF_LogReg": e1_test_metrics,
        "model_pipeline": e1_pipe,
    }


def run_all_baselines() -> None:
    # 1. Synthetic sanity benchmark
    syn_dir = pathlib.Path("data/synthetic")
    if (syn_dir / "train.jsonl").exists():
        syn_results = evaluate_split_dataset(syn_dir, "SYNTHETIC_TEMPLATE")
        with open(REPORTS_DIR / "synthetic_baseline_metrics.json", "w", encoding="utf-8") as f:
            clean_res = {k: v for k, v in syn_results.items() if k != "model_pipeline"}
            json.dump(clean_res, f, indent=2)
        print("Completed Synthetic Benchmark:", clean_res["E1_TFIDF_LogReg"]["macro_f1"])

    # 2. Real verified seeds benchmark
    real_dir = pathlib.Path("data/labels")
    if (real_dir / "train.jsonl").exists():
        real_results = evaluate_split_dataset(real_dir, "MANUALLY_AUTHORED_SEEDS")
        # Save production E1 model
        joblib.dump(real_results["model_pipeline"], MODELS_DIR / "baseline_tfidf_logreg.joblib")
        with open(REPORTS_DIR / "real_baseline_metrics.json", "w", encoding="utf-8") as f:
            clean_res = {k: v for k, v in real_results.items() if k != "model_pipeline"}
            json.dump(clean_res, f, indent=2)
        print("Completed Real Seeds Benchmark:", clean_res["E1_TFIDF_LogReg"]["macro_f1"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_baselines()
