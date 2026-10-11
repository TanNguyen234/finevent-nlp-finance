"""Business logic and operational service layer for FinEvent API.

Maintains clean separation between:
1. Operational event storage (data/processed/operational_events.jsonl)
2. Evaluation benchmark sets (data/labels/ / data/synthetic/)
Provides live model inference, dynamic confidence scoring, and guidance change queries.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional
import os
import joblib
from src.extract.arguments import extract_multi_metric_arguments
from src.extract.change import compare_company_reporting_periods

logger = logging.getLogger(__name__)

OPERATIONAL_DATA_PATH = pathlib.Path("data/processed/operational_events.jsonl")
MODEL_PATH = pathlib.Path("models/baseline_tfidf_logreg.joblib")
E2_DEBERTA_PATH = pathlib.Path("models/E2_DeBERTa_v3_small")
E2_FINBERT_PATH = pathlib.Path("models/E2_FinBERT")


class FinEventService:
    """Service handling operational event queries and inference across E1/E2 backends."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.events_by_id: Dict[str, Dict[str, Any]] = {}
        self.events_by_cik: Dict[str, List[Dict[str, Any]]] = {}
        self.dataset_mode = "REAL_SEC"
        self.backend_type = os.environ.get("FINEVENT_MODEL_BACKEND", "E1_TFIDF")
        self.model_name = "E1_TFIDF_LogReg"
        self.cloud_weights_status = "CLOUD_STORAGE_ONLY"
        self._load_data()
        self.model = self._load_model()

    def _load_data(self) -> None:
        target_path = OPERATIONAL_DATA_PATH
        if not target_path.exists():
            target_path = pathlib.Path("data/labels/gold.jsonl")
            self.dataset_mode = "VERIFIED_SEEDS"

        if not target_path.exists():
            logger.warning("No operational or gold events found.")
            return

        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.events.append(item)
                    self.events_by_id[item["id"]] = item
                    cik = item["cik"]
                    self.events_by_cik.setdefault(cik, []).append(item)

        logger.info("Loaded %d events across %d companies (Mode: %s).", len(self.events), len(self.events_by_cik), self.dataset_mode)

    def _load_model(self):
        # 1. Configured for E2 neural models
        if self.backend_type in {"E2_DEBERTA", "E2_FINBERT"}:
            target_neural_path = E2_DEBERTA_PATH if self.backend_type == "E2_DEBERTA" else E2_FINBERT_PATH
            if target_neural_path.exists():
                try:
                    from transformers import pipeline
                    logger.info("Loading neural model %s from %s", self.backend_type, target_neural_path)
                    pipe = pipeline("text-classification", model=str(target_neural_path))
                    self.model_name = self.backend_type
                    self.cloud_weights_status = "LOADED_LOCAL"
                    return pipe
                except Exception as exc:
                    logger.warning("Failed to load neural weights for %s: %s", self.backend_type, exc)
            logger.info(
                "Neural weights for %s not found locally (Rule 1: weights retained on Cloud). Falling back to E1 TF-IDF.",
                self.backend_type,
            )
            self.cloud_weights_status = "CLOUD_STORAGE_ONLY (FALLBACK_TO_E1)"

        # 2. Default E1 TF-IDF Logistic Regression
        if MODEL_PATH.exists():
            logger.info("Loading baseline model from %s", MODEL_PATH)
            self.model_name = "E1_TFIDF_LogReg"
            return joblib.load(MODEL_PATH)

        logger.warning("No model found. Falling back to E0 Rule Heuristic.")
        self.model_name = "E0_Rule_Heuristic"
        return None

    def get_service_status(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "service": "FinEvent API",
            "active_model": self.model_name,
            "configured_backend": self.backend_type,
            "cloud_weights_status": self.cloud_weights_status,
            "dataset_mode": self.dataset_mode,
            "total_events_indexed": len(self.events),
            "total_loaded_events": len(self.events),
        }

    def list_companies(self) -> List[Dict[str, str]]:
        seen = {}
        for ev in self.events:
            cik = ev["cik"]
            if cik not in seen:
                seen[cik] = ev.get("company_name", "UNKNOWN")
        return [{"cik": c, "company_name": name} for c, name in seen.items()]

    def get_events(self, cik: str, limit: int = 50) -> List[Dict[str, Any]]:
        clean_cik = cik.strip().zfill(10)
        items = self.events_by_cik.get(clean_cik, [])
        return items[:limit]

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        return self.events_by_id.get(event_id)

    def get_changes(self, cik: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        clean_cik = cik.strip().zfill(10)
        company_events = self.events_by_cik.get(clean_cik, [])
        return compare_company_reporting_periods(company_events, from_date, to_date)

    def predict_event(self, text: str) -> Dict[str, Any]:
        """Classify event paragraph and extract multiple arguments."""
        multi_args = extract_multi_metric_arguments(text)
        primary_args = multi_args[0] if multi_args else {}

        if self.model is not None:
            pred_class = str(self.model.predict([text])[0])
            probas = self.model.predict_proba([text])[0]
            classes = self.model.classes_
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probas)}
            confidence = prob_dict.get(pred_class, 0.75)
            model_ver = "E1_TFIDF_LogReg"
        else:
            t_low = text.lower()
            if any(k in t_low for k in ["revis", "expects revenue to", "expect gross margin", "guidance", "outlook"]):
                pred_class = "GUIDANCE_REVISION"
            elif any(k in t_low for k in ["posted quarterly", "reported financial results"]):
                pred_class = "EARNINGS_RESULT"
            elif any(k in t_low for k in ["appointed", "resigned", "depart"]):
                pred_class = "MANAGEMENT_CHANGE"
            else:
                pred_class = "OTHER"
            confidence = 0.80
            prob_dict = {pred_class: 0.80}
            model_ver = "E0_Rule_Heuristic"

        evidence_span = primary_args.get("evidence") or text[:120]

        return {
            "event_type": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "arguments": primary_args,
            "multi_arguments": multi_args,
            "evidence_span": evidence_span,
            "model_version": model_ver,
        }


service = FinEventService()
