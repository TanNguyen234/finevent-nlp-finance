"""Business logic and service layer for FinEvent API."""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, List, Optional
import joblib
from app.schemas import FinancialEvent, GuidanceChange, PredictEventResponse
from src.extract.arguments import extract_guidance_arguments
from src.extract.change import compare_company_reporting_periods

logger = logging.getLogger(__name__)

GOLD_PATH = pathlib.Path("data/labels/gold.jsonl")
MODEL_PATH = pathlib.Path("models/baseline_tfidf_logreg.joblib")


class FinEventService:
    """Service handling event retrieval, change tracking, and live inference."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.events_by_id: Dict[str, Dict[str, Any]] = {}
        self.events_by_cik: Dict[str, List[Dict[str, Any]]] = {}
        self._load_data()
        self.model = self._load_model()

    def _load_data(self) -> None:
        if not GOLD_PATH.exists():
            logger.warning("Data file %s not found.", GOLD_PATH)
            return

        with open(GOLD_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.events.append(item)
                    self.events_by_id[item["id"]] = item
                    cik = item["cik"]
                    self.events_by_cik.setdefault(cik, []).append(item)

        logger.info("Loaded %d events across %d companies.", len(self.events), len(self.events_by_cik))

    def _load_model(self):
        if MODEL_PATH.exists():
            logger.info("Loading baseline model from %s", MODEL_PATH)
            return joblib.load(MODEL_PATH)
        logger.warning("Model file %s not found.", MODEL_PATH)
        return None

    def list_companies(self) -> List[Dict[str, str]]:
        """Return distinct companies in the system."""
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
        """Perform event classification and argument extraction on unseen snippet."""
        args = extract_guidance_arguments(text)

        if self.model is not None:
            pred_class = str(self.model.predict([text])[0])
            probas = self.model.predict_proba([text])[0]
            classes = self.model.classes_
            prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probas)}
            confidence = prob_dict.get(pred_class, 0.90)
            model_ver = "E1_TFIDF_LogReg_v1"
        else:
            pred_class = "GUIDANCE_REVISION" if "expect" in text.lower() else "OTHER"
            prob_dict = {pred_class: 0.85}
            confidence = 0.85
            model_ver = "E0_Rule_Heuristic"

        evidence_span = args.get("evidence") or text[:120]

        return {
            "event_type": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "arguments": args,
            "evidence_span": evidence_span,
            "model_version": model_ver,
        }


# Singleton service instance
service = FinEventService()
