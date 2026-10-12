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


from abc import ABC, abstractmethod

class BaseInferenceBackend(ABC):
    """Abstract interface for all FinEvent inference backends."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the precise model identifier string."""
        ...

    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict event class and class probabilities for a text paragraph.

        Returns:
            dict with keys: 'predicted_class', 'confidence', 'probabilities'.
        """
        ...


class RuleBackend(BaseInferenceBackend):
    """Deterministic keyword baseline."""

    @property
    def model_version(self) -> str:
        return "E0_Rule_Heuristic"

    def predict(self, text: str) -> Dict[str, Any]:
        t_low = text.lower()
        if any(k in t_low for k in ["revis", "expects revenue to", "expect gross margin", "guidance", "outlook"]):
            pred_class = "GUIDANCE_REVISION"
        elif any(k in t_low for k in ["posted quarterly", "reported financial results"]):
            pred_class = "EARNINGS_RESULT"
        elif any(k in t_low for k in ["appointed", "resigned", "depart", "elected"]):
            pred_class = "MANAGEMENT_CHANGE"
        else:
            pred_class = "OTHER"
        return {
            "predicted_class": pred_class,
            "confidence": 0.80,
            "probabilities": {pred_class: 0.80},
        }


class SklearnBackend(BaseInferenceBackend):
    """scikit-learn TF-IDF + Logistic Regression backend."""

    def __init__(self, pipeline: Any, model_version: str = "E1_TFIDF_LogReg") -> None:
        self._pipeline = pipeline
        self._version = model_version

    @property
    def model_version(self) -> str:
        return self._version

    def predict(self, text: str) -> Dict[str, Any]:
        pred_class = str(self._pipeline.predict([text])[0])
        probas = self._pipeline.predict_proba([text])[0]
        classes = self._pipeline.classes_
        prob_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probas)}
        confidence = prob_dict.get(pred_class, 0.75)
        return {
            "event_type": pred_class,
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "scores": prob_dict,
            "probabilities": prob_dict,
            "model_version": self._version,
        }


class HuggingFaceBackend(BaseInferenceBackend):
    """Hugging Face pipeline adapter for DeBERTa / FinBERT."""

    def __init__(self, pipeline: Any, model_version: str = "E2_DEBERTA") -> None:
        self._pipeline = pipeline
        self._version = model_version

    @property
    def model_version(self) -> str:
        return self._version

    def predict(self, text: str) -> Dict[str, Any]:
        results = self._pipeline(text)
        if isinstance(results, list) and len(results) > 0:
            item = results[0]
            if isinstance(item, list):
                score_dict = {str(r["label"]): round(float(r["score"]), 4) for r in item}
                top = max(item, key=lambda x: x["score"])
                top_label = str(top["label"])
                top_score = round(float(top["score"]), 4)
                return {
                    "event_type": top_label,
                    "predicted_class": top_label,
                    "confidence": top_score,
                    "scores": score_dict,
                    "probabilities": score_dict,
                    "model_version": self._version,
                }
            elif isinstance(item, dict):
                score_dict = {str(r.get("label", "OTHER")): round(float(r.get("score", 0.5)), 4) for r in results if isinstance(r, dict)}
                top = max(results, key=lambda x: x.get("score", 0.0)) if all(isinstance(r, dict) for r in results) else item
                top_label = str(top.get("label", "OTHER"))
                top_score = round(float(top.get("score", 0.5)), 4)
                return {
                    "event_type": top_label,
                    "predicted_class": top_label,
                    "confidence": top_score,
                    "scores": score_dict if score_dict else {top_label: top_score},
                    "probabilities": score_dict if score_dict else {top_label: top_score},
                    "model_version": self._version,
                }
        return {
            "event_type": "OTHER",
            "predicted_class": "OTHER",
            "confidence": 0.50,
            "scores": {"OTHER": 0.50},
            "probabilities": {"OTHER": 0.50},
            "model_version": self._version,
        }


class HybridEnsembleBackend(BaseInferenceBackend):
    """Hybrid Serving Stack combining fast linear matcher (E1) with deep semantic classifier (E2).

    Confidence Arbitrator:
    - High-confidence fast path: if E1 confidence >= fast_threshold (0.85) or deep backend unavailable -> returns fast path (<2ms).
    - Ambiguous zone / ensemble: blends normalized probabilities P = alpha * P(E1) + (1 - alpha) * P(E2).
    """

    def __init__(
        self,
        fast_backend: BaseInferenceBackend,
        deep_backend: Optional[BaseInferenceBackend] = None,
        fast_threshold: float = 0.85,
        alpha: float = 0.5,
    ) -> None:
        self.fast_backend = fast_backend
        self.deep_backend = deep_backend
        self.fast_threshold = fast_threshold
        self.alpha = alpha

    @property
    def model_version(self) -> str:
        deep_ver = self.deep_backend.model_version if self.deep_backend else "NONE"
        return f"E3_Hybrid({self.fast_backend.model_version}+{deep_ver})"

    def predict(self, text: str) -> Dict[str, Any]:
        fast_res = self.fast_backend.predict(text)
        fast_conf = fast_res.get("confidence", 0.0)

        # Fast-path exit if no deep backend or high confidence in unambiguous regime
        if self.deep_backend is None or fast_conf >= self.fast_threshold:
            res = dict(fast_res)
            res["model_version"] = self.model_version
            res["serving_path"] = "FAST_PATH"
            return res

        deep_res = self.deep_backend.predict(text)

        # Ensemble class probabilities
        fast_probs = fast_res.get("probabilities") or fast_res.get("scores", {})
        deep_probs = deep_res.get("probabilities") or deep_res.get("scores", {})
        all_classes = set(fast_probs.keys()).union(set(deep_probs.keys()))

        blended_scores: Dict[str, float] = {}
        for c in all_classes:
            p_fast = fast_probs.get(c, 0.0)
            p_deep = deep_probs.get(c, 0.0)
            blended = self.alpha * p_fast + (1.0 - self.alpha) * p_deep
            blended_scores[c] = round(float(blended), 4)

        if blended_scores:
            top_class = max(blended_scores.keys(), key=lambda k: blended_scores[k])
            top_conf = blended_scores[top_class]
        else:
            top_class = fast_res.get("predicted_class", "OTHER")
            top_conf = fast_conf

        return {
            "event_type": top_class,
            "predicted_class": top_class,
            "confidence": top_conf,
            "scores": blended_scores,
            "probabilities": blended_scores,
            "model_version": self.model_version,
            "serving_path": "DEEP_ENSEMBLE",
            "ensemble_components": {
                "fast": fast_res.get("model_version"),
                "deep": deep_res.get("model_version"),
            },
        }


class FinEventService:
    """Service handling operational event queries and inference across unified backends."""

    def __init__(self, backend_override: Optional[BaseInferenceBackend] = None) -> None:
        self.events: List[Dict[str, Any]] = []
        self.events_by_id: Dict[str, Dict[str, Any]] = {}
        self.events_by_cik: Dict[str, List[Dict[str, Any]]] = {}
        self.dataset_mode = "REAL_SEC"
        self.backend_type = os.environ.get("FINEVENT_MODEL_BACKEND", "E1_TFIDF")
        self.cloud_weights_status = "CLOUD_STORAGE_ONLY"
        self.fallback_reason: Optional[str] = None
        self._load_data()
        self.backend: BaseInferenceBackend = backend_override or self._load_backend()

    def _load_data(self) -> None:
        target_path = OPERATIONAL_DATA_PATH
        if not target_path.exists():
            self.dataset_mode = "NO_OPERATIONAL_DATA"
            logger.warning("Operational events file %s not found. Degraded state: benchmark datasets are not served as operational data.", target_path)
            return

        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.events.append(item)
                    self.events_by_id[item["id"]] = item
                    cik = item["cik"]
                    self.events_by_cik.setdefault(cik, []).append(item)

        self.dataset_mode = "REAL_SEC"
        logger.info("Loaded %d events across %d companies (Mode: %s).", len(self.events), len(self.events_by_cik), self.dataset_mode)

    def _load_backend(self) -> BaseInferenceBackend:
        # 0. Check for hybrid ensemble request
        if self.backend_type == "E3_HYBRID":
            fast_backend: BaseInferenceBackend
            if MODEL_PATH.exists():
                pipe = joblib.load(MODEL_PATH)
                fast_backend = SklearnBackend(pipe, model_version="E1_TFIDF_LogReg")
            else:
                fast_backend = RuleBackend()

            deep_backend: Optional[BaseInferenceBackend] = None
            for p, ver in [(E2_DEBERTA_PATH, "E2_DEBERTA"), (E2_FINBERT_PATH, "E2_FINBERT")]:
                if p.exists():
                    try:
                        from transformers import pipeline
                        deep_backend = HuggingFaceBackend(pipeline("text-classification", model=str(p)), model_version=ver)
                        self.cloud_weights_status = "LOADED_LOCAL"
                        break
                    except Exception as exc:
                        logger.warning("Could not load %s for hybrid ensemble: %s", ver, exc)

            if deep_backend is None:
                self.fallback_reason = "E2_WEIGHTS_UNAVAILABLE_LOCAL"

            return HybridEnsembleBackend(fast_backend=fast_backend, deep_backend=deep_backend)

        # 1. Check for requested E2 neural models
        if self.backend_type in {"E2_DEBERTA", "E2_FINBERT"}:
            target_path = E2_DEBERTA_PATH if self.backend_type == "E2_DEBERTA" else E2_FINBERT_PATH
            if target_path.exists():
                try:
                    from transformers import pipeline
                    logger.info("Loading neural model %s from %s", self.backend_type, target_path)
                    pipe = pipeline("text-classification", model=str(target_path))
                    self.cloud_weights_status = "LOADED_LOCAL"
                    return HuggingFaceBackend(pipe, model_version=self.backend_type)
                except Exception as exc:
                    logger.warning("Failed to load neural weights for %s: %s", self.backend_type, exc)
            self.fallback_reason = "E2_WEIGHTS_UNAVAILABLE_LOCAL"
            logger.info("E2 neural weights absent locally (Rule 1). Explicitly falling back to E1 TF-IDF.")

        # 2. E1 TF-IDF Logistic Regression
        if MODEL_PATH.exists():
            logger.info("Loading E1 baseline model from %s", MODEL_PATH)
            pipe = joblib.load(MODEL_PATH)
            return SklearnBackend(pipe, model_version="E1_TFIDF_LogReg")

        # 3. Deterministic rule baseline
        logger.warning("E1 model file not found. Falling back to E0 Rule baseline.")
        return RuleBackend()

    @property
    def model(self):
        """Backwards compatibility accessor for tests checking `service.model is not None`."""
        return getattr(self.backend, "_pipeline", None)

    def get_service_status(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "service": "FinEvent API",
            "active_model": self.backend.model_version,
            "configured_backend": self.backend_type,
            "cloud_weights_status": self.cloud_weights_status,
            "fallback_reason": self.fallback_reason,
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
        """Classify event paragraph and extract multiple arguments through the unified backend."""
        multi_args = extract_multi_metric_arguments(text)
        primary_args = multi_args[0] if multi_args else {}

        pred_res = self.backend.predict(text)
        evidence_span = primary_args.get("evidence") or text[:120]
        event_type = pred_res.get("event_type") or pred_res.get("predicted_class", "OTHER")
        confidence = pred_res.get("confidence", 0.5)
        scores = pred_res.get("scores") or pred_res.get("probabilities", {})

        return {
            "event_type": event_type,
            "confidence": confidence,
            "probabilities": scores,
            "scores": scores,
            "arguments": primary_args,
            "multi_arguments": multi_args,
            "evidence_span": evidence_span,
            "model_version": self.backend.model_version,
        }


service = FinEventService()
