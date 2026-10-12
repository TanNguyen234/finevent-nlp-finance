"""Temporal change detection module for FinEvent disclosures.

Implements rigorous Guidance Identity Tuple matching, unit normalization,
explicit None handling, and prevents invalid comparisons (e.g. Actuals vs Guidance,
different target periods, or incompatible accounting bases).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from src.extract.arguments import UNIT_FACTORS, convert_value_to_base_currency

logger = logging.getLogger(__name__)


def are_units_compatible(unit_a: str, unit_b: str) -> bool:
    """Check if two units belong to the same dimensional family."""
    currency_units = {"USD_B", "USD_M", "USD"}
    if unit_a in currency_units and unit_b in currency_units:
        return True
    if unit_a == unit_b and unit_a not in {"UNKNOWN", ""}:
        return True
    return False


def validate_guidance_identity_compatibility(
    disclosure_a: Dict[str, Any],
    disclosure_b: Dict[str, Any],
) -> Tuple[bool, str]:
    """Validate that two disclosures can be legally compared under Guidance Identity rules.

    Enforces strict fail-closed 7-tuple identity matching:
    (cik, metric, target_fiscal_period, accounting_basis, currency, unit, scope)
    along with chronological ordering and verified company guidance event types.
    Returns (is_compatible, reason).
    """
    # 1. Company CIK match (fail-closed if missing, unknown, or mismatch)
    cik_a = str(disclosure_a.get("cik") or disclosure_a.get("company_cik") or "").strip()
    cik_b = str(disclosure_b.get("cik") or disclosure_b.get("company_cik") or "").strip()
    if not cik_a or not cik_b or cik_a in {"UNKNOWN", ""} or cik_b in {"UNKNOWN", ""} or cik_a != cik_b:
        return False, f"CIK mismatch or missing: '{cik_a}' vs '{cik_b}'"

    # 2. Metric match (fail-closed if missing, unknown, or mismatch)
    metric_a = (disclosure_a.get("metric") or "").lower().strip()
    metric_b = (disclosure_b.get("metric") or "").lower().strip()
    if not metric_a or not metric_b or metric_a in {"unknown", ""} or metric_b in {"unknown", ""} or metric_a != metric_b:
        return False, f"Metric mismatch or missing: '{metric_a}' vs '{metric_b}'"

    # 3. Target fiscal period match (fail-closed if missing, unknown, or mismatch)
    p_a = str(disclosure_a.get("target_fiscal_period") or disclosure_a.get("fiscal_period") or "").strip()
    p_b = str(disclosure_b.get("target_fiscal_period") or disclosure_b.get("fiscal_period") or "").strip()
    if not p_a or not p_b or p_a in {"UNKNOWN", ""} or p_b in {"UNKNOWN", ""} or p_a != p_b:
        return False, f"Target fiscal period mismatch or missing: '{p_a}' vs '{p_b}'"

    # 4. Accounting basis compatibility (fail-closed: reject missing, UNKNOWN, or mismatch)
    basis_a = str(disclosure_a.get("accounting_basis") or "").strip()
    basis_b = str(disclosure_b.get("accounting_basis") or "").strip()
    if not basis_a or not basis_b or basis_a in {"UNKNOWN", ""} or basis_b in {"UNKNOWN", ""}:
        return False, f"Accounting basis unverified or unknown: '{basis_a}' vs '{basis_b}'"
    if basis_a != basis_b:
        return False, f"Accounting basis mismatch: '{basis_a}' vs '{basis_b}'"

    # 5. Currency compatibility (fail-closed: reject missing, UNKNOWN, or mismatch; never assume USD)
    curr_a = str(disclosure_a.get("currency") or "").strip()
    curr_b = str(disclosure_b.get("currency") or "").strip()
    if not curr_a or not curr_b or curr_a in {"UNKNOWN", ""} or curr_b in {"UNKNOWN", ""}:
        return False, f"Currency mismatch or unverified: '{curr_a}' vs '{curr_b}'"
    if curr_a != curr_b:
        return False, f"Currency mismatch or unverified: '{curr_a}' vs '{curr_b}'"

    # 6. Unit dimension compatibility
    u_a = str(disclosure_a.get("unit") or "").strip()
    u_b = str(disclosure_b.get("unit") or "").strip()
    if not u_a or not u_b or u_a in {"UNKNOWN", ""} or u_b in {"UNKNOWN", ""} or not are_units_compatible(u_a, u_b):
        return False, f"Incompatible or missing units: '{u_a}' vs '{u_b}'"

    # 7. Reporting scope compatibility (fail-closed: reject missing, UNKNOWN, or mismatch; never assume CONSOLIDATED)
    scope_a = str(disclosure_a.get("reporting_scope") or "").strip()
    scope_b = str(disclosure_b.get("reporting_scope") or "").strip()
    if not scope_a or not scope_b or scope_a in {"UNKNOWN", ""} or scope_b in {"UNKNOWN", ""}:
        return False, f"Reporting scope mismatch or unverified: '{scope_a}' vs '{scope_b}'"
    if scope_a != scope_b:
        return False, f"Reporting scope mismatch or unverified: '{scope_a}' vs '{scope_b}'"

    # 8. Chronological ordering (prior must have valid date and not be after current)
    f_date_a = str(disclosure_a.get("filing_date") or "").strip()
    f_date_b = str(disclosure_b.get("filing_date") or "").strip()
    if not f_date_a or not f_date_b or f_date_a in {"UNKNOWN", ""} or f_date_b in {"UNKNOWN", ""}:
        return False, f"Filing date missing or unverified: '{f_date_a}' vs '{f_date_b}'"
    if f_date_a > f_date_b:
        return False, f"Chronological inversion: prior '{f_date_a}' is after current '{f_date_b}'"

    # 9. Event type compatibility: both must represent company-issued guidance
    type_a = str(disclosure_a.get("event_type") or "").strip()
    type_b = str(disclosure_b.get("event_type") or "").strip()
    if not type_a or not type_b or type_a in {"UNKNOWN", ""} or type_b in {"UNKNOWN", ""}:
        return False, f"Event type missing or unverified: '{type_a}' vs '{type_b}'"

    # 9a. Never compare actual backward-looking earnings
    if type_a == "EARNINGS_RESULT" or type_b == "EARNINGS_RESULT" or disclosure_a.get("is_actual") or disclosure_b.get("is_actual"):
        return False, "Cannot compare backward-looking EARNINGS_RESULT with forward-looking GUIDANCE"

    # 9b. Never compare analyst consensus as company guidance
    is_consensus_a = (
        disclosure_a.get("is_consensus")
        or disclosure_a.get("source_type") in {"CONSENSUS", "ANALYST_CONSENSUS"}
        or "consensus" in type_a.lower()
        or "consensus" in str(disclosure_a.get("source", "")).lower()
        or "consensus" in str(disclosure_a.get("evidence", "")).lower()
        or "consensus" in str(disclosure_a.get("text", "")).lower()
    )
    is_consensus_b = (
        disclosure_b.get("is_consensus")
        or disclosure_b.get("source_type") in {"CONSENSUS", "ANALYST_CONSENSUS"}
        or "consensus" in type_b.lower()
        or "consensus" in str(disclosure_b.get("source", "")).lower()
        or "consensus" in str(disclosure_b.get("evidence", "")).lower()
        or "consensus" in str(disclosure_b.get("text", "")).lower()
    )
    if is_consensus_a or is_consensus_b or type_a in {"CONSENSUS", "ANALYST_CONSENSUS"} or type_b in {"CONSENSUS", "ANALYST_CONSENSUS"}:
        return False, "Analyst consensus cannot be compared as company guidance"

    # 9c. Both records must be recognized company guidance events
    valid_guidance_types = {"GUIDANCE", "GUIDANCE_REVISION", "GUIDANCE_ISSUANCE", "COMPANY_GUIDANCE"}
    if type_a not in valid_guidance_types or type_b not in valid_guidance_types:
        return False, f"Non-guidance event type cannot form revision: '{type_a}' vs '{type_b}'"

    return True, "COMPATIBLE"

    return True, "COMPATIBLE"


def calculate_evidence_completeness_score(
    old_present: bool,
    new_present: bool,
    evidence: str,
    is_pairwise: bool,
) -> float:
    """Calculate heuristic evidence completeness score based on span coverage and boundary presence."""
    score = 0.50
    if old_present and new_present:
        score += 0.30
    elif new_present:
        score += 0.15
    if evidence and len(evidence.strip()) > 15:
        score += 0.15
    if is_pairwise:
        score += 0.05
    return min(round(score, 2), 1.0)


# Backward compatibility alias
calculate_dynamic_confidence = calculate_evidence_completeness_score


def compute_revision_direction(delta_mid: Optional[float]) -> str:
    """Classify guidance revision direction."""
    if delta_mid is None:
        return "UNKNOWN"
    if delta_mid > 0.001:
        return "UPWARD_REVISION"
    if delta_mid < -0.001:
        return "DOWNWARD_REVISION"
    return "REAFFIRMED"


def detect_guidance_change(
    disclosure_a: Dict[str, Any],
    disclosure_b: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect guidance change either within a single revision disclosure or between two periods."""
    cik = disclosure_a.get("cik", "UNKNOWN")
    company_name = disclosure_a.get("company_name", "UNKNOWN")

    if disclosure_b is not None:
        # Cross-period pairwise comparison
        is_comp, reason = validate_guidance_identity_compatibility(disclosure_a, disclosure_b)
        if not is_comp:
            return {
                "cik": cik,
                "company_name": company_name,
                "metric": disclosure_a.get("metric", "UNKNOWN"),
                "fiscal_period": disclosure_a.get("fiscal_period", "UNKNOWN"),
                "direction": "NOT_COMPARABLE",
                "reason": reason,
                "evidence_completeness_score": 0.0,
                "model_probability": None,
                "confidence": 0.0,
            }

        metric = disclosure_b.get("metric") or disclosure_a.get("metric", "UNKNOWN")
        period = disclosure_b.get("target_fiscal_period") or disclosure_a.get("target_fiscal_period", "UNKNOWN")
        unit = disclosure_b.get("unit") or disclosure_a.get("unit", "UNKNOWN")

        # Explicit None checks (do not use "or" which fails on 0.0)
        old_low = disclosure_a.get("new_low") if disclosure_a.get("new_low") is not None else disclosure_a.get("old_low")
        old_high = disclosure_a.get("new_high") if disclosure_a.get("new_high") is not None else disclosure_a.get("old_high")
        new_low = disclosure_b.get("new_low")
        new_high = disclosure_b.get("new_high")

        # Normalize units to target unit scale if needed (e.g. billion vs million)
        u_a = disclosure_a.get("unit", unit)
        norm_old_low = convert_value_to_base_currency(old_low, u_a)
        norm_new_low = convert_value_to_base_currency(new_low, unit)
        norm_old_high = convert_value_to_base_currency(old_high, u_a)
        norm_new_high = convert_value_to_base_currency(new_high, unit)

        factor_b = UNIT_FACTORS.get(unit, 1.0)
        conv_old_low = (norm_old_low / factor_b) if (norm_old_low is not None and factor_b) else old_low
        conv_old_high = (norm_old_high / factor_b) if (norm_old_high is not None and factor_b) else old_high

        ev_a = disclosure_a.get("evidence") or disclosure_a.get("text", "")
        ev_b = disclosure_b.get("evidence") or disclosure_b.get("text", "")
        evidence = f"[Prior Period]: {ev_a} | [Current Period]: {ev_b}"
        source_url = disclosure_b.get("source_url", "")
        f_date_a = disclosure_a.get("filing_date", "")
        f_date_b = disclosure_b.get("filing_date", "")
    else:
        # Self-contained revision disclosure
        metric = disclosure_a.get("metric", "UNKNOWN")
        period = disclosure_a.get("target_fiscal_period") or disclosure_a.get("fiscal_period", "UNKNOWN")
        unit = disclosure_a.get("unit", "UNKNOWN")
        old_low = disclosure_a.get("old_low")
        old_high = disclosure_a.get("old_high")
        new_low = disclosure_a.get("new_low")
        new_high = disclosure_a.get("new_high")
        conv_old_low = old_low
        conv_old_high = old_high
        evidence = disclosure_a.get("evidence") or disclosure_a.get("text", "")
        source_url = disclosure_a.get("source_url", "")
        f_date_a = disclosure_a.get("filing_date", "")
        f_date_b = f_date_a

    # Calculate Deltas using normalized/converted values on the same unit scale
    delta_low: Optional[float] = None
    delta_high: Optional[float] = None
    delta_mid: Optional[float] = None

    if conv_old_low is not None and new_low is not None:
        delta_low = round(new_low - conv_old_low, 3)

    if conv_old_high is not None and new_high is not None:
        delta_high = round(new_high - conv_old_high, 3)

    if conv_old_low is not None and conv_old_high is not None and new_low is not None and new_high is not None:
        old_mid = (conv_old_low + conv_old_high) / 2.0
        new_mid = (new_low + new_high) / 2.0
        delta_mid = round(new_mid - old_mid, 3)
    elif delta_low is not None:
        delta_mid = delta_low

    direction = compute_revision_direction(delta_mid)
    completeness_score = calculate_evidence_completeness_score(
        old_present=(old_low is not None or old_high is not None),
        new_present=(new_low is not None or new_high is not None),
        evidence=evidence,
        is_pairwise=(disclosure_b is not None),
    )

    return {
        "cik": cik,
        "company_name": company_name,
        "metric": metric,
        "fiscal_period": period,
        "unit": unit,
        "old_low": old_low,
        "old_high": old_high,
        "conv_old_low": conv_old_low,
        "conv_old_high": conv_old_high,
        "new_low": new_low,
        "new_high": new_high,
        "delta_low": delta_low,
        "delta_high": delta_high,
        "delta_mid": delta_mid,
        "direction": direction,
        "evidence": evidence,
        "source_url": source_url,
        "filing_date_prior": f_date_a,
        "filing_date_current": f_date_b,
        "evidence_completeness_score": completeness_score,
        "model_probability": None,
        "confidence": completeness_score,
    }


def compare_company_reporting_periods(
    events: List[Dict[str, Any]],
    from_date: str,
    to_date: str,
) -> List[Dict[str, Any]]:
    """Compare corporate reporting periods respecting Guidance Identity."""
    window_events = [
        e for e in events
        if from_date <= e.get("filing_date", "") <= to_date
    ]
    window_events.sort(key=lambda x: x.get("filing_date", ""))

    changes: List[Dict[str, Any]] = []

    # 1. Self-contained revisions
    for e in window_events:
        if e.get("event_type") == "GUIDANCE_REVISION":
            c = detect_guidance_change(e)
            if c.get("direction") != "NOT_COMPARABLE":
                changes.append(c)

    # 2. Pairwise period tracking
    by_identity: Dict[str, List[Dict[str, Any]]] = {}
    for e in window_events:
        m = e.get("metric")
        p = e.get("target_fiscal_period") or e.get("fiscal_period")
        if m and m != "UNKNOWN" and p and p != "UNKNOWN":
            key = f"{e.get('cik')}:{m}:{p}"
            by_identity.setdefault(key, []).append(e)

    for key, group in by_identity.items():
        if len(group) >= 2:
            prior = group[0]
            current = group[-1]
            if prior.get("accession_number") != current.get("accession_number"):
                c = detect_guidance_change(prior, current)
                if c.get("direction") != "NOT_COMPARABLE":
                    changes.append(c)

    return changes
