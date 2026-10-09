"""Temporal change detection module for FinEvent disclosures.

Implements rigorous Guidance Identity Tuple matching, unit normalization,
explicit None handling, and prevents invalid comparisons (e.g. Actuals vs Guidance,
different target periods, or incompatible accounting bases).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from src.extract.arguments import UNIT_FACTORS, convert_value_to_base_currency

logger = logging.getLogger(__name__)


def are_units_compatible(unit_a: str, unit_b: str) -> bool:
    """Check if two units belong to the same dimensional family."""
    currency_units = {"USD_B", "USD_M", "USD"}
    percentage_units = {"PERCENT", "PERCENTAGE_POINTS"}

    if unit_a in currency_units and unit_b in currency_units:
        return True
    if unit_a in percentage_units and unit_b in percentage_units:
        return True
    if unit_a == unit_b and unit_a not in {"UNKNOWN"}:
        return True
    return False


def validate_guidance_identity_compatibility(
    disclosure_a: Dict[str, Any],
    disclosure_b: Dict[str, Any],
) -> Tuple[bool, str]:
    """Validate that two disclosures can be legally compared under Guidance Identity rules.

    Returns (is_compatible, reason).
    """
    # 1. Company CIK match
    cik_a = disclosure_a.get("cik")
    cik_b = disclosure_b.get("cik")
    if cik_a and cik_b and cik_a != cik_b:
        return False, f"CIK mismatch: {cik_a} vs {cik_b}"

    # 2. Metric match
    metric_a = (disclosure_a.get("metric") or "").lower()
    metric_b = (disclosure_b.get("metric") or "").lower()
    if not metric_a or not metric_b or metric_a == "unknown" or metric_b == "unknown" or metric_a != metric_b:
        return False, f"Metric mismatch: '{metric_a}' vs '{metric_b}'"

    # 3. Target period match
    p_a = disclosure_a.get("target_fiscal_period") or disclosure_a.get("fiscal_period")
    p_b = disclosure_b.get("target_fiscal_period") or disclosure_b.get("fiscal_period")
    if p_a and p_b and p_a != "UNKNOWN" and p_b != "UNKNOWN" and p_a != p_b:
        return False, f"Target fiscal period mismatch: '{p_a}' vs '{p_b}'"

    # 4. Accounting basis compatibility
    basis_a = disclosure_a.get("accounting_basis", "UNKNOWN")
    basis_b = disclosure_b.get("accounting_basis", "UNKNOWN")
    if basis_a in {"GAAP", "NON_GAAP"} and basis_b in {"GAAP", "NON_GAAP"} and basis_a != basis_b:
        return False, f"Accounting basis mismatch: {basis_a} vs {basis_b}"

    # 5. Unit compatibility
    u_a = disclosure_a.get("unit", "UNKNOWN")
    u_b = disclosure_b.get("unit", "UNKNOWN")
    if not are_units_compatible(u_a, u_b):
        return False, f"Incompatible units: {u_a} vs {u_b}"

    # 6. Event type compatibility: do not compare Actual results directly to Guidance as a revision
    type_a = disclosure_a.get("event_type")
    type_b = disclosure_b.get("event_type")
    if type_a == "EARNINGS_RESULT" and type_b in {"GUIDANCE_REVISION", "GUIDANCE_ISSUANCE"}:
        return False, "Cannot compare backward-looking EARNINGS_RESULT with forward-looking GUIDANCE"

    return True, "COMPATIBLE"


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

        # Normalize units to base if needed (e.g. billion vs million)
        norm_old_low = convert_value_to_base_currency(old_low, disclosure_a.get("unit", unit))
        norm_new_low = convert_value_to_base_currency(new_low, unit)
        norm_old_high = convert_value_to_base_currency(old_high, disclosure_a.get("unit", unit))
        norm_new_high = convert_value_to_base_currency(new_high, unit)

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
        norm_old_low = old_low
        norm_new_low = new_low
        norm_old_high = old_high
        norm_new_high = new_high
        evidence = disclosure_a.get("evidence") or disclosure_a.get("text", "")
        source_url = disclosure_a.get("source_url", "")
        f_date_a = disclosure_a.get("filing_date", "")
        f_date_b = f_date_a

    # Calculate Deltas
    delta_low: Optional[float] = None
    delta_high: Optional[float] = None
    delta_mid: Optional[float] = None

    if old_low is not None and new_low is not None:
        delta_low = round(new_low - old_low, 3)

    if old_high is not None and new_high is not None:
        delta_high = round(new_high - old_high, 3)

    if old_low is not None and old_high is not None and new_low is not None and new_high is not None:
        old_mid = (old_low + old_high) / 2.0
        new_mid = (new_low + new_high) / 2.0
        delta_mid = round(new_mid - old_mid, 3)
    elif delta_low is not None:
        delta_mid = delta_low

    direction = compute_revision_direction(delta_mid)

    return {
        "cik": cik,
        "company_name": company_name,
        "metric": metric,
        "fiscal_period": period,
        "unit": unit,
        "old_low": old_low,
        "old_high": old_high,
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
        "confidence": 0.95 if delta_mid is not None else 0.70,
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
