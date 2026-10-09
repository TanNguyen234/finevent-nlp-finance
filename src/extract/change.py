"""Temporal change detection module for FinEvent disclosures.

Compares disclosures across reporting periods, computes delta revisions,
identifies revision direction, and anchors findings to verifiable SEC evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_revision_direction(
    delta_mid: Optional[float],
) -> str:
    """Classify the direction of a guidance revision."""
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
    """Detect guidance change either within a single revision disclosure or between two periods.

    If disclosure_b is provided, disclosure_a is treated as prior (period A)
    and disclosure_b as current (period B).
    """
    cik = disclosure_a.get("cik", "UNKNOWN")
    company_name = disclosure_a.get("company_name", "UNKNOWN")

    if disclosure_b is not None:
        # Cross-period pairwise comparison
        metric = disclosure_b.get("metric") or disclosure_a.get("metric", "UNKNOWN")
        period = disclosure_b.get("fiscal_period") or disclosure_a.get("fiscal_period", "UNKNOWN")
        unit = disclosure_b.get("unit") or disclosure_a.get("unit", "UNKNOWN")

        old_low = disclosure_a.get("new_low") or disclosure_a.get("old_low")
        old_high = disclosure_a.get("new_high") or disclosure_a.get("old_high")
        new_low = disclosure_b.get("new_low")
        new_high = disclosure_b.get("new_high")

        ev_a = disclosure_a.get("evidence") or disclosure_a.get("text", "")
        ev_b = disclosure_b.get("evidence") or disclosure_b.get("text", "")
        evidence = f"[Prior Period]: {ev_a} | [Current Period]: {ev_b}"

        src_a = disclosure_a.get("source_url", "")
        src_b = disclosure_b.get("source_url", "")
        source_url = f"{src_b} (Prior: {src_a})" if src_a else src_b
        filing_date_b = disclosure_b.get("filing_date", "")
        filing_date_a = disclosure_a.get("filing_date", "")
    else:
        # Self-contained revision disclosure
        metric = disclosure_a.get("metric", "UNKNOWN")
        period = disclosure_a.get("fiscal_period", "UNKNOWN")
        unit = disclosure_a.get("unit", "UNKNOWN")
        old_low = disclosure_a.get("old_low")
        old_high = disclosure_a.get("old_high")
        new_low = disclosure_a.get("new_low")
        new_high = disclosure_a.get("new_high")
        evidence = disclosure_a.get("evidence") or disclosure_a.get("text", "")
        source_url = disclosure_a.get("source_url", "")
        filing_date_a = disclosure_a.get("filing_date", "")
        filing_date_b = filing_date_a

    # Compute deltas
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
        "filing_date_prior": filing_date_a,
        "filing_date_current": filing_date_b,
        "confidence": 0.95 if delta_mid is not None else 0.70,
    }


def compare_company_reporting_periods(
    events: List[Dict[str, Any]],
    from_date: str,
    to_date: str,
) -> List[Dict[str, Any]]:
    """Filter events within the requested window and group by metric to compute changes."""
    window_events = [
        e for e in events
        if from_date <= e.get("filing_date", "") <= to_date
    ]

    # Sort chronologically
    window_events.sort(key=lambda x: x.get("filing_date", ""))

    changes: List[Dict[str, Any]] = []

    # Case 1: Self-contained revisions
    for e in window_events:
        if e.get("event_type") == "GUIDANCE_REVISION":
            change = detect_guidance_change(e)
            changes.append(change)

    # Case 2: Pairwise tracking for earnings or repeated metrics
    by_metric: Dict[str, List[Dict[str, Any]]] = {}
    for e in window_events:
        m = e.get("metric")
        if m and m != "UNKNOWN":
            by_metric.setdefault(m, []).append(e)

    for metric_name, group in by_metric.items():
        if len(group) >= 2:
            prior_ev = group[0]
            current_ev = group[-1]
            if prior_ev.get("accession_number") != current_ev.get("accession_number"):
                pair_change = detect_guidance_change(prior_ev, current_ev)
                changes.append(pair_change)

    return changes
