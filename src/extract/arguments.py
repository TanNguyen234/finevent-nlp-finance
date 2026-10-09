"""Argument extraction module for FinEvent disclosures.

Extracts financial metrics, guidance ranges (old_low/high, new_low/high),
measurement units, and fiscal periods with deterministic regex & rule parsing.
Handles abstention (UNKNOWN) when evidence is insufficient.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# Patterns for financial metrics
METRIC_PATTERNS = {
    "revenue": [
        re.compile(r"\b(?:total\s+)?revenue\b", re.IGNORECASE),
        re.compile(r"\bnet\s+sales\b", re.IGNORECASE),
        re.compile(r"\btop[\-\s]line\b", re.IGNORECASE),
    ],
    "diluted_eps": [
        re.compile(r"\bdiluted\s+earnings\s+per\s+share\b", re.IGNORECASE),
        re.compile(r"\bdiluted\s+EPS\b", re.IGNORECASE),
        re.compile(r"\bearnings\s+per\s+diluted\s+share\b", re.IGNORECASE),
    ],
    "gross_margin": [
        re.compile(r"\bgross\s+margin\b", re.IGNORECASE),
        re.compile(r"\bgross\s+profit\s+margin\b", re.IGNORECASE),
    ],
    "operating_expenses": [
        re.compile(r"\boperating\s+expenses\b", re.IGNORECASE),
        re.compile(r"\bOpex\b", re.IGNORECASE),
    ],
    "capex": [
        re.compile(r"\bcapital\s+expenditures?\b", re.IGNORECASE),
        re.compile(r"\bcapex\b", re.IGNORECASE),
    ],
    "dividend": [
        re.compile(r"\b(?:quarterly\s+)?dividends?\b", re.IGNORECASE),
    ],
    "share_buyback": [
        re.compile(r"\bshare\s+repurchases?\b", re.IGNORECASE),
        re.compile(r"\bbuyback\b", re.IGNORECASE),
    ],
}

# Number extraction regexes
RANGE_PATTERN_BETWEEN = re.compile(
    r"between\s+\$?([\d\.]+)\s*(?:billion|million)?\s*(?:and|to)\s+\$?([\d\.]+)\s*(billion|million|percent|%)?",
    re.IGNORECASE,
)
RANGE_PATTERN_TO = re.compile(
    r"\$?([\d\.]+)\s*(?:billion|million)?\s*(?:to|-)\s+\$?([\d\.]+)\s*(billion|million|percent|%)?",
    re.IGNORECASE,
)
SINGLE_NUMBER_PATTERN = re.compile(
    r"(?:of|\bat|\bto|\bwas)\s+\$?([\d\.]+)\s*(billion|million|percent|%)?",
    re.IGNORECASE,
)
REVISE_FROM_PATTERN = re.compile(
    r"(?:revising|up\s+from|down\s+from|compared\s+(?:to|with)|prior\s+range\s+of)\s+(?:previously\s+)?(?:modeled|expectations\s+of|issued\s+of)?\s*\$?([\d\.]+)(?:\s*(?:billion|million)?\s*(?:to|-|and)\s*\$?([\d\.]+))?\s*(billion|million|percent|%)?",
    re.IGNORECASE,
)

PERIOD_PATTERN = re.compile(
    r"\b((?:Q[1-4]|first|second|third|fourth\s+quarter|fiscal\s+year|FY)\s*(?:of\s+)?(?:fiscal\s+)?20\d{2})\b",
    re.IGNORECASE,
)


def extract_metric(text: str) -> Optional[str]:
    """Identify the primary financial metric referenced in the text."""
    for metric_name, patterns in METRIC_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                return metric_name
    return None


def extract_period(text: str) -> Optional[str]:
    """Extract fiscal period (e.g. Q1 2024, FY 2025)."""
    match = PERIOD_PATTERN.search(text)
    if match:
        raw_period = match.group(1).strip()
        # Normalize
        raw_period = re.sub(r"\s+", " ", raw_period)
        return raw_period
    return None


def determine_unit(text: str, matched_unit_str: Optional[str] = None) -> str:
    """Determine standardized financial unit."""
    combined = f"{text} {matched_unit_str or ''}".lower()
    if "billion" in combined or "usd_b" in combined:
        return "USD_B"
    if "million" in combined or "usd_m" in combined:
        return "USD_M"
    if "%" in combined or "percent" in combined or "percentage" in combined:
        return "%"
    if "per share" in combined or "eps" in combined or "$" in combined:
        return "USD"
    if "employees" in combined or "headcount" in combined:
        return "persons"
    return "UNKNOWN"


def extract_guidance_arguments(text: str) -> Dict[str, Any]:
    """Extract metric, range figures, unit, and period from financial disclosure snippet.

    Returns dict with keys:
    metric, period, old_low, old_high, new_low, new_high, unit, evidence
    """
    metric = extract_metric(text)
    period = extract_period(text)

    new_low: Optional[float] = None
    new_high: Optional[float] = None
    old_low: Optional[float] = None
    old_high: Optional[float] = None
    unit_str: Optional[str] = None

    # Check for "revising from" or "down from" prior range
    rev_match = REVISE_FROM_PATTERN.search(text)
    if rev_match:
        val1 = float(rev_match.group(1))
        val2 = float(rev_match.group(2)) if rev_match.group(2) else val1
        old_low = min(val1, val2)
        old_high = max(val1, val2)
        if rev_match.group(3):
            unit_str = rev_match.group(3)

    # Check for current range
    range_match = RANGE_PATTERN_BETWEEN.search(text) or RANGE_PATTERN_TO.search(text)
    if range_match:
        try:
            val1 = float(range_match.group(1))
            val2 = float(range_match.group(2))
            new_low = min(val1, val2)
            new_high = max(val1, val2)
            if range_match.group(3):
                unit_str = unit_str or range_match.group(3)
        except ValueError:
            pass

    # Fallback to single figure
    if new_low is None:
        single_match = SINGLE_NUMBER_PATTERN.search(text)
        if single_match:
            try:
                val = float(single_match.group(1))
                new_low = val
                new_high = val
                if single_match.group(2):
                    unit_str = unit_str or single_match.group(2)
            except ValueError:
                pass

    unit = determine_unit(text, unit_str)

    # Locate evidence span
    evidence = text
    if range_match:
        evidence = range_match.group(0).strip()
    elif rev_match:
        evidence = rev_match.group(0).strip()

    return {
        "metric": metric or "UNKNOWN",
        "period": period or "UNKNOWN",
        "old_low": old_low,
        "old_high": old_high,
        "new_low": new_low,
        "new_high": new_high,
        "unit": unit,
        "evidence": evidence,
    }


if __name__ == "__main__":
    test_snippet = (
        "For Q3 FY24, Apple expects revenue between $81.5 billion and $83.5 billion, "
        "revising prior expectations of $79.0 billion to $80.5 billion."
    )
    res = extract_guidance_arguments(test_snippet)
    print("Extracted arguments:", res)
