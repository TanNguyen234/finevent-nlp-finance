"""Financial argument and multi-metric extraction engine.

Extracts typed financial metrics, numeric guidance ranges, units, target fiscal periods,
accounting basis, and reporting scope. Normalizes units (e.g. billion vs million)
and abstains (UNKNOWN) when evidence is insufficient or ambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Standard financial units
UNIT_FACTORS = {
    "USD_B": 1_000_000_000.0,
    "USD_M": 1_000_000.0,
    "USD": 1.0,
    "PERCENT": 1.0,
    "PERCENTAGE_POINTS": 1.0,
    "BASIS_POINTS": 0.01,
    "USD_PER_SHARE": 1.0,
    "UNKNOWN": 1.0,
}

# Regex definitions for financial metrics
METRIC_DEFINITIONS = [
    {
        "name": "revenue",
        "patterns": [
            re.compile(r"\b(?:total\s+)?revenue\b", re.IGNORECASE),
            re.compile(r"\bnet\s+sales\b", re.IGNORECASE),
            re.compile(r"\btop[\-\s]line\b", re.IGNORECASE),
        ],
        "default_unit_class": "CURRENCY",
    },
    {
        "name": "gross_margin",
        "patterns": [
            re.compile(r"\bgross\s+margin\b", re.IGNORECASE),
            re.compile(r"\bgross\s+profit\s+margin\b", re.IGNORECASE),
        ],
        "default_unit_class": "RATIO",
    },
    {
        "name": "diluted_eps",
        "patterns": [
            re.compile(r"\bdiluted\s+earnings\s+per\s+share\b", re.IGNORECASE),
            re.compile(r"\bdiluted\s+EPS\b", re.IGNORECASE),
            re.compile(r"\bearnings\s+per\s+diluted\s+share\b", re.IGNORECASE),
        ],
        "default_unit_class": "PER_SHARE",
    },
    {
        "name": "operating_expenses",
        "patterns": [
            re.compile(r"\boperating\s+expenses\b", re.IGNORECASE),
            re.compile(r"\bopex\b", re.IGNORECASE),
        ],
        "default_unit_class": "CURRENCY",
    },
    {
        "name": "capex",
        "patterns": [
            re.compile(r"\bcapital\s+expenditures?\b", re.IGNORECASE),
            re.compile(r"\bcapex\b", re.IGNORECASE),
        ],
        "default_unit_class": "CURRENCY",
    },
    {
        "name": "dividend",
        "patterns": [
            re.compile(r"\b(?:quarterly\s+)?dividends?\b", re.IGNORECASE),
        ],
        "default_unit_class": "PER_SHARE",
    },
]

# Patterns for target periods
PERIOD_PATTERN = re.compile(
    r"\b((?:Q[1-4]|first|second|third|fourth\s+quarter|fiscal\s+year|FY)\s*(?:of\s+)?(?:fiscal\s+)?20\d{2})\b",
    re.IGNORECASE,
)

# Number range patterns
RANGE_BETWEEN_PAT = re.compile(
    r"between\s+\$?([\d\.]+)\s*(?:billion|million|percent|%)?\s*(?:and|to)\s+\$?([\d\.]+)\s*(billion|million|percent|%|percentage points|basis points|bps)?",
    re.IGNORECASE,
)
RANGE_TO_PAT = re.compile(
    r"\$?([\d\.]+)\s*(?:billion|million|percent|%)?\s*(?:to|-)\s+\$?([\d\.]+)\s*(billion|million|percent|%|percentage points|basis points|bps)?",
    re.IGNORECASE,
)
SINGLE_VAL_PAT = re.compile(
    r"(?:of|\bat|\bto|\bwas|\breach)\s+\$?([\d\.]+)\s*(billion|million|percent|%|percentage points|basis points|bps)?",
    re.IGNORECASE,
)
REVISE_FROM_PAT = re.compile(
    r"(?:revising|up\s+from|down\s+from|compared\s+(?:to|with)|prior\s+range\s+of)\s+(?:previously\s+)?(?:modeled|expectations\s+of|issued\s+of)?\s*\$?([\d\.]+)(?:\s*(?:billion|million)?\s*(?:to|-|and)\s*\$?([\d\.]+))?\s*(billion|million|percent|%|percentage points|basis points|bps)?",
    re.IGNORECASE,
)


def normalize_unit(unit_raw: Optional[str], text_context: str = "") -> str:
    """Normalize raw unit string into standardized enum."""
    combined = f"{unit_raw or ''} {text_context}".lower()
    if "basis points" in combined or "bps" in combined:
        return "BASIS_POINTS"
    if "percentage points" in combined:
        return "PERCENTAGE_POINTS"
    if "%" in combined or "percent" in combined:
        return "PERCENT"
    if "billion" in combined:
        return "USD_B"
    if "million" in combined:
        return "USD_M"
    if "per share" in combined or "eps" in combined:
        return "USD_PER_SHARE"
    if "$" in combined or "dollar" in combined:
        return "USD"
    return "UNKNOWN"


def convert_value_to_base_currency(value: Optional[float], unit: str) -> Optional[float]:
    """Convert monetary values to base USD if applicable, preventing unit mismatches."""
    if value is None:
        return None
    factor = UNIT_FACTORS.get(unit, 1.0)
    if unit in {"USD_B", "USD_M", "USD"}:
        return value * factor
    return value


def extract_period(text: str) -> str:
    """Extract and normalize fiscal period string."""
    m = PERIOD_PATTERN.search(text)
    if m:
        raw = m.group(1).strip()
        return re.sub(r"\s+", " ", raw)
    return "UNKNOWN"


def detect_accounting_basis(text: str) -> str:
    """Detect whether guidance is GAAP or Non-GAAP."""
    t_low = text.lower()
    if "non-gaap" in t_low or "adjusted" in t_low or "constant currency" in t_low:
        return "NON_GAAP"
    if "gaap" in t_low:
        return "GAAP"
    return "UNKNOWN"


def extract_single_metric_arguments(text: str, metric_name: str) -> Dict[str, Any]:
    """Extract argument numbers and range for a specific metric context."""
    old_low: Optional[float] = None
    old_high: Optional[float] = None
    new_low: Optional[float] = None
    new_high: Optional[float] = None
    unit_str: Optional[str] = None
    evidence_span: str = text

    # Check for "revising from" prior range
    rev_m = REVISE_FROM_PAT.search(text)
    if rev_m:
        try:
            v1 = float(rev_m.group(1))
            v2 = float(rev_m.group(2)) if rev_m.group(2) else v1
            old_low = min(v1, v2)
            old_high = max(v1, v2)
            unit_str = rev_m.group(3)
        except ValueError:
            pass

    # Check for current range
    range_m = RANGE_BETWEEN_PAT.search(text) or RANGE_TO_PAT.search(text)
    if range_m:
        try:
            v1 = float(range_m.group(1))
            v2 = float(range_m.group(2))
            new_low = min(v1, v2)
            new_high = max(v1, v2)
            unit_str = unit_str or range_m.group(3)
            evidence_span = range_m.group(0).strip()
        except ValueError:
            pass
    else:
        # Check single value
        single_m = SINGLE_VAL_PAT.search(text)
        if single_m:
            try:
                v = float(single_m.group(1))
                new_low = v
                new_high = v
                unit_str = unit_str or single_m.group(2)
                evidence_span = single_m.group(0).strip()
            except ValueError:
                pass

    unit = normalize_unit(unit_str, text_context=text)
    period = extract_period(text)
    basis = detect_accounting_basis(text)

    return {
        "metric": metric_name,
        "target_fiscal_period": period,
        "accounting_basis": basis,
        "old_low": old_low,
        "old_high": old_high,
        "new_low": new_low,
        "new_high": new_high,
        "unit": unit,
        "normalized_new_low": convert_value_to_base_currency(new_low, unit),
        "normalized_new_high": convert_value_to_base_currency(new_high, unit),
        "evidence": evidence_span,
    }


def extract_multi_metric_arguments(text: str) -> List[Dict[str, Any]]:
    """Scan paragraph for ALL present financial metrics and extract distinct argument objects.

    For example, if a paragraph mentions both revenue and gross margin,
    produces separate argument dictionaries for each.
    """
    found_metrics = []
    for mdef in METRIC_DEFINITIONS:
        for pat in mdef["patterns"]:
            match = pat.search(text)
            if match:
                found_metrics.append((mdef["name"], match.start(), match.end()))
                break

    if not found_metrics:
        # Fallback single unknown extraction
        single = extract_single_metric_arguments(text, "UNKNOWN")
        return [single]

    # Sort metrics by character position in the paragraph
    found_metrics.sort(key=lambda x: x[1])

    extractions = []
    for i, (m_name, m_start, m_end) in enumerate(found_metrics):
        # Determine start bound: look backwards for last delimiter or default to m_start
        if i == 0:
            start_bound = 0
        else:
            prev_end = found_metrics[i - 1][2]
            interim = text[prev_end:m_start]
            # Find the last clause boundary (comma, semicolon, 'and', 'with')
            sep_match = list(re.finditer(r"(?:;|,|\band\b|\bwith\b|\bwhile\b)", interim, re.IGNORECASE))
            if sep_match:
                start_bound = prev_end + sep_match[-1].end()
            else:
                start_bound = m_start

        # Determine end bound: look forwards for next metric boundary
        if i == len(found_metrics) - 1:
            end_bound = len(text)
        else:
            next_start = found_metrics[i + 1][1]
            interim = text[m_end:next_start]
            sep_match = re.search(r"(?:;|,|\band\b|\bwith\b|\bwhile\b)", interim, re.IGNORECASE)
            if sep_match:
                end_bound = m_end + sep_match.start()
            else:
                end_bound = next_start

        local_clause = text[start_bound:end_bound].strip()
        if len(local_clause) < 10:
            # Fallback if clause is too clipped: take text from m_start onwards
            local_clause = text[m_start:end_bound].strip()

        arg_obj = extract_single_metric_arguments(local_clause, m_name)
        # Inherit global period or basis if local clause missed it
        if arg_obj["target_fiscal_period"] == "UNKNOWN":
            arg_obj["target_fiscal_period"] = extract_period(text)
        if arg_obj["accounting_basis"] == "UNKNOWN":
            arg_obj["accounting_basis"] = detect_accounting_basis(text)

        extractions.append(arg_obj)

    return extractions


def extract_guidance_arguments(text: str) -> Dict[str, Any]:
    """Compatibility interface returning the primary metric extraction."""
    multi = extract_multi_metric_arguments(text)
    if multi:
        return multi[0]
    return extract_single_metric_arguments(text, "UNKNOWN")


def extract_metric(text: str) -> Optional[str]:
    """Extract primary metric name (backwards-compatibility helper)."""
    args = extract_guidance_arguments(text)
    m = args.get("metric")
    return m if m != "UNKNOWN" else None


def determine_unit(text: str, matched_unit_str: Optional[str] = None) -> str:
    """Normalize unit (backwards-compatibility helper)."""
    return normalize_unit(matched_unit_str, text_context=text)
