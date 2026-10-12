"""Test edge cases in argument extraction and change detection."""

from __future__ import annotations

import pytest
from src.extract.arguments import (
    extract_guidance_arguments,
    extract_metric,
    extract_period,
    determine_unit,
)
from src.extract.change import (
    compute_revision_direction,
    detect_guidance_change,
)
from src.extract.segment import is_boilerplate, parse_and_segment_filing


def test_empty_and_whitespace_text():
    """Verify system does not crash on empty or whitespace strings."""
    res = extract_guidance_arguments("")
    assert res["metric"] == "UNKNOWN"
    assert res["new_low"] is None
    assert res["old_low"] is None

    res_spaces = extract_guidance_arguments("   \n\t   ")
    assert res_spaces["metric"] == "UNKNOWN"
    assert res_spaces["new_low"] is None


def test_text_without_financial_numbers():
    """Verify system abstains when text has no figures or metrics."""
    snippet = "The company held its annual meeting of shareholders in Cupertino."
    res = extract_guidance_arguments(snippet)
    assert res["metric"] == "UNKNOWN"
    assert res["new_low"] is None


def test_boilerplate_detection():
    """Verify standard SEC signatures and boilerplate are caught."""
    assert is_boilerplate("SIGNATURES")
    assert is_boilerplate("Pursuant to the requirements of the Securities Exchange Act of 1934")
    assert not is_boilerplate("Apple expects quarterly revenue between $90.0 billion and $93.0 billion for Q1 2025.")


def test_negative_revision_direction():
    """Verify downward revision calculation."""
    dir_down = compute_revision_direction(-1.5)
    assert dir_down == "DOWNWARD_REVISION"

    dir_up = compute_revision_direction(2.0)
    assert dir_up == "UPWARD_REVISION"

    dir_flat = compute_revision_direction(0.0)
    assert dir_flat == "REAFFIRMED"

    dir_none = compute_revision_direction(None)
    assert dir_none == "UNKNOWN"


def test_pairwise_change_calculation():
    """Verify pairwise cross-period change detection."""
    prior = {
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "metric": "revenue",
        "fiscal_period": "Q1 2025",
        "accounting_basis": "GAAP",
        "currency": "USD",
        "reporting_scope": "CONSOLIDATED",
        "event_type": "GUIDANCE_REVISION",
        "new_low": 89.0,
        "new_high": 93.0,
        "unit": "USD_B",
        "evidence": "prior guidance 89 to 93",
        "filing_date": "2024-11-01",
        "source_url": "https://sec.gov/prior",
    }
    current = {
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "metric": "revenue",
        "fiscal_period": "Q1 2025",
        "accounting_basis": "GAAP",
        "currency": "USD",
        "reporting_scope": "CONSOLIDATED",
        "event_type": "GUIDANCE_REVISION",
        "new_low": 94.0,
        "new_high": 96.0,
        "unit": "USD_B",
        "evidence": "revised guidance 94 to 96",
        "filing_date": "2025-01-15",
        "source_url": "https://sec.gov/current",
    }
    change = detect_guidance_change(prior, current)
    assert change["direction"] == "UPWARD_REVISION"
    assert change["delta_low"] == 5.0  # 94 - 89
    assert change["delta_high"] == 3.0  # 96 - 93
    assert change["delta_mid"] == 4.0   # 95 - 91
