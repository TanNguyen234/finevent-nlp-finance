"""Regression tests for financial reasoning and guidance identity constraints."""

import pytest
from src.extract.arguments import (
    extract_guidance_arguments,
    extract_multi_metric_arguments,
    convert_value_to_base_currency,
)
from src.extract.change import (
    detect_guidance_change,
    validate_guidance_identity_compatibility,
    are_units_compatible,
)


def test_actual_result_cannot_compare_with_guidance():
    """Verify actual backward-looking results cannot be paired with forward guidance."""
    actual_event = {
        "cik": "0000320193",
        "metric": "revenue",
        "target_fiscal_period": "Q1 2025",
        "event_type": "EARNINGS_RESULT",
        "new_low": 90.0,
        "new_high": 90.0,
        "unit": "USD_B",
    }
    guidance_event = {
        "cik": "0000320193",
        "metric": "revenue",
        "target_fiscal_period": "Q1 2025",
        "event_type": "GUIDANCE_REVISION",
        "new_low": 95.0,
        "new_high": 100.0,
        "unit": "USD_B",
    }
    is_comp, reason = validate_guidance_identity_compatibility(actual_event, guidance_event)
    assert not is_comp
    assert "EARNINGS_RESULT" in reason


def test_different_target_periods_rejected():
    """Verify different target fiscal periods cannot be paired as a revision."""
    q1_guidance = {
        "cik": "0000320193",
        "metric": "revenue",
        "target_fiscal_period": "Q1 2025",
        "unit": "USD_B",
        "new_low": 90.0,
        "new_high": 95.0,
    }
    q2_guidance = {
        "cik": "0000320193",
        "metric": "revenue",
        "target_fiscal_period": "Q2 2025",
        "unit": "USD_B",
        "new_low": 95.0,
        "new_high": 100.0,
    }
    is_comp, reason = validate_guidance_identity_compatibility(q1_guidance, q2_guidance)
    assert not is_comp
    assert "Target fiscal period mismatch" in reason


def test_accounting_basis_gaap_vs_nongaap():
    """Verify GAAP cannot be paired directly against Non-GAAP."""
    gaap_ev = {
        "cik": "0000789019",
        "metric": "gross_margin",
        "target_fiscal_period": "FY 2025",
        "accounting_basis": "GAAP",
        "unit": "PERCENT",
        "new_low": 45.0,
        "new_high": 46.0,
    }
    non_gaap_ev = {
        "cik": "0000789019",
        "metric": "gross_margin",
        "target_fiscal_period": "FY 2025",
        "accounting_basis": "NON_GAAP",
        "unit": "PERCENT",
        "new_low": 48.0,
        "new_high": 49.0,
    }
    is_comp, reason = validate_guidance_identity_compatibility(gaap_ev, non_gaap_ev)
    assert not is_comp
    assert "Accounting basis mismatch" in reason


def test_zero_numeric_value_handling():
    """Verify zero value (0.0) is not swallowed as missing."""
    prior = {
        "cik": "0000320193",
        "metric": "diluted_eps",
        "target_fiscal_period": "Q1 2025",
        "new_low": 0.0,
        "new_high": 0.10,
        "unit": "USD",
        "evidence": "prior EPS 0.0 to 0.10",
    }
    current = {
        "cik": "0000320193",
        "metric": "diluted_eps",
        "target_fiscal_period": "Q1 2025",
        "new_low": 0.20,
        "new_high": 0.30,
        "unit": "USD",
        "evidence": "revised EPS 0.20 to 0.30",
    }
    change = detect_guidance_change(prior, current)
    assert change["old_low"] == 0.0
    assert change["delta_low"] == 0.20
    assert change["direction"] == "UPWARD_REVISION"


def test_multi_metric_extraction_in_single_paragraph():
    """Verify paragraph with Revenue and Gross margin yields multiple metric objects."""
    paragraph = (
        "For the second quarter of fiscal 2025, Apple expects revenue between $90.0 billion and $94.0 billion, "
        "and gross margin between 45.0 percent and 46.0 percent."
    )
    extractions = extract_multi_metric_arguments(paragraph)
    assert len(extractions) == 2
    metrics = {e["metric"] for e in extractions}
    assert "revenue" in metrics
    assert "gross_margin" in metrics


def test_unit_conversion_billion_million():
    """Verify monetary conversion converts B and M to base USD properly."""
    val_b = convert_value_to_base_currency(5.0, "USD_B")
    val_m = convert_value_to_base_currency(5000.0, "USD_M")
    assert val_b == 5_000_000_000.0
    assert val_m == 5_000_000_000.0
    assert val_b == val_m


def test_incompatible_unit_rejection():
    """Verify percentage gross margin is not comparable to billion dollar revenue."""
    assert not are_units_compatible("PERCENT", "USD_B")


def test_cross_unit_delta_computation_normalized():
    """Verify that comparing $1000M with $1.1B yields a +$0.1B delta, not $1098.9."""
    prior = {
        "cik": "0000320193",
        "metric": "revenue",
        "target_fiscal_period": "Q1 2025",
        "new_low": 1000.0,
        "new_high": 1000.0,
        "unit": "USD_M",
        "evidence": "prior guidance 1000 million",
    }
    current = {
        "cik": "0000320193",
        "metric": "revenue",
        "target_fiscal_period": "Q1 2025",
        "new_low": 1.1,
        "new_high": 1.1,
        "unit": "USD_B",
        "evidence": "revised revenue 1.1 billion",
    }
    change = detect_guidance_change(prior, current)
    assert change["direction"] == "UPWARD_REVISION"
    assert change["delta_low"] == 0.1
    assert change["delta_mid"] == 0.1
    assert change["conv_old_low"] == 1.0

