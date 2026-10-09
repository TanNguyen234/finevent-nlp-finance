"""Test schema validity for FinEvent records."""

from __future__ import annotations

import json
import pathlib
import pytest
from pydantic import ValidationError
from app.schemas import FinancialEvent, GuidanceChange


def test_gold_dataset_schema_conformance():
    """Verify all records in gold.jsonl strictly satisfy the FinancialEvent schema."""
    gold_path = pathlib.Path("data/labels/gold.jsonl")
    assert gold_path.exists(), "gold.jsonl must exist"

    with open(gold_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 15, f"Expected verified gold samples, got {len(lines)}"

    # Also verify isolated synthetic dataset
    syn_path = pathlib.Path("data/synthetic/synthetic_samples.jsonl")
    assert syn_path.exists(), "synthetic_samples.jsonl must exist"

    for idx, line in enumerate(lines):
        data = json.loads(line)
        event = FinancialEvent(**data)
        assert event.cik.isdigit() and len(event.cik) == 10
        assert event.event_type in {"GUIDANCE_REVISION", "EARNINGS_RESULT", "MANAGEMENT_CHANGE", "OTHER"}
        assert len(event.evidence) > 0
        assert event.char_start >= 0
        assert event.char_end >= event.char_start


def test_schema_rejects_invalid_cik():
    """Verify schema raises error on invalid non-10-digit CIK."""
    invalid_data = {
        "id": "err_01",
        "cik": "123",  # Invalid short CIK
        "company_name": "Test Corp",
        "filing_date": "2024-01-01",
        "fiscal_period": "Q1 2024",
        "accession_number": "0000123456-24-000001",
        "source_url": "https://sec.gov/test",
        "event_type": "GUIDANCE_REVISION",
        "text": "Valid text with revenue guidance.",
        "evidence": "revenue guidance",
        "char_start": 10,
        "char_end": 26,
    }
    with pytest.raises(ValidationError):
        FinancialEvent(**invalid_data)


def test_schema_rejects_invalid_event_type():
    """Verify schema rejects unauthorized event classes."""
    invalid_data = {
        "id": "err_02",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "filing_date": "2024-01-01",
        "fiscal_period": "Q1 2024",
        "accession_number": "0000320193-24-000001",
        "source_url": "https://sec.gov/test",
        "event_type": "STOCK_BUY_SIGNAL",  # Illegal non-existent event class
        "text": "Valid text with revenue guidance.",
        "evidence": "revenue guidance",
        "char_start": 10,
        "char_end": 26,
    }
    with pytest.raises(ValidationError):
        FinancialEvent(**invalid_data)


def test_guidance_change_schema():
    """Verify GuidanceChange schema passes for valid delta."""
    change = GuidanceChange(
        cik="0000320193",
        company_name="Apple Inc.",
        metric="revenue",
        fiscal_period="Q2 2024",
        unit="USD_B",
        old_low=80.0,
        old_high=85.0,
        new_low=83.0,
        new_high=87.0,
        delta_low=3.0,
        delta_high=2.0,
        delta_mid=2.5,
        direction="UPWARD_REVISION",
        evidence="Apple expects revenue between 83 and 87",
        source_url="https://sec.gov/sample",
        confidence=0.95,
    )
    assert change.direction == "UPWARD_REVISION"
    assert change.delta_mid == 2.5
