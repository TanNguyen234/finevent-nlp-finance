"""Integration test validating the full pipeline from raw SEC HTML fixture to API response."""

from __future__ import annotations

import hashlib
import pytest
from fastapi.testclient import TestClient
from app.main import app
from src.extract.segment import parse_and_segment_filing
from src.extract.arguments import extract_multi_metric_arguments
from src.extract.change import detect_guidance_change

client = TestClient(app)

# Authentic SEC 8-K HTML snippet fixture (Item 2.02 / Exhibit 99.1 style)
SEC_8K_FIXTURE_HTML = """
<html>
<head><title>Apple Inc. Form 8-K Item 2.02</title></head>
<body>
<div class="header">UNITED STATES SECURITIES AND EXCHANGE COMMISSION</div>
<h2>Item 2.02 Results of Operations and Financial Condition.</h2>
<p>
On October 31, 2024, Apple Inc. announced financial results for its fiscal 2024 fourth quarter ended September 28, 2024.
The Company posted quarterly revenue of $94.9 billion, up 6 percent year over year, and quarterly diluted earnings per share of $0.97.
</p>
<h2>Item 7.01 Regulation FD Disclosure.</h2>
<p>
For the first quarter of fiscal 2025, Apple expects total company revenue to grow between 7.0% and 8.0% year over year,
and gross margin to range between 46.0 percent and 47.0 percent, revising prior internal baseline projections.
</p>
<div class="footer">Pursuant to the requirements of the Securities Exchange Act of 1934.</div>
</body>
</html>
"""


def test_end_to_end_sec_fixture_pipeline():
    """Verify raw SEC HTML fixture correctly propagates through segmentation, extraction, and API."""
    # 1. Compute fixture provenance hash
    fixture_bytes = SEC_8K_FIXTURE_HTML.encode("utf-8")
    sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    assert len(sha256) == 64

    # 2. Segment document
    accession = "0000320193-24-000106"
    segments = parse_and_segment_filing(SEC_8K_FIXTURE_HTML, accession_number=accession)
    assert len(segments) >= 2

    # Check that legal boilerplate was removed
    seg_texts = [s["text"] for s in segments]
    assert not any("Pursuant to the requirements" in t for t in seg_texts)

    # 3. Multi-metric argument extraction on the guidance paragraph
    guidance_seg = [s for s in segments if "gross margin" in s["text"]][0]
    extractions = extract_multi_metric_arguments(guidance_seg["text"])
    assert len(extractions) >= 1

    extracted_metrics = [e["metric"] for e in extractions]
    assert "gross_margin" in extracted_metrics

    gm_arg = [e for e in extractions if e["metric"] == "gross_margin"][0]
    assert gm_arg["new_low"] == 46.0
    assert gm_arg["new_high"] == 47.0
    assert gm_arg["unit"] == "PERCENT"

    # 4. Change detection
    prior_disclosure = {
        "cik": "0000320193",
        "metric": "gross_margin",
        "target_fiscal_period": "Q1 2025",
        "accounting_basis": "GAAP",
        "unit": "PERCENT",
        "new_low": 45.0,
        "new_high": 46.0,
        "filing_date": "2024-08-01",
    }
    current_disclosure = {
        "cik": "0000320193",
        "metric": "gross_margin",
        "target_fiscal_period": "Q1 2025",
        "accounting_basis": "GAAP",
        "unit": "PERCENT",
        "new_low": gm_arg["new_low"],
        "new_high": gm_arg["new_high"],
        "filing_date": "2024-10-31",
    }
    change = detect_guidance_change(prior_disclosure, current_disclosure)
    assert change["direction"] == "UPWARD_REVISION"
    assert change["delta_mid"] == 1.0

    # 5. Live API endpoint verification
    res = client.post("/predict/event", json={"text": guidance_seg["text"]})
    assert res.status_code == 200
    api_data = res.json()
    assert "event_type" in api_data
    assert "confidence" in api_data
    assert api_data["confidence"] > 0.0
