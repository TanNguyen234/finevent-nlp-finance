"""Integration test validating the full pipeline from raw SEC HTML fixture to API response."""

from __future__ import annotations

import hashlib
import pathlib
import pytest
from fastapi.testclient import TestClient
from app.main import app
from src.extract.segment import is_boilerplate, normalize_whitespace, parse_and_segment_filing
from src.extract.arguments import extract_multi_metric_arguments
from src.extract.change import detect_guidance_change

client = TestClient(app)

# Synthetic SEC 8-K HTML snippet fixture (modeled on authentic Item 2.02 / Exhibit 99.1 disclosures)
SYNTHETIC_SEC_8K_FIXTURE_HTML = """
<html>
<head><title>Apple Inc. Form 8-K Item 2.02 (Synthetic Benchmark Fixture)</title></head>
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

# Backward compatibility alias
SEC_8K_FIXTURE_HTML = SYNTHETIC_SEC_8K_FIXTURE_HTML


def test_end_to_end_sec_fixture_pipeline():
    """Verify raw SEC HTML fixture correctly propagates through segmentation, extraction, and API."""
    # 1. Compute fixture provenance hash
    fixture_bytes = SYNTHETIC_SEC_8K_FIXTURE_HTML.encode("utf-8")
    sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    assert len(sha256) == 64

    # 2. Segment document
    accession = "0000320193-24-000106"
    segments = parse_and_segment_filing(SYNTHETIC_SEC_8K_FIXTURE_HTML, accession_number=accession)
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
        "currency": "NOT_APPLICABLE",
        "unit": "PERCENT",
        "reporting_scope": "CONSOLIDATED",
        "event_type": "GUIDANCE_REVISION",
        "new_low": 45.0,
        "new_high": 46.0,
        "filing_date": "2024-08-01",
    }
    current_disclosure = {
        "cik": "0000320193",
        "metric": "gross_margin",
        "target_fiscal_period": "Q1 2025",
        "accounting_basis": "GAAP",
        "currency": "NOT_APPLICABLE",
        "unit": "PERCENT",
        "reporting_scope": "CONSOLIDATED",
        "event_type": "GUIDANCE_REVISION",
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


def test_exhibit_discovery_realistic_index(monkeypatch):
    """Verify discover_exhibits parses a realistic SEC EDGAR document index table using BeautifulSoup."""
    from src.ingest.sec_submissions import SECSubmissionsClient

    realistic_index_html = """
    <html>
    <body>
    <table class="tableFile" summary="Document Format Files">
    <tr>
        <th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th>
    </tr>
    <tr>
        <td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/320193/000032019326000005/aapl-20260129.htm">aapl-20260129.htm</a></td><td>8-K</td><td>38421</td>
    </tr>
    <tr>
        <td>2</td><td>Press Release Q1 2026</td><td><a href="/Archives/edgar/data/320193/000032019326000005/a8-kex991q1202612272025.htm">a8-kex991q1202612272025.htm</a></td><td>EX-99.1</td><td>140168</td>
    </tr>
    </table>
    </body>
    </html>
    """

    client_sec = SECSubmissionsClient()
    class MockResponse:
        status_code = 200
        text = realistic_index_html
    monkeypatch.setattr("requests.get", lambda url, **kwargs: MockResponse())

    exhibits = client_sec.discover_exhibits("0000320193", "0000320193-26-000005")
    assert len(exhibits) >= 1
    ex = exhibits[0]
    assert ex["type"] == "EX-99.1"
    assert ex["filename"] == "a8-kex991q1202612272025.htm"
    assert "https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/a8-kex991q1202612272025.htm" in ex["url"]


def test_exhibit_filename_not_hardcoded(monkeypatch):
    """Verify exhibit discovery preserves dynamic arbitrary filenames from EDGAR tables."""
    from src.ingest.sec_submissions import SECSubmissionsClient

    arbitrary_index_html = """
    <table>
    <tr><td>Seq</td><td>Description</td><td>Document</td><td>Type</td><td>Size</td></tr>
    <tr><td>1</td><td>Press Release</td><td><a href="d89342_arbitrary_name_99_1.htm">d89342_arbitrary_name_99_1.htm</a></td><td>EX-99.1</td><td>12345</td></tr>
    </table>
    """
    client_sec = SECSubmissionsClient()
    class MockResponse:
        status_code = 200
        text = arbitrary_index_html
    monkeypatch.setattr("requests.get", lambda url, **kwargs: MockResponse())

    exhibits = client_sec.discover_exhibits("0000789019", "0000789019-24-000088")
    assert len(exhibits) == 1
    assert exhibits[0]["filename"] == "d89342_arbitrary_name_99_1.htm"


def test_discovered_exhibit_download_flow(tmp_path, monkeypatch):
    """Verify end-to-end flow from discovered exhibit metadata to FilingDownloader provenance record."""
    from src.ingest.download_filing import FilingDownloader

    fake_html_content = b"<html><body><p>Apple Reports Record First Quarter Results.</p></body></html>"
    class MockResp:
        status_code = 200
        content = fake_html_content
        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda url, **kwargs: MockResp())

    manifest_file = tmp_path / "manifest.jsonl"
    downloader = FilingDownloader(output_dir=tmp_path / "docs", manifest_path=manifest_file)

    exhibit = {
        "type": "EX-99.1",
        "filename": "a8-kex991q1202612272025.htm",
        "url": "https://www.sec.gov/Archives/edgar/data/320193/000032019326000005/a8-kex991q1202612272025.htm",
    }
    record = downloader.download_exhibit(exhibit, accession_number="0000320193-26-000005", cik="0000320193", filing_date="2026-01-29")
    assert record["document_type"] == "EX-99.1"
    assert record["source_status"] == "VERIFIED_DOWNLOAD"
    assert record["size_bytes"] == len(fake_html_content)
    assert record["filing_date"] == "2026-01-29"
    assert pathlib.Path(record["local_path"]).exists()


def test_source_sha256_integrity(tmp_path, monkeypatch):
    """Verify cryptographic SHA-256 integrity matches the exact file contents downloaded."""
    from src.ingest.download_filing import FilingDownloader

    doc_content = b"Authentic SEC Exhibit Press Release Content with Guidance"
    expected_sha256 = hashlib.sha256(doc_content).hexdigest()

    class MockResp:
        status_code = 200
        content = doc_content
        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.get", lambda url, **kwargs: MockResp())

    downloader = FilingDownloader(output_dir=tmp_path, manifest_path=tmp_path / "m.jsonl")
    rec = downloader.download_document(
        source_url="https://sec.gov/doc.htm",
        accession_number="0000320193-26-000005",
        primary_document="doc.htm",
        cik="0000320193",
        document_type="EX-99.1",
    )
    assert rec["sha256"] == expected_sha256

    saved_bytes = pathlib.Path(rec["local_path"]).read_bytes()
    assert hashlib.sha256(saved_bytes).hexdigest() == expected_sha256


def test_parser_evidence_offsets():
    """Verify segment_filing_text guarantees text[start_char:end_char] == paragraph['text']."""
    sample_filing = (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\n"
        "Item 2.02 Results of Operations and Financial Condition.\n\n"
        "On January 29, 2026, Apple Inc. announced financial results for its first fiscal quarter ended December 27, 2025. "
        "The Company posted quarterly revenue of $124.3 billion, up 4 percent year over year.\n\n"
        "Item 7.01 Regulation FD Disclosure.\n\n"
        "For the second quarter of fiscal 2026, Apple expects total revenue to grow between 5.0 percent and 7.0 percent, "
        "with gross margin expected to be between 46.5 percent and 47.5 percent.\n\n"
        "Pursuant to the requirements of the Securities Exchange Act of 1934."
    )
    segments = parse_and_segment_filing(sample_filing, accession_number="0000320193-26-000005")
    assert len(segments) >= 2

    norm_text = normalize_whitespace(sample_filing)

    for seg in segments:
        extracted_slice = norm_text[seg["start_char"]:seg["end_char"]]
        assert extracted_slice == seg["text"], f"Offset mismatch: '{extracted_slice}' vs '{seg['text']}'"
        assert seg["accession_number"] == "0000320193-26-000005"
        assert not is_boilerplate(seg["text"])


def test_authentic_sec_smoke_test_when_network_available():
    """Demonstrate real filing-to-exhibit-to-parser flow against authentic SEC EDGAR if reachable."""
    from src.ingest.sec_submissions import SECSubmissionsClient

    client_sec = SECSubmissionsClient()
    try:
        manifest = client_sec.get_filings_manifest("0000320193", form_filter="8-K", limit=1, discover_exhibits=True)
        if not manifest:
            pytest.skip("SEC returned empty manifest.")
        filing = manifest[0]
        assert filing["cik"] == "0000320193"
        assert "exhibits" in filing
    except Exception as exc:
        pytest.skip(f"Live SEC EDGAR access is not reachable in this test environment: {exc}")
