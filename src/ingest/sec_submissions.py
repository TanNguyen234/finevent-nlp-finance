"""SEC Submissions metadata ingestion module.

Queries the official SEC EDGAR API for company submissions metadata,
filters 8-K filings, caches the raw JSON, and produces normalized filing manifests.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
DEFAULT_USER_AGENT = "FinEventNLPResearch/1.0 (academic-contact: researcher@finevent-project.org)"


class SECSubmissionsClient:
    """Client for querying SEC EDGAR submissions API."""

    def __init__(
        self,
        cache_dir: str | pathlib.Path = "data/raw/sec",
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = 30,
    ) -> None:
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.timeout = timeout
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

    def fetch_submissions(self, cik: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch submissions JSON for a 10-digit CIK.

        Uses local cache if available unless force_refresh is True.
        """
        cik_clean = str(cik).strip().zfill(10)
        cache_file = self.cache_dir / f"submissions_{cik_clean}.json"

        if not force_refresh and cache_file.exists():
            logger.info("Loading cached submissions for CIK %s from %s", cik_clean, cache_file)
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        url = SEC_SUBMISSIONS_URL.format(cik=cik_clean)
        logger.info("Fetching SEC submissions from %s", url)

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            return payload
        except Exception as exc:
            logger.warning("Network request failed for CIK %s: %s. Checking cache fallback...", cik_clean, exc)
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise

    def get_filings_manifest(
        self,
        cik: str,
        form_filter: str = "8-K",
        limit: Optional[int] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract a deduplicated list of filing metadata for a given form (default: 8-K)."""
        cik_clean = str(cik).strip().zfill(10)
        payload = self.fetch_submissions(cik_clean, force_refresh=force_refresh)
        cik_int = int(cik_clean)

        recent = payload.get("filings", {}).get("recent", {})
        if not recent or "form" not in recent:
            return []

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])
        items_list = recent.get("items", [])

        records: List[Dict[str, Any]] = []
        seen_accessions = set()

        num_records = len(forms)
        for i in range(num_records):
            current_form = forms[i]
            if form_filter and current_form != form_filter:
                continue

            accession = accessions[i]
            if accession in seen_accessions:
                continue
            seen_accessions.add(accession)

            primary_doc = primary_docs[i]
            accession_no_hyphens = accession.replace("-", "")
            archive_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{accession_no_hyphens}/{primary_doc}"
            )

            records.append(
                {
                    "cik": cik_clean,
                    "company_name": payload.get("name", "UNKNOWN"),
                    "form": current_form,
                    "accession_number": accession,
                    "filing_date": filing_dates[i] if i < len(filing_dates) else "",
                    "report_date": report_dates[i] if i < len(report_dates) else "",
                    "primary_document": primary_doc,
                    "items": items_list[i] if i < len(items_list) else "",
                    "source_url": archive_url,
                }
            )

            if limit and len(records) >= limit:
                break

        return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = SECSubmissionsClient()
    # Test Apple Inc. CIK: 0000320193
    try:
        filings = client.get_filings_manifest("0000320193", form_filter="8-K", limit=5)
        print(f"Retrieved {len(filings)} 8-K filings for Apple (CIK: 0000320193)")
        if filings:
            print("Latest 8-K filing:", filings[0])
    except Exception as err:
        print("SEC fetch error (check network or user-agent):", err)
