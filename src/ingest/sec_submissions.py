"""SEC Submissions metadata ingestion module.

Queries official SEC EDGAR API for company submissions metadata,
supports Form 8-K with Item 2.02, 5.02, 7.01, 8.01 and Exhibit 99.1 detection,
implements exponential backoff, rate-limiting, and caching.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional
import requests

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

logger = logging.getLogger(__name__)

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
DEFAULT_USER_AGENT = os.environ.get(
    "SEC_EDGAR_USER_AGENT",
    "FinEventResearchNLP/1.0 (research-contact: contact@finevent-project.org)",
)

TRACKED_CIKS = {
    "0000320193": "Apple Inc.",
    "0000789019": "Microsoft Corporation",
    "0001045810": "NVIDIA Corporation",
    "0000002488": "Advanced Micro Devices, Inc.",
    "0000019617": "JPMorgan Chase & Co.",
}


class SECSubmissionsClient:
    """Client for querying SEC EDGAR submissions API with rate limiting and retry."""

    def __init__(
        self,
        cache_dir: str | pathlib.Path = "data/raw/sec",
        user_agent: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_request_time = 0.0
        self.min_interval = 0.25  # Max 4 requests/sec (well within SEC 10 req/s limit)

        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _wait_rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def fetch_submissions(self, cik: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch submissions JSON for a 10-digit CIK with caching and exponential backoff."""
        cik_clean = str(cik).strip().zfill(10)
        cache_file = self.cache_dir / f"submissions_{cik_clean}.json"

        if not force_refresh and cache_file.exists():
            logger.info("Loading cached submissions for CIK %s from %s", cik_clean, cache_file)
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        url = SEC_SUBMISSIONS_URL.format(cik=cik_clean)
        backoff = 1.0

        for attempt in range(1, self.max_retries + 1):
            self._wait_rate_limit()
            logger.info("Requesting SEC submissions for CIK %s (Attempt %d/%d)", cik_clean, attempt, self.max_retries)
            try:
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                if response.status_code == 429:
                    logger.warning("SEC 429 Too Many Requests received. Backing off for %.1fs", backoff)
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                response.raise_for_status()
                payload = response.json()

                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                return payload
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Fetch failed on attempt %d: %s", attempt, exc)
                if attempt == self.max_retries:
                    if cache_file.exists():
                        logger.info("Falling back to existing cache file %s", cache_file)
                        with open(cache_file, "r", encoding="utf-8") as f:
                            return json.load(f)
                    raise
                time.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(f"Unable to fetch SEC submissions for CIK {cik_clean}")

    def discover_exhibits(self, cik: str, accession: str) -> List[Dict[str, str]]:
        """Dynamically discover exhibits by parsing the SEC EDGAR filing document index table.

        Uses BeautifulSoup to parse document index table and selects documents by actual document type.
        """
        import re
        import urllib.parse
        from bs4 import BeautifulSoup

        cik_clean = str(cik).strip().zfill(10)
        cik_int = int(cik_clean)
        accession_no_hyphens = accession.replace("-", "")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_hyphens}/{accession}-index.htm"
        base_archive_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_hyphens}/"

        exhibits: List[Dict[str, str]] = []
        try:
            self._wait_rate_limit()
            resp = requests.get(index_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                tables = soup.find_all("table")
                rows = []
                for table in tables:
                    rows.extend(table.find_all("tr"))
                if not rows:
                    rows = soup.find_all("tr")

                for row in rows:
                    cells = row.find_all(["td", "th"])
                    row_text = " ".join(c.get_text(strip=True) for c in cells)
                    if not re.search(r'\b(?:EX-99(?:\.\d+)?|EXHIBIT\s+99(?:\.\d+)?|99\.\d+)\b', row_text, re.IGNORECASE):
                        continue

                    link = row.find("a", href=True)
                    if not link:
                        continue

                    href = link["href"].strip()
                    raw_filename = href.split("/")[-1].split("?")[0]
                    full_url = urllib.parse.urljoin(base_archive_url, raw_filename)

                    doc_type = "EX-99.1"
                    for c in cells:
                        ctext = c.get_text(strip=True)
                        m_type = re.search(r'\b(EX-99(?:\.\d+)?|99\.\d+)\b', ctext, re.IGNORECASE)
                        if m_type:
                            doc_type = m_type.group(1).upper()
                            break

                    doc_desc = link.get_text(strip=True)
                    for c in cells:
                        ctext = c.get_text(strip=True)
                        if ctext and ctext not in {doc_type, raw_filename, link.get_text(strip=True)} and not ctext.isdigit():
                            doc_desc = ctext
                            break

                    exhibits.append({
                        "type": doc_type,
                        "doc_type": doc_type,
                        "filename": raw_filename,
                        "description": doc_desc,
                        "url": full_url,
                    })
        except Exception as exc:
            logger.warning("Could not dynamically resolve exhibits for %s: %s", accession, exc)

        return exhibits

    def get_filings_manifest(
        self,
        cik: str,
        form_filter: str = "8-K",
        limit: Optional[int] = None,
        force_refresh: bool = False,
        discover_exhibits: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract a deduplicated list of filing metadata with dynamic Exhibit discovery."""
        cik_clean = str(cik).strip().zfill(10)
        payload = self.fetch_submissions(cik_clean, force_refresh=force_refresh)
        cik_int = int(cik_clean)

        company_name = payload.get("name", TRACKED_CIKS.get(cik_clean, "UNKNOWN"))
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
            base_archive_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_hyphens}/"
            primary_url = f"{base_archive_url}{primary_doc}"

            exhibits: List[Dict[str, str]] = []
            if discover_exhibits:
                exhibits = self.discover_exhibits(cik_clean, accession)

            records.append(
                {
                    "cik": cik_clean,
                    "company_name": company_name,
                    "form": current_form,
                    "accession_number": accession,
                    "filing_date": filing_dates[i] if i < len(filing_dates) else "",
                    "report_date": report_dates[i] if i < len(report_dates) else "",
                    "primary_document": primary_doc,
                    "items": items_list[i] if i < len(items_list) else "",
                    "source_url": primary_url,
                    "archive_folder_url": base_archive_url,
                    "exhibits": exhibits,
                    "source_status": "VERIFIED_METADATA",
                    "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

            if limit and len(records) >= limit:
                break

        return records
