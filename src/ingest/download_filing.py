"""Filing document downloader module with SHA-256 provenance tracking.

Downloads 8-K filings and Exhibits from SEC EDGAR Archives adhering to the fair-access policy.
"""

from __future__ import annotations

import hashlib
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

DEFAULT_USER_AGENT = os.environ.get(
    "SEC_EDGAR_USER_AGENT",
    "FinEventResearchNLP/1.0 (research-contact: contact@finevent-project.org)",
)


class FilingDownloader:
    """Download SEC filing documents and track cryptographic hashes."""

    def __init__(
        self,
        output_dir: str | pathlib.Path = "data/raw/sec",
        user_agent: Optional[str] = None,
        request_interval_sec: float = 0.25,
        manifest_path: str | pathlib.Path = "data/raw/sec/download_manifest.jsonl",
    ) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = pathlib.Path(manifest_path)
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.request_interval_sec = request_interval_sec
        self.last_request_time = 0.0
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_interval_sec:
            time.sleep(self.request_interval_sec - elapsed)
        self.last_request_time = time.time()

    def record_manifest(self, record: Dict[str, Any]) -> None:
        """Append verified download record to manifest."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def download_document(
        self,
        source_url: str,
        accession_number: str,
        primary_document: str,
        cik: str = "UNKNOWN",
        document_type: str = "PRIMARY_8K",
        filing_date: str = "",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """Download document, calculate SHA-256 hash, and record provenance."""
        safe_accession = accession_number.replace("/", "_")
        safe_filename = f"{safe_accession}_{pathlib.Path(primary_document).name}"
        target_path = self.output_dir / safe_filename

        if not force_download and target_path.exists():
            content = target_path.read_bytes()
            sha256 = hashlib.sha256(content).hexdigest()
            record = {
                "cik": cik,
                "accession_number": accession_number,
                "form": "8-K",
                "filing_date": filing_date,
                "source_url": source_url,
                "document_type": document_type,
                "sha256": sha256,
                "local_path": str(target_path),
                "size_bytes": len(content),
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_status": "VERIFIED_DOWNLOAD",
            }
            return record

        self._rate_limit()
        logger.info("Downloading filing from %s", source_url)

        try:
            resp = requests.get(source_url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            content = resp.content

            target_path.write_bytes(content)
            sha256 = hashlib.sha256(content).hexdigest()

            record = {
                "cik": cik,
                "accession_number": accession_number,
                "form": "8-K",
                "filing_date": filing_date,
                "source_url": source_url,
                "document_type": document_type,
                "sha256": sha256,
                "local_path": str(target_path),
                "size_bytes": len(content),
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_status": "VERIFIED_DOWNLOAD",
            }
            self.record_manifest(record)
            return record
        except Exception as exc:
            logger.error("Failed to download filing from %s: %s", source_url, exc)
            raise

    def download_exhibit(
        self,
        exhibit: Dict[str, Any],
        accession_number: str,
        cik: str = "UNKNOWN",
        filing_date: str = "",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """Download an exhibit document (e.g. EX-99.1 press release) and track provenance."""
        source_url = exhibit.get("url") or exhibit.get("source_url", "")
        filename = exhibit.get("filename") or pathlib.Path(source_url).name
        doc_type = exhibit.get("type") or exhibit.get("doc_type") or "EX-99.1"
        return self.download_document(
            source_url=source_url,
            accession_number=accession_number,
            primary_document=filename,
            cik=cik,
            document_type=doc_type,
            filing_date=filing_date,
            force_download=force_download,
        )
