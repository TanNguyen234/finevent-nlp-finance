"""Filing document downloader module with SHA-256 provenance tracking.

Downloads 8-K filings from SEC EDGAR Archives adhering to the fair-access policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "FinEventNLPResearch/1.0 (academic-contact: researcher@finevent-project.org)"


class FilingDownloader:
    """Download SEC filing documents and track cryptographic hashes."""

    def __init__(
        self,
        output_dir: str | pathlib.Path = "data/raw/sec",
        user_agent: str = DEFAULT_USER_AGENT,
        request_interval_sec: float = 0.2,  # Max ~5 requests/sec (well under SEC 10/sec cap)
    ) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.request_interval_sec = request_interval_sec
        self.last_request_time = 0.0
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _rate_limit(self) -> None:
        """Enforce rate limiting between outbound requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_interval_sec:
            time.sleep(self.request_interval_sec - elapsed)
        self.last_request_time = time.time()

    def download_document(
        self,
        source_url: str,
        accession_number: str,
        primary_document: str,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """Download document and calculate SHA-256 hash."""
        safe_accession = accession_number.replace("/", "_")
        safe_filename = f"{safe_accession}_{pathlib.Path(primary_document).name}"
        target_path = self.output_dir / safe_filename

        if not force_download and target_path.exists():
            content = target_path.read_bytes()
            sha256 = hashlib.sha256(content).hexdigest()
            return {
                "accession_number": accession_number,
                "file_path": str(target_path),
                "sha256": sha256,
                "size_bytes": len(content),
                "source_url": source_url,
                "status": "cached",
            }

        self._rate_limit()
        logger.info("Downloading filing from %s", source_url)

        try:
            resp = requests.get(source_url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            content = resp.content

            target_path.write_bytes(content)
            sha256 = hashlib.sha256(content).hexdigest()

            return {
                "accession_number": accession_number,
                "file_path": str(target_path),
                "sha256": sha256,
                "size_bytes": len(content),
                "source_url": source_url,
                "status": "downloaded",
            }
        except Exception as exc:
            logger.error("Failed to download filing from %s: %s", source_url, exc)
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloader = FilingDownloader()
    print("FilingDownloader ready.")
