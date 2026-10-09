"""HTML Document cleaner and paragraph segmenter with character offset tracking.

Parses SEC 8-K filings HTML, cleans markup, and splits the document into
evidence-ready paragraph snippets with character offsets and section headers.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Common boilerplate phrases to filter or down-weight
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*SIGNATURES?\s*$", re.IGNORECASE),
    re.compile(r"Pursuant to the requirements of the Securities Exchange Act", re.IGNORECASE),
    re.compile(r"^\s*EXHIBIT\s+\d+[\.\d]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*Item\s+\d+\.\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*UNITED STATES SECURITIES AND EXCHANGE COMMISSION\s*$", re.IGNORECASE),
]

ITEM_PATTERN = re.compile(r"(Item\s+\d+\.\d+[\s\w\-\,\:]+)", re.IGNORECASE)


def is_boilerplate(text: str) -> bool:
    """Check if snippet is standard SEC legal boilerplate."""
    stripped = text.strip()
    if len(stripped) < 25:
        return True
    for pattern in BOILERPLATE_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def clean_html_to_text(html_content: str) -> str:
    """Remove scripts, styles, and extract plain text."""
    soup = BeautifulSoup(html_content, "html.parser")
    for elem in soup(["script", "style", "meta", "noscript", "head"]):
        elem.decompose()
    return soup.get_text(separator="\n")


def segment_filing_text(
    text: str,
    accession_number: str = "",
    min_length: int = 40,
) -> List[Dict[str, Any]]:
    """Segment raw text into paragraphs with character offsets.

    Returns a list of paragraph dicts with:
    - paragraph_index: int
    - text: str
    - start_char: int
    - end_char: int
    - section: str (Item 2.02, Item 5.02, etc. if identified)
    - accession_number: str
    """
    paragraphs: List[Dict[str, Any]] = []
    current_section = "GENERAL"

    # Find paragraphs by double newlines or block splits
    raw_blocks = re.split(r"\n\s*\n+", text)
    current_offset = 0

    idx = 0
    for block in raw_blocks:
        block_clean = re.sub(r"[ \t]+", " ", block).strip()
        start_char = text.find(block, current_offset)
        if start_char == -1:
            start_char = current_offset
        end_char = start_char + len(block)
        current_offset = end_char

        # Check if block defines a new section (e.g. Item 2.02)
        item_match = ITEM_PATTERN.search(block_clean)
        if item_match:
            current_section = item_match.group(1).strip()

        # Skip short snippets or boilerplate
        if len(block_clean) < min_length or is_boilerplate(block_clean):
            continue

        paragraphs.append(
            {
                "paragraph_index": idx,
                "text": block_clean,
                "start_char": start_char,
                "end_char": end_char,
                "section": current_section,
                "accession_number": accession_number,
            }
        )
        idx += 1

    return paragraphs


def parse_and_segment_filing(
    html_or_text: str,
    accession_number: str = "",
) -> List[Dict[str, Any]]:
    """Parse HTML/text filing directly into segments."""
    if "<html" in html_or_text.lower() or "<body" in html_or_text.lower():
        cleaned_text = clean_html_to_text(html_or_text)
    else:
        cleaned_text = html_or_text
    return segment_filing_text(cleaned_text, accession_number=accession_number)


if __name__ == "__main__":
    sample = """
    <html><body>
    <h1>Item 2.02 Results of Operations and Financial Condition.</h1>
    <p>On October 31, 2024, Apple Inc. announced its financial results for its fiscal 2024 fourth quarter.
    The Company posted quarterly revenue of $94.9 billion, up 6 percent year over year, and quarterly diluted earnings per share of $0.97.</p>
    <p>Item 7.01 Regulation FD Disclosure.</p>
    <p>The company now expects total revenue for the first quarter of fiscal 2025 to grow between 7.5% and 8.5% compared with the prior year period.</p>
    <p>Pursuant to the requirements of the Securities Exchange Act of 1934, the registrant has duly caused this report to be signed.</p>
    </body></html>
    """
    segments = parse_and_segment_filing(sample, "0000320193-24-000100")
    for s in segments:
        print(f"[{s['paragraph_index']}] ({s['section']}) {s['text'][:80]}... [offsets: {s['start_char']}-{s['end_char']}]")
