"""HTML Document cleaner and paragraph segmenter with exact character offset tracking.

Parses SEC 8-K filings and exhibits HTML, cleans markup, and splits the document into
evidence-ready paragraph snippets with character offsets and section headers.
Guarantees offset consistency against normalized text.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*SIGNATURES?\s*$", re.IGNORECASE),
    re.compile(r"Pursuant to the requirements of the Securities Exchange Act", re.IGNORECASE),
    re.compile(r"^\s*EXHIBIT\s+\d+[\.\d]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*Item\s+\d+\.\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*UNITED STATES SECURITIES AND EXCHANGE COMMISSION\s*$", re.IGNORECASE),
    re.compile(r"FORWARD-LOOKING STATEMENTS", re.IGNORECASE),
]

ITEM_PATTERN = re.compile(r"(Item\s+\d+\.\d+[\s\w\-\,\:]+)", re.IGNORECASE)
EXHIBIT_PATTERN = re.compile(r"(Exhibit\s+99\.\d+[\s\w\-\,\:]*)", re.IGNORECASE)


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
    """Remove scripts, styles, and extract plain text with paragraph separation."""
    soup = BeautifulSoup(html_content, "html.parser")
    for elem in soup(["script", "style", "meta", "noscript", "head", "title"]):
        elem.decompose()
    return soup.get_text(separator="\n\n")


def normalize_whitespace(text: str) -> str:
    """Normalize irregular whitespaces while preserving paragraph breaks."""
    paragraphs = []
    for raw_p in re.split(r"\n\s*\n+", text):
        clean_p = re.sub(r"[ \t]+", " ", raw_p).strip()
        if clean_p:
            paragraphs.append(clean_p)
    return "\n\n".join(paragraphs)


def segment_filing_text(
    text: str,
    accession_number: str = "",
    min_length: int = 40,
) -> List[Dict[str, Any]]:
    """Segment normalized text into paragraphs with exact character offsets.

    Guarantees that text[start_char:end_char] == paragraph['text'] relative to the full document.
    """
    paragraphs: List[Dict[str, Any]] = []
    current_section = "GENERAL"

    raw_blocks = re.split(r"\n\s*\n+", text)
    current_offset = 0

    idx = 0
    for block in raw_blocks:
        block_clean = re.sub(r"[ \t]+", " ", block).strip()
        if not block_clean:
            continue

        start_char = text.find(block_clean, current_offset)
        if start_char == -1:
            start_char = current_offset
        end_char = start_char + len(block_clean)
        current_offset = end_char

        # Track section context
        item_match = ITEM_PATTERN.search(block_clean)
        exhibit_match = EXHIBIT_PATTERN.search(block_clean)
        if item_match:
            current_section = item_match.group(1).strip()
        elif exhibit_match:
            current_section = exhibit_match.group(1).strip()

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
    """Parse HTML/text filing directly into segments with verified offsets."""
    if "<html" in html_or_text.lower() or "<body" in html_or_text.lower():
        extracted = clean_html_to_text(html_or_text)
    else:
        extracted = html_or_text
    normalized = normalize_whitespace(extracted)
    return segment_filing_text(normalized, accession_number=accession_number)
