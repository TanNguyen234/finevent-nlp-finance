"""Real SEC Exhibit Data Harvester and Silver Dataset Bootstrapper.

Pillar 2: Expands the authentic dataset from 15 historical seed samples to 60-80
authentic SEC disclosure samples across AAPL, MSFT, and NVDA.
Guarantees:
1. Every paragraph originates from an authentic SEC Form 8-K Exhibit 99.1 press release.
2. Exact character offsets and SHA-256 provenance hashes are preserved.
3. Chronological/stratified split into train/val/test with 0 text leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Tuple

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from src.ingest.sec_submissions import SECSubmissionsClient, TRACKED_CIKS
from src.ingest.download_filing import FilingDownloader
from src.extract.segment import parse_and_segment_filing, is_boilerplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SEC-Expansion")

SILVER_DIR = pathlib.Path("data/silver")
RAW_SEC_DIR = pathlib.Path("data/raw/sec")

CLASSES = ["GUIDANCE_REVISION", "EARNINGS_RESULT", "MANAGEMENT_CHANGE", "OTHER"]

GUIDANCE_PATTERNS = [
    re.compile(r"\b(?:expects?|project(?:s|ed)?|guidance|outlook|ranges?\s+from|forecast(?:s|ed)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:gross\s+margin\s+(?:to\s+be|between|of)|revenue\s+(?:to\s+be|between|of))\b", re.IGNORECASE),
    re.compile(r"\b(?:fiscal\s+(?:quarter|year)\s+(?:guidance|outlook)|prior\s+guidance|revised?\s+guidance)\b", re.IGNORECASE),
    re.compile(r"\b(?:operating\s+expenses\s+to\s+be|tax\s+rate\s+of\s+approximately)\b", re.IGNORECASE),
]

EARNINGS_PATTERNS = [
    re.compile(r"\b(?:reported|posted|announced)\s+(?:financial|quarterly|fiscal|first|second|third|fourth)\s+results\b", re.IGNORECASE),
    re.compile(r"\b(?:quarterly|total)?\s*revenue\s+(?:was|of|grew|increased)\s+(?:to\s+)?\$[\d\.]+\s*(?:billion|million)\b", re.IGNORECASE),
    re.compile(r"\b(?:diluted\s+(?:earnings\s+per\s+share|eps))\s+(?:was|of|were)\s+\$[\d\.]+\b", re.IGNORECASE),
    re.compile(r"\b(?:net\s+income|operating\s+income|gross\s+profit)\s+(?:was|of)\s+\$[\d\.]+\b", re.IGNORECASE),
    re.compile(r"\b(?:cash\s+dividend\s+of|share\s+repurchase|returned\s+\$[\d\.]+|cash\s+flow\s+from\s+operations)\b", re.IGNORECASE),
    re.compile(r"\b(?:fourth|third|second|first)\s+quarter\s+(?:highlights|financial\s+summary|results)\b", re.IGNORECASE),
]

MANAGEMENT_PATTERNS = [
    re.compile(r"\b(?:appointed|resigned|named|departed?|elected|successor|stepping\s+down)\b", re.IGNORECASE),
    re.compile(r"\b(?:chief\s+executive\s+officer|chief\s+financial\s+officer|board\s+of\s+directors|executive\s+officer)\b", re.IGNORECASE),
    re.compile(r"\b(?:president\s+and\s+ceo|effective\s+(?:immediately|as\s+of))\b", re.IGNORECASE),
]


def classify_text_high_precision(text: str) -> str:
    """Classify SEC paragraph text with high-precision domain rules."""
    t_clean = text.strip()

    # 1. Management Change (very distinct)
    if any(p.search(t_clean) for p in MANAGEMENT_PATTERNS) and any(title in t_clean.lower() for title in ["officer", "director", "ceo", "cfo", "president", "board"]):
        return "MANAGEMENT_CHANGE"

    # 2. Guidance Revision (forward-looking language)
    guidance_hits = sum(1 for p in GUIDANCE_PATTERNS if p.search(t_clean))
    if guidance_hits >= 2 or (guidance_hits >= 1 and any(kw in t_clean.lower() for kw in ["guidance", "outlook", "expects", "forecast", "prior guidance", "full-year", "ranges from"])):
        # Make sure it's not purely a historical earnings statement without forward language
        if not ("diluted earnings per share of $" in t_clean.lower() and not any(k in t_clean.lower() for k in ["expect", "guidance", "outlook"])):
            return "GUIDANCE_REVISION"

    # 3. Earnings Results (backward-looking quarterly reporting)
    earnings_hits = sum(1 for p in EARNINGS_PATTERNS if p.search(t_clean))
    if earnings_hits >= 1 and any(kw in t_clean.lower() for kw in ["revenue", "diluted", "earnings", "net income", "operating income", "quarter ended", "fiscal quarter", "results"]):
        if not any(g in t_clean.lower() for g in ["expects", "outlook", "guidance", "forecast"]):
            return "EARNINGS_RESULT"

    return "OTHER"


def harvest_silver_disclosures() -> List[Dict[str, Any]]:
    """Harvest real 8-K EX-99.1 press release disclosures from tracked companies."""
    client = SECSubmissionsClient(cache_dir=RAW_SEC_DIR)
    downloader = FilingDownloader(output_dir=RAW_SEC_DIR)

    target_ciks = list(TRACKED_CIKS.items())

    all_harvested_samples: List[Dict[str, Any]] = []

    for cik, company_name in target_ciks:
        logger.info("Harvesting exhibits for %s (CIK: %s)...", company_name, cik)
        try:
            manifest = client.get_filings_manifest(
                cik=cik,
                form_filter="8-K",
                limit=12,
                discover_exhibits=True,
            )
        except Exception as exc:
            logger.warning("Could not fetch manifest for %s: %s", cik, exc)
            continue

        filings_with_exhibits = [m for m in manifest if m.get("exhibits")]
        logger.info("Found %d 8-K filings with exhibits for %s.", len(filings_with_exhibits), company_name)

        # Download up to 5 filings with exhibits per company
        for filing in filings_with_exhibits[:5]:
            acc = filing["accession_number"]
            filing_date = filing.get("filing_date", "UNKNOWN")
            for exh in filing.get("exhibits", []):
                doc_type = exh.get("doc_type") or exh.get("type", "EX-99.1")
                if "99" not in doc_type:
                    continue

                try:
                    logger.info("Downloading %s exhibit for %s (%s)...", doc_type, company_name, acc)
                    rec = downloader.download_exhibit(
                        exhibit=exh,
                        accession_number=acc,
                        cik=cik,
                        filing_date=filing_date,
                    )
                    local_path = pathlib.Path(rec["local_path"])
                    if not local_path.exists():
                        continue

                    html_content = local_path.read_text(encoding="utf-8", errors="replace")
                    sha256 = rec.get("sha256") or hashlib.sha256(html_content.encode("utf-8")).hexdigest()

                    # Parse & Segment
                    segments = parse_and_segment_filing(html_content, accession_number=acc)
                    logger.info("Extracted %d valid paragraph segments from %s.", len(segments), local_path.name)

                    for seg in segments:
                        text = seg["text"].strip()
                        if len(text) < 50 or len(text) > 800:
                            continue
                        if is_boilerplate(text):
                            continue

                        event_type = classify_text_high_precision(text)
                        sample = {
                            "id": f"SILVER_{cik}_{acc.replace('-', '')}_{seg['paragraph_index']}",
                            "cik": cik,
                            "company_name": company_name,
                            "filing_date": filing_date,
                            "accession_number": acc,
                            "form_type": "8-K",
                            "document_type": doc_type,
                            "event_type": event_type,
                            "text": text,
                            "start_char": seg["start_char"],
                            "end_char": seg["end_char"],
                            "provenance_hash": sha256,
                            "status": "silver_curated",
                        }
                        all_harvested_samples.append(sample)
                except Exception as exc:
                    logger.warning("Error processing exhibit %s: %s", exh.get("url"), exc)

    return all_harvested_samples


def build_curated_silver_splits(samples: List[Dict[str, Any]], target_total: int = 360) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Curate balanced, non-duplicate samples and split into train/val/test."""
    # Deduplicate by text content
    seen_texts = set()
    unique_samples = []
    for s in samples:
        norm_t = " ".join(s["text"].lower().split())
        if norm_t not in seen_texts:
            seen_texts.add(norm_t)
            unique_samples.append(s)

    # Group by class
    by_class: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CLASSES}
    for s in unique_samples:
        by_class[s["event_type"]].append(s)

    logger.info("Class distribution among unique harvested samples: %s", {c: len(items) for c, items in by_class.items()})

    # Stratify each class across train (70%), val (15%), test (15%)
    train_split: List[Dict[str, Any]] = []
    val_split: List[Dict[str, Any]] = []
    test_split: List[Dict[str, Any]] = []

    per_class_limit = max(70, target_total // len(CLASSES))
    for c in CLASSES:
        items = list(by_class[c][:per_class_limit])
        n_c = len(items)
        if n_c == 0:
            continue
        n_val = max(4, int(n_c * 0.15))
        n_test = max(4, int(n_c * 0.15))
        n_train = n_c - n_val - n_test
        if n_train < 2:
            n_train = max(1, n_c - 2)
            n_val = 1
            n_test = 1

        train_split.extend(items[:n_train])
        val_split.extend(items[n_train:n_train + n_val])
        test_split.extend(items[n_train + n_val:])

    return train_split, val_split, test_split


def save_and_verify_silver_splits(train: List[Dict[str, Any]], val: List[Dict[str, Any]], test: List[Dict[str, Any]]) -> None:
    """Save silver splits and generate split_manifest.json with integrity checks."""
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.jsonl": train,
        "val.jsonl": val,
        "test.jsonl": test,
    }

    manifest: Dict[str, Any] = {
        "dataset_name": "finevent-silver-expansion",
        "description": "Authentic SEC Form 8-K Exhibit 99.1 press release dataset for financial event classification",
        "created_at": "2026-10-12T13:00:00Z",
        "total_samples": len(train) + len(val) + len(test),
        "split_counts": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "classes": CLASSES,
        "splits": {},
    }

    for fname, data in splits.items():
        fpath = SILVER_DIR / fname
        with open(fpath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        sha256 = hashlib.sha256(fpath.read_bytes()).hexdigest()
        class_dist = {c: sum(1 for s in data if s["event_type"] == c) for c in CLASSES}
        manifest["splits"][fname] = {
            "samples": len(data),
            "sha256": sha256,
            "class_distribution": class_dist,
        }

    # Verify zero leakage
    train_texts = {s["text"] for s in train}
    val_texts = {s["text"] for s in val}
    test_texts = {s["text"] for s in test}

    assert len(train_texts.intersection(val_texts)) == 0, "Leakage detected between train and val!"
    assert len(train_texts.intersection(test_texts)) == 0, "Leakage detected between train and test!"
    assert len(val_texts.intersection(test_texts)) == 0, "Leakage detected between val and test!"

    manifest["leakage_check"] = "0 text leakage verified"

    with open(SILVER_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Silver dataset successfully generated at %s (Total: %d samples, Train: %d, Val: %d, Test: %d).",
                SILVER_DIR, len(train) + len(val) + len(test), len(train), len(val), len(test))


def main() -> None:
    logger.info("Starting SEC Silver Dataset Expansion Pipeline...")
    harvested = harvest_silver_disclosures()
    logger.info("Total harvested paragraph candidates: %d", len(harvested))
    train, val, test = build_curated_silver_splits(harvested)
    save_and_verify_silver_splits(train, val, test)
    logger.info("Pillar 2 SEC Silver Dataset Expansion completed successfully.")


if __name__ == "__main__":
    main()
