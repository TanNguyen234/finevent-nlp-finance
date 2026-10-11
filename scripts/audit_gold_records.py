"""Audit all 20 records in gold.jsonl for source authenticity, accession validity, and future dates."""

import json
import pathlib

GOLD_PATH = pathlib.Path("data/labels/gold.jsonl")
lines = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

print(f"Total records in gold.jsonl: {len(lines)}")
for i, r in enumerate(lines):
    rid = r.get("id")
    cik = r.get("cik")
    company = r.get("company_name")
    date = r.get("filing_date")
    acc = r.get("accession_number")
    ev_type = r.get("event_type")
    metric = r.get("metric")
    text = r.get("text")
    char_start = r.get("char_start")
    char_end = r.get("char_end")
    evidence = r.get("evidence")
    
    # Check offset
    offset_ok = (text[char_start:char_end] == evidence)
    
    # Check year
    year = int(date.split("-")[0])
    is_future = (year >= 2025)
    
    # Check consensus
    has_consensus = "consensus" in text.lower()
    
    print(f"Index {i:02d} | ID: {rid} | Date: {date} | Year: {year} | Future: {is_future} | Acc: {acc} | Type: {ev_type} | OffsetOK: {offset_ok} | Consensus: {has_consensus}")
