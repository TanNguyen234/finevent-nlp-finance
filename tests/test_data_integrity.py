"""Data integrity and provenance quality gate tests."""

from __future__ import annotations

import json
import pathlib
import pytest


def test_synthetic_not_in_gold():
    """Verify that no synthetic template record is mixed into gold.jsonl."""
    gold_path = pathlib.Path("data/labels/gold.jsonl")
    assert gold_path.exists()
    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for r in records:
        assert r.get("data_origin") != "SYNTHETIC_TEMPLATE", f"Contamination: synthetic record {r['id']} found in gold.jsonl"
        assert r.get("annotation_status") != "MACHINE_LABELED", f"Unverified machine label {r['id']} found in gold.jsonl"


def test_evidence_offsets_exact():
    """Verify that every record in gold.jsonl satisfies text[start:end] == evidence strictly."""
    gold_path = pathlib.Path("data/labels/gold.jsonl")
    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for r in records:
        text = r["text"]
        ev = r["evidence"]
        start = r["char_start"]
        end = r["char_end"]
        sliced = text[start:end]
        assert sliced == ev, f"Exact offset mismatch in {r['id']}: '{sliced}' != '{ev}'"


def test_accession_metadata_consistency():
    """Verify all records have properly formed CIK and SEC accession numbers."""
    gold_path = pathlib.Path("data/labels/gold.jsonl")
    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for r in records:
        cik = r["cik"]
        acc = r["accession_number"]
        assert len(cik) == 10 and cik.isdigit()
        assert len(acc.split("-")) == 3, f"Invalid SEC accession format in {r['id']}: {acc}"
        assert r["source_url"].startswith("https://www.sec.gov/Archives/edgar/data/")


def test_temporal_split_disjointness():
    """Verify train, val, and test splits have zero accession overlaps."""
    train_recs = [json.loads(l) for l in pathlib.Path("data/labels/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_recs = [json.loads(l) for l in pathlib.Path("data/labels/val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test_recs = [json.loads(l) for l in pathlib.Path("data/labels/test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    train_acc = {r["accession_number"] for r in train_recs}
    val_acc = {r["accession_number"] for r in val_recs}
    test_acc = {r["accession_number"] for r in test_recs}

    assert len(train_acc.intersection(val_acc)) == 0
    assert len(train_acc.intersection(test_acc)) == 0
    assert len(val_acc.intersection(test_acc)) == 0
