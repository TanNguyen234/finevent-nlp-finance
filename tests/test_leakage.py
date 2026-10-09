"""Test data integrity and prevent temporal data leakage."""

from __future__ import annotations

import json
import pathlib
import pytest


def load_jsonl(path: pathlib.Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def test_temporal_split_boundaries():
    """Verify that train, validation, and test sets are strictly partitioned by year."""
    train_file = pathlib.Path("data/labels/train.jsonl")
    val_file = pathlib.Path("data/labels/val.jsonl")
    test_file = pathlib.Path("data/labels/test.jsonl")

    assert train_file.exists() and val_file.exists() and test_file.exists()

    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    test_data = load_jsonl(test_file)

    # 1. Train set must only contain filings from 2019 to 2023
    for r in train_data:
        date_str = r["filing_date"]
        year = int(date_str.split("-")[0])
        assert year <= 2023, f"Data leakage in train set: record {r['id']} has filing year {year} > 2023"

    # 2. Validation set must only contain filings from 2024
    for r in val_data:
        date_str = r["filing_date"]
        year = int(date_str.split("-")[0])
        assert year == 2024, f"Data leakage in val set: record {r['id']} has filing year {year} != 2024"

    # 3. Test set must only contain filings from 2025 to 2026
    for r in test_data:
        date_str = r["filing_date"]
        year = int(date_str.split("-")[0])
        assert year >= 2025, f"Data leakage in test set: record {r['id']} has filing year {year} < 2025"


def test_disjoint_accession_numbers():
    """Verify there is zero overlap between accession numbers in train, val, and test splits."""
    train_acc = {r["accession_number"] for r in load_jsonl(pathlib.Path("data/labels/train.jsonl"))}
    val_acc = {r["accession_number"] for r in load_jsonl(pathlib.Path("data/labels/val.jsonl"))}
    test_acc = {r["accession_number"] for r in load_jsonl(pathlib.Path("data/labels/test.jsonl"))}

    train_val_overlap = train_acc.intersection(val_acc)
    assert len(train_val_overlap) == 0, f"Leakage: overlapping accession numbers between train and val: {train_val_overlap}"

    train_test_overlap = train_acc.intersection(test_acc)
    assert len(train_test_overlap) == 0, f"Leakage: overlapping accession numbers between train and test: {train_test_overlap}"

    val_test_overlap = val_acc.intersection(test_acc)
    assert len(val_test_overlap) == 0, f"Leakage: overlapping accession numbers between val and test: {val_test_overlap}"
