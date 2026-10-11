# FinEvent — Financial Event Extraction and Guidance Revision Reasoning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/UI-React_18-61DAFB.svg)](https://react.dev/)
[![Kaggle GPU T4](https://img.shields.io/badge/Kaggle-GPU_T4-20BEFF.svg)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FinEvent is a domain-specific Natural Language Processing system designed to extract evidence-anchored events and detect corporate guidance revisions from authentic SEC EDGAR Form 8-K filings and Item 2.02 / Exhibit 99.1 earnings releases.

Given a company CIK and historical disclosure filings, FinEvent determines:
1. Event classification across four canonical classes (`GUIDANCE_REVISION`, `EARNINGS_RESULT`, `MANAGEMENT_CHANGE`, `OTHER`).
2. Multi-metric argument extraction (metric name, fiscal period, accounting basis, currency, and numerical range).
3. Exact character-level evidence attribution (`text[char_start:char_end] == evidence`) linked to verified SEC EDGAR source URLs.
4. Pairwise temporal guidance revisions evaluated through a strict 7-tuple Financial Guidance Identity matching engine.

---

## Architecture

```
                       +-------------------------+
                       |  SEC EDGAR Filings API  |
                       |  (8-K & Exhibit 99.1)   |
                       +------------+------------+
                                    | Ingest & Provenance (SHA-256)
                                    v
                       +-------------------------+
                       |   Segment & Offsets     |
                       |  (Normalized Text)      |
                       +------------+------------+
                                    |
          +-------------------------+-------------------------+
          v                                                   v
+-------------------------+                         +-------------------------+
|     E0 / E1 Baseline    |                         |    E2 Transformer GPU   |
|   (Keyword & TF-IDF)    |                         |  (FinBERT / DeBERTa-v3) |
+------------+------------+                         +------------+------------+
             |                                                   |
             +--------------------------+------------------------+
                                        | Event Probabilities
                                        v
                       +-------------------------+
                       | Multi-Metric Extractor  |
                       | (Unit Normalization)    |
                       +------------+------------+
                                        |
                                        v
                       +-------------------------+
                       | Guidance Identity Engine|
                       | (7-Tuple Compatibility) |
                       +------------+------------+
                                        |
          +-----------------------------+-----------------------------+
          v                                                           v
+-------------------------+                                 +-------------------------+
|     FastAPI Service     |                                 |     React Dashboard     |
|   (/companies/.../diff) |                                 |  (Dynamic System State) |
+-------------------------+                                 +-------------------------+
```

---

## Dataset Architecture and Integrity

To eliminate synthetic contamination and ensure research-grade integrity, FinEvent enforces strict separation between data subsets:

- `data/labels/`: Audited Authentic SEC Historical Benchmark (15 records from historical Form 8-K filings of Apple Inc. and Microsoft Corporation between 2019 and 2024 with 0% character-offset error). Partitioned via Chronological Temporal Split:
  - `train.jsonl`: 9 samples (2019–2020)
  - `val.jsonl`: 3 samples (2021)
  - `test.jsonl`: 3 samples (2024 held-out)
  - `split_manifest.json`: Content hash and provenance metadata (`gold_sha256: 1d7d535c1b...`).
- `data/quarantine/`: Quarantined records (`quarantined_records.jsonl`) isolating 5 historical records identified with fabricated future dates (2025–2026) and ungrounded consensus claims during the forensic audit.
- `data/synthetic/`: Isolated synthetic template samples (200 records) reserved exclusively for parser sanity checks and local tests.
- `data/annotations/`: Active annotation queue (`annotation_queue.jsonl`) and guidelines (`annotation_guidelines.md`) for continuous review.
- `data/processed/`: Operational database (`operational_events.jsonl`) serving FastAPI endpoints, preventing benchmark contamination in production serving.

---

## Financial Guidance Identity Matching

A common pitfall in financial NLP is treating ordinary quarter-over-quarter revenue growth or historical earnings results as guidance revisions. FinEvent resolves this by requiring pairwise equivalence across a 7-tuple identity before calculating directional changes:

$$\text{Identity} = (\text{CIK}, \text{Metric}, \text{TargetFiscalPeriod}, \text{AccountingBasis}, \text{Currency}, \text{Unit}, \text{Scope})$$

### Compatibility Rules
- Different target periods (e.g. Q1 vs Q2) are rejected with `Target fiscal period mismatch`.
- Actual earnings results paired with forward-looking guidance are rejected with `Cannot compare backward-looking EARNINGS_RESULT with forward-looking GUIDANCE`.
- Accounting basis mismatches (GAAP vs Non-GAAP) are rejected with `Accounting basis mismatch`.
- Chronological inversions (prior date occurring after current date) are rejected with `Chronological inversion`.
- Units and currencies are normalized (`USD_B` = 1,000 * `USD_M`), but incompatible dimensions (`PERCENT` vs `USD_B`) are rejected with `Incompatible units`.
- Explicit null handling (`is not None`) preserves zero numeric values ($0.00 diluted EPS) from being discarded.

### Ablation Findings
In dynamic execution across cross-domain disclosure pairs (`reports/ablation_results.json`):
- Without Guidance Identity (A1): 3 false positive revisions (100% false alarm rate).
- Strict Guidance Identity matching (A0): 0 false positive revisions (100% precision).

---

## Experimental Benchmarks

### 1. Authentic SEC Held-Out Benchmark (2024 Held-Out Test Set)

Evaluated on the cleansed historical test set (`data/labels/test.jsonl`, 3 held-out 2024 filings across Apple and Microsoft):

| Experiment ID | Architecture | Execution Platform | Test Macro-F1 | Test Weighted-F1 | Accuracy | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **E0_REAL_SEC** | Keyword Rule Baseline | Local CPU | 0.5000 | 0.6667 | 0.6667 | Verified Local |
| **E1_REAL_SEC** | TF-IDF (1,2-gram) + LogReg | Local CPU | 0.7500 | 1.0000 | 1.0000 | Verified Local |

*Honest Assessment*: A 3-sample held-out split is an exploratory sanity check on authentic SEC filings. Scaling the verified benchmark to >=200 human-annotated filings is required for robust generalization claims.

### 2. Historical Kaggle Cloud Synthetic Sanity Run

Forensic audit established that historical Kaggle kernel `foxy0505/finevent-nlp-training-gpu` evaluated on the legacy 290-record synthetic dataset (56-sample test set) fetched from GitHub raw commit `662aabe`:

| Experiment ID | Architecture | Hardware | Split Strategy | Pre-Val F1 | Post-Val F1 | Test Macro-F1 | Status & Audit Finding |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **E2_A_DEBERTA** | `microsoft/deberta-v3-small` | Tesla T4 GPU | Legacy Synthetic Split | 0.0952 | 0.3588 | 0.3677 | Historical Synthetic Sanity |
| **E2_B_FINBERT** | `ProsusAI/finbert` | Tesla T4 GPU | Legacy Synthetic Split | 0.3221 | 1.0000 | 1.0000 | Historical Synthetic Sanity (Memorization) |

*Audit Finding*: FinBERT's 1.0000 F1 on the legacy test set represents template memorization on synthetic permutations, not genuine generalization on noisy SEC filings. Model comparisons across datasets with different sample counts or origins have been completely separated.

---

## Reproduction and Quickstart

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Test Suite and Verification
```powershell
python -m pytest tests/ -v
python -m compileall src app kaggle
```

### 3. Ingestion and Baseline Execution
```powershell
# Run E0 and E1 baselines on cleansed real data
python -m src.train.baseline

# Execute dynamic Financial Guidance Identity ablation
python -m src.eval.run_ablation
```

### 4. Cloud Training & Rule 1 Compliance
In accordance with Rule 1 ("Cloud-First Execution: Never download heavy model checkpoints locally"), heavy neural weights (`*.safetensors`, `*.pt`) remain in cloud storage / Kaggle kernel outputs. Only lightweight evaluation manifests, logs, and metrics reside locally.

---

## Directory Layout

```
finevent-nlp-finance/
├── app/                  # FastAPI backend application
├── data/
│   ├── annotations/      # Annotation guidelines and review queue
│   ├── labels/           # Audited historical SEC gold dataset (train, val, test)
│   ├── processed/        # Operational database for API serving
│   ├── quarantine/       # Quarantined non-grounded / future-dated records
│   └── synthetic/        # Isolated synthetic template datasets
├── frontend/             # React dashboard frontend
├── kaggle/               # Cloud GPU training pipeline and metadata
├── reports/              # Experiment manifests, logs, forensic audits
│   ├── gold_provenance_audit.md   # Record-by-record gold label audit
│   ├── kaggle_forensic_audit.md   # Cloud GPU kernel forensic verification
│   ├── remediation_checklist.md  # Status of all P0 and P1 audit findings
│   └── results.md                 # Partitioned research benchmark report
├── src/
│   ├── eval/             # Evaluation metrics and dynamic ablation runners
│   ├── extract/          # Segmenters, argument extraction, 7-tuple change logic
│   ├── ingest/           # SEC EDGAR crawlers, submissions, and parsers
│   └── train/            # Local E0/E1 baseline training pipelines
└── tests/                # 33 verified unit and integration tests
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
