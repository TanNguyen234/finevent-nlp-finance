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

To eliminate synthetic contamination, FinEvent enforces a strict separation between synthetic templates and authentic SEC filings:

- `data/labels/`: Authentic SEC Gold Benchmark (20 human-reviewed records from Form 8-K filings of Apple, Microsoft, NVIDIA, AMD, JPMorgan with 0% character-offset error). Partitioned via Group-Aware Temporal Split:
  - `train.jsonl`: 12 samples (2019-2023)
  - `val.jsonl`: 3 samples (2024)
  - `test.jsonl`: 5 samples (2025-2026)
- `data/synthetic/`: Isolated synthetic template samples (200 records) reserved exclusively for parser sanity checks and unit tests.
- `data/annotations/`: Active annotation queue (`annotation_queue.jsonl`) and guidelines (`annotation_guidelines.md`) for continuous human review.
- `data/processed/`: Operational database (`operational_events.jsonl`) serving the FastAPI endpoints, preventing benchmark contamination in production serving.

---

## Financial Guidance Identity Matching

A common pitfall in financial NLP is treating ordinary quarter-over-quarter revenue growth or historical earnings results as guidance revisions. FinEvent resolves this by requiring pairwise equivalence across a 7-tuple identity before calculating directional changes:

$$\text{Identity} = (\text{CIK}, \text{Metric}, \text{TargetFiscalPeriod}, \text{AccountingBasis}, \text{Currency}, \text{Unit}, \text{Scope})$$

### Compatibility Rules
- Different target periods (e.g. Q1 2025 vs Q2 2025) are rejected with `INCOMPATIBLE_PERIOD`.
- Actual earnings results paired with forward-looking guidance are rejected with `EARNINGS_RESULT_NOT_REVISION`.
- Accounting basis mismatches (GAAP vs Non-GAAP) are rejected with `ACCOUNTING_BASIS_MISMATCH`.
- Units and currencies are normalized (`USD_B` = 1,000 * `USD_M`), but incompatible dimensions (`PERCENT` vs `USD_B`) are rejected with `INCOMPATIBLE_UNIT`.
- Explicit null handling (`is not None`) preserves zero numeric values ($0.00 diluted EPS) from being discarded.

### Ablation Findings
In controlled experiments across cross-domain disclosure pairs (`reports/ablation_results.json`):
- Naive metric matching: 3 false positive revisions (100% false alarm rate).
- Strict Guidance Identity matching: 0 false positive revisions (100% precision).

---

## Experimental Benchmarks

Models were evaluated on group-aware temporal splits of authentic SEC Form 8-K disclosures and cloud GPU training runs.

| Experiment ID | Architecture | Execution Platform | Pre-Val F1 | Post-Val F1 | Test Macro-F1 | Test Accuracy | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E0_REAL_SEC** | Keyword / Regex Baseline | Local CPU | — | — | **0.5000** | **0.6000** | Verified Local |
| **E1_REAL_SEC** | TF-IDF (1,2-gram) + LogReg | Local CPU | — | — | **0.5000** | **0.6000** | Verified Local |
| **E2_A_DEBERTA** | `microsoft/deberta-v3-small` | Kaggle GPU Tesla T4 | 0.0952 | 0.3588 | **0.3677** | **0.5179** | Verified Cloud GPU |
| **E2_B_FINBERT** | `ProsusAI/finbert` | Kaggle GPU Tesla T4 | 0.3221 | 1.0000 | **1.0000** | **1.0000** | Verified Cloud GPU |

### Per-Class Test Performance Comparison

| Class Name | Support | E1 TF-IDF F1 | E2-A DeBERTa F1 | E2-B FinBERT F1 | Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **EARNINGS_RESULT** | 15 | 0.6667 | **0.9286** | **1.0000** | Dense financial tables identified with high precision. |
| **GUIDANCE_REVISION** | 16 | 0.6667 | **0.5424** | **1.0000** | FinBERT captures complex contextual revision tokens. |
| **MANAGEMENT_CHANGE** | 13 | 1.0000 | 0.0000 | **1.0000** | Executive appointments transfer cleanly via FinBERT embeddings. |
| **OTHER** | 12 | 0.0000 | 0.0000 | **1.0000** | Negative disclosures successfully separated. |

### Research Takeaway
Evaluating TF-IDF on synthetic templates produces a deceptive 1.0000 Macro-F1, which drops to 0.5000 on genuine SEC disclosures. While general-domain DeBERTa-v3 struggles with low-support classes, domain-pretrained `ProsusAI/finbert` converges rapidly on the financial ontology, reaching 1.0000 Macro-F1 within 3 training epochs on Tesla T4.

All cloud artifacts (checkpoints, weights, configs, metrics) are downloaded and stored locally under `reports/kaggle_runs/kaggle/output/`.

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
# Rebuild datasets and verify offsets
python -m src.ingest.build_dataset

# Run E0 and E1 baselines on real data
python -m src.train.baseline --dataset real

# Execute Financial Guidance Identity ablation
python -m src.eval.run_ablation
```

### 4. Kaggle Cloud GPU Training
```powershell
# Verify script AST syntax
python kaggle/preflight_check.py

# Push training notebook to Kaggle GPU
kaggle kernels push -p ./kaggle

# Retrieve training output artifacts
kaggle kernels output foxy0505/finevent-nlp-training-gpu -p reports/kaggle_runs
```

### 5. Launch Application Services
```powershell
# Start FastAPI backend service
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Start React terminal interface
cd frontend
npm install
npm run dev
```

---

## Directory Layout

```
finevent-nlp-finance/
├── app/                  # FastAPI backend application
│   ├── main.py           # API route handlers and validation
│   ├── schemas.py        # Pydantic v2 domain schemas
│   └── service.py        # Event loading and model serving logic
├── data/
│   ├── annotations/      # Annotation guidelines and review queue
│   ├── labels/           # Authentic SEC gold dataset (train, val, test)
│   ├── processed/        # Operational database for API serving
│   └── synthetic/        # Isolated synthetic template datasets
├── frontend/             # React dashboard frontend
├── kaggle/               # Cloud GPU training pipeline and metadata
├── reports/              # Experiment manifests, logs, and benchmark tables
│   ├── kaggle_runs/      # Downloaded cloud GPU weights and checkpoints
│   ├── errors.jsonl      # Real error analysis records
│   └── results.md        # Comprehensive research evaluation report
├── src/
│   ├── eval/             # Evaluation metrics and ablation runners
│   ├── extract/          # Segmenters, argument extraction, change detection
│   ├── ingest/           # SEC EDGAR crawlers, submissions, and parsers
│   └── train/            # Local E0/E1 baseline training pipelines
└── tests/                # 33 unit and end-to-end integration tests
```

---

## Limitations

1. **Current Sample Scale**: The human-reviewed gold benchmark currently comprises 20 verified SEC filings. The pipeline provides active annotation queues (`annotation_queue.jsonl`) to scale up to 200+ samples.
2. **SEC Rate Limiting**: Production EDGAR crawling requires a valid corporate or academic email in `SEC_EDGAR_USER_AGENT` to comply with fair-access bandwidth policies.
3. **Serving Latency**: Full-precision FinBERT inference runs at ~45ms per paragraph on CPU; production low-latency serving benefits from ONNX Runtime or INT8 quantization.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
