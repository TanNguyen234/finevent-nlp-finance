# FinEvent (P01) — NLP × Financial Event & Guidance Revision Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/UI-React_18-61DAFB.svg)](https://react.dev/)
[![Kaggle GPU T4](https://img.shields.io/badge/Kaggle-GPU_T4-20BEFF.svg)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Question**: *Can we extract evidence-anchored events from corporate financial filings (SEC EDGAR 8-K) and detect guidance revisions between reporting periods better than a rule/TF-IDF baseline?*

---

## 📌 Executive Summary

**FinEvent** is an intelligent NLP system engineered for equity analysts and research associates tracking corporate guidance revisions, earnings releases, and executive transitions.

When an analyst selects a target company (CIK) and two disclosure periods (e.g., Q1 vs Q2), FinEvent automatically determines:
1. **Which metric changed** (e.g., Revenue, Diluted EPS, Gross Margin).
2. **Old guidance range vs. New guidance range** (`old_low`/`old_high` → `new_low`/`new_high`).
3. **Exact evidence span** with character offsets and verifiable SEC EDGAR source URLs.
4. **Event classification** across 4 strictly defined classes:
   - `GUIDANCE_REVISION`
   - `EARNINGS_RESULT`
   - `MANAGEMENT_CHANGE`
   - `OTHER`

---

## 🏛️ System Architecture

```
                       ┌─────────────────────────┐
                       │  SEC EDGAR Filings API  │
                       │   (8-K Submissions)     │
                       └────────────┬────────────┘
                                    │ Ingest (Respect Fair-Access)
                                    ▼
                       ┌─────────────────────────┐
                       │   Segment & Offsets     │
                       │  (Clean HTML / Parsers) │
                       └────────────┬────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────┐                           ┌─────────────────────────┐
│     E0 / E1 Baseline    │                           │    E2 Transformer GPU   │
│   (Keyword & TF-IDF)    │                           │  (FinBERT / DeBERTa-v3) │
└────────────┬────────────┘                           └────────────┬────────────┘
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        │ Event Probabilities & Spans
                                        ▼
                       ┌─────────────────────────┐
                       │    Argument Extractor   │
                       │ (Metric, Range, Unit)   │
                       └────────────┬────────────┘
                                        │
                                        ▼
                       ┌─────────────────────────┐
                       │  Temporal Change Engine │
                       │ (Pairwise Period Diff)  │
                       └────────────┬────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────┐                           ┌─────────────────────────┐
│       FastAPI API       │                           │   Financial Terminal    │
│  (/companies/.../diff)  │                           │    (ReactJS Frontend)   │
└─────────────────────────┘                           └─────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/ -v
```

### 3. Start Backend & Frontend
```bash
# Start FastAPI backend
uvicorn app.main:app --reload --port 8000

# Start React Frontend
cd frontend
npm install
npm run dev
```

---

## 🔬 Benchmark & Methodology

| Model ID | Method | Macro-F1 | Precision@1 (Evidence) | Notes |
|---|---|---|---|---|
| **E0** | Keyword / Regex Baseline | *Measured* | *Measured* | Fast, deterministic floor |
| **E1** | TF-IDF (1,2-gram) + LogReg | *Measured* | *Measured* | Strong sparse baseline |
| **E2-A** | `microsoft/deberta-v3-small` | *Measured* | *Measured* | Fine-tuned on Kaggle GPU T4 |
| **E2-B** | `ProsusAI/finbert` | *Measured* | *Measured* | Domain-specific pretrained |

Detailed results and ablation studies are documented in [`reports/results.md`](reports/results.md).
