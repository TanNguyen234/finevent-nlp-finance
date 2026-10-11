# FinEvent Research Evaluation and Benchmark Report

## 1. Executive Summary
This document provides the authoritative, evidence-grounded performance benchmarks for the FinEvent financial event extraction framework. Following an independent forensic audit, benchmarks are strictly partitioned between the exploratory seed benchmark and the historical synthetic sanity run, eliminating invalid cross-dataset comparisons and reflecting true dataset provenance.

---

## 2. Benchmark Table 1: Exploratory Prototype Benchmark (Pending SEC Ground-Truth Verification)

This benchmark evaluates models on the 15-sample exploratory historical seed dataset (`data/labels/test.jsonl`), comprising 3 held-out 2024 records from Apple Inc. and Microsoft Corporation. 

- **Dataset Origin**: `MANUALLY_AUTHORED_SEEDS` (Exploratory Prototype, `PENDING_REVIEW`)
- **Dataset SHA-256**: `1d7d535c1b...` (`data/labels/split_manifest.json`)
- **Sample Distribution**: Train = 9 (2019–2020), Validation = 3 (2021), Test = 3 (2024 held-out)
- **Represented Test Classes**: `GUIDANCE_REVISION` (1), `EARNINGS_RESULT` (1), `MANAGEMENT_CHANGE` (1); `OTHER` (0 support in held-out test).
- **Audit Provenance Note**: Discrepancies exist between seed metadata and SEC filings (e.g. `seed_0012` accession belongs to a Form 10-Q rather than 8-K; `seed_0014` CFO transition accession maps to an Item 2.02 earnings filing). These metrics are presented strictly as a code pipeline sanity check, not as production-grade generalization proof.

| Experiment ID | Model Architecture | Hardware | Split Strategy | Test Macro-F1 | Test Weighted-F1 | Accuracy | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **E0_REAL_SEC** | Keyword Rule Baseline | Local CPU | Chronological (2024) | 0.5000 | 0.6667 | 0.6667 | Verified Local Pipeline |
| **E1_REAL_SEC** | TF-IDF (1-2 gram) + Logistic Regression | Local CPU | Chronological (2024) | 0.7500 | 1.0000 | 1.0000 | Verified Local Pipeline |

### Per-Class Test Performance (Exploratory Test Set)

| Class Name | Test Support | E0 Rule F1 | E1 TF-IDF F1 | Evaluation Note |
| :--- | :---: | :---: | :---: | :--- |
| **EARNINGS_RESULT** | 1 | 1.0000 | 1.0000 | Accurately identifies historical reported metrics and EPS announcements. |
| **GUIDANCE_REVISION** | 1 | 0.0000 | 1.0000 | TF-IDF correctly captures forward-looking revision phrasing where rules miss ungrounded cues. |
| **MANAGEMENT_CHANGE** | 1 | 1.0000 | 1.0000 | Accurately captures executive transitions and board appointments. |
| **OTHER** | 0 | 0.0000 | 0.0000 | Unrepresented in the held-out 2024 test split. |

*Limitation Statement*: A 3-sample held-out test split is an exploratory sanity check, not a statistically comprehensive benchmark. Expanding human-verified gold annotations to >=200 filings directly anchored to downloaded raw SEC HTML documents is required for publication-grade claims.

---

## 3. Benchmark Table 2: Historical Kaggle Cloud Synthetic Sanity Run

This table records the historical execution of `kaggle/train_finevent.py` on Kaggle Cloud GPU (`foxy0505/finevent-nlp-training-gpu`). The forensic audit established that this kernel run loaded the legacy 290-sample dataset from raw GitHub commit `662aabe`, which contained synthetic template-generated samples.

- **Dataset Origin**: `HISTORICAL_SYNTHETIC` (Legacy 290-sample split fetched via raw GitHub URL)
- **Sample Distribution**: Train = 183, Validation = 51, Test = 56
- **Hardware**: Kaggle Cloud Tesla T4 GPU (CUDA 12.x, PyTorch 2.6.0, Transformers 4.49.0)

| Experiment ID | Model Architecture | Hardware | Split Strategy | Pre-Val F1 | Post-Val F1 | Test Macro-F1 | Test Accuracy | Status & Audit Finding |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E2_A_DEBERTA** | `microsoft/deberta-v3-small` | Tesla T4 GPU | Legacy Synthetic Split | 0.0952 | 0.3588 | 0.3677 | 0.5179 | Historical Synthetic Sanity |
| **E2_B_FINBERT** | `ProsusAI/finbert` | Tesla T4 GPU | Legacy Synthetic Split | 0.3221 | 1.0000 | 1.0000 | 1.0000 | Historical Synthetic Sanity (Memorization on Synthetic) |

*Audit Finding*: FinBERT's 1.0000 Macro-F1 on the legacy test set reflects template-memorization on synthetic sentence variations, not generalized financial reasoning on noisy SEC filings.

---

## 4. Financial Guidance Identity Ablation Benchmark

Dynamic execution of ablated functions on candidate revision pairs (`reports/ablation_results.json`):

| Configuration | Test Pairs | Naive False Positives | Strict Identity Rejections | Precision Delta | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **A1: Remove Guidance Identity** | 4 boundary test cases | 3 (100% false alarms) | 0 | 0.0% | Ablated Execution (Accuracy 25%) |
| **A2: Disable Unit Normalization** | 4 boundary test cases | 0 | 1 (Unit failure) | Defective deltas | Ablated Execution (Accuracy 75%) |
| **A0: Full Pipeline (Strict Identity)** | 4 boundary test cases | 0 (0% false alarms) | 3 (100% rejected) | +100.0% precision | Full Production Pipeline (Accuracy 100%) |

**Takeaway**: Enforcing the 7-tuple Guidance Identity `(company, metric, target_fiscal_period, accounting_basis, currency, unit, scope)` completely eliminates false revision inferences between historical earnings results and forward guidance, GAAP vs Non-GAAP disclosures, and mismatched fiscal quarters.
