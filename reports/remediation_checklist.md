# FinEvent Remediation Checklist — Verified Findings & Implementation Plan

| Issue ID | Severity | Description | Affected Files | Implementation Fix | Verification Gate |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **P0-01** | P0 | Kaggle benchmark trained on legacy 290 synthetic samples from commit 662aabe raw URL | `kaggle/train_finevent.py`, `reports/results.md` | Audit Kaggle run, isolate historical metrics to legacy report, create immutable dataset snapshot | `reports/kaggle_forensic_audit.md` |
| **P0-02** | P0 | Incorrect gold-label provenance (future 2025/2026 dates, synthetic accessions, consensus labeled as revision, false HUMAN_REVIEWED) | `data/labels/gold.jsonl`, `src/ingest/build_dataset.py` | Audit all 20 records, quarantine non-grounded records to `data/quarantine/`, relabel to `MANUALLY_AUTHORED` / `PENDING_REVIEW` | Audit table in `reports/gold_provenance_audit.md` |
| **P0-03** | P0 | Hardcoded ablation values in `src/eval/run_ablation.py` (`1/total`, `2/total`) | `src/eval/run_ablation.py` | Implement actual execution of ablated functions (`ablate_no_identity`, `ablate_no_units`) computing true predictions | Run `python -m src.eval.run_ablation` |
| **P0-04** | P0 | Invalid cross-dataset model comparison (E0/E1 on 5 samples vs E2 on 56 samples) | `README.md`, `reports/results.md` | Separate comparison into distinct benchmark tables with clear dataset fingerprints | Table verification against prediction files |
| **P0-05** | P0 | Dataset resolution is mutable (`raw.githubusercontent.com/.../main/...`) | `kaggle/train_finevent.py` | Embed immutable SHA-256 verified dataset snapshot | Content-address verification in script |
| **P1-01** | P1 | Exhibit discovery assumes static `ex99-1.htm` filename | `src/ingest/sec_submissions.py` | Discover exhibits dynamically from SEC filing document index metadata | Test dynamic exhibit detection |
| **P1-02** | P1 | FastAPI only loads E1 TF-IDF, never serves E2 models | `app/service.py`, `app/main.py` | Add dynamic model loader supporting E1/E2 with fallback and actual `/health` metadata | Test `/health` and `/predict/event` |
| **P1-03** | P1 | Metrics reporting inconsistency (accuracy vs confusion matrix support) | `src/eval/metrics.py` | Automate all benchmark metrics from confusion matrix and prediction lists | `pytest tests/test_evaluation_metrics.py` |
| **P1-04** | P1 | Incomplete guidance identity matching | `src/extract/change.py` | Enforce full 7-tuple check with explicit validation | `pytest tests/test_financial_reasoning.py` |
| **P1-05** | P1 | Misleading integration fixture | `tests/test_integration_sec.py` | Clearly mark fixture as synthetic integration fixture and test authentic parser | `pytest tests/test_integration_sec.py` |
| **P1-06** | P1 | Residual hardcoded confidence (0.95, 0.70) | `src/extract/change.py` | Calculate dynamic or raw score confidence | Test dynamic confidence output |
