# Forensic Provenance Audit of Historical Gold Labels
**Audit Date**: October 11, 2026

This document audits every record in the historical 20-sample seed dataset to establish true source provenance, accession authenticity, and eliminate false `HUMAN_REVIEWED` attestations.

---

## 1. Record-by-Record SEC EDGAR Audit Table

| ID | Company | Filing Date | Accession Number | Stated Event Type | Audit Status | SEC EDGAR Direct Verification Finding |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| `seed_0000` | Apple Inc. | 2019-01-29 | `0000320193-19-000010` | `GUIDANCE_REVISION` | **PENDING_SEC_AUDIT** | Accession `0000320193-19-000010` is Apple's Form 10-Q filed 30/01/2019 (not 8-K). Initial guidance issuance mislabeled as revision. |
| `seed_0001` | Apple Inc. | 2019-04-30 | `0000320193-19-000066` | `EARNINGS_RESULT` | **PENDING_SEC_AUDIT** | Manually authored seed based on Q2 2019 earnings release. Awaiting official SEC exhibit text alignment. |
| `seed_0002` | Apple Inc. | 2019-06-27 | `0000320193-19-000085` | `MANAGEMENT_CHANGE` | **PENDING_SEC_AUDIT** | Form 8-K Item 5.02 regarding Jony Ive departure. Text matches historical event; offsets pending formal review. |
| `seed_0003` | Microsoft Corporation | 2019-07-18 | `0000789019-19-000030` | `EARNINGS_RESULT` | **PENDING_SEC_AUDIT** | Manually authored seed from Q4 FY19 earnings disclosure. |
| `seed_0004` | Microsoft Corporation | 2019-10-23 | `0000789019-19-000045` | `GUIDANCE_REVISION` | **PENDING_SEC_AUDIT** | Manually authored seed from Q1 FY20 earnings conference call disclosure. |
| `seed_0005` | Microsoft Corporation | 2019-11-20 | `0000789019-19-000052` | `OTHER` | **PENDING_SEC_AUDIT** | Form 8-K quarterly dividend declaration. Negative class baseline. |
| `seed_0006` | Apple Inc. | 2020-01-28 | `0000320193-20-000010` | `GUIDANCE_REVISION` | **PENDING_SEC_AUDIT** | Q2 2020 initial guidance forecast range. |
| `seed_0007` | Apple Inc. | 2020-02-17 | `0000320193-20-000020` | `GUIDANCE_REVISION` | **PENDING_SEC_AUDIT** | Form 8-K Item 7.01 COVID-19 guidance update withdrawing previous forecast. |
| `seed_0008` | Microsoft Corporation | 2020-03-13 | `0000789019-20-000015` | `MANAGEMENT_CHANGE` | **PENDING_SEC_AUDIT** | Form 8-K Item 5.02 Bill Gates board departure disclosure. |
| `seed_0009` | Microsoft Corporation | 2021-01-26 | `0000789019-21-000005` | `GUIDANCE_REVISION` | **PENDING_SEC_AUDIT** | Manually authored seed from Q2 FY21 earnings release. |
| `seed_0010` | Apple Inc. | 2021-04-28 | `0000320193-21-000055` | `EARNINGS_RESULT` | **PENDING_SEC_AUDIT** | Q2 2021 record revenue announcement. |
| `seed_0011` | Microsoft Corporation | 2021-06-16 | `0000789019-21-000040` | `MANAGEMENT_CHANGE` | **PENDING_SEC_AUDIT** | Form 8-K Satya Nadella Board Chair appointment. |
| `seed_0012` | Apple Inc. | 2024-02-01 | `0000320193-24-000006` | `EARNINGS_RESULT` | **PENDING_SEC_AUDIT** | Accession `0000320193-24-000006` is Apple's Form 10-Q filed 02/02/2024 (not 8-K). Form and filing date discrepancy. |
| `seed_0013` | Microsoft Corporation | 2024-04-25 | `0000789019-24-000028` | `GUIDANCE_REVISION` | **PENDING_SEC_AUDIT** | Manually authored seed from Q3 FY24 forward outlook. |
| `seed_0014` | Apple Inc. | 2024-08-26 | `0000320193-24-000080` | `MANAGEMENT_CHANGE` | **PENDING_SEC_AUDIT** | Accession `0000320193-24-000080` is an 8-K filed 01/08/2024 (Item 2.02 earnings). CFO transition occurred under a separate Item 5.02 filing. |
| `seed_0015` | Apple Inc. | 2025-01-30 | `0000320193-25-000008` | `GUIDANCE_REVISION` | **QUARANTINED** | Synthetic accession number not verified on SEC EDGAR. |
| `seed_0016` | Microsoft Corporation | 2025-04-29 | `0000789019-25-000032` | `GUIDANCE_REVISION` | **QUARANTINED** | Synthetic accession number not verified on SEC EDGAR. |
| `seed_0017` | Microsoft Corporation | 2025-07-22 | `0000789019-25-000055` | `EARNINGS_RESULT` | **QUARANTINED** | Synthetic accession number not verified on SEC EDGAR. |
| `seed_0018` | Apple Inc. | 2026-02-05 | `0000320193-26-000005` | `EARNINGS_RESULT` | **QUARANTINED** | Synthetic accession number not verified on SEC EDGAR. |
| `seed_0019` | Microsoft Corporation | 2026-04-18 | `0000789019-26-000022` | `GUIDANCE_REVISION` | **QUARANTINED** | Synthetic accession number not verified on SEC EDGAR; consensus mislabeled as revision. |

---

## 2. Methodological Clarifications
1. **Quarantine Criteria**: Records `seed_0015` to `seed_0019` (5 records) were quarantined because their SEC accession numbers are synthetic fabrications not matching authentic EDGAR filings, and analyst consensus was incorrectly labeled as guidance revisions. They are isolated in `data/quarantine/quarantined_records.jsonl`.
2. **Current Status of Retained 15 Records**: All 15 retained records are classified strictly as `MANUALLY_AUTHORED` and `PENDING_REVIEW`. They constitute an **Exploratory Prototype Benchmark** (with 3 held-out exploratory test samples in `test.jsonl`), NOT a verified human-reviewed gold benchmark or research-grade ground truth. No extra gold examples have been fabricated, and no fine-tuning is performed on this exploratory seed set.
3. **Discrepancy Resolution**: Filing forms (10-Q vs 8-K) and accession mismatches (e.g. `seed_0000`, `seed_0012`, `seed_0014`) are explicitly cataloged above. Future dataset iterations will directly anchor each seed to authentic raw HTML documents downloaded by `SECSubmissionsClient`.
