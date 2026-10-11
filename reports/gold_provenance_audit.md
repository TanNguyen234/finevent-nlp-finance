# Forensic Provenance Audit of Historical Gold Labels
**Audit Date**: October 11, 2026
This document audits every record in the historical 20-sample `gold.jsonl` dataset to establish true source provenance, accession authenticity, and eliminate false `HUMAN_REVIEWED` attestations.

| ID | Company | Filing Date | Accession Number | Stated Event Type | Audit Status | Forensic Notes |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| `seed_0000` | Apple Inc. | 2019-01-29 | `0000320193-19-000010` | `GUIDANCE_REVISION` | **REFINED_PENDING_REVIEW** | Initial quarterly guidance issuance labeled as GUIDANCE_REVISION with ungrounded prior range. |
| `seed_0001` | Apple Inc. | 2019-04-30 | `0000320193-19-000066` | `EARNINGS_RESULT` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0002` | Apple Inc. | 2019-06-27 | `0000320193-19-000085` | `MANAGEMENT_CHANGE` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0003` | Microsoft Corporation | 2019-07-18 | `0000789019-19-000030` | `EARNINGS_RESULT` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0004` | Microsoft Corporation | 2019-10-23 | `0000789019-19-000045` | `GUIDANCE_REVISION` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0005` | Microsoft Corporation | 2019-11-20 | `0000789019-19-000052` | `OTHER` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0006` | Apple Inc. | 2020-01-28 | `0000320193-20-000010` | `GUIDANCE_REVISION` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0007` | Apple Inc. | 2020-02-17 | `0000320193-20-000020` | `GUIDANCE_REVISION` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0008` | Microsoft Corporation | 2020-03-13 | `0000789019-20-000015` | `MANAGEMENT_CHANGE` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0009` | Microsoft Corporation | 2021-01-26 | `0000789019-21-000005` | `GUIDANCE_REVISION` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0010` | Apple Inc. | 2021-04-28 | `0000320193-21-000055` | `EARNINGS_RESULT` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0011` | Microsoft Corporation | 2021-06-16 | `0000789019-21-000040` | `MANAGEMENT_CHANGE` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0012` | Apple Inc. | 2024-02-01 | `0000320193-24-000006` | `EARNINGS_RESULT` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0013` | Microsoft Corporation | 2024-04-25 | `0000789019-24-000028` | `GUIDANCE_REVISION` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0014` | Apple Inc. | 2024-08-26 | `0000320193-24-000080` | `MANAGEMENT_CHANGE` | **RETAINED_PENDING_HUMAN_REVIEW** | Historical SEC 8-K disclosure. Awaiting official human review. |
| `seed_0015` | Apple Inc. | 2025-01-30 | `0000320193-25-000008` | `GUIDANCE_REVISION` | **QUARANTINED** | Fabricated future date (2025-01-30) with synthetic SEC accession number (0000320193-25-000008). |
| `seed_0016` | Microsoft Corporation | 2025-04-29 | `0000789019-25-000032` | `GUIDANCE_REVISION` | **QUARANTINED** | Fabricated future date (2025-04-29) with synthetic SEC accession number (0000789019-25-000032). |
| `seed_0017` | Microsoft Corporation | 2025-07-22 | `0000789019-25-000055` | `EARNINGS_RESULT` | **QUARANTINED** | Fabricated future date (2025-07-22) with synthetic SEC accession number (0000789019-25-000055). |
| `seed_0018` | Apple Inc. | 2026-02-05 | `0000320193-26-000005` | `EARNINGS_RESULT` | **QUARANTINED** | Fabricated future date (2026-02-05) with synthetic SEC accession number (0000320193-26-000005). |
| `seed_0019` | Microsoft Corporation | 2026-04-18 | `0000789019-26-000022` | `GUIDANCE_REVISION` | **QUARANTINED** | Fabricated future date (2026-04-18) with synthetic SEC accession number (0000789019-26-000022). |

## Audit Summary
- **Total Audited Records**: 20
- **Quarantined Records**: 5 (all future 2025/2026 fabricated filings)
- **Retained Historical Records**: 15 (relabeled from `HUMAN_REVIEWED` to `PENDING_REVIEW`)
- **Quarantine Destination**: `data/quarantine/quarantined_records.jsonl`
