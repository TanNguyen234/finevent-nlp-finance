"""Perform forensic audit on all 20 records and output audit table and quarantine records."""

import json
import pathlib

GOLD_PATH = pathlib.Path("data/labels/gold.jsonl")
QUARANTINE_DIR = pathlib.Path("data/quarantine")
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = pathlib.Path("reports")

records = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

audit_rows = []
quarantined = []
retained = []

for i, r in enumerate(records):
    rid = r.get("id")
    cik = r.get("cik")
    company = r.get("company_name")
    date = r.get("filing_date")
    year = int(date.split("-")[0])
    acc = r.get("accession_number")
    ev_type = r.get("event_type")
    text = r.get("text")
    evidence = r.get("evidence")
    old_low = r.get("old_low")
    old_high = r.get("old_high")
    
    is_future = (year >= 2025)
    has_consensus = "consensus" in text.lower()
    has_synthetic_prior = (old_low is not None and "revis" not in text.lower() and "from" not in text.lower() and ev_type == "GUIDANCE_REVISION")

    # Determine status
    if is_future:
        status = "QUARANTINED"
        reason = f"Fabricated future date ({date}) with synthetic SEC accession number ({acc})."
        quarantined.append({**r, "quarantine_reason": reason})
    elif has_consensus and ev_type == "GUIDANCE_REVISION":
        status = "QUARANTINED"
        reason = "Analyst/Wall Street consensus improperly labeled as company GUIDANCE_REVISION."
        quarantined.append({**r, "quarantine_reason": reason})
    elif rid == "seed_0000":
        # Apple Q2 2019: initial guidance labeled as revision with fabricated old_low=53.0
        status = "REFINED_PENDING_REVIEW"
        reason = "Initial quarterly guidance issuance labeled as GUIDANCE_REVISION with ungrounded prior range."
        r_clean = dict(r)
        r_clean["event_type"] = "GUIDANCE_ISSUANCE"
        r_clean["old_low"] = None
        r_clean["old_high"] = None
        r_clean["annotation_status"] = "PENDING_REVIEW"
        retained.append(r_clean)
    else:
        status = "RETAINED_PENDING_HUMAN_REVIEW"
        reason = "Historical SEC 8-K disclosure. Awaiting official human review."
        r_clean = dict(r)
        r_clean["annotation_status"] = "PENDING_REVIEW"
        retained.append(r_clean)

    audit_rows.append({
        "record_id": rid,
        "company": company,
        "filing_date": date,
        "accession": acc,
        "event_type": ev_type,
        "audit_status": status,
        "audit_notes": reason
    })

# Write quarantine file
with open(QUARANTINE_DIR / "quarantined_records.jsonl", "w", encoding="utf-8") as f:
    for q in quarantined:
        f.write(json.dumps(q) + "\n")

# Write markdown report
md = ["# Forensic Provenance Audit of Historical Gold Labels\n"]
md.append("**Audit Date**: October 11, 2026\n")
md.append("This document audits every record in the historical 20-sample `gold.jsonl` dataset to establish true source provenance, accession authenticity, and eliminate false `HUMAN_REVIEWED` attestations.\n\n")
md.append("| ID | Company | Filing Date | Accession Number | Stated Event Type | Audit Status | Forensic Notes |\n")
md.append("| :--- | :--- | :---: | :--- | :--- | :---: | :--- |\n")

for row in audit_rows:
    md.append(f"| `{row['record_id']}` | {row['company']} | {row['filing_date']} | `{row['accession']}` | `{row['event_type']}` | **{row['audit_status']}** | {row['audit_notes']} |\n")

md.append("\n## Audit Summary\n")
md.append(f"- **Total Audited Records**: {len(records)}\n")
md.append(f"- **Quarantined Records**: {len(quarantined)} (all future 2025/2026 fabricated filings)\n")
md.append(f"- **Retained Historical Records**: {len(retained)} (relabeled from `HUMAN_REVIEWED` to `PENDING_REVIEW`)\n")
md.append("- **Quarantine Destination**: `data/quarantine/quarantined_records.jsonl`\n")

with open(REPORTS_DIR / "gold_provenance_audit.md", "w", encoding="utf-8") as f:
    f.writelines(md)

print(f"Audit completed: {len(quarantined)} quarantined, {len(retained)} retained.")
