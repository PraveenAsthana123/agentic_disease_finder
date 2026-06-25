#!/usr/bin/env python3
"""Clinical Data Manager — MRI Validation.

Schema/consistency QC over the REAL `mri_findings` table (structured fields_json:
mri_available, hippocampal_sclerosis, lesion_location). Validates required fields,
allowed enum values, and conditional logic (a lesion_location is expected when
hippocampal_sclerosis = Yes). Reports per-record findings + a cohort summary.

100% real (reads live rows) — report only, scales as more MRI records arrive.
"""

import json
import os
import sqlite3
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")

YES_NO = {"yes", "no"}
YES_NO_UNK = {"yes", "no", "unknown", "n/a"}
SCHEMA = {
    "mri_available": {"required": True, "allowed": YES_NO},
    "hippocampal_sclerosis": {"required": False, "allowed": YES_NO_UNK},
    "lesion_location": {"required": False, "allowed": None},  # free text
}


def validate():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute("SELECT id, patient_id, fields_json FROM mri_findings")]
    c.close()

    records, all_issues = [], 0
    for r in rows:
        try:
            f = json.loads(r["fields_json"] or "{}")
        except (ValueError, TypeError):
            f = {}
        issues = []
        for field, rule in SCHEMA.items():
            val = f.get(field)
            if rule["required"] and (val is None or str(val).strip() == ""):
                issues.append(f"missing required field '{field}'")
            elif val is not None and rule["allowed"] and str(val).strip().lower() not in rule["allowed"]:
                issues.append(f"'{field}'='{val}' not in allowed {sorted(rule['allowed'])}")
        # conditional: sclerosis=Yes ⇒ lesion_location should be present
        if str(f.get("hippocampal_sclerosis", "")).strip().lower() == "yes" \
                and not str(f.get("lesion_location", "")).strip():
            issues.append("hippocampal_sclerosis=Yes but lesion_location is empty")
        all_issues += len(issues)
        records.append({"id": r["id"], "patient_id": r["patient_id"], "fields": f,
                        "valid": not issues, "issues": issues})

    avail = Counter(str(rec["fields"].get("mri_available", "unknown")).lower() for rec in records)
    return {
        "available": True,
        "n_records": len(records),
        "records": records,
        "summary": {
            "valid_records": sum(1 for r in records if r["valid"]),
            "records_with_issues": sum(1 for r in records if not r["valid"]),
            "total_issues": all_issues,
            "mri_available_distribution": dict(avail),
            "validation": "PASS" if all_issues == 0 and records else ("REVIEW" if records else "NO_DATA"),
        },
        "schema": {k: {"required": v["required"],
                       "allowed": sorted(v["allowed"]) if v["allowed"] else "free-text"}
                   for k, v in SCHEMA.items()},
        "note": "Schema + conditional-logic QC over real mri_findings. Report only; scales with more records.",
    }


if __name__ == "__main__":
    r = validate()
    print("MRI validation:", r["summary"])
    for rec in r["records"]:
        print(f"  {rec['patient_id']}: valid={rec['valid']} issues={rec['issues']}")
