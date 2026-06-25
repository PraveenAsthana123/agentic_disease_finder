#!/usr/bin/env python3
"""Clinical Data Manager — Terminology Mapping + Data Standardization.

Two CDM tasks over the REAL `assessments` table (259 records, 12 instruments):

1. terminology_map(): maps each instrument to a canonical internal CATEGORY +
   clinical domain (mood / cognition / language / swallowing / function / sleep
   / quality-of-life). Internal taxonomy only — does NOT assert external LOINC/
   SNOMED codes it cannot verify (§57.7 honesty).

2. standardize_levels(): normalizes the free-text `level` strings to a canonical
   ordinal severity scale (normal=0 … severe=4) and flags non-conforming /
   empty values for cleanup. Reports the standardization, never mutates rows.

100% real (reads live table) — no synthetic, no destructive action.
"""

import os
import sqlite3
from collections import Counter, defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")

# Instrument → (canonical category, clinical domain). Internal taxonomy.
INSTRUMENT_CATEGORY = {
    "PHQ9": ("depression_screen", "mood"),
    "GAD7": ("anxiety_screen", "mood"),
    "NDDIE": ("depression_epilepsy_screen", "mood"),
    "MOCA": ("global_cognition", "cognition"),
    "MMSE": ("global_cognition", "cognition"),
    "BNT": ("naming", "language"),
    "WAB": ("aphasia_battery", "language"),
    "VERBAL_FLUENCY": ("verbal_fluency", "language"),
    "MASA": ("swallowing", "swallowing"),
    "BARTHEL": ("functional_independence", "function"),
    "EPWORTH": ("daytime_sleepiness", "sleep"),
    "QOLIE31": ("quality_of_life", "quality_of_life"),
}

# Canonical ordinal severity scale (normalize free-text `level` → ordinal).
SEVERITY_ORDINAL = {
    "normal": 0, "minimal": 0, "none": 0,
    "mild": 1,
    "moderate": 2,
    "moderately severe": 3, "moderately_severe": 3,
    "severe": 4,
}


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def terminology_map(patient_id=None):
    """Map live instruments to canonical category + domain; report coverage."""
    c = _conn()
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    present = Counter(r[0] for r in c.execute(f"SELECT instrument FROM assessments {where}", params))
    c.close()

    mapped, unmapped = [], []
    domain_rollup = defaultdict(lambda: {"instruments": [], "records": 0})
    for ins, n in sorted(present.items()):
        if ins in INSTRUMENT_CATEGORY:
            cat, dom = INSTRUMENT_CATEGORY[ins]
            mapped.append({"instrument": ins, "category": cat, "domain": dom, "records": n})
            domain_rollup[dom]["instruments"].append(ins)
            domain_rollup[dom]["records"] += n
        else:
            unmapped.append({"instrument": ins, "records": n})
    return {
        "available": True,
        "n_instruments": len(present),
        "mapped": mapped,
        "unmapped": unmapped,
        "by_domain": {k: v for k, v in sorted(domain_rollup.items())},
        "coverage_pct": round(100 * len(mapped) / len(present), 1) if present else 0.0,
        "note": ("Canonical INTERNAL category/domain taxonomy. External standard codes "
                 "(LOINC/SNOMED) intentionally not asserted without registry verification (§57.7)."),
    }


def standardize_levels(patient_id=None):
    """Normalize free-text `level` to a canonical ordinal scale; flag non-conforming."""
    c = _conn()
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    rows = [dict(r) for r in c.execute(
        f"SELECT id, patient_id, instrument, level, interpretation FROM assessments {where}", params)]
    c.close()

    canonical = []
    nonconforming = []
    dist = Counter()
    for r in rows:
        raw = (r["level"] or "").strip().lower()
        if raw in SEVERITY_ORDINAL:
            ordv = SEVERITY_ORDINAL[raw]
            canonical.append({"id": r["id"], "instrument": r["instrument"],
                              "raw_level": r["level"], "canonical_level": raw, "ordinal": ordv})
            dist[raw] += 1
        else:
            nonconforming.append({"id": r["id"], "patient_id": r["patient_id"],
                                  "instrument": r["instrument"], "raw_level": r["level"],
                                  "interpretation": r["interpretation"],
                                  "issue": "empty level" if not raw else f"unrecognized level '{raw}'"})
    return {
        "available": True,
        "total_records": len(rows),
        "conformant": len(canonical),
        "nonconforming": len(nonconforming),
        "conformance_pct": round(100 * len(canonical) / len(rows), 1) if rows else 0.0,
        "canonical_scale": {"normal": 0, "mild": 1, "moderate": 2, "moderately_severe": 3, "severe": 4},
        "severity_distribution": dict(dist),
        "nonconforming_records": nonconforming,
        "note": ("Report only — assessment rows are not mutated. Non-conforming (empty/unknown) "
                 "levels are flagged for a separate, reviewed cleanup pass."),
    }


def full_report(patient_id=None):
    return {
        "role": "Clinical Data Manager — Terminology & Standardization",
        "terminology_map": terminology_map(patient_id),
        "standardization": standardize_levels(patient_id),
    }


if __name__ == "__main__":
    r = full_report()
    tm, st = r["terminology_map"], r["standardization"]
    print("Terminology mapping: coverage", tm["coverage_pct"], "% domains", list(tm["by_domain"]))
    print("Standardization:", st["conformant"], "/", st["total_records"], "conformant;",
          st["nonconforming"], "flagged")
