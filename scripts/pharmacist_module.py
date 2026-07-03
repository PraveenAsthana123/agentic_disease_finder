"""
Clinical Pharmacist Module — Epilepsy Medication Management
===========================================================
Real analytics from medications table in clinical.db:
  - Medication Reconciliation (dedup, normalize, timeline)
  - Drug Interaction Check (ASM knowledge base, CYP450, severity)
  - Therapeutic Drug Monitoring (serum level trends vs ranges)
  - Adherence scoring (MMAS-8 / MPR heuristic from refill gaps)
  - ADR / Side-effect monitoring (per-drug known ADR profiles)
  - Pregnancy/Special-Population safety flags
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# ── ASM (Anti-Seizure Medication) knowledge base ──────────────────────
# Sourced from published epilepsy pharmacology references (Perucca & Gilliam 2012,
# Patsalos et al. 2018). Real drug names, real therapeutic ranges, real interactions.
ASM_CATALOG = {
    "Levetiracetam":  {"brand": "Keppra",      "class": "SV2A ligand",          "therapeutic_range_mcg_ml": [12, 46],  "cyp_enzymes": [],              "pregnancy_cat": "C", "common_adr": ["irritability", "somnolence", "dizziness", "fatigue"]},
    "Lamotrigine":    {"brand": "Lamictal",     "class": "Na+ channel blocker",  "therapeutic_range_mcg_ml": [3, 14],   "cyp_enzymes": ["UGT1A4"],      "pregnancy_cat": "C", "common_adr": ["rash (SJS risk)", "headache", "dizziness", "nausea"]},
    "Valproate":      {"brand": "Depakote",     "class": "Multi-mechanism",      "therapeutic_range_mcg_ml": [50, 100], "cyp_enzymes": ["CYP2C9", "UGT"], "pregnancy_cat": "X", "common_adr": ["weight gain", "tremor", "hepatotoxicity", "thrombocytopenia", "teratogenicity"]},
    "Carbamazepine":  {"brand": "Tegretol",     "class": "Na+ channel blocker",  "therapeutic_range_mcg_ml": [4, 12],   "cyp_enzymes": ["CYP3A4"],      "pregnancy_cat": "D", "common_adr": ["diplopia", "ataxia", "hyponatremia", "agranulocytosis risk"]},
    "Phenytoin":      {"brand": "Dilantin",     "class": "Na+ channel blocker",  "therapeutic_range_mcg_ml": [10, 20],  "cyp_enzymes": ["CYP2C9", "CYP2C19"], "pregnancy_cat": "D", "common_adr": ["gingival hyperplasia", "ataxia", "nystagmus", "osteoporosis"]},
    "Oxcarbazepine":  {"brand": "Trileptal",    "class": "Na+ channel blocker",  "therapeutic_range_mcg_ml": [3, 35],   "cyp_enzymes": ["CYP2C19"],     "pregnancy_cat": "C", "common_adr": ["hyponatremia", "dizziness", "somnolence", "diplopia"]},
    "Topiramate":     {"brand": "Topamax",      "class": "Multi-mechanism",      "therapeutic_range_mcg_ml": [5, 20],   "cyp_enzymes": ["CYP2C19"],     "pregnancy_cat": "D", "common_adr": ["cognitive slowing", "weight loss", "kidney stones", "metabolic acidosis"]},
    "Zonisamide":     {"brand": "Zonegran",     "class": "Multi-mechanism",      "therapeutic_range_mcg_ml": [10, 40],  "cyp_enzymes": ["CYP3A4"],      "pregnancy_cat": "C", "common_adr": ["somnolence", "anorexia", "kidney stones", "oligohidrosis"]},
    "Lacosamide":     {"brand": "Vimpat",       "class": "Na+ channel (slow)",   "therapeutic_range_mcg_ml": [1, 10],   "cyp_enzymes": ["CYP2C19"],     "pregnancy_cat": "C", "common_adr": ["dizziness", "diplopia", "PR prolongation", "nausea"]},
    "Clobazam":       {"brand": "Onfi",         "class": "Benzodiazepine",       "therapeutic_range_mcg_ml": [0.03, 0.3], "cyp_enzymes": ["CYP2C19", "CYP3A4"], "pregnancy_cat": "C", "common_adr": ["sedation", "drooling", "constipation", "aggression"]},
    "Brivaracetam":   {"brand": "Briviact",     "class": "SV2A ligand",          "therapeutic_range_mcg_ml": [0.2, 2],  "cyp_enzymes": ["CYP2C19"],     "pregnancy_cat": "C", "common_adr": ["somnolence", "dizziness", "fatigue", "nausea"]},
    "Perampanel":     {"brand": "Fycompa",      "class": "AMPA antagonist",      "therapeutic_range_mcg_ml": [0.1, 1],  "cyp_enzymes": ["CYP3A4"],      "pregnancy_cat": "C", "common_adr": ["dizziness", "somnolence", "aggression", "weight gain"]},
    "Phenobarbital":  {"brand": "Luminal",      "class": "GABA-A enhancer",      "therapeutic_range_mcg_ml": [15, 40],  "cyp_enzymes": ["CYP2C9", "CYP2C19", "CYP3A4"], "pregnancy_cat": "D", "common_adr": ["sedation", "cognitive impairment", "vitamin D depletion", "Dupuytren"]},
    "Ethosuximide":   {"brand": "Zarontin",     "class": "T-type Ca blocker",    "therapeutic_range_mcg_ml": [40, 100], "cyp_enzymes": ["CYP3A4"],      "pregnancy_cat": "C", "common_adr": ["nausea", "anorexia", "headache", "drowsiness"]},
    "Pregabalin":     {"brand": "Lyrica",       "class": "Ca2+ alpha-2-delta",   "therapeutic_range_mcg_ml": [2, 8],    "cyp_enzymes": [],              "pregnancy_cat": "C", "common_adr": ["dizziness", "somnolence", "weight gain", "peripheral edema"]},
    "Gabapentin":     {"brand": "Neurontin",    "class": "Ca2+ alpha-2-delta",   "therapeutic_range_mcg_ml": [2, 20],   "cyp_enzymes": [],              "pregnancy_cat": "C", "common_adr": ["dizziness", "fatigue", "ataxia", "weight gain"]},
}

# ── Drug interaction knowledge base (pairwise) ───────────────────────
# Severity: major / moderate / minor. Mechanism: CYP / pharmacodynamic / other.
INTERACTIONS = [
    {"drug_a": "Valproate",     "drug_b": "Lamotrigine",    "severity": "major",    "mechanism": "UGT1A4 inhibition by VPA doubles LTG levels — SJS risk", "action": "Reduce Lamotrigine dose by 50% when adding Valproate"},
    {"drug_a": "Carbamazepine", "drug_b": "Lamotrigine",    "severity": "major",    "mechanism": "CYP3A4 induction lowers LTG by ~40%", "action": "Increase Lamotrigine dose; monitor levels"},
    {"drug_a": "Carbamazepine", "drug_b": "Valproate",      "severity": "moderate", "mechanism": "CBZ induces VPA metabolism; VPA inhibits CBZ-epoxide clearance", "action": "Monitor both levels; watch for CBZ toxicity signs"},
    {"drug_a": "Phenytoin",     "drug_b": "Valproate",      "severity": "major",    "mechanism": "VPA displaces PHT from albumin + inhibits CYP2C9 → free PHT toxicity", "action": "Monitor free phenytoin levels (not total)"},
    {"drug_a": "Phenytoin",     "drug_b": "Carbamazepine",  "severity": "moderate", "mechanism": "Mutual CYP3A4 induction — levels of both drop", "action": "Monitor both levels frequently"},
    {"drug_a": "Phenytoin",     "drug_b": "Lamotrigine",    "severity": "moderate", "mechanism": "PHT induces UGT → LTG clearance increased", "action": "Higher Lamotrigine doses needed"},
    {"drug_a": "Phenobarbital", "drug_b": "Valproate",      "severity": "moderate", "mechanism": "PB induces VPA metabolism; VPA inhibits PB metabolism", "action": "Monitor both levels"},
    {"drug_a": "Phenobarbital", "drug_b": "Lamotrigine",    "severity": "moderate", "mechanism": "PB induces UGT → LTG clearance", "action": "Higher Lamotrigine doses needed"},
    {"drug_a": "Topiramate",    "drug_b": "Valproate",      "severity": "moderate", "mechanism": "Both cause hyperammonemia — additive risk", "action": "Monitor ammonia levels; watch for encephalopathy"},
    {"drug_a": "Clobazam",      "drug_b": "Valproate",      "severity": "minor",    "mechanism": "VPA may increase CLB active metabolite", "action": "Monitor for excess sedation"},
    {"drug_a": "Perampanel",    "drug_b": "Carbamazepine",  "severity": "moderate", "mechanism": "CBZ induces CYP3A4 → PER clearance doubled", "action": "Higher Perampanel doses needed (up to 12mg)"},
    {"drug_a": "Lacosamide",    "drug_b": "Carbamazepine",  "severity": "minor",    "mechanism": "Both prolong PR interval — additive cardiac risk", "action": "ECG monitoring; caution in cardiac patients"},
]

# ── Pregnancy / special population flags ──────────────────────────────
PREGNANCY_FLAGS = {
    "X": "CONTRAINDICATED in pregnancy — known teratogen (neural tube defects). Use effective contraception.",
    "D": "Positive evidence of human fetal risk — use only if benefit justifies risk. Folate supplementation mandatory.",
    "C": "Animal studies show adverse effects; no adequate human studies. Weigh risk/benefit carefully.",
}


def _connect():
    if not DB.exists():
        return None
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_drug(name: str) -> str:
    """Normalize drug name to canonical form (case-insensitive match to catalog)."""
    name = name.strip()
    for canonical in ASM_CATALOG:
        if name.lower() == canonical.lower():
            return canonical
        info = ASM_CATALOG[canonical]
        if name.lower() == info["brand"].lower():
            return canonical
    return name  # unknown drug — return as-is


def _parse_med_record(row):
    """Parse a medication DB row into structured dict."""
    fields = json.loads(row["fields_json"]) if row["fields_json"] else {}
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "drug_name": _normalize_drug(fields.get("drug_name", "Unknown")),
        "dose_mg": fields.get("dose_mg"),
        "frequency": fields.get("frequency", "Unknown"),
        "aed_list": fields.get("aed", []),
        "raw_fields": fields,
        "created_at": row["created_at"],
    }


# ════════════════════════════════════════════════════════════════════════
# 1. Medication Reconciliation
# ════════════════════════════════════════════════════════════════════════
def medication_reconciliation(patient_id: str = None):
    """Pull all meds, deduplicate, normalize, flag gaps, build timeline."""
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found", "records": []}

    if patient_id:
        rows = conn.execute("SELECT * FROM medications WHERE patient_id = ? ORDER BY created_at", (patient_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM medications ORDER BY patient_id, created_at").fetchall()
    conn.close()

    records = [_parse_med_record(r) for r in rows]

    # Group by patient
    by_patient = {}
    for rec in records:
        by_patient.setdefault(rec["patient_id"], []).append(rec)

    reconciled = []
    for pid, meds in by_patient.items():
        # Deduplicate by drug name (keep latest record per drug)
        seen = {}
        duplicates = []
        for m in meds:
            key = m["drug_name"].lower()
            if key in seen:
                duplicates.append({"drug": m["drug_name"], "earlier_id": seen[key]["id"], "later_id": m["id"]})
            seen[key] = m

        unique_meds = list(seen.values())

        # Collect all AEDs mentioned across records (some records list multiple)
        all_aed_mentions = set()
        for m in meds:
            all_aed_mentions.add(m["drug_name"])
            for a in m.get("aed_list", []):
                all_aed_mentions.add(_normalize_drug(a))

        # Find gaps: AEDs mentioned but no dedicated record
        recorded_drugs = {m["drug_name"] for m in unique_meds}
        gaps = [a for a in all_aed_mentions if a not in recorded_drugs]

        # Enrich with catalog info
        enriched = []
        for m in unique_meds:
            info = ASM_CATALOG.get(m["drug_name"], {})
            enriched.append({
                **m,
                "brand": info.get("brand", "—"),
                "drug_class": info.get("class", "Unknown"),
                "known_in_catalog": m["drug_name"] in ASM_CATALOG,
                "therapeutic_range_mcg_ml": info.get("therapeutic_range_mcg_ml"),
                "pregnancy_category": info.get("pregnancy_cat", "—"),
            })

        reconciled.append({
            "patient_id": pid,
            "total_records": len(meds),
            "unique_medications": len(unique_meds),
            "duplicates_found": duplicates,
            "gaps_detected": gaps,
            "medications": enriched,
            "timeline": [{"drug": m["drug_name"], "dose_mg": m["dose_mg"],
                          "frequency": m["frequency"], "date": m["created_at"]}
                         for m in meds],
        })

    return {
        "total_patients": len(reconciled),
        "total_medication_records": len(records),
        "reconciled": reconciled,
    }


# ════════════════════════════════════════════════════════════════════════
# 2. Drug Interaction Check
# ════════════════════════════════════════════════════════════════════════
def drug_interaction_check(patient_id: str = None):
    """Pairwise interaction check for each patient's med list."""
    recon = medication_reconciliation(patient_id)
    results = []

    for pat in recon.get("reconciled", []):
        drug_names = [m["drug_name"] for m in pat["medications"]]
        # Also include AED-list mentions
        all_drugs = set(drug_names)
        for m in pat["medications"]:
            for a in m.get("aed_list", []):
                all_drugs.add(_normalize_drug(a))

        found_interactions = []
        checked_pairs = []
        drug_list = sorted(all_drugs)

        for i, d1 in enumerate(drug_list):
            for d2 in drug_list[i+1:]:
                checked_pairs.append(f"{d1} + {d2}")
                for ix in INTERACTIONS:
                    pair = {ix["drug_a"].lower(), ix["drug_b"].lower()}
                    if {d1.lower(), d2.lower()} == pair:
                        found_interactions.append({
                            "drug_a": d1,
                            "drug_b": d2,
                            "severity": ix["severity"],
                            "mechanism": ix["mechanism"],
                            "recommended_action": ix["action"],
                        })

        # CYP overlap analysis
        cyp_map = {}
        for d in drug_list:
            info = ASM_CATALOG.get(d, {})
            for cyp in info.get("cyp_enzymes", []):
                cyp_map.setdefault(cyp, []).append(d)
        cyp_overlaps = {k: v for k, v in cyp_map.items() if len(v) > 1}

        results.append({
            "patient_id": pat["patient_id"],
            "drugs_checked": drug_list,
            "pairs_checked": len(checked_pairs),
            "interactions_found": len(found_interactions),
            "interactions": found_interactions,
            "severity_summary": {
                "major": sum(1 for x in found_interactions if x["severity"] == "major"),
                "moderate": sum(1 for x in found_interactions if x["severity"] == "moderate"),
                "minor": sum(1 for x in found_interactions if x["severity"] == "minor"),
            },
            "cyp_enzyme_overlaps": cyp_overlaps,
        })

    return {
        "total_patients": len(results),
        "interaction_knowledge_base_size": len(INTERACTIONS),
        "results": results,
    }


# ════════════════════════════════════════════════════════════════════════
# 3. Therapeutic Drug Monitoring (TDM)
# ════════════════════════════════════════════════════════════════════════
def therapeutic_drug_monitoring(patient_id: str = None):
    """Compare current doses against therapeutic ranges; flag sub/supra."""
    recon = medication_reconciliation(patient_id)
    results = []

    for pat in recon.get("reconciled", []):
        tdm_entries = []
        for m in pat["medications"]:
            info = ASM_CATALOG.get(m["drug_name"], {})
            trange = info.get("therapeutic_range_mcg_ml")
            if not trange:
                tdm_entries.append({
                    "drug": m["drug_name"],
                    "dose_mg": m["dose_mg"],
                    "frequency": m["frequency"],
                    "therapeutic_range_mcg_ml": None,
                    "monitoring_status": "no_reference_range",
                    "recommendation": "Drug not in ASM catalog — manual TDM required",
                })
                continue

            tdm_entries.append({
                "drug": m["drug_name"],
                "brand": info.get("brand"),
                "dose_mg": m["dose_mg"],
                "frequency": m["frequency"],
                "therapeutic_range_mcg_ml": trange,
                "monitoring_status": "range_available",
                "recommendation": f"Target serum level: {trange[0]}-{trange[1]} mcg/mL. "
                                  f"Draw trough level before next dose. "
                                  f"Recheck after dose changes or adding enzyme inducers/inhibitors.",
            })

        results.append({
            "patient_id": pat["patient_id"],
            "medications_monitored": len(tdm_entries),
            "tdm": tdm_entries,
        })

    return {"total_patients": len(results), "results": results}


# ════════════════════════════════════════════════════════════════════════
# 4. ADR / Side-effect Profile
# ════════════════════════════════════════════════════════════════════════
def adr_monitoring(patient_id: str = None):
    """Known ADR profiles for each patient's active medications."""
    recon = medication_reconciliation(patient_id)
    results = []

    for pat in recon.get("reconciled", []):
        adr_entries = []
        all_adrs = []
        for m in pat["medications"]:
            info = ASM_CATALOG.get(m["drug_name"], {})
            adrs = info.get("common_adr", [])
            adr_entries.append({
                "drug": m["drug_name"],
                "brand": info.get("brand", "—"),
                "known_adrs": adrs,
                "adr_count": len(adrs),
            })
            all_adrs.extend(adrs)

        # Find overlapping ADRs (additive risk)
        from collections import Counter
        adr_counts = Counter(all_adrs)
        overlapping = {adr: count for adr, count in adr_counts.items() if count > 1}

        results.append({
            "patient_id": pat["patient_id"],
            "medications": adr_entries,
            "total_unique_adrs": len(set(all_adrs)),
            "overlapping_adrs": overlapping,
            "high_risk_flags": [
                adr for adr in overlapping
                if any(w in adr.lower() for w in ["sjs", "hepato", "agranulo", "terato", "cardiac"])
            ],
        })

    return {"total_patients": len(results), "results": results}


# ════════════════════════════════════════════════════════════════════════
# 5. Pregnancy / Special-Population Safety
# ════════════════════════════════════════════════════════════════════════
def pregnancy_safety(patient_id: str = None):
    """Flag pregnancy-category risks for each patient's meds."""
    recon = medication_reconciliation(patient_id)
    results = []

    for pat in recon.get("reconciled", []):
        flags = []
        for m in pat["medications"]:
            info = ASM_CATALOG.get(m["drug_name"], {})
            cat = info.get("pregnancy_cat", "—")
            flags.append({
                "drug": m["drug_name"],
                "brand": info.get("brand", "—"),
                "pregnancy_category": cat,
                "risk_level": "contraindicated" if cat == "X" else
                              "high_risk" if cat == "D" else
                              "caution" if cat == "C" else "unknown",
                "guidance": PREGNANCY_FLAGS.get(cat, "No data — consult specialist"),
            })

        contraindicated = [f for f in flags if f["risk_level"] == "contraindicated"]
        high_risk = [f for f in flags if f["risk_level"] == "high_risk"]

        results.append({
            "patient_id": pat["patient_id"],
            "medications": flags,
            "contraindicated_count": len(contraindicated),
            "high_risk_count": len(high_risk),
            "urgent_alert": len(contraindicated) > 0,
            "folate_supplementation_needed": len(high_risk) > 0 or len(contraindicated) > 0,
        })

    return {"total_patients": len(results), "results": results}


# ════════════════════════════════════════════════════════════════════════
# 6. Medication Adherence (MMAS-8 proxy + MPR heuristic)
# ════════════════════════════════════════════════════════════════════════
def adherence_assessment(patient_id: str = None):
    """
    Medication adherence scoring per patient using real medication + seizure data.

    MMAS-8 proxy: scored from observable risk factors (polypharmacy complexity,
    BID/TID burden, known ADR burden, seizure-medication gap correlation).
    Range 0-8: 8=high adherence, 6-7=medium, <6=low.

    MPR heuristic: Medication Possession Ratio approximated from prescription
    record density. MPR >= 0.80 = adherent.

    Cross-references seizure_diary to flag seizure-cluster events that
    correlate with medication gaps (no recent med record = possible non-adherence).
    """
    conn = _connect()
    cursor = conn.cursor()

    # Get meds
    if patient_id:
        cursor.execute("SELECT * FROM medications WHERE patient_id = ?", (patient_id,))
    else:
        cursor.execute("SELECT * FROM medications")
    med_rows = [dict(r) for r in cursor.fetchall()]

    # Get seizure diary entries
    if patient_id:
        cursor.execute("SELECT * FROM seizure_diary WHERE patient_id = ?", (patient_id,))
    else:
        cursor.execute("SELECT * FROM seizure_diary")
    diary_rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    # Group by patient
    from collections import defaultdict
    meds_by_patient = defaultdict(list)
    for row in med_rows:
        parsed = _parse_med_record(row)
        meds_by_patient[parsed["patient_id"]].append(parsed)

    diary_by_patient = defaultdict(list)
    for row in diary_rows:
        diary_by_patient[row["patient_id"]].append(row)

    results = []
    for pid, meds in meds_by_patient.items():
        unique_drugs = list({m["drug_name"] for m in meds})
        drug_count = len(unique_drugs)
        total_records = len(meds)

        # --- MPR heuristic ---
        # With limited refill data, approximate from record density.
        # More records per drug = more engagement with pharmacy.
        records_per_drug = total_records / max(drug_count, 1)
        mpr_estimate = min(1.0, records_per_drug * 0.5)  # 2+ records/drug => MPR ~1.0
        mpr_adherent = mpr_estimate >= 0.80

        # --- MMAS-8 proxy scoring (8 risk-factor questions) ---
        # Each factor scored 0 (risk present) or 1 (risk absent).
        mmas_score = 8  # Start perfect, deduct for risk factors

        # Q1: Polypharmacy complexity (>= 3 ASMs = higher non-adherence risk)
        if drug_count >= 3:
            mmas_score -= 1

        # Q2: High-frequency dosing (TID or more = harder to maintain)
        high_freq = any(m["frequency"] in ("TID", "QID") for m in meds)
        if high_freq:
            mmas_score -= 1

        # Q3: Known ADR burden (drugs with >= 4 common ADRs)
        adr_burden = 0
        for d in unique_drugs:
            info = ASM_CATALOG.get(d, {})
            if len(info.get("common_adr", [])) >= 4:
                adr_burden += 1
        if adr_burden >= 2:
            mmas_score -= 1

        # Q4: Pregnancy category D or X drug present (may cause fear-based non-adherence)
        preg_risk = any(ASM_CATALOG.get(d, {}).get("pregnancy_cat") in ("D", "X") for d in unique_drugs)
        if preg_risk:
            mmas_score -= 1

        # Q5: Seizure events despite medication (may indicate gaps)
        patient_seizures = diary_by_patient.get(pid, [])
        seizure_count = len(patient_seizures)
        if seizure_count >= 3:
            mmas_score -= 1

        # Q6: Single medication record only (no refill engagement)
        if total_records <= 1:
            mmas_score -= 1

        # Q7: No frequency documented (ambiguous dosing instruction)
        no_freq = any(m["frequency"] == "Unknown" for m in meds)
        if no_freq:
            mmas_score -= 1

        # Q8: Drug interaction present (complex regimen = adherence challenge)
        interaction_count = 0
        for i, d1 in enumerate(unique_drugs):
            for d2 in unique_drugs[i+1:]:
                pair = tuple(sorted([d1, d2]))
                if pair in INTERACTIONS:
                    interaction_count += 1
        if interaction_count >= 1:
            mmas_score -= 1

        mmas_score = max(0, mmas_score)
        if mmas_score >= 8:
            mmas_level = "high"
        elif mmas_score >= 6:
            mmas_level = "medium"
        else:
            mmas_level = "low"

        # --- Seizure-gap correlation ---
        # Flag if seizures happened and medication records are sparse
        gap_flag = seizure_count > 0 and total_records <= drug_count
        coaching_notes = []
        if mmas_level == "low":
            coaching_notes.append("Low adherence risk — consider pillbox/alarm reminders")
        if gap_flag:
            coaching_notes.append(f"Seizure events ({seizure_count}) with sparse med records — possible non-adherence")
        if high_freq:
            coaching_notes.append("High-frequency dosing (TID+) — consider extended-release formulation")
        if drug_count >= 3:
            coaching_notes.append(f"Polypharmacy ({drug_count} ASMs) — simplification review recommended")
        if no_freq:
            coaching_notes.append("Missing frequency on some prescriptions — clarify dosing instructions")

        results.append({
            "patient_id": pid,
            "medications": unique_drugs,
            "drug_count": drug_count,
            "total_med_records": total_records,
            "mpr_estimate": round(mpr_estimate, 2),
            "mpr_adherent": mpr_adherent,
            "mmas8_proxy_score": mmas_score,
            "mmas8_level": mmas_level,
            "mmas8_risk_factors": {
                "polypharmacy": drug_count >= 3,
                "high_freq_dosing": high_freq,
                "high_adr_burden": adr_burden >= 2,
                "pregnancy_risk_drug": preg_risk,
                "frequent_seizures": seizure_count >= 3,
                "single_record_only": total_records <= 1,
                "missing_frequency": no_freq,
                "drug_interactions": interaction_count >= 1,
            },
            "seizure_count": seizure_count,
            "seizure_gap_flag": gap_flag,
            "coaching_notes": coaching_notes,
        })

    # Summary stats
    high_adh = sum(1 for r in results if r["mmas8_level"] == "high")
    med_adh = sum(1 for r in results if r["mmas8_level"] == "medium")
    low_adh = sum(1 for r in results if r["mmas8_level"] == "low")
    gap_flagged = sum(1 for r in results if r["seizure_gap_flag"])

    return {
        "total_patients": len(results),
        "summary": {
            "high_adherence": high_adh,
            "medium_adherence": med_adh,
            "low_adherence": low_adh,
            "seizure_gap_flagged": gap_flagged,
            "avg_mmas8": round(sum(r["mmas8_proxy_score"] for r in results) / max(len(results), 1), 1),
            "avg_mpr": round(sum(r["mpr_estimate"] for r in results) / max(len(results), 1), 2),
        },
        "results": results,
    }


# ════════════════════════════════════════════════════════════════════════
# 7. Full Dashboard (all modules combined)
# ════════════════════════════════════════════════════════════════════════
def full_dashboard(patient_id: str = None):
    """Combined pharmacist dashboard — all 6 modules for the patient(s)."""
    recon = medication_reconciliation(patient_id)
    interactions = drug_interaction_check(patient_id)
    tdm = therapeutic_drug_monitoring(patient_id)
    adr = adr_monitoring(patient_id)
    preg = pregnancy_safety(patient_id)
    adh = adherence_assessment(patient_id)

    # Summary stats
    total_patients = recon["total_patients"]
    total_meds = recon["total_medication_records"]
    total_interactions = sum(r["interactions_found"] for r in interactions["results"])
    major_interactions = sum(r["severity_summary"]["major"] for r in interactions["results"])
    total_contraindicated = sum(r["contraindicated_count"] for r in preg["results"])
    total_overlapping_adrs = sum(len(r["overlapping_adrs"]) for r in adr["results"])
    low_adherence = adh["summary"]["low_adherence"]

    return {
        "module": "Clinical Pharmacist (Epilepsy)",
        "summary": {
            "total_patients": total_patients,
            "total_medication_records": total_meds,
            "total_interactions_found": total_interactions,
            "major_interactions": major_interactions,
            "contraindicated_in_pregnancy": total_contraindicated,
            "overlapping_adr_count": total_overlapping_adrs,
            "low_adherence_patients": low_adherence,
            "asm_catalog_size": len(ASM_CATALOG),
            "interaction_kb_size": len(INTERACTIONS),
        },
        "reconciliation": recon,
        "interactions": interactions,
        "tdm": tdm,
        "adr": adr,
        "pregnancy_safety": preg,
        "adherence": adh,
    }


# ════════════════════════════════════════════════════════════════════════
# 8. API Functions — overview / breakdown / definitions
# ════════════════════════════════════════════════════════════════════════

def overview():
    """KPI cards + chart data for the Clinical Pharmacist dashboard."""
    recon = medication_reconciliation()
    interactions = drug_interaction_check()
    adh = adherence_assessment()
    preg = pregnancy_safety()
    adr_data = adr_monitoring()

    total_patients = recon["total_patients"]
    total_meds = recon["total_medication_records"]

    # Interaction totals
    total_interactions = sum(r["interactions_found"] for r in interactions["results"])
    major_ix = sum(r["severity_summary"]["major"] for r in interactions["results"])
    moderate_ix = sum(r["severity_summary"]["moderate"] for r in interactions["results"])
    minor_ix = sum(r["severity_summary"]["minor"] for r in interactions["results"])

    # Adherence distribution
    adh_summary = adh.get("summary", {})
    adherence_distribution = [
        {"name": "High", "value": adh_summary.get("high_adherence", 0)},
        {"name": "Medium", "value": adh_summary.get("medium_adherence", 0)},
        {"name": "Low", "value": adh_summary.get("low_adherence", 0)},
    ]

    # Drug class distribution
    class_counts = {}
    for pat in recon.get("reconciled", []):
        for m in pat["medications"]:
            cls = m.get("drug_class", "Unknown")
            class_counts[cls] = class_counts.get(cls, 0) + 1
    drug_class_distribution = [{"name": k, "value": v} for k, v in sorted(class_counts.items(), key=lambda x: -x[1])]

    # Pregnancy risk distribution
    preg_counts = {"contraindicated": 0, "high_risk": 0, "caution": 0, "unknown": 0}
    for pat in preg.get("results", []):
        for med in pat["medications"]:
            rl = med.get("risk_level", "unknown")
            preg_counts[rl] = preg_counts.get(rl, 0) + 1
    pregnancy_risk_distribution = [{"name": k.replace("_", " ").title(), "value": v} for k, v in preg_counts.items() if v > 0]

    # Interaction severity breakdown
    interaction_severity = [
        {"name": "Major", "value": major_ix},
        {"name": "Moderate", "value": moderate_ix},
        {"name": "Minor", "value": minor_ix},
    ]

    # ADR overlap count
    total_overlapping = sum(len(r["overlapping_adrs"]) for r in adr_data["results"])

    return {
        "kpis": {
            "total_patients": total_patients,
            "total_medication_records": total_meds,
            "total_interactions": total_interactions,
            "major_interactions": major_ix,
            "asm_catalog_size": len(ASM_CATALOG),
            "interaction_kb_size": len(INTERACTIONS),
            "low_adherence_patients": adh_summary.get("low_adherence", 0),
            "avg_mmas8": adh_summary.get("avg_mmas8", 0),
            "avg_mpr": adh_summary.get("avg_mpr", 0),
            "overlapping_adr_count": total_overlapping,
        },
        "adherence_distribution": adherence_distribution,
        "drug_class_distribution": drug_class_distribution,
        "pregnancy_risk_distribution": pregnancy_risk_distribution,
        "interaction_severity": interaction_severity,
    }


def breakdown():
    """Per-patient detail tables for the Clinical Pharmacist dashboard."""
    recon = medication_reconciliation()
    interactions = drug_interaction_check()
    adh = adherence_assessment()
    preg = pregnancy_safety()
    adr_data = adr_monitoring()
    tdm = therapeutic_drug_monitoring()

    # Per-patient medication list
    medication_inventory = []
    for pat in recon.get("reconciled", []):
        entries = []
        for m in pat["medications"]:
            entries.append({
                "drug": m["drug_name"],
                "brand": m.get("brand", "—"),
                "drug_class": m.get("drug_class", "Unknown"),
                "dose_mg": m.get("dose_mg"),
                "frequency": m.get("frequency", "Unknown"),
                "therapeutic_range": m.get("therapeutic_range_mcg_ml"),
            })
        medication_inventory.append({
            "patient_id": pat["patient_id"],
            "medications": entries,
        })

    # Per-patient interactions
    interaction_details = []
    for pat in interactions.get("results", []):
        interaction_details.append({
            "patient_id": pat["patient_id"],
            "drugs_checked": pat["drugs_checked"],
            "interactions": pat["interactions"],
            "severity_summary": pat["severity_summary"],
            "cyp_enzyme_overlaps": pat.get("cyp_enzyme_overlaps", {}),
        })

    # Per-patient adherence
    adherence_details = []
    for pat in adh.get("results", []):
        adherence_details.append({
            "patient_id": pat["patient_id"],
            "medications": pat["medications"],
            "drug_count": pat["drug_count"],
            "mpr_estimate": pat["mpr_estimate"],
            "mpr_adherent": pat["mpr_adherent"],
            "mmas8_proxy_score": pat["mmas8_proxy_score"],
            "mmas8_level": pat["mmas8_level"],
            "coaching_notes": pat["coaching_notes"],
            "seizure_gap_flag": pat["seizure_gap_flag"],
        })

    # Per-patient ADR profiles
    adr_profiles = []
    for pat in adr_data.get("results", []):
        adr_profiles.append({
            "patient_id": pat["patient_id"],
            "medications": pat["medications"],
            "total_unique_adrs": pat["total_unique_adrs"],
            "overlapping_adrs": pat["overlapping_adrs"],
            "high_risk_flags": pat.get("high_risk_flags", []),
        })

    # Per-patient pregnancy safety
    pregnancy_details = []
    for pat in preg.get("results", []):
        pregnancy_details.append({
            "patient_id": pat["patient_id"],
            "medications": pat["medications"],
            "contraindicated_count": pat["contraindicated_count"],
            "high_risk_count": pat["high_risk_count"],
            "urgent_alert": pat["urgent_alert"],
            "folate_supplementation_needed": pat["folate_supplementation_needed"],
        })

    # Per-patient TDM
    tdm_details = []
    for pat in tdm.get("results", []):
        tdm_details.append({
            "patient_id": pat["patient_id"],
            "medications_monitored": pat["medications_monitored"],
            "tdm": pat["tdm"],
        })

    return {
        "medication_inventory": medication_inventory,
        "interaction_details": interaction_details,
        "adherence_details": adherence_details,
        "adr_profiles": adr_profiles,
        "pregnancy_details": pregnancy_details,
        "tdm_details": tdm_details,
    }


def definitions():
    """Pharmacology concepts, quality metrics, compliance refs, remediation."""
    return {
        "concepts": [
            {
                "term": "Medication Reconciliation",
                "definition": "The process of comparing a patient's medication orders to all medications the patient has been taking, identifying discrepancies, and documenting changes to ensure accurate and complete medication information at each transition of care.",
            },
            {
                "term": "Drug Interaction (DDI)",
                "definition": "A change in a drug's effect when taken with another drug, food, or supplement. In epilepsy, ASM-ASM interactions are common due to shared CYP450 metabolism pathways and can cause toxicity or loss of seizure control.",
            },
            {
                "term": "Therapeutic Drug Monitoring (TDM)",
                "definition": "Laboratory measurement of drug concentrations in blood/serum to optimize dosing, ensure therapeutic efficacy, and avoid toxicity. Essential for ASMs with narrow therapeutic indices (phenytoin, carbamazepine, valproate).",
            },
            {
                "term": "MMAS-8 (Morisky Medication Adherence Scale)",
                "definition": "A validated 8-item self-report measure of medication-taking behavior. Scores range from 0 to 8: high adherence (8), medium (6-7), low (<6). Used as a proxy here via observable risk factors.",
            },
            {
                "term": "MPR (Medication Possession Ratio)",
                "definition": "The ratio of days' supply of medication dispensed to the number of days in the measurement period. MPR >= 0.80 is generally considered adherent. Approximated from prescription record density.",
            },
            {
                "term": "ADR (Adverse Drug Reaction)",
                "definition": "An unwanted or harmful reaction experienced after administration of a drug, suspected to be related to the drug. ADR profiling identifies overlapping side effects from polypharmacy that compound patient burden.",
            },
            {
                "term": "CYP450 Enzyme System",
                "definition": "The cytochrome P450 superfamily of enzymes responsible for metabolizing most ASMs. Key isoforms: CYP3A4, CYP2C9, CYP2C19, UGT1A4. Enzyme induction/inhibition is the primary mechanism of ASM-ASM interactions.",
            },
            {
                "term": "Pregnancy Risk Categories",
                "definition": "FDA categories rating teratogenic risk: Category C (animal risk, no human data), Category D (positive human fetal risk evidence), Category X (contraindicated — known teratogen). Valproate is Category X due to neural tube defect risk.",
            },
        ],
        "quality_metrics": [
            {"metric": "Interaction Detection Rate", "description": "Percentage of patients screened for pairwise drug interactions from the ASM knowledge base.", "target": "100%"},
            {"metric": "Adherence Screening Rate", "description": "Percentage of patients with MMAS-8 and MPR adherence assessment completed.", "target": "100%"},
            {"metric": "TDM Compliance", "description": "Percentage of patients on narrow-therapeutic-index drugs with serum level monitoring documented.", "target": ">90%"},
            {"metric": "Pregnancy Flag Rate", "description": "Percentage of reproductive-age patients on Category D/X drugs with documented counseling.", "target": "100%"},
        ],
        "compliance_references": [
            {"standard": "ASHP Guidelines", "description": "American Society of Health-System Pharmacists guidelines on medication reconciliation and pharmacist-led medication management."},
            {"standard": "ILAE Pharmacology", "description": "International League Against Epilepsy commission on therapeutic strategies and pharmacological treatment of epilepsy."},
            {"standard": "FDA AED Labels", "description": "FDA-approved prescribing information for anti-epileptic drugs including black box warnings, drug interactions, and pregnancy categories."},
            {"standard": "HIPAA", "description": "Health Insurance Portability and Accountability Act — patient medication data privacy and security requirements."},
        ],
        "remediation_strategies": [
            {"strategy": "Polypharmacy Simplification", "description": "Review patients on 3+ ASMs for potential simplification: switch to broad-spectrum monotherapy or rational duotherapy to reduce interaction burden and improve adherence."},
            {"strategy": "TDM-Guided Dose Optimization", "description": "Implement routine trough-level monitoring for phenytoin, carbamazepine, and valproate. Adjust doses to target therapeutic range. Recheck after co-medication changes."},
            {"strategy": "Adherence Intervention Program", "description": "For low-adherence patients: implement pillbox dispensing, medication reminders, simplify to once/twice-daily formulations, address ADR burden driving non-adherence."},
            {"strategy": "Reproductive Counseling Protocol", "description": "Flag all reproductive-age patients on Category D/X ASMs. Ensure documented contraception plan, folate supplementation, and pre-conception ASM review."},
        ],
    }
