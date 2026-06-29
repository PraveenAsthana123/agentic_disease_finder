"""
Patient Portal — Medication Tab Module
=======================================
Patient-facing medication view from clinical.db:
  - My Medications (current drug list with details)
  - Medication Schedule (daily time-slot planner)
  - Adherence Summary (scores + seizure correlation)
  - Medication Recommendations (warnings, interactions, optimization)
  - Side Effect Profile (aggregate + overlapping risk ranking)
"""
import json
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

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

# ── Frequency-to-schedule mapping ─────────────────────────────────────
FREQ_SCHEDULE = {
    "QD":  ["morning"],
    "BID": ["morning", "evening"],
    "TID": ["morning", "noon", "evening"],
    "QID": ["morning", "noon", "evening", "bedtime"],
    "QHS": ["bedtime"],
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
        "created_at": row["created_at"],
    }


def _get_patient_meds(patient_id=None):
    """Fetch and parse medication records, grouped by patient."""
    conn = _connect()
    if not conn:
        return {}

    if patient_id:
        rows = conn.execute(
            "SELECT * FROM medications WHERE patient_id = ? ORDER BY created_at",
            (patient_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM medications ORDER BY patient_id, created_at"
        ).fetchall()
    conn.close()

    by_patient = defaultdict(list)
    for r in rows:
        parsed = _parse_med_record(r)
        by_patient[parsed["patient_id"]].append(parsed)
    return dict(by_patient)


def _unique_meds(meds):
    """Deduplicate meds per patient — keep latest record per drug."""
    seen = {}
    for m in meds:
        seen[m["drug_name"].lower()] = m
    return list(seen.values())


# ════════════════════════════════════════════════════════════════════════
# 1. My Medications
# ════════════════════════════════════════════════════════════════════════
def my_medications(patient_id=None):
    """
    List current medications with drug_name, brand, dose, frequency,
    drug_class, and common side effects. Patient-facing view.
    """
    by_patient = _get_patient_meds(patient_id)
    if not by_patient:
        return {"total_patients": 0, "patients": []}

    patients = []
    all_drugs = set()

    for pid, meds in by_patient.items():
        unique = _unique_meds(meds)
        med_list = []
        for m in unique:
            info = ASM_CATALOG.get(m["drug_name"], {})
            entry = {
                "drug_name": m["drug_name"],
                "brand": info.get("brand", "—"),
                "dose_mg": m["dose_mg"],
                "frequency": m["frequency"],
                "drug_class": info.get("class", "Unknown"),
                "common_side_effects": info.get("common_adr", []),
                "known_in_catalog": m["drug_name"] in ASM_CATALOG,
            }
            med_list.append(entry)
            all_drugs.add(m["drug_name"])

        patients.append({
            "patient_id": pid,
            "medication_count": len(med_list),
            "medications": med_list,
        })

    return {
        "total_patients": len(patients),
        "unique_drugs_overall": sorted(all_drugs),
        "patients": patients,
    }


# ════════════════════════════════════════════════════════════════════════
# 2. Medication Schedule
# ════════════════════════════════════════════════════════════════════════
def medication_schedule(patient_id=None):
    """
    Build a daily schedule (morning/noon/evening/bedtime) from dose
    frequency. BID = morning + evening, TID = morning + noon + evening,
    QD = morning, QHS = bedtime. Shows drug name + dose for each slot.
    """
    by_patient = _get_patient_meds(patient_id)
    if not by_patient:
        return {"total_patients": 0, "patients": []}

    patients = []
    for pid, meds in by_patient.items():
        unique = _unique_meds(meds)

        schedule = {"morning": [], "noon": [], "evening": [], "bedtime": []}
        unscheduled = []

        for m in unique:
            freq = m["frequency"].upper() if m["frequency"] else "Unknown"
            slots = FREQ_SCHEDULE.get(freq)
            info = ASM_CATALOG.get(m["drug_name"], {})
            entry = {
                "drug_name": m["drug_name"],
                "brand": info.get("brand", "—"),
                "dose_mg": m["dose_mg"],
            }

            if slots:
                for slot in slots:
                    schedule[slot].append(entry)
            else:
                # Unknown frequency — flag for review
                unscheduled.append({
                    **entry,
                    "frequency": m["frequency"],
                    "note": "Frequency not recognized — confirm with your provider",
                })

        patients.append({
            "patient_id": pid,
            "daily_schedule": schedule,
            "unscheduled": unscheduled,
            "total_daily_doses": sum(len(v) for v in schedule.values()),
        })

    return {"total_patients": len(patients), "patients": patients}


# ════════════════════════════════════════════════════════════════════════
# 3. Adherence Summary
# ════════════════════════════════════════════════════════════════════════
def adherence_summary(patient_id=None):
    """
    Compute adherence score per patient using medication record count vs
    expected (heuristic: multiple records = refills = better adherence).
    Cross-ref with seizure_diary: more seizures with low adherence = concern.
    Returns per-patient scores + overall stats.
    """
    conn = _connect()
    if not conn:
        return {"total_patients": 0, "summary": {}, "patients": []}

    # Fetch meds
    if patient_id:
        med_rows = conn.execute(
            "SELECT * FROM medications WHERE patient_id = ?", (patient_id,)
        ).fetchall()
    else:
        med_rows = conn.execute("SELECT * FROM medications").fetchall()

    # Fetch seizure diary
    if patient_id:
        sz_rows = conn.execute(
            "SELECT * FROM seizure_diary WHERE patient_id = ?", (patient_id,)
        ).fetchall()
    else:
        sz_rows = conn.execute("SELECT * FROM seizure_diary").fetchall()
    conn.close()

    # Group meds by patient
    meds_by_patient = defaultdict(list)
    for r in med_rows:
        parsed = _parse_med_record(r)
        meds_by_patient[parsed["patient_id"]].append(parsed)

    # Group seizures by patient
    diary_by_patient = defaultdict(list)
    for r in sz_rows:
        diary_by_patient[r["patient_id"]].append(dict(r))

    patients = []
    for pid, meds in meds_by_patient.items():
        unique_drugs = list({m["drug_name"] for m in meds})
        drug_count = len(unique_drugs)
        total_records = len(meds)

        # Adherence heuristic: records per drug (more refills = better engagement)
        records_per_drug = total_records / max(drug_count, 1)
        # Scale: 1 record/drug = 50%, 2+ = 80-100%
        adherence_pct = min(100.0, records_per_drug * 50.0)
        adherence_score = round(adherence_pct, 1)

        if adherence_score >= 80:
            adherence_level = "good"
        elif adherence_score >= 60:
            adherence_level = "moderate"
        else:
            adherence_level = "needs_attention"

        # Seizure correlation
        patient_seizures = diary_by_patient.get(pid, [])
        seizure_count = len(patient_seizures)

        concern_flag = False
        concern_notes = []
        if seizure_count > 0 and adherence_level == "needs_attention":
            concern_flag = True
            concern_notes.append(
                f"Low adherence ({adherence_score}%) with {seizure_count} seizure event(s) — "
                "possible missed doses contributing to breakthrough seizures"
            )
        if seizure_count >= 3:
            concern_notes.append(
                f"{seizure_count} seizure events on record — discuss with your care team"
            )
        if total_records <= drug_count and seizure_count > 0:
            concern_notes.append(
                "Only one medication record per drug — ensure refills are on schedule"
            )

        patients.append({
            "patient_id": pid,
            "medications": unique_drugs,
            "drug_count": drug_count,
            "total_med_records": total_records,
            "records_per_drug": round(records_per_drug, 2),
            "adherence_score_pct": adherence_score,
            "adherence_level": adherence_level,
            "seizure_count": seizure_count,
            "concern_flag": concern_flag,
            "concern_notes": concern_notes,
        })

    # Overall stats
    total = len(patients)
    good = sum(1 for p in patients if p["adherence_level"] == "good")
    moderate = sum(1 for p in patients if p["adherence_level"] == "moderate")
    needs_attn = sum(1 for p in patients if p["adherence_level"] == "needs_attention")
    flagged = sum(1 for p in patients if p["concern_flag"])

    return {
        "total_patients": total,
        "summary": {
            "good_adherence": good,
            "moderate_adherence": moderate,
            "needs_attention": needs_attn,
            "concern_flagged": flagged,
            "avg_adherence_pct": round(
                sum(p["adherence_score_pct"] for p in patients) / max(total, 1), 1
            ),
        },
        "patients": patients,
    }


# ════════════════════════════════════════════════════════════════════════
# 4. Medication Recommendations
# ════════════════════════════════════════════════════════════════════════
def medication_recommendations(patient_id=None):
    """
    Based on patient's current regimen, flag:
      - Pregnancy category warnings
      - Known side effect risks
      - Potential drug-drug interactions
      - Dose optimization suggestions
    """
    by_patient = _get_patient_meds(patient_id)
    if not by_patient:
        return {"total_patients": 0, "patients": []}

    patients = []
    for pid, meds in by_patient.items():
        unique = _unique_meds(meds)
        drug_names = [m["drug_name"] for m in unique]

        # Also collect AED mentions
        all_drugs = set(drug_names)
        for m in meds:
            for a in m.get("aed_list", []):
                all_drugs.add(_normalize_drug(a))
        drug_list = sorted(all_drugs)

        warnings = []

        # --- Pregnancy category warnings ---
        for d in drug_list:
            info = ASM_CATALOG.get(d, {})
            cat = info.get("pregnancy_cat", "—")
            if cat == "X":
                warnings.append({
                    "type": "pregnancy_contraindicated",
                    "severity": "critical",
                    "drug": d,
                    "message": f"{d} ({info.get('brand', d)}) is Category X — "
                               "CONTRAINDICATED in pregnancy. Discuss alternative "
                               "medications and effective contraception with your doctor.",
                })
            elif cat == "D":
                warnings.append({
                    "type": "pregnancy_high_risk",
                    "severity": "high",
                    "drug": d,
                    "message": f"{d} ({info.get('brand', d)}) is Category D — "
                               "evidence of fetal risk. If you are or may become pregnant, "
                               "discuss options with your neurologist. Folate supplementation "
                               "is recommended.",
                })

        # --- Drug interaction warnings ---
        for i, d1 in enumerate(drug_list):
            for d2 in drug_list[i + 1 :]:
                for ix in INTERACTIONS:
                    pair = {ix["drug_a"].lower(), ix["drug_b"].lower()}
                    if {d1.lower(), d2.lower()} == pair:
                        sev = "critical" if ix["severity"] == "major" else ix["severity"]
                        warnings.append({
                            "type": "drug_interaction",
                            "severity": sev,
                            "drugs": [d1, d2],
                            "message": f"Interaction between {d1} and {d2}: "
                                       f"{ix['mechanism']}. "
                                       f"Recommendation: {ix['action']}.",
                        })

        # --- Side effect risk flags ---
        combined_adrs = []
        for d in drug_list:
            info = ASM_CATALOG.get(d, {})
            combined_adrs.extend(info.get("common_adr", []))
        adr_counts = Counter(combined_adrs)
        overlapping = {adr: cnt for adr, cnt in adr_counts.items() if cnt > 1}
        if overlapping:
            for adr, cnt in overlapping.items():
                warnings.append({
                    "type": "overlapping_side_effect",
                    "severity": "moderate",
                    "side_effect": adr,
                    "drug_count": cnt,
                    "message": f"'{adr}' is a side effect of {cnt} of your medications — "
                               "increased risk. Report this symptom to your provider.",
                })

        # --- Dose optimization suggestions ---
        for m in unique:
            info = ASM_CATALOG.get(m["drug_name"], {})
            freq = (m["frequency"] or "").upper()
            if freq in ("TID", "QID"):
                warnings.append({
                    "type": "dose_optimization",
                    "severity": "info",
                    "drug": m["drug_name"],
                    "message": f"{m['drug_name']} is dosed {freq} (multiple times daily). "
                               "Ask your doctor if an extended-release formulation is available "
                               "to simplify your schedule.",
                })
            if not m["dose_mg"]:
                warnings.append({
                    "type": "dose_missing",
                    "severity": "info",
                    "drug": m["drug_name"],
                    "message": f"No dose recorded for {m['drug_name']}. "
                               "Confirm your dose with your pharmacy or provider.",
                })

        # Polypharmacy flag
        if len(drug_list) >= 3:
            warnings.append({
                "type": "polypharmacy",
                "severity": "moderate",
                "drug_count": len(drug_list),
                "message": f"You are taking {len(drug_list)} anti-seizure medications. "
                           "Discuss with your neurologist whether any can be safely reduced.",
            })

        patients.append({
            "patient_id": pid,
            "drugs_assessed": drug_list,
            "total_warnings": len(warnings),
            "critical_warnings": sum(1 for w in warnings if w["severity"] == "critical"),
            "warnings": warnings,
        })

    return {"total_patients": len(patients), "patients": patients}


# ════════════════════════════════════════════════════════════════════════
# 5. Side Effect Profile
# ════════════════════════════════════════════════════════════════════════
def side_effect_profile(patient_id=None):
    """
    Aggregate all side effects across patient's medications, rank by
    frequency, flag overlapping risks from multiple drugs.
    """
    by_patient = _get_patient_meds(patient_id)
    if not by_patient:
        return {"total_patients": 0, "patients": []}

    patients = []
    for pid, meds in by_patient.items():
        unique = _unique_meds(meds)

        # Collect AED mentions too
        all_drugs = set()
        for m in unique:
            all_drugs.add(m["drug_name"])
        for m in meds:
            for a in m.get("aed_list", []):
                all_drugs.add(_normalize_drug(a))

        # Build per-drug side effect list
        per_drug = []
        all_adrs = []
        adr_to_drugs = defaultdict(list)

        for d in sorted(all_drugs):
            info = ASM_CATALOG.get(d, {})
            adrs = info.get("common_adr", [])
            per_drug.append({
                "drug": d,
                "brand": info.get("brand", "—"),
                "side_effects": adrs,
                "count": len(adrs),
            })
            all_adrs.extend(adrs)
            for adr in adrs:
                adr_to_drugs[adr].append(d)

        # Rank side effects by frequency across medications
        adr_counts = Counter(all_adrs)
        ranked = [
            {
                "side_effect": adr,
                "frequency": cnt,
                "from_drugs": adr_to_drugs[adr],
                "overlapping_risk": cnt > 1,
            }
            for adr, cnt in adr_counts.most_common()
        ]

        # High-concern flags (serious ADRs that overlap)
        high_concern = [
            r for r in ranked
            if r["overlapping_risk"]
            and any(
                kw in r["side_effect"].lower()
                for kw in ["sjs", "hepato", "agranulo", "terato", "cardiac", "pr prolongation"]
            )
        ]

        patients.append({
            "patient_id": pid,
            "drugs_analyzed": sorted(all_drugs),
            "total_unique_side_effects": len(set(all_adrs)),
            "per_drug_profile": per_drug,
            "ranked_side_effects": ranked,
            "overlapping_count": sum(1 for r in ranked if r["overlapping_risk"]),
            "high_concern_flags": high_concern,
        })

    return {"total_patients": len(patients), "patients": patients}


# ════════════════════════════════════════════════════════════════════════
# 6. Full Dashboard
# ════════════════════════════════════════════════════════════════════════
def full_dashboard(patient_id=None):
    """
    Combined patient portal medication dashboard — all modules in one call.
    """
    meds = my_medications(patient_id)
    schedule = medication_schedule(patient_id)
    adherence = adherence_summary(patient_id)
    recommendations = medication_recommendations(patient_id)
    side_effects = side_effect_profile(patient_id)

    # Summary stats
    all_drugs = set()
    drug_counter = Counter()
    total_prescriptions = 0
    polypharmacy_count = 0

    by_patient = _get_patient_meds(patient_id)
    for pid, med_list in by_patient.items():
        unique = _unique_meds(med_list)
        total_prescriptions += len(med_list)
        for m in unique:
            all_drugs.add(m["drug_name"])
            drug_counter[m["drug_name"]] += 1
        if len(unique) >= 3:
            polypharmacy_count += 1

    most_common_drug = drug_counter.most_common(1)[0][0] if drug_counter else "—"

    return {
        "module": "Patient Portal — Medication Tab",
        "summary": {
            "total_patients_on_meds": len(by_patient),
            "total_prescriptions": total_prescriptions,
            "unique_drugs": len(all_drugs),
            "most_common_drug": most_common_drug,
            "polypharmacy_count": polypharmacy_count,
        },
        "my_medications": meds,
        "medication_schedule": schedule,
        "adherence_summary": adherence,
        "medication_recommendations": recommendations,
        "side_effect_profile": side_effects,
    }
