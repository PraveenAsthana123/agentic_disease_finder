"""Multidisciplinary Epilepsy Board Review Dashboard — board meeting analytics from clinical.db.

Aggregates data for epilepsy board meetings where neurologists, surgeons,
neuropsychologists, and other experts review patient cases. Provides
case summaries, concordance analysis, surgical candidacy assessment,
medication profiles, and neuropsychological screening results.

Sources:
- clinical_decisions (1 decision — AI prediction, confidence, agreement)
- expert_reviews (3 reviews — role-based findings and AI concordance)
- seizure_diary (25 entries — event severity, triggers, duration)
- mri_findings (40 scans — lesion type, location, laterality)
- medications (9 records — drug name, dose, frequency)
- patient_demographics (30 patients — age, sex, epilepsy type, onset)
- neuropsych (37 assessments — PHQ-9, GAD-7, MoCA, cognitive indices)
- eeg_acquisition (30 recordings — type, duration, sampling rate)
- analyses (AI predictions — disease, confidence)
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, default=0):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


# ------------------------------------------------------------------
#  /api/epilepsy-board/overview
# ------------------------------------------------------------------

def epilepsy_board_overview():
    """Aggregate epilepsy board KPIs: reviewed patients, expert agreement,
    MRI lesion rates, seizure severity, surgical candidacy, chart data."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # -- 1. Core KPIs -----------------------------------------------
    # Total patients reviewed (union of clinical_decisions + expert_reviews)
    reviewed_cd = set()
    for row in _safe_rows(cur, "SELECT DISTINCT patient_id FROM clinical_decisions"):
        reviewed_cd.add(row[0])
    reviewed_er = set()
    for row in _safe_rows(cur, "SELECT DISTINCT patient_id FROM expert_reviews"):
        reviewed_er.add(row[0])
    reviewed_patients = reviewed_cd | reviewed_er
    total_patients_reviewed = len(reviewed_patients)

    total_expert_reviews = _safe(cur, "SELECT count(*) FROM expert_reviews")
    total_clinical_decisions = _safe(cur, "SELECT count(*) FROM clinical_decisions")

    # AI agreement rate (% of expert_reviews where agree_with_ai = 'agree')
    agree_count = _safe(
        cur, "SELECT count(*) FROM expert_reviews WHERE lower(agree_with_ai) = 'agree'")
    ai_agreement_rate = round(
        agree_count / max(total_expert_reviews, 1) * 100, 1)

    # -- 2. MRI lesion positive rate ---------------------------------
    mri_rows = _safe_rows(cur, "SELECT patient_id, fields_json FROM mri_findings")
    mri_total = 0
    mri_lesion_positive = 0
    mri_lesion_types = {}
    mri_temporal_patients = set()
    for pid, fj in mri_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        avail = d.get("mri_available", "")
        if str(avail).lower() != "yes":
            continue
        mri_total += 1
        lt = d.get("lesion_type", "None")
        # Non-lesional codes: NL (No Lesion), NRM (Normal), None, Normal
        non_lesional = ("Normal", "None", "none", "normal", "NL", "NRM")
        if lt and lt not in non_lesional:
            mri_lesion_positive += 1
        lt_key = lt if lt else "None"
        mri_lesion_types[lt_key] = mri_lesion_types.get(lt_key, 0) + 1
        # Track temporal lesion patients for surgical candidacy
        loc = str(d.get("lesion_location", "")).lower()
        if "temporal" in loc and lt not in non_lesional:
            mri_temporal_patients.add(pid)

    mri_lesion_positive_rate = round(
        mri_lesion_positive / max(mri_total, 1) * 100, 1)

    # -- 3. Seizure severity breakdown --------------------------------
    sz_rows = _safe_rows(
        cur, "SELECT patient_id, severity, event_date FROM seizure_diary")
    severity_counts = {"Mild": 0, "Moderate": 0, "Severe": 0}
    seizure_by_patient = {}
    seizure_by_month = {}
    for pid, sev, edate in sz_rows:
        if sev in severity_counts:
            severity_counts[sev] += 1
        seizure_by_patient[pid] = seizure_by_patient.get(pid, 0) + 1
        # Monthly trend
        if edate:
            month_key = str(edate)[:7]  # YYYY-MM
            seizure_by_month[month_key] = seizure_by_month.get(month_key, 0) + 1

    # -- 4. Surgical candidate count ----------------------------------
    # Patients with MRI temporal lesion AND seizure diary entries > 2
    surgical_candidates = set()
    for pid in mri_temporal_patients:
        if seizure_by_patient.get(pid, 0) > 2:
            surgical_candidates.add(pid)
    surgical_candidate_count = len(surgical_candidates)

    # -- 5. Charts: review_by_role ------------------------------------
    role_rows = _safe_rows(
        cur, "SELECT role, count(*) FROM expert_reviews GROUP BY role ORDER BY count(*) DESC")
    review_by_role = [{"role": r, "count": c} for r, c in role_rows]

    # -- 6. Charts: epilepsy_type_distribution -------------------------
    etype_rows = _safe_rows(
        cur, "SELECT epilepsy_type, count(*) FROM patient_demographics GROUP BY epilepsy_type ORDER BY count(*) DESC")
    epilepsy_type_distribution = [{"type": t, "count": c} for t, c in etype_rows]

    # -- 7. Charts: lesion_type_distribution ---------------------------
    lesion_type_distribution = [
        {"type": k, "count": v}
        for k, v in sorted(mri_lesion_types.items(), key=lambda x: -x[1])]

    # -- 8. Charts: seizure_frequency_trend ----------------------------
    seizure_frequency_trend = [
        {"month": m, "count": c}
        for m, c in sorted(seizure_by_month.items())]

    # -- 9. Charts: medication_distribution ----------------------------
    med_rows = _safe_rows(cur, "SELECT fields_json FROM medications")
    drug_counts = {}
    for (fj,) in med_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        drug = d.get("drug_name", "unknown")
        drug_counts[drug] = drug_counts.get(drug, 0) + 1
    medication_distribution = [
        {"drug": k, "count": v}
        for k, v in sorted(drug_counts.items(), key=lambda x: -x[1])]

    # -- 10. Charts: neuropsych_profile (radar-style) ------------------
    np_rows = _safe_rows(cur, "SELECT fields_json FROM neuropsych")
    np_moca, np_phq9, np_gad7 = [], [], []
    np_memory, np_attention = [], []
    for (fj,) in np_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        v = d.get("moca")
        if v is not None:
            np_moca.append(v)
        v = d.get("phq9")
        if v is not None:
            np_phq9.append(v)
        v = d.get("gad7")
        if v is not None:
            np_gad7.append(v)
        v = d.get("memory_index")
        if v is not None:
            np_memory.append(v)
        v = d.get("attention_index")
        if v is not None:
            np_attention.append(v)

    def _avg(lst):
        return round(sum(lst) / max(len(lst), 1), 1) if lst else None

    neuropsych_profile = {
        "avg_moca": _avg(np_moca),
        "avg_phq9": _avg(np_phq9),
        "avg_gad7": _avg(np_gad7),
        "avg_memory_index": _avg(np_memory),
        "avg_attention_index": _avg(np_attention),
        "sample_sizes": {
            "moca": len(np_moca),
            "phq9": len(np_phq9),
            "gad7": len(np_gad7),
            "memory_index": len(np_memory),
            "attention_index": len(np_attention),
        },
    }

    conn.close()

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_patients_reviewed": total_patients_reviewed,
            "total_expert_reviews": total_expert_reviews,
            "total_clinical_decisions": total_clinical_decisions,
            "ai_agreement_rate_pct": ai_agreement_rate,
            "mri_lesion_positive_rate_pct": mri_lesion_positive_rate,
            "seizure_severity_breakdown": severity_counts,
            "surgical_candidate_count": surgical_candidate_count,
        },
        "charts": {
            "review_by_role": review_by_role,
            "lesion_type_distribution": lesion_type_distribution,
            "epilepsy_type_distribution": epilepsy_type_distribution,
            "seizure_frequency_trend": seizure_frequency_trend,
            "medication_distribution": medication_distribution,
            "neuropsych_profile": neuropsych_profile,
        },
    }


# ------------------------------------------------------------------
#  /api/epilepsy-board/breakdown
# ------------------------------------------------------------------

def epilepsy_board_breakdown():
    """Per-patient case summaries for board review: demographics, MRI,
    seizure counts, medications, neuropsych, expert reviews, decisions,
    concordance analysis, and medication summary."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # -- 1. Load patient demographics --------------------------------
    demo_rows = _safe_rows(
        cur,
        "SELECT patient_id, full_name, age, sex, epilepsy_type, "
        "epilepsy_onset_age, years_with_epilepsy, primary_neurologist "
        "FROM patient_demographics ORDER BY patient_id")
    patients = {}
    for pid, name, age, sex, etype, onset, years, neuro in demo_rows:
        patients[pid] = {
            "patient_id": pid,
            "full_name": name,
            "age": age,
            "sex": sex,
            "epilepsy_type": etype,
            "epilepsy_onset_age": onset,
            "years_with_epilepsy": years,
            "primary_neurologist": neuro,
            "mri_lesion": None,
            "mri_location": None,
            "seizure_count": 0,
            "medication_count": 0,
            "medications": [],
            "latest_neuropsych": {},
            "expert_reviews": [],
            "clinical_decision": None,
            "board_recommendation": "Continued monitoring",
        }

    # -- 2. MRI findings per patient ---------------------------------
    mri_rows = _safe_rows(cur, "SELECT patient_id, fields_json FROM mri_findings")
    mri_by_patient = {}
    for pid, fj in mri_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        avail = d.get("mri_available", "")
        if str(avail).lower() != "yes":
            continue
        lt = d.get("lesion_type", "None")
        loc = d.get("lesion_location", "")
        mri_by_patient[pid] = {"lesion_type": lt, "location": loc}

    for pid, mri in mri_by_patient.items():
        if pid in patients:
            patients[pid]["mri_lesion"] = mri["lesion_type"]
            patients[pid]["mri_location"] = mri["location"]

    # -- 3. Seizure counts per patient --------------------------------
    sz_count_rows = _safe_rows(
        cur, "SELECT patient_id, count(*) FROM seizure_diary GROUP BY patient_id")
    for pid, cnt in sz_count_rows:
        if pid in patients:
            patients[pid]["seizure_count"] = cnt

    # -- 4. Medications per patient -----------------------------------
    med_rows = _safe_rows(cur, "SELECT patient_id, fields_json FROM medications")
    med_by_patient = {}
    all_drug_combos = {}
    for pid, fj in med_rows:
        try:
            d = json.loads(fj)
        except Exception:
            continue
        drug = d.get("drug_name", "unknown")
        if pid not in med_by_patient:
            med_by_patient[pid] = []
        med_by_patient[pid].append(drug)

    drug_total_counts = {}
    for pid, drugs in med_by_patient.items():
        if pid in patients:
            patients[pid]["medication_count"] = len(drugs)
            patients[pid]["medications"] = drugs
        for drug in drugs:
            drug_total_counts[drug] = drug_total_counts.get(drug, 0) + 1
        # Track combinations (sorted tuple for uniqueness)
        combo = tuple(sorted(set(drugs)))
        if len(combo) > 1:
            all_drug_combos[combo] = all_drug_combos.get(combo, 0) + 1

    # -- 5. Latest neuropsych per patient -----------------------------
    np_rows = _safe_rows(
        cur, "SELECT patient_id, fields_json, created_at FROM neuropsych ORDER BY created_at DESC")
    np_seen = set()
    for pid, fj, ts in np_rows:
        if pid in np_seen:
            continue
        np_seen.add(pid)
        try:
            d = json.loads(fj)
        except Exception:
            continue
        if pid in patients:
            patients[pid]["latest_neuropsych"] = {
                "moca": d.get("moca"),
                "phq9": d.get("phq9"),
                "gad7": d.get("gad7"),
                "memory_index": d.get("memory_index"),
                "attention_index": d.get("attention_index"),
                "executive_index": d.get("executive_index"),
                "assessed_at": ts,
            }

    # -- 6. Expert reviews per patient --------------------------------
    er_rows = _safe_rows(
        cur,
        "SELECT patient_id, role, expert, finding, agree_with_ai "
        "FROM expert_reviews ORDER BY patient_id")
    for pid, role, expert, finding, agree in er_rows:
        if pid in patients:
            patients[pid]["expert_reviews"].append({
                "role": role,
                "expert": expert,
                "finding": finding,
                "agree_with_ai": agree,
            })

    # -- 7. Clinical decisions per patient ----------------------------
    cd_rows = _safe_rows(
        cur,
        "SELECT patient_id, ai_prediction, ai_confidence, "
        "neurologist_agreement, final_decision, reviewer, note "
        "FROM clinical_decisions ORDER BY patient_id")
    for pid, pred, conf, agree, decision, reviewer, note in cd_rows:
        if pid in patients:
            patients[pid]["clinical_decision"] = {
                "ai_prediction": pred,
                "ai_confidence": conf,
                "neurologist_agreement": agree,
                "final_decision": decision,
                "reviewer": reviewer,
                "note": note,
            }

    # -- 8. Board recommendation logic --------------------------------
    for pid, p in patients.items():
        lesion = p.get("mri_lesion")
        location = str(p.get("mri_location", "")).lower()
        sz_count = p.get("seizure_count", 0)
        non_lesional = (None, "Normal", "None", "none", "normal", "NL", "NRM")
        has_temporal_lesion = (
            lesion not in non_lesional
            and "temporal" in location)
        has_high_seizures = sz_count > 2
        has_lesion = lesion not in non_lesional

        if has_temporal_lesion and has_high_seizures:
            p["board_recommendation"] = "Surgical evaluation"
        elif has_high_seizures and not has_lesion:
            p["board_recommendation"] = "Medication adjustment"
        else:
            p["board_recommendation"] = "Continued monitoring"

    # -- 9. Review concordance (agreement matrix) ---------------------
    concordance_rows = _safe_rows(
        cur,
        "SELECT role, agree_with_ai, count(*) FROM expert_reviews "
        "GROUP BY role, agree_with_ai")
    concordance = {}
    for role, agree, cnt in concordance_rows:
        if role not in concordance:
            concordance[role] = {}
        concordance[role][agree if agree else "unknown"] = cnt

    review_concordance = [
        {"role": role, "responses": responses}
        for role, responses in sorted(concordance.items())]

    # -- 10. Medication summary ---------------------------------------
    medication_summary = {
        "per_drug_counts": [
            {"drug": k, "count": v}
            for k, v in sorted(drug_total_counts.items(), key=lambda x: -x[1])],
        "common_combinations": [
            {"drugs": list(combo), "count": cnt}
            for combo, cnt in sorted(all_drug_combos.items(),
                                     key=lambda x: -x[1])[:10]],
    }

    # Build patient case summaries list
    patient_case_summaries = [
        patients[pid] for pid in sorted(patients.keys())]

    conn.close()

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_case_summaries": patient_case_summaries,
        "review_concordance": review_concordance,
        "medication_summary": medication_summary,
    }


# ------------------------------------------------------------------
#  /api/epilepsy-board/definitions
# ------------------------------------------------------------------

def epilepsy_board_definitions():
    """Metric definitions for the Epilepsy Board Review Dashboard."""
    return {
        "available": True,
        "definitions": [
            {"metric": "Total Patients Reviewed",
             "definition": "Distinct patients appearing in either clinical_decisions or expert_reviews tables — represents cases formally reviewed by the epilepsy board"},
            {"metric": "Total Expert Reviews",
             "definition": "Count of individual expert review records — each represents one specialist's assessment of a patient case (neurologist, EEG technician, neuropsychologist, etc.)"},
            {"metric": "Total Clinical Decisions",
             "definition": "Count of finalized clinical decision records — each captures the AI prediction, neurologist agreement, and final board decision for a case"},
            {"metric": "AI Agreement Rate %",
             "definition": "Percentage of expert reviews where the reviewer agreed with the AI prediction (agree_with_ai = 'agree') — measures concordance between AI and multidisciplinary team"},
            {"metric": "MRI Lesion Positive Rate %",
             "definition": "Percentage of available MRI scans showing a structural lesion (lesion_type not Normal/None) — identifies patients with radiologically confirmed epileptogenic substrates"},
            {"metric": "Seizure Severity Breakdown",
             "definition": "Distribution of seizure diary entries by severity (Mild/Moderate/Severe) — informs treatment urgency and surgical candidacy discussions"},
            {"metric": "Surgical Candidate Count",
             "definition": "Patients with both a temporal-region MRI lesion AND more than 2 seizure diary entries — flags cases for presurgical evaluation per ILAE guidelines"},
            {"metric": "Review by Role",
             "definition": "Bar chart showing the number of expert reviews contributed by each specialist role — ensures multidisciplinary coverage (neurology, neuropsychology, EEG, radiology)"},
            {"metric": "Lesion Type Distribution",
             "definition": "Pie chart of MRI lesion classifications across all scans — common types include hippocampal sclerosis, cortical dysplasia, tumor, vascular malformation, and normal"},
            {"metric": "Epilepsy Type Distribution",
             "definition": "Pie chart of epilepsy syndrome classifications from patient demographics — focal, generalized, combined, and unknown categories per ILAE 2017 classification"},
            {"metric": "Seizure Frequency Trend",
             "definition": "Line chart showing monthly seizure event counts from the seizure diary — tracks treatment response and breakthrough seizure patterns over time"},
            {"metric": "Medication Distribution",
             "definition": "Bar chart of anti-seizure medication usage — drug_name counts from the medications table, showing the most prescribed ASMs in the cohort"},
            {"metric": "Neuropsych Profile",
             "definition": "Radar-style aggregate of neuropsychological screening: avg MoCA (cognitive), PHQ-9 (depression), GAD-7 (anxiety), memory index, and attention index — identifies cognitive comorbidities affecting surgical planning"},
            {"metric": "Board Recommendation",
             "definition": "Algorithmic triage: 'Surgical evaluation' if temporal lesion + >2 seizures, 'Medication adjustment' if high seizures + no lesion, 'Continued monitoring' otherwise — serves as a decision-support suggestion, not a clinical order"},
            {"metric": "Review Concordance",
             "definition": "Agreement matrix showing how often each specialist role agrees or disagrees with AI predictions — highlights roles where AI concordance is high vs where expert override is common"},
            {"metric": "Medication Combinations",
             "definition": "Most common multi-drug regimens — identifies polytherapy patterns and potential drug interaction concerns for the board to review"},
        ],
    }
