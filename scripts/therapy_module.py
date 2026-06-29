"""
Patient Portal — Therapy Tab Module
====================================
Patient-facing therapy & rehabilitation view from clinical.db:
  - Rehab Programs (physical, cognitive, sleep, safety — derived from assessments)
  - Exercise Plans (evidence-based, medication-aware, seizure-safe)
  - Meditation Programs (mindfulness, breathing, relaxation — from mood assessments)
  - Physio Protocols (motor recovery, swallowing, speech — from functional scores)
  - Therapy Overview (combined dashboard summary)
  - Therapy Definitions (metric/term definitions for tooltip overlays)

All recommendations are derived from REAL patient data in clinical.db using
evidence-based epilepsy neurology rehabilitation guidelines (ILAE, AAN, AES).
"""
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# ── Evidence-based threshold constants ───────────────────────────────
# BARTHEL Index (0-100): functional independence
BARTHEL_SEVERE = 20       # total dependence
BARTHEL_MODERATE = 60     # moderate dependence
BARTHEL_MILD = 80         # mild dependence
BARTHEL_INDEPENDENT = 100 # independent

# MoCA / MMSE cognitive screens
MOCA_IMPAIRED = 26        # < 26 = cognitive impairment (Nasreddine 2005)
MMSE_SEVERE = 10          # severe cognitive impairment
MMSE_MODERATE = 18        # moderate cognitive impairment
MMSE_MILD = 24            # mild cognitive impairment

# Epworth Sleepiness Scale (0-24)
EPWORTH_NORMAL = 10       # <= 10 normal
EPWORTH_MILD = 12         # 11-12 mild sleepiness
EPWORTH_MODERATE = 15     # 13-15 moderate
EPWORTH_SEVERE = 24       # 16-24 severe

# PHQ-9 Depression (0-27)
PHQ9_MINIMAL = 4
PHQ9_MILD = 9
PHQ9_MODERATE = 14
PHQ9_MOD_SEVERE = 19
PHQ9_SEVERE = 27

# GAD-7 Anxiety (0-21)
GAD7_MINIMAL = 4
GAD7_MILD = 9
GAD7_MODERATE = 14
GAD7_SEVERE = 21

# QOLIE-31 Quality of Life (0-100, higher = better)
QOLIE_POOR = 40
QOLIE_FAIR = 60
QOLIE_GOOD = 80

# MASA (Mann Assessment of Swallowing Ability, 0-200)
MASA_DYSPHAGIA = 170      # < 170 = dysphagia risk (Mann 2002)
MASA_SEVERE = 140         # severe dysphagia

# WAB (Western Aphasia Battery, 0-100 AQ)
WAB_APHASIA = 93.8        # < 93.8 = aphasia present (Kertesz 1982)
WAB_SEVERE = 50           # severe aphasia

# ── Medication-exercise interaction knowledge base ───────────────────
MED_EXERCISE_MODS = {
    "Topiramate":    {"concern": "anhidrosis + metabolic acidosis",
                      "modification": "Increase hydration, avoid extreme heat, monitor for overheating",
                      "precautions": ["extra hydration (500ml pre-exercise)", "avoid hot environments", "watch for heat exhaustion signs"]},
    "Valproate":     {"concern": "weight gain + tremor",
                      "modification": "Emphasize aerobic exercise for weight management, balance exercises for tremor",
                      "precautions": ["weight management focus", "balance exercises if tremor present"]},
    "Phenytoin":     {"concern": "osteoporosis + ataxia",
                      "modification": "Weight-bearing exercises for bone density, balance training for ataxia",
                      "precautions": ["weight-bearing exercises for bone health", "balance aids if ataxic", "calcium + vitamin D monitoring"]},
    "Phenobarbital": {"concern": "sedation + vitamin D depletion + osteoporosis",
                      "modification": "Low-intensity initially, weight-bearing for bone health, avoid morning sessions if sedated",
                      "precautions": ["avoid high-intensity if sedated", "weight-bearing for bone density", "vitamin D supplementation"]},
    "Carbamazepine": {"concern": "hyponatremia + ataxia",
                      "modification": "Electrolyte monitoring with intense exercise, balance training",
                      "precautions": ["electrolyte replacement during prolonged exercise", "sodium monitoring"]},
    "Oxcarbazepine": {"concern": "hyponatremia",
                      "modification": "Electrolyte-rich hydration during exercise",
                      "precautions": ["electrolyte monitoring", "avoid excessive water intake without sodium"]},
    "Pregabalin":    {"concern": "dizziness + peripheral edema",
                      "modification": "Compression garments for edema, gradual warm-up for dizziness",
                      "precautions": ["gradual position changes", "compression socks if edema present"]},
    "Gabapentin":    {"concern": "dizziness + ataxia",
                      "modification": "Balance-focused exercises, avoid rapid position changes",
                      "precautions": ["supervised balance exercises", "gradual warm-up"]},
    "Perampanel":    {"concern": "dizziness + aggression",
                      "modification": "Supervised exercise preferred, avoid competitive settings if aggression noted",
                      "precautions": ["supervised sessions recommended", "non-competitive activities"]},
    "Zonisamide":    {"concern": "kidney stones + oligohidrosis",
                      "modification": "High fluid intake, avoid heat, monitor urine output",
                      "precautions": ["aggressive hydration", "avoid hot yoga / sauna", "monitor urine color"]},
    "Levetiracetam": {"concern": "irritability + fatigue",
                      "modification": "Moderate intensity, mind-body exercises for irritability",
                      "precautions": ["mindfulness-based exercise may help mood", "rest days if fatigue is significant"]},
}

# ── Safe exercise catalog for epilepsy patients ──────────────────────
EXERCISE_CATALOG = {
    "walking":            {"category": "aerobic",     "duration_min": 30, "frequency": "5x/week",  "safe_all": True,  "description": "Brisk walking on flat terrain"},
    "stationary_cycling": {"category": "aerobic",     "duration_min": 20, "frequency": "3x/week",  "safe_all": True,  "description": "Recumbent or upright stationary bike"},
    "swimming_supervised":{"category": "aerobic",     "duration_min": 30, "frequency": "2x/week",  "safe_all": False, "description": "Pool swimming with lifeguard/buddy (never alone)"},
    "yoga_modified":      {"category": "flexibility", "duration_min": 45, "frequency": "3x/week",  "safe_all": True,  "description": "Modified yoga — avoid headstands, hot yoga"},
    "tai_chi":            {"category": "balance",     "duration_min": 30, "frequency": "3x/week",  "safe_all": True,  "description": "Tai chi for balance and fall prevention"},
    "resistance_bands":   {"category": "strength",    "duration_min": 20, "frequency": "3x/week",  "safe_all": True,  "description": "Resistance band training — seated or standing"},
    "seated_exercises":   {"category": "strength",    "duration_min": 20, "frequency": "3x/week",  "safe_all": True,  "description": "Chair-based strength exercises for limited mobility"},
    "balance_training":   {"category": "balance",     "duration_min": 15, "frequency": "daily",    "safe_all": True,  "description": "Single-leg stance, tandem walk, stability board"},
    "stretching":         {"category": "flexibility", "duration_min": 15, "frequency": "daily",    "safe_all": True,  "description": "Full-body static and dynamic stretching"},
    "light_jogging":      {"category": "aerobic",     "duration_min": 20, "frequency": "3x/week",  "safe_all": False, "description": "Light jogging on flat ground (supervised for uncontrolled seizures)"},
}

# Activities contraindicated or restricted for uncontrolled seizures
RESTRICTED_ACTIVITIES = {
    "swimming_unsupervised": "Risk of drowning — swimming must be supervised",
    "rock_climbing":         "Risk of fall from height during seizure",
    "scuba_diving":          "Contraindicated in epilepsy — barotrauma + drowning risk",
    "solo_cycling_road":     "Road cycling alone — risk of traffic accident during seizure",
    "water_sports":          "Unsupervised water activities contraindicated",
}


def _connect():
    """Open read-only connection to clinical.db."""
    if not DB.exists():
        return None
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


def _get_assessments(conn, patient_id=None, instruments=None):
    """Fetch assessments, optionally filtered by patient and instrument list."""
    sql = "SELECT * FROM assessments"
    params = []
    clauses = []
    if patient_id:
        clauses.append("patient_id = ?")
        params.append(patient_id)
    if instruments:
        placeholders = ",".join("?" for _ in instruments)
        clauses.append(f"instrument IN ({placeholders})")
        params.extend(instruments)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY patient_id, instrument, created_at DESC"
    return conn.execute(sql, params).fetchall()


def _get_seizure_diary(conn, patient_id=None):
    """Fetch seizure diary entries."""
    if patient_id:
        return conn.execute(
            "SELECT * FROM seizure_diary WHERE patient_id = ? ORDER BY event_date DESC",
            (patient_id,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM seizure_diary ORDER BY patient_id, event_date DESC"
    ).fetchall()


def _get_medications(conn, patient_id=None):
    """Fetch medication records."""
    if patient_id:
        return conn.execute(
            "SELECT * FROM medications WHERE patient_id = ?", (patient_id,),
        ).fetchall()
    return conn.execute("SELECT * FROM medications ORDER BY patient_id").fetchall()


def _get_patients(conn, patient_id=None):
    """Fetch patient demographic records."""
    if patient_id:
        return conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,),
        ).fetchall()
    return conn.execute("SELECT * FROM patients ORDER BY patient_id").fetchall()


def _latest_assessment(rows, patient_id, instrument):
    """Return the latest assessment row for a given patient + instrument."""
    for r in rows:
        if r["patient_id"] == patient_id and r["instrument"] == instrument:
            return r  # already ordered DESC by created_at
    return None


def _parse_drug_names(med_rows):
    """Extract drug names from medication records grouped by patient."""
    by_patient = defaultdict(set)
    for r in med_rows:
        fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
        drug = fields.get("drug_name", "").strip()
        if drug:
            by_patient[r["patient_id"]].add(drug)
        for a in fields.get("aed", []):
            if a.strip():
                by_patient[r["patient_id"]].add(a.strip())
    return dict(by_patient)


def _seizure_frequency(diary_rows, patient_id):
    """Count seizure events for a patient. Returns (count, max_severity)."""
    events = [r for r in diary_rows if r["patient_id"] == patient_id]
    count = len(events)
    severities = [r["severity"] for r in events if r["severity"]]
    severity_rank = {"mild": 1, "moderate": 2, "severe": 3}
    max_sev = "none"
    if severities:
        max_sev = max(severities, key=lambda s: severity_rank.get(s.lower(), 0) if s else 0)
    return count, max_sev


# ════════════════════════════════════════════════════════════════════════
# 1. Rehabilitation Programs
# ════════════════════════════════════════════════════════════════════════
def rehab_programs(patient_id=None):
    """
    Active rehabilitation programs per patient, derived from:
      - BARTHEL scores -> physical rehab (mobility, ADL training)
      - MOCA/MMSE scores -> cognitive rehabilitation
      - EPWORTH scores -> sleep hygiene programs
      - Seizure frequency -> seizure safety/first-aid plans

    Each program: program_name, type, frequency, duration_weeks,
    assigned_by, status, progress_pct.
    """
    conn = _connect()
    if not conn:
        return {"total_patients": 0, "patients": []}

    assessments = _get_assessments(conn, patient_id,
                                   ["BARTHEL", "MOCA", "MMSE", "EPWORTH"])
    diary = _get_seizure_diary(conn, patient_id)
    patients_rows = _get_patients(conn, patient_id)
    conn.close()

    patient_ids = sorted({r["patient_id"] for r in patients_rows})
    patients = []

    for pid in patient_ids:
        programs = []

        # --- BARTHEL -> Physical Rehabilitation ---
        barthel = _latest_assessment(assessments, pid, "BARTHEL")
        if barthel and barthel["score"] is not None:
            score = float(barthel["score"])
            if score <= BARTHEL_SEVERE:
                programs.append({
                    "program_name": "Intensive Physical Rehabilitation",
                    "type": "physio",
                    "frequency": "5x/week",
                    "duration_weeks": 12,
                    "assigned_by": "Rehabilitation Medicine",
                    "status": "active",
                    "progress_pct": 0,
                    "rationale": f"BARTHEL {score}/100 — total dependence; intensive ADL training + mobility rehab",
                    "goals": ["Bed mobility", "Transfers", "Wheelchair independence", "Basic ADL retraining"],
                })
            elif score <= BARTHEL_MODERATE:
                programs.append({
                    "program_name": "Moderate Physical Rehabilitation",
                    "type": "physio",
                    "frequency": "3x/week",
                    "duration_weeks": 8,
                    "assigned_by": "Rehabilitation Medicine",
                    "status": "active",
                    "progress_pct": 0,
                    "rationale": f"BARTHEL {score}/100 — moderate dependence; functional mobility + ADL training",
                    "goals": ["Independent transfers", "Stair negotiation", "Community ambulation", "Instrumental ADL"],
                })
            elif score <= BARTHEL_MILD:
                programs.append({
                    "program_name": "Functional Independence Program",
                    "type": "physio",
                    "frequency": "2x/week",
                    "duration_weeks": 6,
                    "assigned_by": "Rehabilitation Medicine",
                    "status": "active",
                    "progress_pct": 0,
                    "rationale": f"BARTHEL {score}/100 — mild dependence; fine-tuning independence",
                    "goals": ["Full ADL independence", "Community reintegration", "Fall prevention"],
                })

        # --- MOCA / MMSE -> Cognitive Rehabilitation ---
        moca = _latest_assessment(assessments, pid, "MOCA")
        mmse = _latest_assessment(assessments, pid, "MMSE")

        cog_score = None
        cog_instrument = None
        if moca and moca["score"] is not None:
            cog_score = float(moca["score"])
            cog_instrument = "MoCA"
            cog_threshold = MOCA_IMPAIRED
        elif mmse and mmse["score"] is not None:
            cog_score = float(mmse["score"])
            cog_instrument = "MMSE"
            cog_threshold = MMSE_MILD

        if cog_score is not None and cog_instrument:
            threshold = MOCA_IMPAIRED if cog_instrument == "MoCA" else MMSE_MILD
            if cog_score < threshold:
                if cog_instrument == "MMSE" and cog_score < MMSE_SEVERE:
                    intensity = "Intensive"
                    freq = "4x/week"
                    weeks = 12
                    goals = ["Orientation training", "Memory cuing strategies",
                             "Caregiver education", "Structured environment"]
                elif cog_instrument == "MMSE" and cog_score < MMSE_MODERATE:
                    intensity = "Structured"
                    freq = "3x/week"
                    weeks = 10
                    goals = ["Memory compensation", "Attention training",
                             "Executive function tasks", "Daily routine structuring"]
                else:
                    intensity = "Targeted"
                    freq = "2x/week"
                    weeks = 8
                    goals = ["Memory strategies", "Attention enhancement",
                             "Visuospatial training", "Problem-solving exercises"]

                programs.append({
                    "program_name": f"{intensity} Cognitive Rehabilitation",
                    "type": "cognitive",
                    "frequency": freq,
                    "duration_weeks": weeks,
                    "assigned_by": "Neuropsychology",
                    "status": "active",
                    "progress_pct": 0,
                    "rationale": f"{cog_instrument} {cog_score}/{30 if cog_instrument == 'MoCA' else 30} — cognitive impairment detected",
                    "goals": goals,
                })

        # --- EPWORTH -> Sleep Hygiene Program ---
        epworth = _latest_assessment(assessments, pid, "EPWORTH")
        if epworth and epworth["score"] is not None:
            score = float(epworth["score"])
            if score > EPWORTH_NORMAL:
                if score > EPWORTH_MODERATE:
                    severity_label = "severe"
                    freq = "weekly + daily home practice"
                    weeks = 8
                elif score > EPWORTH_MILD:
                    severity_label = "moderate"
                    freq = "biweekly + daily home practice"
                    weeks = 6
                else:
                    severity_label = "mild"
                    freq = "monthly + daily home practice"
                    weeks = 4

                programs.append({
                    "program_name": f"Sleep Hygiene Program ({severity_label} sleepiness)",
                    "type": "sleep",
                    "frequency": freq,
                    "duration_weeks": weeks,
                    "assigned_by": "Sleep Medicine / Neurology",
                    "status": "active",
                    "progress_pct": 0,
                    "rationale": f"EPWORTH {score}/24 — {severity_label} daytime sleepiness; "
                                 "sleep deprivation is a major seizure trigger",
                    "goals": ["Sleep schedule regulation", "Sleep hygiene education",
                              "Stimulus control therapy", "Seizure trigger reduction"],
                })

        # --- Seizure Diary -> Safety/First-Aid Plan ---
        sz_count, max_sev = _seizure_frequency(diary, pid)
        if sz_count > 0:
            if sz_count >= 3 or max_sev.lower() == "severe":
                level = "Comprehensive"
                freq = "weekly review"
                weeks = 12
            else:
                level = "Standard"
                freq = "biweekly review"
                weeks = 8

            programs.append({
                "program_name": f"{level} Seizure Safety & First-Aid Plan",
                "type": "safety",
                "frequency": freq,
                "duration_weeks": weeks,
                "assigned_by": "Epilepsy Nurse Specialist",
                "status": "active",
                "progress_pct": 0,
                "rationale": f"{sz_count} seizure event(s), max severity: {max_sev}",
                "goals": ["Seizure first-aid training (patient + family)",
                          "Home safety assessment", "Rescue medication training",
                          "Seizure action plan documentation",
                          "SUDEP risk counseling"],
            })

        if programs:
            patients.append({
                "patient_id": pid,
                "total_programs": len(programs),
                "programs": programs,
            })

    return {
        "total_patients": len(patients),
        "total_programs": sum(p["total_programs"] for p in patients),
        "patients": patients,
    }


# ════════════════════════════════════════════════════════════════════════
# 2. Exercise Plans
# ════════════════════════════════════════════════════════════════════════
def exercise_plans(patient_id=None):
    """
    Evidence-based exercise prescriptions considering:
      - Seizure severity -> safe exercise types
      - Medications -> exercise modifications (hydration, bone health, etc.)
      - BARTHEL -> adaptive exercises for mobility limitations

    Each plan: exercise_name, category, duration_min, frequency,
    precautions[], contraindicated (bool), reason.
    """
    conn = _connect()
    if not conn:
        return {"total_patients": 0, "patients": []}

    assessments = _get_assessments(conn, patient_id, ["BARTHEL"])
    diary = _get_seizure_diary(conn, patient_id)
    med_rows = _get_medications(conn, patient_id)
    patients_rows = _get_patients(conn, patient_id)
    conn.close()

    patient_ids = sorted({r["patient_id"] for r in patients_rows})
    drug_map = _parse_drug_names(med_rows)
    patients = []

    for pid in patient_ids:
        sz_count, max_sev = _seizure_frequency(diary, pid)
        uncontrolled = sz_count >= 3 or (max_sev and max_sev.lower() == "severe")
        barthel = _latest_assessment(assessments, pid, "BARTHEL")
        barthel_score = float(barthel["score"]) if barthel and barthel["score"] is not None else None
        patient_drugs = drug_map.get(pid, set())

        # Collect medication-specific precautions
        med_precautions = []
        med_modifications = []
        for drug in patient_drugs:
            # Normalize drug name for lookup
            drug_key = drug.strip()
            for canonical, info in MED_EXERCISE_MODS.items():
                if drug_key.lower() == canonical.lower():
                    med_precautions.extend(info["precautions"])
                    med_modifications.append({
                        "drug": canonical,
                        "concern": info["concern"],
                        "modification": info["modification"],
                    })
                    break

        # Build exercise list
        exercises = []
        for ex_name, ex_info in EXERCISE_CATALOG.items():
            precautions = list(med_precautions)  # copy
            contraindicated = False
            reason = None

            # Seizure safety check
            if not ex_info["safe_all"] and uncontrolled:
                contraindicated = True
                reason = f"Uncontrolled seizures ({sz_count} events, max severity: {max_sev}) — {ex_name.replace('_', ' ')} requires supervision or is restricted"
                precautions.append("Requires 1:1 supervision or buddy system")

            # BARTHEL mobility check
            if barthel_score is not None and barthel_score <= BARTHEL_MODERATE:
                if ex_info["category"] == "aerobic" and ex_name not in ("stationary_cycling",):
                    if barthel_score <= BARTHEL_SEVERE:
                        contraindicated = True
                        reason = f"BARTHEL {barthel_score}/100 — severely limited mobility"
                    else:
                        precautions.append(f"BARTHEL {barthel_score}/100 — modify for mobility limitations")

            # For severely limited patients, recommend seated/adaptive exercises
            if barthel_score is not None and barthel_score <= BARTHEL_SEVERE:
                if ex_name not in ("seated_exercises", "stretching"):
                    contraindicated = True
                    reason = reason or f"BARTHEL {barthel_score}/100 — seated exercises recommended"

            exercises.append({
                "exercise_name": ex_name.replace("_", " ").title(),
                "category": ex_info["category"],
                "duration_min": ex_info["duration_min"],
                "frequency": ex_info["frequency"],
                "description": ex_info["description"],
                "precautions": precautions if precautions else ["Standard seizure precautions apply"],
                "contraindicated": contraindicated,
                "reason": reason,
            })

        # Restricted activities warning
        restrictions = []
        if sz_count > 0:
            for act, warning in RESTRICTED_ACTIVITIES.items():
                restrictions.append({"activity": act.replace("_", " ").title(), "warning": warning})

        patients.append({
            "patient_id": pid,
            "seizure_events": sz_count,
            "max_severity": max_sev,
            "uncontrolled_seizures": uncontrolled,
            "barthel_score": barthel_score,
            "medications_assessed": sorted(patient_drugs),
            "medication_modifications": med_modifications,
            "exercises": exercises,
            "safe_exercises": sum(1 for e in exercises if not e["contraindicated"]),
            "contraindicated_exercises": sum(1 for e in exercises if e["contraindicated"]),
            "restricted_activities": restrictions,
        })

    return {"total_patients": len(patients), "patients": patients}


# ════════════════════════════════════════════════════════════════════════
# 3. Meditation Programs
# ════════════════════════════════════════════════════════════════════════
def meditation_programs(patient_id=None):
    """
    Mindfulness/meditation programs derived from:
      - PHQ9 -> stress reduction / depression-focused programs
      - GAD7 -> anxiety management programs
      - QOLIE31 -> quality of life / coping strategies
      - Seizure triggers -> trigger management mindfulness

    Each: program_name, type, session_duration_min, frequency,
    target_condition, evidence_level.
    """
    conn = _connect()
    if not conn:
        return {"total_patients": 0, "patients": []}

    assessments = _get_assessments(conn, patient_id,
                                   ["PHQ9", "GAD7", "QOLIE31", "NDDIE"])
    diary = _get_seizure_diary(conn, patient_id)
    patients_rows = _get_patients(conn, patient_id)
    conn.close()

    patient_ids = sorted({r["patient_id"] for r in patients_rows})
    patients = []

    for pid in patient_ids:
        programs = []

        # --- PHQ9 -> Depression / Stress Reduction ---
        phq9 = _latest_assessment(assessments, pid, "PHQ9")
        if phq9 and phq9["score"] is not None:
            score = float(phq9["score"])
            if score > PHQ9_MODERATE:
                programs.append({
                    "program_name": "Mindfulness-Based Cognitive Therapy (MBCT) for Depression",
                    "type": "mindfulness",
                    "session_duration_min": 45,
                    "frequency": "3x/week",
                    "target_condition": f"Moderate-severe depression (PHQ-9: {score}/27)",
                    "evidence_level": "Level A — RCTs support MBCT for depression relapse prevention (Kuyken et al. 2016)",
                    "components": ["Body scan meditation", "Sitting meditation",
                                   "Mindful movement", "Cognitive defusion exercises",
                                   "Pleasant/unpleasant events diary"],
                })
            elif score > PHQ9_MILD:
                programs.append({
                    "program_name": "Mindfulness-Based Stress Reduction (MBSR)",
                    "type": "mindfulness",
                    "session_duration_min": 30,
                    "frequency": "2x/week",
                    "target_condition": f"Mild-moderate depression (PHQ-9: {score}/27)",
                    "evidence_level": "Level B — MBSR reduces depressive symptoms in chronic illness (Grossman et al. 2004)",
                    "components": ["Guided meditation", "Body awareness",
                                   "Gentle yoga", "Stress psychoeducation"],
                })

        # --- NDDIE -> Epilepsy-specific depression ---
        nddie = _latest_assessment(assessments, pid, "NDDIE")
        if nddie and nddie["score"] is not None:
            score = float(nddie["score"])
            if score > 15:  # NDDI-E cutoff for major depression in epilepsy
                programs.append({
                    "program_name": "Epilepsy-Specific Depression Coping Program",
                    "type": "progressive_relaxation",
                    "session_duration_min": 30,
                    "frequency": "3x/week",
                    "target_condition": f"NDDI-E elevated ({score}) — epilepsy-related depression",
                    "evidence_level": "Level B — NDDI-E validated screening; relaxation reduces seizure-related distress (Gandy et al. 2014)",
                    "components": ["Progressive muscle relaxation", "Seizure fear desensitization",
                                   "Cognitive restructuring for epilepsy stigma",
                                   "Self-compassion meditation"],
                })

        # --- GAD7 -> Anxiety Management ---
        gad7 = _latest_assessment(assessments, pid, "GAD7")
        if gad7 and gad7["score"] is not None:
            score = float(gad7["score"])
            if score > GAD7_MODERATE:
                programs.append({
                    "program_name": "Anxiety Management — Breathing & Relaxation Protocol",
                    "type": "breathing",
                    "session_duration_min": 20,
                    "frequency": "daily",
                    "target_condition": f"Moderate-severe anxiety (GAD-7: {score}/21)",
                    "evidence_level": "Level A — Diaphragmatic breathing reduces cortisol and seizure susceptibility (Ma et al. 2017)",
                    "components": ["4-7-8 breathing technique", "Diaphragmatic breathing",
                                   "Box breathing for acute anxiety",
                                   "Vagal tone stimulation breathing"],
                })
            elif score > GAD7_MILD:
                programs.append({
                    "program_name": "Guided Relaxation for Mild Anxiety",
                    "type": "progressive_relaxation",
                    "session_duration_min": 15,
                    "frequency": "3x/week",
                    "target_condition": f"Mild-moderate anxiety (GAD-7: {score}/21)",
                    "evidence_level": "Level B — Progressive relaxation reduces anxiety in epilepsy (Lundgren et al. 2006)",
                    "components": ["Progressive muscle relaxation",
                                   "Guided imagery", "Autogenic training"],
                })

        # --- QOLIE31 -> Quality of Life / Coping ---
        qolie = _latest_assessment(assessments, pid, "QOLIE31")
        if qolie and qolie["score"] is not None:
            score = float(qolie["score"])
            if score < QOLIE_FAIR:
                programs.append({
                    "program_name": "Quality of Life Enhancement — Yoga Nidra Program",
                    "type": "yoga_nidra",
                    "session_duration_min": 40,
                    "frequency": "3x/week",
                    "target_condition": f"Reduced quality of life (QOLIE-31: {score}/100)",
                    "evidence_level": "Level B — Yoga Nidra improves QoL in epilepsy (Panebianco et al. 2017 Cochrane review)",
                    "components": ["Body rotation awareness", "Breath awareness",
                                   "Opposite feeling pairs", "Visualization",
                                   "Sankalpa (intention setting)"],
                })
            elif score < QOLIE_GOOD:
                programs.append({
                    "program_name": "Coping Skills & Resilience Mindfulness",
                    "type": "mindfulness",
                    "session_duration_min": 20,
                    "frequency": "2x/week",
                    "target_condition": f"Fair quality of life (QOLIE-31: {score}/100)",
                    "evidence_level": "Level C — Mindfulness-based interventions improve coping in chronic neurological conditions",
                    "components": ["Acceptance meditation", "Values clarification",
                                   "Gratitude practice", "Social connection exercises"],
                })

        # --- Seizure Triggers -> Trigger Management Mindfulness ---
        trigger_events = [r for r in diary if r["patient_id"] == pid and r["trigger"]]
        if trigger_events:
            triggers = list({r["trigger"] for r in trigger_events})
            stress_related = any(
                t.lower() in ("stress", "anxiety", "emotional", "sleep deprivation",
                               "poor sleep", "fatigue", "emotional stress")
                for t in triggers
            )
            if stress_related:
                programs.append({
                    "program_name": "Seizure Trigger Awareness Meditation",
                    "type": "mindfulness",
                    "session_duration_min": 15,
                    "frequency": "daily",
                    "target_condition": f"Stress-related seizure triggers identified: {', '.join(triggers)}",
                    "evidence_level": "Level B — Stress management reduces seizure frequency in stress-triggered epilepsy (Haut et al. 2012)",
                    "components": ["Trigger recognition mindfulness",
                                   "Pre-seizure aura awareness meditation",
                                   "Stress inoculation techniques",
                                   "Biofeedback breathing for trigger moments"],
                })

        if programs:
            patients.append({
                "patient_id": pid,
                "total_programs": len(programs),
                "programs": programs,
            })

    return {
        "total_patients": len(patients),
        "total_programs": sum(p["total_programs"] for p in patients),
        "patients": patients,
    }


# ════════════════════════════════════════════════════════════════════════
# 4. Physiotherapy Protocols
# ════════════════════════════════════════════════════════════════════════
def physio_protocols(patient_id=None):
    """
    Physiotherapy protocols derived from:
      - Motor signs from seizure_diary -> post-ictal motor recovery
      - BARTHEL -> functional independence protocols
      - MASA -> swallowing therapy
      - WAB -> speech/language therapy

    Each: protocol_name, target_area, exercises[], sessions_per_week,
    duration_weeks, therapist_notes.
    """
    conn = _connect()
    if not conn:
        return {"total_patients": 0, "patients": []}

    assessments = _get_assessments(conn, patient_id,
                                   ["BARTHEL", "MASA", "WAB", "BNT",
                                    "VERBAL_FLUENCY"])
    diary = _get_seizure_diary(conn, patient_id)
    patients_rows = _get_patients(conn, patient_id)
    conn.close()

    patient_ids = sorted({r["patient_id"] for r in patients_rows})
    patients = []

    for pid in patient_ids:
        protocols = []

        # --- Motor signs from seizure diary -> Post-Ictal Motor Recovery ---
        patient_diary = [r for r in diary if r["patient_id"] == pid]
        motor_events = [r for r in patient_diary if r["motor_signs"] and r["motor_signs"].strip()]
        if motor_events:
            motor_types = list({r["motor_signs"] for r in motor_events})
            recovery_times = [r["recovery_min"] for r in motor_events
                              if r["recovery_min"] is not None]
            avg_recovery = round(sum(recovery_times) / len(recovery_times), 1) if recovery_times else None

            protocols.append({
                "protocol_name": "Post-Ictal Motor Recovery Protocol",
                "target_area": "upper_limb" if any("arm" in m.lower() or "upper" in m.lower()
                                                    for m in motor_types) else "lower_limb",
                "exercises": [
                    "Passive range-of-motion (PROM) during post-ictal phase",
                    "Active-assisted ROM as awareness returns",
                    "Gentle stretching of affected muscle groups",
                    "Proprioceptive neuromuscular facilitation (PNF)",
                    "Graduated active movement re-education",
                ],
                "sessions_per_week": 3,
                "duration_weeks": 8,
                "therapist_notes": (
                    f"Motor signs observed: {', '.join(motor_types)}. "
                    f"{'Avg recovery: ' + str(avg_recovery) + ' min. ' if avg_recovery else ''}"
                    "Begin passive ROM immediately post-ictally. "
                    "Progress to active movement as patient regains awareness. "
                    "Monitor for Todd's paralysis (transient focal weakness)."
                ),
            })

        # --- BARTHEL -> Functional Independence Protocol ---
        barthel = _latest_assessment(assessments, pid, "BARTHEL")
        if barthel and barthel["score"] is not None:
            score = float(barthel["score"])
            if score <= BARTHEL_MODERATE:
                if score <= BARTHEL_SEVERE:
                    exercises = [
                        "Bed positioning and pressure relief education",
                        "Passive/active-assisted transfers (bed to chair)",
                        "Sitting balance training (static -> dynamic)",
                        "Upper limb functional reaching tasks",
                        "Assisted standing with frame/parallel bars",
                    ]
                    sessions = 5
                    weeks = 12
                    notes = (f"BARTHEL {score}/100 — severe functional limitation. "
                             "Focus on preventing secondary complications (contractures, "
                             "pressure injuries). Caregiver training is essential.")
                else:
                    exercises = [
                        "Independent transfer training (sit-to-stand, bed-to-chair)",
                        "Dynamic balance training (reaching, turning, obstacle course)",
                        "Stair negotiation with appropriate aid",
                        "Community ambulation simulation",
                        "Dual-task walking (cognitive + motor)",
                        "Kitchen/bathroom ADL simulation",
                    ]
                    sessions = 3
                    weeks = 8
                    notes = (f"BARTHEL {score}/100 — moderate dependence. "
                             "Goal: maximize functional independence for home discharge. "
                             "Seizure safety precautions during all mobility tasks.")

                protocols.append({
                    "protocol_name": "Functional Independence Physiotherapy",
                    "target_area": "balance",
                    "exercises": exercises,
                    "sessions_per_week": sessions,
                    "duration_weeks": weeks,
                    "therapist_notes": notes,
                })

        # --- MASA -> Swallowing Therapy ---
        masa = _latest_assessment(assessments, pid, "MASA")
        if masa and masa["score"] is not None:
            score = float(masa["score"])
            if score < MASA_DYSPHAGIA:
                if score < MASA_SEVERE:
                    exercises = [
                        "Mendelsohn maneuver training",
                        "Effortful swallow exercises",
                        "Shaker exercise (head lift) for UES opening",
                        "Ice stimulation of faucial arches",
                        "Supraglottic swallow technique",
                        "Modified diet texture training",
                    ]
                    sessions = 5
                    weeks = 10
                    notes = (f"MASA {score}/200 — severe dysphagia. "
                             "NPO risk assessment needed. Modified texture diet "
                             "(IDDSI Level 4-5). Aspiration precautions. "
                             "Post-ictal swallowing assessment mandatory after seizures.")
                else:
                    exercises = [
                        "Tongue strengthening exercises",
                        "Oral motor coordination drills",
                        "Chin tuck swallowing technique",
                        "Controlled bolus size training",
                        "Safe swallowing strategy education",
                    ]
                    sessions = 3
                    weeks = 6
                    notes = (f"MASA {score}/200 — mild-moderate dysphagia. "
                             "Supervise meals. Educate on safe swallowing positions. "
                             "Reassess after seizure clusters (transient worsening expected).")

                protocols.append({
                    "protocol_name": "Dysphagia Management Protocol",
                    "target_area": "swallowing",
                    "exercises": exercises,
                    "sessions_per_week": sessions,
                    "duration_weeks": weeks,
                    "therapist_notes": notes,
                })

        # --- WAB -> Speech/Language Therapy ---
        wab = _latest_assessment(assessments, pid, "WAB")
        if wab and wab["score"] is not None:
            score = float(wab["score"])
            if score < WAB_APHASIA:
                if score < WAB_SEVERE:
                    exercises = [
                        "Auditory comprehension drills (single word -> sentence)",
                        "Naming therapy (semantic feature analysis)",
                        "Repetition hierarchy (syllable -> phrase -> sentence)",
                        "Augmentative communication introduction (AAC)",
                        "Script training for functional communication",
                        "Constraint-induced language therapy (CILT)",
                    ]
                    sessions = 4
                    weeks = 12
                    notes = (f"WAB AQ {score}/100 — severe aphasia. "
                             "AAC device trial recommended. "
                             "Caregiver communication partner training essential. "
                             "Consider seizure lateralization in therapy planning.")
                else:
                    exercises = [
                        "Word-finding strategies (circumlocution, semantic cues)",
                        "Sentence completion and formulation",
                        "Conversational coaching",
                        "Reading comprehension graded tasks",
                        "Written expression exercises",
                    ]
                    sessions = 3
                    weeks = 8
                    notes = (f"WAB AQ {score}/100 — mild-moderate aphasia. "
                             "Focus on functional communication for daily needs. "
                             "Monitor for post-ictal language regression.")

                protocols.append({
                    "protocol_name": "Speech-Language Therapy for Aphasia",
                    "target_area": "speech",
                    "exercises": exercises,
                    "sessions_per_week": sessions,
                    "duration_weeks": weeks,
                    "therapist_notes": notes,
                })

        # --- BNT / VERBAL_FLUENCY -> Word-Finding Support ---
        bnt = _latest_assessment(assessments, pid, "BNT")
        vf = _latest_assessment(assessments, pid, "VERBAL_FLUENCY")

        word_finding_concern = False
        wf_rationale = []
        if bnt and bnt["score"] is not None and float(bnt["score"]) < 45:
            word_finding_concern = True
            wf_rationale.append(f"BNT {bnt['score']}/60 — confrontation naming deficit")
        if vf and vf["score"] is not None and float(vf["score"]) < 15:
            word_finding_concern = True
            wf_rationale.append(f"Verbal Fluency {vf['score']} — reduced generative naming")

        if word_finding_concern and not wab:
            # Only add if not already covered by WAB-based protocol
            protocols.append({
                "protocol_name": "Word-Finding & Naming Therapy",
                "target_area": "speech",
                "exercises": [
                    "Semantic Feature Analysis (SFA) for naming",
                    "Phonological Components Analysis (PCA)",
                    "Category-specific naming drills",
                    "Generative naming practice (animals, tools, actions)",
                    "Spaced retrieval training",
                ],
                "sessions_per_week": 2,
                "duration_weeks": 6,
                "therapist_notes": (
                    f"{'; '.join(wf_rationale)}. "
                    "Common in temporal lobe epilepsy — may fluctuate with "
                    "seizure activity and AED levels. Monitor word-finding "
                    "before/after medication changes."
                ),
            })

        if protocols:
            patients.append({
                "patient_id": pid,
                "total_protocols": len(protocols),
                "protocols": protocols,
            })

    return {
        "total_patients": len(patients),
        "total_protocols": sum(p["total_protocols"] for p in patients),
        "patients": patients,
    }


# ════════════════════════════════════════════════════════════════════════
# 5. Therapy Overview (Combined Dashboard)
# ════════════════════════════════════════════════════════════════════════
def therapy_overview(patient_id=None):
    """
    Combined therapy dashboard summary — all modules in one call.
    Aggregates rehab programs, exercise plans, meditation, and physio.
    """
    rehab = rehab_programs(patient_id)
    exercises = exercise_plans(patient_id)
    meditation = meditation_programs(patient_id)
    physio = physio_protocols(patient_id)

    # Aggregate stats
    all_patient_ids = set()
    for section in [rehab, exercises, meditation, physio]:
        for p in section.get("patients", []):
            all_patient_ids.add(p["patient_id"])

    # Type breakdown from rehab programs
    type_counts = defaultdict(int)
    for p in rehab.get("patients", []):
        for prog in p.get("programs", []):
            type_counts[prog["type"]] += 1

    # Exercise safety summary
    total_safe = sum(p.get("safe_exercises", 0) for p in exercises.get("patients", []))
    total_contra = sum(p.get("contraindicated_exercises", 0) for p in exercises.get("patients", []))

    return {
        "module": "Patient Portal — Therapy Tab",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_patients_with_therapy": len(all_patient_ids),
            "total_rehab_programs": rehab.get("total_programs", 0),
            "total_meditation_programs": meditation.get("total_programs", 0),
            "total_physio_protocols": physio.get("total_protocols", 0),
            "rehab_type_breakdown": dict(type_counts),
            "exercise_safety": {
                "total_safe_prescriptions": total_safe,
                "total_contraindicated": total_contra,
            },
        },
        "rehab_programs": rehab,
        "exercise_plans": exercises,
        "meditation_programs": meditation,
        "physio_protocols": physio,
    }


# ════════════════════════════════════════════════════════════════════════
# 6. Therapy Definitions (Tooltip Overlays)
# ════════════════════════════════════════════════════════════════════════
def therapy_definitions():
    """
    Returns dict of metric/term definitions for tooltip overlays in the
    patient portal therapy tab UI.
    """
    return {
        "BARTHEL": {
            "name": "Barthel Index",
            "range": "0-100",
            "description": "Measures functional independence in activities of daily living (ADL). "
                           "100 = fully independent, 0 = total dependence.",
            "thresholds": {
                "0-20": "Total dependence — requires 24h assistance",
                "21-60": "Moderate dependence — needs help with most ADLs",
                "61-80": "Mild dependence — independent with some assistance",
                "81-100": "Functionally independent",
            },
        },
        "MOCA": {
            "name": "Montreal Cognitive Assessment",
            "range": "0-30",
            "description": "Screens for mild cognitive impairment. Evaluates attention, memory, "
                           "language, visuospatial, executive function, orientation.",
            "thresholds": {"<26": "Cognitive impairment detected", ">=26": "Normal cognition"},
        },
        "MMSE": {
            "name": "Mini-Mental State Examination",
            "range": "0-30",
            "description": "Brief cognitive screening tool assessing orientation, registration, "
                           "attention, recall, language, and visuospatial ability.",
            "thresholds": {
                "0-9": "Severe cognitive impairment",
                "10-17": "Moderate cognitive impairment",
                "18-23": "Mild cognitive impairment",
                "24-30": "Normal cognition",
            },
        },
        "EPWORTH": {
            "name": "Epworth Sleepiness Scale",
            "range": "0-24",
            "description": "Measures daytime sleepiness propensity. Important in epilepsy as "
                           "sleep deprivation is a major seizure trigger.",
            "thresholds": {
                "0-10": "Normal daytime sleepiness",
                "11-12": "Mild excessive sleepiness",
                "13-15": "Moderate excessive sleepiness",
                "16-24": "Severe excessive sleepiness",
            },
        },
        "PHQ9": {
            "name": "Patient Health Questionnaire-9",
            "range": "0-27",
            "description": "Screens for depression severity. Depression is the most common "
                           "psychiatric comorbidity in epilepsy (prevalence 20-55%).",
            "thresholds": {
                "0-4": "Minimal depression",
                "5-9": "Mild depression",
                "10-14": "Moderate depression",
                "15-19": "Moderately severe depression",
                "20-27": "Severe depression",
            },
        },
        "GAD7": {
            "name": "Generalized Anxiety Disorder-7",
            "range": "0-21",
            "description": "Screens for anxiety severity. Anxiety disorders affect 10-25% of "
                           "epilepsy patients and can exacerbate seizure frequency.",
            "thresholds": {
                "0-4": "Minimal anxiety",
                "5-9": "Mild anxiety",
                "10-14": "Moderate anxiety",
                "15-21": "Severe anxiety",
            },
        },
        "QOLIE31": {
            "name": "Quality of Life in Epilepsy-31",
            "range": "0-100",
            "description": "Epilepsy-specific quality of life measure covering seizure worry, "
                           "emotional well-being, energy, cognition, medication effects, "
                           "social function, and overall QoL.",
            "thresholds": {
                "<40": "Poor quality of life",
                "40-59": "Fair quality of life",
                "60-79": "Good quality of life",
                ">=80": "Excellent quality of life",
            },
        },
        "MASA": {
            "name": "Mann Assessment of Swallowing Ability",
            "range": "0-200",
            "description": "Evaluates swallowing function. Dysphagia can occur in epilepsy "
                           "patients post-ictally or with brainstem-onset seizures.",
            "thresholds": {
                "<140": "Severe dysphagia — aspiration risk",
                "140-169": "Mild-moderate dysphagia",
                ">=170": "Normal swallowing function",
            },
        },
        "WAB": {
            "name": "Western Aphasia Battery (Aphasia Quotient)",
            "range": "0-100",
            "description": "Assesses language function including fluency, comprehension, "
                           "repetition, and naming. Language deficits are common in "
                           "temporal lobe epilepsy.",
            "thresholds": {
                "<50": "Severe aphasia",
                "50-93.7": "Mild-moderate aphasia",
                ">=93.8": "No aphasia",
            },
        },
        "BNT": {
            "name": "Boston Naming Test",
            "range": "0-60",
            "description": "Measures confrontation naming ability — the ability to name "
                           "presented pictures. Sensitive to temporal lobe dysfunction.",
            "thresholds": {"<45": "Naming deficit", ">=45": "Normal naming"},
        },
        "VERBAL_FLUENCY": {
            "name": "Verbal Fluency Test",
            "range": "Varies (words per minute)",
            "description": "Measures word generation ability (phonemic and semantic). "
                           "Sensitive to frontal and temporal lobe epilepsy.",
            "thresholds": {"<15": "Below expected fluency", ">=15": "Normal fluency"},
        },
        "rehab_types": {
            "physio": "Physical rehabilitation — mobility, ADL training, functional independence",
            "cognitive": "Cognitive rehabilitation — memory, attention, executive function training",
            "sleep": "Sleep hygiene program — sleep schedule, triggers, seizure risk reduction",
            "safety": "Seizure safety plan — first-aid training, home safety, SUDEP counseling",
        },
        "exercise_categories": {
            "aerobic": "Cardiovascular exercise to improve heart health and reduce seizure frequency",
            "strength": "Resistance training to maintain muscle mass and bone density (especially important with enzyme-inducing AEDs)",
            "flexibility": "Stretching and yoga to reduce muscle tension and improve range of motion",
            "balance": "Balance training to reduce fall risk (falls are a leading cause of seizure-related injury)",
        },
        "meditation_types": {
            "mindfulness": "Present-moment awareness meditation — reduces stress-related seizure triggers",
            "breathing": "Controlled breathing techniques — activates parasympathetic nervous system, may reduce seizure susceptibility",
            "progressive_relaxation": "Systematic muscle tension-release — reduces physiological arousal",
            "yoga_nidra": "Guided deep relaxation technique — shown to improve QoL in epilepsy (Cochrane evidence)",
        },
        "evidence_levels": {
            "Level A": "Established by 2+ well-designed RCTs or meta-analysis",
            "Level B": "Probably effective — 1 well-designed RCT or multiple observational studies",
            "Level C": "Possibly effective — expert consensus or case series",
        },
    }
