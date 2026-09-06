"""Patient-Facing Report Dashboard — simplified clinical summaries for patients and caregivers.

This is step 22 in the EEG AI RAG pipeline ("Final Output" → "patient-facing report").
It is NOT a clinical tool for doctors. It summarises what happened in plain English,
reminds patients about medications and follow-ups, and always defers diagnosis to the
treating physician.

All data is pulled from data/clinical.db — no fabricated values.

Tables used:
  patients          — patient_id, name, age, gender, disease
  analyses          — patient_id, disease, predicted_label, confidence, signal_quality, result_json
  assessments       — patient_id, instrument, score, max_score, interpretation, level, alert
  appointments      — patient_id, provider, department, appt_type, status, scheduled_for
  patient_appointments — patient_id, appointment_type, provider_name, appointment_date, status
  medication_adherence — patient_id, drug_name, dose_mg, frequency, taken, log_date
  medication_refills — patient_id, drug_name, refill_date, days_supply, pharmacy
  seizure_diary     — patient_id, event_date, severity, trigger, duration_sec
  education_modules — patient_id, module_name, completion_pct, quiz_score
  caregivers        — patient_id, name, role, rescue_med_trained, seizure_first_aid_confidence

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confidence_to_risk(confidence: float) -> str:
    """Convert a model confidence score to a patient-friendly risk label."""
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.55:
        return "Moderate"
    return "Low"


def _risk_color(risk: str) -> str:
    return {"Low": "#22c55e", "Moderate": "#f97316", "High": "#ef4444"}.get(risk, "#94a3b8")


def _plain_signal_quality(sq: str) -> str:
    """Translate technical signal quality into a patient-friendly phrase."""
    mapping = {
        "Good": "Your brain-wave recording came out clearly — the equipment worked well.",
        "Fair": "Your brain-wave recording was mostly clear with a few noisy moments.",
        "Poor": "Parts of your brain-wave recording had interference. Your care team may ask for a repeat.",
    }
    return mapping.get(sq, "Your brain-wave recording has been reviewed by the system.")


def _plain_assessment_level(instrument: str, level: str, score) -> str:
    """Return a plain-English sentence for common assessment scores."""
    level_lower = (level or "").lower()
    instrument_upper = (instrument or "").upper()

    if instrument_upper == "PHQ9":
        phrases = {
            "minimal": "Your mood questionnaire showed minimal signs of low mood.",
            "mild": "Your mood questionnaire picked up mild signs of low mood. Your doctor may want to discuss this.",
            "moderate": "Your mood questionnaire showed moderate signs of low mood. Bring this up at your next visit.",
            "moderately severe": "Your mood questionnaire flagged moderately severe low mood. Please talk to your care team soon.",
            "severe": "Your mood questionnaire showed severe low mood. Please contact your care team right away.",
            "normal": "Your mood questionnaire results are in the normal range.",
        }
        return phrases.get(level_lower, f"Your mood questionnaire score was {score}. Your doctor will review this.")

    if instrument_upper == "GAD7":
        phrases = {
            "minimal": "Your anxiety questionnaire showed minimal anxiety.",
            "mild": "Your anxiety questionnaire showed mild anxiety.",
            "moderate": "Your anxiety questionnaire showed moderate anxiety. Your doctor may want to discuss this.",
            "severe": "Your anxiety questionnaire showed high anxiety. Please mention this at your next visit.",
            "normal": "Your anxiety questionnaire results are in the normal range.",
        }
        return phrases.get(level_lower, f"Your anxiety questionnaire score was {score}.")

    if instrument_upper == "MOCA":
        if isinstance(score, (int, float)):
            if score >= 26:
                return "Your thinking-and-memory check showed results in the normal range."
            if score >= 18:
                return "Your thinking-and-memory check showed mild changes. Your doctor will discuss next steps."
            return "Your thinking-and-memory check showed some changes. Your care team will review this with you."
        return f"Your thinking-and-memory check was completed (score: {score})."

    if instrument_upper == "EPWORTH":
        if isinstance(score, (int, float)):
            if score <= 10:
                return "Your daytime sleepiness check is in the normal range."
            return "Your daytime sleepiness check suggests you may be sleepier than usual. Mention this to your doctor."
        return f"Your sleepiness questionnaire score was {score}."

    if instrument_upper == "QOLIE31":
        return f"Your quality-of-life survey was completed (score: {score}). Your doctor will review this with you."

    if instrument_upper == "NDDIE":
        phrases = {
            "normal": "Your mood screen for people with epilepsy showed normal results.",
            "mild": "Your mood screen showed mild concerns. Mention this to your care team.",
            "moderate": "Your mood screen showed moderate concerns. Your care team will follow up.",
            "severe": "Your mood screen flagged a concern. Please speak with your care team soon.",
        }
        return phrases.get(level_lower, f"Your {instrument} questionnaire score was {score}.")

    return f"Your {instrument} assessment was completed (score: {score}, level: {level})."


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview() -> dict:
    """
    Top-level summary card for a patient-facing dashboard.

    Returns
    -------
    dict with keys:
      total_reports          – number of patients who have at least one analysis
      patients_with_followup – number of patients with upcoming appointments
      risk_distribution      – {Low: N, Moderate: N, High: N}
      avg_risk_level         – most common simplified risk category across all reports
      recent_reports         – list of per-patient summaries in plain language
    """
    analyses = _rows(
        "SELECT patient_id, predicted_label, confidence, signal_quality, disease, created_at "
        "FROM analyses ORDER BY created_at DESC"
    )

    if not analyses:
        return {
            "total_reports": 0,
            "patients_with_followup": 0,
            "avg_risk_level": "N/A",
            "risk_distribution": {"Low": 0, "Moderate": 0, "High": 0},
            "recent_reports": [],
            "message": "No reports have been generated yet. Upload your EEG recording to get started.",
        }

    # Latest analysis per patient
    latest_per_patient = {}
    for row in analyses:
        pid = row["patient_id"]
        if pid not in latest_per_patient:
            latest_per_patient[pid] = row

    # Risk distribution
    risk_dist: Counter = Counter()
    for row in latest_per_patient.values():
        risk_dist[_confidence_to_risk(row["confidence"] or 0)] += 1

    total_risk = sum(risk_dist.values()) or 1
    avg_risk = max(risk_dist, key=risk_dist.get) if risk_dist else "N/A"

    # Patients with upcoming appointments (either table)
    appt_patients_1 = _rows(
        "SELECT DISTINCT patient_id FROM appointments WHERE status IN ('booked','confirmed')"
    )
    appt_patients_2 = _rows(
        "SELECT DISTINCT patient_id FROM patient_appointments WHERE status IN ('scheduled','confirmed')"
    )
    followup_pids = {r["patient_id"] for r in appt_patients_1} | {r["patient_id"] for r in appt_patients_2}
    patients_with_followup = len(followup_pids & set(latest_per_patient.keys()))

    # Build recent report summaries
    patients_info = {
        r["patient_id"]: r
        for r in _rows("SELECT patient_id, name, age, gender, disease FROM patients")
    }

    recent_reports = []
    for pid, analysis in sorted(latest_per_patient.items(), key=lambda x: x[1]["created_at"] or "", reverse=True)[:20]:
        p = patients_info.get(pid, {})
        confidence = analysis["confidence"] or 0
        risk = _confidence_to_risk(confidence)
        sq = analysis.get("signal_quality") or "Unknown"

        # Pull latest mood/anxiety score for this patient
        mood_rows = _rows(
            "SELECT instrument, score, level FROM assessments "
            "WHERE patient_id=? AND instrument IN ('PHQ9','GAD7') "
            "ORDER BY created_at DESC LIMIT 1",
            (pid,),
        )
        mood_note = ""
        if mood_rows:
            m = mood_rows[0]
            mood_note = _plain_assessment_level(m["instrument"], m["level"], m["score"])

        has_followup = pid in followup_pids

        recent_reports.append({
            "patient_id": pid,
            "patient_name": p.get("name") or "Patient " + pid,
            "age": p.get("age"),
            "gender": p.get("gender") or "Not recorded",
            "condition": (p.get("disease") or analysis.get("disease") or "epilepsy").title(),
            "report_date": (analysis.get("created_at") or "")[:10],
            "signal_quality_plain": _plain_signal_quality(sq),
            "risk_level": risk,
            "risk_color": _risk_color(risk),
            "confidence_pct": round(confidence * 100),
            "mood_note": mood_note,
            "has_upcoming_appointment": has_followup,
            "plain_summary": (
                f"Your brain-wave recording from {(analysis.get('created_at') or '')[:10]} "
                f"has been reviewed by the AI system. {_plain_signal_quality(sq)} "
                f"The system flagged a {risk.lower()} activity level in the recording. "
                f"Your doctor will discuss what this means for you."
            ),
        })

    return {
        "total_reports": len(latest_per_patient),
        "patients_with_followup": patients_with_followup,
        "avg_risk_level": avg_risk,
        "risk_distribution": dict(risk_dist),
        "recent_reports": recent_reports,
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown() -> dict:
    """
    Detailed per-patient breakdown for caregivers and patients.

    Returns
    -------
    dict with keys:
      medication_reminders  – current medication info per patient, in plain language
      biomarker_summaries   – plain-English EEG band-power summaries per patient
      followup_schedule     – upcoming appointments across all patients
      lifestyle_tips        – static seizure-safety tips (always shown)
    """

    # ------------------------------------------------------------------
    # 1. Medication reminders
    # ------------------------------------------------------------------
    med_rows = _rows(
        "SELECT patient_id, drug_name, dose_mg, frequency, "
        "       MAX(log_date) as last_log, "
        "       SUM(CASE WHEN taken='yes' THEN 1 ELSE 0 END) as taken_count, "
        "       COUNT(*) as total_count "
        "FROM medication_adherence "
        "GROUP BY patient_id, drug_name "
        "ORDER BY patient_id, drug_name"
    )

    med_by_patient: dict = defaultdict(list)
    for m in med_rows:
        pid = m["patient_id"]
        total = m["total_count"] or 1
        adherence_pct = round((m["taken_count"] or 0) / total * 100)
        freq_plain = {
            "QD": "once a day", "BID": "twice a day", "TID": "three times a day",
            "QID": "four times a day", "PRN": "as needed",
        }.get((m["frequency"] or "").upper(), m["frequency"] or "as prescribed")

        med_by_patient[pid].append({
            "drug_name": m["drug_name"],
            "dose_mg": m["dose_mg"],
            "frequency": m["frequency"],
            "frequency_plain": freq_plain,
            "last_logged": m["last_log"],
            "adherence_pct": adherence_pct,
            "plain_reminder": (
                f"Take {m['drug_name']} {m['dose_mg']} mg {freq_plain}. "
                f"You took it on time about {adherence_pct}% of the time we tracked. "
                + ("Great job keeping up with your medication!" if adherence_pct >= 85
                   else "Try to take it at the same time each day — skipping doses can trigger seizures.")
            ),
        })

    # Fall back to medications table if no adherence data
    if not med_by_patient:
        fallback_meds = _rows("SELECT patient_id, fields_json FROM medications")
        for r in fallback_meds:
            pid = r["patient_id"]
            try:
                fields = json.loads(r["fields_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                fields = {}
            drug = fields.get("drug_name", "your medication")
            dose = fields.get("dose_mg", "")
            freq = fields.get("frequency", "as prescribed")
            med_by_patient[pid].append({
                "drug_name": drug,
                "dose_mg": dose,
                "frequency": freq,
                "frequency_plain": freq,
                "last_logged": None,
                "adherence_pct": None,
                "plain_reminder": f"Remember to take {drug} {dose} mg {freq} as prescribed by your doctor.",
            })

    medication_reminders = [
        {"patient_id": pid, "medications": meds}
        for pid, meds in sorted(med_by_patient.items())
    ]

    # ------------------------------------------------------------------
    # 2. Biomarker summaries  (from analyses.result_json band powers)
    # ------------------------------------------------------------------
    analyses = _rows(
        "SELECT patient_id, result_json, confidence, signal_quality, created_at "
        "FROM analyses ORDER BY created_at DESC"
    )

    seen_pids: set = set()
    biomarker_summaries = []

    for row in analyses:
        pid = row["patient_id"]
        if pid in seen_pids:
            continue
        seen_pids.add(pid)

        try:
            rj = json.loads(row["result_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            rj = {}

        bands = rj.get("analysis", {}).get("band_power_relative", {})
        delta = bands.get("delta", 0)
        theta = bands.get("theta", 0)
        alpha = bands.get("alpha", 0)
        beta = bands.get("beta", 0)
        gamma = bands.get("gamma", 0)

        # Plain-language interpretation of dominant brain rhythm
        dominant = max(bands, key=bands.get) if bands else None
        dominant_plain = {
            "delta": "slow, deep rhythms (delta) — common during deep sleep or in some brain conditions",
            "theta": "slow-medium rhythms (theta) — often seen during drowsiness or certain brain patterns",
            "alpha": "relaxed rhythms (alpha) — typical when you are calm and resting",
            "beta": "fast rhythms (beta) — typical when you are awake and alert",
            "gamma": "very fast rhythms (gamma) — associated with focused thinking",
        }.get(dominant, "mixed brain rhythms")

        biomarker_summaries.append({
            "patient_id": pid,
            "recording_date": (row.get("created_at") or "")[:10],
            "signal_quality": row.get("signal_quality") or "Unknown",
            "band_powers": {
                "delta_pct": round(delta * 100, 1),
                "theta_pct": round(theta * 100, 1),
                "alpha_pct": round(alpha * 100, 1),
                "beta_pct": round(beta * 100, 1),
                "gamma_pct": round(gamma * 100, 1),
            },
            "dominant_rhythm": dominant,
            "plain_biomarker": (
                f"Your brain-wave recording mainly showed {dominant_plain}. "
                f"This is one of many things your care team will look at. "
                f"Your doctor will discuss what these patterns mean for you."
            ),
            "ai_confidence_pct": round((row.get("confidence") or 0) * 100),
            "disclaimer": (
                "These brain-wave measurements are produced by a computer program. "
                "They are NOT a diagnosis. Only your doctor can tell you what they mean for your health."
            ),
        })

    # ------------------------------------------------------------------
    # 3. Follow-up schedule
    # ------------------------------------------------------------------
    # Use patient_appointments (richer data) first, then appointments
    appt_rows = _rows(
        "SELECT patient_id, appointment_type, provider_name, "
        "       appointment_date, appointment_time, status, location, notes "
        "FROM patient_appointments "
        "WHERE status IN ('scheduled','confirmed','rescheduled') "
        "ORDER BY appointment_date ASC"
    )

    if not appt_rows:
        appt_rows = _rows(
            "SELECT patient_id, appt_type as appointment_type, provider as provider_name, "
            "       scheduled_for as appointment_date, '' as appointment_time, "
            "       status, department as location, notes "
            "FROM appointments "
            "WHERE status IN ('booked','confirmed') "
            "ORDER BY scheduled_for ASC"
        )

    followup_schedule = []
    for a in appt_rows:
        appt_type = a.get("appointment_type") or "Appointment"
        provider = a.get("provider_name") or "your care team"
        date = (a.get("appointment_date") or "")[:10]
        time = a.get("appointment_time") or ""
        location = a.get("location") or ""
        status = (a.get("status") or "").capitalize()

        followup_schedule.append({
            "patient_id": a["patient_id"],
            "appointment_type": appt_type,
            "provider": provider,
            "date": date,
            "time": time,
            "location": location,
            "status": status,
            "plain_reminder": (
                f"You have a {appt_type} with {provider} on {date}"
                + (f" at {time}" if time else "")
                + (f" at {location}" if location else "")
                + ". Please don't forget this appointment."
            ),
        })

    # ------------------------------------------------------------------
    # 4. Static lifestyle & seizure-safety tips
    # ------------------------------------------------------------------
    lifestyle_tips = [
        {
            "category": "Medication",
            "tip": "Take your anti-seizure medication at the same time every day. Missing doses is one of the most common reasons seizures come back.",
            "icon": "pill",
        },
        {
            "category": "Sleep",
            "tip": "Aim for 7–9 hours of sleep every night. Lack of sleep is a common seizure trigger.",
            "icon": "moon",
        },
        {
            "category": "Stress",
            "tip": "Try to manage stress with deep breathing, gentle exercise, or talking to someone you trust. High stress can lower your seizure threshold.",
            "icon": "heart",
        },
        {
            "category": "Alcohol & Caffeine",
            "tip": "Limit or avoid alcohol. Even small amounts can lower your seizure threshold. Drink caffeine in moderation.",
            "icon": "no-drink",
        },
        {
            "category": "Safety at Home",
            "tip": "Use a shower instead of a bath when possible, and keep the bathroom door unlocked. Swim only with a buddy who knows about your condition.",
            "icon": "shield",
        },
        {
            "category": "Driving",
            "tip": "Check your state or country's rules about driving with epilepsy. Many places require a seizure-free period before you can drive.",
            "icon": "car",
        },
        {
            "category": "Emergency Plan",
            "tip": "Make sure a family member or caregiver knows what to do during a seizure — do not hold the person down, time the seizure, and call 911 if it lasts more than 5 minutes.",
            "icon": "alert",
        },
        {
            "category": "Seizure Diary",
            "tip": "Keep a record of when seizures happen, how long they last, and any triggers you notice. This helps your doctor adjust your care plan.",
            "icon": "notebook",
        },
        {
            "category": "Follow-Up Visits",
            "tip": "Keep all scheduled appointments with your neurologist — even when you feel well. Regular check-ups help your doctor catch changes early.",
            "icon": "calendar",
        },
        {
            "category": "Rescue Medication",
            "tip": "If you have been prescribed a rescue medication (such as diazepam nasal spray), keep it with you at all times and make sure your caregiver knows how to use it.",
            "icon": "syringe",
        },
    ]

    return {
        "medication_reminders": medication_reminders,
        "biomarker_summaries": biomarker_summaries,
        "followup_schedule": followup_schedule,
        "lifestyle_tips": lifestyle_tips,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions() -> dict:
    """
    Plain-language glossary and disclaimer for the patient-facing dashboard.

    Returns a single dict that can be serialised directly to JSON for the
    frontend Definitions tab.
    """
    return {
        "title": "Understanding Your Report — Plain Language Guide",
        "intro": (
            "This report was created by a computer program that looked at your brain-wave (EEG) "
            "recording. It is meant to help you and your family understand what happened, remind "
            "you about medications and appointments, and prepare questions for your doctor. "
            "It does NOT replace a consultation with your doctor."
        ),
        "sections": [
            {
                "heading": "What Is an EEG?",
                "body": (
                    "An EEG (electroencephalogram) records the electrical activity of your brain "
                    "through small sensors placed on your scalp. It does not hurt. The recording "
                    "shows patterns of brain activity that your care team can review for signs of "
                    "epilepsy or other conditions."
                ),
            },
            {
                "heading": "What Is the AI System Doing?",
                "body": (
                    "The AI is a computer program trained on thousands of EEG recordings. It looks "
                    "for patterns that have been seen in people with epilepsy. It gives a score (the "
                    "confidence percentage) that says how strongly the pattern in your recording "
                    "matches those it has seen before. A high score does NOT mean you definitely "
                    "have epilepsy — and a low score does NOT mean you are clear. Only your doctor "
                    "can interpret results in the context of your full medical history."
                ),
            },
            {
                "heading": "What Does the Risk Level Mean?",
                "body": (
                    "The risk level (Low / Moderate / High) is a simple label the system assigns "
                    "based on the AI confidence score:\n"
                    "  • Low — the pattern did not look very similar to epilepsy patterns in the AI's training data.\n"
                    "  • Moderate — the pattern had some similarity to epilepsy patterns.\n"
                    "  • High — the pattern looked very similar to epilepsy patterns.\n"
                    "This label is a guide for your doctor, not a diagnosis."
                ),
            },
            {
                "heading": "What Are Brain-Wave Rhythms?",
                "body": (
                    "Your brain produces electrical signals at different speeds (measured in Hz — "
                    "cycles per second). Scientists group them into bands:\n"
                    "  • Delta (0.5–4 Hz) — very slow waves, seen in deep sleep.\n"
                    "  • Theta (4–8 Hz) — slow waves, seen in drowsiness or some brain conditions.\n"
                    "  • Alpha (8–13 Hz) — relaxed, resting waves.\n"
                    "  • Beta (13–30 Hz) — alert, thinking waves.\n"
                    "  • Gamma (30–100 Hz) — high-frequency waves linked to focused thought.\n"
                    "The proportion of each band in your recording is one signal your care team uses."
                ),
            },
            {
                "heading": "Medication Adherence",
                "body": (
                    "Adherence means 'taking your medication as prescribed'. The dashboard shows "
                    "what percentage of your logged doses you took on time. Missing or delaying "
                    "anti-seizure medications is one of the most common reasons seizures return. "
                    "If you are having trouble remembering or affording your medication, tell your "
                    "pharmacist or nurse — there are often programmes that can help."
                ),
            },
            {
                "heading": "Mood and Anxiety Questionnaires",
                "body": (
                    "PHQ-9 (Patient Health Questionnaire-9) measures depression symptoms. "
                    "GAD-7 (Generalised Anxiety Disorder-7) measures anxiety symptoms. "
                    "NDDIE (Neurological Disorders Depression Inventory for Epilepsy) screens "
                    "for depression in people with epilepsy. These are NOT diagnoses — they are "
                    "screening tools that help your doctor decide whether to ask more questions."
                ),
            },
            {
                "heading": "Quality-of-Life Score (QOLIE-31)",
                "body": (
                    "QOLIE-31 is a questionnaire designed for people with epilepsy. It asks about "
                    "how seizures, medications, and the condition affect your daily life, work, "
                    "driving, emotions, and memory. A higher score means better quality of life. "
                    "Your doctor uses changes in this score over time to see whether treatment is helping."
                ),
            },
            {
                "heading": "Follow-Up Appointments",
                "body": (
                    "Your follow-up appointments are listed in the dashboard. Please attend even "
                    "when you feel well — your neurologist needs regular visits to adjust your "
                    "medication, review new recordings, and check for side effects. If you cannot "
                    "attend, call the clinic to reschedule rather than skipping."
                ),
            },
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalogram — a painless recording of your brain's electrical activity."},
            {"term": "Epilepsy", "definition": "A brain condition in which the electrical signals sometimes fire in an unusual pattern, causing seizures."},
            {"term": "Seizure", "definition": "A sudden burst of unusual electrical activity in the brain that can cause shaking, staring, loss of awareness, or other symptoms."},
            {"term": "Anti-seizure medication (ASM)", "definition": "Medicine prescribed to reduce how often or how severe seizures are. Also called anti-epileptic drugs (AEDs)."},
            {"term": "Confidence score", "definition": "A number (0–100%) that shows how closely your recording matches the AI's training examples. Higher does NOT mean 'definitely epilepsy'."},
            {"term": "Risk level", "definition": "A simplified Low / Moderate / High label based on the AI confidence score. This is a guide for your doctor."},
            {"term": "Signal quality", "definition": "How clearly the EEG electrodes recorded your brain signals. Good quality means less interference from movement or other sources."},
            {"term": "PHQ-9", "definition": "A 9-question mood questionnaire that screens for signs of depression."},
            {"term": "GAD-7", "definition": "A 7-question questionnaire that screens for signs of anxiety."},
            {"term": "QOLIE-31", "definition": "A 31-item quality-of-life questionnaire designed specifically for people with epilepsy."},
            {"term": "NDDIE", "definition": "A 6-item questionnaire that screens for depression in people with epilepsy."},
            {"term": "MoCA", "definition": "Montreal Cognitive Assessment — a short test of thinking and memory."},
            {"term": "Delta waves", "definition": "Very slow brain waves (0.5–4 Hz) — normal in deep sleep, sometimes elevated in some brain conditions."},
            {"term": "Theta waves", "definition": "Slow brain waves (4–8 Hz) — normal in drowsiness."},
            {"term": "Alpha waves", "definition": "Medium brain waves (8–13 Hz) — typical when relaxed and eyes closed."},
            {"term": "Beta waves", "definition": "Faster brain waves (13–30 Hz) — typical when alert and thinking."},
            {"term": "Gamma waves", "definition": "Very fast brain waves (30–100 Hz) — associated with focused mental activity."},
            {"term": "Rescue medication", "definition": "A fast-acting medication (such as diazepam nasal spray or diazepam rectal gel) used to stop a prolonged seizure."},
            {"term": "Neurologist", "definition": "A doctor who specialises in conditions of the brain and nervous system, including epilepsy."},
            {"term": "Caregiver", "definition": "A family member, friend, or professional who helps care for someone with a health condition."},
        ],
        "disclaimer": {
            "heading": "Important Notice",
            "body": (
                "This report is generated by an artificial intelligence (AI) system and is intended "
                "for informational purposes only. It does NOT constitute a medical diagnosis, "
                "medical advice, or a substitute for professional medical care.\n\n"
                "The AI system has not been validated as a standalone diagnostic device. Results "
                "must always be interpreted by a qualified healthcare professional who knows your "
                "full medical history.\n\n"
                "If you are experiencing a seizure or a medical emergency, call 911 (or your local "
                "emergency number) immediately.\n\n"
                "If you have questions about your results, please contact your neurologist or the "
                "clinic where your EEG was performed."
            ),
            "emergency_note": "IN AN EMERGENCY — CALL 911 IMMEDIATELY. Do not rely on this report.",
        },
        "pipeline_step": "Step 22 — Final Output: Patient-Facing Report",
        "report_language": "Plain English (8th-grade reading level target)",
    }
