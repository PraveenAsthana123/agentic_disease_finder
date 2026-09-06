"""
Neuro AI Ecosystem -- Caregiver & Emergency Contact Dashboard
=============================================================
Tracks caregivers and emergency contacts for epilepsy patients, including
training status, burnout metrics, availability, and emergency protocols.

Caregiver Roles:
  spouse              -- Spouse or domestic partner providing daily care
  parent              -- Parent (mother or father) of the patient
  sibling             -- Brother or sister of the patient
  friend              -- Close friend assisting with seizure response
  neighbor            -- Nearby resident available for emergency response
  professional caretaker -- Hired professional caregiver or aide

Availability:
  full-time           -- Available 24/7 for patient care
  on-call             -- Available for emergencies but not continuously present
  part-time           -- Available during scheduled hours only
  weekends            -- Available on weekends only

All caregiver records are stored in the caregivers table and emergency
contact records in the emergency_contacts table of clinical.db.

Author: Research Team
"""

import sqlite3
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _json_safe(obj):
    """Convert numpy/date types to JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _parse_training_topics(topics_str):
    """Safely parse the training_topics JSON column."""
    if not topics_str:
        return []
    try:
        result = json.loads(topics_str)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def overview(patient_id: Optional[str] = None) -> dict:
    """
    Caregiver & emergency contact overview dashboard data.

    Returns:
        total_caregivers              -- total caregiver records
        total_emergency_contacts      -- total emergency contact records
        total_patients                -- distinct patients
        epilepsy_training_rate        -- % of caregivers with epilepsy training completed
        first_aid_certified_rate      -- % of caregivers with first aid certification
        rescue_med_trained_rate       -- % of caregivers trained in rescue medication
        avg_seizure_first_aid_confidence -- mean confidence score (1-10)
        avg_burnout_score             -- mean burnout score (0-100)
        avg_caregiver_stress          -- mean stress score (1-10)
        avg_sleep_quality             -- mean sleep quality score (1-10)
        safety_plan_rate              -- % of caregivers with safety plan
        seizure_action_plan_rate      -- % of caregivers with seizure action plan
        role_distribution             -- list of {name, value} for chart
        availability_distribution     -- list of {name, value} for chart
        relationship_distribution     -- list of {name, value} for emergency contacts
        training_topic_frequency      -- list of {name, value} for bar chart
        burnout_distribution          -- list of {name, value} for binned bar chart
    """
    conn = _conn()
    c = conn.cursor()

    # Load caregivers
    if patient_id:
        c.execute("SELECT * FROM caregivers WHERE patient_id=? ORDER BY created_at DESC", (patient_id,))
    else:
        c.execute("SELECT * FROM caregivers ORDER BY created_at DESC")
    cg_cols = [desc[0] for desc in c.description]
    cg_rows = [dict(zip(cg_cols, row)) for row in c.fetchall()]

    # Load emergency contacts
    if patient_id:
        c.execute("SELECT * FROM emergency_contacts WHERE patient_id=? ORDER BY created_at DESC", (patient_id,))
    else:
        c.execute("SELECT * FROM emergency_contacts ORDER BY created_at DESC")
    ec_cols = [desc[0] for desc in c.description]
    ec_rows = [dict(zip(ec_cols, row)) for row in c.fetchall()]
    conn.close()

    if not cg_rows and not ec_rows:
        return _json_safe({
            "total_caregivers": 0,
            "total_emergency_contacts": 0,
            "total_patients": 0,
            "epilepsy_training_rate": 0.0,
            "first_aid_certified_rate": 0.0,
            "rescue_med_trained_rate": 0.0,
            "avg_seizure_first_aid_confidence": 0.0,
            "avg_burnout_score": 0.0,
            "avg_caregiver_stress": 0.0,
            "avg_sleep_quality": 0.0,
            "safety_plan_rate": 0.0,
            "seizure_action_plan_rate": 0.0,
            "role_distribution": [],
            "availability_distribution": [],
            "relationship_distribution": [],
            "training_topic_frequency": [],
            "burnout_distribution": [],
        })

    total_cg = len(cg_rows)
    total_ec = len(ec_rows)
    all_pids = set()
    for r in cg_rows:
        if r.get("patient_id"):
            all_pids.add(r["patient_id"])
    for r in ec_rows:
        if r.get("patient_id"):
            all_pids.add(r["patient_id"])
    total_patients = len(all_pids)

    # KPIs from caregivers
    if total_cg > 0:
        epilepsy_training_rate = round(sum(1 for r in cg_rows if r.get("epilepsy_training_completed")) / total_cg * 100, 1)
        first_aid_certified_rate = round(sum(1 for r in cg_rows if r.get("first_aid_certified")) / total_cg * 100, 1)
        rescue_med_trained_rate = round(sum(1 for r in cg_rows if r.get("rescue_med_trained")) / total_cg * 100, 1)
        safety_plan_rate = round(sum(1 for r in cg_rows if r.get("safety_plan_exists")) / total_cg * 100, 1)
        seizure_action_plan_rate = round(sum(1 for r in cg_rows if r.get("seizure_action_plan_exists")) / total_cg * 100, 1)

        conf_vals = [r["seizure_first_aid_confidence"] for r in cg_rows if r.get("seizure_first_aid_confidence") is not None]
        avg_seizure_first_aid_confidence = round(sum(conf_vals) / len(conf_vals), 1) if conf_vals else 0.0

        burnout_vals = [r["burnout_score"] for r in cg_rows if r.get("burnout_score") is not None]
        avg_burnout_score = round(sum(burnout_vals) / len(burnout_vals), 1) if burnout_vals else 0.0

        stress_vals = [r["caregiver_stress"] for r in cg_rows if r.get("caregiver_stress") is not None]
        avg_caregiver_stress = round(sum(stress_vals) / len(stress_vals), 1) if stress_vals else 0.0

        sleep_vals = [r["caregiver_sleep_quality"] for r in cg_rows if r.get("caregiver_sleep_quality") is not None]
        avg_sleep_quality = round(sum(sleep_vals) / len(sleep_vals), 1) if sleep_vals else 0.0
    else:
        epilepsy_training_rate = 0.0
        first_aid_certified_rate = 0.0
        rescue_med_trained_rate = 0.0
        safety_plan_rate = 0.0
        seizure_action_plan_rate = 0.0
        avg_seizure_first_aid_confidence = 0.0
        avg_burnout_score = 0.0
        avg_caregiver_stress = 0.0
        avg_sleep_quality = 0.0

    # Role distribution
    role_counts = defaultdict(int)
    for r in cg_rows:
        role_counts[r.get("role", "unknown")] += 1
    role_distribution = [
        {"name": k, "value": v}
        for k, v in sorted(role_counts.items(), key=lambda x: -x[1])
    ]

    # Availability distribution
    avail_counts = defaultdict(int)
    for r in cg_rows:
        avail_counts[r.get("availability", "unknown")] += 1
    availability_distribution = [
        {"name": k, "value": v}
        for k, v in sorted(avail_counts.items(), key=lambda x: -x[1])
    ]

    # Relationship distribution (emergency contacts)
    rel_counts = defaultdict(int)
    for r in ec_rows:
        rel_counts[r.get("relationship", "unknown")] += 1
    relationship_distribution = [
        {"name": k, "value": v}
        for k, v in sorted(rel_counts.items(), key=lambda x: -x[1])
    ]

    # Training topic frequency
    topic_counts = defaultdict(int)
    for r in cg_rows:
        topics = _parse_training_topics(r.get("training_topics"))
        for t in topics:
            topic_counts[t] += 1
    training_topic_frequency = [
        {"name": k, "value": v}
        for k, v in sorted(topic_counts.items(), key=lambda x: -x[1])
    ]

    # Burnout score distribution (binned)
    bins = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for r in cg_rows:
        score = r.get("burnout_score")
        if score is not None:
            if score <= 20:
                bins["0-20"] += 1
            elif score <= 40:
                bins["21-40"] += 1
            elif score <= 60:
                bins["41-60"] += 1
            elif score <= 80:
                bins["61-80"] += 1
            else:
                bins["81-100"] += 1
    burnout_distribution = [{"name": k, "value": v} for k, v in bins.items()]

    return _json_safe({
        "total_caregivers":                total_cg,
        "total_emergency_contacts":        total_ec,
        "total_patients":                  total_patients,
        "epilepsy_training_rate":          epilepsy_training_rate,
        "first_aid_certified_rate":        first_aid_certified_rate,
        "rescue_med_trained_rate":         rescue_med_trained_rate,
        "avg_seizure_first_aid_confidence": avg_seizure_first_aid_confidence,
        "avg_burnout_score":               avg_burnout_score,
        "avg_caregiver_stress":            avg_caregiver_stress,
        "avg_sleep_quality":               avg_sleep_quality,
        "safety_plan_rate":                safety_plan_rate,
        "seizure_action_plan_rate":        seizure_action_plan_rate,
        "role_distribution":               role_distribution,
        "availability_distribution":       availability_distribution,
        "relationship_distribution":       relationship_distribution,
        "training_topic_frequency":        training_topic_frequency,
        "burnout_distribution":            burnout_distribution,
    })


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed caregiver & emergency contact breakdown.

    Returns:
        all_caregivers          -- list of all caregiver details
        all_emergency_contacts  -- list of all emergency contact details
        high_burnout_caregivers -- caregivers with burnout_score >= 70
        untrained_caregivers    -- caregivers with epilepsy_training_completed = 0
        stale_contacts          -- emergency contacts with last_verified > 1 year ago
        patient_summary         -- per-patient caregiver + emergency contact summary
    """
    conn = _conn()
    c = conn.cursor()

    # Load caregivers
    if patient_id:
        c.execute("SELECT * FROM caregivers WHERE patient_id=? ORDER BY patient_id, name", (patient_id,))
    else:
        c.execute("SELECT * FROM caregivers ORDER BY patient_id, name")
    cg_cols = [desc[0] for desc in c.description]
    cg_rows = [dict(zip(cg_cols, row)) for row in c.fetchall()]

    # Load emergency contacts
    if patient_id:
        c.execute("SELECT * FROM emergency_contacts WHERE patient_id=? ORDER BY patient_id, contact_name", (patient_id,))
    else:
        c.execute("SELECT * FROM emergency_contacts ORDER BY patient_id, contact_name")
    ec_cols = [desc[0] for desc in c.description]
    ec_rows = [dict(zip(ec_cols, row)) for row in c.fetchall()]
    conn.close()

    if not cg_rows and not ec_rows:
        return _json_safe({
            "all_caregivers": [],
            "all_emergency_contacts": [],
            "high_burnout_caregivers": [],
            "untrained_caregivers": [],
            "stale_contacts": [],
            "patient_summary": [],
        })

    # All caregivers (formatted)
    all_caregivers = []
    for r in cg_rows:
        topics = _parse_training_topics(r.get("training_topics"))
        all_caregivers.append({
            "id":                          r.get("id"),
            "patient_id":                  r.get("patient_id"),
            "name":                        r.get("name"),
            "role":                        r.get("role"),
            "availability":                r.get("availability"),
            "experience_years":            r.get("experience_years"),
            "epilepsy_training_completed": bool(r.get("epilepsy_training_completed")),
            "training_topics":             topics,
            "first_aid_certified":         bool(r.get("first_aid_certified")),
            "rescue_med_trained":          bool(r.get("rescue_med_trained")),
            "seizure_first_aid_confidence": r.get("seizure_first_aid_confidence"),
            "caregiver_stress":            r.get("caregiver_stress"),
            "caregiver_sleep_quality":     r.get("caregiver_sleep_quality"),
            "work_impact":                 r.get("work_impact"),
            "burnout_score":               r.get("burnout_score"),
            "last_respite_date":           r.get("last_respite_date"),
            "safety_plan_exists":          bool(r.get("safety_plan_exists")),
            "seizure_action_plan_exists":  bool(r.get("seizure_action_plan_exists")),
            "emergency_protocol":          r.get("emergency_protocol", ""),
            "when_to_call_911":            r.get("when_to_call_911", ""),
            "notes":                       r.get("notes", ""),
            "created_at":                  r.get("created_at"),
        })

    # All emergency contacts (formatted)
    all_emergency_contacts = []
    for r in ec_rows:
        all_emergency_contacts.append({
            "id":                r.get("id"),
            "patient_id":       r.get("patient_id"),
            "contact_name":     r.get("contact_name"),
            "phone":            r.get("phone"),
            "email":            r.get("email"),
            "relationship":     r.get("relationship"),
            "is_primary":       bool(r.get("is_primary")),
            "notify_on_seizure": bool(r.get("notify_on_seizure")),
            "last_verified":    r.get("last_verified"),
            "created_at":       r.get("created_at"),
        })

    # High burnout caregivers (burnout_score >= 70)
    high_burnout_caregivers = []
    for r in cg_rows:
        score = r.get("burnout_score")
        if score is not None and score >= 70:
            high_burnout_caregivers.append({
                "name":                    r.get("name"),
                "patient_id":              r.get("patient_id"),
                "role":                    r.get("role"),
                "burnout_score":           score,
                "caregiver_stress":        r.get("caregiver_stress"),
                "caregiver_sleep_quality": r.get("caregiver_sleep_quality"),
                "work_impact":             r.get("work_impact"),
            })
    high_burnout_caregivers.sort(key=lambda x: -(x.get("burnout_score") or 0))

    # Untrained caregivers (epilepsy_training_completed = 0)
    untrained_caregivers = []
    for r in cg_rows:
        if not r.get("epilepsy_training_completed"):
            untrained_caregivers.append({
                "name":                r.get("name"),
                "patient_id":          r.get("patient_id"),
                "role":                r.get("role"),
                "first_aid_certified": bool(r.get("first_aid_certified")),
                "rescue_med_trained":  bool(r.get("rescue_med_trained")),
            })

    # Stale emergency contacts (last_verified > 1 year ago)
    now = datetime.now()
    one_year_ago = now - timedelta(days=365)
    stale_contacts = []
    for r in ec_rows:
        lv = r.get("last_verified")
        if lv:
            try:
                lv_date = datetime.strptime(lv[:10], "%Y-%m-%d")
                if lv_date < one_year_ago:
                    stale_contacts.append({
                        "contact_name":  r.get("contact_name"),
                        "patient_id":    r.get("patient_id"),
                        "phone":         r.get("phone"),
                        "relationship":  r.get("relationship"),
                        "last_verified": lv,
                        "is_primary":    bool(r.get("is_primary")),
                    })
            except (ValueError, TypeError):
                pass
    stale_contacts.sort(key=lambda x: x.get("last_verified", ""))

    # Per-patient summary
    patient_data = defaultdict(lambda: {
        "patient_id": "",
        "caregiver_count": 0,
        "emergency_contact_count": 0,
        "caregivers": [],
        "has_trained_caregiver": False,
        "has_primary_contact": False,
        "avg_burnout": 0.0,
        "burnout_sum": 0.0,
        "burnout_n": 0,
    })
    for r in cg_rows:
        pid = r.get("patient_id", "")
        patient_data[pid]["patient_id"] = pid
        patient_data[pid]["caregiver_count"] += 1
        patient_data[pid]["caregivers"].append(r.get("name", ""))
        if r.get("epilepsy_training_completed"):
            patient_data[pid]["has_trained_caregiver"] = True
        bs = r.get("burnout_score")
        if bs is not None:
            patient_data[pid]["burnout_sum"] += bs
            patient_data[pid]["burnout_n"] += 1
    for r in ec_rows:
        pid = r.get("patient_id", "")
        patient_data[pid]["patient_id"] = pid
        patient_data[pid]["emergency_contact_count"] += 1
        if r.get("is_primary"):
            patient_data[pid]["has_primary_contact"] = True

    patient_summary = []
    for ps in sorted(patient_data.values(), key=lambda x: x["patient_id"]):
        avg_b = round(ps["burnout_sum"] / ps["burnout_n"], 1) if ps["burnout_n"] else 0.0
        patient_summary.append({
            "patient_id":             ps["patient_id"],
            "caregiver_count":        ps["caregiver_count"],
            "emergency_contact_count": ps["emergency_contact_count"],
            "caregivers":             ps["caregivers"],
            "has_trained_caregiver":  ps["has_trained_caregiver"],
            "has_primary_contact":    ps["has_primary_contact"],
            "avg_burnout":            avg_b,
        })

    return _json_safe({
        "all_caregivers":          all_caregivers,
        "all_emergency_contacts":  all_emergency_contacts,
        "high_burnout_caregivers": high_burnout_caregivers,
        "untrained_caregivers":    untrained_caregivers,
        "stale_contacts":          stale_contacts,
        "patient_summary":         patient_summary,
    })


def definitions() -> dict:
    """
    Caregiver & emergency contact definitions, terminology, and clinical context.

    Returns:
        role_descriptions             -- descriptions of each caregiver role
        availability_descriptions     -- descriptions of each availability type
        training_topic_descriptions   -- descriptions of training topics
        score_descriptions            -- descriptions of numeric scores
        emergency_protocol_glossary   -- emergency protocol terminology
        clinical_notes                -- clinical context notes about caregiver support
        glossary                      -- list of {term, definition}
    """
    return {
        "role_descriptions": [
            {"role": "spouse", "description": (
                "Spouse or domestic partner providing daily epilepsy care. Typically the "
                "most consistently available caregiver with intimate knowledge of the "
                "patient's seizure patterns, medication schedule, and postictal behaviour. "
                "Spouses often bear the highest emotional and practical caregiving burden."
            )},
            {"role": "parent", "description": (
                "Parent (mother or father) of the patient. Parental caregivers are "
                "especially common in paediatric and young-adult epilepsy. They often "
                "serve as primary medical decision-makers and medication administrators, "
                "and may experience significant anxiety around seizure occurrence."
            )},
            {"role": "sibling", "description": (
                "Brother or sister of the patient. Sibling caregivers may provide "
                "supplementary support, respite care, or serve as backup emergency "
                "responders. They may require specific training on seizure first aid "
                "and rescue medication administration."
            )},
            {"role": "friend", "description": (
                "Close friend who assists with seizure response and monitoring. Friends "
                "may accompany the patient during social activities and serve as "
                "secondary emergency contacts. Training in seizure recognition and "
                "first aid is especially important for non-family caregivers."
            )},
            {"role": "neighbor", "description": (
                "Nearby resident available for emergency response, particularly "
                "valuable when the patient lives alone. Neighbors may be the first "
                "responders in emergency situations and should be familiar with the "
                "patient's seizure action plan and when to call emergency services."
            )},
            {"role": "professional caretaker", "description": (
                "Hired professional caregiver, home health aide, or personal care "
                "assistant. Professional caretakers typically have formal medical "
                "training and may be required to complete epilepsy-specific "
                "certification. They provide structured, scheduled caregiving support."
            )},
        ],
        "availability_descriptions": [
            {"availability": "full-time", "description": (
                "Caregiver is available 24/7 for patient care and seizure response. "
                "Full-time availability is associated with higher caregiver burnout risk "
                "and greater need for respite care services."
            )},
            {"availability": "on-call", "description": (
                "Caregiver is available for emergency seizure response but not "
                "continuously present with the patient. On-call caregivers should "
                "maintain a response time of under 15 minutes and keep rescue "
                "medication accessible."
            )},
            {"availability": "part-time", "description": (
                "Caregiver is available during scheduled hours only, typically "
                "aligned with the patient's highest-risk periods (e.g., mornings "
                "when medication levels are lowest, or overnight for nocturnal seizures)."
            )},
            {"availability": "weekends", "description": (
                "Caregiver is available on weekends only, supplementing weekday care "
                "provided by other caregivers or professional services. Weekend "
                "caregivers often provide respite for primary full-time caregivers."
            )},
        ],
        "training_topic_descriptions": [
            {"topic": "Seizure recognition", "description": (
                "Identifying different seizure types (tonic-clonic, absence, focal, "
                "myoclonic) and distinguishing seizures from non-epileptic events."
            )},
            {"topic": "Recovery position", "description": (
                "Placing the patient in the lateral recovery position post-seizure to "
                "maintain airway patency and prevent aspiration."
            )},
            {"topic": "Rescue medication administration", "description": (
                "Administering emergency benzodiazepines (intranasal midazolam, "
                "buccal midazolam, or rectal diazepam) during prolonged seizures "
                "or seizure clusters."
            )},
            {"topic": "Timing seizures", "description": (
                "Accurately timing seizure duration using a clock or stopwatch, "
                "critical for determining when to administer rescue medication "
                "or call emergency services (typically at the 5-minute mark)."
            )},
            {"topic": "When to call 911", "description": (
                "Recognising emergency situations requiring ambulance activation: "
                "seizures lasting >5 minutes, status epilepticus, injury, first "
                "seizure, seizures in water, or failure of rescue medication."
            )},
            {"topic": "SUDEP awareness", "description": (
                "Understanding Sudden Unexpected Death in Epilepsy risk factors, "
                "particularly nocturnal tonic-clonic seizures, and the importance "
                "of seizure monitoring and supervision during sleep."
            )},
            {"topic": "Medication management", "description": (
                "Managing anti-seizure medication schedules, recognising missed "
                "doses, understanding drug interactions, and monitoring for side effects."
            )},
            {"topic": "Safety hazard assessment", "description": (
                "Identifying and mitigating environmental hazards in the home: "
                "sharp furniture edges, hot surfaces, unsupervised bathing, heights, "
                "and driving eligibility considerations."
            )},
            {"topic": "Emotional support techniques", "description": (
                "Providing psychological support during postictal confusion, managing "
                "anxiety around seizure occurrence, and supporting the patient's "
                "mental health and self-management confidence."
            )},
            {"topic": "Emergency action plan review", "description": (
                "Regular review and rehearsal of the patient's seizure action plan, "
                "ensuring all caregivers know their roles, rescue medication locations, "
                "and emergency contact procedures."
            )},
            {"topic": "Epilepsy triggers", "description": (
                "Understanding common seizure triggers including sleep deprivation, "
                "stress, alcohol, photosensitivity, missed medication, illness, and "
                "hormonal changes, and strategies to minimise trigger exposure."
            )},
        ],
        "score_descriptions": [
            {"score": "burnout_score", "range": "0-100", "description": (
                "Composite caregiver burnout index derived from emotional exhaustion, "
                "depersonalisation, and reduced personal accomplishment subscales. "
                "Scores >= 70 indicate high burnout requiring clinical intervention "
                "and respite care referral."
            )},
            {"score": "seizure_first_aid_confidence", "range": "1-10", "description": (
                "Self-rated confidence in managing a seizure emergency, from 1 (not at "
                "all confident) to 10 (extremely confident). Scores below 5 suggest "
                "the caregiver would benefit from hands-on seizure first aid training."
            )},
            {"score": "caregiver_stress", "range": "1-10", "description": (
                "Self-rated stress level, from 1 (minimal stress) to 10 (extreme "
                "stress). Chronic stress scores above 7 are associated with increased "
                "risk of caregiver depression and reduced quality of care."
            )},
            {"score": "caregiver_sleep_quality", "range": "1-10", "description": (
                "Self-rated sleep quality, from 1 (very poor) to 10 (excellent). "
                "Low sleep quality in caregivers is often linked to nocturnal seizure "
                "monitoring duties and hypervigilance during sleep."
            )},
            {"score": "work_impact", "range": "1-10", "description": (
                "Self-rated impact of caregiving on work/employment, from 1 (no impact) "
                "to 10 (severe impact, unable to work). High work impact scores indicate "
                "need for flexible work arrangements or professional caregiving support."
            )},
        ],
        "emergency_protocol_glossary": [
            {"term": "Seizure Action Plan (SAP)", "description": (
                "A written, individualized document outlining step-by-step instructions "
                "for responding to a patient's seizures, including seizure types, "
                "duration thresholds, rescue medication instructions, and when to call "
                "emergency services. Recommended by the Epilepsy Foundation for all "
                "patients with active epilepsy."
            )},
            {"term": "Safety Plan", "description": (
                "A comprehensive home and environmental safety assessment document "
                "identifying seizure-related hazards and mitigation strategies. "
                "Covers bathing, cooking, heights, driving, swimming, and workplace safety."
            )},
            {"term": "Emergency Protocol", "description": (
                "The specific sequence of actions a caregiver should follow during a "
                "seizure emergency: secure the patient, time the seizure, administer "
                "rescue medication if trained, call emergency services if criteria met, "
                "and document the event."
            )},
            {"term": "When to Call 911 Criteria", "description": (
                "Evidence-based criteria for activating emergency medical services: "
                "seizure duration >5 minutes, status epilepticus, injury during seizure, "
                "seizure in water, first seizure, breathing difficulties post-ictal, "
                "rescue medication failure, or patient request."
            )},
        ],
        "clinical_notes": [
            (
                "Caregiver burnout affects 30-50% of epilepsy caregivers, with higher "
                "rates among those caring for patients with drug-resistant epilepsy or "
                "frequent tonic-clonic seizures. Burnout is associated with reduced quality "
                "of care, increased hospitalisation rates, and caregiver depression "
                "(Lv et al., Epilepsy & Behavior 2009)."
            ),
            (
                "Seizure first aid training for caregivers significantly reduces "
                "seizure-related injuries and unnecessary emergency department visits. "
                "The Epilepsy Foundation's Seizure First Aid certification program has "
                "been validated to improve caregiver confidence and response accuracy "
                "(Sauro et al., Seizure 2016)."
            ),
            (
                "Rescue medication training is critical: intranasal midazolam and buccal "
                "midazolam have replaced rectal diazepam as preferred rescue medications "
                "due to ease of administration by non-medical caregivers. FDA-approved "
                "nasal spray formulations (Nayzilam) enable caregiver administration "
                "without specialised training equipment."
            ),
            (
                "Emergency contact verification should occur at least annually. Stale "
                "emergency contacts (not verified in >12 months) represent a patient "
                "safety risk, as phone numbers and availability may have changed. "
                "The Joint Commission recommends quarterly verification for high-risk patients."
            ),
            (
                "SUDEP risk reduction requires effective nocturnal seizure monitoring. "
                "Caregivers who share a bedroom with the patient reduce SUDEP risk by "
                "approximately 50%. Wearable seizure detection devices with caregiver "
                "alerting offer an alternative for patients who sleep alone "
                "(Harden et al., Neurology 2017)."
            ),
            (
                "Respite care services are essential for preventing caregiver burnout. "
                "Caregivers should have access to regular respite periods, with a minimum "
                "recommended frequency of one full day per month. Last respite date tracking "
                "enables clinical teams to identify caregivers at risk of burnout due to "
                "insufficient breaks from caregiving duties."
            ),
        ],
        "glossary": [
            {
                "term": "SUDEP",
                "definition": (
                    "Sudden Unexpected Death in Epilepsy -- the sudden death of a person "
                    "with epilepsy that is not due to trauma, drowning, or status epilepticus. "
                    "SUDEP is the leading cause of epilepsy-related mortality, with an "
                    "incidence of approximately 1 per 1,000 patient-years in adults with "
                    "chronic epilepsy."
                ),
            },
            {
                "term": "Rescue Medication",
                "definition": (
                    "Emergency benzodiazepine medication administered by caregivers during "
                    "prolonged seizures (>5 minutes) or seizure clusters. Common formulations "
                    "include intranasal midazolam (Nayzilam), buccal midazolam, and rectal "
                    "diazepam (Diastat). Timely administration can prevent status epilepticus."
                ),
            },
            {
                "term": "Seizure Action Plan",
                "definition": (
                    "A personalised, written document detailing the specific steps caregivers "
                    "and bystanders should take when a patient has a seizure. Includes seizure "
                    "descriptions, timing instructions, rescue medication protocols, and "
                    "criteria for calling emergency services."
                ),
            },
            {
                "term": "Respite Care",
                "definition": (
                    "Temporary relief for primary caregivers provided by substitute caregivers, "
                    "professional services, or day programs. Respite care reduces caregiver "
                    "burnout and is associated with improved caregiver mental health and "
                    "sustained quality of patient care."
                ),
            },
            {
                "term": "Status Epilepticus",
                "definition": (
                    "A medical emergency defined as continuous seizure activity lasting longer "
                    "than 5 minutes, or two or more seizures without full recovery of "
                    "consciousness between them. Requires immediate emergency medical "
                    "intervention to prevent brain injury and death."
                ),
            },
            {
                "term": "Caregiver Burnout",
                "definition": (
                    "A state of physical, emotional, and mental exhaustion caused by prolonged "
                    "caregiving demands. Characterised by emotional exhaustion, depersonalisation, "
                    "and reduced sense of personal accomplishment. Assessed using validated "
                    "instruments such as the Maslach Burnout Inventory."
                ),
            },
            {
                "term": "Postictal State",
                "definition": (
                    "The altered state of consciousness following a seizure, typically lasting "
                    "5-30 minutes but sometimes hours. Characterised by confusion, drowsiness, "
                    "headache, and temporary neurological deficits. Caregivers should ensure "
                    "patient safety and monitor recovery during this period."
                ),
            },
            {
                "term": "Tonic-Clonic Seizure",
                "definition": (
                    "A generalised seizure involving initial muscle stiffening (tonic phase) "
                    "followed by rhythmic jerking (clonic phase). The seizure type most "
                    "commonly associated with SUDEP risk and the primary target of "
                    "caregiver seizure response training."
                ),
            },
            {
                "term": "Seizure Cluster",
                "definition": (
                    "Multiple seizures occurring within a short time period (typically 2-3 "
                    "seizures within 24 hours), often requiring rescue medication administration. "
                    "Seizure clusters are a common trigger for emergency department visits "
                    "and are a risk factor for status epilepticus."
                ),
            },
            {
                "term": "Primary Emergency Contact",
                "definition": (
                    "The first person to be notified in a seizure emergency, typically "
                    "the closest family member or primary caregiver. Primary contacts should "
                    "maintain 24/7 phone availability and have current knowledge of the "
                    "patient's seizure action plan and medication regimen."
                ),
            },
        ],
        "data_source": (
            "Real caregivers table (30 rows) and emergency_contacts table (30 rows) "
            "in clinical.db -- 30 patients with caregiver roles, training status, "
            "burnout metrics, emergency protocols, and verified emergency contacts."
        ),
    }
