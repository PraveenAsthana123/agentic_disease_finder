"""Referral Records Dashboard — backend analytics for referral_records table."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def _conn():
    return sqlite3.connect(DB)

def overview():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total = c.execute("SELECT COUNT(*) FROM referral_records").fetchone()[0]
    patients = c.execute("SELECT COUNT(DISTINCT patient_id) FROM referral_records").fetchone()[0]
    avg_score = c.execute("SELECT ROUND(AVG(triage_score),1) FROM referral_records").fetchone()[0]
    urgent_emergent = c.execute("SELECT COUNT(*) FROM referral_records WHERE urgency IN ('urgent','emergent')").fetchone()[0]
    completed = c.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status='completed'").fetchone()[0]
    completion_rate = round(completed / total * 100, 1) if total else 0
    pending = c.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status='pending_triage'").fetchone()[0]

    source_dist = [dict(r) for r in c.execute(
        "SELECT referral_source AS source, COUNT(*) AS count FROM referral_records GROUP BY referral_source ORDER BY count DESC")]

    reason_dist = [dict(r) for r in c.execute(
        "SELECT referral_reason AS reason, COUNT(*) AS count FROM referral_records GROUP BY referral_reason ORDER BY count DESC")]

    urgency_dist = [dict(r) for r in c.execute(
        "SELECT urgency, COUNT(*) AS count FROM referral_records GROUP BY urgency ORDER BY count DESC")]

    triage_status_dist = [dict(r) for r in c.execute(
        "SELECT triage_status AS status, COUNT(*) AS count FROM referral_records GROUP BY triage_status ORDER BY count DESC")]

    monthly_trend = [dict(r) for r in c.execute("""
        SELECT SUBSTR(referral_date,1,7) AS month,
               COUNT(*) AS total,
               SUM(CASE WHEN urgency IN ('urgent','emergent') THEN 1 ELSE 0 END) AS urgent_emergent
        FROM referral_records GROUP BY month ORDER BY month
    """)]

    assigned_dist = [dict(r) for r in c.execute(
        "SELECT assigned_to, COUNT(*) AS count FROM referral_records WHERE assigned_to IS NOT NULL GROUP BY assigned_to ORDER BY count DESC")]

    avg_score_by_urgency = [dict(r) for r in c.execute(
        "SELECT urgency, ROUND(AVG(triage_score),1) AS avg_score FROM referral_records GROUP BY urgency ORDER BY avg_score DESC")]

    avg_score_by_source = [dict(r) for r in c.execute(
        "SELECT referral_source AS source, ROUND(AVG(triage_score),1) AS avg_score FROM referral_records GROUP BY referral_source ORDER BY avg_score DESC")]

    conn.close()
    return {
        "total_referrals": total,
        "total_patients": patients,
        "avg_triage_score": avg_score,
        "urgent_emergent_count": urgent_emergent,
        "completion_rate": completion_rate,
        "pending_count": pending,
        "source_distribution": source_dist,
        "reason_distribution": reason_dist,
        "urgency_distribution": urgency_dist,
        "triage_status_distribution": triage_status_dist,
        "monthly_trend": monthly_trend,
        "assigned_to_distribution": assigned_dist,
        "avg_triage_score_by_urgency": avg_score_by_urgency,
        "avg_triage_score_by_source": avg_score_by_source,
    }

def breakdown():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    referrals = [dict(r) for r in c.execute(
        "SELECT * FROM referral_records ORDER BY referral_date DESC")]

    patient_summary = [dict(r) for r in c.execute("""
        SELECT patient_id,
               COUNT(*) AS total_referrals,
               ROUND(AVG(triage_score),1) AS avg_triage_score,
               MAX(referral_date) AS latest_referral_date,
               (SELECT referral_source FROM referral_records r2
                WHERE r2.patient_id = r1.patient_id
                GROUP BY referral_source ORDER BY COUNT(*) DESC LIMIT 1) AS top_source,
               (SELECT urgency FROM referral_records r2
                WHERE r2.patient_id = r1.patient_id
                GROUP BY urgency ORDER BY COUNT(*) DESC LIMIT 1) AS top_urgency
        FROM referral_records r1 GROUP BY patient_id ORDER BY total_referrals DESC
    """)]

    by_source = [dict(r) for r in c.execute("""
        SELECT referral_source AS source,
               COUNT(*) AS count,
               ROUND(AVG(triage_score),1) AS avg_triage_score,
               ROUND(SUM(CASE WHEN triage_status='completed' THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS completion_rate,
               (SELECT referral_reason FROM referral_records r2
                WHERE r2.referral_source = r1.referral_source
                GROUP BY referral_reason ORDER BY COUNT(*) DESC LIMIT 1) AS top_reason
        FROM referral_records r1 GROUP BY referral_source ORDER BY count DESC
    """)]

    conn.close()
    return {
        "referrals": referrals,
        "patient_summary": patient_summary,
        "by_source": by_source,
    }

def definitions():
    return {
        "title": "Referral Records Dashboard — Definitions",
        "concepts": [
            {"name": "Referral Source — Primary Care", "description": "Referral initiated by the patient's primary care physician or general practitioner, typically after initial evaluation of neurological symptoms."},
            {"name": "Referral Source — Neurology Clinic", "description": "Referral from a general neurology clinic for specialized epilepsy evaluation, monitoring, or surgical assessment."},
            {"name": "Referral Source — Emergency", "description": "Referral originating from an emergency department visit, often for acute seizure presentation, status epilepticus, or first-time seizure."},
            {"name": "Referral Source — Pediatrics", "description": "Referral from a pediatric provider for evaluation of childhood-onset seizures, febrile seizures, or developmental regression with epileptic features."},
            {"name": "Referral Source — Psychiatry", "description": "Referral from a psychiatrist, often for differentiation of psychogenic nonepileptic events vs epileptic seizures, or psychiatric comorbidity management."},
            {"name": "Referral Source — Other Specialist", "description": "Referral from non-neurology specialists (e.g., cardiology for syncope workup, sleep medicine for nocturnal events)."},
            {"name": "Referral Source — Self-Referral", "description": "Patient-initiated referral, typically for second opinion, self-identified seizure symptoms, or direct access clinic appointments."},
            {"name": "Referral Reason — New Onset Seizure", "description": "First-time seizure requiring diagnostic workup including EEG, neuroimaging, and metabolic evaluation to establish etiology and treatment plan."},
            {"name": "Referral Reason — Refractory Epilepsy", "description": "Drug-resistant epilepsy (failure of ≥2 adequate antiseizure medication trials) requiring advanced evaluation including surgical candidacy assessment."},
            {"name": "Referral Reason — Pre-Surgical Evaluation", "description": "Comprehensive workup for epilepsy surgery candidacy including video-EEG, MRI, PET, neuropsychological testing, and Wada test."},
            {"name": "Referral Reason — Status Epilepticus", "description": "Prolonged or repetitive seizures without recovery, a neurological emergency requiring immediate evaluation and aggressive treatment."},
            {"name": "Referral Reason — Medication Review", "description": "Assessment of current antiseizure medication regimen for efficacy, side effects, drug interactions, or planned medication transition."},
            {"name": "Referral Reason — Headache Workup", "description": "Evaluation of headache presentations that may overlap with epilepsy, including postictal headache, migraine-epilepsy comorbidity, or migraine aura mimicking seizure."},
            {"name": "Referral Reason — EEG Abnormality", "description": "Referral for interpretation of abnormal EEG findings such as epileptiform discharges, focal slowing, or subclinical seizure patterns."},
            {"name": "Referral Reason — Cognitive Decline", "description": "Evaluation of progressive cognitive impairment potentially related to epilepsy, chronic seizure activity, or antiseizure medication effects."},
            {"name": "Referral Reason — Psychogenic Nonepileptic", "description": "Evaluation for psychogenic nonepileptic seizures (PNES), requiring video-EEG to confirm absence of epileptiform correlate and psychiatric assessment."},
            {"name": "Urgency — Routine", "description": "Standard priority referral with no acute clinical concern. Typically scheduled within normal appointment availability (2-6 weeks)."},
            {"name": "Urgency — Elective", "description": "Non-urgent, planned referral for optimization of care (e.g., medication review, follow-up evaluation). Scheduled at patient and provider convenience."},
            {"name": "Urgency — Urgent", "description": "Referral requiring expedited evaluation, typically within 1-2 weeks. Indicates worsening seizure control, significant medication side effects, or new neurological deficits."},
            {"name": "Urgency — Emergent", "description": "Highest priority referral requiring same-day or next-day evaluation. Includes status epilepticus, acute symptomatic seizures, or life-threatening conditions."},
            {"name": "Triage Status — Pending Triage", "description": "Referral received but not yet reviewed or prioritized by the triage team. Awaiting initial clinical assessment."},
            {"name": "Triage Status — Triaged", "description": "Referral reviewed and assigned a priority level, but appointment not yet scheduled. Clinical urgency has been assessed."},
            {"name": "Triage Status — Scheduled", "description": "Appointment has been booked with the assigned provider. Patient has been notified of date and time."},
            {"name": "Triage Status — In Progress", "description": "Patient is currently being seen or undergoing the evaluation process. Active clinical engagement."},
            {"name": "Triage Status — Completed", "description": "Referral evaluation finished. Clinical findings documented and recommendations communicated to referring provider."},
            {"name": "Triage Status — Cancelled", "description": "Referral withdrawn by patient, referring provider, or receiving clinic. May be due to resolution, patient preference, or duplicate referral."},
            {"name": "Triage Score", "description": "Numeric score (0-100) assigned during triage to quantify clinical priority. Higher scores indicate greater urgency. Factors include seizure frequency, comorbidities, functional impairment, and time sensitivity of the referral reason."},
        ],
        "data_sources": [
            "referral_records table — 84 referrals, 41 patients",
            "Referral management system integrated with EHR",
            "Triage scoring algorithm based on clinical urgency criteria",
        ],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
