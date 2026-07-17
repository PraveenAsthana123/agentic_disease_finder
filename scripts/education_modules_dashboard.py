"""Education Modules Dashboard — patient education analytics from clinical.db.

Tracks patient engagement with epilepsy education modules including completion
rates, quiz scores, time spent, and per-module/per-patient progress.

Clinically this matters because:
- Patient education improves seizure self-management, medication adherence, and
  reduces emergency visits (Bradley & Lindsay, Cochrane 2008).
- Epilepsy-specific education on SUDEP awareness, seizure first aid, and AED
  management has direct safety implications.
- Tracking completion and quiz performance identifies knowledge gaps that may
  correlate with adverse outcomes.

Sources:
- education_modules table  (clinical.db)
- patients table           (clinical.db)
"""

import pathlib
import sqlite3
from collections import defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def overview():
    con = _conn()
    c = con.cursor()

    total_enrollments = _safe(c, "SELECT COUNT(*) FROM education_modules")
    total_patients = _safe(c, "SELECT COUNT(DISTINCT patient_id) FROM education_modules")
    total_modules = _safe(c, "SELECT COUNT(DISTINCT module_name) FROM education_modules")
    avg_completion = round(_safe(c, "SELECT AVG(completion_pct) FROM education_modules", default=0.0), 1)
    avg_time = round(_safe(c, "SELECT AVG(time_spent_minutes) FROM education_modules", default=0.0), 1)
    completed_count = _safe(c, "SELECT COUNT(*) FROM education_modules WHERE completion_pct = 100")
    completion_rate = round(completed_count / total_enrollments * 100, 1) if total_enrollments else 0
    not_started = _safe(c, "SELECT COUNT(*) FROM education_modules WHERE completion_pct = 0")
    in_progress = _safe(c, "SELECT COUNT(*) FROM education_modules WHERE completion_pct > 0 AND completion_pct < 100")

    # Quiz stats
    quiz_taken = _safe(c, "SELECT COUNT(*) FROM education_modules WHERE quiz_score IS NOT NULL")
    avg_quiz = round(_safe(c, "SELECT AVG(quiz_score) FROM education_modules WHERE quiz_score IS NOT NULL", default=0.0), 1)
    quiz_pass_rate = 0
    if quiz_taken:
        passing = _safe(c, "SELECT COUNT(*) FROM education_modules WHERE quiz_score IS NOT NULL AND quiz_score >= 70")
        quiz_pass_rate = round(passing / quiz_taken * 100, 1)

    total_time = _safe(c, "SELECT SUM(time_spent_minutes) FROM education_modules", default=0)

    # Module name distribution
    c.execute("""SELECT module_name, COUNT(*) as cnt,
        ROUND(AVG(completion_pct), 1) as avg_comp,
        ROUND(AVG(time_spent_minutes), 1) as avg_time,
        SUM(CASE WHEN completion_pct = 100 THEN 1 ELSE 0 END) as completed
        FROM education_modules GROUP BY module_name ORDER BY cnt DESC""")
    module_dist = [dict(r) for r in c.fetchall()]

    # Format distribution
    c.execute("""SELECT format, COUNT(*) as cnt,
        ROUND(AVG(completion_pct), 1) as avg_completion,
        ROUND(AVG(time_spent_minutes), 1) as avg_time
        FROM education_modules GROUP BY format ORDER BY cnt DESC""")
    format_dist = [dict(r) for r in c.fetchall()]

    # Completion distribution buckets
    c.execute("""SELECT
        CASE
            WHEN completion_pct = 0 THEN 'Not Started (0%)'
            WHEN completion_pct < 25 THEN '1-24%'
            WHEN completion_pct < 50 THEN '25-49%'
            WHEN completion_pct < 75 THEN '50-74%'
            WHEN completion_pct < 100 THEN '75-99%'
            ELSE 'Complete (100%)'
        END as bucket,
        COUNT(*) as cnt
        FROM education_modules GROUP BY bucket ORDER BY bucket""")
    completion_dist = [dict(r) for r in c.fetchall()]

    # Monthly enrollment trend (by started_at)
    c.execute("""SELECT
        SUBSTR(started_at, 1, 7) as month,
        COUNT(*) as enrollments,
        SUM(CASE WHEN completion_pct = 100 THEN 1 ELSE 0 END) as completed,
        ROUND(AVG(completion_pct), 1) as avg_completion
        FROM education_modules
        WHERE started_at IS NOT NULL
        GROUP BY month ORDER BY month""")
    monthly_trend = [dict(r) for r in c.fetchall()]

    # Quiz score distribution buckets
    c.execute("""SELECT
        CASE
            WHEN quiz_score < 60 THEN 'Below 60'
            WHEN quiz_score < 70 THEN '60-69'
            WHEN quiz_score < 80 THEN '70-79'
            WHEN quiz_score < 90 THEN '80-89'
            ELSE '90-100'
        END as bucket,
        COUNT(*) as cnt
        FROM education_modules
        WHERE quiz_score IS NOT NULL
        GROUP BY bucket ORDER BY bucket""")
    quiz_dist = [dict(r) for r in c.fetchall()]

    con.close()
    return {
        "total_enrollments": total_enrollments,
        "total_patients": total_patients,
        "total_modules": total_modules,
        "avg_completion": avg_completion,
        "avg_time_minutes": avg_time,
        "completed_count": completed_count,
        "completion_rate": completion_rate,
        "not_started": not_started,
        "in_progress": in_progress,
        "quiz_taken": quiz_taken,
        "avg_quiz_score": avg_quiz,
        "quiz_pass_rate": quiz_pass_rate,
        "total_time_hours": round(total_time / 60, 1) if total_time else 0,
        "module_distribution": module_dist,
        "format_distribution": format_dist,
        "completion_distribution": completion_dist,
        "monthly_trend": monthly_trend,
        "quiz_score_distribution": quiz_dist,
    }


def breakdown():
    con = _conn()
    c = con.cursor()

    # Per-patient summary
    c.execute("""SELECT patient_id,
        COUNT(*) as total_modules,
        SUM(CASE WHEN completion_pct = 100 THEN 1 ELSE 0 END) as completed,
        ROUND(AVG(completion_pct), 1) as avg_completion,
        ROUND(AVG(time_spent_minutes), 1) as avg_time,
        ROUND(AVG(CASE WHEN quiz_score IS NOT NULL THEN quiz_score END), 1) as avg_quiz,
        SUM(time_spent_minutes) as total_time
        FROM education_modules
        GROUP BY patient_id
        ORDER BY avg_completion DESC""")
    per_patient = [dict(r) for r in c.fetchall()]

    # Per-module completion rates
    c.execute("""SELECT module_name,
        COUNT(*) as enrolled,
        SUM(CASE WHEN completion_pct = 100 THEN 1 ELSE 0 END) as completed,
        ROUND(100.0 * SUM(CASE WHEN completion_pct = 100 THEN 1 ELSE 0 END) / COUNT(*), 1) as completion_rate,
        ROUND(AVG(completion_pct), 1) as avg_progress,
        ROUND(AVG(time_spent_minutes), 1) as avg_time,
        ROUND(AVG(CASE WHEN quiz_score IS NOT NULL THEN quiz_score END), 1) as avg_quiz,
        COUNT(CASE WHEN quiz_score IS NOT NULL THEN 1 END) as quiz_count
        FROM education_modules
        GROUP BY module_name
        ORDER BY completion_rate DESC""")
    per_module = [dict(r) for r in c.fetchall()]

    # Low-engagement patients (avg completion < 30%)
    c.execute("""SELECT patient_id,
        COUNT(*) as modules,
        ROUND(AVG(completion_pct), 1) as avg_completion,
        SUM(CASE WHEN completion_pct = 0 THEN 1 ELSE 0 END) as not_started
        FROM education_modules
        GROUP BY patient_id
        HAVING AVG(completion_pct) < 30
        ORDER BY avg_completion ASC""")
    low_engagement = [dict(r) for r in c.fetchall()]

    # Module x format cross-tab
    c.execute("""SELECT module_name, format,
        COUNT(*) as cnt,
        ROUND(AVG(completion_pct), 1) as avg_completion
        FROM education_modules
        GROUP BY module_name, format
        ORDER BY module_name, format""")
    module_format = [dict(r) for r in c.fetchall()]

    # Recent enrollments (latest 20)
    c.execute("""SELECT id, patient_id, module_name, completion_pct, quiz_score,
        time_spent_minutes, format, started_at, completed_at
        FROM education_modules
        ORDER BY started_at DESC LIMIT 20""")
    recent = [dict(r) for r in c.fetchall()]

    # Top quiz performers
    c.execute("""SELECT patient_id, module_name, quiz_score, completion_pct, format
        FROM education_modules
        WHERE quiz_score IS NOT NULL
        ORDER BY quiz_score DESC LIMIT 15""")
    top_quiz = [dict(r) for r in c.fetchall()]

    con.close()
    return {
        "per_patient": per_patient,
        "per_module": per_module,
        "low_engagement": low_engagement,
        "module_format_crosstab": module_format,
        "recent_enrollments": recent,
        "top_quiz_performers": top_quiz,
    }


def definitions():
    return {
        "title": "Education Modules Dashboard — Definitions",
        "glossary": [
            {"term": "Completion Rate", "definition": "Percentage of enrolled modules where the patient reached 100% progress. A key engagement metric."},
            {"term": "Quiz Score", "definition": "Assessment score (0-100) measuring knowledge retention after module completion. Passing threshold: 70%."},
            {"term": "Time Spent", "definition": "Total minutes a patient actively engaged with the module content. Does not include idle time."},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy. Leading cause of epilepsy-related death. Education on risk factors and prevention is critical."},
            {"term": "AED", "definition": "Anti-Epileptic Drug. Education covers mechanism, adherence importance, side effects, and drug interactions."},
            {"term": "Ketogenic Diet", "definition": "High-fat, low-carbohydrate diet therapy that reduces seizures in ~50% of drug-resistant epilepsy patients, especially children."},
            {"term": "VNS", "definition": "Vagus Nerve Stimulation. Implantable device therapy for drug-resistant epilepsy. Education covers device operation and magnet use."},
            {"term": "Seizure Diary", "definition": "Systematic recording of seizure events, triggers, and patterns. Critical for treatment optimization and surgical evaluation."},
            {"term": "Seizure First Aid", "definition": "Evidence-based response protocol during seizure events: protect from injury, do not restrain, time the seizure, call 911 if >5 minutes."},
            {"term": "Interactive Module", "definition": "Education format with embedded simulations, decision trees, or case-based scenarios for active learning."},
            {"term": "Self-Management", "definition": "Patient-driven seizure management including medication adherence, trigger avoidance, lifestyle modification, and emergency planning."},
            {"term": "Health Literacy", "definition": "Patient ability to understand and act on health information. Education modules should be accessible across literacy levels."},
        ],
        "module_descriptions": [
            {"module": "SUDEP Awareness", "description": "Risk factors, prevention strategies, and when to seek emergency care. Based on AES/AAN SUDEP prevention guidelines."},
            {"module": "AED Basics", "description": "How anti-epileptic drugs work, common side effects, importance of adherence, drug interactions, and what to do if a dose is missed."},
            {"module": "Seizure First Aid", "description": "Step-by-step guide for bystanders and caregivers on responding to tonic-clonic, absence, and focal seizures safely."},
            {"module": "Seizure Diary Training", "description": "How to accurately record seizure type, duration, triggers, and recovery. Includes app-based and paper diary options."},
            {"module": "Ketogenic Diet", "description": "Overview of ketogenic and modified Atkins diets for epilepsy, meal planning, monitoring ketosis, and managing side effects."},
            {"module": "VNS Therapy Guide", "description": "Understanding vagus nerve stimulation: device function, magnet use, programming, battery replacement, and MRI safety."},
            {"module": "Epilepsy Surgery Overview", "description": "Candidacy criteria, pre-surgical evaluation (video-EEG, MRI, Wada test), surgical options, outcomes, and recovery expectations."},
            {"module": "Women & Epilepsy", "description": "Contraception interactions with AEDs, pre-conception planning, pregnancy management, breastfeeding, and catamenial epilepsy."},
            {"module": "Mental Health & Epilepsy", "description": "Comorbid depression and anxiety, stigma, cognitive effects of epilepsy and AEDs, and when to seek psychological support."},
            {"module": "Lifestyle & Triggers", "description": "Common seizure triggers (sleep deprivation, alcohol, stress, photosensitivity), lifestyle modifications, and driving regulations."},
            {"module": "Emergency Preparedness", "description": "Creating a seizure action plan, medical ID, emergency contacts, rescue medication (midazolam, diazepam), and status epilepticus recognition."},
            {"module": "Driving & Legal Rights", "description": "State-specific seizure-free driving requirements, employment protections (ADA), insurance considerations, and disability resources."},
        ],
        "format_descriptions": [
            {"format": "video", "description": "Pre-recorded educational videos with clinical demonstrations, patient testimonials, and expert explanations. Avg 15-30 minutes."},
            {"format": "article", "description": "Written educational content with diagrams and illustrations. Self-paced reading at various health literacy levels."},
            {"format": "quiz", "description": "Assessment-based modules with multiple-choice and scenario questions. Provides immediate feedback and score."},
            {"format": "interactive", "description": "Multimedia modules with simulations, decision trees, and case-based scenarios. Highest engagement but longest completion time."},
        ],
        "clinical_notes": [
            "Epilepsy education programs reduce seizure frequency and improve quality of life (Bradley & Lindsay, Cochrane 2008).",
            "SUDEP risk education should be provided to all patients with epilepsy per AAN practice guideline (Harden et al., 2017).",
            "Health literacy assessment should guide module format selection — video and interactive formats improve comprehension for low-literacy patients.",
            "Quiz scores below 70% suggest knowledge gaps that may require one-on-one reinforcement or alternative teaching methods.",
            "Caregiver education (seizure first aid, emergency preparedness) is equally important and should be tracked separately.",
        ],
        "data_sources": [
            "education_modules table (clinical.db) — 179 records, 30 patients",
            "12 distinct epilepsy education modules",
            "4 delivery formats (video, article, quiz, interactive)",
            "Quiz scores available for completed quiz-format modules",
        ],
    }


if __name__ == "__main__":
    import json
    print("=== Overview ===")
    print(json.dumps(overview(), indent=2, default=str)[:2000])
    print("\n=== Breakdown (keys) ===")
    bd = breakdown()
    for k, v in bd.items():
        print(f"  {k}: {len(v) if isinstance(v, list) else v}")
    print("\n=== Definitions ===")
    d = definitions()
    print(f"  {len(d['glossary'])} glossary terms, {len(d['module_descriptions'])} modules, {len(d['format_descriptions'])} formats")
