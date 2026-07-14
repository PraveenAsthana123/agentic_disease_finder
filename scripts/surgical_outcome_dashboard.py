"""Surgical Outcome Dashboard — epilepsy surgery analytics from clinical.db.

Provides Engel classification outcomes, ILAE outcome scale, seizure freedom
rates, complication tracking, and surgery type analysis for post-operative
epilepsy surgery evaluation.

Clinically this matters because:
- Epilepsy surgery achieves seizure freedom (Engel IA) in ~60-70% of temporal
  lobe cases — superior to continued medical therapy for drug-resistant epilepsy
  (Wiebe et al., NEJM 2001).
- Complication rates and outcome distributions guide informed consent and
  surgical candidate selection.
- ILAE outcome scale (Wieser et al., 2001) provides standardized reporting
  complementary to the Engel classification.

Sources:
- surgical_outcomes table  (clinical.db)
- patients table           (clinical.db)
"""

import pathlib
import sqlite3
from collections import Counter, defaultdict

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

    total_surgeries = _safe(c, "SELECT COUNT(*) FROM surgical_outcomes")
    total_patients = _safe(c, "SELECT COUNT(DISTINCT patient_id) FROM surgical_outcomes")
    seizure_free_count = _safe(c, "SELECT COUNT(*) FROM surgical_outcomes WHERE seizure_free = 1")
    seizure_free_rate = round(seizure_free_count / total_surgeries * 100, 1) if total_surgeries else 0
    mean_followup = _safe(c, "SELECT ROUND(AVG(follow_up_months), 1) FROM surgical_outcomes", default=0.0)
    complication_count = _safe(c, "SELECT COUNT(*) FROM surgical_outcomes WHERE complications IS NOT NULL")
    complication_rate = round(complication_count / total_surgeries * 100, 1) if total_surgeries else 0
    aed_reduction_count = _safe(c, "SELECT COUNT(*) FROM surgical_outcomes WHERE aed_reduction = 1")
    aed_reduction_rate = round(aed_reduction_count / total_surgeries * 100, 1) if total_surgeries else 0

    # Engel class distribution (group by major class: I, II, III, IV)
    c.execute("""SELECT engel_class, COUNT(*) as cnt FROM surgical_outcomes
        GROUP BY engel_class ORDER BY engel_class""")
    engel_detail = [{"class": r["engel_class"], "count": r["cnt"]} for r in c.fetchall()]

    # Major Engel classes
    c.execute("""SELECT
        CASE
            WHEN engel_class LIKE 'I%' AND engel_class NOT LIKE 'II%'
                 AND engel_class NOT LIKE 'III%' AND engel_class NOT LIKE 'IV%' THEN 'I'
            WHEN engel_class LIKE 'II%' AND engel_class NOT LIKE 'III%'
                 AND engel_class NOT LIKE 'IV%' THEN 'II'
            WHEN engel_class LIKE 'III%' THEN 'III'
            WHEN engel_class LIKE 'IV%' THEN 'IV'
        END as major_class,
        COUNT(*) as cnt
        FROM surgical_outcomes GROUP BY major_class ORDER BY major_class""")
    engel_major = [{"class": r["major_class"], "count": r["cnt"]} for r in c.fetchall()]

    # Surgery type distribution
    c.execute("""SELECT surgery_type, COUNT(*) as cnt FROM surgical_outcomes
        GROUP BY surgery_type ORDER BY cnt DESC""")
    surgery_type_dist = [{"type": r["surgery_type"], "count": r["cnt"]} for r in c.fetchall()]

    # ILAE outcome distribution
    c.execute("""SELECT ilae_outcome, COUNT(*) as cnt FROM surgical_outcomes
        GROUP BY ilae_outcome ORDER BY ilae_outcome""")
    ilae_dist = [{"outcome": r["ilae_outcome"], "count": r["cnt"]} for r in c.fetchall()]

    # Outcome by surgery type
    c.execute("""SELECT surgery_type,
        COUNT(*) as total,
        SUM(CASE WHEN seizure_free = 1 THEN 1 ELSE 0 END) as seizure_free,
        ROUND(AVG(follow_up_months), 1) as avg_followup,
        SUM(CASE WHEN complications IS NOT NULL THEN 1 ELSE 0 END) as complications
        FROM surgical_outcomes GROUP BY surgery_type ORDER BY total DESC""")
    outcome_by_type = [dict(r) for r in c.fetchall()]

    # Follow-up trend (outcome vs follow-up duration buckets)
    c.execute("""SELECT
        CASE
            WHEN follow_up_months <= 12 THEN '0-12m'
            WHEN follow_up_months <= 24 THEN '13-24m'
            WHEN follow_up_months <= 36 THEN '25-36m'
            WHEN follow_up_months <= 48 THEN '37-48m'
            ELSE '49-60m'
        END as period,
        COUNT(*) as total,
        SUM(CASE WHEN seizure_free = 1 THEN 1 ELSE 0 END) as seizure_free
        FROM surgical_outcomes GROUP BY period ORDER BY period""")
    followup_trend = [dict(r) for r in c.fetchall()]

    con.close()
    return {
        "total_surgeries": total_surgeries,
        "total_patients": total_patients,
        "seizure_free_count": seizure_free_count,
        "seizure_free_rate": seizure_free_rate,
        "mean_followup": mean_followup,
        "complication_count": complication_count,
        "complication_rate": complication_rate,
        "aed_reduction_count": aed_reduction_count,
        "aed_reduction_rate": aed_reduction_rate,
        "engel_detail": engel_detail,
        "engel_major": engel_major,
        "surgery_type_distribution": surgery_type_dist,
        "ilae_distribution": ilae_dist,
        "outcome_by_type": outcome_by_type,
        "followup_trend": followup_trend,
    }


def breakdown():
    con = _conn()
    c = con.cursor()

    # Per-patient surgery details
    c.execute("""SELECT id, patient_id, surgery_type, surgery_date, hemisphere,
        pathology, engel_class, ilae_outcome, follow_up_months, seizure_free,
        aed_reduction, complications, complication_severity,
        pre_surgery_frequency, post_surgery_frequency, notes
        FROM surgical_outcomes ORDER BY surgery_date DESC""")
    surgeries = [dict(r) for r in c.fetchall()]

    # Outcome by pathology
    c.execute("""SELECT pathology,
        COUNT(*) as total,
        SUM(CASE WHEN seizure_free = 1 THEN 1 ELSE 0 END) as seizure_free,
        ROUND(AVG(post_surgery_frequency), 1) as avg_post_freq,
        ROUND(AVG(pre_surgery_frequency), 1) as avg_pre_freq
        FROM surgical_outcomes GROUP BY pathology ORDER BY total DESC""")
    outcome_by_pathology = [dict(r) for r in c.fetchall()]

    # Hemisphere distribution
    c.execute("""SELECT hemisphere, COUNT(*) as cnt,
        SUM(CASE WHEN seizure_free = 1 THEN 1 ELSE 0 END) as seizure_free
        FROM surgical_outcomes GROUP BY hemisphere ORDER BY cnt DESC""")
    hemisphere_dist = [dict(r) for r in c.fetchall()]

    # Complication breakdown
    c.execute("""SELECT complications, complication_severity, COUNT(*) as cnt
        FROM surgical_outcomes WHERE complications IS NOT NULL
        GROUP BY complications, complication_severity ORDER BY cnt DESC""")
    complication_breakdown = [dict(r) for r in c.fetchall()]

    # Cases with complications (detail)
    c.execute("""SELECT patient_id, surgery_type, surgery_date, complications,
        complication_severity, engel_class
        FROM surgical_outcomes WHERE complications IS NOT NULL
        ORDER BY surgery_date DESC""")
    complication_cases = [dict(r) for r in c.fetchall()]

    # Pre vs Post seizure frequency comparison
    c.execute("""SELECT patient_id, surgery_type, pre_surgery_frequency,
        post_surgery_frequency, engel_class
        FROM surgical_outcomes ORDER BY pre_surgery_frequency DESC""")
    freq_comparison = [dict(r) for r in c.fetchall()]

    # Engel by follow-up duration
    c.execute("""SELECT
        CASE
            WHEN follow_up_months <= 12 THEN '0-12m'
            WHEN follow_up_months <= 24 THEN '13-24m'
            WHEN follow_up_months <= 36 THEN '25-36m'
            ELSE '37-60m'
        END as period,
        engel_class, COUNT(*) as cnt
        FROM surgical_outcomes GROUP BY period, engel_class
        ORDER BY period, engel_class""")
    engel_by_followup = [dict(r) for r in c.fetchall()]

    con.close()
    return {
        "surgeries": surgeries,
        "outcome_by_pathology": outcome_by_pathology,
        "hemisphere_distribution": hemisphere_dist,
        "complication_breakdown": complication_breakdown,
        "complication_cases": complication_cases,
        "freq_comparison": freq_comparison,
        "engel_by_followup": engel_by_followup,
    }


def definitions():
    return {
        "title": "Surgical Outcome Dashboard — Definitions",
        "engel_classification": [
            {"class": "I", "label": "Free of disabling seizures",
             "subclasses": [
                 {"sub": "IA", "definition": "Completely seizure-free since surgery."},
                 {"sub": "IB", "definition": "Non-disabling simple partial seizures only."},
                 {"sub": "IC", "definition": "Some disabling seizures after surgery, but free of disabling seizures for at least 2 years."},
                 {"sub": "ID", "definition": "Generalized convulsions with AED discontinuation only."},
             ]},
            {"class": "II", "label": "Rare disabling seizures",
             "subclasses": [
                 {"sub": "IIA", "definition": "Initially free of disabling seizures, but now has rare seizures."},
                 {"sub": "IIB", "definition": "Rare disabling seizures since surgery."},
                 {"sub": "IIC", "definition": "More than rare disabling seizures after surgery, but rare seizures for the last 2 years."},
                 {"sub": "IID", "definition": "Nocturnal seizures only."},
             ]},
            {"class": "III", "label": "Worthwhile improvement",
             "subclasses": [
                 {"sub": "IIIA", "definition": "Worthwhile seizure reduction (>90% reduction)."},
                 {"sub": "IIIB", "definition": "Prolonged seizure-free intervals amounting to >50% of follow-up."},
             ]},
            {"class": "IV", "label": "No worthwhile improvement",
             "subclasses": [
                 {"sub": "IVA", "definition": "Significant seizure reduction (>50% reduction)."},
                 {"sub": "IVB", "definition": "No appreciable change."},
                 {"sub": "IVC", "definition": "Seizures worse."},
             ]},
        ],
        "ilae_scale": [
            {"outcome": 1, "definition": "Completely seizure-free, no auras."},
            {"outcome": 2, "definition": "Only auras, no other seizures."},
            {"outcome": 3, "definition": "One to three seizure days per year, with or without auras."},
            {"outcome": 4, "definition": "Four seizure days per year to 50% reduction of baseline seizure days."},
            {"outcome": 5, "definition": "Less than 50% reduction of baseline seizure days to 100% increase."},
            {"outcome": 6, "definition": "More than 100% increase of baseline seizure days."},
        ],
        "surgery_types": [
            {"type": "Temporal lobectomy", "description": "Resection of the temporal lobe, including amygdalohippocampectomy. Most common epilepsy surgery. Best outcomes for mesial temporal sclerosis (~65-70% Engel I)."},
            {"type": "Lesionectomy", "description": "Targeted removal of a discrete epileptogenic lesion (FCD, cavernoma, tumor). Outcome depends on complete resection of the epileptogenic zone."},
            {"type": "Hemispherectomy", "description": "Disconnection or removal of one cerebral hemisphere. Reserved for catastrophic hemispheric epilepsy (Rasmussen, Sturge-Weber, large FCD). High seizure freedom (~70-80%)."},
            {"type": "Corpus callosotomy", "description": "Section of the corpus callosum to prevent bilateral seizure spread. Palliative — reduces drop attacks/generalized seizures, rarely curative."},
            {"type": "LITT (Laser Interstitial Thermal Therapy)", "description": "MRI-guided stereotactic laser ablation. Minimally invasive alternative to open resection for deep/mesial lesions. Engel I ~55-60%."},
            {"type": "VNS implant (Vagus Nerve Stimulation)", "description": "Palliative neuromodulation device. Reduces seizure frequency by ~50% in ~50% of patients (\"Rule of 50s\"). Not curative."},
            {"type": "RNS implant (Responsive Neurostimulation)", "description": "Closed-loop neurostimulation detecting and disrupting seizure onset. For bilateral or eloquent cortex foci not amenable to resection. Median 75% reduction at 9 years."},
            {"type": "Multiple subpial transections (MST)", "description": "Disrupts horizontal cortical connections while preserving vertical columns. For epileptogenic zones in eloquent cortex (language, motor)."},
        ],
        "data_sources": [
            "surgical_outcomes table (clinical.db) — ~28 records, ~22 patients",
            "patients table (clinical.db) — demographics for cross-reference",
            "Engel classification (Engel, 1993) — standard surgical outcome scale",
            "ILAE outcome scale (Wieser et al., Epilepsia 2001) — complementary outcome measure",
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
    print(f"  {len(d['engel_classification'])} Engel classes, {len(d['ilae_scale'])} ILAE levels, {len(d['surgery_types'])} surgery types")
