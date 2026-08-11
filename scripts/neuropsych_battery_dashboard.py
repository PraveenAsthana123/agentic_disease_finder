"""Neuropsychological Battery Dashboard — detailed neuropsych assessments from clinical.db.

Provides cognitive index profiles (Memory / Attention / Executive / Language /
Processing Speed), screening scores (MoCA, MMSE, PHQ-9, GAD-7), impairment
classification, lateralization hypothesis distribution, and per-patient battery
detail — all drawn from the neuropsych table (37 records, 37 patients).

Clinical rationale:
- Neuropsychological testing is mandatory before epilepsy surgery to establish
  baseline cognition and identify eloquent-cortex risk (NAEC Level 4 centres).
- Cognitive decline is reported in 30-50 % of patients with temporal lobe
  epilepsy; tracking index profiles over time guides AED selection and rehab
  referral (Hermann et al., Epilepsia 2006).
- PHQ-9 ≥ 10 and GAD-7 ≥ 10 flag psychiatric co-morbidity requiring
  concurrent psychiatric review before surgical candidacy is confirmed.

Sources:
  neuropsych table  (clinical.db) — 37 rows, fields_json JSONB payload
  patients table    (clinical.db) — demographics cross-reference
"""

import json
import pathlib
import sqlite3
import statistics
from collections import Counter, defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _load_rows():
    """Return list of merged dicts (base columns + parsed fields_json)."""
    con = _conn()
    c = con.cursor()
    c.execute("SELECT * FROM neuropsych ORDER BY id")
    raw = c.fetchall()
    con.close()
    data = []
    for r in raw:
        row = dict(r)
        try:
            fields = json.loads(row.get("fields_json") or "{}")
            row.update(fields)
        except Exception:
            pass
        data.append(row)
    return data


def _bucket_moca(score):
    if score is None:
        return "Unknown"
    if score >= 26:
        return "Normal (≥26)"
    if score >= 18:
        return "Mild Impairment (18-25)"
    return "Moderate Impairment (<18)"


def _bucket_phq9(score):
    if score is None:
        return "Unknown"
    if score <= 4:
        return "Minimal (0-4)"
    if score <= 9:
        return "Mild (5-9)"
    if score <= 14:
        return "Moderate (10-14)"
    if score <= 19:
        return "Mod-Severe (15-19)"
    return "Severe (20-27)"


def _bucket_gad7(score):
    if score is None:
        return "Unknown"
    if score <= 4:
        return "Minimal (0-4)"
    if score <= 9:
        return "Mild (5-9)"
    if score <= 14:
        return "Moderate (10-14)"
    return "Severe (15-21)"


def _avg(vals):
    valid = [v for v in vals if v is not None]
    return round(statistics.mean(valid), 1) if valid else None


def overview():
    data = _load_rows()
    n = len(data)

    # KPI counts
    impairment_dist = Counter(d.get("impairment_flag") for d in data if d.get("impairment_flag"))
    impaired_count = sum(v for k, v in impairment_dist.items() if k in ("mild", "moderate"))

    battery_dist = Counter(d.get("battery_type") for d in data if d.get("battery_type"))
    lateral_dist = Counter(d.get("lateralization_hypothesis") for d in data if d.get("lateralization_hypothesis"))
    referral_dist = Counter(d.get("referral_reason") for d in data if d.get("referral_reason"))

    # Cognitive index means
    index_profile = [
        {"domain": "Memory",           "key": "memory_index",          "mean": _avg([d.get("memory_index") for d in data])},
        {"domain": "Attention",         "key": "attention_index",        "mean": _avg([d.get("attention_index") for d in data])},
        {"domain": "Executive",         "key": "executive_index",        "mean": _avg([d.get("executive_index") for d in data])},
        {"domain": "Language",          "key": "language_index",         "mean": _avg([d.get("language_index") for d in data])},
        {"domain": "Processing Speed",  "key": "processing_speed_index", "mean": _avg([d.get("processing_speed_index") for d in data])},
    ]

    # Screening score means
    avg_moca  = _avg([d.get("moca")  for d in data])
    avg_mmse  = _avg([d.get("mmse")  for d in data])
    avg_phq9  = _avg([d.get("phq9")  for d in data])
    avg_gad7  = _avg([d.get("gad7")  for d in data])

    phq9_elevated = sum(1 for d in data if (d.get("phq9") or 0) >= 10)
    gad7_elevated = sum(1 for d in data if (d.get("gad7") or 0) >= 10)
    moca_impaired = sum(1 for d in data if (d.get("moca") is not None) and d["moca"] < 26)

    # MoCA distribution
    moca_dist = Counter(_bucket_moca(d.get("moca")) for d in data)

    # Impairment distribution with colours
    imp_colour = {"none": "#22c55e", "mild": "#f59e0b", "moderate": "#ef4444"}
    impairment_list = [
        {"flag": k, "count": v, "color": imp_colour.get(k, "#94a3b8")}
        for k, v in sorted(impairment_dist.items())
    ]

    return {
        "total_assessments": n,
        "total_patients": len({d["patient_id"] for d in data}),
        "impaired_count": impaired_count,
        "impairment_rate_pct": round(impaired_count / n * 100, 1) if n else 0,
        "moca_impaired": moca_impaired,
        "phq9_elevated": phq9_elevated,
        "gad7_elevated": gad7_elevated,
        "avg_moca": avg_moca,
        "avg_mmse": avg_mmse,
        "avg_phq9": avg_phq9,
        "avg_gad7": avg_gad7,
        "index_profile": index_profile,
        "impairment_distribution": impairment_list,
        "battery_type_distribution": [{"type": k, "count": v} for k, v in battery_dist.most_common()],
        "lateralization_distribution": [{"hypothesis": k, "count": v} for k, v in lateral_dist.most_common()],
        "referral_reason_distribution": [{"reason": k.replace("_", " ").title(), "count": v} for k, v in referral_dist.most_common()],
        "moca_distribution": [{"bucket": k, "count": v} for k, v in sorted(moca_dist.items())],
    }


def breakdown():
    data = _load_rows()

    # PHQ-9 severity distribution
    phq9_dist = Counter(_bucket_phq9(d.get("phq9")) for d in data)
    gad7_dist = Counter(_bucket_gad7(d.get("gad7")) for d in data)

    # Per-patient summary (sorted by patient_id)
    patients = []
    for d in sorted(data, key=lambda x: x.get("patient_id", "")):
        patients.append({
            "patient_id": d.get("patient_id"),
            "battery_type": d.get("battery_type"),
            "impairment_flag": d.get("impairment_flag"),
            "lateralization": d.get("lateralization_hypothesis"),
            "referral_reason": (d.get("referral_reason") or "").replace("_", " ").title(),
            "moca": d.get("moca"),
            "mmse": d.get("mmse"),
            "phq9": d.get("phq9"),
            "gad7": d.get("gad7"),
            "memory_index": d.get("memory_index"),
            "attention_index": d.get("attention_index"),
            "executive_index": d.get("executive_index"),
            "language_index": d.get("language_index"),
            "processing_speed_index": d.get("processing_speed_index"),
            "verbal_memory_raw": d.get("verbal_memory_raw"),
            "visual_memory_raw": d.get("visual_memory_raw"),
            "trail_a_seconds": d.get("trail_a_seconds"),
            "trail_b_seconds": d.get("trail_b_seconds"),
            "assessor": d.get("assessor"),
            "created_at": d.get("created_at"),
        })

    # Domain weakness analysis: count patients below 85 in each domain
    domain_weakness = []
    for domain, key in [
        ("Memory",          "memory_index"),
        ("Attention",       "attention_index"),
        ("Executive",       "executive_index"),
        ("Language",        "language_index"),
        ("Processing Speed","processing_speed_index"),
    ]:
        below = sum(1 for d in data if (d.get(key) or 100) < 85)
        domain_weakness.append({"domain": domain, "below_85_count": below,
                                 "pct": round(below / len(data) * 100, 1) if data else 0})

    # Pre-surgical subset
    presurgical = [d for d in data if d.get("referral_reason") == "pre-surgical"]
    presurgical_impaired = sum(1 for d in presurgical if d.get("impairment_flag") in ("mild","moderate"))

    return {
        "phq9_severity_distribution": [{"severity": k, "count": v} for k, v in sorted(phq9_dist.items())],
        "gad7_severity_distribution": [{"severity": k, "count": v} for k, v in sorted(gad7_dist.items())],
        "domain_weakness": domain_weakness,
        "presurgical_count": len(presurgical),
        "presurgical_impaired": presurgical_impaired,
        "patients": patients,
    }


def definitions():
    return {
        "dashboard": "Neuropsychological Battery Dashboard",
        "data_source": "neuropsych table (clinical.db) — 37 assessments",
        "terms": [
            {
                "term": "Cognitive Domain Indices",
                "definition": (
                    "Composite scores (50–150 scale) derived from neuropsychological "
                    "sub-tests covering Memory, Attention, Executive Function, Language, "
                    "and Processing Speed. Scores < 85 are flagged as below-average."
                ),
            },
            {
                "term": "Impairment Flag",
                "definition": (
                    "'none' — all domain indices ≥ 85; 'mild' — 1–2 domains < 85; "
                    "'moderate' — 3 or more domains < 85. Thresholds follow "
                    "APA/WAIS-IV clinical interpretation guidelines."
                ),
            },
            {
                "term": "MoCA (Montreal Cognitive Assessment)",
                "definition": (
                    "30-point global screen. Score ≥ 26 = normal; 18–25 = mild "
                    "cognitive impairment; < 18 = moderate impairment. Sensitivity "
                    "for MCI ~90 % (Nasreddine et al., 2005)."
                ),
            },
            {
                "term": "MMSE (Mini-Mental State Examination)",
                "definition": (
                    "30-point bedside screen. Score 24–30 = normal; 18–23 = mild "
                    "dementia; 0–17 = severe dementia. Less sensitive than MoCA for "
                    "frontal/executive deficits."
                ),
            },
            {
                "term": "PHQ-9 (Patient Health Questionnaire-9)",
                "definition": (
                    "Depression severity: 0–4 minimal; 5–9 mild; 10–14 moderate; "
                    "15–19 moderately severe; 20–27 severe. Score ≥ 10 triggers "
                    "psychiatric review before epilepsy surgery."
                ),
            },
            {
                "term": "GAD-7 (Generalised Anxiety Disorder-7)",
                "definition": (
                    "Anxiety severity: 0–4 minimal; 5–9 mild; 10–14 moderate; "
                    "15–21 severe. Score ≥ 10 clinically significant anxiety."
                ),
            },
            {
                "term": "Lateralization Hypothesis",
                "definition": (
                    "Neuropsychological inference of hemisphere most affected by "
                    "epileptic activity: left (language-dominant) / right / bilateral "
                    "/ non-lateralizing. Informs WADA / fMRI language mapping."
                ),
            },
            {
                "term": "Battery Types",
                "definition": (
                    "Full — complete 3-4 hour battery (all indices + Trail Making + "
                    "digit span); Screening — 60-90 min abbreviated battery; "
                    "Follow-up — re-test to track change over time or post-surgery."
                ),
            },
            {
                "term": "Trail Making Test",
                "definition": (
                    "Trail A (seconds) measures psychomotor speed and visual scanning; "
                    "Trail B measures executive function / cognitive flexibility. "
                    "Normal Trail A < 90 s; Trail B < 180 s (adult norms)."
                ),
            },
            {
                "term": "Standards",
                "definition": (
                    "NAEC (National Association of Epilepsy Centers) Level 4 "
                    "requirements; ILAE Neuropsychology Task Force; APA/WAIS-IV "
                    "normative interpretation; QOLIE-31 integration guidelines."
                ),
            },
        ],
        "abbreviations": {
            "MoCA": "Montreal Cognitive Assessment",
            "MMSE": "Mini-Mental State Examination",
            "PHQ-9": "Patient Health Questionnaire-9",
            "GAD-7": "Generalised Anxiety Disorder-7",
            "NAEC": "National Association of Epilepsy Centers",
            "ILAE": "International League Against Epilepsy",
            "WAIS": "Wechsler Adult Intelligence Scale",
            "TMT":  "Trail Making Test",
            "DRE":  "Drug-Resistant Epilepsy",
        },
    }
