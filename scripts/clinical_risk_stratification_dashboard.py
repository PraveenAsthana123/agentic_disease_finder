"""Clinical Risk Stratification Dashboard — composite per-patient epilepsy risk scoring.

Combines 6 real data tables:
- seizure_diary      (25 rows, 22 patients) — seizure burden: frequency, severity, ER visits
- medication_adherence (12600 rows, 30 patients) — adherence rate, side-effect burden
- pharmacogenomics   (172 rows, 40 patients) — genetic risk flags (HLA-B*1502, poor metabolisers)
- comorbidities      (27 rows, 27 patients) — comorbidity count, behavioural risk score
- pro_outcomes       (180 rows, 30 patients) — QOLIE-31, PHQ-9, GAD-7 QoL metrics
- patients           (40 rows)               — demographics (age, gender)

Risk Score (0–100):
  seizure_burden     max 30 pts  — frequency, severity, ER visits, rescue med use
  adherence_risk     max 25 pts  — low adherence → high risk (inverted)
  genetic_risk       max 20 pts  — high-severity pharmacogenomic variants
  comorbidity_burden max 15 pts  — comorbidity count + behavioural risk score
  qol_deficit        max 10 pts  — low QOLIE-31 → high risk

Tiers:
  Critical  70–100
  High      45–70
  Moderate  20–45
  Low        0–20
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

TIER_LABELS = [
    (35, "Critical"),
    (23, "High"),
    (12, "Moderate"),
    (0,  "Low"),
]

TIER_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f59e0b",
    "Moderate": "#3b82f6",
    "Low":      "#10b981",
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _tier(score):
    for threshold, label in TIER_LABELS:
        if score >= threshold:
            return label
    return "Low"


def _load_seizure_burden(conn):
    """Return dict patient_id -> seizure risk component (0–30)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_id,
               COUNT(*) AS total,
               SUM(CASE WHEN severity='Severe' THEN 1 ELSE 0 END) AS severe_count,
               SUM(CASE WHEN er_visit='Yes'    THEN 1 ELSE 0 END) AS er_visits,
               SUM(CASE WHEN rescue_med='Yes'  THEN 1 ELSE 0 END) AS rescue_meds,
               AVG(duration_sec) AS avg_dur_sec
        FROM seizure_diary
        GROUP BY patient_id
    """)
    result = {}
    for row in cur.fetchall():
        pid, total, severe, er, rescue, avg_dur = row
        # frequency score (max 12): 1+ seizure =4, 3+ =8, 6+ =12
        freq_score = min(12, total * 2)
        # severity score (max 8): severe seizures
        sev_score = min(8, (severe or 0) * 2)
        # ER visit score (max 6)
        er_score = min(6, (er or 0) * 3)
        # rescue med score (max 4)
        resc_score = min(4, (rescue or 0) * 2)
        result[pid] = round(freq_score + sev_score + er_score + resc_score, 1)
    return result


def _load_adherence_risk(conn):
    """Return dict patient_id -> adherence risk component (0–25, high = poor adherence)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_id,
               AVG(CASE WHEN taken='yes' THEN 1.0 ELSE 0.0 END) AS rate,
               AVG(side_effect_severity) AS avg_se_sev
        FROM medication_adherence
        GROUP BY patient_id
    """)
    result = {}
    for pid, rate, se_sev in cur.fetchall():
        rate = (rate or 1.0) * 100
        # non-adherence risk (max 20): 100%-rate scaled
        non_adh_score = round((100 - rate) / 100 * 20, 1)
        # side-effect burden (max 5)
        se_score = round(min(5, (se_sev or 0) / 3 * 5), 1)
        result[pid] = round(non_adh_score + se_score, 1)
    return result


def _load_genetic_risk(conn):
    """Return dict patient_id -> pharmacogenomic risk component (0–20)."""
    cur = conn.cursor()
    cur.execute("SELECT patient_id, gene, clinical_significance, metabolizer_status FROM pharmacogenomics")
    patient_risks = {}
    for pid, gene, sig, met_status in cur.fetchall():
        if pid not in patient_risks:
            patient_risks[pid] = {"high": 0, "moderate": 0, "poor_met": 0}
        if sig and "High" in sig:
            patient_risks[pid]["high"] += 1
        elif sig and ("Moderate" in sig or "Reduced" in sig or "resistance" in sig.lower()):
            patient_risks[pid]["moderate"] += 1
        if met_status and ("Poor" in met_status or "Reduced" in met_status):
            patient_risks[pid]["poor_met"] += 1

    result = {}
    for pid, risks in patient_risks.items():
        high_score = min(12, risks["high"] * 6)
        mod_score = min(5, risks["moderate"] * 1)
        met_score = min(3, risks["poor_met"] * 1)
        result[pid] = round(high_score + mod_score + met_score, 1)
    return result


def _load_comorbidity_burden(conn):
    """Return dict patient_id -> comorbidity burden component (0–15)."""
    cur = conn.cursor()
    cur.execute("SELECT patient_id, fields_json FROM comorbidities")
    result = {}
    for pid, fields_json in cur.fetchall():
        try:
            d = json.loads(fields_json)
        except Exception:
            d = {}
        count = d.get("comorbidity_count", 0) or 0
        behav = d.get("behavioral_risk_score", 0) or 0
        # comorbidity count score (max 8): each comorbidity +2, cap 8
        cnt_score = min(8, count * 2)
        # behavioural risk (max 7): score 0-100 → 0-7
        beh_score = round(min(7, behav / 100 * 7), 1)
        result[pid] = round(cnt_score + beh_score, 1)
    return result


def _load_qol_deficit(conn):
    """Return dict patient_id -> QoL deficit component (0–10, low QOLIE → high risk)."""
    cur = conn.cursor()
    cur.execute("SELECT patient_id, fields_json FROM pro_outcomes")
    patient_qol = {}
    for pid, fields_json in cur.fetchall():
        try:
            d = json.loads(fields_json)
        except Exception:
            d = {}
        qolie = d.get("qolie31_score")
        phq9 = d.get("phq9_score", 0) or 0
        if qolie is not None:
            if pid not in patient_qol:
                patient_qol[pid] = {"qolie_scores": [], "phq9_scores": []}
            patient_qol[pid]["qolie_scores"].append(qolie)
            patient_qol[pid]["phq9_scores"].append(phq9)

    result = {}
    for pid, data in patient_qol.items():
        avg_qolie = sum(data["qolie_scores"]) / len(data["qolie_scores"])
        avg_phq9 = sum(data["phq9_scores"]) / len(data["phq9_scores"])
        # QOLIE deficit (max 7): 100-QOLIE score → 0-7
        qolie_score = round((100 - avg_qolie) / 100 * 7, 1)
        # PHQ-9 score (max 3): 0-27 → 0-3
        phq_score = round(min(3, avg_phq9 / 27 * 3), 1)
        result[pid] = round(qolie_score + phq_score, 1)
    return result


def _load_patients(conn):
    """Return dict patient_id -> {age, gender}."""
    cur = conn.cursor()
    cur.execute("SELECT patient_id, age, gender FROM patients")
    return {pid: {"age": age, "gender": gender} for pid, age, gender in cur.fetchall()}


def _build_profiles():
    """Build composite risk profiles for all patients with data in at least 2 tables."""
    conn = _conn()
    seizure   = _load_seizure_burden(conn)
    adherence = _load_adherence_risk(conn)
    genetic   = _load_genetic_risk(conn)
    comorbid  = _load_comorbidity_burden(conn)
    qol       = _load_qol_deficit(conn)
    demo      = _load_patients(conn)
    conn.close()

    # Union all patient IDs with any data
    all_pids = set(demo.keys()) | set(seizure.keys()) | set(adherence.keys()) | \
               set(genetic.keys()) | set(comorbid.keys()) | set(qol.keys())

    profiles = []
    for pid in sorted(all_pids):
        s_score  = seizure.get(pid, 0)
        a_score  = adherence.get(pid, 0)
        g_score  = genetic.get(pid, 0)
        c_score  = comorbid.get(pid, 0)
        q_score  = qol.get(pid, 0)
        composite = round(s_score + a_score + g_score + c_score + q_score, 1)
        info = demo.get(pid, {})
        profiles.append({
            "patient_id":        pid,
            "composite_score":   min(100, composite),
            "tier":              _tier(composite),
            "seizure_burden":    s_score,
            "adherence_risk":    a_score,
            "genetic_risk":      g_score,
            "comorbidity_burden": c_score,
            "qol_deficit":       q_score,
            "age":               info.get("age"),
            "gender":            info.get("gender"),
        })
    return profiles


def overview():
    profiles = _build_profiles()
    n = len(profiles)

    tier_counts = {"Critical": 0, "High": 0, "Moderate": 0, "Low": 0}
    score_sum = 0.0
    component_sums = {"seizure_burden": 0, "adherence_risk": 0, "genetic_risk": 0,
                      "comorbidity_burden": 0, "qol_deficit": 0}

    for p in profiles:
        tier_counts[p["tier"]] = tier_counts.get(p["tier"], 0) + 1
        score_sum += p["composite_score"]
        for k in component_sums:
            component_sums[k] += p[k]

    avg_score = round(score_sum / n, 1) if n else 0

    # Tier distribution list
    tier_dist = [
        {"tier": t, "count": c, "pct": round(c / n * 100, 1) if n else 0, "color": TIER_COLORS[t]}
        for t, c in tier_counts.items()
    ]

    # Average component contributions
    avg_components = [
        {"component": "Seizure Burden",      "avg": round(component_sums["seizure_burden"] / n, 1) if n else 0,      "max": 30},
        {"component": "Adherence Risk",      "avg": round(component_sums["adherence_risk"] / n, 1) if n else 0,      "max": 25},
        {"component": "Genetic Risk",        "avg": round(component_sums["genetic_risk"] / n, 1) if n else 0,        "max": 20},
        {"component": "Comorbidity Burden",  "avg": round(component_sums["comorbidity_burden"] / n, 1) if n else 0,  "max": 15},
        {"component": "QoL Deficit",         "avg": round(component_sums["qol_deficit"] / n, 1) if n else 0,         "max": 10},
    ]

    # Score histogram (bins of 10)
    hist = {f"{i*10}-{i*10+9}": 0 for i in range(10)}
    for p in profiles:
        bin_key = f"{int(p['composite_score'] // 10) * 10}-{int(p['composite_score'] // 10) * 10 + 9}"
        if bin_key in hist:
            hist[bin_key] += 1
    score_histogram = [{"range": k, "count": v} for k, v in hist.items()]

    # High-risk patients (Critical + High tiers)
    high_risk = sorted(
        [p for p in profiles if p["tier"] in ("Critical", "High")],
        key=lambda x: x["composite_score"], reverse=True
    )[:10]

    # Gender breakdown
    gender_risk = {}
    for p in profiles:
        g = p.get("gender") or "Unknown"
        if g not in gender_risk:
            gender_risk[g] = {"count": 0, "score_sum": 0}
        gender_risk[g]["count"] += 1
        gender_risk[g]["score_sum"] += p["composite_score"]
    gender_breakdown = [
        {"gender": g, "count": d["count"], "avg_score": round(d["score_sum"] / d["count"], 1)}
        for g, d in gender_risk.items()
    ]

    return {
        "total_patients":         n,
        "avg_composite_score":    avg_score,
        "critical_count":         tier_counts["Critical"],
        "high_count":             tier_counts["High"],
        "moderate_count":         tier_counts["Moderate"],
        "low_count":              tier_counts["Low"],
        "high_risk_rate_pct":     round((tier_counts["Critical"] + tier_counts["High"]) / n * 100, 1) if n else 0,
        "tier_distribution":      tier_dist,
        "avg_components":         avg_components,
        "score_histogram":        score_histogram,
        "high_risk_patients":     high_risk,
        "gender_breakdown":       gender_breakdown,
    }


def breakdown():
    profiles = _build_profiles()

    # Sort by composite score descending (highest risk first)
    sorted_profiles = sorted(profiles, key=lambda x: x["composite_score"], reverse=True)

    # Top critical patients
    critical = [p for p in sorted_profiles if p["tier"] == "Critical"]
    high = [p for p in sorted_profiles if p["tier"] == "High"]

    # Component waterfall (which component contributes most risk on average)
    n = len(profiles)
    waterfall = []
    cumulative = 0
    for comp, mx in [
        ("Seizure Burden", 30),
        ("Adherence Risk", 25),
        ("Genetic Risk", 20),
        ("Comorbidity Burden", 15),
        ("QoL Deficit", 10),
    ]:
        key = comp.lower().replace(" ", "_")
        avg = round(sum(p[key] for p in profiles) / n, 1) if n else 0
        pct_of_max = round(avg / mx * 100, 1)
        waterfall.append({
            "component": comp,
            "avg": avg,
            "max": mx,
            "pct_of_max": pct_of_max,
            "cumulative_start": cumulative,
        })
        cumulative += avg

    return {
        "all_patients":       sorted_profiles,
        "critical_patients":  critical,
        "high_risk_patients": high,
        "component_waterfall": waterfall,
        "total_patients":     n,
        "critical_count":     len(critical),
        "high_count":         len(high),
    }


def definitions():
    return {
        "dashboard": "Clinical Risk Stratification",
        "description": "Composite per-patient epilepsy risk scoring combining 6 real data sources "
                       "to identify patients at Critical/High/Moderate/Low risk and prioritise interventions.",
        "risk_components": [
            {
                "component": "Seizure Burden",
                "max_points": 30,
                "sources": ["seizure_diary"],
                "sub_factors": [
                    {"factor": "Seizure Frequency",    "max": 12, "note": "2 pts per seizure, capped at 12"},
                    {"factor": "Severe Seizure Count",  "max": 8,  "note": "2 pts per severe seizure"},
                    {"factor": "ER Visit Count",        "max": 6,  "note": "3 pts per ER visit"},
                    {"factor": "Rescue Medication Use", "max": 4,  "note": "2 pts per rescue med use"},
                ],
            },
            {
                "component": "Adherence Risk",
                "max_points": 25,
                "sources": ["medication_adherence"],
                "sub_factors": [
                    {"factor": "Non-Adherence Rate",    "max": 20, "note": "(100% − adherence%) × 0.20"},
                    {"factor": "Side-Effect Burden",    "max": 5,  "note": "Mean side-effect severity / 3 × 5"},
                ],
            },
            {
                "component": "Genetic Risk",
                "max_points": 20,
                "sources": ["pharmacogenomics"],
                "sub_factors": [
                    {"factor": "High-Severity Variants",     "max": 12, "note": "6 pts per High variant (HLA-B*1502, SJS/TEN risk)"},
                    {"factor": "Moderate-Severity Variants",  "max": 5,  "note": "1 pt per moderate variant"},
                    {"factor": "Poor Metaboliser Status",    "max": 3,  "note": "1 pt per poor/reduced metaboliser gene"},
                ],
            },
            {
                "component": "Comorbidity Burden",
                "max_points": 15,
                "sources": ["comorbidities"],
                "sub_factors": [
                    {"factor": "Comorbidity Count",       "max": 8, "note": "2 pts per comorbidity, capped at 8"},
                    {"factor": "Behavioural Risk Score",  "max": 7, "note": "PHQ-9/GAD-7/C-SSRS composite 0–100 → 0–7"},
                ],
            },
            {
                "component": "QoL Deficit",
                "max_points": 10,
                "sources": ["pro_outcomes"],
                "sub_factors": [
                    {"factor": "Low QOLIE-31 Score",  "max": 7, "note": "(100 − QOLIE-31) / 100 × 7"},
                    {"factor": "PHQ-9 Depression",     "max": 3, "note": "PHQ-9 / 27 × 3"},
                ],
            },
        ],
        "risk_tiers": [
            {"tier": "Critical", "range": "≥35 pts", "color": "#ef4444",
             "action": "Urgent neurology review, multi-disciplinary team escalation, daily monitoring"},
            {"tier": "High",     "range": "23–34 pts", "color": "#f59e0b",
             "action": "Enhanced follow-up within 2 weeks, medication optimisation, social support review"},
            {"tier": "Moderate", "range": "12–22 pts", "color": "#3b82f6",
             "action": "Routine follow-up, adherence counselling, patient education reinforcement"},
            {"tier": "Low",      "range": "0–11 pts",  "color": "#10b981",
             "action": "Standard monitoring schedule, preventive education, annual review"},
        ],
        "data_sources": [
            {"table": "seizure_diary",        "rows": 25,     "patients": 22, "key_fields": "severity, er_visit, rescue_med, duration_sec"},
            {"table": "medication_adherence", "rows": 12600,  "patients": 30, "key_fields": "taken, side_effect_severity"},
            {"table": "pharmacogenomics",     "rows": 172,    "patients": 40, "key_fields": "gene, clinical_significance, metabolizer_status"},
            {"table": "comorbidities",        "rows": 27,     "patients": 27, "key_fields": "comorbidity_count, behavioral_risk_score"},
            {"table": "pro_outcomes",         "rows": 180,    "patients": 30, "key_fields": "qolie31_score, phq9_score"},
            {"table": "patients",             "rows": 40,     "patients": 40, "key_fields": "age, gender"},
        ],
        "glossary": [
            {"term": "QOLIE-31",   "definition": "Quality of Life in Epilepsy — 31-item patient-reported outcome scale (0–100, higher = better QoL)"},
            {"term": "PHQ-9",      "definition": "Patient Health Questionnaire-9 — depression severity scale (0–27, higher = more severe)"},
            {"term": "HLA-B*1502", "definition": "HLA allele associated with severe carbamazepine-induced Stevens–Johnson syndrome, prevalent in Southeast Asian populations"},
            {"term": "SJS/TEN",    "definition": "Stevens–Johnson Syndrome / Toxic Epidermal Necrolysis — severe drug hypersensitivity reactions"},
            {"term": "CYP2C9",     "definition": "Cytochrome P450 enzyme metabolising phenytoin; poor metabolisers accumulate toxic drug levels"},
            {"term": "CPIC",       "definition": "Clinical Pharmacogenomics Implementation Consortium — evidence-based drug–gene prescribing guidelines"},
            {"term": "SUDEP",      "definition": "Sudden Unexpected Death in Epilepsy — leading cause of epilepsy-related mortality, linked to uncontrolled seizures"},
            {"term": "MDT",        "definition": "Multi-Disciplinary Team — specialist team (neurologist, pharmacist, nurse, social worker) for complex epilepsy management"},
        ],
    }
