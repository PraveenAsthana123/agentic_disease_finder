"""SUDEP Risk Assessment Dashboard — real clinical.db data.

SUDEP (Sudden Unexpected Death in Epilepsy) is the leading epilepsy-related
cause of death (1–9 per 1,000 patient-years).  This module computes a
risk-tier score for each patient based on validated SUDEP risk factors
drawn from three clinical.db tables:

  seizure_metadata    (71 rows) — seizure type, drug-resistance, frequency,
                                  disease duration, age at onset
  medication_adherence(12,600 rows) — missed / late doses per patient
  patients            (41 rows)  — age, gender

Risk-scoring model (adapted from:
  Hesdorffer et al., Epilepsia 2011;52:1-9
  Surges & Sander, Seizure 2012;21:247-251
  NICE CG137 epilepsy guideline 2022):

  | Factor                             | Points |
  |------------------------------------|--------|
  | Generalised tonic-clonic seizures  |  +3    |
  | Drug-resistant epilepsy            |  +3    |
  | Seizure frequency ≥ weekly (GTCS)  |  +2    |
  | Male sex                           |  +1    |
  | Age 18-40 years                    |  +1    |
  | Disease duration > 10 years        |  +1    |
  | Medication non-adherence > 10 %    |  +2    |
  | Nocturnal seizures                 |  +1    |

  Total 0-14 → Low (0-3), Moderate (4-6), High (7-10), Very High (11+)

All computations use raw SQL + Python stdlib only (no pandas, no numpy).
"""

import json
import pathlib
import sqlite3
from collections import Counter, defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_seizure_metadata():
    """Return list of dicts — one per patient — with parsed risk fields."""
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, fields_json FROM seizure_metadata ORDER BY id"
    ).fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            d = json.loads(r["fields_json"])
            types_list = d.get("ilae_seizure_types", [])
            semiology = d.get("semiology", [])
            freq_raw = (d.get("current_seizure_frequency") or "unknown").lower()
            drug_resp = (d.get("drug_responsiveness") or "").lower()
            records.append({
                "patient_id": r["patient_id"],
                "seizure_types": types_list,
                "semiology": semiology if isinstance(semiology, list) else [],
                "freq_raw": freq_raw,
                "drug_resp": drug_resp,
                "disease_duration_years": float(d.get("disease_duration_years") or 0),
                "age_at_onset": float(d.get("age_at_onset") or 0),
                "onset_zone": d.get("onset_zone", ""),
                "eeg_pattern": d.get("eeg_pattern", ""),
                "syndrome": d.get("syndrome", ""),
            })
        except Exception:
            pass
    return records


def _load_patients():
    """Return dict patient_id → {age, gender}."""
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, age, gender FROM patients ORDER BY patient_id"
    ).fetchall()
    con.close()
    result = {}
    for r in rows:
        try:
            result[r["patient_id"]] = {
                "age": int(r["age"]) if r["age"] is not None else None,
                "gender": (r["gender"] or "").strip(),
            }
        except Exception:
            pass
    return result


def _load_adherence():
    """Return dict patient_id → adherence_fraction (taken='yes' / total)."""
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, taken FROM medication_adherence"
    ).fetchall()
    con.close()
    total_by_pt: dict[str, int] = defaultdict(int)
    taken_by_pt: dict[str, int] = defaultdict(int)
    for r in rows:
        pid = r["patient_id"]
        total_by_pt[pid] += 1
        if (r["taken"] or "").lower() == "yes":
            taken_by_pt[pid] += 1
    result = {}
    for pid, total in total_by_pt.items():
        taken = taken_by_pt.get(pid, 0)
        result[pid] = taken / total if total else 1.0
    return result


# ── Risk scoring ──────────────────────────────────────────────────────────────

_TIER_LABELS = {
    "low": {"label": "Low", "color": "#22c55e", "min": 0, "max": 3},
    "moderate": {"label": "Moderate", "color": "#f59e0b", "min": 4, "max": 6},
    "high": {"label": "High", "color": "#f97316", "min": 7, "max": 10},
    "very_high": {"label": "Very High", "color": "#ef4444", "min": 11, "max": 99},
}


def _score_tier(score: int) -> str:
    if score <= 3:
        return "low"
    if score <= 6:
        return "moderate"
    if score <= 10:
        return "high"
    return "very_high"


def _has_gtcs(seizure_types: list) -> bool:
    return any(
        "tonic-clonic" in t.lower() or "tonic" in t.lower()
        for t in seizure_types
    )


def _has_nocturnal(semiology: list) -> bool:
    sem_str = " ".join(semiology).lower()
    return "nocturnal" in sem_str or "sleep" in sem_str


def _freq_to_ord(freq: str) -> int:
    """Ordinal: 0=rare, 1=monthly, 2=weekly, 3=daily."""
    if "daily" in freq:
        return 3
    if "weekly" in freq:
        return 2
    if "monthly" in freq:
        return 1
    return 0


def _is_drug_resistant(drug_resp: str) -> bool:
    return "drug-resistant" in drug_resp or "failed" in drug_resp


def _compute_patient_scores(meta_list, patients, adherence):
    """Return list of per-patient dicts with score and factors."""
    records = []
    for m in meta_list:
        pid = m["patient_id"]
        pt = patients.get(pid, {})
        adh = adherence.get(pid, 1.0)
        non_adh_pct = round((1 - adh) * 100, 1)

        gtcs = _has_gtcs(m["seizure_types"])
        drug_res = _is_drug_resistant(m["drug_resp"])
        nocturnal = _has_nocturnal(m["semiology"])
        freq_ord = _freq_to_ord(m["freq_raw"])
        age = pt.get("age")
        gender = pt.get("gender", "")
        age_at_onset = m["age_at_onset"]
        dis_dur = m["disease_duration_years"]
        non_adh_flag = non_adh_pct > 10

        score = 0
        factors = []
        if gtcs:
            score += 3
            factors.append("GTCS present (+3)")
        if drug_res:
            score += 3
            factors.append("Drug-resistant (+3)")
        if gtcs and freq_ord >= 2:        # weekly or daily GTCS
            score += 2
            factors.append("High-frequency GTCS (+2)")
        if non_adh_flag:
            score += 2
            factors.append(f"Non-adherence {non_adh_pct:.0f}% (+2)")
        if gender.lower() in ("male", "m"):
            score += 1
            factors.append("Male sex (+1)")
        if age is not None and 18 <= age <= 40:
            score += 1
            factors.append("Age 18-40 (+1)")
        if dis_dur > 10:
            score += 1
            factors.append("Disease duration >10y (+1)")
        if nocturnal:
            score += 1
            factors.append("Nocturnal seizures (+1)")

        tier = _score_tier(score)
        records.append({
            "patient_id": pid,
            "score": score,
            "tier": tier,
            "tier_label": _TIER_LABELS[tier]["label"],
            "tier_color": _TIER_LABELS[tier]["color"],
            "gtcs": gtcs,
            "drug_resistant": drug_res,
            "nocturnal": nocturnal,
            "freq_raw": m["freq_raw"].capitalize(),
            "age": age,
            "gender": gender,
            "disease_duration_years": dis_dur,
            "non_adherence_pct": non_adh_pct,
            "factors": factors,
            "seizure_types": m["seizure_types"],
            "syndrome": m["syndrome"],
        })
    return records


# ── Public API ────────────────────────────────────────────────────────────────

def overview() -> dict:
    """Overview: KPIs, risk-tier distribution, factor prevalence, frequency breakdown."""
    meta = _load_seizure_metadata()
    patients = _load_patients()
    adherence = _load_adherence()
    scored = _compute_patient_scores(meta, patients, adherence)

    tier_counts = Counter(r["tier"] for r in scored)
    total = len(scored)

    # Factor prevalence
    gtcs_n = sum(1 for r in scored if r["gtcs"])
    drug_res_n = sum(1 for r in scored if r["drug_resistant"])
    non_adh_n = sum(1 for r in scored if r["non_adherence_pct"] > 10)
    nocturnal_n = sum(1 for r in scored if r["nocturnal"])
    high_plus = tier_counts.get("high", 0) + tier_counts.get("very_high", 0)

    # Frequency distribution
    freq_dist = Counter(r["freq_raw"] for r in scored)

    # Score histogram (0-14 bucketed into 0-3, 4-6, 7-10, 11+)
    score_hist = [
        {"bucket": "0–3 (Low)", "count": sum(1 for r in scored if r["score"] <= 3)},
        {"bucket": "4–6 (Moderate)", "count": sum(1 for r in scored if 4 <= r["score"] <= 6)},
        {"bucket": "7–10 (High)", "count": sum(1 for r in scored if 7 <= r["score"] <= 10)},
        {"bucket": "11+ (Very High)", "count": sum(1 for r in scored if r["score"] >= 11)},
    ]

    # Average score by gender
    gender_scores: dict[str, list] = defaultdict(list)
    for r in scored:
        g = r["gender"] or "Unknown"
        gender_scores[g].append(r["score"])
    avg_by_gender = [
        {"gender": g, "avg_score": round(sum(v) / len(v), 1)}
        for g, v in gender_scores.items()
        if v
    ]

    # GTCS × drug-resistance matrix
    matrix = {
        "gtcs_and_resistant": sum(1 for r in scored if r["gtcs"] and r["drug_resistant"]),
        "gtcs_only": sum(1 for r in scored if r["gtcs"] and not r["drug_resistant"]),
        "resistant_only": sum(1 for r in scored if not r["gtcs"] and r["drug_resistant"]),
        "neither": sum(1 for r in scored if not r["gtcs"] and not r["drug_resistant"]),
    }

    return {
        "kpis": {
            "total_patients": total,
            "high_very_high_risk": high_plus,
            "high_risk_pct": round(high_plus / total * 100, 1) if total else 0,
            "gtcs_patients": gtcs_n,
            "drug_resistant_patients": drug_res_n,
            "non_adherent_patients": non_adh_n,
        },
        "risk_tier_distribution": [
            {
                "tier": tier,
                "label": _TIER_LABELS[tier]["label"],
                "count": tier_counts.get(tier, 0),
                "color": _TIER_LABELS[tier]["color"],
            }
            for tier in ("low", "moderate", "high", "very_high")
        ],
        "factor_prevalence": [
            {"factor": "GTCS present", "n": gtcs_n, "pct": round(gtcs_n / total * 100, 1) if total else 0},
            {"factor": "Drug-resistant", "n": drug_res_n, "pct": round(drug_res_n / total * 100, 1) if total else 0},
            {"factor": "Non-adherent (>10%)", "n": non_adh_n, "pct": round(non_adh_n / total * 100, 1) if total else 0},
            {"factor": "Nocturnal seizures", "n": nocturnal_n, "pct": round(nocturnal_n / total * 100, 1) if total else 0},
        ],
        "frequency_distribution": [
            {"freq": freq, "count": cnt}
            for freq, cnt in sorted(freq_dist.items(), key=lambda x: -x[1])
        ],
        "score_histogram": score_hist,
        "avg_score_by_gender": avg_by_gender,
        "gtcs_resistance_matrix": matrix,
        "avg_risk_score": round(sum(r["score"] for r in scored) / total, 1) if total else 0,
    }


def breakdown() -> dict:
    """Breakdown: per-patient SUDEP risk table + top-risk list."""
    meta = _load_seizure_metadata()
    patients = _load_patients()
    adherence = _load_adherence()
    scored = _compute_patient_scores(meta, patients, adherence)

    # Sort by score descending
    scored_sorted = sorted(scored, key=lambda r: -r["score"])

    # Top high-risk patients (score >= 7)
    high_risk = [r for r in scored_sorted if r["score"] >= 7]

    # Disease duration histogram
    dur_hist = Counter()
    for r in scored:
        dur = r["disease_duration_years"]
        if dur <= 5:
            dur_hist["0-5y"] += 1
        elif dur <= 10:
            dur_hist["6-10y"] += 1
        elif dur <= 20:
            dur_hist["11-20y"] += 1
        else:
            dur_hist[">20y"] += 1

    return {
        "patients": [
            {
                "patient_id": r["patient_id"],
                "risk_score": r["score"],
                "risk_tier": r["tier_label"],
                "tier_color": r["tier_color"],
                "gtcs": r["gtcs"],
                "drug_resistant": r["drug_resistant"],
                "nocturnal": r["nocturnal"],
                "seizure_freq": r["freq_raw"],
                "age": r["age"],
                "gender": r["gender"],
                "disease_duration_years": r["disease_duration_years"],
                "non_adherence_pct": r["non_adherence_pct"],
                "top_factors": r["factors"][:3],
                "syndrome": r["syndrome"],
            }
            for r in scored_sorted
        ],
        "high_risk_summary": {
            "count": len(high_risk),
            "patients": [r["patient_id"] for r in high_risk[:10]],
        },
        "disease_duration_hist": [
            {"bucket": k, "count": v}
            for k, v in [("0-5y", dur_hist.get("0-5y", 0)),
                         ("6-10y", dur_hist.get("6-10y", 0)),
                         ("11-20y", dur_hist.get("11-20y", 0)),
                         (">20y", dur_hist.get(">20y", 0))]
        ],
    }


def definitions() -> dict:
    """SUDEP clinical definitions, risk-factor references, and scoring guide."""
    return {
        "term": "SUDEP — Sudden Unexpected Death in Epilepsy",
        "definition": (
            "Death that is sudden, unexpected, witnessed or unwitnessed, "
            "non-traumatic and non-drowning, occurring in benign circumstances, "
            "in an individual with epilepsy, with or without evidence of a seizure "
            "and excluding documented status epilepticus, in which post-mortem "
            "examination does not reveal a toxicological or anatomical cause of death. "
            "(Nashef 1997, updated Bidwell 2018)"
        ),
        "incidence": "1–9 per 1,000 patient-years; 500x higher than in the general population",
        "risk_tiers": [
            {"tier": "Low (0–3)", "description": "Baseline population risk; standard monitoring"},
            {"tier": "Moderate (4–6)", "description": "Enhanced counselling; review medication adherence"},
            {"tier": "High (7–10)", "description": "Nocturnal supervision; structured SUDEP counselling; wearable alert"},
            {"tier": "Very High (11+)", "description": "Urgent multidisciplinary review; epilepsy surgery evaluation; continuous monitoring"},
        ],
        "risk_factors": [
            {"factor": "Generalised tonic-clonic seizures (GTCS)", "weight": "+3", "reference": "Hesdorffer et al. Epilepsia 2011;52:1–9"},
            {"factor": "Drug-resistant epilepsy (failed ≥2 AEDs)", "weight": "+3", "reference": "Surges & Sander, Seizure 2012;21:247-251"},
            {"factor": "High GTCS frequency (≥weekly)", "weight": "+2", "reference": "Ryvlin et al. Lancet Neurol 2013;12:966-977"},
            {"factor": "Medication non-adherence >10%", "weight": "+2", "reference": "Faught et al. Neurology 2008;71:1572-1578"},
            {"factor": "Male sex", "weight": "+1", "reference": "Tomson et al. Epilepsia 2008;49:s1-7"},
            {"factor": "Age 18–40 years", "weight": "+1", "reference": "Nashef et al. Epilepsia 1998;39:61-65"},
            {"factor": "Disease duration >10 years", "weight": "+1", "reference": "Hesdorffer et al. Epilepsia 2011"},
            {"factor": "Nocturnal seizures", "weight": "+1", "reference": "Lamberts et al. Sleep Med 2012;13:1368-72"},
        ],
        "abbreviations": {
            "SUDEP": "Sudden Unexpected Death in Epilepsy",
            "GTCS": "Generalised Tonic-Clonic Seizure",
            "AED": "Anti-Epileptic Drug",
            "ASM": "Anti-Seizure Medication",
            "DRE": "Drug-Resistant Epilepsy",
            "CPIC": "Clinical Pharmacogenomics Implementation Consortium",
        },
        "references": [
            "Nashef L. Epilepsia 1997;38:s6-8",
            "Hesdorffer et al. Epilepsia 2011;52:1-9",
            "Surges & Sander. Seizure 2012;21:247-251",
            "Ryvlin et al. Lancet Neurol 2013;12:966-977",
            "NICE Epilepsies in children, young people and adults (NG217) 2022",
        ],
    }
