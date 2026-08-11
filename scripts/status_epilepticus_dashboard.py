"""Status Epilepticus (SE) Dashboard — real clinical.db data.

Status Epilepticus is defined as continuous seizure activity ≥ 5 minutes, or
two or more seizures without full recovery of consciousness between them
(Neurocritical Care Society 2012 / ILAE 2015).

SE is a neurological emergency with mortality 7–39 % (DeLorenzo et al., 1996;
Rossetti et al., 2008). Early IV benzodiazepine treatment within 5 minutes
is the cornerstone of the Salzburg–NCS treatment algorithm.

Data sources (clinical.db):
  hospitalization     (115 rows) — admission_reason = 'status_epilepticus'
                                   or 'seizure_cluster', admission_type
  seizure_metadata    (71 rows)  — seizure types, syndrome, drug-resistance
  patients            (41 rows)  — age, gender
  medications         (9 rows)   — ASM profile
  outcomes                       — functional outcomes at discharge

All computations: raw SQL + Python stdlib only (no pandas, no numpy).
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

def _load_hospitalizations():
    """Return all hospitalization records as list of dicts."""
    con = _conn()
    rows = con.execute("SELECT * FROM hospitalization ORDER BY id").fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            # hospitalization table has unnamed columns; col[1]=patient_id, col[2]=fields_json
            raw = dict(r)
            # Try to extract fields regardless of column naming
            vals = list(raw.values())
            pid = vals[1] if len(vals) > 1 else None
            fields_raw = vals[2] if len(vals) > 2 else None
            date_raw = vals[3] if len(vals) > 3 else None
            if not pid or not fields_raw:
                continue
            d = json.loads(fields_raw)
            records.append({
                "patient_id": pid,
                "admission_date": d.get("admission_date") or date_raw,
                "discharge_date": d.get("discharge_date"),
                "admission_type": (d.get("admission_type") or "unknown").lower(),
                "admission_reason": (d.get("admission_reason") or "unknown").lower(),
                "ward": d.get("ward", "Unknown"),
                "los_days": float(d.get("length_of_stay_days") or 1),
                "discharge_disposition": (d.get("discharge_disposition") or "unknown").lower(),
                "readmission_30d": bool(d.get("readmission_within_30d", False)),
                "seizure_free_at_discharge": bool(d.get("seizure_free_at_discharge", False)),
                "complications": d.get("complications"),
                "insurance_type": (d.get("insurance_type") or "unknown").lower(),
                "total_cost_usd": float(d.get("total_cost_usd") or 0),
            })
        except Exception:
            pass
    return records


def _load_seizure_meta():
    """Return dict patient_id → parsed metadata."""
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, fields_json FROM seizure_metadata ORDER BY id"
    ).fetchall()
    con.close()
    result = {}
    for r in rows:
        try:
            d = json.loads(r["fields_json"])
            result[r["patient_id"]] = {
                "seizure_types": d.get("ilae_seizure_types", []),
                "syndrome": d.get("syndrome", ""),
                "drug_resp": (d.get("drug_responsiveness") or "").lower(),
                "etiology": (d.get("etiology") or "unknown").lower(),
                "onset_zone": d.get("onset_zone", ""),
                "disease_duration_years": float(d.get("disease_duration_years") or 0),
                "age_at_onset": float(d.get("age_at_onset") or 0),
            }
        except Exception:
            pass
    return result


def _load_patients():
    con = _conn()
    rows = con.execute("SELECT patient_id, age, gender FROM patients").fetchall()
    con.close()
    return {r["patient_id"]: {"age": r["age"], "gender": r["gender"]} for r in rows}


# ── SE classification helpers ─────────────────────────────────────────────────

_SE_REASONS = {"status_epilepticus", "seizure_cluster", "breakthrough_seizure"}


def _is_se_related(hosp: dict) -> bool:
    """True if admission reason is SE or seizure cluster."""
    return hosp["admission_reason"] in _se_reasons()


def _se_reasons():
    return {"status_epilepticus", "seizure_cluster"}


def _classify_se(hosp: dict) -> str:
    """Classify SE admission subtype."""
    reason = hosp["admission_reason"]
    if reason == "status_epilepticus":
        return "Convulsive SE (CSE)"
    if reason == "seizure_cluster":
        return "Seizure Cluster / NCSE"
    return "Emergency Seizure"


def _ncs_stage(los: float, complications: str | None) -> str:
    """Approximate NCS urgency stage from LOS + complications proxy."""
    if los >= 5 or (complications and complications.lower() not in ("null", "none", "")):
        return "Stage 3 — Refractory SE"
    if los >= 2:
        return "Stage 2 — Established SE"
    return "Stage 1 — Early SE"


# ── Public API ────────────────────────────────────────────────────────────────

def overview() -> dict:
    """Overview KPIs, admission-reason breakdown, NCS stage distribution,
    ward distribution, outcomes, and cost summary."""
    hosps = _load_hospitalizations()
    meta = _load_seizure_meta()
    patients = _load_patients()

    total = len(hosps)
    emergency = [h for h in hosps if h["admission_type"] == "emergency"]
    se_direct = [h for h in hosps if h["admission_reason"] == "status_epilepticus"]
    clusters = [h for h in hosps if h["admission_reason"] == "seizure_cluster"]
    se_all = [h for h in hosps if h["admission_reason"] in _se_reasons()]

    # NCS stage distribution across all SE/cluster admissions
    stage_ctr: Counter = Counter()
    for h in se_all:
        stage_ctr[_ncs_stage(h["los_days"], h["complications"])] += 1

    # Admission reason breakdown
    reason_ctr = Counter(h["admission_reason"] for h in hosps)

    # Ward distribution
    ward_ctr = Counter(h["ward"] for h in se_all)

    # Disposition at discharge for SE patients
    dispo_ctr = Counter(h["discharge_disposition"] for h in se_all)

    # LOS stats for SE
    los_vals = [h["los_days"] for h in se_all] if se_all else [0]
    avg_los = round(sum(los_vals) / len(los_vals), 1)

    # Seizure-free at discharge
    seizure_free = sum(1 for h in se_all if h["seizure_free_at_discharge"])

    # 30-day readmission
    readmission_n = sum(1 for h in se_all if h["readmission_30d"])

    # Total cost for SE admissions
    total_cost = sum(h["total_cost_usd"] for h in se_all)
    avg_cost = round(total_cost / len(se_all), 0) if se_all else 0

    # Complications
    comps = [h["complications"] for h in se_all
             if h["complications"] and str(h["complications"]).lower() not in ("none", "null")]
    comp_ctr = Counter(comps)

    # Etiology breakdown from seizure_meta for SE patients
    se_pids = {h["patient_id"] for h in se_all}
    etiology_ctr: Counter = Counter()
    for pid, m in meta.items():
        if pid in se_pids:
            etiology_ctr[m["etiology"] or "unknown"] += 1

    return {
        "kpis": {
            "total_admissions": total,
            "se_direct_admissions": len(se_direct),
            "seizure_cluster_admissions": len(clusters),
            "emergency_admissions": len(emergency),
            "se_related_total": len(se_all),
            "seizure_free_at_discharge": seizure_free,
            "seizure_free_pct": round(seizure_free / len(se_all) * 100, 1) if se_all else 0,
            "readmission_30d": readmission_n,
            "avg_los_days": avg_los,
            "avg_cost_usd": int(avg_cost),
        },
        "admission_reason_breakdown": [
            {"reason": k.replace("_", " ").title(), "count": v}
            for k, v in sorted(reason_ctr.items(), key=lambda x: -x[1])
        ],
        "ncs_stage_distribution": [
            {"stage": stage, "count": cnt}
            for stage, cnt in sorted(stage_ctr.items(), key=lambda x: -x[1])
        ],
        "ward_distribution": [
            {"ward": w, "count": c}
            for w, c in sorted(ward_ctr.items(), key=lambda x: -x[1])
        ],
        "discharge_disposition": [
            {"disposition": d.replace("_", " ").title(), "count": c}
            for d, c in sorted(dispo_ctr.items(), key=lambda x: -x[1])
        ],
        "complications": [
            {"complication": comp, "count": cnt}
            for comp, cnt in sorted(comp_ctr.items(), key=lambda x: -x[1])
        ],
        "etiology_breakdown": [
            {"etiology": e.replace("_", " ").title(), "count": c}
            for e, c in sorted(etiology_ctr.items(), key=lambda x: -x[1])
        ],
        "cost_summary": {
            "total_se_cost_usd": int(total_cost),
            "avg_se_cost_usd": int(avg_cost),
            "se_admissions_n": len(se_all),
        },
    }


def breakdown() -> dict:
    """Per-patient SE admission table, LOS histogram, insurance breakdown,
    drug-resistance prevalence in SE patients, and syndrome breakdown."""
    hosps = _load_hospitalizations()
    meta = _load_seizure_meta()
    patients = _load_patients()

    se_all = [h for h in hosps if h["admission_reason"] in _se_reasons()]

    # Build per-patient detail rows
    rows = []
    for h in se_all:
        pid = h["patient_id"]
        pt = patients.get(pid, {})
        m = meta.get(pid, {})
        rows.append({
            "patient_id": pid,
            "age": pt.get("age"),
            "gender": pt.get("gender"),
            "admission_reason": h["admission_reason"].replace("_", " ").title(),
            "admission_type": h["admission_type"].title(),
            "ward": h["ward"],
            "los_days": h["los_days"],
            "ncs_stage": _ncs_stage(h["los_days"], h["complications"]),
            "discharge_disposition": h["discharge_disposition"].replace("_", " ").title(),
            "seizure_free_at_discharge": h["seizure_free_at_discharge"],
            "readmission_30d": h["readmission_30d"],
            "complications": h["complications"],
            "cost_usd": int(h["total_cost_usd"]),
            "insurance_type": h["insurance_type"].replace("_", " ").title(),
            "syndrome": m.get("syndrome", ""),
            "etiology": (m.get("etiology") or "unknown").replace("_", " ").title(),
            "drug_resistant": "drug-resistant" in m.get("drug_resp", ""),
        })

    # Sort by LOS descending (severity proxy)
    rows.sort(key=lambda r: -r["los_days"])

    # LOS histogram
    los_buckets: Counter = Counter()
    for h in se_all:
        d = h["los_days"]
        if d <= 1:
            los_buckets["1 day"] += 1
        elif d <= 3:
            los_buckets["2-3 days"] += 1
        elif d <= 7:
            los_buckets["4-7 days"] += 1
        else:
            los_buckets[">7 days"] += 1

    # Insurance breakdown
    ins_ctr = Counter(h["insurance_type"] for h in se_all)

    # Drug-resistance in SE patients
    dr_in_se = sum(1 for r in rows if r["drug_resistant"])

    # Syndrome breakdown in SE patients
    syndrome_ctr = Counter(r["syndrome"] or "Unknown" for r in rows)

    # Month trend from admission_date
    month_ctr: Counter = Counter()
    for h in hosps:
        if h["admission_reason"] in _se_reasons():
            try:
                month = h["admission_date"][:7]  # YYYY-MM
                month_ctr[month] += 1
            except Exception:
                pass

    return {
        "patients": rows,
        "los_histogram": [
            {"bucket": k, "count": v}
            for k, v in [
                ("1 day", los_buckets.get("1 day", 0)),
                ("2-3 days", los_buckets.get("2-3 days", 0)),
                ("4-7 days", los_buckets.get("4-7 days", 0)),
                (">7 days", los_buckets.get(">7 days", 0)),
            ]
        ],
        "insurance_breakdown": [
            {"type": t.replace("_", " ").title(), "count": c}
            for t, c in sorted(ins_ctr.items(), key=lambda x: -x[1])
        ],
        "drug_resistant_in_se": {
            "count": dr_in_se,
            "total": len(rows),
            "pct": round(dr_in_se / len(rows) * 100, 1) if rows else 0,
        },
        "syndrome_breakdown": [
            {"syndrome": s, "count": c}
            for s, c in sorted(syndrome_ctr.items(), key=lambda x: -x[1])
        ],
        "monthly_trend": [
            {"month": m, "count": c}
            for m, c in sorted(month_ctr.items())
        ],
    }


def definitions() -> dict:
    """SE clinical definitions, NCS treatment algorithm stages, and references."""
    return {
        "term": "Status Epilepticus (SE)",
        "definition": (
            "Continuous seizure activity lasting ≥ 5 minutes, or two or more discrete "
            "seizures without full recovery of consciousness in between. "
            "(Neurocritical Care Society 2012; ILAE 2015 operational definition)"
        ),
        "incidence": "10–41 per 100,000 person-years; mortality 7–39% (DeLorenzo 1996)",
        "se_types": [
            {
                "type": "Convulsive SE (CSE)",
                "description": "Generalised tonic-clonic activity ≥ 5 min; most common and immediately life-threatening",
            },
            {
                "type": "Non-Convulsive SE (NCSE)",
                "description": "Electrographic seizure without prominent motor features; often missed clinically",
            },
            {
                "type": "Focal Motor SE",
                "description": "Sustained focal clonic activity (≥ 5 min) without impaired consciousness",
            },
            {
                "type": "Febrile SE",
                "description": "SE in children associated with fever, absent prior CNS infection; risk factor for temporal lobe epilepsy",
            },
            {
                "type": "Refractory SE (RSE)",
                "description": "SE that fails to respond to first- and second-line agents (benzodiazepine + antiseizure medication)",
            },
            {
                "type": "Super-Refractory SE (SRSE)",
                "description": "SE that continues ≥ 24 h after general anaesthetic therapy",
            },
        ],
        "ncs_treatment_stages": [
            {
                "stage": "Stage 1 — Early SE (0-5 min)",
                "agents": "IV/IM Lorazepam 4 mg · IM Midazolam 10 mg · IV Diazepam 10 mg",
                "target": "Terminate seizure within 5 minutes",
            },
            {
                "stage": "Stage 2 — Established SE (5-30 min)",
                "agents": "IV Levetiracetam 60 mg/kg (max 4.5 g) · IV Valproate 40 mg/kg · IV Fosphenytoin 20 mg PE/kg",
                "target": "Terminate seizure within 30 minutes",
            },
            {
                "stage": "Stage 3 — Refractory SE (30-60 min)",
                "agents": "IV Midazolam infusion · IV Propofol · IV Pentobarbital (burst-suppression target)",
                "target": "EEG burst-suppression pattern; ICU monitoring mandatory",
            },
        ],
        "prognostic_factors": [
            {"factor": "Age > 65", "impact": "Higher mortality, worse functional recovery"},
            {"factor": "Etiology (acute symptomatic > cryptogenic)", "impact": "Acute symptomatic worst prognosis"},
            {"factor": "Duration > 60 min", "impact": "Substantially increased morbidity/mortality"},
            {"factor": "Non-convulsive SE", "impact": "Diagnosis delayed → longer duration → worse outcomes"},
            {"factor": "Drug-resistant epilepsy", "impact": "Higher SE recurrence rate"},
        ],
        "abbreviations": {
            "SE": "Status Epilepticus",
            "CSE": "Convulsive Status Epilepticus",
            "NCSE": "Non-Convulsive Status Epilepticus",
            "RSE": "Refractory Status Epilepticus",
            "SRSE": "Super-Refractory Status Epilepticus",
            "NCS": "Neurocritical Care Society",
            "ILAE": "International League Against Epilepsy",
            "LOS": "Length of Stay",
            "ICU": "Intensive Care Unit",
            "AED / ASM": "Anti-Epileptic Drug / Anti-Seizure Medication",
        },
        "references": [
            "DeLorenzo RJ et al. Neurology 1996;46:1029-1035",
            "Rossetti AO et al. Neurocrit Care 2008;8:374-380",
            "Brophy GM et al. (NCS) Neurocrit Care 2012;17:3-23",
            "Trinka E et al. (ILAE) Epilepsia 2015;56:1515-1523",
            "Glauser T et al. Epilepsy Curr 2016;16:48-61",
            "NICE NG217 Epilepsies in children, young people and adults (2022)",
        ],
        "clinical_pearls": [
            "Early benzodiazepine within 5 min improves outcomes (Alldredge 2001)",
            "Continuous EEG monitoring mandatory for NCSE detection in ICU",
            "SE in DRE patients has higher recurrence — prioritise seizure-prevention protocols",
            "Ictal-interictal continuum: consult EEG in prolonged altered consciousness",
            "SUDEP risk elevated post-SE — trigger SUDEP counselling at discharge",
        ],
    }
