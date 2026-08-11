"""Drug-Resistant Epilepsy (DRE) Dashboard — real clinical.db data.

Drug-Resistant Epilepsy (DRE) is defined by the ILAE (2010) as failure of
adequate trials of two tolerated, appropriately chosen and used antiepileptic
drug (AED) schedules (whether as monotherapies or in combination) to achieve
sustained seizure freedom.

DRE affects ~30% of all epilepsy patients and is a major driver of:
  • surgical candidacy evaluation
  • quality-of-life impairment
  • increased SUDEP risk
  • healthcare resource utilisation

Reference: Kwan P et al., Epilepsia 2010;51(6):1069-1077 (ILAE Task Force).

Data sources (clinical.db):
  seizure_metadata     (71 rows) — drug_responsiveness, aed_trials,
                                   ilae_seizure_types, disease_duration,
                                   onset_zone, syndrome
  medication_adherence (12,600 rows) — adherence per patient
  patients             (41 rows) — age, gender

DRE classification criteria used in this module
  (derived from fields_json in seizure_metadata):
  • drug_responsiveness contains "drug-resistant" OR "failed"
  • OR aed_trials count ≥ 2 AND seizures not controlled (freq != remission)

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


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_seizure_metadata():
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, fields_json FROM seizure_metadata ORDER BY id"
    ).fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            d = json.loads(r["fields_json"])
            drug_resp = (d.get("drug_responsiveness") or "").lower()
            aed_raw = d.get("aed_trials") or d.get("aeds_tried") or []
            if isinstance(aed_raw, str):
                # may be comma-separated
                aed_raw = [a.strip() for a in aed_raw.split(",") if a.strip()]
            aed_count = len(aed_raw) if isinstance(aed_raw, list) else 0
            freq_raw = (d.get("current_seizure_frequency") or "unknown").lower()
            onset_zone = (d.get("onset_zone") or "unknown").strip()
            syndrome = (d.get("syndrome") or "").strip()
            dis_dur = float(d.get("disease_duration_years") or 0)
            age_onset = float(d.get("age_at_onset") or 0)
            ilae_types = d.get("ilae_seizure_types", [])
            semiology = d.get("semiology", [])
            records.append({
                "patient_id": r["patient_id"],
                "drug_resp": drug_resp,
                "aed_trials": aed_raw if isinstance(aed_raw, list) else [],
                "aed_count": aed_count,
                "freq_raw": freq_raw,
                "onset_zone": onset_zone,
                "syndrome": syndrome,
                "disease_duration_years": dis_dur,
                "age_at_onset": age_onset,
                "ilae_types": ilae_types,
                "semiology": semiology if isinstance(semiology, list) else [],
            })
        except Exception:
            pass
    return records


def _load_patients():
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
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, taken FROM medication_adherence"
    ).fetchall()
    con.close()
    total_by_pt: dict = defaultdict(int)
    taken_by_pt: dict = defaultdict(int)
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


# ── DRE classification ─────────────────────────────────────────────────────────

def _is_dre(drug_resp: str, aed_count: int, freq_raw: str) -> bool:
    """ILAE 2010: drug-resistant if ≥2 AED trials failed OR explicitly labelled."""
    if "drug-resistant" in drug_resp or "failed" in drug_resp:
        return True
    if aed_count >= 2 and "remission" not in freq_raw and "controlled" not in freq_raw:
        return True
    return False


def _surgical_candidate(onset_zone: str, dre: bool, dis_dur: float) -> bool:
    """Simple proxy for surgical candidacy: focal onset + DRE + duration ≥ 2 years."""
    focal_zones = {"temporal", "frontal", "parietal", "occipital", "mesial", "lateral"}
    zone_lower = onset_zone.lower()
    is_focal = any(z in zone_lower for z in focal_zones)
    return dre and is_focal and dis_dur >= 2.0


def _seizure_control_label(freq_raw: str) -> str:
    if "remission" in freq_raw or "controlled" in freq_raw or "seizure-free" in freq_raw:
        return "Seizure-Free"
    if "daily" in freq_raw:
        return "Daily"
    if "weekly" in freq_raw:
        return "Weekly"
    if "monthly" in freq_raw:
        return "Monthly"
    if "yearly" in freq_raw or "rare" in freq_raw:
        return "Rare"
    return "Unknown"


def _classify_patients(meta_list, patients, adherence):
    records = []
    for m in meta_list:
        pid = m["patient_id"]
        pt = patients.get(pid, {})
        adh = adherence.get(pid, 1.0)
        non_adh_pct = round((1 - adh) * 100, 1)

        dre = _is_dre(m["drug_resp"], m["aed_count"], m["freq_raw"])
        surgical = _surgical_candidate(m["onset_zone"], dre, m["disease_duration_years"])
        control = _seizure_control_label(m["freq_raw"])

        # AED burden bucket
        ac = m["aed_count"]
        if ac == 0:
            aed_bucket = "0 (naïve)"
        elif ac == 1:
            aed_bucket = "1 (mono)"
        elif ac == 2:
            aed_bucket = "2 (ILAE threshold)"
        elif ac <= 4:
            aed_bucket = "3–4"
        else:
            aed_bucket = "5+"

        records.append({
            "patient_id": pid,
            "dre": dre,
            "aed_count": m["aed_count"],
            "aed_bucket": aed_bucket,
            "aed_trials": m["aed_trials"],
            "drug_resp_raw": m["drug_resp"],
            "seizure_control": control,
            "surgical_candidate": surgical,
            "onset_zone": m["onset_zone"],
            "syndrome": m["syndrome"],
            "disease_duration_years": m["disease_duration_years"],
            "age_at_onset": m["age_at_onset"],
            "age": pt.get("age"),
            "gender": pt.get("gender", ""),
            "non_adherence_pct": non_adh_pct,
            "ilae_types": m["ilae_types"],
        })
    return records


# ── Public API ─────────────────────────────────────────────────────────────────

def overview() -> dict:
    """Overview KPIs, DRE prevalence, AED burden distribution, onset-zone breakdown,
    surgical candidacy rate, seizure-control distribution."""
    meta = _load_seizure_metadata()
    patients = _load_patients()
    adherence = _load_adherence()
    classified = _classify_patients(meta, patients, adherence)

    total = len(classified)
    dre_n = sum(1 for r in classified if r["dre"])
    non_dre_n = total - dre_n
    surgical_n = sum(1 for r in classified if r["surgical_candidate"])
    dre_pct = round(dre_n / total * 100, 1) if total else 0
    surgical_pct = round(surgical_n / dre_n * 100, 1) if dre_n else 0

    # AED burden distribution
    aed_dist = Counter(r["aed_bucket"] for r in classified)
    aed_chart = [
        {"label": k, "count": v}
        for k, v in sorted(aed_dist.items())
    ]

    # Onset zone distribution (DRE only)
    zone_dist = Counter(r["onset_zone"] for r in classified if r["dre"])
    zone_chart = [
        {"label": z or "Unknown", "count": c}
        for z, c in zone_dist.most_common(8)
    ]

    # Seizure control distribution
    ctrl_dist = Counter(r["seizure_control"] for r in classified)
    ctrl_chart = [
        {"label": c, "count": n}
        for c, n in ctrl_dist.most_common()
    ]

    # DRE by gender
    gender_dre: dict = defaultdict(lambda: {"total": 0, "dre": 0})
    for r in classified:
        g = r["gender"] or "Unknown"
        gender_dre[g]["total"] += 1
        if r["dre"]:
            gender_dre[g]["dre"] += 1
    gender_chart = [
        {
            "gender": g,
            "total": v["total"],
            "dre": v["dre"],
            "pct": round(v["dre"] / v["total"] * 100, 1) if v["total"] else 0,
        }
        for g, v in gender_dre.items()
    ]

    # Disease duration histogram (DRE patients)
    dur_buckets = {"<2y": 0, "2–5y": 0, "5–10y": 0, "10–20y": 0, ">20y": 0}
    for r in classified:
        if not r["dre"]:
            continue
        d = r["disease_duration_years"]
        if d < 2:
            dur_buckets["<2y"] += 1
        elif d < 5:
            dur_buckets["2–5y"] += 1
        elif d < 10:
            dur_buckets["5–10y"] += 1
        elif d < 20:
            dur_buckets["10–20y"] += 1
        else:
            dur_buckets[">20y"] += 1
    dur_chart = [{"label": k, "count": v} for k, v in dur_buckets.items()]

    # Mean AED trials: DRE vs non-DRE
    dre_aeds = [r["aed_count"] for r in classified if r["dre"]]
    non_dre_aeds = [r["aed_count"] for r in classified if not r["dre"]]
    avg_aed_dre = round(sum(dre_aeds) / len(dre_aeds), 1) if dre_aeds else 0
    avg_aed_non = round(sum(non_dre_aeds) / len(non_dre_aeds), 1) if non_dre_aeds else 0

    return {
        "kpis": {
            "total_patients": total,
            "dre_patients": dre_n,
            "non_dre_patients": non_dre_n,
            "dre_prevalence_pct": dre_pct,
            "surgical_candidates": surgical_n,
            "surgical_of_dre_pct": surgical_pct,
            "avg_aed_trials_dre": avg_aed_dre,
            "avg_aed_trials_non_dre": avg_aed_non,
        },
        "aed_burden_chart": aed_chart,
        "onset_zone_chart": zone_chart,
        "seizure_control_chart": ctrl_chart,
        "gender_dre_chart": gender_chart,
        "disease_duration_chart": dur_chart,
        "updated_at": "2026-08-11",
        "source": "clinical.db — seizure_metadata + patients + medication_adherence",
        "ilae_reference": "Kwan et al., Epilepsia 2010;51(6):1069-1077",
    }


def breakdown() -> dict:
    """Per-patient DRE table with classification, AED count, seizure control,
    surgical candidacy, disease duration, onset zone, syndrome."""
    meta = _load_seizure_metadata()
    patients = _load_patients()
    adherence = _load_adherence()
    classified = _classify_patients(meta, patients, adherence)

    # Sort: DRE first, then by AED count descending
    classified_sorted = sorted(
        classified,
        key=lambda r: (not r["dre"], -r["aed_count"]),
    )

    rows = []
    for r in classified_sorted:
        rows.append({
            "patient_id": r["patient_id"],
            "dre": r["dre"],
            "dre_label": "DRE" if r["dre"] else "Responsive",
            "aed_count": r["aed_count"],
            "aed_bucket": r["aed_bucket"],
            "seizure_control": r["seizure_control"],
            "surgical_candidate": r["surgical_candidate"],
            "onset_zone": r["onset_zone"],
            "syndrome": r["syndrome"],
            "disease_duration_years": round(r["disease_duration_years"], 1),
            "age": r["age"],
            "gender": r["gender"],
            "non_adherence_pct": r["non_adherence_pct"],
            "top_aeds": r["aed_trials"][:4],
        })

    # High-burden summary: DRE + surgical candidates
    surgical_list = [r for r in rows if r["surgical_candidate"]]

    # AED trial frequency across all DRE patients
    all_aed_names: list = []
    for r in classified:
        if r["dre"]:
            all_aed_names.extend(r["aed_trials"])
    aed_name_freq = Counter(all_aed_names).most_common(10)
    aed_freq_chart = [{"label": a, "count": c} for a, c in aed_name_freq]

    return {
        "patients": rows,
        "surgical_candidates": surgical_list,
        "top_aeds_in_dre": aed_freq_chart,
        "total": len(rows),
        "dre_total": sum(1 for r in rows if r["dre"]),
    }


def definitions() -> dict:
    """Definitions, ILAE DRE criteria, AED classification table, glossary,
    surgical pathway note, and source references."""
    return {
        "title": "Drug-Resistant Epilepsy (DRE) — Definitions & Reference",
        "ilae_definition": (
            "Drug-Resistant Epilepsy (DRE) is defined as failure of adequate trials "
            "of two tolerated, appropriately chosen and used AED schedules (whether "
            "as monotherapies or in combination) to achieve sustained seizure freedom "
            "(ILAE 2010, Kwan et al.)."
        ),
        "dre_criteria_used": [
            "drug_responsiveness field contains 'drug-resistant' or 'failed', OR",
            "aed_trials count ≥ 2 AND current seizure frequency is not 'remission' / 'controlled'",
        ],
        "surgical_candidacy_proxy": (
            "Focal-onset DRE + disease duration ≥ 2 years — flags potential surgical "
            "evaluation candidates (Phase II pre-surgical work-up recommended)."
        ),
        "prevalence": "~30% of epilepsy patients develop DRE (Kwan & Brodie, NEJM 2000).",
        "risk_tiers": [
            {"tier": "DRE", "color": "#ef4444", "criteria": "≥2 AED failures / explicit label", "action": "Refer to Epilepsy Surgery Program"},
            {"tier": "Responsive", "color": "#22c55e", "criteria": "<2 AED failures, no DRE label", "action": "Continue optimised medical management"},
        ],
        "aed_generation_note": (
            "First-generation AEDs (phenytoin, carbamazepine, valproate) differ from "
            "second-generation (lamotrigine, levetiracetam, oxcarbazepine) and "
            "third-generation (lacosamide, brivaracetam, cenobamate) in mechanism "
            "and tolerability profile. ILAE DRE definition applies regardless of generation."
        ),
        "surgical_pathway": [
            "Phase I: Confirm DRE (≥2 AED failures, seizure diary review)",
            "Phase II: Non-invasive work-up (MRI 3T, Video-EEG, PET, neuropsychology)",
            "Phase III: Invasive monitoring if discordant (SEEG / ECoG)",
            "Phase IV: Surgical resection / neuromodulation (VNS, RNS, DBS)",
        ],
        "glossary": [
            {"term": "AED", "definition": "Antiepileptic Drug (also: ASM — Anti-Seizure Medication)"},
            {"term": "DRE", "definition": "Drug-Resistant Epilepsy"},
            {"term": "ILAE", "definition": "International League Against Epilepsy"},
            {"term": "VNS", "definition": "Vagus Nerve Stimulation"},
            {"term": "RNS", "definition": "Responsive Neurostimulation"},
            {"term": "DBS", "definition": "Deep Brain Stimulation"},
            {"term": "SEEG", "definition": "Stereo-EEG (invasive electrode monitoring)"},
            {"term": "ECoG", "definition": "Electrocorticography"},
        ],
        "references": [
            "Kwan P et al. Definition of drug resistant epilepsy. Epilepsia 2010;51(6):1069-1077.",
            "Kwan P, Brodie MJ. Early identification of refractory epilepsy. NEJM 2000;342:314-319.",
            "Engel J et al. Practice parameter: temporal lobe and localized neocortical resections. Neurology 2003;60:538-547.",
            "NICE Guideline NG217. Epilepsies: diagnosis and management. 2022.",
        ],
        "data_source": "clinical.db — seizure_metadata (71 rows), patients (41 rows), medication_adherence (12,600 rows)",
    }
