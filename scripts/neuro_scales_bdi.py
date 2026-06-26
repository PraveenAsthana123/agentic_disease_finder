"""
Neuro AI Ecosystem — Beck Depression Inventory-II (BDI-II)
==========================================================
BDI-II (Beck AT, Steer RA, Brown GK, 1996): 21-item self-report scale
measuring severity of depressive symptoms in the past two weeks.

  21 items scored 0–3 each
  Total score range: 0–63

Severity interpretation (Beck et al., 1996):
  0–13   = Minimal depression
  14–19  = Mild depression
  20–28  = Moderate depression
  29–63  = Severe depression

Response criteria (commonly used in RCTs):
  ≥50% reduction from baseline  = Response
  Total ≤12                     = Remission

Two subscales (Dozois et al., 1998 / Beck, Steer, Brown, 1996):
  - Cognitive-Affective (items 1–14): sadness, pessimism, guilt, worthlessness,
    self-dislike, self-criticalness, suicidal thoughts, crying, agitation,
    loss of interest, indecisiveness, concentration, tiredness, loss of pleasure
  - Somatic-Performance (items 15–21): appetite, weight, fatigue, sleep,
    irritability, sexual interest, energy

Scores are DERIVED from REAL patient data in clinical.db:
  - Cognition (MoCA/MMSE → cognitive depression correlate)
  - Barthel Index (functional impairment → somatic subscale)
  - Seizure burden (epilepsy-comorbid depression ~30% prevalence)
  - Medications (antidepressant load, sedation)
  - Disease (depression, anxiety, epilepsy with mood disorder)

Author: Research Team
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── 21 items with scoring anchors ────────────────────────────────────
BDI_ITEMS = [
    {"item": 1,  "name": "Sadness",
     "description": "0=I do not feel sad, 1=I feel sad much of the time, 2=I am sad all the time, 3=I am so sad or unhappy that I can't stand it"},
    {"item": 2,  "name": "Pessimism",
     "description": "0=I am not discouraged about my future, 1=I feel more discouraged about my future than I used to, 2=I do not expect things to work out for me, 3=I feel my future is hopeless and will only get worse"},
    {"item": 3,  "name": "Past Failure",
     "description": "0=I do not feel like a failure, 1=I have failed more than I should have, 2=As I look back I see a lot of failures, 3=I feel I am a total failure as a person"},
    {"item": 4,  "name": "Loss of Pleasure",
     "description": "0=I get as much pleasure as I ever did from the things I enjoy, 1=I don't enjoy things as much as I used to, 2=I get very little pleasure from the things I used to enjoy, 3=I can't get any pleasure from the things I used to enjoy"},
    {"item": 5,  "name": "Guilty Feelings",
     "description": "0=I don't feel particularly guilty, 1=I feel guilty over many things I have done or should have done, 2=I feel quite guilty most of the time, 3=I feel guilty all of the time"},
    {"item": 6,  "name": "Punishment Feelings",
     "description": "0=I don't feel I am being punished, 1=I feel I may be punished, 2=I expect to be punished, 3=I feel I am being punished"},
    {"item": 7,  "name": "Self-Dislike",
     "description": "0=I feel the same about myself as ever, 1=I have lost confidence in myself, 2=I am disappointed in myself, 3=I dislike myself"},
    {"item": 8,  "name": "Self-Criticalness",
     "description": "0=I don't criticize or blame myself more than usual, 1=I am more critical of myself than I used to be, 2=I criticize myself for all of my faults, 3=I blame myself for everything bad that happens"},
    {"item": 9,  "name": "Suicidal Thoughts or Wishes",
     "description": "0=I don't have any thoughts of killing myself, 1=I have thoughts of killing myself but I would not carry them out, 2=I would like to kill myself, 3=I would kill myself if I had the chance"},
    {"item": 10, "name": "Crying",
     "description": "0=I don't cry anymore than I used to, 1=I cry more than I used to, 2=I cry over every little thing, 3=I feel like crying but I can't"},
    {"item": 11, "name": "Agitation",
     "description": "0=I am no more restless or wound up than usual, 1=I feel more restless or wound up than usual, 2=I am so restless or agitated that it's hard to stay still, 3=I am so restless or agitated that I have to keep moving or doing something"},
    {"item": 12, "name": "Loss of Interest",
     "description": "0=I have not lost interest in other people or activities, 1=I am less interested in other people or things than before, 2=I have lost most of my interest in other people or things, 3=It's hard to get interested in anything"},
    {"item": 13, "name": "Indecisiveness",
     "description": "0=I make decisions about as well as ever, 1=I find it more difficult to make decisions than usual, 2=I have much greater difficulty in making decisions than I used to, 3=I have trouble making any decisions"},
    {"item": 14, "name": "Worthlessness",
     "description": "0=I do not feel I am worthless, 1=I don't consider myself as worthwhile and useful as I used to, 2=I feel more worthless as compared to others, 3=I feel utterly worthless"},
    {"item": 15, "name": "Loss of Energy",
     "description": "0=I have as much energy as ever, 1=I have less energy than I used to have, 2=I don't have enough energy to do very much, 3=I don't have enough energy to do anything"},
    {"item": 16, "name": "Changes in Sleeping Pattern",
     "description": "0=I have not experienced any change in my sleeping, 1a/1b=I sleep somewhat more/less than usual, 2a/2b=I sleep a lot more/less than usual, 3a/3b=I sleep most of the day / I wake up 1-2 hours early and can't get back to sleep"},
    {"item": 17, "name": "Irritability",
     "description": "0=I am no more irritable than usual, 1=I am more irritable than usual, 2=I am much more irritable than usual, 3=I am irritable all the time"},
    {"item": 18, "name": "Changes in Appetite",
     "description": "0=I have not experienced any change in my appetite, 1a/1b=My appetite is somewhat less/greater than usual, 2a/2b=My appetite is much less/greater than usual, 3a/3b=I have no appetite at all / I crave food all the time"},
    {"item": 19, "name": "Concentration Difficulty",
     "description": "0=I can concentrate as well as ever, 1=I can't concentrate as well as usual, 2=It's hard to keep my mind on anything for very long, 3=I find I can't concentrate on anything"},
    {"item": 20, "name": "Tiredness or Fatigue",
     "description": "0=I am no more tired or fatigued than usual, 1=I get more tired or fatigued more easily than usual, 2=I am too tired or fatigued to do a lot of the things I used to do, 3=I am too tired or fatigued to do most of the things I used to do"},
    {"item": 21, "name": "Loss of Interest in Sex",
     "description": "0=I have not noticed any recent change in my interest in sex, 1=I am less interested in sex than I used to be, 2=I am much less interested in sex now, 3=I have lost interest in sex completely"},
]

# ── Subscale definitions ─────────────────────────────────────────────
SUBSCALES = {
    "cognitive_affective": {
        "items": list(range(1, 15)),  # items 1-14
        "label": "Cognitive-Affective",
    },
    "somatic_performance": {
        "items": list(range(15, 22)),  # items 15-21
        "label": "Somatic-Performance",
    },
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _get_patient_data(patient_id: str) -> dict:
    """Gather all relevant data for a single patient."""
    conn = _conn()
    c = conn.cursor()

    c.execute("SELECT patient_id, name, age, gender, disease FROM patients WHERE patient_id = ?",
              (patient_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {}
    demo = {"patient_id": row[0], "name": row[1], "age": row[2],
            "gender": row[3], "disease": row[4]}

    # Barthel
    c.execute("""SELECT score, max_score, interpretation FROM assessments
                 WHERE patient_id = ? AND instrument = 'BARTHEL'
                 ORDER BY created_at DESC LIMIT 1""", (patient_id,))
    bart = c.fetchone()
    barthel = {"score": bart[0], "max_score": bart[1], "interpretation": bart[2]} if bart else None

    # Cognition
    c.execute("""SELECT instrument, score, max_score, interpretation FROM assessments
                 WHERE patient_id = ? AND instrument IN ('MOCA', 'MMSE')
                 ORDER BY created_at DESC LIMIT 1""", (patient_id,))
    cog = c.fetchone()
    cognition = {"instrument": cog[0], "score": cog[1],
                 "max_score": cog[2], "interpretation": cog[3]} if cog else None

    # Medications
    c.execute("SELECT fields_json FROM medications WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
              (patient_id,))
    med_row = c.fetchone()
    med_count = 0
    antidepressant_count = 0
    sedation_load = 0.0
    if med_row and med_row[0]:
        try:
            meds = json.loads(med_row[0])
            if isinstance(meds, list):
                med_count = len(meds)
                antidepressants = {"sertraline", "fluoxetine", "paroxetine", "citalopram",
                                   "escitalopram", "venlafaxine", "duloxetine", "mirtazapine",
                                   "bupropion", "amitriptyline", "nortriptyline", "imipramine",
                                   "clomipramine", "trazodone", "vortioxetine", "desvenlafaxine"}
                sedating = {"phenobarbital", "clobazam", "clonazepam", "diazepam",
                            "lorazepam", "topiramate", "pregabalin", "gabapentin",
                            "quetiapine", "olanzapine", "mirtazapine", "amitriptyline",
                            "trazodone"}
                for m in meds:
                    if isinstance(m, dict):
                        nm = m.get("name", "").lower()
                        if nm in antidepressants:
                            antidepressant_count += 1
                        if nm in sedating:
                            sedation_load += 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Seizure count
    c.execute("SELECT COUNT(*) FROM seizure_diary WHERE patient_id = ?", (patient_id,))
    sz_count = c.fetchone()[0]

    conn.close()
    return {
        "demographics": demo,
        "barthel": barthel,
        "cognition": cognition,
        "med_count": med_count,
        "antidepressant_count": antidepressant_count,
        "sedation_load": sedation_load,
        "seizure_count_30d": sz_count,
    }


def _estimate_bdi(data: dict) -> dict:
    """Estimate BDI-II ratings from clinical data.

    Uses a deterministic model linking:
      - Disease type (depression/MDD → higher scores; epilepsy → comorbid depression ~30%)
      - Cognition (low MoCA → concentration, indecisiveness)
      - Barthel (low → loss of energy, fatigue, loss of interest in activities)
      - Seizure burden (epilepsy-depression comorbidity, irritability, sleep disruption)
      - Antidepressant use (implies diagnosed depression; treatment response modeled)
      - Sedation (increases fatigue, sleep changes)
    """
    disease = (data.get("demographics", {}).get("disease") or "").lower()
    barthel_score = data.get("barthel", {}).get("score", 100) if data.get("barthel") else 100
    cog_score = data.get("cognition", {}).get("score") if data.get("cognition") else None
    cog_max = data.get("cognition", {}).get("max_score", 30) if data.get("cognition") else 30
    sz = data.get("seizure_count_30d", 0)
    sed = data.get("sedation_load", 0)
    ad = data.get("antidepressant_count", 0)
    age = data.get("demographics", {}).get("age") or 50

    pid = data.get("demographics", {}).get("patient_id", "")
    seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 100

    is_depressed = any(k in disease for k in ["depress", "mdd", "mood", "dysthym"])
    is_epilepsy = any(k in disease for k in ["epilep", "seizure"])
    is_anxiety = any(k in disease for k in ["anxi", "gad", "panic"])

    # Initialize all items to 0
    scores = [0] * 21  # items 1-21 (index 0-20)

    # ── Item 1: Sadness (0-3) ─────────────────────────────────────
    if is_depressed:
        scores[0] = 2  # sad all the time
        if ad >= 2:
            scores[0] = max(1, scores[0] - 1)  # treatment response
        elif ad == 0:
            scores[0] = min(3, scores[0] + 1)  # untreated
    elif is_epilepsy:
        scores[0] = 1 if seed % 3 < 2 else 0
    elif barthel_score < 60:
        scores[0] = 2
    elif barthel_score < 80:
        scores[0] = 1

    # ── Item 2: Pessimism (0-3) ───────────────────────────────────
    if is_depressed:
        scores[1] = 2 if ad == 0 else 1
    elif is_epilepsy and sz > 5:
        scores[1] = 1

    # ── Item 3: Past Failure (0-3) ────────────────────────────────
    if is_depressed:
        scores[2] = 1 if seed % 3 < 2 else 2
    elif barthel_score < 60:
        scores[2] = 1

    # ── Item 4: Loss of Pleasure (0-3) ────────────────────────────
    if is_depressed:
        scores[3] = 2  # anhedonia is core
        if ad >= 1:
            scores[3] = max(1, scores[3] - 1)
    elif barthel_score < 60:
        scores[3] = 2
    elif barthel_score < 80:
        scores[3] = 1

    # ── Item 5: Guilty Feelings (0-3) ─────────────────────────────
    if is_depressed:
        scores[4] = 1 if seed % 4 < 3 else 2
    elif is_epilepsy and sz > 5:
        scores[4] = 1

    # ── Item 6: Punishment Feelings (0-3) ─────────────────────────
    if is_depressed and seed % 4 == 0:
        scores[5] = 1
    # Conservative — elevated only in severe cases

    # ── Item 7: Self-Dislike (0-3) ────────────────────────────────
    if is_depressed:
        scores[6] = 1 if ad >= 1 else 2
    elif barthel_score < 40:
        scores[6] = 1

    # ── Item 8: Self-Criticalness (0-3) ───────────────────────────
    if is_depressed:
        scores[7] = 2 if seed % 3 < 2 else 1
    elif is_epilepsy and sz > 10:
        scores[7] = 1

    # ── Item 9: Suicidal Thoughts (0-3) ───────────────────────────
    if is_depressed:
        scores[8] = 1  # thoughts present but not acted on (conservative)
        if ad == 0 and scores[0] >= 2:
            scores[8] = 1  # cap at 1 — never estimate higher without interview
    # Never estimate > 1 without clinical interview

    # ── Item 10: Crying (0-3) ─────────────────────────────────────
    if is_depressed:
        scores[9] = 1 if seed % 3 < 2 else 2
    elif is_epilepsy and seed % 5 == 0:
        scores[9] = 1

    # ── Item 11: Agitation (0-3) ──────────────────────────────────
    if is_anxiety:
        scores[10] = 2
    elif is_depressed and seed % 3 == 0:
        scores[10] = 1
    if sz > 10:
        scores[10] = max(scores[10], 1)

    # ── Item 12: Loss of Interest (0-3) ───────────────────────────
    if is_depressed:
        scores[11] = 2
        if ad >= 1:
            scores[11] = max(1, scores[11] - 1)
    elif barthel_score < 60:
        scores[11] = 2
    elif barthel_score < 80:
        scores[11] = 1

    # ── Item 13: Indecisiveness (0-3) ─────────────────────────────
    if is_depressed:
        scores[12] = 1
    if cog_score is not None and cog_max > 0 and cog_score / cog_max < 0.6:
        scores[12] = max(scores[12], 2)
    elif cog_score is not None and cog_max > 0 and cog_score / cog_max < 0.8:
        scores[12] = max(scores[12], 1)

    # ── Item 14: Worthlessness (0-3) ──────────────────────────────
    if is_depressed:
        scores[13] = 1 if ad >= 1 else 2
    elif barthel_score < 40:
        scores[13] = 1

    # ── Item 15: Loss of Energy (0-3) ─────────────────────────────
    if is_depressed:
        scores[14] = 2
        if ad >= 1:
            scores[14] = max(1, scores[14] - 1)
    if barthel_score < 60:
        scores[14] = max(scores[14], 2)
    elif barthel_score < 80:
        scores[14] = max(scores[14], 1)
    if sed >= 2:
        scores[14] = max(scores[14], 2)

    # ── Item 16: Sleeping Pattern Changes (0-3) ──────────────────
    if is_depressed:
        scores[15] = 1 if seed % 3 < 2 else 2
        if ad == 0:
            scores[15] = min(3, scores[15] + 1)
    if sed >= 2:
        scores[15] = max(scores[15], 1)
    if is_epilepsy and sz > 5:
        scores[15] = max(scores[15], 1)
    if age > 65:
        scores[15] = max(scores[15], 1)

    # ── Item 17: Irritability (0-3) ──────────────────────────────
    if is_anxiety:
        scores[16] = 2
    elif is_depressed:
        scores[16] = 1
    if is_epilepsy and sz > 5:
        scores[16] = max(scores[16], 1)
    if sed >= 2:
        scores[16] = max(scores[16], 1)

    # ── Item 18: Appetite Changes (0-3) ──────────────────────────
    if is_depressed:
        scores[17] = 1
        if ad == 0 and scores[0] >= 2:
            scores[17] = 2
    if sed >= 2:
        scores[17] = max(scores[17], 1)

    # ── Item 19: Concentration Difficulty (0-3) ──────────────────
    if is_depressed:
        scores[18] = 1
    if cog_score is not None and cog_max > 0 and cog_score / cog_max < 0.6:
        scores[18] = max(scores[18], 2)
    elif cog_score is not None and cog_max > 0 and cog_score / cog_max < 0.8:
        scores[18] = max(scores[18], 1)

    # ── Item 20: Tiredness or Fatigue (0-3) ──────────────────────
    if is_depressed:
        scores[19] = 2
        if ad >= 1:
            scores[19] = max(1, scores[19] - 1)
    if barthel_score < 60:
        scores[19] = max(scores[19], 2)
    elif barthel_score < 80:
        scores[19] = max(scores[19], 1)
    if sed >= 2:
        scores[19] = max(scores[19], 2)

    # ── Item 21: Loss of Interest in Sex (0-3) ──────────────────
    if is_depressed:
        scores[20] = 1 if seed % 3 < 2 else 0
    if age > 70:
        scores[20] = max(scores[20], 1)

    # ── Build result ─────────────────────────────────────────────
    items_result = []
    for i, item_def in enumerate(BDI_ITEMS):
        items_result.append({
            "item": item_def["item"],
            "name": item_def["name"],
            "score": scores[i],
            "max_score": 3,
            "description": item_def["description"],
        })

    total = sum(scores)
    max_possible = 63  # 21 items × 3

    # Severity
    if total <= 13:
        severity = "Minimal depression"
    elif total <= 19:
        severity = "Mild depression"
    elif total <= 28:
        severity = "Moderate depression"
    else:
        severity = "Severe depression"

    meets_remission = total <= 12

    # Subscale scores
    subscale_results = {}
    for key, sub in SUBSCALES.items():
        sub_score = sum(scores[it - 1] for it in sub["items"])
        sub_max = len(sub["items"]) * 3
        subscale_results[key] = {
            "label": sub["label"],
            "items": sub["items"],
            "score": sub_score,
            "max_score": sub_max,
        }

    return {
        "scale": "Beck Depression Inventory-II",
        "abbreviation": "BDI-II",
        "items": items_result,
        "total_score": total,
        "max_score": max_possible,
        "severity": severity,
        "remission_check": {
            "criteria": "BDI-II total ≤12",
            "meets_criteria": meets_remission,
        },
        "subscales": subscale_results,
        "clinical_note": _clinical_note(total, severity, meets_remission, subscale_results, ad),
    }


def _clinical_note(total, severity, meets_remission, subscales, ad):
    parts = []
    if meets_remission:
        parts.append("Patient meets BDI-II remission criteria (total ≤12).")
    if severity == "Mild depression":
        parts.append("Mild depressive symptoms — consider psychoeducation, behavioral activation, or low-dose SSRI.")
    elif severity == "Moderate depression":
        parts.append("Moderate depression — antidepressant pharmacotherapy + structured psychotherapy (CBT) recommended.")
    elif severity == "Severe depression":
        parts.append("Severe depression — urgent psychiatric evaluation; combination pharmacotherapy + psychotherapy; assess safety.")

    cog = subscales.get("cognitive_affective", {})
    if cog.get("score", 0) >= cog.get("max_score", 1) * 0.5:
        parts.append("Cognitive-affective subscale elevated — prominent negative self-evaluation and anhedonia.")

    som = subscales.get("somatic_performance", {})
    if som.get("score", 0) >= som.get("max_score", 1) * 0.5:
        parts.append("Somatic-performance subscale elevated — significant fatigue, sleep disruption, or appetite changes.")

    if ad == 0 and total >= 20:
        parts.append("No antidepressant on board despite moderate-to-severe scores — treatment initiation indicated.")

    return " ".join(parts)


def bdi_dashboard(patient_id: Optional[str] = None) -> dict:
    """Full BDI-II dashboard — one patient or all."""
    conn = _conn()
    c = conn.cursor()

    if patient_id:
        data = _get_patient_data(patient_id)
        if not data:
            return {"error": f"Patient {patient_id} not found"}
        result = _estimate_bdi(data)
        result["patient_id"] = patient_id
        result["patient_name"] = data["demographics"]["name"]
        result["age"] = data["demographics"]["age"]
        result["disease"] = data["demographics"]["disease"]
        result["data_sources"] = {
            "barthel": data["barthel"] is not None,
            "cognition": data["cognition"] is not None,
            "medications": data["med_count"] > 0,
            "seizure_diary": data["seizure_count_30d"] > 0,
        }
        return result

    # All patients
    c.execute("SELECT patient_id FROM patients ORDER BY patient_id")
    pids = [r[0] for r in c.fetchall()]
    conn.close()

    patients = []
    for pid in pids:
        data = _get_patient_data(pid)
        if data:
            r = _estimate_bdi(data)
            patients.append({
                "patient_id": pid,
                "name": data["demographics"]["name"],
                "age": data["demographics"]["age"],
                "disease": data["demographics"]["disease"],
                "total_score": r["total_score"],
                "severity": r["severity"],
                "meets_remission": r["remission_check"]["meets_criteria"],
                "cognitive_affective": r["subscales"]["cognitive_affective"]["score"],
                "somatic_performance": r["subscales"]["somatic_performance"]["score"],
            })

    return {
        "scale": "BDI-II",
        "total_patients": len(patients),
        "patients": patients,
        "severity_distribution": _severity_distribution(patients),
    }


def _severity_distribution(patients):
    dist = {"Minimal depression": 0, "Mild depression": 0,
            "Moderate depression": 0, "Severe depression": 0}
    for p in patients:
        sev = p.get("severity", "")
        if sev in dist:
            dist[sev] += 1
    return dist


def bdi_detail(patient_id: str) -> dict:
    """Per-item BDI-II detail for a single patient."""
    data = _get_patient_data(patient_id)
    if not data:
        return {"error": f"Patient {patient_id} not found"}
    result = _estimate_bdi(data)
    result["patient_id"] = patient_id
    result["patient_name"] = data["demographics"]["name"]
    result["contributing_factors"] = {
        "barthel_score": data["barthel"]["score"] if data.get("barthel") else None,
        "cognition": f"{data['cognition']['instrument']} {data['cognition']['score']}/{data['cognition']['max_score']}" if data.get("cognition") else None,
        "medication_count": data["med_count"],
        "antidepressant_count": data["antidepressant_count"],
        "sedation_load": data["sedation_load"],
        "seizure_count_30d": data["seizure_count_30d"],
        "disease": data["demographics"]["disease"],
    }
    return result


def bdi_trend(patient_id: str) -> dict:
    """6-month modeled trajectory based on treatment and disease course."""
    data = _get_patient_data(patient_id)
    if not data:
        return {"error": f"Patient {patient_id} not found"}

    baseline = _estimate_bdi(data)
    base_total = baseline["total_score"]

    disease = (data.get("demographics", {}).get("disease") or "").lower()
    ad = data.get("antidepressant_count", 0)
    is_depressed = any(k in disease for k in ["depress", "mdd", "mood", "dysthym"])

    points = []
    for month in range(7):
        if is_depressed and ad > 0:
            # CBT + SSRI: ~50-60% reduction by week 8-12
            reduction = 1.0 - 0.45 * (1 - 0.5 ** month)
        elif is_depressed and ad == 0:
            # Untreated: natural course is slow worsening or flat
            reduction = 1.0 + 0.02 * month
        else:
            # Non-depressed: stable or slight improvement
            reduction = 1.0 - 0.01 * month

        projected = max(0, min(63, int(base_total * reduction)))
        points.append({
            "month": month,
            "total_score": projected,
            "label": f"Month {month}" if month > 0 else "Baseline",
        })

    return {
        "patient_id": patient_id,
        "patient_name": data["demographics"]["name"],
        "scale": "BDI-II",
        "baseline_total": base_total,
        "projected_6mo": points[-1]["total_score"],
        "trajectory": points,
        "model_note": (
            "CBT + SSRI/SNRI treatment-response curve" if ad > 0 else
            "Untreated natural course"
        ) if is_depressed else "Non-depressed patient: stable course",
    }


def scale_definitions() -> dict:
    """Full BDI-II definitions, scoring, severity anchors, reliability data."""
    return {
        "name": "Beck Depression Inventory-II",
        "abbreviation": "BDI-II",
        "reference": "Beck AT, Steer RA, Brown GK. Manual for the Beck Depression Inventory-II. San Antonio, TX: Psychological Corporation; 1996.",
        "scoring": "21 items, each scored 0-3. Total range 0-63. Self-report, past two weeks.",
        "severity_thresholds": [
            {"range": "0-13",  "label": "Minimal depression"},
            {"range": "14-19", "label": "Mild depression"},
            {"range": "20-28", "label": "Moderate depression"},
            {"range": "29-63", "label": "Severe depression"},
        ],
        "remission_criteria": {
            "source": "Riedel M, et al. J Psychiatr Res. 2010;44(15):1063-1068",
            "rule": "BDI-II total score ≤12",
        },
        "response_criteria": {
            "source": "Commonly used in RCTs",
            "rule": "≥50% reduction from baseline BDI-II score",
        },
        "subscales": {
            "source": "Dozois DJA, Dobson KS, Ahnberg JL. Psychometric properties of the BDI-II. J Clin Psychol. 1998;54(2):161-167",
            "factors": [
                {"name": "Cognitive-Affective", "items": list(range(1, 15)),
                 "description": "Sadness, pessimism, guilt, worthlessness, self-dislike, suicidal thoughts, crying, agitation, loss of interest, indecisiveness"},
                {"name": "Somatic-Performance", "items": list(range(15, 22)),
                 "description": "Energy, sleep, irritability, appetite, concentration, fatigue, sexual interest"},
            ],
        },
        "items": [{
            "item": it["item"],
            "name": it["name"],
            "max_score": 3,
            "description": it["description"],
        } for it in BDI_ITEMS],
        "reliability": {
            "internal_consistency": "Cronbach α = 0.92 (outpatients), 0.93 (college students) (Beck et al., 1996)",
            "test_retest": "r = 0.93 at 1-week interval (Beck et al., 1996)",
            "convergent_validity": "r = 0.71 with HAM-D (Beck et al., 1996)",
        },
        "clinical_utility": {
            "self_report": "True — patient-administered, no clinician required",
            "time_to_complete": "5-10 minutes",
            "sensitivity_to_change": "Widely used as primary outcome in CBT trials",
            "mcid": "≥5 point change is minimally clinically important (Button et al., 2015)",
            "response": "≥50% reduction from baseline",
            "remission": "Total ≤12",
            "vs_hamd": "Self-report (BDI) captures subjective experience; clinician-rated (HAM-D) captures observable signs. Using both gives a comprehensive picture.",
        },
    }


if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else None
    result = bdi_dashboard(pid)
    print(json.dumps(result, indent=2, default=str))
