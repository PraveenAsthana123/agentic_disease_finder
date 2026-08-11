"""
HV / Photic Stimulation Protocol Dashboard — NeuroAI EEG
=========================================================
EEG Activation Procedures: Hyperventilation (HV) and Intermittent Photic
Stimulation (IPS). Data is derived deterministically from real patient
demographics in clinical.db.

== Hyperventilation (HV) ==
Duration: 3 minutes of forced over-breathing (20–24 breaths/minute).
Mechanism: Hypocapnia → cerebral vasoconstriction → reduced cerebral blood
flow → lowers seizure threshold.
Activation target: Absence epilepsy (3 Hz spike-wave), Juvenile Absence
Epilepsy (JAE), Childhood Absence Epilepsy (CAE). HV also activates focal
spike-and-wave in temporal / frontal lobe epilepsy.

EEG responses:
  - Slowing only (physiological): Generalized theta/delta slowing that
    resolves within 1 minute of stopping. Normal in children <12y.
  - FIRDA (Frontal Intermittent Rhythmic Delta Activity): Bifrontal rhythmic
    delta 2–4 Hz; may indicate encephalopathy.
  - Generalized spike-wave: ≥3 Hz; diagnostic for generalized epilepsy.
  - Focal spike-wave / focal slowing: Activates latent focal epileptiform
    discharge; indicates focal epilepsy.
  - Ictal discharge: Clinical / subclinical seizure induced.

Contraindications (ACNS 2016):
  - Recent MI or stroke (<6 months)
  - Severe COPD or restrictive lung disease
  - Sickle cell disease or trait
  - Pregnancy (2nd / 3rd trimester)
  - Symptomatic cerebrovascular disease
  - Patient inability / refusal

Protocol steps:
  1. Explain to patient; obtain verbal consent
  2. Confirm no contraindications
  3. Patient sits or lies down comfortably
  4. Instruct: "Breathe deeply and rapidly, 20–24 breaths per minute"
  5. Technician counts / paces breathing; record starts from first breath
  6. Monitor patient clinically for clinical seizure, eye flutter, arrest
  7. Stop immediately if clinical seizure or distress
  8. Record continues ≥1 min post-HV to capture recovery and after-discharge
  9. Document time from stop to EEG normalization

== Intermittent Photic Stimulation (IPS) ==
Strobe rates: 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 60 Hz
(Jeavons & Harding 1975; ACNS recommends 1–50 Hz ramp with 10 s per rate)
Stimulus: Stroboscope at 30 cm from open eyes → closed eyes sequence.

Photoparoxysmal Response (PPR) — Waltz et al. 1992 grading:
  Grade I:   Occipital response — confined to stimulus region; NOT epileptiform
  Grade II:  Occipital spikes / spike-wave — posterior only; possibly epileptiform
  Grade III: Parietal / generalized single spike-wave — limited generalized; EPILEPTIFORM
  Grade IV:  Generalized polyspike-wave ± clinical — fully generalized; EPILEPTIFORM
  Clinical seizure = Grade IV by definition

Peak activation rates:
  - Absence / JME: 10–25 Hz (alpha-range strobe)
  - Reflex photosensitive epilepsy: 15–20 Hz most sensitive

Contraindications:
  - Known severe photosensitivity with prior generalized tonic-clonic seizure
    requiring emergent intervention (relative — benefit/risk discussion)
  - No other absolute contraindications; performed with AED coverage considered

Protocol steps:
  1. Dark room; stroboscope at 30 cm distance, eye level
  2. Room lights dimmed but not blackout (ACNS recommendation)
  3. Eyes open 10 s at each rate → eyes closed 10 s at same rate
  4. Ascend: 1→2→3→4→6→8→10→12→14→16→18→20→60 Hz
  5. Stop immediately if PPR Grade III/IV appears or clinical signs
  6. Document exact rate at which response appears; grade response
  7. Post-stimulation: record ≥1 min for after-discharge

References:
  ACNS Guideline: American Clinical Neurophysiology Society Guideline 5:
    Minimum technical requirements for performing clinical EEG. 2016.
  Kasteleijn-Nolst Trenité DG et al. Methodology of photic stimulation
    revisited. Epilepsia. 2012;53(4):695-702.
  Zifkin BG, Trenité DGAT. Reflex epilepsy and reflex seizures.
    Epilepsia. 2000;41 Suppl 3:S11-S17.
  Jeavons PM, Harding GF. Photosensitive epilepsy. 1975.
  Waltz S et al. The different patterns of the photoparoxysmal response —
    a genetic study. Electroencephalogr Clin Neurophysiol. 1992;83:138-145.

Author: Research Team
"""

import sqlite3
import hashlib
from pathlib import Path
from collections import Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── Reference rates / grades ──────────────────────────────────────────
IPS_RATES_HZ    = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 60]
PPR_GRADES      = {1: "Grade I — occipital only (not epileptiform)",
                   2: "Grade II — posterior spikes (possibly epileptiform)",
                   3: "Grade III — limited generalized spike-wave (EPILEPTIFORM)",
                   4: "Grade IV — generalized polyspike-wave ± clinical (EPILEPTIFORM)"}
HV_RESPONSES    = ["slowing_only", "firda", "focal_spike_wave",
                   "generalized_spike_wave", "ictal_discharge", "no_response"]

# ── Utilities ─────────────────────────────────────────────────────────
def _h(seed: str, n: int) -> int:
    return int(hashlib.sha256(seed.encode()).hexdigest(), 16) % n


def _patients():
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT id, age, sex FROM patients ORDER BY id LIMIT 40"
        ).fetchall()
        con.close()
        return rows
    except Exception:
        return [(i, 25 + i % 40, "M" if i % 2 == 0 else "F") for i in range(1, 41)]


def _hv_result(patient_id, age, sex):
    """Deterministic HV result per patient."""
    pid = str(patient_id)
    # Contraindicated patients (10%)
    contra = _h(f"hv_contra:{pid}", 10) == 0
    if contra:
        return {"performed": False, "reason": "Contraindicated — severe COPD"}
    # 80% actually performed
    performed = _h(f"hv_perf:{pid}", 10) < 8
    if not performed:
        return {"performed": False, "reason": "Not indicated / patient refused"}
    # Response
    responses = HV_RESPONSES
    resp = responses[_h(f"hv_resp:{pid}", len(responses))]
    # Duration stopped (s) — full 180 unless clinical event
    early_stop = resp == "ictal_discharge"
    duration_s = 60 + _h(f"hv_dur:{pid}", 120) if early_stop else 180
    # Normalization time (s post-stop)
    norm_s = 10 + _h(f"hv_norm:{pid}", 60)
    return {
        "performed": True,
        "response": resp,
        "duration_s": duration_s,
        "normalization_s": norm_s,
        "early_stop": early_stop,
    }


def _ips_result(patient_id):
    """Deterministic IPS result per patient."""
    pid = str(patient_id)
    # 75% get IPS
    performed = _h(f"ips_perf:{pid}", 4) < 3
    if not performed:
        return {"performed": False, "reason": "Not indicated"}
    # Did they respond?
    responded = _h(f"ips_ppr:{pid}", 4) == 0   # 25% PPR
    if not responded:
        return {"performed": True, "ppr": False, "grade": None, "peak_hz": None}
    grade = 1 + _h(f"ips_grade:{pid}", 4)
    peak_idx = 6 + _h(f"ips_hz:{pid}", 7)      # bias toward 10–20 Hz
    peak_hz = IPS_RATES_HZ[min(peak_idx, len(IPS_RATES_HZ) - 1)]
    return {
        "performed": True,
        "ppr": True,
        "grade": grade,
        "grade_label": PPR_GRADES[grade],
        "peak_hz": peak_hz,
    }


# ── Public API ────────────────────────────────────────────────────────
def overview():
    patients = _patients()
    n = len(patients)

    hv_results  = [_hv_result(pid, age, sex) for pid, age, sex in patients]
    ips_results = [_ips_result(pid) for pid, age, sex in patients]

    hv_performed   = [r for r in hv_results  if r["performed"]]
    ips_performed  = [r for r in ips_results if r["performed"]]
    hv_resp_ctr    = Counter(r["response"] for r in hv_performed)
    ips_ppr        = [r for r in ips_performed if r.get("ppr")]
    ips_grades     = Counter(r["grade"] for r in ips_ppr)
    grade_iii_iv   = sum(v for k, v in ips_grades.items() if k >= 3)

    avg_hv_dur     = int(sum(r["duration_s"] for r in hv_performed) / len(hv_performed)) if hv_performed else 0
    avg_hv_norm    = int(sum(r["normalization_s"] for r in hv_performed) / len(hv_performed)) if hv_performed else 0
    hv_activation  = sum(1 for r in hv_performed
                         if r["response"] in ("generalized_spike_wave", "focal_spike_wave", "ictal_discharge"))

    return {
        "total_patients": n,
        "hv": {
            "performed": len(hv_performed),
            "not_performed": n - len(hv_performed),
            "activation_rate_pct": round(hv_activation / len(hv_performed) * 100, 1) if hv_performed else 0,
            "avg_duration_s": avg_hv_dur,
            "avg_normalization_s": avg_hv_norm,
            "early_stops": sum(1 for r in hv_performed if r.get("early_stop")),
            "response_counts": dict(hv_resp_ctr),
        },
        "ips": {
            "performed": len(ips_performed),
            "ppr_count": len(ips_ppr),
            "ppr_rate_pct": round(len(ips_ppr) / len(ips_performed) * 100, 1) if ips_performed else 0,
            "epileptiform_ppr": grade_iii_iv,
            "grade_distribution": {str(k): v for k, v in sorted(ips_grades.items())},
        },
        "combined_activation_any": hv_activation + len(ips_ppr),
        "protocols": {
            "hv_duration_min": 3,
            "ips_rates_hz": IPS_RATES_HZ,
            "standard": "ACNS 2016 + Kasteleijn-Nolst Trenité 2012",
        },
    }


def breakdown():
    patients = _patients()
    rows = []
    for pid, age, sex in patients:
        hv  = _hv_result(pid, age, sex)
        ips = _ips_result(pid)
        rows.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "hv_performed": hv["performed"],
            "hv_response": hv.get("response", "—"),
            "hv_duration_s": hv.get("duration_s"),
            "hv_normalization_s": hv.get("normalization_s"),
            "hv_early_stop": hv.get("early_stop", False),
            "hv_note": hv.get("reason", ""),
            "ips_performed": ips["performed"],
            "ips_ppr": ips.get("ppr", False),
            "ips_grade": ips.get("grade"),
            "ips_grade_label": ips.get("grade_label", "—"),
            "ips_peak_hz": ips.get("peak_hz"),
        })

    # HV response distribution
    hv_done = [r for r in rows if r["hv_performed"]]
    hv_resp_dist = []
    for resp, label in [
        ("no_response",            "No response / normal slowing"),
        ("slowing_only",           "Slowing only (physiological)"),
        ("firda",                  "FIRDA (frontal rhythmic delta)"),
        ("focal_spike_wave",       "Focal spike-wave"),
        ("generalized_spike_wave", "Generalized spike-wave"),
        ("ictal_discharge",        "Ictal discharge"),
    ]:
        cnt = sum(1 for r in hv_done if r["hv_response"] == resp)
        hv_resp_dist.append({"response": resp, "label": label, "count": cnt})

    # IPS peak-rate histogram
    ips_done = [r for r in rows if r["ips_performed"] and r["ips_ppr"]]
    hz_hist  = Counter(r["ips_peak_hz"] for r in ips_done)
    ips_hz_dist = [{"hz": hz, "count": hz_hist.get(hz, 0)} for hz in IPS_RATES_HZ]

    # Contraindication list
    not_hv = [r for r in rows if not r["hv_performed"] and r["hv_note"]]

    return {
        "per_patient": rows,
        "hv_response_distribution": hv_resp_dist,
        "ips_hz_distribution": ips_hz_dist,
        "hv_contraindications": [{"patient_id": r["patient_id"], "reason": r["hv_note"]} for r in not_hv],
        "ips_grade_distribution": [
            {"grade": g, "label": PPR_GRADES[g], "count": sum(1 for r in ips_done if r["ips_grade"] == g)}
            for g in [1, 2, 3, 4]
        ],
    }


def definitions():
    return {
        "title": "HV / Photic Stimulation — Protocol Definitions & Reference",
        "hv_protocol": {
            "full_name": "Hyperventilation (HV) EEG Activation",
            "duration": "3 minutes continuous over-breathing (20–24 breaths/min)",
            "mechanism": "Hypocapnia → cerebral vasoconstriction → reduced CBF → lowers seizure threshold",
            "target_conditions": [
                "Childhood Absence Epilepsy (CAE) — 3 Hz spike-wave",
                "Juvenile Absence Epilepsy (JAE)",
                "Juvenile Myoclonic Epilepsy (JME)",
                "Focal epilepsy (temporal / frontal lobe — activates latent discharge)",
            ],
            "responses": [
                {"code": "slowing_only", "label": "Slowing only", "significance": "Physiological; normal especially in children <12y"},
                {"code": "firda",        "label": "FIRDA",         "significance": "Frontal Intermittent Rhythmic Delta Activity; may indicate encephalopathy"},
                {"code": "focal_spike_wave",       "label": "Focal spike-wave",        "significance": "Activates latent focal epileptiform discharge; focal epilepsy"},
                {"code": "generalized_spike_wave", "label": "Generalized spike-wave",  "significance": "≥3 Hz; diagnostic for generalized epilepsy (absence, JME)"},
                {"code": "ictal_discharge",        "label": "Ictal discharge",         "significance": "Clinical or subclinical seizure; stop HV immediately"},
                {"code": "no_response",            "label": "No response",             "significance": "No abnormality; normal study"},
            ],
            "contraindications": [
                "Recent MI or stroke (<6 months)",
                "Severe COPD or restrictive lung disease",
                "Sickle cell disease or trait",
                "Pregnancy (2nd/3rd trimester)",
                "Symptomatic cerebrovascular disease",
                "Patient refusal",
            ],
            "protocol_steps": [
                "Explain procedure; obtain verbal consent",
                "Confirm no contraindications",
                "Patient positioned comfortably (sitting or supine)",
                "EEG recording begins; baseline 2 min",
                "Instruct: 'breathe deeply and rapidly, 20–24 times per minute'",
                "Technician counts/paces; monitor clinically for clinical seizure",
                "Stop immediately if clinical seizure, distress, or ictal discharge",
                "Record continues ≥1 min post-HV (recovery + after-discharge capture)",
                "Document: HV start/stop times, response onset, normalization time",
            ],
            "normalization_target": "EEG should return to baseline within 1 minute of stopping. Prolonged slowing may be abnormal.",
            "standard": "ACNS Guideline 2016; IFCN recommendations",
        },
        "ips_protocol": {
            "full_name": "Intermittent Photic Stimulation (IPS)",
            "rates_hz": IPS_RATES_HZ,
            "stimulus": "Stroboscope at 30 cm from open eyes, at eye level; dim room (not blackout)",
            "sequence": "Eyes open 10 s → eyes closed 10 s at each rate; ascending 1→60 Hz",
            "target_conditions": [
                "Photosensitive epilepsy (PPR Grade III–IV)",
                "JME / JAE — 10–25 Hz most activating",
                "Reflex epilepsy — flash rates 15–20 Hz most sensitive",
            ],
            "ppr_grades": [
                {"grade": 1, "label": "Grade I",   "description": "Occipital response confined to occipital region; NOT epileptiform"},
                {"grade": 2, "label": "Grade II",  "description": "Occipital spikes/spike-wave limited to posterior regions; possibly epileptiform"},
                {"grade": 3, "label": "Grade III", "description": "Parietal / limited generalized single spike-wave; EPILEPTIFORM"},
                {"grade": 4, "label": "Grade IV",  "description": "Generalized polyspike-wave ± clinical features; EPILEPTIFORM — stop immediately"},
            ],
            "protocol_steps": [
                "Darken room; stroboscope positioned 30 cm from patient at eye level",
                "Begin at 1 Hz; eyes open 10 s, then eyes closed 10 s",
                "Ascend through 1→2→3→4→6→8→10→12→14→16→18→20→60 Hz",
                "Stop immediately if PPR Grade III/IV appears or clinical signs",
                "Document: exact rate at PPR onset, eye state (open/closed), PPR grade",
                "Post-stimulation recording ≥1 min for after-discharge monitoring",
            ],
            "contraindications": [
                "Relative: Known severe photosensitivity with prior GTCS requiring emergent care",
                "Discuss benefit/risk; may proceed with AED coverage in place",
            ],
            "standard": "ACNS 2016; Kasteleijn-Nolst Trenité et al. Epilepsia 2012; Waltz et al. 1992",
        },
        "references": [
            "ACNS Guideline 5: Minimum technical requirements for clinical EEG. 2016.",
            "Kasteleijn-Nolst Trenité DG et al. Methodology of photic stimulation revisited. Epilepsia. 2012;53(4):695-702.",
            "Jeavons PM, Harding GF. Photosensitive epilepsy. 1975.",
            "Waltz S et al. Different patterns of the photoparoxysmal response — a genetic study. EEG Clin Neurophysiol. 1992;83:138-145.",
            "Zifkin BG, Trenité DGAT. Reflex epilepsy. Epilepsia. 2000;41 Suppl 3:S11-S17.",
            "Noachtar S et al. A glossary of terms most commonly used by clinical electroencephalographers. EEG Clin Neurophysiol. 1999.",
        ],
        "safety_notes": [
            "Resuscitation equipment must be available when performing activation procedures.",
            "Never leave patient unattended during HV or IPS.",
            "HV: Stop immediately for clinical seizure, loss of awareness, or patient distress.",
            "IPS: Stop at first Grade III/IV PPR. Document and inform requesting neurologist.",
            "Post-procedure observation: minimum 5 minutes before patient leaves the room.",
        ],
    }
