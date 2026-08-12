"""VNS Therapy Monitoring Dashboard — Vagus Nerve Stimulation for drug-resistant epilepsy.

Vagus Nerve Stimulation (VNS) is a neuromodulation therapy approved for adjunctive
treatment of drug-resistant epilepsy in patients who are not surgical candidates or
who have failed epilepsy surgery. An implanted pulse generator delivers electrical
stimulation to the left vagus nerve in the neck, modulating brain excitability
through brainstem and thalamo-cortical pathways.

Device: LivaNova SenTiva™ / Demipulse® series
Approval: FDA 1997 (≥12 yr), CE mark, Health Canada

Key parameters:
- Output current (mA): primary efficacy driver
- Frequency (Hz): typically 20–30 Hz
- Pulse width (μs): typically 250–500 μs
- ON time (sec): stimulation burst duration
- OFF time (min): inter-burst interval (duty cycle)
- AutoStim: cardiac-based ictal detection → immediate stimulation

Response definition: ≥50% reduction in seizure frequency from baseline (ITT).
"""

import os
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), '..')


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# ── Simulated cohort data (30 epilepsy patients; 12 on VNS) ──────────────────
# All figures derived from published VNS meta-analyses and product specifications.
# Clinical.db patients match the existing 30-patient EEG cohort.

VNS_PATIENTS = [
    {"patient_id": "P004", "age": 34, "sex": "F", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2019, "model": "SenTiva",
     "output_current_ma": 2.25, "frequency_hz": 30, "pulse_width_us": 500,
     "on_time_sec": 30, "off_time_min": 5, "autostim": True,
     "baseline_seizures_month": 18, "current_seizures_month": 7,
     "pct_reduction": 61, "response": "responder",
     "battery_pct": 72, "battery_years_left": 3.1,
     "side_effects": ["hoarseness", "cough"],
     "therapy_years": 5.2},
    {"patient_id": "P007", "age": 28, "sex": "M", "epilepsy_type": "Generalized",
     "drug_resistant": True, "implant_year": 2020, "model": "SenTiva",
     "output_current_ma": 1.75, "frequency_hz": 20, "pulse_width_us": 500,
     "on_time_sec": 30, "off_time_min": 5, "autostim": False,
     "baseline_seizures_month": 24, "current_seizures_month": 11,
     "pct_reduction": 54, "response": "responder",
     "battery_pct": 55, "battery_years_left": 2.4,
     "side_effects": ["hoarseness"],
     "therapy_years": 4.1},
    {"patient_id": "P011", "age": 42, "sex": "F", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2017, "model": "Demipulse",
     "output_current_ma": 2.75, "frequency_hz": 30, "pulse_width_us": 500,
     "on_time_sec": 30, "off_time_min": 5, "autostim": False,
     "baseline_seizures_month": 30, "current_seizures_month": 8,
     "pct_reduction": 73, "response": "responder",
     "battery_pct": 22, "battery_years_left": 0.7,
     "side_effects": ["hoarseness", "dyspnea", "paresthesia"],
     "therapy_years": 7.3},
    {"patient_id": "P015", "age": 19, "sex": "M", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2022, "model": "SenTiva",
     "output_current_ma": 1.25, "frequency_hz": 20, "pulse_width_us": 250,
     "on_time_sec": 30, "off_time_min": 5, "autostim": True,
     "baseline_seizures_month": 12, "current_seizures_month": 7,
     "pct_reduction": 42, "response": "partial",
     "battery_pct": 88, "battery_years_left": 4.2,
     "side_effects": ["cough"],
     "therapy_years": 2.0},
    {"patient_id": "P019", "age": 55, "sex": "F", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2018, "model": "Demipulse",
     "output_current_ma": 2.50, "frequency_hz": 30, "pulse_width_us": 500,
     "on_time_sec": 30, "off_time_min": 5, "autostim": False,
     "baseline_seizures_month": 20, "current_seizures_month": 9,
     "pct_reduction": 55, "response": "responder",
     "battery_pct": 38, "battery_years_left": 1.5,
     "side_effects": ["hoarseness", "dysphagia"],
     "therapy_years": 6.2},
    {"patient_id": "P022", "age": 31, "sex": "M", "epilepsy_type": "Generalized",
     "drug_resistant": True, "implant_year": 2021, "model": "SenTiva",
     "output_current_ma": 1.00, "frequency_hz": 20, "pulse_width_us": 250,
     "on_time_sec": 30, "off_time_min": 3, "autostim": True,
     "baseline_seizures_month": 15, "current_seizures_month": 13,
     "pct_reduction": 13, "response": "non-responder",
     "battery_pct": 65, "battery_years_left": 3.4,
     "side_effects": [],
     "therapy_years": 3.1},
    {"patient_id": "P025", "age": 47, "sex": "F", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2016, "model": "Demipulse",
     "output_current_ma": 3.00, "frequency_hz": 30, "pulse_width_us": 500,
     "on_time_sec": 30, "off_time_min": 5, "autostim": False,
     "baseline_seizures_month": 35, "current_seizures_month": 11,
     "pct_reduction": 69, "response": "responder",
     "battery_pct": 14, "battery_years_left": 0.4,
     "side_effects": ["hoarseness", "paresthesia"],
     "therapy_years": 8.5},
    {"patient_id": "P028", "age": 23, "sex": "M", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2023, "model": "SenTiva",
     "output_current_ma": 0.75, "frequency_hz": 20, "pulse_width_us": 250,
     "on_time_sec": 30, "off_time_min": 5, "autostim": True,
     "baseline_seizures_month": 10, "current_seizures_month": 6,
     "pct_reduction": 40, "response": "partial",
     "battery_pct": 94, "battery_years_left": 4.8,
     "side_effects": [],
     "therapy_years": 1.3},
    {"patient_id": "P002", "age": 38, "sex": "F", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2020, "model": "SenTiva",
     "output_current_ma": 2.00, "frequency_hz": 25, "pulse_width_us": 500,
     "on_time_sec": 30, "off_time_min": 5, "autostim": True,
     "baseline_seizures_month": 22, "current_seizures_month": 8,
     "pct_reduction": 64, "response": "responder",
     "battery_pct": 58, "battery_years_left": 2.6,
     "side_effects": ["hoarseness", "cough"],
     "therapy_years": 4.3},
    {"patient_id": "P009", "age": 52, "sex": "M", "epilepsy_type": "Generalized",
     "drug_resistant": True, "implant_year": 2019, "model": "Demipulse",
     "output_current_ma": 2.25, "frequency_hz": 30, "pulse_width_us": 500,
     "on_time_sec": 30, "off_time_min": 5, "autostim": False,
     "baseline_seizures_month": 28, "current_seizures_month": 17,
     "pct_reduction": 39, "response": "partial",
     "battery_pct": 33, "battery_years_left": 1.2,
     "side_effects": ["hoarseness"],
     "therapy_years": 5.4},
    {"patient_id": "P013", "age": 29, "sex": "F", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2021, "model": "SenTiva",
     "output_current_ma": 1.50, "frequency_hz": 25, "pulse_width_us": 500,
     "on_time_sec": 30, "off_time_min": 5, "autostim": True,
     "baseline_seizures_month": 16, "current_seizures_month": 5,
     "pct_reduction": 69, "response": "responder",
     "battery_pct": 71, "battery_years_left": 3.3,
     "side_effects": ["cough"],
     "therapy_years": 3.0},
    {"patient_id": "P030", "age": 44, "sex": "M", "epilepsy_type": "Focal",
     "drug_resistant": True, "implant_year": 2022, "model": "SenTiva",
     "output_current_ma": 1.00, "frequency_hz": 20, "pulse_width_us": 250,
     "on_time_sec": 30, "off_time_min": 5, "autostim": False,
     "baseline_seizures_month": 8, "current_seizures_month": 8,
     "pct_reduction": 0, "response": "non-responder",
     "battery_pct": 83, "battery_years_left": 3.9,
     "side_effects": [],
     "therapy_years": 1.9},
]

# Monthly seizure frequency for 3 representative patients (12 months retrospective)
MONTHLY_TRENDS = {
    "P004": [17, 16, 15, 13, 12, 10, 9, 8, 8, 7, 7, 7],
    "P011": [28, 25, 22, 18, 15, 12, 10, 9, 8, 8, 8, 8],
    "P022": [15, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13],
}

MONTHS_LABELS = ["Sep'25", "Oct'25", "Nov'25", "Dec'25", "Jan'26", "Feb'26",
                 "Mar'26", "Apr'26", "May'26", "Jun'26", "Jul'26", "Aug'26"]


def _response_counts():
    resp = partial = non = 0
    for p in VNS_PATIENTS:
        if p["response"] == "responder":
            resp += 1
        elif p["response"] == "partial":
            partial += 1
        else:
            non += 1
    return resp, partial, non


def _side_effect_counts():
    from collections import Counter
    se_all = []
    for p in VNS_PATIENTS:
        se_all.extend(p["side_effects"])
    return Counter(se_all)


# ── Public API ────────────────────────────────────────────────────────────────

def vns_overview():
    """Top-level KPIs for VNS Therapy Monitoring Dashboard."""
    total = len(VNS_PATIENTS)
    resp, partial, non = _response_counts()
    mean_reduction = round(sum(p["pct_reduction"] for p in VNS_PATIENTS) / total, 1)
    low_battery = [p for p in VNS_PATIENTS if p["battery_pct"] < 20]
    autostim_on = sum(1 for p in VNS_PATIENTS if p["autostim"])
    senTiva = sum(1 for p in VNS_PATIENTS if p["model"] == "SenTiva")
    focal = sum(1 for p in VNS_PATIENTS if p["epilepsy_type"] == "Focal")

    return {
        "generated_at": _now_iso(),
        "kpis": {
            "total_vns_patients": total,
            "pct_of_cohort": round(total / 30 * 100, 1),
            "responder_rate_pct": round(resp / total * 100, 1),
            "responders": resp,
            "partial_responders": partial,
            "non_responders": non,
            "mean_seizure_reduction_pct": mean_reduction,
            "autostim_enabled": autostim_on,
            "low_battery_alert": len(low_battery),
            "model_sentiva": senTiva,
            "model_demipulse": total - senTiva,
            "focal_epilepsy_pct": round(focal / total * 100, 1),
            "mean_therapy_years": round(sum(p["therapy_years"] for p in VNS_PATIENTS) / total, 1),
        },
        "response_distribution": {
            "responder": resp,
            "partial": partial,
            "non_responder": non,
        },
        "battery_alerts": [
            {"patient_id": p["patient_id"], "battery_pct": p["battery_pct"],
             "years_left": p["battery_years_left"]}
            for p in sorted(VNS_PATIENTS, key=lambda x: x["battery_pct"])[:3]
        ],
        "parameter_summary": {
            "mean_output_current_ma": round(
                sum(p["output_current_ma"] for p in VNS_PATIENTS) / total, 2),
            "freq_20hz_count": sum(1 for p in VNS_PATIENTS if p["frequency_hz"] == 20),
            "freq_25hz_count": sum(1 for p in VNS_PATIENTS if p["frequency_hz"] == 25),
            "freq_30hz_count": sum(1 for p in VNS_PATIENTS if p["frequency_hz"] == 30),
            "pulse_250us_count": sum(1 for p in VNS_PATIENTS if p["pulse_width_us"] == 250),
            "pulse_500us_count": sum(1 for p in VNS_PATIENTS if p["pulse_width_us"] == 500),
        },
        "monthly_trends": {
            "months": MONTHS_LABELS,
            "patients": {
                pid: {"seizures": vals, "label": f"Patient {pid}"}
                for pid, vals in MONTHLY_TRENDS.items()
            }
        },
        "side_effects_summary": dict(_side_effect_counts()),
        "references": [
            "LivaNova SenTiva VNS Therapy System Physician's Manual 2023",
            "Fisher RS et al. VNS for Epilepsy — A Summary of Evidence. Epilepsy Behav 2017;88:11-20",
            "Englot DJ et al. Predictors of Seizure Freedom After VNS. J Neurosurg 2011;115:1248-55",
            "NICE NG217: Epilepsies in Children, Young People and Adults. 2022",
            "Handforth A et al. VNS Therapy for Partial-Onset Seizures. Neurology 1998;51:48-55",
        ],
    }


def vns_breakdown():
    """Per-patient VNS parameters, response, battery, and side-effect detail."""
    se_counts = _side_effect_counts()
    patients_out = []
    for p in sorted(VNS_PATIENTS, key=lambda x: -x["pct_reduction"]):
        patients_out.append({
            "patient_id": p["patient_id"],
            "age": p["age"],
            "sex": p["sex"],
            "epilepsy_type": p["epilepsy_type"],
            "model": p["model"],
            "implant_year": p["implant_year"],
            "therapy_years": p["therapy_years"],
            "output_current_ma": p["output_current_ma"],
            "frequency_hz": p["frequency_hz"],
            "pulse_width_us": p["pulse_width_us"],
            "on_time_sec": p["on_time_sec"],
            "off_time_min": p["off_time_min"],
            "autostim": p["autostim"],
            "baseline_sz_month": p["baseline_seizures_month"],
            "current_sz_month": p["current_seizures_month"],
            "pct_reduction": p["pct_reduction"],
            "response": p["response"],
            "battery_pct": p["battery_pct"],
            "battery_years_left": p["battery_years_left"],
            "battery_alert": p["battery_pct"] < 20,
            "side_effects": p["side_effects"],
        })

    se_by_response = {"responder": {}, "partial": {}, "non_responder": {}}
    for p in VNS_PATIENTS:
        key = p["response"].replace("-", "_")
        for se in p["side_effects"]:
            se_by_response[key][se] = se_by_response[key].get(se, 0) + 1

    return {
        "generated_at": _now_iso(),
        "patients": patients_out,
        "side_effects": {
            "all_patients": dict(se_counts),
            "by_response_group": se_by_response,
            "ranked": [
                {"effect": k, "count": v, "pct": round(v / len(VNS_PATIENTS) * 100, 1)}
                for k, v in sorted(se_counts.items(), key=lambda x: -x[1])
            ],
        },
        "parameter_distributions": {
            "current_bins": {
                "<1.0 mA": sum(1 for p in VNS_PATIENTS if p["output_current_ma"] < 1.0),
                "1.0–1.9 mA": sum(1 for p in VNS_PATIENTS if 1.0 <= p["output_current_ma"] < 2.0),
                "2.0–2.9 mA": sum(1 for p in VNS_PATIENTS if 2.0 <= p["output_current_ma"] < 3.0),
                "≥3.0 mA": sum(1 for p in VNS_PATIENTS if p["output_current_ma"] >= 3.0),
            },
            "frequency_dist": {
                "20 Hz": sum(1 for p in VNS_PATIENTS if p["frequency_hz"] == 20),
                "25 Hz": sum(1 for p in VNS_PATIENTS if p["frequency_hz"] == 25),
                "30 Hz": sum(1 for p in VNS_PATIENTS if p["frequency_hz"] == 30),
            },
            "battery_bins": {
                "Critical (<20%)": sum(1 for p in VNS_PATIENTS if p["battery_pct"] < 20),
                "Low (20–40%)": sum(1 for p in VNS_PATIENTS if 20 <= p["battery_pct"] < 40),
                "Moderate (40–70%)": sum(1 for p in VNS_PATIENTS if 40 <= p["battery_pct"] < 70),
                "Good (≥70%)": sum(1 for p in VNS_PATIENTS if p["battery_pct"] >= 70),
            },
        },
        "autostim_comparison": {
            "autostim_on": {
                "n": sum(1 for p in VNS_PATIENTS if p["autostim"]),
                "mean_reduction": round(
                    sum(p["pct_reduction"] for p in VNS_PATIENTS if p["autostim"]) /
                    max(1, sum(1 for p in VNS_PATIENTS if p["autostim"])), 1),
            },
            "autostim_off": {
                "n": sum(1 for p in VNS_PATIENTS if not p["autostim"]),
                "mean_reduction": round(
                    sum(p["pct_reduction"] for p in VNS_PATIENTS if not p["autostim"]) /
                    max(1, sum(1 for p in VNS_PATIENTS if not p["autostim"])), 1),
            },
        },
        "therapy_duration_bands": {
            "<2 years": sum(1 for p in VNS_PATIENTS if p["therapy_years"] < 2),
            "2–4 years": sum(1 for p in VNS_PATIENTS if 2 <= p["therapy_years"] < 4),
            "4–6 years": sum(1 for p in VNS_PATIENTS if 4 <= p["therapy_years"] < 6),
            "≥6 years": sum(1 for p in VNS_PATIENTS if p["therapy_years"] >= 6),
        },
    }


def definitions():
    """Clinical definitions, parameter references, and evidence base for VNS Therapy."""
    return {
        "generated_at": _now_iso(),
        "what_is_vns": (
            "Vagus Nerve Stimulation (VNS) is an adjunctive neuromodulation therapy for "
            "drug-resistant epilepsy. A pulse generator implanted subcutaneously in the chest "
            "delivers intermittent electrical stimulation to the left vagus nerve via a helical "
            "electrode. The vagus nerve projects to the nucleus tractus solitarius, which relays "
            "signals to the locus coeruleus, raphe nuclei, and thalamus, broadly modulating "
            "cortical excitability and seizure threshold."
        ),
        "indications": [
            "Drug-resistant focal epilepsy (≥2 AED failures at appropriate doses)",
            "Generalized epilepsy refractory to medication",
            "Lennox-Gastaut syndrome",
            "Patients not suitable for resective surgery or failed prior surgery",
            "Age ≥4 years (approved range varies by country)",
        ],
        "device_models": {
            "SenTiva (Model 1000)": (
                "LivaNova's newest implantable pulse generator (introduced 2017). "
                "Features cardiac-based seizure detection (AutoStim™), AspireSR® "
                "responsive stimulation, smartphone remote-patient-management app, "
                "and longer battery life (~10 years at standard settings)."
            ),
            "Demipulse (Model 103)": (
                "Previous-generation device with standard programmed stimulation cycles. "
                "Reliable workhorse; no AutoStim. Battery life ~3–5 years at standard settings."
            ),
        },
        "stimulation_parameters": {
            "output_current_ma": {
                "range": "0.25–3.50 mA",
                "typical_start": "0.25 mA",
                "typical_target": "1.5–3.0 mA (titrated over 3–12 months)",
                "note": "Primary efficacy driver; titrated upward as tolerated",
            },
            "frequency_hz": {
                "range": "1–145 Hz",
                "standard": "20–30 Hz",
                "note": "Higher frequency may increase efficacy but reduces battery life",
            },
            "pulse_width_us": {
                "range": "130–1000 μs",
                "standard": "250–500 μs",
                "note": "Wider pulse activates more fibers; affects tolerance",
            },
            "on_time_sec": {
                "standard": "30 sec",
                "note": "Duration of each stimulation burst",
            },
            "off_time_min": {
                "standard": "5 min",
                "duty_cycle": "~9%",
                "note": "Shorter off-times increase duty cycle; can reduce battery life",
            },
            "autostim": {
                "description": "Heart rate increase detected → immediate extra stimulation burst",
                "ictal_hr_threshold_pct": 20,
                "note": "Detects tachycardic signature of many focal seizures; not all seizures trigger",
            },
        },
        "response_definitions": {
            "responder": "≥50% reduction in seizure frequency from pre-implant baseline (ITT)",
            "partial_responder": "25–49% reduction",
            "non_responder": "<25% reduction",
            "seizure_free": "<1 seizure per 3 months (rare with VNS monotherapy)",
        },
        "efficacy_evidence": {
            "e03_e05_trials": {
                "responder_rate": "23–31% (3-month acute phase)",
                "note": "Pivotal FDA registration trials; 3-month high vs. low stimulation",
                "reference": "Handforth et al. Neurology 1998",
            },
            "long_term_meta_analysis": {
                "responder_rate": "~55–60% at 2 years",
                "mean_seizure_reduction": "~50%",
                "reference": "Fisher et al. Epilepsy Behav 2017",
                "note": "Efficacy improves with longer device exposure (VNS 'honeymoon' effect)",
            },
            "englot_predictors": {
                "positive_predictors": ["Focal onset", "Older age", "Post-traumatic etiology",
                                        "Higher output current"],
                "negative_predictors": ["Generalized epilepsy", "Lennox-Gastaut", "Tuberous sclerosis"],
                "reference": "Englot et al. J Neurosurg 2011",
            },
        },
        "side_effects": {
            "hoarseness_laryngeal_paresthesia": {
                "prevalence_pct": 66,
                "management": "Reduce output current or pulse width; usually improves with dose titration",
            },
            "cough": {
                "prevalence_pct": 33,
                "management": "Intermittent; typically during ON phase; rarely requires intervention",
            },
            "dyspnea": {
                "prevalence_pct": 17,
                "management": "Reduce duty cycle; switch to biphasic pulse waveform",
            },
            "dysphagia": {
                "prevalence_pct": 8,
                "management": "Dose reduction; usually mild",
            },
            "paresthesia_neck": {
                "prevalence_pct": 25,
                "management": "Reduce output current; most patients adapt over time",
            },
            "mood_benefit": {
                "note": "VNS has adjunctive antidepressant effect (FDA-approved for TRD 2005) — relevant in epilepsy comorbid depression",
            },
        },
        "battery_management": {
            "eos_warning": "Battery < 20% → schedule replacement surgery within 3–6 months",
            "replacement": "Outpatient surgery; lead is typically preserved and reconnected",
            "monitoring_interval": "6–12 months for battery check via programmer wand",
        },
        "ai_integration": {
            "autostim_detection": "Cardiac-based (tachycardia at seizure onset) — misses seizures without HR change",
            "wearable_extension": "EEG + wrist-worn motion + HR → multi-modal ictal detection (research phase)",
            "parameter_optimization": "ML-guided parameter titration based on seizure diary + objective response tracking (research)",
            "seizure_forecasting_role": "VNS magnet can be activated prophylactically in high-risk windows identified by AI forecast",
        },
        "nice_guidance": {
            "source": "NICE NG217: Epilepsies in Children, Young People and Adults. 2022",
            "recommendation": "Offer VNS to people with drug-resistant epilepsy for whom surgery is not appropriate or has been unsuccessful",
            "review_schedule": "6-monthly clinic review for device check, seizure frequency, and dose optimisation",
        },
    }
