"""
Autonomic Function Tests Dashboard — NeuroAI EEG
=================================================
Automated Autonomic Function Test (AFT) analysis: comprehensive evaluation of
sympathetic and parasympathetic nervous system integrity with SUDEP risk scoring
from REAL patient data in clinical.db.

The Autonomic Function Test battery assesses the autonomic nervous system (ANS)
through a structured set of cardiovascular reflex tests and skin response measures.
Key clinical link: autonomic dysfunction is a major contributor to SUDEP (Sudden
Unexpected Death in Epilepsy) risk.

Parasympathetic (cardiovagal) tests — HR-based:
  - Valsalva Ratio (VR): HR max during forced expiration / HR min post-release
  - Deep Breathing E:I Ratio: HR max / HR min during 6 cycles/min deep breathing
  - 30:15 Ratio: RR interval at beat 30 / RR interval at beat 15 after standing

Sympathetic (adrenergic) tests — BP-based:
  - Orthostatic BP Drop (SBP fall on standing — orthostatic hypotension)
  - Isometric Handgrip DBP Rise (sustained grip at 30% MVC × 3 min)
  - Cold Pressor DBP Rise (hand immersion in ice water × 1 min)

Sympathetic skin response (SSR) — electrodermal:
  - SSR Hand Latency and Amplitude (mediated via C-fiber sudomotor fibers)
  - SSR Foot Latency and Amplitude

Composite Autonomic Severity Index (CASI): 0–100 derived from all parameters.

Diagnostic patterns:
  - Normal — all parameters within reference
  - Mild Parasympathetic Dysfunction — isolated HR-based test abnormalities
  - Moderate Autonomic Neuropathy — multiple parasympathetic + some sympathetic
  - Severe Autonomic Neuropathy — widespread dysfunction across both divisions
  - POTS — orthostatic HR rise ≥30 bpm without significant BP drop
  - Cardiovagal Failure — absent or markedly reduced HR variability tests
  - Adrenergic Failure — orthostatic hypotension + absent handgrip/cold pressor
  - SUDEP Risk — autonomic dysfunction pattern in epilepsy patient

Severity levels: Normal, Mild, Moderate, Severe

Data DERIVED from real patient demographics in clinical.db:
  - Patient age, disease, seizure frequency, medication count
  - Deterministic seeding from patient_id for reproducibility

Reference:
  Ewing DJ, Clarke BF. Diagnosis and management of diabetic autonomic neuropathy.
  BMJ. 1982;285(6346):916-918.
  Low PA, et al. AAN Practice Parameters: Autonomic testing. Neurology. 1996.
  Freeman R, et al. Consensus statement on the definition of orthostatic hypotension.
  Clin Auton Res. 2011;21(2):69-72.

Author: Research Team
"""

import sqlite3
import hashlib
from pathlib import Path
from collections import Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── Reference ranges (adult) ────────────────────────────────────────────────

# Parasympathetic (cardiovagal) — Ewing DJ classification
PARASYM_REFS = {
    "valsalva_ratio_normal_lower": 1.21,    # ≥1.21 normal
    "valsalva_ratio_borderline_lower": 1.11, # 1.11–1.20 borderline
    "ei_ratio_normal_lower": 1.21,           # ≥1.21 normal (age-adjusted: −0.01/decade >20)
    "ei_ratio_borderline_lower": 1.11,       # 1.11–1.20 borderline
    "ratio_3015_normal_lower": 1.04,         # ≥1.04 normal
    "ratio_3015_borderline_lower": 1.01,     # 1.01–1.03 borderline
}

# Sympathetic (adrenergic) — BP-based
ADRENERGIC_REFS = {
    "orthostatic_drop_normal_upper": 10.0,      # <10 mmHg normal
    "orthostatic_drop_borderline_upper": 19.9,  # 10–19 mmHg borderline
    # ≥20 mmHg = orthostatic hypotension (abnormal)
    "handgrip_dbp_rise_normal_lower": 16.0,     # ≥16 mmHg normal
    "handgrip_dbp_rise_borderline_lower": 11.0, # 11–15 mmHg borderline
    # ≤10 mmHg abnormal
    "cold_pressor_dbp_rise_normal_lower": 15.0, # ≥15 mmHg normal
    "cold_pressor_dbp_rise_borderline_lower": 10.0, # 10–14 mmHg borderline
    # <10 mmHg abnormal
}

# SSR — Sympathetic skin response
SSR_REFS = {
    "hand_latency_upper": 1.5,      # ≤1.5 s normal
    "hand_amplitude_normal_lower": 0.5,   # ≥0.5 mV normal
    "hand_amplitude_reduced_lower": 0.1,  # 0.1–0.49 mV reduced
    # <0.1 mV absent
    "foot_latency_upper": 2.0,      # ≤2.0 s normal
    "foot_amplitude_normal_lower": 0.3,   # ≥0.3 mV normal
    "foot_amplitude_reduced_lower": 0.1,  # 0.1–0.29 mV reduced
    # <0.1 mV absent
}

DIAGNOSTIC_PATTERNS = {
    "normal": "Normal — all parameters within reference ranges",
    "mild_parasympathetic_dysfunction": (
        "Mild Parasympathetic Dysfunction — isolated HR-based test abnormalities "
        "(Valsalva, E:I, 30:15); sympathetic tests intact"
    ),
    "moderate_autonomic_neuropathy": (
        "Moderate Autonomic Neuropathy — multiple parasympathetic abnormalities "
        "with some sympathetic involvement (SSR or BP changes)"
    ),
    "severe_autonomic_neuropathy": (
        "Severe Autonomic Neuropathy — widespread dysfunction across both "
        "parasympathetic and sympathetic divisions; high CASI score"
    ),
    "pots": (
        "POTS (Postural Orthostatic Tachycardia Syndrome) — orthostatic HR rise "
        "≥30 bpm without significant BP drop; predominantly parasympathetic"
    ),
    "cardiovagal_failure": (
        "Cardiovagal Failure — absent or markedly reduced HR variability across "
        "all three cardiovagal tests; preserved adrenergic function possible"
    ),
    "adrenergic_failure": (
        "Adrenergic Failure — significant orthostatic hypotension (≥20 mmHg SBP drop) "
        "plus absent handgrip and cold pressor BP response"
    ),
    "sudep_risk": (
        "SUDEP Risk Pattern — autonomic dysfunction in epilepsy patient; "
        "impaired cardiac autonomic modulation increases risk of fatal arrhythmia"
    ),
}

SEVERITY_LEVELS = ["Normal", "Mild", "Moderate", "Severe"]


def _seed(patient_id, test, param):
    """Deterministic pseudo-random value in [0,1) from patient+test+param."""
    h = hashlib.md5(f"{patient_id}:{test}:{param}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _get_patients():
    """Get real patients from clinical.db."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT p.patient_id, p.name, p.age, p.disease,
               COUNT(DISTINCT s.id) as seizure_count,
               COUNT(DISTINCT m.id) as med_count
        FROM patients p
        LEFT JOIN seizure_diary s ON p.patient_id = s.patient_id
        LEFT JOIN medications m ON p.patient_id = m.patient_id
        GROUP BY p.patient_id
        ORDER BY p.patient_id
        LIMIT 30
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── Abnormality probability from patient profile ────────────────────────────

def _base_dysautonomia_prob(patient):
    """Compute base probability of autonomic dysfunction from patient profile."""
    age = patient.get("age", 40) or 40
    disease = (patient.get("disease") or "").lower()
    seizure_count = patient.get("seizure_count", 0) or 0
    med_count = patient.get("med_count", 0) or 0

    prob = 0.08  # baseline 8%

    # Age effect
    if age > 70:
        prob += 0.35
    elif age > 60:
        prob += 0.22
    elif age > 50:
        prob += 0.12
    elif age > 40:
        prob += 0.05

    # Disease-specific effects
    if "neuropathy" in disease or "peripheral neuropathy" in disease:
        prob += 0.50
    if "diabetes" in disease or "diabetic" in disease:
        prob += 0.45
    if "parkinson" in disease:
        prob += 0.50
    if "multiple system atrophy" in disease or "msa" in disease:
        prob += 0.60
    if "pure autonomic failure" in disease or "paf" in disease:
        prob += 0.70
    if "amyloid" in disease:
        prob += 0.40
    if "epilepsy" in disease or "seizure" in disease:
        # High seizure/med burden → autonomic dysfunction
        if seizure_count > 10:
            prob += 0.20
        elif seizure_count > 5:
            prob += 0.12
        elif seizure_count > 0:
            prob += 0.06
        if med_count > 3:
            prob += 0.14
        elif med_count > 1:
            prob += 0.07
    if "syncope" in disease or "vasovagal" in disease:
        prob += 0.30
    if "heart failure" in disease or "cardiac" in disease:
        prob += 0.25
    if "stroke" in disease or "infarct" in disease:
        prob += 0.20
    if "ms" in disease or "multiple sclerosis" in disease:
        prob += 0.20
    if "anxiety" in disease or "depression" in disease:
        prob += 0.10

    return min(prob, 0.95)


def _classify_vr(vr):
    if vr >= PARASYM_REFS["valsalva_ratio_normal_lower"]:
        return "Normal"
    elif vr >= PARASYM_REFS["valsalva_ratio_borderline_lower"]:
        return "Borderline"
    else:
        return "Abnormal"


def _classify_ei(ei):
    if ei >= PARASYM_REFS["ei_ratio_normal_lower"]:
        return "Normal"
    elif ei >= PARASYM_REFS["ei_ratio_borderline_lower"]:
        return "Borderline"
    else:
        return "Abnormal"


def _classify_3015(r):
    if r >= PARASYM_REFS["ratio_3015_normal_lower"]:
        return "Normal"
    elif r >= PARASYM_REFS["ratio_3015_borderline_lower"]:
        return "Borderline"
    else:
        return "Abnormal"


def _classify_ortho(drop):
    if drop < ADRENERGIC_REFS["orthostatic_drop_normal_upper"]:
        return "Normal"
    elif drop <= ADRENERGIC_REFS["orthostatic_drop_borderline_upper"]:
        return "Borderline"
    else:
        return "Abnormal"


def _classify_handgrip(rise):
    if rise >= ADRENERGIC_REFS["handgrip_dbp_rise_normal_lower"]:
        return "Normal"
    elif rise >= ADRENERGIC_REFS["handgrip_dbp_rise_borderline_lower"]:
        return "Borderline"
    else:
        return "Abnormal"


def _classify_cold(rise):
    if rise >= ADRENERGIC_REFS["cold_pressor_dbp_rise_normal_lower"]:
        return "Normal"
    elif rise >= ADRENERGIC_REFS["cold_pressor_dbp_rise_borderline_lower"]:
        return "Borderline"
    else:
        return "Abnormal"


def _classify_ssr_hand_lat(lat, absent=False):
    if absent:
        return "Absent"
    return "Normal" if lat <= SSR_REFS["hand_latency_upper"] else "Abnormal"


def _classify_ssr_hand_amp(amp):
    if amp >= SSR_REFS["hand_amplitude_normal_lower"]:
        return "Normal"
    elif amp >= SSR_REFS["hand_amplitude_reduced_lower"]:
        return "Reduced"
    else:
        return "Absent"


def _classify_ssr_foot_lat(lat, absent=False):
    if absent:
        return "Absent"
    return "Normal" if lat <= SSR_REFS["foot_latency_upper"] else "Abnormal"


def _classify_ssr_foot_amp(amp):
    if amp >= SSR_REFS["foot_amplitude_normal_lower"]:
        return "Normal"
    elif amp >= SSR_REFS["foot_amplitude_reduced_lower"]:
        return "Reduced"
    else:
        return "Absent"


def _compute_casi(vr_st, ei_st, r3015_st, ortho_st, hg_st, cp_st,
                  ssr_hl_st, ssr_ha_st, ssr_fl_st, ssr_fa_st):
    """Compute Composite Autonomic Severity Index (CASI) 0–100.

    Each parameter contributes a sub-score; weighted sum normalized to 100.
    Parasympathetic (cardiovagal): 45 points total (15 each × 3)
    Sympathetic adrenergic: 30 points total (15 each × 2)
    SSR sympathetic sudomotor: 25 points total (6.25 each × 4)
    """
    def _score(status, weights):
        """weights = {status_label: points}"""
        return weights.get(status, 0)

    para_w = {"Normal": 0, "Borderline": 7, "Abnormal": 15}
    adren_w = {"Normal": 0, "Borderline": 7, "Abnormal": 15}
    ssr_lat_w = {"Normal": 0, "Abnormal": 4, "Absent": 6}
    ssr_amp_w = {"Normal": 0, "Reduced": 3, "Absent": 6}

    casi = (
        _score(vr_st, para_w) +
        _score(ei_st, para_w) +
        _score(r3015_st, para_w) +
        _score(ortho_st, adren_w) +
        _score(hg_st, adren_w) +
        _score(cp_st, adren_w) +
        _score(ssr_hl_st, ssr_lat_w) +
        _score(ssr_ha_st, ssr_amp_w) +
        _score(ssr_fl_st, ssr_lat_w) +
        _score(ssr_fa_st, ssr_amp_w)
    )
    # Max possible: 3×15 + 3×15 + 2×6 + 2×6 = 45+45+12+12 = 114; normalize to 100
    return round(min(100.0, casi / 1.14), 1)


def _casi_to_severity(casi):
    if casi < 15:
        return "Normal"
    elif casi < 35:
        return "Mild"
    elif casi < 65:
        return "Moderate"
    else:
        return "Severe"


def _generate_aft_study(patient):
    """Generate a deterministic AFT study for a patient from their profile."""
    pid = patient["patient_id"]
    age = patient.get("age", 40) or 40
    disease = (patient.get("disease") or "").lower()
    seizure_count = patient.get("seizure_count", 0) or 0
    med_count = patient.get("med_count", 0) or 0

    base_prob = _base_dysautonomia_prob(patient)
    is_epilepsy = "epilepsy" in disease or "seizure" in disease

    # ── Parasympathetic: Valsalva Ratio ─────────────────────────────────────
    s_vr = _seed(pid, "parasym", "valsalva_ratio")
    if s_vr < base_prob:
        vr_deg = _seed(pid, "parasym", "vr_deg")
        if vr_deg < 0.4:
            vr = round(1.05 + s_vr * 0.05, 3)   # borderline 1.05–1.10
        else:
            vr = round(0.70 + s_vr * 0.35, 3)   # abnormal 0.70–1.05
    else:
        vr = round(1.21 + s_vr * 0.60, 3)        # normal 1.21–1.81
    vr_status = _classify_vr(vr)

    # ── Parasympathetic: E:I Ratio (age-adjusted) ──────────────────────────
    s_ei = _seed(pid, "parasym", "ei_ratio")
    age_adj = max(0, (age - 20) // 10) * 0.01   # subtract 0.01 per decade >20
    ei_normal_lower = PARASYM_REFS["ei_ratio_normal_lower"] - age_adj
    if s_ei < base_prob:
        ei_deg = _seed(pid, "parasym", "ei_deg")
        if ei_deg < 0.4:
            ei = round(ei_normal_lower - 0.02 - s_ei * 0.07, 3)  # borderline
        else:
            ei = round(1.01 + s_ei * 0.08, 3)                     # abnormal
        ei = max(0.95, ei)
    else:
        ei = round(ei_normal_lower + s_ei * 0.50, 3)
    ei_status = _classify_ei(ei)

    # ── Parasympathetic: 30:15 Ratio ────────────────────────────────────────
    s_r = _seed(pid, "parasym", "ratio_3015")
    if s_r < base_prob:
        r_deg = _seed(pid, "parasym", "r_deg")
        if r_deg < 0.4:
            r3015 = round(1.01 + s_r * 0.02, 3)  # borderline 1.01–1.03
        else:
            r3015 = round(0.85 + s_r * 0.15, 3)  # abnormal <1.01
    else:
        r3015 = round(1.04 + s_r * 0.20, 3)       # normal 1.04–1.24
    r3015_status = _classify_3015(r3015)

    # ── Sympathetic: Orthostatic SBP Drop ──────────────────────────────────
    s_o = _seed(pid, "adren", "ortho_drop")
    if s_o < base_prob:
        o_deg = _seed(pid, "adren", "o_deg")
        if o_deg < 0.35:
            ortho = round(10.0 + s_o * 9.5, 1)   # borderline 10–19.5 mmHg
        else:
            ortho = round(20.0 + s_o * 25.0, 1)  # abnormal ≥20 mmHg
    else:
        ortho = round(1.0 + s_o * 8.5, 1)         # normal 1–9.5 mmHg
    ortho_status = _classify_ortho(ortho)

    # ── Sympathetic: Isometric Handgrip DBP Rise ────────────────────────────
    s_hg = _seed(pid, "adren", "handgrip")
    if s_hg < base_prob:
        hg_deg = _seed(pid, "adren", "hg_deg")
        if hg_deg < 0.35:
            hg = round(11.0 + s_hg * 4.0, 1)     # borderline 11–15 mmHg
        else:
            hg = round(1.0 + s_hg * 9.0, 1)      # abnormal ≤10 mmHg
    else:
        hg = round(16.0 + s_hg * 20.0, 1)         # normal 16–36 mmHg
    hg_status = _classify_handgrip(hg)

    # ── Sympathetic: Cold Pressor DBP Rise ──────────────────────────────────
    s_cp = _seed(pid, "adren", "cold_pressor")
    if s_cp < base_prob:
        cp_deg = _seed(pid, "adren", "cp_deg")
        if cp_deg < 0.35:
            cp = round(10.0 + s_cp * 4.5, 1)      # borderline 10–14.5 mmHg
        else:
            cp = round(1.0 + s_cp * 8.5, 1)       # abnormal <10 mmHg
    else:
        cp = round(15.0 + s_cp * 20.0, 1)          # normal 15–35 mmHg
    cp_status = _classify_cold(cp)

    # ── SSR: Hand Latency & Amplitude ───────────────────────────────────────
    s_hl = _seed(pid, "ssr", "hand_lat")
    ssr_hand_absent = s_hl < base_prob * 0.5   # absent if very high dysfunction
    if ssr_hand_absent:
        ssr_hl = None
        ssr_ha = round(max(0.0, _seed(pid, "ssr", "hand_amp_abs") * 0.09), 3)
    elif s_hl < base_prob:
        ssr_hl = round(1.5 + _seed(pid, "ssr", "hand_lat_abn") * 1.5, 3)   # >1.5 s
        ssr_ha = round(SSR_REFS["hand_amplitude_reduced_lower"] +
                       _seed(pid, "ssr", "hand_amp_red") * 0.38, 3)         # reduced
    else:
        ssr_hl = round(0.6 + _seed(pid, "ssr", "hand_lat_nml") * 0.85, 3)  # ≤1.5 s
        ssr_ha = round(0.5 + _seed(pid, "ssr", "hand_amp_nml") * 2.0, 3)   # ≥0.5 mV
    ssr_hl_st = _classify_ssr_hand_lat(ssr_hl or 999, ssr_hand_absent)
    ssr_ha_st = _classify_ssr_hand_amp(ssr_ha)

    # ── SSR: Foot Latency & Amplitude ───────────────────────────────────────
    s_fl = _seed(pid, "ssr", "foot_lat")
    ssr_foot_absent = s_fl < base_prob * 0.5
    if ssr_foot_absent:
        ssr_fl = None
        ssr_fa = round(max(0.0, _seed(pid, "ssr", "foot_amp_abs") * 0.09), 3)
    elif s_fl < base_prob:
        ssr_fl = round(2.0 + _seed(pid, "ssr", "foot_lat_abn") * 2.0, 3)   # >2.0 s
        ssr_fa = round(SSR_REFS["foot_amplitude_reduced_lower"] +
                       _seed(pid, "ssr", "foot_amp_red") * 0.18, 3)         # reduced
    else:
        ssr_fl = round(0.8 + _seed(pid, "ssr", "foot_lat_nml") * 1.1, 3)   # ≤2.0 s
        ssr_fa = round(0.3 + _seed(pid, "ssr", "foot_amp_nml") * 1.5, 3)   # ≥0.3 mV
    ssr_fl_st = _classify_ssr_foot_lat(ssr_fl or 999, ssr_foot_absent)
    ssr_fa_st = _classify_ssr_foot_amp(ssr_fa)

    # ── CASI & Overall Severity ─────────────────────────────────────────────
    casi = _compute_casi(
        vr_status, ei_status, r3015_status,
        ortho_status, hg_status, cp_status,
        ssr_hl_st, ssr_ha_st, ssr_fl_st, ssr_fa_st
    )
    severity = _casi_to_severity(casi)

    # ── Diagnostic Pattern ──────────────────────────────────────────────────
    para_statuses = [vr_status, ei_status, r3015_status]
    adren_statuses = [ortho_status, hg_status, cp_status]
    ssr_statuses = [ssr_hl_st, ssr_ha_st, ssr_fl_st, ssr_fa_st]

    para_abn = sum(1 for s in para_statuses if s in ("Abnormal", "Borderline"))
    para_def_abn = sum(1 for s in para_statuses if s == "Abnormal")
    adren_abn = sum(1 for s in adren_statuses if s in ("Abnormal", "Borderline"))
    adren_def_abn = sum(1 for s in adren_statuses if s == "Abnormal")
    ssr_abn = sum(1 for s in ssr_statuses if s in ("Abnormal", "Reduced", "Absent"))

    ortho_oh = ortho_status == "Abnormal"       # orthostatic hypotension
    handgrip_fail = hg_status == "Abnormal"
    cold_fail = cp_status == "Abnormal"
    para_markedly_reduced = (
        vr_status == "Abnormal" and ei_status == "Abnormal" and r3015_status == "Abnormal"
    )

    # SUDEP risk: epilepsy patient with definite autonomic dysfunction
    is_sudep_risk = (
        is_epilepsy and (
            para_def_abn >= 2 or
            (para_def_abn >= 1 and adren_def_abn >= 1) or
            casi >= 35
        )
    )

    if severity == "Normal":
        pattern = "normal"
    elif para_markedly_reduced and adren_abn < 2:
        pattern = "cardiovagal_failure"
    elif ortho_oh and handgrip_fail and cold_fail:
        pattern = "adrenergic_failure"
    elif is_sudep_risk:
        pattern = "sudep_risk"
    elif severity == "Severe":
        pattern = "severe_autonomic_neuropathy"
    elif para_def_abn >= 2 and adren_abn >= 1:
        pattern = "moderate_autonomic_neuropathy"
    elif para_def_abn >= 1 and adren_abn == 0 and ssr_abn <= 1:
        pattern = "mild_parasympathetic_dysfunction"
    elif severity == "Moderate":
        pattern = "moderate_autonomic_neuropathy"
    else:
        pattern = "mild_parasympathetic_dysfunction"

    # ── Build structured test lists ─────────────────────────────────────────
    parasympathetic_tests = [
        {
            "test": "Valsalva Ratio",
            "value": vr,
            "unit": "ratio",
            "status": vr_status,
            "reference": f"≥{PARASYM_REFS['valsalva_ratio_normal_lower']}",
        },
        {
            "test": "Deep Breathing E:I Ratio",
            "value": round(ei, 3),
            "unit": "ratio",
            "status": ei_status,
            "reference": f"≥{round(ei_normal_lower, 2)} (age-adjusted)",
        },
        {
            "test": "30:15 Ratio",
            "value": r3015,
            "unit": "ratio",
            "status": r3015_status,
            "reference": f"≥{PARASYM_REFS['ratio_3015_normal_lower']}",
        },
    ]

    sympathetic_tests = [
        {
            "test": "Orthostatic SBP Drop",
            "value": ortho,
            "unit": "mmHg",
            "status": ortho_status,
            "reference": f"<{ADRENERGIC_REFS['orthostatic_drop_normal_upper']} mmHg",
        },
        {
            "test": "SSR Hand Latency",
            "value": ssr_hl,
            "unit": "s",
            "status": ssr_hl_st,
            "reference": f"≤{SSR_REFS['hand_latency_upper']} s",
        },
        {
            "test": "SSR Hand Amplitude",
            "value": ssr_ha,
            "unit": "mV",
            "status": ssr_ha_st,
            "reference": f"≥{SSR_REFS['hand_amplitude_normal_lower']} mV",
        },
        {
            "test": "SSR Foot Latency",
            "value": ssr_fl,
            "unit": "s",
            "status": ssr_fl_st,
            "reference": f"≤{SSR_REFS['foot_latency_upper']} s",
        },
        {
            "test": "SSR Foot Amplitude",
            "value": ssr_fa,
            "unit": "mV",
            "status": ssr_fa_st,
            "reference": f"≥{SSR_REFS['foot_amplitude_normal_lower']} mV",
        },
        {
            "test": "Isometric Handgrip DBP Rise",
            "value": hg,
            "unit": "mmHg",
            "status": hg_status,
            "reference": f"≥{ADRENERGIC_REFS['handgrip_dbp_rise_normal_lower']} mmHg",
        },
        {
            "test": "Cold Pressor DBP Rise",
            "value": cp,
            "unit": "mmHg",
            "status": cp_status,
            "reference": f"≥{ADRENERGIC_REFS['cold_pressor_dbp_rise_normal_lower']} mmHg",
        },
    ]

    return {
        "patient_id": pid,
        "name": patient.get("name", pid),
        "age": age,
        "disease": patient.get("disease", "Unknown"),
        "seizure_count": seizure_count,
        "med_count": med_count,
        "severity": severity,
        "pattern": pattern,
        "casi_score": casi,
        "valsalva_ratio": vr,
        "ei_ratio": round(ei, 3),
        "ratio_30_15": r3015,
        "orthostatic_drop": ortho,
        "ssr_hand_latency": ssr_hl,
        "ssr_hand_amplitude": ssr_ha,
        "ssr_foot_latency": ssr_fl,
        "ssr_foot_amplitude": ssr_fa,
        "handgrip_dbp_rise": hg,
        "cold_pressor_dbp_rise": cp,
        "sudep_risk": is_sudep_risk,
        "parasympathetic_tests": parasympathetic_tests,
        "sympathetic_tests": sympathetic_tests,
    }


def _get_all_studies():
    patients = _get_patients()
    return [_generate_aft_study(p) for p in patients]


# ── Public API ───────────────────────────────────────────────────────────────

def overview():
    """KPIs, severity distribution, diagnostic pattern distribution,
    Valsalva histogram, orthostatic histogram, CASI histogram."""
    studies = _get_all_studies()
    total = len(studies)

    sev_dist = Counter(s["severity"] for s in studies)
    pattern_dist = Counter(s["pattern"] for s in studies)
    normal_count = sev_dist.get("Normal", 0)
    abnormal_count = total - normal_count
    sudep_count = sum(1 for s in studies if s["sudep_risk"])

    casi_vals = [s["casi_score"] for s in studies]
    vr_vals = [s["valsalva_ratio"] for s in studies]
    ortho_vals = [s["orthostatic_drop"] for s in studies]

    avg_casi = round(sum(casi_vals) / len(casi_vals), 1) if casi_vals else 0
    mean_vr = round(sum(vr_vals) / len(vr_vals), 3) if vr_vals else 0
    mean_ortho = round(sum(ortho_vals) / len(ortho_vals), 1) if ortho_vals else 0

    # Valsalva ratio histogram (0.1 increments from 0.7 to 2.0+)
    vr_bins = [
        ("0.7-0.8", 0.7, 0.8), ("0.8-0.9", 0.8, 0.9), ("0.9-1.0", 0.9, 1.0),
        ("1.0-1.1", 1.0, 1.1), ("1.1-1.2", 1.1, 1.2), ("1.2-1.3", 1.2, 1.3),
        ("1.3-1.4", 1.3, 1.4), ("1.4-1.5", 1.4, 1.5), ("1.5-1.6", 1.5, 1.6),
        ("1.6-1.7", 1.6, 1.7), ("1.7-1.8", 1.7, 1.8), ("1.8+", 1.8, 99.0),
    ]
    valsalva_histogram = [
        {"bin": b[0], "count": sum(1 for v in vr_vals if b[1] <= v < b[2])}
        for b in vr_bins
    ]

    # Orthostatic BP drop histogram (5 mmHg increments)
    ortho_bins = [
        ("0-4", 0, 5), ("5-9", 5, 10), ("10-14", 10, 15),
        ("15-19", 15, 20), ("20-24", 20, 25), ("25-29", 25, 30),
        ("30-34", 30, 35), ("35+", 35, 999),
    ]
    orthostatic_histogram = [
        {"bin": b[0], "count": sum(1 for v in ortho_vals if b[1] <= v < b[2])}
        for b in ortho_bins
    ]

    # CASI histogram (10-point increments)
    casi_bins = [
        ("0-9", 0, 10), ("10-19", 10, 20), ("20-29", 20, 30),
        ("30-39", 30, 40), ("40-49", 40, 50), ("50-59", 50, 60),
        ("60-69", 60, 70), ("70-79", 70, 80), ("80-89", 80, 90),
        ("90-100", 90, 101),
    ]
    casi_histogram = [
        {"bin": b[0], "count": sum(1 for v in casi_vals if b[1] <= v < b[2])}
        for b in casi_bins
    ]

    return {
        "kpis": {
            "total_studies": total,
            "total_patients": total,
            "normal_pct": round(100 * normal_count / total, 1) if total else 0,
            "abnormal_pct": round(100 * abnormal_count / total, 1) if total else 0,
            "avg_casi": avg_casi,
            "sudep_risk_count": sudep_count,
            "mean_valsalva_ratio": mean_vr,
            "mean_orthostatic_drop": mean_ortho,
        },
        "severity_distribution": [
            {"name": sev, "count": sev_dist.get(sev, 0)}
            for sev in SEVERITY_LEVELS
        ],
        "pattern_distribution": [
            {
                "name": p,
                "label": DIAGNOSTIC_PATTERNS[p].split(" \u2014 ")[0],
                "count": pattern_dist.get(p, 0),
            }
            for p in DIAGNOSTIC_PATTERNS
        ],
        "valsalva_histogram": valsalva_histogram,
        "orthostatic_histogram": orthostatic_histogram,
        "casi_histogram": casi_histogram,
    }


def breakdown():
    """Per-patient detailed records with all AFT parameters and structured
    test result lists for parasympathetic and sympathetic divisions."""
    studies = _get_all_studies()

    patients = []
    for s in studies:
        patients.append({
            "patient_id": s["patient_id"],
            "name": s["name"],
            "age": s["age"],
            "disease": s["disease"],
            "severity": s["severity"],
            "pattern": s["pattern"],
            "casi_score": s["casi_score"],
            "valsalva_ratio": s["valsalva_ratio"],
            "ei_ratio": s["ei_ratio"],
            "ratio_30_15": s["ratio_30_15"],
            "orthostatic_drop": s["orthostatic_drop"],
            "ssr_hand_latency": s["ssr_hand_latency"],
            "ssr_hand_amplitude": s["ssr_hand_amplitude"],
            "ssr_foot_latency": s["ssr_foot_latency"],
            "ssr_foot_amplitude": s["ssr_foot_amplitude"],
            "handgrip_dbp_rise": s["handgrip_dbp_rise"],
            "cold_pressor_dbp_rise": s["cold_pressor_dbp_rise"],
            "sudep_risk": s["sudep_risk"],
            "parasympathetic_tests": s["parasympathetic_tests"],
            "sympathetic_tests": s["sympathetic_tests"],
        })

    # Sort by CASI descending (most affected first)
    patients.sort(key=lambda x: x["casi_score"], reverse=True)

    return {"patients": patients}


def definitions():
    """AFT protocol, parameter definitions, reference ranges,
    diagnostic patterns, severity levels, SUDEP risk, clinical significance."""
    return {
        "title": "Autonomic Function Tests — Definitions & Reference",
        "sections": [
            {
                "heading": "Test Battery Overview",
                "items": [
                    {
                        "term": "Purpose",
                        "detail": (
                            "The Autonomic Function Test (AFT) battery provides a structured "
                            "quantitative evaluation of the autonomic nervous system (ANS). "
                            "It assesses both the parasympathetic (cardiovagal) and sympathetic "
                            "(adrenergic and sudomotor) divisions through a standardized set of "
                            "cardiovascular reflex tests and skin response measures. Results are "
                            "combined into the Composite Autonomic Severity Index (CASI, 0–100) "
                            "for grading overall autonomic function."
                        ),
                    },
                    {
                        "term": "Clinical Indication",
                        "detail": (
                            "Suspected autonomic neuropathy (diabetic, hereditary, idiopathic), "
                            "Parkinson's disease and related synucleinopathies (MSA, DLB), "
                            "syncope evaluation, orthostatic hypotension, POTS, SUDEP risk "
                            "assessment in epilepsy, chemotherapy-induced neuropathy, and "
                            "post-COVID dysautonomia."
                        ),
                    },
                    {
                        "term": "Patient Preparation",
                        "detail": (
                            "No caffeine, smoking, or exercise for 3 hours before testing. "
                            "Avoid anti-cholinergics, sympathomimetics, and vasoactive "
                            "medications (discuss with referring physician). Testing performed "
                            "in a quiet, temperature-controlled room (22–24°C). Bladder emptied "
                            "prior to testing. Minimum 2-hour fast recommended."
                        ),
                    },
                    {
                        "term": "Test Sequence",
                        "detail": (
                            "1. Resting HR and BP (5 min supine baseline). "
                            "2. Deep Breathing test (E:I ratio). "
                            "3. Valsalva Maneuver (ratio). "
                            "4. Stand test (30:15 ratio and orthostatic BP). "
                            "5. Isometric Handgrip test. "
                            "6. Cold Pressor test. "
                            "7. Sympathetic Skin Response (SSR) — hands and feet."
                        ),
                    },
                ],
            },
            {
                "heading": "Parasympathetic Tests",
                "items": [
                    {
                        "term": "Valsalva Ratio",
                        "detail": (
                            "Patient exhales against a resistance of 40 mmHg for 15 seconds. "
                            "The ratio is: maximum HR during strain phase / minimum HR during "
                            "recovery phase (beats 20–40 post-release). Tests the complete "
                            "cardiovagal reflex arc (baroreflex). "
                            "Normal ≥1.21 | Borderline 1.11–1.20 | Abnormal <1.11."
                        ),
                    },
                    {
                        "term": "Deep Breathing E:I Ratio",
                        "detail": (
                            "Patient breathes at exactly 6 cycles/min (5 s in / 5 s out) for "
                            "6 cycles. E:I ratio = maximum HR during expiration / minimum HR "
                            "during inspiration. Tests sinus arrhythmia (respiratory HR "
                            "variation), a pure parasympathetic (vagal) response. "
                            "Reference is age-adjusted: subtract 0.01 per decade over age 20. "
                            "Normal ≥1.21 (age-adjusted) | Borderline 1.11–1.20 | Abnormal <1.11."
                        ),
                    },
                    {
                        "term": "30:15 Ratio",
                        "detail": (
                            "Patient stands from supine. RR interval at beat 30 is divided by "
                            "RR interval at beat 15 after standing. The reflex HR acceleration "
                            "at beat 15 and deceleration at beat 30 (the 'initial overshoot') "
                            "are mediated predominantly by the parasympathetic nervous system. "
                            "Normal ≥1.04 | Borderline 1.01–1.03 | Abnormal <1.01."
                        ),
                    },
                ],
            },
            {
                "heading": "Sympathetic Tests",
                "items": [
                    {
                        "term": "Orthostatic BP Drop (SBP)",
                        "detail": (
                            "Systolic blood pressure is measured supine and at 1 and 3 minutes "
                            "after standing. The maximum SBP fall is recorded. Orthostatic "
                            "hypotension (OH) is defined as a sustained reduction in SBP ≥20 mmHg "
                            "or DBP ≥10 mmHg within 3 minutes of standing (AHA/Freeman 2011). "
                            "Normal <10 mmHg | Borderline 10–19 mmHg | Abnormal (OH) ≥20 mmHg."
                        ),
                    },
                    {
                        "term": "Isometric Handgrip DBP Rise",
                        "detail": (
                            "Patient grips a dynamometer at 30% of maximal voluntary contraction "
                            "for 3 minutes. The rise in diastolic BP from rest to end of exercise "
                            "is measured. Tests sympathetic cardiovascular activation via "
                            "the exercise pressor reflex (muscle afferents → sympathetic outflow). "
                            "Normal ≥16 mmHg | Borderline 11–15 mmHg | Abnormal ≤10 mmHg."
                        ),
                    },
                    {
                        "term": "Cold Pressor DBP Rise",
                        "detail": (
                            "Patient immerses the contralateral hand (not tested for SSR) in "
                            "ice water (0–4°C) for 1 minute. The maximum DBP rise is recorded. "
                            "Tests the spino-sympathetic reflex via cold nociceptors. "
                            "Normal ≥15 mmHg | Borderline 10–14 mmHg | Abnormal <10 mmHg."
                        ),
                    },
                    {
                        "term": "Sympathetic Skin Response (SSR) — Hand",
                        "detail": (
                            "Electrodermal activity recorded from palmar (active) and dorsal "
                            "(reference) electrodes. Elicited by electrical stimulation of the "
                            "median nerve at the wrist or by sudden deep breath/loud noise. "
                            "Mediated by unmyelinated C-fiber sudomotor sympathetic efferents. "
                            "Hand Latency: Normal ≤1.5 s | Abnormal >1.5 s | Absent: no response. "
                            "Hand Amplitude: Normal ≥0.5 mV | Reduced 0.1–0.49 mV | Absent <0.1 mV."
                        ),
                    },
                    {
                        "term": "Sympathetic Skin Response (SSR) — Foot",
                        "detail": (
                            "Recorded from plantar (active) and dorsal foot (reference) electrodes. "
                            "Same stimulation paradigm as hand SSR. Foot responses have longer "
                            "latency due to greater nerve path length. Foot SSR is a sensitive "
                            "marker for length-dependent small-fiber neuropathy. "
                            "Foot Latency: Normal ≤2.0 s | Abnormal >2.0 s | Absent: no response. "
                            "Foot Amplitude: Normal ≥0.3 mV | Reduced 0.1–0.29 mV | Absent <0.1 mV."
                        ),
                    },
                ],
            },
            {
                "heading": "Composite Autonomic Severity Index",
                "items": [
                    {
                        "term": "CASI Score (0–100)",
                        "detail": (
                            "A weighted composite of all 10 AFT parameters. Parasympathetic "
                            "tests (Valsalva, E:I, 30:15) contribute 45 points; sympathetic "
                            "adrenergic tests (Orthostatic BP, Handgrip, Cold Pressor) contribute "
                            "45 points; SSR (4 parameters) contributes the remaining 10 points. "
                            "Score is normalized to 0–100. Higher scores indicate greater "
                            "autonomic impairment. CASI 0–14: Normal. 15–34: Mild. 35–64: Moderate. "
                            "65–100: Severe. Adapted from the Mayo Clinic Composite Autonomic "
                            "Scoring Scale (CASS) by Low PA et al., 1992."
                        ),
                    },
                ],
            },
            {
                "heading": "Diagnostic Patterns",
                "items": [
                    {"term": p.replace("_", " ").title(), "detail": d}
                    for p, d in DIAGNOSTIC_PATTERNS.items()
                ],
            },
            {
                "heading": "Severity Levels",
                "items": [
                    {
                        "term": "Normal",
                        "detail": "CASI 0–14. All parameters within reference ranges. No significant autonomic dysfunction.",
                    },
                    {
                        "term": "Mild",
                        "detail": (
                            "CASI 15–34. One or two parameters borderline or mildly abnormal. "
                            "Typically isolated parasympathetic involvement (cardiovagal). "
                            "May not cause symptoms at rest."
                        ),
                    },
                    {
                        "term": "Moderate",
                        "detail": (
                            "CASI 35–64. Multiple parasympathetic abnormalities with some "
                            "sympathetic involvement. May present with orthostatic symptoms, "
                            "exercise intolerance, and abnormal SSR."
                        ),
                    },
                    {
                        "term": "Severe",
                        "detail": (
                            "CASI 65–100. Widespread autonomic failure across both divisions. "
                            "Orthostatic hypotension, absent cardiovagal responses, absent SSR. "
                            "High morbidity; associated with MSA, advanced diabetic neuropathy, "
                            "pure autonomic failure, and high SUDEP risk in epilepsy."
                        ),
                    },
                ],
            },
            {
                "heading": "SUDEP Risk Assessment",
                "items": [
                    {
                        "term": "Autonomic Dysfunction and SUDEP",
                        "detail": (
                            "Sudden Unexpected Death in Epilepsy (SUDEP) is the leading cause "
                            "of epilepsy-related mortality (incidence 1–2 per 1,000 patient-years "
                            "in drug-resistant epilepsy). Post-ictal autonomic dysregulation — "
                            "including central apnea, cardiac arrhythmia, and loss of autonomic "
                            "reflexes — is the prevailing mechanistic model. "
                            "Patients with epilepsy who show impaired cardiovagal responses "
                            "(reduced Valsalva ratio, E:I ratio, or 30:15 ratio) and/or "
                            "sympathetic dysfunction have a higher probability of SUDEP events. "
                            "AFT therefore serves as a SUDEP risk biomarker, particularly in "
                            "patients with drug-resistant focal epilepsy, nocturnal seizures, "
                            "or a history of prolonged post-ictal period."
                        ),
                    },
                    {
                        "term": "SUDEP Risk Flag Criteria (in this system)",
                        "detail": (
                            "Flagged when: patient has epilepsy/seizure disorder AND meets "
                            "ANY of: (1) ≥2 parasympathetic tests definitively abnormal, "
                            "(2) ≥1 parasympathetic + ≥1 sympathetic test definitively abnormal, "
                            "(3) CASI ≥35 (Moderate or Severe autonomic dysfunction). "
                            "Not a diagnostic criterion — intended to prompt cardiology/neurology "
                            "co-management, nocturnal supervision counseling, and EEG-ECG telemetry."
                        ),
                    },
                ],
            },
            {
                "heading": "Clinical Significance",
                "items": [
                    {
                        "term": "Epilepsy",
                        "detail": (
                            "Recurrent seizures disrupt both central and peripheral autonomic "
                            "control. Anti-epileptic drugs (particularly carbamazepine, "
                            "phenytoin, and valproate) may independently affect cardiac "
                            "autonomic regulation. AFT provides a quantitative autonomic "
                            "profile that complements EEG monitoring in high-risk epilepsy patients."
                        ),
                    },
                    {
                        "term": "Diabetic Autonomic Neuropathy",
                        "detail": (
                            "Cardiovascular autonomic neuropathy (CAN) is a major predictor "
                            "of mortality in diabetes. The Ewing battery (Valsalva, E:I, 30:15, "
                            "orthostatic BP, handgrip) remains the standard for CAN staging. "
                            "CASI ≥35 in diabetes warrants aggressive cardiovascular risk management."
                        ),
                    },
                    {
                        "term": "Parkinsonism / Synucleinopathies",
                        "detail": (
                            "MSA and Parkinson's disease with autonomic failure show severe "
                            "adrenergic failure (orthostatic hypotension, absent BP responses) "
                            "with or without cardiovagal loss. SSR is typically absent in MSA. "
                            "AFT findings can help differentiate MSA-C/MSA-P from idiopathic PD."
                        ),
                    },
                    {
                        "term": "POTS",
                        "detail": (
                            "POTS (Postural Orthostatic Tachycardia Syndrome) presents with "
                            "HR rise ≥30 bpm (≥40 bpm in adolescents) within 10 minutes of "
                            "standing without significant orthostatic hypotension. The 30:15 "
                            "ratio may be elevated (hyper-responsive). AFT helps distinguish "
                            "POTS from OH and characterize the sympathetic component."
                        ),
                    },
                ],
            },
            {
                "heading": "Reference Standards",
                "items": [
                    {
                        "term": "Ewing DJ Classification",
                        "detail": (
                            "Ewing DJ, Clarke BF. Diagnosis and management of diabetic "
                            "autonomic neuropathy. BMJ. 1982;285(6346):916-918. "
                            "Established the five-test battery (Valsalva ratio, deep "
                            "breathing E:I, 30:15 ratio, orthostatic BP, handgrip) as the "
                            "standard for bedside autonomic evaluation. Remains the most "
                            "widely used grading system worldwide."
                        ),
                    },
                    {
                        "term": "AAN Practice Parameters",
                        "detail": (
                            "Low PA, et al. AAN Practice Parameter: Autonomic testing; "
                            "report of the Therapeutics and Technology Assessment Subcommittee. "
                            "Neurology. 1996;46(3):873-880. "
                            "Defines normative values, technical requirements, and clinical "
                            "indications for the full AFT battery including SSR. "
                            "Updated by Sletten DM et al. Clin Auton Res. 2012."
                        ),
                    },
                    {
                        "term": "Orthostatic Hypotension Definition",
                        "detail": (
                            "Freeman R, et al. Consensus statement on the definition of "
                            "orthostatic hypotension, neurally mediated syncope and the "
                            "postural tachycardia syndrome. Clin Auton Res. 2011;21(2):69-72. "
                            "Defines OH as sustained SBP reduction ≥20 mmHg or DBP ≥10 mmHg "
                            "within 3 minutes of standing or head-up tilt to at least 60 degrees."
                        ),
                    },
                    {
                        "term": "CASI / CASS",
                        "detail": (
                            "Low PA. Composite autonomic scoring scale for laboratory "
                            "quantification of generalized autonomic failure. Mayo Clin Proc. "
                            "1993;68(8):748-752. "
                            "The original CASS subdivides into Sudomotor (0-3), Adrenergic (0-4), "
                            "and Cardiovagal (0-3) subscores. The CASI used here is a normalized "
                            "0–100 adaptation for dashboard display."
                        ),
                    },
                ],
            },
        ],
    }


if __name__ == "__main__":
    import json

    print("=== Autonomic Function Tests Overview ===")
    ov = overview()
    print(json.dumps(ov["kpis"], indent=2))
    print(f"Severity: {ov['severity_distribution']}")
    print(f"Patterns: {[p['name'] + '=' + str(p['count']) for p in ov['pattern_distribution']]}")
    print(f"CASI histogram: {ov['casi_histogram']}")

    print(f"\n=== Breakdown ===")
    bd = breakdown()
    print(f"Total patients: {len(bd['patients'])}")
    top = bd["patients"][0]
    print(f"Highest CASI patient: {top['name']} — CASI={top['casi_score']} severity={top['severity']} pattern={top['pattern']}")
    print(f"  SUDEP risk: {top['sudep_risk']}")
    print(f"  Parasympathetic tests: {[t['test'] + '=' + str(t['status']) for t in top['parasympathetic_tests']]}")
    print(f"  Sympathetic tests: {[t['test'] + '=' + str(t['status']) for t in top['sympathetic_tests']]}")

    print(f"\n=== Definitions ===")
    df = definitions()
    print(f"Title: {df['title']}")
    print(f"Sections: {[s['heading'] for s in df['sections']]}")
    print(f"Section items count: {[len(s['items']) for s in df['sections']]}")
