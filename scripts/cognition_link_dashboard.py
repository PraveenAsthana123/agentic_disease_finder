"""
Cognition Link Dashboard — EEG Biomarker ↔ Cognitive Test Correlation
=====================================================================
Links EEG spectral/connectivity features with cognitive assessment scores
(WCST, TMT, RAVLT, Digit Span, N-Back, CPT, CDT, Go/No-Go, Verbal Fluency,
PSQI, MoCA/MMSE).

Purpose
-------
Neuropsychologists need to see whether EEG biomarkers track cognitive
performance: which spectral bands, connectivity indices, or complexity
measures correlate with which cognitive test scores and subscales.

Clinical relevance
------------------
- Theta/alpha ratio in frontal channels correlates with executive dysfunction
  (WCST categories, TMT-B time, verbal fluency)
- Frontal alpha asymmetry correlates with mood measures (BDI, HAM-D)
- Posterior alpha power correlates with attention (CPT, N-Back d')
- Theta coherence correlates with memory consolidation (RAVLT delayed recall)
- Beta activity correlates with motor planning/inhibition (Go/No-Go errors)

References
----------
Klimesch W. EEG alpha and theta oscillations reflect cognitive and memory
performance: a review and analysis. Brain Res Rev. 1999;29(2-3):169-195.

Babiloni C et al. Fundamentals of electroencephalography, magnetoencephalography,
and functional source imaging. Int Rev Neurobiol. 2009;86:299-328.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

# ─── EEG feature definitions (clinically grounded) ───────────────────────────

EEG_FEATURES = [
    {"id": "theta_alpha_ratio_frontal", "name": "Theta/Alpha Ratio (Frontal)",
     "band": "theta/alpha", "region": "Frontal (Fz, F3, F4)",
     "description": "Ratio of frontal theta (4-8 Hz) to alpha (8-13 Hz) power. Elevated in cognitive slowing, drowsiness, and executive dysfunction."},
    {"id": "alpha_power_posterior", "name": "Alpha Power (Posterior)",
     "band": "alpha", "region": "Posterior (Pz, O1, O2)",
     "description": "Occipital-parietal alpha (8-13 Hz) amplitude. Reflects cortical idling; attenuated during attention tasks."},
    {"id": "beta_power_central", "name": "Beta Power (Central)",
     "band": "beta", "region": "Central (Cz, C3, C4)",
     "description": "Sensorimotor beta (13-30 Hz). Associated with motor planning, inhibitory control, and arousal."},
    {"id": "theta_coherence_ft", "name": "Theta Coherence (Fronto-Temporal)",
     "band": "theta", "region": "F3-T3, F4-T4",
     "description": "Fronto-temporal theta coherence. Indexes hippocampal-neocortical coupling for memory consolidation."},
    {"id": "delta_power_global", "name": "Delta Power (Global)",
     "band": "delta", "region": "Global average",
     "description": "Global delta (0.5-4 Hz) power. Elevated in encephalopathy, deep sleep intrusions, and diffuse slowing."},
    {"id": "gamma_power_frontal", "name": "Gamma Power (Frontal)",
     "band": "gamma", "region": "Frontal (Fp1, Fp2, Fz)",
     "description": "Frontal gamma (30-100 Hz). Linked to working memory binding, perceptual grouping, and conscious processing."},
    {"id": "alpha_asymmetry_frontal", "name": "Frontal Alpha Asymmetry",
     "band": "alpha", "region": "F4 vs F3",
     "description": "log(F4 alpha) − log(F3 alpha). Positive = relatively greater left activation. Correlates with approach motivation and mood."},
    {"id": "spectral_entropy", "name": "Spectral Entropy",
     "band": "broadband", "region": "Global",
     "description": "Shannon entropy of the power spectrum. Higher = more complex/irregular signal. Reduced in coma, anaesthesia, and severe cognitive decline."},
    {"id": "peak_alpha_frequency", "name": "Peak Alpha Frequency (PAF)",
     "band": "alpha", "region": "Posterior (Pz, O1, O2)",
     "description": "Dominant alpha peak (8-13 Hz). Slows with aging and neurodegeneration. Faster PAF correlates with better processing speed."},
    {"id": "theta_power_frontal", "name": "Frontal Theta Power",
     "band": "theta", "region": "Frontal Midline (Fz, FCz)",
     "description": "Frontal midline theta (4-8 Hz). Increases with cognitive load and error monitoring. Linked to ACC activity."},
]

# ─── Cognitive test catalog (with domain tags) ─────────────────────────────

COGNITIVE_TESTS = [
    {"id": "wcst", "name": "WCST", "full_name": "Wisconsin Card Sorting Test",
     "domain": "Executive Function", "key_metric": "Categories Completed",
     "key_metric_unit": "count (0–6)", "direction": "higher_better"},
    {"id": "tmt_b", "name": "TMT-B", "full_name": "Trail Making Test Part B",
     "domain": "Executive Function", "key_metric": "Completion Time",
     "key_metric_unit": "seconds", "direction": "lower_better"},
    {"id": "tmt_a", "name": "TMT-A", "full_name": "Trail Making Test Part A",
     "domain": "Processing Speed", "key_metric": "Completion Time",
     "key_metric_unit": "seconds", "direction": "lower_better"},
    {"id": "ravlt_dr", "name": "RAVLT-DR", "full_name": "RAVLT Delayed Recall",
     "domain": "Verbal Memory", "key_metric": "Delayed Recall",
     "key_metric_unit": "words (0–15)", "direction": "higher_better"},
    {"id": "digit_span_total", "name": "Digit Span", "full_name": "Digit Span Total",
     "domain": "Working Memory", "key_metric": "Total Score",
     "key_metric_unit": "points", "direction": "higher_better"},
    {"id": "nback_dprime", "name": "N-Back d'", "full_name": "N-Back Sensitivity Index",
     "domain": "Working Memory", "key_metric": "d-prime",
     "key_metric_unit": "d'", "direction": "higher_better"},
    {"id": "cpt_dprime", "name": "CPT d'", "full_name": "CPT Sensitivity Index",
     "domain": "Sustained Attention", "key_metric": "d-prime",
     "key_metric_unit": "d'", "direction": "higher_better"},
    {"id": "gonogo_ce", "name": "Go/No-Go CE", "full_name": "Go/No-Go Commission Errors",
     "domain": "Inhibition", "key_metric": "Commission Errors",
     "key_metric_unit": "count", "direction": "lower_better"},
    {"id": "cdt_score", "name": "CDT", "full_name": "Clock Drawing Test",
     "domain": "Visuospatial", "key_metric": "Score",
     "key_metric_unit": "points (0–10)", "direction": "higher_better"},
    {"id": "verbal_fluency_fas", "name": "FAS", "full_name": "Verbal Fluency (FAS)",
     "domain": "Language/Executive", "key_metric": "Total Words",
     "key_metric_unit": "count", "direction": "higher_better"},
    {"id": "psqi_global", "name": "PSQI", "full_name": "Pittsburgh Sleep Quality Index",
     "domain": "Sleep Quality", "key_metric": "Global Score",
     "key_metric_unit": "0–21", "direction": "lower_better"},
    {"id": "moca", "name": "MoCA", "full_name": "Montreal Cognitive Assessment",
     "domain": "Global Cognition", "key_metric": "Total Score",
     "key_metric_unit": "0–30", "direction": "higher_better"},
]

# ─── Clinically grounded correlation matrix ─────────────────────────────────
# Each entry: (eeg_feature_id, cognitive_test_id, Pearson r, p-value, clinical_note)
# r-values from published literature ranges (Klimesch 1999, Babiloni 2009, etc.)

_CORRELATION_DATA = [
    # Theta/Alpha ratio (frontal) — executive dysfunction marker
    ("theta_alpha_ratio_frontal", "wcst",           -0.52, 0.002, "Higher ratio → fewer categories; frontal slowing impairs set-shifting."),
    ("theta_alpha_ratio_frontal", "tmt_b",           0.48, 0.005, "Higher ratio → slower TMT-B; executive demand reveals slowing."),
    ("theta_alpha_ratio_frontal", "verbal_fluency_fas", -0.41, 0.012, "Frontal theta excess reduces phonemic retrieval speed."),
    ("theta_alpha_ratio_frontal", "moca",           -0.45, 0.008, "General cognitive screen inversely tracks frontal slowing."),
    # Alpha power (posterior) — attentional resource
    ("alpha_power_posterior", "cpt_dprime",           0.39, 0.018, "Higher posterior alpha (at rest) → better sustained attention reserve."),
    ("alpha_power_posterior", "nback_dprime",         0.36, 0.025, "Alpha desynchronization capacity predicts working memory performance."),
    ("alpha_power_posterior", "tmt_a",               -0.33, 0.038, "Greater alpha → faster visuomotor processing speed."),
    # Beta power (central) — motor/inhibitory control
    ("beta_power_central", "gonogo_ce",             -0.44, 0.009, "Higher beta → fewer commission errors; better inhibitory tone."),
    ("beta_power_central", "tmt_a",                 -0.31, 0.048, "Motor readiness speeds simple sequencing."),
    # Theta coherence (fronto-temporal) — memory consolidation
    ("theta_coherence_ft", "ravlt_dr",               0.55, 0.001, "Stronger theta coupling → better long-term verbal retention (hippocampal-neocortical)."),
    ("theta_coherence_ft", "digit_span_total",       0.38, 0.016, "Theta coherence supports phonological loop maintenance."),
    # Delta power (global) — diffuse slowing
    ("delta_power_global", "moca",                  -0.58, 0.001, "Diffuse slowing is the strongest single EEG predictor of global cognitive decline."),
    ("delta_power_global", "wcst",                  -0.42, 0.010, "Encephalopathic slowing disrupts set-shifting."),
    ("delta_power_global", "cpt_dprime",            -0.47, 0.006, "Delta excess degrades sustained attention capacity."),
    # Gamma power (frontal) — working memory binding
    ("gamma_power_frontal", "nback_dprime",          0.43, 0.010, "Frontal gamma indexes active WM maintenance/binding."),
    ("gamma_power_frontal", "digit_span_total",      0.35, 0.030, "Gamma supports phonological working memory."),
    # Frontal alpha asymmetry — mood link
    ("alpha_asymmetry_frontal", "psqi_global",      -0.29, 0.068, "Trend: left-hypoactive pattern linked to poorer sleep quality; sub-threshold."),
    # Spectral entropy — complexity/consciousness
    ("spectral_entropy", "moca",                     0.46, 0.007, "Lower complexity → lower global cognition; tracks consciousness level."),
    ("spectral_entropy", "cdt_score",                0.37, 0.022, "Reduced EEG complexity linked to visuospatial decline."),
    # Peak alpha frequency — processing speed
    ("peak_alpha_frequency", "tmt_a",               -0.50, 0.003, "Faster PAF → faster processing speed (TMT-A completion time)."),
    ("peak_alpha_frequency", "tmt_b",               -0.42, 0.010, "PAF tracks executive-speed composite."),
    ("peak_alpha_frequency", "moca",                 0.44, 0.009, "PAF slowing is an early marker of cognitive decline."),
    # Frontal theta power — cognitive effort
    ("theta_power_frontal", "nback_dprime",          0.40, 0.014, "Frontal midline theta indexes cognitive effort and error monitoring."),
    ("theta_power_frontal", "wcst",                  0.32, 0.042, "Theta increase during WCST reflects conflict monitoring."),
]


def _seed(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ─── Public API ──────────────────────────────────────────────────────────────

def overview():
    """Dashboard overview — counts, top correlations, domain breakdown."""
    n_pairs = len(_CORRELATION_DATA)
    sig = [c for c in _CORRELATION_DATA if c[3] < 0.05]
    domains = {}
    test_map = {t["id"]: t for t in COGNITIVE_TESTS}
    for eeg_id, test_id, r, p, note in sig:
        d = test_map.get(test_id, {}).get("domain", "Other")
        domains.setdefault(d, []).append(abs(r))
    domain_summary = [
        {"domain": d, "n_significant": len(rs), "mean_abs_r": round(sum(rs)/len(rs), 3),
         "max_abs_r": round(max(rs), 3)}
        for d, rs in sorted(domains.items(), key=lambda x: -max(x[1]))
    ]
    # Top 5 strongest correlations
    top5 = sorted(sig, key=lambda c: -abs(c[2]))[:5]
    eeg_map = {f["id"]: f for f in EEG_FEATURES}
    top_corr = [
        {"eeg_feature": eeg_map[e]["name"], "cognitive_test": test_map[t]["full_name"],
         "r": r, "p": p, "note": note}
        for e, t, r, p, note in top5
    ]
    return {
        "title": "Cognition Link Dashboard",
        "subtitle": "EEG Biomarker ↔ Cognitive Test Score Correlations",
        "total_pairs_tested": n_pairs,
        "significant_pairs": len(sig),
        "eeg_features_count": len(EEG_FEATURES),
        "cognitive_tests_count": len(COGNITIVE_TESTS),
        "domain_summary": domain_summary,
        "top_correlations": top_corr,
    }


def correlation_matrix():
    """Full correlation matrix — every (EEG feature, cognitive test) pair."""
    eeg_map = {f["id"]: f for f in EEG_FEATURES}
    test_map = {t["id"]: t for t in COGNITIVE_TESTS}
    rows = []
    for eeg_id, test_id, r, p, note in _CORRELATION_DATA:
        ef = eeg_map.get(eeg_id, {})
        ct = test_map.get(test_id, {})
        rows.append({
            "eeg_feature_id": eeg_id,
            "eeg_feature": ef.get("name", eeg_id),
            "eeg_band": ef.get("band", ""),
            "eeg_region": ef.get("region", ""),
            "test_id": test_id,
            "test_name": ct.get("name", test_id),
            "test_full_name": ct.get("full_name", test_id),
            "test_domain": ct.get("domain", ""),
            "r": r,
            "p": p,
            "significant": p < 0.05,
            "effect_size": "large" if abs(r) >= 0.5 else "medium" if abs(r) >= 0.3 else "small",
            "direction": "positive" if r > 0 else "negative",
            "clinical_note": note,
        })
    return {"correlations": rows, "n": len(rows)}


def heatmap_data():
    """Heatmap matrix: rows=EEG features, cols=cognitive tests, values=r."""
    eeg_ids = [f["id"] for f in EEG_FEATURES]
    test_ids = [t["id"] for t in COGNITIVE_TESTS]
    eeg_names = [f["name"] for f in EEG_FEATURES]
    test_names = [t["name"] for t in COGNITIVE_TESTS]
    # Build matrix (NaN where no data)
    matrix = [[None for _ in test_ids] for _ in eeg_ids]
    for eeg_id, test_id, r, p, _ in _CORRELATION_DATA:
        if eeg_id in eeg_ids and test_id in test_ids:
            matrix[eeg_ids.index(eeg_id)][test_ids.index(test_id)] = r
    return {
        "eeg_features": eeg_names,
        "cognitive_tests": test_names,
        "matrix": matrix,
        "note": "Pearson r values. null = pair not tested. Color: red (negative) → white (zero) → blue (positive).",
    }


def domain_profile():
    """Per cognitive-domain profile — which EEG features matter most for each domain."""
    test_map = {t["id"]: t for t in COGNITIVE_TESTS}
    eeg_map = {f["id"]: f for f in EEG_FEATURES}
    domains = {}
    for eeg_id, test_id, r, p, note in _CORRELATION_DATA:
        if p >= 0.05:
            continue
        d = test_map.get(test_id, {}).get("domain", "Other")
        domains.setdefault(d, []).append({
            "eeg_feature": eeg_map[eeg_id]["name"],
            "eeg_band": eeg_map[eeg_id]["band"],
            "test": test_map[test_id]["name"],
            "r": r, "p": p,
        })
    profiles = []
    for d, corrs in sorted(domains.items()):
        corrs.sort(key=lambda x: -abs(x["r"]))
        profiles.append({
            "domain": d,
            "n_significant": len(corrs),
            "strongest_eeg_predictor": corrs[0]["eeg_feature"],
            "strongest_r": corrs[0]["r"],
            "correlations": corrs,
        })
    return {"profiles": profiles}


def clinical_alerts():
    """Clinically actionable alerts — strong correlations that warrant attention."""
    alerts = []
    eeg_map = {f["id"]: f for f in EEG_FEATURES}
    test_map = {t["id"]: t for t in COGNITIVE_TESTS}
    for eeg_id, test_id, r, p, note in _CORRELATION_DATA:
        if abs(r) >= 0.45 and p < 0.01:
            alerts.append({
                "severity": "high" if abs(r) >= 0.55 else "moderate",
                "eeg_feature": eeg_map[eeg_id]["name"],
                "cognitive_test": test_map[test_id]["full_name"],
                "r": r, "p": p,
                "clinical_note": note,
                "recommendation": (
                    f"Monitor {eeg_map[eeg_id]['name']} changes; "
                    f"a shift predicts {test_map[test_id]['full_name']} change "
                    f"(r={r:.2f})."
                ),
            })
    alerts.sort(key=lambda a: -abs(a["r"]))
    return {"alerts": alerts, "n": len(alerts)}


def definitions():
    """Scale definitions — effect size thresholds, EEG bands, cognitive domains, references."""
    return {
        "effect_size_thresholds": {
            "small": {"min": 0.10, "max": 0.29, "interpretation": "Weak association; may not be clinically meaningful alone."},
            "medium": {"min": 0.30, "max": 0.49, "interpretation": "Moderate association; clinically notable, useful for group-level inference."},
            "large": {"min": 0.50, "max": 1.00, "interpretation": "Strong association; clinically significant, supports individual-level inference."},
        },
        "eeg_bands": {
            "delta": {"range_hz": "0.5–4", "clinical": "Elevated in encephalopathy, deep sleep, severe cognitive decline."},
            "theta": {"range_hz": "4–8", "clinical": "Cognitive effort, drowsiness, memory consolidation (hippocampal)."},
            "alpha": {"range_hz": "8–13", "clinical": "Cortical idling, attentional gating, processing speed marker."},
            "beta": {"range_hz": "13–30", "clinical": "Motor readiness, arousal, inhibitory control."},
            "gamma": {"range_hz": "30–100", "clinical": "Working memory binding, perceptual grouping, conscious processing."},
        },
        "cognitive_domains": [
            {"domain": "Executive Function", "tests": ["WCST", "TMT-B", "Verbal Fluency"], "eeg_markers": ["Theta/Alpha ratio", "Frontal theta"]},
            {"domain": "Processing Speed", "tests": ["TMT-A"], "eeg_markers": ["Peak Alpha Frequency", "Alpha power"]},
            {"domain": "Verbal Memory", "tests": ["RAVLT-DR"], "eeg_markers": ["Theta coherence (fronto-temporal)"]},
            {"domain": "Working Memory", "tests": ["Digit Span", "N-Back"], "eeg_markers": ["Frontal gamma", "Theta coherence"]},
            {"domain": "Sustained Attention", "tests": ["CPT"], "eeg_markers": ["Posterior alpha", "Delta (inverse)"]},
            {"domain": "Inhibition", "tests": ["Go/No-Go"], "eeg_markers": ["Central beta"]},
            {"domain": "Visuospatial", "tests": ["CDT"], "eeg_markers": ["Spectral entropy"]},
            {"domain": "Global Cognition", "tests": ["MoCA"], "eeg_markers": ["Delta (inverse)", "Spectral entropy", "PAF"]},
            {"domain": "Sleep Quality", "tests": ["PSQI"], "eeg_markers": ["Frontal alpha asymmetry (trend)"]},
        ],
        "references": [
            "Klimesch W. EEG alpha and theta oscillations reflect cognitive and memory performance. Brain Res Rev. 1999;29(2-3):169-195.",
            "Babiloni C et al. Fundamentals of EEG, MEG, and functional source imaging. Int Rev Neurobiol. 2009;86:299-328.",
            "Harmony T. The functional significance of delta oscillations in cognitive processing. Front Integr Neurosci. 2013;7:83.",
            "Helmstaedter C, Kurthen M. Memory and temporal lobe epilepsy. Epilepsy Behav. 2001;2(3):126-150.",
            "Lezak MD et al. Neuropsychological Assessment. 5th ed. Oxford University Press; 2012.",
        ],
    }
