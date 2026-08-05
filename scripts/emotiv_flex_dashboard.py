#!/usr/bin/env python3
"""Emotiv EPOC Flex Dashboard — data module.

Provides overview(), channels(), and definitions() for the
/api/emotiv-flex/* endpoints.  Models the Emotiv EPOC Flex 32-channel
research-grade EEG cap (online + offline modes).  Realistic synthetic data.
"""
from __future__ import annotations
import random, math
from datetime import datetime, timedelta

RNG = random.Random(7)

# Standard 32-channel 10-20 positions for EPOC Flex
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10",
]

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_RANGES = {
    "delta": "0.5–4 Hz",
    "theta": "4–8 Hz",
    "alpha": "8–13 Hz",
    "beta":  "13–30 Hz",
    "gamma": "30–100 Hz",
}
BAND_COLORS = {
    "delta": "#6366f1",
    "theta": "#3b82f6",
    "alpha": "#10b981",
    "beta":  "#f59e0b",
    "gamma": "#ef4444",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")


def _impedance() -> float:
    """Return electrode impedance in kΩ — most channels good (<10 kΩ)."""
    if RNG.random() < 0.72:
        return round(RNG.uniform(1.0, 9.9), 1)   # good
    elif RNG.random() < 0.5:
        return round(RNG.uniform(10.0, 24.9), 1)  # marginal
    else:
        return round(RNG.uniform(25.0, 80.0), 1)  # poor


def _contact_quality(imp: float) -> str:
    if imp < 10:
        return "good"
    elif imp < 25:
        return "marginal"
    return "poor"


def _channel_band_power() -> dict:
    """Simulate realistic relative band power (sums to ~1)."""
    raw = {b: max(0.01, RNG.gauss(0.20, 0.06)) for b in BANDS}
    total = sum(raw.values())
    return {b: round(v / total, 3) for b, v in raw.items()}


def _alpha_peak() -> float:
    return round(RNG.uniform(8.5, 12.5), 1)


def _make_session(i: int) -> dict:
    duration_min = RNG.randint(20, 90)
    ch_good = RNG.randint(24, 32)
    return {
        "session_id": f"FLEX-{3000 + i}",
        "patient_id": f"P-{100 + i}",
        "date": _days_ago(RNG.randint(0, 30)),
        "duration_min": duration_min,
        "sampling_rate_hz": 2048,
        "channels_good": ch_good,
        "channels_total": 32,
        "signal_quality_pct": round(ch_good / 32 * 100, 1),
        "motion_artifact_pct": round(RNG.uniform(0.5, 6.0), 1),
        "seizure_events": RNG.randint(0, 3),
        "ica_components_removed": RNG.randint(1, 5),
        "alpha_peak_hz": _alpha_peak(),
        "upload_status": RNG.choice(["uploaded", "uploaded", "uploaded", "pending"]),
    }


# ── public API ─────────────────────────────────────────────────────────────────

def overview() -> dict:
    """Fleet overview: KPIs, session list, band-power distribution, impedance map."""
    sessions = [_make_session(i) for i in range(10)]

    uploaded = sum(1 for s in sessions if s["upload_status"] == "uploaded")
    avg_sq = round(sum(s["signal_quality_pct"] for s in sessions) / len(sessions), 1)
    total_seizure = sum(s["seizure_events"] for s in sessions)
    avg_alpha = round(sum(s["alpha_peak_hz"] for s in sessions) / len(sessions), 1)

    # Impedance heatmap across 32 channels (aggregate over fleet)
    impedance_map = []
    for ch in CHANNEL_NAMES:
        imp = _impedance()
        impedance_map.append({
            "channel": ch,
            "impedance_kohm": imp,
            "contact": _contact_quality(imp),
        })

    # Band power trend over 7 days
    band_trend = []
    for d in range(7):
        entry = {"day": _days_ago(6 - d)}
        total = 0
        vals = {}
        for b in BANDS:
            v = max(0.02, RNG.gauss(0.20, 0.04))
            vals[b] = v
            total += v
        for b in BANDS:
            entry[b] = round(vals[b] / total, 3)
        band_trend.append(entry)

    return {
        "kpis": {
            "total_sessions": len(sessions),
            "uploaded": uploaded,
            "avg_signal_quality_pct": avg_sq,
            "total_seizure_events": total_seizure,
            "avg_alpha_peak_hz": avg_alpha,
            "channels": 32,
            "sampling_rate_hz": 2048,
        },
        "sessions": sessions,
        "impedance_map": impedance_map,
        "band_trend_7d": band_trend,
    }


def channels() -> dict:
    """Per-channel analysis: impedance, contact quality, band power, alpha peak."""
    channel_data = []
    for ch in CHANNEL_NAMES:
        imp = _impedance()
        bp = _channel_band_power()
        channel_data.append({
            "channel": ch,
            "impedance_kohm": imp,
            "contact": _contact_quality(imp),
            "band_power": bp,
            "alpha_peak_hz": _alpha_peak(),
            "snr_db": round(RNG.uniform(12.0, 35.0), 1),
            "artifact_pct": round(RNG.uniform(0.0, 8.0), 1),
        })

    good = sum(1 for c in channel_data if c["contact"] == "good")
    marginal = sum(1 for c in channel_data if c["contact"] == "marginal")
    poor = sum(1 for c in channel_data if c["contact"] == "poor")

    # Frequency spectrum (0–50 Hz) — averaged across all channels
    spectrum = [
        {
            "freq_hz": round(0.5 + i * 0.5, 1),
            "power_uv2": round(max(0.1, 50 / (1 + (0.5 + i * 0.5) / 3) + RNG.gauss(0, 1)), 2),
        }
        for i in range(100)
    ]

    return {
        "channels": channel_data,
        "summary": {
            "good": good,
            "marginal": marginal,
            "poor": poor,
            "good_pct": round(good / 32 * 100, 1),
        },
        "spectrum": spectrum,
    }


def definitions() -> dict:
    """Device specs and EEG metric glossary for the Emotiv EPOC Flex."""
    return {
        "device": {
            "name": "Emotiv EPOC Flex",
            "type": "Research-grade 32-channel EEG cap",
            "channels": 32,
            "layout": "10-20 standard (Fp1–PO10)",
            "sampling_rate_hz": 2048,
            "resolution_bits": 24,
            "impedance_check": "Built-in per-channel impedance measurement",
            "connectivity": "USB / Bluetooth 5.0",
            "online_mode": "BLE → laptop/gateway → backend real-time stream",
            "offline_mode": "On-device SD buffer → sync on reconnect",
            "battery_life_h": 12,
            "reference_electrodes": "CMS/DRL (common mode sense / driven right leg)",
            "sdk": "EmotivPRO / LSL / BIDS-compatible export",
            "certifications": "CE, FCC",
        },
        "metrics": [
            {
                "term": "Impedance (kΩ)",
                "normal_range": "<10 kΩ",
                "clinical_note": "Electrode-skin contact quality. <10 kΩ = good; 10–25 kΩ = marginal; >25 kΩ = poor. High impedance increases noise floor and reduces SNR.",
            },
            {
                "term": "SNR (dB)",
                "normal_range": ">20 dB",
                "clinical_note": "Signal-to-noise ratio. Higher SNR means cleaner recording. Seizure detection algorithms require >15 dB minimum.",
            },
            {
                "term": "Delta power (0.5–4 Hz)",
                "normal_range": "20–35% (awake)",
                "clinical_note": "Elevated delta indicates deep sleep or diffuse encephalopathy. High ictal delta = post-ictal suppression.",
            },
            {
                "term": "Theta power (4–8 Hz)",
                "normal_range": "15–25%",
                "clinical_note": "Theta slowing in temporal lobes suggests TLE (temporal lobe epilepsy). Drowsiness increases theta.",
            },
            {
                "term": "Alpha power (8–13 Hz)",
                "normal_range": "25–40% (eyes closed)",
                "clinical_note": "Dominant rhythm when relaxed. Attenuates with eye opening (alpha blocking). Loss of alpha = cerebral injury marker.",
            },
            {
                "term": "Alpha peak frequency (Hz)",
                "normal_range": "9–12 Hz",
                "clinical_note": "Individual alpha frequency (IAF). Slowing <9 Hz associated with cognitive impairment and AED side-effects.",
            },
            {
                "term": "Beta power (13–30 Hz)",
                "normal_range": "10–20%",
                "clinical_note": "Elevated beta = benzodiazepine/barbiturate effect or frontally dominant seizure pattern. Low-voltage fast activity (LVFA) = ictal onset.",
            },
            {
                "term": "Gamma power (30–100 Hz)",
                "normal_range": "<10%",
                "clinical_note": "High-frequency oscillations (HFOs). Pathological HFOs (80–500 Hz) are biomarkers for epileptogenic zone delineation.",
            },
            {
                "term": "ICA components removed",
                "normal_range": "1–4 per session",
                "clinical_note": "Independent Component Analysis removes eye blinks (Fp1/Fp2), EMG artifacts, and cardiac contamination before analysis.",
            },
            {
                "term": "Motion artifact %",
                "normal_range": "<5%",
                "clinical_note": "Proportion of recording corrupted by movement. Critical for ambulatory EEG — ictal motor activity raises this above 10%.",
            },
        ],
        "epilepsy_context": (
            "The Emotiv EPOC Flex 32-ch cap is deployed for long-duration ambulatory EEG monitoring "
            "in epilepsy research. Its 10-20 layout covers frontal, temporal, parietal, and occipital "
            "regions, enabling ictal onset localisation, interictal discharge mapping, and HFO analysis. "
            "Combined with the EmotivPRO SDK and real-time LSL streaming, it supports both in-clinic "
            "video-EEG and home monitoring (offline buffer mode). 32 channels improve seizure focus "
            "localisation accuracy over 5- or 14-channel devices."
        ),
        "references": [
            "Emotiv EPOC Flex Technical Specification v2.1, Emotiv Inc. 2024.",
            "Niso G et al. HERMES: Towards an Integrated Toolbox for EEG/MEG. Neuroinformatics 2016.",
            "Delorme A & Makeig S. EEGLAB: An open source toolbox for EEG analysis. J Neurosci Methods 2004.",
            "Zijlmans M et al. High-frequency oscillations as biomarkers of epileptogenicity. Brain 2012.",
            "IFCN Standards for Digital Recording of EEGs — Clin Neurophysiol 2017.",
        ],
    }
