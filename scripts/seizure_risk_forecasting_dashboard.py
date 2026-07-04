"""Seizure Risk Forecasting Dashboard — pre-ictal risk tiering with alert
thresholds and escalation actions.  Surfaces gap analysis data from
config/feature_gaps.json plus simulated per-patient risk forecasting scores."""

import json
import os
import hashlib
import math

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _seed_float(seed_str, lo=0.0, hi=1.0):
    """Deterministic pseudo-random float from a seed string."""
    h = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    return lo + (h % 10000) / 10000.0 * (hi - lo)


def _risk_tier(score):
    if score >= 0.75:
        return "critical"
    if score >= 0.50:
        return "high"
    if score >= 0.25:
        return "moderate"
    return "low"


_ESCALATION_ACTIONS = {
    "critical": ["Page on-call neurologist", "Activate caregiver SOS alert",
                 "Prepare rescue medication protocol", "Start continuous EEG monitoring"],
    "high":     ["Notify care team via dashboard", "Alert caregiver app",
                 "Increase EEG sampling rate"],
    "moderate": ["Log risk event", "Schedule follow-up review",
                 "Notify patient wearable"],
    "low":      ["Continue routine monitoring"],
}

_HORIZON_BUCKETS = ["0-15 min", "15-30 min", "30-60 min", "1-4 hr", "4-24 hr"]

# 30 simulated patients for consistent demo data
_PATIENT_IDS = [f"P{str(i).zfill(3)}" for i in range(1, 31)]


def overview():
    """Summary KPIs + charts for seizure-risk forecasting."""
    fg = _load('feature_gaps.json')
    if not fg:
        return {"available": False, "note": "feature_gaps.json missing"}

    gaps = fg.get('gaps', [])

    # Build per-patient risk scores (deterministic)
    patients = []
    for pid in _PATIENT_IDS:
        score = round(_seed_float(f"risk_{pid}", 0.05, 0.95), 3)
        tier = _risk_tier(score)
        horizon = _HORIZON_BUCKETS[int(_seed_float(f"hz_{pid}", 0, 4.99))]
        patients.append({
            "patient_id": pid,
            "risk_score": score,
            "risk_tier": tier,
            "forecast_horizon": horizon,
        })

    tier_counts = {}
    for p in patients:
        t = p["risk_tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    horizon_counts = {}
    for p in patients:
        h = p["forecast_horizon"]
        horizon_counts[h] = horizon_counts.get(h, 0) + 1

    avg_score = round(sum(p["risk_score"] for p in patients) / len(patients), 3)
    critical_pct = round(tier_counts.get("critical", 0) / len(patients) * 100, 1)
    high_pct = round(tier_counts.get("high", 0) / len(patients) * 100, 1)

    # Gap analysis summary
    forecasting_gaps = [g for g in gaps if g.get("category") == "decision_ai"
                        or "forecast" in g.get("feature", "").lower()
                        or "predict" in g.get("feature", "").lower()
                        or "pre-ictal" in g.get("feature", "").lower()]
    gaps_addressed = sum(1 for g in forecasting_gaps if g.get("in_project") == "built")
    gaps_partial = sum(1 for g in forecasting_gaps if g.get("in_project") == "partial")
    gaps_missing = sum(1 for g in forecasting_gaps if g.get("in_project") in (False, "planned", False))

    # Threshold config (static reference)
    thresholds = [
        {"tier": "critical", "min_score": 0.75, "max_response_min": 5,
         "escalation": "Page neurologist + caregiver SOS"},
        {"tier": "high", "min_score": 0.50, "max_response_min": 15,
         "escalation": "Care team notification + caregiver alert"},
        {"tier": "moderate", "min_score": 0.25, "max_response_min": 60,
         "escalation": "Log event + schedule follow-up"},
        {"tier": "low", "min_score": 0.00, "max_response_min": None,
         "escalation": "Routine monitoring"},
    ]

    kpis = {
        "total_patients_monitored": len(patients),
        "avg_risk_score": avg_score,
        "critical_pct": critical_pct,
        "high_pct": high_pct,
        "patients_critical": tier_counts.get("critical", 0),
        "patients_high": tier_counts.get("high", 0),
        "forecasting_gaps_total": len(forecasting_gaps),
        "gaps_partial": gaps_partial,
    }

    charts = {
        "risk_tier_pie": [{"name": k, "value": v} for k, v in tier_counts.items()],
        "horizon_bar": [{"name": h, "value": horizon_counts.get(h, 0)}
                        for h in _HORIZON_BUCKETS],
        "risk_distribution": _build_risk_histogram(patients),
        "escalation_actions_bar": [
            {"name": tier, "value": len(actions)}
            for tier, actions in _ESCALATION_ACTIONS.items()
        ],
    }

    return {
        "available": True,
        "kpis": kpis,
        "charts": charts,
        "thresholds": thresholds,
    }


def _build_risk_histogram(patients):
    """Bucket risk scores into 0.1-wide bins for a histogram chart."""
    bins = {f"{i/10:.1f}-{(i+1)/10:.1f}": 0 for i in range(10)}
    for p in patients:
        idx = min(int(p["risk_score"] * 10), 9)
        key = f"{idx/10:.1f}-{(idx+1)/10:.1f}"
        bins[key] += 1
    return [{"name": k, "value": v} for k, v in bins.items()]


def breakdown():
    """Detailed per-patient forecasting, escalation log, gap analysis."""
    fg = _load('feature_gaps.json')
    if not fg:
        return {"available": False, "note": "feature_gaps.json missing"}

    gaps = fg.get('gaps', [])

    patients = []
    escalation_log = []
    for pid in _PATIENT_IDS:
        score = round(_seed_float(f"risk_{pid}", 0.05, 0.95), 3)
        tier = _risk_tier(score)
        horizon = _HORIZON_BUCKETS[int(_seed_float(f"hz_{pid}", 0, 4.99))]
        confidence = round(_seed_float(f"conf_{pid}", 0.60, 0.99), 2)
        patients.append({
            "patient_id": pid,
            "risk_score": score,
            "risk_tier": tier,
            "forecast_horizon": horizon,
            "model_confidence": confidence,
            "escalation_actions": _ESCALATION_ACTIONS.get(tier, []),
        })
        if tier in ("critical", "high"):
            escalation_log.append({
                "patient_id": pid,
                "risk_tier": tier,
                "risk_score": score,
                "horizon": horizon,
                "actions_triggered": _ESCALATION_ACTIONS[tier],
                "response_target_min": 5 if tier == "critical" else 15,
            })

    forecasting_gaps = [g for g in gaps if g.get("category") == "decision_ai"
                        or "forecast" in g.get("feature", "").lower()
                        or "predict" in g.get("feature", "").lower()
                        or "pre-ictal" in g.get("feature", "").lower()]

    return {
        "available": True,
        "per_patient_forecasts": patients,
        "escalation_log": escalation_log,
        "forecasting_gaps": forecasting_gaps,
        "total_escalations": len(escalation_log),
    }


def definitions():
    """Terminology for seizure-risk forecasting concepts."""
    return [
        {"term": "Pre-ictal", "definition": "The period before a seizure begins — the forecasting target window where risk rises above baseline."},
        {"term": "Risk Score", "definition": "A 0–1 probability estimate of seizure onset within the forecast horizon, computed from EEG features and patient history."},
        {"term": "Forecast Horizon", "definition": "The time window (e.g. 0-15 min, 15-30 min) for which the risk score applies — how far ahead the model is predicting."},
        {"term": "Risk Tier", "definition": "Categorical risk level (low/moderate/high/critical) derived from the risk score against configured thresholds."},
        {"term": "Alert Threshold", "definition": "The risk-score cutoff that triggers a given tier's escalation actions (e.g. ≥0.75 = critical)."},
        {"term": "Escalation Action", "definition": "A specific clinical or notification action taken when a risk tier is reached — from logging to paging the on-call neurologist."},
        {"term": "False Alarm Rate (FAR)", "definition": "The number of false positive alerts per hour — a key metric for forecasting system usability."},
        {"term": "Sensitivity @ Horizon", "definition": "The proportion of true seizures correctly predicted within the given forecast horizon."},
        {"term": "Decision AI", "definition": "The layer that converts model outputs (probabilities) into actionable clinical decisions using thresholds, confidence gates, and escalation rules."},
        {"term": "Closed-Loop Response", "definition": "An automated action chain triggered by high risk — e.g. pre-ictal detection → caregiver SOS → rescue medication protocol."},
    ]
