#!/usr/bin/env python3
"""
AgenticFinder Drift Monitoring Dashboard
=========================================

Read-only dashboard module that surfaces drift detection results from
``jobs/reports/drift_latest.json`` (produced by the DRIFT cron job running
``scripts/drift_monitor.py``).

Functions return plain dicts/lists suitable for JSON serialisation so they
can be consumed by FastAPI endpoints, CLI reporters, or frontend charts.

All data is REAL -- read from disk, never fabricated.  When the report file
is missing the functions return ``{"available": false, ...}`` with guidance
on how to generate it.

PSI severity thresholds (aligned with drift_monitor.py):
    < 0.10   No significant drift
    0.10-0.25  Small / moderate drift
    0.25-0.50  Medium drift
    > 0.50   Large / severe drift
"""

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPORTS_DIR = str(Path(__file__).parent.parent / "jobs" / "reports")
LATEST_PATH = str(Path(REPORTS_DIR) / "drift_latest.json")

# ── PSI severity bands ──────────────────────────────────────────────
PSI_THRESHOLDS = {
    "no_drift": 0.10,
    "small": 0.25,
    "medium": 0.50,
}

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "none": 3}

_NOT_AVAILABLE = {
    "available": False,
    "note": "Run scripts/drift_job.py to generate drift data",
}


# ── helpers ──────────────────────────────────────────────────────────
def _load_latest() -> Optional[Dict[str, Any]]:
    """Load the latest drift report; return *None* if absent/corrupt."""
    p = Path(LATEST_PATH)
    if not p.exists():
        return None
    try:
        with open(p, "r") as fh:
            data = json.load(fh)
        if not data.get("available", False):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _classify_psi(psi: float) -> str:
    """Return human-readable severity band for a PSI value."""
    if psi < PSI_THRESHOLDS["no_drift"]:
        return "no_drift"
    if psi < PSI_THRESHOLDS["small"]:
        return "small"
    if psi < PSI_THRESHOLDS["medium"]:
        return "medium"
    return "large"


def _recommend(frac_drifted: float, n_high: int, n_features: int) -> str:
    """Return a one-paragraph recommendation based on drift severity."""
    if frac_drifted == 0.0:
        return (
            "No drift detected.  Model inputs match the training "
            "distribution.  Continue routine monitoring."
        )
    if frac_drifted < 0.10:
        return (
            "Minor drift in a small subset of features.  Monitor over "
            "the next few runs; no immediate action required."
        )
    if frac_drifted < 0.30:
        return (
            "Moderate drift detected.  Investigate the top-drifted "
            "features for data-pipeline changes or population shifts.  "
            "Consider retraining if accuracy degrades."
        )
    if frac_drifted < 0.70:
        return (
            "Significant drift across many features.  Model predictions "
            "may be unreliable.  Retrain on recent data or apply domain "
            "adaptation before using predictions clinically."
        )
    return (
        "SEVERE drift -- virtually all features have shifted.  Model "
        "confidence is NOT trustworthy.  Human oversight is required for "
        "every prediction until a retrained model is deployed."
    )


# ── public API ───────────────────────────────────────────────────────
def drift_overview() -> Dict[str, Any]:
    """High-level drift summary.

    Returns dict with keys: available, verdict, frac_drifted,
    severity_breakdown, last_run, n_reference, n_live, n_features,
    disease, recommendation, interpretation.
    """
    data = _load_latest()
    if data is None:
        return _NOT_AVAILABLE

    top_drift: List[Dict] = data.get("top_drift", [])
    # Count by severity across ALL reported features
    breakdown = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for feat in top_drift:
        sev = feat.get("severity", "none").lower()
        if sev in breakdown:
            breakdown[sev] += 1
        else:
            breakdown["none"] += 1

    n_features = data.get("n_features", 0)
    n_high = data.get("n_high_drift", 0)
    frac = data.get("frac_drifted", 0.0)

    return {
        "available": True,
        "verdict": data.get("verdict", "unknown"),
        "frac_drifted": round(frac, 4),
        "severity_breakdown": breakdown,
        "last_run": data.get("run_at_local", ""),
        "n_reference": data.get("n_reference", 0),
        "n_live": data.get("n_live", 0),
        "n_features": n_features,
        "n_high_drift": n_high,
        "disease": data.get("disease", ""),
        "method": data.get("method", ""),
        "recommendation": _recommend(frac, n_high, n_features),
        "interpretation": data.get("interpretation", ""),
    }


def drift_features(
    sort_by: str = "psi", limit: int = 20
) -> Dict[str, Any]:
    """Per-feature drift table.

    Parameters
    ----------
    sort_by : str
        ``"psi"`` (default) or ``"ks"`` -- sort descending by that metric.
    limit : int
        Maximum number of features to return (default 20).

    Returns dict with ``features`` list, each entry containing:
        feature, psi, psi_band, ks_stat, ks_p, severity.
    """
    data = _load_latest()
    if data is None:
        return _NOT_AVAILABLE

    top_drift: List[Dict] = data.get("top_drift", [])

    enriched = []
    for f in top_drift:
        psi_val = f.get("psi", 0.0)
        enriched.append({
            "feature": f.get("feature", ""),
            "psi": round(psi_val, 4),
            "psi_band": _classify_psi(psi_val),
            "ks_stat": round(f.get("ks_stat", 0.0), 4),
            "ks_p": f.get("ks_p", 1.0),
            "severity": f.get("severity", "none"),
        })

    key = "psi" if sort_by.lower().startswith("psi") else "ks_stat"
    enriched.sort(key=lambda x: x[key], reverse=True)

    return {
        "available": True,
        "total_features": len(enriched),
        "showing": min(limit, len(enriched)),
        "sort_by": key,
        "features": enriched[:limit],
    }


def drift_severity_distribution() -> Dict[str, Any]:
    """Severity counts suitable for pie/bar chart rendering.

    Returns dict with ``labels`` and ``values`` lists (parallel arrays)
    plus ``total``.
    """
    data = _load_latest()
    if data is None:
        return _NOT_AVAILABLE

    top_drift: List[Dict] = data.get("top_drift", [])
    counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for f in top_drift:
        sev = f.get("severity", "none").lower()
        if sev in counts:
            counts[sev] += 1
        else:
            counts["none"] += 1

    # Exclude zero-count buckets for cleaner charts
    labels = [k for k, v in counts.items() if v > 0]
    values = [counts[k] for k in labels]

    return {
        "available": True,
        "labels": labels,
        "values": values,
        "total": sum(values),
        "colors": {
            "high": "#ef4444",
            "medium": "#f59e0b",
            "low": "#3b82f6",
            "none": "#22c55e",
        },
    }


def drift_alerts(psi_threshold: float = 0.25) -> Dict[str, Any]:
    """Actionable alerts for features exceeding *psi_threshold*.

    Each alert includes the feature name, scores, and a recommended action.
    """
    data = _load_latest()
    if data is None:
        return _NOT_AVAILABLE

    top_drift: List[Dict] = data.get("top_drift", [])
    alerts: List[Dict[str, Any]] = []

    for f in top_drift:
        psi_val = f.get("psi", 0.0)
        if psi_val < psi_threshold:
            continue
        band = _classify_psi(psi_val)
        if band == "large":
            action = (
                "Severe shift -- investigate data pipeline for this feature; "
                "consider excluding or re-engineering before retraining."
            )
        elif band == "medium":
            action = (
                "Moderate shift -- check for recording/preprocessing changes; "
                "monitor over next 2-3 runs."
            )
        else:
            action = "Minor shift -- continue monitoring."

        alerts.append({
            "feature": f.get("feature", ""),
            "psi": round(psi_val, 4),
            "ks_stat": round(f.get("ks_stat", 0.0), 4),
            "ks_p": f.get("ks_p", 1.0),
            "severity": f.get("severity", "none"),
            "psi_band": band,
            "recommended_action": action,
        })

    alerts.sort(key=lambda x: x["psi"], reverse=True)

    return {
        "available": True,
        "psi_threshold_used": psi_threshold,
        "n_alerts": len(alerts),
        "alerts": alerts,
    }


def drift_trend() -> Dict[str, Any]:
    """Historical drift trend from all ``drift_*.json`` files.

    Returns time-series lists for ``timestamps``, ``frac_drifted``,
    and ``n_high_drift`` (parallel arrays), sorted chronologically.
    """
    pattern = str(Path(REPORTS_DIR) / "drift_*.json")
    files = glob.glob(pattern)

    if not files:
        return {
            "available": False,
            "note": "No historical drift reports found in jobs/reports/",
        }

    points: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r") as fh:
                d = json.load(fh)
            if not d.get("available", False):
                continue
            points.append({
                "timestamp": d.get("run_at_local", ""),
                "frac_drifted": round(d.get("frac_drifted", 0.0), 4),
                "n_high_drift": d.get("n_high_drift", 0),
                "n_features": d.get("n_features", 0),
                "verdict": d.get("verdict", ""),
                "disease": d.get("disease", ""),
                "source_file": Path(fp).name,
            })
        except (json.JSONDecodeError, OSError):
            continue

    if not points:
        return {
            "available": False,
            "note": "Historical drift files exist but none contain valid data",
        }

    # Sort chronologically
    points.sort(key=lambda x: x["timestamp"])

    return {
        "available": True,
        "n_points": len(points),
        "timestamps": [p["timestamp"] for p in points],
        "frac_drifted": [p["frac_drifted"] for p in points],
        "n_high_drift": [p["n_high_drift"] for p in points],
        "points": points,
    }


def scale_definitions() -> Dict[str, Any]:
    """Reference definitions for PSI thresholds, KS-test, severity levels.

    Returns a static dict documenting the drift metrics and their
    interpretation -- useful for tooltip / help panels.
    """
    return {
        "available": True,
        "psi": {
            "name": "Population Stability Index (PSI)",
            "description": (
                "Measures how much a feature's distribution has shifted "
                "between the reference (training) and live (serving) data. "
                "Based on binned log-likelihood ratios."
            ),
            "thresholds": {
                "no_drift": {
                    "range": "PSI < 0.10",
                    "label": "No significant drift",
                    "action": "No action needed",
                },
                "small": {
                    "range": "0.10 <= PSI < 0.25",
                    "label": "Small / moderate drift",
                    "action": "Monitor; investigate if persistent",
                },
                "medium": {
                    "range": "0.25 <= PSI < 0.50",
                    "label": "Medium drift",
                    "action": "Investigate root cause; consider retraining",
                },
                "large": {
                    "range": "PSI >= 0.50",
                    "label": "Large / severe drift",
                    "action": (
                        "Model predictions unreliable; retrain or apply "
                        "human oversight immediately"
                    ),
                },
            },
        },
        "ks_test": {
            "name": "Kolmogorov-Smirnov Test",
            "description": (
                "Non-parametric test comparing the empirical CDFs of "
                "reference and live distributions.  Returns a statistic "
                "(0-1, higher = more different) and a p-value."
            ),
            "interpretation": {
                "ks_stat": (
                    "Maximum absolute difference between CDFs.  "
                    "Values near 0 indicate similar distributions; "
                    "values near 1 indicate completely different."
                ),
                "ks_p": (
                    "p-value of the KS test.  p < 0.05 typically "
                    "indicates a statistically significant difference "
                    "between distributions."
                ),
            },
        },
        "severity_levels": {
            "high": "PSI >= 0.25 -- distribution has changed substantially",
            "medium": "0.10 <= PSI < 0.25 -- noticeable shift, monitor closely",
            "low": "PSI < 0.10 -- minimal change, within normal variation",
        },
        "verdicts": {
            "SEVERE drift": (
                "Majority of features have high drift; model outputs "
                "should not be trusted without human review."
            ),
            "MODERATE drift": (
                "A meaningful fraction of features have drifted; "
                "investigate and plan retraining."
            ),
            "MINOR drift": (
                "A few features show small shifts; routine monitoring "
                "is sufficient."
            ),
            "NO drift": "All features are stable relative to training data.",
        },
    }


# ── CLI convenience ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    commands = {
        "overview": drift_overview,
        "features": lambda: drift_features(sort_by="psi", limit=20),
        "severity": drift_severity_distribution,
        "alerts": drift_alerts,
        "trend": drift_trend,
        "scales": scale_definitions,
    }

    cmd = sys.argv[1] if len(sys.argv) > 1 else "overview"
    if cmd in ("--help", "-h"):
        print(f"Usage: {sys.argv[0]} [{' | '.join(commands.keys())}]")
        sys.exit(0)

    fn = commands.get(cmd)
    if fn is None:
        print(f"Unknown command: {cmd}. Use one of: {', '.join(commands.keys())}")
        sys.exit(1)

    result = fn()
    print(json.dumps(result, indent=2))
