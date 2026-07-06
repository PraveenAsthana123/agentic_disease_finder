#!/usr/bin/env python3
"""
Device Telemetry Dashboard
==========================

Monitors REAL device telemetry by cross-referencing:

  1. **iot_devices** — battery_pct, signal_strength_dbm, status, latency_ms
  2. **wearable_devices** — battery_level, connectivity, status, last_sync
  3. **iot_alerts** — alert_type, severity, resolved, acknowledged
  4. **iot_gateways** — gateway metadata and connectivity

The dashboard visualises fleet-wide device health:
  battery health → signal quality → connection status → degradation alerts

Functions:
  overview()     — fleet KPIs, battery/signal distributions, alert severity breakdown
  breakdown()    — per-device telemetry detail, per-alert-type summary, gateway health
  definitions()  — telemetry glossary, thresholds, clinical references
"""

import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, List

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    if not os.path.exists(_DB_PATH):
        return []
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _avg(vals):
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return round(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)), 4)


def _pct(num, denom):
    return round(num / denom, 4) if denom else 0.0


def _safe_float(val, default=0.0):
    """Convert value to float, returning default for None/NaN/Inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

def _load_iot_devices():
    """Return parsed IoT device records."""
    raw = _rows("SELECT id, patient_id, fields_json, created_at FROM iot_devices")
    out = []
    for r in raw:
        fields = _safe_json(r.get("fields_json", ""))
        fields["db_id"] = r["id"]
        fields["db_patient_id"] = r["patient_id"]
        fields["created_at"] = r["created_at"]
        out.append(fields)
    return out


def _load_wearable_devices():
    """Return parsed wearable device records."""
    raw = _rows("SELECT id, patient_id, fields_json, created_at FROM wearable_devices")
    out = []
    for r in raw:
        fields = _safe_json(r.get("fields_json", ""))
        fields["db_id"] = r["id"]
        fields["db_patient_id"] = r["patient_id"]
        fields["created_at"] = r["created_at"]
        out.append(fields)
    return out


def _load_iot_alerts():
    """Return parsed IoT alerts."""
    raw = _rows("SELECT id, patient_id, fields_json, created_at FROM iot_alerts")
    out = []
    for r in raw:
        fields = _safe_json(r.get("fields_json", ""))
        fields["db_id"] = r["id"]
        fields["db_patient_id"] = r["patient_id"]
        fields["created_at"] = r["created_at"]
        out.append(fields)
    return out


def _load_iot_gateways():
    """Return parsed IoT gateway records."""
    raw = _rows("SELECT id, patient_id, fields_json, created_at FROM iot_gateways")
    out = []
    for r in raw:
        fields = _safe_json(r.get("fields_json", ""))
        fields["db_id"] = r["id"]
        fields["db_patient_id"] = r["patient_id"]
        fields["created_at"] = r["created_at"]
        out.append(fields)
    return out


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview() -> Dict[str, Any]:
    iot = _load_iot_devices()
    wearables = _load_wearable_devices()
    alerts = _load_iot_alerts()

    # -- Fleet totals --
    total_iot = len(iot)
    total_wearable = len(wearables)
    total_devices = total_iot + total_wearable

    # -- Online / offline counts --
    iot_online = sum(1 for d in iot if str(d.get("status", "")).lower() == "online")
    iot_offline = total_iot - iot_online
    wear_active = sum(1 for d in wearables if str(d.get("status", "")).lower() == "active")
    wear_offline = total_wearable - wear_active
    online_count = iot_online + wear_active
    offline_count = iot_offline + wear_offline
    online_pct = _pct(online_count, total_devices)
    offline_pct = _pct(offline_count, total_devices)

    # -- Battery health --
    battery_vals = []
    for d in iot:
        v = d.get("battery_pct")
        if v is not None:
            battery_vals.append(_safe_float(v))
    for d in wearables:
        v = d.get("battery_level")
        if v is not None:
            battery_vals.append(_safe_float(v))

    avg_battery = _avg(battery_vals)
    low_battery_count = sum(1 for b in battery_vals if b < 30)

    # -- Battery distribution histogram --
    battery_buckets = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for b in battery_vals:
        if b < 20:
            battery_buckets["0-20"] += 1
        elif b < 40:
            battery_buckets["20-40"] += 1
        elif b < 60:
            battery_buckets["40-60"] += 1
        elif b < 80:
            battery_buckets["60-80"] += 1
        else:
            battery_buckets["80-100"] += 1
    battery_distribution = [{"bucket": k, "count": v} for k, v in battery_buckets.items()]

    # -- Signal strength (IoT only — dBm) --
    signal_vals = []
    for d in iot:
        v = d.get("signal_strength_dbm")
        if v is not None:
            signal_vals.append(_safe_float(v))

    avg_signal = _avg(signal_vals)
    weak_signal_count = sum(1 for s in signal_vals if s < -70)

    # -- Latency (IoT only) --
    latency_vals = []
    for d in iot:
        v = d.get("latency_ms")
        if v is not None:
            latency_vals.append(_safe_float(v))
    avg_latency = _avg(latency_vals)

    # -- Alerts summary --
    total_alerts = len(alerts)
    resolved_count = sum(1 for a in alerts if a.get("resolved"))
    unresolved_count = total_alerts - resolved_count

    severity_counts = Counter(a.get("severity", "unknown") for a in alerts)
    severity_breakdown = [
        {"severity": k, "count": v, "pct": _pct(v, total_alerts)}
        for k, v in severity_counts.most_common()
    ]

    # -- Device type breakdown --
    type_counter = Counter()
    for d in iot:
        type_counter[d.get("device_type", "unknown")] += 1
    for d in wearables:
        type_counter[d.get("device_type", "unknown")] += 1
    device_type_breakdown = [
        {"device_type": k, "count": v, "pct": _pct(v, total_devices)}
        for k, v in type_counter.most_common()
    ]

    # -- Recent alerts (last 10 by timestamp desc) --
    sorted_alerts = sorted(
        alerts,
        key=lambda a: a.get("timestamp") or a.get("created_at") or "",
        reverse=True,
    )
    recent_alerts = []
    for a in sorted_alerts[:10]:
        recent_alerts.append({
            "alert_type": a.get("alert_type"),
            "severity": a.get("severity"),
            "device_id": a.get("device_id"),
            "resolved": bool(a.get("resolved")),
            "acknowledged": bool(a.get("acknowledged")),
            "timestamp": a.get("timestamp") or a.get("created_at"),
        })

    return {
        "kpis": {
            "total_devices": total_devices,
            "total_iot": total_iot,
            "total_wearable": total_wearable,
            "online_count": online_count,
            "offline_count": offline_count,
            "online_pct": online_pct,
            "offline_pct": offline_pct,
            "avg_battery": avg_battery,
            "low_battery_count": low_battery_count,
            "avg_signal_dbm": avg_signal,
            "weak_signal_count": weak_signal_count,
            "avg_latency_ms": avg_latency,
            "total_alerts": total_alerts,
            "unresolved_alerts": unresolved_count,
            "resolved_alerts": resolved_count,
        },
        "severity_breakdown": severity_breakdown,
        "battery_distribution": battery_distribution,
        "device_type_breakdown": device_type_breakdown,
        "recent_alerts": recent_alerts,
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown() -> Dict[str, Any]:
    iot = _load_iot_devices()
    wearables = _load_wearable_devices()
    alerts = _load_iot_alerts()
    gateways = _load_iot_gateways()

    # -- Per-device telemetry (IoT) sorted by battery ascending (worst first) --
    iot_devices = []
    for d in iot:
        iot_devices.append({
            "device_id": d.get("device_id"),
            "device_type": d.get("device_type"),
            "patient_id": d.get("patient_id") or d.get("db_patient_id"),
            "battery_pct": _safe_float(d.get("battery_pct")),
            "signal_strength_dbm": _safe_float(d.get("signal_strength_dbm")),
            "status": d.get("status"),
            "latency_ms": _safe_float(d.get("latency_ms")),
            "last_seen": d.get("last_seen"),
            "firmware_version": d.get("firmware_version"),
            "location": d.get("location"),
        })
    iot_devices.sort(key=lambda x: x["battery_pct"])

    # -- Per-device telemetry (wearables) sorted by battery ascending --
    wearable_details = []
    for d in wearables:
        wearable_details.append({
            "device_id": d.get("device_id"),
            "device_type": d.get("device_type"),
            "brand": d.get("brand"),
            "battery_level": _safe_float(d.get("battery_level")),
            "connectivity": d.get("connectivity"),
            "status": d.get("status"),
            "last_sync": d.get("last_sync"),
            "firmware_version": d.get("firmware_version"),
            "seizure_detection_enabled": bool(d.get("seizure_detection_enabled")),
            "patient_id": d.get("patient_id") or d.get("db_patient_id"),
        })
    wearable_details.sort(key=lambda x: x["battery_level"])

    # -- Per-alert-type summary --
    alert_type_groups = defaultdict(list)
    for a in alerts:
        alert_type_groups[a.get("alert_type", "unknown")].append(a)

    per_alert_type = []
    for atype, items in sorted(alert_type_groups.items()):
        total = len(items)
        resolved = sum(1 for a in items if a.get("resolved"))
        unresolved = total - resolved
        pct_resolved = _pct(resolved, total)
        per_alert_type.append({
            "alert_type": atype,
            "count": total,
            "unresolved": unresolved,
            "resolved": resolved,
            "pct_resolved": pct_resolved,
        })

    # -- Gateway health --
    gateway_health = []
    for g in gateways:
        gateway_health.append({
            "gateway_id": g.get("gateway_id"),
            "status": g.get("status"),
            "connected_devices": g.get("connected_devices", 0),
            "location": g.get("location"),
            "firmware_version": g.get("firmware_version"),
            "uptime_hours": _safe_float(g.get("uptime_hours")),
        })

    return {
        "iot_devices": iot_devices,
        "wearable_devices": wearable_details,
        "per_alert_type": per_alert_type,
        "gateway_health": gateway_health,
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    return {
        "title": "Device Telemetry — Thresholds, Glossary & Clinical References",
        "signal_strength_thresholds": [
            {"level": "Excellent", "range": "> -50 dBm", "description": "Strong signal, minimal packet loss, reliable real-time streaming."},
            {"level": "Good", "range": "-50 to -65 dBm", "description": "Adequate for continuous monitoring with occasional retransmissions."},
            {"level": "Fair", "range": "-65 to -75 dBm", "description": "Intermittent connectivity possible; buffered transmission recommended."},
            {"level": "Poor", "range": "< -75 dBm", "description": "High packet loss risk; device may fall back to store-and-forward mode."},
        ],
        "battery_thresholds": [
            {"level": "Critical", "range": "< 15%", "description": "Immediate replacement or recharge required. Alerts auto-generated."},
            {"level": "Low", "range": "15–30%", "description": "Proactive replacement recommended within 24 hours."},
            {"level": "Normal", "range": "30–80%", "description": "Adequate charge for standard monitoring duty cycles."},
            {"level": "Full", "range": "> 80%", "description": "Fully charged; no action needed."},
        ],
        "alert_severity_definitions": [
            {"severity": "critical", "description": "Immediate clinical or operational impact. Device failure, patient safety risk, or complete signal loss. Requires response within 15 minutes."},
            {"severity": "warning", "description": "Degraded performance or approaching threshold. Low battery, weak signal, firmware outdated. Requires response within 4 hours."},
            {"severity": "info", "description": "Informational event. Routine status change, successful sync, or planned maintenance. No immediate action required."},
        ],
        "device_types_glossary": [
            {"type": "EEG headband", "description": "Non-invasive scalp electroencephalography device for continuous brainwave monitoring. Detects epileptiform discharges and seizure onset patterns."},
            {"type": "Wrist wearable", "description": "Accelerometer + photoplethysmography (PPG) device for heart rate, HRV, and motion-based seizure detection."},
            {"type": "Smart patch", "description": "Adhesive biosensor for long-term ECG, temperature, and electrodermal activity (EDA) monitoring."},
            {"type": "Gateway", "description": "Local hub that aggregates data from multiple BLE/Zigbee devices and relays to the cloud via Wi-Fi or cellular."},
            {"type": "Environmental sensor", "description": "Ambient temperature, humidity, and air quality monitor for contextual seizure trigger analysis."},
            {"type": "Pulse oximeter", "description": "Continuous SpO2 and pulse rate monitor, critical for detecting ictal hypoxemia."},
        ],
        "clinical_importance": [
            "Continuous device telemetry is essential for ensuring uninterrupted seizure monitoring. A single missed seizure event due to device failure can have life-threatening consequences.",
            "Battery degradation follows a non-linear curve — devices may report 20% charge and then fail within minutes. Proactive replacement at 30% is the clinical standard.",
            "Signal strength below -70 dBm in hospital environments correlates with >5% packet loss, potentially missing critical EEG segments during seizure events.",
            "Firmware updates must be validated against clinical certification (IEC 62304) before deployment to avoid introducing software-related hazards.",
            "Device downtime metrics feed directly into regulatory reporting (FDA 21 CFR Part 820) for post-market surveillance of medical-grade wearables.",
        ],
        "clinical_references": [
            "Patel S et al. A review of wearable sensors and systems with application in rehabilitation. J NeuroEngineering and Rehabilitation. 2012;9(1):21.",
            "Johansson D et al. Wearable sensors for clinical applications in epilepsy, Parkinson's disease, and stroke: a mixed-methods systematic review. J Neurol. 2020;267(8):2137-2163.",
            "Nasseri M et al. Signal quality and patient experience with wearable devices for epilepsy management. Epilepsia. 2020;61(Suppl 1):S25-S35.",
            "Bruno E et al. Wearable technology in epilepsy: The views of patients, caregivers, and healthcare professionals. Epilepsy & Behavior. 2018;85:141-149.",
        ],
        "data_tables_used": [
            "iot_devices (device_id, battery_pct, signal_strength_dbm, status, latency_ms)",
            "wearable_devices (device_id, battery_level, connectivity, status, last_sync)",
            "iot_alerts (alert_type, severity, resolved, acknowledged)",
            "iot_gateways (gateway_id, status, connected_devices)",
        ],
    }


if __name__ == "__main__":
    import json as _json
    print("=== OVERVIEW ===")
    print(_json.dumps(overview(), indent=2, default=str)[:2000])
    print("\n=== BREAKDOWN ===")
    print(_json.dumps(breakdown(), indent=2, default=str)[:2000])
    print("\n=== DEFINITIONS ===")
    print(_json.dumps(definitions(), indent=2, default=str)[:1000])
