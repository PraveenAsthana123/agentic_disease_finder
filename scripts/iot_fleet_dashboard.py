#!/usr/bin/env python3
"""
IoT Fleet Dashboard
====================

Analyses REAL IoT fleet data from clinical.db:

  1. **iot_devices**  — device_id, device_type, firmware_version, battery_pct,
     signal_strength_dbm, status, patient_id, last_seen, location, latency_ms
  2. **iot_gateways** — gateway_id, location, status, uptime_pct,
     connected_devices, last_heartbeat, firmware_version
  3. **iot_alerts**   — alert_type, severity, device_id, gateway_id, patient_id,
     acknowledged, resolved, timestamp

The dashboard visualises fleet health end-to-end:
  device status → gateway health → alert pipeline → patient coverage

Functions:
  overview()    — KPIs, status distribution, device type breakdown, alert severity
  breakdown()   — per-device detail, gateway table, unresolved alerts, patient coverage
  definitions() — glossary, alert types, severity levels, connectivity modes, references
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


def _parse_json_rows(rows):
    """Expand fields_json column into flat dicts."""
    out = []
    for r in rows:
        base = dict(r)
        fj = base.pop("fields_json", None)
        if fj:
            try:
                base.update(json.loads(fj))
            except Exception:
                pass
        out.append(base)
    return out


def _avg(vals):
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _pct(num, denom):
    return round(num / denom, 4) if denom else 0.0


def _safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# overview — fleet-level KPIs + distributions
# ---------------------------------------------------------------------------

def overview() -> Dict[str, Any]:
    raw_devices = _rows("SELECT * FROM iot_devices")
    raw_gateways = _rows("SELECT * FROM iot_gateways")
    raw_alerts = _rows("SELECT * FROM iot_alerts")

    devices = _parse_json_rows(raw_devices)
    gateways = _parse_json_rows(raw_gateways)
    alerts = _parse_json_rows(raw_alerts)

    total_devices = len(devices)
    total_gateways = len(gateways)
    total_alerts = len(alerts)

    if total_devices == 0 and total_gateways == 0:
        return {
            "kpis": {},
            "device_status_distribution": [],
            "device_type_distribution": [],
            "alert_severity_distribution": [],
            "gateway_status_distribution": [],
            "location_distribution": [],
        }

    # ── Device KPIs ──────────────────────────────────────────────────
    online_devices = sum(1 for d in devices if d.get("status") == "online")
    batteries = [_safe_float(d.get("battery_pct")) for d in devices if d.get("battery_pct") is not None]
    latencies = [_safe_float(d.get("latency_ms")) for d in devices if d.get("latency_ms") is not None]
    signals = [_safe_float(d.get("signal_strength_dbm")) for d in devices if d.get("signal_strength_dbm") is not None]
    low_battery = sum(1 for d in devices if _safe_float(d.get("battery_pct", 100)) < 20)

    unique_patients = len(set(d.get("patient_id") for d in devices if d.get("patient_id")))

    # ── Gateway KPIs ─────────────────────────────────────────────────
    gw_online = sum(1 for g in gateways if g.get("status") == "online")
    uptimes = [_safe_float(g.get("uptime_pct")) for g in gateways if g.get("uptime_pct") is not None]
    connected = [_safe_float(g.get("connected_devices")) for g in gateways if g.get("connected_devices") is not None]

    # ── Alert KPIs ───────────────────────────────────────────────────
    unresolved = sum(1 for a in alerts if not a.get("resolved"))
    critical_alerts = sum(1 for a in alerts if a.get("severity") == "critical" and not a.get("resolved"))
    unacknowledged = sum(1 for a in alerts if not a.get("acknowledged"))

    kpis = {
        "total_devices": total_devices,
        "online_devices": online_devices,
        "device_availability_rate": _pct(online_devices, total_devices),
        "avg_battery_pct": _avg(batteries),
        "low_battery_devices": low_battery,
        "avg_latency_ms": _avg(latencies),
        "avg_signal_dbm": _avg(signals),
        "total_gateways": total_gateways,
        "online_gateways": gw_online,
        "avg_gateway_uptime_pct": _avg(uptimes),
        "total_connected_devices": int(sum(connected)),
        "total_alerts": total_alerts,
        "unresolved_alerts": unresolved,
        "critical_unresolved": critical_alerts,
        "unacknowledged_alerts": unacknowledged,
        "unique_patients_covered": unique_patients,
    }

    # ── Distributions ────────────────────────────────────────────────
    status_counts = Counter(d.get("status", "unknown") for d in devices)
    device_status_dist = [{"name": s, "count": c, "pct": _pct(c, total_devices)}
                          for s, c in status_counts.most_common()]

    type_counts = Counter(d.get("device_type", "unknown") for d in devices)
    device_type_dist = [{"name": t, "count": c, "pct": _pct(c, total_devices)}
                        for t, c in type_counts.most_common()]

    alert_sev_counts = Counter(a.get("severity", "unknown") for a in alerts)
    sev_order = ["critical", "warning", "info"]
    alert_sev_dist = []
    for sev in sev_order:
        if sev in alert_sev_counts:
            alert_sev_dist.append({"name": sev, "count": alert_sev_counts[sev],
                                   "pct": _pct(alert_sev_counts[sev], total_alerts)})

    gw_status_counts = Counter(g.get("status", "unknown") for g in gateways)
    gw_status_dist = [{"name": s, "count": c} for s, c in gw_status_counts.most_common()]

    loc_counts = Counter(d.get("location", "unknown") for d in devices)
    location_dist = [{"name": loc, "count": c} for loc, c in loc_counts.most_common()]

    return {
        "kpis": kpis,
        "device_status_distribution": device_status_dist,
        "device_type_distribution": device_type_dist,
        "alert_severity_distribution": alert_sev_dist,
        "gateway_status_distribution": gw_status_dist,
        "location_distribution": location_dist,
    }


# ---------------------------------------------------------------------------
# breakdown — per-device, gateway table, alerts, patient coverage
# ---------------------------------------------------------------------------

def breakdown() -> Dict[str, Any]:
    raw_devices = _rows("SELECT * FROM iot_devices ORDER BY created_at DESC")
    raw_gateways = _rows("SELECT * FROM iot_gateways ORDER BY created_at DESC")
    raw_alerts = _rows("SELECT * FROM iot_alerts ORDER BY created_at DESC")

    devices = _parse_json_rows(raw_devices)
    gateways = _parse_json_rows(raw_gateways)
    alerts = _parse_json_rows(raw_alerts)

    # ── Device detail table (top 30) ──────────────────────────────────
    device_table = []
    for d in devices[:30]:
        device_table.append({
            "device_id": d.get("device_id", f"dev-{d.get('id')}"),
            "type": d.get("device_type", "unknown"),
            "status": d.get("status", "unknown"),
            "patient_id": d.get("patient_id"),
            "location": d.get("location", ""),
            "battery_pct": d.get("battery_pct"),
            "signal_dbm": d.get("signal_strength_dbm"),
            "latency_ms": d.get("latency_ms"),
            "firmware": d.get("firmware_version", ""),
            "last_seen": d.get("last_seen", ""),
        })

    # ── Gateway table ─────────────────────────────────────────────────
    gateway_table = []
    for g in gateways:
        gateway_table.append({
            "gateway_id": g.get("gateway_id", f"gw-{g.get('id')}"),
            "location": g.get("location", ""),
            "status": g.get("status", "unknown"),
            "uptime_pct": g.get("uptime_pct"),
            "connected_devices": g.get("connected_devices"),
            "firmware": g.get("firmware_version", ""),
            "last_heartbeat": g.get("last_heartbeat", ""),
        })

    # ── Unresolved alerts ─────────────────────────────────────────────
    unresolved_alerts = []
    for a in alerts:
        if not a.get("resolved"):
            unresolved_alerts.append({
                "alert_type": a.get("alert_type", ""),
                "severity": a.get("severity", ""),
                "device_id": a.get("device_id"),
                "gateway_id": a.get("gateway_id"),
                "patient_id": a.get("patient_id"),
                "acknowledged": a.get("acknowledged", False),
                "timestamp": a.get("timestamp", ""),
            })

    # ── Alert type breakdown ──────────────────────────────────────────
    alert_type_counts = Counter(a.get("alert_type", "unknown") for a in alerts)
    alert_type_breakdown = [{"type": t, "total": c,
                              "unresolved": sum(1 for a in alerts if a.get("alert_type") == t and not a.get("resolved"))}
                             for t, c in alert_type_counts.most_common()]

    # ── Patient coverage ──────────────────────────────────────────────
    pat_devices = defaultdict(list)
    for d in devices:
        pid = d.get("patient_id")
        if pid:
            pat_devices[pid].append(d)

    patient_coverage = []
    for pid in sorted(pat_devices.keys()):
        devs = pat_devices[pid]
        patient_coverage.append({
            "patient_id": pid,
            "devices": len(devs),
            "types": list(set(d.get("device_type", "?") for d in devs)),
            "online": sum(1 for d in devs if d.get("status") == "online"),
            "avg_battery": _avg([_safe_float(d.get("battery_pct")) for d in devs if d.get("battery_pct") is not None]),
        })
    patient_coverage.sort(key=lambda x: x["devices"], reverse=True)

    # ── Low-battery devices ───────────────────────────────────────────
    low_battery_devices = [
        {"device_id": d.get("device_id", "?"), "patient_id": d.get("patient_id"), "battery_pct": d.get("battery_pct"),
         "type": d.get("device_type"), "location": d.get("location")}
        for d in devices if _safe_float(d.get("battery_pct", 100)) < 30
    ]
    low_battery_devices.sort(key=lambda x: _safe_float(x["battery_pct"], 100))

    return {
        "device_table": device_table,
        "gateway_table": gateway_table,
        "unresolved_alerts": unresolved_alerts,
        "alert_type_breakdown": alert_type_breakdown,
        "patient_coverage": patient_coverage,
        "low_battery_devices": low_battery_devices,
    }


# ---------------------------------------------------------------------------
# definitions — glossary and reference
# ---------------------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    return {
        "device_types": {
            "implantable_rns": "Responsive Neurostimulation System — implanted device that detects and responds to seizure activity",
            "wearable_eeg": "Consumer or clinical EEG headset worn by the patient for ambulatory monitoring",
            "smartwatch": "Wrist-worn device capturing HR, SpO2, accelerometry, fall detection",
            "ecg_patch": "Adhesive patch recording continuous ECG for cardiac correlation with seizures",
            "seizure_band": "Embrace-style wristband detecting electro-dermal activity surges indicative of seizures",
            "mobile_app": "Patient or caregiver smartphone app for diary entry, alerts, and SOS",
        },
        "connectivity_modes": {
            "online": "Device streams data in real-time to the clinical platform via gateway or direct WiFi/LTE",
            "offline": "Device buffers data locally when connectivity is lost; sync occurs on reconnect",
            "hybrid": "Edge inference on device; summarised events streamed; raw data synced on connectivity",
        },
        "alert_types": {
            "low_battery": "Device battery below threshold (< 20%) — requires charging or replacement",
            "signal_lost": "Device has not reported telemetry within the expected interval",
            "firmware_outdated": "Device firmware version is below the minimum supported version",
            "gateway_offline": "Gateway has missed consecutive heartbeats — downstream devices may be unreachable",
            "seizure_detected": "On-device classifier triggered a seizure detection event requiring clinical review",
            "high_latency": "Round-trip telemetry latency exceeds clinical threshold (> 200 ms)",
        },
        "severity_levels": {
            "critical": "Immediate clinical or operational action required — escalated to on-call team",
            "warning": "Degraded performance or approaching threshold — schedule intervention within 24 h",
            "info": "Informational — no immediate action needed; logged for audit",
        },
        "kpi_definitions": [
            {"kpi": "Device Availability Rate", "desc": "Fraction of registered devices currently online"},
            {"kpi": "Avg Battery %", "desc": "Mean battery level across all devices with battery telemetry"},
            {"kpi": "Low-Battery Devices", "desc": "Count of devices with battery < 20%"},
            {"kpi": "Avg Latency (ms)", "desc": "Mean round-trip telemetry latency across reporting devices"},
            {"kpi": "Gateway Uptime %", "desc": "Mean uptime fraction across all registered gateways"},
            {"kpi": "Unresolved Alerts", "desc": "Count of alerts not yet marked resolved"},
            {"kpi": "Critical Unresolved", "desc": "Severity=critical alerts still open — highest priority"},
            {"kpi": "Patients Covered", "desc": "Distinct patients with at least one registered device"},
        ],
        "clinical_references": [
            "IEC 62304: Medical device software lifecycle",
            "FDA: Software as a Medical Device (SaMD) guidance",
            "AES: Wearable Technology in Epilepsy Monitoring (2023)",
            "NICE: Implanted brain stimulation for epilepsy (IPG416)",
            "MQTT Protocol v5.0 — ISO/IEC 20922",
        ],
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN ===")
    pprint.pprint(breakdown())
    print("\n=== DEFINITIONS ===")
    pprint.pprint(definitions())
