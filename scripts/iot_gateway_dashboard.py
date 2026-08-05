#!/usr/bin/env python3
"""
IoT Gateway Dashboard
=====================

Analyses REAL gateway data from clinical.db:

  **iot_gateways** — gateway_id, location, status, uptime_pct,
                     connected_devices, last_heartbeat, firmware_version
  **iot_devices**  — device_id, device_type, status, patient_id, location
  **iot_alerts**   — alert_type, severity, gateway_id, resolved, acknowledged

The dashboard focuses on gateway infrastructure health:
  gateway uptime → connected device load → firmware distribution
  → alert pipeline → per-location coverage

Functions:
  overview()    — KPIs, uptime distribution, firmware versions, alert summary
  breakdown()   — per-gateway table, location map, firmware gap, device load
  definitions() — glossary, connectivity modes, firmware tiers, references
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
# overview
# ---------------------------------------------------------------------------

def overview() -> Dict[str, Any]:
    gateways = _parse_json_rows(_rows("SELECT * FROM iot_gateways"))
    devices = _parse_json_rows(_rows("SELECT * FROM iot_devices"))
    alerts = _parse_json_rows(_rows("SELECT * FROM iot_alerts"))

    total_gw = len(gateways)
    online_gw = sum(1 for g in gateways if g.get("status") == "online")
    uptimes = [_safe_float(g.get("uptime_pct")) for g in gateways]
    avg_uptime = _avg(uptimes)
    min_uptime = round(min(uptimes), 1) if uptimes else 0.0
    max_uptime = round(max(uptimes), 1) if uptimes else 0.0

    total_connected = sum(int(g.get("connected_devices", 0)) for g in gateways)
    avg_load = _avg([int(g.get("connected_devices", 0)) for g in gateways])

    # Uptime tiers
    uptime_tiers = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    for u in uptimes:
        if u >= 95:
            uptime_tiers["excellent"] += 1
        elif u >= 90:
            uptime_tiers["good"] += 1
        elif u >= 80:
            uptime_tiers["fair"] += 1
        else:
            uptime_tiers["poor"] += 1

    # Firmware distribution
    fw_counter = Counter(g.get("firmware_version", "unknown") for g in gateways)
    firmware_dist = [
        {"version": v, "count": c, "pct": _pct(c, total_gw)}
        for v, c in fw_counter.most_common()
    ]

    # Alert summary for gateways
    gw_alerts = [a for a in alerts if a.get("gateway_id")]
    total_gw_alerts = len(gw_alerts)
    unresolved = sum(1 for a in gw_alerts if not a.get("resolved"))
    critical_alerts = sum(1 for a in gw_alerts if a.get("severity") == "critical")

    # Per-location summary
    location_map = {}
    for g in gateways:
        loc = g.get("location", "Unknown")
        location_map[loc] = {
            "gateway_id": g.get("gateway_id"),
            "uptime_pct": _safe_float(g.get("uptime_pct")),
            "connected_devices": int(g.get("connected_devices", 0)),
            "status": g.get("status", "unknown"),
        }

    # Latest heartbeat age (days from most-recent record)
    heartbeats = [g.get("last_heartbeat", "") for g in gateways if g.get("last_heartbeat")]

    return {
        "kpis": {
            "total_gateways": total_gw,
            "online_gateways": online_gw,
            "gateway_availability_rate": _pct(online_gw, total_gw),
            "avg_uptime_pct": avg_uptime,
            "min_uptime_pct": min_uptime,
            "max_uptime_pct": max_uptime,
            "total_connected_devices": total_connected,
            "avg_devices_per_gateway": round(avg_load, 1),
            "total_gateway_alerts": total_gw_alerts,
            "unresolved_gateway_alerts": unresolved,
            "critical_alerts": critical_alerts,
            "unique_firmware_versions": len(fw_counter),
        },
        "uptime_tier_distribution": [
            {"tier": t, "count": c, "pct": _pct(c, total_gw)}
            for t, c in uptime_tiers.items()
        ],
        "firmware_distribution": firmware_dist,
        "location_summary": list(location_map.values()),
        "last_heartbeats": sorted(heartbeats, reverse=True)[:6],
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown(gateway_id: str = None) -> Dict[str, Any]:
    gateways = _parse_json_rows(_rows("SELECT * FROM iot_gateways"))
    devices = _parse_json_rows(_rows("SELECT * FROM iot_devices"))
    alerts = _parse_json_rows(_rows("SELECT * FROM iot_alerts"))

    # Per-gateway table
    gw_alert_counts = Counter(a.get("gateway_id") for a in alerts if a.get("gateway_id"))
    gw_unresolved = Counter(
        a.get("gateway_id")
        for a in alerts
        if a.get("gateway_id") and not a.get("resolved")
    )

    gateway_table = []
    for g in gateways:
        gid = g.get("gateway_id", "")
        alert_count = gw_alert_counts.get(gid, 0)
        unresolved_count = gw_unresolved.get(gid, 0)
        gateway_table.append({
            "gateway_id": gid,
            "location": g.get("location", ""),
            "status": g.get("status", "unknown"),
            "uptime_pct": _safe_float(g.get("uptime_pct")),
            "connected_devices": int(g.get("connected_devices", 0)),
            "last_heartbeat": g.get("last_heartbeat", ""),
            "firmware_version": g.get("firmware_version", ""),
            "alerts": alert_count,
            "unresolved_alerts": unresolved_count,
        })
    gateway_table.sort(key=lambda x: x["uptime_pct"], reverse=True)

    # Filter by gateway if requested
    if gateway_id:
        gateway_table = [g for g in gateway_table if g["gateway_id"] == gateway_id]

    # Firmware gap analysis
    latest_fw = "4.0.0"
    outdated_gw = [
        g for g in gateway_table if g["firmware_version"] != latest_fw
    ]

    # Device load analysis per gateway
    # Group devices by location to approximate gateway coverage
    loc_device_count = Counter(d.get("location", "Unknown") for d in devices)
    device_load = []
    for g in gateways:
        loc = g.get("location", "")
        device_load.append({
            "gateway_id": g.get("gateway_id"),
            "location": loc,
            "reported_connected": int(g.get("connected_devices", 0)),
            "devices_in_location": loc_device_count.get(loc, 0),
        })

    # Alert log for gateways (most recent 20)
    gw_alerts = sorted(
        [a for a in alerts if a.get("gateway_id")],
        key=lambda x: x.get("timestamp", ""),
        reverse=True,
    )[:20]

    alert_log = []
    for a in gw_alerts:
        alert_log.append({
            "gateway_id": a.get("gateway_id", ""),
            "alert_type": a.get("alert_type", ""),
            "severity": a.get("severity", ""),
            "resolved": bool(a.get("resolved")),
            "acknowledged": bool(a.get("acknowledged")),
            "timestamp": a.get("timestamp", ""),
        })

    # Outdated firmware list
    outdated_list = [
        {"gateway_id": g["gateway_id"], "location": g["location"], "firmware_version": g["firmware_version"]}
        for g in outdated_gw
    ]

    return {
        "gateway_table": gateway_table,
        "device_load": device_load,
        "alert_log": alert_log,
        "outdated_firmware": outdated_list,
        "outdated_count": len(outdated_gw),
        "latest_firmware": latest_fw,
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    return {
        "glossary": [
            {"term": "Gateway", "definition": "Edge computing node that aggregates device data via BLE/MQTT and routes to backend."},
            {"term": "Uptime %", "definition": "Percentage of time a gateway is online and reachable (heartbeat check)."},
            {"term": "Connected Devices", "definition": "Number of EEG/wearable devices actively paired to a gateway."},
            {"term": "Last Heartbeat", "definition": "Most recent timestamp when the gateway sent a health ping."},
            {"term": "Firmware Version", "definition": "Embedded software version installed on the gateway hardware."},
            {"term": "MQTT", "definition": "Message Queuing Telemetry Transport — lightweight IoT protocol for device↔gateway messaging."},
            {"term": "BLE", "definition": "Bluetooth Low Energy — short-range wireless protocol used by wearables and EEG headsets."},
            {"term": "Device Load", "definition": "Number of devices handled by a single gateway; high load may increase latency."},
        ],
        "uptime_tiers": [
            {"tier": "Excellent", "range": "≥ 95%", "action": "No action needed."},
            {"tier": "Good", "range": "90–94%", "action": "Monitor; schedule maintenance window."},
            {"tier": "Fair", "range": "80–89%", "action": "Investigate drops; check network."},
            {"tier": "Poor", "range": "< 80%", "action": "Immediate remediation required."},
        ],
        "firmware_policy": [
            {"rule": "Latest version", "version": "4.0.0", "status": "Current — deploy to all gateways."},
            {"rule": "Supported", "version": "3.2.x", "status": "Security patches only."},
            {"rule": "EOL", "version": "≤ 3.1.x", "status": "End-of-life — upgrade required."},
        ],
        "connectivity_modes": [
            {"mode": "MQTT over TCP", "use": "High-throughput EEG stream from wired gateways (Ward/EMU/ICU)."},
            {"mode": "BLE relay", "use": "Short-range wearable pairing; gateway bridges to backend."},
            {"mode": "Offline buffer", "use": "Gateway stores data locally if backend unreachable; auto-syncs on reconnect."},
        ],
        "data_sources": [
            {"table": "iot_gateways", "rows": "6", "fields": "gateway_id, location, status, uptime_pct, connected_devices, last_heartbeat, firmware_version"},
            {"table": "iot_devices", "rows": "35", "fields": "device_id, device_type, status, patient_id, location, battery_pct, signal_strength_dbm"},
            {"table": "iot_alerts", "rows": "50", "fields": "alert_type, severity, gateway_id, device_id, patient_id, resolved, acknowledged, timestamp"},
        ],
        "references": [
            {"title": "MQTT v5.0 Specification", "source": "OASIS Standard, 2019"},
            {"title": "Bluetooth Low Energy (BLE 5.3)", "source": "Bluetooth SIG"},
            {"title": "Edge Computing for Epilepsy Monitoring", "source": "J. Neural Eng., 2022"},
        ],
    }
