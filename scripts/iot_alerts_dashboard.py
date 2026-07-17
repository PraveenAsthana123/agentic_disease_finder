"""
Neuro AI Ecosystem -- IoT Alerts Dashboard
===========================================
Tracks IoT device alerts for epilepsy monitoring infrastructure, including
seizure detection sensors, wearable gateways, and signal quality events.

Alert Types:
  low_battery        -- Device battery level below operational threshold
  firmware_outdated  -- Device firmware requires update
  gateway_offline    -- Communication gateway unreachable
  signal_lost        -- EEG/sensor signal interrupted
  sos_seizure        -- Seizure detection event triggered by device
  high_impedance     -- Electrode impedance above acceptable range

Severity Levels:
  info       -- Informational, no immediate action required
  warning    -- Attention needed within operational window
  critical   -- Immediate clinical or technical response required

All alert records are stored in the iot_alerts table of clinical.db.
The fields_json column contains structured alert metadata parsed at
query time via json.loads().

Author: Research Team
"""

import sqlite3
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _json_safe(obj):
    """Convert numpy/date types to JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _parse_fields(fields_json_str: str) -> dict:
    """Safely parse the fields_json column."""
    if not fields_json_str:
        return {}
    try:
        return json.loads(fields_json_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _load_all_alerts(conn, patient_id: Optional[str] = None) -> list:
    """Load all iot_alerts rows, parsing fields_json into flat dicts."""
    c = conn.cursor()
    if patient_id:
        c.execute(
            "SELECT id, patient_id, fields_json, created_at "
            "FROM iot_alerts WHERE patient_id=? ORDER BY created_at DESC",
            (patient_id,),
        )
    else:
        c.execute(
            "SELECT id, patient_id, fields_json, created_at "
            "FROM iot_alerts ORDER BY created_at DESC"
        )
    rows = []
    for row in c.fetchall():
        fields = _parse_fields(row[2])
        rows.append({
            "id":           row[0],
            "patient_id":   row[1],
            "alert_type":   fields.get("alert_type", "unknown"),
            "severity":     fields.get("severity", "unknown"),
            "device_id":    fields.get("device_id", "unknown"),
            "gateway_id":   fields.get("gateway_id", "unknown"),
            "acknowledged": fields.get("acknowledged", False),
            "resolved":     fields.get("resolved", False),
            "timestamp":    fields.get("timestamp", row[3]),
            "created_at":   row[3],
        })
    return rows


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def overview(patient_id: Optional[str] = None) -> dict:
    """
    IoT alerts overview dashboard data.

    Returns:
        total_alerts            -- total alert records
        critical_count          -- alerts with severity=critical
        unresolved_count        -- alerts not yet resolved
        unacknowledged_count    -- alerts not yet acknowledged
        acknowledgment_rate     -- percentage of acknowledged alerts
        resolution_rate         -- percentage of resolved alerts
        severity_distribution   -- list of {name, value} for pie chart
        alert_type_distribution -- list of {name, value} for bar chart
        monthly_trend           -- list of {month, critical, warning, info} for line chart
        total_patients          -- distinct patients with alerts
        total_devices           -- distinct devices generating alerts
    """
    conn = _conn()
    alerts = _load_all_alerts(conn, patient_id)
    conn.close()

    if not alerts:
        return _json_safe({
            "total_alerts": 0,
            "critical_count": 0,
            "unresolved_count": 0,
            "unacknowledged_count": 0,
            "acknowledgment_rate": 0.0,
            "resolution_rate": 0.0,
            "severity_distribution": [],
            "alert_type_distribution": [],
            "monthly_trend": [],
            "total_patients": 0,
            "total_devices": 0,
        })

    total_alerts = len(alerts)
    critical_count = sum(1 for a in alerts if a["severity"] == "critical")
    warning_count = sum(1 for a in alerts if a["severity"] == "warning")
    info_count = sum(1 for a in alerts if a["severity"] == "info")
    unresolved_count = sum(1 for a in alerts if not a["resolved"])
    unacknowledged_count = sum(1 for a in alerts if not a["acknowledged"])
    acknowledged_count = sum(1 for a in alerts if a["acknowledged"])
    resolved_count = sum(1 for a in alerts if a["resolved"])

    acknowledgment_rate = round(acknowledged_count / total_alerts * 100, 1) if total_alerts else 0.0
    resolution_rate = round(resolved_count / total_alerts * 100, 1) if total_alerts else 0.0

    # Severity distribution (pie chart)
    sev_counts = defaultdict(int)
    for a in alerts:
        sev_counts[a["severity"]] += 1
    severity_distribution = [
        {"name": sev, "value": cnt}
        for sev, cnt in sorted(sev_counts.items(), key=lambda x: -x[1])
    ]

    # Alert type distribution (bar chart)
    type_counts = defaultdict(int)
    for a in alerts:
        type_counts[a["alert_type"]] += 1
    alert_type_distribution = [
        {"name": atype, "value": cnt}
        for atype, cnt in sorted(type_counts.items(), key=lambda x: -x[1])
    ]

    # Monthly trend (line chart)
    month_data = defaultdict(lambda: {"critical": 0, "warning": 0, "info": 0})
    for a in alerts:
        ts = a["timestamp"] or a["created_at"] or ""
        if len(ts) >= 7:
            month_key = ts[:7]  # YYYY-MM
            sev = a["severity"]
            if sev in month_data[month_key]:
                month_data[month_key][sev] += 1
    monthly_trend = [
        {"month": m, "critical": d["critical"], "warning": d["warning"], "info": d["info"]}
        for m, d in sorted(month_data.items())
    ]

    # Unique patients and devices
    total_patients = len({a["patient_id"] for a in alerts if a["patient_id"]})
    total_devices = len({a["device_id"] for a in alerts if a["device_id"] != "unknown"})

    return _json_safe({
        "total_alerts":            total_alerts,
        "critical_count":          critical_count,
        "warning_count":           warning_count,
        "info_count":              info_count,
        "unresolved_count":        unresolved_count,
        "unacknowledged_count":    unacknowledged_count,
        "acknowledgment_rate":     acknowledgment_rate,
        "resolution_rate":         resolution_rate,
        "severity_distribution":   severity_distribution,
        "alert_type_distribution": alert_type_distribution,
        "monthly_trend":           monthly_trend,
        "total_patients":          total_patients,
        "total_devices":           total_devices,
    })


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed IoT alerts breakdown.

    Returns:
        recent_alerts        -- list of alert details (last 20)
        alerts_by_device     -- device-level aggregation
        unresolved_alerts    -- filtered list of unresolved alerts
        patient_alert_summary -- per-patient alert statistics
    """
    conn = _conn()
    alerts = _load_all_alerts(conn, patient_id)
    conn.close()

    if not alerts:
        return _json_safe({
            "recent_alerts": [],
            "alerts_by_device": [],
            "unresolved_alerts": [],
            "patient_alert_summary": [],
        })

    # Recent alerts (last 20)
    recent_alerts = []
    for a in alerts[:20]:
        recent_alerts.append({
            "patient_id":   a["patient_id"],
            "alert_type":   a["alert_type"],
            "severity":     a["severity"],
            "device_id":    a["device_id"],
            "acknowledged": a["acknowledged"],
            "resolved":     a["resolved"],
            "timestamp":    a["timestamp"],
        })

    # Alerts by device
    device_stats = defaultdict(lambda: {"device_id": "", "total": 0, "critical": 0, "unresolved": 0})
    for a in alerts:
        did = a["device_id"]
        device_stats[did]["device_id"] = did
        device_stats[did]["total"] += 1
        if a["severity"] == "critical":
            device_stats[did]["critical"] += 1
        if not a["resolved"]:
            device_stats[did]["unresolved"] += 1
    alerts_by_device = sorted(device_stats.values(), key=lambda x: -x["total"])

    # Unresolved alerts
    unresolved_alerts = []
    for a in alerts:
        if not a["resolved"]:
            unresolved_alerts.append({
                "patient_id":   a["patient_id"],
                "alert_type":   a["alert_type"],
                "severity":     a["severity"],
                "device_id":    a["device_id"],
                "gateway_id":   a["gateway_id"],
                "acknowledged": a["acknowledged"],
                "resolved":     a["resolved"],
                "timestamp":    a["timestamp"],
            })

    # Patient alert summary
    patient_stats = defaultdict(lambda: {"patient_id": "", "total": 0, "critical": 0, "resolved": 0})
    for a in alerts:
        pid = a["patient_id"]
        patient_stats[pid]["patient_id"] = pid
        patient_stats[pid]["total"] += 1
        if a["severity"] == "critical":
            patient_stats[pid]["critical"] += 1
        if a["resolved"]:
            patient_stats[pid]["resolved"] += 1
    patient_alert_summary = []
    for ps in sorted(patient_stats.values(), key=lambda x: -x["total"]):
        ps["resolved_pct"] = round(ps["resolved"] / ps["total"] * 100, 1) if ps["total"] else 0.0
        patient_alert_summary.append(ps)

    return _json_safe({
        "recent_alerts":         recent_alerts,
        "alerts_by_device":      alerts_by_device,
        "unresolved_alerts":     unresolved_alerts,
        "patient_alert_summary": patient_alert_summary,
    })


def definitions() -> dict:
    """
    IoT alerting definitions, terminology, and clinical context.

    Returns:
        alert_types      -- descriptions of each alert type
        severity_levels  -- descriptions of each severity
        glossary         -- list of {term, definition} for IoT alerting terminology
        clinical_notes   -- list of clinical context notes about IoT alerts in epilepsy
    """
    return {
        "alert_types": [
            {"type": "low_battery", "description": (
                "The device battery level has dropped below the operational threshold "
                "(typically <20%). Continued monitoring may be interrupted if the battery "
                "is not replaced or recharged within the expected remaining runtime."
            )},
            {"type": "firmware_outdated", "description": (
                "The device is running firmware that is behind the current recommended "
                "version. Outdated firmware may lack critical bug fixes, security patches, "
                "or algorithm improvements for seizure detection accuracy."
            )},
            {"type": "gateway_offline", "description": (
                "The communication gateway responsible for relaying device data to the "
                "cloud has become unreachable. While the gateway is offline, real-time "
                "alerts and data streaming are interrupted for all devices connected "
                "through that gateway."
            )},
            {"type": "signal_lost", "description": (
                "The EEG or physiological sensor signal has been interrupted. This may be "
                "caused by electrode displacement, cable disconnect, or excessive patient "
                "movement. Signal loss gaps create blind spots in seizure detection coverage."
            )},
            {"type": "sos_seizure", "description": (
                "A seizure event has been detected by the on-device algorithm. This alert "
                "is classified as critical and triggers immediate clinical notification. "
                "The detection may be based on accelerometer, EEG, or multimodal fusion "
                "depending on the device type."
            )},
            {"type": "high_impedance", "description": (
                "Electrode impedance has exceeded the acceptable range (typically >20 kOhm). "
                "High impedance degrades signal quality, increases noise, and reduces the "
                "reliability of seizure detection algorithms. Electrode re-application or "
                "gel refresh is recommended."
            )},
        ],
        "severity_levels": [
            {"level": "info", "description": (
                "Informational alert that does not require immediate action. Logged for "
                "operational awareness and trend monitoring. Examples include routine "
                "firmware update availability or scheduled gateway maintenance."
            )},
            {"level": "warning", "description": (
                "Alert requiring attention within the standard operational window "
                "(typically within 4-8 hours). Examples include low battery, high impedance, "
                "or intermittent signal loss. Failure to address may escalate to critical."
            )},
            {"level": "critical", "description": (
                "Alert demanding immediate clinical or technical response. Examples include "
                "seizure detection events (sos_seizure), prolonged gateway outages affecting "
                "multiple patients, or complete signal loss during active monitoring. "
                "Critical alerts trigger push notifications and pager escalation."
            )},
        ],
        "glossary": [
            {
                "term": "IoT Gateway",
                "definition": (
                    "A network relay device that aggregates data from multiple bedside "
                    "or wearable sensors and transmits it to the cloud platform. Each "
                    "gateway serves a ward or patient cluster. Gateway failure disrupts "
                    "monitoring for all connected devices."
                ),
            },
            {
                "term": "Impedance",
                "definition": (
                    "Electrical resistance at the electrode-skin interface, measured in "
                    "kilo-ohms (kOhm). Low impedance (<5 kOhm) ensures high-fidelity EEG "
                    "recording. Values above 20 kOhm degrade signal-to-noise ratio and "
                    "compromise seizure detection accuracy."
                ),
            },
            {
                "term": "Signal-to-Noise Ratio (SNR)",
                "definition": (
                    "The ratio of meaningful EEG signal to background noise. Higher SNR "
                    "improves seizure detection sensitivity. IoT alerts for high impedance "
                    "and signal loss directly impact SNR."
                ),
            },
            {
                "term": "Seizure Detection Confidence",
                "definition": (
                    "A probability score (0.0-1.0) assigned by the on-device seizure "
                    "detection algorithm reflecting certainty that a detected event is a "
                    "true seizure. Thresholds for clinical alerting are typically set at "
                    ">= 0.80 to balance sensitivity and false alarm rate."
                ),
            },
            {
                "term": "Device Uptime",
                "definition": (
                    "The percentage of time a monitoring device is operational and "
                    "transmitting data. Uptime is reduced by battery depletion, firmware "
                    "crashes, signal loss, and gateway outages. Target uptime for clinical "
                    "monitoring is >= 99.5%."
                ),
            },
            {
                "term": "Acknowledgment",
                "definition": (
                    "The act of a clinician or technician marking an alert as seen. "
                    "Acknowledgment does not imply resolution -- it indicates awareness. "
                    "Unacknowledged critical alerts trigger escalation after a configurable "
                    "timeout (default 15 minutes)."
                ),
            },
            {
                "term": "Resolution",
                "definition": (
                    "The act of marking an alert as resolved after the underlying issue "
                    "has been addressed (e.g., battery replaced, electrode reapplied, "
                    "gateway restarted). Resolved alerts are excluded from active "
                    "dashboards but retained for historical analysis."
                ),
            },
            {
                "term": "Alert Fatigue",
                "definition": (
                    "A well-documented clinical phenomenon where excessive non-actionable "
                    "alerts desensitise staff, leading to delayed response to genuine "
                    "critical events. IoT alert systems must balance sensitivity with "
                    "specificity to minimise fatigue."
                ),
            },
            {
                "term": "Over-the-Air (OTA) Update",
                "definition": (
                    "A wireless firmware update pushed to IoT devices without physical "
                    "access. OTA updates address firmware_outdated alerts by deploying "
                    "patches, algorithm improvements, and security fixes remotely."
                ),
            },
            {
                "term": "Edge Computing",
                "definition": (
                    "Processing data on the device or gateway rather than sending raw "
                    "data to the cloud. Seizure detection algorithms running at the edge "
                    "enable real-time sos_seizure alerts with minimal latency."
                ),
            },
        ],
        "clinical_notes": [
            (
                "Continuous EEG monitoring via IoT devices is recommended for patients "
                "with drug-resistant epilepsy (DRE) to capture subclinical seizures and "
                "nocturnal events that may otherwise go undetected (ILAE guidelines, 2022)."
            ),
            (
                "Gateway offline alerts affecting multiple patients simultaneously should "
                "trigger immediate IT response, as loss of monitoring coverage in an "
                "epilepsy monitoring unit (EMU) poses patient safety risks."
            ),
            (
                "High impedance alerts are the most common cause of false-negative "
                "seizure detection. Nursing protocols should include impedance checks "
                "every 4 hours during long-term monitoring (LTM) sessions."
            ),
            (
                "Seizure detection devices with accelerometer-only algorithms have higher "
                "false positive rates compared to multimodal (EEG + accelerometer) systems. "
                "Alert type sos_seizure should be interpreted in the context of device "
                "modality (FDA cleared devices: NightWatch, Embrace2, EpiCare)."
            ),
            (
                "Alert fatigue studies in epilepsy monitoring units report that >60% of "
                "IoT alerts are non-actionable. Severity-based filtering and smart "
                "suppression of repeat low_battery/firmware_outdated alerts during active "
                "seizure monitoring windows can reduce clinician burden by up to 40% "
                "(Beniczky et al., Epilepsia 2021)."
            ),
        ],
        "data_source": (
            "Real iot_alerts table in clinical.db — 50 alert records across "
            "24 devices, 6 gateways, 20 patients. Alert types include device, "
            "gateway, and seizure SOS events with severity, acknowledgment, "
            "and resolution tracking."
        ),
    }
