"""
Neuro AI Ecosystem -- Wearable Devices Dashboard
=================================================
Tracks wearable monitoring devices used in epilepsy care, including
smartwatches, wrist EEG bands, patch monitors, and ring sensors.

Device Types:
  Ankle Sensor      -- Accelerometer-based seizure/fall detector worn on ankle
  Chest Strap       -- ECG and respiration monitor for cardiac-linked seizure detection
  Patch Monitor     -- Adhesive biosensor patch for continuous EEG or EMG recording
  Ring Sensor       -- Finger-worn sensor tracking HRV, SpO2, and sleep patterns
  Smartwatch        -- Consumer wrist device with accelerometer and heart rate sensor
  Wrist EEG Band    -- Medical-grade wrist-worn EEG with seizure detection algorithm

Statuses:
  active            -- Device is online and transmitting data
  charging          -- Device is connected to charger, not actively monitoring
  maintenance       -- Device is offline for firmware update or calibration
  offline           -- Device is unreachable or powered off

All device records are stored in the wearable_devices table of clinical.db.
The fields_json column contains structured device metadata parsed at
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


def _load_all_devices(conn, patient_id: Optional[str] = None) -> list:
    """Load all wearable_devices rows, parsing fields_json into flat dicts."""
    c = conn.cursor()
    if patient_id:
        c.execute(
            "SELECT id, patient_id, fields_json, created_at "
            "FROM wearable_devices WHERE patient_id=? ORDER BY created_at DESC",
            (patient_id,),
        )
    else:
        c.execute(
            "SELECT id, patient_id, fields_json, created_at "
            "FROM wearable_devices ORDER BY created_at DESC"
        )
    rows = []
    for row in c.fetchall():
        fields = _parse_fields(row[2])
        rows.append({
            "id":                       row[0],
            "patient_id":               row[1],
            "device_id":                fields.get("device_id", "unknown"),
            "device_type":              fields.get("device_type", "unknown"),
            "brand":                    fields.get("brand", "unknown"),
            "firmware_version":         fields.get("firmware_version", "unknown"),
            "registered_date":          fields.get("registered_date", ""),
            "battery_level":            fields.get("battery_level", 0),
            "connectivity":             fields.get("connectivity", "unknown"),
            "status":                   fields.get("status", "unknown"),
            "last_sync":                fields.get("last_sync", ""),
            "seizure_detection_enabled": fields.get("seizure_detection_enabled", False),
            "fall_detection_enabled":   fields.get("fall_detection_enabled", False),
            "heart_rate_monitoring":    fields.get("heart_rate_monitoring", False),
            "sleep_tracking":           fields.get("sleep_tracking", False),
            "stress_tracking":          fields.get("stress_tracking", False),
            "created_at":               row[3],
        })
    return rows


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def overview(patient_id: Optional[str] = None) -> dict:
    """
    Wearable devices overview dashboard data.

    Returns:
        total_devices              -- total device records
        active_count               -- devices with status=active
        offline_count              -- devices with status=offline
        charging_count             -- devices with status=charging
        maintenance_count          -- devices with status=maintenance
        avg_battery_level          -- mean battery level across all devices
        low_battery_count          -- devices with battery < 20%
        seizure_detection_rate     -- percentage of devices with seizure detection enabled
        fall_detection_rate        -- percentage of devices with fall detection enabled
        total_patients             -- distinct patients with devices
        status_distribution        -- list of {name, value} for pie chart
        device_type_distribution   -- list of {name, value} for bar chart
        brand_distribution         -- list of {name, value} for bar chart
        connectivity_distribution  -- list of {name, value} for pie chart
        feature_coverage           -- list of {feature, enabled, total, pct} for bar chart
    """
    conn = _conn()
    devices = _load_all_devices(conn, patient_id)
    conn.close()

    if not devices:
        return _json_safe({
            "total_devices": 0,
            "active_count": 0,
            "offline_count": 0,
            "charging_count": 0,
            "maintenance_count": 0,
            "avg_battery_level": 0.0,
            "low_battery_count": 0,
            "seizure_detection_rate": 0.0,
            "fall_detection_rate": 0.0,
            "total_patients": 0,
            "status_distribution": [],
            "device_type_distribution": [],
            "brand_distribution": [],
            "connectivity_distribution": [],
            "feature_coverage": [],
        })

    total_devices = len(devices)
    active_count = sum(1 for d in devices if d["status"] == "active")
    offline_count = sum(1 for d in devices if d["status"] == "offline")
    charging_count = sum(1 for d in devices if d["status"] == "charging")
    maintenance_count = sum(1 for d in devices if d["status"] == "maintenance")

    # Battery stats
    battery_levels = [d["battery_level"] for d in devices if d["battery_level"] is not None]
    avg_battery_level = round(sum(battery_levels) / len(battery_levels), 1) if battery_levels else 0.0
    low_battery_count = sum(1 for b in battery_levels if b < 20)

    # Feature detection rates
    seizure_enabled = sum(1 for d in devices if d["seizure_detection_enabled"])
    fall_enabled = sum(1 for d in devices if d["fall_detection_enabled"])
    seizure_detection_rate = round(seizure_enabled / total_devices * 100, 1) if total_devices else 0.0
    fall_detection_rate = round(fall_enabled / total_devices * 100, 1) if total_devices else 0.0

    # Unique patients
    total_patients = len({d["patient_id"] for d in devices if d["patient_id"]})

    # Status distribution (pie chart)
    status_counts = defaultdict(int)
    for d in devices:
        status_counts[d["status"]] += 1
    status_distribution = [
        {"name": s, "value": cnt}
        for s, cnt in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    # Device type distribution (bar chart)
    type_counts = defaultdict(int)
    for d in devices:
        type_counts[d["device_type"]] += 1
    device_type_distribution = [
        {"name": dt, "value": cnt}
        for dt, cnt in sorted(type_counts.items(), key=lambda x: -x[1])
    ]

    # Brand distribution (bar chart)
    brand_counts = defaultdict(int)
    for d in devices:
        brand_counts[d["brand"]] += 1
    brand_distribution = [
        {"name": b, "value": cnt}
        for b, cnt in sorted(brand_counts.items(), key=lambda x: -x[1])
    ]

    # Connectivity distribution (pie chart)
    conn_counts = defaultdict(int)
    for d in devices:
        conn_counts[d["connectivity"]] += 1
    connectivity_distribution = [
        {"name": c, "value": cnt}
        for c, cnt in sorted(conn_counts.items(), key=lambda x: -x[1])
    ]

    # Feature coverage (bar chart)
    features = [
        ("seizure_detection_enabled", "Seizure Detection"),
        ("fall_detection_enabled", "Fall Detection"),
        ("heart_rate_monitoring", "Heart Rate Monitoring"),
        ("sleep_tracking", "Sleep Tracking"),
        ("stress_tracking", "Stress Tracking"),
    ]
    feature_coverage = []
    for field, label in features:
        enabled = sum(1 for d in devices if d[field])
        pct = round(enabled / total_devices * 100, 1) if total_devices else 0.0
        feature_coverage.append({
            "feature": label,
            "enabled": enabled,
            "total": total_devices,
            "pct": pct,
        })

    return _json_safe({
        "total_devices":             total_devices,
        "active_count":              active_count,
        "offline_count":             offline_count,
        "charging_count":            charging_count,
        "maintenance_count":         maintenance_count,
        "avg_battery_level":         avg_battery_level,
        "low_battery_count":         low_battery_count,
        "seizure_detection_rate":    seizure_detection_rate,
        "fall_detection_rate":       fall_detection_rate,
        "total_patients":            total_patients,
        "status_distribution":       status_distribution,
        "device_type_distribution":  device_type_distribution,
        "brand_distribution":        brand_distribution,
        "connectivity_distribution": connectivity_distribution,
        "feature_coverage":          feature_coverage,
    })


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed wearable devices breakdown.

    Returns:
        all_devices           -- list of all device details
        low_battery_devices   -- devices with battery < 20%
        offline_devices       -- devices with status=offline
        patient_device_summary -- per-patient device statistics
        devices_by_brand      -- brand-level aggregation
    """
    conn = _conn()
    devices = _load_all_devices(conn, patient_id)
    conn.close()

    if not devices:
        return _json_safe({
            "all_devices": [],
            "low_battery_devices": [],
            "offline_devices": [],
            "patient_device_summary": [],
            "devices_by_brand": [],
        })

    # All devices detail list
    all_devices = []
    for d in devices:
        all_devices.append({
            "patient_id":        d["patient_id"],
            "device_id":         d["device_id"],
            "device_type":       d["device_type"],
            "brand":             d["brand"],
            "status":            d["status"],
            "battery_level":     d["battery_level"],
            "connectivity":      d["connectivity"],
            "last_sync":         d["last_sync"],
            "seizure_detection": d["seizure_detection_enabled"],
            "fall_detection":    d["fall_detection_enabled"],
        })

    # Low battery devices (battery < 20)
    low_battery_devices = []
    for d in devices:
        if d["battery_level"] is not None and d["battery_level"] < 20:
            low_battery_devices.append({
                "patient_id":    d["patient_id"],
                "device_id":     d["device_id"],
                "device_type":   d["device_type"],
                "brand":         d["brand"],
                "battery_level": d["battery_level"],
                "status":        d["status"],
            })
    low_battery_devices.sort(key=lambda x: x["battery_level"])

    # Offline devices
    offline_devices = []
    for d in devices:
        if d["status"] == "offline":
            offline_devices.append({
                "patient_id":    d["patient_id"],
                "device_id":     d["device_id"],
                "device_type":   d["device_type"],
                "brand":         d["brand"],
                "battery_level": d["battery_level"],
                "last_sync":     d["last_sync"],
            })

    # Patient device summary
    patient_stats = defaultdict(lambda: {
        "patient_id": "", "device_count": 0, "device_types": set(),
        "battery_sum": 0.0, "battery_n": 0, "any_offline": False,
    })
    for d in devices:
        pid = d["patient_id"]
        patient_stats[pid]["patient_id"] = pid
        patient_stats[pid]["device_count"] += 1
        patient_stats[pid]["device_types"].add(d["device_type"])
        if d["battery_level"] is not None:
            patient_stats[pid]["battery_sum"] += d["battery_level"]
            patient_stats[pid]["battery_n"] += 1
        if d["status"] == "offline":
            patient_stats[pid]["any_offline"] = True
    patient_device_summary = []
    for ps in sorted(patient_stats.values(), key=lambda x: -x["device_count"]):
        avg_bat = round(ps["battery_sum"] / ps["battery_n"], 1) if ps["battery_n"] else 0.0
        patient_device_summary.append({
            "patient_id":   ps["patient_id"],
            "device_count": ps["device_count"],
            "device_types": sorted(ps["device_types"]),
            "avg_battery":  avg_bat,
            "any_offline":  ps["any_offline"],
        })

    # Devices by brand
    brand_stats = defaultdict(lambda: {
        "brand": "", "total": 0, "active": 0,
        "battery_sum": 0.0, "battery_n": 0,
    })
    for d in devices:
        b = d["brand"]
        brand_stats[b]["brand"] = b
        brand_stats[b]["total"] += 1
        if d["status"] == "active":
            brand_stats[b]["active"] += 1
        if d["battery_level"] is not None:
            brand_stats[b]["battery_sum"] += d["battery_level"]
            brand_stats[b]["battery_n"] += 1
    devices_by_brand = []
    for bs in sorted(brand_stats.values(), key=lambda x: -x["total"]):
        avg_bat = round(bs["battery_sum"] / bs["battery_n"], 1) if bs["battery_n"] else 0.0
        devices_by_brand.append({
            "brand":       bs["brand"],
            "total":       bs["total"],
            "active":      bs["active"],
            "avg_battery": avg_bat,
        })

    return _json_safe({
        "all_devices":           all_devices,
        "low_battery_devices":   low_battery_devices,
        "offline_devices":       offline_devices,
        "patient_device_summary": patient_device_summary,
        "devices_by_brand":      devices_by_brand,
    })


def definitions() -> dict:
    """
    Wearable device definitions, terminology, and clinical context.

    Returns:
        device_types              -- descriptions of each device type
        status_descriptions       -- descriptions of each device status
        connectivity_descriptions -- descriptions of each connectivity type
        feature_descriptions      -- descriptions of the 5 boolean features
        glossary                  -- list of {term, definition} for wearable monitoring terminology
        clinical_notes            -- list of clinical context notes about wearables in epilepsy
    """
    return {
        "device_types": [
            {"type": "Ankle Sensor", "description": (
                "Accelerometer-based sensor worn on the ankle, optimised for detecting "
                "generalised tonic-clonic seizures (GTCS) through rhythmic limb movement "
                "patterns. Ankle placement offers higher sensitivity for lower-extremity "
                "motor seizures compared to wrist-worn devices (Beniczky et al., 2018)."
            )},
            {"type": "Chest Strap", "description": (
                "Thoracic band with embedded ECG and respiratory sensors. Monitors heart "
                "rate variability (HRV) and respiration rate, both of which show "
                "characteristic peri-ictal changes. Particularly valuable for detecting "
                "seizure-related cardiac arrhythmias and postictal respiratory suppression "
                "linked to SUDEP risk."
            )},
            {"type": "Patch Monitor", "description": (
                "Adhesive biosensor patch applied to the skin surface for continuous "
                "ambulatory EEG or electromyography (EMG) recording. Patches offer "
                "multi-day recording with minimal patient burden, enabling long-term "
                "monitoring outside the epilepsy monitoring unit (EMU)."
            )},
            {"type": "Ring Sensor", "description": (
                "Finger-worn sensor measuring photoplethysmography (PPG), skin temperature, "
                "and SpO2. Tracks heart rate variability and sleep architecture. Compact "
                "form factor improves long-term compliance. Clinically relevant for "
                "detecting nocturnal autonomic changes associated with sleep-related seizures."
            )},
            {"type": "Smartwatch", "description": (
                "Consumer wrist-worn device with accelerometer, gyroscope, and optical "
                "heart rate sensor. When paired with validated seizure detection algorithms "
                "(e.g., Embrace2/EmbracePlus by Empatica), smartwatches provide FDA-cleared "
                "tonic-clonic seizure detection with automated caregiver alerts."
            )},
            {"type": "Wrist EEG Band", "description": (
                "Medical-grade wrist-worn device with dry EEG electrodes and on-board "
                "seizure detection algorithm. Captures electrodermal activity (EDA) and "
                "peripheral EEG signals. Offers higher seizure detection specificity than "
                "accelerometer-only devices by combining motion and neurophysiological data."
            )},
        ],
        "status_descriptions": [
            {"status": "active", "description": (
                "Device is powered on, connected to its gateway or paired smartphone, and "
                "actively transmitting physiological data. Seizure detection and alerting "
                "features are operational."
            )},
            {"status": "charging", "description": (
                "Device is connected to its charger and not currently monitoring the "
                "patient. During charging periods, seizure detection coverage is interrupted. "
                "Charging sessions should be scheduled during low-risk windows when possible."
            )},
            {"status": "maintenance", "description": (
                "Device has been taken offline for firmware update, sensor calibration, or "
                "hardware inspection. Maintenance status is manually set by clinical "
                "engineering staff and should be cleared once the device passes quality checks."
            )},
            {"status": "offline", "description": (
                "Device is unreachable -- either powered off, out of wireless range, or "
                "experiencing a hardware fault. Offline devices require immediate attention "
                "as seizure monitoring coverage is lost. Prolonged offline status triggers "
                "escalation alerts."
            )},
        ],
        "connectivity_descriptions": [
            {"type": "BLE", "description": (
                "Bluetooth Low Energy -- short-range, low-power wireless protocol. "
                "Requires a paired smartphone or bedside gateway within approximately "
                "10 metres. Most common connectivity for wearable seizure detectors."
            )},
            {"type": "BLE+WiFi", "description": (
                "Dual-mode connectivity combining Bluetooth Low Energy for device pairing "
                "and WiFi for high-bandwidth data upload. Enables local seizure alerting "
                "via BLE while streaming continuous EEG data over WiFi."
            )},
            {"type": "Cellular", "description": (
                "Built-in cellular modem (LTE-M or NB-IoT) enabling direct cloud "
                "connectivity without a smartphone or gateway. Provides monitoring "
                "coverage outside the home, including during commutes and outdoor activity."
            )},
            {"type": "WiFi", "description": (
                "Standard WiFi connectivity for data transmission. Higher bandwidth than "
                "BLE, suitable for devices streaming continuous waveform data. Range "
                "limited to WiFi network coverage area."
            )},
        ],
        "feature_descriptions": [
            {"feature": "seizure_detection_enabled", "description": (
                "On-device algorithm for real-time seizure detection, typically targeting "
                "generalised tonic-clonic seizures (GTCS). When enabled, the device "
                "analyses accelerometer, EDA, or EEG signals and triggers automated alerts "
                "to caregivers and clinicians upon seizure detection."
            )},
            {"feature": "fall_detection_enabled", "description": (
                "Accelerometer and gyroscope-based fall detection algorithm. Detects sudden "
                "postural changes indicative of seizure-related falls. Particularly important "
                "for patients with atonic (drop) seizures who are at high risk of "
                "fall-related injuries."
            )},
            {"feature": "heart_rate_monitoring", "description": (
                "Continuous optical or ECG-based heart rate tracking. Monitors for peri-ictal "
                "tachycardia, bradycardia, and heart rate variability changes that commonly "
                "accompany seizures. HRV depression is a known biomarker for SUDEP risk."
            )},
            {"feature": "sleep_tracking", "description": (
                "Actigraphy and heart rate-based sleep stage classification. Tracks sleep "
                "onset, wake episodes, REM/NREM staging, and total sleep time. Sleep "
                "disruption is both a seizure trigger and consequence; 70-80% of nocturnal "
                "seizures occur during NREM sleep stages."
            )},
            {"feature": "stress_tracking", "description": (
                "Electrodermal activity (EDA) and HRV-based stress level estimation. "
                "Psychosocial stress is a well-documented seizure precipitant. Continuous "
                "stress monitoring enables identification of individual stress-seizure "
                "patterns and supports lifestyle modification counselling."
            )},
        ],
        "glossary": [
            {
                "term": "Electrodermal Activity (EDA)",
                "definition": (
                    "A measure of skin conductance reflecting sympathetic nervous system "
                    "arousal. EDA surges are observed during generalised tonic-clonic "
                    "seizures and are used by devices such as the Empatica Embrace2 as a "
                    "primary seizure detection signal."
                ),
            },
            {
                "term": "Heart Rate Variability (HRV)",
                "definition": (
                    "The variation in time intervals between consecutive heartbeats. "
                    "Reduced HRV is associated with autonomic dysfunction in epilepsy "
                    "and is considered a biomarker for increased SUDEP risk. Wearable "
                    "devices track HRV continuously to detect peri-ictal autonomic changes."
                ),
            },
            {
                "term": "SUDEP",
                "definition": (
                    "Sudden Unexpected Death in Epilepsy -- the sudden, unexpected death "
                    "of a person with epilepsy without a clear structural or toxicological "
                    "cause. Wearable seizure detectors aim to reduce SUDEP risk by enabling "
                    "rapid caregiver intervention during nocturnal tonic-clonic seizures."
                ),
            },
            {
                "term": "Photoplethysmography (PPG)",
                "definition": (
                    "An optical technique used in wearable devices to measure blood volume "
                    "changes in the microvascular bed of tissue. PPG sensors in smartwatches "
                    "and rings provide continuous heart rate and SpO2 estimation."
                ),
            },
            {
                "term": "Actigraphy",
                "definition": (
                    "Wrist or ankle accelerometer-based monitoring of motor activity "
                    "patterns over extended periods. Used in epilepsy for seizure "
                    "detection, sleep-wake classification, and circadian rhythm assessment. "
                    "Multi-day actigraphy can identify seizure clustering patterns."
                ),
            },
            {
                "term": "Tonic-Clonic Seizure",
                "definition": (
                    "A generalised seizure involving initial muscle stiffening (tonic phase) "
                    "followed by rhythmic jerking (clonic phase). This seizure type produces "
                    "the most distinctive accelerometer and EDA signatures and is the primary "
                    "target of most wearable seizure detection algorithms."
                ),
            },
            {
                "term": "Compliance Rate",
                "definition": (
                    "The percentage of prescribed monitoring time during which the patient "
                    "actually wears the device. Low compliance reduces seizure detection "
                    "coverage and data completeness. Device comfort, battery life, and "
                    "social acceptability are major compliance determinants."
                ),
            },
            {
                "term": "False Alarm Rate (FAR)",
                "definition": (
                    "The number of false seizure detections per unit time, typically "
                    "expressed as false alarms per day. High FAR leads to alert fatigue "
                    "and device abandonment. FDA-cleared devices target FAR < 1 per day."
                ),
            },
            {
                "term": "Sensitivity",
                "definition": (
                    "The proportion of true seizures correctly detected by the wearable "
                    "device algorithm, expressed as a percentage. FDA-cleared seizure "
                    "detection wearables for GTCS typically achieve sensitivity of 90-100% "
                    "in clinical validation studies."
                ),
            },
            {
                "term": "Multimodal Fusion",
                "definition": (
                    "The algorithmic combination of multiple physiological signals "
                    "(e.g., accelerometer + EDA + HRV + EEG) to improve seizure detection "
                    "accuracy. Multimodal approaches reduce false alarm rates compared to "
                    "single-signal detectors while maintaining high sensitivity."
                ),
            },
        ],
        "clinical_notes": [
            (
                "FDA-cleared wearable seizure detection devices (Embrace2/EmbracePlus by "
                "Empatica, NightWatch) have demonstrated >90% sensitivity for generalised "
                "tonic-clonic seizures in prospective clinical studies, with automated "
                "caregiver alerting reducing seizure-related injury risk (Onorati et al., "
                "Epilepsia 2017; van Andel et al., Neurology 2017)."
            ),
            (
                "Wearable device compliance in epilepsy outpatient populations averages "
                "60-80% of prescribed wear time. Key barriers include device discomfort, "
                "skin irritation from adhesive patches, social stigma, and battery life "
                "limitations requiring daily charging (Johansson et al., Epilepsy & "
                "Behavior 2018)."
            ),
            (
                "Nocturnal seizure monitoring is a primary clinical use case for wearable "
                "devices, as unwitnessed nocturnal tonic-clonic seizures are the strongest "
                "risk factor for SUDEP. Devices capable of detecting seizures during sleep "
                "and alerting co-sleeping caregivers can significantly reduce SUDEP risk "
                "(Ryvlin et al., Lancet Neurology 2013)."
            ),
            (
                "Heart rate variability (HRV) derived from wearable devices shows promise "
                "as a biomarker for seizure prediction. Pre-ictal HRV changes have been "
                "observed 30-60 minutes before seizure onset in some patients, potentially "
                "enabling prospective seizure warnings (Jeppesen et al., Epilepsia 2019)."
            ),
            (
                "Multi-device monitoring (combining wrist, ankle, and chest sensors) "
                "improves seizure detection coverage across seizure types beyond GTCS, "
                "including focal seizures with impaired awareness. However, multi-device "
                "setups increase patient burden and require careful balancing of detection "
                "benefit versus compliance impact (Ulate-Campos et al., Seizure 2016)."
            ),
        ],
        "data_source": (
            "Real wearable_devices table in clinical.db -- 30 device records across "
            "30 patients. Device types include smartwatches, wrist EEG bands, patch "
            "monitors, ring sensors, chest straps, and ankle sensors with status, "
            "battery, connectivity, and feature tracking."
        ),
    }
