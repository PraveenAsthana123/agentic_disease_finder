"""IoT Devices Dashboard — Emotiv + IoT + Mobile device fleet registry,
connectivity model, online/offline strategy, and alert pipeline, from
config/iot_devices.json."""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: total devices, status distribution, type breakdown, connectivity modes."""
    cfg = _load('iot_devices.json')
    if not cfg:
        return {"available": False, "note": "iot_devices.json missing"}

    devices = cfg.get('devices', [])
    total = len(devices)

    status_counts = Counter(d.get('status', 'unknown') for d in devices)
    type_counts = Counter(d.get('type', 'unknown') for d in devices)

    # Count connectivity modes across all devices
    mode_counts = Counter()
    for d in devices:
        for m in d.get('modes', []):
            mode_counts[m] += 1

    # Data streams across fleet
    all_data = []
    for d in devices:
        all_data.extend(d.get('data', []))
    unique_data_streams = len(set(all_data))

    # Devices with alerts
    alert_devices = sum(1 for d in devices if d.get('alert'))

    status_distribution = [
        {"name": s.replace('_', ' ').title(), "value": c}
        for s, c in sorted(status_counts.items())
    ]

    type_distribution = [
        {"name": t, "value": c}
        for t, c in sorted(type_counts.items())
    ]

    mode_distribution = [
        {"name": m.title(), "value": c}
        for m, c in sorted(mode_counts.items())
    ]

    device_summary = []
    for d in devices:
        device_summary.append({
            "id": d.get('id'),
            "name": d.get('name'),
            "type": d.get('type'),
            "channels": d.get('channels'),
            "modes": d.get('modes', []),
            "data_streams": len(d.get('data', [])),
            "has_alert": bool(d.get('alert')),
            "status": d.get('status'),
        })

    connectivity = cfg.get('connectivity_model', {})
    summary = cfg.get('summary', {})

    return {
        "available": True,
        "summary": {
            "total_devices": total,
            "built": status_counts.get('built', 0),
            "partial": status_counts.get('partial', 0),
            "planned": status_counts.get('planned', 0),
            "unique_data_streams": unique_data_streams,
            "alert_capable": alert_devices,
            "device_types": len(type_counts),
            "honest_note": summary.get('honest_note', ''),
        },
        "status_distribution": status_distribution,
        "type_distribution": type_distribution,
        "mode_distribution": mode_distribution,
        "device_summary": device_summary,
        "connectivity_model": connectivity,
        "alert_pipeline": cfg.get('alert_pipeline', ''),
    }


def breakdown():
    """Per-device detail: data streams, connectivity, alerts, online/offline behavior."""
    cfg = _load('iot_devices.json')
    if not cfg:
        return {"available": False}

    devices = cfg.get('devices', [])
    by_type = {}
    for d in devices:
        dtype = d.get('type', 'unknown')
        if dtype not in by_type:
            by_type[dtype] = []
        by_type[dtype].append({
            "id": d.get('id'),
            "name": d.get('name'),
            "channels": d.get('channels'),
            "modes": d.get('modes', []),
            "data": d.get('data', []),
            "online": d.get('online', ''),
            "offline": d.get('offline', ''),
            "alert": d.get('alert', ''),
            "status": d.get('status'),
        })

    offline_strategy = cfg.get('offline_strategy', {})

    # Cross-reference: device × mode matrix
    all_modes = sorted(set(m for d in devices for m in d.get('modes', [])))
    mode_matrix = []
    for d in devices:
        row = {"device": d.get('name'), "id": d.get('id'), "status": d.get('status')}
        for m in all_modes:
            row[m] = m in d.get('modes', [])
        mode_matrix.append(row)

    # Data stream matrix: which device has which data
    all_data = sorted(set(s for d in devices for s in d.get('data', [])))
    data_matrix = []
    for d in devices:
        row = {"device": d.get('name'), "id": d.get('id'), "status": d.get('status')}
        for s in all_data:
            row[s] = s in d.get('data', [])
        data_matrix.append(row)

    return {
        "available": True,
        "by_type": by_type,
        "mode_matrix": {"modes": all_modes, "rows": mode_matrix},
        "data_matrix": {"streams": all_data, "rows": data_matrix},
        "offline_strategy": offline_strategy,
    }


def definitions():
    """Definitions: connectivity model, device types, status legend, glossary."""
    cfg = _load('iot_devices.json')
    connectivity = cfg.get('connectivity_model', {}) if cfg else {}
    offline = cfg.get('offline_strategy', {}) if cfg else {}

    return {
        "connectivity_model": {
            k: {"label": k.title(), "description": v}
            for k, v in connectivity.items()
        },
        "status_legend": [
            {"status": "built", "description": "Fully implemented and tested"},
            {"status": "partial", "description": "Connectivity model designed; integration simulated, not live hardware"},
            {"status": "planned", "description": "Spec defined; implementation not started"},
        ],
        "device_types": [
            {"type": "EEG headset", "description": "Multi-channel EEG acquisition device (Emotiv family)"},
            {"type": "EEG cap", "description": "High-density research-grade EEG cap"},
            {"type": "wearable", "description": "Body-worn sensor (watch, band, patch) for physiological monitoring"},
            {"type": "mobile (online)", "description": "Smartphone app for patient/caregiver interaction"},
            {"type": "gateway", "description": "Home hub aggregating all device streams for cloud forwarding"},
        ],
        "offline_strategy": offline,
        "glossary": [
            {"term": "BLE", "definition": "Bluetooth Low Energy — wireless protocol for wearable/EEG streaming"},
            {"term": "EDA", "definition": "Electrodermal Activity — skin conductance, used in seizure detection bands"},
            {"term": "HRV", "definition": "Heart Rate Variability — autonomic nervous system marker from wearables"},
            {"term": "SpO2", "definition": "Blood oxygen saturation measured by pulse oximetry"},
            {"term": "EDF", "definition": "European Data Format — standard file format for EEG recordings"},
            {"term": "MQTT", "definition": "Message Queuing Telemetry Transport — IoT messaging protocol"},
            {"term": "LSL", "definition": "Lab Streaming Layer — real-time data streaming for EEG/biosignals"},
            {"term": "FCM", "definition": "Firebase Cloud Messaging — push notification service for mobile apps"},
            {"term": "SOS", "definition": "Emergency alert triggered by seizure detection → caregiver notification"},
            {"term": "Edge inference", "definition": "Running seizure model on-device without cloud, enabling offline alerts"},
        ],
        "clinical_notes": [
            "Emotiv devices are consumer-grade EEG; clinical-grade recording uses medical EEG systems",
            "Seizure detection wristbands detect convulsive seizures only (not absence/focal)",
            "Offline-first design ensures zero data loss during connectivity gaps",
            "All device data is encrypted in transit (TLS) and at rest (AES-256)",
        ],
        "references": [
            "Emotiv SDK documentation (Cortex API v2)",
            "IEEE 11073 — Point-of-care medical device communication",
            "FDA guidance on mobile medical applications (2019)",
            "IEC 62304 — Medical device software lifecycle",
        ],
    }
