"""IoT Engineer Dashboard — wearable EEG device fleet management, gateway health,
streaming infrastructure monitoring, and alert/SOS tracking for remote epilepsy
monitoring.

Populates and reads from:
  - iot_devices    (wearable EEG devices: type, firmware, battery, signal, status)
  - iot_gateways   (network gateways: location, uptime, connected devices)
  - iot_alerts     (SOS seizure alerts, system alerts, severity, resolution)

Uses real patient_ids from the patients table.
IEC 62304 / IEC 80001 / HIPAA device standards applied throughout.
"""

import json
import os
import random
import sqlite3
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

DEVICE_TYPES = ['wearable_eeg', 'ambulatory_eeg', 'home_monitor', 'implantable_rns']
DEVICE_STATUSES = ['online', 'offline', 'maintenance']
GATEWAY_LOCATIONS = ['Ward A', 'Ward B', 'ICU', 'EMU', 'Outpatient Clinic', 'Remote Hub']
GATEWAY_STATUSES = ['online', 'offline']

ALERT_TYPES = [
    'sos_seizure', 'low_battery', 'signal_lost',
    'high_impedance', 'gateway_offline', 'firmware_outdated',
]
ALERT_SEVERITIES = ['critical', 'warning', 'info']

FIRMWARE_VERSIONS = ['1.0.3', '1.1.0', '1.2.1', '2.0.0', '2.0.1', '2.1.0']
GATEWAY_FIRMWARE = ['3.0.0', '3.1.2', '3.2.0', '4.0.0']

random.seed(99)


def _db_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = _db_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
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


def _avg(values):
    return round(sum(values) / len(values), 2) if values else 0


def _generate_device_data(device_id, patient_id, rng):
    """Generate realistic wearable EEG device data."""
    device_type = rng.choice(DEVICE_TYPES)
    firmware = rng.choice(FIRMWARE_VERSIONS)
    battery_pct = round(rng.gauss(65, 25), 1)
    battery_pct = max(2.0, min(100.0, battery_pct))
    signal_strength_dbm = round(rng.gauss(-55, 18), 1)
    signal_strength_dbm = max(-100.0, min(-20.0, signal_strength_dbm))
    status = rng.choices(
        DEVICE_STATUSES,
        weights=[0.72, 0.18, 0.10]
    )[0]
    # Simulate last_seen relative to status
    if status == 'online':
        last_seen = f'2025-{rng.randint(6, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 22):02d}:{rng.randint(0, 59):02d}:00'
    elif status == 'offline':
        last_seen = f'2025-{rng.randint(1, 5):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00'
    else:
        last_seen = f'2025-{rng.randint(3, 8):02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 17):02d}:{rng.randint(0, 59):02d}:00'
    location = rng.choice(GATEWAY_LOCATIONS)
    latency_ms = round(rng.gauss(45, 20), 1)
    latency_ms = max(5.0, min(200.0, latency_ms))
    return {
        'device_id': device_id,
        'device_type': device_type,
        'firmware_version': firmware,
        'battery_pct': battery_pct,
        'signal_strength_dbm': signal_strength_dbm,
        'status': status,
        'patient_id': patient_id,
        'last_seen': last_seen,
        'location': location,
        'latency_ms': latency_ms,
    }


def _generate_gateway_data(gateway_id, location, rng):
    """Generate realistic gateway data."""
    status = rng.choices(GATEWAY_STATUSES, weights=[0.85, 0.15])[0]
    uptime_pct = round(rng.gauss(96, 4), 1)
    uptime_pct = max(70.0, min(100.0, uptime_pct))
    connected_devices = rng.randint(2, 10)
    last_heartbeat = f'2025-{rng.randint(6, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(8, 22):02d}:{rng.randint(0, 59):02d}:00'
    firmware = rng.choice(GATEWAY_FIRMWARE)
    return {
        'gateway_id': gateway_id,
        'location': location,
        'status': status,
        'uptime_pct': uptime_pct,
        'connected_devices': connected_devices,
        'last_heartbeat': last_heartbeat,
        'firmware_version': firmware,
    }


def _generate_alert_data(alert_id, device_ids, gateway_ids, patient_ids, rng):
    """Generate realistic IoT alert data."""
    alert_type = rng.choice(ALERT_TYPES)
    # Assign severity based on alert type
    if alert_type in ('sos_seizure', 'gateway_offline'):
        severity = rng.choices(ALERT_SEVERITIES, weights=[0.7, 0.25, 0.05])[0]
    elif alert_type in ('low_battery', 'signal_lost', 'high_impedance'):
        severity = rng.choices(ALERT_SEVERITIES, weights=[0.2, 0.6, 0.2])[0]
    else:
        severity = rng.choices(ALERT_SEVERITIES, weights=[0.1, 0.3, 0.6])[0]

    device_id = rng.choice(device_ids) if alert_type != 'gateway_offline' else None
    gateway_id = rng.choice(gateway_ids) if alert_type == 'gateway_offline' else None
    patient_id = rng.choice(patient_ids) if device_id else None
    acknowledged = rng.random() < 0.75
    resolved = acknowledged and rng.random() < 0.80
    timestamp = f'2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00'
    return {
        'alert_type': alert_type,
        'severity': severity,
        'device_id': device_id,
        'gateway_id': gateway_id,
        'patient_id': patient_id,
        'acknowledged': acknowledged,
        'resolved': resolved,
        'timestamp': timestamp,
    }


def _populate_if_empty():
    """Populate the three IoT Engineer tables with realistic data if empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        # Create tables if not exist
        conn.execute('''CREATE TABLE IF NOT EXISTS iot_devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            fields_json TEXT,
            created_at TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS iot_gateways (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            fields_json TEXT,
            created_at TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS iot_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            fields_json TEXT,
            created_at TEXT
        )''')
        conn.commit()

        count = conn.execute('SELECT COUNT(*) FROM iot_devices').fetchone()[0]
        if count > 0:
            return  # already populated

        # Get real patient IDs
        patients = conn.execute(
            'SELECT patient_id, name, age, gender FROM patients'
        ).fetchall()
        patients = [dict(p) for p in patients]

        epat = [p for p in patients if p['patient_id'].startswith('EPAT')]
        others = [p for p in patients if not p['patient_id'].startswith('EPAT')]
        ordered = epat + others
        target_patients = ordered[:30]
        patient_ids = [p['patient_id'] for p in target_patients]

        rng = random.Random(99)  # deterministic seed

        # 1. Generate ~35 devices
        device_ids = []
        for i in range(35):
            device_id = f'DEV-{i + 1:03d}'
            device_ids.append(device_id)
            # Assign to a patient (cycle through patient list)
            pid = patient_ids[i % len(patient_ids)]
            dev = _generate_device_data(device_id, pid, rng)
            conn.execute(
                'INSERT INTO iot_devices (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                (pid, json.dumps(dev))
            )

        # 2. Generate 6 gateways
        gateway_ids = []
        for i, location in enumerate(GATEWAY_LOCATIONS):
            gateway_id = f'GW-{i + 1:03d}'
            gateway_ids.append(gateway_id)
            gw = _generate_gateway_data(gateway_id, location, rng)
            conn.execute(
                'INSERT INTO iot_gateways (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                (gateway_id, json.dumps(gw))
            )

        # 3. Generate ~50 alerts
        for i in range(50):
            alert = _generate_alert_data(i + 1, device_ids, gateway_ids, patient_ids, rng)
            alert_pid = alert.get('patient_id') or alert.get('gateway_id', '')
            conn.execute(
                'INSERT INTO iot_alerts (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                (alert_pid, json.dumps(alert))
            )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def _load_all_data():
    """Load all three IoT tables and return structured data."""
    _populate_if_empty()

    devices = []
    for r in _db_query('SELECT patient_id, fields_json FROM iot_devices'):
        devices.append(_safe_json(r['fields_json']))

    gateways = []
    for r in _db_query('SELECT patient_id, fields_json FROM iot_gateways'):
        gateways.append(_safe_json(r['fields_json']))

    alerts = []
    for r in _db_query('SELECT patient_id, fields_json FROM iot_alerts'):
        alerts.append(_safe_json(r['fields_json']))

    patients = {r['patient_id']: r for r in _db_query('SELECT * FROM patients')}

    return devices, gateways, alerts, patients


def overview():
    """Return KPI cards + chart data for the IoT Engineer overview tab."""
    devices, gateways, alerts, patients = _load_all_data()

    total_devices = len(devices)

    # ── Devices online
    devices_online = sum(1 for d in devices if d.get('status') == 'online')
    devices_online_pct = round(100 * devices_online / total_devices, 1) if total_devices else 0

    # ── Gateway uptime (avg %)
    gateway_uptimes = [g.get('uptime_pct', 0) for g in gateways]
    avg_gateway_uptime = _avg(gateway_uptimes)

    # ── Stream latency (ms) — avg latency across devices
    latencies = [d.get('latency_ms', 0) for d in devices if d.get('latency_ms')]
    avg_latency_ms = _avg(latencies)

    # ── SOS alerts fired
    sos_alerts = sum(1 for a in alerts if a.get('alert_type') == 'sos_seizure')

    # ── Battery/signal low
    battery_low = sum(1 for d in devices if d.get('battery_pct', 100) < 20)
    signal_low = sum(1 for d in devices if d.get('signal_strength_dbm', 0) < -80)
    battery_signal_low = battery_low + signal_low

    # ── Device status pie (online/offline/maintenance)
    status_counts = Counter(d.get('status', 'offline') for d in devices)
    device_status_dist = [
        {'status': s, 'count': status_counts.get(s, 0)} for s in DEVICE_STATUSES
    ]

    # ── Battery histogram
    battery_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    battery_histogram = []
    for lo, hi in battery_bins:
        cnt = sum(1 for d in devices
                  if lo <= d.get('battery_pct', 0) < hi or (hi == 100 and d.get('battery_pct', 0) == 100))
        battery_histogram.append({'range': f'{lo}-{hi}%', 'count': cnt})

    # ── Alert type distribution
    alert_type_counts = Counter(a.get('alert_type', '') for a in alerts)
    alert_type_dist = [
        {'type': t, 'count': alert_type_counts.get(t, 0)} for t in ALERT_TYPES
    ]

    # ── Gateway uptime bars
    gateway_uptime_bars = [
        {
            'gateway_id': g.get('gateway_id', ''),
            'location': g.get('location', ''),
            'uptime_pct': g.get('uptime_pct', 0),
            'status': g.get('status', 'offline'),
        }
        for g in gateways
    ]

    # ── Firmware version spread
    fw_counts = Counter(d.get('firmware_version', 'unknown') for d in devices)
    firmware_dist = [
        {'version': v, 'count': fw_counts.get(v, 0)} for v in FIRMWARE_VERSIONS
    ]

    # ── Device type distribution
    type_counts = Counter(d.get('device_type', '') for d in devices)
    device_type_dist = [
        {'type': t, 'count': type_counts.get(t, 0)} for t in DEVICE_TYPES
    ]

    # ── Alert severity breakdown
    sev_counts = Counter(a.get('severity', '') for a in alerts)
    severity_dist = [
        {'severity': s, 'count': sev_counts.get(s, 0)} for s in ALERT_SEVERITIES
    ]

    # ── Resolution rate
    total_alerts = len(alerts)
    resolved_alerts = sum(1 for a in alerts if a.get('resolved'))
    resolution_rate = round(100 * resolved_alerts / total_alerts, 1) if total_alerts else 0

    return {
        'kpis': {
            'total_devices': total_devices,
            'devices_online': devices_online,
            'devices_online_pct': devices_online_pct,
            'avg_gateway_uptime_pct': avg_gateway_uptime,
            'avg_stream_latency_ms': avg_latency_ms,
            'sos_alerts_fired': sos_alerts,
            'battery_signal_low': battery_signal_low,
            'battery_low_count': battery_low,
            'signal_low_count': signal_low,
            'total_gateways': len(gateways),
            'total_alerts': total_alerts,
            'alert_resolution_rate': resolution_rate,
        },
        'device_status_distribution': device_status_dist,
        'device_type_distribution': device_type_dist,
        'battery_histogram': battery_histogram,
        'alert_type_distribution': alert_type_dist,
        'alert_severity_distribution': severity_dist,
        'gateway_uptime_bars': gateway_uptime_bars,
        'firmware_distribution': firmware_dist,
    }


def breakdown():
    """Return per-device detail with assigned patient, gateway, battery, signal, alerts."""
    devices, gateways, alerts, patients = _load_all_data()

    # Index alerts by device_id
    alerts_by_device = defaultdict(list)
    for a in alerts:
        did = a.get('device_id')
        if did:
            alerts_by_device[did].append(a)

    # Build gateway lookup by location
    gateway_by_location = {}
    for g in gateways:
        gateway_by_location[g.get('location', '')] = g

    profiles = []
    for dev in sorted(devices, key=lambda d: d.get('device_id', '')):
        device_id = dev.get('device_id', '')
        pid = dev.get('patient_id', '')
        pt = patients.get(pid, {})
        location = dev.get('location', '')
        gw = gateway_by_location.get(location, {})
        dev_alerts = alerts_by_device.get(device_id, [])

        profiles.append({
            'device_id': device_id,
            'device_type': dev.get('device_type', ''),
            'firmware_version': dev.get('firmware_version', ''),
            'status': dev.get('status', 'offline'),
            'battery_pct': dev.get('battery_pct', 0),
            'signal_strength_dbm': dev.get('signal_strength_dbm', 0),
            'latency_ms': dev.get('latency_ms', 0),
            'last_seen': dev.get('last_seen', ''),
            'location': location,
            # Patient info
            'patient_id': pid,
            'patient_name': pt.get('name') or pid,
            'patient_age': pt.get('age'),
            'patient_gender': pt.get('gender'),
            # Gateway info
            'gateway_id': gw.get('gateway_id', ''),
            'gateway_status': gw.get('status', ''),
            'gateway_uptime_pct': gw.get('uptime_pct', 0),
            # Alerts
            'alert_count': len(dev_alerts),
            'alerts': dev_alerts,
        })

    # Also include gateway detail
    gateway_detail = []
    for g in sorted(gateways, key=lambda x: x.get('gateway_id', '')):
        gw_alerts = [a for a in alerts if a.get('gateway_id') == g.get('gateway_id')]
        gateway_detail.append({
            'gateway_id': g.get('gateway_id', ''),
            'location': g.get('location', ''),
            'status': g.get('status', 'offline'),
            'uptime_pct': g.get('uptime_pct', 0),
            'connected_devices': g.get('connected_devices', 0),
            'last_heartbeat': g.get('last_heartbeat', ''),
            'firmware_version': g.get('firmware_version', ''),
            'offline_alerts': len(gw_alerts),
        })

    return {
        'devices': profiles,
        'gateways': gateway_detail,
    }


def definitions():
    """Return IoT/biomedical device monitoring terminology, IEC standards, and protocols."""
    return {
        'title': 'IoT Engineer — Definitions & Standards (IEC 62304, IEC 80001, HIPAA)',
        'sections': [
            {
                'heading': 'Wearable EEG Device Types',
                'items': [
                    {'term': 'Wearable EEG (wearable_eeg)', 'detail': 'Lightweight headband or cap-style device with dry or semi-dry electrodes for ambulatory seizure monitoring. Typically 4-8 channels, Bluetooth Low Energy (BLE) data transmission. Battery life 24-72 hours. IEC 60601-1 medical electrical equipment safety compliance required.'},
                    {'term': 'Ambulatory EEG (ambulatory_eeg)', 'detail': 'Portable multi-channel EEG recorder (16-32 channels) with local storage and periodic gateway sync. Uses clinical-grade gel electrodes. Sampling rate >= 256 Hz. Supports 24-72 hour continuous recording for outpatient epilepsy monitoring.'},
                    {'term': 'Home Monitor (home_monitor)', 'detail': 'Consumer-grade seizure detection device using accelerometry, EMG, and limited EEG. Triggers SOS alerts to caregivers and clinical team. Lower channel count but optimized for false-alarm reduction. FDA Class II 510(k) pathway.'},
                    {'term': 'Implantable RNS (implantable_rns)', 'detail': 'Responsive Neurostimulation System (NeuroPace). Implanted device with depth/strip electrodes that detects electrographic seizure patterns and delivers responsive cortical stimulation. Data telemetered via near-field wand to gateway. IEC 62304 Class C software lifecycle required.'},
                ],
            },
            {
                'heading': 'Gateway Architecture',
                'items': [
                    {'term': 'Clinical Gateway', 'detail': 'Edge computing node that aggregates data from multiple wearable devices via BLE/Wi-Fi. Performs local signal quality checks, data buffering, and MQTT/HTTPS upload to cloud. Deployed per ward/unit with redundancy. IEC 80001-1 risk management for IT-networks incorporating medical devices.'},
                    {'term': 'Gateway Heartbeat', 'detail': 'Periodic status message (every 30-60 seconds) confirming gateway is operational, reporting connected device count, CPU/memory utilization, and network throughput. Missing heartbeats trigger gateway_offline alerts after 3 consecutive failures.'},
                    {'term': 'Connected Devices', 'detail': 'Number of active wearable devices currently paired and streaming data through the gateway. Maximum capacity depends on BLE bandwidth (~7 concurrent BLE connections per radio, scalable with multiple radios).'},
                    {'term': 'Uptime Percentage', 'detail': 'Ratio of operational time to total time over a rolling 30-day window. Target >= 99.5% for clinical gateways. Downtime includes planned maintenance, firmware updates, and unplanned outages. Tracked per IEC 80001-1 availability requirements.'},
                ],
            },
            {
                'heading': 'Device Monitoring Metrics',
                'items': [
                    {'term': 'Battery Percentage', 'detail': 'Current battery charge level. Critical threshold < 20% triggers low_battery alert. Devices must maintain >= 10% reserve for safe shutdown and data preservation. Lithium-polymer cells with typical 500 charge cycles.'},
                    {'term': 'Signal Strength (dBm)', 'detail': 'Received Signal Strength Indicator (RSSI) for BLE/Wi-Fi link. Good: > -60 dBm, Acceptable: -60 to -80 dBm, Poor: < -80 dBm. Poor signal causes data packet loss and increased latency. Affected by patient mobility and ward construction materials.'},
                    {'term': 'Stream Latency (ms)', 'detail': 'End-to-end delay from EEG sample acquisition at the device to arrival at the cloud analytics engine. Target < 100 ms for real-time seizure detection. Components: device buffer (5-10 ms), BLE transmission (10-30 ms), gateway processing (5-15 ms), network upload (10-50 ms).'},
                    {'term': 'Firmware Version', 'detail': 'Software version running on the device microcontroller. Must be tracked for IEC 62304 software lifecycle compliance. Firmware updates deployed OTA (over-the-air) via gateway, with rollback capability. Version mismatch triggers firmware_outdated alert.'},
                ],
            },
            {
                'heading': 'Alert Protocols',
                'items': [
                    {'term': 'SOS Seizure Alert (sos_seizure)', 'detail': 'Critical alert triggered by on-device seizure detection algorithm detecting sustained ictal patterns (rhythmic activity > 3 seconds). Immediately notifies clinical team via dashboard, SMS, and pager. Response target: < 2 minutes acknowledgment. Requires human-in-the-loop confirmation to reduce false positives.'},
                    {'term': 'Low Battery Alert (low_battery)', 'detail': 'Warning triggered when device battery drops below 20%. Escalates to critical at < 10%. Nursing staff notified to replace/recharge device. Unresolved low battery leads to data gaps in continuous monitoring.'},
                    {'term': 'Signal Lost Alert (signal_lost)', 'detail': 'Warning triggered when device stops transmitting data for > 60 seconds. Causes: patient moved out of range, device malfunction, electrode disconnection. Auto-resolves when signal resumes. Persistent signal loss (> 5 min) escalates to critical.'},
                    {'term': 'High Impedance Alert (high_impedance)', 'detail': 'Warning triggered when electrode-scalp impedance exceeds 10 kOhm on wearable EEG. Indicates poor electrode contact, dried gel, or skin preparation issues. Requires technician intervention to re-gel or replace electrode.'},
                    {'term': 'Gateway Offline Alert (gateway_offline)', 'detail': 'Critical alert triggered when a gateway misses 3+ consecutive heartbeats. All devices on that gateway lose cloud connectivity. Failover to adjacent gateway (if available) or local device buffering. On-call IoT engineer dispatched for physical inspection.'},
                    {'term': 'Firmware Outdated Alert (firmware_outdated)', 'detail': 'Info-level alert when device firmware is > 2 versions behind current release. IEC 62304 requires documented rationale for deferred updates. Security patches are mandatory within 30 days per HIPAA technical safeguard requirements.'},
                ],
            },
            {
                'heading': 'Regulatory Standards',
                'items': [
                    {'term': 'IEC 62304 (Medical Device Software Lifecycle)', 'detail': 'International standard for software lifecycle processes for medical device software. Classifies software into Class A (no injury), B (non-serious injury), C (death or serious injury). Wearable EEG firmware is typically Class B or C. Requires documented software development plan, risk analysis, verification, and release.'},
                    {'term': 'IEC 80001-1 (Medical IT-Network Risk Management)', 'detail': 'Addresses risks from incorporating medical devices into IT networks. Covers safety, effectiveness, and data/system security. Applies to gateway infrastructure, cloud connectivity, and hospital network integration. Requires risk management file for each networked configuration.'},
                    {'term': 'HIPAA Device Requirements', 'detail': 'Protected Health Information (PHI) transmitted by wearable devices must be encrypted in transit (TLS 1.2+) and at rest (AES-256). Device identifiers must not expose patient identity. Access logging required for all device data. Breach notification within 60 days per HIPAA Breach Notification Rule.'},
                    {'term': 'FDA 21 CFR Part 820 (Quality System Regulation)', 'detail': 'Quality management system requirements for medical device manufacturers. Covers design controls, production controls, corrective/preventive actions (CAPA), and device tracking. Wearable EEG devices must maintain Device History Record (DHR) and Device Master Record (DMR).'},
                    {'term': 'IEC 60601-1 (Medical Electrical Equipment Safety)', 'detail': 'General safety and essential performance requirements for medical electrical equipment. Wearable EEG devices must meet leakage current limits, electromagnetic compatibility (EMC via IEC 60601-1-2), and biocompatibility requirements for skin-contact electrodes.'},
                ],
            },
            {
                'heading': 'Continuous Monitoring Infrastructure',
                'items': [
                    {'term': 'MQTT Broker', 'detail': 'Message Queuing Telemetry Transport broker for lightweight publish/subscribe device-to-cloud communication. QoS Level 1 (at least once delivery) for EEG data streams. QoS Level 2 (exactly once) for alerts and commands. Retained messages for device last-will status.'},
                    {'term': 'Data Throughput', 'detail': 'Aggregate data rate from all devices. Single-channel EEG at 256 Hz, 16-bit = ~4 kbps. 8-channel wearable = ~32 kbps. 35-device fleet = ~1.1 Mbps aggregate. Gateway must handle burst capacity for simultaneous seizure events.'},
                    {'term': 'Edge Processing', 'detail': 'On-gateway signal quality assessment, artifact rejection, and preliminary seizure detection before cloud upload. Reduces cloud bandwidth by ~40% and provides sub-100ms local alerting. Uses TensorFlow Lite or ONNX Runtime for inference.'},
                    {'term': 'OTA Firmware Update', 'detail': 'Over-the-air firmware deployment via gateway. Uses differential binary patches to minimize transfer size. Requires pre-update battery check (>= 30%), integrity verification (SHA-256), and automatic rollback on failed boot. IEC 62304 mandates documented update procedure and verification.'},
                ],
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN (first 3 devices) ===')
    bd = breakdown()
    for d in bd['devices'][:3]:
        d_display = {k: v for k, v in d.items() if k != 'alerts'}
        pprint.pprint(d_display)
    print(f'\nTotal devices: {len(bd["devices"])}')
    print(f'Total gateways: {len(bd["gateways"])}')
    print('\n=== GATEWAY DETAIL ===')
    for g in bd['gateways']:
        pprint.pprint(g)
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Sections: {len(df["sections"])}')
    for s in df['sections']:
        print(f'  {s["heading"]}: {len(s["items"])} items')
