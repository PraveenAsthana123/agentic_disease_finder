"""Seizure Band Dashboard — wearable seizure-detection devices (Empatica Embrace2,
Wrist EEG Bands, Ankle Sensors) fleet status, seizure-event detection, fall alerts,
seizure-risk scoring, and per-patient summaries. Real data: wearable_devices (30 rows),
wearable_readings (900 rows, 112 seizure events, 6 fall events)."""

import json
import os
import sqlite3
import statistics
from collections import Counter, defaultdict

_DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

# Device types that count as "seizure-band" class
_BAND_TYPES = {'Wrist EEG Band', 'Ankle Sensor'}
_BAND_BRANDS = {'Empatica Embrace2', 'Byteflies Sensor Dot'}


def _con():
    c = sqlite3.connect(_DB)
    c.row_factory = sqlite3.Row
    return c


def _load_devices(con):
    rows = con.execute('SELECT patient_id, fields_json FROM wearable_devices').fetchall()
    devices = []
    for row in rows:
        d = json.loads(row['fields_json'])
        d['patient_id'] = row['patient_id']
        devices.append(d)
    return devices


def _load_readings(con):
    rows = con.execute('SELECT patient_id, device_id, fields_json FROM wearable_readings').fetchall()
    readings = []
    for row in rows:
        d = json.loads(row['fields_json'])
        d['patient_id'] = row['patient_id']
        d['device_id'] = row['device_id']
        readings.append(d)
    return readings


def overview():
    """Summary KPIs: fleet status, seizure-detection events, fall alerts,
    battery health, risk scoring, and per-patient alert summary."""
    try:
        con = _con()
        devices = _load_devices(con)
        readings = _load_readings(con)
        con.close()
    except Exception as e:
        return {'available': False, 'error': str(e)}

    total_devices = len(devices)
    total_patients = len(set(d['patient_id'] for d in devices))
    total_readings = len(readings)

    # Device status breakdown
    status_counts = Counter(d.get('status', 'unknown') for d in devices)
    active = status_counts.get('active', 0)
    offline = status_counts.get('offline', 0)

    # Battery health
    batteries = [d.get('battery_level', 0) for d in devices]
    avg_battery = round(statistics.mean(batteries), 1) if batteries else 0
    low_battery = sum(1 for b in batteries if b < 20)

    # Seizure/fall events from readings
    seizure_events = [r for r in readings if r.get('seizure_detected')]
    fall_events = [r for r in readings if r.get('fall_detected')]
    seizure_count = len(seizure_events)
    fall_count = len(fall_events)

    # Detection confidence and risk
    confidences = [r.get('seizure_detection_confidence', 0) for r in readings]
    risks = [r.get('seizure_risk_score', 0) for r in readings]
    health_scores = [r.get('health_score', 0) for r in readings]
    avg_confidence = round(statistics.mean(confidences), 3) if confidences else 0
    avg_risk = round(statistics.mean(risks), 1) if risks else 0
    avg_health = round(statistics.mean(health_scores), 1) if health_scores else 0

    # Patients with seizure events
    patients_with_seizure = len(set(r['patient_id'] for r in seizure_events))

    # Device type breakdown
    type_counts = Counter(d.get('device_type', 'unknown') for d in devices)
    type_distribution = [
        {'name': k, 'count': v}
        for k, v in sorted(type_counts.items(), key=lambda x: -x[1])
    ]

    # Brand breakdown
    brand_counts = Counter(d.get('brand', 'unknown') for d in devices)
    brand_distribution = [
        {'name': k, 'count': v}
        for k, v in sorted(brand_counts.items(), key=lambda x: -x[1])
    ]

    # Status distribution
    status_distribution = [
        {'name': k, 'count': v}
        for k, v in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    # Seizure risk tiers
    risk_tiers = {'Low (0-25)': 0, 'Moderate (25-50)': 0, 'High (50-75)': 0, 'Critical (75+)': 0}
    for r in readings:
        score = r.get('seizure_risk_score', 0)
        if score < 25:
            risk_tiers['Low (0-25)'] += 1
        elif score < 50:
            risk_tiers['Moderate (25-50)'] += 1
        elif score < 75:
            risk_tiers['High (50-75)'] += 1
        else:
            risk_tiers['Critical (75+)'] += 1
    risk_distribution = [
        {'tier': k, 'count': v, 'pct': round(v / total_readings, 3) if total_readings else 0}
        for k, v in risk_tiers.items()
    ]

    # Recent seizure events (last 10)
    recent_seizures = sorted(seizure_events, key=lambda r: r.get('reading_date', ''), reverse=True)[:10]
    recent_list = [
        {
            'patient_id': r.get('patient_id'),
            'device_id': r.get('device_id'),
            'date': r.get('reading_date'),
            'confidence': r.get('seizure_detection_confidence'),
            'risk_score': r.get('seizure_risk_score'),
            'hr': r.get('heart_rate_avg'),
            'health_score': r.get('health_score'),
        }
        for r in recent_seizures
    ]

    # Per-patient summary
    patient_readings = defaultdict(list)
    for r in readings:
        patient_readings[r['patient_id']].append(r)

    per_patient = []
    for pid, rs in sorted(patient_readings.items()):
        seizures_p = [r for r in rs if r.get('seizure_detected')]
        falls_p = [r for r in rs if r.get('fall_detected')]
        avg_risk_p = round(statistics.mean([r.get('seizure_risk_score', 0) for r in rs]), 1)
        avg_health_p = round(statistics.mean([r.get('health_score', 0) for r in rs]), 1)
        per_patient.append({
            'patient_id': pid,
            'readings': len(rs),
            'seizure_events': len(seizures_p),
            'fall_events': len(falls_p),
            'avg_risk_score': avg_risk_p,
            'avg_health_score': avg_health_p,
            'risk_tier': (
                'Critical' if avg_risk_p >= 75 else
                'High' if avg_risk_p >= 50 else
                'Moderate' if avg_risk_p >= 25 else 'Low'
            ),
        })
    per_patient.sort(key=lambda x: -x['avg_risk_score'])

    return {
        'available': True,
        'kpis': {
            'total_devices': total_devices,
            'total_patients': total_patients,
            'active_devices': active,
            'offline_devices': offline,
            'avg_battery_pct': avg_battery,
            'low_battery_devices': low_battery,
            'total_readings': total_readings,
            'seizure_events_detected': seizure_count,
            'fall_events_detected': fall_count,
            'patients_with_seizure_event': patients_with_seizure,
            'avg_seizure_confidence': avg_confidence,
            'avg_risk_score': avg_risk,
            'avg_health_score': avg_health,
        },
        'status_distribution': status_distribution,
        'type_distribution': type_distribution,
        'brand_distribution': brand_distribution,
        'risk_distribution': risk_distribution,
        'recent_seizure_events': recent_list,
        'per_patient_summary': per_patient,
    }


def breakdown():
    """Per-device drill-down: device roster, connectivity, seizure-band specific reading
    aggregates (EDA proxy via stress_score, accelerometer via fall_detected), monthly trend."""
    try:
        con = _con()
        devices = _load_devices(con)
        readings = _load_readings(con)
        con.close()
    except Exception as e:
        return {'available': False, 'error': str(e)}

    # Device roster with reading counts
    device_map = {d['device_id']: d for d in devices}
    reading_counts = Counter(r['device_id'] for r in readings)
    seizure_by_device = Counter(r['device_id'] for r in readings if r.get('seizure_detected'))

    device_roster = []
    for d in devices:
        did = d.get('device_id', '')
        device_roster.append({
            'device_id': did,
            'patient_id': d.get('patient_id'),
            'device_type': d.get('device_type'),
            'brand': d.get('brand'),
            'status': d.get('status'),
            'battery_level': d.get('battery_level'),
            'connectivity': d.get('connectivity'),
            'firmware_version': d.get('firmware_version'),
            'seizure_detection_enabled': d.get('seizure_detection_enabled', False),
            'fall_detection_enabled': d.get('fall_detection_enabled', False),
            'last_sync': d.get('last_sync'),
            'total_readings': reading_counts.get(did, 0),
            'seizure_events': seizure_by_device.get(did, 0),
        })
    device_roster.sort(key=lambda x: -x['seizure_events'])

    # Monthly seizure event trend
    monthly = Counter()
    for r in readings:
        if r.get('seizure_detected'):
            date = r.get('reading_date', '')
            if len(date) >= 7:
                monthly[date[:7]] += 1
    monthly_trend = [
        {'month': m, 'seizure_events': c}
        for m, c in sorted(monthly.items())
    ]

    # Connectivity breakdown
    conn_counts = Counter(d.get('connectivity', 'unknown') for d in devices)
    connectivity_dist = [
        {'name': k, 'count': v}
        for k, v in sorted(conn_counts.items(), key=lambda x: -x[1])
    ]

    # Firmware version breakdown
    fw_counts = Counter(d.get('firmware_version', '?') for d in devices)
    firmware_dist = [
        {'version': k, 'count': v}
        for k, v in sorted(fw_counts.items(), key=lambda x: -x[1])
    ]

    # EDA/stress proxy: average stress_score (acts as EDA proxy for seizure band)
    stress_scores = [r.get('stress_score', 0) for r in readings if r.get('stress_score') is not None]
    avg_stress = round(statistics.mean(stress_scores), 1) if stress_scores else 0

    # SpO2 stats (important for seizure monitoring)
    spo2_vals = [r.get('spo2', 0) for r in readings if r.get('spo2')]
    avg_spo2 = round(statistics.mean(spo2_vals), 1) if spo2_vals else 0
    low_spo2 = sum(1 for v in spo2_vals if v < 95)

    # Seizure confidence distribution
    conf_buckets = {'<0.1': 0, '0.1-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7+': 0}
    for r in readings:
        c = r.get('seizure_detection_confidence', 0)
        if c < 0.1:
            conf_buckets['<0.1'] += 1
        elif c < 0.3:
            conf_buckets['0.1-0.3'] += 1
        elif c < 0.5:
            conf_buckets['0.3-0.5'] += 1
        elif c < 0.7:
            conf_buckets['0.5-0.7'] += 1
        else:
            conf_buckets['0.7+'] += 1
    confidence_dist = [
        {'bucket': k, 'count': v}
        for k, v in conf_buckets.items()
    ]

    return {
        'available': True,
        'device_roster': device_roster,
        'monthly_seizure_trend': monthly_trend,
        'connectivity_distribution': connectivity_dist,
        'firmware_distribution': firmware_dist,
        'confidence_distribution': confidence_dist,
        'physiological_summary': {
            'avg_stress_score': avg_stress,
            'avg_spo2': avg_spo2,
            'low_spo2_readings': low_spo2,
            'avg_heart_rate': round(
                statistics.mean([r.get('heart_rate_avg', 0) for r in readings]), 1
            ),
        },
    }


def definitions():
    """Seizure band glossary: device types, detection methods, data streams, alert pipeline."""
    return {
        'available': True,
        'title': 'Seizure Band — Device & Detection Glossary',
        'sections': {
            'Device Classes': {
                'label': 'Device Classes',
                'items': [
                    {
                        'term': 'Seizure-Detection Wristband (Embrace-style)',
                        'definition': (
                            'Ambulatory wrist-worn device that continuously monitors electrodermal '
                            'activity (EDA), accelerometer, and skin temperature to detect convulsive '
                            'seizures. Empatica Embrace2 is the clinical gold-standard example. '
                            'Alert triggered on EDA spike + motion burst pattern.'
                        ),
                    },
                    {
                        'term': 'Wrist EEG Band',
                        'definition': (
                            'Wrist-worn EEG sensor that captures surface neural signals supplemented '
                            'with HR, HRV, and motion data. Combines EEG features with autonomic '
                            'signals for multi-modal seizure detection.'
                        ),
                    },
                    {
                        'term': 'Ankle Sensor',
                        'definition': (
                            'Ankle-worn inertial measurement unit (IMU) for detecting fall events, '
                            'tonic-clonic movements, and gait anomalies associated with seizure onset.'
                        ),
                    },
                ],
            },
            'Data Streams': {
                'label': 'Data Streams',
                'items': [
                    {
                        'term': 'EDA (Electrodermal Activity)',
                        'definition': (
                            'Skin conductance signal reflecting sympathetic nervous system activation. '
                            'Sharp EDA spike is the primary seizure biomarker for Empatica Embrace2. '
                            'Proxied here by stress_score in wearable_readings.'
                        ),
                    },
                    {
                        'term': 'Accelerometer / IMU',
                        'definition': (
                            '3-axis inertial measurement for tonic-clonic limb movements. '
                            'Detects rhythmic jerking patterns that characterize convulsive seizures. '
                            'Fall events in wearable_readings represent IMU-triggered fall detection.'
                        ),
                    },
                    {
                        'term': 'Seizure Detection Confidence',
                        'definition': (
                            'Model output probability [0–1] that the current sensor reading reflects '
                            'a true seizure event. Threshold for alert: ≥0.5. Values shown per-reading '
                            'in wearable_readings.seizure_detection_confidence.'
                        ),
                    },
                    {
                        'term': 'Seizure Risk Score',
                        'definition': (
                            'Composite per-patient risk score [0–100] aggregating seizure history, '
                            'detection confidence, autonomic instability, and medication adherence. '
                            'Tiers: Low (<25) · Moderate (25–50) · High (50–75) · Critical (≥75).'
                        ),
                    },
                ],
            },
            'Alert Pipeline': {
                'label': 'Alert Pipeline',
                'items': [
                    {
                        'term': 'On-Device Detection',
                        'definition': (
                            'Edge ML model runs locally on device. Detects convulsive seizure in <3 s '
                            'without cloud round-trip. Seizure_detected flag set in reading payload.'
                        ),
                    },
                    {
                        'term': 'Caregiver SOS Notification',
                        'definition': (
                            'On confirmed detection: BLE → phone app → push notification to emergency '
                            'contact within 10 s. If phone unreachable: Cellular backup SOS to on-call.'
                        ),
                    },
                    {
                        'term': 'Offline Buffering',
                        'definition': (
                            'Device buffers readings in on-device ring-buffer when offline. '
                            'Syncs to backend on BLE reconnect. No data loss; delayed inference only.'
                        ),
                    },
                ],
            },
            'Compliance': {
                'label': 'Regulatory & Compliance',
                'items': [
                    {
                        'term': 'Empatica Embrace2 FDA Clearance',
                        'definition': (
                            'FDA De Novo authorization (DEN180031) for convulsive seizure detection. '
                            'CE-marked under MDR Class IIa. Used in clinical trials for non-convulsive '
                            'seizure monitoring research.'
                        ),
                    },
                    {
                        'term': 'Data Handling',
                        'definition': (
                            'All readings stored in clinical.db wearable_readings with patient linkage. '
                            'PHI encrypted at rest. Access restricted to treating clinicians.'
                        ),
                    },
                ],
            },
        },
    }
