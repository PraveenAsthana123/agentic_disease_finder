"""Data Privacy Dashboard — PII/PHI exposure analytics from clinical.db.

Monitors personally identifiable information (PII) exposure, protected health
information (PHI) access patterns, de-identification coverage, conversation PHI
leakage, and per-patient privacy profiles for the clinical EEG/epilepsy platform.

Sources:
- patients table (40 records): PII fields — name, age, gender, disease, department
- patient_master table (2 records): de-identification pipeline coverage
- transaction_log table (645+ events): PHI access by actor/component
- conversation_log table (360 messages): PHI leakage via patient ID references
- hitl_reviews table: clinical review records with PHI
- uploads / analyses tables: file-level PHI exposure per patient
"""

import sqlite3
import os
import re
from collections import defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


# ── Component sensitivity classification ──
SENSITIVITY_MAP = {
    'clinical_trust': 'HIGH',
    'expert_review': 'HIGH',
    'medications': 'HIGH',
    'patient_master': 'HIGH',
    'patient_chat': 'HIGH',
    'seizure_diary': 'HIGH',
    'eeg_upload': 'MEDIUM',
    'training': 'MEDIUM',
    'eeg_analysis': 'MEDIUM',
    'hitl_review': 'MEDIUM',
    'clinical_decisions': 'MEDIUM',
    'scheduled_train': 'MEDIUM',
}

PATIENT_ID_PATTERN = re.compile(r'P\d{4}')


def _classify_sensitivity(component):
    return SENSITIVITY_MAP.get((component or '').lower().strip(), 'LOW')


def data_privacy_overview():
    """Full privacy dashboard KPIs, PII exposure, PHI access trends, and distributions."""
    conn = _conn()
    cur = conn.cursor()

    # ── PII field exposure across patients table ──
    pii_name_count = _safe(cur, 'SELECT COUNT(*) FROM patients WHERE name IS NOT NULL AND name != ""')
    pii_age_count = _safe(cur, 'SELECT COUNT(*) FROM patients WHERE age IS NOT NULL')
    pii_gender_count = _safe(cur, 'SELECT COUNT(*) FROM patients WHERE gender IS NOT NULL AND gender != ""')
    pii_disease_count = _safe(cur, 'SELECT COUNT(*) FROM patients WHERE disease IS NOT NULL AND disease != ""')
    pii_department_count = _safe(cur, 'SELECT COUNT(*) FROM patients WHERE department IS NOT NULL AND department != ""')
    total_pii_fields = pii_name_count + pii_age_count + pii_gender_count + pii_disease_count + pii_department_count

    # ── Total patient records with PII (name populated) ──
    patients_with_pii = pii_name_count
    total_patients = _safe(cur, 'SELECT COUNT(*) FROM patients')

    # ── PHI access events (transaction_log with patient_id not null) ──
    phi_access_events = _safe(cur, '''
        SELECT COUNT(*) FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
    ''')

    # ── Unique actors accessing PHI ──
    unique_phi_actors = _safe(cur, '''
        SELECT COUNT(DISTINCT actor) FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
    ''')

    # ── De-identification coverage ──
    patient_master_count = _safe(cur, 'SELECT COUNT(*) FROM patient_master')
    deidentification_coverage = round(patient_master_count / max(total_patients, 1) * 100, 1)

    # ── Conversation PHI exposure: scan for patient ID patterns ──
    conv_rows = _safe_rows(cur, 'SELECT id, text FROM conversation_log WHERE text IS NOT NULL')
    conv_phi_count = 0
    for row in conv_rows:
        if PATIENT_ID_PATTERN.search(row[1] or ''):
            conv_phi_count += 1
    total_conversations = _safe(cur, 'SELECT COUNT(*) FROM conversation_log')
    conv_phi_rate = round(conv_phi_count / max(total_conversations, 1) * 100, 1)

    # ── Total transaction events (for context) ──
    total_events = _safe(cur, 'SELECT COUNT(*) FROM transaction_log')

    # ── PII field distribution ──
    pii_field_distribution = [
        {'field': 'name', 'non_null_count': pii_name_count},
        {'field': 'age', 'non_null_count': pii_age_count},
        {'field': 'gender', 'non_null_count': pii_gender_count},
        {'field': 'disease', 'non_null_count': pii_disease_count},
        {'field': 'department', 'non_null_count': pii_department_count},
    ]

    # ── PHI access by component ──
    comp_rows = _safe_rows(cur, '''
        SELECT component, COUNT(*) as cnt
        FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
        GROUP BY component
        ORDER BY cnt DESC
    ''')
    phi_access_by_component = [{'component': r[0], 'events': r[1]} for r in comp_rows]

    # ── Daily PHI access trend ──
    daily_rows = _safe_rows(cur, '''
        SELECT DATE(ts_local) as day, COUNT(*) as cnt
        FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
          AND ts_local IS NOT NULL
        GROUP BY DATE(ts_local)
        ORDER BY day
    ''')
    daily_phi_trend = [{'date': r[0], 'events': r[1]} for r in daily_rows]

    # ── PHI access by action type ──
    action_rows = _safe_rows(cur, '''
        SELECT action, COUNT(*) as cnt
        FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
        GROUP BY action
        ORDER BY cnt DESC
    ''')
    phi_access_by_action = [{'action': r[0], 'events': r[1]} for r in action_rows]

    conn.close()

    return {
        'available': True,
        'kpis': {
            'total_pii_fields_exposed': total_pii_fields,
            'patients_with_pii': patients_with_pii,
            'total_patients': total_patients,
            'phi_access_events': phi_access_events,
            'unique_phi_actors': unique_phi_actors,
            'deidentification_coverage_pct': deidentification_coverage,
            'conversation_phi_messages': conv_phi_count,
            'conversation_phi_rate_pct': conv_phi_rate,
        },
        'pii_field_distribution': pii_field_distribution,
        'phi_access_by_component': phi_access_by_component,
        'daily_phi_trend': daily_phi_trend,
        'phi_access_by_action': phi_access_by_action,
    }


def data_privacy_breakdown():
    """Per-patient privacy profiles, conversation PHI scan, actor matrix, sensitivity, access log, upload privacy."""
    conn = _conn()
    cur = conn.cursor()

    # ── Per-patient privacy profile ──
    patient_ids = _safe_rows(cur, '''
        SELECT DISTINCT patient_id FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
        ORDER BY patient_id
    ''')
    patient_profiles = []
    for (pid,) in patient_ids:
        event_count = _safe(cur, '''
            SELECT COUNT(*) FROM transaction_log WHERE patient_id = ?
        ''', (pid,))
        components = _safe_rows(cur, '''
            SELECT DISTINCT component FROM transaction_log WHERE patient_id = ?
        ''', (pid,))
        actors = _safe_rows(cur, '''
            SELECT DISTINCT actor FROM transaction_log WHERE patient_id = ?
        ''', (pid,))
        patient_profiles.append({
            'patient_id': pid,
            'access_count': event_count,
            'components': [c[0] for c in components],
            'actors': [a[0] for a in actors],
        })

    # ── Conversation PHI scan ──
    conv_rows = _safe_rows(cur, '''
        SELECT id, role, text, ts_utc FROM conversation_log
        WHERE text IS NOT NULL
        ORDER BY id
    ''')
    conversation_phi_scan = []
    for cid, role, text, ts in conv_rows:
        matches = PATIENT_ID_PATTERN.findall(text or '')
        if matches:
            conversation_phi_scan.append({
                'message_id': cid,
                'role': role,
                'timestamp': ts,
                'patient_ids_found': list(set(matches)),
                'snippet': (text or '')[:120],
            })

    # ── Actor PHI access matrix ──
    actor_matrix_rows = _safe_rows(cur, '''
        SELECT actor, patient_id, COUNT(*) as cnt
        FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
        GROUP BY actor, patient_id
        ORDER BY actor, cnt DESC
    ''')
    actor_matrix = defaultdict(list)
    for actor, pid, cnt in actor_matrix_rows:
        actor_matrix[actor].append({'patient_id': pid, 'access_count': cnt})
    actor_phi_matrix = [{'actor': a, 'patients': plist, 'total_patients': len(plist)}
                        for a, plist in sorted(actor_matrix.items())]

    # ── Component data sensitivity ──
    comp_rows = _safe_rows(cur, '''
        SELECT component, COUNT(*) as cnt
        FROM transaction_log
        GROUP BY component
        ORDER BY cnt DESC
    ''')
    component_sensitivity = []
    for comp, cnt in comp_rows:
        component_sensitivity.append({
            'component': comp,
            'events': cnt,
            'sensitivity': _classify_sensitivity(comp),
        })

    # ── Recent PHI access log (last 30) ──
    recent_rows = _safe_rows(cur, '''
        SELECT id, patient_id, component, action, actor, detail, ts_local
        FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
        ORDER BY ts_utc DESC LIMIT 30
    ''')
    recent_phi_log = []
    for r in recent_rows:
        recent_phi_log.append({
            'id': r[0],
            'patient_id': r[1],
            'component': r[2],
            'action': r[3],
            'actor': r[4],
            'detail': (r[5] or '')[:100],
            'timestamp': r[6],
        })

    # ── Upload privacy: patients with file uploads ──
    upload_rows = _safe_rows(cur, '''
        SELECT patient_id, COUNT(*) as file_count
        FROM uploads
        WHERE patient_id IS NOT NULL AND patient_id != ''
        GROUP BY patient_id
        ORDER BY file_count DESC
    ''')
    upload_privacy = [{'patient_id': r[0], 'file_count': r[1]} for r in upload_rows]

    conn.close()

    return {
        'available': True,
        'patient_profiles': patient_profiles,
        'conversation_phi_scan': conversation_phi_scan,
        'actor_phi_matrix': actor_phi_matrix,
        'component_sensitivity': component_sensitivity,
        'recent_phi_log': recent_phi_log,
        'upload_privacy': upload_privacy,
    }


def data_privacy_definitions():
    """Data privacy metric definitions for tooltip overlays."""
    return {
        'available': True,
        'definitions': {
            'concepts': [
                {'term': 'PII (Personally Identifiable Information)', 'definition': 'Any data that can identify a specific individual — in this system: patient name, age, gender. PII exposure is tracked by counting non-null PII fields across the patients table.'},
                {'term': 'PHI (Protected Health Information)', 'definition': 'Health-related data linked to an individual — includes diagnoses, EEG results, medications, seizure diaries. PHI access is tracked via transaction_log entries that reference a patient_id.'},
                {'term': 'De-identification', 'definition': 'The process of removing or obscuring PII/PHI so data cannot be linked to a specific patient. Coverage is measured by the ratio of patient_master records (processed through the master pipeline) to total patient records.'},
                {'term': 'Pseudonymization', 'definition': 'Replacing direct identifiers with artificial IDs (e.g., P0001) while maintaining a separate mapping. The system uses patient_id codes as pseudonyms, but conversation logs may still leak these IDs.'},
                {'term': 'Data Minimization', 'definition': 'Collecting and retaining only the minimum PII/PHI necessary for the clinical purpose. Measured by the ratio of populated PII fields to total possible fields across all patient records.'},
                {'term': 'Consent Management', 'definition': 'Tracking patient consent for data collection, processing, and sharing. Requires explicit records of what each patient consented to and when — currently assessed by audit trail completeness.'},
                {'term': 'Access Control', 'definition': 'Restricting PHI access to authorized actors and components. Measured by the number of distinct actors accessing patient data and whether access aligns with their clinical role.'},
                {'term': 'Audit Trail', 'definition': 'A chronological record of all PHI access events — who accessed which patient data, through which component, performing what action, and when. Sourced from the transaction_log table.'},
            ],
            'quality_metrics': [
                {'metric': 'PII Exposure Rate', 'description': 'Percentage of PII fields (name, age, gender, disease, department) that contain non-null values across all patient records. Lower rates indicate better data minimization.'},
                {'metric': 'PHI Access Frequency', 'description': 'Total number of transaction_log events that reference a patient_id — measures how often patient-linked data is touched by system components.'},
                {'metric': 'De-identification Coverage', 'description': 'Percentage of patients who have a corresponding patient_master record, indicating they have been processed through the de-identification/master pipeline.'},
                {'metric': 'Conversation PHI Leakage', 'description': 'Percentage of conversation_log messages that contain patient ID patterns (P0001, P0002, etc.) — indicates PHI exposure in unstructured text channels.'},
                {'metric': 'Actor Spread', 'description': 'Number of distinct actors (users, agents, services) that have accessed PHI — wider spread increases the attack surface and complicates access control auditing.'},
                {'metric': 'Component Sensitivity Score', 'description': 'Classification of each system component as HIGH (clinical_trust, expert_review, medications), MEDIUM (eeg_upload, training), or LOW sensitivity based on the type of patient data it handles.'},
            ],
            'clinical_relevance': [
                {'standard': 'HIPAA Privacy Rule (45 CFR 164.502)', 'requirement': 'Requires covered entities to limit uses and disclosures of PHI to the minimum necessary, implement safeguards to protect PHI, and provide patients with rights over their health information.'},
                {'standard': 'HIPAA Security Rule (45 CFR 164.312)', 'requirement': 'Mandates technical safeguards including access controls, audit controls, integrity controls, and transmission security for all electronic PHI (ePHI).'},
                {'standard': 'GDPR Article 9 (Special Categories)', 'requirement': 'Health data is a special category requiring explicit consent or specific legal basis for processing. Requires data protection impact assessments and records of processing activities.'},
                {'standard': 'FDA 21 CFR Part 11', 'requirement': 'Electronic records and signatures must ensure data integrity, audit trails, and access controls — applies to clinical AI systems that generate or store patient health records.'},
                {'standard': 'EU AI Act (Data Governance)', 'requirement': 'High-risk AI systems must implement data governance measures including data quality checks, bias detection, and privacy-preserving techniques for training and validation datasets.'},
                {'standard': 'IEC 62304 (Data Integrity)', 'requirement': 'Medical device software must maintain data integrity throughout the software lifecycle, including secure storage, transmission, and processing of patient health information.'},
            ],
            'remediation': [
                {'action': 'Implement PII masking pipeline', 'description': 'Deploy automated masking for name, age, and gender fields before data enters analytics pipelines. Use k-anonymity (k>=5) for quasi-identifiers and suppress direct identifiers in non-clinical views.'},
                {'action': 'Add consent tracking', 'description': 'Create a consent_records table linking each patient to explicit consent grants — what data, what purpose, when granted, when it expires. Block processing for patients without active consent.'},
                {'action': 'Enable field-level encryption', 'description': 'Encrypt PII fields (name, age, gender) at rest using AES-256. Implement key management with role-based decryption so only authorized clinical actors can view plaintext PII.'},
                {'action': 'Deploy automated PHI scanning', 'description': 'Add a pre-commit hook and runtime scanner that detects patient ID patterns (P0001-P9999) in conversation logs, commit messages, and unstructured text fields — alert and redact automatically.'},
                {'action': 'Implement data retention policies', 'description': 'Define retention periods per data category (e.g., raw EEG 7 years, conversation logs 1 year, analytics aggregates indefinite). Automate purging of expired records with audit logging.'},
                {'action': 'Add privacy impact assessments', 'description': 'Conduct and document a Privacy Impact Assessment (PIA) for each new component or data flow that touches PHI. Store assessments alongside the component registry and review quarterly.'},
            ],
        },
    }
