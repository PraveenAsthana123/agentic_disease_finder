"""IS SOP Dashboard — Information Security Standard Operating Procedures for
clinical EEG/epilepsy monitoring facilities. Tracks SOP lifecycle, compliance
scoring, audit findings, and review cadence across data protection, access
control, device security, incident management, compliance, and training domains.

Populates and reads from:
  - is_sop_procedures  (SOPs: title, version, status, category, compliance score)
  - is_sop_audits      (audit records: findings, severity, corrective actions)

Uses real patient_ids from the patients table.
HIPAA Security Rule, 21 CFR Part 11, IEC 62443, NIST CSF, ISO 27001 applied throughout.
"""

import json
import os
import random
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

SOP_TITLES = [
    'EEG Data Handling and Storage',
    'AI Model Validation Protocol',
    'Incident Response Procedure',
    'Patient Data Access Control',
    'HIPAA PHI De-identification',
    'EEG Recording Equipment Calibration',
    'Clinical Decision Support Audit',
    'Data Backup and Recovery',
    'User Account Management',
    'Network Security for EEG Devices',
    'Vulnerability Management',
    'Change Management Protocol',
    'Physical Security - EMU',
    'Third-Party Vendor Assessment',
    'Data Retention and Disposal',
    'Emergency Access Procedure',
    'Audit Log Review',
    'Mobile Device Management',
    'Cloud Data Security',
    'Staff Security Training',
]

SOP_CATEGORIES = [
    'Data Protection',
    'Data Protection',
    'Incident Management',
    'Access Control',
    'Compliance',
    'Device Security',
    'Compliance',
    'Data Protection',
    'Access Control',
    'Device Security',
    'Device Security',
    'Compliance',
    'Device Security',
    'Compliance',
    'Data Protection',
    'Access Control',
    'Compliance',
    'Device Security',
    'Data Protection',
    'Training',
]

SOP_OWNERS = [
    'IT Security',
    'Clinical Engineering',
    'IT Security',
    'IT Security',
    'Compliance',
    'Clinical Engineering',
    'Compliance',
    'IT Security',
    'IT Security',
    'Clinical Engineering',
    'IT Security',
    'Compliance',
    'Clinical Engineering',
    'Compliance',
    'Compliance',
    'IT Security',
    'Compliance',
    'IT Security',
    'IT Security',
    'Neurology',
]

SOP_STATUSES = ['published', 'draft', 'under_review', 'retired']
FINDING_TYPES = ['compliant', 'minor_nonconformance', 'major_nonconformance', 'observation']
AUDIT_STATUSES = ['open', 'closed', 'in_progress']
SEVERITIES = ['low', 'medium', 'high', 'critical']
APPLICABLE_STANDARDS_POOL = [
    'HIPAA', '21 CFR Part 11', 'IEC 62443', 'NIST CSF', 'ISO 27001',
    'HITRUST', 'SOC 2 Type II', 'GDPR',
]
APPROVER_NAMES = [
    'Dr. Sarah Mitchell', 'James Kowalski', 'Dr. Anita Patel',
    'Robert Chen', 'Lisa Fernandez', 'Dr. Marcus Webb',
]
AUDITOR_NAMES = [
    'External: Deloitte Health', 'Internal: Compliance Team',
    'External: KPMG Cyber', 'Internal: IT Security',
    'External: Ernst & Young', 'Internal: Clinical Engineering',
]

FINDING_DESCRIPTIONS = [
    'Access logs reviewed and complete for reporting period',
    'Two user accounts lacked MFA enrollment past grace period',
    'Backup restoration test not performed within SOP-defined 90-day window',
    'EEG device firmware update applied without change request documentation',
    'PHI de-identification process fully compliant with Safe Harbor method',
    'Incident response drill completed within target 4-hour window',
    'Vendor risk assessment documentation incomplete for one cloud provider',
    'Password complexity policy enforced across all clinical workstations',
    'Network segmentation verified between EMU and administrative networks',
    'Mobile device encryption confirmed on all registered tablets',
    'Data retention schedule followed; no records past retention period found',
    'Emergency access procedure invoked once; properly documented and reviewed',
    'Audit log review cadence missed for two consecutive months',
    'Staff training completion rate below 90% threshold for Q2',
    'Cloud data encryption at rest verified (AES-256)',
    'Physical access badge audit revealed three inactive badges not deactivated',
    'AI model validation checklist partially completed; two items pending',
    'Change management board approval obtained for all Q3 changes',
    'Vulnerability scan identified two medium-severity unpatched systems',
    'Third-party SOC 2 report reviewed and accepted for primary EHR vendor',
    'EEG calibration records complete and within tolerance specifications',
    'User provisioning followed least-privilege principle; one exception noted',
    'Incident response plan updated to include ransomware-specific playbook',
    'Data backup integrity check passed; RPO and RTO targets met',
    'Clinical decision support audit trail complete for all AI-assisted reads',
    'Network intrusion detection system alerts reviewed weekly as required',
    'Equipment disposal certificates obtained for all decommissioned devices',
    'Security awareness phishing simulation results improved over baseline',
    'Role-based access control matrix reviewed and approved by department heads',
    'Wireless network security assessment completed; WPA3 enforced',
]

CORRECTIVE_ACTIONS = [
    None,
    'Enforce MFA enrollment within 7 days; disable non-compliant accounts',
    'Schedule backup restoration test within 14 days; update SOP timeline',
    'Implement pre-deployment change request checklist for firmware updates',
    None,
    None,
    'Complete vendor risk assessment for remaining cloud providers by Q4',
    None,
    None,
    None,
    None,
    None,
    'Assign dedicated reviewer; set calendar reminders for monthly review',
    'Schedule make-up training sessions; notify department heads of deadline',
    None,
    'Deactivate inactive badges immediately; implement quarterly badge audit',
    'Complete remaining validation checklist items within 30 days',
    None,
    'Prioritize patching for identified systems within 30-day SLA',
    None,
    None,
    'Review and remediate the access exception within 7 days',
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]

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


def _generate_procedure(idx, rng):
    """Generate a single IS SOP procedure record."""
    sop_id = f'SOP-{idx + 1:03d}'
    title = SOP_TITLES[idx]
    category = SOP_CATEGORIES[idx]
    owner = SOP_OWNERS[idx]

    # Status distribution: mostly published
    status = rng.choices(SOP_STATUSES, weights=[0.60, 0.15, 0.15, 0.10])[0]

    version = rng.choice(['1.0', '1.1', '1.2', '2.0', '2.1', '3.0'])
    revision_count = rng.randint(1, 8)

    # Compliance score: generally high but some lower
    compliance_score = rng.randint(55, 100)

    # Applicable standards: 2-4 from pool
    num_standards = rng.randint(2, 4)
    applicable_standards = rng.sample(APPLICABLE_STANDARDS_POOL, num_standards)

    approver = rng.choice(APPROVER_NAMES)

    # Last reviewed: within past 18 months
    days_ago = rng.randint(30, 540)
    last_reviewed_dt = datetime(2025, 6, 15) - timedelta(days=days_ago)
    last_reviewed = last_reviewed_dt.strftime('%Y-%m-%d')

    # Next review due: some in the past (overdue), some in the future
    review_interval_days = rng.choice([90, 180, 365])
    next_review_dt = last_reviewed_dt + timedelta(days=review_interval_days)
    next_review_due = next_review_dt.strftime('%Y-%m-%d')

    return {
        'sop_id': sop_id,
        'title': title,
        'version': version,
        'status': status,
        'category': category,
        'owner': owner,
        'last_reviewed': last_reviewed,
        'next_review_due': next_review_due,
        'compliance_score': compliance_score,
        'applicable_standards': applicable_standards,
        'approver': approver,
        'revision_count': revision_count,
    }


def _generate_audit(idx, sop_ids, rng):
    """Generate a single IS SOP audit record."""
    audit_id = f'AUD-{idx + 1:03d}'
    sop_id = rng.choice(sop_ids)

    # Audit date within past 12 months
    days_ago = rng.randint(7, 365)
    audit_dt = datetime(2025, 6, 15) - timedelta(days=days_ago)
    audit_date = audit_dt.strftime('%Y-%m-%d')

    auditor = rng.choice(AUDITOR_NAMES)

    # Finding type distribution
    finding_type = rng.choices(
        FINDING_TYPES, weights=[0.40, 0.30, 0.15, 0.15]
    )[0]

    finding_idx = idx % len(FINDING_DESCRIPTIONS)
    finding_description = FINDING_DESCRIPTIONS[finding_idx]

    # Corrective action: required for nonconformances, optional otherwise
    if finding_type in ('minor_nonconformance', 'major_nonconformance'):
        corrective_action = CORRECTIVE_ACTIONS[finding_idx] or \
            'Develop and implement corrective action plan within 30 days'
    else:
        corrective_action = None

    # Severity distribution
    if finding_type == 'major_nonconformance':
        severity = rng.choices(SEVERITIES, weights=[0.0, 0.15, 0.55, 0.30])[0]
    elif finding_type == 'minor_nonconformance':
        severity = rng.choices(SEVERITIES, weights=[0.20, 0.50, 0.25, 0.05])[0]
    elif finding_type == 'observation':
        severity = rng.choices(SEVERITIES, weights=[0.50, 0.40, 0.10, 0.0])[0]
    else:
        severity = 'low'

    # Status distribution
    if finding_type == 'compliant':
        audit_status = 'closed'
    else:
        audit_status = rng.choices(AUDIT_STATUSES, weights=[0.30, 0.45, 0.25])[0]

    return {
        'audit_id': audit_id,
        'sop_id': sop_id,
        'audit_date': audit_date,
        'auditor': auditor,
        'finding_type': finding_type,
        'finding_description': finding_description,
        'corrective_action': corrective_action,
        'status': audit_status,
        'severity': severity,
    }


def _populate_if_empty():
    """Populate IS SOP tables with realistic data if empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        conn.execute('''CREATE TABLE IF NOT EXISTS is_sop_procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            fields_json TEXT,
            created_at TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS is_sop_audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            fields_json TEXT,
            created_at TEXT
        )''')
        conn.commit()

        count = conn.execute('SELECT COUNT(*) FROM is_sop_procedures').fetchone()[0]
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

        rng = random.Random(99)

        # 1. Generate 20 IS SOP procedures
        sop_ids = []
        for i in range(len(SOP_TITLES)):
            proc = _generate_procedure(i, rng)
            sop_ids.append(proc['sop_id'])
            pid = patient_ids[i % len(patient_ids)]
            conn.execute(
                'INSERT INTO is_sop_procedures (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                (pid, json.dumps(proc))
            )

        # 2. Generate 30 audit records
        for i in range(30):
            audit = _generate_audit(i, sop_ids, rng)
            pid = patient_ids[i % len(patient_ids)]
            conn.execute(
                'INSERT INTO is_sop_audits (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                (pid, json.dumps(audit))
            )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def _load_all_data():
    """Load all IS SOP tables and return structured data."""
    _populate_if_empty()

    procedures = []
    for r in _db_query('SELECT patient_id, fields_json FROM is_sop_procedures'):
        procedures.append(_safe_json(r['fields_json']))

    audits = []
    for r in _db_query('SELECT patient_id, fields_json FROM is_sop_audits'):
        audits.append(_safe_json(r['fields_json']))

    return procedures, audits


def overview():
    """Return KPI cards + chart data for the IS SOP overview tab."""
    procedures, audits = _load_all_data()

    total_sops = len(procedures)
    published_count = sum(1 for p in procedures if p.get('status') == 'published')
    draft_count = sum(1 for p in procedures if p.get('status') == 'draft')
    under_review_count = sum(1 for p in procedures if p.get('status') == 'under_review')
    retired_count = sum(1 for p in procedures if p.get('status') == 'retired')

    # Review due = next_review_due < today or within 30 days
    today = datetime.now()
    reviews_due = 0
    overdue_reviews = 0
    for p in procedures:
        nrd = p.get('next_review_due', '')
        if nrd:
            try:
                nrd_dt = datetime.strptime(nrd, '%Y-%m-%d')
                if nrd_dt < today:
                    overdue_reviews += 1
                    reviews_due += 1
                elif nrd_dt <= today + timedelta(days=30):
                    reviews_due += 1
            except ValueError:
                pass

    # Compliance scores
    compliance_scores = [p.get('compliance_score', 0) for p in procedures]
    avg_compliance_score = _avg(compliance_scores)

    # Compliance by category
    cat_scores = {}
    cat_counts = {}
    for p in procedures:
        cat = p.get('category', 'Unknown')
        score = p.get('compliance_score', 0)
        cat_scores.setdefault(cat, []).append(score)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    compliance_by_category = [
        {'category': cat, 'avg_score': _avg(scores), 'count': len(scores)}
        for cat, scores in sorted(cat_scores.items())
    ]

    # Status distribution
    status_counts = Counter(p.get('status', 'draft') for p in procedures)
    status_distribution = [
        {'status': st, 'count': status_counts.get(st, 0)} for st in SOP_STATUSES
    ]

    # Category distribution
    category_counts = Counter(p.get('category', 'Unknown') for p in procedures)
    category_distribution = [
        {'category': cat, 'count': cnt}
        for cat, cnt in sorted(category_counts.items())
    ]

    # Audit KPIs
    total_audits = len(audits)
    open_findings = sum(1 for a in audits if a.get('status') == 'open')
    closed_findings = sum(1 for a in audits if a.get('status') == 'closed')

    # Finding type distribution
    finding_type_counts = Counter(a.get('finding_type', '') for a in audits)
    finding_type_distribution = [
        {'finding_type': ft, 'count': finding_type_counts.get(ft, 0)}
        for ft in FINDING_TYPES
    ]

    # Severity distribution
    severity_counts = Counter(a.get('severity', '') for a in audits)
    severity_distribution = [
        {'severity': sev, 'count': severity_counts.get(sev, 0)}
        for sev in SEVERITIES
    ]

    # Top noncompliant SOPs (bottom 5 by compliance score)
    sorted_procs = sorted(procedures, key=lambda p: p.get('compliance_score', 0))
    top_noncompliant_sops = [
        {
            'sop_id': p.get('sop_id', ''),
            'title': p.get('title', ''),
            'compliance_score': p.get('compliance_score', 0),
            'category': p.get('category', ''),
            'status': p.get('status', ''),
        }
        for p in sorted_procs[:5]
    ]

    return {
        'available': True,
        'total_sops': total_sops,
        'published': published_count,
        'draft': draft_count,
        'under_review': under_review_count,
        'retired': retired_count,
        'reviews_due': reviews_due,
        'overdue': overdue_reviews,
        'avg_compliance': avg_compliance_score,
        'total_audits': total_audits,
        'open_findings': open_findings,
        'closed_findings': closed_findings,
        'compliance_by_category': compliance_by_category,
        'status_distribution': status_distribution,
        'category_distribution': category_distribution,
        'finding_type_distribution': finding_type_distribution,
        'severity_distribution': severity_distribution,
        'top_noncompliant_sops': top_noncompliant_sops,
    }


def breakdown():
    """Return per-procedure and per-audit detail for the IS SOP dashboard."""
    procedures, audits = _load_all_data()

    # Build audit lookup by sop_id
    audit_by_sop = {}
    for a in audits:
        sid = a.get('sop_id', '')
        audit_by_sop.setdefault(sid, []).append(a)

    procedure_details = []
    for p in sorted(procedures, key=lambda x: x.get('sop_id', '')):
        sop_id = p.get('sop_id', '')
        associated_audits = audit_by_sop.get(sop_id, [])
        procedure_details.append({
            'sop_id': sop_id,
            'title': p.get('title', ''),
            'version': p.get('version', ''),
            'status': p.get('status', ''),
            'category': p.get('category', ''),
            'owner': p.get('owner', ''),
            'last_reviewed': p.get('last_reviewed', ''),
            'next_review_due': p.get('next_review_due', ''),
            'compliance_score': p.get('compliance_score', 0),
            'applicable_standards': p.get('applicable_standards', []),
            'approver': p.get('approver', ''),
            'revision_count': p.get('revision_count', 0),
            'audits': [
                {
                    'audit_id': a.get('audit_id', ''),
                    'audit_date': a.get('audit_date', ''),
                    'auditor': a.get('auditor', ''),
                    'finding_type': a.get('finding_type', ''),
                    'finding_description': a.get('finding_description', ''),
                    'corrective_action': a.get('corrective_action'),
                    'status': a.get('status', ''),
                    'severity': a.get('severity', ''),
                }
                for a in sorted(associated_audits, key=lambda x: x.get('audit_date', ''))
            ],
        })

    audit_details = []
    for a in sorted(audits, key=lambda x: x.get('audit_id', '')):
        audit_details.append({
            'audit_id': a.get('audit_id', ''),
            'sop_id': a.get('sop_id', ''),
            'audit_date': a.get('audit_date', ''),
            'auditor': a.get('auditor', ''),
            'finding_type': a.get('finding_type', ''),
            'finding_description': a.get('finding_description', ''),
            'corrective_action': a.get('corrective_action'),
            'status': a.get('status', ''),
            'severity': a.get('severity', ''),
        })

    return {
        'procedures': procedure_details,
        'audits': audit_details,
    }


def definitions():
    """Return IS/SOP terminology, compliance frameworks, and clinical security standards."""
    return {
        'concepts': [
            {
                'name': 'HIPAA Security Rule',
                'description': 'The Health Insurance Portability and Accountability Act (HIPAA) Security Rule establishes national standards for protecting electronic Protected Health Information (ePHI). It requires covered entities and business associates to implement administrative safeguards (security management, workforce training, contingency planning), physical safeguards (facility access controls, workstation security, device disposal), and technical safeguards (access control, audit controls, transmission security, integrity controls). For epilepsy monitoring units, this covers EEG recordings, patient demographics, seizure logs, and AI-assisted clinical decision support outputs.',
            },
            {
                'name': '21 CFR Part 11 (Electronic Records)',
                'description': 'FDA regulation governing electronic records and electronic signatures. Requires that electronic records are trustworthy, reliable, and equivalent to paper records. Key requirements include: audit trails that record who created/modified/deleted records and when, system validation to ensure accuracy and reliability, authority checks limiting system access to authorized individuals, and device checks to determine validity of data input sources. Applicable to AI model validation records, EEG analysis reports, and clinical decision support documentation in epilepsy monitoring.',
            },
            {
                'name': 'IEC 62443 (Industrial Cybersecurity)',
                'description': 'International standard series for cybersecurity of Industrial Automation and Control Systems (IACS), extended to medical device networks. Defines security levels (SL1-SL4) and zones/conduits for network segmentation. For clinical EEG environments, applies to: networked EEG amplifiers, video-EEG synchronization systems, bedside monitoring stations, and AI inference servers. Requires threat modeling, security risk assessments, and defense-in-depth architectures for medical device networks.',
            },
            {
                'name': 'NIST Cybersecurity Framework (CSF)',
                'description': 'A voluntary framework from the National Institute of Standards and Technology providing guidelines for managing cybersecurity risk. Organized into five core functions: Identify (asset management, risk assessment), Protect (access control, training, data security), Detect (anomaly detection, continuous monitoring), Respond (incident response, communications), and Recover (recovery planning, improvements). Healthcare organizations use NIST CSF to structure their security programs and demonstrate due diligence in protecting patient data and clinical systems.',
            },
            {
                'name': 'ISO 27001 for Healthcare',
                'description': 'International standard for Information Security Management Systems (ISMS). Provides a systematic approach to managing sensitive information through risk assessment and treatment. Annex A contains 114 controls across 14 domains including access control, cryptography, physical security, operations security, and supplier relationships. Healthcare implementations must additionally address medical device security, clinical workflow continuity, patient safety implications of security controls, and regulatory compliance mapping (HIPAA, HITECH, state laws).',
            },
            {
                'name': 'SOP Lifecycle',
                'description': 'The standard progression of a Standard Operating Procedure through defined stages: Draft (initial creation, content development, stakeholder input) -> Under Review (formal review by subject matter experts, compliance team, and department heads; comment resolution) -> Published (approved, effective, and enforceable; staff notified and trained) -> Retired (superseded by newer version or no longer applicable; archived for audit trail). Each transition requires documented approval and version control.',
            },
            {
                'name': 'Review Cadence',
                'description': 'The scheduled interval at which each SOP must be formally reviewed for continued accuracy and relevance. Standard cadences: High-risk SOPs (incident response, emergency access) = 90 days; Medium-risk SOPs (access control, data handling) = 180 days; Low-risk SOPs (training, vendor assessment) = 365 days. Overdue reviews are flagged as compliance gaps and tracked in audit findings. Reviews must be documented with reviewer name, date, and outcome (no changes / minor update / major revision).',
            },
            {
                'name': 'Compliance Score Methodology',
                'description': 'A composite score (0-100) assessing adherence to each SOP. Components weighted as follows: Documentation completeness (20%), Implementation evidence (30%), Audit history (20%), Training completion (15%), Review timeliness (15%). Scores below 70 trigger mandatory corrective action plans. Scores below 50 require immediate management escalation.',
            },
            {
                'name': 'Audit Finding Classifications',
                'description': 'Audit findings are categorized as: Compliant (fully meets SOP requirements, no corrective action), Minor Nonconformance (deviation not directly impacting patient safety or compliance, 30-90 day remediation), Major Nonconformance (significant deviation that could impact patient safety or regulatory compliance, 7-30 day remediation with root cause analysis), Observation (opportunity for improvement, no mandatory corrective action). Severity levels: Low (90-day window), Medium (30-day), High (14-day), Critical (same-day escalation, 7-day remediation).',
            },
            {
                'name': 'EEG Data Security',
                'description': 'Specific controls for protecting electroencephalography data throughout its lifecycle. Includes: encryption at rest (AES-256) and in transit (TLS 1.3), access logging for all EEG file access, de-identification procedures for research use (removing patient identifiers while preserving clinical utility), secure archival with integrity verification (checksums), and secure deletion when retention periods expire. EEG data is classified as ePHI under HIPAA and requires the full spectrum of Security Rule safeguards.',
            },
            {
                'name': 'AI Model Security',
                'description': 'Controls specific to machine learning models used in clinical decision support for epilepsy diagnosis. Includes: model versioning and provenance tracking, validation dataset security (preventing data leakage), inference API access controls, model output audit trails, adversarial robustness testing, bias monitoring across patient demographics, and secure model deployment pipelines. AI models that influence clinical decisions are subject to FDA SaMD (Software as a Medical Device) guidance and require documented validation protocols.',
            },
            {
                'name': 'Network Segmentation for Clinical Devices',
                'description': 'Architectural control separating clinical device networks from administrative and guest networks. EEG amplifiers, video-EEG systems, and bedside monitors are placed on isolated VLANs with strict firewall rules. Only authorized clinical workstations can access device networks. AI inference servers may reside in a separate compute zone with controlled API access. Network segmentation limits the blast radius of security incidents and prevents lateral movement by attackers between clinical and non-clinical systems.',
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN (first 3 procedures) ===')
    bd = breakdown()
    for p in bd['procedures'][:3]:
        pprint.pprint(p)
    print(f'\nTotal procedures: {len(bd["procedures"])}')
    print(f'Total audits: {len(bd["audits"])}')
    print('\n=== AUDIT DETAIL (first 3) ===')
    for a in bd['audits'][:3]:
        pprint.pprint(a)
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Concepts: {len(df["concepts"])}')
    for c in df['concepts']:
        print(f'  {c["name"]}')
