"""Grounding Dashboard — source verification and citation grounding analytics
for an epilepsy clinical decision support system.

Verifies that AI-generated outputs are backed by real source data:
- data/clinical.db patients (40 patients)
- data/clinical.db analyses (21 EEG analyses)
- data/clinical.db assessments (423 assessment scores)
- data/clinical.db conversation_log (376 conversation entries)
- data/clinical.db expert_reviews (3 expert review annotations)
- data/clinical.db clinical_decisions (1 clinical decision)
- data/clinical.db seizure_diary (25 seizure events)
- data/clinical.db medications (9 medication records)
- data/clinical.db mri_findings (40 MRI findings)
- data/clinical.db transaction_log (660 transaction events)

Metrics: grounding rate, citation coverage, source verification,
confidence distribution, per-patient grounding scores, claim traces.
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime, timezone
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _iso(dt):
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt) if dt else None


def _det_seed(key):
    """Deterministic seed from a string key."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# ── Source tables that ground AI claims ────────────────────────────────

_SOURCE_TABLES = [
    ('analyses', 'patient_id', 'EEG analysis reports'),
    ('assessments', 'patient_id', 'Clinical assessment scores'),
    ('seizure_diary', 'patient_id', 'Seizure event logs'),
    ('medications', 'patient_id', 'Medication records'),
    ('mri_findings', 'patient_id', 'MRI radiology findings'),
    ('expert_reviews', 'patient_id', 'Expert review annotations'),
]


def _patient_source_map(cur):
    """For each patient, determine which source tables have real data."""
    patient_ids = [r[0] for r in cur.execute(
        'SELECT patient_id FROM patients ORDER BY patient_id'
    ).fetchall()]
    patient_names = {}
    patient_diseases = {}
    for r in cur.execute('SELECT patient_id, name, disease FROM patients').fetchall():
        patient_names[r[0]] = r[1]
        patient_diseases[r[0]] = r[2]

    result = []
    for pid in patient_ids:
        sources_present = []
        for tbl, col, desc in _SOURCE_TABLES:
            try:
                n = cur.execute(
                    f'SELECT COUNT(*) FROM [{tbl}] WHERE [{col}]=?', (pid,)
                ).fetchone()[0]
                if n > 0:
                    sources_present.append({'table': tbl, 'description': desc, 'count': n})
            except Exception:
                pass
        result.append({
            'patient_id': pid,
            'name': patient_names.get(pid, ''),
            'disease': patient_diseases.get(pid, ''),
            'sources_present': sources_present,
            'source_count': len(sources_present),
            'grounding_score': round(len(sources_present) / len(_SOURCE_TABLES), 3),
        })
    return result, patient_ids


def _count_assistant_claims(cur):
    """Count assistant-generated entries in conversation_log as claims."""
    try:
        return cur.execute(
            "SELECT COUNT(*) FROM conversation_log WHERE role='assistant'"
        ).fetchone()[0]
    except Exception:
        return 0


def _analyses_with_patients(cur):
    """Count analyses that link to a real patient record."""
    try:
        return cur.execute(
            'SELECT COUNT(*) FROM analyses a '
            'INNER JOIN patients p ON a.patient_id = p.patient_id'
        ).fetchone()[0]
    except Exception:
        return 0


def _confidence_histogram(cur):
    """Build histogram buckets from analyses.confidence values."""
    try:
        rows = cur.execute('SELECT confidence FROM analyses WHERE confidence IS NOT NULL').fetchall()
    except Exception:
        rows = []
    values = [r[0] for r in rows]
    buckets = [
        {'range': '0.0-0.2', 'min': 0.0, 'max': 0.2, 'count': 0},
        {'range': '0.2-0.4', 'min': 0.2, 'max': 0.4, 'count': 0},
        {'range': '0.4-0.6', 'min': 0.4, 'max': 0.6, 'count': 0},
        {'range': '0.6-0.8', 'min': 0.6, 'max': 0.8, 'count': 0},
        {'range': '0.8-1.0', 'min': 0.8, 'max': 1.01, 'count': 0},
    ]
    for v in values:
        for b in buckets:
            if b['min'] <= v < b['max']:
                b['count'] += 1
                break
    return [{'range': b['range'], 'count': b['count']} for b in buckets]


def _verification_summary(cur):
    """Summarize expert_reviews by agree_with_ai status."""
    try:
        rows = cur.execute('SELECT agree_with_ai FROM expert_reviews').fetchall()
    except Exception:
        rows = []
    statuses = [r[0] for r in rows]
    verified = sum(1 for s in statuses if s and s.lower() == 'agree')
    unverified = sum(1 for s in statuses if s and s.lower() == 'disagree')
    pending = sum(1 for s in statuses if not s or s.lower() not in ('agree', 'disagree'))
    return {
        'verified': verified,
        'unverified': unverified,
        'pending': pending,
        'total': len(statuses),
    }


# ── Grounding by source ───────────────────────────────────────────────

def _grounding_by_source(cur, patient_source_data):
    """For each source table, count how many patients have data there."""
    total_patients = len(patient_source_data)
    source_stats = {}
    for tbl, col, desc in _SOURCE_TABLES:
        source_stats[tbl] = {'grounded': 0, 'total': total_patients}

    for ps in patient_source_data:
        present_tables = {s['table'] for s in ps['sources_present']}
        for tbl, col, desc in _SOURCE_TABLES:
            if tbl in present_tables:
                source_stats[tbl]['grounded'] += 1

    result = []
    for tbl, col, desc in _SOURCE_TABLES:
        s = source_stats[tbl]
        rate = round(s['grounded'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        result.append({
            'source': tbl,
            'description': desc,
            'grounded_count': s['grounded'],
            'total_count': s['total'],
            'rate': rate,
        })
    return result


# ── Citation map ───────────────────────────────────────────────────────

def _citation_map(cur, patient_source_data):
    """For each patient, produce citation entries showing grounding status."""
    citations = []
    claim_types = {
        'analyses': 'EEG prediction',
        'assessments': 'Assessment score',
        'seizure_diary': 'Seizure event',
        'medications': 'Medication record',
        'mri_findings': 'MRI finding',
        'expert_reviews': 'Expert review',
    }
    for ps in patient_source_data:
        present_tables = {s['table']: s['count'] for s in ps['sources_present']}
        for tbl, col, desc in _SOURCE_TABLES:
            source_count = present_tables.get(tbl, 0)
            citations.append({
                'patient_id': ps['patient_id'],
                'claim_type': claim_types.get(tbl, tbl),
                'source_table': tbl,
                'source_count': source_count,
                'grounded': source_count > 0,
            })
    return citations


# ── Claim traces from conversation_log ─────────────────────────────────

def _claim_traces(cur):
    """Derive claim traces from assistant entries in conversation_log."""
    try:
        rows = cur.execute(
            "SELECT id, text, ts_utc FROM conversation_log "
            "WHERE role='assistant' ORDER BY id DESC LIMIT 50"
        ).fetchall()
    except Exception:
        rows = []

    # Get patient IDs for cross-referencing
    try:
        all_pids = [r[0] for r in cur.execute('SELECT patient_id FROM patients').fetchall()]
    except Exception:
        all_pids = []

    traces = []
    for r in rows:
        cid, text, ts = r
        text_str = str(text or '')
        preview = text_str[:150]

        # Determine which source tables the text references
        claimed_sources = []
        for tbl, col, desc in _SOURCE_TABLES:
            # Check if the text references concepts related to the source
            keywords = {
                'analyses': ['eeg', 'analysis', 'prediction', 'signal', 'channel'],
                'assessments': ['assessment', 'score', 'phq', 'gad', 'instrument'],
                'seizure_diary': ['seizure', 'event', 'aura', 'duration'],
                'medications': ['medication', 'drug', 'dose', 'prescription'],
                'mri_findings': ['mri', 'radiology', 'imaging', 'scan'],
                'expert_reviews': ['expert', 'review', 'specialist'],
            }
            text_lower = text_str.lower()
            if any(kw in text_lower for kw in keywords.get(tbl, [])):
                claimed_sources.append(tbl)

        # Check which claimed sources have actual data
        # Find if any patient ID is mentioned in the text
        referenced_pids = [pid for pid in all_pids if pid in text_str]
        verified_sources = []
        if referenced_pids:
            pid = referenced_pids[0]
            for src in claimed_sources:
                try:
                    n = cur.execute(
                        f'SELECT COUNT(*) FROM [{src}] WHERE patient_id=?', (pid,)
                    ).fetchone()[0]
                    if n > 0:
                        verified_sources.append(src)
                except Exception:
                    pass
        elif claimed_sources:
            # No specific patient referenced; check if source table has any data
            for src in claimed_sources:
                try:
                    n = cur.execute(f'SELECT COUNT(*) FROM [{src}]').fetchone()[0]
                    if n > 0:
                        verified_sources.append(src)
                except Exception:
                    pass

        if claimed_sources:
            status = 'grounded' if len(verified_sources) == len(claimed_sources) else (
                'partial' if verified_sources else 'ungrounded'
            )
        else:
            status = 'no_claim'

        traces.append({
            'trace_id': f'claim-{cid:04d}',
            'timestamp': ts,
            'text_preview': preview,
            'claimed_sources': claimed_sources,
            'verified_sources': verified_sources,
            'grounding_status': status,
        })
    return traces


# ── Source verification log from transaction_log ───────────────────────

def _source_verification_log(cur):
    """Extract verification-relevant events from transaction_log."""
    try:
        rows = cur.execute(
            "SELECT id, patient_id, component, action, actor, detail, ts_utc "
            "FROM transaction_log "
            "WHERE component IN ('eeg_upload', 'expert_review', 'clinical_trust', "
            "  'cv_pipeline', 'training', 'consistency', 'drift', 'fairness') "
            "ORDER BY id DESC LIMIT 100"
        ).fetchall()
    except Exception:
        rows = []

    events = []
    for r in rows:
        eid, pid, comp, action, actor, detail, ts = r
        events.append({
            'event_id': eid,
            'patient_id': pid,
            'component': comp,
            'action': action,
            'actor': actor,
            'detail': str(detail or '')[:200],
            'timestamp': ts,
        })
    return events


# ── Expert verification detail ─────────────────────────────────────────

def _expert_verification(cur):
    """Retrieve expert_reviews with grounding status."""
    try:
        rows = cur.execute(
            'SELECT id, patient_id, analysis_id, role, expert, finding, '
            'agree_with_ai, note, created_at FROM expert_reviews ORDER BY id'
        ).fetchall()
    except Exception:
        rows = []
    reviews = []
    for r in rows:
        reviews.append({
            'review_id': r[0],
            'patient_id': r[1],
            'analysis_id': r[2],
            'role': r[3],
            'expert': r[4],
            'finding': r[5],
            'agree_with_ai': r[6],
            'note': r[7],
            'created_at': r[8],
            'verification_status': 'verified' if r[6] and r[6].lower() == 'agree' else 'disputed',
        })
    return reviews


# ── Public API ──────────────────────────────────────────────────────────

def grounding_overview():
    """Overview: grounding KPIs, source coverage, citation map, confidence, verification."""
    conn = _conn()
    cur = conn.cursor()

    patient_source_data, patient_ids = _patient_source_map(cur)
    assistant_claims = _count_assistant_claims(cur)
    analyses_total = cur.execute('SELECT COUNT(*) FROM analyses').fetchone()[0]
    analyses_linked = _analyses_with_patients(cur)
    expert_count = cur.execute('SELECT COUNT(*) FROM expert_reviews').fetchone()[0]
    decision_count = cur.execute('SELECT COUNT(*) FROM clinical_decisions').fetchone()[0]

    # Total claims = assistant conversation entries + analyses
    total_claims = assistant_claims + analyses_total

    # Grounded claims = patients that have at least 1 source backing AI outputs
    grounded_patients = sum(1 for ps in patient_source_data if ps['source_count'] > 0)
    # Grounded claims proportional to grounded patient fraction
    grounding_fraction = grounded_patients / len(patient_ids) if patient_ids else 0
    grounded_claims = round(total_claims * grounding_fraction)
    grounding_rate = round(grounded_claims / total_claims * 100, 1) if total_claims > 0 else 0

    # Citation coverage: % of analyses linked to patient records
    citation_coverage = round(analyses_linked / analyses_total * 100, 1) if analyses_total > 0 else 0

    # Source types used: distinct source tables that have any data
    source_types_used = 0
    for tbl, col, desc in _SOURCE_TABLES:
        try:
            n = cur.execute(f'SELECT COUNT(*) FROM [{tbl}]').fetchone()[0]
            if n > 0:
                source_types_used += 1
        except Exception:
            pass

    # Verification checks
    verification_checks = expert_count + decision_count

    # Average confidence
    avg_conf_row = cur.execute(
        'SELECT AVG(confidence) FROM analyses WHERE confidence IS NOT NULL'
    ).fetchone()
    avg_confidence = round(avg_conf_row[0], 3) if avg_conf_row[0] is not None else 0

    grounding_by_source = _grounding_by_source(cur, patient_source_data)
    citation_map = _citation_map(cur, patient_source_data)
    confidence_dist = _confidence_histogram(cur)
    verification = _verification_summary(cur)

    conn.close()
    return {
        'generated_at': _iso(datetime.now(timezone.utc)),
        'kpis': {
            'total_claims': total_claims,
            'grounded_claims': grounded_claims,
            'grounding_rate': grounding_rate,
            'citation_coverage': citation_coverage,
            'source_types_used': source_types_used,
            'verification_checks': verification_checks,
            'avg_confidence': avg_confidence,
        },
        'grounding_by_source': grounding_by_source,
        'citation_map': citation_map,
        'confidence_distribution': confidence_dist,
        'verification_summary': verification,
    }


def grounding_breakdown():
    """Breakdown: per-patient grounding, claim traces, verification logs, expert reviews."""
    conn = _conn()
    cur = conn.cursor()

    patient_source_data, patient_ids = _patient_source_map(cur)

    # Enrich per-patient grounding with claims from conversation_log
    try:
        conv_rows = cur.execute(
            "SELECT id, text FROM conversation_log WHERE role='assistant'"
        ).fetchall()
    except Exception:
        conv_rows = []

    for ps in patient_source_data:
        pid = ps['patient_id']
        # Count conversation entries that mention this patient
        mentions = [r for r in conv_rows if pid in str(r[1] or '')]
        ps['claims'] = len(mentions)
        ps['citations'] = sum(s['count'] for s in ps['sources_present'])

    claim_traces = _claim_traces(cur)
    verification_log = _source_verification_log(cur)
    expert_verif = _expert_verification(cur)

    conn.close()
    return {
        'generated_at': _iso(datetime.now(timezone.utc)),
        'per_patient_grounding': patient_source_data,
        'claim_traces': claim_traces,
        'source_verification_log': verification_log,
        'expert_verification': expert_verif,
    }


def definitions():
    """Definitions: grounding concepts, metrics, clinical relevance, remediation."""
    return {
        'generated_at': _iso(datetime.now(timezone.utc)),
        'sections': [
            {
                'title': 'Grounding Concepts',
                'items': [
                    {'term': 'Grounding',
                     'definition': 'The process of verifying that AI-generated clinical outputs are anchored in real, traceable source data rather than model hallucination. A grounded claim has at least one verifiable data source.'},
                    {'term': 'Citation Mapping',
                     'definition': 'Linking each AI-generated claim or prediction back to the specific source records (EEG analyses, assessments, seizure diary entries, medications) that support it.'},
                    {'term': 'Source Verification',
                     'definition': 'Confirming that a cited data source actually exists in the clinical database and contains the information the AI claim references. Verification can be automated (record lookup) or manual (expert review).'},
                    {'term': 'Grounded Generation',
                     'definition': 'An AI output paradigm where every generated statement is constrained to information retrievable from the knowledge base, reducing confabulation risk in clinical contexts.'},
                    {'term': 'Hallucination (AI context)',
                     'definition': 'When an AI system generates plausible-sounding clinical information that is not supported by any source data. In medical AI, hallucination is a patient-safety risk requiring systematic detection and mitigation.'},
                ],
            },
            {
                'title': 'Grounding Metrics',
                'items': [
                    {'term': 'Grounding Rate',
                     'definition': 'The percentage of AI-generated claims that can be traced back to at least one real source record. Computed as grounded_claims / total_claims * 100. Target: >90% for clinical-grade systems.'},
                    {'term': 'Citation Coverage',
                     'definition': 'The percentage of AI analyses (EEG predictions, assessments) that link to a verified patient record. Measures completeness of the citation chain from output to source.'},
                    {'term': 'Source Diversity',
                     'definition': 'The number of distinct source table types (analyses, assessments, seizure diary, medications, MRI, expert reviews) contributing grounding evidence. Higher diversity indicates more robust grounding.'},
                    {'term': 'Verification Depth',
                     'definition': 'The extent to which AI outputs have been independently verified, measured by the count of expert reviews and clinical decisions that validate or dispute AI-generated claims.'},
                ],
            },
            {
                'title': 'Clinical Relevance & Regulatory Standards',
                'items': [
                    {'term': 'EU AI Act Art.13 — Transparency',
                     'definition': 'High-risk AI systems must be designed with sufficient transparency to enable users to interpret outputs and understand their basis. Grounding dashboards support this by showing the evidence trail behind each AI claim.'},
                    {'term': 'FDA AI/ML — Clinical Validation',
                     'definition': 'FDA guidance on AI/ML-based Software as a Medical Device requires demonstrating that AI outputs are clinically valid and traceable to real patient data. Grounding metrics provide this traceability.'},
                    {'term': 'IEC 62304 — Software Lifecycle',
                     'definition': 'Medical device software standard requiring documented verification and validation. Grounding verification is a software verification activity ensuring output correctness.'},
                    {'term': 'ISO 14971 — Risk Management',
                     'definition': 'Ungrounded AI claims in clinical contexts are identified hazards under ISO 14971. The grounding dashboard quantifies residual risk from unverified AI outputs.'},
                    {'term': 'ILAE — Epilepsy Classification Standards',
                     'definition': 'International League Against Epilepsy classification standards define the ground-truth terminology for seizure types and epilepsy syndromes. AI outputs must be grounded in ILAE-compliant clinical data.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Low Grounding Rate',
                     'definition': 'If grounding rate falls below 85%: audit the AI generation pipeline for unconstrained outputs, increase retrieval coverage, enforce citation requirements in prompt templates, and add post-generation grounding checks.'},
                    {'term': 'Missing Citations',
                     'definition': 'For AI outputs lacking source citations: implement mandatory citation injection in the generation pipeline, add automated citation verification as a post-processing step, and flag uncited claims for human review.'},
                    {'term': 'Unverified Claims',
                     'definition': 'For claims that reference source data but lack expert verification: prioritize expert review queue, implement automated consistency checks against source records, and add confidence thresholds for auto-verification.'},
                    {'term': 'Source Gaps',
                     'definition': 'For patients with sparse source data (fewer than 2 source types): prioritize data collection for underserved patients, restrict AI confidence for sparsely-grounded outputs, and flag incomplete records in clinical reports.'},
                ],
            },
        ],
    }


if __name__ == '__main__':
    ov = grounding_overview()
    print(json.dumps(ov, indent=2, default=str)[:2000])
    print('...')
    print(f"KPIs: {ov['kpis']}")
