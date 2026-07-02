"""Knowledge Management Dashboard — knowledge lifecycle tracking:
create, approve, publish, expiry, archive from real clinical.db data.

Sources:
- conversation_log (382 rows) — knowledge items created from AI conversations
- analyses (21 rows) — analysis knowledge articles (published findings)
- assessments (423 rows) — assessment knowledge base (clinical instruments)
- expert_reviews (3 rows) — expert-approved/rejected knowledge items
- hitl_reviews (2 rows) — human-in-the-loop knowledge approvals
- transaction_log (666 rows) — lifecycle events (create/approve/publish/archive)
- uploads (21 rows) — uploaded knowledge artifacts (EDF files, reports)
- medications (9 rows) — medication knowledge entries
- mri_findings (40 rows) — imaging knowledge articles
- seizure_diary (25 rows) — patient-contributed knowledge
"""

import sqlite3
import os
from datetime import datetime, timezone, timedelta

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def _safe_query(cur, sql):
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


# ── Knowledge lifecycle stage classification ────────────────────
LIFECYCLE_STAGES = ['created', 'approved', 'published', 'expired', 'archived']

KNOWLEDGE_TYPES = [
    'Clinical Analysis',
    'Assessment Instrument',
    'Expert Review',
    'Medication Record',
    'Imaging Finding',
    'Patient Diary',
    'EEG Upload',
    'Conversation Knowledge',
]


def _classify_knowledge_stage(item_type, item):
    """Determine lifecycle stage based on data source and content."""
    if item_type == 'expert_review':
        if item.get('agree_with_ai') == 'agree':
            return 'approved'
        return 'created'
    elif item_type == 'hitl_review':
        return 'approved'
    elif item_type == 'analysis':
        conf = item.get('confidence') or 0
        if conf >= 0.7:
            return 'published'
        elif conf >= 0.5:
            return 'approved'
        return 'created'
    elif item_type == 'assessment':
        if item.get('alert') and str(item.get('alert')).lower() in ('yes', '1', 'true'):
            return 'published'
        return 'approved'
    elif item_type == 'upload':
        return 'published'
    elif item_type == 'medication':
        return 'published'
    elif item_type == 'mri_finding':
        return 'published'
    elif item_type == 'seizure_diary':
        return 'created'
    elif item_type == 'conversation':
        return 'created'
    return 'created'


def _build_knowledge_register(cur):
    """Build a unified knowledge register from all data sources."""
    register = []

    # Analyses → Clinical Analysis knowledge
    analyses = _safe_query(cur,
        "SELECT id, patient_id, disease, predicted_label, confidence, "
        "created_at FROM analyses")
    for a in analyses:
        stage = _classify_knowledge_stage('analysis', a)
        register.append({
            "id": f"KA-{a['id']:04d}",
            "type": "Clinical Analysis",
            "source_table": "analyses",
            "source_id": a['id'],
            "patient_id": a.get('patient_id'),
            "title": f"{a.get('disease', 'Unknown')} analysis — {a.get('predicted_label', 'N/A')}",
            "stage": stage,
            "confidence": a.get('confidence'),
            "created_at": a.get('created_at'),
        })

    # Assessments → Assessment Instrument knowledge
    assessments = _safe_query(cur,
        "SELECT id, patient_id, instrument, score, max_score, interpretation, "
        "level, alert, examiner, created_at FROM assessments")
    for a in assessments:
        stage = _classify_knowledge_stage('assessment', a)
        register.append({
            "id": f"KI-{a['id']:04d}",
            "type": "Assessment Instrument",
            "source_table": "assessments",
            "source_id": a['id'],
            "patient_id": a.get('patient_id'),
            "title": f"{a.get('instrument', 'Unknown')} — {a.get('interpretation', 'N/A')}",
            "stage": stage,
            "confidence": (a.get('score') or 0) / max(a.get('max_score') or 1, 1),
            "created_at": a.get('created_at'),
        })

    # Expert reviews → Expert Review knowledge
    expert = _safe_query(cur,
        "SELECT id, patient_id, role, expert, finding, agree_with_ai, "
        "note, created_at FROM expert_reviews")
    for e in expert:
        stage = _classify_knowledge_stage('expert_review', e)
        register.append({
            "id": f"KE-{e['id']:04d}",
            "type": "Expert Review",
            "source_table": "expert_reviews",
            "source_id": e['id'],
            "patient_id": e.get('patient_id'),
            "title": f"Expert {e.get('expert', 'Unknown')}: {(e.get('finding') or '')[:60]}",
            "stage": stage,
            "confidence": 1.0 if e.get('agree_with_ai') == 'agree' else 0.5,
            "created_at": e.get('created_at'),
        })

    # Uploads → EEG Upload knowledge
    uploads = _safe_query(cur,
        "SELECT id, patient_id, file_name, disease, department, "
        "created_at FROM uploads")
    for u in uploads:
        stage = _classify_knowledge_stage('upload', u)
        register.append({
            "id": f"KU-{u['id']:04d}",
            "type": "EEG Upload",
            "source_table": "uploads",
            "source_id": u['id'],
            "patient_id": u.get('patient_id'),
            "title": f"Upload: {u.get('file_name', 'Unknown')} ({u.get('disease', '')})",
            "stage": stage,
            "confidence": None,
            "created_at": u.get('created_at'),
        })

    # Medications → Medication Record knowledge (fields_json schema)
    medications = _safe_query(cur,
        "SELECT id, patient_id, fields_json, created_at FROM medications")
    for m in medications:
        import json as _json
        fields = {}
        try:
            fields = _json.loads(m.get('fields_json') or '{}')
        except Exception:
            pass
        drug = fields.get('drug_name', 'Unknown')
        dose = fields.get('dose_mg', '')
        stage = _classify_knowledge_stage('medication', m)
        register.append({
            "id": f"KM-{m['id']:04d}",
            "type": "Medication Record",
            "source_table": "medications",
            "source_id": m['id'],
            "patient_id": m.get('patient_id'),
            "title": f"Medication: {drug} {dose}mg" if dose else f"Medication: {drug}",
            "stage": stage,
            "confidence": None,
            "created_at": m.get('created_at'),
        })

    # MRI findings → Imaging Finding knowledge (fields_json schema)
    mri = _safe_query(cur,
        "SELECT id, patient_id, fields_json, created_at FROM mri_findings")
    for m in mri:
        fields = {}
        try:
            fields = _json.loads(m.get('fields_json') or '{}')
        except Exception:
            pass
        region = fields.get('lesion_location', 'Unknown')
        finding = fields.get('lesion_description', fields.get('lesion_label', ''))
        stage = _classify_knowledge_stage('mri_finding', m)
        register.append({
            "id": f"KF-{m['id']:04d}",
            "type": "Imaging Finding",
            "source_table": "mri_findings",
            "source_id": m['id'],
            "patient_id": m.get('patient_id'),
            "title": f"MRI {region}: {(finding or '')[:50]}",
            "stage": stage,
            "confidence": None,
            "created_at": m.get('created_at'),
        })

    # Seizure diary → Patient Diary knowledge
    diary = _safe_query(cur,
        "SELECT id, patient_id, event_date, duration_sec, severity "
        "FROM seizure_diary")
    for d in diary:
        stage = _classify_knowledge_stage('seizure_diary', d)
        register.append({
            "id": f"KD-{d['id']:04d}",
            "type": "Patient Diary",
            "source_table": "seizure_diary",
            "source_id": d['id'],
            "patient_id": d.get('patient_id'),
            "title": f"Seizure event: {d.get('severity', 'Unknown')} ({d.get('event_date', '')})",
            "stage": stage,
            "confidence": None,
            "created_at": d.get('event_date'),
        })

    # HITL reviews → approved knowledge (add to existing items by tagging)
    hitl = _safe_query(cur,
        "SELECT id, patient_id, analysis_id, created_at FROM hitl_reviews")
    for h in hitl:
        register.append({
            "id": f"KH-{h['id']:04d}",
            "type": "Expert Review",
            "source_table": "hitl_reviews",
            "source_id": h['id'],
            "patient_id": h.get('patient_id'),
            "title": f"HITL review for analysis #{h.get('analysis_id', '?')}",
            "stage": "approved",
            "confidence": 1.0,
            "created_at": h.get('created_at'),
        })

    return register


def _lifecycle_events_from_transactions(cur):
    """Extract knowledge lifecycle events from transaction_log."""
    action_to_stage = {
        'create': 'created',
        'ingest': 'created',
        'submit': 'created',
        'add': 'created',
        'sign-off': 'approved',
        'human_decision': 'approved',
        'analyze': 'published',
        'process': 'published',
        'extract': 'published',
        'build': 'published',
        'delete': 'archived',
        'blocked': 'expired',
    }

    events = _safe_query(cur,
        "SELECT id, patient_id, component, action, actor, detail, ts_utc "
        "FROM transaction_log ORDER BY ts_utc DESC")

    lifecycle = []
    for e in events:
        action = (e.get('action') or '').lower()
        stage = action_to_stage.get(action)
        if stage:
            lifecycle.append({
                "event_id": e['id'],
                "patient_id": e.get('patient_id'),
                "component": e.get('component'),
                "action": e.get('action'),
                "stage": stage,
                "actor": e.get('actor'),
                "detail": (e.get('detail') or '')[:100],
                "timestamp": e.get('ts_utc'),
            })

    return lifecycle


def knowledge_overview():
    """Aggregate knowledge management KPIs."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    register = _build_knowledge_register(cur)
    lifecycle = _lifecycle_events_from_transactions(cur)

    # Stage distribution
    stage_counts = {}
    for item in register:
        s = item.get('stage', 'created')
        stage_counts[s] = stage_counts.get(s, 0) + 1

    # Type distribution
    type_counts = {}
    for item in register:
        t = item.get('type', 'Unknown')
        type_counts[t] = type_counts.get(t, 0) + 1

    # Total knowledge items
    total_items = len(register)

    # Published rate
    published = stage_counts.get('published', 0)
    approved = stage_counts.get('approved', 0)
    publish_rate = round(published / max(total_items, 1) * 100, 1)

    # Approval rate
    approval_rate = round((approved + published) / max(total_items, 1) * 100, 1)

    # Unique patients with knowledge
    patients_with_knowledge = len(set(
        item.get('patient_id') for item in register
        if item.get('patient_id')))

    # Average confidence (where available)
    confs = [item['confidence'] for item in register
             if item.get('confidence') is not None]
    avg_confidence = round(sum(confs) / max(len(confs), 1), 3)

    # Lifecycle events count
    total_lifecycle_events = len(lifecycle)

    # Daily activity trend from transaction_log
    daily_activity = {}
    for e in lifecycle:
        ts = e.get('timestamp') or ''
        day = ts[:10] if len(ts) >= 10 else 'unknown'
        daily_activity[day] = daily_activity.get(day, 0) + 1
    activity_trend = [{"date": d, "events": c}
                      for d, c in sorted(daily_activity.items())
                      if d != 'unknown']

    # Stage distribution for chart
    stage_distribution = [{"stage": s, "count": c}
                          for s, c in sorted(stage_counts.items())]

    # Type distribution for chart
    type_distribution = [{"type": t, "count": c}
                         for t, c in sorted(type_counts.items())]

    # Knowledge source breakdown (by source table)
    source_counts = {}
    for item in register:
        src = item.get('source_table', 'unknown')
        source_counts[src] = source_counts.get(src, 0) + 1
    source_breakdown = [{"source": s, "count": c}
                        for s, c in sorted(source_counts.items())]

    conn.close()

    return {
        "available": True,
        "total_knowledge_items": total_items,
        "published_count": published,
        "approved_count": approved,
        "created_count": stage_counts.get('created', 0),
        "expired_count": stage_counts.get('expired', 0),
        "archived_count": stage_counts.get('archived', 0),
        "publish_rate_pct": publish_rate,
        "approval_rate_pct": approval_rate,
        "patients_with_knowledge": patients_with_knowledge,
        "avg_confidence": avg_confidence,
        "total_lifecycle_events": total_lifecycle_events,
        "knowledge_types_count": len(type_counts),
        "stage_distribution": stage_distribution,
        "type_distribution": type_distribution,
        "source_breakdown": source_breakdown,
        "activity_trend": activity_trend,
    }


def knowledge_breakdown():
    """Detailed knowledge register + lifecycle event log."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    register = _build_knowledge_register(cur)
    lifecycle = _lifecycle_events_from_transactions(cur)

    # Per-patient knowledge profile
    patient_map = {}
    for item in register:
        pid = item.get('patient_id') or 'unknown'
        if pid not in patient_map:
            patient_map[pid] = {"patient_id": pid, "items": 0,
                                "types": set(), "stages": set()}
        patient_map[pid]["items"] += 1
        patient_map[pid]["types"].add(item.get('type', 'Unknown'))
        patient_map[pid]["stages"].add(item.get('stage', 'created'))

    patient_profiles = []
    for pid, prof in sorted(patient_map.items()):
        patient_profiles.append({
            "patient_id": prof["patient_id"],
            "total_items": prof["items"],
            "types": sorted(prof["types"]),
            "stages": sorted(prof["stages"]),
        })

    # Lifecycle stage flow (from → to transition counts)
    stage_order = {s: i for i, s in enumerate(LIFECYCLE_STAGES)}
    stage_flow = {}
    for e in lifecycle:
        stage = e.get('stage', 'created')
        action = e.get('action', '')
        key = f"{action} → {stage}"
        stage_flow[key] = stage_flow.get(key, 0) + 1
    flow_entries = [{"transition": k, "count": v}
                    for k, v in sorted(stage_flow.items(),
                                       key=lambda x: -x[1])]

    # Top knowledge register entries (first 100)
    register_sample = register[:100]

    # Recent lifecycle events (first 50)
    recent_lifecycle = lifecycle[:50]

    conn.close()

    return {
        "available": True,
        "knowledge_register": register_sample,
        "patient_profiles": patient_profiles,
        "lifecycle_events": recent_lifecycle,
        "stage_flow": flow_entries,
        "total_register_items": len(register),
        "total_lifecycle_events": len(lifecycle),
    }


def knowledge_definitions():
    """Knowledge Management metric definitions + compliance references."""
    return {
        "concepts": [
            {
                "term": "Knowledge Article",
                "definition": "A discrete unit of clinical or AI-generated knowledge "
                              "(analysis, assessment, expert finding, imaging result) "
                              "tracked through a lifecycle from creation to archival.",
            },
            {
                "term": "Knowledge Lifecycle",
                "definition": "Five-stage pipeline: Created → Approved → Published → "
                              "Expired → Archived. Each stage has governance requirements.",
            },
            {
                "term": "Publish Rate",
                "definition": "Percentage of knowledge items that have reached "
                              "'published' stage, meaning they are validated and "
                              "available for clinical use.",
            },
            {
                "term": "Approval Rate",
                "definition": "Percentage of items that have been approved or published, "
                              "reflecting expert review coverage.",
            },
            {
                "term": "Knowledge Register",
                "definition": "Unified catalog of all knowledge items across sources — "
                              "analyses, assessments, expert reviews, uploads, medication "
                              "records, imaging findings, patient diary entries.",
            },
            {
                "term": "Lifecycle Event",
                "definition": "A transaction_log entry mapped to a knowledge stage "
                              "transition (create, sign-off/approve, process/publish, "
                              "delete/archive, blocked/expire).",
            },
            {
                "term": "Knowledge Expiry",
                "definition": "When a knowledge item becomes outdated or is blocked — "
                              "requires re-validation before returning to published state.",
            },
        ],
        "metrics": [
            {
                "name": "Total Knowledge Items",
                "description": "Count of all knowledge articles across all source tables.",
            },
            {
                "name": "Publish Rate %",
                "description": "published / total × 100 — target ≥ 60%.",
            },
            {
                "name": "Approval Rate %",
                "description": "(approved + published) / total × 100 — target ≥ 75%.",
            },
            {
                "name": "Avg Confidence",
                "description": "Mean confidence score of knowledge items that have one.",
            },
            {
                "name": "Patients with Knowledge",
                "description": "Unique patients whose data contributes to the knowledge base.",
            },
        ],
        "compliance": [
            "EU AI Act Art. 13 — Transparency: knowledge provenance and lifecycle must be traceable",
            "EU AI Act Art. 14 — Human Oversight: approval stage ensures expert review",
            "FDA AI/ML SaMD — Post-market surveillance via knowledge lifecycle monitoring",
            "ISO 14971 — Risk management: expired/archived knowledge signals safety review",
            "IEC 62304 — Software lifecycle: knowledge articles map to validated outputs",
            "NIST AI RMF — Govern 1.3: information integrity through managed knowledge lifecycle",
            "HIPAA § 164.530 — Documentation retention: archived knowledge meets retention rules",
        ],
        "remediation": [
            "Low publish rate → increase expert review capacity or lower approval threshold "
            "for low-risk items.",
            "High expiry rate → investigate root causes (model drift, data quality) and "
            "retrain or re-validate.",
            "Missing patient coverage → ensure all active patients have at least one "
            "knowledge article created.",
            "Stale knowledge → implement automated expiry checks (e.g., 90-day re-validation "
            "for high-risk articles).",
        ],
    }
