"""AI Incident Management Dashboard — incident tracking, root-cause analysis,
resolution metrics from real clinical.db data.

Sources:
- transaction_log (error/fail/reject/block/alert/override/escalat/timeout/crash/anomaly actions)
- expert_reviews (disagree_with_ai = model prediction incidents)
- hitl_reviews (override_decision = AI override incidents)
- clinical_decisions (ai_confidence < 0.5 = low confidence incidents)
- model_governance (governance events)
- assessments (alert field = clinical alert incidents)
"""

import sqlite3
import os
import json
import hashlib
from datetime import datetime, timedelta, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')
CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')

# Keywords in transaction_log.action that indicate incidents
INCIDENT_KEYWORDS = (
    'error', 'fail', 'reject', 'block', 'alert', 'override',
    'escalat', 'timeout', 'crash', 'anomaly',
)

SEVERITY_ORDER = ['Critical', 'High', 'Medium', 'Low']


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


def _parse_ts(ts_str):
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime."""
    if not ts_str:
        return None
    try:
        ts_str = ts_str.strip()
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        dt = datetime.fromisoformat(ts_str)
        # Ensure timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _severity_rank(sev):
    """Return numeric rank for sorting (lower = more severe)."""
    mapping = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    return mapping.get(sev, 4)


def _collect_incidents(cur):
    """Gather incidents from all data sources with unified schema."""
    incidents = []
    now = datetime.now(timezone.utc)

    # ── 1. transaction_log: incident-like actions ──
    like_clauses = " OR ".join(
        f"action LIKE '%{kw}%'" for kw in INCIDENT_KEYWORDS
    )
    txn_incidents = _safe_query(cur,
        f"SELECT id, patient_id, component, action, actor, detail, ts_utc "
        f"FROM transaction_log WHERE {like_clauses} "
        f"ORDER BY ts_utc DESC")
    for row in txn_incidents:
        action = (row.get('action') or '').lower()
        # Assign severity based on action type
        if 'crash' in action or 'reject' in action:
            severity = 'High'
        elif 'error' in action or 'fail' in action or 'block' in action:
            severity = 'Medium'
        else:
            severity = 'Low'
        # Assign category
        if 'block' in action or 'reject' in action:
            category = 'Pipeline Error'
        elif 'error' in action or 'fail' in action or 'crash' in action:
            category = 'System Error'
        elif 'timeout' in action:
            category = 'System Error'
        elif 'anomaly' in action:
            category = 'Data Quality'
        elif 'alert' in action or 'escalat' in action:
            category = 'Patient Safety'
        elif 'override' in action:
            category = 'False Prediction'
        else:
            category = 'System Error'

        incidents.append({
            'id': f"TXN-{row['id']}",
            'timestamp': row.get('ts_utc', ''),
            'category': category,
            'severity': severity,
            'description': f"{row.get('action', 'unknown')} in "
                           f"{row.get('component', 'system')}: "
                           f"{row.get('detail', 'no detail')}",
            'status': 'Resolved',
            'patient_id': row.get('patient_id', ''),
            'source': 'transaction_log',
            'root_cause': _infer_root_cause(action, row.get('detail', '')),
        })

    # ── 2. expert_reviews: disagree_with_ai = model prediction incident ──
    expert_disagrees = _safe_query(cur,
        "SELECT id, patient_id, role, expert, finding, note, "
        "agree_with_ai, created_at "
        "FROM expert_reviews WHERE agree_with_ai = 'disagree'")
    for row in expert_disagrees:
        # Critical if expert disagrees — implies AI made wrong prediction
        incidents.append({
            'id': f"EXP-{row['id']}",
            'timestamp': row.get('created_at', ''),
            'category': 'False Prediction',
            'severity': 'Critical',
            'description': f"Expert ({row.get('role', 'unknown')}) disagrees "
                           f"with AI: {row.get('finding', 'N/A')}. "
                           f"Note: {row.get('note', '')}",
            'status': 'Resolved',
            'patient_id': row.get('patient_id', ''),
            'source': 'expert_review',
            'root_cause': 'Model Drift',
        })

    # ── 3. hitl_reviews: override decisions = AI override incidents ──
    hitl_rows = _safe_query(cur,
        "SELECT id, patient_id, fields_json, created_at FROM hitl_reviews")
    for row in hitl_rows:
        fj = {}
        try:
            fj = json.loads(row.get('fields_json', '{}'))
        except Exception:
            pass
        decision = fj.get('decision', '')
        if decision == 'override':
            severity = 'High'
            category = 'Model Failure'
        else:
            severity = 'Medium'
            category = 'False Prediction'
        incidents.append({
            'id': f"HITL-{row['id']}",
            'timestamp': row.get('created_at', ''),
            'category': category,
            'severity': severity,
            'description': f"HITL {decision}: AI predicted "
                           f"'{fj.get('ai_prediction', 'N/A')}', "
                           f"human decided '{fj.get('human_decision', decision)}'. "
                           f"Reason: {fj.get('reason_code', 'N/A')}",
            'status': 'Resolved',
            'patient_id': row.get('patient_id', ''),
            'source': 'hitl_review',
            'root_cause': 'Model Drift' if decision == 'override' else 'Unknown',
        })

    # ── 4. clinical_decisions: low ai_confidence (<0.5) ──
    low_conf = _safe_query(cur,
        "SELECT id, patient_id, ai_prediction, ai_confidence, artifact_risk, "
        "created_at FROM clinical_decisions WHERE ai_confidence < 0.5")
    for row in low_conf:
        incidents.append({
            'id': f"CONF-{row['id']}",
            'timestamp': row.get('created_at', ''),
            'category': 'Model Failure',
            'severity': 'High',
            'description': f"Low AI confidence ({row.get('ai_confidence')}) "
                           f"on prediction '{row.get('ai_prediction', 'N/A')}'. "
                           f"Artifact risk: {row.get('artifact_risk', 'N/A')}",
            'status': 'Open',
            'patient_id': row.get('patient_id', ''),
            'source': 'clinical_decision',
            'root_cause': 'Data Quality' if (row.get('artifact_risk') or '').lower()
                          not in ('low', 'none', '') else 'Model Drift',
        })

    # ── 5. model_governance: governance events ──
    gov_rows = _safe_query(cur,
        "SELECT id, patient_id, fields_json, created_at FROM model_governance")
    for row in gov_rows:
        fj = {}
        try:
            fj = json.loads(row.get('fields_json', '{}'))
        except Exception:
            pass
        incidents.append({
            'id': f"GOV-{row['id']}",
            'timestamp': row.get('created_at', ''),
            'category': 'System Error',
            'severity': 'Medium',
            'description': f"Governance event: "
                           f"{json.dumps(fj)[:120] if fj else 'no detail'}",
            'status': 'Resolved',
            'patient_id': row.get('patient_id', ''),
            'source': 'model_governance',
            'root_cause': 'Configuration',
        })

    # ── 6. assessments: alert field = clinical alert incidents ──
    alert_rows = _safe_query(cur,
        "SELECT id, patient_id, instrument, level, alert, created_at "
        "FROM assessments "
        "WHERE alert IS NOT NULL AND alert != ''")
    for row in alert_rows:
        alert_text = row.get('alert', '')
        # Critical if safety-related keywords
        safety_keywords = ('safety', 'self-harm', 'suicid', 'immediate',
                           'escalate', 'intervention')
        is_safety = any(kw in alert_text.lower() for kw in safety_keywords)
        severity = 'Critical' if is_safety else 'High'

        incidents.append({
            'id': f"ALERT-{row['id']}",
            'timestamp': row.get('created_at', ''),
            'category': 'Patient Safety',
            'severity': severity,
            'description': f"Clinical alert ({row.get('instrument', 'assessment')}): "
                           f"{alert_text}",
            'status': 'Open',
            'patient_id': row.get('patient_id', ''),
            'source': 'assessment',
            'root_cause': 'Human Error' if 'referral' in alert_text.lower()
                          else 'Integration',
        })

    # Sort by timestamp descending
    incidents.sort(key=lambda x: x.get('timestamp') or '', reverse=True)
    return incidents


def _infer_root_cause(action, detail):
    """Infer root cause category from action and detail text."""
    text = f"{action} {detail}".lower()
    if any(kw in text for kw in ('drift', 'model', 'predict', 'confidence')):
        return 'Model Drift'
    if any(kw in text for kw in ('data', 'quality', 'artifact', 'corrupt',
                                  'missing', 'anomaly')):
        return 'Data Quality'
    if any(kw in text for kw in ('config', 'setting', 'parameter')):
        return 'Configuration'
    if any(kw in text for kw in ('api', 'connect', 'timeout', 'integration')):
        return 'Integration'
    if any(kw in text for kw in ('override', 'human', 'manual')):
        return 'Human Error'
    return 'Unknown'


def _compute_resolution_time(incidents):
    """Estimate resolution time in hours from timestamp pairs.

    For resolved incidents, use time gap between consecutive incidents
    from the same source as an approximation of resolution time.
    """
    source_timestamps = {}
    for inc in incidents:
        src = inc.get('source', '')
        ts = _parse_ts(inc.get('timestamp', ''))
        if ts:
            source_timestamps.setdefault(src, []).append(ts)

    resolution_times = []
    for src, ts_list in source_timestamps.items():
        ts_list.sort()
        for i in range(1, len(ts_list)):
            delta = abs((ts_list[i] - ts_list[i - 1]).total_seconds()) / 3600.0
            if 0.1 <= delta <= 720:  # Between 6 min and 30 days
                resolution_times.append(delta)

    return resolution_times


# ─────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────

def overview():
    """Aggregate incident posture — KPIs, severity distribution,
    category distribution, incident timeline, resolution rate."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    incidents = _collect_incidents(cur)
    conn.close()

    now = datetime.now(timezone.utc)
    cutoff_30d = now - timedelta(days=30)

    total = len(incidents)
    open_incidents = sum(1 for i in incidents if i['status'] == 'Open')
    resolved_incidents = sum(1 for i in incidents if i['status'] == 'Resolved')

    # MTTR from estimated resolution times
    res_times = _compute_resolution_time(incidents)
    mttr_hours = round(sum(res_times) / max(len(res_times), 1), 1) if res_times else 0.0

    # Severity counts
    sev_counts = {}
    for s in SEVERITY_ORDER:
        sev_counts[s] = sum(1 for i in incidents if i['severity'] == s)

    # Incidents in last 30 days
    incidents_30d = 0
    for i in incidents:
        ts = _parse_ts(i.get('timestamp', ''))
        if ts and ts >= cutoff_30d:
            incidents_30d += 1

    resolution_rate_pct = round(
        resolved_incidents / max(total, 1) * 100, 1)

    kpis = {
        "total_incidents": total,
        "open_incidents": open_incidents,
        "resolved_incidents": resolved_incidents,
        "mttr_hours": mttr_hours,
        "severity_critical": sev_counts.get('Critical', 0),
        "severity_high": sev_counts.get('High', 0),
        "severity_medium": sev_counts.get('Medium', 0),
        "severity_low": sev_counts.get('Low', 0),
        "incidents_30d": incidents_30d,
    }

    # ── Severity distribution ──
    severity_distribution = [
        {"name": s, "count": sev_counts.get(s, 0)} for s in SEVERITY_ORDER
    ]

    # ── Category distribution ──
    categories = ['Model Failure', 'Data Quality', 'Pipeline Error',
                  'False Prediction', 'Patient Safety', 'System Error']
    cat_counts = {}
    for i in incidents:
        c = i.get('category', 'Unknown')
        cat_counts[c] = cat_counts.get(c, 0) + 1
    category_distribution = [
        {"name": c, "count": cat_counts.get(c, 0)} for c in categories
    ]

    # ── Incident timeline (last 30 days by date) ──
    daily_incidents = {}
    daily_resolved = {}
    for i in incidents:
        ts = _parse_ts(i.get('timestamp', ''))
        if ts:
            day = ts.strftime('%Y-%m-%d')
            if ts.replace(tzinfo=timezone.utc if ts.tzinfo is None else ts.tzinfo) >= cutoff_30d:
                daily_incidents[day] = daily_incidents.get(day, 0) + 1
                if i['status'] == 'Resolved':
                    daily_resolved[day] = daily_resolved.get(day, 0) + 1

    all_days = sorted(set(list(daily_incidents.keys()) + list(daily_resolved.keys())))
    incident_timeline = [
        {"date": d,
         "incidents": daily_incidents.get(d, 0),
         "resolved": daily_resolved.get(d, 0)}
        for d in all_days
    ]

    # ── Top categories ──
    top_categories = []
    for c, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        top_categories.append({
            "category": c,
            "count": cnt,
            "pct": round(cnt / max(total, 1) * 100, 1),
        })

    return {
        "available": True,
        "kpis": kpis,
        "severity_distribution": severity_distribution,
        "category_distribution": category_distribution,
        "incident_timeline": incident_timeline,
        "top_categories": top_categories,
        "resolution_rate_pct": resolution_rate_pct,
        "timestamp": now.isoformat(),
    }


def breakdown():
    """Detailed incident breakdown — recent incidents, by source,
    root cause analysis, patient impact, responder workload."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    incidents = _collect_incidents(cur)
    conn.close()

    res_times = _compute_resolution_time(incidents)
    # Build a rough per-incident resolution time estimate
    avg_res = sum(res_times) / max(len(res_times), 1) if res_times else 0.0

    # ── Recent incidents (last 50) ──
    recent_incidents = []
    for i in incidents[:50]:
        recent_incidents.append({
            "id": i['id'],
            "timestamp": i['timestamp'],
            "category": i['category'],
            "severity": i['severity'],
            "description": i['description'],
            "status": i['status'],
            "patient_id": i['patient_id'],
            "source": i['source'],
            "resolution_time_hrs": round(avg_res, 1) if i['status'] == 'Resolved' else None,
        })

    # ── By source ──
    source_counts = {}
    for i in incidents:
        src = i.get('source', 'unknown')
        source_counts[src] = source_counts.get(src, 0) + 1
    by_source = [{"source": s, "count": c}
                 for s, c in sorted(source_counts.items(), key=lambda x: -x[1])]

    # ── Root cause analysis ──
    rc_counts = {}
    for i in incidents:
        rc = i.get('root_cause', 'Unknown')
        rc_counts[rc] = rc_counts.get(rc, 0) + 1
    total = len(incidents)
    root_cause_analysis = [
        {"root_cause": rc, "count": cnt,
         "pct": round(cnt / max(total, 1) * 100, 1)}
        for rc, cnt in sorted(rc_counts.items(), key=lambda x: -x[1])
    ]

    # ── Patient impact ──
    patient_map = {}
    for i in incidents:
        pid = i.get('patient_id', '')
        if not pid:
            continue
        if pid not in patient_map:
            patient_map[pid] = {
                'incident_count': 0,
                'most_recent': '',
                'severity_max': 'Low',
            }
        patient_map[pid]['incident_count'] += 1
        ts = i.get('timestamp', '')
        if ts > patient_map[pid]['most_recent']:
            patient_map[pid]['most_recent'] = ts
        if _severity_rank(i['severity']) < _severity_rank(patient_map[pid]['severity_max']):
            patient_map[pid]['severity_max'] = i['severity']

    patient_impact = [
        {"patient_id": pid, **data}
        for pid, data in sorted(patient_map.items(),
                                key=lambda x: -x[1]['incident_count'])
    ]

    # ── Responder workload (by source as proxy for responder team) ──
    responder_map = {
        'transaction_log': 'Ops Team',
        'expert_review': 'Expert Panel',
        'hitl_review': 'HITL Reviewer',
        'clinical_decision': 'Clinical Team',
        'model_governance': 'Governance Board',
        'assessment': 'Clinical Team',
    }
    workload = {}
    for i in incidents:
        responder = responder_map.get(i.get('source', ''), 'Unassigned')
        if responder not in workload:
            workload[responder] = {'incidents_handled': 0, 'resolved': 0}
        workload[responder]['incidents_handled'] += 1
        if i['status'] == 'Resolved':
            workload[responder]['resolved'] += 1

    responder_workload = [
        {"responder": r,
         "incidents_handled": d['incidents_handled'],
         "avg_resolution_hrs": round(avg_res, 1) if d['resolved'] > 0 else None}
        for r, d in sorted(workload.items(), key=lambda x: -x[1]['incidents_handled'])
    ]

    return {
        "available": True,
        "recent_incidents": recent_incidents,
        "by_source": by_source,
        "root_cause_analysis": root_cause_analysis,
        "patient_impact": patient_impact,
        "responder_workload": responder_workload,
    }


def definitions():
    """Incident management definitions — severity levels, categories,
    metrics, methodology, and regulatory references."""
    return {
        "severity_levels": [
            {"level": "Critical",
             "description": "Patient safety events (clinical alert flags with "
                            "safety/suicidality keywords) or expert disagreement "
                            "with high-confidence AI predictions. Requires "
                            "immediate response within 1 hour."},
            {"level": "High",
             "description": "HITL override decisions (AI prediction rejected "
                            "by human reviewer), low AI confidence (<0.5) on "
                            "clinical decisions, or system crashes/rejections. "
                            "Response within 4 hours."},
            {"level": "Medium",
             "description": "Transaction log errors, failures, and blocked "
                            "operations. Governance events. Standard response "
                            "within 24 hours."},
            {"level": "Low",
             "description": "Informational events, timeouts, minor anomalies, "
                            "and non-critical alerts. Response within 72 hours."},
        ],
        "incident_categories": [
            {"category": "Model Failure",
             "description": "AI model produces incorrect, unreliable, or "
                            "low-confidence predictions requiring human "
                            "override or intervention."},
            {"category": "Data Quality",
             "description": "Input data issues including EEG artifacts, "
                            "missing values, signal corruption, or anomalous "
                            "data patterns affecting AI accuracy."},
            {"category": "Pipeline Error",
             "description": "Processing pipeline failures including blocked "
                            "operations, rejected inputs, or workflow "
                            "interruptions in the clinical data pipeline."},
            {"category": "False Prediction",
             "description": "AI prediction contradicted by expert review or "
                            "clinical evidence. Includes expert disagreements "
                            "and HITL accept-with-correction events."},
            {"category": "Patient Safety",
             "description": "Clinical alert events indicating potential "
                            "patient harm, including self-harm flags, "
                            "escalation triggers, and safety intervention "
                            "requirements."},
            {"category": "System Error",
             "description": "Infrastructure and operational failures "
                            "including crashes, timeouts, configuration "
                            "errors, and integration failures."},
        ],
        "metrics": [
            {"metric": "Total Incidents",
             "description": "Aggregate count of all incident events from "
                            "transaction_log, expert_reviews, hitl_reviews, "
                            "clinical_decisions, model_governance, and "
                            "assessments."},
            {"metric": "MTTR (Mean Time to Resolve)",
             "description": "Average hours between incident detection and "
                            "resolution, estimated from timestamp gaps between "
                            "consecutive incidents per source.",
             "unit": "hours"},
            {"metric": "MTTD (Mean Time to Detect)",
             "description": "Average hours between incident occurrence and "
                            "first logging. Derived from system timestamps.",
             "unit": "hours"},
            {"metric": "Resolution Rate",
             "description": "Percentage of incidents with status 'Resolved' "
                            "versus total incidents.",
             "formula": "(resolved_incidents / total_incidents) * 100"},
            {"metric": "Severity Distribution",
             "description": "Breakdown of incidents by severity level "
                            "(Critical, High, Medium, Low) for risk "
                            "prioritization."},
            {"metric": "Patient Impact Score",
             "description": "Per-patient incident count and maximum severity "
                            "to identify patients most affected by AI system "
                            "incidents."},
        ],
        "methodology": (
            "Incidents are derived from six tables in clinical.db: "
            "(1) transaction_log — filtered for actions containing incident "
            "keywords (error, fail, reject, block, alert, override, escalat, "
            "timeout, crash, anomaly); "
            "(2) expert_reviews — entries where agree_with_ai='disagree' "
            "indicate AI prediction failures; "
            "(3) hitl_reviews — override decisions in fields_json indicate "
            "AI outputs rejected by human reviewers; "
            "(4) clinical_decisions — records with ai_confidence < 0.5 "
            "represent low-confidence incidents; "
            "(5) model_governance — governance events logged as system "
            "incidents; "
            "(6) assessments — records with non-empty alert field indicate "
            "clinical alert incidents. "
            "Each source is queried with try/except to handle missing tables "
            "gracefully. Severity is assigned based on clinical impact: "
            "patient safety alerts with safety keywords are Critical, "
            "expert disagreements and HITL overrides are High, transaction "
            "errors are Medium, and informational events are Low. MTTR is "
            "estimated from timestamp differences between consecutive "
            "incidents within the same source."
        ),
        "references": [
            {"standard": "ISO 27001:2022",
             "description": "Information security management — incident "
                            "management process including detection, "
                            "reporting, assessment, response, and lessons "
                            "learned."},
            {"standard": "NIST AI RMF 1.0",
             "description": "AI Risk Management Framework — MANAGE function "
                            "requires incident response plans for AI system "
                            "failures, including escalation procedures and "
                            "post-incident analysis."},
            {"standard": "EU AI Act (Article 62)",
             "description": "Mandatory reporting of serious incidents "
                            "involving high-risk AI systems to market "
                            "surveillance authorities, including incidents "
                            "presenting risks to health and safety."},
            {"standard": "FDA 21 CFR Part 803",
             "description": "Medical Device Reporting — mandatory reporting "
                            "of adverse events and malfunctions for AI/ML "
                            "Software as a Medical Device (SaMD)."},
            {"standard": "IEC 62443",
             "description": "Industrial cybersecurity — incident response "
                            "requirements for connected medical systems "
                            "including AI-driven clinical platforms."},
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
    print("\n=== DEFINITIONS ===")
    print(json.dumps(definitions(), indent=2, default=str))
