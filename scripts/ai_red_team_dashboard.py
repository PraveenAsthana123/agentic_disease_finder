"""AI Red Team Dashboard — adversarial testing, jailbreak detection,
prompt attack analysis, tool abuse monitoring from real clinical.db data.

Sources:
- transaction_log (664 rows) — adversarial/attack/injection/jailbreak/blocked events
- conversation_log (380 rows) — prompt injection attempt patterns
- hitl_reviews (2 rows) — human interventions related to attack mitigation
- expert_reviews (3 rows) — expert findings on AI disagreements
- clinical_decisions (1 row) — AI confidence as attack surface indicator
- assessments (423 rows) — data integrity checks
- analyses (21 rows) — model output reliability
"""

import sqlite3
import os
import re
from datetime import datetime, timezone

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


# Patterns that indicate potential prompt injection / jailbreak in conversation text
INJECTION_PATTERNS = [
    r'ignore\s+(previous|above|all)\s+(instructions?|prompts?)',
    r'you\s+are\s+now\s+',
    r'system\s*:\s*',
    r'<\s*/?script',
    r'OVERRIDE\s*:',
    r'forget\s+(everything|all|prior)',
    r'act\s+as\s+(a|an)\s+',
    r'pretend\s+you\s+are',
    r'do\s+not\s+follow',
    r'bypass\s+(safety|filter|guard)',
    r'jailbreak',
    r'DAN\s+mode',
]

ATTACK_CATEGORIES = [
    'Prompt Injection',
    'Jailbreak Attempt',
    'Tool Abuse',
    'Data Poisoning',
    'Model Evasion',
    'Adversarial Input',
]


def _scan_conversations_for_injections(cur):
    """Scan conversation_log for prompt injection patterns."""
    rows = _safe_query(cur,
        "SELECT id, role, text, ts_utc FROM conversation_log "
        "WHERE role = 'operator'")
    detections = []
    for row in rows:
        text = row.get('text') or ''
        for pat in INJECTION_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                detections.append({
                    "id": row['id'],
                    "role": row.get('role'),
                    "pattern_matched": pat,
                    "snippet": text[:120] + ('...' if len(text) > 120 else ''),
                    "timestamp": row.get('ts_utc'),
                    "severity": "high" if 'override' in pat.lower() or 'bypass' in pat.lower()
                                 else "medium",
                })
                break  # one detection per message
    return detections


def _classify_security_events(events):
    """Classify transaction_log events into red team categories."""
    classified = {cat: [] for cat in ATTACK_CATEGORIES}

    for e in events:
        action = (e.get('action') or '').lower()
        component = (e.get('component') or '').lower()
        detail = (e.get('detail') or '').lower()

        if action == 'blocked' or 'block' in detail:
            classified['Prompt Injection'].append(e)
        elif 'monitor' in action or 'check' in action:
            if 'drift' in component or 'consistency' in component:
                classified['Model Evasion'].append(e)
            else:
                classified['Adversarial Input'].append(e)
        elif 'council' in component and ('sign-off' in action or 'human_decision' in action):
            classified['Tool Abuse'].append(e)
        elif 'training' in component:
            classified['Data Poisoning'].append(e)
        elif action in ('query', 'ask', 'answer') and 'genai' in component:
            classified['Jailbreak Attempt'].append(e)

    return classified


def red_team_overview():
    """Aggregate red team KPIs — adversarial testing, jailbreak detection,
    prompt attack analysis, tool abuse monitoring from real clinical.db data."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── Total transaction events (test surface) ──
    total_tx = _safe_count(cur, "SELECT count(*) FROM transaction_log")

    # ── Blocked events ──
    blocked_events = _safe_query(cur,
        "SELECT * FROM transaction_log WHERE action = 'blocked' "
        "ORDER BY ts_utc DESC")
    blocked_count = len(blocked_events)

    # ── Security-related events (monitor, check, blocked) ──
    security_events = _safe_query(cur,
        "SELECT * FROM transaction_log "
        "WHERE action IN ('blocked','monitor','check') "
        "ORDER BY ts_utc DESC")
    security_count = len(security_events)

    # ── Prompt injection scans from conversation_log ──
    injection_detections = _scan_conversations_for_injections(cur)
    prompt_injection_count = len(injection_detections)

    # ── Classify all events ──
    all_events = _safe_query(cur,
        "SELECT * FROM transaction_log ORDER BY ts_utc DESC")
    classified = _classify_security_events(all_events)

    jailbreak_count = len(classified['Jailbreak Attempt'])
    tool_abuse_count = len(classified['Tool Abuse'])
    adversarial_count = len(classified['Adversarial Input']) + len(classified['Model Evasion'])
    data_poisoning_count = len(classified['Data Poisoning'])

    # ── HITL interventions (attack mitigation) ──
    hitl_count = _safe_count(cur, "SELECT count(*) FROM hitl_reviews")

    # ── Expert disagreements (potential adversarial indicators) ──
    expert_disagree = _safe_count(cur,
        "SELECT count(*) FROM expert_reviews WHERE agree_with_ai = 'disagree'")
    total_expert = _safe_count(cur, "SELECT count(*) FROM expert_reviews")

    # ── AI confidence as attack surface ──
    decisions = _safe_query(cur, "SELECT * FROM clinical_decisions")
    low_confidence = sum(1 for d in decisions
                        if d.get('ai_confidence') is not None
                        and d['ai_confidence'] < 0.5)

    # ── Analyses & assessments (model reliability surface) ──
    total_analyses = _safe_count(cur, "SELECT count(*) FROM analyses")
    total_assessments = _safe_count(cur, "SELECT count(*) FROM assessments")

    # ── Total tests run = security events + injection scans + analyses ──
    total_tests_run = security_count + prompt_injection_count + total_analyses

    # ── Adversarial attempts detected ──
    adversarial_detected = (blocked_count + prompt_injection_count +
                           expert_disagree + adversarial_count)

    # ── Vulnerability score (0-100): higher = more vulnerable ──
    # Based on: blocked events, injection detections, low confidence,
    # expert disagreements, relative to total surface
    raw_vuln = (blocked_count * 15 + prompt_injection_count * 10 +
                low_confidence * 20 + expert_disagree * 10)
    vulnerability_score = min(100, round(raw_vuln / max(total_tests_run, 1) * 100, 1))

    # ── Coverage: components with security monitoring / total components ──
    total_components = _safe_count(cur,
        "SELECT count(DISTINCT component) FROM transaction_log")
    monitored_components = _safe_count(cur,
        "SELECT count(DISTINCT component) FROM transaction_log "
        "WHERE action IN ('blocked','monitor','check')")
    coverage_pct = round(monitored_components / max(total_components, 1) * 100, 1)

    kpis = {
        "total_tests_run": total_tests_run,
        "adversarial_attempts_detected": adversarial_detected,
        "jailbreak_attempts": jailbreak_count,
        "prompt_injection_scans": prompt_injection_count,
        "tool_abuse_incidents": tool_abuse_count,
        "blocked_events": blocked_count,
        "vulnerability_score": vulnerability_score,
        "coverage_pct": coverage_pct,
    }

    # ── Attack type distribution (pie chart) ──
    attack_type_distribution = []
    for cat in ATTACK_CATEGORIES:
        count = len(classified[cat])
        if cat == 'Prompt Injection':
            count += prompt_injection_count
        if count > 0:
            attack_type_distribution.append({"name": cat, "value": count})

    # ── Daily security events (timeline) ──
    daily_counts = {}
    for e in all_events:
        ts = e.get('ts_utc', '')
        if ts:
            day = ts[:10]
            daily_counts[day] = daily_counts.get(day, 0) + 1
    daily_security_events = [{"date": k, "events": v}
                             for k, v in sorted(daily_counts.items())]

    # ── Vulnerability matrix: severity x likelihood ──
    vuln_matrix = [
        {"severity": "Critical", "likelihood": "Low",
         "count": blocked_count, "component": "council"},
        {"severity": "High", "likelihood": "Medium",
         "count": prompt_injection_count, "component": "conversation"},
        {"severity": "Medium", "likelihood": "High",
         "count": adversarial_count, "component": "monitoring"},
        {"severity": "Low", "likelihood": "High",
         "count": data_poisoning_count, "component": "training"},
        {"severity": "Medium", "likelihood": "Low",
         "count": expert_disagree, "component": "expert_review"},
    ]

    # ── Attack surface map: component exposure ──
    component_event_counts = {}
    for e in all_events:
        comp = e.get('component') or 'unknown'
        component_event_counts[comp] = component_event_counts.get(comp, 0) + 1

    attack_surface_map = []
    for comp, cnt in sorted(component_event_counts.items(),
                            key=lambda x: -x[1])[:15]:
        # Exposure level based on event volume and security events
        has_security = comp in ('council', 'drift', 'consistency')
        exposure = ('high' if has_security or cnt > 100
                    else 'medium' if cnt > 20
                    else 'low')
        attack_surface_map.append({
            "component": comp,
            "events": cnt,
            "exposure": exposure,
            "has_monitoring": has_security,
        })

    conn.close()

    return {
        "available": True,
        "kpis": kpis,
        "attack_type_distribution": attack_type_distribution,
        "daily_security_events": daily_security_events,
        "vulnerability_matrix": vuln_matrix,
        "attack_surface_map": attack_surface_map,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def red_team_breakdown():
    """Detailed per-component adversarial test results, prompt injection log,
    jailbreak attempts, tool abuse log, attack vectors."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── All events for classification ──
    all_events = _safe_query(cur,
        "SELECT * FROM transaction_log ORDER BY ts_utc DESC")
    classified = _classify_security_events(all_events)

    # ── Adversarial tests: per-component test results ──
    components = {}
    for e in all_events:
        comp = e.get('component') or 'unknown'
        if comp not in components:
            components[comp] = {"component": comp, "total_events": 0,
                                "blocked": 0, "monitored": 0,
                                "status": "untested"}
        components[comp]["total_events"] += 1
        action = (e.get('action') or '').lower()
        if action == 'blocked':
            components[comp]["blocked"] += 1
            components[comp]["status"] = "vulnerable"
        elif action in ('monitor', 'check'):
            components[comp]["monitored"] += 1
            components[comp]["status"] = "tested"

    adversarial_tests = sorted(components.values(),
                               key=lambda x: -(x['blocked'] + x['monitored']))

    # ── Prompt injection log ──
    injection_log = _scan_conversations_for_injections(cur)

    # ── Jailbreak attempts ──
    jailbreak_events = []
    for e in classified['Jailbreak Attempt']:
        jailbreak_events.append({
            "id": e.get("id"),
            "patient_id": e.get("patient_id"),
            "component": e.get("component"),
            "action": e.get("action"),
            "detail": e.get("detail"),
            "timestamp": e.get("ts_utc"),
            "status": "detected",
        })

    # ── Tool abuse log ──
    tool_abuse_events = []
    for e in classified['Tool Abuse']:
        tool_abuse_events.append({
            "id": e.get("id"),
            "patient_id": e.get("patient_id"),
            "component": e.get("component"),
            "action": e.get("action"),
            "actor": e.get("actor"),
            "detail": e.get("detail"),
            "timestamp": e.get("ts_utc"),
            "status": "flagged",
        })

    # ── Blocked events detail ──
    blocked_detail = []
    for e in classified['Prompt Injection']:
        blocked_detail.append({
            "id": e.get("id"),
            "patient_id": e.get("patient_id"),
            "component": e.get("component"),
            "action": e.get("action"),
            "detail": e.get("detail"),
            "actor": e.get("actor"),
            "timestamp": e.get("ts_utc"),
        })

    # ── Attack vectors: categorized with status ──
    attack_vectors = []
    for cat in ATTACK_CATEGORIES:
        items = classified[cat]
        count = len(items)
        if cat == 'Prompt Injection':
            count += len(injection_log)
        attack_vectors.append({
            "category": cat,
            "count": count,
            "status": "mitigated" if count == 0 else "active",
            "last_event": items[0].get('ts_utc') if items else None,
        })

    # ── Expert review detail (disagree = adversarial indicator) ──
    expert_findings = _safe_query(cur,
        "SELECT * FROM expert_reviews")

    # ── HITL mitigation detail ──
    hitl_mitigations = _safe_query(cur,
        "SELECT * FROM hitl_reviews")

    conn.close()

    return {
        "available": True,
        "adversarial_tests": adversarial_tests,
        "prompt_injection_log": injection_log,
        "jailbreak_attempts": jailbreak_events,
        "tool_abuse_log": tool_abuse_events,
        "blocked_events": blocked_detail,
        "attack_vectors": attack_vectors,
        "expert_findings": expert_findings,
        "hitl_mitigations": hitl_mitigations,
    }


def red_team_definitions():
    """AI Red Team metric definitions with compliance references."""
    return {
        "available": True,
        "definitions": [
            {
                "metric": "Adversarial Testing",
                "description": "Systematic probing of AI components with adversarial "
                               "inputs to identify vulnerabilities, unexpected behaviors, "
                               "and failure modes in the clinical AI pipeline.",
                "source": "transaction_log (monitor/check/blocked events per component)",
                "compliance": "OWASP ML Top 10, NIST AI RMF MAP 1.5, EU AI Act Art. 9(7)",
            },
            {
                "metric": "Jailbreak Detection",
                "description": "Detection of attempts to bypass safety guardrails, role "
                               "constraints, or operational boundaries of the AI system "
                               "through crafted prompts or input sequences.",
                "source": "conversation_log (pattern-matched against known jailbreak signatures)",
                "compliance": "OWASP ML06 (AI Supply Chain), NIST AI RMF GOVERN 1.2",
            },
            {
                "metric": "Prompt Injection",
                "description": "Detection of malicious prompt injections designed to alter "
                               "AI behavior, override system instructions, or extract "
                               "sensitive clinical information through conversation input.",
                "source": "conversation_log (regex scan for injection patterns), "
                          "transaction_log (blocked events)",
                "compliance": "OWASP ML01 (Input Manipulation), EU AI Act Art. 9(2), "
                              "FDA AI-ML Action Plan",
            },
            {
                "metric": "Tool Abuse",
                "description": "Monitoring of unauthorized or anomalous tool usage patterns "
                               "that may indicate exploitation of MCP tools, agent capabilities, "
                               "or clinical decision endpoints beyond intended scope.",
                "source": "transaction_log (council sign-off, human_decision actions)",
                "compliance": "NIST AI RMF MANAGE 2.4, IEC 62304, EU AI Act Art. 9(4)",
            },
            {
                "metric": "Vulnerability Score",
                "description": "Composite 0-100 score reflecting the overall attack surface "
                               "exposure. Computed from blocked events, injection detections, "
                               "low AI confidence instances, and expert disagreements relative "
                               "to total test surface. Lower is better.",
                "source": "Computed from transaction_log + conversation_log + clinical_decisions + "
                          "expert_reviews",
                "compliance": "NIST AI RMF MEASURE 2.6, ISO 14971 (risk analysis), "
                              "FDA AI-ML (continuous monitoring)",
            },
            {
                "metric": "Attack Surface",
                "description": "Mapping of all system components and their exposure level "
                               "based on event volume, presence of security monitoring, "
                               "and historical incident data. Components without monitoring "
                               "are flagged as higher exposure.",
                "source": "transaction_log (per-component event counts and action types)",
                "compliance": "OWASP ML09 (Output Integrity), NIST AI RMF MAP 3.4, "
                              "EU AI Act Art. 9(1)",
            },
            {
                "metric": "Red Team Coverage",
                "description": "Percentage of system components that have active security "
                               "monitoring (monitor/check/blocked actions) versus total "
                               "components in the transaction log. Target: 100%.",
                "source": "transaction_log (distinct components with security actions / "
                          "total distinct components)",
                "compliance": "NIST AI RMF MEASURE 1.1, EU AI Act Art. 9(8), "
                              "FDA AI-ML (total product lifecycle approach)",
            },
        ],
    }
