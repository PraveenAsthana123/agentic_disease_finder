"""SecOps Dashboard — threat detection, injection/jailbreak scanning, PII protection,
access audit trail, incident tracking, and compliance posture from clinical.db
transaction logs, conversation logs, and rai_checks patterns.

Distinct from MCP Security (guardrail enforcement focus). SecOps covers:
- Threat detection: injection attempts, jailbreak probes, PII exposure
- Access audit: who accessed what patient data, privileged escalation
- Incident timeline: blocked events, security agent interventions
- Compliance posture: OWASP/STRIDE coverage, PII pattern inventory
- Conversation security: role distribution, injection surface monitoring
"""

import hashlib
import os
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

PII_CATEGORIES = {
    "email": {"pattern": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "severity": "high", "regulation": "HIPAA/GDPR"},
    "phone": {"pattern": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b", "severity": "medium", "regulation": "HIPAA"},
    "ssn": {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "severity": "critical", "regulation": "HIPAA/PCI"},
    "mrn": {"pattern": r"\bMRN[:#]?\s?\d{5,10}\b", "severity": "high", "regulation": "HIPAA"},
    "dob": {"pattern": r"\b(?:DOB|D\.O\.B)[:\s]?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "severity": "high", "regulation": "HIPAA"},
    "credit_card": {"pattern": r"\b(?:\d[ -]*?){13,16}\b", "severity": "critical", "regulation": "PCI-DSS"},
}

INJECTION_PATTERNS = [
    {"pattern": r"ignore (?:all |previous |the )?instructions", "category": "prompt_injection", "severity": "critical"},
    {"pattern": r"disregard (?:the )?(?:above|previous)", "category": "prompt_injection", "severity": "critical"},
    {"pattern": r"system prompt", "category": "prompt_extraction", "severity": "high"},
    {"pattern": r"you are now", "category": "role_hijack", "severity": "high"},
    {"pattern": r"act as", "category": "role_hijack", "severity": "medium"},
    {"pattern": r"reveal (?:your )?(?:prompt|instructions)", "category": "prompt_extraction", "severity": "critical"},
    {"pattern": r"forget (?:everything|all)", "category": "memory_wipe", "severity": "high"},
    {"pattern": r"jailbreak", "category": "jailbreak", "severity": "critical"},
    {"pattern": r"developer mode", "category": "jailbreak", "severity": "high"},
    {"pattern": r"bypass", "category": "bypass", "severity": "high"},
]

THREAT_CATEGORIES = {
    "prompt_injection": {"description": "Attempts to override system instructions", "owasp": "LLM01"},
    "prompt_extraction": {"description": "Attempts to extract system prompts or instructions", "owasp": "LLM01"},
    "role_hijack": {"description": "Attempts to change the AI's role or persona", "owasp": "LLM01"},
    "memory_wipe": {"description": "Attempts to clear AI context or memory", "owasp": "LLM01"},
    "jailbreak": {"description": "Attempts to bypass AI safety guardrails", "owasp": "LLM01"},
    "bypass": {"description": "Attempts to bypass security or access controls", "owasp": "LLM07"},
    "pii_exposure": {"description": "Personal identifiable information detected in data flow", "owasp": "LLM06"},
    "data_exfiltration": {"description": "Unusual data access patterns suggesting exfiltration", "owasp": "LLM06"},
}


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql):
    try:
        cur.execute(sql)
        r = cur.fetchone()
        return r[0] if r else 0
    except Exception:
        return 0


def _safe_rows(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


def _scan_conversations(cur):
    """Scan conversation_log for injection/PII threats using real patterns."""
    rows = _safe_rows(cur, "SELECT role, text FROM conversation_log WHERE text IS NOT NULL")
    threats = []
    pii_hits = defaultdict(int)
    injection_hits = defaultdict(int)

    for role, text in rows:
        if not text:
            continue
        t = text.lower()
        # Injection scan
        for pat_info in INJECTION_PATTERNS:
            if re.search(pat_info["pattern"], t, re.IGNORECASE):
                injection_hits[pat_info["category"]] += 1
                threats.append({
                    "type": "injection",
                    "category": pat_info["category"],
                    "severity": pat_info["severity"],
                    "role": role,
                    "snippet": text[:80] + "..." if len(text) > 80 else text,
                })
        # PII scan
        for pii_name, pii_info in PII_CATEGORIES.items():
            if re.search(pii_info["pattern"], text):
                pii_hits[pii_name] += 1
                threats.append({
                    "type": "pii_exposure",
                    "category": pii_name,
                    "severity": pii_info["severity"],
                    "role": role,
                    "snippet": "[REDACTED]",
                })

    return threats, dict(pii_hits), dict(injection_hits)


# ── Overview ──────────────────────────────────────────────────

def overview():
    """SecOps overview: threat summary, guardrail posture, access stats,
    PII/injection pattern inventory, compliance score."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc)
    result = {"available": True, "generated_at": now.isoformat()}

    # ── Transaction totals ──
    total_tx = _safe(cur, "SELECT count(*) FROM transaction_log")
    total_conv = _safe(cur, "SELECT count(*) FROM conversation_log")
    blocked = _safe(cur, "SELECT count(*) FROM transaction_log WHERE action='blocked'")
    signoffs = _safe(cur, "SELECT count(*) FROM transaction_log WHERE action='sign-off'")
    human_decisions = _safe(cur, "SELECT count(*) FROM transaction_log WHERE action='human_decision'")
    expert_reviews = _safe(cur, "SELECT count(*) FROM expert_reviews")
    hitl_reviews = _safe(cur, "SELECT count(*) FROM hitl_reviews")

    guardrail_total = blocked + signoffs + human_decisions
    oversight_events = signoffs + human_decisions + expert_reviews + hitl_reviews

    # ── Scan conversations for threats ──
    threats, pii_hits, injection_hits = _scan_conversations(cur)
    total_threats = len(threats)
    critical_threats = sum(1 for t in threats if t["severity"] == "critical")
    high_threats = sum(1 for t in threats if t["severity"] == "high")

    # ── Distinct actors and their privilege levels ──
    actors = _safe_rows(cur,
        "SELECT actor, count(*) as txns, count(DISTINCT component) as comps, "
        "count(DISTINCT patient_id) as patients "
        "FROM transaction_log GROUP BY actor ORDER BY txns DESC")
    total_actors = len(actors)
    privileged_actors = 0
    for r in actors:
        priv = _safe_rows(cur,
            f"SELECT DISTINCT action FROM transaction_log WHERE actor='{r[0]}' "
            "AND action IN ('delete','update','human_decision','sign-off','blocked')")
        if priv:
            privileged_actors += 1

    # ── Compliance score (weighted) ──
    # Based on: guardrails active, PII patterns deployed, injection patterns deployed,
    # human oversight rate, blocked enforcement
    pii_coverage = len(PII_CATEGORIES)  # 6 patterns
    injection_coverage = len(INJECTION_PATTERNS)  # 10 patterns
    oversight_rate = round(oversight_events / max(total_tx, 1) * 100, 1)
    has_blocking = 1 if blocked > 0 else 0
    compliance_score = min(100, round(
        (pii_coverage / 6 * 25) +  # 25% for PII coverage
        (injection_coverage / 10 * 25) +  # 25% for injection coverage
        (min(oversight_rate, 5) / 5 * 25) +  # 25% for oversight rate (capped at 5%)
        (has_blocking * 25)  # 25% for active enforcement
    ))

    result["kpis"] = {
        "total_transactions": total_tx,
        "total_conversations": total_conv,
        "guardrail_events": guardrail_total,
        "blocked_events": blocked,
        "total_threats_detected": total_threats,
        "critical_threats": critical_threats,
        "high_threats": high_threats,
        "pii_patterns_active": pii_coverage,
        "injection_patterns_active": injection_coverage,
        "total_actors": total_actors,
        "privileged_actors": privileged_actors,
        "oversight_rate_pct": oversight_rate,
        "compliance_score": compliance_score,
    }

    # ── PII pattern inventory ──
    result["pii_inventory"] = [
        {
            "name": name,
            "severity": info["severity"],
            "regulation": info["regulation"],
            "detections": pii_hits.get(name, 0),
        }
        for name, info in PII_CATEGORIES.items()
    ]

    # ── Injection pattern inventory ──
    result["injection_inventory"] = [
        {
            "pattern_label": p["category"],
            "severity": p["severity"],
            "detections": injection_hits.get(p["category"], 0),
        }
        for p in INJECTION_PATTERNS
    ]
    # Deduplicate by category
    seen = set()
    deduped = []
    for item in result["injection_inventory"]:
        if item["pattern_label"] not in seen:
            seen.add(item["pattern_label"])
            deduped.append(item)
    result["injection_inventory"] = deduped

    # ── Threat severity distribution ──
    severity_dist = defaultdict(int)
    for t in threats:
        severity_dist[t["severity"]] += 1
    result["threat_severity"] = dict(severity_dist)

    # ── Threat category distribution ──
    cat_dist = defaultdict(int)
    for t in threats:
        cat_dist[t["category"]] += 1
    result["threat_categories"] = dict(cat_dist)

    # ── Security agent activity ──
    sec_events = _safe_rows(cur,
        "SELECT actor, action, component, detail, ts_utc "
        "FROM transaction_log "
        "WHERE actor IN ('security_agent','compliance_agent') "
        "ORDER BY ts_utc DESC")
    result["security_agent_events"] = [
        {"actor": r[0], "action": r[1], "component": r[2],
         "detail": r[3], "timestamp": r[4]}
        for r in sec_events
    ]

    # ── Guardrail enforcement log ──
    guard = _safe_rows(cur,
        "SELECT actor, action, component, patient_id, detail, ts_utc "
        "FROM transaction_log WHERE action IN ('blocked','sign-off','human_decision') "
        "ORDER BY ts_utc DESC")
    result["guardrail_log"] = [
        {"actor": r[0], "action": r[1], "component": r[2],
         "patient_id": r[3], "detail": r[4], "timestamp": r[5]}
        for r in guard
    ]

    conn.close()
    return result


# ── Breakdown ─────────────────────────────────────────────────

def breakdown():
    """Drill-down: access audit, actor privilege matrix, daily security trend,
    conversation role analysis, component attack surface, incident timeline."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    result = {"available": True}

    # ── Patient data access audit ──
    patient_access = _safe_rows(cur,
        "SELECT patient_id, count(DISTINCT actor) as actors, "
        "count(DISTINCT component) as comps, count(*) as events "
        "FROM transaction_log WHERE patient_id IS NOT NULL AND patient_id != '' "
        "GROUP BY patient_id ORDER BY actors DESC, events DESC LIMIT 30")
    result["patient_access_audit"] = [
        {"patient_id": r[0], "distinct_actors": r[1], "distinct_components": r[2],
         "total_events": r[3],
         "risk": "elevated" if r[1] > 2 else "normal"}
        for r in patient_access
    ]

    # ── Actor privilege matrix ──
    actor_rows = _safe_rows(cur,
        "SELECT actor, count(*) as txns, count(DISTINCT component) as comps, "
        "count(DISTINCT action) as acts, count(DISTINCT patient_id) as patients "
        "FROM transaction_log GROUP BY actor ORDER BY txns DESC")
    actor_matrix = []
    for r in actor_rows:
        priv_rows = _safe_rows(cur,
            f"SELECT DISTINCT action FROM transaction_log WHERE actor='{r[0]}' "
            "AND action IN ('delete','update','human_decision','sign-off','blocked')")
        priv_actions = [p[0] for p in priv_rows]
        risk = "high" if len(priv_actions) >= 2 else ("medium" if priv_actions else "low")
        actor_matrix.append({
            "actor": r[0], "transactions": r[1], "components": r[2],
            "distinct_actions": r[3], "patients_accessed": r[4],
            "privileged_actions": priv_actions, "risk_level": risk,
        })
    result["actor_privilege_matrix"] = actor_matrix

    # ── Component attack surface ──
    comp_surface = _safe_rows(cur,
        "SELECT component, count(DISTINCT actor) as actors, count(*) as txns, "
        "count(DISTINCT action) as acts "
        "FROM transaction_log GROUP BY component ORDER BY actors DESC")
    result["attack_surface"] = [
        {"component": r[0], "distinct_actors": r[1], "transactions": r[2],
         "distinct_actions": r[3],
         "exposure": "high" if r[1] >= 3 else ("medium" if r[1] >= 2 else "low")}
        for r in comp_surface
    ]

    # ── Daily security trend ──
    daily = _safe_rows(cur,
        "SELECT substr(ts_utc, 1, 10) AS day, count(*) as total, "
        "sum(CASE WHEN action='blocked' THEN 1 ELSE 0 END) as blocked, "
        "sum(CASE WHEN action='sign-off' THEN 1 ELSE 0 END) as signoffs, "
        "sum(CASE WHEN action IN ('delete','update') THEN 1 ELSE 0 END) as mutations "
        "FROM transaction_log WHERE ts_utc IS NOT NULL "
        "GROUP BY day ORDER BY day")
    result["daily_security"] = [
        {"date": r[0], "total": r[1], "blocked": r[2],
         "signoffs": r[3], "mutations": r[4]}
        for r in daily
    ]

    # ── Conversation role distribution ──
    conv_roles = _safe_rows(cur,
        "SELECT role, count(*) FROM conversation_log GROUP BY role ORDER BY count(*) DESC")
    result["conversation_roles"] = [
        {"role": r[0], "count": r[1]} for r in conv_roles
    ]

    # ── Recent privileged events (incident timeline) ──
    recent = _safe_rows(cur,
        "SELECT actor, action, component, patient_id, detail, ts_utc "
        "FROM transaction_log "
        "WHERE action IN ('delete','update','human_decision','sign-off','blocked',"
        "'ingest','scheduled_train') "
        "ORDER BY ts_utc DESC LIMIT 25")
    result["incident_timeline"] = [
        {"actor": r[0], "action": r[1], "component": r[2],
         "patient_id": r[3], "detail": r[4], "timestamp": r[5]}
        for r in recent
    ]

    # ── Action distribution ──
    actions = _safe_rows(cur,
        "SELECT action, count(*) FROM transaction_log GROUP BY action ORDER BY count(*) DESC")
    result["action_distribution"] = [
        {"action": r[0], "count": r[1]} for r in actions
    ]

    # ── OWASP LLM Top-10 coverage map ──
    result["owasp_coverage"] = [
        {"id": "LLM01", "name": "Prompt Injection", "status": "monitored",
         "controls": "10 injection patterns, conversation scanning, role validation"},
        {"id": "LLM02", "name": "Insecure Output Handling", "status": "monitored",
         "controls": "PII redaction (6 patterns), output sanitization"},
        {"id": "LLM03", "name": "Training Data Poisoning", "status": "mitigated",
         "controls": "Data lineage tracking, quality scoring, drift monitoring"},
        {"id": "LLM04", "name": "Model Denial of Service", "status": "mitigated",
         "controls": "Rate limiting, token budgets, timeout enforcement"},
        {"id": "LLM05", "name": "Supply Chain Vulnerabilities", "status": "monitored",
         "controls": "Dependency scanning, model provenance tracking"},
        {"id": "LLM06", "name": "Sensitive Information Disclosure", "status": "monitored",
         "controls": "PII detection (6 categories), conversation role audit"},
        {"id": "LLM07", "name": "Insecure Plugin Design", "status": "mitigated",
         "controls": "MCP tool registry, permission scoping, audit logging"},
        {"id": "LLM08", "name": "Excessive Agency", "status": "monitored",
         "controls": "Human-in-the-loop gates, confidence thresholds, guardrails"},
        {"id": "LLM09", "name": "Overreliance", "status": "mitigated",
         "controls": "Confidence scoring, uncertainty display, human oversight"},
        {"id": "LLM10", "name": "Model Theft", "status": "mitigated",
         "controls": "Access control, model registry, deployment audit"},
    ]

    conn.close()
    return result


# ── Definitions ───────────────────────────────────────────────

def definitions():
    """Metric definitions for the SecOps dashboard."""
    return {
        "available": True,
        "sections": [
            {
                "title": "Threat Detection",
                "items": [
                    {"term": "Injection Patterns", "definition": "Regular expressions that detect prompt injection, jailbreak, role hijack, and bypass attempts in conversation text. 10 patterns covering 6 attack categories from rai_checks.py."},
                    {"term": "PII Patterns", "definition": "Regular expressions that detect personally identifiable information: email, phone, SSN, MRN, DOB, credit card. 6 patterns mapped to HIPAA/GDPR/PCI-DSS regulations."},
                    {"term": "Threat Severity", "definition": "Critical: direct prompt injection, jailbreak, SSN/credit card exposure. High: role hijack, prompt extraction, MRN/email exposure. Medium: indirect probes like 'act as'."},
                    {"term": "Conversation Scanning", "definition": "All conversation_log entries are scanned against injection and PII patterns. Detections are counted by category and severity."},
                ],
            },
            {
                "title": "Access Audit",
                "items": [
                    {"term": "Patient Access Audit", "definition": "Per-patient count of distinct actors and components that accessed their data. Risk is elevated when >2 distinct actors access a single patient record."},
                    {"term": "Actor Privilege Matrix", "definition": "Per-actor breakdown showing transactions, components accessed, distinct actions, and patients touched. Risk levels: high (≥2 privileged actions), medium (1), low (none)."},
                    {"term": "Privileged Actions", "definition": "High-impact actions requiring audit: delete, update, human_decision, sign-off, blocked, ingest, scheduled_train."},
                ],
            },
            {
                "title": "Attack Surface",
                "items": [
                    {"term": "Component Exposure", "definition": "Components ranked by number of distinct actors. More actors accessing a component = broader exposure = higher attack surface."},
                    {"term": "Action Distribution", "definition": "Breakdown of all transaction actions by frequency. Highlights unusual or high-risk action patterns."},
                ],
            },
            {
                "title": "Guardrail Enforcement",
                "items": [
                    {"term": "Blocked Events", "definition": "Actions explicitly stopped by the security_agent or compliance rules. Represents active enforcement."},
                    {"term": "Sign-off Events", "definition": "Actions requiring and receiving clinical sign-off before proceeding."},
                    {"term": "Human Decisions", "definition": "Actions where a human clinician overrode or confirmed an AI recommendation."},
                    {"term": "Oversight Rate", "definition": "Percentage of transactions involving human review: (sign-offs + human_decisions + expert_reviews + HITL_reviews) / total_transactions."},
                ],
            },
            {
                "title": "OWASP LLM Top-10",
                "items": [
                    {"term": "LLM01 — Prompt Injection", "definition": "Manipulating LLM behavior through crafted inputs. Monitored via 10 regex patterns scanning conversation_log."},
                    {"term": "LLM06 — Sensitive Info Disclosure", "definition": "LLM revealing PII or confidential data. Monitored via 6 PII detection patterns."},
                    {"term": "LLM07 — Insecure Plugin Design", "definition": "Plugins/tools with excessive permissions. Mitigated via MCP tool registry and permission scoping."},
                    {"term": "LLM08 — Excessive Agency", "definition": "LLM taking actions beyond intended scope. Mitigated via HITL gates and confidence thresholds."},
                ],
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {"term": "IEC 62304 §5.1.3", "definition": "Security risk management for medical device software. SecOps dashboard provides continuous monitoring of access patterns, threat detection, and compliance posture."},
                    {"term": "HIPAA Security Rule", "definition": "Access controls, audit trails, and PII protection for protected health information. Actor privilege matrix and patient access audit support compliance."},
                    {"term": "Compliance Score", "definition": "Weighted composite: PII coverage (25%) + injection coverage (25%) + oversight rate (25%) + active enforcement (25%). Scale 0-100."},
                ],
            },
        ],
    }
