"""Agent Memory Dashboard — real per-patient memory analytics from clinical.db.

Tracks how completely the system "remembers" each patient across data domains
(demographics, medications, assessments, seizure diary, analyses, expert reviews,
clinical decisions, uploads, clinical history), conversation context retention,
memory staleness, recall patterns, and coverage gaps.

Sources:
- patients (40)          — demographics
- patient_master (2)     — master file index
- medications (9)        — active medications
- assessments (259)      — instrument scores
- seizure_diary (25)     — seizure event log
- analyses (21)          — AI predictions
- expert_reviews (3)     — specialist findings
- clinical_decisions (1) — human-AI decisions
- uploads (21)           — file attachments
- conversation_log (227) — chat context
- transaction_log (558)  — action audit trail
- feedback (1)           — correction/learning signal
- hitl_reviews (2)       — human-in-the-loop overrides
"""

import os
import sqlite3
import json
from datetime import datetime, timezone
from collections import Counter, defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

# Memory domains — each maps to a table with patient_id
_DOMAINS = [
    ("demographics", "patients", "patient_id"),
    ("medications", "medications", "patient_id"),
    ("assessments", "assessments", "patient_id"),
    ("seizure_diary", "seizure_diary", "patient_id"),
    ("analyses", "analyses", "patient_id"),
    ("expert_reviews", "expert_reviews", "patient_id"),
    ("clinical_decisions", "clinical_decisions", "patient_id"),
    ("uploads", "uploads", "patient_id"),
    ("feedback", "feedback", "patient_id"),
    ("hitl_reviews", "hitl_reviews", "patient_id"),
]


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


# ── Overview ──────────────────────────────────────────────────────

def memory_overview():
    """Aggregate agent memory KPIs: coverage, completeness, staleness,
    conversation context depth, domain fill rates."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    now = datetime.now(timezone.utc)
    result = {"available": True, "generated_at": now.isoformat()}

    # ── Total patients ──
    total_patients = _safe(cur, "SELECT count(*) FROM patients")

    # ── Per-domain record counts and patient coverage ──
    domain_stats = []
    all_domain_patients = defaultdict(set)  # patient_id -> set of domains
    total_records = 0

    for domain_name, table, pid_col in _DOMAINS:
        count = _safe(cur, f"SELECT count(*) FROM [{table}]")
        patients_with = _safe(cur, f"SELECT count(DISTINCT {pid_col}) FROM [{table}]")
        coverage_pct = round(patients_with / max(total_patients, 1) * 100, 1)

        # Track which patients have this domain
        rows = _safe_rows(cur, f"SELECT DISTINCT {pid_col} FROM [{table}]")
        for r in rows:
            all_domain_patients[r[0]].add(domain_name)

        domain_stats.append({
            "domain": domain_name,
            "total_records": count,
            "patients_with_data": patients_with,
            "coverage_pct": coverage_pct,
        })
        total_records += count

    # ── Memory completeness per patient (how many domains filled) ──
    total_domains = len(_DOMAINS)
    completeness_scores = []
    for pid, domains in all_domain_patients.items():
        completeness_scores.append(round(len(domains) / total_domains * 100, 1))

    # Also count patients with ZERO memory
    patients_with_any = len(all_domain_patients)
    patients_with_none = max(total_patients - patients_with_any, 0)

    avg_completeness = (
        round(sum(completeness_scores) / len(completeness_scores), 1)
        if completeness_scores else 0
    )

    # Completeness distribution buckets
    comp_buckets = {"0%": patients_with_none, "1-25%": 0, "26-50%": 0,
                    "51-75%": 0, "76-100%": 0}
    for s in completeness_scores:
        if s <= 25:
            comp_buckets["1-25%"] += 1
        elif s <= 50:
            comp_buckets["26-50%"] += 1
        elif s <= 75:
            comp_buckets["51-75%"] += 1
        else:
            comp_buckets["76-100%"] += 1

    completeness_distribution = [
        {"bucket": k, "count": v} for k, v in comp_buckets.items()
    ]

    # ── Conversation context depth ──
    total_conversations = _safe(cur, "SELECT count(*) FROM conversation_log")
    assistant_turns = _safe(cur, "SELECT count(*) FROM conversation_log WHERE role='assistant'")
    user_turns = _safe(cur, "SELECT count(*) FROM conversation_log WHERE role='user'")
    system_turns = _safe(cur, "SELECT count(*) FROM conversation_log WHERE role='system'")

    # Average conversation length (chars) per assistant response
    avg_response_len = _safe(
        cur,
        "SELECT avg(length(text)) FROM conversation_log WHERE role='assistant' AND text IS NOT NULL"
    )
    avg_response_len = round(avg_response_len or 0, 0)

    # ── Memory staleness — latest timestamp per domain ──
    staleness = []
    ts_columns = {
        "patients": "created_at",
        "medications": "created_at",
        "assessments": "updated_at",
        "seizure_diary": "created_at",
        "analyses": "created_at",
        "expert_reviews": "created_at",
        "clinical_decisions": "created_at",
        "uploads": "created_at",
        "feedback": "created_at",
        "hitl_reviews": "created_at",
    }
    for domain_name, table, _ in _DOMAINS:
        ts_col = ts_columns.get(table, "created_at")
        latest = _safe(
            cur,
            f"SELECT max({ts_col}) FROM [{table}]"
        )
        staleness.append({
            "domain": domain_name,
            "latest_update": str(latest) if latest else None,
        })

    # ── Transaction log — action types as memory write patterns ──
    action_dist = _safe_rows(
        cur,
        "SELECT action, count(*) FROM transaction_log GROUP BY action ORDER BY count(*) DESC LIMIT 15"
    )
    memory_write_patterns = [{"action": r[0], "count": r[1]} for r in action_dist]

    # ── Daily memory activity (transaction volume) ──
    daily_activity = _safe_rows(
        cur,
        "SELECT substr(ts_utc, 1, 10) as day, count(*) "
        "FROM transaction_log WHERE ts_utc IS NOT NULL "
        "GROUP BY day ORDER BY day"
    )
    daily_trend = [{"date": r[0], "transactions": r[1]} for r in daily_activity]

    # ── Patient master coverage ──
    master_count = _safe(cur, "SELECT count(*) FROM patient_master")

    conn.close()

    result["summary"] = {
        "total_patients": total_patients,
        "patients_with_memory": patients_with_any,
        "patients_no_memory": patients_with_none,
        "memory_coverage_pct": round(patients_with_any / max(total_patients, 1) * 100, 1),
        "total_memory_records": total_records,
        "total_domains": total_domains,
        "avg_completeness_pct": avg_completeness,
        "master_index_count": master_count,
        "total_conversations": total_conversations,
        "assistant_turns": assistant_turns,
        "avg_response_length": int(avg_response_len),
    }
    result["domain_fill_rates"] = domain_stats
    result["completeness_distribution"] = completeness_distribution
    result["conversation_context"] = {
        "total_turns": total_conversations,
        "assistant": assistant_turns,
        "user": user_turns,
        "system": system_turns,
    }
    result["daily_activity_trend"] = daily_trend
    result["memory_write_patterns"] = memory_write_patterns
    result["staleness"] = staleness

    return result


# ── Breakdown ─────────────────────────────────────────────────────

def memory_breakdown():
    """Per-patient memory profile, domain cross-tab, recall patterns,
    coverage gaps, component memory attribution."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── Per-patient memory profile ──
    patients = _safe_rows(
        cur,
        "SELECT patient_id, name, age, gender, disease, department FROM patients ORDER BY patient_id"
    )

    patient_profiles = []
    for pid, name, age, gender, disease, dept in patients:
        domains_filled = []
        record_counts = {}
        for domain_name, table, pid_col in _DOMAINS:
            cnt = _safe(cur, f"SELECT count(*) FROM [{table}] WHERE {pid_col}='{pid}'")
            if cnt > 0:
                domains_filled.append(domain_name)
                record_counts[domain_name] = cnt

        completeness = round(len(domains_filled) / len(_DOMAINS) * 100, 1)
        patient_profiles.append({
            "patient_id": pid,
            "name": name or pid,
            "age": age,
            "gender": gender,
            "disease": disease,
            "department": dept,
            "domains_filled": len(domains_filled),
            "total_domains": len(_DOMAINS),
            "completeness_pct": completeness,
            "filled_domains": domains_filled,
            "record_counts": record_counts,
        })

    # Sort by completeness descending
    patient_profiles.sort(key=lambda x: x["completeness_pct"], reverse=True)

    # ── Domain cross-tab: which domains co-occur ──
    domain_cooccurrence = []
    domain_names = [d[0] for d in _DOMAINS]
    for i, d1 in enumerate(domain_names):
        for d2 in domain_names[i + 1:]:
            # Count patients that have BOTH domains
            both = sum(
                1 for p in patient_profiles
                if d1 in p["filled_domains"] and d2 in p["filled_domains"]
            )
            if both > 0:
                domain_cooccurrence.append({
                    "domain_a": d1,
                    "domain_b": d2,
                    "patients_with_both": both,
                })

    # ── Coverage gaps — domains with lowest fill rate ──
    coverage_gaps = []
    for domain_name, table, pid_col in _DOMAINS:
        filled = _safe(cur, f"SELECT count(DISTINCT {pid_col}) FROM [{table}]")
        missing = max(0, len(patients) - filled)
        if missing > 0:
            coverage_gaps.append({
                "domain": domain_name,
                "patients_missing": missing,
                "fill_rate_pct": round(filled / max(len(patients), 1) * 100, 1),
            })
    coverage_gaps.sort(key=lambda x: x["fill_rate_pct"])

    # ── Component memory attribution (which components write memory) ──
    component_writes = _safe_rows(
        cur,
        "SELECT component, count(*) FROM transaction_log "
        "GROUP BY component ORDER BY count(*) DESC"
    )
    component_attribution = [
        {"component": r[0], "writes": r[1]} for r in component_writes
    ]

    # ── Actor memory attribution (who creates memory) ──
    actor_writes = _safe_rows(
        cur,
        "SELECT actor, count(*) FROM transaction_log "
        "GROUP BY actor ORDER BY count(*) DESC"
    )
    actor_attribution = [{"actor": r[0], "writes": r[1]} for r in actor_writes]

    # ── Per-disease memory depth ──
    disease_memory = defaultdict(lambda: {"patients": 0, "total_records": 0, "domains": set()})
    for p in patient_profiles:
        d = p.get("disease") or "unknown"
        disease_memory[d]["patients"] += 1
        disease_memory[d]["total_records"] += sum(p["record_counts"].values())
        disease_memory[d]["domains"].update(p["filled_domains"])

    disease_depth = [
        {
            "disease": d,
            "patients": v["patients"],
            "total_records": v["total_records"],
            "avg_records_per_patient": round(v["total_records"] / max(v["patients"], 1), 1),
            "unique_domains": len(v["domains"]),
        }
        for d, v in sorted(disease_memory.items(), key=lambda x: x[1]["total_records"], reverse=True)
    ]

    # ── Recent memory writes (last 20 transactions) ──
    recent = _safe_rows(
        cur,
        "SELECT id, patient_id, component, action, actor, detail, ts_utc "
        "FROM transaction_log ORDER BY id DESC LIMIT 20"
    )
    recent_writes = [
        {
            "id": r[0], "patient_id": r[1], "component": r[2],
            "action": r[3], "actor": r[4],
            "detail": (r[5] or "")[:120], "ts": r[6] or "",
        }
        for r in recent
    ]

    conn.close()

    return {
        "available": True,
        "patient_profiles": patient_profiles,
        "domain_cooccurrence": domain_cooccurrence,
        "coverage_gaps": coverage_gaps,
        "component_attribution": component_attribution,
        "actor_attribution": actor_attribution,
        "disease_memory_depth": disease_depth,
        "recent_memory_writes": recent_writes,
    }


# ── Definitions ───────────────────────────────────────────────────

def memory_definitions():
    """Metric definitions for the Agent Memory dashboard."""
    return {
        "definitions": [
            {
                "metric": "Memory Coverage",
                "definition": "Percentage of patients that have at least one record in any memory domain (demographics, medications, assessments, etc.).",
            },
            {
                "metric": "Avg Completeness",
                "definition": "Mean percentage of memory domains filled per patient. 100% means every patient has data in all 10 tracked domains.",
            },
            {
                "metric": "Total Memory Records",
                "definition": "Sum of all records across all memory domains (medications + assessments + seizure diary + analyses + expert reviews + clinical decisions + uploads + feedback + HITL reviews).",
            },
            {
                "metric": "Domain Fill Rate",
                "definition": "Per-domain percentage of patients who have at least one record in that domain.",
            },
            {
                "metric": "Memory Staleness",
                "definition": "Latest update timestamp per domain. Older timestamps indicate stale memory that may need refreshing.",
            },
            {
                "metric": "Completeness Distribution",
                "definition": "Histogram showing how many patients fall into each completeness bucket (0%, 1-25%, 26-50%, 51-75%, 76-100%).",
            },
            {
                "metric": "Conversation Context Depth",
                "definition": "Total conversation turns (user + assistant + system) representing the system's conversational memory.",
            },
            {
                "metric": "Memory Write Patterns",
                "definition": "Distribution of transaction log action types showing what kinds of memory writes occur most frequently.",
            },
            {
                "metric": "Component Attribution",
                "definition": "Which system components (classifier, chat, assessment, etc.) are responsible for creating memory records.",
            },
            {
                "metric": "Domain Co-occurrence",
                "definition": "Pairs of memory domains that tend to be filled together for the same patients, indicating correlated data capture.",
            },
            {
                "metric": "Coverage Gaps",
                "definition": "Domains with the lowest patient fill rates, highlighting where memory capture needs improvement.",
            },
            {
                "metric": "Disease Memory Depth",
                "definition": "Per-disease breakdown of total records and unique domains covered, showing which conditions have the richest agent memory.",
            },
        ]
    }
