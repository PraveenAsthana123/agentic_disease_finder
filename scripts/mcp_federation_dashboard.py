"""MCP Federation Dashboard — real cross-component federation analytics from clinical.db.

Analyses MCP (Model Context Protocol) federation patterns: how components
interconnect, which tools/actions flow between services, federation topology,
protocol compliance (guardrails, security checks), actor participation across
federated nodes, and service mesh health.

Sources:
- transaction_log (558)   — cross-component action audit trail
- conversation_log (225+) — inter-service message flow
- analyses (21)           — federated analysis pipeline outputs
- component_findings      — cross-component finding reports
- expert_reviews (3)      — human oversight across federation
- clinical_decisions (1)  — federated decision routing
- hitl_reviews (2)        — human-in-the-loop federation gate
"""

import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


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


# ── MCP node classification ──────────────────────────────────────

_NODE_CLASSES = {
    "patient_master": "data-ingestion",
    "eeg_upload": "data-ingestion",
    "video_frames": "data-ingestion",
    "cv_pipeline": "ml-processing",
    "training": "ml-processing",
    "drift": "ml-processing",
    "consistency": "ml-processing",
    "fairness": "ml-processing",
    "clinical_trust": "ml-processing",
    "assessment": "clinical-service",
    "medications": "clinical-service",
    "seizure_diary": "clinical-service",
    "expert_review": "clinical-oversight",
    "consultant:neurologist": "clinical-oversight",
    "council": "clinical-oversight",
    "component_finding": "clinical-oversight",
    "patient_chat": "communication",
    "team_chat": "communication",
    "chat_group": "communication",
    "genai_bot": "ai-agent",
    "graph_db": "data-store",
    "form": "workflow",
    "feedback": "feedback-loop",
}


def _classify_node(component):
    return _NODE_CLASSES.get(component, "other")


# ── Overview ─────────────────────────────────────────────────────

def federation_overview():
    """Federation KPIs: node count, edge count, throughput, actor distribution,
    class breakdown, cross-component rate, daily trend."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc)
    result = {"available": True, "generated_at": now.isoformat()}

    # ── Federation nodes (distinct components) ──
    components = _safe_rows(
        cur, "SELECT DISTINCT component FROM transaction_log ORDER BY component"
    )
    node_list = [r[0] for r in components]
    total_nodes = len(node_list)

    # ── Per-node transaction counts ──
    node_volumes = _safe_rows(
        cur,
        "SELECT component, count(*) FROM transaction_log "
        "GROUP BY component ORDER BY count(*) DESC"
    )
    node_volume_map = {r[0]: r[1] for r in node_volumes}

    # Classify nodes
    node_class_counts = Counter()
    node_details = []
    for comp in node_list:
        cls = _classify_node(comp)
        node_class_counts[cls] += 1
        vol = node_volume_map.get(comp, 0)
        node_details.append({
            "node": comp,
            "class": cls,
            "transactions": vol,
        })

    node_class_distribution = [
        {"class": k, "count": v}
        for k, v in sorted(node_class_counts.items(), key=lambda x: -x[1])
    ]

    # ── Total transactions (federation throughput) ──
    total_tx = _safe(cur, "SELECT count(*) FROM transaction_log")

    # ── Distinct actors across federation ──
    actors = _safe_rows(
        cur, "SELECT actor, count(*) FROM transaction_log GROUP BY actor ORDER BY count(*) DESC"
    )
    actor_distribution = [{"actor": r[0], "transactions": r[1]} for r in actors]
    total_actors = len(actors)

    # ── Distinct actions (protocol verbs) ──
    actions = _safe_rows(
        cur, "SELECT action, count(*) FROM transaction_log GROUP BY action ORDER BY count(*) DESC"
    )
    action_distribution = [{"action": r[0], "count": r[1]} for r in actions]
    total_actions = len(actions)

    # ── Cross-component federation edges ──
    # Two components are "federated" if the same patient_id has transactions in both
    patient_components = defaultdict(set)
    pc_rows = _safe_rows(
        cur,
        "SELECT patient_id, component FROM transaction_log "
        "WHERE patient_id IS NOT NULL AND patient_id != ''"
    )
    for pid, comp in pc_rows:
        patient_components[pid].add(comp)

    edge_counter = Counter()
    for pid, comps in patient_components.items():
        for a, b in combinations(sorted(comps), 2):
            edge_counter[(a, b)] += 1

    federation_edges = [
        {"source": e[0], "target": e[1], "shared_patients": c}
        for e, c in edge_counter.most_common(30)
    ]
    total_edges = len(edge_counter)

    # Cross-component rate: % of patients touched by 2+ components
    multi_component_patients = sum(
        1 for comps in patient_components.values() if len(comps) >= 2
    )
    total_patients_in_tx = len(patient_components)
    cross_component_rate = round(
        multi_component_patients / max(total_patients_in_tx, 1) * 100, 1
    )

    # ── Daily federation volume ──
    daily = _safe_rows(
        cur,
        "SELECT substr(ts_utc, 1, 10) as day, count(*) "
        "FROM transaction_log WHERE ts_utc IS NOT NULL "
        "GROUP BY day ORDER BY day"
    )
    daily_trend = [{"date": r[0], "transactions": r[1]} for r in daily]

    # ── Hourly pattern ──
    hourly = _safe_rows(
        cur,
        "SELECT substr(ts_utc, 12, 2) as hour, count(*) "
        "FROM transaction_log WHERE ts_utc IS NOT NULL AND length(ts_utc)>=13 "
        "GROUP BY hour ORDER BY hour"
    )
    hourly_pattern = [{"hour": r[0], "count": r[1]} for r in hourly]

    # ── Protocol compliance signals ──
    # Security/compliance actors present?
    compliance_actors = _safe(
        cur,
        "SELECT count(*) FROM transaction_log "
        "WHERE actor IN ('compliance_agent','security_agent')"
    )
    # Sign-off actions (human gate)
    signoff_count = _safe(
        cur,
        "SELECT count(*) FROM transaction_log WHERE action='sign-off'"
    )
    # Expert reviews (oversight)
    expert_count = _safe(cur, "SELECT count(*) FROM expert_reviews")
    # HITL reviews
    hitl_count = _safe(cur, "SELECT count(*) FROM hitl_reviews")

    oversight_total = compliance_actors + signoff_count + expert_count + hitl_count
    oversight_rate = round(oversight_total / max(total_tx, 1) * 100, 1)

    # ── Conversation federation — messages across patients ──
    conv_total = _safe(cur, "SELECT count(*) FROM conversation_log")
    conv_patients = _safe(
        cur, "SELECT count(DISTINCT patient_id) FROM conversation_log"
    )

    conn.close()

    result["summary"] = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "total_transactions": total_tx,
        "total_actors": total_actors,
        "total_protocol_verbs": total_actions,
        "cross_component_rate_pct": cross_component_rate,
        "multi_component_patients": multi_component_patients,
        "total_patients_in_federation": total_patients_in_tx,
        "oversight_events": oversight_total,
        "oversight_rate_pct": oversight_rate,
        "conversation_messages": conv_total,
        "conversation_patients": conv_patients,
    }
    result["node_details"] = node_details
    result["node_class_distribution"] = node_class_distribution
    result["actor_distribution"] = actor_distribution
    result["action_distribution"] = action_distribution
    result["daily_trend"] = daily_trend
    result["hourly_pattern"] = hourly_pattern
    result["top_federation_edges"] = federation_edges

    return result


# ── Breakdown ────────────────────────────────────────────────────

def federation_breakdown():
    """Detailed federation topology: per-patient component coverage,
    service mesh adjacency, protocol verb matrix, node-class cross-tab,
    recent federation events, component pair analysis."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── Per-patient federation profile ──
    patients = _safe_rows(
        cur,
        "SELECT patient_id, name, disease, department FROM patients ORDER BY patient_id"
    )

    patient_federation = []
    for pid, name, disease, dept in patients:
        comps = _safe_rows(
            cur,
            f"SELECT DISTINCT component FROM transaction_log WHERE patient_id='{pid}'"
        )
        comp_list = [r[0] for r in comps]
        tx_count = _safe(
            cur,
            f"SELECT count(*) FROM transaction_log WHERE patient_id='{pid}'"
        )
        actors_involved = _safe_rows(
            cur,
            f"SELECT DISTINCT actor FROM transaction_log WHERE patient_id='{pid}'"
        )
        actor_list = [r[0] for r in actors_involved]

        patient_federation.append({
            "patient_id": pid,
            "name": name or pid,
            "disease": disease,
            "department": dept,
            "components_touched": len(comp_list),
            "component_list": comp_list,
            "total_transactions": tx_count,
            "actors_involved": actor_list,
        })

    patient_federation.sort(key=lambda x: x["components_touched"], reverse=True)

    # ── Service mesh adjacency matrix ──
    # Component-to-component co-occurrence per patient
    patient_components = defaultdict(set)
    pc_rows = _safe_rows(
        cur,
        "SELECT patient_id, component FROM transaction_log "
        "WHERE patient_id IS NOT NULL AND patient_id != ''"
    )
    for pid, comp in pc_rows:
        patient_components[pid].add(comp)

    # Build adjacency
    adjacency = defaultdict(int)
    for pid, comps in patient_components.items():
        for a, b in combinations(sorted(comps), 2):
            adjacency[(a, b)] += 1

    mesh_links = [
        {"source": k[0], "target": k[1], "weight": v}
        for k, v in sorted(adjacency.items(), key=lambda x: -x[1])
    ]

    # ── Protocol verb matrix: component × action ──
    verb_matrix_rows = _safe_rows(
        cur,
        "SELECT component, action, count(*) "
        "FROM transaction_log GROUP BY component, action "
        "ORDER BY component, count(*) DESC"
    )
    verb_matrix = defaultdict(dict)
    for comp, action, cnt in verb_matrix_rows:
        verb_matrix[comp][action] = cnt

    verb_matrix_table = [
        {"component": comp, "actions": actions}
        for comp, actions in sorted(verb_matrix.items())
    ]

    # ── Node-class cross-tab: class × action totals ──
    class_action = defaultdict(lambda: Counter())
    for comp, action, cnt in verb_matrix_rows:
        cls = _classify_node(comp)
        class_action[cls][action] += cnt

    class_action_table = [
        {"class": cls, "actions": dict(acts)}
        for cls, acts in sorted(class_action.items())
    ]

    # ── Component pair analysis: most connected pairs ──
    pair_details = []
    for (src, tgt), weight in sorted(adjacency.items(), key=lambda x: -x[1])[:15]:
        src_cls = _classify_node(src)
        tgt_cls = _classify_node(tgt)
        pair_details.append({
            "source": src,
            "source_class": src_cls,
            "target": tgt,
            "target_class": tgt_cls,
            "shared_patients": weight,
            "federation_type": f"{src_cls} ↔ {tgt_cls}",
        })

    # ── Actor × component cross-tab ──
    actor_comp_rows = _safe_rows(
        cur,
        "SELECT actor, component, count(*) "
        "FROM transaction_log GROUP BY actor, component "
        "ORDER BY actor, count(*) DESC"
    )
    actor_component = defaultdict(dict)
    for actor, comp, cnt in actor_comp_rows:
        actor_component[actor][comp] = cnt

    actor_component_table = [
        {"actor": actor, "components": comps}
        for actor, comps in sorted(actor_component.items())
    ]

    # ── Recent federation events (last 20 transactions) ──
    recent = _safe_rows(
        cur,
        "SELECT patient_id, component, action, actor, detail, ts_utc "
        "FROM transaction_log ORDER BY id DESC LIMIT 20"
    )
    recent_events = [
        {
            "patient_id": r[0],
            "component": r[1],
            "action": r[2],
            "actor": r[3],
            "detail": (r[4] or "")[:120],
            "timestamp": r[5],
        }
        for r in recent
    ]

    # ── Federation depth per disease ──
    disease_fed = defaultdict(lambda: {"patients": 0, "components": set(), "transactions": 0})
    for p in patient_federation:
        d = p.get("disease") or "unknown"
        disease_fed[d]["patients"] += 1
        disease_fed[d]["components"].update(p["component_list"])
        disease_fed[d]["transactions"] += p["total_transactions"]

    disease_federation = [
        {
            "disease": d,
            "patients": v["patients"],
            "unique_components": len(v["components"]),
            "total_transactions": v["transactions"],
            "avg_components_per_patient": round(
                sum(p["components_touched"] for p in patient_federation if (p.get("disease") or "unknown") == d)
                / max(v["patients"], 1), 1
            ),
        }
        for d, v in sorted(disease_fed.items(), key=lambda x: -x[1]["transactions"])
    ]

    conn.close()

    return {
        "available": True,
        "patient_federation": patient_federation,
        "mesh_links": mesh_links,
        "verb_matrix": verb_matrix_table,
        "class_action_crosstab": class_action_table,
        "component_pairs": pair_details,
        "actor_component_crosstab": actor_component_table,
        "recent_events": recent_events,
        "disease_federation": disease_federation,
    }


# ── Definitions ──────────────────────────────────────────────────

def federation_definitions():
    """Metric definitions for the MCP Federation dashboard."""
    return {
        "metrics": [
            {
                "name": "Total Nodes",
                "definition": "Distinct MCP components (services) observed in the transaction log.",
                "source": "transaction_log.component",
            },
            {
                "name": "Total Edges",
                "definition": "Unique component pairs that share at least one patient — indicates federation link.",
                "source": "transaction_log (patient_id × component co-occurrence)",
            },
            {
                "name": "Cross-Component Rate",
                "definition": "Percentage of patients whose data flows through 2+ MCP components.",
                "source": "transaction_log patient_id with ≥2 distinct components",
            },
            {
                "name": "Oversight Rate",
                "definition": "Ratio of oversight events (sign-off, expert review, HITL, compliance/security agent) to total transactions.",
                "source": "transaction_log (action='sign-off') + expert_reviews + hitl_reviews + compliance/security actors",
            },
            {
                "name": "Node Class",
                "definition": "Functional category of a component: data-ingestion, ml-processing, clinical-service, clinical-oversight, communication, ai-agent, data-store, workflow, feedback-loop.",
                "source": "Hardcoded classification of transaction_log.component values.",
            },
            {
                "name": "Federation Edge Weight",
                "definition": "Number of distinct patients shared between two components — higher weight = tighter coupling.",
                "source": "transaction_log patient_id intersection per component pair.",
            },
            {
                "name": "Protocol Verb",
                "definition": "The action field in transaction_log: ingest, create, query, sign-off, analyze, etc. Represents MCP protocol operations.",
                "source": "transaction_log.action",
            },
            {
                "name": "Service Mesh",
                "definition": "The graph of component-to-component connections weighted by shared patient data flow.",
                "source": "Derived from transaction_log co-occurrence analysis.",
            },
            {
                "name": "Federation Depth (per disease)",
                "definition": "How many unique MCP components are involved in managing patients with a given disease.",
                "source": "patients.disease + transaction_log.component per patient.",
            },
        ]
    }
