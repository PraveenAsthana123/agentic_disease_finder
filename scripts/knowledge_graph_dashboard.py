#!/usr/bin/env python3
"""
Knowledge Graph Dashboard — real entity-relationship analytics
==============================================================

Reads REAL data from:
  - data/clinical.db   — patients, analyses, medications, mri_findings,
                          neuropsych, hitl_reviews, seizure_metadata, comorbidities
  - data/vector_db/chroma.sqlite3 — embedded document counts per patient/type

Builds a knowledge graph of entity nodes (patients, diseases, medications,
departments, document types) and relationship edges (diagnosed_with,
prescribed, analyzed_by, reviewed_by, embedded_as, etc.) from actual
clinical data — no fabricated nodes.

Functions:
  - knowledge_graph_overview   — KPIs, node/edge counts, entity type distribution
  - knowledge_graph_breakdown  — full node list, edge list, per-patient subgraph stats
  - knowledge_graph_definitions — metric definitions
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CLINICAL_DB = _PROJECT_ROOT / "data" / "clinical.db"
_CHROMA_DB = _PROJECT_ROOT / "data" / "vector_db" / "chroma.sqlite3"


def _clinical_conn():
    if not _CLINICAL_DB.exists():
        return None
    return sqlite3.connect(str(_CLINICAL_DB))


def _chroma_conn():
    if not _CHROMA_DB.exists():
        return None
    return sqlite3.connect(str(_CHROMA_DB))


# ── Graph builder ───────────────────────────────────────────────────
def _build_graph() -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """Extract nodes and edges from clinical.db + ChromaDB.

    Returns (nodes, edges, stats).
    """
    nodes: Dict[str, Dict[str, Any]] = {}  # id -> {id, type, label, ...}
    edges: List[Dict[str, Any]] = []
    stats: Dict[str, int] = defaultdict(int)

    def add_node(nid: str, ntype: str, label: str, **extra):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "type": ntype, "label": label, **extra}
            stats[f"node_{ntype}"] += 1

    def add_edge(src: str, tgt: str, rel: str, **extra):
        edges.append({"source": src, "target": tgt, "relation": rel, **extra})
        stats[f"edge_{rel}"] += 1

    conn = _clinical_conn()
    if conn is None:
        return [], [], dict(stats)
    cur = conn.cursor()

    # --- Patients ---
    cur.execute("SELECT patient_id, name, age, gender, disease, department FROM patients")
    for pid, name, age, gender, disease, dept in cur.fetchall():
        add_node(f"patient:{pid}", "patient", pid, age=age, gender=gender)
        if disease:
            disease_id = f"disease:{disease}"
            add_node(disease_id, "disease", disease)
            add_edge(f"patient:{pid}", disease_id, "diagnosed_with")
        if dept:
            dept_id = f"department:{dept}"
            add_node(dept_id, "department", dept)
            add_edge(f"patient:{pid}", dept_id, "treated_in")

    # --- Analyses ---
    cur.execute("SELECT id, patient_id, disease, predicted_label, confidence FROM analyses")
    for aid, pid, disease, label, conf in cur.fetchall():
        analysis_id = f"analysis:{aid}"
        add_node(analysis_id, "analysis", f"Analysis #{aid}",
                 predicted=label, confidence=round(conf, 2) if conf else None)
        add_edge(f"patient:{pid}", analysis_id, "has_analysis")
        if disease:
            add_edge(analysis_id, f"disease:{disease}", "classifies")

    # --- Medications ---
    cur.execute("SELECT id, patient_id, fields_json FROM medications")
    for mid, pid, fields_raw in cur.fetchall():
        fields = {}
        if fields_raw:
            try:
                fields = json.loads(fields_raw)
            except (json.JSONDecodeError, TypeError):
                pass
        drug = fields.get("drug_name", f"med_{mid}")
        drug_id = f"medication:{drug}"
        add_node(drug_id, "medication", drug,
                 dose_mg=fields.get("dose_mg"), frequency=fields.get("frequency"))
        add_edge(f"patient:{pid}", drug_id, "prescribed")
        # AED list (additional anti-epileptic drugs)
        for aed in fields.get("aed", []):
            aed_id = f"medication:{aed}"
            add_node(aed_id, "medication", aed)
            add_edge(f"patient:{pid}", aed_id, "prescribed")

    # --- MRI findings ---
    cur.execute("SELECT id, patient_id, fields_json FROM mri_findings")
    for fid, pid, fields_raw in cur.fetchall():
        mri_id = f"mri:{fid}"
        add_node(mri_id, "mri_finding", f"MRI #{fid}")
        add_edge(f"patient:{pid}", mri_id, "has_mri")

    # --- Neuropsych ---
    cur.execute("SELECT id, patient_id FROM neuropsych")
    for nid, pid in cur.fetchall():
        np_id = f"neuropsych:{nid}"
        add_node(np_id, "neuropsych", f"Neuropsych #{nid}")
        add_edge(f"patient:{pid}", np_id, "has_neuropsych")

    # --- HITL reviews ---
    cur.execute("SELECT id, patient_id, analysis_id FROM hitl_reviews")
    for rid, pid, aid in cur.fetchall():
        review_id = f"review:{rid}"
        add_node(review_id, "hitl_review", f"HITL Review #{rid}")
        add_edge(f"patient:{pid}", review_id, "reviewed_by")
        if aid:
            add_edge(review_id, f"analysis:{aid}", "reviews")

    # --- Seizure metadata ---
    try:
        cur.execute("SELECT id, patient_id FROM seizure_metadata")
        for sid, pid in cur.fetchall():
            sz_id = f"seizure:{sid}"
            add_node(sz_id, "seizure_event", f"Seizure #{sid}")
            add_edge(f"patient:{pid}", sz_id, "has_seizure")
    except Exception:
        pass

    conn.close()

    # --- ChromaDB document embeddings ---
    chroma = _chroma_conn()
    if chroma:
        cc = chroma.cursor()
        try:
            cc.execute(
                "SELECT em_pid.string_value, em_type.string_value, count(*) "
                "FROM embedding_metadata em_pid "
                "JOIN embedding_metadata em_type "
                "  ON em_pid.id = em_type.id "
                "WHERE em_pid.key = 'patient_id' AND em_type.key = 'type' "
                "GROUP BY em_pid.string_value, em_type.string_value"
            )
            for pid, doc_type, cnt in cc.fetchall():
                if pid and doc_type:
                    dt_id = f"doctype:{doc_type}"
                    add_node(dt_id, "document_type", doc_type)
                    add_edge(f"patient:{pid}", dt_id, "embedded_as",
                             chunk_count=cnt)
        except Exception:
            pass
        chroma.close()

    stats["total_nodes"] = len(nodes)
    stats["total_edges"] = len(edges)

    return list(nodes.values()), edges, dict(stats)


# ── Overview ────────────────────────────────────────────────────────
def knowledge_graph_overview() -> Dict[str, Any]:
    """Top-level KPIs: node/edge counts, entity type distribution, connectivity."""
    nodes, edges, stats = _build_graph()
    if not nodes:
        return {"available": False, "note": "clinical.db not found — no graph data."}

    # Entity type distribution
    type_counts = Counter(n["type"] for n in nodes)
    type_distribution = [
        {"type": t, "count": c}
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
    ]

    # Relation type distribution
    rel_counts = Counter(e["relation"] for e in edges)
    relation_distribution = [
        {"relation": r, "count": c}
        for r, c in sorted(rel_counts.items(), key=lambda x: -x[1])
    ]

    # Connectivity — degree per node
    degree: Dict[str, int] = defaultdict(int)
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    degrees = list(degree.values()) if degree else [0]
    avg_degree = sum(degrees) / len(degrees)
    max_degree = max(degrees)
    hub_nodes = [
        {"id": nid, "degree": d, "type": next((n["type"] for n in nodes if n["id"] == nid), "?")}
        for nid, d in sorted(degree.items(), key=lambda x: -x[1])[:5]
    ]

    # Isolated nodes (degree 0)
    connected_ids = set()
    for e in edges:
        connected_ids.add(e["source"])
        connected_ids.add(e["target"])
    isolated = [n for n in nodes if n["id"] not in connected_ids]

    return {
        "available": True,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": len(type_counts),
            "relation_types": len(rel_counts),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max_degree,
            "isolated_nodes": len(isolated),
            "density": round(2 * len(edges) / (len(nodes) * (len(nodes) - 1)), 6)
            if len(nodes) > 1 else 0,
        },
        "type_distribution": type_distribution,
        "relation_distribution": relation_distribution,
        "hub_nodes": hub_nodes,
    }


# ── Breakdown ───────────────────────────────────────────────────────
def knowledge_graph_breakdown() -> Dict[str, Any]:
    """Full node list, edge list, per-patient subgraph stats."""
    nodes, edges, stats = _build_graph()
    if not nodes:
        return {"available": False, "note": "clinical.db not found."}

    # Per-patient subgraph: how many edges each patient has
    patient_nodes = [n for n in nodes if n["type"] == "patient"]
    patient_subgraphs = []
    for pn in patient_nodes:
        pid = pn["id"]
        p_edges = [e for e in edges if e["source"] == pid or e["target"] == pid]
        neighbors: Set[str] = set()
        for e in p_edges:
            neighbors.add(e["source"])
            neighbors.add(e["target"])
        neighbors.discard(pid)
        rel_types = Counter(e["relation"] for e in p_edges)
        patient_subgraphs.append({
            "patient_id": pn["label"],
            "edges": len(p_edges),
            "neighbors": len(neighbors),
            "relations": dict(rel_types),
        })
    patient_subgraphs.sort(key=lambda x: -x["edges"])

    # Disease clusters — which diseases connect to how many patients
    disease_nodes = [n for n in nodes if n["type"] == "disease"]
    disease_clusters = []
    for dn in disease_nodes:
        d_edges = [e for e in edges if e["target"] == dn["id"] and e["relation"] == "diagnosed_with"]
        disease_clusters.append({
            "disease": dn["label"],
            "patient_count": len(d_edges),
        })
    disease_clusters.sort(key=lambda x: -x["patient_count"])

    # Medication network — which meds are most prescribed
    med_nodes = [n for n in nodes if n["type"] == "medication"]
    med_usage = []
    for mn in med_nodes:
        m_edges = [e for e in edges if e["target"] == mn["id"] and e["relation"] == "prescribed"]
        med_usage.append({
            "medication": mn["label"],
            "patients": len(m_edges),
            "dose_mg": mn.get("dose_mg"),
            "frequency": mn.get("frequency"),
        })
    med_usage.sort(key=lambda x: -x["patients"])

    return {
        "available": True,
        "generated_at": datetime.now().isoformat(),
        "graph_stats": stats,
        "patient_subgraphs": patient_subgraphs,
        "disease_clusters": disease_clusters,
        "medication_network": med_usage,
        "nodes": nodes,
        "edges": edges[:500],  # cap for response size
    }


# ── Definitions ─────────────────────────────────────────────────────
def knowledge_graph_definitions() -> Dict[str, Any]:
    return {
        "available": True,
        "metrics": [
            {
                "name": "Total Nodes",
                "description": "Number of unique entities (patients, diseases, medications, analyses, etc.) in the knowledge graph.",
                "unit": "count",
            },
            {
                "name": "Total Edges",
                "description": "Number of relationships connecting entities (diagnosed_with, prescribed, has_analysis, etc.).",
                "unit": "count",
            },
            {
                "name": "Entity Types",
                "description": "Number of distinct entity categories: patient, disease, medication, department, analysis, mri_finding, neuropsych, hitl_review, seizure_event, document_type.",
                "unit": "count",
            },
            {
                "name": "Relation Types",
                "description": "Number of distinct edge types: diagnosed_with, prescribed, has_analysis, classifies, treated_in, has_mri, has_neuropsych, reviewed_by, reviews, embedded_as.",
                "unit": "count",
            },
            {
                "name": "Avg Degree",
                "description": "Average number of connections per node. Higher = denser graph.",
                "unit": "edges/node",
            },
            {
                "name": "Max Degree",
                "description": "Highest connection count for any single node (hub node).",
                "unit": "edges",
            },
            {
                "name": "Graph Density",
                "description": "Ratio of actual edges to maximum possible edges (0 = sparse, 1 = fully connected).",
                "unit": "ratio",
            },
            {
                "name": "Hub Nodes",
                "description": "Top-5 most connected entities. Hub nodes are critical for graph traversal and indicate key entities.",
                "unit": "list",
            },
            {
                "name": "Patient Subgraph",
                "description": "Per-patient view showing how many edges and neighbors each patient has — measures clinical data richness.",
                "unit": "per-patient",
            },
            {
                "name": "Disease Cluster",
                "description": "How many patients cluster around each disease node.",
                "unit": "per-disease",
            },
            {
                "name": "Medication Network",
                "description": "Which medications are prescribed to how many patients, with dosage and frequency.",
                "unit": "per-medication",
            },
        ],
        "entity_types": [
            {"type": "patient", "source": "clinical.db patients table", "description": "Individual patient records"},
            {"type": "disease", "source": "clinical.db patients.disease", "description": "Diagnosed conditions (epilepsy, etc.)"},
            {"type": "medication", "source": "clinical.db medications.fields_json", "description": "Prescribed drugs and AEDs"},
            {"type": "department", "source": "clinical.db patients.department", "description": "Clinical departments"},
            {"type": "analysis", "source": "clinical.db analyses table", "description": "EEG classification results"},
            {"type": "mri_finding", "source": "clinical.db mri_findings table", "description": "MRI scan findings"},
            {"type": "neuropsych", "source": "clinical.db neuropsych table", "description": "Neuropsychological assessments"},
            {"type": "hitl_review", "source": "clinical.db hitl_reviews table", "description": "Human-in-the-loop reviews"},
            {"type": "seizure_event", "source": "clinical.db seizure_metadata", "description": "Recorded seizure events"},
            {"type": "document_type", "source": "ChromaDB embedding_metadata", "description": "Types of embedded RAG documents"},
        ],
        "relation_types": [
            {"relation": "diagnosed_with", "description": "Patient → Disease edge"},
            {"relation": "prescribed", "description": "Patient → Medication edge"},
            {"relation": "has_analysis", "description": "Patient → Analysis edge"},
            {"relation": "classifies", "description": "Analysis → Disease edge"},
            {"relation": "treated_in", "description": "Patient → Department edge"},
            {"relation": "has_mri", "description": "Patient → MRI Finding edge"},
            {"relation": "has_neuropsych", "description": "Patient → Neuropsych Assessment edge"},
            {"relation": "reviewed_by", "description": "Patient → HITL Review edge"},
            {"relation": "reviews", "description": "HITL Review → Analysis edge"},
            {"relation": "has_seizure", "description": "Patient → Seizure Event edge"},
            {"relation": "embedded_as", "description": "Patient → Document Type edge (from vector DB)"},
        ],
    }
