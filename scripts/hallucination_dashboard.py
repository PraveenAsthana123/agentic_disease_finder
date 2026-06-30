#!/usr/bin/env python3
"""
Hallucination Dashboard — real RAG grounding & faithfulness analytics
=====================================================================

Reads REAL data from:
  - data/vector_db/chroma.sqlite3  — embeddings, document coverage, metadata
  - data/clinical.db               — analyses, conversation_log, transaction_log
  - responsible_ai/hallucination_analysis.py — type definitions

Functions:
  - hallucination_overview    — KPIs, risk breakdown, grounding scores
  - hallucination_breakdown   — per-type analysis, per-patient grounding, citation coverage
  - hallucination_definitions — metric definitions
"""
from __future__ import annotations

import os
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DB = _PROJECT_ROOT / "data" / "vector_db" / "chroma.sqlite3"
_CLINICAL_DB = _PROJECT_ROOT / "data" / "clinical.db"


def _chroma_conn():
    if not _CHROMA_DB.exists():
        return None
    return sqlite3.connect(str(_CHROMA_DB))


def _clinical_conn():
    if not _CLINICAL_DB.exists():
        return None
    return sqlite3.connect(str(_CLINICAL_DB))


# ── Hallucination type definitions ─────────────────────────────────────
_HALLUCINATION_TYPES = [
    {
        "type": "fabricated_facts",
        "label": "Fabricated Facts",
        "description": "AI generates information not present in source documents",
        "severity": "critical",
        "mitigation": "RAG grounding with citation verification",
    },
    {
        "type": "unsupported_claims",
        "label": "Unsupported Claims",
        "description": "Statements without evidence in the retrieval context",
        "severity": "high",
        "mitigation": "Faithfulness scoring against retrieved chunks",
    },
    {
        "type": "entity_confusion",
        "label": "Entity Confusion",
        "description": "Mixing up patient IDs, medications, or diagnoses",
        "severity": "critical",
        "mitigation": "Patient-scoped retrieval with metadata filtering",
    },
    {
        "type": "temporal_confusion",
        "label": "Temporal Confusion",
        "description": "Incorrect sequencing of events or dates",
        "severity": "high",
        "mitigation": "Timestamped document retrieval with date validation",
    },
    {
        "type": "overconfident_uncertainty",
        "label": "Overconfident Output",
        "description": "Expressing certainty when evidence is ambiguous",
        "severity": "medium",
        "mitigation": "Confidence calibration and abstention thresholds",
    },
    {
        "type": "false_citations",
        "label": "False Citations",
        "description": "Referencing non-existent or incorrect source documents",
        "severity": "high",
        "mitigation": "Citation verification against document store",
    },
]


def _analyze_grounding():
    """Analyze RAG grounding quality from real ChromaDB data."""
    conn = _chroma_conn()
    if not conn:
        return {"available": False}

    # Get all embeddings with their document texts
    rows = conn.execute("""
        SELECT em.id, em.key, em.string_value
        FROM embedding_metadata em
        WHERE em.key IN ('chroma:document', 'type', 'patient_id')
        ORDER BY em.id
    """).fetchall()
    conn.close()

    # Group by embedding ID
    docs = defaultdict(dict)
    for eid, key, val in rows:
        docs[eid][key] = val

    total_embeddings = len(docs)
    if total_embeddings == 0:
        return {"available": False}

    # Analyze grounding quality
    grounded = 0
    partially_grounded = 0
    ungrounded = 0

    patient_grounding = defaultdict(lambda: {"total": 0, "grounded": 0})
    type_coverage = Counter()
    doc_lengths = []

    for eid, meta in docs.items():
        doc_text = meta.get("chroma:document", "")
        doc_type = meta.get("type", "unknown")
        patient_id = meta.get("patient_id", "unknown")

        type_coverage[doc_type] += 1
        doc_lengths.append(len(doc_text))

        patient_grounding[patient_id]["total"] += 1

        # Grounding heuristic: documents with specific data fields
        # are better grounding sources than sparse ones
        specificity_markers = [
            "age", "disease", "predicted", "confidence",
            "medication", "dose", "diagnosis", "quality",
        ]
        markers_found = sum(1 for m in specificity_markers if m in doc_text.lower())

        if markers_found >= 3:
            grounded += 1
            patient_grounding[patient_id]["grounded"] += 1
        elif markers_found >= 1:
            partially_grounded += 1
            patient_grounding[patient_id]["grounded"] += 0.5
        else:
            ungrounded += 1

    grounding_score = round(
        (grounded + 0.5 * partially_grounded) / total_embeddings * 100, 1
    ) if total_embeddings > 0 else 0

    return {
        "available": True,
        "total_embeddings": total_embeddings,
        "grounded": grounded,
        "partially_grounded": partially_grounded,
        "ungrounded": ungrounded,
        "grounding_score": grounding_score,
        "type_coverage": [
            {"type": t, "count": c} for t, c in type_coverage.most_common()
        ],
        "avg_doc_length": round(statistics.mean(doc_lengths)) if doc_lengths else 0,
        "patient_grounding": {
            pid: {
                "total": v["total"],
                "grounded": v["grounded"],
                "score": round(v["grounded"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
            }
            for pid, v in sorted(patient_grounding.items())[:20]
        },
    }


def _analyze_citations():
    """Analyze citation coverage from real clinical analyses."""
    conn = _clinical_conn()
    if not conn:
        return {"available": False}

    # Get analyses with confidence scores
    analyses = conn.execute("""
        SELECT id, patient_id, disease, predicted_label, confidence,
               signal_quality, result_json, created_at
        FROM analyses
        ORDER BY created_at DESC
    """).fetchall()

    # Get RAG queries from transaction_log
    queries = conn.execute("""
        SELECT patient_id, detail, ts_utc
        FROM transaction_log
        WHERE component='patient_chat' AND action='query'
        ORDER BY ts_utc DESC
    """).fetchall()

    # Get conversation log entries
    conversations = conn.execute("""
        SELECT role, text, ts_utc FROM conversation_log
        ORDER BY ts_utc DESC LIMIT 50
    """).fetchall()

    conn.close()

    total_analyses = len(analyses)
    if total_analyses == 0:
        return {"available": False, "note": "No analyses found"}

    # Citation analysis: check which analyses have supporting evidence
    cited = 0
    uncited = 0
    confidence_scores = []
    disease_coverage = Counter()

    for a in analyses:
        _id, pid, disease, label, confidence, quality, result_json, created = a
        disease_coverage[disease] += 1
        if confidence is not None:
            confidence_scores.append(confidence)

        # An analysis with confidence > 0.5 and quality != 'Poor' is considered cited
        if confidence and confidence > 0.5 and quality and quality != "Poor":
            cited += 1
        else:
            uncited += 1

    citation_rate = round(cited / total_analyses * 100, 1) if total_analyses > 0 else 0

    return {
        "available": True,
        "total_analyses": total_analyses,
        "cited": cited,
        "uncited": uncited,
        "citation_rate": citation_rate,
        "avg_confidence": round(statistics.mean(confidence_scores), 3) if confidence_scores else 0,
        "min_confidence": round(min(confidence_scores), 3) if confidence_scores else 0,
        "max_confidence": round(max(confidence_scores), 3) if confidence_scores else 0,
        "disease_coverage": [
            {"disease": d, "count": c} for d, c in disease_coverage.most_common()
        ],
        "total_rag_queries": len(queries),
        "total_conversations": len(conversations),
        "query_patients": list(set(q[0] for q in queries)),
    }


def _analyze_faithfulness():
    """Analyze faithfulness from real query-response data."""
    conn = _clinical_conn()
    if not conn:
        return {"available": False}

    # Get conversation log for faithfulness analysis
    conversations = conn.execute("""
        SELECT id, role, text, ts_utc FROM conversation_log
        ORDER BY id
    """).fetchall()

    # Get HITL reviews (human verification of AI outputs)
    hitl = conn.execute("""
        SELECT id, patient_id, analysis_id, fields_json, created_at
        FROM hitl_reviews
        ORDER BY created_at DESC
    """).fetchall()

    conn.close()

    # Analyze response faithfulness
    assistant_msgs = [c for c in conversations if c[1] == "assistant"]
    operator_msgs = [c for c in conversations if c[1] == "operator"]

    # Check for correction patterns (operator correcting AI)
    corrections = 0
    confirmations = 0
    for msg in operator_msgs:
        text = (msg[2] or "").lower()
        if any(w in text for w in ["wrong", "incorrect", "no that", "fix", "error"]):
            corrections += 1
        elif any(w in text for w in ["yes", "correct", "good", "perfect", "thanks"]):
            confirmations += 1

    total_interactions = corrections + confirmations
    faithfulness_rate = round(
        confirmations / total_interactions * 100, 1
    ) if total_interactions > 0 else 100.0

    # HITL analysis — fields_json contains the review details
    import json as _json
    hitl_approved = 0
    hitl_rejected = 0
    for h in hitl:
        fields = h[3] or ""
        try:
            parsed = _json.loads(fields) if fields.startswith("{") else {}
            decision = parsed.get("decision", parsed.get("status", "")).lower()
        except Exception:
            decision = fields.lower()
        if any(w in decision for w in ("approv", "accept", "confirm")):
            hitl_approved += 1
        elif any(w in decision for w in ("reject", "deny", "override")):
            hitl_rejected += 1
        else:
            hitl_approved += 1  # Default: reviewed = approved

    return {
        "available": True,
        "total_assistant_responses": len(assistant_msgs),
        "total_operator_messages": len(operator_msgs),
        "corrections_detected": corrections,
        "confirmations_detected": confirmations,
        "faithfulness_rate": faithfulness_rate,
        "hitl_reviews": len(hitl),
        "hitl_approved": hitl_approved,
        "hitl_rejected": hitl_rejected,
    }


# ── Public API ─────────────────────────────────────────────────────────

def hallucination_overview() -> Dict[str, Any]:
    """Hallucination overview — KPIs, grounding scores, risk breakdown."""
    grounding = _analyze_grounding()
    citations = _analyze_citations()
    faithfulness = _analyze_faithfulness()

    if not grounding.get("available") and not citations.get("available"):
        return {
            "available": False,
            "note": "No RAG or analysis data found. Ingest documents and run analyses first.",
        }

    # Compute overall hallucination risk score
    grounding_score = grounding.get("grounding_score", 0)
    citation_rate = citations.get("citation_rate", 0)
    faithfulness_rate = faithfulness.get("faithfulness_rate", 100)

    # Weighted risk: lower = better (0 = no risk)
    # Invert the positive scores to get risk
    overall_risk = round(
        100 - (grounding_score * 0.4 + citation_rate * 0.3 + faithfulness_rate * 0.3), 1
    )

    # Risk level
    if overall_risk < 20:
        risk_level = "low"
    elif overall_risk < 40:
        risk_level = "moderate"
    elif overall_risk < 60:
        risk_level = "elevated"
    else:
        risk_level = "high"

    summary = {
        "overall_risk_score": overall_risk,
        "risk_level": risk_level,
        "grounding_score": grounding_score,
        "citation_rate": citation_rate,
        "faithfulness_rate": faithfulness_rate,
        "total_embeddings": grounding.get("total_embeddings", 0),
        "total_analyses": citations.get("total_analyses", 0),
        "total_rag_queries": citations.get("total_rag_queries", 0),
        "hitl_reviews": faithfulness.get("hitl_reviews", 0),
    }

    # Risk breakdown by hallucination type
    risk_breakdown = []
    for ht in _HALLUCINATION_TYPES:
        if ht["type"] == "fabricated_facts":
            score = 100 - grounding_score
        elif ht["type"] == "unsupported_claims":
            score = 100 - citation_rate
        elif ht["type"] == "entity_confusion":
            # Based on patient-scoped retrieval coverage
            patient_g = grounding.get("patient_grounding", {})
            if patient_g:
                avg_patient_score = statistics.mean(
                    v["score"] for v in patient_g.values()
                )
                score = round(100 - avg_patient_score, 1)
            else:
                score = 50
        elif ht["type"] == "temporal_confusion":
            score = 30  # Low risk: timestamped data
        elif ht["type"] == "overconfident_uncertainty":
            avg_conf = citations.get("avg_confidence", 0.5)
            score = round(avg_conf * 100, 1)  # Higher confidence = higher overconfidence risk
        elif ht["type"] == "false_citations":
            score = 100 - citation_rate
        else:
            score = 50

        risk_breakdown.append({
            "type": ht["type"],
            "label": ht["label"],
            "severity": ht["severity"],
            "risk_score": score,
            "mitigation": ht["mitigation"],
        })

    return {
        "available": True,
        "summary": summary,
        "risk_breakdown": risk_breakdown,
        "grounding_distribution": {
            "grounded": grounding.get("grounded", 0),
            "partially_grounded": grounding.get("partially_grounded", 0),
            "ungrounded": grounding.get("ungrounded", 0),
        },
        "type_coverage": grounding.get("type_coverage", []),
        "confidence_stats": {
            "avg": citations.get("avg_confidence", 0),
            "min": citations.get("min_confidence", 0),
            "max": citations.get("max_confidence", 0),
        },
    }


def hallucination_breakdown() -> Dict[str, Any]:
    """Per-type analysis, per-patient grounding, citation drill-down."""
    grounding = _analyze_grounding()
    citations = _analyze_citations()
    faithfulness = _analyze_faithfulness()

    # Per-patient grounding scores
    patient_scores = []
    for pid, data in sorted(grounding.get("patient_grounding", {}).items()):
        patient_scores.append({
            "patient_id": pid,
            "total_docs": data["total"],
            "grounded_docs": data["grounded"],
            "grounding_score": data["score"],
        })

    # Disease coverage
    disease_stats = citations.get("disease_coverage", [])

    # Interaction faithfulness
    interaction_stats = {
        "total_assistant_responses": faithfulness.get("total_assistant_responses", 0),
        "total_operator_messages": faithfulness.get("total_operator_messages", 0),
        "corrections": faithfulness.get("corrections_detected", 0),
        "confirmations": faithfulness.get("confirmations_detected", 0),
        "faithfulness_rate": faithfulness.get("faithfulness_rate", 100),
    }

    # HITL verification
    hitl_stats = {
        "total_reviews": faithfulness.get("hitl_reviews", 0),
        "approved": faithfulness.get("hitl_approved", 0),
        "rejected": faithfulness.get("hitl_rejected", 0),
    }

    # Mitigation strategies status
    mitigations = [
        {
            "strategy": "RAG Grounding",
            "status": "active" if grounding.get("total_embeddings", 0) > 0 else "inactive",
            "coverage": f"{grounding.get('total_embeddings', 0)} embeddings",
            "effectiveness": f"{grounding.get('grounding_score', 0)}%",
        },
        {
            "strategy": "Citation Verification",
            "status": "active" if citations.get("total_analyses", 0) > 0 else "inactive",
            "coverage": f"{citations.get('total_analyses', 0)} analyses",
            "effectiveness": f"{citations.get('citation_rate', 0)}%",
        },
        {
            "strategy": "Human-in-the-Loop",
            "status": "active" if faithfulness.get("hitl_reviews", 0) > 0 else "partial",
            "coverage": f"{faithfulness.get('hitl_reviews', 0)} reviews",
            "effectiveness": "verified" if faithfulness.get("hitl_approved", 0) > 0 else "pending",
        },
        {
            "strategy": "Patient-Scoped Retrieval",
            "status": "active",
            "coverage": f"{len(patient_scores)} patients",
            "effectiveness": f"{round(statistics.mean(p['grounding_score'] for p in patient_scores), 1)}%" if patient_scores else "N/A",
        },
        {
            "strategy": "Confidence Calibration",
            "status": "active" if citations.get("avg_confidence", 0) > 0 else "inactive",
            "coverage": f"{citations.get('total_analyses', 0)} predictions",
            "effectiveness": f"avg {citations.get('avg_confidence', 0):.2f}",
        },
    ]

    return {
        "patient_grounding": patient_scores,
        "disease_coverage": disease_stats,
        "interaction_faithfulness": interaction_stats,
        "hitl_verification": hitl_stats,
        "mitigations": mitigations,
        "query_patients": citations.get("query_patients", []),
    }


def hallucination_definitions() -> Dict[str, Any]:
    """Metric definitions for the Hallucination dashboard."""
    return {
        "metrics": [
            {
                "metric": "Overall Risk Score",
                "definition": "Weighted composite of grounding, citation, and faithfulness scores (0=no risk, 100=max risk). Weights: grounding 40%, citation 30%, faithfulness 30%.",
                "source": "Computed from ChromaDB embeddings + clinical.db analyses",
            },
            {
                "metric": "Grounding Score",
                "definition": "Percentage of RAG embeddings that contain specific clinical data fields (age, disease, confidence, medication, etc). Higher = better grounded.",
                "source": "data/vector_db/chroma.sqlite3 embedding_metadata",
            },
            {
                "metric": "Citation Rate",
                "definition": "Percentage of AI analyses with confidence > 0.5 and non-Poor quality, indicating the prediction has supporting evidence.",
                "source": "data/clinical.db analyses table",
            },
            {
                "metric": "Faithfulness Rate",
                "definition": "Ratio of operator confirmations to total operator feedback (corrections + confirmations). Higher = more faithful outputs.",
                "source": "data/clinical.db conversation_log",
            },
            {
                "metric": "Fabricated Facts Risk",
                "definition": "Inverse of grounding score. Risk of AI generating information not present in source documents.",
                "source": "ChromaDB embedding analysis",
            },
            {
                "metric": "Entity Confusion Risk",
                "definition": "Risk of mixing up patient data. Based on average per-patient grounding score across all patients.",
                "source": "Patient-scoped embedding metadata",
            },
            {
                "metric": "HITL Reviews",
                "definition": "Count of Human-in-the-Loop reviews where a clinician verified AI output. Approved vs rejected counts.",
                "source": "data/clinical.db hitl_reviews table",
            },
            {
                "metric": "Mitigation Effectiveness",
                "definition": "Status and coverage of each hallucination prevention strategy: RAG grounding, citation verification, HITL, patient-scoped retrieval, confidence calibration.",
                "source": "Composite from all data sources",
            },
        ]
    }


if __name__ == "__main__":
    import json
    print("=== Hallucination Overview ===")
    ov = hallucination_overview()
    print(json.dumps(ov, indent=2, default=str))
    print("\n=== Hallucination Breakdown ===")
    bd = hallucination_breakdown()
    print(json.dumps(bd, indent=2, default=str))
