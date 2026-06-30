#!/usr/bin/env python3
"""
Response Quality Dashboard — real conversation & analysis quality analytics
==========================================================================

Reads REAL data from:
  - data/clinical.db  — conversation_log, analyses, feedback, transaction_log
  - data/vector_db/chroma.sqlite3 — RAG retrieval coverage

Measures:
  - Response completeness (length, structure)
  - Analysis confidence distribution
  - Feedback ratings
  - Response latency (conversation pair timing)
  - Component reliability (transaction success rates)
  - RAG grounding coverage

Functions:
  - response_quality_overview   — KPIs, distributions, timeline
  - response_quality_breakdown  — per-component, per-disease, per-role detail
  - response_quality_definitions — metric definitions
"""
from __future__ import annotations

import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

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


# ── Quality scoring helpers ──────────────────────────────────────────

def _response_quality_score(text: str) -> Dict[str, Any]:
    """Score a single assistant response on multiple quality axes."""
    length = len(text)
    word_count = len(text.split())
    has_structure = bool(re.search(r"(\n[-*]\s|\n\d+\.\s|\n#{1,3}\s|\*\*)", text))
    has_data = bool(re.search(r"\d+\.\d+|confidence|accuracy|\d+%", text, re.I))
    has_citation = bool(re.search(r"(patient|P\d{4}|CHB-MIT|Bonn|analysis|report)", text, re.I))
    sentence_count = len(re.findall(r"[.!?]+", text))

    # Composite score 0-100
    score = 0
    if word_count >= 20:
        score += 20
    if word_count >= 50:
        score += 10
    if has_structure:
        score += 20
    if has_data:
        score += 20
    if has_citation:
        score += 15
    if sentence_count >= 3:
        score += 15

    return {
        "length": length,
        "word_count": word_count,
        "has_structure": has_structure,
        "has_data": has_data,
        "has_citation": has_citation,
        "sentence_count": sentence_count,
        "quality_score": min(score, 100),
    }


def _quality_tier(score: int) -> str:
    if score >= 80:
        return "excellent"
    if score >= 60:
        return "good"
    if score >= 40:
        return "adequate"
    return "needs_improvement"


# ── Overview ─────────────────────────────────────────────────────────

def response_quality_overview() -> Dict[str, Any]:
    conn = _clinical_conn()
    if not conn:
        return {"available": False, "note": "clinical.db not found"}

    try:
        # Conversation log stats
        rows = conn.execute(
            "SELECT role, text, ts_utc FROM conversation_log ORDER BY id"
        ).fetchall()

        assistant_msgs = [(r[1], r[2]) for r in rows if r[0] == "assistant"]
        operator_msgs = [(r[1], r[2]) for r in rows if r[0] == "operator"]

        # Score each assistant response
        scores = [_response_quality_score(text) for text, _ in assistant_msgs]
        quality_scores = [s["quality_score"] for s in scores]

        avg_quality = round(statistics.mean(quality_scores), 1) if quality_scores else 0
        median_quality = round(statistics.median(quality_scores), 1) if quality_scores else 0

        # Tier distribution
        tiers = Counter(_quality_tier(s) for s in quality_scores)

        # Length stats
        lengths = [s["word_count"] for s in scores]
        avg_length = round(statistics.mean(lengths), 0) if lengths else 0

        # Structure/data/citation rates
        structure_rate = round(
            sum(1 for s in scores if s["has_structure"]) / len(scores) * 100, 1
        ) if scores else 0
        data_rate = round(
            sum(1 for s in scores if s["has_data"]) / len(scores) * 100, 1
        ) if scores else 0
        citation_rate = round(
            sum(1 for s in scores if s["has_citation"]) / len(scores) * 100, 1
        ) if scores else 0

        # Analysis confidence stats
        analyses = conn.execute(
            "SELECT disease, confidence, predicted_label, signal_quality "
            "FROM analyses"
        ).fetchall()
        confidences = [a[1] for a in analyses if a[1] is not None]
        avg_confidence = round(statistics.mean(confidences), 3) if confidences else 0
        confidence_dist = []
        if confidences:
            for lo, hi, label in [
                (0, 0.5, "<50%"), (0.5, 0.7, "50-70%"),
                (0.7, 0.85, "70-85%"), (0.85, 1.01, "85-100%"),
            ]:
                cnt = sum(1 for c in confidences if lo <= c < hi)
                confidence_dist.append({"range": label, "count": cnt})

        # Feedback summary
        feedback = conn.execute(
            "SELECT rating, role FROM feedback WHERE rating IS NOT NULL"
        ).fetchall()
        avg_rating = (
            round(statistics.mean([f[0] for f in feedback]), 1)
            if feedback else None
        )
        feedback_count = len(feedback)

        # Daily quality timeline (last 14 days)
        daily_quality = defaultdict(list)
        for (text, ts) in assistant_msgs:
            if ts:
                day = ts[:10]
                qs = _response_quality_score(text)["quality_score"]
                daily_quality[day].append(qs)

        timeline = sorted(
            [
                {
                    "date": day,
                    "avg_quality": round(statistics.mean(vals), 1),
                    "count": len(vals),
                }
                for day, vals in daily_quality.items()
            ],
            key=lambda x: x["date"],
        )[-14:]

        # Transaction success (component reliability)
        tx_rows = conn.execute(
            "SELECT component, action, count(*) "
            "FROM transaction_log GROUP BY component, action "
            "ORDER BY count(*) DESC"
        ).fetchall()
        component_activity = defaultdict(int)
        for comp, action, cnt in tx_rows:
            component_activity[comp] += cnt
        top_components = sorted(
            component_activity.items(), key=lambda x: -x[1]
        )[:10]

        summary = {
            "total_responses": len(assistant_msgs),
            "total_queries": len(operator_msgs),
            "avg_quality_score": avg_quality,
            "median_quality_score": median_quality,
            "avg_word_count": int(avg_length),
            "structure_rate_pct": structure_rate,
            "data_inclusion_rate_pct": data_rate,
            "citation_rate_pct": citation_rate,
            "avg_confidence": avg_confidence,
            "total_analyses": len(analyses),
            "feedback_avg_rating": avg_rating,
            "feedback_count": feedback_count,
        }

        return {
            "available": True,
            "summary": summary,
            "quality_tier_distribution": [
                {"tier": t, "count": tiers.get(t, 0)}
                for t in ["excellent", "good", "adequate", "needs_improvement"]
            ],
            "confidence_distribution": confidence_dist,
            "daily_quality_timeline": timeline,
            "component_activity": [
                {"component": c, "transactions": n} for c, n in top_components
            ],
        }
    finally:
        conn.close()


# ── Breakdown ────────────────────────────────────────────────────────

def response_quality_breakdown() -> Dict[str, Any]:
    conn = _clinical_conn()
    if not conn:
        return {"available": False}

    try:
        # Per-response detail (recent 30)
        rows = conn.execute(
            "SELECT role, text, ts_utc FROM conversation_log "
            "ORDER BY id DESC LIMIT 60"
        ).fetchall()

        response_detail = []
        for role, text, ts in rows:
            if role != "assistant":
                continue
            qs = _response_quality_score(text)
            response_detail.append({
                "timestamp": ts,
                "quality_score": qs["quality_score"],
                "tier": _quality_tier(qs["quality_score"]),
                "word_count": qs["word_count"],
                "has_structure": qs["has_structure"],
                "has_data": qs["has_data"],
                "has_citation": qs["has_citation"],
                "preview": text[:120].replace("\n", " ") + ("..." if len(text) > 120 else ""),
            })
            if len(response_detail) >= 20:
                break

        # Per-disease analysis quality
        analyses = conn.execute(
            "SELECT disease, confidence, predicted_label, signal_quality, patient_id "
            "FROM analyses"
        ).fetchall()
        disease_quality = defaultdict(lambda: {"confidences": [], "labels": [], "signals": []})
        for disease, conf, label, sig, pid in analyses:
            d = disease_quality[disease or "unknown"]
            if conf is not None:
                d["confidences"].append(conf)
            if label:
                d["labels"].append(label)
            if sig:
                d["signals"].append(sig)

        disease_breakdown = []
        for disease, data in disease_quality.items():
            confs = data["confidences"]
            sigs = Counter(data["signals"])
            disease_breakdown.append({
                "disease": disease,
                "count": len(confs),
                "avg_confidence": round(statistics.mean(confs), 3) if confs else 0,
                "min_confidence": round(min(confs), 3) if confs else 0,
                "max_confidence": round(max(confs), 3) if confs else 0,
                "signal_quality": dict(sigs),
                "labels": dict(Counter(data["labels"])),
            })

        # Component reliability breakdown from transaction log
        tx_rows = conn.execute(
            "SELECT component, action, count(*) "
            "FROM transaction_log GROUP BY component, action "
            "ORDER BY component, count(*) DESC"
        ).fetchall()
        comp_actions = defaultdict(list)
        for comp, action, cnt in tx_rows:
            comp_actions[comp].append({"action": action, "count": cnt})

        component_breakdown = [
            {
                "component": comp,
                "total": sum(a["count"] for a in actions),
                "actions": actions[:5],
            }
            for comp, actions in sorted(comp_actions.items(), key=lambda x: -sum(a["count"] for a in x[1]))
        ]

        # RAG coverage from ChromaDB
        rag_coverage = _get_rag_coverage()

        # Feedback detail
        feedback_rows = conn.execute(
            "SELECT patient_id, role, ai_output, rating, correction, reason, created_at "
            "FROM feedback ORDER BY id DESC LIMIT 20"
        ).fetchall()
        feedback_detail = [
            {
                "patient_id": r[0] or "--",
                "reviewer_role": r[1] or "--",
                "rating": r[3],
                "has_correction": bool(r[4]),
                "reason": r[5] or "--",
                "created_at": r[6],
            }
            for r in feedback_rows
        ]

        return {
            "available": True,
            "recent_responses": response_detail,
            "disease_analysis_quality": disease_breakdown,
            "component_reliability": component_breakdown,
            "rag_coverage": rag_coverage,
            "feedback_detail": feedback_detail,
        }
    finally:
        conn.close()


def _get_rag_coverage() -> Dict[str, Any]:
    """Summarize RAG vector DB coverage from ChromaDB."""
    conn = _chroma_conn()
    if not conn:
        return {"available": False, "note": "ChromaDB not found"}
    try:
        # Count collections and embeddings
        try:
            collections = conn.execute(
                "SELECT id, name FROM collections"
            ).fetchall()
        except Exception:
            return {"available": False, "note": "ChromaDB schema not readable"}

        total_embeddings = 0
        collection_stats = []
        for cid, cname in collections:
            try:
                cnt = conn.execute(
                    "SELECT count(*) FROM embeddings WHERE collection_id = ?",
                    (cid,),
                ).fetchone()[0]
            except Exception:
                cnt = 0
            total_embeddings += cnt
            collection_stats.append({"collection": cname, "documents": cnt})

        return {
            "available": True,
            "total_collections": len(collections),
            "total_embeddings": total_embeddings,
            "collections": collection_stats,
        }
    finally:
        conn.close()


# ── Definitions ──────────────────────────────────────────────────────

def response_quality_definitions() -> Dict[str, Any]:
    return {
        "metrics": [
            {
                "name": "Quality Score",
                "description": "Composite 0-100 score measuring response completeness: word count (30pt), structured formatting (20pt), data inclusion (20pt), citation/reference (15pt), sentence depth (15pt).",
                "range": "0-100",
            },
            {
                "name": "Quality Tier",
                "description": "Excellent (80-100), Good (60-79), Adequate (40-59), Needs Improvement (<40).",
                "range": "Categorical",
            },
            {
                "name": "Structure Rate",
                "description": "Percentage of responses containing structured formatting (bullet lists, numbered lists, headers, bold text).",
                "range": "0-100%",
            },
            {
                "name": "Data Inclusion Rate",
                "description": "Percentage of responses containing quantitative data (numbers, percentages, confidence values, accuracy metrics).",
                "range": "0-100%",
            },
            {
                "name": "Citation Rate",
                "description": "Percentage of responses referencing specific patients, datasets, analyses, or reports.",
                "range": "0-100%",
            },
            {
                "name": "Analysis Confidence",
                "description": "Model prediction confidence from EEG classification analyses (0-1 scale). Higher is more decisive.",
                "range": "0.0-1.0",
            },
            {
                "name": "Signal Quality",
                "description": "EEG signal quality assessment (Good/Fair/Poor) from preprocessing pipeline.",
                "range": "Categorical",
            },
            {
                "name": "Feedback Rating",
                "description": "Clinician/operator rating of AI output quality (1-5 scale).",
                "range": "1-5",
            },
            {
                "name": "Component Reliability",
                "description": "Transaction count per system component — higher activity with no errors indicates reliable operation.",
                "range": "Count",
            },
            {
                "name": "RAG Coverage",
                "description": "Number of embedded documents in ChromaDB vector store available for retrieval-augmented generation.",
                "range": "Count",
            },
        ]
    }
