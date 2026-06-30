#!/usr/bin/env python3
"""
Content Freshness Dashboard — real RAG document age & staleness analytics
=========================================================================

Reads REAL data from:
  - data/vector_db/chroma.sqlite3  — embeddings, metadata (patient_id, type,
    chroma:document), embeddings_queue (ingestion pipeline)
  - data/clinical.db               — transaction_log (update activity),
    analyses (clinical analysis timestamps)

Functions:
  - content_freshness_overview    — KPIs, staleness distribution, decay risk,
                                    ingestion timeline, queue stats
  - content_freshness_breakdown   — per-patient freshness, per-type detail,
                                    update activity, refresh recommendations
  - content_freshness_definitions — metric definitions
"""
from __future__ import annotations

import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DB = _PROJECT_ROOT / "data" / "vector_db" / "chroma.sqlite3"
_CLINICAL_DB = _PROJECT_ROOT / "data" / "clinical.db"

# Freshness decay: 168 hours (1 week) = score drops from 100 to 0
_DECAY_WINDOW_HOURS = 168


def _chroma_conn():
    if not _CHROMA_DB.exists():
        return None
    return sqlite3.connect(str(_CHROMA_DB))


def _clinical_conn():
    if not _CLINICAL_DB.exists():
        return None
    return sqlite3.connect(str(_CLINICAL_DB))


def _freshness_score(age_hours: float) -> float:
    """Compute freshness score: 100 = just ingested, 0 = 1 week+ old."""
    return round(max(0.0, 100.0 - (age_hours / _DECAY_WINDOW_HOURS * 100.0)), 1)


def _age_hours(created_at_str: str, now: datetime) -> float:
    """Parse a created_at timestamp and return age in hours from *now*."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"):
        try:
            dt = datetime.strptime(created_at_str.strip(), fmt)
            delta = now - dt
            return max(0.0, delta.total_seconds() / 3600.0)
        except (ValueError, TypeError):
            continue
    return 0.0


def _staleness_bucket(age_hours: float) -> str:
    if age_hours < 24:
        return "< 24h"
    elif age_hours < 72:
        return "1-3 days"
    elif age_hours < 168:
        return "3-7 days"
    else:
        return "> 7 days"


def _decay_risk_level(age_hours: float) -> str:
    if age_hours < 24:
        return "fresh"
    elif age_hours < 72:
        return "aging"
    elif age_hours < 168:
        return "stale"
    else:
        return "critical"


# ─── Overview ────────────────────────────────────────────────────────
def content_freshness_overview() -> Dict[str, Any]:
    """Top-level KPIs: document counts, freshness scores, staleness
    distribution, ingestion timeline, decay risk, and queue stats."""
    conn = _chroma_conn()
    if conn is None:
        return {"available": False, "note": "ChromaDB not found — no freshness data."}

    try:
        cur = conn.cursor()
        now = datetime.utcnow()

        # ── Gather all embeddings with created_at ───────────────────
        cur.execute("SELECT id, created_at FROM embeddings")
        embeddings_raw = cur.fetchall()
        total_documents = len(embeddings_raw)

        # Map embedding id -> age_hours
        id_age: Dict[int, float] = {}
        for eid, created_at in embeddings_raw:
            if created_at:
                id_age[eid] = _age_hours(str(created_at), now)

        all_ages = list(id_age.values()) if id_age else [0.0]

        # ── Document types ──────────────────────────────────────────
        cur.execute(
            "SELECT em.id, em.string_value FROM embedding_metadata em "
            "WHERE em.key = 'type'"
        )
        id_type: Dict[int, str] = {}
        type_set: set = set()
        for eid, sval in cur.fetchall():
            if sval:
                id_type[eid] = sval
                type_set.add(sval)

        total_doc_types = len(type_set)

        # ── Freshness by type ──────────────────────────────────────
        type_ages: Dict[str, List[float]] = defaultdict(list)
        for eid, age in id_age.items():
            doc_type = id_type.get(eid, "unknown")
            type_ages[doc_type].append(age)

        freshness_by_type: List[Dict[str, Any]] = []
        for dtype in sorted(type_ages.keys()):
            ages = type_ages[dtype]
            freshest = min(ages)
            stalest = max(ages)
            avg = statistics.mean(ages)
            freshness_by_type.append({
                "type": dtype,
                "count": len(ages),
                "avg_age_hours": round(avg, 1),
                "freshest": round(freshest, 1),
                "stalest": round(stalest, 1),
                "freshness_score": _freshness_score(avg),
            })

        # ── Staleness distribution ─────────────────────────────────
        bucket_counts: Dict[str, int] = {
            "< 24h": 0, "1-3 days": 0, "3-7 days": 0, "> 7 days": 0,
        }
        for age in all_ages:
            bucket_counts[_staleness_bucket(age)] += 1

        staleness_distribution: List[Dict[str, Any]] = []
        for bucket_name in ("< 24h", "1-3 days", "3-7 days", "> 7 days"):
            cnt = bucket_counts[bucket_name]
            pct = round(cnt / max(1, total_documents) * 100, 1)
            staleness_distribution.append({
                "bucket": bucket_name,
                "count": cnt,
                "percentage": pct,
            })

        # ── Ingestion timeline (documents per day) ─────────────────
        day_counts: Dict[str, int] = defaultdict(int)
        for _eid, created_at in embeddings_raw:
            if created_at:
                day_str = str(created_at)[:10]
                day_counts[day_str] += 1

        ingestion_timeline = [
            {"date": d, "count": c}
            for d, c in sorted(day_counts.items())
        ]

        # ── Decay risk ─────────────────────────────────────────────
        risk_groups: Dict[str, Dict[str, Any]] = {}
        risk_thresholds = {
            "fresh": 24, "aging": 72, "stale": 168, "critical": 9999,
        }
        for eid, age in id_age.items():
            level = _decay_risk_level(age)
            if level not in risk_groups:
                risk_groups[level] = {"count": 0, "doc_types": set()}
            risk_groups[level]["count"] += 1
            risk_groups[level]["doc_types"].add(id_type.get(eid, "unknown"))

        decay_risk: List[Dict[str, Any]] = []
        for level in ("fresh", "aging", "stale", "critical"):
            grp = risk_groups.get(level, {"count": 0, "doc_types": set()})
            decay_risk.append({
                "risk_level": level,
                "count": grp["count"],
                "threshold_hours": risk_thresholds[level],
                "doc_types": sorted(grp["doc_types"]) if isinstance(grp["doc_types"], set) else [],
            })

        # ── Queue stats ────────────────────────────────────────────
        cur.execute("SELECT count(*) FROM embeddings_queue")
        total_queued = cur.fetchone()[0]

        pending_operations = 0
        recent_ingestions = 0
        try:
            cur.execute(
                "SELECT count(*) FROM embeddings_queue WHERE operation = 'ADD'"
            )
            recent_ingestions = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM embeddings_queue WHERE operation != 'ADD'"
            )
            pending_operations = cur.fetchone()[0]
        except Exception:
            pending_operations = total_queued
            recent_ingestions = total_queued

        queue_stats = {
            "total_queued": total_queued,
            "pending_operations": pending_operations,
            "recent_ingestions": recent_ingestions,
        }

        # ── Summary KPIs ───────────────────────────────────────────
        avg_age = statistics.mean(all_ages)
        freshest_doc_age = round(min(all_ages), 1)
        stalest_doc_age = round(max(all_ages), 1)

        summary = {
            "total_documents": total_documents,
            "total_doc_types": total_doc_types,
            "avg_age_hours": round(avg_age, 1),
            "freshest_doc_age": freshest_doc_age,
            "stalest_doc_age": stalest_doc_age,
            "freshness_score": _freshness_score(avg_age),
        }

        return {
            "available": True,
            "generated_at": now.isoformat(),
            "summary": summary,
            "freshness_by_type": freshness_by_type,
            "staleness_distribution": staleness_distribution,
            "ingestion_timeline": ingestion_timeline,
            "decay_risk": decay_risk,
            "queue_stats": queue_stats,
        }
    finally:
        conn.close()


# ─── Breakdown ───────────────────────────────────────────────────────
def content_freshness_breakdown() -> Dict[str, Any]:
    """Per-patient freshness, per-type document details, update activity
    from transaction_log, and refresh recommendations."""
    chroma = _chroma_conn()
    if chroma is None:
        return {"available": False, "note": "ChromaDB not found."}

    try:
        cur = chroma.cursor()
        now = datetime.utcnow()

        # ── Build lookup maps ──────────────────────────────────────
        cur.execute("SELECT id, created_at FROM embeddings")
        id_created: Dict[int, str] = {}
        id_age_map: Dict[int, float] = {}
        for eid, created_at in cur.fetchall():
            ca = str(created_at) if created_at else ""
            id_created[eid] = ca
            id_age_map[eid] = _age_hours(ca, now) if ca else 0.0

        # Metadata maps
        cur.execute(
            "SELECT id, key, string_value FROM embedding_metadata "
            "WHERE key IN ('patient_id', 'type', 'chroma:document')"
        )
        id_patient: Dict[int, str] = {}
        id_type: Dict[int, str] = {}
        id_doc: Dict[int, str] = {}
        for eid, key, sval in cur.fetchall():
            if not sval:
                continue
            if key == "patient_id":
                id_patient[eid] = sval
            elif key == "type":
                id_type[eid] = sval
            elif key == "chroma:document":
                id_doc[eid] = sval

        # Embedding IDs (for display)
        cur.execute("SELECT id, embedding_id FROM embeddings")
        id_emb_id: Dict[int, str] = {r[0]: r[1] for r in cur.fetchall() if r[1]}

        # ── Per-patient freshness ──────────────────────────────────
        patient_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"ages": [], "types": set(), "last_updated": ""}
        )
        for eid, age in id_age_map.items():
            pid = id_patient.get(eid, "unknown")
            pd = patient_data[pid]
            pd["ages"].append(age)
            pd["types"].add(id_type.get(eid, "unknown"))
            ca = id_created.get(eid, "")
            if ca > pd["last_updated"]:
                pd["last_updated"] = ca

        per_patient_freshness: List[Dict[str, Any]] = []
        for pid in sorted(patient_data.keys()):
            pd = patient_data[pid]
            ages = pd["ages"]
            avg = statistics.mean(ages) if ages else 0.0
            per_patient_freshness.append({
                "patient_id": pid,
                "doc_count": len(ages),
                "doc_types": sorted(pd["types"]),
                "avg_age_hours": round(avg, 1),
                "freshness_score": _freshness_score(avg),
                "last_updated": pd["last_updated"],
            })

        # ── Per-type detail ────────────────────────────────────────
        type_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for eid in id_age_map:
            dtype = id_type.get(eid, "unknown")
            doc_text = id_doc.get(eid, "")
            preview = doc_text[:60] if doc_text else ""
            type_docs[dtype].append({
                "embedding_id": id_emb_id.get(eid, str(eid)),
                "patient_id": id_patient.get(eid, "unknown"),
                "age_hours": round(id_age_map[eid], 1),
                "content_preview": preview,
                "created_at": id_created.get(eid, ""),
            })

        per_type_detail: List[Dict[str, Any]] = []
        for dtype in sorted(type_docs.keys()):
            docs = sorted(type_docs[dtype], key=lambda d: d["age_hours"])
            per_type_detail.append({
                "type": dtype,
                "documents": docs,
            })

    finally:
        chroma.close()

    # ── Update activity from transaction_log ───────────────────────
    update_activity: List[Dict[str, Any]] = []
    clinical = _clinical_conn()
    if clinical is not None:
        try:
            ccur = clinical.cursor()
            ccur.execute(
                "SELECT date(ts_utc) AS day, count(*) AS txns, "
                "       group_concat(DISTINCT component) AS components "
                "FROM transaction_log "
                "GROUP BY date(ts_utc) ORDER BY day"
            )
            for day, txns, components in ccur.fetchall():
                update_activity.append({
                    "date": day,
                    "transactions": txns,
                    "components": components.split(",") if components else [],
                })
        except Exception:
            pass
        finally:
            clinical.close()

    # ── Refresh recommendations ────────────────────────────────────
    refresh_recommendations: List[Dict[str, Any]] = []
    for entry in per_type_detail:
        dtype = entry["type"]
        docs = entry["documents"]
        if not docs:
            continue
        ages = [d["age_hours"] for d in docs]
        max_age = max(ages)
        avg_age = statistics.mean(ages)

        if max_age > 168:
            refresh_recommendations.append({
                "type": dtype,
                "reason": f"Oldest document is {round(max_age, 1)}h old (> 1 week)",
                "priority": "high",
                "affected_count": sum(1 for a in ages if a > 168),
            })
        elif max_age > 72:
            refresh_recommendations.append({
                "type": dtype,
                "reason": f"Documents aging — oldest is {round(max_age, 1)}h old",
                "priority": "medium",
                "affected_count": sum(1 for a in ages if a > 72),
            })
        elif avg_age > 48:
            refresh_recommendations.append({
                "type": dtype,
                "reason": f"Average age {round(avg_age, 1)}h approaching staleness",
                "priority": "low",
                "affected_count": len(ages),
            })

    return {
        "available": True,
        "generated_at": datetime.utcnow().isoformat(),
        "per_patient_freshness": per_patient_freshness,
        "per_type_detail": per_type_detail,
        "update_activity": update_activity,
        "refresh_recommendations": refresh_recommendations,
    }


# ─── Definitions ─────────────────────────────────────────────────────
def content_freshness_definitions() -> Dict[str, Any]:
    """Metric definitions for the Content Freshness Dashboard."""
    return {
        "available": True,
        "metrics": [
            {
                "name": "Total Documents",
                "description": "Number of embedded document chunks stored in ChromaDB.",
                "unit": "count",
            },
            {
                "name": "Total Doc Types",
                "description": "Number of distinct document categories (patient, analysis, medications, mri_findings, neuropsych, hitl_reviews, survey).",
                "unit": "count",
            },
            {
                "name": "Avg Age (hours)",
                "description": "Mean age of all documents in hours, measured from their created_at timestamp to now.",
                "unit": "hours",
            },
            {
                "name": "Freshest Doc Age",
                "description": "Age of the most recently ingested document.",
                "unit": "hours",
            },
            {
                "name": "Stalest Doc Age",
                "description": "Age of the oldest document in the store.",
                "unit": "hours",
            },
            {
                "name": "Freshness Score",
                "description": "0-100 score computed as max(0, 100 - (age_hours / 168 * 100)). 100 = just ingested, 0 = 1 week or older. Applied per-document and averaged for aggregates.",
                "unit": "score (0-100)",
            },
            {
                "name": "Staleness Distribution",
                "description": "Histogram bucketing documents into age ranges: < 24h, 1-3 days, 3-7 days, > 7 days.",
                "unit": "count per bucket",
            },
            {
                "name": "Ingestion Timeline",
                "description": "Daily count of documents ingested, derived from embeddings.created_at dates.",
                "unit": "count per day",
            },
            {
                "name": "Decay Risk",
                "description": "Documents grouped by risk level: fresh (< 24h), aging (24-72h), stale (72-168h), critical (> 168h). Shows which doc types are at each risk tier.",
                "unit": "risk tier",
            },
            {
                "name": "Queue Stats",
                "description": "Embeddings queue pipeline metrics: total queued operations, pending non-ADD operations, and recent ingestion ADDs from embeddings_queue.",
                "unit": "count",
            },
            {
                "name": "Per-Patient Freshness",
                "description": "For each patient: document count, doc types present, average age, freshness score, and last-updated timestamp.",
                "unit": "composite",
            },
            {
                "name": "Per-Type Detail",
                "description": "For each document type: list of individual documents with embedding ID, patient ID, age, content preview (first 60 characters), and created_at.",
                "unit": "composite",
            },
            {
                "name": "Update Activity",
                "description": "Daily transaction counts from clinical.db transaction_log, showing which components had activity each day.",
                "unit": "transactions per day",
            },
            {
                "name": "Refresh Recommendations",
                "description": "Actionable suggestions for document types that need re-ingestion, prioritized as high (> 1 week old), medium (> 3 days), or low (average age approaching staleness).",
                "unit": "recommendation",
            },
        ],
        "clinical_relevance": (
            "Document freshness directly impacts clinical decision support quality. "
            "Stale EEG analyses, outdated medication lists, or old neuropsych reports "
            "can lead to recommendations based on superseded data. Monitoring freshness "
            "ensures the RAG pipeline serves current, clinically accurate information "
            "and flags documents that need re-ingestion after patient updates."
        ),
        "data_sources": {
            "chroma_db": "data/vector_db/chroma.sqlite3 — embeddings (75 rows), embedding_metadata (patient_id, type, chroma:document), embeddings_queue (709 rows)",
            "clinical_db": "data/clinical.db — transaction_log (539 rows) for update activity tracking",
        },
    }
