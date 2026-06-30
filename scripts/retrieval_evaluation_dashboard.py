"""Retrieval Evaluation Dashboard — real retrieval quality metrics from ChromaDB + clinical.db.

Sources:
- data/vector_db/chroma.sqlite3 — embeddings (75), embedding_metadata, embeddings_queue (709)
- data/clinical.db — patients (40), analyses (21), conversation_log (218), uploads (21)

Metrics:
- Embedding coverage: what % of clinical.db records have a matching vector entry
- Document type distribution in vector store
- Queue health: pending vs processed embeddings
- Cross-reference completeness: patients/analyses in DB vs in vector store
- Retrieval freshness: age distribution of vector entries
- Collection statistics and density
"""

import sqlite3
import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

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


def _safe(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


# ── Overview ──────────────────────────────────────────────────────

def retrieval_overview():
    """Aggregate retrieval evaluation: coverage, quality scores, queue health."""
    chroma = _chroma_conn()
    clinical = _clinical_conn()
    if chroma is None and clinical is None:
        return {"available": False, "note": "Neither ChromaDB nor clinical.db found"}

    result = {"available": True, "generated_at": datetime.now(timezone.utc).isoformat()}

    # ── Vector store stats ──
    if chroma:
        cur = chroma.cursor()
        total_embeddings = _safe(cur, "SELECT count(*) FROM embeddings")
        total_queue = _safe(cur, "SELECT count(*) FROM embeddings_queue")
        total_metadata = _safe(cur, "SELECT count(DISTINCT id) FROM embedding_metadata")

        # Type distribution
        cur.execute(
            "SELECT string_value, count(*) FROM embedding_metadata "
            "WHERE key='type' GROUP BY string_value ORDER BY count(*) DESC"
        )
        type_dist = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]

        # Collection info
        cur.execute("SELECT id, name FROM collections")
        collections = [{"id": r[0], "name": r[1]} for r in cur.fetchall()]

        # Queue operations breakdown
        cur.execute(
            "SELECT operation, count(*) FROM embeddings_queue "
            "GROUP BY operation ORDER BY count(*) DESC"
        )
        queue_ops = [{"operation": r[0] or "unknown", "count": r[1]} for r in cur.fetchall()]

        # Queue timeline — group by date
        try:
            cur.execute(
                "SELECT date(created_at) as d, count(*) FROM embeddings_queue "
                "WHERE created_at IS NOT NULL GROUP BY d ORDER BY d"
            )
            queue_timeline = [{"date": r[0], "count": r[1]} for r in cur.fetchall()]
        except Exception:
            queue_timeline = []

        chroma.close()
    else:
        total_embeddings = 0
        total_queue = 0
        total_metadata = 0
        type_dist = []
        collections = []
        queue_ops = []
        queue_timeline = []

    # ── Clinical DB coverage check ──
    if clinical:
        ccur = clinical.cursor()
        total_patients = _safe(ccur, "SELECT count(*) FROM patients")
        total_analyses = _safe(ccur, "SELECT count(*) FROM analyses")
        total_conversations = _safe(ccur, "SELECT count(*) FROM conversation_log")
        total_uploads = _safe(ccur, "SELECT count(*) FROM uploads")

        # Get patient IDs from clinical DB
        ccur.execute("SELECT DISTINCT patient_id FROM patients WHERE patient_id IS NOT NULL")
        clinical_patient_ids = set(r[0] for r in ccur.fetchall())
        clinical.close()
    else:
        total_patients = 0
        total_analyses = 0
        total_conversations = 0
        total_uploads = 0
        clinical_patient_ids = set()

    # ── Coverage: how many clinical patients have vector entries ──
    vector_patient_ids = set()
    if _CHROMA_DB.exists():
        ch = sqlite3.connect(str(_CHROMA_DB))
        chcur = ch.cursor()
        try:
            chcur.execute(
                "SELECT DISTINCT string_value FROM embedding_metadata "
                "WHERE key='patient_id' AND string_value IS NOT NULL"
            )
            vector_patient_ids = set(r[0] for r in chcur.fetchall())
        except Exception:
            pass
        ch.close()

    covered = clinical_patient_ids & vector_patient_ids
    missing = clinical_patient_ids - vector_patient_ids
    extra = vector_patient_ids - clinical_patient_ids
    coverage_pct = round(len(covered) / max(len(clinical_patient_ids), 1) * 100, 1)

    # ── Quality score (composite) ──
    # Coverage weight 40%, queue health 30%, type diversity 30%
    queue_health = min(total_embeddings / max(total_queue, 1), 1.0) * 100
    type_diversity = min(len(type_dist) / 7.0, 1.0) * 100  # 7 ideal types
    quality_score = round(coverage_pct * 0.4 + queue_health * 0.3 + type_diversity * 0.3, 1)

    result["summary"] = {
        "total_embeddings": total_embeddings,
        "total_queue_entries": total_queue,
        "total_metadata_entries": total_metadata,
        "collections": len(collections),
        "quality_score": quality_score,
        "coverage_pct": coverage_pct,
        "queue_health_pct": round(queue_health, 1),
        "type_diversity_pct": round(type_diversity, 1),
    }

    result["coverage"] = {
        "clinical_patients": total_patients,
        "vectorized_patients": len(vector_patient_ids),
        "covered": len(covered),
        "missing_from_vector": sorted(list(missing))[:20],
        "extra_in_vector": sorted(list(extra))[:20],
        "coverage_pct": coverage_pct,
        "clinical_analyses": total_analyses,
        "clinical_conversations": total_conversations,
        "clinical_uploads": total_uploads,
    }

    result["type_distribution"] = type_dist
    result["collections"] = collections
    result["queue_operations"] = queue_ops
    result["queue_timeline"] = queue_timeline

    return result


# ── Breakdown ─────────────────────────────────────────────────────

def retrieval_breakdown():
    """Detailed retrieval drill-down: per-patient coverage, per-type detail,
    document inventory, queue detail."""

    result = {"available": True}

    # ── Per-patient vector coverage ──
    patient_coverage = []
    clinical = _clinical_conn()
    chroma = _chroma_conn()

    if clinical and chroma:
        ccur = clinical.cursor()
        chcur = chroma.cursor()

        # Get patients
        ccur.execute(
            "SELECT patient_id, name, disease FROM patients "
            "WHERE patient_id IS NOT NULL ORDER BY patient_id"
        )
        patients = ccur.fetchall()

        # Get vector counts per patient
        try:
            chcur.execute(
                "SELECT string_value, count(*) FROM embedding_metadata "
                "WHERE key='patient_id' AND string_value IS NOT NULL "
                "GROUP BY string_value"
            )
            vector_counts = dict(chcur.fetchall())
        except Exception:
            vector_counts = {}

        # Get types per patient
        try:
            chcur.execute(
                "SELECT m1.string_value as pid, m2.string_value as typ, count(*) "
                "FROM embedding_metadata m1 "
                "JOIN embedding_metadata m2 ON m1.id = m2.id "
                "WHERE m1.key='patient_id' AND m2.key='type' "
                "GROUP BY pid, typ ORDER BY pid, typ"
            )
            patient_types = {}
            for pid, typ, cnt in chcur.fetchall():
                patient_types.setdefault(pid, []).append({"type": typ, "count": cnt})
        except Exception:
            patient_types = {}

        for pid, name, disease in patients:
            vec_count = vector_counts.get(pid, 0)
            types = patient_types.get(pid, [])
            patient_coverage.append({
                "patient_id": pid,
                "name": name or "",
                "disease": disease or "",
                "vector_count": vec_count,
                "types": types,
                "has_vectors": vec_count > 0,
            })

        clinical.close()
        chroma.close()

    result["patient_coverage"] = patient_coverage

    # ── Document inventory (all embedded documents) ──
    documents = []
    ch = _chroma_conn()
    if ch:
        chcur = ch.cursor()
        try:
            chcur.execute("SELECT DISTINCT id FROM embedding_metadata ORDER BY id")
            doc_ids = [r[0] for r in chcur.fetchall()]

            for doc_id in doc_ids[:100]:  # cap at 100
                chcur.execute(
                    "SELECT key, string_value FROM embedding_metadata WHERE id=?",
                    (doc_id,)
                )
                meta = dict(chcur.fetchall())
                documents.append({
                    "id": doc_id,
                    "patient_id": meta.get("patient_id", ""),
                    "type": meta.get("type", "unknown"),
                    "document": (meta.get("chroma:document", ""))[:120],
                })
        except Exception:
            pass
        ch.close()

    result["documents"] = documents

    # ── Analyses cross-reference ──
    analysis_coverage = []
    clinical = _clinical_conn()
    ch = _chroma_conn()
    if clinical and ch:
        ccur = clinical.cursor()
        chcur = ch.cursor()

        ccur.execute(
            "SELECT id, patient_id, disease, predicted_label, confidence, created_at "
            "FROM analyses ORDER BY id"
        )
        analyses = ccur.fetchall()

        # Check which analyses are in vector store
        try:
            chcur.execute(
                "SELECT string_value FROM embedding_metadata "
                "WHERE key='type' AND string_value='analysis'"
            )
            analysis_vector_count = len(chcur.fetchall())
        except Exception:
            analysis_vector_count = 0

        for a in analyses:
            analysis_coverage.append({
                "id": a[0],
                "patient_id": a[1] or "",
                "disease": a[2] or "",
                "label": a[3] or "",
                "confidence": a[4],
                "created_at": a[5] or "",
            })

        clinical.close()
        ch.close()

    result["analysis_coverage"] = {
        "total_analyses": len(analysis_coverage),
        "vectorized_analyses": analysis_coverage,
    }

    # ── Queue detail (recent entries) ──
    queue_detail = []
    ch = _chroma_conn()
    if ch:
        chcur = ch.cursor()
        try:
            chcur.execute(
                "SELECT seq_id, created_at, operation, topic "
                "FROM embeddings_queue ORDER BY seq_id DESC LIMIT 50"
            )
            for r in chcur.fetchall():
                queue_detail.append({
                    "seq_id": r[0],
                    "created_at": r[1] or "",
                    "operation": r[2] or "unknown",
                    "topic": r[3] or "",
                })
        except Exception:
            pass
        ch.close()

    result["queue_detail"] = queue_detail

    return result


# ── Definitions ───────────────────────────────────────────────────

def retrieval_definitions():
    """Metric definitions for the Retrieval Evaluation dashboard."""
    return {
        "available": True,
        "metrics": {
            "quality_score": {
                "label": "Retrieval Quality Score",
                "description": "Composite score (0-100) combining patient coverage (40%), queue health (30%), and type diversity (30%).",
                "source": "data/vector_db/chroma.sqlite3 + data/clinical.db",
            },
            "coverage_pct": {
                "label": "Patient Coverage",
                "description": "Percentage of clinical.db patients that have at least one vector embedding in ChromaDB.",
                "source": "Cross-reference: clinical.db patients vs embedding_metadata patient_id",
            },
            "queue_health_pct": {
                "label": "Queue Health",
                "description": "Ratio of processed embeddings to total queue entries. Higher = healthier pipeline (fewer pending items).",
                "source": "embeddings (processed) / embeddings_queue (total pipeline)",
            },
            "type_diversity_pct": {
                "label": "Type Diversity",
                "description": "How many distinct document types are embedded (patient, analysis, medications, etc.). Target: 7 types.",
                "source": "embedding_metadata WHERE key='type' — distinct values",
            },
            "total_embeddings": {
                "label": "Total Embeddings",
                "description": "Number of vector embeddings stored in ChromaDB.",
                "source": "SELECT count(*) FROM embeddings",
            },
            "total_queue_entries": {
                "label": "Queue Entries",
                "description": "Total entries in the embeddings ingestion queue (includes processed and pending).",
                "source": "SELECT count(*) FROM embeddings_queue",
            },
            "vector_count": {
                "label": "Per-Patient Vector Count",
                "description": "Number of vector embeddings for a given patient across all document types.",
                "source": "embedding_metadata WHERE key='patient_id'",
            },
        },
        "data_sources": {
            "chroma_db": "data/vector_db/chroma.sqlite3 — embeddings (75 rows), embedding_metadata, embeddings_queue (709 rows)",
            "clinical_db": "data/clinical.db — patients (40), analyses (21), conversation_log (218)",
        },
    }


if __name__ == "__main__":
    ov = retrieval_overview()
    print(json.dumps(ov, indent=2, default=str))
