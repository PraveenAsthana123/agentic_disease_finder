"""RAG Metadata Filter Dashboard — metadata-driven retrieval analytics.

Sources:
- data/vector_db/chroma.sqlite3 — embeddings, embedding_metadata, collections, embeddings_queue
- data/clinical.db — patients (for cross-ref), transaction_log (for query history)

Metrics:
- Metadata key inventory: which keys exist, value distributions
- Per-type embedding counts and coverage
- Per-patient vector coverage with metadata richness
- Metadata filter applicability (how many queries could benefit from filters)
- Cross-tab: type × patient distribution
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

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

def overview():
    """Aggregate metadata filter metrics: key inventory, type distribution,
    patient coverage, filter readiness."""
    chroma = _chroma_conn()
    clinical = _clinical_conn()
    if chroma is None:
        return {"available": False, "note": "ChromaDB not found"}

    result = {"available": True, "generated_at": datetime.now(timezone.utc).isoformat()}
    chcur = chroma.cursor()

    # ── Total embeddings ──
    total_embeddings = _safe(chcur, "SELECT count(*) FROM embeddings")

    # ── Metadata key inventory ──
    chcur.execute(
        "SELECT key, count(*) as cnt, count(DISTINCT string_value) as uniq "
        "FROM embedding_metadata WHERE key != 'chroma:document' "
        "GROUP BY key ORDER BY cnt DESC"
    )
    metadata_keys = [
        {"key": r[0], "total_entries": r[1], "distinct_values": r[2]}
        for r in chcur.fetchall()
    ]

    # ── Type distribution ──
    chcur.execute(
        "SELECT string_value, count(*) FROM embedding_metadata "
        "WHERE key='type' GROUP BY string_value ORDER BY count(*) DESC"
    )
    type_distribution = [
        {"type": r[0] or "unknown", "count": r[1]}
        for r in chcur.fetchall()
    ]

    # ── Patient coverage ──
    chcur.execute(
        "SELECT count(DISTINCT string_value) FROM embedding_metadata "
        "WHERE key='patient_id' AND string_value IS NOT NULL"
    )
    patients_with_vectors = chcur.fetchone()[0] or 0

    total_patients = 0
    if clinical:
        total_patients = _safe(clinical.cursor(), "SELECT count(*) FROM patients")

    coverage_pct = round(patients_with_vectors / max(total_patients, 1) * 100, 1)

    # ── Per-patient embedding counts ──
    chcur.execute(
        "SELECT string_value, count(*) FROM embedding_metadata "
        "WHERE key='patient_id' AND string_value IS NOT NULL "
        "GROUP BY string_value ORDER BY count(*) DESC"
    )
    patient_embedding_counts = [
        {"patient_id": r[0], "embedding_count": r[1]}
        for r in chcur.fetchall()
    ]

    # ── Metadata richness: how many embeddings have BOTH patient_id AND type ──
    chcur.execute(
        "SELECT count(DISTINCT em1.id) FROM embedding_metadata em1 "
        "INNER JOIN embedding_metadata em2 ON em1.id = em2.id "
        "WHERE em1.key='patient_id' AND em1.string_value IS NOT NULL "
        "AND em2.key='type' AND em2.string_value IS NOT NULL"
    )
    dual_metadata_count = chcur.fetchone()[0] or 0
    filter_readiness_pct = round(dual_metadata_count / max(total_embeddings, 1) * 100, 1)

    # ── Collections ──
    chcur.execute("SELECT id, name, dimension FROM collections")
    collections = [
        {"id": r[0], "name": r[1], "dimension": r[2]}
        for r in chcur.fetchall()
    ]

    # ── Queue status ──
    queue_pending = _safe(chcur, "SELECT count(*) FROM embeddings_queue")

    result["summary"] = {
        "total_embeddings": total_embeddings,
        "metadata_keys_count": len(metadata_keys),
        "document_types": len(type_distribution),
        "patients_with_vectors": patients_with_vectors,
        "total_patients": total_patients,
        "coverage_pct": coverage_pct,
        "filter_readiness_pct": filter_readiness_pct,
        "dual_metadata_embeddings": dual_metadata_count,
        "queue_pending": queue_pending,
    }

    result["metadata_keys"] = metadata_keys
    result["type_distribution"] = type_distribution
    result["patient_embedding_counts"] = patient_embedding_counts
    result["collections"] = collections

    chroma.close()
    if clinical:
        clinical.close()

    return result


# ── Breakdown ─────────────────────────────────────────────────────

def breakdown():
    """Detailed metadata filter drill-down: cross-tab type×patient,
    per-patient type matrix, recent embeddings, filterable query analysis."""
    chroma = _chroma_conn()
    clinical = _clinical_conn()
    if chroma is None:
        return {"available": False, "note": "ChromaDB not found"}

    result = {"available": True}
    chcur = chroma.cursor()

    # ── Cross-tab: type × patient ──
    chcur.execute(
        "SELECT em_type.string_value as doc_type, "
        "em_pid.string_value as patient_id, count(*) as cnt "
        "FROM embedding_metadata em_type "
        "INNER JOIN embedding_metadata em_pid ON em_type.id = em_pid.id "
        "WHERE em_type.key='type' AND em_pid.key='patient_id' "
        "AND em_type.string_value IS NOT NULL AND em_pid.string_value IS NOT NULL "
        "GROUP BY doc_type, patient_id ORDER BY cnt DESC"
    )
    cross_tab_raw = chcur.fetchall()

    # Build per-type patient count summary
    type_patient_summary = {}
    for doc_type, patient_id, cnt in cross_tab_raw:
        if doc_type not in type_patient_summary:
            type_patient_summary[doc_type] = {"type": doc_type, "patients": 0, "total_embeddings": 0}
        type_patient_summary[doc_type]["patients"] += 1
        type_patient_summary[doc_type]["total_embeddings"] += cnt

    type_patient_data = sorted(type_patient_summary.values(), key=lambda x: -x["total_embeddings"])

    # ── Per-patient type matrix (top 20 patients) ──
    patient_types = {}
    for doc_type, patient_id, cnt in cross_tab_raw:
        if patient_id not in patient_types:
            patient_types[patient_id] = {"patient_id": patient_id, "types": {}, "total": 0}
        patient_types[patient_id]["types"][doc_type] = cnt
        patient_types[patient_id]["total"] += cnt

    patient_type_matrix = sorted(patient_types.values(), key=lambda x: -x["total"])[:20]

    # Enrich with patient name/disease from clinical.db
    if clinical:
        ccur = clinical.cursor()
        for pm in patient_type_matrix:
            try:
                ccur.execute(
                    "SELECT name, disease FROM patients WHERE patient_id=?",
                    (pm["patient_id"],)
                )
                row = ccur.fetchone()
                if row:
                    pm["patient_name"] = row[0] or ""
                    pm["disease"] = row[1] or ""
            except Exception:
                pass

    # ── Recent embeddings with metadata ──
    chcur.execute(
        "SELECT e.id, e.embedding_id, e.created_at FROM embeddings e "
        "ORDER BY e.created_at DESC LIMIT 30"
    )
    recent_embeddings = []
    for eid, emb_id, created_at in chcur.fetchall():
        chcur.execute(
            "SELECT key, string_value FROM embedding_metadata WHERE id=? AND key != 'chroma:document'",
            (eid,)
        )
        meta = {r[0]: r[1] for r in chcur.fetchall()}
        recent_embeddings.append({
            "id": eid,
            "embedding_id": emb_id or "",
            "created_at": created_at or "",
            "patient_id": meta.get("patient_id", ""),
            "type": meta.get("type", ""),
        })

    # ── Metadata completeness per embedding ──
    chcur.execute("SELECT id FROM embeddings")
    all_ids = [r[0] for r in chcur.fetchall()]
    has_patient = set()
    has_type = set()
    chcur.execute(
        "SELECT DISTINCT id FROM embedding_metadata WHERE key='patient_id' AND string_value IS NOT NULL"
    )
    has_patient = {r[0] for r in chcur.fetchall()}
    chcur.execute(
        "SELECT DISTINCT id FROM embedding_metadata WHERE key='type' AND string_value IS NOT NULL"
    )
    has_type = {r[0] for r in chcur.fetchall()}

    total = len(all_ids)
    completeness = {
        "total_embeddings": total,
        "has_patient_id": len(has_patient),
        "has_type": len(has_type),
        "has_both": len(has_patient & has_type),
        "has_neither": len(set(all_ids) - has_patient - has_type),
        "pct_patient_id": round(len(has_patient) / max(total, 1) * 100, 1),
        "pct_type": round(len(has_type) / max(total, 1) * 100, 1),
        "pct_both": round(len(has_patient & has_type) / max(total, 1) * 100, 1),
    }

    # ── Filterable query analysis (from transaction_log) ──
    filterable_queries = []
    if clinical:
        ccur = clinical.cursor()
        try:
            ccur.execute(
                "SELECT t.id, t.patient_id, t.detail, t.ts_utc "
                "FROM transaction_log t "
                "WHERE t.component='patient_chat' "
                "ORDER BY t.ts_utc DESC LIMIT 30"
            )
            for r in ccur.fetchall():
                query_text = r[2] or ""
                # Detect if query mentions a type keyword
                type_keywords = ["analysis", "medication", "mri", "neuropsych", "survey", "review"]
                detected_types = [kw for kw in type_keywords if kw in query_text.lower()]
                filterable_queries.append({
                    "id": r[0],
                    "patient_id": r[1] or "",
                    "query_text": query_text[:200],
                    "timestamp": r[3] or "",
                    "detected_type_filters": detected_types,
                    "has_patient_filter": bool(r[1]),
                    "filter_applicable": bool(r[1]) or bool(detected_types),
                })
        except Exception:
            pass

    filter_applicable_count = sum(1 for q in filterable_queries if q["filter_applicable"])

    result["type_patient_summary"] = type_patient_data
    result["patient_type_matrix"] = patient_type_matrix
    result["recent_embeddings"] = recent_embeddings
    result["metadata_completeness"] = completeness
    result["filterable_queries"] = filterable_queries
    result["filter_applicability"] = {
        "total_queries_sampled": len(filterable_queries),
        "filter_applicable": filter_applicable_count,
        "pct_applicable": round(filter_applicable_count / max(len(filterable_queries), 1) * 100, 1),
    }

    chroma.close()
    if clinical:
        clinical.close()

    return result


# ── Definitions ───────────────────────────────────────────────────

def definitions():
    """Metric definitions and glossary for the RAG Metadata Filter dashboard."""
    return {
        "metrics": [
            {
                "name": "Total Embeddings",
                "description": "Number of vector embeddings stored in ChromaDB.",
                "source": "embeddings table",
            },
            {
                "name": "Metadata Keys",
                "description": "Distinct metadata keys attached to embeddings (excluding chroma:document). Used as filter dimensions.",
                "source": "embedding_metadata table",
            },
            {
                "name": "Document Types",
                "description": "Distinct values of the 'type' metadata key (analysis, medications, mri_findings, neuropsych, patient, survey, hitl_reviews).",
                "source": "embedding_metadata WHERE key='type'",
            },
            {
                "name": "Patient Coverage",
                "description": "Percentage of clinical patients that have at least one vector embedding with patient_id metadata.",
                "source": "embedding_metadata WHERE key='patient_id' vs patients table",
            },
            {
                "name": "Filter Readiness",
                "description": "Percentage of embeddings that have BOTH patient_id AND type metadata, making them fully filterable by both dimensions.",
                "source": "embedding_metadata cross-join on id",
            },
            {
                "name": "Filter Applicability",
                "description": "Percentage of recent queries that could benefit from metadata filtering (have patient context or mention a document type).",
                "source": "transaction_log WHERE component='patient_chat'",
            },
        ],
        "filter_dimensions": [
            {
                "key": "patient_id",
                "description": "Filter retrieval results to a specific patient. Narrows context window and improves relevance.",
                "example": "Retrieve only embeddings for patient EPAT001",
            },
            {
                "key": "type",
                "description": "Filter by document type (analysis, medications, mri_findings, neuropsych, patient, survey, hitl_reviews).",
                "example": "Retrieve only medication-related embeddings for drug interaction queries",
            },
            {
                "key": "combined",
                "description": "Apply both patient_id AND type filters simultaneously for maximum precision.",
                "example": "Retrieve MRI findings for patient EPAT003 only",
            },
        ],
        "glossary": {
            "Embedding": "A vector representation of a text chunk stored in ChromaDB for similarity search.",
            "Metadata Filter": "A pre-retrieval constraint that limits which embeddings are searched, improving relevance and speed.",
            "Hybrid Retrieval": "Combining vector similarity with keyword search and metadata filters for optimal recall.",
            "Filter Readiness": "The proportion of embeddings that carry enough metadata to support filtered retrieval.",
            "Coverage": "The proportion of clinical patients that have at least one vectorized document.",
            "Cross-tab": "A matrix showing the distribution of embeddings across two metadata dimensions (type x patient).",
        },
    }
