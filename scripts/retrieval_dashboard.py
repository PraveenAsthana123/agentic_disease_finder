"""Retrieval Dashboard — patient-chat retrieval operations analytics.

Sources:
- data/clinical.db — transaction_log (component='patient_chat'), conversation_log, patients, analyses
- data/vector_db/chroma.sqlite3 — embeddings, embedding_metadata, embeddings_queue, collections

Metrics:
- Query volume and rate from patient_chat transactions
- Per-patient retrieval activity and vector coverage
- Vector store size, queue health, collection stats
- Query text analysis and embedding timeline
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DB = _PROJECT_ROOT / "data" / "vector_db" / "chroma.sqlite3"
_CLINICAL_DB = _PROJECT_ROOT / "data" / "clinical.db"

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "about", "above", "below", "between",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "if", "then", "else", "up", "out", "off",
}


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
    """Aggregate retrieval metrics: query volume, vector store health, coverage."""
    clinical = _clinical_conn()
    chroma = _chroma_conn()
    if clinical is None and chroma is None:
        return {"available": False, "note": "Neither clinical.db nor ChromaDB found"}

    result = {"available": True, "generated_at": datetime.now(timezone.utc).isoformat()}

    # ── Transaction log query stats ──
    total_queries = 0
    unique_patients_queried = 0
    query_rate_per_day = 0.0
    query_volume_daily = []
    query_by_patient = []

    if clinical:
        ccur = clinical.cursor()
        total_queries = _safe(
            ccur,
            "SELECT count(*) FROM transaction_log WHERE component='patient_chat'"
        )
        unique_patients_queried = _safe(
            ccur,
            "SELECT count(DISTINCT patient_id) FROM transaction_log "
            "WHERE component='patient_chat' AND patient_id IS NOT NULL"
        )

        # Query rate per day
        try:
            ccur.execute(
                "SELECT min(date(ts_utc)), max(date(ts_utc)) "
                "FROM transaction_log WHERE component='patient_chat'"
            )
            row = ccur.fetchone()
            if row and row[0] and row[1]:
                d0 = datetime.strptime(row[0], "%Y-%m-%d")
                d1 = datetime.strptime(row[1], "%Y-%m-%d")
                span = max((d1 - d0).days, 1)
                query_rate_per_day = round(total_queries / span, 2)
        except Exception:
            pass

        # Daily volume
        try:
            ccur.execute(
                "SELECT date(ts_utc) as d, count(*) FROM transaction_log "
                "WHERE component='patient_chat' GROUP BY d ORDER BY d"
            )
            query_volume_daily = [{"date": r[0], "count": r[1]} for r in ccur.fetchall()]
        except Exception:
            pass

        # Queries by patient
        try:
            ccur.execute(
                "SELECT t.patient_id, p.name, p.disease, count(*) as cnt "
                "FROM transaction_log t "
                "LEFT JOIN patients p ON t.patient_id = p.patient_id "
                "WHERE t.component='patient_chat' AND t.patient_id IS NOT NULL "
                "GROUP BY t.patient_id ORDER BY cnt DESC"
            )
            query_by_patient = [
                {"patient_id": r[0], "patient_name": r[1] or "", "disease": r[2] or "", "query_count": r[3]}
                for r in ccur.fetchall()
            ]
        except Exception:
            pass

    # ── Vector store stats ──
    vector_store_size = 0
    queue_size = 0
    total_metadata = 0
    collections_count = 0
    queue_ops = []

    if chroma:
        chcur = chroma.cursor()
        vector_store_size = _safe(chcur, "SELECT count(*) FROM embeddings")
        queue_size = _safe(chcur, "SELECT count(*) FROM embeddings_queue")
        total_metadata = _safe(chcur, "SELECT count(DISTINCT id) FROM embedding_metadata")
        collections_count = _safe(chcur, "SELECT count(*) FROM collections")

        try:
            chcur.execute(
                "SELECT operation, count(*) FROM embeddings_queue "
                "GROUP BY operation ORDER BY count(*) DESC"
            )
            queue_ops = [{"operation": r[0] or "unknown", "count": r[1]} for r in chcur.fetchall()]
        except Exception:
            pass

        chroma.close()

    # ── Coverage: patients with at least 1 vector entry ──
    total_patients = 0
    patients_with_vectors = 0

    if clinical:
        total_patients = _safe(ccur, "SELECT count(*) FROM patients")
        clinical.close()

    if _CHROMA_DB.exists():
        ch = sqlite3.connect(str(_CHROMA_DB))
        chcur = ch.cursor()
        try:
            chcur.execute(
                "SELECT count(DISTINCT string_value) FROM embedding_metadata "
                "WHERE key='patient_id' AND string_value IS NOT NULL"
            )
            patients_with_vectors = chcur.fetchone()[0] or 0
        except Exception:
            pass
        ch.close()

    coverage_pct = round(patients_with_vectors / max(total_patients, 1) * 100, 1)
    retrieval_readiness_pct = coverage_pct

    result["summary"] = {
        "total_queries": total_queries,
        "unique_patients_queried": unique_patients_queried,
        "query_rate_per_day": query_rate_per_day,
        "vector_store_size": vector_store_size,
        "queue_size": queue_size,
        "retrieval_readiness_pct": retrieval_readiness_pct,
    }

    result["query_volume_daily"] = query_volume_daily
    result["query_by_patient"] = query_by_patient

    result["vector_summary"] = {
        "total_embeddings": vector_store_size,
        "total_metadata": total_metadata,
        "collections_count": collections_count,
        "queue_pending": queue_size,
        "queue_operations": queue_ops,
    }

    result["coverage_rate"] = {
        "total_patients": total_patients,
        "patients_with_vectors": patients_with_vectors,
        "coverage_pct": coverage_pct,
    }

    return result


# ── Breakdown ─────────────────────────────────────────────────────

def retrieval_breakdown():
    """Detailed retrieval drill-down: recent queries, per-patient retrieval,
    type distribution, embedding timeline, query text analysis."""

    result = {"available": True}

    clinical = _clinical_conn()
    chroma = _chroma_conn()

    # ── Recent query detail ──
    query_detail = []
    if clinical:
        ccur = clinical.cursor()
        try:
            ccur.execute(
                "SELECT t.id, t.patient_id, p.name, t.detail, t.actor, t.ts_utc "
                "FROM transaction_log t "
                "LEFT JOIN patients p ON t.patient_id = p.patient_id "
                "WHERE t.component='patient_chat' "
                "ORDER BY t.ts_utc DESC LIMIT 50"
            )
            query_detail = [
                {
                    "id": r[0],
                    "patient_id": r[1] or "",
                    "patient_name": r[2] or "",
                    "query_text": r[3] or "",
                    "actor": r[4] or "",
                    "timestamp": r[5] or "",
                }
                for r in ccur.fetchall()
            ]
        except Exception:
            pass

    result["query_detail"] = query_detail

    # ── Per-patient retrieval stats ──
    per_patient_retrieval = []
    if clinical and chroma:
        ccur = clinical.cursor()
        chcur = chroma.cursor()

        try:
            ccur.execute(
                "SELECT patient_id, name, disease FROM patients "
                "WHERE patient_id IS NOT NULL ORDER BY patient_id"
            )
            patients = ccur.fetchall()
        except Exception:
            patients = []

        # Query counts per patient
        query_counts = {}
        try:
            ccur.execute(
                "SELECT patient_id, count(*) FROM transaction_log "
                "WHERE component='patient_chat' AND patient_id IS NOT NULL "
                "GROUP BY patient_id"
            )
            query_counts = dict(ccur.fetchall())
        except Exception:
            pass

        # Vector counts per patient
        vector_counts = {}
        try:
            chcur.execute(
                "SELECT string_value, count(*) FROM embedding_metadata "
                "WHERE key='patient_id' AND string_value IS NOT NULL "
                "GROUP BY string_value"
            )
            vector_counts = dict(chcur.fetchall())
        except Exception:
            pass

        # Types per patient
        patient_types = {}
        try:
            chcur.execute(
                "SELECT m1.string_value as pid, m2.string_value as typ, count(*) "
                "FROM embedding_metadata m1 "
                "JOIN embedding_metadata m2 ON m1.id = m2.id "
                "WHERE m1.key='patient_id' AND m2.key='type' "
                "GROUP BY pid, typ ORDER BY pid, typ"
            )
            for pid, typ, cnt in chcur.fetchall():
                patient_types.setdefault(pid, []).append({"type": typ, "count": cnt})
        except Exception:
            pass

        for pid, name, disease in patients:
            qc = query_counts.get(pid, 0)
            vc = vector_counts.get(pid, 0)
            types = patient_types.get(pid, [])
            per_patient_retrieval.append({
                "patient_id": pid,
                "name": name or "",
                "disease": disease or "",
                "query_count": qc,
                "vector_count": vc,
                "has_vectors": vc > 0,
                "types": types,
            })

    result["per_patient_retrieval"] = per_patient_retrieval

    # ── Type distribution ──
    type_distribution = []
    if chroma:
        chcur = chroma.cursor()
        try:
            chcur.execute(
                "SELECT string_value, count(*) FROM embedding_metadata "
                "WHERE key='type' GROUP BY string_value ORDER BY count(*) DESC"
            )
            type_distribution = [{"type": r[0], "count": r[1]} for r in chcur.fetchall()]
        except Exception:
            pass

    result["type_distribution"] = type_distribution

    # ── Embedding timeline ──
    embedding_timeline = []
    if chroma:
        chcur = chroma.cursor()
        try:
            chcur.execute(
                "SELECT date(created_at) as d, count(*) FROM embeddings_queue "
                "WHERE created_at IS NOT NULL GROUP BY d ORDER BY d"
            )
            embedding_timeline = [{"date": r[0], "count": r[1]} for r in chcur.fetchall()]
        except Exception:
            pass

    result["embedding_timeline"] = embedding_timeline

    # ── Query text analysis (word frequency) ──
    query_text_analysis = []
    if clinical:
        ccur = clinical.cursor()
        try:
            ccur.execute(
                "SELECT detail FROM transaction_log "
                "WHERE component='patient_chat' AND detail IS NOT NULL"
            )
            word_freq = {}
            for (text,) in ccur.fetchall():
                for word in text.lower().split():
                    word = word.strip(".,;:!?\"'()[]{}<>/-_")
                    if word and len(word) > 1 and word not in _STOPWORDS:
                        word_freq[word] = word_freq.get(word, 0) + 1
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            query_text_analysis = [{"word": w, "count": c} for w, c in sorted_words]
        except Exception:
            pass

    result["query_text_analysis"] = query_text_analysis

    # Close connections
    if clinical:
        try:
            clinical.close()
        except Exception:
            pass
    if chroma:
        try:
            chroma.close()
        except Exception:
            pass

    return result


# ── Definitions ───────────────────────────────────────────────────

def retrieval_definitions():
    """Metric definitions for the Retrieval Dashboard."""
    return {
        "available": True,
        "metrics": [
            {
                "name": "total_queries",
                "description": "Total number of patient_chat transactions recorded in the transaction log.",
                "source": "clinical.db — transaction_log WHERE component='patient_chat'",
            },
            {
                "name": "unique_patients_queried",
                "description": "Count of distinct patients who have had at least one patient_chat query.",
                "source": "clinical.db — transaction_log DISTINCT patient_id WHERE component='patient_chat'",
            },
            {
                "name": "query_rate_per_day",
                "description": "Average number of patient_chat queries per day across the observed date range.",
                "source": "clinical.db — total_queries / date span of transaction_log",
            },
            {
                "name": "vector_store_size",
                "description": "Total number of vector embeddings stored in ChromaDB.",
                "source": "chroma.sqlite3 — SELECT count(*) FROM embeddings",
            },
            {
                "name": "queue_size",
                "description": "Number of entries in the embeddings ingestion queue (pending and processed).",
                "source": "chroma.sqlite3 — SELECT count(*) FROM embeddings_queue",
            },
            {
                "name": "retrieval_readiness_pct",
                "description": "Percentage of patients in clinical.db that have at least one vector embedding in ChromaDB.",
                "source": "Cross-reference: clinical.db patients vs embedding_metadata patient_id",
            },
            {
                "name": "query_volume_daily",
                "description": "Daily time series of patient_chat query counts.",
                "source": "clinical.db — transaction_log GROUP BY date(ts_utc)",
            },
            {
                "name": "query_by_patient",
                "description": "Per-patient breakdown of query counts, joined with patient name and disease.",
                "source": "clinical.db — transaction_log JOIN patients",
            },
            {
                "name": "per_patient_retrieval",
                "description": "Combined view of each patient's query count, vector count, and embedded document types.",
                "source": "clinical.db transaction_log + chroma.sqlite3 embedding_metadata",
            },
            {
                "name": "type_distribution",
                "description": "Distribution of document types stored in the vector database.",
                "source": "chroma.sqlite3 — embedding_metadata WHERE key='type'",
            },
            {
                "name": "embedding_timeline",
                "description": "Daily time series of embeddings queue entries by creation date.",
                "source": "chroma.sqlite3 — embeddings_queue GROUP BY date(created_at)",
            },
            {
                "name": "query_text_analysis",
                "description": "Top 20 most frequent words from patient_chat query texts (stopwords excluded).",
                "source": "clinical.db — transaction_log detail column, word frequency analysis",
            },
            {
                "name": "coverage_rate",
                "description": "Patient vector coverage: total patients, patients with vectors, and coverage percentage.",
                "source": "clinical.db patients vs chroma.sqlite3 embedding_metadata patient_id",
            },
        ],
        "data_sources": {
            "clinical_db": "data/clinical.db — transaction_log (component='patient_chat'), conversation_log, patients, analyses",
            "chroma_db": "data/vector_db/chroma.sqlite3 — embeddings, embedding_metadata, embeddings_queue, collections",
        },
    }


if __name__ == "__main__":
    ov = retrieval_overview()
    print(json.dumps(ov, indent=2, default=str))
