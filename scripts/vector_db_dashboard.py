#!/usr/bin/env python3
"""
Vector DB Dashboard — real ChromaDB / vector store monitoring
=============================================================

Reads REAL data from data/vector_db/chroma.sqlite3:
  - Collection metadata (name, dimension, count)
  - Embedding records with metadata (patient_id, type, document)
  - Queue operations (inserts/updates/deletes)
  - Storage size on disk

Functions:
  - vector_db_overview     — KPIs, collection stats, storage, health
  - vector_db_collections  — per-collection breakdown, record samples
  - vector_db_operations   — queue ops, ingestion timeline, throughput
  - vector_db_definitions  — metric definitions
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DB = _PROJECT_ROOT / "data" / "vector_db" / "chroma.sqlite3"


def _conn():
    """Return a read-only SQLite connection to the ChromaDB file."""
    if not _CHROMA_DB.exists():
        return None
    return sqlite3.connect(str(_CHROMA_DB))


def _sizeof_fmt(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# ─── Overview ──────────────────────────────────────────────────────────
def vector_db_overview() -> Dict[str, Any]:
    """Top-level KPIs: total vectors, collections, dimensions, storage, health."""
    conn = _conn()
    if conn is None:
        return {"error": "ChromaDB not found", "status": "offline"}

    cur = conn.cursor()

    # Collections
    cur.execute("SELECT id, name, dimension FROM collections")
    collections = [{"id": r[0], "name": r[1], "dimension": r[2]} for r in cur.fetchall()]

    # Embedding count
    cur.execute("SELECT count(*) FROM embeddings")
    total_vectors = cur.fetchone()[0]

    # Queue depth
    cur.execute("SELECT count(*) FROM embeddings_queue")
    queue_depth = cur.fetchone()[0]

    # Metadata keys
    cur.execute("SELECT DISTINCT key FROM embedding_metadata WHERE key != 'chroma:document'")
    meta_keys = [r[0] for r in cur.fetchall()]

    # Unique patients
    cur.execute("SELECT count(DISTINCT string_value) FROM embedding_metadata WHERE key='patient_id'")
    unique_patients = cur.fetchone()[0]

    # Date range
    cur.execute("SELECT min(created_at), max(created_at) FROM embeddings")
    date_range = cur.fetchone()

    # Storage
    db_size = os.path.getsize(str(_CHROMA_DB))
    hnsw_dir = _CHROMA_DB.parent
    total_storage = sum(
        f.stat().st_size for f in hnsw_dir.rglob("*") if f.is_file()
    )

    # Fulltext index size (fts5 may not be available)
    try:
        cur.execute("SELECT count(*) FROM embedding_fulltext_search")
        fts_count = cur.fetchone()[0]
    except Exception:
        fts_count = 0

    # Types distribution
    cur.execute(
        "SELECT string_value, count(*) FROM embedding_metadata "
        "WHERE key='type' GROUP BY string_value ORDER BY count(*) DESC"
    )
    type_dist = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]

    conn.close()

    dimension = collections[0]["dimension"] if collections else None

    return {
        "summary": {
            "total_vectors": total_vectors,
            "total_collections": len(collections),
            "dimension": dimension,
            "queue_depth": queue_depth,
            "unique_patients": unique_patients,
            "storage_bytes": total_storage,
            "storage_human": _sizeof_fmt(total_storage),
            "db_size_human": _sizeof_fmt(db_size),
            "fts_indexed": fts_count,
            "metadata_keys": meta_keys,
            "status": "online",
            "health": "healthy" if total_vectors > 0 else "empty",
        },
        "collections": collections,
        "type_distribution": type_dist,
        "date_range": {
            "earliest": date_range[0] if date_range else None,
            "latest": date_range[1] if date_range else None,
        },
        "source": "data/vector_db/chroma.sqlite3",
        "generated": datetime.now().isoformat(),
    }


# ─── Collections detail ───────────────────────────────────────────────
def vector_db_collections() -> Dict[str, Any]:
    """Per-collection breakdown: dimension, vector count, sample records."""
    conn = _conn()
    if conn is None:
        return {"error": "ChromaDB not found"}

    cur = conn.cursor()

    cur.execute("SELECT id, name, dimension FROM collections")
    colls = cur.fetchall()

    result = []
    for cid, cname, dim in colls:
        # Count vectors in this collection's segments
        cur.execute(
            "SELECT count(*) FROM embeddings e "
            "JOIN segments s ON e.segment_id = s.id "
            "WHERE s.collection = ?",
            (cid,),
        )
        vec_count = cur.fetchone()[0]

        # Sample documents
        cur.execute(
            "SELECT e.embedding_id, em_pid.string_value, em_doc.string_value "
            "FROM embeddings e "
            "JOIN segments s ON e.segment_id = s.id "
            "LEFT JOIN embedding_metadata em_pid ON em_pid.id = e.id AND em_pid.key = 'patient_id' "
            "LEFT JOIN embedding_metadata em_doc ON em_doc.id = e.id AND em_doc.key = 'chroma:document' "
            "WHERE s.collection = ? "
            "ORDER BY e.created_at DESC LIMIT 10",
            (cid,),
        )
        samples = []
        for eid, pid, doc in cur.fetchall():
            samples.append({
                "embedding_id": eid,
                "patient_id": pid,
                "document_preview": (doc[:120] + "...") if doc and len(doc) > 120 else doc,
            })

        # Patient distribution for this collection
        cur.execute(
            "SELECT em.string_value, count(*) FROM embedding_metadata em "
            "JOIN embeddings e ON em.id = e.id "
            "JOIN segments s ON e.segment_id = s.id "
            "WHERE s.collection = ? AND em.key = 'patient_id' "
            "GROUP BY em.string_value ORDER BY count(*) DESC LIMIT 15",
            (cid,),
        )
        patient_dist = [{"patient_id": r[0], "vectors": r[1]} for r in cur.fetchall()]

        result.append({
            "id": cid,
            "name": cname,
            "dimension": dim,
            "vector_count": vec_count,
            "samples": samples,
            "patient_distribution": patient_dist,
        })

    conn.close()
    return {
        "collections": result,
        "generated": datetime.now().isoformat(),
    }


# ─── Operations / queue ───────────────────────────────────────────────
def vector_db_operations() -> Dict[str, Any]:
    """Queue operations, ingestion timeline, throughput metrics."""
    conn = _conn()
    if conn is None:
        return {"error": "ChromaDB not found"}

    cur = conn.cursor()

    # Operation type counts (1=INSERT, 2=UPDATE, 3=DELETE in Chroma)
    op_labels = {1: "INSERT", 2: "UPDATE", 3: "DELETE"}
    cur.execute("SELECT operation, count(*) FROM embeddings_queue GROUP BY operation")
    ops_summary = [
        {"operation": op_labels.get(r[0], f"OP_{r[0]}"), "count": r[1]}
        for r in cur.fetchall()
    ]

    # Ingestion timeline (by day)
    cur.execute(
        "SELECT date(created_at) as day, count(*) FROM embeddings_queue "
        "GROUP BY day ORDER BY day"
    )
    timeline = [{"date": r[0], "operations": r[1]} for r in cur.fetchall()]

    # Recent queue entries
    cur.execute(
        "SELECT seq_id, created_at, operation, id FROM embeddings_queue "
        "ORDER BY seq_id DESC LIMIT 15"
    )
    recent = [
        {
            "seq_id": r[0],
            "created_at": r[1],
            "operation": op_labels.get(r[2], f"OP_{r[2]}"),
            "embedding_id": r[3],
        }
        for r in cur.fetchall()
    ]

    # Max seq_id
    cur.execute("SELECT max(seq_id) FROM embeddings_queue")
    max_seq = cur.fetchone()[0] or 0

    # Throughput: total ops / days span
    cur.execute("SELECT min(created_at), max(created_at), count(*) FROM embeddings_queue")
    row = cur.fetchone()
    throughput = None
    if row[0] and row[1]:
        try:
            t0 = datetime.fromisoformat(row[0])
            t1 = datetime.fromisoformat(row[1])
            days = max((t1 - t0).total_seconds() / 86400, 1)
            throughput = {
                "total_ops": row[2],
                "days_span": round(days, 1),
                "ops_per_day": round(row[2] / days, 1),
            }
        except Exception:
            pass

    conn.close()
    return {
        "operations_summary": ops_summary,
        "timeline": timeline,
        "recent_queue": recent,
        "max_seq_id": max_seq,
        "throughput": throughput,
        "generated": datetime.now().isoformat(),
    }


# ─── Definitions ──────────────────────────────────────────────────────
def vector_db_definitions() -> Dict[str, Any]:
    """Metric definitions for the Vector DB Dashboard."""
    return {
        "definitions": [
            {
                "metric": "Total Vectors",
                "description": "Number of embedding vectors stored in the database",
                "unit": "count",
            },
            {
                "metric": "Dimension",
                "description": "Dimensionality of each embedding vector (e.g. 768 for all-MiniLM-L6-v2)",
                "unit": "int",
            },
            {
                "metric": "Queue Depth",
                "description": "Number of pending operations in the embeddings queue (INSERT/UPDATE/DELETE)",
                "unit": "count",
            },
            {
                "metric": "Storage",
                "description": "Total disk space used by the vector DB (SQLite + HNSW index files)",
                "unit": "bytes",
            },
            {
                "metric": "Unique Patients",
                "description": "Number of distinct patient_id values across all embeddings",
                "unit": "count",
            },
            {
                "metric": "FTS Indexed",
                "description": "Documents indexed in the full-text search index",
                "unit": "count",
            },
            {
                "metric": "Ops / Day",
                "description": "Average queue operations per day over the data time span",
                "unit": "ops/day",
            },
            {
                "metric": "HNSW",
                "description": "Hierarchical Navigable Small World graph — the ANN index used by ChromaDB for nearest-neighbor search",
                "unit": "index",
            },
        ],
        "technology": {
            "database": "ChromaDB (SQLite + HNSW)",
            "distance_metric": "L2 (Euclidean)",
            "index_type": "HNSW (ef_construction=100, ef_search=100, max_neighbors=16)",
        },
        "generated": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(vector_db_overview(), indent=2, default=str))
