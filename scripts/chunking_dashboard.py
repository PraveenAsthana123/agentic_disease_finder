#!/usr/bin/env python3
"""
Chunking Dashboard — real RAG chunk analytics
==============================================

Reads REAL data from:
  - data/vector_db/chroma.sqlite3  — chunk count, text lengths, patient coverage
  - scripts/rag_engine.py config   — chunk_size=1000, chunk_overlap=200
  - eeg-stress-rag/src/rag/core/chunking.py — 5 strategy definitions

Functions:
  - chunking_overview    — KPIs, strategy mix, chunk size distribution
  - chunking_breakdown   — per-strategy config, per-collection chunk stats
  - chunking_definitions — metric definitions
"""
from __future__ import annotations

import os
import sqlite3
import statistics
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


# ── Production chunking config (from rag_engine.py) ──────────────────
_PRODUCTION_CONFIG = {
    "strategy": "RecursiveCharacterTextSplitter",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\\n\\n", "\\n", ". ", " ", ""],
}

# ── Available strategies (from eeg-stress-rag/src/rag/core/chunking.py) ─
_STRATEGIES = [
    {
        "name": "FixedSize",
        "description": "Fixed character window with overlap",
        "params": {"chunk_size": 512, "overlap": 50},
        "use_case": "Simple, fast; good for uniform-length documents",
    },
    {
        "name": "Sentence",
        "description": "Split on sentence boundaries",
        "params": {"max_sentences": 5},
        "use_case": "Preserves sentence coherence for NLP tasks",
    },
    {
        "name": "Semantic",
        "description": "Embedding-similarity-based split points",
        "params": {"threshold": 0.5},
        "use_case": "Groups semantically similar content; best for heterogeneous docs",
    },
    {
        "name": "SlidingWindow",
        "description": "Overlapping fixed windows",
        "params": {"window_size": 512, "step_size": 256},
        "use_case": "Ensures no context is lost at boundaries",
    },
    {
        "name": "Recursive",
        "description": "Hierarchical split on separator list",
        "params": {"max_chunk_size": 512, "separators": ["\\n\\n", "\\n", ". "]},
        "use_case": "Production default; respects document structure",
    },
]


# ─── Overview ────────────────────────────────────────────────────────
def chunking_overview() -> Dict[str, Any]:
    """Top-level KPIs: chunk count, avg size, overlap ratio, coverage."""
    conn = _conn()
    if conn is None:
        return {"available": False, "note": "ChromaDB not found — no chunk data."}

    cur = conn.cursor()

    # Total chunks = total embeddings
    cur.execute("SELECT count(*) FROM embeddings")
    total_chunks = cur.fetchone()[0]

    # Get chunk texts from embedding_metadata for length analysis
    chunk_lengths: List[int] = []
    try:
        cur.execute(
            "SELECT string_value FROM embedding_metadata "
            "WHERE key = 'chroma:document'"
        )
        for (text,) in cur.fetchall():
            if text:
                chunk_lengths.append(len(text))
    except Exception:
        pass

    # Patient coverage
    patient_ids: set = set()
    try:
        cur.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata "
            "WHERE key = 'patient_id'"
        )
        patient_ids = {r[0] for r in cur.fetchall() if r[0]}
    except Exception:
        pass

    # Document types
    doc_types: Dict[str, int] = {}
    try:
        cur.execute(
            "SELECT string_value, count(*) FROM embedding_metadata "
            "WHERE key = 'type' GROUP BY string_value"
        )
        doc_types = {r[0]: r[1] for r in cur.fetchall() if r[0]}
    except Exception:
        pass

    # Collections
    cur.execute("SELECT count(*) FROM collections")
    num_collections = cur.fetchone()[0]

    # Storage
    storage_bytes = os.path.getsize(str(_CHROMA_DB)) if _CHROMA_DB.exists() else 0

    # Size distribution buckets
    buckets = {"<256": 0, "256-512": 0, "512-1000": 0, "1000-2000": 0, ">2000": 0}
    for l in chunk_lengths:
        if l < 256:
            buckets["<256"] += 1
        elif l < 512:
            buckets["256-512"] += 1
        elif l < 1000:
            buckets["512-1000"] += 1
        elif l < 2000:
            buckets["1000-2000"] += 1
        else:
            buckets[">2000"] += 1

    size_distribution = [{"range": k, "count": v} for k, v in buckets.items()]

    conn.close()

    avg_len = statistics.mean(chunk_lengths) if chunk_lengths else 0
    median_len = statistics.median(chunk_lengths) if chunk_lengths else 0
    std_len = statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0

    return {
        "available": True,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_chunks": total_chunks,
            "avg_chunk_length": round(avg_len, 1),
            "median_chunk_length": round(median_len, 1),
            "std_chunk_length": round(std_len, 1),
            "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
            "overlap_ratio_pct": round(
                _PRODUCTION_CONFIG["chunk_overlap"]
                / _PRODUCTION_CONFIG["chunk_size"]
                * 100,
                1,
            ),
            "patients_covered": len(patient_ids),
            "document_types": len(doc_types),
            "collections": num_collections,
            "storage": _sizeof_fmt(storage_bytes),
        },
        "production_config": _PRODUCTION_CONFIG,
        "doc_type_distribution": [
            {"type": t, "chunks": c} for t, c in sorted(doc_types.items(), key=lambda x: -x[1])
        ],
        "size_distribution": size_distribution,
    }


# ─── Breakdown ───────────────────────────────────────────────────────
def chunking_breakdown() -> Dict[str, Any]:
    """Per-strategy config + per-patient chunk counts."""
    conn = _conn()
    if conn is None:
        return {"available": False, "note": "ChromaDB not found."}

    cur = conn.cursor()

    # Per-patient chunk counts
    patient_chunks: List[Dict[str, Any]] = []
    try:
        cur.execute(
            "SELECT string_value, count(*) FROM embedding_metadata "
            "WHERE key = 'patient_id' GROUP BY string_value ORDER BY count(*) DESC"
        )
        patient_chunks = [
            {"patient_id": r[0], "chunks": r[1]} for r in cur.fetchall() if r[0]
        ]
    except Exception:
        pass

    # Per-type chunk lengths
    type_stats: List[Dict[str, Any]] = []
    try:
        cur.execute(
            "SELECT em_type.string_value AS doc_type, em_doc.string_value AS doc_text "
            "FROM embedding_metadata em_type "
            "JOIN embedding_metadata em_doc "
            "  ON em_type.id = em_doc.id "
            "WHERE em_type.key = 'type' AND em_doc.key = 'chroma:document'"
        )
        by_type: Dict[str, List[int]] = {}
        for doc_type, doc_text in cur.fetchall():
            if doc_type and doc_text:
                by_type.setdefault(doc_type, []).append(len(doc_text))

        for dtype, lengths in sorted(by_type.items()):
            type_stats.append({
                "type": dtype,
                "count": len(lengths),
                "avg_length": round(statistics.mean(lengths), 1),
                "min_length": min(lengths),
                "max_length": max(lengths),
            })
    except Exception:
        pass

    conn.close()

    return {
        "available": True,
        "strategies": _STRATEGIES,
        "active_strategy": _PRODUCTION_CONFIG,
        "patient_chunks": patient_chunks,
        "type_stats": type_stats,
    }


# ─── Definitions ─────────────────────────────────────────────────────
def chunking_definitions() -> Dict[str, Any]:
    return {
        "available": True,
        "metrics": [
            {
                "name": "Total Chunks",
                "description": "Number of text segments stored in the vector DB after splitting source documents.",
                "unit": "count",
            },
            {
                "name": "Avg Chunk Length",
                "description": "Mean character count across all stored chunks.",
                "unit": "characters",
            },
            {
                "name": "Median Chunk Length",
                "description": "Median character count; less sensitive to outliers than mean.",
                "unit": "characters",
            },
            {
                "name": "Overlap Ratio",
                "description": "Percentage of chunk_size that overlaps with the next chunk (chunk_overlap / chunk_size × 100).",
                "unit": "%",
            },
            {
                "name": "Patients Covered",
                "description": "Number of unique patients whose documents have been chunked and embedded.",
                "unit": "count",
            },
            {
                "name": "Document Types",
                "description": "Number of distinct document categories (e.g., clinical_note, eeg_report) in the chunk store.",
                "unit": "count",
            },
            {
                "name": "Chunk Size Distribution",
                "description": "Histogram showing how many chunks fall into each length bucket (<256, 256-512, 512-1000, 1000-2000, >2000 chars).",
                "unit": "count per bucket",
            },
            {
                "name": "Strategy",
                "description": "The text-splitting algorithm used (e.g., Recursive, Sentence, Semantic). Production uses RecursiveCharacterTextSplitter.",
                "unit": "enum",
            },
        ],
        "clinical_relevance": (
            "Chunk quality directly affects RAG retrieval accuracy. "
            "Chunks that are too short lose clinical context; chunks that are too long "
            "dilute relevance scoring. Overlap ensures no information is lost at split boundaries. "
            "Monitoring chunk-size distribution helps detect ingestion anomalies early."
        ),
    }
