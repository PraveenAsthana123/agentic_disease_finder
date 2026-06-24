#!/usr/bin/env python3
"""Vector DB ingest (cron target, §87 VECTOR-INGEST).

Reads clinical records from data/clinical.db, embeds each with a local Ollama
embedding model, and upserts into a persistent ChromaDB collection so the
patient RAG chat can do real semantic search (not just keyword).

Usage: python scripts/vector_ingest.py
Idempotent: re-running upserts by stable id (no duplicates).
"""
from __future__ import annotations

import json
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "clinical.db"
VDB = ROOT / "data" / "vector_db"
OLLAMA = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def embed(text: str):
    body = json.dumps({"model": EMBED_MODEL, "prompt": text[:2000]}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["embedding"]


def collect_docs():
    """Build (id, text, metadata) per clinical record."""
    if not DB.exists():
        return []
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    docs = []

    for r in c.execute("SELECT * FROM patients").fetchall():
        d = dict(r)
        docs.append((f"patient:{d['patient_id']}",
                     f"Patient {d['patient_id']} {d.get('name','')} age {d.get('age')} {d.get('gender','')} disease {d.get('disease','')}",
                     {"type": "patient", "patient_id": d["patient_id"]}))

    for r in c.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 500").fetchall():
        d = dict(r)
        docs.append((f"analysis:{d['id']}",
                     f"Analysis for {d.get('patient_id')}: {d.get('disease')} predicted {d.get('predicted_label')} "
                     f"confidence {d.get('confidence')} quality {d.get('signal_quality')}",
                     {"type": "analysis", "patient_id": d.get("patient_id") or ""}))

    # Clinical capture tables (generic fields_json).
    for t in ["medications", "mri_findings", "outcomes", "neuropsych", "seizure_metadata",
              "hitl_reviews", "clinical_history", "eeg_interpretation"]:
        try:
            rows = c.execute(f"SELECT * FROM {t} ORDER BY id DESC LIMIT 300").fetchall()
        except Exception:
            continue
        for r in rows:
            d = dict(r)
            fields = json.loads(d.get("fields_json") or "{}")
            txt = f"{t} for {d.get('patient_id')}: " + ", ".join(f"{k}={v}" for k, v in fields.items())
            docs.append((f"{t}:{d['id']}", txt, {"type": t, "patient_id": d.get("patient_id") or ""}))

    for r in c.execute("SELECT * FROM surveys ORDER BY id DESC LIMIT 300").fetchall():
        d = dict(r)
        ans = json.loads(d.get("answers_json") or "{}")
        docs.append((f"survey:{d['id']}",
                     f"Survey ({d.get('kind')}) for {d.get('patient_id')}: " + ", ".join(f"{k}={v}" for k, v in ans.items()),
                     {"type": "survey", "patient_id": d.get("patient_id") or ""}))
    c.close()
    return docs


def main() -> int:
    import chromadb
    VDB.mkdir(parents=True, exist_ok=True)
    docs = collect_docs()
    if not docs:
        print("No clinical records to ingest."); return 0

    client = chromadb.PersistentClient(path=str(VDB))
    col = client.get_or_create_collection("clinical")

    ids, texts, metas, embs = [], [], [], []
    failed = 0
    for did, text, meta in docs:
        try:
            embs.append(embed(text))
            ids.append(did); texts.append(text); metas.append(meta)
        except Exception:
            failed += 1
    if ids:
        col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[vector_ingest] {ts}")
    print(f"  embedded {len(ids)} / {len(docs)} records (failed {failed}) via {EMBED_MODEL}")
    print(f"  collection 'clinical' total: {col.count()} @ {VDB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
