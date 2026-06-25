#!/usr/bin/env python3
"""Clinical Data Manager — Dataset Versioning.

Produces a reproducible version manifest of the REAL dataset + model artifacts:
content SHA-256 hashes of every sample .npz and model .joblib on disk, plus a
live snapshot of the `uploads` table. A composite fingerprint changes iff any
tracked artifact changes — the basis for dataset/model lineage (§12, §41.6).

100% real (hashes actual file bytes) — no synthetic, no mutation.
"""

import glob
import hashlib
import os
import sqlite3
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "clinical.db")


def _sha256(path, _buf=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_buf), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _npz_shape(path):
    """Best-effort (n_samples, n_features) from an .npz numeric X array.

    Uses the default safe loader (no object/pickle deserialization) — only the
    numeric feature matrix shape is read; failures degrade to None.
    """
    try:
        import numpy as np
        with np.load(path) as z:  # allow_pickle defaults to False (safe)
            if "X" in z.files:
                x = z["X"]
                return {"n_samples": int(x.shape[0]), "n_features": int(x.shape[1]) if x.ndim > 1 else 1}
    except Exception:
        return None
    return None


def _artifacts(pattern, kind):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, pattern))):
        rel = os.path.relpath(p, ROOT)
        st = os.stat(p)
        entry = {"artifact": rel, "kind": kind, "bytes": st.st_size,
                 "sha256": _sha256(p),
                 "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if kind == "dataset":
            shape = _npz_shape(p)
            if shape:
                entry["shape"] = shape
        out.append(entry)
    return out


def version_manifest():
    """Full dataset/model version manifest with a composite fingerprint."""
    datasets = _artifacts("data/*/sample/*.npz", "dataset")
    models = _artifacts("models/*.joblib", "model")

    # uploads table snapshot (live)
    uploads = {"available": False}
    try:
        c = sqlite3.connect(DB_PATH)
        n = c.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        by_disease = dict(c.execute("SELECT disease, COUNT(*) FROM uploads GROUP BY disease").fetchall())
        by_dept = dict((d or "unassigned", k) for d, k in
                       c.execute("SELECT department, COUNT(*) FROM uploads GROUP BY department").fetchall())
        c.close()
        uploads = {"available": True, "total": n, "by_disease": by_disease, "by_department": by_dept}
    except sqlite3.OperationalError:
        pass

    # composite fingerprint = hash of all artifact hashes (order-stable)
    parts = [a["sha256"] or "" for a in datasets + models]
    composite = hashlib.sha256("".join(parts).encode()).hexdigest()[:16] if parts else None

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "composite_fingerprint": composite,
        "datasets": datasets,
        "models": models,
        "uploads_snapshot": uploads,
        "summary": {
            "n_datasets": len(datasets), "n_models": len(models),
            "total_dataset_bytes": sum(a["bytes"] for a in datasets),
            "total_model_bytes": sum(a["bytes"] for a in models),
        },
        "note": ("Content SHA-256 over real artifact bytes — the composite fingerprint changes "
                 "iff any dataset/model file changes. Basis for dataset→model lineage (§12)."),
    }


if __name__ == "__main__":
    m = version_manifest()
    print("Dataset version manifest:")
    print("  fingerprint:", m["composite_fingerprint"])
    print("  summary:", m["summary"])
    for d in m["datasets"][:4]:
        print(f"  {d['artifact']}: {(d['sha256'] or '')[:12]}… {d.get('shape')}")
