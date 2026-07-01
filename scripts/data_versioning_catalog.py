"""
Data Versioning & Catalog — NeuroAI EEG
========================================
Tracks the data catalog and versioning status across all dataset
directories using REAL data:
  - Dataset dirs:   data/<disease>/, data/eeg_datasets/<disease>/
  - Model artifacts: models/*.joblib
  - SQLite DBs:     data/*.db
  - Track log:      jobs/logs/track.jsonl  (data-related events)
  - Git log:        recent commits touching data/ paths
"""

import json, os, pathlib, sqlite3, subprocess
from datetime import datetime, timedelta, timezone
from collections import Counter

MDT = timezone(timedelta(hours=-6))
BASE = pathlib.Path(__file__).resolve().parent.parent
NOW = datetime.now(MDT)

# Staleness thresholds (days)
FRESH_DAYS = 7
ACTIVE_DAYS = 30

# Directories to skip when scanning datasets
_SKIP_DIRS = {"__pycache__", ".git", "node_modules"}
_SKIP_FILES = {"__init__.py", "__pycache__"}


# ── Helpers ────────────────────────────────────────────────────────

def _staleness_status(mtime):
    """Return fresh / active / stale based on last-modified datetime."""
    age = (NOW - mtime).days
    if age < FRESH_DAYS:
        return "fresh"
    if age < ACTIVE_DAYS:
        return "active"
    return "stale"


def _git_version_for_path(rel_path):
    """Return short hash of the last git commit touching *rel_path*, or 'v1.0'."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%h", "--", rel_path],
            cwd=str(BASE), stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        h = out.strip()
        return h if h else "v1.0"
    except Exception:
        return "v1.0"


def _scan_directory(dirpath):
    """Return (file_count, size_bytes, formats_set, latest_mtime) for a dir."""
    file_count = 0
    total_size = 0
    formats = set()
    latest = None

    if not dirpath.is_dir():
        return 0, 0, set(), None

    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            if fname in _SKIP_FILES:
                continue
            fp = pathlib.Path(root) / fname
            try:
                stat = fp.stat()
            except OSError:
                continue
            file_count += 1
            total_size += stat.st_size
            ext = fp.suffix.lower()
            if ext:
                formats.add(ext)
            mt = datetime.fromtimestamp(stat.st_mtime, tz=MDT)
            if latest is None or mt > latest:
                latest = mt

    return file_count, total_size, formats, latest


def _get_dataset_dirs():
    """Collect all dataset directories under data/ and data/eeg_datasets/."""
    dirs = []
    data_dir = BASE / "data"
    if not data_dir.is_dir():
        return dirs

    # Top-level data/ subdirectories (skip files, __pycache__, eeg_datasets handled separately)
    skip_top = {"eeg_datasets", "__pycache__", ".git"}
    for entry in sorted(data_dir.iterdir()):
        if entry.is_dir() and entry.name not in skip_top and entry.name not in _SKIP_DIRS:
            dirs.append(entry)

    # data/eeg_datasets/ subdirectories
    eeg_dir = data_dir / "eeg_datasets"
    if eeg_dir.is_dir():
        for entry in sorted(eeg_dir.iterdir()):
            if entry.is_dir() and entry.name not in _SKIP_DIRS:
                dirs.append(entry)

    return dirs


def _build_catalog():
    """Build the full dataset catalog list."""
    dataset_dirs = _get_dataset_dirs()
    catalog = []
    for d in dataset_dirs:
        file_count, size_bytes, formats, latest_mtime = _scan_directory(d)
        if file_count == 0:
            continue
        rel = str(d.relative_to(BASE))
        version = _git_version_for_path(rel)
        status = _staleness_status(latest_mtime) if latest_mtime else "stale"
        catalog.append({
            "name": d.name,
            "path": rel,
            "file_count": file_count,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "formats": sorted(formats),
            "last_modified": latest_mtime.isoformat() if latest_mtime else None,
            "version": version,
            "status": status,
        })
    return catalog


def _get_db_files():
    """Scan for SQLite .db files under data/."""
    data_dir = BASE / "data"
    dbs = []
    if not data_dir.is_dir():
        return dbs
    for f in sorted(data_dir.glob("*.db")):
        stat = f.stat()
        table_count = 0
        try:
            conn = sqlite3.connect(str(f))
            cur = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            )
            table_count = cur.fetchone()[0]
            conn.close()
        except Exception:
            pass
        dbs.append({
            "name": f.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "tables": table_count,
            "path": str(f.relative_to(BASE)),
        })
    return dbs


def _get_model_artifacts():
    """Scan models/*.joblib."""
    model_dir = BASE / "models"
    artifacts = []
    if not model_dir.is_dir():
        return artifacts
    for f in sorted(model_dir.glob("*.joblib")):
        stat = f.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=MDT)
        artifacts.append({
            "name": f.stem,
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": mtime.isoformat(),
        })
    return artifacts


def _get_git_data_history(n=20):
    """Recent git commits touching data/ paths."""
    try:
        out = subprocess.check_output(
            ["git", "log", "--oneline", "--format=%H|%ai|%s",
             "-n", str(n), "--", "data/"],
            cwd=str(BASE), stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        commits = []
        for line in out.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            # count files changed in that commit
            try:
                fc_out = subprocess.check_output(
                    ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", parts[0]],
                    cwd=str(BASE), stderr=subprocess.DEVNULL, text=True, timeout=5,
                )
                files_changed = len([l for l in fc_out.strip().split("\n") if l])
            except Exception:
                files_changed = 0
            commits.append({
                "hash": parts[0][:8],
                "message": parts[2][:120],
                "date": parts[1][:10],
                "files_changed": files_changed,
            })
        return commits
    except Exception:
        return []


def _get_data_events(limit=30):
    """Pull data-related events from track.jsonl."""
    track_path = BASE / "jobs" / "logs" / "track.jsonl"
    events = []
    if not track_path.exists():
        return events
    keywords = ["data", "dataset", "ingest"]
    try:
        with open(track_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                evt = rec.get("event", "").lower()
                if any(kw in evt for kw in keywords):
                    events.append({
                        "timestamp": rec.get("ts_local", rec.get("ts_utc", "")),
                        "event": rec.get("event", ""),
                        "agent": rec.get("user", rec.get("host", "")),
                    })
    except Exception:
        pass
    return events[-limit:]


def _build_lineage(catalog, artifacts):
    """Build simple data lineage: source dataset -> model artifact via train script."""
    lineage = []
    artifact_names = {a["name"] for a in artifacts}
    for entry in catalog:
        ds_name = entry["name"]
        model_name = f"{ds_name}_model"
        if model_name in artifact_names:
            lineage.append({
                "source": entry["path"],
                "artifact": f"models/{model_name}.joblib",
                "pipeline": "train.py",
            })
    return lineage


# ── Public API ─────────────────────────────────────────────────────

def overview():
    """KPIs, catalog, format distribution, size by dataset."""
    catalog = _build_catalog()
    db_files = _get_db_files()
    artifacts = _get_model_artifacts()

    if not catalog and not db_files:
        return {"available": False, "reason": "No datasets or databases found"}

    total_files = sum(e["file_count"] for e in catalog)
    total_size = sum(e["size_mb"] for e in catalog)

    # Last updated across everything
    all_mtimes = [e["last_modified"] for e in catalog if e["last_modified"]]
    for a in artifacts:
        all_mtimes.append(a["modified"])
    last_updated = max(all_mtimes) if all_mtimes else None

    # Format distribution
    fmt_counter = Counter()
    for entry in catalog:
        for fmt in entry["formats"]:
            fmt_counter[fmt] += 1
    # Recount by actual files, not by datasets — re-scan dirs
    fmt_file_counter = Counter()
    for entry in catalog:
        dirpath = BASE / entry["path"]
        if not dirpath.is_dir():
            continue
        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for fname in files:
                if fname in _SKIP_FILES:
                    continue
                ext = pathlib.Path(fname).suffix.lower()
                if ext:
                    fmt_file_counter[ext] += 1

    format_distribution = [
        {"format": fmt, "count": cnt}
        for fmt, cnt in fmt_file_counter.most_common()
    ]

    size_by_dataset = [
        {"name": e["name"], "size_mb": e["size_mb"]}
        for e in sorted(catalog, key=lambda x: x["size_mb"], reverse=True)
    ]

    return {
        "available": True,
        "kpis": {
            "total_datasets": len(catalog),
            "total_files": total_files,
            "total_size_mb": round(total_size, 2),
            "db_count": len(db_files),
            "model_artifacts": len(artifacts),
            "last_updated": last_updated,
        },
        "catalog": catalog,
        "format_distribution": format_distribution,
        "size_by_dataset": size_by_dataset,
    }


def breakdown():
    """Databases, model artifacts, recent changes, data events, staleness, lineage."""
    catalog = _build_catalog()
    db_files = _get_db_files()
    artifacts = _get_model_artifacts()
    recent_changes = _get_git_data_history(20)
    data_events = _get_data_events(30)

    # Staleness per dataset
    staleness = []
    for entry in catalog:
        if entry["last_modified"]:
            mt = datetime.fromisoformat(entry["last_modified"])
            days = (NOW - mt).days
        else:
            days = 999
        staleness.append({
            "name": entry["name"],
            "days_since_update": days,
            "status": _staleness_status(
                datetime.fromisoformat(entry["last_modified"])
            ) if entry["last_modified"] else "stale",
        })

    lineage = _build_lineage(catalog, artifacts)

    return {
        "available": True,
        "databases": db_files,
        "model_artifacts": artifacts,
        "recent_changes": recent_changes,
        "data_events": data_events,
        "staleness": staleness,
        "lineage": lineage,
    }


def definitions():
    """DVC concepts, data catalog terminology, versioning stages, thresholds."""
    return {
        "stages": [
            {
                "name": "Ingestion",
                "description": "Raw data arrives from clinical EEG devices, public datasets (CHB-MIT, TUSZ, BCI), or manual upload. Files land in data/<disease>/.",
            },
            {
                "name": "Cataloging",
                "description": "Each dataset directory is scanned, indexed by name, file count, total size, file formats, and last modification time.",
            },
            {
                "name": "Versioning",
                "description": "Datasets are versioned via git commits. The version tag is the short hash of the most recent commit that touched the dataset path.",
            },
            {
                "name": "Validation",
                "description": "Data quality checks (schema, completeness, drift) run on ingested data before training pipelines consume it.",
            },
            {
                "name": "Lineage Tracking",
                "description": "Source-to-artifact lineage links each dataset directory to its downstream model (.joblib) and the training script that produced it.",
            },
            {
                "name": "Archival / Retirement",
                "description": "Stale datasets (>30 days without update) are flagged. Archived datasets are retained for reproducibility but excluded from active training.",
            },
        ],
        "metrics": [
            {"name": "total_datasets", "description": "Count of dataset directories cataloged across data/ and data/eeg_datasets/."},
            {"name": "total_files", "description": "Sum of all files across every cataloged dataset directory."},
            {"name": "total_size_mb", "description": "Aggregate disk usage of all cataloged datasets in megabytes."},
            {"name": "db_count", "description": "Number of SQLite .db database files found under data/."},
            {"name": "model_artifacts", "description": "Number of .joblib serialized model files in models/."},
            {"name": "last_updated", "description": "ISO-8601 timestamp of the most recently modified file across all datasets and models."},
            {"name": "file_count", "description": "Number of files within a single dataset directory."},
            {"name": "size_mb", "description": "Disk size of a single dataset directory in megabytes."},
            {"name": "formats", "description": "List of unique file extensions (e.g. .csv, .npy, .edf) found in a dataset."},
            {"name": "version", "description": "Short git hash of the last commit touching the dataset path. Falls back to 'v1.0' if no git history."},
            {"name": "status", "description": "Staleness indicator: fresh (<7 days), active (7-30 days), stale (>30 days since last modification)."},
            {"name": "days_since_update", "description": "Number of days elapsed since the most recent file modification in a dataset."},
        ],
        "concepts": [
            {
                "term": "Data Versioning",
                "definition": "Tracking changes to datasets over time using git commits as version identifiers, enabling reproducibility of experiments.",
            },
            {
                "term": "Data Catalog",
                "definition": "A centralized inventory of all datasets, their metadata (size, format, location), and their current status.",
            },
            {
                "term": "Data Lineage",
                "definition": "The mapping from raw source data through processing pipelines to final model artifacts, showing how each model's training data originated.",
            },
            {
                "term": "Staleness",
                "definition": "A measure of how recently a dataset was updated. Thresholds: fresh (<7 days), active (7-30 days), stale (>30 days).",
            },
            {
                "term": "DVC (Data Version Control)",
                "definition": "A paradigm for versioning large data files alongside code. This dashboard implements lightweight DVC using git history for version tracking.",
            },
            {
                "term": "Model Artifact",
                "definition": "A serialized trained model file (.joblib) stored in models/. Linked to its source dataset via lineage tracking.",
            },
            {
                "term": "Format Distribution",
                "definition": "Breakdown of file types across all datasets, showing which formats (CSV, NPY, EDF, etc.) dominate the data landscape.",
            },
            {
                "term": "SQLite Catalog DB",
                "definition": "Database files (clinical.db, eeg_clinical.db, neuroai.db) storing structured clinical and EEG metadata alongside file-based datasets.",
            },
        ],
    }


# ── CLI quick-test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    print("=== overview ===")
    pprint.pprint(overview())
    print("\n=== breakdown ===")
    pprint.pprint(breakdown())
    print("\n=== definitions ===")
    pprint.pprint(definitions())
