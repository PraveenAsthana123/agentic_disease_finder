"""Database Ops Dashboard — real SQLite health metrics, table sizes, row counts,
WAL status, backup history, query volume, and integrity checks from clinical.db."""

import sqlite3
import os
import glob
from datetime import datetime, timezone


DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')
BACKUP_DIR = os.path.join(os.path.dirname(__file__), '..', 'backups')


def _conn():
    return sqlite3.connect(DB)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_overview():
    """Database health overview — size, tables, rows, WAL, integrity, backups."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # File size
    db_size_bytes = os.path.getsize(DB)
    wal_path = DB + "-wal"
    wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0

    # Page info
    cur.execute("PRAGMA page_size")
    page_size = cur.fetchone()[0]
    cur.execute("PRAGMA page_count")
    page_count = cur.fetchone()[0]
    cur.execute("PRAGMA freelist_count")
    freelist = cur.fetchone()[0]

    # Journal mode
    cur.execute("PRAGMA journal_mode")
    journal_mode = cur.fetchone()[0]

    # Integrity
    cur.execute("PRAGMA integrity_check")
    integrity = cur.fetchone()[0]

    # All tables with row counts and column counts
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = []
    total_rows = 0
    for (tname,) in cur.fetchall():
        cur.execute(f"SELECT count(*) FROM [{tname}]")
        rows = cur.fetchone()[0]
        total_rows += rows
        cur.execute(f"PRAGMA table_info([{tname}])")
        cols = len(cur.fetchall())
        tables.append({"name": tname, "rows": rows, "columns": cols})

    # Sort by rows desc for the overview
    tables.sort(key=lambda t: t["rows"], reverse=True)

    # Index count
    cur.execute("SELECT count(*) FROM sqlite_master WHERE type='index'")
    index_count = cur.fetchone()[0]

    # Trigger count
    cur.execute("SELECT count(*) FROM sqlite_master WHERE type='trigger'")
    trigger_count = cur.fetchone()[0]

    # Backups
    backup_files = sorted(glob.glob(os.path.join(BACKUP_DIR, "clinical_db_*.db"))) if os.path.isdir(BACKUP_DIR) else []
    backup_count = len(backup_files)
    latest_backup = None
    oldest_backup = None
    total_backup_size = 0
    if backup_files:
        latest_backup = os.path.basename(backup_files[-1])
        oldest_backup = os.path.basename(backup_files[0])
        total_backup_size = sum(os.path.getsize(f) for f in backup_files)

    conn.close()

    return {
        "available": True,
        "generated_at": _now(),
        "summary": {
            "db_size_kb": round(db_size_bytes / 1024, 1),
            "wal_size_kb": round(wal_size / 1024, 1),
            "total_tables": len(tables),
            "total_rows": total_rows,
            "total_indexes": index_count,
            "total_triggers": trigger_count,
            "page_size": page_size,
            "page_count": page_count,
            "freelist_pages": freelist,
            "fragmentation_pct": round(freelist / max(page_count, 1) * 100, 1),
            "journal_mode": journal_mode,
            "integrity": integrity,
        },
        "tables": tables,
        "backups": {
            "count": backup_count,
            "latest": latest_backup,
            "oldest": oldest_backup,
            "total_size_kb": round(total_backup_size / 1024, 1),
            "retention_days": 14,
        },
    }


def db_breakdown():
    """Per-table breakdown: rows, columns, indexes, estimated size, recent activity."""
    if not os.path.exists(DB):
        return {"available": False, "tables": []}

    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    all_tables = [r[0] for r in cur.fetchall()]

    # Gather indexes per table
    cur.execute("SELECT tbl_name, name FROM sqlite_master WHERE type='index'")
    idx_map = {}
    for tbl, idx in cur.fetchall():
        idx_map.setdefault(tbl, []).append(idx)

    page_size = cur.execute("PRAGMA page_size").fetchone()[0]

    result = []
    for tname in all_tables:
        cur.execute(f"SELECT count(*) FROM [{tname}]")
        rows = cur.fetchone()[0]
        cur.execute(f"PRAGMA table_info([{tname}])")
        columns = cur.fetchall()
        col_names = [c[1] for c in columns]
        col_types = [c[2] for c in columns]
        indexes = idx_map.get(tname, [])

        # Check for common timestamp columns to detect recent activity
        last_activity = None
        for ts_col in ["ts_utc", "created_at", "timestamp", "date", "ts", "updated_at"]:
            if ts_col in col_names:
                try:
                    cur.execute(f"SELECT max([{ts_col}]) FROM [{tname}]")
                    val = cur.fetchone()[0]
                    if val:
                        last_activity = str(val)
                        break
                except Exception:
                    pass

        # Has NULL values in any column?
        nullable_cols = 0
        for c in columns:
            if c[3] == 0:  # notnull = 0 means nullable
                nullable_cols += 1

        result.append({
            "name": tname,
            "rows": rows,
            "columns": len(columns),
            "column_names": col_names,
            "column_types": col_types,
            "indexes": indexes,
            "index_count": len(indexes),
            "nullable_columns": nullable_cols,
            "last_activity": last_activity,
        })

    result.sort(key=lambda t: t["rows"], reverse=True)
    conn.close()

    return {"available": True, "tables": result}


def db_definitions():
    """Database ops metric definitions for tooltip overlays."""
    return {
        "available": True,
        "definitions": [
            {"term": "WAL Mode", "meaning": "Write-Ahead Logging — allows concurrent reads during writes; standard for SQLite production use."},
            {"term": "Page Size", "meaning": "Size in bytes of each database page. Default is 4096. Affects I/O efficiency."},
            {"term": "Page Count", "meaning": "Total number of pages in the database file. DB size = page_count x page_size."},
            {"term": "Freelist Pages", "meaning": "Pages that were previously used but are now empty. High count suggests VACUUM needed."},
            {"term": "Fragmentation %", "meaning": "Percentage of pages on the freelist. >10% suggests running VACUUM."},
            {"term": "Integrity Check", "meaning": "SQLite PRAGMA integrity_check result. 'ok' means no corruption detected."},
            {"term": "Indexes", "meaning": "Database indexes speed up queries on indexed columns. Missing indexes cause slow queries."},
            {"term": "Row Count", "meaning": "Total number of records in a table."},
            {"term": "Backup Retention", "meaning": "Number of days backup files are retained. Older backups are automatically pruned."},
            {"term": "Last Activity", "meaning": "Most recent timestamp found in the table, indicating when it was last written to."},
        ],
    }
