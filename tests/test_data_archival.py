#!/usr/bin/env python3
"""Data Archival / Retention report tests — reads real table timestamps.

Positive: report shape + per-table row counts equal live COUNTs (§57.7).
Negative: clinical tables MUST never be archival candidates (retention=None →
retain indefinitely); a regression that flags clinical records for archival
is a data-loss hazard. Report must be non-destructive (no rows removed).
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.data_archival as da  # noqa: E402

DB = ROOT / "data" / "clinical.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="clinical.db not present")


def test_report_shape():
    r = da.archival_report()
    assert r["available"] is True
    assert r["summary"]["tables_tracked"] >= 1
    assert {"clinical", "operational_logs"} <= set(r["policy"])


def test_row_counts_match_real_tables():
    """No fabricated counts — each table's rows == live COUNT (§57.7)."""
    r = da.archival_report()
    c = sqlite3.connect(str(DB))
    for t in r["tables"]:
        real = c.execute(f"SELECT COUNT(*) FROM {t['table']}").fetchone()[0]
        assert t["rows"] == real, f"{t['table']} count mismatch"
    c.close()


def test_clinical_tables_never_archival_candidates():
    """SAFETY: clinical (retention=None) tables must have 0 archival candidates."""
    r = da.archival_report()
    for t in r["tables"]:
        if t["class"] == "clinical":
            assert t["archival_candidates"] == 0, f"clinical {t['table']} must never be archived"


def test_report_is_non_destructive():
    """Running the report must not change any row counts."""
    c = sqlite3.connect(str(DB))
    before = c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    da.archival_report()
    after = c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    c.close()
    assert before == after
