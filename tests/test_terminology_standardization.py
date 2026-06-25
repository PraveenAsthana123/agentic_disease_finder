#!/usr/bin/env python3
"""Terminology Mapping + Data Standardization tests — real assessments vocab.

Positive: every present instrument maps to a category; level normalization
produces canonical ordinals.
Negative: the empty/unknown NDDI-E level MUST be flagged non-conforming (a
regression that silently passes empty levels hides data-quality debt); the
report must be non-destructive (assessment row count unchanged).
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.terminology_standardization as ts  # noqa: E402

DB = ROOT / "data" / "clinical.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="clinical.db not present")


def test_terminology_maps_all_present_instruments():
    r = ts.terminology_map()
    assert r["available"] is True
    # every instrument present in the DB must be either mapped or explicitly unmapped
    assert r["n_instruments"] == len(r["mapped"]) + len(r["unmapped"])
    # domains rolled up
    assert "mood" in r["by_domain"] and "cognition" in r["by_domain"]


def test_standardization_flags_nonconforming():
    """Empty/unknown levels MUST be flagged (data-quality lock)."""
    r = ts.standardize_levels()
    assert r["conformant"] + r["nonconforming"] == r["total_records"]
    # the dataset has at least one empty NDDI-E level
    assert r["nonconforming"] >= 1
    assert any(n["issue"].startswith("empty") or "unrecognized" in n["issue"]
               for n in r["nonconforming_records"])


def test_canonical_ordinals_are_ordered():
    scale = ts.standardize_levels()["canonical_scale"]
    assert scale["normal"] < scale["mild"] < scale["moderate"] < scale["severe"]


def test_report_is_non_destructive():
    c = sqlite3.connect(str(DB))
    before = c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    ts.full_report()
    after = c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    c.close()
    assert before == after
