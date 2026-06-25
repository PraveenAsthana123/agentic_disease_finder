#!/usr/bin/env python3
"""
Occupational Therapist (OT) module tests
========================================

Locks the real bug fixed on 2026-06-25: the `medications` table stores drug
info inside a `fields_json` blob (columns: id, patient_id, fields_json,
created_at) — NOT as flat `drug_name`/`dose`/`frequency` columns. The OT
fall-risk assessment must parse `fields_json`; querying `drug_name` as a
column raises `sqlite3.OperationalError: no such column: drug_name` (→ 500).

Positive: every OT module function returns its expected shape.
Negative: medications has no `drug_name` column (so the JSON-parse path is
the only correct one) AND fall_risk_assessment never raises on real data.

Author: AgenticFinder Research Team
License: MIT
"""

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.ot_module as ot  # noqa: E402

DB = ROOT / "data" / "clinical.db"


@pytest.mark.skipif(not DB.exists(), reason="clinical.db not present")
def test_medications_has_no_drug_name_column():
    """NEGATIVE: drug info lives in fields_json, not a flat column.

    This is the regression lock — if a future edit reintroduces
    `SELECT ... drug_name FROM medications`, the OT endpoint 500s again.
    """
    c = sqlite3.connect(str(DB))
    cols = [r[1] for r in c.execute("PRAGMA table_info(medications)")]
    c.close()
    assert "drug_name" not in cols, (
        "medications has no flat drug_name column; OT must parse fields_json"
    )
    assert "fields_json" in cols


@pytest.mark.skipif(not DB.exists(), reason="clinical.db not present")
def test_fall_risk_does_not_raise():
    """The exact crash path — must run clean on real medications data."""
    r = ot.fall_risk_assessment()
    assert isinstance(r, dict)


@pytest.mark.skipif(not DB.exists(), reason="clinical.db not present")
def test_full_dashboard_shape():
    """POSITIVE: the four OT modules are all present."""
    r = ot.full_dashboard()
    assert set(r.get("modules", {})) >= {
        "adl_assessment",
        "fall_risk",
        "return_to_work",
        "cognitive_function",
    }


@pytest.mark.skipif(not DB.exists(), reason="clinical.db not present")
@pytest.mark.parametrize(
    "fn",
    [
        ot.adl_assessment,
        ot.fall_risk_assessment,
        ot.return_to_work,
        ot.cognitive_function_ot,
    ],
)
def test_each_module_returns_dict(fn):
    assert isinstance(fn(), dict)
