#!/usr/bin/env python3
"""Clinical Psychologist module tests — real PHQ-9/GAD-7/NDDI-E/QOLIE-31 data.

Positive: every module function returns its expected shape.
Negative: the suicide-risk escalation flag MUST fire when PHQ-9 item 9 ≥ 1
(safety-critical — a regression that drops this flag silently hides
at-risk patients), and therapy_planning MUST mark those patients urgent.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.psychologist_module as ps  # noqa: E402

DB = ROOT / "data" / "clinical.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="clinical.db not present")


def test_full_dashboard_shape():
    r = ps.full_dashboard()
    assert set(r.get("modules", {})) == {
        "depression_anxiety", "coping_resilience", "seizure_emotion", "therapy_planning"
    }


def test_depression_anxiety_real_scores():
    r = ps.depression_anxiety()
    assert r["available"] is True
    assert r["summary"]["patients_assessed"] > 0
    # PHQ-9 + GAD-7 + NDDI-E all queried
    assert {"PHQ9", "GAD7", "NDDIE"} <= set(r["instruments"])


def test_suicide_risk_flag_fires():
    """SAFETY-CRITICAL: PHQ-9 item 9 ≥ 1 must produce a suicidal_ideation alert.

    The dataset has 8 such patients — if this drops to 0, the escalation
    path is broken and at-risk patients are silently hidden.
    """
    r = ps.depression_anxiety()
    si_alerts = [a for a in r["alerts"] if a["type"] == "suicidal_ideation"]
    assert len(si_alerts) >= 1, "suicide-risk escalation flag must fire on real data"
    assert all(a["severity"] in ("HIGH", "MODERATE") for a in si_alerts)


def test_suicidal_patients_are_urgent_in_therapy_plan():
    """Patients with suicidal ideation must be prioritized urgent."""
    da = ps.depression_anxiety()
    si_pids = {a["patient_id"] for a in da["alerts"] if a["type"] == "suicidal_ideation"}
    tp = ps.therapy_planning()
    urgent_pids = {p["patient_id"] for p in tp["plans"] if p["priority"] == "urgent"}
    assert si_pids <= urgent_pids, "every suicidal-ideation patient must be urgent in the therapy plan"


@pytest.mark.parametrize("fn", [
    ps.depression_anxiety, ps.coping_resilience,
    ps.seizure_emotion_correlation, ps.therapy_planning,
])
def test_each_module_returns_dict(fn):
    assert isinstance(fn(), dict)
