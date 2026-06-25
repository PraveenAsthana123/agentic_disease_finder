#!/usr/bin/env python3
"""Sleep State Dashboard tests — real Sleep-EDF hypnograms."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mne = pytest.importorskip("mne")
import scripts.sleep_staging as ss  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "**" / "*Hypnogram.edf"), recursive=True))
pytestmark = pytest.mark.skipif(not HAS, reason="no hypnograms")


def test_list_recordings():
    r = ss.list_sleep_recordings()
    assert r["available"] is True and r["n_total"] >= 1


def test_architecture_real_metrics():
    r = ss.sleep_architecture()
    assert r["available"] is True
    assert r["total_sleep_time_min"] > 0
    assert set(r["stages"]) == {"W", "N1", "N2", "N3", "REM"}


def test_sleep_efficiency_in_range():
    """After sleep-period clipping, efficiency must be a sane 0-100% (not >100 or <5)."""
    r = ss.sleep_architecture()
    assert 5 <= r["sleep_efficiency_pct"] <= 100


def test_sleep_stage_pcts_sum_to_100():
    """N1+N2+N3+REM as % of TST must sum ~100 (no double-count / W leakage into sleep%)."""
    r = ss.sleep_architecture()
    s = sum(r["stages"][st]["pct_of_sleep"] for st in ["N1", "N2", "N3", "REM"])
    assert 95 <= s <= 105


def test_quality_matches_flags():
    r = ss.sleep_architecture()
    assert r["quality"] == ("PASS" if r["flags"] == ["within normative ranges"] else "REVIEW")
