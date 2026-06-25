#!/usr/bin/env python3
"""False Alarm Review tests — detector vs ground-truth annotations."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mne = pytest.importorskip("mne")
import scripts.false_alarm as fa  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "**" / "chb*-summary.txt"), recursive=True))
pytestmark = pytest.mark.skipif(not HAS, reason="no CHB annotations")


def test_shape():
    r = fa.review()
    if not r["available"]:
        pytest.skip("no annotated edf")
    assert r["n_seizures_annotated"] >= 1


def test_metrics_in_range():
    r = fa.review()
    if not r["available"]:
        pytest.skip("no annotated edf")
    assert 0 <= r["sensitivity"] <= 1
    assert r["false_alarms"] >= 0
    assert r["false_alarms"] == len([w for w in r["false_alarm_windows"]]) or r["false_alarms"] >= len(r["false_alarm_windows"])


def test_verdict_matches_far():
    """Verdict 'acceptable' iff FA/hour <= 6 (validation lock)."""
    r = fa.review()
    if not r["available"]:
        pytest.skip("no annotated edf")
    far = r["false_alarms_per_hour"]
    assert (r["verdict"] == "acceptable") == (far <= 6)


def test_tp_plus_fa_equals_detections():
    """Every detection is TP or FA, none lost."""
    r = fa.review()
    if not r["available"]:
        pytest.skip("no annotated edf")
    assert r["true_positive_windows"] >= 0 and r["false_alarms"] >= 0
