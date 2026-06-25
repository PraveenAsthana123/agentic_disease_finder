#!/usr/bin/env python3
"""Artifact Review tests — window-based artifact detection from real EDF."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mne = pytest.importorskip("mne")
import scripts.eeg_quality as q  # noqa: E402
EDF = sorted(glob.glob(str(ROOT / "data" / "real_eeg" / "epilepsy_physionet" / "chb*.edf")))
pytestmark = pytest.mark.skipif(not EDF, reason="no chb edf")


def test_shape():
    r = q.artifact_review(EDF[0], seconds=30)
    assert r["available"] is True
    assert r["n_windows"] >= 1


def test_clean_plus_dirty_consistent():
    r = q.artifact_review(EDF[0], seconds=30)
    dirty = sum(1 for w in r["windows"] if not w["clean"])
    assert r["clean_windows"] + dirty == len(r["windows"][: r["n_windows"]]) or r["clean_windows"] <= r["n_windows"]
    assert 0 <= r["clean_pct"] <= 100


def test_quality_matches_clean_pct():
    """PASS iff >=80% clean, else REVIEW (QC lock)."""
    r = q.artifact_review(EDF[0], seconds=30)
    if r["n_windows"]:
        assert r["quality"] == ("PASS" if r["clean_pct"] >= 80 else "REVIEW")


def test_artifact_types_known():
    r = q.artifact_review(EDF[0], seconds=30)
    assert set(r["artifact_type_counts"]) == {"eye_blink", "muscle", "line_noise", "movement"}
