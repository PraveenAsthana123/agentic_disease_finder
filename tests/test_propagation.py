#!/usr/bin/env python3
"""Seizure Propagation Map tests — onset timing from annotated EDF."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mne = pytest.importorskip("mne")
import scripts.propagation as P  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "**" / "chb*-summary.txt"), recursive=True))
pytestmark = pytest.mark.skipif(not HAS, reason="no CHB annotations")


def test_shape():
    r = P.propagation()
    if not r["available"]:
        pytest.skip("no annotated edf")
    assert r["n_activated"] >= 1
    assert r["onset_leaders"]


def test_onsets_non_negative_and_ordered():
    """Onsets are within the ictal window (>=0) and ascending (interictal-baseline fix)."""
    r = P.propagation()
    if not r["available"]:
        pytest.skip("no annotated edf")
    times = [o["onset_s"] for o in r["propagation_order"]]
    assert all(t >= 0 for t in times), "onsets must be within the seizure window (>=0)"
    assert times == sorted(times)


def test_leaders_are_earliest():
    r = P.propagation()
    if not r["available"]:
        pytest.skip("no annotated edf")
    assert r["onset_leaders"][0]["onset_s"] == r["propagation_order"][0]["onset_s"]
