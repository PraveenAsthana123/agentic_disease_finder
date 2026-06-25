#!/usr/bin/env python3
"""Ictal vs Interictal tests — real CHB-MIT seizure annotations + EDF."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mne = pytest.importorskip("mne")
import scripts.ictal_analysis as ia  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "**" / "chb*-summary.txt"), recursive=True))
pytestmark = pytest.mark.skipif(not HAS, reason="no CHB summary files")


def test_annotations_parse():
    a = ia.parse_seizure_annotations()
    # every annotated file has >=1 seizure with start<end
    for rec in a:
        assert rec["n_seizures"] >= 1
        for s in rec["seizures"]:
            assert s["start_s"] < s["end_s"]


def test_ictal_interictal_real_contrast():
    r = ia.ictal_interictal()
    if not r["available"]:
        pytest.skip("no annotated EDF on disk")
    assert set(r["band_power"]["ictal"]) >= {"delta", "alpha"}
    # windows are non-degenerate
    assert r["seizure_window"]["end_s"] > r["seizure_window"]["start_s"]


def test_signature_verdict_consistent():
    """Verdict must match the actual delta/alpha shift direction (no cosmetic verdict)."""
    r = ia.ictal_interictal()
    if not r["available"]:
        pytest.skip("no annotated EDF")
    sig = r["ictal_signature"]
    expected_ictal = sig["delta_shift"] > 0.05 and sig["alpha_shift"] < 0
    assert ("consistent with seizure" in sig["verdict"]) == expected_ictal


def test_band_power_sums_reasonable():
    r = ia.ictal_interictal()
    if not r["available"]:
        pytest.skip("no annotated EDF")
    for phase in ("ictal", "interictal"):
        s = sum(r["band_power"][phase].values())
        assert 0.5 < s <= 1.5  # relative band powers ~sum to 1
