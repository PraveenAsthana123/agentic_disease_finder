#!/usr/bin/env python3
"""EEG Complexity tests — AntroPy + Nolds features on real EDF."""
import glob, sys, math
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("mne"); pytest.importorskip("antropy"); pytest.importorskip("nolds")
import scripts.eeg_complexity as ec  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "real_eeg" / "epilepsy_physionet" / "chb*.edf")))
pytestmark = pytest.mark.skipif(not HAS, reason="no chb edf")


def test_shape():
    r = ec.complexity(seconds=5)
    assert r["available"] is True
    assert r["n_channels"] >= 1


def test_entropy_values_normalized():
    """Spectral/permutation entropy are normalized [0,1] (real measure bounds)."""
    r = ec.complexity(seconds=5)
    for row in r["per_channel"]:
        if "spectral_entropy" in row:
            assert 0 <= row["spectral_entropy"] <= 1.001
            assert 0 <= row["permutation_entropy"] <= 1.001


def test_summary_means_finite():
    r = ec.complexity(seconds=5)
    for v in r["summary"].values():
        assert v is None or math.isfinite(v)


def test_libraries_declared():
    r = ec.complexity(seconds=5)
    assert "antropy" in r["libraries"] and "nolds" in r["libraries"]
