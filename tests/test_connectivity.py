#!/usr/bin/env python3
"""EEG Connectivity tests — mne-connectivity coherence on real EDF."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("mne"); pytest.importorskip("mne_connectivity")
import scripts.connectivity as C  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "real_eeg" / "epilepsy_physionet" / "chb*.edf")))
pytestmark = pytest.mark.skipif(not HAS, reason="no chb edf")


def test_shape():
    r = C.connectivity(seconds=10)
    assert r["available"] is True
    assert r["n_channels"] >= 2
    assert len(r["matrix"]) == r["n_channels"]


def test_coherence_bounds():
    """Coherence is in [0,1] (real measure bounds)."""
    r = C.connectivity(seconds=10)
    assert 0 <= r["mean_coherence"] <= 1
    assert 0 <= r["max_coherence"] <= 1.001


def test_matrix_symmetric():
    r = C.connectivity(seconds=10)
    m = r["matrix"]
    n = len(m)
    for i in range(min(n, 5)):
        for j in range(min(n, 5)):
            assert abs(m[i][j] - m[j][i]) < 1e-6


def test_hub_strength_descending():
    r = C.connectivity(seconds=10)
    strengths = [h["strength"] for h in r["hub_channels"]]
    assert strengths == sorted(strengths, reverse=True)
