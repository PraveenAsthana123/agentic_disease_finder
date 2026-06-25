#!/usr/bin/env python3
"""TSFEL feature extraction tests — real EDF."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("mne"); pytest.importorskip("tsfel")
import scripts.tsfel_features as T  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "real_eeg" / "epilepsy_physionet" / "chb*.edf")))
pytestmark = pytest.mark.skipif(not HAS, reason="no chb edf")


def test_shape():
    r = T.extract(seconds=5, max_channels=3)
    assert r["available"] is True
    assert r["n_features_per_channel"] > 0


def test_per_channel_features():
    r = T.extract(seconds=5, max_channels=3)
    ok = [c for c in r["per_channel"] if "n_features" in c]
    assert ok, "at least one channel must extract features"
    for c in ok:
        assert c["n_features"] == r["n_features_per_channel"]


def test_domains_declared():
    r = T.extract(seconds=5, max_channels=2)
    assert set(r["feature_domains"]) == {"statistical", "temporal"}
