#!/usr/bin/env python3
"""Montage Comparison tests — real monopolar EDF re-referencing."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mne = pytest.importorskip("mne")
import scripts.montage_compare as mc  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "real_eeg" / "depression_figshare" / "*.edf")))
pytestmark = pytest.mark.skipif(not HAS, reason="no monopolar edf")


def test_three_montages():
    r = mc.compare()
    assert r["available"] is True
    assert set(r["montages"]) == {"referential_original", "common_average", "bipolar_longitudinal"}


def test_bipolar_has_one_fewer_channel():
    """Bipolar longitudinal = N-1 sequential differences (real montage math)."""
    r = mc.compare()
    ref = r["montages"]["referential_original"]["n_channels"]
    bip = r["montages"]["bipolar_longitudinal"]["n_channels"]
    assert bip == ref - 1


def test_band_power_real():
    r = mc.compare()
    for m in r["montages"].values():
        s = sum(m["band_power"].values())
        assert 0.3 < s <= 1.2
        assert m["mean_amplitude_uv"] > 0
