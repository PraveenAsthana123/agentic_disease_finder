#!/usr/bin/env python3
"""Localization Dashboard tests — seizure focus from annotated CHB-MIT EDF."""
import glob, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
mne = pytest.importorskip("mne")
import scripts.localization as L  # noqa: E402
HAS = bool(glob.glob(str(ROOT / "data" / "**" / "chb*-summary.txt"), recursive=True))
pytestmark = pytest.mark.skipif(not HAS, reason="no CHB annotations")


def test_localize_shape():
    r = L.localize()
    if not r["available"]:
        pytest.skip("no annotated edf")
    assert r["top_focus_channels"]
    assert "region" in r["localized_focus"] and "hemisphere" in r["localized_focus"]


def test_ranking_is_descending():
    """Top focus channels must be sorted by ictal increase descending (real ranking)."""
    r = L.localize()
    if not r["available"]:
        pytest.skip("no annotated edf")
    xs = [c["ictal_increase_x"] for c in r["all_channels_ranked"]]
    assert xs == sorted(xs, reverse=True)


def test_peak_channel_is_top():
    r = L.localize()
    if not r["available"]:
        pytest.skip("no annotated edf")
    assert r["localized_focus"]["peak_increase_x"] == r["top_focus_channels"][0]["ictal_increase_x"]


def test_hemisphere_region_valid():
    r = L.localize()
    if not r["available"]:
        pytest.skip("no annotated edf")
    for c in r["top_focus_channels"]:
        assert c["hemisphere"] in {"left", "right", "bilateral", "midline"}
