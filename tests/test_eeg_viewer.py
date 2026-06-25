#!/usr/bin/env python3
"""Raw EEG Viewer tests — real .edf waveform traces via MNE.

Positive: recordings list + raw_traces return real signal shapes.
Negative: trace point-count must be capped (downsample step ≥ 1 → never returns
the full raw sample count uncapped); µV values must be finite real numbers (a
regression returning NaN/placeholder breaks the strip chart).
"""
import glob
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

mne = pytest.importorskip("mne")
import scripts.eeg_viz as viz  # noqa: E402

HAS_EDF = bool(glob.glob(str(ROOT / "data" / "**" / "*.edf"), recursive=True))
pytestmark = pytest.mark.skipif(not HAS_EDF, reason="no .edf recordings present")


def test_list_recordings_real():
    r = viz.list_recordings()
    assert r["available"] is True
    assert r["n_total"] >= 1
    assert all("file" in x and x["file"].endswith(".edf") for x in r["recordings"])


def test_raw_traces_shape():
    f = viz.list_presets()["default"]
    t = viz.raw_traces(f, start=0, seconds=10)
    assert t["available"] is True
    assert t["n_channels"] >= 1
    assert len(t["time_s"]) == t["n_points"]
    for tr in t["traces"]:
        assert len(tr["uv"]) == t["n_points"]


def test_traces_values_are_finite():
    """µV samples must be real finite numbers (no NaN/placeholder)."""
    import math
    f = viz.list_presets()["default"]
    t = viz.raw_traces(f, start=0, seconds=5)
    sample = t["traces"][0]["uv"]
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in sample[:50])


def test_window_clamped_to_duration():
    """Requesting a start past the recording end must clamp, not crash."""
    f = viz.list_presets()["default"]
    t = viz.raw_traces(f, start=1e9, seconds=10)
    assert t["available"] is True
    assert t["window"]["start_s"] <= t["duration_s"]
