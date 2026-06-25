#!/usr/bin/env python3
"""Bad Channel Dashboard tests — per-channel QC from real EDF.

Positive: every channel gets a valid verdict; quality matches bad count.
Negative: the quality verdict MUST be REVIEW iff any bad channel exists
(a regression that PASSes a recording with a flat/disconnected channel
hides unusable data); every verdict must be from the allowed set.
"""
import glob
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

mne = pytest.importorskip("mne")
import scripts.eeg_quality as q  # noqa: E402

EDF = sorted(glob.glob(str(ROOT / "data" / "real_eeg" / "epilepsy_physionet" / "chb*.edf")))
pytestmark = pytest.mark.skipif(not EDF, reason="no chb edf present")
VALID = {"good", "flat", "disconnected", "noisy", "line-noise"}


def test_report_shape():
    r = q.bad_channels(EDF[0])
    assert r["available"] is True
    assert r["n_channels"] >= 1


def test_all_channels_valid_verdict():
    r = q.bad_channels(EDF[0])
    for c in r["channels"]:
        assert c["verdict"] in VALID


def test_quality_matches_bad_count():
    """PASS iff n_bad==0, REVIEW otherwise (QC lock)."""
    r = q.bad_channels(EDF[0])
    assert r["quality"] == ("PASS" if r["n_bad"] == 0 else "REVIEW")
    # bad list excludes 'good'
    assert all(c["verdict"] != "good" for c in r["bad_channels"])
    assert r["n_bad"] == len(r["bad_channels"])


def test_metrics_are_real_numbers():
    import math
    r = q.bad_channels(EDF[0])
    for c in r["channels"][:5]:
        assert math.isfinite(c["std_uv"]) and math.isfinite(c["line_noise_rel"])
